#!/usr/bin/env python3
"""
Gateway Monitor — Polls the inference subnet gateway and sends Telegram alerts.

Monitors:
  - Miner liveness (up/down transitions)
  - Challenge pass rates (< 90% threshold)
  - Error rate spikes (> 5% in trailing window)
  - Gateway reachability
  - Latency degradation (TTFT and TPS)

Usage:
  python gateway_monitor.py --gateway-url http://localhost:8081 --interval 30

Environment variables:
  TELEGRAM_BOT_TOKEN  — Telegram bot token
  TELEGRAM_CHAT_ID    — Chat ID for alerts
  MONITORING_KEY      — Bearer token for authenticated endpoints (optional)
  METRICS_LOG         — Path for JSON metrics log (default: gateway_metrics.log)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web

# ── Logging ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("gateway_monitor")

# ── Constants ────────────────────────────────────────────────────────────

DEFAULT_INTERVAL = 30           # seconds between polls
ALERT_COOLDOWN = 300            # 5 min between identical alerts
PASS_RATE_THRESHOLD = 0.90      # alert when challenge pass rate < this
ERROR_RATE_THRESHOLD = 0.05     # alert when error rate > 5%
ERROR_WINDOW = 300              # 5-minute sliding window for error rate
LATENCY_TTFT_FACTOR = 2.0      # alert if TTFT > 2x baseline
LATENCY_TPS_FACTOR = 0.5       # alert if TPS < 50% of baseline
BASELINE_WARMUP_POLLS = 5       # polls before latency baseline is set
TELEGRAM_TIMEOUT = 10           # seconds


# ── Data structures ──────────────────────────────────────────────────────

@dataclass
class MinerSnapshot:
    uid: int
    alive: bool
    reliability: float = 0.0
    served: int = 0
    failed: int = 0
    avg_ttft_ms: float = 0.0
    avg_tps: float = 0.0
    pass_rate: float = 1.0


@dataclass
class GatewaySnapshot:
    ts: float
    reachable: bool
    status: str = ""
    model: str = ""
    miners_total: int = 0
    miners_alive: int = 0
    epoch: int = 0
    total_organic: int = 0
    total_synthetic: int = 0
    challenges_total: int = 0
    challenges_passed: int = 0
    challenges_failed: int = 0
    miners: list[MinerSnapshot] = field(default_factory=list)

    def total_requests(self) -> int:
        return self.total_organic + self.total_synthetic

    def challenge_pass_rate(self) -> float:
        total = self.challenges_passed + self.challenges_failed
        if total == 0:
            return 1.0
        return self.challenges_passed / total

    def to_dict(self) -> dict:
        d = {
            "ts": self.ts,
            "reachable": self.reachable,
            "status": self.status,
            "model": self.model,
            "miners_total": self.miners_total,
            "miners_alive": self.miners_alive,
            "epoch": self.epoch,
            "total_organic": self.total_organic,
            "total_synthetic": self.total_synthetic,
            "challenges_total": self.challenges_total,
            "challenges_passed": self.challenges_passed,
            "challenges_failed": self.challenges_failed,
        }
        if self.miners:
            d["miners"] = [
                {
                    "uid": m.uid, "alive": m.alive, "reliability": m.reliability,
                    "served": m.served, "failed": m.failed,
                    "avg_ttft_ms": m.avg_ttft_ms, "avg_tps": m.avg_tps,
                    "pass_rate": m.pass_rate,
                }
                for m in self.miners
            ]
        return d


# ── Alerting ─────────────────────────────────────────────────────────────

class AlertManager:
    """Sends Telegram alerts with per-key cooldowns."""

    def __init__(self, bot_token: str | None, chat_id: str | None):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._last_alert: dict[str, float] = {}  # alert_key -> timestamp
        self._enabled = bool(bot_token and chat_id)
        if not self._enabled:
            log.warning("Telegram alerts disabled (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set)")

    def _cooldown_ok(self, key: str) -> bool:
        now = time.time()
        last = self._last_alert.get(key, 0)
        if now - last < ALERT_COOLDOWN:
            return False
        self._last_alert[key] = now
        return True

    async def send(self, key: str, message: str, *, force: bool = False):
        """Send an alert if cooldown allows. `key` deduplicates alerts."""
        if not force and not self._cooldown_ok(key):
            log.debug("Alert suppressed (cooldown): %s", key)
            return

        log.warning("ALERT [%s]: %s", key, message)

        if not self._enabled:
            return

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=TELEGRAM_TIMEOUT)) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        log.error("Telegram API error %d: %s", resp.status, body[:200])
        except Exception as exc:
            log.error("Failed to send Telegram alert: %s", exc)


# ── Monitor ──────────────────────────────────────────────────────────────

class GatewayMonitor:
    """Core monitor: poll, detect anomalies, alert."""

    def __init__(
        self,
        gateway_url: str,
        interval: int,
        alert_mgr: AlertManager,
        monitoring_key: str | None = None,
        metrics_log: str = "gateway_metrics.log",
    ):
        self.gateway_url = gateway_url.rstrip("/")
        self.interval = interval
        self.alert = alert_mgr
        self.monitoring_key = monitoring_key
        self.metrics_log = Path(metrics_log)

        # State
        self.prev: GatewaySnapshot | None = None
        self.prev_miners: dict[int, MinerSnapshot] = {}  # uid -> last snapshot
        self._error_history: list[tuple[float, int, int]] = []  # (ts, served, failed)
        self._ttft_baseline: float | None = None
        self._tps_baseline: float | None = None
        self._baseline_samples: list[tuple[float, float]] = []  # (ttft, tps)
        self._poll_count = 0
        self._gateway_was_reachable = True

        # Latest snapshot for /status endpoint
        self.latest: GatewaySnapshot | None = None

    def _headers(self, auth: bool = False) -> dict[str, str]:
        h: dict[str, str] = {}
        if auth and self.monitoring_key:
            h["Authorization"] = f"Bearer {self.monitoring_key}"
        return h

    # ── Polling ──────────────────────────────────────────────────────

    async def poll(self) -> GatewaySnapshot:
        """Fetch /v1/health (with auth for details) and return a snapshot."""
        snap = GatewaySnapshot(ts=time.time(), reachable=False)
        try:
            async with aiohttp.ClientSession() as session:
                # Health endpoint — try with auth first for detail, fall back
                async with session.get(
                    f"{self.gateway_url}/v1/health",
                    headers=self._headers(auth=True),
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status != 200:
                        log.error("Health endpoint returned %d", resp.status)
                        return snap
                    data = await resp.json()

                snap.reachable = True
                snap.status = data.get("status", "")
                snap.model = data.get("model", "")
                snap.miners_total = data.get("miners_total", 0)
                snap.miners_alive = data.get("miners_alive", 0)
                snap.epoch = data.get("epoch", 0)
                snap.total_organic = data.get("total_organic", 0)
                snap.total_synthetic = data.get("total_synthetic", 0)

                # Challenge stats (only present with auth)
                challenges = data.get("challenges", {})
                snap.challenges_total = challenges.get("total", 0)
                snap.challenges_passed = challenges.get("passed", 0)
                snap.challenges_failed = challenges.get("failed", 0)

                # Per-miner detail (only present with auth)
                for md in data.get("miners_detail", []):
                    snap.miners.append(MinerSnapshot(
                        uid=md["uid"],
                        alive=md["alive"],
                        reliability=md.get("reliability", 0),
                        served=md.get("served", 0),
                        failed=md.get("failed", 0),
                        avg_ttft_ms=md.get("avg_ttft_ms", 0),
                        avg_tps=md.get("avg_tps", 0),
                    ))

                # If we got per-miner detail, enrich with scoreboard pass_rate
                if snap.miners and self.monitoring_key:
                    try:
                        async with session.get(
                            f"{self.gateway_url}/v1/scoreboard",
                            headers=self._headers(auth=True),
                            timeout=aiohttp.ClientTimeout(total=10),
                        ) as sb_resp:
                            if sb_resp.status == 200:
                                sb_data = await sb_resp.json()
                                sb_by_uid = {m["uid"]: m for m in sb_data.get("miners", [])}
                                for miner in snap.miners:
                                    sb = sb_by_uid.get(miner.uid, {})
                                    miner.pass_rate = sb.get("pass_rate", 1.0)
                    except Exception as exc:
                        log.debug("Scoreboard fetch failed: %s", exc)

        except asyncio.TimeoutError:
            log.error("Gateway poll timed out")
        except aiohttp.ClientError as exc:
            log.error("Gateway connection error: %s", exc)
        except Exception as exc:
            log.error("Unexpected poll error: %s", exc)

        return snap

    # ── Analysis ─────────────────────────────────────────────────────

    async def analyze(self, snap: GatewaySnapshot):
        """Compare snapshot to previous state, emit alerts."""
        self._poll_count += 1

        # 1. Gateway reachability
        if not snap.reachable:
            if self._gateway_was_reachable:
                await self.alert.send(
                    "gateway_down",
                    f"<b>Gateway DOWN</b>\n"
                    f"URL: {self.gateway_url}\n"
                    f"Time: {_fmt_time(snap.ts)}",
                )
                self._gateway_was_reachable = False
            return  # no further analysis if unreachable

        if not self._gateway_was_reachable:
            await self.alert.send(
                "gateway_up",
                f"<b>Gateway RECOVERED</b>\n"
                f"URL: {self.gateway_url}\n"
                f"Miners alive: {snap.miners_alive}/{snap.miners_total}\n"
                f"Time: {_fmt_time(snap.ts)}",
                force=True,
            )
            self._gateway_was_reachable = True

        # 2. Miner liveness transitions
        curr_miners = {m.uid: m for m in snap.miners}
        for uid, miner in curr_miners.items():
            prev = self.prev_miners.get(uid)
            if prev is None:
                continue  # new miner, no transition to report
            if prev.alive and not miner.alive:
                await self.alert.send(
                    f"miner_down_{uid}",
                    f"<b>Miner {uid} DOWN</b>\n"
                    f"Was alive, now unreachable.\n"
                    f"Last reliability: {prev.reliability:.3f}\n"
                    f"Time: {_fmt_time(snap.ts)}",
                )
            elif not prev.alive and miner.alive:
                await self.alert.send(
                    f"miner_up_{uid}",
                    f"<b>Miner {uid} RECOVERED</b>\n"
                    f"Back online.\n"
                    f"Time: {_fmt_time(snap.ts)}",
                    force=True,
                )

        # 3. Challenge pass rate
        pass_rate = snap.challenge_pass_rate()
        if snap.challenges_total > 0 and pass_rate < PASS_RATE_THRESHOLD:
            await self.alert.send(
                "low_pass_rate",
                f"<b>Challenge pass rate LOW</b>\n"
                f"Rate: {pass_rate:.1%} (threshold: {PASS_RATE_THRESHOLD:.0%})\n"
                f"Passed: {snap.challenges_passed}, Failed: {snap.challenges_failed}\n"
                f"Time: {_fmt_time(snap.ts)}",
            )

        # Per-miner pass rate check
        for miner in snap.miners:
            if miner.pass_rate < PASS_RATE_THRESHOLD and miner.served > 5:
                await self.alert.send(
                    f"miner_low_pass_{miner.uid}",
                    f"<b>Miner {miner.uid} pass rate LOW</b>\n"
                    f"Rate: {miner.pass_rate:.1%}\n"
                    f"Served: {miner.served}, Failed: {miner.failed}\n"
                    f"Time: {_fmt_time(snap.ts)}",
                )

        # 4. Error rate spike (sliding window)
        total_served = sum(m.served for m in snap.miners)
        total_failed = sum(m.failed for m in snap.miners)
        now = snap.ts
        self._error_history.append((now, total_served, total_failed))
        # Prune old entries
        self._error_history = [(t, s, f) for t, s, f in self._error_history if now - t <= ERROR_WINDOW]

        if len(self._error_history) >= 2:
            oldest_ts, oldest_served, oldest_failed = self._error_history[0]
            delta_served = total_served - oldest_served
            delta_failed = total_failed - oldest_failed
            delta_total = delta_served + delta_failed
            if delta_total > 0:
                error_rate = delta_failed / delta_total if delta_total > 10 else 0
                if error_rate > ERROR_RATE_THRESHOLD:
                    await self.alert.send(
                        "high_error_rate",
                        f"<b>Error rate SPIKE</b>\n"
                        f"Rate: {error_rate:.1%} over last {int(now - oldest_ts)}s\n"
                        f"Failed: {delta_failed}, Total: {delta_total}\n"
                        f"Time: {_fmt_time(snap.ts)}",
                    )

        # 5. Latency degradation
        alive_miners = [m for m in snap.miners if m.alive and m.avg_tps > 0]
        if alive_miners:
            avg_ttft = sum(m.avg_ttft_ms for m in alive_miners) / len(alive_miners)
            avg_tps = sum(m.avg_tps for m in alive_miners) / len(alive_miners)

            if self._poll_count <= BASELINE_WARMUP_POLLS:
                self._baseline_samples.append((avg_ttft, avg_tps))
                if self._poll_count == BASELINE_WARMUP_POLLS:
                    self._ttft_baseline = sum(s[0] for s in self._baseline_samples) / len(self._baseline_samples)
                    self._tps_baseline = sum(s[1] for s in self._baseline_samples) / len(self._baseline_samples)
                    log.info("Latency baselines set: TTFT=%.1f ms, TPS=%.1f", self._ttft_baseline, self._tps_baseline)
            else:
                if self._ttft_baseline and avg_ttft > self._ttft_baseline * LATENCY_TTFT_FACTOR:
                    await self.alert.send(
                        "high_ttft",
                        f"<b>TTFT degraded</b>\n"
                        f"Current: {avg_ttft:.0f} ms (baseline: {self._ttft_baseline:.0f} ms)\n"
                        f"Factor: {avg_ttft / self._ttft_baseline:.1f}x\n"
                        f"Time: {_fmt_time(snap.ts)}",
                    )
                if self._tps_baseline and avg_tps < self._tps_baseline * LATENCY_TPS_FACTOR:
                    await self.alert.send(
                        "low_tps",
                        f"<b>TPS degraded</b>\n"
                        f"Current: {avg_tps:.1f} tok/s (baseline: {self._tps_baseline:.1f} tok/s)\n"
                        f"Factor: {avg_tps / self._tps_baseline:.2f}x\n"
                        f"Time: {_fmt_time(snap.ts)}",
                    )

        # Update state
        self.prev_miners = curr_miners
        self.prev = snap

    # ── Metrics logging ──────────────────────────────────────────────

    def log_metrics(self, snap: GatewaySnapshot):
        """Append snapshot as JSON line to metrics log."""
        try:
            with open(self.metrics_log, "a") as f:
                f.write(json.dumps(snap.to_dict()) + "\n")
        except Exception as exc:
            log.error("Failed to write metrics log: %s", exc)

    # ── Main loop ────────────────────────────────────────────────────

    async def run(self):
        """Main polling loop."""
        log.info("Starting gateway monitor")
        log.info("  Gateway URL: %s", self.gateway_url)
        log.info("  Poll interval: %ds", self.interval)
        log.info("  Auth: %s", "yes" if self.monitoring_key else "no (limited data)")
        log.info("  Metrics log: %s", self.metrics_log)

        while True:
            try:
                snap = await self.poll()
                self.latest = snap

                if snap.reachable:
                    log.info(
                        "Poll OK: miners=%d/%d epoch=%d organic=%d synthetic=%d pass_rate=%.1f%%",
                        snap.miners_alive, snap.miners_total, snap.epoch,
                        snap.total_organic, snap.total_synthetic,
                        snap.challenge_pass_rate() * 100,
                    )
                else:
                    log.warning("Poll FAILED: gateway unreachable")

                await self.analyze(snap)
                self.log_metrics(snap)

            except Exception as exc:
                log.error("Monitor loop error: %s", exc, exc_info=True)

            await asyncio.sleep(self.interval)


# ── Status HTTP server ───────────────────────────────────────────────────

class StatusServer:
    """Tiny HTTP server exposing /status with the latest snapshot."""

    def __init__(self, monitor: GatewayMonitor, port: int = 9100):
        self.monitor = monitor
        self.port = port

    async def handle_status(self, request: web.Request) -> web.Response:
        snap = self.monitor.latest
        if snap is None:
            return web.json_response({"status": "starting", "message": "No data yet"}, status=503)

        data: dict[str, Any] = {
            "monitor_status": "ok",
            "gateway_reachable": snap.reachable,
            "last_poll": _fmt_time(snap.ts),
            "last_poll_unix": snap.ts,
        }
        if snap.reachable:
            data.update({
                "gateway_status": snap.status,
                "model": snap.model,
                "miners_alive": snap.miners_alive,
                "miners_total": snap.miners_total,
                "epoch": snap.epoch,
                "total_organic": snap.total_organic,
                "total_synthetic": snap.total_synthetic,
                "challenge_pass_rate": round(snap.challenge_pass_rate(), 4),
            })
            if snap.miners:
                data["miners"] = [
                    {
                        "uid": m.uid,
                        "alive": m.alive,
                        "reliability": m.reliability,
                        "avg_ttft_ms": m.avg_ttft_ms,
                        "avg_tps": m.avg_tps,
                    }
                    for m in snap.miners
                ]
        return web.json_response(data)

    async def handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def start(self):
        app = web.Application()
        app.router.add_get("/status", self.handle_status)
        app.router.add_get("/health", self.handle_health)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self.port)
        await site.start()
        log.info("Status server listening on port %d", self.port)


# ── Helpers ──────────────────────────────────────────────────────────────

def _fmt_time(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(ts))


# ── CLI ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Monitor the inference subnet gateway and send Telegram alerts.",
    )
    p.add_argument(
        "--gateway-url",
        default="http://localhost:8081",
        help="Base URL of the gateway (default: http://localhost:8081)",
    )
    p.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL,
        help=f"Polling interval in seconds (default: {DEFAULT_INTERVAL})",
    )
    p.add_argument(
        "--status-port",
        type=int,
        default=9100,
        help="Port for the /status HTTP endpoint (default: 9100, 0 to disable)",
    )
    p.add_argument(
        "--metrics-log",
        default=os.environ.get("METRICS_LOG", "gateway_metrics.log"),
        help="Path for JSON-lines metrics log (default: gateway_metrics.log)",
    )
    p.add_argument(
        "--monitoring-key",
        default=os.environ.get("MONITORING_KEY", ""),
        help="Bearer token for authenticated gateway endpoints (env: MONITORING_KEY)",
    )
    p.add_argument(
        "--pass-rate-threshold",
        type=float,
        default=PASS_RATE_THRESHOLD,
        help=f"Challenge pass rate alert threshold (default: {PASS_RATE_THRESHOLD})",
    )
    p.add_argument(
        "--error-rate-threshold",
        type=float,
        default=ERROR_RATE_THRESHOLD,
        help=f"Error rate alert threshold (default: {ERROR_RATE_THRESHOLD})",
    )
    return p.parse_args()


async def main():
    args = parse_args()

    # Override thresholds from CLI
    global PASS_RATE_THRESHOLD, ERROR_RATE_THRESHOLD
    PASS_RATE_THRESHOLD = args.pass_rate_threshold
    ERROR_RATE_THRESHOLD = args.error_rate_threshold

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TAU_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID") or os.environ.get("TELEGRAM_OWNER_ID")
    alert_mgr = AlertManager(bot_token, chat_id)

    monitor = GatewayMonitor(
        gateway_url=args.gateway_url,
        interval=args.interval,
        alert_mgr=alert_mgr,
        monitoring_key=args.monitoring_key or None,
        metrics_log=args.metrics_log,
    )

    tasks = [asyncio.create_task(monitor.run())]

    if args.status_port > 0:
        status_server = StatusServer(monitor, port=args.status_port)
        tasks.append(asyncio.create_task(status_server.start()))

    # Run until interrupted
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        log.info("Monitor shutting down")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Interrupted, exiting.")
        sys.exit(0)
