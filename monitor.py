#!/usr/bin/env python3
"""
Gateway monitoring & Telegram alerting.

Polls the gateway's /v1/health and /metrics endpoints and sends Telegram alerts
when issues are detected:
  - Miner drops (alive count decreases)
  - Error rate spikes (failures > threshold)
  - Latency degradation (TTFT or TPS anomalies)
  - Gateway unreachable

Usage:
    # Run as background process
    python monitor.py --gateway http://127.0.0.1:8081 --interval 30

    # With Telegram alerts via arbos.py
    python monitor.py --gateway http://127.0.0.1:8081 --arbos-dir /Arbos

    # Standalone with direct Telegram bot
    python monitor.py --gateway http://127.0.0.1:8081 --bot-token TOKEN --chat-id CHAT_ID
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("monitor")


@dataclass
class AlertState:
    """Tracks state to avoid duplicate alerts."""
    last_miners_alive: int = -1
    last_miners_total: int = -1
    last_alert_time: float = 0.0
    consecutive_failures: int = 0
    gateway_was_down: bool = False
    last_error_count: int = 0
    last_challenge_fail_count: int = 0
    alert_cooldown_s: float = 300.0  # 5 min cooldown between same alerts
    history: list = field(default_factory=list)  # Recent health snapshots
    per_miner_alert_time: dict = field(default_factory=dict)  # uid -> last alert time


class GatewayMonitor:
    """Monitors gateway health and sends alerts."""

    def __init__(
        self,
        gateway_url: str,
        arbos_dir: Optional[str] = None,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        poll_interval: float = 30.0,
        alert_cooldown: float = 300.0,
    ):
        self.gateway_url = gateway_url.rstrip("/")
        self.arbos_dir = arbos_dir
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.poll_interval = poll_interval
        self.state = AlertState(alert_cooldown_s=alert_cooldown)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _send_alert(self, message: str, level: str = "WARN"):
        """Send alert via available channels."""
        prefix = {"CRIT": "\u26a0\ufe0f CRITICAL", "WARN": "\u26a0\ufe0f WARNING", "INFO": "\u2139\ufe0f INFO", "OK": "\u2705 RESOLVED"}.get(level, level)
        full_msg = f"{prefix}: {message}"
        log.warning(full_msg)

        # Send via arbos.py (preferred — uses existing Telegram integration)
        if self.arbos_dir:
            try:
                subprocess.run(
                    [sys.executable, "arbos.py", "send", "--new", full_msg],
                    cwd=self.arbos_dir,
                    capture_output=True,
                    timeout=30,
                )
            except Exception as e:
                log.error(f"Failed to send via arbos: {e}")

        # Direct Telegram fallback
        elif self.bot_token and self.chat_id:
            try:
                import requests
                requests.post(
                    f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
                    json={"chat_id": self.chat_id, "text": full_msg},
                    timeout=10,
                )
            except Exception as e:
                log.error(f"Failed to send Telegram: {e}")

    def _can_alert(self) -> bool:
        """Check cooldown."""
        now = time.time()
        if now - self.state.last_alert_time < self.state.alert_cooldown_s:
            return False
        self.state.last_alert_time = now
        return True

    async def _fetch_health(self) -> Optional[dict]:
        """Fetch gateway health data."""
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.gateway_url}/v1/health",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                log.warning(f"Health check returned {resp.status}")
                return None
        except Exception as e:
            log.warning(f"Health check failed: {e}")
            return None

    async def _fetch_metrics(self) -> Optional[str]:
        """Fetch Prometheus metrics."""
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.gateway_url}/metrics",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    return await resp.text()
                return None
        except Exception:
            return None

    def _parse_metric(self, metrics_text: str, name: str) -> float:
        """Extract a single metric value from Prometheus text."""
        for line in metrics_text.split("\n"):
            if line.startswith(name + " "):
                try:
                    return float(line.split()[-1])
                except (ValueError, IndexError):
                    pass
        return 0.0

    async def check_once(self):
        """Run one monitoring check cycle."""
        health = await self._fetch_health()

        # Gateway unreachable
        if health is None:
            self.state.consecutive_failures += 1
            if self.state.consecutive_failures >= 3 and not self.state.gateway_was_down:
                self.state.gateway_was_down = True
                self._send_alert(
                    f"Gateway UNREACHABLE at {self.gateway_url} "
                    f"({self.state.consecutive_failures} consecutive failures)",
                    "CRIT",
                )
            return

        # Gateway recovered
        if self.state.gateway_was_down:
            self.state.gateway_was_down = False
            self.state.consecutive_failures = 0
            self._send_alert(f"Gateway recovered at {self.gateway_url}", "OK")

        self.state.consecutive_failures = 0

        miners_alive = health.get("miners_alive", 0)
        miners_total = health.get("miners_total", 0)
        epoch = health.get("epoch", 0)
        challenges = health.get("challenges", {})

        log.info(
            f"Poll OK: miners={miners_alive}/{miners_total} epoch={epoch} "
            f"challenges={challenges.get('passed', 0)}/{challenges.get('total', 0)}"
        )

        # Store snapshot
        self.state.history.append({
            "time": time.time(),
            "miners_alive": miners_alive,
            "miners_total": miners_total,
            "epoch": epoch,
            "organic": health.get("total_organic", 0),
            "synthetic": health.get("total_synthetic", 0),
            "challenges_passed": challenges.get("passed", 0),
            "challenges_failed": challenges.get("failed", 0),
        })
        # Keep last 100 snapshots
        if len(self.state.history) > 100:
            self.state.history = self.state.history[-100:]

        # ── Alert: Miner dropped ──
        if self.state.last_miners_alive >= 0:
            if miners_alive < self.state.last_miners_alive:
                dropped = self.state.last_miners_alive - miners_alive
                self._send_alert(
                    f"Miner(s) dropped: {miners_alive}/{miners_total} alive "
                    f"(was {self.state.last_miners_alive}), {dropped} lost",
                    "WARN",
                )

            # Alert: all miners down
            if miners_alive == 0 and miners_total > 0:
                self._send_alert(
                    f"ALL MINERS DOWN: 0/{miners_total} alive!",
                    "CRIT",
                )

        self.state.last_miners_alive = miners_alive
        self.state.last_miners_total = miners_total

        # ── Alert: Challenge failures spike ──
        failed = challenges.get("failed", 0)
        if failed > self.state.last_challenge_fail_count:
            new_fails = failed - self.state.last_challenge_fail_count
            total = challenges.get("total", 0)
            if total > 0:
                fail_rate = failed / total
                if fail_rate > 0.1 and new_fails >= 3:  # >10% fail rate with 3+ new
                    self._send_alert(
                        f"Challenge failure spike: {failed}/{total} failed "
                        f"({fail_rate:.0%}), {new_fails} new since last check",
                        "WARN",
                    )
        self.state.last_challenge_fail_count = failed

        # ── Alert: Latency degradation (from per-miner details) ──
        miners_detail = health.get("miners_detail", [])
        now = time.time()
        for m in miners_detail:
            uid = m.get("uid", "?")
            alive = m.get("alive", True)
            ttft = m.get("avg_ttft_ms", 0)
            tps = m.get("avg_tps", 0)
            reliability = m.get("reliability", 1.0)

            # Skip dead miners — they're already tracked by the miner-drop alert
            if not alive:
                continue

            # Per-miner cooldown to avoid spamming Telegram
            last_miner_alert = self.state.per_miner_alert_time.get(uid, 0)
            miner_can_alert = (now - last_miner_alert) >= self.state.alert_cooldown_s

            if ttft > 2000 and miner_can_alert:  # TTFT > 2 seconds is bad
                self.state.per_miner_alert_time[uid] = now
                self._send_alert(
                    f"Miner {uid}: high TTFT={ttft:.0f}ms (>2s threshold)",
                    "WARN",
                )
            if reliability < 0.5 and miner_can_alert:  # Below 50% reliability
                self.state.per_miner_alert_time[uid] = now
                self._send_alert(
                    f"Miner {uid}: low reliability={reliability:.0%} (<50%)",
                    "WARN",
                )

        # ── Fetch and check Prometheus metrics ──
        metrics = await self._fetch_metrics()
        if metrics:
            timeouts = self._parse_metric(metrics, "gateway_error_timeouts")
            miner_errors = self._parse_metric(metrics, "gateway_error_miner_errors")
            total_errors = timeouts + miner_errors
            if total_errors > self.state.last_error_count:
                new_errors = total_errors - self.state.last_error_count
                if new_errors >= 5:
                    self._send_alert(
                        f"Error spike: {new_errors} new errors "
                        f"(timeouts={timeouts:.0f}, miner_errors={miner_errors:.0f})",
                        "WARN",
                    )
            self.state.last_error_count = total_errors

    async def run(self):
        """Main monitoring loop."""
        log.info(f"Starting gateway monitor")
        log.info(f"  Gateway: {self.gateway_url}")
        log.info(f"  Poll interval: {self.poll_interval}s")
        log.info(f"  Alert cooldown: {self.state.alert_cooldown_s}s")
        if self.arbos_dir:
            log.info(f"  Alerts via: arbos.py ({self.arbos_dir})")
        elif self.bot_token:
            log.info(f"  Alerts via: direct Telegram")
        else:
            log.info(f"  Alerts via: log only (no Telegram configured)")

        while True:
            try:
                await self.check_once()
            except Exception as e:
                log.error(f"Monitor check error: {e}", exc_info=True)
            await asyncio.sleep(self.poll_interval)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    def get_status(self) -> dict:
        """Return current monitoring status for API/dashboard."""
        recent = self.state.history[-1] if self.state.history else {}
        return {
            "gateway_url": self.gateway_url,
            "last_check": recent.get("time", 0),
            "miners_alive": self.state.last_miners_alive,
            "miners_total": self.state.last_miners_total,
            "consecutive_failures": self.state.consecutive_failures,
            "gateway_down": self.state.gateway_was_down,
            "history_length": len(self.state.history),
        }


def _load_arbos_env():
    """Load arbos encrypted env if available."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("arbos", "/Arbos/arbos.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod._init_env()
    except Exception as e:
        log.debug(f"Could not load arbos env: {e}")


def main():
    _load_arbos_env()
    parser = argparse.ArgumentParser(description="Gateway Monitor & Alert System")
    parser.add_argument(
        "--gateway",
        default="http://127.0.0.1:8081",
        help="Gateway URL to monitor",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=30.0,
        help="Poll interval in seconds (default: 30)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=300.0,
        help="Alert cooldown in seconds (default: 300)",
    )
    parser.add_argument(
        "--arbos-dir",
        default=None,
        help="Path to Arbos directory (for Telegram via arbos.py send)",
    )
    parser.add_argument(
        "--bot-token",
        default=None,
        help="Direct Telegram bot token",
    )
    parser.add_argument(
        "--chat-id",
        default=None,
        help="Direct Telegram chat ID",
    )
    args = parser.parse_args()

    monitor = GatewayMonitor(
        gateway_url=args.gateway,
        arbos_dir=args.arbos_dir,
        bot_token=args.bot_token,
        chat_id=args.chat_id,
        poll_interval=args.interval,
        alert_cooldown=args.cooldown,
    )

    try:
        asyncio.run(monitor.run())
    except KeyboardInterrupt:
        log.info("Monitor stopped")
    finally:
        asyncio.run(monitor.close())


if __name__ == "__main__":
    main()
