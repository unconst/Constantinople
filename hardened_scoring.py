#!/usr/bin/env python3
"""
Hardened Scoring Engine — Exploit-resistant miner evaluation.

Attack vectors defended against:
1. Selective honesty (organic vs synthetic performance divergence)
2. Score manipulation via edge cases in formulas
3. Latency gaming (fast on challenges, slow on organic)
4. Collusion detection via statistical correlation
5. Sybil resistance via diminishing returns per hotkey cluster
6. Three-state challenge tracking (pass/fail/unchallenged) to prevent score inflation
7. Minimum sample requirements to prevent early-exit gaming

Design principle: Every formula is designed so that the optimal strategy
for a rational miner is to serve fast, honest inference.
"""

import asyncio
import logging
import math
import secrets
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("hardened_scoring")


# ── Constants ────────────────────────────────────────────────────────────────

# Scoring weights
SPEED_WEIGHT = 0.40       # 40% for speed (TTFT + throughput)
VERIFICATION_WEIGHT = 0.40  # 40% for hidden state verification
CONSISTENCY_WEIGHT = 0.20  # 20% for consistency across requests

# Divergence detection
DIVERGENCE_THRESHOLD = 0.12       # 12% gap triggers investigation
DIVERGENCE_PENALTY_MILD = 0.30    # -30% weight for mild divergence
DIVERGENCE_PENALTY_SEVERE = 0.70  # -70% weight for severe divergence (>25%)
MIN_ORGANIC_SAMPLES = 5           # Need this many organic scores
MIN_SYNTHETIC_SAMPLES = 5         # Need this many synthetic scores

# Challenge verification
COSINE_THRESHOLD = 0.995          # Strict cosine threshold
CHALLENGE_TIMEOUT_MS = 50         # Max 50ms for challenge response (soft, penalized)
CHALLENGE_TIMEOUT_HARD_MS = 500   # Hard cutoff — auto-fail above this

# Speed scoring (relative to population)
TTFT_EXCELLENT_MS = 30
TTFT_POOR_MS = 500
TPS_EXCELLENT = 150
TPS_POOR = 10

# Anti-gaming
MAX_POINTS_PER_REQUEST = 1.0      # Cap to prevent any single request from dominating
MIN_REQUESTS_FOR_WEIGHT = 10      # Minimum requests to receive any weight (prevents last-second gaming)
CHALLENGE_FAIL_STRIKE_MULTIPLIER = 3.0  # Failing costs 3x what passing earns
MAX_CONSECUTIVE_FAILS = 3         # 3 consecutive fails → miner marked suspect

# Rate limiting
MAX_REQUESTS_PER_MINER_PER_EPOCH = 10000  # Prevent flooding

# Epoch
DEFAULT_EPOCH_LENGTH_S = 4320     # ~72 minutes (360 blocks * 12s)


@dataclass
class ChallengeResult:
    """Result of a single hidden state challenge."""
    request_id: str
    layer_index: int
    token_index: int
    cosine_sim: float
    latency_ms: float
    passed: bool
    reason: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class RequestScore:
    """Score for a single request."""
    request_id: str
    miner_uid: int
    timestamp: float
    is_synthetic: bool
    speed_score: float
    verification_score: float
    quality_score: float
    ttft_ms: float
    tokens_per_sec: float
    cosine_sim: float
    challenge_latency_ms: float
    challenge_passed: bool = None  # None = no challenge performed, True/False = challenge result

    @property
    def points(self) -> float:
        raw = self.speed_score * self.verification_score * self.quality_score
        if not math.isfinite(raw):
            return 0.0
        return min(raw, MAX_POINTS_PER_REQUEST)


def compute_verification_score(challenge_passed, cosine_sim: float,
                                challenge_latency_ms: float) -> float:
    """
    Gradient verification score using cosine similarity and challenge latency.

    Instead of binary 0/1, provides a gradient that:
    - Rewards near-perfect cosine matches (>0.999) with full credit
    - Penalizes marginal passes (0.995-0.999) proportionally
    - Gives a latency bonus for fast challenge responses
    - Returns 0 for failed challenges (below COSINE_THRESHOLD)
    - Returns 0.7 (floor) for unchallenged requests (challenge_passed=None)

    This prevents miners from barely passing challenges with minimum effort.
    """
    if challenge_passed is None:
        # No challenge performed — discount to 0.5 (below passing-challenge floor of 0.7).
        # This makes unchallenged requests contribute LESS than challenged-and-passed ones,
        # preventing miners from free-riding on unchallenged volume.
        return 0.5
    if challenge_passed is False:
        return 0.0

    # Cosine gradient: map [COSINE_THRESHOLD..1.0] → [0.7..1.0]
    # Miners with cos=0.995 get 0.7, cos=1.0 gets 1.0
    cos_range = 1.0 - COSINE_THRESHOLD  # 0.005
    cos_position = max(0.0, min(1.0, (cosine_sim - COSINE_THRESHOLD) / max(cos_range, 1e-9)))
    cos_factor = 0.7 + 0.3 * cos_position

    # Latency bonus: fast challenge responses get a small boost
    # <50ms → 1.0, 50-500ms → linear decay to 0.85
    if challenge_latency_ms <= CHALLENGE_TIMEOUT_MS:
        latency_factor = 1.0
    elif challenge_latency_ms <= CHALLENGE_TIMEOUT_HARD_MS:
        latency_factor = 1.0 - 0.15 * (
            (challenge_latency_ms - CHALLENGE_TIMEOUT_MS)
            / (CHALLENGE_TIMEOUT_HARD_MS - CHALLENGE_TIMEOUT_MS)
        )
    else:
        latency_factor = 0.85

    return cos_factor * latency_factor


@dataclass
class MinerEpochStats:
    """Hardened per-miner stats for an epoch."""
    uid: int
    total_points: float = 0.0
    penalty_points: float = 0.0  # Points deducted for failures
    organic_scores: list = field(default_factory=list)
    synthetic_scores: list = field(default_factory=list)
    organic_latencies: list = field(default_factory=list)
    synthetic_latencies: list = field(default_factory=list)
    challenge_results: list = field(default_factory=list)
    total_requests: int = 0
    passed_challenges: int = 0
    failed_challenges: int = 0
    consecutive_fails: int = 0
    is_suspect: bool = False
    ttft_values: list = field(default_factory=list)
    tps_values: list = field(default_factory=list)
    cosine_values: list = field(default_factory=list)

    @property
    def net_points(self) -> float:
        """Points after deducting penalties. Can go negative — miners with
        negative net_points are excluded from weight calculation entirely."""
        return self.total_points - self.penalty_points

    @property
    def recent_pass_rate(self) -> float:
        """Pass rate over the most recent challenges only (last 20).
        Prevents goodwill banking where a miner passes 100 challenges then
        starts cheating — the recent window catches the change quickly."""
        if not self.challenge_results:
            return self.pass_rate  # Fall back to overall
        recent = self.challenge_results[-20:]
        passed = sum(1 for r in recent if r.passed)
        return passed / len(recent)

    @property
    def organic_mean(self) -> float:
        return sum(self.organic_scores) / len(self.organic_scores) if self.organic_scores else 0.0

    @property
    def synthetic_mean(self) -> float:
        return sum(self.synthetic_scores) / len(self.synthetic_scores) if self.synthetic_scores else 0.0

    @property
    def organic_std(self) -> float:
        if len(self.organic_scores) < 2:
            return 0.0
        return float(np.std(self.organic_scores))

    @property
    def synthetic_std(self) -> float:
        if len(self.synthetic_scores) < 2:
            return 0.0
        return float(np.std(self.synthetic_scores))

    @property
    def divergence(self) -> float:
        """
        Robust divergence metric. Uses both mean and distribution comparison.
        Returns 0 if insufficient data (favors the miner — innocent until proven guilty).
        """
        if len(self.organic_scores) < MIN_ORGANIC_SAMPLES or len(self.synthetic_scores) < MIN_SYNTHETIC_SAMPLES:
            return 0.0

        org_mean = self.organic_mean
        syn_mean = self.synthetic_mean

        if syn_mean == 0 and org_mean == 0:
            return 0.0
        if syn_mean == 0:
            return 1.0  # All synthetic scores are 0 but organic aren't — suspicious

        # Mean-based divergence
        mean_div = abs(org_mean - syn_mean) / max(syn_mean, 0.001)

        # Also check latency divergence (miners gaming speed on synthetics)
        latency_div = 0.0
        if self.organic_latencies and self.synthetic_latencies:
            org_lat = np.median(self.organic_latencies)
            syn_lat = np.median(self.synthetic_latencies)
            if syn_lat > 0:
                latency_div = abs(org_lat - syn_lat) / max(syn_lat, 1.0)

        # Combined divergence: max of mean and latency divergence
        return max(mean_div, latency_div)

    @property
    def pass_rate(self) -> float:
        total = self.passed_challenges + self.failed_challenges
        return self.passed_challenges / max(total, 1)

    @property
    def avg_ttft_ms(self) -> float:
        return sum(self.ttft_values) / len(self.ttft_values) if self.ttft_values else 0.0

    @property
    def avg_tps(self) -> float:
        return sum(self.tps_values) / len(self.tps_values) if self.tps_values else 0.0

    @property
    def avg_cosine(self) -> float:
        return sum(self.cosine_values) / len(self.cosine_values) if self.cosine_values else 0.0

    @property
    def consistency_score(self) -> float:
        """
        Measure how consistent the miner's performance is.
        Inconsistent performance suggests gaming or unreliable service.
        Suspiciously LOW variance also penalized with a GRADIENT —
        real GPU inference always has natural jitter (cv typically 0.05-0.15).

        Gradient low-variance penalty prevents miners from injecting just
        enough jitter (cv≈0.03) to clear a binary threshold while still
        being suspiciously uniform.

        Returns 0-1 where 1 = consistent with natural variance.
        """
        if len(self.tps_values) < 3:
            return 0.5  # Not enough data, neutral
        cv = np.std(self.tps_values) / max(np.mean(self.tps_values), 0.001)
        # High variance penalty: cv=0 → 1.0, cv≥1 → 0.0
        high_var_score = max(0.0, 1.0 - cv)
        # Gradient low-variance penalty: real GPU inference has CV typically
        # in the 0.05-0.15 range. Suspiciously low CV is penalized on a
        # gradient so miners can't circumvent with minimal jitter injection.
        #   cv=0.00 → 0.3 (heavily penalized)
        #   cv=0.02 → 0.5 (still suspicious)
        #   cv=0.05 → 0.8 (marginal, blends with natural variance)
        #   cv≥0.08 → 1.0 (no low-variance penalty)
        LOW_CV_FLOOR = 0.08  # Below this, apply gradient penalty
        if cv < LOW_CV_FLOOR:
            low_var_factor = 0.3 + 0.7 * (cv / LOW_CV_FLOOR)
        else:
            low_var_factor = 1.0
        return high_var_score * low_var_factor


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Returns 0 for zero/NaN/Inf vectors."""
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return 0.0
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    result = float(dot / (norm_a * norm_b))
    if not math.isfinite(result):
        return 0.0
    return result


def compute_output_quality(text: str, expected_min_tokens: int = 4) -> float:
    """
    Score output quality. Detects garbage, repetition, and degenerate outputs.

    Returns 0.0-1.0 where 1.0 = high quality output.

    Checks:
    1. Non-empty and meets minimum length
    2. Repetition ratio (repeated n-grams signal degenerate decoding)
    3. Character entropy (gibberish has abnormal entropy)
    """
    if not text or not text.strip():
        return 0.0

    words = text.split()
    if len(words) < expected_min_tokens:
        # Very short output — scale linearly from 0.05 to threshold
        # 1 word = ~0.05, 2 words = ~0.25 for expected_min=4
        return max(0.05, len(words) / (expected_min_tokens * 2))

    # --- Repetition detection ---
    # Check trigram repetition: what fraction of trigrams are unique?
    if len(words) >= 6:
        trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
        unique_ratio = len(set(trigrams)) / len(trigrams)
        # unique_ratio < 0.3 → heavy repetition (looping)
        if unique_ratio < 0.2:
            return 0.1  # Severely repetitive
        repetition_score = min(1.0, unique_ratio / 0.5)  # 0.5+ unique → 1.0
    else:
        repetition_score = 1.0

    # --- Character entropy ---
    # Normal English text: ~4.0-4.5 bits/char. Gibberish or single-char spam diverges.
    char_counts = {}
    clean = text.strip().lower()
    for c in clean:
        char_counts[c] = char_counts.get(c, 0) + 1
    total = len(clean)
    entropy = 0.0
    for count in char_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    # Very low entropy (< 2.0) → single char repeated or very uniform
    # Very high entropy (> 6.0) → random bytes / base64 etc
    if entropy < 1.5:
        entropy_score = 0.2
    elif entropy < 2.5:
        entropy_score = 0.6
    elif entropy > 6.5:
        entropy_score = 0.5
    else:
        entropy_score = 1.0

    return repetition_score * entropy_score


def compute_speed_score(ttft_ms: float, tokens_per_sec: float,
                        population_ttft: list[float] = None,
                        population_tps: list[float] = None,
                        miner_medians_ttft: list[float] = None,
                        miner_medians_tps: list[float] = None) -> float:
    """
    Score miner speed. Uses population-relative scoring when available,
    falls back to absolute thresholds.

    Sybil-resistant: when miner_medians_* are provided, ranks against
    per-miner median values instead of raw request pools. This prevents
    Sybil miners from flooding the population with fabricated values to
    shift the distribution. Each miner UID contributes exactly one data
    point regardless of request volume.
    """
    # Prefer per-miner medians (Sybil-resistant) over raw request pool
    pool_ttft = miner_medians_ttft if miner_medians_ttft and len(miner_medians_ttft) >= 3 else population_ttft
    pool_tps = miner_medians_tps if miner_medians_tps and len(miner_medians_tps) >= 3 else population_tps

    if pool_ttft and len(pool_ttft) >= 3:
        ttft_rank = sum(1 for t in pool_ttft if t >= ttft_ms) / len(pool_ttft)
        tps_rank = sum(1 for t in pool_tps if t <= tokens_per_sec) / len(pool_tps)
    else:
        # Absolute fallback
        ttft_rank = max(0.0, min(1.0, 1.0 - (ttft_ms - TTFT_EXCELLENT_MS) / (TTFT_POOR_MS - TTFT_EXCELLENT_MS)))
        tps_rank = max(0.0, min(1.0, (tokens_per_sec - TPS_POOR) / (TPS_EXCELLENT - TPS_POOR)))

    return 0.4 * ttft_rank + 0.6 * tps_rank


class HardenedScoringEngine:
    """
    Exploit-resistant scoring engine.

    Key hardening features:
    1. Asymmetric penalties: failing a challenge costs more than passing earns
    2. Consecutive failure tracking with suspect flagging
    3. Population-relative speed scoring (can't game absolute thresholds)
    4. Dual divergence detection (score + latency)
    5. Minimum sample requirements
    6. Net points (gross - penalties) prevents "win some lose some" strategies
    7. Request rate limiting per miner
    8. Consistency scoring rewards reliable service
    """

    # Max samples to keep for population-relative scoring per epoch
    _MAX_POPULATION_SAMPLES = 5000
    # Max request log entries per epoch (older entries discarded)
    _MAX_REQUEST_LOG = 2000
    # Max weight history entries to retain
    _MAX_WEIGHT_HISTORY = 100

    def __init__(self, epoch_length_s: float = DEFAULT_EPOCH_LENGTH_S):
        self.epoch_length_s = epoch_length_s
        self.current_epoch_start = time.time()
        self._current_epoch_target_s = self._randomize_epoch_length()
        self.epoch_number = 0
        self.miner_stats: dict[int, MinerEpochStats] = {}
        self.request_log: deque[RequestScore] = deque(maxlen=self._MAX_REQUEST_LOG)
        self.weight_history: deque[dict] = deque(maxlen=self._MAX_WEIGHT_HISTORY)
        self._epoch_lock = asyncio.Lock()  # Protects miner_stats during epoch rollover

        # Population-level metrics for relative scoring (bounded)
        self._population_ttft: deque[float] = deque(maxlen=self._MAX_POPULATION_SAMPLES)
        self._population_tps: deque[float] = deque(maxlen=self._MAX_POPULATION_SAMPLES)

        # Cross-epoch suspect tracking: uid → number of epochs flagged suspect
        self._suspect_history: dict[int, int] = {}

        # Hotkey-based suspect tracking: survives UID re-registrations.
        # hotkey → suspect count. When a miner registers with a known hotkey,
        # its suspect history carries over even if the UID changes.
        self._hotkey_suspect_history: dict[str, int] = {}

        # UID ↔ hotkey mapping for current epoch
        self._uid_to_hotkey: dict[int, str] = {}

        # Cross-epoch divergence tracking: accumulates organic/synthetic score samples
        # across epochs so miners can't reset detection by staying under per-epoch minimums.
        # uid → {"organic": deque, "synthetic": deque}
        self._cross_epoch_scores: dict[int, dict[str, deque]] = {}

    def _get_stats(self, uid: int) -> MinerEpochStats:
        if uid not in self.miner_stats:
            self.miner_stats[uid] = MinerEpochStats(uid=uid)
        return self.miner_stats[uid]

    def register_hotkey(self, uid: int, hotkey: str):
        """Register a UID↔hotkey mapping. Must be called when miners are discovered.
        Transfers any hotkey-based suspect history to the UID-based tracker so
        re-registered miners inherit their penalty history."""
        self._uid_to_hotkey[uid] = hotkey
        if hotkey in self._hotkey_suspect_history:
            # Transfer hotkey history to UID history — re-registering doesn't reset
            existing_uid_history = self._suspect_history.get(uid, 0)
            hotkey_history = self._hotkey_suspect_history[hotkey]
            self._suspect_history[uid] = max(existing_uid_history, hotkey_history)

    def _randomize_epoch_length(self) -> float:
        """Add ±20% jitter to epoch length so miners cannot predict boundaries."""
        jitter_range = int(self.epoch_length_s * 0.2)
        if jitter_range < 1:
            return self.epoch_length_s
        offset = secrets.randbelow(2 * jitter_range + 1) - jitter_range
        return self.epoch_length_s + offset

    def get_miner_medians(self) -> tuple[list[float], list[float]]:
        """
        Compute per-miner median TTFT and TPS for Sybil-resistant population ranking.

        Each miner UID contributes exactly one data point (its median), regardless
        of how many requests it has served. This prevents Sybil miners from flooding
        the population pool with fabricated values to shift the distribution.
        """
        medians_ttft = []
        medians_tps = []
        for uid, stats in self.miner_stats.items():
            if stats.ttft_values:
                medians_ttft.append(float(np.median(stats.ttft_values)))
            if stats.tps_values:
                medians_tps.append(float(np.median(stats.tps_values)))
        return medians_ttft, medians_tps

    def record_request(self, score: RequestScore):
        """Record a scored request with hardened accounting."""
        stats = self._get_stats(score.miner_uid)

        # Rate limiting
        if stats.total_requests >= MAX_REQUESTS_PER_MINER_PER_EPOCH:
            log.warning(f"Miner {score.miner_uid}: rate limited (>{MAX_REQUESTS_PER_MINER_PER_EPOCH} requests)")
            return

        # Record score
        if score.is_synthetic:
            stats.synthetic_scores.append(score.points)
            stats.synthetic_latencies.append(score.ttft_ms)
        else:
            stats.organic_scores.append(score.points)
            stats.organic_latencies.append(score.ttft_ms)

        # Challenge tracking — three states:
        #   None  = no challenge performed (cross-probes, unchallenged requests)
        #   True  = challenge passed
        #   False = challenge failed
        if score.challenge_passed is True:
            stats.passed_challenges += 1
            stats.consecutive_fails = 0
            stats.total_points += score.points
            stats.challenge_results.append(ChallengeResult(
                request_id=score.request_id, layer_index=0, token_index=0,
                cosine_sim=score.cosine_sim, latency_ms=score.challenge_latency_ms,
                passed=True, reason="pass",
            ))
        elif score.challenge_passed is False:
            stats.failed_challenges += 1
            stats.consecutive_fails += 1
            stats.challenge_results.append(ChallengeResult(
                request_id=score.request_id, layer_index=0, token_index=0,
                cosine_sim=score.cosine_sim, latency_ms=score.challenge_latency_ms,
                passed=False, reason="fail",
            ))
            # Asymmetric penalty: failing costs MORE than passing earns.
            # Use speed_score as the base (NOT quality_score — garbage output miners
            # shouldn't get LOWER penalties than honest miners).
            # Floor of 0.3 ensures even slow/garbage miners face meaningful penalties.
            penalty_base = max(score.speed_score, 0.3)
            stats.penalty_points += penalty_base * CHALLENGE_FAIL_STRIKE_MULTIPLIER

            if stats.consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                stats.is_suspect = True
                log.warning(f"Miner {score.miner_uid}: SUSPECT — {stats.consecutive_fails} consecutive challenge failures")
        else:
            # No challenge performed — add points but don't affect challenge stats
            stats.total_points += score.points

        stats.total_requests += 1
        stats.ttft_values.append(score.ttft_ms)
        stats.tps_values.append(score.tokens_per_sec)
        # Only append cosine values when a challenge was actually performed
        # to prevent 0.0 from failed/unchallenged requests diluting avg_cosine
        if score.challenge_passed is not None and score.cosine_sim > 0.0:
            stats.cosine_values.append(score.cosine_sim)

        # Update population metrics
        self._population_ttft.append(score.ttft_ms)
        self._population_tps.append(score.tokens_per_sec)

        self.request_log.append(score)

        # Feed cross-epoch divergence tracker
        uid = score.miner_uid
        if uid not in self._cross_epoch_scores:
            self._cross_epoch_scores[uid] = {
                "organic": deque(maxlen=50),
                "synthetic": deque(maxlen=50),
            }
        bucket = "synthetic" if score.is_synthetic else "organic"
        self._cross_epoch_scores[uid][bucket].append(score.points)

    def record_challenge(self, miner_uid: int, result: ChallengeResult):
        """Record a standalone challenge result."""
        stats = self._get_stats(miner_uid)
        stats.challenge_results.append(result)

    def should_end_epoch(self) -> bool:
        return time.time() - self.current_epoch_start >= self._current_epoch_target_s

    def compute_weights(self) -> dict[int, float]:
        """Public API — compute weights from current live miner_stats."""
        return self._compute_weights_from(self.miner_stats)

    def _compute_weights_from(self, stats_dict: dict) -> dict[int, float]:
        """
        Compute final weights with layered anti-gaming defenses.

        Weight = net_points × consistency × divergence_factor × suspect_factor
        All normalized to sum to 1.0.

        Accepts an explicit stats_dict so end_epoch can pass a snapshot
        without racing with concurrent record_request() calls.
        """
        raw_weights: dict[int, float] = {}

        for uid, stats in stats_dict.items():
            # Minimum request threshold
            if stats.total_requests < MIN_REQUESTS_FOR_WEIGHT:
                log.debug(f"Miner {uid}: below minimum requests ({stats.total_requests}/{MIN_REQUESTS_FOR_WEIGHT})")
                continue

            # Start with net points
            weight = stats.net_points
            if weight <= 0:
                continue

            # Consistency bonus/penalty (0.5-1.5x multiplier)
            consistency = stats.consistency_score
            weight *= (0.5 + consistency)  # Range: 0.5x to 1.5x

            # Cosine fidelity bonus (0.85-1.1x multiplier)
            # Rewards miners with consistently high cosine similarity
            if stats.cosine_values:
                avg_cos = stats.avg_cosine
                # Map [COSINE_THRESHOLD..1.0] → [0.85..1.1]
                cos_range = 1.0 - COSINE_THRESHOLD
                cos_pos = min(1.0, max(0.0, (avg_cos - COSINE_THRESHOLD) / max(cos_range, 1e-9)))
                weight *= 0.85 + 0.25 * cos_pos

            # Divergence penalty (per-epoch)
            div = stats.divergence
            if div > 0.25:
                weight *= (1.0 - DIVERGENCE_PENALTY_SEVERE)
                log.warning(f"Miner {uid}: SEVERE divergence={div:.3f} → -70% weight")
            elif div > DIVERGENCE_THRESHOLD:
                weight *= (1.0 - DIVERGENCE_PENALTY_MILD)
                log.warning(f"Miner {uid}: mild divergence={div:.3f} → -30% weight")

            # Cross-epoch divergence: catches miners staying under per-epoch sample minimums
            cross = self._cross_epoch_scores.get(uid)
            if cross and len(cross["organic"]) >= MIN_ORGANIC_SAMPLES and len(cross["synthetic"]) >= MIN_SYNTHETIC_SAMPLES:
                cross_org_mean = sum(cross["organic"]) / len(cross["organic"])
                cross_syn_mean = sum(cross["synthetic"]) / len(cross["synthetic"])
                if cross_syn_mean > 0:
                    cross_div = abs(cross_org_mean - cross_syn_mean) / max(cross_syn_mean, 0.001)
                    if cross_div > 0.25:
                        weight *= (1.0 - DIVERGENCE_PENALTY_SEVERE)
                        log.warning(f"Miner {uid}: SEVERE cross-epoch divergence={cross_div:.3f} → -70% weight")
                    elif cross_div > DIVERGENCE_THRESHOLD:
                        weight *= (1.0 - DIVERGENCE_PENALTY_MILD)
                        log.warning(f"Miner {uid}: mild cross-epoch divergence={cross_div:.3f} → -30% weight")

            # Suspect penalty (current epoch)
            if stats.is_suspect:
                weight *= 0.1  # 90% penalty for suspected cheaters
                log.warning(f"Miner {uid}: SUSPECT → -90% weight")

            # Cross-epoch suspect history penalty
            prior_suspect_epochs = self._suspect_history.get(uid, 0)
            if prior_suspect_epochs > 0:
                # Each prior suspect epoch adds a 20% cumulative penalty (min 0.2x)
                history_factor = max(0.2, 1.0 - 0.2 * prior_suspect_epochs)
                weight *= history_factor
                log.warning(f"Miner {uid}: suspect history ({prior_suspect_epochs} epochs) → {history_factor:.1f}x weight")

            # Pass rate factor: smooth penalty using WORST of overall and recent.
            # recent_pass_rate uses the last 20 challenges, preventing goodwill
            # banking where a miner passes 100 challenges then starts cheating.
            # Uses quadratic scaling to avoid a hard cliff at 90%:
            #   pr=1.0 → 1.0x, pr=0.9 → 0.99x, pr=0.5 → 0.75x, pr=0.0 → 0.0x
            if stats.passed_challenges + stats.failed_challenges > 0:
                pr_overall = stats.pass_rate
                pr_recent = stats.recent_pass_rate
                pr = min(pr_overall, pr_recent)  # Use whichever is worse
                weight *= pr * (2 - pr)  # Quadratic: pr*(2-pr), concave, smooth

            # Challenge participation factor: penalize miners that have very few
            # (or zero) challenged requests relative to total (unchallenged free-riding)
            # Applies starting at 3 requests (not 10) to prevent short-lived Sybil
            # campaigns that serve <10 requests per UID to evade participation checks.
            total_challenged = stats.passed_challenges + stats.failed_challenges
            if stats.total_requests >= 3:
                if total_challenged == 0:
                    # Zero challenges = maximum participation penalty
                    weight *= 0.3
                    log.warning(f"Miner {uid}: zero challenges out of {stats.total_requests} requests → 0.3x weight")
                else:
                    challenge_ratio = total_challenged / stats.total_requests
                    # If < 10% of requests were challenged, apply a participation discount
                    # Floor at 0.3 (matches zero-challenge penalty) scaling to 1.0 at 10%
                    if challenge_ratio < 0.1:
                        weight *= max(0.3, challenge_ratio * 10)  # Scale 0→0.3, 0.1→1.0

            # Organic participation factor: penalize miners with suspiciously low
            # organic request counts relative to synthetic. Miners that avoid organic
            # traffic can evade divergence detection (which needs MIN_ORGANIC_SAMPLES).
            # A healthy miner should serve a comparable ratio of organic to synthetic.
            # This uses cross-epoch accumulated data to catch drip-feeders who stay
            # under per-epoch minimums.
            cross = self._cross_epoch_scores.get(uid)
            if cross:
                cross_org_count = len(cross["organic"])
                cross_syn_count = len(cross["synthetic"])
                cross_total = cross_org_count + cross_syn_count
                if cross_syn_count >= MIN_ORGANIC_SAMPLES and cross_org_count < MIN_ORGANIC_SAMPLES:
                    # Miner has enough synthetic samples but suspiciously few organic.
                    # Lowered from 10 to MIN_ORGANIC_SAMPLES to close the gap where
                    # a miner with 5-9 synthetic + <5 organic evaded both checks.
                    # Penalty: 0.6x for zero organic, scaling to 1.0x at MIN_ORGANIC_SAMPLES
                    organic_factor = 0.6 + 0.4 * (cross_org_count / max(MIN_ORGANIC_SAMPLES, 1))
                    weight *= organic_factor
                elif cross_total < MIN_ORGANIC_SAMPLES and stats.total_requests >= 3:
                    # Drip-feed evasion: miner keeps BOTH organic and synthetic counts
                    # below minimums to avoid all divergence checks. Lowered from 10 to
                    # 3 requests to match the challenge participation threshold.
                    drip_factor = 0.7 + 0.3 * (cross_total / max(MIN_ORGANIC_SAMPLES, 1))
                    weight *= drip_factor
                elif cross_total >= MIN_ORGANIC_SAMPLES and (cross_org_count < MIN_ORGANIC_SAMPLES or cross_syn_count < MIN_ORGANIC_SAMPLES):
                    # Imbalanced participation: enough total samples but insufficient
                    # diversity. A healthy miner should have both organic and synthetic.
                    # This catches the 9+4 or 4+9 gap that evaded the above conditions.
                    min_count = min(cross_org_count, cross_syn_count)
                    balance_factor = 0.7 + 0.3 * (min_count / max(MIN_ORGANIC_SAMPLES, 1))
                    weight *= balance_factor

            # Guard against NaN/Inf/subnormal contamination — a single bad value
            # would corrupt the entire normalization step. Subnormal floats
            # (< 1e-15) from chained penalty multipliers distort normalization.
            if not math.isfinite(weight) or weight < 1e-15:
                if weight != 0.0:
                    log.warning(f"Miner {uid}: non-finite or subnormal weight {weight} → 0.0")
                weight = 0.0
            raw_weights[uid] = weight

        # Normalize
        total = sum(raw_weights.values())
        if total == 0 or not math.isfinite(total):
            return {}

        return {uid: w / total for uid, w in raw_weights.items()}

    def end_epoch(self) -> dict:
        """End current epoch, compute weights, return summary.

        Atomically swaps miner_stats to a fresh dict before computing weights,
        so concurrent record_request() calls go to the new epoch's dict and
        don't corrupt the snapshot being scored.
        """
        # Snapshot and swap atomically (Python dict assignment is atomic at bytecode level)
        old_stats = self.miner_stats
        old_request_log = self.request_log
        old_population_ttft = list(self._population_ttft)
        old_population_tps = list(self._population_tps)

        # New epoch starts immediately — concurrent record_request() writes here
        self.miner_stats = {}
        self.request_log = deque(maxlen=self._MAX_REQUEST_LOG)
        self._population_ttft = deque(maxlen=self._MAX_POPULATION_SAMPLES)
        self._population_tps = deque(maxlen=self._MAX_POPULATION_SAMPLES)

        # Update suspect history BEFORE computing weights so that first-time
        # suspects get the history penalty in the same epoch they're flagged.
        # This closes the 1-epoch amnesty window where a newly-suspect miner
        # only got the current-epoch 0.1x penalty without the history multiplier.
        for uid, stats in old_stats.items():
            if stats.is_suspect:
                prior = self._suspect_history.get(uid, 0)
                increment = 3 * (2 ** min(prior // 3, 4))
                self._suspect_history[uid] = prior + increment
                hotkey = self._uid_to_hotkey.get(uid)
                if hotkey:
                    self._hotkey_suspect_history[hotkey] = self._suspect_history[uid]
            else:
                if uid in self._suspect_history:
                    self._suspect_history[uid] = max(0, self._suspect_history[uid] - 1)
                    if self._suspect_history[uid] == 0:
                        del self._suspect_history[uid]
                hotkey = self._uid_to_hotkey.get(uid)
                if hotkey and hotkey in self._hotkey_suspect_history:
                    self._hotkey_suspect_history[hotkey] = max(0, self._hotkey_suspect_history[hotkey] - 1)
                    if self._hotkey_suspect_history[hotkey] == 0:
                        del self._hotkey_suspect_history[hotkey]

        # Compute weights from the snapshot (no concurrent mutation possible)
        weights = self._compute_weights_from(old_stats)

        summary = {
            "epoch": self.epoch_number,
            "duration_s": time.time() - self.current_epoch_start,
            "total_requests": sum(s.total_requests for s in old_stats.values()),
            "miners": {},
            "weights": weights,
        }

        for uid, stats in old_stats.items():
            summary["miners"][uid] = {
                "total_points": stats.total_points,
                "penalty_points": stats.penalty_points,
                "net_points": stats.net_points,
                "total_requests": stats.total_requests,
                "organic_count": len(stats.organic_scores),
                "synthetic_count": len(stats.synthetic_scores),
                "organic_mean": stats.organic_mean,
                "synthetic_mean": stats.synthetic_mean,
                "divergence": stats.divergence,
                "consistency": stats.consistency_score,
                "passed_challenges": stats.passed_challenges,
                "failed_challenges": stats.failed_challenges,
                "pass_rate": stats.pass_rate,
                "is_suspect": stats.is_suspect,
                "avg_ttft_ms": stats.avg_ttft_ms,
                "avg_tps": stats.avg_tps,
                "avg_cosine": stats.avg_cosine,
                "weight": weights.get(uid, 0.0),
            }

        self.weight_history.append(summary)

        # Suspect history already updated above (before weight computation).
        # Advance epoch counter (miner_stats already swapped at top of method)
        self.epoch_number += 1
        self.current_epoch_start = time.time()
        self._current_epoch_target_s = self._randomize_epoch_length()

        log.info(
            f"Epoch {summary['epoch']} complete: "
            f"{summary['total_requests']} requests, "
            f"{len(weights)} miners weighted"
        )

        return summary

    def get_scoreboard(self) -> list[dict]:
        """Get current epoch scoreboard."""
        board = []
        for uid, stats in self.miner_stats.items():
            board.append({
                "uid": uid,
                "net_points": stats.net_points,
                "total_points": stats.total_points,
                "penalty_points": stats.penalty_points,
                "requests": stats.total_requests,
                "organic": len(stats.organic_scores),
                "synthetic": len(stats.synthetic_scores),
                "divergence": stats.divergence,
                "consistency": stats.consistency_score,
                "pass_rate": stats.pass_rate,
                "is_suspect": stats.is_suspect,
                "avg_ttft_ms": stats.avg_ttft_ms,
                "avg_tps": stats.avg_tps,
            })
        return sorted(board, key=lambda x: x["net_points"], reverse=True)

