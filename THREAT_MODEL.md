# Inference Subnet — Threat Model & Attack Analysis

## Overview

This document analyzes all known attack vectors for the inference subnet
and how the hardened validator defends against each.

## Attack Vectors

### 1. Selective Honesty (Challenge Detection)

**Attack**: Miner detects which requests are validator challenges vs organic
traffic, and only performs well on challenges.

**How miner might detect challenges:**
- Different prompt patterns (synthetic vs real user)
- Timing patterns (challenges at regular intervals)
- Request source IP identification
- Challenge format differs from organic

**Defenses (implemented):**
- Synthetic prompts are curated to match organic patterns
- 30% of synthetics reuse actual recent organic prompts (identical content)
- Timing jitter: ±50% on synthetic intervals, cryptographic randomness
- All traffic goes through same gateway endpoint
- Dual divergence detection: if miner performs differently on organic vs
  synthetic (score OR latency), they get penalized

**Residual risk**: LOW — reusing organic prompts with randomized timing
makes detection impractical.

### 2. Hidden State Spoofing

**Attack**: Miner pre-computes or caches hidden states without running
the real model.

**Subvariants:**
- Pre-compute for common prompts
- Run a smaller/quantized model
- Proxy to an API service

**Defenses (implemented):**
- Hidden states are model-specific AND input-specific
- Cosine threshold 0.995 catches quantized/different models
- Timing requirement (<50ms) proves states are in GPU VRAM
- Multi-point challenges: 20% of challenges check 3+ (layer,position)
  pairs — pre-computing all combinations is exponentially expensive
- Challenge parameters use `secrets` module (crypto-random)
- Each challenge has a unique nonce preventing replay

**Residual risk**: VERY LOW — would require running the exact model
anyway, making cheating more expensive than honesty.

### 3. Latency Gaming

**Attack**: Miner is fast on benchmark/challenges but slow on organic traffic.

**Defenses (implemented):**
- Dual divergence detection includes latency comparison
  (organic_latency vs synthetic_latency)
- Consistency scoring: coefficient of variation of TPS
  rewards steady performance, penalizes spiky behavior
- Speed scoring is population-relative (ranked against all miners)
  so gaming absolute thresholds doesn't work

**Residual risk**: LOW — latency divergence detection catches this.

### 4. Score Manipulation

**Attack**: Exploit edge cases in scoring formulas to inflate points.

**Subvariants:**
- Flood requests to accumulate more points
- Game the speed scoring thresholds
- Exploit insufficient divergence data windows

**Defenses (implemented):**
- Points capped at 1.0 per request
- Rate limiting: 10,000 requests per miner per epoch max
- Minimum 3 requests to receive any weight
- Minimum 5 organic + 5 synthetic samples for divergence
- Population-relative speed scoring (no fixed thresholds to game)
- Asymmetric penalties: failing a challenge costs 3x what passing earns
- Net points = total - penalties (can't "win some lose some")

**Residual risk**: LOW — multiple overlapping caps and minimums.

### 5. KV Cache Cheating

**Attack**: Claim KV cache hits without actually maintaining the cache.

**Defenses (implemented):**
- Hidden state challenges verify the model was actually run
- Challenge timing proves states are in VRAM (<50ms)
- Session-aware routing sends multi-turn conversations to same miner,
  then verifies the cache is real
- Cache hits from previous turns can be challenged at any layer/position

**Additional defenses (implemented — Phase 2):**
- **KV Cache Prober** (`kv_cache_prober.py`):
  - Sends turn-1 (long prefix), waits 2-8s (randomized), sends turn-2 (short continuation)
  - Measures TTFT ratio (turn2/turn1) — real cache yields ratio < 0.7
  - Hidden state challenge on turn-2 proves model actually ran
  - Median aggregation across 3+ probes per epoch (robust to outliers)
  - Score feeds into weight calculation as multiplier (±15%)
  - Probe timing and content use crypto randomness (indistinguishable from organic)

**Residual risk**: LOW — TTFT ratio measurement combined with hidden state
verification on the continuation turn makes faking cache hits unprofitable.

### 6. Collusion

**Attack**: Miners coordinating to share work or answers.

**Subvariants:**
- Two miners sharing hidden states over network
- One miner fronting for another's compute

**Defenses (implemented):**
- 50ms timing requirement makes network-shared states impractical
  (network round-trip typically >5ms, plus serialization)
- Each challenge has a unique nonce (can't pre-share answers)
- Multi-point challenges require multiple states per request

**Additional defenses (implemented — Phase 2):**
- **Collusion Detector** (`collusion_detector.py`):
  - Response fingerprinting: LCS-based token similarity between miners sent same prompt
  - Timing correlation: Pearson correlation of binned latencies across miners
  - Error correlation: Jaccard similarity of failure time windows
  - Hidden state exactness: detects bit-exact matches (cosine > 0.9999) between
    miners (same model + different GPU should have FP variation)
  - 4 orthogonal signals weighted: response 35%, timing 30%, error 15%, exact 20%
  - Minimum sample requirements prevent false positives
  - Flagged pairs get 30-70% weight penalty on both miners

**Residual risk**: LOW — four independent detection methods make collusion
extremely difficult to sustain without detection across an epoch.

### 7. Resource Misrepresentation

**Attack**: Claiming more GPU memory/compute than actually available.

**Defenses (implemented):**
- Speed is measured empirically (TTFT, TPS) not self-reported
- Population-relative scoring means only actual performance matters
- Consistency scoring catches intermittent performance (sharing GPU)

**Future enhancement:**
- GPU fingerprinting: challenge that requires specific VRAM amount
- Concurrent load testing: send multiple requests simultaneously

**Residual risk**: LOW — empirical measurement is the gold standard.

### 8. Optimization Penalty — RESOLVED

**Problem**: Hidden-state verification previously required near-exact cosine match
(>0.995), effectively banning speculative decoding, quantization, TP scaling, and
KV compression.

**Solution (IMPLEMENTED)**: Tiered cosine verification in hardened_scoring.py:
- cos >= 0.99: full credit (1.0) — standard float16/float32 inference
- cos >= 0.90: high credit (~0.90) — quantized (INT8/FP8) or TP-sharded
- cos >= 0.70: base credit (~0.75) — speculative decoding, aggressive optimization
- cos < 0.70: fail (0.0) — cheaters / wrong model (typically produce < 0.3)

The speed scoring (40% weight) means optimized miners running 2x faster with
quantization (score 0.90) will outperform standard miners (score 1.0) because
the speed bonus more than compensates for the small verification penalty.

COSINE_THRESHOLD lowered from 0.995 to 0.70 in both hardened_scoring.py and
challenge_engine.py. Void range (0.2-0.7 shallow, 0.01-0.7 deep) handles
computational divergence without penalizing miners.

**Residual risk**: LOW — tiered system allows optimization while catching cheaters.

### 9. Centralization Pressure

**Problem**: Speed scoring (40% weight) + expensive validation creates incentives
for a single dominant validator, causing miners to co-locate geographically for
latency advantages. This recreates past subnet centralization patterns.

**Solution**: Distributed gateway architecture:
- Multiple gateway nodes in different geographic regions
- Miners compete on raw inference speed, not proximity to one validator
- Load balancing across gateways prevents single-point-of-failure
- Regional scoring normalizes for network latency differences

**Residual risk**: MEDIUM — requires architectural work but is solvable.

### 10. Timing Defense Gap — RESOLVED

**Problem**: Challenge latency measured by the validator includes network RTT
(10-120ms depending on datacenter), making the 50ms VRAM proof threshold
meaningless over WAN. A miner could re-run inference from scratch and still
appear "fast enough" if the raw timing check doesn't account for network overhead.

**Solution (IMPLEMENTED)**: RTT-corrected timing defense in audit_validator.py:

1. **RTT baseline tracking**: Validator pings each miner's `/health` endpoint
   on every discovery cycle (~10s), building a rolling median RTT baseline per miner.
2. **Net extraction time**: `net_time = raw_latency - rtt_baseline` estimates
   actual server-side work, filtering out network latency.
3. **Tiered scoring on net extraction time** (hardened_scoring.py):
   - <50ms → 1.0 (VRAM cache proven — extraction from cached KV states)
   - 50-200ms → decay to 0.90 (slow cache or CPU-side HF extraction)
   - 200-1000ms → decay to 0.70 (suspicious — partial re-computation)
   - >1000ms → 0.50 (strong evidence of full re-inference from scratch)
   - >2000ms → auto-fail (CHALLENGE_TIMEOUT_HARD_MS)
4. **Re-inference alerting**: Net extraction >1000ms triggers a log warning
   for manual review. Qwen 7B forward pass takes 2-10s on GPU, so >1000ms
   net time is a strong signal the miner didn't have VRAM cache.
5. **Miner-reported time cross-check**: Miner returns its own extraction time;
   validator logs it alongside the RTT-corrected estimate. Miners can lie about
   self-reported time, but large discrepancies (miner says 2ms, validator sees
   800ms net) flag suspicious behavior.

**Why this matters**: Without RTT correction, all miners show 50-150ms
"challenge latency" regardless of whether they extracted from VRAM cache (2ms)
or ran a full forward pass (5s) — the network RTT dominates. With RTT correction,
VRAM-cached miners score net ~5ms while re-inference miners score net ~3000ms,
making the timing defense actually discriminative.

**Residual risk**: LOW — RTT baseline uses p10 percentile (not median) so miners
cannot inflate baseline by adding artificial latency to health pings. The fastest
pings reveal the true network RTT. Requires 3+ measurements to activate.

### 11. Validator Cost Scaling

**Problem**: If every request required a full forward pass on the validator,
validators would need the same GPU resources as miners, scaling linearly.
Even at 20% spot-checking, validator costs scale linearly with traffic.
At 1000 req/s across the subnet, that's 200 forward passes/s on the validator.

**Current mitigation**: Adaptive per-miner challenge rates (IMPLEMENTED).

**Adaptive challenge rate system** (hardened_scoring.py + audit_validator.py):
- New miners (< 10 challenges): 90% rate — high scrutiny
- Base rate: 30% after proving trustworthy (10 clean passes)
- Decay formula: `rate = 0.3 / sqrt(streak / 10)` — decays as trust builds
  - 10 passes → 30%, 40 passes → 15%, 90 passes → 10%, 360+ → 5% floor
- Any failure: resets streak to 0, boosts to 100% for next 5 challenges
- Suspect miners: always 100%
- Floor: 5% — never drops below this (statistical minimum for detection)

This keeps total validator GPU compute roughly constant as the fleet grows:
trusted miners free up budget for higher scrutiny of new/suspicious ones.

**Future scaling solutions** (not yet needed):
- **Batch/amortized verification**: One hidden-state check cross-validates multiple responses
- **Delegated verification**: Multiple validators share the challenge load
- **Statistical confidence targeting**: Math-driven per-miner rates targeting 99.9% detection

**Implementation priority**: DONE (basic adaptive). Future: batch verification.

**Residual risk**: LOW — adaptive system in place, further optimization possible.

### 12. Streaming Commitment Bypass — RESOLVED

**Problem**: The streaming endpoint (`/inference/stream`) did not inject inline
commitment challenges (`commit_layers`, `commit_positions`). A miner that exclusively
served streaming requests bypassed the primary verification path entirely.

**Solution (IMPLEMENTED)**: `_stream_response()` now calls `_generate_commitment_request()`
and injects commitment fields into the streaming payload. Commitments are extracted
from SSE metadata events. The R2 audit record includes `commitment_requested=True`.

**Residual risk**: LOW — streaming path now has equivalent verification to non-streaming.

### 13. Proxy/Relay Attack — PARTIALLY MITIGATED (nonce binding flawed)

**Attack**: A cheating miner runs no model. It forwards inference requests to another
honest miner or public API, and relays responses including hidden state commitments.
Multiple Sybil UIDs can leech off one honest miner's hardware.

**Sub-variants:**
- **External API relay**: Forward to OpenRouter/Together/Groq, relay tokens back.
  Hidden states will fail cosine because external APIs don't expose them.
- **Intra-network relay**: Forward to another miner on the SAME subnet (through
  the gateway or directly querying their axon). Hidden states are real (from a genuine
  model on the network), so cosine verification passes. The attacker needs zero GPU.
- **Validator impersonation**: Miner registers as a validator, queries other miners
  via the standard validator path, and relays their responses.

**Defenses (IMPLEMENTED):**

1. **Miner-bound HMAC nonce** (`proxy_gateway.py`, `vllm_miner.py`):
   - Gateway computes `nonce = HMAC-SHA256(request_id:miner_uid, gateway_secret)` for
     each request. Different nonce per miner, per request.
   - Nonce is sent alongside the inference payload. Miner includes it in commitment
     hashes: `hash(hidden_state || nonce)`.
   - Audit validator re-derives the expected nonce from the R2 audit record (which
     stores request_id + miner_uid) and verifies the commitment hash matches.
   - **VULNERABILITY (red-team finding 2026-03-14)**: The nonce is transmitted in
     cleartext to the miner. A relay miner (A) receives the nonce and forwards it
     verbatim to the target miner (B). B computes hidden states and produces
     `hash(hidden_state || forwarded_nonce)`, which passes verification. The nonce
     binding only prevents forging a commitment hash WITHOUT seeing the nonce — but
     the miner always sees it.
   - **Proposed fix**: Hotkey signature binding — require miners to sign commitments
     with their Bittensor hotkey private key. The validator verifies against the
     miner's on-chain registered hotkey. A relay miner cannot forward a private key.
     This requires bittensor wallet integration in the miner code.

2. **Prompt fingerprint detection** (`proxy_gateway.py`):
   - Gateway tracks SHA-256 hashes of all prompts over a 30s sliding window.
   - If the same prompt hash appears for two different miner UIDs within 30s, the
     gateway logs a relay suspect alert and tracks per-miner relay hit counts.
   - This catches the case where miner A relays through the gateway (the prompt
     appears twice — once for A's original request, once when A sends it to B).
   - **Limitation**: Doesn't catch direct axon-to-axon relay (bypasses gateway).

3. **External API defense** (existing):
   - External APIs don't expose hidden states, so cosine verification catches this.
   - Timing: external APIs add 100-500ms latency, caught by RTT-corrected timing.

4. **RTT-corrected timing** (existing, `audit_validator.py`):
   - Each miner has a p10 RTT baseline from direct pings.
   - Relay adds round-trip latency (miner A → B → A), which inflates extraction time.
   - Net extraction time = raw_latency - rtt_baseline; anomalously high values flag.
   - **Strongest practical defense**: relay reliably adds 50-200ms+ latency.

5. **Cross-miner output correlation** (existing, `collusion_detector.py`):
   - LCS token similarity between miners given the same prompt.
   - If miner A consistently returns identical outputs to miner B (because A is
     relaying B's responses), the collusion detector flags the pair.

**Residual risk**: MEDIUM — nonce binding is weaker than initially assessed (cleartext
nonce can be forwarded). The primary relay defenses are (a) timing/RTT detection,
(b) prompt fingerprinting for gateway-routed relay, and (c) output correlation.
Speed scoring (40% weight) provides economic disincentive since relay adds latency.
Hotkey signing would close the gap definitively — tracked as future improvement.

### 14. Synthetic Probe Fingerprinting — RESOLVED

**Problem**: The synthetic audit loop used only 10 fixed topics and 3 fixed styles
(30 unique prompts). A miner could fingerprint these and cheat only on non-synthetic traffic.

**Solution (IMPLEMENTED)**: Expanded to 48 topics × 12 styles × 12 random prefixes
(~6900 unique combinations). Max tokens uses continuous range (32-256) instead of
fixed [64, 128, 256]. Bandwidth probes similarly expanded to 22 topics × 4 styles.

**Residual risk**: LOW — combinatorial explosion makes fingerprinting impractical.

### 15. TPS Inflation via Token Count — RESOLVED

**Problem**: Gateway trusted the miner's `output_tokens` count for TPS calculation.
A miner could inflate this value to boost apparent throughput.

**Solution (IMPLEMENTED)**: Gateway-side token counting uses `all_token_ids` from the
miner response (ground truth). If `all_token_ids` is absent, falls back to character-based
estimate (~4 chars/token). Miner's self-reported `output_tokens` is only trusted when
it agrees with the gateway's count within 20%.

**Residual risk**: LOW — `all_token_ids` can theoretically be padded, but the miner
must produce tokens that make sense with the response text.

### 16. Distilled Model + Speed Bonus — RESOLVED

**Problem**: A miner running a quantized/distilled model (cosine 0.70-0.85) could
compensate the verification penalty with the TPS bonus (up to 1.5x), making cheating
profitable. E.g., distilled model: fidelity=0.89x × TPS bonus=1.33x = 1.18x weight,
vs honest: fidelity=1.09x × TPS bonus=1.07x = 1.17x.

**Solution (IMPLEMENTED)**: TPS bonus is now damped by cosine fidelity. Full TPS bonus
only available at cosine >= 0.95. At cosine < 0.80, TPS bonus is zero. Formula:
`cos_damper = clamp((avg_cosine - 0.80) / 0.15, 0, 1)`.

**Residual risk**: LOW — makes distilled model attacks strictly unprofitable.

### 17. R2 Record Tampering

**Attack**: If a miner obtains R2 credentials, they can write fabricated audit records
showing perfect scores for their UID.

**Defenses (planned):**
- Gateway-signed audit records (HMAC with gateway-only secret)
- Auditor verifies signatures before trusting records

**Residual risk**: MEDIUM — R2 credentials are in `.env` on the validator machine only,
but defense-in-depth requires signed records.

### 18. Dual-Model Generation Attack — PARTIALLY MITIGATED (red-team finding 2026-03-14)

**Attack**: Miner runs TWO models simultaneously:
- Model A: fast quantized/distilled model for generating tokens (actual inference output)
- Model B: correct baseline model for hidden state extraction

The miner generates tokens with model A (fast, different token distribution), then
extracts hidden states from model B using the SAME token IDs from model A. Since the
validator uses the miner's `all_token_ids` to compute reference hidden states, the cosine
similarity will be ~1.0 because both validator and miner are computing over the same tokens.

**Defenses (IMPLEMENTED):**
1. **Output perplexity spot-check**: 10% of committed records get a windowed perplexity
   check. Validator computes log-likelihood of 16-32 output tokens under the reference
   model. If the output was generated by a different model, perplexity will be significantly
   higher (>50 vs <10 for self-generated tokens). Alert threshold at 50 (conservative
   to account for high-temperature/creative sampling).
2. **Multi-layer commitments**: 2-3 layers checked per request (not just 1). A dual-model
   attacker must run model B's full forward pass for hidden state extraction at multiple
   random layers, negating any speed advantage from model A.
3. **TPS bonus dampened by cosine**: Full TPS bonus only at cosine >= 0.95. A different
   model's tokens processed through model B still produce correct hidden states, but the
   output quality (measured separately) may diverge.

**Practical constraints:**
- Requires running two models (double memory — ~14GB quantized + ~14GB baseline)
- Multi-layer commitments force full model B forward pass, negating speed savings
- Perplexity check catches statistically divergent output distributions

**Residual risk**: LOW — perplexity spot-checking + multi-layer commitments + cosine
dampening makes this attack unprofitable. Full closure: continuous perplexity tracking
(not just spot-check) and token logit verification at commitment positions.

### 19. Chat Template Tampering — HARDENED (red-team finding 2026-03-14)

**Attack**: Miner modifies the Jinja2 chat template used by `tokenizer.apply_chat_template()`
to alter how messages are tokenized. This could strip system prompts, change formatting tokens,
or inject hidden instructions. Since the inline commitment path uses the miner's self-reported
`all_token_ids` for reference computation, a tampered template that passes the prefix check
would result in cosine=1.0 (correct hidden states for the wrong token sequence).

**Severity**: MEDIUM

**Defenses implemented**:
1. **Prompt token prefix verification**: Validator independently applies `apply_chat_template`
   and compares against the miner's reported token prefix. >2% mismatch = template tamper
   detected (tightened from 10% on 2026-03-14).
2. **Sequence length check**: Miner's prompt tokens must be >=80% of expected length.
3. **Tokenization override on tamper**: When tampering detected, validator overrides with its
   own tokenization — miner's hidden states will mismatch, causing a hard failure.
4. **Legacy audit path immunity**: Post-hoc audit re-tokenizes independently from `messages`,
   never trusting miner's `all_token_ids`.
5. **Multi-turn synthetic probes**: ~30% of synthetic probes now include system prompts and
   multi-turn conversation context, exercising more of the chat template (added 2026-03-14).
6. **Minor mismatch logging**: Even sub-threshold mismatches (>0 but <2%) are logged for
   monitoring — catches subtle template tweaks before they become exploits.

**Residual risk**: LOW — legitimate miners produce identical prompt tokens, so the 2% threshold
is generous. Remaining gap: a miner that modifies only the generation prompt suffix (a few
tokens) could stay under threshold. Would need periodic canonical template fingerprinting probes
to fully close.

## Summary Table

| Attack | Severity | Defense Level | Residual Risk |
|--------|----------|---------------|---------------|
| Selective honesty | HIGH | Strong | LOW |
| Hidden state spoofing | CRITICAL | Very strong | VERY LOW |
| Latency gaming | MEDIUM | Strong | LOW |
| Score manipulation | MEDIUM | Strong | LOW |
| KV cache cheating | HIGH | Strong (Phase 2) | LOW |
| Collusion | HIGH | Strong (Phase 2) | LOW |
| Resource misrepresentation | MEDIUM | Strong | LOW |
| Optimization penalty | HIGH | Tiered cosine (IMPLEMENTED) | LOW |
| Centralization pressure | MEDIUM | Planned (distributed gw) | MEDIUM |
| Timing defense gap | HIGH | RTT-corrected p10 timing (IMPLEMENTED) | LOW |
| Validator cost scaling | MEDIUM | Adaptive per-miner rates (IMPLEMENTED) | LOW |
| Streaming commitment bypass | HIGH | Fixed (IMPLEMENTED) | LOW |
| Proxy/relay attack | HIGH | Timing + fingerprinting + correlation (nonce FLAWED) | MEDIUM |
| Synthetic fingerprinting | HIGH | Fixed (IMPLEMENTED) | LOW |
| TPS inflation | MEDIUM | Fixed (IMPLEMENTED) | LOW |
| Distilled model + speed bonus | MEDIUM | Fixed (IMPLEMENTED) | LOW |
| R2 record tampering | CRITICAL | Planned (gateway-signed records) | MEDIUM |
| Dual-model generation | MEDIUM | Perplexity + multi-layer + cosine dampening | LOW |
| Chat template tampering | MEDIUM | Strong (2% threshold + multi-turn probes) | LOW |

### 20. Commitment Position Predictability — RESOLVED (red-team finding 2026-03-14)

**Problem**: Original commitment system checked only 1 layer per request, with 60% of
positions being "last" (sequence end) and 40% being small negative offsets (-1 to -10).
A miner could pre-compute hidden states for just the probable positions rather than
maintaining the full activation cache.

**Solution (IMPLEMENTED)**:
1. **Multi-layer commitments**: 2-3 layers per request (80% chance of 2, 20% chance of 3)
   instead of just 1. Layers sampled without replacement from all eligible layers.
2. **Uniform random positions**: Position distribution is now:
   - 40% "last" (end of sequence)
   - 30% random negative offset -1 to -20 (output region)
   - 30% random positive offset 5-54 (prompt region)
   This forces miners to maintain activations across the full sequence.
3. **Cross-layer validation**: A cheater must produce correct hidden states at multiple
   random layers simultaneously, which requires a full forward pass.

**Residual risk**: VERY LOW — exponential cost to pre-compute all layer×position combinations.

### 21. Concurrency Throughput Gaming — NEW (incentive alignment 2026-03-14)

**Problem**: Miners could optimize only for single-request throughput (the primary scoring
metric) while using poor batching/scheduling that degrades under concurrent load. Since
real-world traffic is concurrent, this misaligns incentives.

**Solution (IMPLEMENTED)**:
1. **Bandwidth probes**: Every ~5 minutes, validator sends 4 simultaneous requests to each
   miner and measures aggregate TPS vs baseline TPS.
2. **Concurrency bonus** in weight calculation: miners maintaining >70% of baseline TPS
   under concurrent load get up to 25% weight bonus. This directly incentivizes continuous
   batching, PagedAttention, and efficient KV cache management.
3. **Bandwidth probe data feeds into scoring**: the `concurrent_tps_ratios` field tracks
   per-miner concurrency performance across the epoch.

**Residual risk**: LOW — miners are now incentivized to optimize for both single-request
speed AND concurrent throughput, aligning with real production workload patterns.

## Architecture

```
External Clients → POST /v1/chat/completions (OpenAI-compatible)
                 → Hardened Gateway Validator
                 → Intelligent Router (reliability + load + session affinity)
                 → Miner (serves inference + hidden states from GPU VRAM)
                 → Challenge Engine (crypto-random multi-point verification)
                 → Hardened Scoring Engine (net points, asymmetric penalties)
                 → R2 Publisher (immutable audit log)
```

### 23. Bandwidth Probe Fingerprinting

**Attack**: Miner detects bandwidth probes (concurrent burst, missing fields) and
prioritizes probe requests over organic traffic for inflated concurrency bonus.

**Detection signals (pre-fix)**: Missing nonce/commit_layers/messages fields, fixed
CONCURRENCY=4 burst, fixed max_tokens=64.

**Defense**: Bandwidth probes now include all fields matching organic/synthetic format
(nonce, commit_layers, commit_positions, messages). Concurrency varies 2-5, max_tokens
varies 32-128. Status: **FIXED** (step 9).

### 24. Selective Organic Throttling (Verified Miner)

**Attack**: Miner with pass_rate=1.0 detects synthetic probes (by source IP or request
patterns), serves them at full speed, but throttles organic traffic to save GPU for
other tasks. The pass_rate=1.0 divergence exemption meant no penalty was applied.

**Defense**: Severe divergence (>25% gap) now triggers a 30% penalty even at 100% pass
rate, for both per-epoch and cross-epoch checks. Mild divergence at pass_rate=1.0 is
still tolerated (natural noise). Status: **FIXED** (step 9).

**Remaining risk**: Source IP fingerprinting was addressed in Vector 25.

### 25. Source IP Fingerprinting (Auditor vs Gateway)

**Attack**: Miner inspects HTTP source IP via FastAPI `request.client.host` to
distinguish auditor probes (from audit_validator) vs organic traffic (from
proxy_gateway). If deployed on separate machines, source IPs differ. Miner
cheats on auditor requests while behaving honestly for organic traffic.

**Defense**: Gateway relay endpoint (`/internal/relay`). The auditor sends
synthetic probes and bandwidth probes through the gateway's connection, so
miners always see the gateway's IP. Authenticated with a shared secret
(`INTERNAL_RELAY_SECRET`). Falls back to direct connection if relay fails.

**Status**: FIXED. Both processes share `INTERNAL_RELAY_SECRET`; auditor reads
`GATEWAY_RELAY_URL` env var. When set, all probes route through gateway.
Nonces are rewritten by the gateway to use its own secret, maintaining
nonce consistency regardless of request origin.

### 26. Commitment Cherry-Picking

**Attack**: When the gateway or auditor requests hidden state commitments for
multiple layers (e.g., layers [5, 18, 23]), the miner only returns commitments
for the easiest/shallowest layers where its hidden states match best. It omits
commitments for deeper layers where divergence is higher. The auditor would
verify the cherry-picked subset and see 100% pass rate.

**Defense**: Added `requested_layers` and `requested_positions` fields to R2
audit records. The auditor now compares returned commitment layers against
requested layers and logs `[CHERRY-PICK]` warnings when layers are missing.
Currently monitoring-only — if cherry-picking is detected in practice, will
be escalated to auto-fail.

**Status**: DETECTED. Monitoring deployed. No cherry-picking observed from
current miners (all return cosine=1.0000 for all requested layers).

**Remaining risk**: A sophisticated miner could return commitments for all
requested layers but recompute hidden states for just the committed ones,
serving garbage for others. This is partially addressed by the separate
hidden-state challenge (which queries arbitrary layers, not just committed ones).

### 27. Synthetic Probe Interval Fingerprinting

**Attack**: Miner monitors inter-request arrival times and builds a histogram.
Synthetic probes previously used a uniform 8-20s jitter between probes, creating
a distinctive cluster in the timing distribution. A miner detecting this cluster
can identify ~90% of synthetic probes, allowing it to serve honest responses only
for those while degrading service for organic traffic.

**Defense**: Replaced uniform 8-20s jitter with exponential distribution
(-15*ln(U), clamped [5, 45] seconds). Exponential inter-arrivals mimic a Poisson
process, which is statistically indistinguishable from organic traffic arrival
patterns. The distribution has the memoryless property: knowing when the last
request arrived gives no information about the next one.

**Status**: FIXED. Deployed in audit_validator.py.

### 28. Bandwidth Probe Burst Fingerprinting

**Attack**: Bandwidth probes send 2-5 concurrent requests in a burst, arriving
at ~5-minute intervals (300s ± 30s). This narrow jitter window and distinctive
burst pattern let miners detect bandwidth probes with high confidence. A miner
could prioritize these burst requests for inflated concurrency scores.

**Defense**: (1) Widened interval jitter from ±30s to exponential distribution
(-300*ln(U), clamped [180, 480] seconds). (2) Moved CONCURRENCY and MAX_TOKENS
randomization inside the per-cycle loop so they're freshly random each probe,
not fixed for the process lifetime.

**Status**: FIXED. Deployed in audit_validator.py.

### 29. Post-Inference Challenge Timing Correlation

**Attack**: When the auditor sends a synthetic inference probe followed
immediately by a `/hidden_state` challenge, the miner can observe a
distinctive pattern: inference request → response → challenge within 0-2
seconds. By tracking inter-request timing, a miner can detect which
requests will be challenged (those followed by a rapid `/hidden_state`
request) vs those that won't. The miner can then only be honest on
challenged requests.

**Fingerprint signals**:
- `/hidden_state` request arrives within ~2 seconds of an inference response
- The challenged request_id matches the most recent inference
- Pattern is deterministic: challenge always follows inference immediately

**Defense**: Deferred challenge queue. Instead of calling `audit_record()`
inline in the synthetic probe loop, challenges are pushed to an async queue
(`_deferred_queue`) and processed by a separate `deferred_challenge_worker`
coroutine. The worker applies an exponential delay (mean ~60s, clamped
[30, 180]s) before issuing the `/hidden_state` challenge. This completely
breaks the temporal correlation between inference and challenge — miners
see challenge requests arriving 30-180 seconds after the original inference,
indistinguishable from the audit_loop's R2-based deferred auditing.

Combined with inline commitments (COMMIT_RATE=1.0), we now have two
fully-deferred verification paths:
1. Commitment verification via audit_loop (reads R2 records ~15s later)
2. /hidden_state challenges via deferred worker (30-180s delay)

**Status**: FIXED. Deployed in audit_validator.py.

### 30. Challenge Rate Decay Exploitation

**Attack**: The adaptive challenge rate decays from 90% (new miner) down to
5% (trusted miner with long clean streak). A miner can track its own
challenge frequency over time to infer its trust level:
- If receiving challenges on ~90% of requests → new/untrusted, stay honest
- If challenges drop to ~30% → gaining trust, stay honest
- If challenges drop to ~5% → fully trusted, start cheating on unchallenged

The miner exploits the trust it built up by being honest only during the
high-challenge phase, then defecting once challenge rate is low enough.

**Analysis**: This is mitigated by COMMIT_RATE=1.0 — every request gets
inline commitment verification regardless of `/hidden_state` challenge rate.
Commitments are verified later in `audit_loop` using the same reference model.
A miner returning fake commitments would fail verification. A miner not
returning commitments gets tracked as cache_miss.

The remaining gap: commitment verification only checks the specific layers
and positions requested. A miner with the correct model loaded but serving
degraded inference (lower temperature, truncated output, quantized differently)
would pass commitment checks while providing lower quality.

**Residual risk**: LOW — commitment coverage at 100% makes the window for
undetected cheating very small. The few-layer check per request combined
with multi-request statistical accumulation catches systematic quality
degradation. The perplexity spot-check (now fixed) adds a second signal.

**Status**: ACCEPTED (low risk). No code change needed — existing defenses
(COMMIT_RATE=1.0, perplexity checks, divergence detection) provide adequate
coverage. Monitor for anomalies in pass_rate vs organic quality.

## Files

| File | Purpose |
|------|---------|
| `hardened_gateway.py` | Main validator with OpenAI API, routing, challenges |
| `hardened_scoring.py` | Exploit-resistant scoring with 7 defense layers |
| `challenge_engine.py` | Cryptographic challenge generation and verification |
| `test_hardened.py` | 29 tests covering all anti-cheat mechanisms |
| `kv_cache_prober.py` | KV cache verification probes (TTFT ratio measurement) |
| `collusion_detector.py` | Cross-miner collusion detection (4 signal types) |
| `test_kv_cache_and_collusion.py` | 20 tests for cache prober and collusion detector |
| `real_miner.py` | Reference miner implementation (HuggingFace) |
| `multi_gpu_miner.py` | Multi-GPU load-balanced miner |
