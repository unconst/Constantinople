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
