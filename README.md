# Constantinople

**Decentralized LLM Inference Subnet for Bittensor — Subnet 97**

Constantinople is a hardened inference subnet that provides trustless AI inference through hidden state verification. Validators route organic traffic to GPU miners and cryptographically verify that miners actually execute the model, not proxy or fake results.

## Architecture

```
Clients → OpenAI-compatible API → Hardened Gateway Validator
                                       ├── Intelligent Router (picks best miner)
                                       ├── Challenge Engine (hidden state verification)
                                       ├── Scoring Engine (speed × verification × consistency)
                                       └── Chain Weight Setter (on-chain rewards)
                                                    │
                    ┌───────────────────────────────┼────────────────────────┐
                    │                               │                        │
              GPU Miner 1                     GPU Miner 2              GPU Miner N
           (vLLM + Qwen 7B)              (vLLM + Llama 8B)               ...
```

## Key Features

- **OpenAI-compatible API** — Drop-in replacement at `/v1/chat/completions`
- **Hidden state verification** — Cryptographic challenges prove miners run the actual model
- **Anti-cheat scoring** — Multi-dimensional: speed (40%) × verification (40%) × consistency (20%)
- **KV cache verification** — Detects miners faking cache hits via TTFT ratio analysis
- **Collusion detection** — 4 orthogonal signals detect coordinated cheating
- **Metagraph discovery** — Auto-discovers miners from Bittensor chain
- **Streaming support** — Server-sent events for real-time token generation
- **Audit logging** — All requests/responses logged for transparency

## Quick Start

### Validator

```bash
# Clone and configure
git clone https://github.com/unconst/Constantinople.git
cd Constantinople
cp .env.example .env
# Edit .env with your wallet and settings

# Run with docker compose (includes watchtower for auto-updates)
docker compose up -d
```

### Miner

```bash
# On a GPU machine (e.g., Lium pod with RTX 4090/5090)
cd Constantinople/miner

# Run with vLLM (recommended for production)
python3 vllm_miner.py \
  --port 8091 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.70

# Or with Docker
docker build -t constantinople-miner .
docker run --gpus all -p 8091:8091 constantinople-miner
```

### Register on Subnet 97

```bash
# Install bittensor CLI
pip install bittensor

# Register validator
btcli subnet register --netuid 97 --wallet.name validator --wallet.hotkey default

# Register miner
btcli subnet register --netuid 97 --wallet.name miner --wallet.hotkey default
```

## Scoring

Miners are scored on three dimensions each epoch:

| Dimension | Weight | What it measures |
|-----------|--------|------------------|
| Speed | 40% | TTFT + tokens/sec (population-relative) |
| Verification | 40% | Hidden state cosine similarity ≥ 0.995 |
| Consistency | 20% | Variance in performance over time |

**Anti-gaming features:**
- Asymmetric penalties (cheating costs 3× what passing earns)
- Divergence detection (organic vs synthetic performance)
- Minimum sample requirements (prevents last-minute gaming)
- Multi-point challenges (20% check 3+ hidden state positions)

## Miner Guide

See [MINER_GUIDE.md](miner/MINER_GUIDE.md) for detailed miner setup instructions.

## Threat Model

See [THREAT_MODEL.md](docs/THREAT_MODEL.md) for the full security analysis covering:
- Selective honesty attacks
- Hidden state spoofing
- Latency gaming
- KV cache cheating
- Collusion/Sybil attacks

## License

MIT
