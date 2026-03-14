# Constantinople — Decentralized LLM Inference (Bittensor SN97)

**Live at [constantinople.cloud](https://www.constantinople.cloud)** — OpenAI-compatible API powered by decentralized GPU miners.

Hidden state verification protocol for trustless AI inference on Bittensor.

## Quick Start

Use the API with any OpenAI-compatible client:

```bash
curl https://api.constantinople.cloud/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer cst-YOUR_API_KEY' \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256,
    "stream": true
  }'
```

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.constantinople.cloud/v1",
    api_key="cst-YOUR_API_KEY",
)

for chunk in client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="")
```

Get an API key at [constantinople.cloud/api](https://www.constantinople.cloud/api).

## How It Works

```
Client → POST /v1/chat/completions (OpenAI-compatible)
       → Proxy Gateway (stateless forwarder, routes + logs)
       → Miner (serves inference + returns hidden state commitments)
       → Audit Validator (async challenges, cosine similarity check)
       → Scoring Engine (speed × verification × consistency)
       → On-chain weight setting via commit-reveal
```

Miners prove they're actually running the LLM by returning hidden state vectors that match the validator's reference model (cosine similarity > 0.70 to pass, > 0.99 for full credit).

## Validator

The validator runs as two split services:

- **Proxy Gateway** (`proxy_gateway.py`): Stateless request forwarder with metagraph discovery, inline commitment injection, R2 audit logging, and wall-clock TPS validation.
- **Audit Validator** (`audit_validator.py`): Async auditor that reads R2 records, sends deferred hidden state challenges (30-180s delay), runs reference forward passes, and sets on-chain weights.

### Run with Docker

```bash
# Requires: wallet registered on subnet 97
docker compose up -d
# Watchtower auto-pulls new images every 5 minutes
```

### Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `WALLET_NAME` | `validator` | Bittensor wallet name |
| `HOTKEY_NAME` | `default` | Hotkey name |
| `WALLET_PATH` | `~/.bittensor/wallets` | Path to wallet directory |
| `VALIDATOR_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | HF model for verification |
| `EPOCH_LENGTH` | `4320` | Scoring epoch length (seconds) |
| `VALIDATOR_PORT` | `8080` | Proxy gateway port |

## Miner

Miners serve LLM inference with hidden state extraction for verification. The current required model is **Qwen/Qwen2.5-7B-Instruct**.

```bash
# vLLM miner (recommended, high throughput)
python vllm_miner.py --port 8091 --model Qwen/Qwen2.5-7B-Instruct --gpu-memory-utilization 0.70 --hf-device cpu

# Multi-GPU
python vllm_miner.py --port 8091 --model Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 2

# HuggingFace miner (simpler, lower throughput)
python real_miner.py --port 8091 --model Qwen/Qwen2.5-7B-Instruct
```

See [MINER_GUIDE.md](MINER_GUIDE.md) for detailed setup instructions, scoring breakdown, and troubleshooting.

## Architecture

| Component | File | Description |
|-----------|------|-------------|
| Proxy Gateway | `proxy_gateway.py` | Stateless forwarder: routing, commitment injection, R2 logging |
| Audit Validator | `audit_validator.py` | Async auditor: deferred challenges, reference model, weight setting |
| Scoring | `hardened_scoring.py` | Multi-dimensional exploit-resistant scoring (7 defense layers) |
| Challenges | `challenge_engine.py` | Cryptographic challenge generation & verification |
| KV Cache | `kv_cache_prober.py` | KV cache verification via TTFT ratio analysis |
| Collusion | `collusion_detector.py` | Cross-miner collusion detection (4 signal types) |
| Monitor | `monitor.py` | Fleet monitoring with Telegram alerts |
| R2 Audit | `r2_publisher.py` | Audit log publishing to Cloudflare R2 |
| vLLM Miner | `vllm_miner.py` | High-throughput vLLM-based miner |
| HF Miner | `real_miner.py` | HuggingFace transformer miner |

See [THREAT_MODEL.md](THREAT_MODEL.md) for attack analysis and defense details (29+ vectors analyzed).

## License

MIT
