# Constantinople — Decentralized LLM Inference (Bittensor SN97)

**Live at [tau.ninja](https://tau.ninja)** — OpenAI-compatible API powered by decentralized GPU miners.

Hidden state verification protocol for trustless AI inference on Bittensor.

## Quick Start

Use the API with any OpenAI-compatible client:

```bash
curl https://vercel-app-rosy-kappa.vercel.app/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen-7b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256,
    "stream": true
  }'
```

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://vercel-app-rosy-kappa.vercel.app/v1",
    api_key="unused",
)

for chunk in client.chat.completions.create(
    model="qwen-7b",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="")
```

## How It Works

```
Client → POST /v1/chat/completions (OpenAI-compatible)
       → Validator Gateway (routes + verifies)
       → Miner (serves inference + exposes hidden states)
       → Challenge Engine (cosine similarity check)
       → Scoring Engine (speed × verification × consistency)
       → On-chain weight setting via commit-reveal
```

Miners prove they're actually running the LLM by returning hidden state vectors that match the validator's reference model within cosine similarity > 0.995.

## Validator

The validator is an OpenAI-compatible inference gateway that:
- Discovers miners from the Bittensor metagraph (subnet 97)
- Routes requests using intelligent load balancing with session affinity
- Embeds verification challenges inline with organic traffic
- Scores miners across speed, verification accuracy, and consistency
- Detects KV cache cheating and miner collusion
- Sets on-chain weights proportional to miner quality

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
| `VALIDATOR_MODEL` | `mock` | HF model for verification (`mock` or e.g. `Qwen/Qwen2.5-7B-Instruct`) |
| `EPOCH_LENGTH` | `300` | Scoring epoch length (seconds) |
| `VALIDATOR_PORT` | `8080` | Gateway port |

## Miner

Miners serve LLM inference with hidden state extraction for verification.

```bash
# vLLM miner (recommended, high throughput)
python vllm_miner.py --port 8091 --model Qwen/Qwen2.5-7B-Instruct --gpu-memory-utilization 0.70

# Multi-GPU
python vllm_miner.py --port 8091 --model Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 2

# HuggingFace miner (simpler)
python real_miner.py --port 8091 --model Qwen/Qwen2.5-7B-Instruct
```

See [MINER_GUIDE.md](MINER_GUIDE.md) for detailed setup instructions.

## Architecture

| Component | File | Description |
|-----------|------|-------------|
| Gateway | `hardened_gateway.py` | Main validator: routing, scoring, challenges, dashboard |
| Scoring | `hardened_scoring.py` | Multi-dimensional exploit-resistant scoring (7 defense layers) |
| Challenges | `challenge_engine.py` | Cryptographic challenge generation & verification |
| KV Cache | `kv_cache_prober.py` | KV cache verification via TTFT ratio analysis |
| Collusion | `collusion_detector.py` | Cross-miner collusion detection (4 signal types) |
| Monitor | `monitor.py` | Fleet monitoring with Telegram alerts |
| Mock Model | `model.py` | Deterministic mock for testing |
| R2 Audit | `r2_publisher.py` | Audit log publishing |
| vLLM Miner | `vllm_miner.py` | High-throughput vLLM-based miner |
| HF Miner | `real_miner.py` | HuggingFace transformer miner |

See [THREAT_MODEL.md](docs/THREAT_MODEL.md) for attack analysis and defense details.

## License

MIT
