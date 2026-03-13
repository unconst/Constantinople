# Constantinople — Decentralized LLM Inference (Bittensor SN97)

Hidden state verification protocol for trustless AI inference on Bittensor.

## How it works

```
Client → POST /v1/chat/completions (OpenAI-compatible)
       → Validator Gateway (routes + verifies)
       → Miner (serves inference + exposes hidden states)
       → Challenge Engine (cosine similarity check)
       → Scoring Engine (speed × verification × consistency)
       → On-chain weight setting
```

Miners prove they're actually running the LLM by returning hidden state vectors that match the validator's reference model within cosine similarity > 0.995.

## Validator

The validator is an OpenAI-compatible inference gateway that:
- Discovers miners from the Bittensor metagraph (subnet 97)
- Routes requests using intelligent load balancing
- Embeds verification challenges inline with organic traffic
- Scores miners across speed, verification accuracy, and consistency
- Sets on-chain weights proportional to miner quality

### Run with Docker

```bash
# Copy wallet files to the deployment machine
# Requires: wallet registered on subnet 97

docker compose up -d
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
# HuggingFace miner
python real_miner.py --port 8091 --model Qwen/Qwen2.5-7B-Instruct

# vLLM miner (high throughput)
python vllm_miner.py --port 8091 --model Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 2
```

See [MINER_GUIDE.md](MINER_GUIDE.md) for detailed setup instructions.

## API

The gateway exposes an OpenAI-compatible API:

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen-7b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Architecture

| Component | File | Description |
|-----------|------|-------------|
| Gateway | `hardened_gateway.py` | Main validator: routing, scoring, challenges |
| Scoring | `hardened_scoring.py` | Multi-dimensional exploit-resistant scoring |
| Challenges | `challenge_engine.py` | Cryptographic challenge generation & verification |
| KV Cache | `kv_cache_prober.py` | KV cache verification via TTFT analysis |
| Collusion | `collusion_detector.py` | Cross-miner collusion detection |
| Mock Model | `model.py` | Deterministic mock for testing |
| R2 Audit | `r2_publisher.py` | Audit log publishing |
| Miner | `real_miner.py` | HuggingFace transformer miner |
| vLLM Miner | `vllm_miner.py` | High-throughput vLLM-based miner |

## License

MIT
