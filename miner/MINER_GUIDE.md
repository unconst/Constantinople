# Miner Onboarding Guide

A practical guide for running an inference miner on the Bittensor inference subnet.

---

## 1. Overview

The inference subnet provides decentralized LLM inference on Bittensor. Miners run real transformer models, serve inference requests from the validator gateway, and earn rewards proportional to their speed, reliability, and honesty.

**How miners earn rewards:**

- Serve inference requests routed by the gateway validator.
- Pass hidden state verification challenges that prove you are running the actual model (not proxying or faking outputs).
- Accumulate a composite score each epoch. Scores translate directly to on-chain weight, which determines your share of emissions.

**What makes a good miner:**

- **Fast time-to-first-token (TTFT)** -- the gateway measures wall-clock latency, so you cannot fake this.
- **High tokens-per-second (TPS)** -- throughput is scored relative to the miner population.
- **Consistent challenge passing** -- failing a hidden state challenge costs 3x what passing one earns. Three consecutive failures mark you as suspect.
- **High availability** -- disconnections and timeouts reduce your consistency score.

---

## 2. Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA, 16GB+ VRAM (RTX 4090, A5000, A100) | RTX 4090 (24GB) or A100/H100 |
| CPU | 8+ cores | 16+ cores |
| RAM | 32GB | 64GB |
| Network | 100 Mbps upload, low latency | 1 Gbps |
| Storage | 100GB SSD | 200GB+ NVMe SSD |

**VRAM notes:**

- The default model (Qwen2.5-1.5B-Instruct) needs ~4GB in fp16. Good for testing.
- Llama 3 8B needs ~16GB in fp16.
- Larger models (70B) require multi-GPU or quantization.
- Model weights are downloaded from HuggingFace on first run and cached in `~/.cache/huggingface`.

**GPU driver:** NVIDIA driver 525+ with CUDA 12.x. Verify with `nvidia-smi`.

---

## 3. Quick Start (Docker)

Docker is the fastest way to get running. You need Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.

### 3.1 Pull the image

```bash
docker pull thebes1618/inference-subnet-miner:latest
```

### 3.2 Create a `.env` file

```bash
# .env
MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
MINER_PORT=8091
CACHE_SIZE=200
```

### 3.3 Run with docker-compose

Save the project's `docker-compose.yml` to your working directory, then:

```bash
docker compose up -d miner
```

This starts the miner with GPU access and a health check. The compose file also includes a Watchtower service that auto-pulls new images every 5 minutes:

```bash
# To run miner + auto-updater:
docker compose up -d miner watchtower
```

### 3.4 Run standalone (without compose)

```bash
docker run -d \
  --name inference-miner \
  --gpus all \
  -p 8091:8091 \
  -v huggingface-cache:/root/.cache/huggingface \
  thebes1618/inference-subnet-miner:latest \
  --port 8091 --model "Qwen/Qwen2.5-1.5B-Instruct"
```

### 3.5 Verify health

```bash
curl http://localhost:8091/health
```

Expected response:

```json
{
  "status": "ok",
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "num_layers": 28,
  "hidden_dim": 1536,
  "total_requests": 0
}
```

---

## 4. Manual Setup

For direct deployment without Docker (e.g., on a bare-metal GPU server or Lium pod).

### 4.1 Install Python 3.10+

```bash
python3 --version  # must be 3.10 or higher
```

### 4.2 Install dependencies

```bash
pip install torch transformers accelerate fastapi "uvicorn[standard]" numpy aiohttp pydantic
```

On system Python (e.g., Lium pods), you may need:

```bash
pip install --break-system-packages torch transformers accelerate fastapi "uvicorn[standard]" numpy aiohttp pydantic
```

### 4.3 Start the miner

```bash
python3 real_miner.py \
  --port 8091 \
  --model "Qwen/Qwen2.5-1.5B-Instruct" \
  --cache-size 200
```

CLI arguments:

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | 8091 | HTTP port to listen on |
| `--host` | 0.0.0.0 | Bind address |
| `--model` | Qwen/Qwen2.5-1.5B-Instruct | HuggingFace model name or local path |
| `--cache-size` | 200 | Max inference requests to cache hidden states for |

The first run will download model weights from HuggingFace (may take several minutes depending on model size and bandwidth).

### 4.4 Run in background

```bash
nohup python3 real_miner.py --port 8091 --model "Qwen/Qwen2.5-1.5B-Instruct" > miner.log 2>&1 &
```

### 4.5 Use the deploy script

The `deploy_gpu.sh` script automates the full setup (install deps, check GPU, start miner):

```bash
# Miner only
bash deploy_gpu.sh --miner-only

# Custom model
MODEL_NAME="meta-llama/Llama-3-8B-Instruct" bash deploy_gpu.sh --miner-only
```

### 4.6 Verify it's running

```bash
curl http://localhost:8091/health
```

Test inference:

```bash
curl -X POST http://localhost:8091/inference \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 32}'
```

---

## 5. How Scoring Works

Your miner is scored on three dimensions each epoch. Scores are combined into a final weight that determines your share of emissions.

### 5.1 Score Components

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| **Speed** | 40% | TTFT and tokens-per-second, measured wall-clock by the gateway |
| **Verification** | 40% | Hidden state challenge pass rate (cosine similarity must exceed 0.995) |
| **Consistency** | 20% | Stable performance across organic and synthetic requests |

### 5.2 Speed Score

- **TTFT** is scored on a curve: 30ms or below is excellent, above 500ms is poor.
- **TPS** is scored similarly: 150+ tok/s is excellent, below 10 tok/s is poor.
- The gateway measures wall-clock time, so you cannot fake speed metrics.

### 5.3 Verification Score (Hidden State Challenges)

The gateway sends challenges bundled inline with inference requests. Your miner cannot distinguish challenge traffic from organic traffic. For each challenge:

1. The gateway picks a random layer and token position from the inference you just ran.
2. Your miner returns the cached hidden state vector at that position.
3. The gateway runs its own partial forward pass and compares with cosine similarity.
4. **Pass threshold:** cosine similarity > 0.995.
5. **Latency:** responses under 50ms are ideal. Above 500ms is an automatic failure.

**Asymmetric penalties:** Failing a challenge costs 3x what passing one earns. Three consecutive failures flag your miner as suspect.

### 5.4 Consistency Score

The gateway tracks whether your performance diverges between organic and synthetic requests:

- A gap of >12% between organic and synthetic performance triggers a penalty (-30% weight).
- A gap of >25% triggers a severe penalty (-70% weight).
- You need at least 5 samples of each type before divergence detection activates.

### 5.5 Anti-Gaming Protections

- Minimum 10 requests per epoch to receive any weight (prevents last-second sniping).
- Maximum 10,000 requests per miner per epoch (prevents flooding).
- Score per request is capped at 1.0 (prevents single-request domination).
- Challenges are cryptographically unpredictable and indistinguishable from organic traffic.

---

## 6. Optimization Tips

### Use the right model

Match your model to your hardware. An overloaded GPU leads to high TTFT and low TPS, both of which hurt your score.

### Enable KV cache (it is tested)

The gateway's KV cache prober verifies that your miner maintains KV cache for multi-turn sessions. The default `real_miner.py` handles this via its `HiddenStateCache`, but if you write a custom miner, make sure you cache hidden states across requests.

### Keep `--cache-size` large enough

The default of 200 cached requests works for moderate traffic. If you see "cache_miss" errors in your logs, increase it. Each cached request stores hidden states for all layers, so monitor GPU/CPU memory.

### Use vLLM for higher throughput (recommended)

For production, use `vllm_miner.py` instead of `real_miner.py`:

```bash
pip install vllm
python3 vllm_miner.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --port 8091 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85
```

vLLM provides continuous batching, PagedAttention, and significantly higher TPS. The `vllm_miner.py` uses vLLM's AsyncLLMEngine for generation and a separate HuggingFace model instance for hidden state extraction, giving you the best of both worlds.

Or use the automated deploy script:

```bash
MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct bash deploy_vllm.sh --miner-only
```

### Enable flash attention

If your GPU supports it (Ampere or newer), install and enable flash attention:

```bash
pip install flash-attn --no-build-isolation
```

This reduces memory usage and improves speed, particularly for longer sequences.

### Monitor GPU utilization

```bash
# Live monitoring
watch -n 1 nvidia-smi

# Check for thermal throttling
nvidia-smi -q -d PERFORMANCE
```

Keep GPU utilization high (80-95%) but not maxed. Sustained 100% utilization often means requests are queuing, which increases TTFT.

### Network latency matters

Place your miner close to the validator gateway. TTFT is measured wall-clock by the gateway, so network round-trip time is included in your score.

---

## 7. Troubleshooting

### Common Errors

**`CUDA out of memory`**

Your model is too large for your GPU. Options:
- Use a smaller model
- Reduce `--cache-size` (each cached request stores hidden states)
- Use quantization (load model in int8/int4)

**`cache_miss` in challenge responses**

The validator requested a hidden state for a request that was already evicted from cache. Increase `--cache-size` or check that your miner is not restarting between inference and challenge.

**Miner health check fails**

```bash
# Check if the process is running
ps aux | grep real_miner

# Check logs for errors
tail -50 miner.log

# Check GPU status
nvidia-smi
```

**Model download hangs or fails**

```bash
# Pre-download the model
python3 -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')"
```

If behind a firewall, download weights manually and pass a local path to `--model`.

**Low cosine similarity (challenge failures)**

This means your miner's hidden states do not match the validator's. Common causes:
- Model mismatch: you must be running the exact same model the validator expects.
- Precision mismatch: the miner uses fp16 and normalizes hidden states before returning them. Make sure you are not applying extra post-processing.
- Token alignment: the miner returns `all_token_ids` so the validator can align its forward pass. Make sure this field is populated correctly.

**Port already in use**

```bash
# Find and kill the existing process
lsof -ti :8091 | xargs kill -9
```

### Checking Logs

```bash
# Follow live logs
tail -f miner.log

# Search for errors
grep -i "error\|exception\|fail" miner.log

# Check challenge results
grep "Challenge" miner.log
```

### Verifying Challenge Responses Manually

```bash
# Run inference
RESPONSE=$(curl -s -X POST http://localhost:8091/inference \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is 2+2?"}], "max_tokens": 16}')

echo "$RESPONSE" | python3 -m json.tool

# Extract request_id and query hidden state
REQUEST_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['request_id'])")

curl -s -X POST http://localhost:8091/hidden_state \
  -H "Content-Type: application/json" \
  -d "{\"request_id\": \"$REQUEST_ID\", \"layer_index\": 0, \"token_index\": 0}" | python3 -m json.tool
```

If the hidden state endpoint returns a vector with `latency_ms` under 50ms, your miner is correctly caching and serving challenge responses.

---

## Endpoints Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check. Returns model info and request count. |
| `/inference` | POST | Run inference. Accepts `prompt` or `messages` (chat format). Returns generated text and optionally inline challenge results. |
| `/inference/stream` | POST | SSE streaming inference. Same input as `/inference`, streams tokens as they are generated. |
| `/hidden_state` | POST | Return cached hidden state at a given layer and token position. Used by the validator for verification challenges. |

---

## Quick Reference

```bash
# Docker (simplest)
docker run -d --gpus all -p 8091:8091 thebes1618/inference-subnet-miner:latest

# Manual
pip install torch transformers accelerate fastapi "uvicorn[standard]" numpy
python3 real_miner.py --port 8091 --model "Qwen/Qwen2.5-1.5B-Instruct"

# Verify
curl http://localhost:8091/health
```
