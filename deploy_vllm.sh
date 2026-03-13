#!/bin/bash
# Deploy inference miner with vLLM on a GPU pod (RTX 4090 / 24GB VRAM)
#
# vLLM serves fast inference while hidden states are extracted via
# a lightweight HuggingFace model instance for challenge verification.
#
# Usage:
#   scp deploy_vllm.sh tools/inference_subnet/*.py user@pod:/workspace/
#   ssh user@pod 'bash /workspace/deploy_vllm.sh'
#
# Environment:
#   MODEL_NAME      — HuggingFace model (default: meta-llama/Meta-Llama-3-8B-Instruct)
#   MINER_PORT      — Miner port (default: 8091)
#   VALIDATOR_PORT  — Validator gateway port (default: 8080)
#   VLLM_PORT       — vLLM OpenAI server port (default: 8000)
#   HF_TOKEN        — HuggingFace token for gated models (Llama 3)
#   API_KEYS        — Comma-separated API keys for gateway auth
#   MONITORING_KEYS — Comma-separated keys for monitoring endpoints

set -e

MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
MINER_PORT="${MINER_PORT:-8091}"
VALIDATOR_PORT="${VALIDATOR_PORT:-8080}"
VLLM_PORT="${VLLM_PORT:-8000}"
MODE="${1:---both}"
WORK_DIR="${WORK_DIR:-/workspace/inference_subnet}"
LOG_DIR="${LOG_DIR:-/workspace}"

echo "=== Inference Subnet vLLM Deploy ==="
echo "Model: $MODEL_NAME"
echo "Mode: $MODE"
echo "vLLM port: $VLLM_PORT"

# ── Check GPU ─────────────────────────────────────────────────────────────
echo ""
echo "=== GPU Status ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || {
    echo "ERROR: No GPU detected. vLLM requires a GPU."
    exit 1
}

# ── Install deps ──────────────────────────────────────────────────────────
echo ""
echo "=== Installing dependencies ==="
pip install --break-system-packages -q \
    vllm \
    fastapi uvicorn[standard] aiohttp numpy pydantic boto3 \
    torch transformers accelerate \
    2>&1 | tail -10

echo "Dependencies installed."

# ── HuggingFace auth (for gated models like Llama 3) ─────────────────────
if [ -n "$HF_TOKEN" ]; then
    echo "Logging into HuggingFace..."
    huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true
fi

# ── Copy code ─────────────────────────────────────────────────────────────
mkdir -p "$WORK_DIR"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for f in model.py hardened_scoring.py challenge_engine.py hardened_gateway.py \
         r2_publisher.py kv_cache_prober.py collusion_detector.py \
         real_miner.py vllm_miner.py; do
    if [ -f "$SCRIPT_DIR/$f" ]; then
        cp "$SCRIPT_DIR/$f" "$WORK_DIR/"
    fi
done

cd "$WORK_DIR"

# ── Kill existing processes ───────────────────────────────────────────────
echo ""
echo "=== Cleaning up old processes ==="
pkill -f "vllm.entrypoints" 2>/dev/null || true
pkill -f "vllm_miner.py" 2>/dev/null || true
pkill -f "real_miner.py" 2>/dev/null || true
pkill -f "hardened_gateway.py" 2>/dev/null || true
sleep 2

# ── Start vLLM server ────────────────────────────────────────────────────
if [ "$MODE" = "--miner-only" ] || [ "$MODE" = "--both" ]; then
    echo ""
    echo "=== Starting vLLM server (this may take a few minutes to load model) ==="

    VLLM_ARGS="--model $MODEL_NAME --port $VLLM_PORT --dtype auto --max-model-len 4096"

    # For 24GB GPUs, enable tensor parallelism if needed
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ "$GPU_COUNT" -gt 1 ]; then
        VLLM_ARGS="$VLLM_ARGS --tensor-parallel-size $GPU_COUNT"
    fi

    nohup python3 -m vllm.entrypoints.openai.api_server $VLLM_ARGS \
        > $LOG_DIR/vllm.log 2>&1 &
    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID"

    # Wait for vLLM to load model (can take 2-5 minutes)
    echo "Waiting for vLLM to load model (this may take several minutes)..."
    for i in $(seq 1 300); do
        if curl -sf "http://127.0.0.1:$VLLM_PORT/health" > /dev/null 2>&1; then
            echo "vLLM ready! (${i}s)"
            break
        fi
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "ERROR: vLLM process died. Check $LOG_DIR/vllm.log"
            tail -30 $LOG_DIR/vllm.log
            exit 1
        fi
        if [ $((i % 30)) -eq 0 ]; then
            echo "  Still loading... (${i}s)"
        fi
        sleep 1
    done

    echo ""
    echo "=== Starting miner (vLLM backend) ==="
    nohup python3 vllm_miner.py \
        --port "$MINER_PORT" \
        --vllm-url "http://127.0.0.1:$VLLM_PORT" \
        --model "$MODEL_NAME" \
        > $LOG_DIR/miner.log 2>&1 &
    MINER_PID=$!
    echo "Miner PID: $MINER_PID"

    echo "Waiting for miner..."
    for i in $(seq 1 60); do
        if curl -sf "http://127.0.0.1:$MINER_PORT/health" > /dev/null 2>&1; then
            echo "Miner ready! (${i}s)"
            break
        fi
        if ! kill -0 $MINER_PID 2>/dev/null; then
            echo "ERROR: Miner process died. Check $LOG_DIR/miner.log"
            tail -20 $LOG_DIR/miner.log
            exit 1
        fi
        sleep 1
    done
fi

if [ "$MODE" = "--validator-only" ] || [ "$MODE" = "--both" ]; then
    MINER_URL="http://127.0.0.1:$MINER_PORT"
    echo ""
    echo "=== Starting validator gateway ==="
    echo "  Miners: $MINER_URL"
    echo "  Model: $MODEL_NAME"

    EXTRA_ARGS=""
    if [ -n "$API_KEYS" ]; then
        EXTRA_ARGS="--api-keys $API_KEYS"
    fi
    if [ -n "$MONITORING_KEYS" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --monitoring-keys $MONITORING_KEYS"
    fi

    nohup python3 hardened_gateway.py \
        --miners "$MINER_URL" \
        --port "$VALIDATOR_PORT" \
        --model "$MODEL_NAME" \
        --epoch-length 300 \
        --synthetic-interval 10 \
        $EXTRA_ARGS \
        > $LOG_DIR/validator.log 2>&1 &
    VALIDATOR_PID=$!
    echo "Validator PID: $VALIDATOR_PID"

    echo "Waiting for validator..."
    for i in $(seq 1 120); do
        if curl -sf "http://127.0.0.1:$VALIDATOR_PORT/v1/health" > /dev/null 2>&1; then
            echo "Validator ready! (${i}s)"
            break
        fi
        if ! kill -0 $VALIDATOR_PID 2>/dev/null; then
            echo "ERROR: Validator process died. Check $LOG_DIR/validator.log"
            tail -20 $LOG_DIR/validator.log
            exit 1
        fi
        sleep 1
    done
fi

echo ""
echo "=== Deployment Complete ==="
echo "vLLM:      http://127.0.0.1:$VLLM_PORT/health"
echo "Miner:     http://127.0.0.1:$MINER_PORT/health"
echo "Validator: http://127.0.0.1:$VALIDATOR_PORT/v1/health"
echo "API:       http://127.0.0.1:$VALIDATOR_PORT/v1/chat/completions"
echo ""
echo "Logs:"
echo "  tail -f $LOG_DIR/vllm.log"
echo "  tail -f $LOG_DIR/miner.log"
echo "  tail -f $LOG_DIR/validator.log"
echo ""
echo "Test:"
echo "  curl -X POST http://127.0.0.1:$VALIDATOR_PORT/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"max_tokens\":32}'"
