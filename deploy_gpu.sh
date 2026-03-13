#!/bin/bash
# Deploy inference subnet on a Lium GPU pod (direct, no Docker)
#
# Usage:
#   scp deploy_gpu.sh user@pod:/workspace/
#   ssh user@pod 'bash /workspace/deploy_gpu.sh'
#
# Or run locally on the pod:
#   bash deploy_gpu.sh [--miner-only|--validator-only|--both]
#
# Environment:
#   MODEL_NAME    — HuggingFace model (default: Qwen/Qwen2.5-1.5B-Instruct)
#   MINER_PORT    — Miner port (default: 8091)
#   VALIDATOR_PORT — Validator port (default: 8080)
#   API_KEYS      — Comma-separated API keys for gateway auth (default: none)
#   MONITORING_KEYS — Comma-separated keys for monitoring endpoints (default: none)
#   WORK_DIR      — Working directory (default: /workspace/inference_subnet)
#   LOG_DIR       — Log directory (default: /workspace)

set -e

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
MINER_PORT="${MINER_PORT:-8091}"
VALIDATOR_PORT="${VALIDATOR_PORT:-8080}"
MODE="${1:---both}"
WORK_DIR="${WORK_DIR:-/workspace/inference_subnet}"
LOG_DIR="${LOG_DIR:-/workspace}"

echo "=== Inference Subnet GPU Deploy ==="
echo "Model: $MODEL_NAME"
echo "Mode: $MODE"

# ── Install deps ──────────────────────────────────────────────────────────
echo ""
echo "=== Installing dependencies ==="
pip install -q \
    fastapi uvicorn[standard] aiohttp numpy pydantic boto3 \
    torch transformers accelerate 2>&1 | tail -5

echo "Dependencies installed."

# ── Check GPU ─────────────────────────────────────────────────────────────
echo ""
echo "=== GPU Status ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "No GPU detected (mock mode)"

# ── Copy code ─────────────────────────────────────────────────────────────
mkdir -p "$WORK_DIR"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Copy all Python modules
for f in model.py hardened_scoring.py challenge_engine.py hardened_gateway.py \
         r2_publisher.py kv_cache_prober.py collusion_detector.py \
         real_miner.py multi_gpu_miner.py; do
    if [ -f "$SCRIPT_DIR/$f" ]; then
        cp "$SCRIPT_DIR/$f" "$WORK_DIR/"
    fi
done

cd "$WORK_DIR"

# ── Start services ────────────────────────────────────────────────────────
echo ""
echo "=== Starting services ==="

# Kill any existing processes
pkill -f "real_miner.py" 2>/dev/null || true
pkill -f "hardened_gateway.py" 2>/dev/null || true
sleep 1

if [ "$MODE" = "--miner-only" ] || [ "$MODE" = "--both" ]; then
    echo "Starting miner on port $MINER_PORT with model $MODEL_NAME ..."
    nohup python3 real_miner.py \
        --port "$MINER_PORT" \
        --model "$MODEL_NAME" \
        --cache-size 200 \
        > $LOG_DIR/miner.log 2>&1 &
    MINER_PID=$!
    echo "Miner PID: $MINER_PID"

    # Wait for miner health
    echo "Waiting for miner to load model..."
    for i in $(seq 1 120); do
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
    echo "Starting validator on port $VALIDATOR_PORT ..."
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

    # Wait for validator health
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
echo "Miner:     http://127.0.0.1:$MINER_PORT/health"
echo "Validator: http://127.0.0.1:$VALIDATOR_PORT/v1/health"
echo "API:       http://127.0.0.1:$VALIDATOR_PORT/v1/chat/completions"
echo "Scores:    http://127.0.0.1:$VALIDATOR_PORT/v1/scoreboard"
echo ""
echo "Logs:"
echo "  tail -f $LOG_DIR/miner.log"
echo "  tail -f $LOG_DIR/validator.log"
echo ""
echo "Test:"
echo "  curl -X POST http://127.0.0.1:$VALIDATOR_PORT/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"max_tokens\":32}'"
