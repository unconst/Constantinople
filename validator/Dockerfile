# Constantinople Validator — Subnet 97
# Hardened inference gateway with hidden state verification
#
# Build:
#   docker build -t constantinople-validator validator/
#
# Run (with chain integration):
#   docker run --gpus all -p 8080:8080 \
#     -v ~/.bittensor/wallets:/root/.bittensor/wallets:ro \
#     constantinople-validator \
#     --discover --wallet validator --hotkey default --netuid 97 --network finney

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    aiohttp \
    numpy \
    pydantic \
    boto3 \
    bittensor \
    torch \
    transformers \
    accelerate

# Copy validator code
COPY hardened_gateway.py ./
COPY hardened_scoring.py ./
COPY challenge_engine.py ./
COPY kv_cache_prober.py ./
COPY collusion_detector.py ./
COPY r2_publisher.py ./
COPY model.py ./

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=120s \
    CMD curl -f http://localhost:8080/v1/health || exit 1

EXPOSE 8080

ENTRYPOINT ["python3", "hardened_gateway.py"]
CMD ["--discover", "--wallet", "validator", "--hotkey", "default", "--netuid", "97", "--network", "finney", "--port", "8080", "--epoch-length", "300", "--synthetic-interval", "10"]
