# Constantinople Miner — vLLM + Hidden State Extraction
# High-throughput GPU miner for Subnet 97
#
# Build:
#   docker build -t constantinople-miner miner/
#
# Run:
#   docker run --gpus all -p 8091:8091 constantinople-miner \
#     --model Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 2

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

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
    torch \
    transformers \
    accelerate \
    vllm

# Copy miner code
COPY vllm_miner.py ./
COPY real_miner.py ./
COPY model.py ./

HEALTHCHECK --interval=30s --timeout=10s --retries=5 --start-period=180s \
    CMD curl -f http://localhost:8091/health || exit 1

EXPOSE 8091

ENTRYPOINT ["python3", "vllm_miner.py"]
CMD ["--port", "8091", "--model", "Qwen/Qwen2.5-7B-Instruct"]
