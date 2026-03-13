#!/usr/bin/env python3
"""
Multi-GPU Miner — Load balanced across multiple GPU workers.

One registration, one endpoint, many GPUs:
  Client → Miner Endpoint → GPU Worker 0
                           → GPU Worker 1
                           → GPU Worker 2
                           → GPU Worker N

Each worker runs its own model instance (mock for PoC, vLLM for production).
The miner endpoint routes requests via round-robin and tracks which worker
handled each request_id for hidden state challenges.

Usage:
    # Single GPU (default)
    python multi_gpu_miner.py --port 8091

    # Multi-GPU (simulated with multiple worker ports)
    python multi_gpu_miner.py --port 8091 --num-workers 4

    # Multi-GPU with explicit worker ports
    python multi_gpu_miner.py --port 8091 --worker-ports 9001 9002 9003 9004
"""

import argparse
import asyncio
import json
import logging
import time
import uuid
from collections import OrderedDict

import aiohttp
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model import MockModel, ModelConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("multi-gpu-miner")


# ── Request/Response Models ──────────────────────────────────────────────────

class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 64
    request_id: str | None = None
    stream: bool = False
    # Inline challenge fields — bundled with inference to prevent endpoint fingerprinting
    challenge_layer: int | None = None
    challenge_token: int | None = None
    challenge_extra: list[list[int]] | None = None  # [[layer, token], ...]

class InferenceResponse(BaseModel):
    request_id: str
    text: str
    input_tokens: int
    output_tokens: int
    ttft_ms: float
    total_ms: float
    tokens_per_sec: float
    challenge_result: dict | None = None  # Inline challenge response

class HiddenStateRequest(BaseModel):
    request_id: str
    layer_index: int
    token_index: int

class HiddenStateResponse(BaseModel):
    request_id: str
    layer_index: int
    token_index: int
    hidden_state: list[float]
    latency_ms: float


# ── GPU Worker ───────────────────────────────────────────────────────────────

class GPUWorker:
    """
    A single GPU worker running a model instance.
    In production, this wraps a vLLM instance.
    For PoC, uses the mock model.
    """

    def __init__(self, worker_id: int, model: MockModel = None):
        self.worker_id = worker_id
        self.model = model or MockModel(ModelConfig())
        self.cache: OrderedDict[str, dict] = OrderedDict()
        self.cache_metadata: dict[str, dict] = {}
        self.max_cache = 500
        self.busy = False
        self.requests_served = 0

    def run_inference(self, prompt: str, max_tokens: int, request_id: str) -> dict:
        """Run model inference and cache hidden states."""
        t_start = time.perf_counter()
        result = self.model.generate(prompt, max_tokens)
        t_end = time.perf_counter()

        # Cache hidden states
        if len(self.cache) >= self.max_cache:
            evicted_id, _ = self.cache.popitem(last=False)
            self.cache_metadata.pop(evicted_id, None)

        self.cache[request_id] = result["hidden_states"]
        self.cache_metadata[request_id] = {
            "tokens": result["all_tokens"],
            "timestamp": time.time(),
        }

        total_ms = (t_end - t_start) * 1000
        output_count = len(result["output_tokens"])
        tps = output_count / max(total_ms / 1000, 0.001)

        self.requests_served += 1

        return {
            "text": result["text"],
            "input_tokens": len(result["input_tokens"]),
            "output_tokens": output_count,
            "ttft_ms": total_ms,  # mock: same as total
            "total_ms": total_ms,
            "tokens_per_sec": tps,
        }

    def get_hidden_state(self, request_id: str, layer_index: int, token_index: int) -> np.ndarray | None:
        """Retrieve cached hidden state."""
        if request_id not in self.cache:
            return None
        states = self.cache[request_id]
        if layer_index not in states:
            return None
        if token_index not in states[layer_index]:
            return None
        self.cache.move_to_end(request_id)
        return states[layer_index][token_index]

    def has_request(self, request_id: str) -> bool:
        return request_id in self.cache


# ── Multi-GPU Miner ─────────────────────────────────────────────────────────

class MultiGPUMiner:
    """
    Load-balanced miner across multiple GPU workers.
    Routes requests via round-robin and tracks which worker handled each request.
    """

    def __init__(self, num_workers: int = 1):
        self.workers = [GPUWorker(i) for i in range(num_workers)]
        self.request_routing: OrderedDict[str, int] = OrderedDict()
        self.max_routing_entries = 10000
        self._next_worker = 0
        self.total_requests = 0
        self.total_challenges = 0
        self.challenges_passed = 0

    def _select_worker(self) -> GPUWorker:
        """Round-robin worker selection."""
        worker = self.workers[self._next_worker % len(self.workers)]
        self._next_worker += 1
        return worker

    def _track_routing(self, request_id: str, worker_id: int):
        """Track which worker handled which request for challenge routing."""
        if len(self.request_routing) >= self.max_routing_entries:
            self.request_routing.popitem(last=False)
        self.request_routing[request_id] = worker_id

    def run_inference(self, request: InferenceRequest) -> InferenceResponse:
        """Route to a worker, run inference, cache hidden states."""
        request_id = request.request_id or str(uuid.uuid4())
        worker = self._select_worker()

        result = worker.run_inference(request.prompt, request.max_tokens, request_id)
        self._track_routing(request_id, worker.worker_id)
        self.total_requests += 1

        # Handle inline challenge if requested
        challenge_result = None
        if request.challenge_layer is not None and request.challenge_token is not None:
            challenge_result = self._serve_inline_challenge(
                worker, request_id, request.challenge_layer, request.challenge_token,
                request.challenge_extra,
            )

        log.info(
            f"Inference {request_id[:8]}... → worker {worker.worker_id} | "
            f"{result['input_tokens']} in + {result['output_tokens']} out | "
            f"{result['total_ms']:.1f}ms | {result['tokens_per_sec']:.0f} tok/s"
            f"{' +challenge' if challenge_result else ''}"
        )

        return InferenceResponse(
            request_id=request_id,
            text=result["text"],
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            ttft_ms=result["ttft_ms"],
            total_ms=result["total_ms"],
            tokens_per_sec=result["tokens_per_sec"],
            challenge_result=challenge_result,
        )

    def _serve_inline_challenge(
        self,
        worker: "GPUWorker",
        request_id: str,
        layer_index: int,
        token_index: int,
        extra_points: list[list[int]] | None = None,
    ) -> dict:
        """Serve hidden state challenge inline from the same worker that ran inference."""
        t_start = time.perf_counter()
        self.total_challenges += 1

        state = worker.get_hidden_state(request_id, layer_index, token_index)
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000

        if state is None:
            return {"error": "cache_miss", "latency_ms": latency_ms}

        self.challenges_passed += 1

        result = {
            "hidden_state": state.tolist(),
            "layer_index": layer_index,
            "token_index": token_index,
            "latency_ms": latency_ms,
        }

        if extra_points:
            extra_states = []
            for point in extra_points:
                if len(point) >= 2:
                    extra_state = worker.get_hidden_state(request_id, point[0], point[1])
                    if extra_state is not None:
                        extra_states.append({
                            "layer_index": point[0],
                            "token_index": point[1],
                            "hidden_state": extra_state.tolist(),
                        })
                    else:
                        extra_states.append({
                            "layer_index": point[0],
                            "token_index": point[1],
                            "error": "cache_miss",
                        })
            result["extra_states"] = extra_states

        return result

    def get_hidden_state(self, request: HiddenStateRequest) -> HiddenStateResponse:
        """Route challenge to the correct worker."""
        t_start = time.perf_counter()
        self.total_challenges += 1

        # Find which worker handled this request
        worker_id = self.request_routing.get(request.request_id)
        if worker_id is None:
            raise HTTPException(
                status_code=404,
                detail=f"No routing entry for request {request.request_id}"
            )

        worker = self.workers[worker_id]
        state = worker.get_hidden_state(request.request_id, request.layer_index, request.token_index)

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000

        if state is None:
            log.warning(
                f"Challenge MISS {request.request_id[:8]}... → worker {worker_id} | "
                f"layer={request.layer_index} pos={request.token_index}"
            )
            raise HTTPException(
                status_code=404,
                detail=f"No cached hidden state for request {request.request_id}"
            )

        self.challenges_passed += 1

        log.info(
            f"Challenge HIT {request.request_id[:8]}... → worker {worker_id} | "
            f"layer={request.layer_index} pos={request.token_index} | {latency_ms:.2f}ms"
        )

        return HiddenStateResponse(
            request_id=request.request_id,
            layer_index=request.layer_index,
            token_index=request.token_index,
            hidden_state=state.tolist(),
            latency_ms=latency_ms,
        )


# ── FastAPI App ──────────────────────────────────────────────────────────────

def create_miner_app(num_workers: int = 1) -> tuple[FastAPI, MultiGPUMiner]:
    miner = MultiGPUMiner(num_workers=num_workers)
    app = FastAPI(title="Multi-GPU Inference Miner")

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": miner.workers[0].model.config.name,
            "num_workers": len(miner.workers),
            "total_requests": miner.total_requests,
            "total_challenges": miner.total_challenges,
            "challenges_passed": miner.challenges_passed,
            "worker_stats": [
                {"worker_id": w.worker_id, "requests": w.requests_served, "cache_size": len(w.cache)}
                for w in miner.workers
            ],
        }

    @app.post("/inference", response_model=InferenceResponse)
    async def inference(request: InferenceRequest):
        return miner.run_inference(request)

    @app.post("/hidden_state", response_model=HiddenStateResponse)
    async def hidden_state(request: HiddenStateRequest):
        return miner.get_hidden_state(request)

    return app, miner


# Global for module-level imports in tests
_app, _miner = create_miner_app(num_workers=1)
app = _app
miner = _miner


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Inference Miner")
    parser.add_argument("--port", type=int, default=8091, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of GPU workers")
    args = parser.parse_args()

    global app, miner, _app, _miner
    _app, _miner = create_miner_app(num_workers=args.num_workers)
    app = _app
    miner = _miner

    log.info(f"Starting multi-GPU miner on {args.host}:{args.port} ({args.num_workers} workers)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
