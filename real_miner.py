#!/usr/bin/env python3
"""
Real inference miner using vLLM with hidden state extraction.

Serves LLM inference and caches hidden states from the last transformer layer
for verification challenges. Uses a small model (Qwen2.5-1.5B) that fits
easily on a single GPU.

Endpoints:
    POST /inference     — Run real LLM inference, cache hidden states
    POST /hidden_state  — Return cached hidden state at (layer, position)
    GET  /health        — Health check with model info
"""

import argparse
import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import OrderedDict

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("real_miner")


# ── Request/Response Models ──────────────────────────────────────────────────

class InferenceRequest(BaseModel):
    prompt: str = ""
    messages: list[dict] | None = None  # Chat messages for chat template
    max_tokens: int = 128
    request_id: str | None = None
    # Sampling parameters — forwarded from OpenAI-compatible gateway
    temperature: float | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: list[str] | str | None = None
    # Inline challenge fields — challenges are bundled with inference to prevent
    # miners from distinguishing challenge traffic via separate endpoints.
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
    all_token_ids: list[int] | None = None  # Full token sequence for verification
    # Inline challenge response — returned when challenge fields were in the request
    challenge_result: dict | None = None  # {hidden_state, layer_index, token_index, latency_ms, extra_states}

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


# ── Hidden State Cache ───────────────────────────────────────────────────────

class HiddenStateCache:
    """LRU cache for hidden states from inference runs."""

    def __init__(self, max_requests: int = 200):
        self.max_requests = max_requests
        self.cache: OrderedDict[str, dict] = OrderedDict()

    def store(self, request_id: str, hidden_states: dict):
        """Store hidden states. hidden_states = {layer_idx: tensor(seq_len, hidden_dim)}"""
        if len(self.cache) >= self.max_requests:
            self.cache.popitem(last=False)
        self.cache[request_id] = hidden_states

    def get(self, request_id: str, layer_index: int, token_index: int) -> np.ndarray | None:
        if request_id not in self.cache:
            return None
        states = self.cache[request_id]
        if layer_index not in states:
            return None
        layer_tensor = states[layer_index]
        if token_index >= layer_tensor.shape[0]:
            return None
        self.cache.move_to_end(request_id)
        return layer_tensor[token_index].numpy()

    @property
    def size(self):
        return len(self.cache)


# ── Real Model Wrapper ───────────────────────────────────────────────────────

class RealModelMiner:
    """Miner using a real transformer model with hidden state extraction."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", cache_size: int = 200):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        self.model_name = model_name
        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_dim = self.model.config.hidden_size

        self.cache = HiddenStateCache(max_requests=cache_size)
        self.total_requests = 0
        self.total_challenges = 0
        self.challenges_passed = 0

        log.info(f"Model loaded: {model_name} | {self.num_layers} layers | hidden_dim={self.hidden_dim}")
        log.info(f"Device: {next(self.model.parameters()).device}")

    def _build_generate_kwargs(self, inputs: dict, request: InferenceRequest) -> dict:
        """Build HuggingFace generate() kwargs from an InferenceRequest, wiring sampling params."""
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=request.max_tokens,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        temp = request.temperature
        do_sample = (temp is not None and temp > 0) or request.top_p is not None
        if do_sample:
            gen_kwargs["do_sample"] = True
            if temp is not None:
                gen_kwargs["temperature"] = max(temp, 1e-7)  # Clamp to avoid div-by-zero
            if request.top_p is not None:
                gen_kwargs["top_p"] = request.top_p
        else:
            gen_kwargs["do_sample"] = False
        # Map OpenAI frequency/presence penalty to HuggingFace repetition_penalty.
        # HF uses a multiplicative repetition_penalty (>1 = penalize, 1 = no effect).
        # OpenAI uses additive penalties in [-2, 2]. Approximate: 1.0 + max(freq, pres).
        fp = request.frequency_penalty or 0.0
        pp = request.presence_penalty or 0.0
        combined = max(fp, pp)
        if combined > 0:
            gen_kwargs["repetition_penalty"] = 1.0 + combined
        # Stop sequences
        stop = request.stop
        if stop:
            stop_list = [stop] if isinstance(stop, str) else list(stop)
            if stop_list:
                from transformers import StoppingCriteria, StoppingCriteriaList
                tokenizer = self.tokenizer

                class StopOnSequences(StoppingCriteria):
                    def __init__(self, stops):
                        self.stop_ids = [tokenizer.encode(s, add_special_tokens=False) for s in stops if s]
                    def __call__(self, input_ids, scores, **kwargs):
                        for sid in self.stop_ids:
                            if len(sid) > 0 and input_ids[0][-len(sid):].tolist() == sid:
                                return True
                        return False

                gen_kwargs["stopping_criteria"] = StoppingCriteriaList([StopOnSequences(stop_list)])
        return gen_kwargs

    @torch.no_grad()
    def run_inference(self, request: InferenceRequest) -> InferenceResponse:
        request_id = request.request_id or str(uuid.uuid4())

        # Tokenize — use chat template if messages provided
        if request.messages:
            text = self.tokenizer.apply_chat_template(
                request.messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        else:
            inputs = self.tokenizer(request.prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        t_start = time.perf_counter()

        # Generate with hidden states output
        gen_kwargs = self._build_generate_kwargs(inputs, request)
        outputs = self.model.generate(**gen_kwargs)

        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000

        # Extract generated tokens
        generated_ids = outputs.sequences[0][input_len:]
        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        output_len = len(generated_ids)

        # Extract and cache hidden states for the PREFILL step (input tokens)
        # outputs.hidden_states is a tuple:
        #   - First element: prefill hidden states (tuple of layer tensors)
        #   - Remaining elements: one per generated token (each is tuple of layer tensors)
        hidden_states_cache = {}

        if outputs.hidden_states:
            # Prefill: outputs.hidden_states[0] is tuple of (num_layers+1) tensors
            # Each tensor shape: (batch=1, seq_len, hidden_dim)
            prefill_states = outputs.hidden_states[0]
            for layer_idx in range(self.num_layers):
                # +1 because index 0 is embedding, layers start at index 1
                layer_tensor = prefill_states[layer_idx + 1][0].cpu().float()  # (seq_len, hidden_dim)
                hidden_states_cache[layer_idx] = layer_tensor

            # Also cache generated token hidden states
            # outputs.hidden_states[step] for step >= 1 has shape (1, 1, hidden_dim) per layer
            for step_idx in range(1, len(outputs.hidden_states)):
                step_states = outputs.hidden_states[step_idx]
                for layer_idx in range(self.num_layers):
                    step_tensor = step_states[layer_idx + 1][0].cpu().float()  # (1, hidden_dim)
                    # Append to existing layer cache
                    hidden_states_cache[layer_idx] = torch.cat(
                        [hidden_states_cache[layer_idx], step_tensor], dim=0
                    )

        self.cache.store(request_id, hidden_states_cache)
        self.total_requests += 1

        ttft_ms = total_ms / max(output_len, 1)  # approximate
        tps = output_len / max(total_ms / 1000, 0.001)

        # Handle inline challenge if requested
        challenge_result = None
        if request.challenge_layer is not None and request.challenge_token is not None:
            challenge_result = self._serve_inline_challenge(
                request_id, request.challenge_layer, request.challenge_token,
                request.challenge_extra,
            )

        log.info(
            f"Inference {request_id[:8]}... | "
            f"{input_len} in + {output_len} out | "
            f"{total_ms:.1f}ms | {tps:.0f} tok/s | "
            f"cache: {self.cache.size}"
            f"{' +challenge' if challenge_result else ''}"
        )

        return InferenceResponse(
            request_id=request_id,
            text=output_text,
            input_tokens=input_len,
            output_tokens=output_len,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            tokens_per_sec=tps,
            all_token_ids=outputs.sequences[0].tolist(),
            challenge_result=challenge_result,
        )

    @torch.no_grad()
    def run_inference_streaming(self, request: InferenceRequest):
        """
        Generator that yields SSE events with tokens as they are generated.
        Uses TextIteratorStreamer for real token-by-token streaming.
        Hidden states are captured from the full generate() call (via a background thread).

        Yields SSE-formatted strings: "data: {json}\n\n"
        """
        from threading import Thread
        from transformers import TextIteratorStreamer

        request_id = request.request_id or str(uuid.uuid4())

        # Tokenize
        if request.messages:
            text = self.tokenizer.apply_chat_template(
                request.messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        else:
            inputs = self.tokenizer(request.prompt, return_tensors="pt").to(self.model.device)

        input_len = inputs["input_ids"].shape[1]

        # Set up streaming
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # generate() blocks, so run in thread — reuse shared param builder
        generate_kwargs = self._build_generate_kwargs(inputs, request)
        generate_kwargs["streamer"] = streamer

        # We need the outputs object for hidden states, so we capture it
        generation_result = {}

        def generate_thread():
            try:
                result = self.model.generate(**generate_kwargs)
                generation_result["outputs"] = result
            except Exception as e:
                generation_result["error"] = str(e)

        t_start = time.perf_counter()
        thread = Thread(target=generate_thread)
        thread.start()

        # Yield tokens as they arrive from the streamer
        first_token = True
        generated_text = ""
        for text_chunk in streamer:
            if first_token:
                ttft_ms = (time.perf_counter() - t_start) * 1000
                first_token = False
            generated_text += text_chunk
            yield f"data: {json.dumps({'request_id': request_id, 'token': text_chunk, 'finish_reason': None})}\n\n"

        thread.join()
        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000

        # Extract hidden states and cache them
        outputs = generation_result.get("outputs")
        if outputs and outputs.hidden_states:
            hidden_states_cache = {}
            prefill_states = outputs.hidden_states[0]
            for layer_idx in range(self.num_layers):
                hidden_states_cache[layer_idx] = prefill_states[layer_idx + 1][0].cpu().float()
            for step_idx in range(1, len(outputs.hidden_states)):
                step_states = outputs.hidden_states[step_idx]
                for layer_idx in range(self.num_layers):
                    step_tensor = step_states[layer_idx + 1][0].cpu().float()
                    hidden_states_cache[layer_idx] = torch.cat(
                        [hidden_states_cache[layer_idx], step_tensor], dim=0
                    )
            self.cache.store(request_id, hidden_states_cache)

        all_token_ids = outputs.sequences[0].tolist() if outputs else []
        output_len = len(all_token_ids) - input_len if all_token_ids else 0
        tps = output_len / max(total_ms / 1000, 0.001)
        if first_token:
            ttft_ms = total_ms  # no tokens generated

        self.total_requests += 1

        # Handle inline challenge if requested (same as non-streaming path)
        challenge_result = None
        if request.challenge_layer is not None and request.challenge_token is not None:
            challenge_result = self._serve_inline_challenge(
                request_id, request.challenge_layer, request.challenge_token,
                request.challenge_extra,
            )

        # Final SSE event with metadata (includes challenge result if requested)
        final_meta = {
            'request_id': request_id, 'token': '', 'finish_reason': 'stop',
            'input_tokens': input_len, 'output_tokens': output_len,
            'ttft_ms': round(ttft_ms, 2), 'total_ms': round(total_ms, 2),
            'tokens_per_sec': round(tps, 1), 'all_token_ids': all_token_ids,
        }
        if challenge_result is not None:
            final_meta['challenge_result'] = challenge_result
        yield f"data: {json.dumps(final_meta)}\n\n"
        yield "data: [DONE]\n\n"

        log.info(
            f"Stream {request_id[:8]}... | "
            f"{input_len} in + {output_len} out | "
            f"{total_ms:.1f}ms | {tps:.0f} tok/s"
            f"{' +challenge' if challenge_result else ''}"
        )

    def _serve_inline_challenge(
        self,
        request_id: str,
        layer_index: int,
        token_index: int,
        extra_points: list[list[int]] | None = None,
    ) -> dict:
        """Serve challenge inline after inference — no separate endpoint needed."""
        t_start = time.perf_counter()
        state = self.cache.get(request_id, layer_index, token_index)
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000
        self.total_challenges += 1

        if state is None:
            log.warning(f"Inline challenge MISS {request_id[:8]}... | layer={layer_index} pos={token_index}")
            return {"error": "cache_miss", "latency_ms": latency_ms}

        self.challenges_passed += 1
        norm = np.linalg.norm(state)
        if norm > 0:
            state = state / norm

        result = {
            "hidden_state": state.tolist(),
            "layer_index": layer_index,
            "token_index": token_index,
            "latency_ms": latency_ms,
        }

        # Serve extra challenge points if requested
        if extra_points:
            extra_states = []
            for point in extra_points:
                if len(point) >= 2:
                    extra_state = self.cache.get(request_id, point[0], point[1])
                    if extra_state is not None:
                        norm_e = np.linalg.norm(extra_state)
                        if norm_e > 0:
                            extra_state = extra_state / norm_e
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
        t_start = time.perf_counter()
        state = self.cache.get(request.request_id, request.layer_index, request.token_index)
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000

        self.total_challenges += 1

        if state is None:
            log.warning(f"Challenge MISS {request.request_id[:8]}... | layer={request.layer_index} pos={request.token_index}")
            raise HTTPException(status_code=404, detail=f"No cached hidden state for request {request.request_id}")

        self.challenges_passed += 1

        # Normalize for consistency
        norm = np.linalg.norm(state)
        if norm > 0:
            state = state / norm

        log.info(f"Challenge HIT {request.request_id[:8]}... | layer={request.layer_index} pos={request.token_index} | {latency_ms:.2f}ms")

        return HiddenStateResponse(
            request_id=request.request_id,
            layer_index=request.layer_index,
            token_index=request.token_index,
            hidden_state=state.tolist(),
            latency_ms=latency_ms,
        )


# ── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(title="Real Inference Miner")
miner: RealModelMiner | None = None


@app.get("/health")
async def health():
    # Note: cache_size, total_challenges, challenges_passed intentionally omitted
    # to prevent validators/observers from fingerprinting challenge vs organic traffic
    # or inferring probe rates from cache size oscillations.
    return {
        "status": "ok",
        "model": miner.model_name if miner else "not loaded",
        "num_layers": miner.num_layers if miner else 0,
        "hidden_dim": miner.hidden_dim if miner else 0,
        "total_requests": miner.total_requests if miner else 0,
    }


@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    return miner.run_inference(request)


@app.post("/inference/stream")
async def inference_stream(request: InferenceRequest):
    """SSE streaming inference endpoint. Streams tokens as they are generated."""
    return StreamingResponse(
        miner.run_inference_streaming(request),
        media_type="text/event-stream",
    )


@app.post("/hidden_state", response_model=HiddenStateResponse)
async def hidden_state(request: HiddenStateRequest):
    return miner.get_hidden_state(request)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    global miner

    parser = argparse.ArgumentParser(description="Real Inference Miner")
    parser.add_argument("--port", type=int, default=8091, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="HuggingFace model name")
    parser.add_argument("--cache-size", type=int, default=200, help="Max cached requests")
    args = parser.parse_args()

    miner = RealModelMiner(model_name=args.model, cache_size=args.cache_size)
    log.info(f"Starting real miner on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
