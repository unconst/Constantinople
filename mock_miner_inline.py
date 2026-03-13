#!/usr/bin/env python3
"""
Mock miner with inline challenge support for gateway integration testing.
Supports the hardened gateway's inline challenge protocol.
"""

import asyncio
import os
import sys
import uuid

import numpy as np
from aiohttp import web

sys.path.insert(0, os.path.dirname(__file__))
from model import MockModel, ModelConfig


class InlineMockMiner:
    def __init__(self, port: int = 9502):
        self.port = port
        self.model = MockModel(ModelConfig())
        self.app = web.Application()
        self.app.router.add_post("/inference", self.handle_inference)
        self.app.router.add_post("/inference/stream", self.handle_stream)
        self.app.router.add_get("/health", self.handle_health)

    async def handle_inference(self, request):
        data = await request.json()
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_tokens", 64)
        request_id = data.get("request_id", str(uuid.uuid4()))

        # Generate tokens
        result = self.model.generate(prompt, max_tokens)
        all_tokens = result["all_tokens"]
        n_input = len(result["input_tokens"])
        n_output = len(result["output_tokens"])

        ttft_ms = 15.0 + np.random.uniform(0, 10)
        tps = min(n_output / 0.05, 120.0)

        response = {
            "text": result["text"],
            "request_id": request_id,
            "input_tokens": n_input,
            "output_tokens": n_output,
            "ttft_ms": ttft_ms,
            "tokens_per_sec": tps,
            "all_tokens": all_tokens,
            "all_token_ids": all_tokens,
        }

        # Handle inline challenge
        challenge_layer = data.get("challenge_layer")
        challenge_token = data.get("challenge_token")
        if challenge_layer is not None and challenge_token is not None:
            challenge_result = {}
            try:
                if challenge_token < len(all_tokens):
                    hs = self.model.compute_hidden_state_at(
                        all_tokens, challenge_layer, challenge_token
                    )
                    challenge_result = {
                        "hidden_state": hs.tolist(),
                        "layer_index": challenge_layer,
                        "token_index": challenge_token,
                    }

                    # Handle extra challenge points
                    extra_points = data.get("challenge_extra", [])
                    if extra_points:
                        extra_states = []
                        for pt in extra_points:
                            lyr, tok = pt[0], pt[1]
                            if tok < len(all_tokens):
                                es = self.model.compute_hidden_state_at(all_tokens, lyr, tok)
                                extra_states.append({
                                    "hidden_state": es.tolist(),
                                    "layer_index": lyr,
                                    "token_index": tok,
                                })
                            else:
                                extra_states.append({"error": "out of bounds"})
                        challenge_result["extra_states"] = extra_states
                else:
                    challenge_result = {"error": "token index out of bounds"}
            except Exception as e:
                challenge_result = {"error": str(e)}

            response["challenge_result"] = challenge_result

        return web.json_response(response)

    async def handle_stream(self, request):
        """Streaming endpoint — simplified for testing."""
        data = await request.json()
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_tokens", 64)
        request_id = data.get("request_id", str(uuid.uuid4()))

        result = self.model.generate(prompt, max_tokens)
        all_tokens = result["all_tokens"]

        challenge_layer = data.get("challenge_layer")
        challenge_token = data.get("challenge_token")

        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={"Content-Type": "text/event-stream"},
        )
        await response.prepare(request)

        # Send tokens one at a time with small delay for realism
        import json as _json
        for i, token_text in enumerate(result["text"].split()):
            chunk = {
                "token": token_text,
                "token_id": result["output_tokens"][i] if i < len(result["output_tokens"]) else 0,
            }
            await response.write(f"data: {_json.dumps(chunk)}\n\n".encode())
            await asyncio.sleep(0.01)  # ~10ms per token

        # Send final metadata (finish_reason="stop" is how the gateway identifies this event)
        meta = {
            "finish_reason": "stop",
            "request_id": request_id,
            "all_token_ids": all_tokens,
            "prompt_tokens": len(result["input_tokens"]),
            "input_tokens": len(result["input_tokens"]),
            "output_tokens": len(result["output_tokens"]),
            "ttft_ms": 15.0 + np.random.uniform(0, 10),
            "tokens_per_sec": min(len(result["output_tokens"]) / 0.05, 120.0),
            "total_ms": 50.0 + np.random.uniform(0, 20),
        }

        # Inline challenge for streaming
        if challenge_layer is not None and challenge_token is not None:
            if challenge_token < len(all_tokens):
                hs = self.model.compute_hidden_state_at(all_tokens, challenge_layer, challenge_token)
                challenge_result = {
                    "hidden_state": hs.tolist(),
                    "layer_index": challenge_layer,
                    "token_index": challenge_token,
                }

                # Handle extra challenge points (multi-point verification)
                extra_points = data.get("challenge_extra", [])
                if extra_points:
                    extra_states = []
                    for pt in extra_points:
                        lyr, tok = pt[0], pt[1]
                        if tok < len(all_tokens):
                            es = self.model.compute_hidden_state_at(all_tokens, lyr, tok)
                            extra_states.append({
                                "hidden_state": es.tolist(),
                                "layer_index": lyr,
                                "token_index": tok,
                            })
                        else:
                            extra_states.append({"error": "out of bounds"})
                    challenge_result["extra_states"] = extra_states

                meta["challenge_result"] = challenge_result

        await response.write(f"data: {_json.dumps(meta)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        return response

    async def handle_health(self, request):
        return web.json_response({"status": "ok", "model": "mock-inline"})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9502)
    args = parser.parse_args()

    miner = InlineMockMiner(port=args.port)
    web.run_app(miner.app, host="127.0.0.1", port=args.port)
