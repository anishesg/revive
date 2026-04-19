"""Pipeline stage worker.

Loads a slice [layer_start, layer_end) of a causal LM and exposes:

  POST /forward   — run forward through our layers, return hidden state
                    (or sampled token if this is the last stage)
  POST /reset     — drop KV cache for a seq_id
  GET  /info      — stage metadata
  GET  /health    — liveness

Per-seq KV cache lives locally. Only hidden states traverse the wire.

Run:
  python -m pipeline.worker \\
      --model Qwen/Qwen3-0.6B \\
      --layer-start 0 --layer-end 14 \\
      --first --port 50100
"""
from __future__ import annotations
import argparse
import asyncio
import logging
import time
from typing import Optional

import numpy as np
import torch
from aiohttp import web
from transformers import AutoConfig, AutoModelForCausalLM

from .protocol import Frame, Response

log = logging.getLogger("worker")


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class PipelineStage:
    def __init__(self, model_name: str, layer_start: int, layer_end: int,
                 is_first: bool, is_last: bool, device: Optional[str] = None):
        self.model_name = model_name
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.is_first = is_first
        self.is_last = is_last
        self.device = device or pick_device()
        # MPS float16 has some gotchas with complex ops, but Qwen3 is standard. Use fp16 on GPU, fp32 on CPU.
        self.dtype = torch.float32 if self.device == "cpu" else torch.float16

        log.info(f"loading {model_name} on {self.device} dtype={self.dtype}")
        t0 = time.time()
        full = AutoModelForCausalLM.from_pretrained(model_name, dtype=self.dtype, low_cpu_mem_usage=True)
        full.eval()
        self.config = full.config
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers
        if layer_end > self.num_layers:
            raise ValueError(f"layer_end={layer_end} > num_layers={self.num_layers}")

        # Extract the slice. Keep what we need; let GC drop the rest.
        self.embed = full.model.embed_tokens.to(self.device) if is_first else None
        kept = torch.nn.ModuleList(
            [layer.to(self.device) for layer in full.model.layers[layer_start:layer_end]]
        )
        # Re-index extracted layers so the internal KV-cache layer_idx is local.
        for local_idx, layer in enumerate(kept):
            layer.self_attn.layer_idx = local_idx
        self.layers = kept
        # rotary_emb is stateless (precomputed inv_freq), move to device.
        self.rotary_emb = full.model.rotary_emb.to(self.device)
        self.norm = full.model.norm.to(self.device) if is_last else None
        self.lm_head = full.lm_head.to(self.device) if is_last else None

        # Drop references to the full model so Python can free weights we don't own.
        del full
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # Per-seq KV cache. We keep one dict-of-lists per seq_id using the
        # Transformers DynamicCache API indexed by our local layer index.
        from transformers import DynamicCache  # lazy import: version-dependent
        self._DynamicCache = DynamicCache
        self.caches: dict[str, object] = {}
        self.positions: dict[str, int] = {}  # tokens-already-seen counter
        # Track recently-generated tokens per seq_id for presence penalty —
        # this is the single biggest lever against "Paris. Paris. Paris."
        # collapse on small Qwen3 models.
        self.recent_tokens: dict[str, list[int]] = {}
        self._recent_window = 128

        load_s = time.time() - t0
        log.info(
            f"stage ready: layers [{layer_start}..{layer_end}) of {self.num_layers}, "
            f"first={is_first} last={is_last} H={self.hidden_size} load={load_s:.2f}s"
        )

    def reset(self, seq_id: str):
        self.caches.pop(seq_id, None)
        self.positions.pop(seq_id, None)
        self.recent_tokens.pop(seq_id, None)

    def cache_for(self, seq_id: str):
        cache = self.caches.get(seq_id)
        if cache is None:
            # Transformers 5.x DynamicCache needs a config to pre-allocate layer
            # slots — otherwise `update()` silently no-ops on lazy init paths and
            # prefill produces right-output-but-empty-cache, breaking decode.
            cache = self._DynamicCache(config=self.config)
            self.caches[seq_id] = cache
            self.positions[seq_id] = 0
        return cache

    @torch.inference_mode()
    def forward(self, frame: Frame) -> Response:
        t0 = time.time()
        cache = self.cache_for(frame.seq_id)
        cache_len = cache.get_seq_length() if cache is not None and len(cache) > 0 else 0

        if self.is_first:
            # Incoming tensor is int32 token ids, shape [T]
            token_ids = torch.from_numpy(frame.tensor.astype(np.int64)).to(self.device).unsqueeze(0)  # [1, T]
            T = token_ids.shape[1]
            hidden = self.embed(token_ids)  # [1, T, H]
        else:
            hidden = torch.from_numpy(frame.tensor).to(device=self.device, dtype=self.dtype)  # [1, T, H]
            T = hidden.shape[1]

        # position_ids come from the frame (coordinator knows absolute positions)
        position_ids = torch.tensor(frame.positions, device=self.device, dtype=torch.long).unsqueeze(0)
        assert position_ids.shape[1] == T, f"positions {position_ids.shape} vs T={T}"

        # Precompute RoPE (cos, sin) — stateless, same function every stage would run.
        position_embeddings = self.rotary_emb(hidden, position_ids)

        # Attention mask: pass None and let the SDPA / FlashAttention kernels use
        # their native causal handling. Qwen3Attention reads position_embeddings +
        # cache_position and the cache's stored keys handle the history correctly.
        attention_mask = None
        cache_position = torch.arange(cache_len, cache_len + T, device=self.device, dtype=torch.long)

        for layer in self.layers:
            out = layer(
                hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,  # transformers 5.x renamed this (plural)
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            # Transformers 5.x Qwen3DecoderLayer returns the hidden tensor directly,
            # but older/other models return a (hidden, ...) tuple. Handle both.
            hidden = out[0] if isinstance(out, tuple) else out

        self.positions[frame.seq_id] = self.positions.get(frame.seq_id, 0) + T

        if self.is_last:
            hidden = self.norm(hidden)
            # Only need the last position's logits for next-token sampling
            logits = self.lm_head(hidden[:, -1:, :])  # [1, 1, V]
            token_id, eos = self._sample(logits, frame)
            dt = (time.time() - t0) * 1000
            return Response(
                seq_id=frame.seq_id,
                shape=[],
                dtype="int32",
                tensor=b"",
                token_id=int(token_id),
                eos=bool(eos),
                latency_ms=dt,
                tokens_per_second=T * 1000 / max(dt, 1e-6),
            )
        else:
            out_arr = hidden.detach().to(torch.float16).cpu().numpy()
            dt = (time.time() - t0) * 1000
            return Response(
                seq_id=frame.seq_id,
                shape=list(out_arr.shape),
                dtype="float16",
                tensor=out_arr.tobytes(),
                latency_ms=dt,
                tokens_per_second=T * 1000 / max(dt, 1e-6),
            )

    def _sample(self, logits: torch.Tensor, frame: Frame) -> tuple[int, bool]:
        """Sample from logits [1, 1, V] using Qwen3-recommended params:
        repetition penalty (presence) → top_k → top_p → min_p → temperature
        → multinomial. Matches vLLM / Qwen team recommendations."""
        logits = logits[0, -1, :].float()  # [V]

        # (1) Repetition / presence penalty — applied BEFORE temperature so
        #     it meaningfully shifts the ranking even at low temps. This is
        #     the key fix for the "Paris. Paris. Paris." death spiral on
        #     Qwen3-0.6B.
        recent = self.recent_tokens.get(frame.seq_id) or []
        if recent:
            penalty = 1.15    # 1.0 = off, ~1.1–1.3 is typical
            seen = torch.tensor(list(set(recent[-self._recent_window:])),
                                 device=logits.device, dtype=torch.long)
            sel = logits.index_select(0, seen)
            # Multiplicative penalty with sign-aware branch (standard HF behavior)
            sel = torch.where(sel > 0, sel / penalty, sel * penalty)
            logits = logits.index_copy(0, seen, sel)

        # (2) Temperature
        temp = max(frame.temperature, 1e-5)
        logits = logits / temp

        # (3) top_k
        if frame.top_k and frame.top_k > 0:
            topk = min(frame.top_k, logits.size(-1))
            vals, idx = torch.topk(logits, topk)
            mask = torch.full_like(logits, float("-inf"))
            mask.scatter_(-1, idx, vals)
            logits = mask

        # (4) top_p (nucleus)
        if frame.top_p and frame.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumprobs = torch.cumsum(probs, dim=-1)
            cutoff = cumprobs > frame.top_p
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False
            sorted_logits[cutoff] = float("-inf")
            logits = torch.full_like(logits, float("-inf")).scatter_(-1, sorted_idx, sorted_logits)

        # (5) min_p — prune tokens with probability below min_p × max_prob.
        #     Qwen team recommends min_p=0 (disabled) by default. The frame
        #     protocol doesn't carry min_p right now; left here for future.

        probs = torch.softmax(logits, dim=-1)
        token = int(torch.multinomial(probs, num_samples=1).item())

        # Track recently-sampled token for the next step's penalty
        rt = self.recent_tokens.setdefault(frame.seq_id, [])
        rt.append(token)
        if len(rt) > self._recent_window:
            del rt[:-self._recent_window]

        raw = getattr(self.config, "eos_token_id", None)
        if raw is None:
            eos_ids: set[int] = set()
        elif isinstance(raw, int):
            eos_ids = {raw}
        else:
            eos_ids = set(raw)
        return token, token in eos_ids


# ─── HTTP surface ────────────────────────────────────────────────────────────

def make_app(stage: PipelineStage) -> web.Application:
    app = web.Application(client_max_size=64 * 1024 * 1024)

    async def handle_forward(request: web.Request) -> web.Response:
        raw = await request.read()
        frame = Frame.decode(raw)
        loop = asyncio.get_event_loop()
        # Torch forward is blocking/GIL-held; run in executor so the event
        # loop isn't starved (important when coordinator sends concurrent reqs).
        resp = await loop.run_in_executor(None, stage.forward, frame)
        return web.Response(body=resp.encode(), content_type="application/octet-stream")

    async def handle_reset(request: web.Request) -> web.Response:
        body = await request.json()
        stage.reset(body["seq_id"])
        return web.json_response({"ok": True})

    async def handle_info(request: web.Request) -> web.Response:
        return web.json_response({
            "model": stage.model_name,
            "layer_start": stage.layer_start,
            "layer_end": stage.layer_end,
            "num_layers_total": stage.num_layers,
            "is_first": stage.is_first,
            "is_last": stage.is_last,
            "hidden_size": stage.hidden_size,
            "device": stage.device,
            "dtype": str(stage.dtype),
            "active_seqs": len(stage.caches),
        })

    async def handle_health(request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    app.router.add_post("/forward", handle_forward)
    app.router.add_post("/reset", handle_reset)
    app.router.add_get("/info", handle_info)
    app.router.add_get("/health", handle_health)
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer-start", type=int, required=True)
    parser.add_argument("--layer-end", type=int, required=True)
    parser.add_argument("--first", action="store_true")
    parser.add_argument("--last", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(name)s:%(process)d] %(message)s")

    stage = PipelineStage(
        args.model, args.layer_start, args.layer_end,
        is_first=args.first, is_last=args.last, device=args.device,
    )
    app = make_app(stage)
    log.info(f"serving stage on http://{args.host}:{args.port}")
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
