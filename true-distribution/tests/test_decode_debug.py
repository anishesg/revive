"""Debug the decode step directly.

For each step compare:
  (A) reference full model fed [prompt + all_generated_so_far] as fresh forward
  (B) reference full model using its own KV cache with past_key_values
  (C) distributed stages using our KV cache decode path

If A==B, reference cache works. If A==C, our distributed decode is correct.
If A==B but A!=C, our cache handling is wrong.
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from pipeline.protocol import Frame
from pipeline.worker import PipelineStage, pick_device

MODEL = "Qwen/Qwen3-0.6B"
SPLIT = 14
PROMPT = "What is the capital of France?"
STEPS = 4


def ref_fresh(m, tok_ids):
    ids = torch.tensor([tok_ids], device=m.device)
    with torch.inference_mode():
        out = m(ids)
    return int(out.logits[0, -1, :].argmax().item())


def ref_cached(m, prompt_ids, extras):
    """Prefill with prompt_ids then decode extras one by one using DynamicCache."""
    cache = DynamicCache()
    with torch.inference_mode():
        ids = torch.tensor([prompt_ids], device=m.device)
        out = m(ids, past_key_values=cache, use_cache=True)
        nxt = int(out.logits[0, -1, :].argmax().item())
        generated = [nxt]
        for tok in extras:
            ids = torch.tensor([[tok]], device=m.device)
            out = m(ids, past_key_values=cache, use_cache=True)
            nxt = int(out.logits[0, -1, :].argmax().item())
            generated.append(nxt)
        return generated


def dist_decode(stage_a, stage_b, prompt_ids):
    seq_id = "dbg"
    generated = []

    def step(inputs, positions):
        fa = Frame(seq_id=seq_id, stage_kind="first", positions=positions,
                   tensor=np.array(inputs, dtype=np.int32))
        ra = stage_a.forward(fa)
        h = np.frombuffer(ra.tensor, dtype=np.float16).reshape(ra.shape).copy()
        fb = Frame(seq_id=seq_id, stage_kind="last", positions=positions,
                   tensor=h, temperature=1e-6, top_k=1, top_p=1.0)
        rb = stage_b.forward(fb)
        return rb.token_id

    # Prefill
    nxt = step(prompt_ids, list(range(len(prompt_ids))))
    generated.append(nxt)

    for i in range(STEPS - 1):
        pos = [len(prompt_ids) + i]
        nxt = step([nxt], pos)
        generated.append(nxt)

    return generated


def main():
    device = pick_device()
    dtype = torch.float32 if device == "cpu" else torch.float16
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompt_ids = tok(PROMPT).input_ids

    # Reference model (full)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(device)
    m.eval()

    # (A) fresh forward each step
    fresh_ids = []
    cur = list(prompt_ids)
    for _ in range(STEPS):
        nxt = ref_fresh(m, cur)
        fresh_ids.append(nxt)
        cur.append(nxt)
    print(f"(A) ref FRESH:  {fresh_ids}   {tok.decode(fresh_ids)!r}")

    # (B) reference with its own KV cache
    cached_ids = ref_cached(m, prompt_ids, extras=fresh_ids[:-1])
    print(f"(B) ref CACHED: {cached_ids}   {tok.decode(cached_ids)!r}")
    print(f"    A == B: {fresh_ids == cached_ids}")

    # (C) distributed
    del m
    stage_a = PipelineStage(MODEL, 0, SPLIT, is_first=True, is_last=False, device=device)
    stage_b = PipelineStage(MODEL, SPLIT, stage_a.num_layers, is_first=False, is_last=True, device=device)
    dist_ids = dist_decode(stage_a, stage_b, prompt_ids)
    print(f"(C) DIST:       {dist_ids}   {tok.decode(dist_ids)!r}")
    print(f"    A == C: {fresh_ids == dist_ids}")
    print(f"    B == C: {cached_ids == dist_ids}")


if __name__ == "__main__":
    main()
