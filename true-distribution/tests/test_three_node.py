"""3-stage distributed generation test.

Splits Qwen3-0.6B (28 layers) into three contiguous slices and verifies
the output matches the full reference model token-for-token under greedy
sampling. Catches any bug introduced by mid-stage (non-first, non-last)
hidden-state forwarding.
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeline.protocol import Frame
from pipeline.worker import PipelineStage, pick_device


MODEL = "Qwen/Qwen3-0.6B"
SPLITS = [(0, 10), (10, 19), (19, 28)]   # contiguous, sums to 28
PROMPT = "What is the capital of France?"
MAX_NEW = 8


def reference_generate(device, dtype):
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(device)
    m.eval()
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        out = m.generate(
            ids, max_new_tokens=MAX_NEW, do_sample=False,
            temperature=1.0, top_p=1.0, top_k=0,
            pad_token_id=tok.eos_token_id,
        )
    return out[0, ids.shape[1]:].tolist()


def distributed_generate(device, dtype):
    tok = AutoTokenizer.from_pretrained(MODEL)
    stages = []
    for i, (s, e) in enumerate(SPLITS):
        is_first = (i == 0)
        is_last = (i == len(SPLITS) - 1)
        stages.append(PipelineStage(MODEL, s, e, is_first=is_first, is_last=is_last,
                                     device=device))

    prompt_ids = tok(PROMPT).input_ids
    seq_id = "tn3"
    positions = list(range(len(prompt_ids)))

    def step(inputs, positions):
        # Stage 0: tokens → hidden
        f = Frame(seq_id=seq_id, stage_kind="first", positions=positions,
                  tensor=np.array(inputs, dtype=np.int32),
                  temperature=1e-6, top_k=1, top_p=1.0)
        r = stages[0].forward(f)
        h = np.frombuffer(r.tensor, dtype=np.float16).reshape(r.shape).copy()
        # Mid stages
        for st in stages[1:-1]:
            f = Frame(seq_id=seq_id, stage_kind="mid", positions=positions,
                      tensor=h, temperature=1e-6, top_k=1, top_p=1.0)
            r = st.forward(f)
            h = np.frombuffer(r.tensor, dtype=np.float16).reshape(r.shape).copy()
        # Last
        f = Frame(seq_id=seq_id, stage_kind="last", positions=positions,
                  tensor=h, temperature=1e-6, top_k=1, top_p=1.0)
        r = stages[-1].forward(f)
        return r.token_id

    generated = []
    nxt = step(prompt_ids, positions)
    generated.append(nxt)

    for i in range(MAX_NEW - 1):
        pos = [len(prompt_ids) + i]
        nxt = step([nxt], pos)
        generated.append(nxt)

    return generated


def main():
    device = pick_device()
    dtype = torch.float32 if device == "cpu" else torch.float16
    print(f"device={device}  splits={SPLITS}")

    print("\n=== reference (full model, greedy) ===")
    ref = reference_generate(device, dtype)
    tok = AutoTokenizer.from_pretrained(MODEL)
    print(f"ids:  {ref}")
    print(f"text: {tok.decode(ref)!r}")

    print("\n=== distributed (3-stage, greedy) ===")
    dist = distributed_generate(device, dtype)
    print(f"ids:  {dist}")
    print(f"text: {tok.decode(dist)!r}")

    match = ref == dist
    print(f"\n{'✓ MATCH — 3-stage ring is correct' if match else '✗ MISMATCH'}")
    if not match:
        for i, (a, b) in enumerate(zip(ref, dist)):
            marker = "  " if a == b else "! "
            print(f"  {marker}pos {i}: ref={a} dist={b}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
