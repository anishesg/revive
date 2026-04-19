"""Full generation test without HTTP — calls stage.forward() directly.

Compares against transformers' own .generate() for the same prompt with greedy
sampling. If this agrees, then any remaining bug is in the HTTP/wire layer.
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeline.protocol import Frame
from pipeline.worker import PipelineStage, pick_device


MODEL = "Qwen/Qwen3-0.6B"
SPLIT = 14
PROMPT = "What is the capital of France?"
MAX_NEW = 10


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
    gen = out[0, ids.shape[1]:].tolist()
    return gen, tok.decode(gen, skip_special_tokens=False)


def distributed_generate(device, dtype):
    tok = AutoTokenizer.from_pretrained(MODEL)
    stage_a = PipelineStage(MODEL, 0, SPLIT, is_first=True, is_last=False, device=device)
    stage_b = PipelineStage(MODEL, SPLIT, stage_a.num_layers, is_first=False, is_last=True, device=device)

    prompt_ids = tok(PROMPT).input_ids
    seq_id = "gen-1"
    positions = list(range(len(prompt_ids)))

    generated = []
    # Prefill
    frame_a = Frame(seq_id=seq_id, stage_kind="first", positions=positions,
                    tensor=np.array(prompt_ids, dtype=np.int32))
    resp_a = stage_a.forward(frame_a)
    hidden = np.frombuffer(resp_a.tensor, dtype=np.float16).reshape(resp_a.shape).copy()
    frame_b = Frame(seq_id=seq_id, stage_kind="last", positions=positions,
                    tensor=hidden, temperature=1e-6, top_k=1, top_p=1.0)  # greedy
    resp_b = stage_b.forward(frame_b)
    next_tok = resp_b.token_id
    generated.append(next_tok)
    print(f"  prefill -> {next_tok} ({tok.decode([next_tok])!r})")

    # Decode
    for step in range(MAX_NEW - 1):
        pos = [len(prompt_ids) + step]
        frame_a = Frame(seq_id=seq_id, stage_kind="first", positions=pos,
                        tensor=np.array([next_tok], dtype=np.int32))
        resp_a = stage_a.forward(frame_a)
        hidden = np.frombuffer(resp_a.tensor, dtype=np.float16).reshape(resp_a.shape).copy()
        frame_b = Frame(seq_id=seq_id, stage_kind="last", positions=pos,
                        tensor=hidden, temperature=1e-6, top_k=1, top_p=1.0)
        resp_b = stage_b.forward(frame_b)
        next_tok = resp_b.token_id
        generated.append(next_tok)
        print(f"  step {step+1} (pos {pos[0]}) -> {next_tok} ({tok.decode([next_tok])!r})")

    return generated, tok.decode(generated, skip_special_tokens=False)


def main():
    device = pick_device()
    dtype = torch.float32 if device == "cpu" else torch.float16
    print(f"device={device} dtype={dtype}")

    print("\n=== reference (full model .generate greedy) ===")
    ref_ids, ref_text = reference_generate(device, dtype)
    print(f"ids:  {ref_ids}")
    print(f"text: {ref_text!r}")

    print("\n=== distributed (2-stage greedy) ===")
    dist_ids, dist_text = distributed_generate(device, dtype)
    print(f"ids:  {dist_ids}")
    print(f"text: {dist_text!r}")

    match = ref_ids == dist_ids
    print(f"\n{'✓ MATCH' if match else '✗ MISMATCH'}")
    if not match:
        for i, (a, b) in enumerate(zip(ref_ids, dist_ids)):
            marker = "  " if a == b else "! "
            print(f"  {marker}pos {i}: ref={a} dist={b}")


if __name__ == "__main__":
    main()
