"""One-stage decode test — covers all layers in a single worker.

If this fails too, the bug is in my PipelineStage decode logic, not the split.
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeline.protocol import Frame
from pipeline.worker import PipelineStage, pick_device

MODEL = "Qwen/Qwen3-0.6B"
PROMPT = "What is the capital of France?"
STEPS = 4


def main():
    device = pick_device()
    dtype = torch.float32 if device == "cpu" else torch.float16
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompt_ids = tok(PROMPT).input_ids

    # One stage that covers all layers and is BOTH first AND last
    stage = PipelineStage(MODEL, 0, 28, is_first=True, is_last=True, device=device)

    seq_id = "single"
    positions = list(range(len(prompt_ids)))
    frame = Frame(seq_id=seq_id, stage_kind="first", positions=positions,
                  tensor=np.array(prompt_ids, dtype=np.int32),
                  temperature=1e-6, top_k=1, top_p=1.0)
    resp = stage.forward(frame)
    nxt = resp.token_id
    generated = [nxt]
    print(f"prefill -> {nxt} ({tok.decode([nxt])!r})")

    for i in range(STEPS - 1):
        pos = [len(prompt_ids) + i]
        frame = Frame(seq_id=seq_id, stage_kind="first", positions=pos,
                      tensor=np.array([nxt], dtype=np.int32),
                      temperature=1e-6, top_k=1, top_p=1.0)
        resp = stage.forward(frame)
        nxt = resp.token_id
        generated.append(nxt)
        print(f"decode {i+1} -> {nxt} ({tok.decode([nxt])!r})")

    print(f"\nfinal: {generated}")
    print(f"text:  {tok.decode(generated)!r}")


if __name__ == "__main__":
    main()
