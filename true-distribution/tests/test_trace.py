"""Trace hidden state magnitudes layer-by-layer.

Prefill works in both paths, so the divergence must be during decode.
Print |hidden|.mean() after each layer for both reference and distributed.
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL = "Qwen/Qwen3-0.6B"
PROMPT = "What is the capital of France?"


def trace_reference(m, prompt_ids, decode_tok, device):
    print("\n=== REFERENCE ===")
    cache = DynamicCache()
    # prefill silently
    with torch.inference_mode():
        ids = torch.tensor([prompt_ids], device=device)
        m(ids, past_key_values=cache, use_cache=True)

    # Decode one token, hook each layer's output
    traces = []
    hooks = []
    for i, layer in enumerate(m.model.layers):
        def hook(mod, inp, out, idx=i):
            h = out[0] if isinstance(out, tuple) else out
            traces.append((idx, float(h.abs().mean().item()), tuple(h.shape)))
        hooks.append(layer.register_forward_hook(hook))

    try:
        with torch.inference_mode():
            ids = torch.tensor([[decode_tok]], device=device)
            out = m(ids, past_key_values=cache, use_cache=True)
            next_tok = int(out.logits[0, -1].argmax().item())
    finally:
        for h in hooks:
            h.remove()

    for idx, mag, shape in traces:
        print(f"  layer {idx:2d}: shape={shape} |h|.mean={mag:.4f}")
    print(f"  -> next_tok = {next_tok}")
    return traces, next_tok


def trace_distributed(prompt_ids, decode_tok, device, dtype):
    from pipeline.worker import PipelineStage
    from pipeline.protocol import Frame

    print("\n=== DISTRIBUTED (single stage) ===")
    stage = PipelineStage(MODEL, 0, 28, is_first=True, is_last=True, device=device)
    # Prefill
    f = Frame(seq_id="t", stage_kind="first",
              positions=list(range(len(prompt_ids))),
              tensor=np.array(prompt_ids, dtype=np.int32),
              temperature=1e-6, top_k=1)
    stage.forward(f)

    # Hook each layer
    traces = []
    hooks = []
    for i, layer in enumerate(stage.layers):
        def hook(mod, inp, out, idx=i):
            h = out[0] if isinstance(out, tuple) else out
            traces.append((idx, float(h.abs().mean().item()), tuple(h.shape)))
        hooks.append(layer.register_forward_hook(hook))

    try:
        f2 = Frame(seq_id="t", stage_kind="first",
                   positions=[len(prompt_ids)],
                   tensor=np.array([decode_tok], dtype=np.int32),
                   temperature=1e-6, top_k=1)
        resp = stage.forward(f2)
    finally:
        for h in hooks:
            h.remove()

    for idx, mag, shape in traces:
        print(f"  layer {idx:2d}: shape={shape} |h|.mean={mag:.4f}")
    print(f"  -> next_tok = {resp.token_id}")
    return traces, resp.token_id


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if dev != "cpu" else torch.float32
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompt_ids = tok(PROMPT).input_ids
    DECODE = 576  # ' The' — what both paths agreed on as the first generated token

    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(dev)
    m.eval()
    ref_traces, ref_tok = trace_reference(m, prompt_ids, DECODE, dev)
    del m

    dist_traces, dist_tok = trace_distributed(prompt_ids, DECODE, dev, dtype)

    print("\n=== DIFFS ===")
    for (i_a, m_a, s_a), (i_b, m_b, s_b) in zip(ref_traces, dist_traces):
        diff = abs(m_a - m_b)
        marker = "!!" if diff > 0.01 else "  "
        print(f"  layer {i_a:2d} ref={m_a:.4f} dist={m_b:.4f} diff={diff:.4f} {marker}")
    print(f"\nref_tok={ref_tok} dist_tok={dist_tok}")


if __name__ == "__main__":
    main()
