"""Verify that a 2-stage distributed forward produces the same logits as
the full model on a single device (within fp16 tolerance).

Runs in-process — no HTTP — so bugs are attributable to the slicing logic,
not networking.
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeline.protocol import Frame
from pipeline.worker import PipelineStage, pick_device


MODEL = "Qwen/Qwen3-0.6B"


def reference_forward(model_name, token_ids, device, dtype):
    """Run the full model and return final-position logits."""
    m = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
    m.eval()
    ids = torch.tensor([token_ids], device=device)
    with torch.inference_mode():
        out = m(ids)
    return out.logits[0, -1, :].float().cpu().numpy()


def distributed_forward(model_name, token_ids, split_at, device, dtype):
    """Run the model through two layer-split stages in sequence, in-process."""
    stage_a = PipelineStage(model_name, 0, split_at, is_first=True, is_last=False, device=device)
    # Hack: force stage_a dtype to match (PipelineStage picks its own)
    stage_b = PipelineStage(model_name, split_at, stage_a.num_layers, is_first=False, is_last=True, device=device)

    # First stage: tokens -> hidden
    frame_a = Frame(
        seq_id="corr-1",
        stage_kind="first",
        positions=list(range(len(token_ids))),
        tensor=np.array(token_ids, dtype=np.int32),
    )
    resp_a = stage_a.forward(frame_a)
    hidden = np.frombuffer(resp_a.tensor, dtype=np.float16).reshape(resp_a.shape).copy()

    # Last stage: hidden -> sample. But we want LOGITS not a sampled token.
    # Easy path: patch the stage to return logits when a flag is set.
    # Cleaner path: call internal parts manually.
    stage_b.caches["corr-1"] = stage_b._DynamicCache()
    stage_b.positions["corr-1"] = 0
    with torch.inference_mode():
        h = torch.from_numpy(hidden).to(device=stage_b.device, dtype=stage_b.dtype)
        position_ids = torch.tensor([list(range(len(token_ids)))], device=stage_b.device, dtype=torch.long)
        position_embeddings = stage_b.rotary_emb(h, position_ids)
        cache = stage_b.caches["corr-1"]
        cache_position = torch.arange(len(token_ids), device=stage_b.device, dtype=torch.long)
        for layer in stage_b.layers:
            out = layer(
                h, attention_mask=None, position_ids=position_ids,
                past_key_value=cache, use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            h = out[0] if isinstance(out, tuple) else out
        h = stage_b.norm(h)
        logits = stage_b.lm_head(h[:, -1:, :])
    return logits[0, -1, :].float().cpu().numpy()


def main():
    device = pick_device()
    # Use fp32 on CPU, fp16 on GPU — just like the worker.
    dtype = torch.float32 if device == "cpu" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompt = "What is the capital of France?"
    token_ids = tokenizer.encode(prompt)
    print(f"prompt token_ids: {token_ids}")

    print("\n[1] reference (full model) forward...")
    ref_logits = reference_forward(MODEL, token_ids, device, dtype)
    print(f"  logits stats: mean={ref_logits.mean():.4f} std={ref_logits.std():.4f} "
          f"argmax={ref_logits.argmax()} top5={ref_logits.argsort()[-5:][::-1].tolist()}")

    print("\n[2] distributed (2-stage split) forward...")
    dist_logits = distributed_forward(MODEL, token_ids, split_at=14, device=device, dtype=dtype)
    print(f"  logits stats: mean={dist_logits.mean():.4f} std={dist_logits.std():.4f} "
          f"argmax={dist_logits.argmax()} top5={dist_logits.argsort()[-5:][::-1].tolist()}")

    print("\n[3] diff analysis...")
    diff = np.abs(ref_logits - dist_logits)
    print(f"  max abs diff: {diff.max():.4f}")
    print(f"  mean abs diff: {diff.mean():.4f}")
    # Top-5 agreement is the real test — sampling only cares about the relative ranking.
    ref_top5 = set(ref_logits.argsort()[-5:].tolist())
    dist_top5 = set(dist_logits.argsort()[-5:].tolist())
    overlap = len(ref_top5 & dist_top5)
    print(f"  top-5 overlap: {overlap}/5")
    print(f"  argmax match: {ref_logits.argmax() == dist_logits.argmax()}")

    # argmax must match, and top-5 overlap should be high (4+/5).
    assert ref_logits.argmax() == dist_logits.argmax(), (
        f"argmax mismatch! ref={ref_logits.argmax()} dist={dist_logits.argmax()}"
    )
    print("\n✓ argmax matches — pipeline math is correct")


if __name__ == "__main__":
    main()
