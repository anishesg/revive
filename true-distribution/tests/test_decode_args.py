"""Check if passing MY position_ids/attention_mask/cache_position to the
reference model breaks it. If so, my computed args are wrong.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL = "Qwen/Qwen3-0.6B"
PROMPT = "What is the capital of France?"


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if dev != "cpu" else torch.float32
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompt_ids = tok(PROMPT).input_ids
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(dev)
    m.eval()

    # Path 1: reference decoder, no extra args — baseline
    cache1 = DynamicCache()
    with torch.inference_mode():
        ids = torch.tensor([prompt_ids], device=dev)
        out = m(ids, past_key_values=cache1, use_cache=True)
        tok1 = int(out.logits[0, -1].argmax().item())
        # decode one more
        ids2 = torch.tensor([[tok1]], device=dev)
        out2 = m(ids2, past_key_values=cache1, use_cache=True)
        tok2_baseline = int(out2.logits[0, -1].argmax().item())
    print(f"baseline path: step1={tok1}, step2={tok2_baseline}")

    # Path 2: same but I pass position_ids and cache_position explicitly
    cache2 = DynamicCache()
    with torch.inference_mode():
        ids = torch.tensor([prompt_ids], device=dev)
        P = len(prompt_ids)
        pos_ids = torch.arange(P, device=dev).unsqueeze(0)
        cache_pos = torch.arange(P, device=dev)
        out = m(ids, past_key_values=cache2, use_cache=True,
                position_ids=pos_ids, cache_position=cache_pos)
        tok1_v2 = int(out.logits[0, -1].argmax().item())
        # decode with explicit pos/cache_pos
        ids2 = torch.tensor([[tok1_v2]], device=dev)
        pos_ids2 = torch.tensor([[P]], device=dev)
        cache_pos2 = torch.tensor([P], device=dev)
        out2 = m(ids2, past_key_values=cache2, use_cache=True,
                 position_ids=pos_ids2, cache_position=cache_pos2)
        tok2_v2 = int(out2.logits[0, -1].argmax().item())
    print(f"my-args path:  step1={tok1_v2}, step2={tok2_v2}")

    # Path 3: same as path 2 but ALSO pass explicit 4D attention_mask like I do
    cache3 = DynamicCache()
    with torch.inference_mode():
        ids = torch.tensor([prompt_ids], device=dev)
        P = len(prompt_ids)
        pos_ids = torch.arange(P, device=dev).unsqueeze(0)
        cache_pos = torch.arange(P, device=dev)
        # prefill causal mask
        mask = torch.zeros((1, 1, P, P), device=dev, dtype=dtype)
        for i in range(P):
            mask[0, 0, i, i + 1 :] = float("-inf")
        out = m(ids, past_key_values=cache3, use_cache=True,
                position_ids=pos_ids, cache_position=cache_pos,
                attention_mask=mask)
        tok1_v3 = int(out.logits[0, -1].argmax().item())
        # decode with 1-row mask covering all kv positions
        ids2 = torch.tensor([[tok1_v3]], device=dev)
        pos_ids2 = torch.tensor([[P]], device=dev)
        cache_pos2 = torch.tensor([P], device=dev)
        mask2 = torch.zeros((1, 1, 1, P + 1), device=dev, dtype=dtype)  # all zeros = no blocking
        out2 = m(ids2, past_key_values=cache3, use_cache=True,
                 position_ids=pos_ids2, cache_position=cache_pos2,
                 attention_mask=mask2)
        tok2_v3 = int(out2.logits[0, -1].argmax().item())
    print(f"my-mask path:  step1={tok1_v3}, step2={tok2_v3}")


if __name__ == "__main__":
    main()
