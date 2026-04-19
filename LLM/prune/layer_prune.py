#!/usr/bin/env python3
"""Role-aware transformer block pruning.

Implements the block-importance heuristic from ShortGPT (arXiv 2403.17887):
for each transformer block i, score it by the angular distance between
h_i (hidden state entering the block) and h_{i+1} (hidden state leaving it),
averaged over calibration prompts. Low angular change = low importance =
safe to drop. Keep the top-N highest-importance blocks; delete the rest
from the model config.

Usage:
  python3 -m LLM.prune.layer_prune --role spotter \\
      --in LLM/output/merged/spotter/hf \\
      --out LLM/output/merged/spotter/hf-pruned
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def load_profiles() -> dict:
    path = Path(__file__).parent / "prune_profiles.yaml"
    return yaml.safe_load(path.open())


def score_blocks(model, tokenizer, prompts: list[str], device: str = "cuda") -> list[float]:
    """Return per-block angular-distance scores. Higher = more important."""
    import torch

    model.eval()
    num_layers = model.config.num_hidden_layers
    running = [0.0] * num_layers
    count = 0

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, use_cache=False)
        # hidden_states is a tuple of length (num_layers + 1)
        hs = out.hidden_states
        for i in range(num_layers):
            a = hs[i].float().flatten(start_dim=1)
            b = hs[i + 1].float().flatten(start_dim=1)
            cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()
            # angular distance: smaller cos -> larger change -> more important
            running[i] += 1.0 - cos
        count += 1

    return [r / max(count, 1) for r in running]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True)
    parser.add_argument("--in", dest="in_dir", required=True)
    parser.add_argument("--out", dest="out_dir", required=True)
    parser.add_argument("--calibration", default=None,
                        help="Path to calibration .txt (default: data/calibration_prompts/{role}.txt)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    profiles = load_profiles()
    profile_name = profiles["roles"].get(args.role, "none")
    profile = profiles["profiles"][profile_name]

    if profile["strategy"] == "none" or profile["drop_fraction"] <= 0:
        print(f"[prune] role={args.role} profile={profile_name}: no-op, copying through")
        import shutil
        shutil.copytree(args.in_dir, args.out_dir, dirs_exist_ok=True)
        return

    calib = Path(args.calibration) if args.calibration else (
        REPO_ROOT / "LLM" / "data" / "calibration_prompts" / f"{args.role}.txt"
    )
    if not calib.exists():
        raise SystemExit(f"Calibration prompts missing: {calib}")
    prompts = [ln.strip() for ln in calib.open() if ln.strip()]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"[prune] loading {args.in_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.in_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.in_dir,
        torch_dtype=torch.float16,
        device_map=args.device,
    )

    print(f"[prune] scoring {model.config.num_hidden_layers} blocks over {len(prompts)} prompts")
    scores = score_blocks(model, tokenizer, prompts, device=args.device)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i])

    total = model.config.num_hidden_layers
    target_drop = int(total * profile["drop_fraction"])
    keep_count = max(profile["min_keep"], total - target_drop)
    drop_count = total - keep_count
    drop_indices = sorted(ranked[:drop_count])
    keep_indices = sorted(i for i in range(total) if i not in set(drop_indices))

    print(f"[prune] dropping {drop_count}/{total} blocks (keep {keep_count}): {drop_indices}")

    # Physically remove the dropped blocks from the decoder layer ModuleList.
    decoder = model.model.layers
    new_layers = torch.nn.ModuleList([decoder[i] for i in keep_indices])
    model.model.layers = new_layers
    model.config.num_hidden_layers = keep_count

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

    meta = {
        "role": args.role,
        "profile": profile_name,
        "original_layers": total,
        "kept_layers": keep_count,
        "dropped_indices": drop_indices,
        "kept_indices": keep_indices,
        "scores": scores,
    }
    (out_dir / "prune_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[prune] wrote {out_dir}")


if __name__ == "__main__":
    main()
