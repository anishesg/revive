#!/usr/bin/env python3
"""Orchestrator: merged HF checkpoint -> (role x tier) GGUF matrix.

Steps per (role, tier) cell:
  1. Start from merged HF checkpoint at output/merged/{role}/hf (or hf-pruned
     for tiers that request pruning).
  2. Apply prune profile if the tier's profile is not 'none'.
  3. Convert to fp16 GGUF.
  4. Generate role-specific imatrix (shared across tiers for the same role).
  5. Quantize to the tier's target format.
  6. Emit revive-{role}-qwen3-{size}-{tier}-{quant}.gguf + update manifest.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from LLM.common import gguf_io  # noqa: E402
from LLM.common.role_registry import ALL_ROLE_NAMES, get as get_role  # noqa: E402
from LLM.export.manifest import write_manifest  # noqa: E402

MERGED_DIR = REPO_ROOT / "LLM" / "output" / "merged"
GGUF_DIR = REPO_ROOT / "LLM" / "output" / "gguf"
IMATRIX_DIR = REPO_ROOT / "LLM" / "output" / "imatrix"
CALIB_DIR = REPO_ROOT / "LLM" / "data" / "calibration_prompts"

TIERS_YAML = REPO_ROOT / "LLM" / "common" / "device_tiers.yaml"
PRUNE_PROFILES = REPO_ROOT / "LLM" / "prune" / "prune_profiles.yaml"


def load_tiers() -> dict:
    return yaml.safe_load(TIERS_YAML.open())["tiers"]


def load_prune_profiles() -> dict:
    return yaml.safe_load(PRUNE_PROFILES.open())


def run(cmd: list[str]) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def ensure_pruned(role: str, profile_name: str) -> Path:
    """Return HF dir to use for GGUF conversion, pruning if needed."""
    base = MERGED_DIR / role / "hf"
    if profile_name == "none":
        return base
    pruned = MERGED_DIR / role / f"hf-{profile_name}"
    if pruned.exists() and (pruned / "config.json").exists():
        return pruned
    run([
        "python3", "-m", "LLM.prune.layer_prune",
        "--role", role,
        "--in", str(base),
        "--out", str(pruned),
    ])
    return pruned


def ensure_fp16(role: str, profile_name: str, hf_dir: Path) -> Path:
    fp16 = GGUF_DIR / f"_fp16-{role}-{profile_name}.gguf"
    if fp16.exists():
        return fp16
    gguf_io.convert_hf_to_gguf(hf_dir, fp16, outtype="f16")
    return fp16


def ensure_imatrix(role: str, profile_name: str, fp16: Path) -> Path:
    imatrix = IMATRIX_DIR / f"{role}-{profile_name}.dat"
    if imatrix.exists():
        return imatrix
    calib = CALIB_DIR / f"{role}.txt"
    if not calib.exists():
        raise SystemExit(f"Missing {calib}. Run data/build_calibration.py.")
    gguf_io.run_imatrix(fp16, calib, imatrix)
    return imatrix


def tier_uses_pruning(tier_cfg: dict) -> bool:
    return tier_cfg.get("prune_profile", "none") != "none"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", default="all")
    parser.add_argument("--tier", default="all")
    parser.add_argument("--skip-imatrix", action="store_true",
                        help="Quantize without imatrix (lower quality on Q2/Q3)")
    args = parser.parse_args()

    tiers = load_tiers()
    prune_cfg = load_prune_profiles()

    target_tiers = list(tiers.keys()) if args.tier == "all" else [args.tier]
    target_roles = list(ALL_ROLE_NAMES) if args.role == "all" else [args.role]

    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    IMATRIX_DIR.mkdir(parents=True, exist_ok=True)

    emitted: list[Path] = []

    for role in target_roles:
        r = get_role(role)
        base_hf = MERGED_DIR / role / "hf"
        if not base_hf.exists():
            print(f"[skip] {role}: no merged checkpoint at {base_hf}")
            continue

        # Decide which prune variants we need across the requested tiers.
        needed_profiles: set[str] = set()
        for tier in target_tiers:
            cfg = tiers[tier]
            if role not in cfg.get("roles", []):
                continue
            profile = cfg.get("prune_profile", "none")
            if profile != "none":
                profile = prune_cfg["roles"].get(role, "none")
            needed_profiles.add(profile)

        # Build fp16 + imatrix per distinct profile.
        fp16_by_profile: dict[str, Path] = {}
        imatrix_by_profile: dict[str, Path | None] = {}
        for profile in needed_profiles:
            hf = ensure_pruned(role, profile)
            fp16 = ensure_fp16(role, profile, hf)
            fp16_by_profile[profile] = fp16
            if args.skip_imatrix:
                imatrix_by_profile[profile] = None
            else:
                imatrix_by_profile[profile] = ensure_imatrix(role, profile, fp16)

        for tier in target_tiers:
            cfg = tiers[tier]
            if role not in cfg.get("roles", []):
                print(f"[skip] {role} @ {tier}: tier excludes this role")
                continue
            requested_profile = cfg.get("prune_profile", "none")
            profile = requested_profile if requested_profile == "none" else (
                prune_cfg["roles"].get(role, "none")
            )
            fp16 = fp16_by_profile[profile]
            imatrix = imatrix_by_profile[profile]
            quant = cfg["quant"]
            out_name = gguf_io.gguf_name(role, r.size_label, tier, quant)
            out_path = GGUF_DIR / out_name
            gguf_io.quantize(fp16, out_path, quant, imatrix=imatrix)
            print(f"[ok] {out_name}: {gguf_io.human_size(out_path)}")
            emitted.append(out_path)

    if emitted:
        manifest_path = GGUF_DIR / "manifest.json"
        write_manifest(emitted, manifest_path)
        print(f"[manifest] {manifest_path}")


if __name__ == "__main__":
    main()
