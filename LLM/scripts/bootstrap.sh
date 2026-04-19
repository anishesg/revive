#!/usr/bin/env bash
# End-to-end green-path: generate data -> distill -> train all roles ->
# export standard tier for all roles. Minimum viable ship.
#
# Required env: ANTHROPIC_API_KEY
# Optional env: REVIVE_TEACHER (path to Qwen3-4B GGUF; defaults to
#               ~/revive/models/qwen3-4b-q4_k_m.gguf via distill script)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

: "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY required for Haiku data generation}"

N="${REVIVE_DATASET_SIZE:-750}"

echo "=== [1/5] Seeding calibration prompts ==="
python3 -m LLM.data.build_calibration

echo "=== [2/5] Generating Haiku dataset (n=${N}/role) ==="
python3 -m LLM.data.generate_expanded_dataset --role all --n "${N}"

echo "=== [3/5] Distilling from Qwen3-4B teacher (n=${N}/role) ==="
if [ -n "${REVIVE_TEACHER:-}" ]; then
    python3 -m LLM.data.distill_from_qwen4b --role all --n "${N}" --teacher "${REVIVE_TEACHER}"
else
    python3 -m LLM.data.distill_from_qwen4b --role all --n "${N}"
fi

echo "=== [4/5] Merging datasets ==="
python3 -m LLM.data.merge_datasets

echo "=== [5/5] Training all 8 roles ==="
bash LLM/train/train_all.sh

echo ""
echo "=== Exporting standard tier ==="
python3 -m LLM.export.export_tier_matrix --tier standard

echo ""
echo "Bootstrap complete. GGUFs in LLM/output/gguf/."
