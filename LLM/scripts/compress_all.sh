#!/usr/bin/env bash
# Full tier matrix export: 8 roles x 4 tiers = up to 32 GGUFs + manifest.json.
# Assumes bootstrap.sh has already produced merged HF checkpoints.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

python3 -m LLM.export.export_tier_matrix --role all --tier all
