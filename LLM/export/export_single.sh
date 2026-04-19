#!/usr/bin/env bash
# Single (role, tier) export. Useful for dev iteration.
# Usage: bash export_single.sh <role> <tier>
set -euo pipefail

ROLE="${1:-spotter}"
TIER="${2:-standard}"

python3 -m LLM.export.export_tier_matrix --role "${ROLE}" --tier "${TIER}"
