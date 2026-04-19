#!/usr/bin/env bash
# Smoke test: verify llama.cpp binaries exist; optionally run a spotter-only
# export to confirm the full pipeline is wired up.
#
# Usage:
#   bash quick_test.sh --check-only   # just verify binaries
#   bash quick_test.sh                # check + spotter standard-tier export
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

MODE="${1:-full}"

echo "=== Checking llama.cpp binaries ==="
python3 -c "
import sys
sys.path.insert(0, 'LLM')
from common import gguf_io
status = gguf_io.check_binaries()
for k, v in status.items():
    print(f'  {\"✓\" if v else \"✗\"} {k}')
if not all(status.values()):
    print()
    print('Missing binaries. Run one of:')
    print('  bash macos/setup.sh     # Apple Silicon/Intel Mac')
    print('  bash rpi/setup.sh       # Raspberry Pi')
    print('Some binaries (llama-imatrix) may need an extra build target.')
    sys.exit(1 if '${MODE}' == '--check-only' else 0)
"

if [ "${MODE}" = "--check-only" ]; then
    exit 0
fi

echo ""
echo "=== Checking merged checkpoint for spotter ==="
if [ ! -d "LLM/output/merged/spotter/hf" ]; then
    echo "No merged checkpoint at LLM/output/merged/spotter/hf."
    echo "Run bootstrap.sh first (or at minimum train_qwen_role.py --role spotter)."
    exit 1
fi

echo ""
echo "=== Running spotter standard-tier export ==="
python3 -m LLM.export.export_tier_matrix --role spotter --tier standard

echo ""
echo "=== Sanity-loading the result ==="
python3 -c "
from pathlib import Path
import sys
sys.path.insert(0, 'LLM')
from common import gguf_io
path = Path('LLM/output/gguf/revive-spotter-qwen3-0.6b-standard-Q4_K_M.gguf')
print(f'  {path}: {gguf_io.human_size(path)}')
ok = gguf_io.sanity_load(path)
print(f'  round-trip load: {\"ok\" if ok else \"FAILED\"}')
sys.exit(0 if ok else 1)
"
