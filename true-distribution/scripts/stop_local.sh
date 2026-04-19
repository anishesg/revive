#!/usr/bin/env bash
# Kill locally-launched workers.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -f "$ROOT/logs/.pids" ]; then
  for pid in $(cat "$ROOT/logs/.pids"); do
    kill "$pid" 2>/dev/null && echo "killed $pid" || true
  done
  rm "$ROOT/logs/.pids"
fi
# Catch-all for orphaned python workers
pkill -f "pipeline.worker" 2>/dev/null || true
echo "done"
