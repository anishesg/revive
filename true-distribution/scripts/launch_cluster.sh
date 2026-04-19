#!/usr/bin/env bash
# Launch the BMC-managed distributed inference cluster end to end.
#
# Starts N worker processes, then the dashboard server (which hosts the
# embedded BMC simulator + controller + coordinator). Open
# http://127.0.0.1:4100 when it says "Dashboard ready".
#
# Usage:
#   scripts/launch_cluster.sh [MODEL] [NUM_WORKERS]
#     MODEL defaults to Qwen/Qwen3-0.6B
#     NUM_WORKERS defaults to 2

set -euo pipefail

MODEL="${1:-Qwen/Qwen3-0.6B}"
NUM_WORKERS="${2:-2}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$ROOT"
source .venv/bin/activate

NUM_LAYERS=$(python3 -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('$MODEL'); print(c.num_hidden_layers)")
echo "Model: $MODEL | layers: $NUM_LAYERS | workers: $NUM_WORKERS"

# Compute layer ranges: even split, remainder on the last worker.
PER=$((NUM_LAYERS / NUM_WORKERS))
REM=$((NUM_LAYERS - PER * NUM_WORKERS))

mkdir -p logs
# Clean up any leftover pids
scripts/stop_local.sh >/dev/null 2>&1 || true

WORKER_URLS=()
for i in $(seq 0 $((NUM_WORKERS - 1))); do
  START=$((i * PER))
  if [ "$i" -eq "$((NUM_WORKERS - 1))" ]; then
    END=$NUM_LAYERS
  else
    END=$((START + PER))
  fi
  PORT=$((50100 + i))
  FIRST=""
  LAST=""
  if [ "$i" -eq 0 ]; then FIRST="--first"; fi
  if [ "$i" -eq "$((NUM_WORKERS - 1))" ]; then LAST="--last"; fi

  python -m pipeline.worker \
    --model "$MODEL" \
    --layer-start "$START" --layer-end "$END" \
    $FIRST $LAST --port "$PORT" \
    > "logs/worker_$i.log" 2>&1 &
  PID=$!
  echo "  worker $i: PID=$PID port=$PORT layers [$START..$END)"
  echo "$PID" >> logs/.pids
  WORKER_URLS+=("127.0.0.1:$PORT")
done

echo ""
echo "Waiting for workers to come up..."
for url in "${WORKER_URLS[@]}"; do
  for i in $(seq 1 60); do
    if curl -sf "http://$url/health" >/dev/null 2>&1; then
      echo "  $url ready"; break
    fi
    sleep 2
    if [ "$i" -eq 60 ]; then
      echo "  $url FAILED — check logs/worker_*.log"; exit 1
    fi
  done
done

echo ""
echo "Launching dashboard on http://127.0.0.1:4100 ..."
echo "  (BMC simulator + coordinator embedded in the same process)"
echo ""
echo "Press Ctrl-C to stop. Workers will keep running — run scripts/stop_local.sh to stop them."
echo ""

EXTRA=()
# If $SERIAL env var is set (e.g. SERIAL=/dev/tty.usbmodem14101), use the real
# Arduino BMC. If SERIAL=auto, try to detect one.
if [ "${SERIAL:-}" = "auto" ]; then
  EXTRA+=(--auto-serial)
elif [ -n "${SERIAL:-}" ]; then
  EXTRA+=(--serial-device "$SERIAL")
fi

exec python -m dashboard.server \
  --model "$MODEL" \
  --workers "${WORKER_URLS[@]}" \
  --bmc-port 45555 \
  --port 4100 \
  --log-level INFO \
  "${EXTRA[@]}"
