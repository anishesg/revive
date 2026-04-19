#!/usr/bin/env bash
# Spawn 2 pipeline workers + print coordinator command.
# Usage: scripts/launch_local.sh [MODEL]
#   MODEL defaults to Qwen/Qwen3-0.6B (28 layers)

set -euo pipefail

MODEL="${1:-Qwen/Qwen3-0.6B}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$ROOT"
source .venv/bin/activate

# Figure out layer count from the config so the split is correct.
NUM_LAYERS=$(python3 -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('$MODEL'); print(c.num_hidden_layers)")
SPLIT=$((NUM_LAYERS / 2))
echo "Model: $MODEL | total layers: $NUM_LAYERS | split at: $SPLIT"

mkdir -p logs
# Worker A: layers [0..SPLIT), first stage
python -m pipeline.worker \
  --model "$MODEL" \
  --layer-start 0 --layer-end "$SPLIT" \
  --first --port 50100 \
  > logs/worker_a.log 2>&1 &
PID_A=$!
echo "Worker A (first): PID=$PID_A port=50100 layers [0..$SPLIT)"

# Worker B: layers [SPLIT..NUM_LAYERS), last stage
python -m pipeline.worker \
  --model "$MODEL" \
  --layer-start "$SPLIT" --layer-end "$NUM_LAYERS" \
  --last --port 50101 \
  > logs/worker_b.log 2>&1 &
PID_B=$!
echo "Worker B (last):  PID=$PID_B port=50101 layers [$SPLIT..$NUM_LAYERS)"

echo "$PID_A $PID_B" > logs/.pids
echo ""
echo "Tail logs:  tail -f logs/worker_a.log logs/worker_b.log"
echo "Stop:       scripts/stop_local.sh"
echo ""
echo "Waiting for workers to be ready..."
for port in 50100 50101; do
  for i in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      echo "  port $port ready"
      break
    fi
    sleep 2
    if [ "$i" -eq 60 ]; then
      echo "  port $port FAILED to come up — check logs/worker_*.log"
      exit 1
    fi
  done
done

echo ""
echo "Ring is up. Try a query:"
echo "  python -m pipeline.coordinator --model $MODEL \\"
echo "    --workers 127.0.0.1:50100 127.0.0.1:50101 \\"
echo "    --prompt 'What is the capital of France?' --max-tokens 40"
