#!/usr/bin/env bash
# Everything after bootstrap, in one go, so the instance is never idle while
# billing. Each stage writes into runs/ or evidence/ and the log is the record
# of what happened.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"

while pgrep -f bootstrap.sh >/dev/null; do sleep 20; done
echo "=== bootstrap finished $(date -u +%H:%M:%S) ==="

echo "=== training $(date -u +%H:%M:%S) ==="
uv run signalrank train --epochs 3 --batch-size 64 --negatives-per-query 2 \
  --out runs/dual-encoder

echo "=== encoding with the trained model $(date -u +%H:%M:%S) ==="
uv run signalrank encode --model runs/dual-encoder --out runs/vectors

echo "=== evaluating $(date -u +%H:%M:%S) ==="
uv run signalrank evaluate --vectors runs/vectors --label dual-encoder

# The second control from the plan: same shape, no training. A metric this
# scores well on is not measuring the model.
echo "=== encoding with an untrained encoder $(date -u +%H:%M:%S) ==="
uv run signalrank encode --model runs/dual-encoder --out runs/vectors-untrained --untrained

echo "=== evaluating the untrained control $(date -u +%H:%M:%S) ==="
uv run signalrank evaluate --vectors runs/vectors-untrained --label untrained-control \
  --latency-seconds 5

echo "=== done $(date -u +%H:%M:%S) ==="
