#!/usr/bin/env bash
# Everything after bootstrap, in one chain so the instance is never idle while
# it bills. Each stage drops a marker file when it finishes.
#
# The markers exist because the instance is not durable. Twice a run was lost
# whole: once when a spot instance was preempted, once when the run-duration
# cap deleted it along with its disk. The driver on the other end watches for
# these and copies each artifact down as it appears, so a loss costs the
# current stage rather than everything.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

mark() { date -u +%H:%M:%S > "runs/.$1-done"; echo "=== $1 finished $(date -u +%H:%M:%S) ==="; }
mkdir -p runs

echo "=== training $(date -u +%H:%M:%S) ==="
uv run signalrank train --epochs 2 --batch-size 64 --negatives-per-query 2 \
  --out runs/dual-encoder
mark training

echo "=== encoding $(date -u +%H:%M:%S) ==="
uv run signalrank encode --model runs/dual-encoder --out runs/vectors
mark encoding

echo "=== evaluating $(date -u +%H:%M:%S) ==="
uv run signalrank evaluate --vectors runs/vectors --label dual-encoder
mark evaluate

echo "=== untrained control $(date -u +%H:%M:%S) ==="
uv run signalrank encode --model runs/dual-encoder --out runs/vectors-untrained --untrained
uv run signalrank evaluate --vectors runs/vectors-untrained --label untrained-control \
  --latency-seconds 5
mark control

echo "=== done $(date -u +%H:%M:%S) ==="
