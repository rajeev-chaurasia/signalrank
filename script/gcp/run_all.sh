#!/usr/bin/env bash
# Everything after bootstrap, in one chain so the instance is never idle while
# it bills. Each stage drops a marker file when it finishes.
#
# Paths are absolute, resolved from the script's own location. A relative
# path here failed once with the file plainly present, and a run that costs
# money is the wrong place to find out which directory a command inherited.
#
# The markers exist because the instance is not durable. Twice a run was lost
# whole: once when a spot instance was preempted, once when the run-duration
# cap deleted it along with its disk. The driver on the other end watches for
# these and copies each artifact down as it appears, so a loss costs the
# current stage rather than everything.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

mark() { date -u +%H:%M:%S > "$ROOT/runs/.$1-done"; echo "=== $1 finished $(date -u +%H:%M:%S) ==="; }
mkdir -p "$ROOT/runs"

echo "=== training $(date -u +%H:%M:%S) ==="
uv run signalrank train --epochs 2 --batch-size 64 --negatives-per-query 2 \
  --build "$ROOT/data/build" --out "$ROOT/runs/dual-encoder"
mark training

echo "=== encoding $(date -u +%H:%M:%S) ==="
uv run signalrank encode --model "$ROOT/runs/dual-encoder" --build "$ROOT/data/build" --out "$ROOT/runs/vectors"
mark encoding

echo "=== evaluating $(date -u +%H:%M:%S) ==="
uv run signalrank evaluate --vectors "$ROOT/runs/vectors" --build "$ROOT/data/build" --evidence "$ROOT/evidence" --label dual-encoder
mark evaluate

echo "=== untrained control $(date -u +%H:%M:%S) ==="
uv run signalrank encode --model "$ROOT/runs/dual-encoder" --build "$ROOT/data/build" --out "$ROOT/runs/vectors-untrained" --untrained
uv run signalrank evaluate --vectors "$ROOT/runs/vectors-untrained" --build "$ROOT/data/build" --evidence "$ROOT/evidence" --label untrained-control \
  --latency-seconds 5
mark control

echo "=== done $(date -u +%H:%M:%S) ==="
