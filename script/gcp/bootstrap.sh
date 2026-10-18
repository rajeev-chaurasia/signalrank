#!/usr/bin/env bash
# Prepare a freshly created instance: dependencies, corpus, BM25, negatives.
#
# Everything here is CPU work on a machine that bills for a GPU, which is worth
# about fifteen cents and saves uploading a 400 MB corpus over a home
# connection. The corpus is rebuilt rather than copied because the sampling is
# seeded, so the instance produces the same 1.2 million passages the laptop did.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv sync --all-groups
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

mkdir -p "$ROOT/data/raw"
"$ROOT/script/fetch_corpus.sh"
tar xzf "$ROOT"/data/raw/collection.tar.gz -C "$ROOT/data/raw"
tar xzf "$ROOT"/data/raw/queries.tar.gz -C "$ROOT/data/raw"

uv run signalrank build-corpus
uv run signalrank build-bm25
uv run signalrank bm25-baseline
# Skipped when negatives were carried in. Mining is single threaded and took
# 55 minutes here against 10 on the laptop that produced the same corpus,
# all of it with the GPU idle and billing.
if [ -s "$ROOT/data/build/negatives.tsv" ]; then
  echo "using the negatives already present: $(wc -l < "$ROOT/data/build/negatives.tsv") lines"
else
  uv run signalrank mine-negatives
fi
echo "bootstrap complete"
