#!/usr/bin/env bash
# Prepare a freshly created instance: dependencies, corpus, BM25, negatives.
#
# Everything here is CPU work on a machine that bills for a GPU, which is worth
# about fifteen cents and saves uploading a 400 MB corpus over a home
# connection. The corpus is rebuilt rather than copied because the sampling is
# seeded, so the instance produces the same 1.2 million passages the laptop did.
set -euo pipefail
cd "$(dirname "$0")/../.."

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv sync --all-groups
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

./data/raw/fetch.sh
tar xzf data/raw/collection.tar.gz -C data/raw
tar xzf data/raw/queries.tar.gz -C data/raw

uv run signalrank build-corpus
uv run signalrank build-bm25
uv run signalrank bm25-baseline
uv run signalrank mine-negatives
echo "bootstrap complete"
