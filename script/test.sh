#!/usr/bin/env bash
# Two invocations, and a thread limit, both for the same reason: faiss, torch
# and the BLAS behind bm25s each bring their own OpenMP runtime. Two of them in
# one process abort at the first threaded matmul. The only other fix on offer
# is a flag whose own authors warn it can silently produce incorrect results,
# which is not a trade this repository can make.
#
# The split is by directory rather than by marker because pytest imports every
# test module while collecting, so a marker would still load torch into the
# faiss process.
set -euo pipefail
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "--- torch and the rest ---"
uv run pytest --ignore=tests/faiss "$@"

echo "--- faiss, on its own ---"
uv run pytest tests/faiss "$@"
