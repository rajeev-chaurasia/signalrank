#!/usr/bin/env bash
# Lives in script/ rather than beside the files it downloads, because
# data/ is ignored and a fetch script that does not ship with the
# repository cannot be run on a fresh machine.
#
# Only what hard-negative mining needs. The published triples are not
# downloaded: this project mines its own negatives with BM25, which is the
# thing being tested, and the triples file expands to 27 GB to supply an
# answer the mining is supposed to produce.
set -euo pipefail
cd "$(dirname "$0")/../data/raw"
mkdir -p . 2>/dev/null || true
BASE=https://msmarco.z22.web.core.windows.net/msmarcoranking
for f in collection.tar.gz queries.tar.gz qrels.train.tsv qrels.dev.small.tsv; do
  [ -f "$f" ] && { echo "have $f"; continue; }
  echo "fetching $f"
  curl -fL --retry 5 --retry-delay 2 -C - -o "$f" "$BASE/$f"
done
