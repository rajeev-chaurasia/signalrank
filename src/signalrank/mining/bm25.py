"""BM25 over the sampled corpus.

It is here for two jobs that happen to need the same index. The first is the
baseline: a neural retriever that cannot beat BM25 has not earned anything, and
most small sentence encoders have already been trained on MS MARCO, so using one
as the baseline would measure contamination instead.

The second is mining hard negatives. Training a dual encoder against random
passages teaches it to separate a query from an arbitrary document, which it can
already do. The passages BM25 ranks highly and that are nonetheless wrong are
the ones that carry signal, and they are what the in-batch negatives alone
cannot supply.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import bm25s
import numpy as np

from signalrank.core import Qrels, Run, relevant_for


@dataclass
class BM25Index:
    """A built index plus the passage identifiers its internal positions mean."""

    retriever: bm25s.BM25
    passage_ids: np.ndarray

    @classmethod
    def build(cls, passage_ids: Sequence[int], texts: Sequence[str]) -> BM25Index:
        tokens = bm25s.tokenize(texts, stopwords="en", show_progress=False)
        retriever = bm25s.BM25()
        retriever.index(tokens, show_progress=False)
        return cls(retriever=retriever, passage_ids=np.asarray(passage_ids, dtype=np.int64))

    def save(self, path: Path) -> None:
        self.retriever.save(str(path))
        np.save(path / "passage_ids.npy", self.passage_ids)

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        return cls(
            retriever=bm25s.BM25.load(str(path), load_corpus=False),
            passage_ids=np.load(path / "passage_ids.npy"),
        )

    def search(self, queries: Sequence[str], k: int) -> list[list[int]]:
        """Rank passages for each query, returning identifiers rather than positions.

        The retriever works in its own dense position space. Translating back
        here means nothing downstream has to know that, and a position leaking
        into a run would silently score against the wrong passage.
        """
        if not queries:
            return []
        tokens = bm25s.tokenize(queries, stopwords="en", show_progress=False)
        depth = min(k, len(self.passage_ids))
        positions, _ = self.retriever.retrieve(tokens, k=depth, show_progress=False)
        return [self.passage_ids[row].tolist() for row in positions]


def run_for(index: BM25Index, queries: dict[int, str], k: int) -> Run:
    query_ids = sorted(queries)
    ranked = index.search([queries[query_id] for query_id in query_ids], k)
    return dict(zip(query_ids, ranked, strict=True))


def mine_negatives(
    index: BM25Index,
    queries: dict[int, str],
    qrels: Qrels,
    *,
    depth: int,
    per_query: int,
) -> dict[int, list[int]]:
    """Top-ranked passages that are not judged relevant for the query.

    Judged positives are removed rather than kept and relabelled. A passage the
    judgements call relevant is not a negative however highly BM25 ranks it, and
    training against one teaches the encoder the opposite of the thing it is
    being measured on.

    What remains are still only presumed negatives: MS MARCO judges a tiny
    fraction of its passages, so some of these are relevant and simply unjudged.
    That is a known property of training this way and not something this code
    can fix, which is why the number kept per query is small.
    """
    query_ids = sorted(queries)
    ranked = index.search([queries[query_id] for query_id in query_ids], depth)
    mined: dict[int, list[int]] = {}
    for query_id, candidates in zip(query_ids, ranked, strict=True):
        positives = relevant_for(qrels, query_id)
        negatives = [pid for pid in candidates if pid not in positives]
        mined[query_id] = negatives[:per_query]
    return mined
