"""Metrics and the evaluation rule, free of any model or index.

Every number this repository publishes is computed here. Keeping it free of
torch and faiss is what lets the metrics be checked against worked examples
rather than against another run of the same pipeline, which is the only way a
scoring bug shows up as a failing test instead of as a better result.
"""

from signalrank.core.controls import shuffled_qrels
from signalrank.core.judgements import Qrels, Run, relevant_for
from signalrank.core.metrics import evaluate, ndcg_at_k, recall_at_k, reciprocal_rank

__all__ = [
    "Qrels",
    "Run",
    "evaluate",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "relevant_for",
    "shuffled_qrels",
]
