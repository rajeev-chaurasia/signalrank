"""Recall, NDCG and reciprocal rank, written out rather than imported.

These are four lines each and every number in the README rests on them, so they
are here where they can be read and checked against a worked example. A library
would be fine until the day its definition of recall turns out to be the capped
one and a published figure moves without any model changing.

Recall here is uncapped: the denominator is every passage judged relevant for
the query, not the smaller of that and k. For a query with more relevant
passages than k, uncapped recall cannot reach 1, which is the honest reading.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from signalrank.core.judgements import Qrels, Run, relevant_for, scorable


def recall_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        raise ValueError("recall is undefined for a query with no relevant passages")
    found = len(set(ranked[:k]) & relevant)
    return found / len(relevant)


def reciprocal_rank(ranked: list[int], relevant: set[int], k: int) -> float:
    for position, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: list[int], grades: dict[int, int], k: int) -> float:
    """Discounted gain over the best achievable gain for this query.

    Gains are the usual exponential form, so a grade of two counts for three
    times a grade of one rather than twice. On binary judgements the two forms
    agree, and stating it matters only if a graded evaluation set is added
    later.
    """
    gained = sum(
        (2 ** grades.get(doc_id, 0) - 1) / math.log2(position + 1)
        for position, doc_id in enumerate(ranked[:k], start=1)
    )
    ideal_grades = sorted((grade for grade in grades.values() if grade > 0), reverse=True)
    ideal = sum(
        (2**grade - 1) / math.log2(position + 1)
        for position, grade in enumerate(ideal_grades[:k], start=1)
    )
    return 0.0 if ideal == 0 else gained / ideal


@dataclass(frozen=True)
class Scores:
    """What one evaluation produced, including what it could not score."""

    queries_scored: int
    queries_unscorable: int
    recall_at_20: float
    recall_at_100: float
    ndcg_at_10: float
    mrr_at_10: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "queries_scored": self.queries_scored,
            "queries_unscorable": self.queries_unscorable,
            "recall_at_20": round(self.recall_at_20, 5),
            "recall_at_100": round(self.recall_at_100, 5),
            "ndcg_at_10": round(self.ndcg_at_10, 5),
            "mrr_at_10": round(self.mrr_at_10, 5),
        }


def evaluate(run: Run, qrels: Qrels) -> Scores:
    """Average every metric over the queries that can be scored.

    Macro averaged, one vote per query. Micro averaging would let the handful of
    queries with many relevant passages dominate, and the claim is about queries
    rather than about judgements.
    """
    scorable_ids = scorable(qrels, run)
    recall_20 = recall_100 = ndcg_10 = mrr_10 = 0.0
    for query_id in scorable_ids:
        ranked = run[query_id]
        relevant = relevant_for(qrels, query_id)
        recall_20 += recall_at_k(ranked, relevant, 20)
        recall_100 += recall_at_k(ranked, relevant, 100)
        ndcg_10 += ndcg_at_k(ranked, qrels[query_id], 10)
        mrr_10 += reciprocal_rank(ranked, relevant, 10)

    count = len(scorable_ids)
    if count == 0:
        return Scores(0, len(run), 0.0, 0.0, 0.0, 0.0)
    return Scores(
        queries_scored=count,
        queries_unscorable=len(run) - count,
        recall_at_20=recall_20 / count,
        recall_at_100=recall_100 / count,
        ndcg_at_10=ndcg_10 / count,
        mrr_at_10=mrr_10 / count,
    )
