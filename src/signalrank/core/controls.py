"""The negative controls, which run on the same code path as the real scoring.

A retrieval number that goes up is not evidence that anything works. An
evaluation with a bug in it produces numbers that go up too, and the usual way
that happens is a passage identifier that lines up by accident, or judgements
joined to the wrong query. Scoring a trained model against deliberately wrong
judgements is the cheapest way to find out whether the harness is measuring
relevance or measuring itself.
"""

from __future__ import annotations

import random

from signalrank.core.judgements import Qrels


def shuffled_qrels(qrels: Qrels, seed: int) -> Qrels:
    """Reassign each query's judgements to a different query.

    A permutation rather than random identifiers, so the shuffled judgements
    have exactly the same shape as the real ones: the same number of queries,
    each with the same number of relevant passages drawn from the same corpus.
    Only the pairing is wrong. A harness that still reports relevance against
    these is finding something that is not there.
    """
    query_ids = sorted(qrels)
    if len(query_ids) < 2:
        raise ValueError("a permutation of fewer than two queries cannot derange anything")

    shifted = query_ids[1:] + query_ids[:1]
    return {source: dict(qrels[target]) for source, target in zip(query_ids, shifted, strict=True)}


def randomised_qrels(qrels: Qrels, seed: int) -> Qrels:
    """The same idea with a seeded shuffle rather than a rotation.

    A rotation is a derangement by construction but is also a single fixed
    permutation, so it cannot show that the result holds for permutations in
    general. This one is drawn, and retried until no query keeps its own
    judgements.
    """
    query_ids = sorted(qrels)
    if len(query_ids) < 2:
        raise ValueError("a permutation of fewer than two queries cannot derange anything")

    rng = random.Random(seed)
    for _ in range(100):
        shuffled = query_ids[:]
        rng.shuffle(shuffled)
        if all(source != target for source, target in zip(query_ids, shuffled, strict=True)):
            return {
                source: dict(qrels[target])
                for source, target in zip(query_ids, shuffled, strict=True)
            }
    raise RuntimeError("could not draw a derangement in 100 attempts")
