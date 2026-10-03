"""Metrics checked against worked examples rather than against another run.

Every figure this repository publishes comes out of these functions, so the
expected values below are computed by hand in the docstrings. A metric that is
only ever compared to its own previous output cannot tell a fixed bug from a
new one.
"""

from __future__ import annotations

import math

import pytest

from signalrank.core import Qrels, evaluate, ndcg_at_k, recall_at_k, reciprocal_rank
from signalrank.core.controls import randomised_qrels, shuffled_qrels
from signalrank.core.judgements import relevant_for, scorable


def test_recall_counts_relevant_in_the_top_k() -> None:
    """Two relevant, one of them inside the cut, so a half."""
    assert recall_at_k([1, 2, 3], {1, 3}, 2) == 0.5


def test_recall_is_uncapped_so_it_cannot_reach_one_here() -> None:
    """Three relevant and a cut of two: the best possible score is two thirds.

    The capped definition would call this 1.0. Stating which one is in use is
    the difference between a published number that can be reproduced and one
    that cannot.
    """
    assert recall_at_k([1, 2, 9], {1, 2, 3}, 2) == pytest.approx(2 / 3)


def test_recall_ignores_rank_order_inside_the_cut() -> None:
    assert recall_at_k([3, 1], {1, 3}, 2) == recall_at_k([1, 3], {1, 3}, 2)


def test_recall_without_relevant_passages_is_refused() -> None:
    """Returning zero would quietly punish a system for a gap in the judgements."""
    with pytest.raises(ValueError):
        recall_at_k([1, 2], set(), 10)


def test_reciprocal_rank_is_one_over_the_first_hit() -> None:
    assert reciprocal_rank([5, 1, 3], {1, 3}, 10) == 0.5


def test_reciprocal_rank_is_zero_when_nothing_relevant_is_in_the_cut() -> None:
    assert reciprocal_rank([5, 1], {3}, 10) == 0.0


def test_reciprocal_rank_respects_the_cut() -> None:
    assert reciprocal_rank([5, 6, 1], {1}, 2) == 0.0


def test_ndcg_matches_the_hand_computed_value() -> None:
    """Ranking [1, 2, 3] with 1 and 3 relevant at grade one.

    DCG  = 1/log2(2) + 0 + 1/log2(4) = 1 + 0.5          = 1.5
    IDCG = 1/log2(2) + 1/log2(3)     = 1 + 0.630929     = 1.630929
    NDCG = 1.5 / 1.630929                               = 0.919721
    """
    expected = 1.5 / (1 + 1 / math.log2(3))
    assert ndcg_at_k([1, 2, 3], {1: 1, 3: 1}, 10) == pytest.approx(expected)
    assert ndcg_at_k([1, 2, 3], {1: 1, 3: 1}, 10) == pytest.approx(0.919721, abs=1e-6)


def test_ndcg_of_the_ideal_ranking_is_one() -> None:
    assert ndcg_at_k([1, 3, 2], {1: 1, 3: 1}, 10) == pytest.approx(1.0)


def test_ndcg_uses_exponential_gain() -> None:
    """A grade of two is worth three, not two, under the exponential form."""
    assert ndcg_at_k([1], {1: 2}, 10) == pytest.approx(1.0)
    both = ndcg_at_k([1, 2], {1: 1, 2: 2}, 10)
    swapped = ndcg_at_k([2, 1], {1: 1, 2: 2}, 10)
    assert swapped > both


def test_ndcg_without_any_positive_grade_is_zero() -> None:
    assert ndcg_at_k([1, 2], {1: 0}, 10) == 0.0


def test_a_zero_grade_is_not_relevant() -> None:
    """Judged and rejected is not the same as never judged."""
    assert relevant_for({7: {1: 0, 2: 1}}, 7) == {2}


def test_queries_without_judgements_are_excluded_not_zeroed() -> None:
    qrels: Qrels = {1: {10: 1}, 2: {}}
    assert scorable(qrels, {1: [10], 2: [11]}) == [1]


def test_evaluate_reports_what_it_could_not_score() -> None:
    qrels: Qrels = {1: {10: 1}, 2: {}}
    scores = evaluate({1: [10], 2: [11]}, qrels)
    assert (scores.queries_scored, scores.queries_unscorable) == (1, 1)
    assert scores.recall_at_20 == 1.0


def test_evaluate_macro_averages_over_queries() -> None:
    """One perfect query and one empty: a half, not weighted by judgement count."""
    qrels: Qrels = {1: {10: 1}, 2: {20: 1, 21: 1, 22: 1}}
    scores = evaluate({1: [10], 2: [99]}, qrels)
    assert scores.recall_at_20 == pytest.approx(0.5)


def test_evaluate_with_nothing_scorable_returns_zeros() -> None:
    scores = evaluate({1: [10]}, {1: {}})
    assert scores.queries_scored == 0
    assert scores.recall_at_20 == 0.0


def test_the_shuffle_gives_every_query_someone_elses_judgements() -> None:
    qrels: Qrels = {1: {10: 1}, 2: {20: 1}, 3: {30: 1}}
    shuffled = shuffled_qrels(qrels, seed=0)
    assert set(shuffled) == set(qrels)
    assert all(shuffled[q] != qrels[q] for q in qrels)


def test_the_shuffle_keeps_the_shape_of_the_judgements() -> None:
    """Same queries, same counts, only the pairing is wrong."""
    qrels: Qrels = {1: {10: 1}, 2: {20: 1, 21: 1}, 3: {30: 1}}
    shuffled = shuffled_qrels(qrels, seed=0)
    assert sorted(len(v) for v in shuffled.values()) == sorted(len(v) for v in qrels.values())


def test_the_randomised_control_is_also_a_derangement() -> None:
    qrels: Qrels = {q: {q * 10: 1} for q in range(1, 20)}
    randomised = randomised_qrels(qrels, seed=7)
    assert all(randomised[q] != qrels[q] for q in qrels)


def test_the_randomised_control_is_reproducible() -> None:
    qrels: Qrels = {q: {q * 10: 1} for q in range(1, 20)}
    assert randomised_qrels(qrels, seed=7) == randomised_qrels(qrels, seed=7)


def test_a_single_query_cannot_be_deranged() -> None:
    with pytest.raises(ValueError):
        shuffled_qrels({1: {10: 1}}, seed=0)
