"""Tests for the baseline and the negative miner.

Two failures here would be invisible downstream. A ranking returned in the
index's internal position space rather than in passage identifiers scores
against the wrong passages and looks like a bad model. A judged positive left
in the mined negatives trains the encoder to push away the thing it is measured
on, and looks like a bad training recipe.
"""

from __future__ import annotations

from pathlib import Path

from signalrank.core import Qrels
from signalrank.mining.bm25 import BM25Index, mine_negatives, run_for

PASSAGES = {
    500: "the manhattan project built the first atomic bomb",
    501: "atomic clocks measure time using caesium atoms",
    502: "a recipe for sourdough bread with a rye starter",
    503: "bread baking temperature and crumb structure",
    504: "the manhattan skyline at night from brooklyn",
    505: "nuclear fission releases energy from heavy atoms",
}


def index() -> BM25Index:
    ids = sorted(PASSAGES)
    return BM25Index.build(ids, [PASSAGES[i] for i in ids])


def test_search_returns_passage_identifiers_not_positions() -> None:
    """Identifiers here start at 500, so a leaked position would be obvious."""
    ranked = index().search(["atomic bomb"], k=3)[0]
    assert all(doc_id in PASSAGES for doc_id in ranked)


def test_search_puts_the_obvious_match_first() -> None:
    ranked = index().search(["sourdough bread recipe"], k=3)[0]
    assert ranked[0] == 502


def test_search_respects_the_requested_depth() -> None:
    assert len(index().search(["atomic"], k=2)[0]) == 2


def test_a_depth_beyond_the_corpus_is_clamped() -> None:
    assert len(index().search(["atomic"], k=999)[0]) == len(PASSAGES)


def test_no_queries_returns_nothing() -> None:
    assert index().search([], k=5) == []


def test_run_for_keys_results_by_query_identifier() -> None:
    run = run_for(index(), {11: "atomic bomb", 12: "sourdough"}, k=2)
    assert sorted(run) == [11, 12]
    assert len(run[11]) == 2


def test_mined_negatives_exclude_judged_positives() -> None:
    """A judged positive is not a negative however highly BM25 ranks it."""
    qrels: Qrels = {11: {500: 1, 505: 1}}
    mined = mine_negatives(index(), {11: "atomic bomb energy"}, qrels, depth=6, per_query=5)
    assert 500 not in mined[11]
    assert 505 not in mined[11]


def test_mined_negatives_are_capped_per_query() -> None:
    qrels: Qrels = {11: {500: 1}}
    mined = mine_negatives(index(), {11: "atomic"}, qrels, depth=6, per_query=2)
    assert len(mined[11]) == 2


def test_a_zero_graded_passage_is_still_available_as_a_negative() -> None:
    """Judged and rejected is exactly what a hard negative is."""
    qrels: Qrels = {11: {500: 1, 505: 0}}
    mined = mine_negatives(index(), {11: "atomic energy"}, qrels, depth=6, per_query=5)
    assert 505 in mined[11]


def test_mining_a_query_with_no_judgements_still_yields_negatives() -> None:
    mined = mine_negatives(index(), {99: "bread"}, {}, depth=6, per_query=3)
    assert len(mined[99]) == 3


def test_an_index_survives_a_round_trip_through_disk(tmp_path: Path) -> None:
    built = index()
    built.save(tmp_path)
    loaded = BM25Index.load(tmp_path)
    assert loaded.search(["sourdough bread"], k=3)[0] == built.search(["sourdough bread"], k=3)[0]
