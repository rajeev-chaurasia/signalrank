"""Tests for the two properties that would quietly invalidate every result.

If a passage judged relevant for an evaluation query is missing from the
sampled corpus, recall becomes partly a statement about the sample. If an
evaluation query appears in training, every number is reported on data the
model was fitted to. Neither failure shows up as an error at run time; both
show up as a better score.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from signalrank.core import relevant_for
from signalrank.corpus import CorpusSpec, build_corpus
from signalrank.corpus.marco import load_qrels, load_queries


def write_raw(root: Path, *, dev_queries: int = 40, train_queries: int = 60) -> Path:
    """A miniature MS MARCO: enough structure to exercise the real code path."""
    raw = root / "raw"
    raw.mkdir(parents=True)

    (raw / "collection.tsv").write_text(
        "".join(f"{pid}\tpassage number {pid}\n" for pid in range(5_000)), encoding="utf-8"
    )
    (raw / "queries.dev.tsv").write_text(
        "".join(f"{qid}\tdev query {qid}\n" for qid in range(dev_queries)), encoding="utf-8"
    )
    (raw / "qrels.dev.small.tsv").write_text(
        "".join(f"{qid}\t0\t{1000 + qid}\t1\n" for qid in range(dev_queries)), encoding="utf-8"
    )

    base = 10_000
    (raw / "queries.train.tsv").write_text(
        "".join(f"{base + qid}\ttrain query {qid}\n" for qid in range(train_queries)),
        encoding="utf-8",
    )
    # Every third training query gets two relevant passages, so there is a
    # multi-intent pool to draw from, as in the real judgements.
    lines = []
    for qid in range(train_queries):
        lines.append(f"{base + qid}\t0\t{2000 + qid}\t1\n")
        if qid % 3 == 0:
            lines.append(f"{base + qid}\t0\t{3000 + qid}\t1\n")
    (raw / "qrels.train.tsv").write_text("".join(lines), encoding="utf-8")
    return raw


@pytest.fixture
def built(tmp_path: Path) -> tuple[Path, CorpusSpec]:
    raw = write_raw(tmp_path)
    spec = CorpusSpec(
        passages=2_000,
        held_out_queries=10,
        multi_intent_queries=5,
        training_pairs=30,
        seed=11,
    )
    build_corpus(raw, tmp_path / "build", spec)
    return tmp_path / "build", spec


def passage_ids(build: Path) -> set[int]:
    return {
        int(line.split("\t")[0])
        for line in (build / "passages.tsv").read_text(encoding="utf-8").splitlines()
    }


def test_every_relevant_passage_for_an_evaluation_query_is_in_the_corpus(
    built: tuple[Path, CorpusSpec],
) -> None:
    build, _ = built
    present = passage_ids(build)
    for name in ("qrels.heldout.tsv", "qrels.multi.tsv"):
        qrels = load_qrels(build / name)
        needed = {pid for qid in qrels for pid in relevant_for(qrels, qid)}
        assert needed <= present, f"{name}: {len(needed - present)} relevant passages missing"


def test_no_evaluation_query_appears_in_training(built: tuple[Path, CorpusSpec]) -> None:
    build, _ = built
    pairs = [
        line.split("\t")
        for line in (build / "pairs.train.tsv").read_text(encoding="utf-8").splitlines()
    ]
    trained_on = {int(qid) for qid, _ in pairs}
    multi = set(load_qrels(build / "qrels.multi.tsv"))
    assert trained_on & multi == set()


def test_the_multi_intent_split_only_holds_multi_intent_queries(
    built: tuple[Path, CorpusSpec],
) -> None:
    build, _ = built
    qrels = load_qrels(build / "qrels.multi.tsv")
    assert qrels
    assert all(len(relevant_for(qrels, qid)) > 1 for qid in qrels)


def test_every_split_query_has_its_text(built: tuple[Path, CorpusSpec]) -> None:
    build, _ = built
    for queries, qrels in (
        ("queries.heldout.tsv", "qrels.heldout.tsv"),
        ("queries.multi.tsv", "qrels.multi.tsv"),
    ):
        assert set(load_qrels(build / qrels)) <= set(load_queries(build / queries))


def test_the_corpus_is_the_size_it_was_asked_for(built: tuple[Path, CorpusSpec]) -> None:
    build, spec = built
    assert len(passage_ids(build)) == spec.passages


def test_the_same_seed_builds_the_same_corpus(tmp_path: Path) -> None:
    raw = write_raw(tmp_path)
    spec = CorpusSpec(
        passages=2_000, held_out_queries=10, multi_intent_queries=5, training_pairs=30, seed=11
    )
    first = tmp_path / "a"
    second = tmp_path / "b"
    build_corpus(raw, first, spec)
    build_corpus(raw, second, spec)
    for name in ("passages.tsv", "pairs.train.tsv", "qrels.multi.tsv", "qrels.heldout.tsv"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_a_different_seed_builds_a_different_corpus(tmp_path: Path) -> None:
    raw = write_raw(tmp_path)
    base = CorpusSpec(
        passages=2_000, held_out_queries=10, multi_intent_queries=5, training_pairs=30, seed=11
    )
    other = CorpusSpec(
        passages=2_000, held_out_queries=10, multi_intent_queries=5, training_pairs=30, seed=12
    )
    build_corpus(raw, tmp_path / "a", base)
    build_corpus(raw, tmp_path / "b", other)
    assert (tmp_path / "a" / "passages.tsv").read_bytes() != (
        tmp_path / "b" / "passages.tsv"
    ).read_bytes()


def test_a_corpus_too_small_to_hold_the_judgements_is_refused(tmp_path: Path) -> None:
    """Silently dropping them would make recall a measure of the sample."""
    raw = write_raw(tmp_path)
    spec = CorpusSpec(
        passages=5, held_out_queries=10, multi_intent_queries=5, training_pairs=30, seed=11
    )
    with pytest.raises(ValueError, match="more than"):
        build_corpus(raw, tmp_path / "build", spec)


def test_a_shortfall_of_multi_intent_queries_is_recorded(tmp_path: Path) -> None:
    """The plan says publish the shortfall rather than widen the definition."""
    raw = write_raw(tmp_path)
    spec = CorpusSpec(
        passages=2_000, held_out_queries=10, multi_intent_queries=10_000, training_pairs=30, seed=11
    )
    report = build_corpus(raw, tmp_path / "build", spec)
    assert report.multi_intent_queries < 10_000
    assert any("short of" in note for note in report.notes)
