"""Cutting MS MARCO into the corpus and splits the claim is measured on.

Two decisions here are the ones that could quietly invalidate every number
downstream, so both are made once, by seed, and written into a manifest.

The sample must contain every passage judged relevant for every evaluation
query. Without that guarantee, a recall of 0.85 would partly be a statement
about which passages the sample happened to include, and a model could be made
to look better by sampling differently.

The evaluation queries must never appear in training. The multi-intent set is
drawn from the training judgements, because the development set contains only
390 queries with more than one relevant passage and the claim is stated over
2,000. Drawing them from training is sound only if they are removed from it,
which is done here rather than left to the training script to remember.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from signalrank.core import Qrels, relevant_for
from signalrank.corpus.marco import (
    load_qrels,
    load_queries,
    stream_collection,
    write_qrels,
    write_queries,
)


@dataclass(frozen=True)
class CorpusSpec:
    passages: int = 1_200_000
    held_out_queries: int = 3_000
    multi_intent_queries: int = 2_000
    training_pairs: int = 160_000
    seed: int = 20261103


@dataclass
class BuildReport:
    passages_written: int = 0
    passages_forced: int = 0
    held_out_queries: int = 0
    multi_intent_queries: int = 0
    multi_intent_available: int = 0
    training_pairs: int = 0
    training_queries: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "passages_written": self.passages_written,
            "passages_forced": self.passages_forced,
            "held_out_queries": self.held_out_queries,
            "multi_intent_queries": self.multi_intent_queries,
            "multi_intent_available": self.multi_intent_available,
            "training_pairs": self.training_pairs,
            "training_queries": self.training_queries,
            "notes": self.notes,
        }


def _multi_intent(qrels: Qrels) -> list[int]:
    return sorted(q for q in qrels if len(relevant_for(qrels, q)) > 1)


def _subset(qrels: Qrels, query_ids: list[int]) -> Qrels:
    return {query_id: dict(qrels[query_id]) for query_id in query_ids}


def build_corpus(raw: Path, out: Path, spec: CorpusSpec) -> BuildReport:
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(spec.seed)
    report = BuildReport()

    dev_qrels = load_qrels(raw / "qrels.dev.small.tsv")
    train_qrels = load_qrels(raw / "qrels.train.tsv")

    held_out_ids = sorted(dev_qrels)
    rng.shuffle(held_out_ids)
    held_out_ids = sorted(held_out_ids[: spec.held_out_queries])

    candidates = _multi_intent(train_qrels)
    report.multi_intent_available = len(candidates)
    rng.shuffle(candidates)
    multi_ids = sorted(candidates[: spec.multi_intent_queries])
    if len(multi_ids) < spec.multi_intent_queries:
        report.notes.append(
            f"only {len(multi_ids)} training queries have more than one relevant passage, "
            f"short of the {spec.multi_intent_queries} the plan assumes"
        )

    # Held out before any pair is drawn, so a training pair cannot reference one
    # of these queries however the sampling below changes.
    withheld = set(multi_ids)
    training_ids = [q for q in sorted(train_qrels) if q not in withheld]
    rng.shuffle(training_ids)

    pairs: list[tuple[int, int]] = []
    used_training: list[int] = []
    for query_id in training_ids:
        if len(pairs) >= spec.training_pairs:
            break
        positives = sorted(relevant_for(train_qrels, query_id))
        if not positives:
            continue
        used_training.append(query_id)
        for passage_id in positives:
            if len(pairs) >= spec.training_pairs:
                break
            pairs.append((query_id, passage_id))

    held_out_qrels = _subset(dev_qrels, held_out_ids)
    multi_qrels = _subset(train_qrels, multi_ids)

    forced: set[int] = set()
    for qrels in (held_out_qrels, multi_qrels):
        for query_id in qrels:
            forced |= relevant_for(qrels, query_id)
    forced |= {passage_id for _, passage_id in pairs}
    report.passages_forced = len(forced)

    collection = raw / "collection.tsv"
    keep = _choose_passages(collection, forced, spec)
    report.passages_written = _write_passages(collection, out / "passages.tsv", keep)
    if report.passages_written != spec.passages:
        raise RuntimeError(
            f"wrote {report.passages_written} passages, not the {spec.passages} asked for"
        )

    write_queries(
        out / "queries.heldout.tsv",
        load_queries(raw / "queries.dev.tsv", set(held_out_ids)),
    )
    write_qrels(out / "qrels.heldout.tsv", held_out_qrels)

    train_query_text = load_queries(raw / "queries.train.tsv", withheld | set(used_training))
    write_queries(
        out / "queries.multi.tsv",
        {q: train_query_text[q] for q in multi_ids if q in train_query_text},
    )
    write_qrels(out / "qrels.multi.tsv", multi_qrels)
    write_queries(
        out / "queries.train.tsv",
        {q: train_query_text[q] for q in used_training if q in train_query_text},
    )

    with (out / "pairs.train.tsv").open("w", encoding="utf-8") as handle:
        for query_id, passage_id in pairs:
            handle.write(f"{query_id}\t{passage_id}\n")

    report.held_out_queries = len(held_out_ids)
    report.multi_intent_queries = len(multi_ids)
    report.training_pairs = len(pairs)
    report.training_queries = len(used_training)

    (out / "manifest.json").write_text(
        json.dumps({"spec": spec.__dict__, "report": report.as_dict()}, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _choose_passages(collection: Path, forced: set[int], spec: CorpusSpec) -> set[int]:
    """Every passage an evaluation needs, then a random fill to the target size.

    The fill is drawn from the identifiers the collection actually contains,
    read off the file rather than assumed from a constant. A constant that
    drifts from the data produces a corpus quietly smaller than the plan
    specifies, and a corpus missing passages scores better, so the failure
    would arrive looking like a result.
    """
    if len(forced) > spec.passages:
        raise ValueError(
            f"{len(forced)} passages are needed by the splits, more than the "
            f"{spec.passages} the corpus is meant to hold"
        )

    with collection.open(encoding="utf-8") as handle:
        present = np.fromiter((int(line.split("\t", 1)[0]) for line in handle), dtype=np.int64)
    available = present[~np.isin(present, np.fromiter(forced, dtype=np.int64, count=len(forced)))]
    needed = spec.passages - len(forced)
    if needed > available.size:
        raise ValueError(
            f"the collection holds {present.size} passages, too few to fill a corpus of "
            f"{spec.passages}"
        )
    chosen = np.random.default_rng(spec.seed).choice(available, size=needed, replace=False)
    return forced | set(chosen.tolist())


def _write_passages(source: Path, destination: Path, keep: set[int]) -> int:
    written = 0
    with destination.open("w", encoding="utf-8") as handle:
        for passage_id, text in stream_collection(source):
            if passage_id in keep:
                handle.write(f"{passage_id}\t{text}\n")
                written += 1
    return written
