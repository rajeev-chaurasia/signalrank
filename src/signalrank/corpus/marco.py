"""Reading the raw MS MARCO files.

The collection is streamed rather than loaded. It is 8.8 million passages and
2.9 GB of text, and holding it alongside a model and an index is the difference
between a run that fits in memory and one that swaps.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from signalrank.core import Qrels


def stream_collection(path: Path) -> Iterator[tuple[int, str]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            passage_id, _, text = line.partition("\t")
            yield int(passage_id), text.rstrip("\n")


def load_queries(path: Path, keep: set[int] | None = None) -> dict[int, str]:
    queries: dict[int, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            query_id, _, text = line.partition("\t")
            identifier = int(query_id)
            if keep is None or identifier in keep:
                queries[identifier] = text.rstrip("\n")
    return queries


def load_qrels(path: Path) -> Qrels:
    """Read a TREC-format judgement file.

    Grades of zero are kept rather than dropped. They mean a passage was looked
    at and rejected, and `relevant_for` is where that distinction is applied, so
    losing it here would make the two indistinguishable downstream.
    """
    qrels: Qrels = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) != 4:
                raise ValueError(f"{path}: expected four fields, found {len(fields)}: {line!r}")
            query_id, _, passage_id, grade = fields
            qrels.setdefault(int(query_id), {})[int(passage_id)] = int(grade)
    return qrels


def write_queries(path: Path, queries: dict[int, str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for query_id in sorted(queries):
            handle.write(f"{query_id}\t{queries[query_id]}\n")


def write_qrels(path: Path, qrels: Qrels) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for query_id in sorted(qrels):
            for passage_id in sorted(qrels[query_id]):
                handle.write(f"{query_id}\t0\t{passage_id}\t{qrels[query_id][passage_id]}\n")
