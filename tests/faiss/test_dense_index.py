"""Tests for the two indexes, and for the rerank that is the claim's hinge.

The exact index is the ceiling. The quantized one is what would actually
serve. The repository claims the gap between them is small, so what matters
here is that both answer the same call, that neither leaks an internal
position in place of a passage identifier, and that reranking recovers
accuracy the compression threw away.
"""

from __future__ import annotations

import numpy as np
import pytest

from signalrank.index.dense import FlatIndex, IvfPqIndex, search_to_run

DIM = 16
COUNT = 512
FIRST_ID = 90_000


def corpus() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(5)
    vectors = rng.standard_normal((COUNT, DIM)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    # Identifiers deliberately far from positions, so a leaked position shows.
    return vectors, np.arange(FIRST_ID, FIRST_ID + COUNT, dtype=np.int64)


def test_exact_search_returns_identifiers_not_positions() -> None:
    vectors, ids = corpus()
    found = FlatIndex(vectors, ids).search(vectors[:3], k=5)
    assert all(doc >= FIRST_ID for row in found for doc in row)


def test_exact_search_finds_a_vector_by_itself() -> None:
    vectors, ids = corpus()
    found = FlatIndex(vectors, ids).search(vectors[7:8], k=1)
    assert found[0][0] == ids[7]


def test_exact_search_respects_the_cut() -> None:
    vectors, ids = corpus()
    assert len(FlatIndex(vectors, ids).search(vectors[:1], k=9)[0]) == 9


def quantized(vectors: np.ndarray, ids: np.ndarray) -> IvfPqIndex:
    index = IvfPqIndex(vectors, ids, nlist=4, m=4, nbits=4)
    index.nprobe = 4
    return index


def test_quantized_search_returns_identifiers_not_positions() -> None:
    vectors, ids = corpus()
    found = quantized(vectors, ids).search(vectors[:3], k=5)
    assert all(doc >= FIRST_ID for row in found for doc in row)


def test_quantization_that_divides_the_dimension_is_required() -> None:
    vectors, ids = corpus()
    with pytest.raises(ValueError, match="divide"):
        IvfPqIndex(vectors, ids, nlist=4, m=5, nbits=4)


def agreement(found: list[list[int]], truth: list[list[int]]) -> float:
    return float(
        np.mean([len(set(a) & set(b)) / len(b) for a, b in zip(found, truth, strict=True)])
    )


def test_reranking_recovers_accuracy_the_compression_lost() -> None:
    """The mechanism the second half of the claim depends on.

    Compressed codes pick the candidates; the full-precision vectors decide the
    order. Fetching deeper before rescoring should move the result towards what
    exact search would have returned.
    """
    vectors, ids = corpus()
    truth = FlatIndex(vectors, ids).search(vectors[:40], k=10)
    index = quantized(vectors, ids)
    shallow = agreement(index.search(vectors[:40], k=10, rerank_depth=1), truth)
    deep = agreement(index.search(vectors[:40], k=10, rerank_depth=8), truth)
    assert deep >= shallow
    assert deep > 0.5


def test_a_deeper_rerank_never_returns_more_than_asked_for() -> None:
    vectors, ids = corpus()
    found = quantized(vectors, ids).search(vectors[:5], k=10, rerank_depth=8)
    assert all(len(row) <= 10 for row in found)


def test_the_quantized_index_is_reproducible() -> None:
    vectors, ids = corpus()
    first = quantized(vectors, ids).search(vectors[:10], k=5, rerank_depth=4)
    second = quantized(vectors, ids).search(vectors[:10], k=5, rerank_depth=4)
    assert first == second


def test_a_run_is_keyed_by_query_identifier() -> None:
    vectors, ids = corpus()
    run = search_to_run(FlatIndex(vectors, ids), [11, 12], vectors[:2], k=3)
    assert sorted(run) == [11, 12]
    assert len(run[11]) == 3


def test_both_indexes_answer_the_same_call() -> None:
    """If they diverged here, the two published numbers would not be comparable."""
    vectors, ids = corpus()
    exact = FlatIndex(vectors, ids).search(vectors[:4], k=5)
    approximate = quantized(vectors, ids).search(vectors[:4], k=5)
    assert len(exact) == len(approximate) == 4
    assert all(len(row) == 5 for row in exact)
