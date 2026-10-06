"""The two indexes the corpus is searched with.

The whole second half of the claim lives in the fact that these two classes
have the same interface. Quality is normally measured with exact search over
full-precision vectors, and then the system ships an approximate index over
compressed ones. Making both answer the same `search` call means the evaluation
cannot tell them apart, so the same code produces both numbers and the
difference between them is the thing being reported.

Nothing here imports torch, and that is load bearing rather than tidy. faiss
and torch each bring their own OpenMP runtime, and a process that loads both
aborts on macOS. The documented workaround sets a flag that the OpenMP authors
say may silently produce incorrect results, which is not a trade a repository
about measuring things correctly can make. So encoding writes vectors to a
memory-mapped file and indexing reads them back, in a separate process.
"""

from __future__ import annotations

from collections.abc import Sequence

import faiss
import numpy as np

from signalrank.core import Run


class FlatIndex:
    """Exact inner product over full-precision vectors.

    The vectors are unit length, so inner product is cosine and this is the
    ceiling the quantized index is measured against. It holds the whole matrix
    in memory, which is the reason it is not what serves.
    """

    def __init__(self, vectors: np.ndarray, passage_ids: np.ndarray) -> None:
        self.passage_ids = passage_ids
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(np.ascontiguousarray(vectors))

    def search(self, queries: np.ndarray, k: int) -> list[list[int]]:
        _, positions = self.index.search(np.ascontiguousarray(queries), k)
        return [self.passage_ids[row].tolist() for row in positions]


class IvfPqIndex:
    """Inverted lists over product-quantized codes, with an exact rerank.

    Three knobs decide what this costs and what it gives up. `nlist` sets how
    many lists the corpus is cut into and `nprobe` how many are opened per
    query, which together trade latency for recall. `m` sets how many bytes a
    vector is compressed to, which is where the quality loss the claim is about
    actually happens.

    The rerank is the part that makes the loss recoverable: candidates come back
    from the compressed codes, and the top of that list is rescored against the
    full-precision vectors before the cut. Fetching deeper costs only the
    rescore, so the depth is a dial on how much of the quantization loss is
    bought back.
    """

    def __init__(
        self,
        vectors: np.ndarray,
        passage_ids: np.ndarray,
        *,
        nlist: int = 4096,
        m: int = 96,
        nbits: int = 8,
        seed: int = 20261103,
    ) -> None:
        dim = vectors.shape[1]
        if dim % m:
            raise ValueError(f"product quantization needs m to divide {dim}, given {m}")
        self.passage_ids = passage_ids
        self.vectors = vectors
        quantizer = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
        self.index.cp.seed = seed

        # Training on a sample rather than the whole corpus: the centroids only
        # need the distribution, and a full pass costs minutes for no gain.
        sample = min(len(vectors), max(nlist * 40, 100_000))
        rng = np.random.default_rng(seed)
        rows = (
            rng.choice(len(vectors), size=sample, replace=False) if sample < len(vectors) else None
        )
        training = np.ascontiguousarray(vectors if rows is None else vectors[np.sort(rows)])
        self.index.train(training)
        self.index.add(np.ascontiguousarray(vectors))

    @property
    def nprobe(self) -> int:
        return int(self.index.nprobe)

    @nprobe.setter
    def nprobe(self, value: int) -> None:
        self.index.nprobe = value

    def search(self, queries: np.ndarray, k: int, rerank_depth: int = 1) -> list[list[int]]:
        depth = max(k, k * rerank_depth)
        _, positions = self.index.search(np.ascontiguousarray(queries), depth)
        if rerank_depth <= 1:
            return [self.passage_ids[row[row >= 0]][:k].tolist() for row in positions]

        ranked: list[list[int]] = []
        for query, row in zip(queries, positions, strict=True):
            candidates = row[row >= 0]
            if candidates.size == 0:
                ranked.append([])
                continue
            exact = self.vectors[candidates] @ query
            best = candidates[np.argsort(-exact)[:k]]
            ranked.append(self.passage_ids[best].tolist())
        return ranked


def search_to_run(
    index: FlatIndex | IvfPqIndex,
    query_ids: Sequence[int],
    vectors: np.ndarray,
    k: int,
    **kwargs: int,
) -> Run:
    ranked = index.search(vectors, k, **kwargs)
    return dict(zip(query_ids, ranked, strict=True))
