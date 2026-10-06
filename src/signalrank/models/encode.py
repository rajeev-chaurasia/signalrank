"""Turning text into vectors, and writing them where faiss can read them.

Kept apart from the indexing code on purpose. faiss and torch each link their
own OpenMP runtime, and a process that loads both aborts on macOS; the
documented workaround is a flag whose own authors warn it may silently produce
incorrect results. Encoding therefore ends at a file on disk, and indexing
starts from one.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from signalrank.models.dual_encoder import MAX_PASSAGE_TOKENS, MAX_QUERY_TOKENS, DualEncoder


def _encode(
    model: DualEncoder,
    texts: Sequence[str],
    max_tokens: int,
    batch_size: int,
    device: torch.device,
    into: np.ndarray | None = None,
) -> np.ndarray:
    out = np.empty((len(texts), model.dim), dtype=np.float32) if into is None else into
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            chunk = list(texts[start : start + batch_size])
            batch = {k: v.to(device) for k, v in model.tokenize(chunk, max_tokens).items()}
            out[start : start + len(chunk)] = model(**batch).float().cpu().numpy()
    return out


def encode_passages(
    model: DualEncoder,
    passage_ids: Sequence[int],
    texts: Sequence[str],
    *,
    out: Path,
    device: torch.device,
    batch_size: int = 128,
) -> Path:
    """Encode the corpus into a memory-mapped array beside its identifiers.

    At 768 dimensions 1.2 million passages are 3.7 GB in float32, which does
    not want to be resident alongside the model that produced it.
    """
    out.mkdir(parents=True, exist_ok=True)
    vectors = np.lib.format.open_memmap(
        out / "passages.npy", mode="w+", dtype=np.float32, shape=(len(texts), model.dim)
    )
    _encode(model, texts, MAX_PASSAGE_TOKENS, batch_size, device, into=vectors)
    vectors.flush()
    np.save(out / "passage_ids.npy", np.asarray(passage_ids, dtype=np.int64))
    return out


def encode_queries(
    model: DualEncoder,
    texts: Sequence[str],
    *,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    return _encode(model, texts, MAX_QUERY_TOKENS, batch_size, device)
