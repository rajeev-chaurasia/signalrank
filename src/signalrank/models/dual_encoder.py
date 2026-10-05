"""The dual encoder, and the loss that trains it.

Written out rather than taken from a library, because the thing being claimed is
that mined hard negatives help, and the loss is where that claim lives. A
library call would make the in-batch and the mined negatives into a
configuration flag, and the one line that matters, the shape of the score
matrix, would be somewhere else.

Queries and passages share one set of weights. Separate towers are the other
reasonable choice and usually score a little better, at twice the parameters to
train and twice the memory; with 160,000 pairs the shared tower is the one that
fits the budget, and the comparison here is against BM25 rather than against a
larger model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer

BASE_MODEL = "distilbert-base-uncased"
MAX_QUERY_TOKENS = 32
MAX_PASSAGE_TOKENS = 192


def mean_pool(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    """Average the tokens that are not padding.

    Mean pooling rather than the first token: the first token's representation
    is only meaningful when a model has been trained to use it that way, and the
    base model here was not trained for retrieval at all.
    """
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    counted = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counted


class DualEncoder(nn.Module):
    def __init__(self, base_model: str = BASE_MODEL) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

    @property
    def dim(self) -> int:
        return int(self.encoder.config.hidden_size)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # Normalised here rather than at the call sites, so a dot product is a
        # cosine everywhere and the index cannot disagree with the loss.
        return functional.normalize(mean_pool(hidden, attention_mask), p=2, dim=-1)

    def tokenize(self, texts: list[str], max_tokens: int) -> dict[str, Tensor]:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_tokens,
            return_tensors="pt",
        )
        return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}


def contrastive_loss(
    queries: Tensor,
    positives: Tensor,
    negatives: Tensor | None,
    temperature: float,
) -> Tensor:
    """Cross entropy over every passage in the batch, with the positive as the label.

    The score matrix is the whole point. Each query is scored against its own
    positive, against every other query's positive, and against every mined
    negative in the batch. The other queries' positives are the in-batch
    negatives and are nearly free; the mined ones are the expensive half, and
    they are what the claim is about.

    Passing `negatives=None` gives in-batch training alone, which is the
    comparison that shows whether the mining earned its cost.
    """
    batch = queries.size(0)
    candidates = positives if negatives is None else torch.cat([positives, negatives], dim=0)
    scores = queries @ candidates.t() / temperature
    # Candidate i is query i's positive for i below the batch size, so the label
    # is the row index and the mined negatives occupy the columns after them.
    labels = torch.arange(batch, device=queries.device)
    return functional.cross_entropy(scores, labels)
