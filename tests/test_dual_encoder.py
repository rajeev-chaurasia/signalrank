"""Tests for the pooling and the loss, which are where the training claim lives.

The encoder itself is a pretrained transformer and is not what is being
claimed. What is claimed is that mined hard negatives help, and that rests
entirely on the shape of the score matrix below, so that is what is pinned
here. These run on tensors alone and load no model.
"""

from __future__ import annotations

import torch

from signalrank.models.dual_encoder import contrastive_loss, mean_pool


def test_mean_pool_ignores_padding() -> None:
    """A padded position must not drag the average towards zero."""
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert torch.allclose(mean_pool(hidden, mask), torch.tensor([[2.0, 2.0]]))


def test_mean_pool_of_a_single_token_is_that_token() -> None:
    hidden = torch.tensor([[[4.0, 8.0], [0.0, 0.0]]])
    mask = torch.tensor([[1, 0]])
    assert torch.allclose(mean_pool(hidden, mask), torch.tensor([[4.0, 8.0]]))


def test_mean_pool_survives_an_all_padding_row() -> None:
    """Truncation can produce one, and a division by zero here would poison the batch."""
    hidden = torch.tensor([[[1.0, 1.0]]])
    mask = torch.tensor([[0]])
    assert torch.isfinite(mean_pool(hidden, mask)).all()


def unit(rows: list[list[float]]) -> torch.Tensor:
    return torch.nn.functional.normalize(torch.tensor(rows), p=2, dim=-1)


def test_a_perfect_alignment_has_almost_no_loss() -> None:
    queries = unit([[1.0, 0.0], [0.0, 1.0]])
    positives = queries.clone()
    assert contrastive_loss(queries, positives, None, temperature=0.02).item() < 0.01


def test_swapping_the_positives_is_punished() -> None:
    queries = unit([[1.0, 0.0], [0.0, 1.0]])
    aligned = contrastive_loss(queries, queries.clone(), None, temperature=0.05)
    swapped = contrastive_loss(queries, queries.flip(0), None, temperature=0.05)
    assert swapped > aligned


def test_a_hard_negative_raises_the_loss() -> None:
    """The point of mining: a near miss in the batch is what the model must learn.

    Same queries and positives either way. The only difference is a negative
    sitting close to query one, and the loss has to notice it.
    """
    queries = unit([[1.0, 0.0], [0.0, 1.0]])
    positives = queries.clone()
    near_miss = unit([[0.99, 0.14], [0.0, 1.0]])
    without = contrastive_loss(queries, positives, None, temperature=0.05)
    with_negative = contrastive_loss(queries, positives, near_miss, temperature=0.05)
    assert with_negative > without


def test_a_distant_negative_barely_moves_the_loss() -> None:
    """Random negatives teach little, which is the argument for mining them."""
    queries = unit([[1.0, 0.0], [0.0, 1.0]])
    positives = queries.clone()
    far = unit([[-1.0, 0.0], [0.0, -1.0]])
    without = contrastive_loss(queries, positives, None, temperature=0.05)
    with_far = contrastive_loss(queries, positives, far, temperature=0.05)
    assert abs(with_far.item() - without.item()) < 0.05


def test_the_label_is_the_row_index() -> None:
    """Negatives occupy the columns after the positives, never before them.

    If they were concatenated first, every label would point at a negative and
    the model would train towards the opposite of the objective while the loss
    still looked like it was falling.
    """
    queries = unit([[1.0, 0.0], [0.0, 1.0]])
    positives = queries.clone()
    negatives = unit([[0.9, 0.3]])
    loss = contrastive_loss(queries, positives, negatives, temperature=0.05)
    assert loss.item() < 1.0


def test_a_lower_temperature_sharpens_the_penalty() -> None:
    queries = unit([[1.0, 0.0], [0.0, 1.0]])
    swapped = queries.flip(0)
    assert contrastive_loss(queries, swapped, None, 0.02) > contrastive_loss(
        queries, swapped, None, 0.5
    )


def test_the_loss_carries_a_gradient() -> None:
    queries = unit([[1.0, 0.0], [0.0, 1.0]]).requires_grad_(True)
    contrastive_loss(queries, queries.detach().clone(), None, 0.05).backward()
    assert queries.grad is not None and torch.isfinite(queries.grad).all()
