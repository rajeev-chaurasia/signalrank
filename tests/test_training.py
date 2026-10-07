"""One real pass of the training loop, at a size that fits anywhere.

Not a quality test. It exists because every expensive failure in training is a
plumbing failure: a label pointing at the wrong column, negatives that are
never looked up, a checkpoint that does not round trip. Those cost an hour of
rented GPU to discover and three seconds to catch here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from signalrank.models.training import TrainConfig, pick_device, train_dual_encoder

pytest.importorskip("transformers")


def fixture_data() -> tuple[
    list[tuple[int, int]], dict[int, list[int]], dict[int, str], dict[int, str]
]:
    queries = {q: f"what is topic {q % 7}" for q in range(32)}
    passages = {p: f"a passage about topic {p % 7} numbered {p}" for p in range(200)}
    pairs = [(q, q) for q in range(32)]
    negatives = {q: [p for p in range(100, 120)] for q in range(32)}
    return pairs, negatives, queries, passages


def model_is_available() -> bool:
    from transformers import AutoConfig

    try:
        AutoConfig.from_pretrained("distilbert-base-uncased")
    except Exception:
        return False
    return True


needs_model = pytest.mark.skipif(
    not model_is_available(), reason="distilbert-base-uncased is not reachable"
)


@needs_model
def test_a_short_run_reduces_the_loss(tmp_path: Path) -> None:
    """Plumbing, not quality: a loop wired backwards does not go down."""
    pairs, negatives, queries, passages = fixture_data()
    report = train_dual_encoder(
        config=TrainConfig(epochs=1, batch_size=8, negatives_per_query=2, max_steps=4, log_every=1),
        pairs=pairs,
        negatives=negatives,
        queries=queries,
        passages=passages,
        out=tmp_path / "run",
    )
    assert report.steps == 4
    losses = [entry["loss"] for entry in report.losses]
    assert losses[-1] < losses[0], f"loss did not fall across four steps: {losses}"


@needs_model
def test_a_checkpoint_is_written_and_can_be_reloaded(tmp_path: Path) -> None:
    """An hour of training that saves something unloadable is an hour lost."""
    from signalrank.models.dual_encoder import DualEncoder

    pairs, negatives, queries, passages = fixture_data()
    out = tmp_path / "run"
    train_dual_encoder(
        config=TrainConfig(epochs=1, batch_size=8, max_steps=2),
        pairs=pairs,
        negatives=negatives,
        queries=queries,
        passages=passages,
        out=out,
    )
    assert (out / "training.json").exists()
    reloaded = DualEncoder(str(out))
    assert reloaded.dim > 0


@needs_model
def test_training_runs_without_any_mined_negatives(tmp_path: Path) -> None:
    """In-batch only is the comparison that shows whether mining earned its cost."""
    pairs, _, queries, passages = fixture_data()
    report = train_dual_encoder(
        config=TrainConfig(epochs=1, batch_size=8, max_steps=2),
        pairs=pairs,
        negatives={},
        queries=queries,
        passages=passages,
        out=tmp_path / "run",
    )
    assert report.steps == 2


def test_the_device_choice_is_one_torch_actually_has() -> None:
    import torch

    device = pick_device()
    assert device.type in {"cuda", "mps", "cpu"}
    if device.type == "mps":
        assert torch.backends.mps.is_available()
