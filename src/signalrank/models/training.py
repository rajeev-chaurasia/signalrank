"""Training the dual encoder.

The loop is deliberately plain. The one decision in it worth defending is that
mined negatives are concatenated after the positives rather than mixed in,
because the label for query i has to stay at column i; getting that backwards
trains the model towards the opposite of the objective while the loss still
falls, which is the kind of bug that is only visible in the evaluation.

Passages are loaded for the identifiers training actually touches, not for the
whole corpus. The corpus is 1.2 million passages and training sees about a
third of them, so holding all of it would cost a few hundred megabytes to no
purpose while a model is on the device.
"""

from __future__ import annotations

import json
import random
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor
from torch.optim import AdamW

from signalrank.models.dual_encoder import (
    MAX_PASSAGE_TOKENS,
    MAX_QUERY_TOKENS,
    DualEncoder,
    contrastive_loss,
)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 3
    batch_size: int = 32
    negatives_per_query: int = 2
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    temperature: float = 0.05
    max_steps: int | None = None
    seed: int = 20261103
    log_every: int = 100
    # bfloat16 on a card that has tensor cores, which an L4 does. The first
    # full run was left in float32 and spent 1.3 seconds a step where the
    # arithmetic says 0.37. bfloat16 rather than float16 because it keeps
    # float32's exponent range, so a contrastive loss over normalised
    # vectors needs no gradient scaler and cannot silently overflow.
    mixed_precision: bool = True


@dataclass
class TrainReport:
    steps: int = 0
    epochs: int = 0
    pairs: int = 0
    seconds: float = 0.0
    device: str = ""
    precision: str = ""
    losses: list[dict[str, float]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "steps": self.steps,
            "epochs": self.epochs,
            "pairs": self.pairs,
            "seconds": round(self.seconds, 1),
            "device": self.device,
            "precision": self.precision,
            "losses": self.losses,
        }


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _batches(
    pairs: list[tuple[int, int]], size: int, rng: random.Random
) -> list[list[tuple[int, int]]]:
    order = pairs[:]
    rng.shuffle(order)
    return [order[start : start + size] for start in range(0, len(order), size)]


def train_dual_encoder(
    *,
    config: TrainConfig,
    pairs: list[tuple[int, int]],
    negatives: dict[int, list[int]],
    queries: dict[int, str],
    passages: dict[int, str],
    out: Path,
) -> TrainReport:
    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)
    device = pick_device()

    model = DualEncoder().to(device)
    model.train()
    optimiser = AdamW(model.parameters(), lr=config.learning_rate)

    usable = [(q, p) for q, p in pairs if q in queries and p in passages]
    batches = _batches(usable, config.batch_size, rng)
    total = len(batches) * config.epochs if config.max_steps is None else config.max_steps
    warmup = max(1, int(total * config.warmup_ratio))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser,
        lambda step: (
            min(1.0, (step + 1) / warmup) * max(0.0, (total - step) / max(1, total - warmup))
        ),
    )

    precision = "bfloat16" if config.mixed_precision and device.type == "cuda" else "float32"
    report = TrainReport(pairs=len(usable), device=str(device), precision=precision)
    started = time.perf_counter()
    step = 0

    for epoch in range(config.epochs):
        for batch in _batches(usable, config.batch_size, rng) if epoch else batches:
            if config.max_steps is not None and step >= config.max_steps:
                break
            query_text = [queries[q] for q, _ in batch]
            positive_text = [passages[p] for _, p in batch]
            negative_text = _negatives_for(batch, negatives, passages, config, rng)

            loss = _step(model, query_text, positive_text, negative_text, config, device)
            # torch ships no annotation for backward, and loosening
            # disallow_untyped_calls across the project to accept one call
            # would give up the check everywhere else.
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            schedule.step()
            optimiser.zero_grad(set_to_none=True)

            step += 1
            if step % config.log_every == 0 or step == 1:
                entry = {"step": step, "loss": round(loss.item(), 5)}
                report.losses.append(entry)
                # Printed as well as collected. The report is only written when
                # training ends, so a run with an hour left to go was otherwise
                # indistinguishable from a run that had hung.
                rate = step / max(time.perf_counter() - started, 1e-9)
                remaining = (total - step) / rate if rate else 0.0
                print(
                    f"step {step}/{total} loss {entry['loss']} "
                    f"{rate:.2f} steps/s, about {remaining / 60:.0f} min left",
                    flush=True,
                )
        report.epochs = epoch + 1
        if config.max_steps is not None and step >= config.max_steps:
            break

    report.steps = step
    report.seconds = time.perf_counter() - started

    out.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(out)
    model.tokenizer.save_pretrained(out)
    (out / "training.json").write_text(
        json.dumps({"config": config.__dict__, "report": report.as_dict()}, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _negatives_for(
    batch: list[tuple[int, int]],
    negatives: dict[int, list[int]],
    passages: dict[int, str],
    config: TrainConfig,
    rng: random.Random,
) -> list[str]:
    """One run of mined negatives per query, drawn fresh each time it is seen.

    Drawing rather than taking the top few means a query contributes different
    near misses across epochs, which is most of the benefit of mining more of
    them than a batch can hold.
    """
    chosen: list[str] = []
    for query_id, _ in batch:
        pool = [p for p in negatives.get(query_id, []) if p in passages]
        if not pool:
            continue
        picked = rng.sample(pool, min(config.negatives_per_query, len(pool)))
        chosen.extend(passages[p] for p in picked)
    return chosen


def _autocast(config: TrainConfig, device: torch.device) -> AbstractContextManager[None]:
    """Half precision where the hardware has tensor cores, and nowhere else.

    Only CUDA is worth it. Apple's backend runs this model in float32 at much
    the same speed either way, and turning it on there would change the
    numerics of a local smoke run against the numbers the GPU produced.
    """
    if not config.mixed_precision or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _step(
    model: DualEncoder,
    query_text: list[str],
    positive_text: list[str],
    negative_text: list[str],
    config: TrainConfig,
    device: torch.device,
) -> Tensor:
    def encode(texts: list[str], max_tokens: int) -> Tensor:
        batch = {k: v.to(device) for k, v in model.tokenize(texts, max_tokens).items()}
        encoded: Tensor = model(**batch)
        return encoded

    with _autocast(config, device):
        queries = encode(query_text, MAX_QUERY_TOKENS)
        positives = encode(positive_text, MAX_PASSAGE_TOKENS)
        negatives = encode(negative_text, MAX_PASSAGE_TOKENS) if negative_text else None
        # The loss itself is computed in float32. The score matrix is a
        # softmax over a few hundred columns divided by a small temperature,
        # which is exactly where reduced precision costs accuracy for no speed.
        return contrastive_loss(
            queries.float(),
            positives.float(),
            None if negatives is None else negatives.float(),
            config.temperature,
        )
