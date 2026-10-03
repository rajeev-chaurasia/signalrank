"""Command line entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from signalrank.corpus import CorpusSpec, build_corpus

app = typer.Typer(
    add_completion=False,
    help="Retrieval whose quality is re-measured under the index that serves it.",
)


@app.callback()
def main() -> None:
    """Keep subcommands addressable by name.

    Without a callback typer collapses a single-command application into a
    bare one, and the subcommand name then parses as a stray argument.
    """


@app.command("build-corpus")
def build_corpus_command(
    raw: Annotated[Path, typer.Option(help="Directory holding the MS MARCO files.")] = Path(
        "data/raw"
    ),
    out: Annotated[Path, typer.Option(help="Where to write the sampled corpus.")] = Path(
        "data/build"
    ),
    passages: Annotated[int, typer.Option()] = 1_200_000,
    held_out: Annotated[int, typer.Option()] = 3_000,
    multi_intent: Annotated[int, typer.Option()] = 2_000,
    pairs: Annotated[int, typer.Option()] = 160_000,
    seed: Annotated[int, typer.Option()] = 20261103,
) -> None:
    """Sample the corpus and cut the splits, reproducibly from the seed."""
    spec = CorpusSpec(
        passages=passages,
        held_out_queries=held_out,
        multi_intent_queries=multi_intent,
        training_pairs=pairs,
        seed=seed,
    )
    report = build_corpus(raw, out, spec)
    typer.echo(json.dumps(report.as_dict(), indent=2))


if __name__ == "__main__":
    app()
