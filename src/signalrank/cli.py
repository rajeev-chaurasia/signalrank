"""Command line entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from signalrank.corpus import CorpusSpec, build_corpus
from signalrank.corpus.marco import load_qrels, load_queries, stream_collection
from signalrank.mining.bm25 import BM25Index, mine_negatives, run_for

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


@app.command("build-bm25")
def build_bm25_command(
    build: Annotated[Path, typer.Option(help="Directory holding the sampled corpus.")] = Path(
        "data/build"
    ),
) -> None:
    """Index the sampled corpus for the baseline and for mining."""
    passage_ids: list[int] = []
    texts: list[str] = []
    for passage_id, text in stream_collection(build / "passages.tsv"):
        passage_ids.append(passage_id)
        texts.append(text)
    typer.echo(f"indexing {len(passage_ids)} passages")
    index = BM25Index.build(passage_ids, texts)
    out = build / "bm25"
    out.mkdir(parents=True, exist_ok=True)
    index.save(out)
    typer.echo(f"saved to {out}")


@app.command("bm25-baseline")
def bm25_baseline_command(
    build: Annotated[Path, typer.Option()] = Path("data/build"),
    evidence: Annotated[Path, typer.Option()] = Path("evidence"),
    depth: Annotated[int, typer.Option(help="Ranking depth to score.")] = 100,
) -> None:
    """Score BM25 on both evaluation splits, which is the baseline to beat."""
    from signalrank.core import evaluate

    index = BM25Index.load(build / "bm25")
    evidence.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {}
    for split in ("heldout", "multi"):
        queries = load_queries(build / f"queries.{split}.tsv")
        qrels = load_qrels(build / f"qrels.{split}.tsv")
        run = run_for(index, queries, depth)
        results[split] = evaluate(run, qrels).as_dict()
        typer.echo(f"{split}: {results[split]}")
    (evidence / "bm25.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


@app.command("mine-negatives")
def mine_negatives_command(
    build: Annotated[Path, typer.Option()] = Path("data/build"),
    raw: Annotated[Path, typer.Option(help="Directory holding the MS MARCO files.")] = Path(
        "data/raw"
    ),
    depth: Annotated[int, typer.Option(help="How deep to look for negatives.")] = 100,
    per_query: Annotated[int, typer.Option(help="Negatives kept per query.")] = 8,
) -> None:
    """Mine hard negatives for the training queries.

    Judged against the full training qrels rather than the sampled splits, so a
    passage this query is known to answer is excluded even when it was not
    drawn into the corpus.
    """
    index = BM25Index.load(build / "bm25")
    queries = load_queries(build / "queries.train.tsv")
    qrels = load_qrels(raw / "qrels.train.tsv")
    mined = mine_negatives(index, queries, qrels, depth=depth, per_query=per_query)
    out = build / "negatives.tsv"
    with out.open("w", encoding="utf-8") as handle:
        for query_id in sorted(mined):
            for passage_id in mined[query_id]:
                handle.write(f"{query_id}\t{passage_id}\n")
    total = sum(len(v) for v in mined.values())
    typer.echo(f"mined {total} negatives for {len(mined)} queries into {out}")
