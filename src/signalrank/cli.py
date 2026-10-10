"""Command line entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from signalrank.corpus import CorpusSpec, build_corpus
from signalrank.corpus.marco import load_qrels, load_queries, stream_collection

# bm25s, faiss and torch are imported inside the commands that need them, never
# here. Each brings its own OpenMP runtime, and a process holding two of them
# either aborts or runs an order of magnitude slower while the runtimes fight
# over threads. Importing them at module scope would put all three in every
# command, including the ones that touch none of them.

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


@app.command("build-bm25")
def build_bm25_command(
    build: Annotated[Path, typer.Option(help="Directory holding the sampled corpus.")] = Path(
        "data/build"
    ),
) -> None:
    """Index the sampled corpus for the baseline and for mining."""
    from signalrank.mining.bm25 import BM25Index

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
    from signalrank.mining.bm25 import BM25Index, run_for

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
    from signalrank.mining.bm25 import BM25Index, mine_negatives

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


def _load_training_inputs(
    build: Path,
) -> tuple[list[tuple[int, int]], dict[int, list[int]], dict[int, str], dict[int, str]]:
    """Read the splits, and only the passages training will actually touch.

    The corpus is 1.2 million passages and training sees the positives plus
    their mined negatives, which is a fraction of it. Loading the rest would
    cost a few hundred megabytes while a model is on the device.
    """
    pairs: list[tuple[int, int]] = []
    for line in (build / "pairs.train.tsv").read_text(encoding="utf-8").splitlines():
        query_id, passage_id = line.split("\t")
        pairs.append((int(query_id), int(passage_id)))

    negatives: dict[int, list[int]] = {}
    for line in (build / "negatives.tsv").read_text(encoding="utf-8").splitlines():
        query_id, passage_id = line.split("\t")
        negatives.setdefault(int(query_id), []).append(int(passage_id))

    queries = load_queries(build / "queries.train.tsv")
    wanted = {passage_id for _, passage_id in pairs}
    wanted |= {passage_id for ids in negatives.values() for passage_id in ids}
    passages = {
        passage_id: text
        for passage_id, text in stream_collection(build / "passages.tsv")
        if passage_id in wanted
    }
    return pairs, negatives, queries, passages


@app.command("train")
def train_command(
    build: Annotated[Path, typer.Option()] = Path("data/build"),
    out: Annotated[Path, typer.Option()] = Path("runs/dual-encoder"),
    epochs: Annotated[int, typer.Option()] = 3,
    batch_size: Annotated[int, typer.Option()] = 32,
    negatives_per_query: Annotated[int, typer.Option()] = 2,
    learning_rate: Annotated[float, typer.Option()] = 2e-5,
    max_steps: Annotated[int, typer.Option(help="Stop early, for a dry run.")] = 0,
    seed: Annotated[int, typer.Option()] = 20261103,
) -> None:
    """Fine-tune the dual encoder on mined and in-batch negatives."""
    from signalrank.models.training import TrainConfig, train_dual_encoder

    pairs, negatives, queries, passages = _load_training_inputs(build)
    typer.echo(
        f"{len(pairs)} pairs, {len(queries)} queries, {len(passages)} passages, "
        f"{sum(len(v) for v in negatives.values())} negatives"
    )
    config = TrainConfig(
        epochs=epochs,
        batch_size=batch_size,
        negatives_per_query=negatives_per_query,
        learning_rate=learning_rate,
        max_steps=max_steps or None,
        seed=seed,
    )
    report = train_dual_encoder(
        config=config,
        pairs=pairs,
        negatives=negatives,
        queries=queries,
        passages=passages,
        out=out,
    )
    typer.echo(json.dumps(report.as_dict(), indent=2))


@app.command("encode")
def encode_command(
    model_path: Annotated[Path, typer.Option("--model", help="Checkpoint, or a base name.")],
    build: Annotated[Path, typer.Option()] = Path("data/build"),
    out: Annotated[Path, typer.Option()] = Path("runs/vectors"),
    batch_size: Annotated[int, typer.Option()] = 256,
    untrained: Annotated[
        bool, typer.Option(help="Random weights, for the negative control.")
    ] = False,
) -> None:
    """Encode the corpus and both query splits into arrays on disk.

    Separate from evaluation on purpose: this loads torch, evaluation loads
    faiss, and a process holding both aborts.
    """
    import torch
    from transformers import AutoConfig, AutoModel

    from signalrank.models.dual_encoder import DualEncoder
    from signalrank.models.encode import encode_passages, encode_queries

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = DualEncoder(str(model_path))
    if untrained:
        # Same shape, no learning. A metric this scores well is not measuring
        # the model, which is the whole reason the control exists.
        torch.manual_seed(0)
        config = AutoConfig.from_pretrained(str(model_path))
        model.encoder = AutoModel.from_config(config)  # type: ignore[no-untyped-call]
    model = model.to(device)
    typer.echo(f"encoding on {device}")

    passage_ids: list[int] = []
    texts: list[str] = []
    for passage_id, text in stream_collection(build / "passages.tsv"):
        passage_ids.append(passage_id)
        texts.append(text)
    encode_passages(model, passage_ids, texts, out=out, device=device, batch_size=batch_size)
    del texts

    for split in ("heldout", "multi"):
        queries = load_queries(build / f"queries.{split}.tsv")
        ids = sorted(queries)
        vectors = encode_queries(model, [queries[q] for q in ids], device=device)
        np.save(out / f"queries.{split}.npy", vectors)
        np.save(out / f"query_ids.{split}.npy", np.asarray(ids, dtype=np.int64))
    typer.echo(f"wrote vectors to {out}")


@app.command("evaluate")
def evaluate_command(
    vectors: Annotated[Path, typer.Option()] = Path("runs/vectors"),
    build: Annotated[Path, typer.Option()] = Path("data/build"),
    evidence: Annotated[Path, typer.Option()] = Path("evidence"),
    label: Annotated[str, typer.Option(help="Name this run goes under.")] = "dual-encoder",
    depth: Annotated[int, typer.Option(help="Ranking depth to score.")] = 100,
    nlist: Annotated[int, typer.Option()] = 4096,
    pq_bytes: Annotated[int, typer.Option(help="Bytes per compressed vector.")] = 96,
    nprobe: Annotated[int, typer.Option()] = 32,
    qps: Annotated[float, typer.Option(help="Arrival rate for the latency run.")] = 100.0,
    latency_seconds: Annotated[float, typer.Option()] = 30.0,
) -> None:
    """Score the same model under exact and quantized search, and time the latter.

    Both indexes answer the same call, so the difference between the two sets
    of numbers is the quantization cost and nothing else. Loads faiss and never
    torch, which is why encoding is a separate command.
    """
    from signalrank.core import evaluate as score
    from signalrank.core.controls import randomised_qrels
    from signalrank.index.dense import FlatIndex, IvfPqIndex, search_to_run
    from signalrank.index.latency import measure

    passages = np.load(vectors / "passages.npy", mmap_mode="r")
    passage_ids = np.load(vectors / "passage_ids.npy")
    typer.echo(f"{passages.shape[0]} passages at {passages.shape[1]} dimensions")

    results: dict[str, object] = {"label": label, "corpus": int(passages.shape[0])}
    resident = np.ascontiguousarray(passages)

    flat = FlatIndex(resident, passage_ids)
    exact_runs = {}
    for split in ("heldout", "multi"):
        queries = np.load(vectors / f"queries.{split}.npy")
        ids = np.load(vectors / f"query_ids.{split}.npy").tolist()
        qrels = load_qrels(build / f"qrels.{split}.tsv")
        run = search_to_run(flat, ids, queries, depth)
        exact_runs[split] = (run, qrels, queries, ids)
        results[f"exact_{split}"] = score(run, qrels).as_dict()
        typer.echo(f"exact {split}: {results[f'exact_{split}']}")

    # The negative control, on the same code path as every number above. A
    # harness that still finds relevance against deranged judgements is finding
    # something that is not there.
    run, qrels, _, _ = exact_runs["heldout"]
    results["control_shuffled_labels"] = score(run, randomised_qrels(qrels, seed=7)).as_dict()
    typer.echo(f"shuffled labels: {results['control_shuffled_labels']}")

    quantized = IvfPqIndex(resident, passage_ids, nlist=nlist, m=pq_bytes, nbits=8)
    quantized.nprobe = nprobe
    results["index"] = {
        "nlist": nlist,
        "pq_bytes": pq_bytes,
        "nprobe": nprobe,
        "compression": round(passages.shape[1] * 4 / pq_bytes, 1),
    }
    for rerank in (1, 4, 10):
        for split in ("heldout", "multi"):
            _, qrels, queries, ids = exact_runs[split]
            run = search_to_run(quantized, ids, queries, depth, rerank_depth=rerank)
            key = f"quantized_{split}_rerank{rerank}"
            results[key] = score(run, qrels).as_dict()
            typer.echo(f"{key}: {results[key]}")

    _, _, queries, _ = exact_runs["heldout"]
    for rerank in (1, 10):

        def searcher(batch: np.ndarray, rerank_depth: int = rerank) -> object:
            return quantized.search(batch, depth, rerank_depth=rerank_depth)

        report = measure(searcher, queries, target_qps=qps, seconds=latency_seconds)
        results[f"latency_rerank{rerank}"] = report.as_dict()
        typer.echo(f"latency rerank{rerank}: {report.as_dict()}")

    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / f"{label}.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    typer.echo(f"wrote {evidence / (label + '.json')}")


if __name__ == "__main__":
    app()
