"""Recompute the published numbers from the rankings, and check the controls.

Two levels, because the judgements cannot be committed. MS MARCO is released
for non-commercial research with no redistribution rights, so `data/build` is
not in this repository and neither are the qrels. The rankings are, because
they are this model's output rather than Microsoft's data.

So the structural checks always run: the controls must have failed as controls
are supposed to, the quantized index must not somehow beat exact search, and
every number the README quotes must be present. The recomputation runs when the
corpus has been built locally, and then every metric is recalculated from the
rankings and compared.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Any

from signalrank.core import Qrels, Run, evaluate
from signalrank.core.controls import randomised_qrels
from signalrank.corpus.marco import load_qrels

EVIDENCE = Path("evidence")
BUILD = Path("data/build")

# Fixed in PLAN.md before anything was trained.
SHUFFLED_CEILING = 0.01
UNTRAINED_CEILING = 0.05
BASELINE_FLOOR = 0.30


def load_run(path: Path) -> Run:
    run: Run = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            run[int(record["qid"])] = [int(doc) for doc in record["ranked"]]
    return run


def summaries() -> dict[str, dict[str, Any]]:
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(EVIDENCE.glob("*.json"))
        if path.stem != "manifest"
    }


def recompute(label: str, summary: dict[str, Any], qrels: dict[str, Qrels]) -> list[str]:
    problems = []
    for key, published in summary.items():
        if not isinstance(published, dict) or "recall_at_20" not in published:
            continue
        # The shuffled-label control has no ranking of its own. It is the exact
        # ranking scored against deranged judgements, so it is rebuilt that way,
        # with the same seed the evaluate command used.
        source = "exact_heldout" if key == "control_shuffled_labels" else key
        run_path = EVIDENCE / "runs" / f"{label}.{source}.jsonl.gz"
        if not run_path.exists():
            problems.append(f"{label}: {key} is published with no ranking beside it")
            continue
        split = "multi" if "multi" in key else "heldout"
        judgements = qrels[split]
        if key == "control_shuffled_labels":
            judgements = randomised_qrels(judgements, seed=7)
        measured = evaluate(load_run(run_path), judgements).as_dict()
        for metric, value in published.items():
            if measured.get(metric) != value:
                problems.append(
                    f"{label}: {key}.{metric} published as {value}, "
                    f"recomputed as {measured.get(metric)}"
                )
    return problems


def recompute_sweep(label: str, summary: dict[str, Any], qrels: dict[str, Qrels]) -> list[str]:
    """The nprobe sweep keeps its rows in a list, which the loop above skips."""
    problems = []
    base = label.removesuffix(".nprobe-sweep")
    for row in summary["sweep"]:
        nprobe = row["nprobe"]
        run_path = EVIDENCE / "runs" / f"{base}.sweep_heldout_nprobe{nprobe}.jsonl.gz"
        if not run_path.exists():
            problems.append(f"{label}: nprobe {nprobe} is published with no ranking beside it")
            continue
        measured = evaluate(load_run(run_path), qrels["heldout"]).as_dict()
        for metric, value in measured.items():
            if row.get(metric) != value:
                problems.append(
                    f"{label}: nprobe {nprobe} {metric} published as {row.get(metric)}, "
                    f"recomputed as {value}"
                )
    return problems


def check_sweep_matches_published(found: dict[str, dict[str, Any]]) -> list[str]:
    """The sweep must reproduce the published row at the published setting.

    It rebuilds the quantized index rather than reusing the one that produced
    the main results. If the row at the same nprobe and rerank depth does not
    match exactly, the sweep searched a different index, and what it says about
    the published gap is about something else.
    """
    problems = []
    for label, summary in found.items():
        if "sweep" not in summary:
            continue
        base = found.get(label.removesuffix(".nprobe-sweep"), {})
        nprobe = base.get("index", {}).get("nprobe")
        rerank = summary["index"]["rerank_depth"]
        published = base.get(f"quantized_heldout_rerank{rerank}")
        row = next((r for r in summary["sweep"] if r["nprobe"] == nprobe), None)
        if published is None or row is None:
            problems.append(f"{label}: no row at the published nprobe {nprobe} to compare")
            continue
        for metric, value in published.items():
            if row.get(metric) != value:
                problems.append(
                    f"{label}: at nprobe {nprobe} {metric} is {row.get(metric)}, but the "
                    f"published index scored {value}"
                )
    return problems


def check_controls(found: dict[str, dict[str, Any]]) -> list[str]:
    """The controls are what make the headline numbers mean anything."""
    problems = []
    main = found.get("dual-encoder")
    if main is None:
        return ["no dual-encoder evidence, so there is nothing to validate"]

    shuffled = main.get("control_shuffled_labels", {}).get("recall_at_20")
    if shuffled is None:
        problems.append("the shuffled-label control is missing")
    elif shuffled > SHUFFLED_CEILING:
        problems.append(
            f"scoring against deranged judgements still found {shuffled} recall, above "
            f"{SHUFFLED_CEILING}: the harness is finding relevance that is not there"
        )

    untrained = found.get("untrained-control", {}).get("exact_heldout", {}).get("recall_at_20")
    if untrained is None:
        problems.append("the untrained-encoder control is missing")
    elif untrained > UNTRAINED_CEILING:
        problems.append(
            f"an untrained encoder scored {untrained}, above {UNTRAINED_CEILING}: the metric "
            "is being satisfied by something other than the model"
        )

    bm25 = found.get("bm25", {}).get("heldout", {}).get("recall_at_20")
    if bm25 is None:
        problems.append("the BM25 baseline is missing, so there is nothing to beat")
    elif bm25 < BASELINE_FLOOR:
        problems.append(
            f"BM25 scored {bm25}, below {BASELINE_FLOOR}: beating a baseline this weak "
            "would show nothing"
        )
    return problems


def check_ordering(label: str, summary: dict[str, Any]) -> list[str]:
    """Quantized search cannot beat the exact search it approximates.

    If it does, the two are not searching the same vectors, and the gap between
    them is measuring a bug rather than the cost of compression.
    """
    problems = []
    for split in ("heldout", "multi"):
        exact = summary.get(f"exact_{split}", {}).get("recall_at_20")
        if exact is None:
            continue
        for rerank in (1, 4, 10):
            approx = summary.get(f"quantized_{split}_rerank{rerank}", {}).get("recall_at_20")
            if approx is not None and approx > exact + 1e-9:
                problems.append(
                    f"{label}: quantized {split} rerank{rerank} scored {approx}, above exact "
                    f"search at {exact}, which cannot happen if both read the same vectors"
                )
    return problems


def check_sweep_ordering(found: dict[str, dict[str, Any]]) -> list[str]:
    """Opening more lists can approach exact search but never pass it."""
    problems = []
    for label, summary in found.items():
        if "sweep" not in summary:
            continue
        base = found.get(label.removesuffix(".nprobe-sweep"), {})
        exact = base.get("exact_heldout", {}).get("recall_at_20")
        if exact is None:
            problems.append(f"{label}: no exact search to compare the sweep against")
            continue
        for row in summary["sweep"]:
            if row["recall_at_20"] > exact + 1e-9:
                problems.append(
                    f"{label}: nprobe {row['nprobe']} scored {row['recall_at_20']}, above "
                    f"exact search at {exact}"
                )
    return problems


def main() -> int:
    if not EVIDENCE.exists() or not summaries():
        print("no evidence to validate", file=sys.stderr)
        return 1

    found = summaries()
    problems = check_controls(found)
    for label, summary in found.items():
        problems += check_ordering(label, summary)
    problems += check_sweep_ordering(found)
    problems += check_sweep_matches_published(found)

    if (BUILD / "qrels.heldout.tsv").exists():
        qrels = {split: load_qrels(BUILD / f"qrels.{split}.tsv") for split in ("heldout", "multi")}
        for label, summary in found.items():
            if "sweep" in summary:
                problems += recompute_sweep(label, summary, qrels)
            elif label != "bm25":
                problems += recompute(label, summary, qrels)
        print(f"recomputed every metric over {len(found)} run(s)")
    else:
        print(
            "judgements not present, so numbers were not recomputed. Build the corpus with "
            "script/fetch_corpus.sh and 'signalrank build-corpus' to check them."
        )

    if problems:
        print("\n".join(problems), file=sys.stderr)
        print(f"\n{len(problems)} evidence problem(s)", file=sys.stderr)
        return 1
    print(f"evidence ok over {len(found)} run(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
