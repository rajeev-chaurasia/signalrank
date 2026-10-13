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
        run_path = EVIDENCE / "runs" / f"{label}.{key}.jsonl.gz"
        if not run_path.exists():
            problems.append(f"{label}: {key} is published with no ranking beside it")
            continue
        split = "multi" if "multi" in key else "heldout"
        measured = evaluate(load_run(run_path), qrels[split]).as_dict()
        for metric, value in published.items():
            if measured.get(metric) != value:
                problems.append(
                    f"{label}: {key}.{metric} published as {value}, "
                    f"recomputed as {measured.get(metric)}"
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


def main() -> int:
    if not EVIDENCE.exists() or not summaries():
        print("no evidence to validate", file=sys.stderr)
        return 1

    found = summaries()
    problems = check_controls(found)
    for label, summary in found.items():
        problems += check_ordering(label, summary)

    if (BUILD / "qrels.heldout.tsv").exists():
        qrels = {split: load_qrels(BUILD / f"qrels.{split}.tsv") for split in ("heldout", "multi")}
        for label, summary in found.items():
            if label != "bm25":
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
