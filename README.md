# signalrank

A dual encoder trained on mined hard negatives, a cross-encoder reranker
distilled into it, and both re-measured under the quantized index that actually
serves them rather than under the exact search they were tuned with.

The claim, the thresholds and the negative controls are fixed in
[PLAN.md](PLAN.md). The measurements are not collected yet, and this file is
written from them once they are, so nothing here states a number the repository
has not produced.
