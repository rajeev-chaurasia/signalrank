# Known misses

Written against the repository rather than for it.

## The first two runs were lost entirely

One spot instance was preempted at 104 minutes and one on-demand instance hit
its six hour cap, and both were configured to delete themselves on termination,
which took the disk and everything on it. Nothing was left billing either time,
which was the point of the flag, but nothing was produced either. Artifacts are
now copied off the instance as each stage finishes.

## Training ran in full precision for those runs

The L4 has tensor cores and the training loop does not use them: there is no
autocast and no gradient scaler, so a model small enough to train in half
precision was trained in float32 throughout. The published run would have
finished in roughly a third of the time, at a cost in numerical precision that
a contrastive objective over normalised vectors would almost certainly not have
noticed.

This was not a considered trade. It was an omission, found while watching a
run take four times longer than the arithmetic allowed. Training now runs in
bfloat16 on CUDA and in float32 everywhere else, so a laptop smoke run and the
GPU run do not quietly differ in numerics for no reason.

## Progress was invisible for the first run

The training loop collected its losses in memory and wrote them only when it
finished, so a run with an hour to go looked exactly like a run that had hung.
That is fixed now, but the first full run was watched by checking that the GPU
was still drawing power.

## The quantized index serves a system worse than BM25

This is the central miss. The plan allowed at most 0.03 of Recall@20 to be lost
between exact search and IVF-PQ, and at the committed configuration 0.0928 was
lost, taking the served system from 7.8 points above BM25 to 1.5 below it.

The cause is traced rather than guessed. Reranking ten times as many candidates
recovers no more than reranking four times as many, so the loss is not in how
compressed codes order what they return. Opening more lists recovers it almost
entirely, so it is in which lists are opened: 32 of 4,096 was too few for these
vectors, and the right passages were filed in lists the search never read. The
sweep that shows this was run after the miss, and the setting it finds is not
retroactively counted as meeting the threshold.

What was wrong in advance was the choice of nprobe without a sweep. A list
count and a probe count picked for their latency, and never measured for
recall on the vectors they would hold, are exactly how a measured gain
disappears in deployment, which is the failure this repository was set up to
catch. It caught it here, in its own configuration.

## Recall@20 is 0.73, not 0.85

The training target was missed by twelve points under exact search. That is
not traced to a single cause, and the run does not contain the evidence to
trace it. One configuration was trained once: two epochs, 160,000 pairs, a
batch of 64 and two mined negatives a query, from a base model with no
retrieval pretraining. Every one of those is a lever that was not moved. The
loss was still moving between 0.28 and 0.52 over the last thousand steps, which
is consistent with more training helping and does not show it.

The 0.85 was set as a target rather than derived from anything, and in
hindsight it was set without asking what this base model and this budget had
reached elsewhere.

## The NDCG threshold measured something that was not built

The plan asked for an NDCG@10 gain of 0.08 from reranking and distillation on
the multi-intent subset. Neither a cross-encoder nor distillation was built.
The dual encoder alone gains 0.0454 over BM25 there under exact search, and
0.0151 under the quantized index. The threshold is recorded as missed because
the plan said what it would measure and it was not measured.

## Latency was measured on the instance, not locally

The plan says the latency threshold is measured locally. It was measured on the
GPU instance's eight CPU cores, as the stage after training in the same
pipeline, rather than after copying 3.7 GB of vectors back to a laptop whose
memory training had already exhausted once. The search runs on CPU either way,
but the machine is not the one the plan named.
