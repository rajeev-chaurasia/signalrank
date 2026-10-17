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
