# Known misses

Written against the repository rather than for it.

## Training ran in full precision

The L4 has tensor cores and the training loop does not use them: there is no
autocast and no gradient scaler, so a model small enough to train in half
precision was trained in float32 throughout. The published run would have
finished in roughly a third of the time, at a cost in numerical precision that
a contrastive objective over normalised vectors would almost certainly not have
noticed.

This was not a considered trade. It was an omission, found while watching the
run take longer than it should have, and it is left as it was rather than
changed under a run already in flight.

## Progress was invisible for the first run

The training loop collected its losses in memory and wrote them only when it
finished, so a run with an hour to go looked exactly like a run that had hung.
That is fixed now, but the first full run was watched by checking that the GPU
was still drawing power.
