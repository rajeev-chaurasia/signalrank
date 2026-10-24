# Non-goals

What this repository deliberately does not attempt, so that the claim it does
make can be read narrowly and checked.

## End-to-end query latency

The latency figures time the index and nothing else. Queries are encoded before
the clock starts, and the harness issues the resulting vectors on a fixed
schedule, so the published p95 is the cost of finding neighbours for a vector
already in hand. Serving a typed query would add a forward pass through the
query encoder on top, and on a CPU that pass would dominate the total. The
numbers say whether the quantized index keeps up at 100 queries a second, not
whether the system answers a person that fast.

## Comparison with published MS MARCO results

The corpus is 1.2 million passages sampled from 8.8 million, and the sample is
built to contain every passage judged relevant for an evaluated query. That
removes about six in seven of the distractors a query would face in the full
collection, so every recall number here is higher than the same model would
score against all of it. The numbers compare systems against each other on the
same sample and say nothing about where they would land on a leaderboard.

## A cross-encoder or distillation

The plan's NDCG threshold names reranking and distillation together. The
reranking built here is the exact rescoring of quantized candidates against
full-precision vectors, which recovers compression loss and nothing more. No
cross-encoder scores query and passage jointly, and no larger model's scores
are distilled into the dual encoder. Both are the usual next step and neither is
claimed.

## Variance

One training run, one seed, one sample. No confidence interval is published
because none was measured, and a difference of a point or two between two
numbers here should not be read as settled.

## Redistributable weights

MS MARCO is licensed for non-commercial research, with no rights granted to
redistribute it or anything presented as derived from it. The trained encoder
is not published. The rankings are, because they are this model's output rather
than Microsoft's data, and they are what the published numbers are recomputed
from.
