# Plan

Written and committed before the models were trained and before any number was
measured, so the thresholds below cannot be chosen to fit the results. The
history shows when each was fixed.

## The claim this repository has to earn

> Fine-tuning a dual encoder on BM25-mined hard negatives, and distilling a
> cross-encoder reranker into it, retrieves better than BM25 over 1.2 million
> passages, and the improvement survives the quantization and approximation
> needed to answer 100 queries per second.

The first half of that is ordinary. The second half is the part worth testing,
because it is the half that usually fails quietly: quality is measured with
exact search over float32 vectors, the system ships with an approximate index
over compressed ones, and nobody re-measures. A gain of eight points that
becomes two after product quantization is not a gain of eight points.

So every quality number here is reported twice, once under exact search and
once under the index that actually serves it.

## Thresholds, fixed before the run

| what | threshold | why this number |
| --- | ---: | --- |
| BM25 Recall@20, the baseline | reported, not targeted | the comparison is only honest if the baseline is a real one, so it is measured rather than assumed |
| fine-tuned dual encoder Recall@20, exact search | at least 0.85 | the target the training is aimed at, on 3,000 held-out queries |
| Recall@20 lost to IVF-PQ against exact search | at most 0.03 absolute | the whole second half of the claim; above this the served system is not the measured one |
| NDCG@10 gain from reranking and distillation | at least 0.08 | measured on the multi-intent subset defined below |
| IVF-PQ p95 latency at 100 queries per second | at most 190 ms | over 1,200,000 passages, measured locally |
| training pairs | 160,000 | query and passage pairs with mined negatives |
| shuffled-label Recall@20, the negative control | at most 0.01 | chance level; above this the evaluation is not measuring relevance |
| untrained encoder Recall@20, the negative control | at most 0.05 | an encoder that learned nothing must score near nothing |

A threshold that is missed is published as missed, with the cause traced, and
pinned by the validator at its measured value so it cannot drift quietly.

## Definitions fixed in advance

These are the ones that could otherwise be adjusted after the fact to flatter a
result, so they are settled here.

**The corpus** is a 1,200,000 passage sample of the MS MARCO passage collection,
drawn with a fixed seed, always including every passage judged relevant for a
query in any evaluation set. Sampling without that guarantee would make recall a
measure of the sample rather than of the model.

**Held-out queries** are 3,000 drawn from the development set, disjoint from
every training pair, fixed by seed before training.

**The multi-intent subset** is operationally defined as evaluation queries with
more than one passage judged relevant. That is a proxy, not a label: MS MARCO
does not mark intent. The real size of that subset is reported, and if it falls
short of the 2,000 queries the NDCG threshold assumes, the shortfall is
published rather than the definition widened.

**The baseline is BM25**, not an off-the-shelf neural retriever. Most small
sentence encoders have already been trained on MS MARCO, so using one as the
baseline measures contamination rather than improvement. An untrained encoder is
also scored, to separate what fine-tuning contributes from what dense retrieval
contributes.

## The negative controls

A retrieval number that goes up is not evidence that anything works, because an
evaluation harness with a bug in it produces numbers that go up too. Three
controls run on the same code path as the real evaluation:

1. **Shuffled labels.** The same trained model, scored against a random
   permutation of the relevance judgements, must collapse to chance. If it does
   not, the harness is finding relevance that is not there.
2. **Untrained encoder.** A randomly initialised encoder of the same shape, with
   no training at all, must score near zero. If it does not, the metric is being
   satisfied by something other than the model.
3. **A baseline that is not straw.** BM25 must score well above chance. A
   fine-tuned model beating a broken baseline shows nothing, so the baseline's
   own number is a threshold the run has to clear.

## What is not claimed

Nothing here is a statement about retrieval quality in general, about other
corpora, or about how these numbers compare to published work on the full 8.8
million passage collection. The corpus is a 1.2 million passage sample and the
models are small enough to train in a few hours on one GPU.

MS MARCO is released by Microsoft for non-commercial research only, with no
licence or intellectual property rights granted in the data. Nothing derived
from it here is redistributable, the corpus is not committed to this repository,
and no trained weights are published.

## Order of work

1. Metrics and the evaluation rule, with no dependency on a model or an index.
2. Corpus sampling and the held-out splits, fixed by seed.
3. BM25, for the baseline and for mining hard negatives.
4. The dual encoder, trained with in-batch and mined negatives.
5. The cross-encoder reranker, and distilling its judgements into the retriever.
6. FAISS IVF-PQ, and the latency measurement at 100 queries per second.
7. The run, then the README written from what it produced.
