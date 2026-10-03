"""What was judged relevant, and what a system returned.

Both are plain dictionaries keyed by the identifiers MS MARCO uses, which are
integers. Wrapping them in classes would buy nothing: the only invariant worth
enforcing is that a query being scored has at least one relevant passage, and
that is checked where the scoring happens rather than at construction, so the
count of unscorable queries can be reported instead of raised.
"""

from __future__ import annotations

# Query identifier to passage identifier to relevance grade. MS MARCO passage
# judgements are binary, so the grade is 1, but the type carries graded
# relevance because NDCG is defined over it and a graded evaluation set would
# otherwise need a second type.
Qrels = dict[int, dict[int, int]]

# Query identifier to passage identifiers, best first.
Run = dict[int, list[int]]


def relevant_for(qrels: Qrels, query_id: int) -> set[int]:
    """Passages judged relevant at a grade above zero.

    A grade of zero appears in some judgement files to mark a passage that was
    looked at and rejected, which is not the same as one that was never judged,
    and counting it as relevant would inflate every metric here.
    """
    judged = qrels.get(query_id, {})
    return {doc_id for doc_id, grade in judged.items() if grade > 0}


def scorable(qrels: Qrels, run: Run) -> list[int]:
    """Queries that can be scored at all, in a fixed order.

    A query with no relevant passage has an undefined recall and a zero ideal
    gain, so averaging it in as zero would quietly punish a system for a gap in
    the judgements. Those queries are excluded and counted instead.
    """
    return sorted(query_id for query_id in run if relevant_for(qrels, query_id))
