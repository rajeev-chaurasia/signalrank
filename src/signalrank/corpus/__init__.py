"""Reading MS MARCO and cutting it into the splits the claim is measured on."""

from signalrank.corpus.marco import load_qrels, load_queries, stream_collection
from signalrank.corpus.splits import CorpusSpec, build_corpus

__all__ = ["CorpusSpec", "build_corpus", "load_qrels", "load_queries", "stream_collection"]
