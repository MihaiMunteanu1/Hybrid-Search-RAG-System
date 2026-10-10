"""The retrieval stage behind one call: question in, best chunks out.

    candidates   dense + BM25, fused with RRF       cheap, over the whole corpus
    precision    cross-encoder over the candidates  expensive, over 20 chunks

The rest of the code (generation, evaluation, API) calls only `retrieve()`.
"""
from __future__ import annotations

from rag.ingestion.indexing import Index
from rag.retrieval.rerank import rerank
from rag.retrieval.search import Hit, search

# The final configuration, chosen on the development set (see docs/DESIGN.md).
TOP_N = 5            # chunks handed to the language model
LIST_K = 5           # results kept from each search list before fusion
CANDIDATES = 20      # fused candidates the reranker reads
PER_LANGUAGE = True  # one dense and one BM25 list per corpus language


def retrieve(index: Index, question: str, *, top_n: int = TOP_N, use_reranker: bool = True,
             candidates: int = CANDIDATES, list_k: int = LIST_K, mode: str = "hybrid",
             per_language: bool = PER_LANGUAGE, sources: list[str] | None = None) -> list[Hit]:
    """The best `top_n` chunks for a question.

    With `use_reranker=False` the fused ranking is cut to `top_n`, so the evaluation
    can measure what the reranker adds through this same function.
    `sources` restricts the search to those documents (a folder).
    """
    hits = search(index, question, limit=max(candidates, top_n), mode=mode, dense_k=list_k,
                  sparse_k=list_k, per_language=per_language, sources=sources)
    if not use_reranker:
        return hits[:top_n]
    return rerank(question, hits, top_n=top_n)
