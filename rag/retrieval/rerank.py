"""Reranking with a cross-encoder.

The embedding model encodes the question and each chunk separately, which is fast but
never lets the two texts see each other. A cross-encoder reads them together as one
input and scores the pair, which is far more precise and far too slow for the whole
corpus, so it only reorders the candidates that hybrid search already found.
"""
from __future__ import annotations

from dataclasses import replace

from rag.retrieval.search import Hit

# Multilingual MiniLM trained on mMARCO (MS MARCO translated into 14 languages).
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# Tokens per (question, chunk) pair. Must cover a whole chunk (up to 400 tokens);
# 512 is the model's limit.
RERANK_MAX_LENGTH = 512

_RERANKER = None


def get_reranker(name: str = RERANK_MODEL, max_length: int = RERANK_MAX_LENGTH):
    """The cross-encoder, loaded once, on CPU."""
    global _RERANKER
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder
        _RERANKER = CrossEncoder(name, device="cpu", max_length=max_length)
    return _RERANKER


def rerank(question: str, hits: list[Hit], top_n: int) -> list[Hit]:
    """Score every candidate against the question and keep the best `top_n`.

    Returns copies, so the input keeps the fused order. Scores are raw logits:
    higher is better, but they are not probabilities.
    """
    if not hits:
        return []
    scores = get_reranker().predict([(question, hit.text) for hit in hits],
                                    show_progress_bar=False)
    scored = [replace(hit,
                      ranks={**hit.ranks, "fused": position},
                      scores={**hit.scores, "rerank": float(score)},
                      score=float(score))
              for position, (hit, score) in enumerate(zip(hits, scores), start=1)]
    # Equal scores keep the order chosen by fusion.
    scored.sort(key=lambda hit: (-hit.score, hit.ranks["fused"]))
    return scored[:top_n]
