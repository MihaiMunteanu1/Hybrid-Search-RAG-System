"""Hybrid search: dense (embeddings) and sparse (BM25), fused with Reciprocal Rank Fusion.

The two searches fail in different places. Dense search finds a passage by meaning
even when it shares no words with the question; BM25 finds exact terms such as
"Partea 3" or an article number, which an embedding barely distinguishes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rag.embedding import embed_queries
from rag.ingestion.indexing import Index, bm25_tokens

RRF_K = 60          # damping constant of RRF
DENSE_K = 10        # results taken from each search before fusion
SPARSE_K = 10
CANDIDATES = 10     # fused results returned

MODES = ("hybrid", "dense", "sparse")


@dataclass
class Hit:
    """One retrieved chunk, with its rank and score in every search that found it."""
    chunk_id: str
    score: float    # cosine for dense-only, BM25 for sparse-only, RRF for hybrid
    chunk: dict
    ranks: dict[str, int] = field(default_factory=dict)     # 1-based, per search
    scores: dict[str, float] = field(default_factory=dict)  # raw score, per search

    @property
    def source(self) -> str:
        return self.chunk["source"]

    @property
    def text(self) -> str:
        return self.chunk["text"]


def search_dense(index: Index, question: str, limit: int = DENSE_K,
                 sources: list[str] | None = None) -> list[tuple[str, float]]:
    """The chunks whose vectors are closest to the question's, as (chunk_id, cosine).

    `sources` restricts the search to those documents. Qdrant applies the filter during
    the search, so the result still holds `limit` chunks from the allowed documents.
    """
    from qdrant_client import models

    vector = embed_queries([question])[0]
    query_filter = None if sources is None else models.Filter(must=[models.FieldCondition(
        key="source", match=models.MatchAny(any=sources))])
    hits = index.client.query_points(collection_name=index.collection, query=vector.tolist(),
                                     limit=limit, query_filter=query_filter).points
    return [(hit.payload["chunk_id"], float(hit.score)) for hit in hits]


def search_sparse(index: Index, question: str, limit: int = SPARSE_K,
                  sources: list[str] | None = None) -> list[tuple[str, float]]:
    """BM25 over the same chunks, as (chunk_id, score).

    The question is tokenized exactly like the documents (accents folded, stemmed),
    otherwise its words would not match the stems in the index. Chunks scoring zero
    share no term with the question and are left out instead of receiving a rank.
    """
    if not index.chunks:
        return []
    if index.bm25 is None:
        raise ValueError("this index has no bm25.pkl; rebuild it with python -m rag.ingestion")
    scores = index.bm25.get_scores(bm25_tokens(question))
    allowed = None if sources is None else set(sources)
    order = sorted((i for i in range(len(scores))
                    if allowed is None or index.chunks[i]["source"] in allowed),
                   key=lambda i: scores[i], reverse=True)
    return [(index.chunks[i]["chunk_id"], float(scores[i]))
            for i in order[:limit] if scores[i] > 0]


def reciprocal_rank_fusion(rankings: dict[str, list[str]], k: int = RRF_K
                           ) -> list[tuple[str, float, dict[str, int]]]:
    """Fuse ranked lists by position: score = sum of 1 / (k + rank).

    Positions are used instead of scores because cosine and BM25 scores are on
    unrelated scales. `k` flattens the top of each list, so a chunk found by both
    searches outranks one that only a single search put first.

    Returns (chunk_id, fused score, rank in each list), best first.
    """
    fused: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for name, ids in rankings.items():
        for position, chunk_id in enumerate(ids, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + position)
            ranks.setdefault(chunk_id, {})[name] = position

    # Exact ties are common (dense #1 + sparse #4 equals dense #4 + sparse #1). They are
    # broken by the dense rank, the stronger search here, and then by chunk_id, so the
    # order is the same on every run.
    def dense_rank(cid: str) -> int:
        return min((r for name, r in ranks[cid].items() if name.startswith("dense")),
                   default=10**6)

    ordered = sorted(fused, key=lambda cid: (-fused[cid], dense_rank(cid), cid))
    return [(cid, fused[cid], ranks[cid]) for cid in ordered]


def search(index: Index, question: str, *, limit: int = CANDIDATES, mode: str = "hybrid",
           dense_k: int = DENSE_K, sparse_k: int = SPARSE_K, k: int = RRF_K,
           per_language: bool = False, sources: list[str] | None = None) -> list[Hit]:
    """Retrieve fused candidates for a question.

    Args:
        mode: "hybrid", or "dense" / "sparse" alone (used by the evaluation to compare
            them through the same code path).
        per_language: run each search once per corpus language and fuse all the lists.
            Otherwise a Romanian question favours Romanian chunks, and an English chunk
            that answers it can fall below the cut.
        sources: restrict the search to these documents (a folder).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    allowed = set(index.language) if sources is None else set(sources) & set(index.language)
    if not allowed:  # an empty folder, or sources the index does not hold
        return []
    if per_language:
        groups: dict[str, list[str]] = {}
        for source in sorted(allowed):
            groups.setdefault(f":{index.language[source]}", []).append(source)
        # One language and no folder: a filter listing every document excludes nothing
        # but slows Qdrant down considerably on a large corpus, so it is dropped.
        if sources is None and len(groups) == 1:
            groups = dict.fromkeys(groups)
    else:
        groups = {"": None if sources is None else sorted(allowed)}

    rankings, raw = {}, {}
    for suffix, group in groups.items():
        if mode in ("hybrid", "dense"):
            dense = search_dense(index, question,
                                 max(dense_k, limit if mode == "dense" else 0), group)
            rankings["dense" + suffix] = [cid for cid, _ in dense]
            raw.setdefault("dense", {}).update(dense)
        if mode in ("hybrid", "sparse"):
            sparse = search_sparse(index, question,
                                   max(sparse_k, limit if mode == "sparse" else 0), group)
            rankings["sparse" + suffix] = [cid for cid, _ in sparse]
            raw.setdefault("sparse", {}).update(sparse)

    hits = []
    for chunk_id, fused, ranks in reciprocal_rank_fusion(rankings, k=k)[:limit]:
        chunk = index.by_id(chunk_id)
        if chunk is None:  # a point Qdrant holds but the chunk list does not
            continue
        # "dense:ro" -> "dense": a chunk has one language, so the names stay unique.
        ranks = {name.split(":")[0]: rank for name, rank in ranks.items()}
        scores = {name: raw[name][chunk_id] for name in ranks}
        # A single search shows its own score; per language, scores come from different
        # lists, so the fused one is shown instead.
        score = fused if mode == "hybrid" or per_language else scores[mode]
        hits.append(Hit(chunk_id=chunk_id, score=score, chunk=chunk, ranks=ranks,
                        scores=scores))
    return hits
