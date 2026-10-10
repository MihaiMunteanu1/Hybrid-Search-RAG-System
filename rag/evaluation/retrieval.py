"""Retrieval evaluation: does a relevant chunk come back for each question?

Metrics, over the questions that have an answer in the corpus:
    recall@k   share of questions with at least one relevant chunk in the top k
    MRR        mean of 1 / rank of the first relevant chunk (0 if not in the top 10)
For unanswerable questions the top score is reported instead, which is what a refusal
threshold would have to separate from the answerable ones.

Results are saved per question in `data/eval/results/<name>.json`, so every number in
the paper can be traced back to the questions behind it.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from rag.config import DATA_DIR
from rag.evaluation.gold import DOCUMENTS, QUESTIONS, attach_spans, is_relevant, read_jsonl
from rag.ingestion.indexing import load
from rag.retrieval.pipeline import CANDIDATES, LIST_K, retrieve

KS = (1, 3, 5, 10)
MAX_K = max(KS)
RESULTS = DATA_DIR / "eval" / "results"


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson confidence interval for a proportion.

    The sets are small (one question is 4-5 points), so every rate is reported with
    its interval. Wilson stays valid near 0% and 100%, where the normal approximation
    breaks down.
    """
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def first_relevant_rank(hits, question: dict) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if is_relevant(hit.chunk, question):
            return rank
    return None


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    out = {"n": n}
    for k in KS:
        found = sum(1 for r in rows if r["rank"] is not None and r["rank"] <= k)
        low, high = wilson(found, n)
        out[f"recall@{k}"] = {"found": found, "value": found / n if n else 0.0,
                              "ci95": [low, high]}
    out["mrr"] = sum(1 / r["rank"] for r in rows if r["rank"]) / n if n else 0.0
    return out


def evaluate(strategy: str, mode: str = "hybrid", per_language: bool = False,
             questions_path: Path = QUESTIONS, rerank: bool = False,
             list_k: int = LIST_K, candidates: int = CANDIDATES) -> dict:
    """Run every question through `retrieve()`, the function the app uses, and score it."""
    questions = read_jsonl(questions_path)
    problems = attach_spans(questions, read_jsonl(DOCUMENTS))
    if problems:
        # A quote that cannot be found would silently count as a retrieval miss.
        raise ValueError("evaluation set out of sync with the corpus:\n  " + "\n  ".join(problems))

    rows = []
    with load(DATA_DIR / "index" / strategy) as index:
        for q in questions:
            hits = retrieve(index, q["question"], top_n=MAX_K, use_reranker=rerank,
                            candidates=candidates, list_k=list_k, mode=mode,
                            per_language=per_language)
            rows.append({
                "id": q["id"], "type": q["type"], "verified": q["verified"],
                "rank": first_relevant_rank(hits, q) if q["gold"] else None,
                "top_score": hits[0].score if hits else None,
                "top": [{"source": h.source, "page": h.chunk.get("page"),
                         "heading_path": h.chunk["heading_path"], "score": round(h.score, 4),
                         "ranks": h.ranks}
                        for h in hits[:5]],
                # The relevant chunk's rank in each search before fusion.
                "ranks_by_search": next((h.ranks for h in hits if is_relevant(h.chunk, q)), {}),
            })

    answerable = [r for r, q in zip(rows, questions) if q["gold"]]
    unanswerable = [r for r, q in zip(rows, questions) if not q["gold"]]
    by_type = {}
    for r in answerable:
        by_type.setdefault(r["type"], []).append(r)

    return {
        "strategy": strategy,
        "mode": mode,
        "per_language": per_language,
        "rerank": rerank, "list_k": list_k, "candidates": candidates,
        "verified": sum(q["verified"] for q in questions),
        "total": len(questions),
        "overall": summarize(answerable),
        "by_type": {t: summarize(rs) for t, rs in sorted(by_type.items())},
        "unanswerable_top_scores": [r["top_score"] for r in unanswerable],
        "answerable_top_scores": sorted(r["top_score"] for r in answerable),
        "questions": rows,
    }


def print_report(result: dict) -> None:
    o = result["overall"]
    if result["verified"] < result["total"]:
        print(f"[provisional] only {result['verified']}/{result['total']} questions "
              f"verified by hand\n")
    print(f"strategy: {result['strategy']}   mode: {result['mode']}"
          f"{' per-language' if result['per_language'] else ''}"
          f"{' + rerank' if result.get('rerank') else ''}   "
          f"answerable questions: {o['n']}\n")
    print(f"{'':16}{'n':>3}  " + "  ".join(f"{'R@' + str(k):>7}" for k in KS) + "     MRR")
    for name, s in [("ALL", o)] + list(result["by_type"].items()):
        cells = "  ".join(f"{s[f'recall@{k}']['value']:>7.0%}" for k in KS)
        print(f"{name:16}{s['n']:>3}  {cells}   {s['mrr']:.3f}")
    low, high = o["recall@5"]["ci95"]
    print(f"\nrecall@5 = {o['recall@5']['found']}/{o['n']}, 95% CI [{low:.0%}, {high:.0%}]")

    misses = [r for r in result["questions"] if r["type"] != "unanswerable"
              and (r["rank"] is None or r["rank"] > 5)]
    if misses:
        print("\nnot in the top 5:")
        for r in misses:
            where = f"rank {r['rank']}" if r["rank"] else f"not in top {MAX_K}"
            got = ", ".join(f"{t['source'].split('/')[-1][:28]} p.{t['page']}" for t in r["top"][:3])
            print(f"  {r['id']} {r['type']:13} {where:16} top 3: {got}")

    un = result["unanswerable_top_scores"]
    ans = result["answerable_top_scores"]
    print(f"\ntop-1 score, unanswerable: {', '.join(f'{s:.3f}' for s in un)}")
    print(f"top-1 score, answerable:   min {ans[0]:.3f}, median {ans[len(ans) // 2]:.3f}")


def save_result(result: dict, name: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{name}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
