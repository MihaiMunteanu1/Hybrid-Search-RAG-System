"""Answer evaluation: the full pipeline over an evaluation set, in resumable batches.

Each run answers the next `batch` questions not yet in `data/eval/answers/<config>.jsonl`
and appends them, so a long set is done in several short runs.

Measured automatically, without reading the answers:
    answered     an answerable question got an answer, not a refusal
    cites gold   at least one cited source overlaps the gold quote
    refused      an unanswerable question got a refusal
Correctness is judged by hand: `report()` prints each answer next to the expected one,
and the verdict is written into the file.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from rag.config import DATA_DIR
from rag.evaluation.gold import DOCUMENTS, QUESTIONS, attach_spans, is_relevant, read_jsonl
from rag.generation.answer import answer
from rag.generation.llm import DEFAULT_MODEL, TEMPERATURE, start_server
from rag.ingestion.indexing import load

ANSWERS = DATA_DIR / "eval" / "answers"


def span(hit) -> dict:
    """Where a source is, so its relevance can be recomputed against a corrected gold."""
    return {"source": hit.source, "start_char": hit.chunk["start_char"],
            "end_char": hit.chunk["end_char"], "page": hit.chunk.get("page")}


def record(q: dict, result, settings: dict) -> dict:
    return {
        "id": q["id"], "type": q["type"], "question": q["question"],
        "expected": q["answer"], "settings": settings,
        "found": result.found, "refusal": result.refusal,
        "text": result.text, "raw": result.raw, "first_raw": result.first_raw,
        # Relevance is not stored; report() recomputes it from the current gold.
        "sources": [span(h) for h in result.sources],
        "cited": [{"n": n, **span(h)} for n, h in zip(result.cited, result.cited_sources)],
        "invalid_citations": result.invalid_citations,
        "timings": result.timings,
        # Filled in by hand: "correct", "partial", "wrong", or for a refusal "ok" / "missed".
        "verdict": None,
    }


def run(config: str, top_n: int, per_language: bool, batch: int, strategy: str,
        ask_for_citations: bool = True, ids: list[str] | None = None,
        use_reranker: bool = True,
        questions_path: Path = QUESTIONS) -> None:
    """Answer the next `batch` unanswered questions and append them to the config's file."""
    questions = read_jsonl(questions_path)
    if ids:
        questions = [q for q in questions if q["id"] in ids]
    problems = attach_spans(questions, read_jsonl(DOCUMENTS))
    if problems:
        raise ValueError("evaluation set out of sync with the corpus:\n  " + "\n  ".join(problems))

    out = ANSWERS / f"{config}.jsonl"
    done = {r["id"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [q for q in questions if q["id"] not in done][:batch]
    if not todo:
        print(f"{config}: all {len(questions)} questions answered")
        return

    settings = {"strategy": strategy, "top_n": top_n, "per_language": per_language,
                "reranker": use_reranker,
                "ask_for_citations": ask_for_citations, "model": Path(DEFAULT_MODEL).name,
                "temperature": TEMPERATURE, "questions": questions_path.as_posix()}
    ANSWERS.mkdir(parents=True, exist_ok=True)
    with load(DATA_DIR / "index" / strategy) as index, start_server() as llm:
        for q in todo:
            started = time.time()
            result = answer(index, llm, q["question"], top_n=top_n, per_language=per_language,
                            use_reranker=use_reranker,
                            ask_for_citations=ask_for_citations)
            row = record(q, result, settings)
            # One line at a time, so a stopped run keeps what it finished.
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            status = "refuz" if not result.found else f"citează {[c['n'] for c in row['cited']]}"
            if result.first_raw is not None:
                status += "  (după cererea de citări)"
            print(f"{q['id']} {q['type']:13} {time.time() - started:5.0f} s  {status}")
    left = len(questions) - len(done) - len(todo)
    print(f"\n{config}: {len(done) + len(todo)}/{len(questions)} done, {left} left")


def report(config: str) -> None:
    """Print every answer next to the expected one, then the automatic measures."""
    rows = read_jsonl(ANSWERS / f"{config}.jsonl")
    # Judged against the current question file, whose gold may have been corrected
    # since the answers were generated.
    questions_path = Path(rows[0]["settings"].get("questions", QUESTIONS)) if rows else QUESTIONS
    questions = {q["id"]: q for q in read_jsonl(questions_path)}
    attach_spans(list(questions.values()), read_jsonl(DOCUMENTS))
    for r in rows:
        q = questions[r["id"]]
        r["type"], r["expected"] = q["type"], q["answer"]
        for c in r["cited"]:
            c["relevant"] = bool(q["gold"]) and is_relevant(c, q)
        r["gold_in_sources"] = bool(q["gold"]) and any(is_relevant(s, q) for s in r["sources"])
    answerable = [r for r in rows if r["type"] != "unanswerable"]
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]

    for r in rows:
        print(f"\n--- {r['id']} {r['type']}  {r['question']}")
        print(f"    așteptat: {r['expected']}")
        if r.get("first_raw") is not None:
            print(f"    inițial (fără citări): {r['first_raw'][:300]}")
        print(f"    primit:   {r['text']}")
        if r["cited"]:
            print("    citări:  " + ", ".join(
                f"[{c['n']}] {c['source'].split('/')[-1][:30]}{' ✓' if c['relevant'] else ''}"
                for c in r["cited"]))
        if r["refusal"]:
            print(f"    refuz:   {r['refusal']}   (sursa corectă era printre surse: "
                  f"{'da' if r['gold_in_sources'] else 'nu'})")

    def share(n, d):
        return f"{n}/{d}" + (f" ({n / d:.0%})" if d else "")

    answered = sum(r["found"] for r in answerable)
    cites_gold = sum(any(c["relevant"] for c in r["cited"]) for r in answerable)
    had_gold = sum(r["gold_in_sources"] for r in answerable)
    refused = sum(not r["found"] for r in unanswerable)
    invalid = sum(bool(r["invalid_citations"]) for r in rows)
    seconds = [r["timings"]["generation"] for r in rows if r["timings"].get("generation")]
    tokens = [r["timings"]["prompt_tokens"] for r in rows if r["timings"].get("prompt_tokens")]
    print(f"\n=== {config}: {len(rows)} questions, settings {rows[0]['settings'] if rows else {}}")
    print(f"  răspunsul corect era printre surse:   {share(had_gold, len(answerable))}")
    print(f"  întrebări cu răspuns, au răspuns:      {share(answered, len(answerable))}")
    print(f"  ... și citează sursa corectă:          {share(cites_gold, len(answerable))}")
    print(f"  întrebări fără răspuns, refuzate:      {share(refused, len(unanswerable))}")
    print(f"  răspunsuri cu citări inventate:        {share(invalid, len(rows))}")
    if seconds:
        print(f"  generare: medie {sum(seconds) / len(seconds):.0f} s, "
              f"{sum(tokens) // max(1, len(tokens))} tokeni citiți în medie")
    judged = [r for r in rows if r["verdict"]]
    print(f"  verdicte citite de mână: {len(judged)}/{len(rows)}")
