"""The evaluation sets, and the rule that decides whether a retrieved chunk is relevant.

Each question's `gold` lists the places that answer it, as {"source", "quote"}; any one
of them is enough. An empty list marks a question the corpus cannot answer. Quotes are
located in `data/processed/documents.jsonl` at run time, never stored as offsets, so
they stay valid when the loaders change.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from rag.config import DATA_DIR

QUESTIONS = DATA_DIR / "eval" / "questions.jsonl"             # development set
TEST_QUESTIONS = DATA_DIR / "eval" / "test_questions.jsonl"   # held-out test set
DOCUMENTS = DATA_DIR / "processed" / "documents.jsonl"


def quote_pattern(quote: str) -> re.Pattern:
    """The quote as a regex in which any whitespace, or a "#" heading marker inside it,
    matches any whitespace. PDF text wraps lines where the quote's author did not."""
    words = [re.escape(w) for w in quote.split()]
    return re.compile(r"(?:\s+#{1,6})?\s+".join(words))


def find_spans(text: str, quote: str) -> list[tuple[int, int]]:
    """Character spans of every occurrence of `quote` in `text`."""
    return [(m.start(), m.end()) for m in quote_pattern(quote).finditer(text)]


def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def attach_spans(questions: list[dict], documents: list[dict]) -> list[str]:
    """Locate every gold quote in its document (adds `spans`); returns the problems found."""
    texts = {d["source"]: d["text"] for d in documents}
    problems = []
    for q in questions:
        for gold in q["gold"]:
            text = texts.get(gold["source"])
            if text is None:
                problems.append(f"{q['id']}: unknown source {gold['source']!r}")
                gold["spans"] = []
                continue
            gold["spans"] = find_spans(text, gold["quote"])
            if not gold["spans"]:
                problems.append(f"{q['id']}: quote not found in {gold['source']}: "
                                f"{gold['quote'][:60]!r}")
    return problems


def is_relevant(chunk: dict, question: dict) -> bool:
    """A chunk is relevant if it overlaps any occurrence of a gold quote.

    Overlap rather than containment: a chunk boundary may fall inside the quote,
    and chunking strategies should not be judged by where their cuts happen to fall.
    """
    span = (chunk["start_char"], chunk["end_char"])
    return any(chunk["source"] == gold["source"] and any(overlaps(span, s) for s in gold["spans"])
               for gold in question["gold"])


def check(questions_path: Path, strategy: str | None = None) -> list[str]:
    """Print where each question's answer is found, and return the problems.

    With `strategy`, also show how many chunks of that index hold each answer.
    """
    questions = read_jsonl(questions_path)
    issues = attach_spans(questions, read_jsonl(DOCUMENTS))
    # The index's own chunk list: documents uploaded through the library never pass
    # through data/processed/chunks.*.jsonl.
    chunks = (read_jsonl(DATA_DIR / "index" / strategy / "chunks.jsonl")
              if strategy else [])

    ids = [q["id"] for q in questions]
    if len(set(ids)) != len(ids):
        issues.append("duplicate question ids")

    for q in questions:
        where = "; ".join(f"{g['source']} x{len(g['spans'])}" for g in q["gold"]) or "(no answer)"
        line = f"{q['id']} {q['type']:13} {'✓' if q['verified'] else '·'}  {where}"
        if chunks and q["gold"]:
            holding = [c for c in chunks if is_relevant(c, q)]
            line += f"  -> {len(holding)} chunk(s)" + (" [!]" if not holding else "")
            if not holding:
                issues.append(f"{q['id']}: no {strategy} chunk overlaps the answer")
        print(line)

    by_type: dict[str, int] = {}
    for q in questions:
        by_type[q["type"]] = by_type.get(q["type"], 0) + 1
    verified = sum(q["verified"] for q in questions)
    print(f"\n{len(questions)} questions {by_type}; verified by hand: {verified}/{len(questions)}")
    for issue in issues:
        print(f"[error] {issue}")
    return issues
