"""A local page for checking an answer run by hand: the questions and their gold, and the
verdict on each answer.

    python -m rag.evaluation review [--config dev-rerank test2-final] [--port 8010]

For each answer it shows the question, the expected answer, the gold quotes in their
context, the model's reply and the five sources it was given (gold overlaps highlighted),
next to the proposed verdict. What the reviewer sets is written straight back:
`verdict` and `review_note` into data/eval/answers/<config>.jsonl; `verified`, `answer`,
`gold` and `note` into the question file the run was made from. Every other field and the
line order stay as they are. It needs neither the index nor the model, so it can run next
to the app.
"""
from __future__ import annotations

import bisect
import json
import mimetypes
import os
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from rag.config import DATA_DIR
from rag.evaluation.answers import ANSWERS
from rag.evaluation.gold import DOCUMENTS, QUESTIONS, find_spans, overlaps, read_jsonl

PAGE = Path(__file__).with_name("review.html")
RAW = DATA_DIR / "raw"
# The final runs first, then the earlier ones the paper compares them with.
CONFIGS = ["dev-rerank", "test2-final", "t0-top5", "t0-lang-top5", "test-t0-top5",
           "dev-fixes", "test2-fixes", "tables-before", "tables-after", "tables-after2",
           "tables2-before", "tables2-after"]
TYPES = ["fact", "paraphrase", "table", "cross_lingual", "unanswerable"]
VERDICTS = {"answerable": ["correct", "partial", "wrong", "refused"],
            "unanswerable": ["ok", "missed"]}
CONTEXT = 500   # characters shown on each side of a gold quote


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Rewrite a JSONL file in the format it was written in, line endings included (a
    Windows checkout has CRLF), replacing it in one step."""
    with path.open("rb") as f:
        newline = "\r\n" if f.readline().endswith(b"\r\n") else "\n"
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline=newline) as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


class Review:
    def __init__(self, configs: list[str]):
        self.configs = configs
        self.lock = threading.Lock()
        self.documents = {d["source"]: d for d in read_jsonl(DOCUMENTS)}

    def paths(self, config: str) -> tuple[Path, Path]:
        if config not in self.configs:
            raise KeyError(f"unknown run {config!r}")
        answers = ANSWERS / f"{config}.jsonl"
        first = read_jsonl(answers)[0]
        # The earliest runs did not record their question file: the development set.
        return answers, Path(first["settings"].get("questions", QUESTIONS))

    def page_of(self, source: str, position: int) -> int | None:
        starts = self.documents[source].get("metadata", {}).get("page_starts")
        return bisect.bisect_right(starts, position) if starts else None

    def gold_view(self, gold: list[dict]) -> list[dict]:
        """Each gold quote with every place it occurs and the text around the first one."""
        out = []
        for g in gold:
            document = self.documents.get(g["source"])
            spans = find_spans(document["text"], g["quote"]) if document else []
            entry = {"source": g["source"], "quote": g["quote"], "found": len(spans)}
            if spans:
                start, end = spans[0]
                lo = max(0, start - CONTEXT)
                entry.update(before=document["text"][lo:start],
                             match=document["text"][start:end],
                             after=document["text"][end:end + CONTEXT],
                             page=self.page_of(g["source"], start))
            entry["spans"] = spans
            out.append(entry)
        return out

    def item(self, row: dict, question: dict) -> dict:
        gold = self.gold_view(question["gold"])
        cited = {c["n"] for c in row["cited"]}
        sources = []
        for n, s in enumerate(row["sources"], 1):
            text = self.documents[s["source"]]["text"][s["start_char"]:s["end_char"]]
            # Parts of the chunk that overlap a gold quote, relative to the chunk.
            marks = sorted((max(a, s["start_char"]) - s["start_char"],
                            min(b, s["end_char"]) - s["start_char"])
                           for g in gold if g["source"] == s["source"]
                           for a, b in g["spans"] if overlaps((a, b), (s["start_char"], s["end_char"])))
            sources.append({"n": n, "source": s["source"], "page": s.get("page"), "text": text,
                            "marks": marks, "gold": bool(marks), "cited": n in cited})
        for g in gold:
            del g["spans"]
        kind = "unanswerable" if question["type"] == "unanswerable" else "answerable"
        return {
            "id": row["id"], "type": question["type"], "lang": question.get("lang"),
            "question": question["question"], "asked": row["question"],
            "answer": question["answer"], "expected_at_run": row["expected"],
            "gold": gold, "note": question.get("note", ""), "verified": question["verified"],
            "found": row["found"], "refusal": row["refusal"], "text": row["text"],
            "first_raw": row["first_raw"], "invalid_citations": row["invalid_citations"],
            "sources": sources, "verdicts": VERDICTS[kind],
            "verdict": row["verdict"], "verdict_proposed": row.get("verdict_proposed"),
            "verdict_note": row.get("verdict_note", ""), "review_note": row.get("review_note", ""),
        }

    def load(self, config: str) -> dict:
        answers_path, questions_path = self.paths(config)
        questions = {q["id"]: q for q in read_jsonl(questions_path)}
        items = [self.item(r, questions[r["id"]]) for r in read_jsonl(answers_path)]
        return {"config": config, "answers_file": answers_path.as_posix(),
                "questions_file": questions_path.as_posix(), "items": items,
                "documents": sorted(self.documents), "types": TYPES}

    def save(self, change: dict) -> dict:
        """Apply one reviewer's change to both files and return the item as it now is."""
        with self.lock:
            answers_path, questions_path = self.paths(change["config"])
            rows = read_jsonl(answers_path)
            questions = read_jsonl(questions_path)
            row = next(r for r in rows if r["id"] == change["id"])
            question = next(q for q in questions if q["id"] == change["id"])

            if "gold" in change:
                gold = [{"source": g["source"], "quote": " ".join(g["quote"].split())}
                        for g in change["gold"]]
                for g in gold:
                    document = self.documents.get(g["source"])
                    if document is None:
                        raise ValueError(f"unknown source {g['source']!r}")
                    if not g["quote"] or not find_spans(document["text"], g["quote"]):
                        raise ValueError(f"quote not found in {g['source']}: {g['quote'][:80]!r}")
                question["gold"] = gold
            if "type" in change:
                if change["type"] not in TYPES:
                    raise ValueError(f"unknown type {change['type']!r}")
                question["type"] = change["type"]
            if (question["type"] == "unanswerable") != (not question["gold"]):
                raise ValueError("an unanswerable question has no gold quote, any other has one")
            if "answer" in change:
                question["answer"] = change["answer"].strip()
            if "note" in change:
                note = change["note"].strip()
                if note:
                    question["note"] = note
                else:
                    question.pop("note", None)
            if "verified" in change:
                question["verified"] = bool(change["verified"])

            kind = "unanswerable" if question["type"] == "unanswerable" else "answerable"
            if row["verdict"] not in VERDICTS[kind]:
                row["verdict"] = None   # judged under the other kind of question
            if "verdict" in change:
                if change["verdict"] not in VERDICTS[kind] + [None]:
                    raise ValueError(f"verdict {change['verdict']!r} not valid for a {kind} question")
                row["verdict"] = change["verdict"]
            if "review_note" in change:
                note = change["review_note"].strip()
                if note:
                    row["review_note"] = note
                else:
                    row.pop("review_note", None)

            write_jsonl(questions_path, questions)
            write_jsonl(answers_path, rows)
            return self.item(row, question)

    def raw_file(self, source: str) -> Path:
        """An original document, only from inside data/raw."""
        raw = RAW.resolve()
        path = (raw / source).resolve()
        if source not in self.documents or not path.is_relative_to(raw) or not path.is_file():
            raise KeyError(source)
        return path


def handler(review: Review):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status: int, body: bytes, content_type: str, headers: dict = {}) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, data, status: int = 200) -> None:
            self.send(status, json.dumps(data, ensure_ascii=False).encode("utf-8"),
                      "application/json; charset=utf-8")

        def do_GET(self) -> None:
            path = unquote(urlparse(self.path).path)
            try:
                if path == "/":
                    self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                elif path == "/api/configs":
                    self.send_json(review.configs)
                elif path.startswith("/api/run/"):
                    self.send_json(review.load(path.removeprefix("/api/run/")))
                elif path.startswith("/file/"):
                    file = review.raw_file(path.removeprefix("/file/"))
                    media_type = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
                    # Like the app: only a PDF is shown unsandboxed.
                    headers = {} if media_type == "application/pdf" else {
                        "Content-Security-Policy": "sandbox"}
                    self.send(200, file.read_bytes(), media_type, headers)
                else:
                    self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except KeyError as e:
                self.send_json({"error": f"not found: {e}"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/save":
                self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            try:
                change = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.send_json(review.save(change))
            except (KeyError, ValueError, StopIteration) as e:
                self.send_json({"error": str(e) or type(e).__name__}, HTTPStatus.BAD_REQUEST)

        def log_message(self, format, *args) -> None:
            pass   # one line per request is noise here

    return Handler


def serve(configs: list[str], port: int, open_browser: bool = True) -> None:
    review = Review(configs)
    for config in configs:
        review.paths(config)   # fail now on a missing run, not on the page
    server = ThreadingHTTPServer(("127.0.0.1", port), handler(review))
    url = f"http://127.0.0.1:{port}/"
    print(f"review page: {url}  (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
