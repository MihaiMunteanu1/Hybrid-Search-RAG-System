"""The document library: folders of documents over one live index.

A folder is a directory under `data/raw/`, and a document's source is "<folder>/<file>".
Every upload is also kept on disk, so a full rebuild from `data/raw/`
(`python -m rag.ingestion`) reproduces the library exactly.

Adding a document embeds only its own chunks, then rebuilds BM25; removing one deletes
its points and rebuilds BM25. `data/processed/documents.jsonl` is kept in step, because
the evaluation locates its gold answers in it.
"""
from __future__ import annotations

import csv
import datetime
import io
import json
import random
import re
import shutil
import threading
import time
from dataclasses import asdict
from pathlib import Path

import openpyxl
from openpyxl.utils import get_column_letter

from rag.config import DATA_DIR
from rag.ingestion.chunking import chunk_document
from rag.ingestion.indexing import (Index, delete_sources, embed_chunks, open_or_create,
                                    save, upsert, write_atomic)
from rag.ingestion.loaders import LOADERS, decode, load_file

# A folder name becomes a directory name on Windows and Linux: letters (with
# diacritics), digits, spaces, dots, dashes and underscores, starting with a letter
# or digit.
FOLDER_NAME = re.compile(r"^[\w][\w .-]{0,63}$", re.UNICODE)
# Device names that Windows refuses as file or directory names.
RESERVED = re.compile(r"^(con|prn|aux|nul|com\d|lpt\d)(\..*)?$", re.IGNORECASE)
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class Library:
    """Folders and documents, with search and answers scoped to a folder."""

    def __init__(self, data_dir: Path = DATA_DIR, strategy: str = "structure",
                 with_heading: bool = True):
        self.raw_dir = data_dir / "raw"
        self.index_dir = data_dir / "index" / strategy
        self.documents_file = data_dir / "processed" / "documents.jsonl"
        self.strategy = strategy
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.index: Index = open_or_create(self.index_dir, with_heading)
        # The API serves requests on several threads; changes to the index are serialized.
        self._lock = threading.Lock()

    def close(self) -> None:
        self.index.close()

    def __enter__(self) -> Library:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------------------------------------------------------------- folders

    def folders(self) -> list[dict]:
        """Every folder, with how many documents and chunks it holds."""
        counts: dict[str, dict] = {}
        for chunk in self.index.chunks:
            entry = counts.setdefault(folder_of(chunk["source"]), {"documents": set(), "chunks": 0})
            entry["documents"].add(chunk["source"])
            entry["chunks"] += 1
        names = {p.name for p in self.raw_dir.iterdir() if p.is_dir()} | set(counts)
        return [{"name": name, "documents": len(counts.get(name, {}).get("documents", ())),
                 "chunks": counts.get(name, {}).get("chunks", 0)}
                for name in sorted(names, key=str.lower)]

    def create_folder(self, name: str) -> str:
        name = check_folder_name(name)
        path = self.raw_dir / name
        if path.exists():
            raise FileExistsError(f"folder {name!r} already exists")
        path.mkdir()
        return name

    def delete_folder(self, name: str, with_documents: bool = False) -> int:
        """Remove a folder and return how many documents went with it.

        A folder that still holds documents is only deleted with `with_documents=True`.
        """
        path = self.raw_dir / check_folder_name(name)
        if not path.is_dir():
            raise KeyError(f"no folder {name!r}")
        sources = self.sources(name)
        if sources and not with_documents:
            raise ValueError(f"folder {name!r} holds {len(sources)} document(s); "
                             f"remove them first or delete with_documents=True")
        with self._lock:
            if sources:
                self._drop(sources)
            shutil.rmtree(path)
        return len(sources)

    # -------------------------------------------------------------- documents

    def sources(self, folder: str | None = None) -> list[str]:
        """The indexed documents, optionally only those of one folder."""
        return sorted(s for s in self.index.language
                      if folder is None or folder_of(s) == folder)

    def documents(self, folder: str | None = None) -> list[dict]:
        """One summary per document (language, chunks, pages), derived from its chunks."""
        by_source: dict[str, dict] = {}
        for c in self.index.chunks:
            if folder is not None and folder_of(c["source"]) != folder:
                continue
            entry = by_source.setdefault(c["source"], {
                "source": c["source"], "folder": folder_of(c["source"]),
                "language": self.index.language.get(c["source"]),
                "chunks": 0, "chars": 0, "pages": 0})
            entry["chunks"] += 1
            entry["chars"] += c["num_chars"]
            entry["pages"] = max(entry["pages"], c.get("page") or 0)
        return [by_source[s] for s in sorted(by_source)]

    def file_path(self, source: str) -> Path:
        """Where an indexed document's original file is. Only indexed sources resolve, and
        only to a file inside data/raw, so a crafted source cannot reach anything else."""
        if source not in self.index.language:
            raise KeyError(f"no document {source!r}")
        raw = self.raw_dir.resolve()
        path = (raw / source).resolve()
        if not path.is_relative_to(raw) or not path.is_file():
            raise KeyError(f"no file for {source!r}")
        return path

    def document_text(self, source: str) -> dict:
        """A document's text as the loaders extracted it (markdown headings, tables), with
        the character offset where each PDF page starts."""
        self.file_path(source)
        with self.documents_file.open(encoding="utf-8") as f:
            for line in f:
                document = json.loads(line)
                if document["source"] == source:
                    metadata = document.get("metadata", {})
                    return {"source": source, "format": metadata.get("format"),
                            "text": document["text"],
                            "page_starts": metadata.get("page_starts")}
        raise KeyError(f"no text for {source!r}")

    def sheets(self, source: str) -> dict:
        """A spreadsheet or CSV as rows of cell text, sheet by sheet, for display."""
        return {"source": source, "sheets": read_sheets(self.file_path(source))}

    def add(self, folder: str, filename: str, data: bytes, replace: bool = False) -> dict:
        """Read, chunk, embed and index one file, and return a short report.

        The file is first written under a temporary name and only takes its final place
        once it has been read and chunked, so an unreadable file never replaces the
        version already indexed.
        """
        folder = check_folder_name(folder)
        if not (self.raw_dir / folder).is_dir():
            raise KeyError(f"no folder {folder!r}; create it first")
        name = check_file_name(filename)
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError(f"file is {len(data) / 2**20:.0f} MB; "
                             f"the limit is {MAX_UPLOAD_BYTES // 2**20} MB")
        source = f"{folder}/{name}"
        final = self.raw_dir / folder / name
        if (source in self.index.language or final.exists()) and not replace:
            raise FileExistsError(f"{source} already exists; pass replace=True to update it")

        started = time.perf_counter()
        tmp = final.with_name(f".upload-{name}")
        tmp.write_bytes(data)
        try:
            document = asdict(load_file(tmp, source))
            chunks = [asdict(c) for c in chunk_document(document, self.strategy)]
            if not chunks:
                raise ValueError("the file was read but produced no chunks")
            vectors = embed_chunks(chunks, self.index.with_heading)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

        with self._lock:
            replaced = sum(1 for c in self.index.chunks if c["source"] == source)
            delete_sources(self.index.client, [source])
            upsert(self.index.client, chunks, vectors)
            self.index.chunks = [c for c in self.index.chunks if c["source"] != source] + chunks
            tmp.replace(final)
            self._save([source], add=[document])

        return {"source": source, "chunks": len(chunks), "replaced_chunks": replaced,
                "chars": document["metadata"]["num_chars"],
                "pages": document["metadata"].get("num_pages"),
                "language": self.index.language[source],
                "seconds": round(time.perf_counter() - started, 1)}

    def remove(self, source: str) -> None:
        """Remove one document from the index and from disk."""
        if source not in self.index.language:
            raise KeyError(f"no document {source!r}")
        with self._lock:
            self._drop([source])

    def _drop(self, sources: list[str]) -> None:
        delete_sources(self.index.client, sources)
        self.index.chunks = [c for c in self.index.chunks if c["source"] not in sources]
        for source in sources:
            (self.raw_dir / source).unlink(missing_ok=True)
        self._save(sources)

    def _save(self, removed: list[str], add: list[dict] = ()) -> None:
        """Save the index and update the processed document file to match."""
        save(self.index, self.index_dir)
        documents = []
        if self.documents_file.exists():
            with self.documents_file.open(encoding="utf-8") as f:
                documents = [json.loads(line) for line in f if line.strip()]
        documents = [d for d in documents if d["source"] not in removed] + list(add)
        documents.sort(key=lambda d: d["source"])
        self.documents_file.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(self.documents_file,
                     "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in documents))

    def topics(self, folder: str | None = None, limit: int = 4) -> list[dict]:
        """Section headings from the indexed documents, at random, one per document where
        possible. The interface offers them as starting questions."""
        sources = None if folder is None else set(self._scope(folder))
        first: dict[tuple[str, str], dict] = {}
        documents_with: dict[str, set[str]] = {}
        for c in self.index.chunks:
            heading = clean_heading(c["heading_path"].split(" > ")[-1]) if c["heading_path"] else ""
            if not usable_heading(heading):
                continue
            documents_with.setdefault(heading.lower(), set()).add(c["source"])
            if sources is None or c["source"] in sources:
                first.setdefault((c["source"], heading.lower()),
                                 {"source": c["source"], "heading": heading, "page": c.get("page")})
        # A heading found in several documents is a letterhead ("Faculty of ..."), not a topic.
        candidates = [t for (_, key), t in first.items() if len(documents_with[key]) == 1]
        random.shuffle(candidates)
        picked, used = [], set()
        for candidate in candidates:            # spread over documents first
            if candidate["source"] not in used:
                picked.append(candidate)
                used.add(candidate["source"])
        picked += [c for c in candidates if c not in picked]
        return picked[:limit]

    # ------------------------------------------------------------- questions

    def search(self, question: str, folder: str | None = None, limit: int = 5, **kwargs):
        """The chunks the model would be given for a question."""
        from rag.retrieval.pipeline import retrieve

        return retrieve(self.index, question, top_n=limit, sources=self._scope(folder), **kwargs)

    def ask_stream(self, question: str, llm, folder: str | None = None, **kwargs):
        """The answer events of `rag.generation.answer.answer_stream`, within a folder."""
        from rag.generation.answer import answer_stream

        return answer_stream(self.index, llm, question, sources=self._scope(folder), **kwargs)

    def _scope(self, folder: str | None) -> list[str] | None:
        if folder is None:
            return None
        if not (self.raw_dir / folder).is_dir():
            raise KeyError(f"no folder {folder!r}")
        return self.sources(folder)


def folder_of(source: str) -> str:
    return source.split("/", 1)[0]


# ------------------------------------------------------------ spreadsheet view

MAX_SHEET_ROWS = 2000    # enough to read a sheet; a table of 100 000 rows would stall the page
MAX_SHEET_COLS = 60


def cell_text(value) -> str:
    """A cell as it would read in a spreadsheet: 3 not 3.0, dates without a midnight time."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:.10g}"
    if isinstance(value, datetime.datetime):
        return value.date().isoformat() if value.time() == datetime.time() else value.isoformat(" ")
    if isinstance(value, (datetime.date, datetime.time)):
        return value.isoformat()
    return str(value)


def sheet_json(name: str, rows: list[list[str]], merges: list[dict], widths: list[float],
               total_rows: int) -> dict:
    """One sheet, without its empty trailing rows and columns."""
    while rows and not any(rows[-1]):
        rows.pop()
    used = max((max((i for i, v in enumerate(r) if v), default=-1) for r in rows), default=-1) + 1
    rows = [r[:used] + [""] * (used - len(r[:used])) for r in rows]
    # A column with no width set (every CSV column) is sized to its content, like Excel's
    # auto-fit, within reason.
    fitted = [min(max((len(r[c]) for r in rows[:200]), default=0) + 2, 40) for c in range(used)]
    widths = [(widths[c] if c < len(widths) and widths[c] else max(fitted[c], 6)) for c in range(used)]
    return {"name": name, "rows": rows,
            "merges": [m for m in merges if m["r"] < len(rows) and m["c"] < used],
            "widths": widths, "truncated": total_rows > MAX_SHEET_ROWS}


def read_sheets(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        raw = decode(path.read_bytes(), path.name)
        try:
            delimiter = csv.Sniffer().sniff(raw[:4096], delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = ","
        rows = [r[:MAX_SHEET_COLS] for r in csv.reader(io.StringIO(raw), delimiter=delimiter)]
        return [sheet_json(path.stem, rows[:MAX_SHEET_ROWS], [], [], len(rows))]
    if suffix != ".xlsx":
        raise ValueError(f"{path.name} is not a spreadsheet")
    # Not read_only: merged ranges and column widths are only available in the full mode.
    book = openpyxl.load_workbook(path, data_only=True)
    try:
        sheets = []
        for ws in book.worksheets:
            rows = [[cell_text(v) for v in row] for row in ws.iter_rows(
                max_row=min(ws.max_row, MAX_SHEET_ROWS), max_col=min(ws.max_column, MAX_SHEET_COLS),
                values_only=True)]
            merges = [{"r": m.min_row - 1, "c": m.min_col - 1,
                       "rows": m.max_row - m.min_row + 1, "cols": m.max_col - m.min_col + 1}
                      for m in ws.merged_cells.ranges]
            widths = [ws.column_dimensions[get_column_letter(c)].width or 0
                      for c in range(1, min(ws.max_column, MAX_SHEET_COLS) + 1)]
            sheets.append(sheet_json(ws.title, rows, merges, widths, ws.max_row))
        return sheets
    finally:
        book.close()


def clean_heading(text: str) -> str:
    """A heading without markdown emphasis, leading numbering ("2.", "B.", "IV.") or
    surrounding punctuation."""
    text = re.sub(r"[*_`]+", "", text).strip()
    text = re.sub(r"^(\d+(\.\d+)*|[A-Z]|[IVXLC]+)[.)]\s+", "", text)
    return text.strip(" .:-–")


def usable_heading(text: str) -> bool:
    """A heading that reads as a topic: a few words starting with a capital, with no
    digits (dates, decision numbers), no abbreviations ("Prof. dr."), not ALL CAPS."""
    words = text.split()
    real_words = [w for w in words if sum(c.isalpha() for c in w) >= 4]   # not "CM, CE, C"
    return (3 <= len(words) <= 10 and 15 <= len(text) <= 80
            and len(real_words) >= 2
            and text[:1].isupper()
            and any(c.islower() for c in text)
            and not any(c.isdigit() for c in text)
            and not re.search(r"\b\w{1,5}\.\s", text)
            and not re.match(r"^(art|articolul|article|capitolul|chapter|anexa|annex)\b", text, re.I)
            and not text.endswith(("?", ";", ",")))


def check_folder_name(name: str) -> str:
    name = name.strip()
    if not FOLDER_NAME.match(name) or RESERVED.match(name) or name.endswith((".", " ")):
        raise ValueError("a folder name may hold letters, digits, spaces, dots, dashes and "
                         "underscores, start with a letter or digit, and be 1-64 characters")
    return name


def check_file_name(filename: str) -> str:
    """The bare name of an uploaded file. Browsers can send a path ("C:\\Users\\x\\doc.pdf",
    "../../etc/x"), so only the last component is kept."""
    name = re.split(r"[\\/]", filename.strip())[-1]
    if not name or name.startswith(".") or len(name) > 160 or RESERVED.match(name):
        raise ValueError(f"not a usable file name: {filename!r}")
    suffix = Path(name).suffix.lower()
    if suffix not in LOADERS:
        raise ValueError(f"unsupported format {suffix or '(no extension)'}; "
                         f"supported: {', '.join(sorted(LOADERS))}")
    return name
