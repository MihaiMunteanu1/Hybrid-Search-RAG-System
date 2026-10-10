"""The HTTP API and the web interface (FastAPI). Started with `python -m rag.api`.

One process owns the library (Qdrant in local mode locks its directory) and one
llama-server: started here, or reached at RAG_LLM_URL when it runs on its own. Endpoints are plain `def`, so FastAPI runs them on a thread pool and a
long upload does not block other requests. Generated API docs: /docs.
"""
from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import threading
import time
from contextlib import ExitStack, asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag.config import DATA_DIR, LLM_URL
from rag.generation.llm import Cancelled
from rag.library import MAX_UPLOAD_BYTES, Library

# The React interface built by `npm run build` in frontend/.
FRONTEND = Path(__file__).resolve().parents[2] / "frontend" / "dist"


class State:
    library: Library | None = None
    llm = None
    model: str | None = None
    # One answer at a time: on a CPU two parallel generations are each half as fast.
    generating = threading.Lock()


state = State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    with ExitStack() as stack:
        state.library = stack.enter_context(Library(DATA_DIR))
        if os.environ.get("RAG_NO_LLM") == "1":
            pass
        elif LLM_URL:
            # A llama-server run separately, as in docker-compose.yml.
            from rag.generation.llm import LLM

            state.llm = LLM(LLM_URL)
            stack.callback(state.llm.close)
            state.llm.wait_until_healthy(timeout=900)
            state.model = state.llm.model_name()
        else:
            from rag.generation.llm import DEFAULT_MODEL, start_server

            state.llm = stack.enter_context(start_server())
            state.model = DEFAULT_MODEL.name
        # Load the embedding and reranking models now, not on the first question.
        state.library.search("warm-up", limit=1)
        yield
        state.library = state.llm = None


app = FastAPI(title="RAG UBB", lifespan=lifespan)


@app.middleware("http")
async def revalidate_pages(request, call_next):
    """HTML is always revalidated: after a rebuild of the interface, a cached index.html
    would still point at the old, deleted script files. The scripts themselves have a
    hash in their names, so they can stay cached."""
    response = await call_next(request)
    if response.headers.get("content-type", "").startswith("text/html"):
        response.headers.setdefault("Cache-Control", "no-cache")
    return response


def library() -> Library:
    if state.library is None:
        raise HTTPException(503, "the library is not open yet")
    return state.library


def errors(call):
    """Map the library's exceptions to HTTP: missing -> 404, duplicate -> 409, invalid -> 400."""
    try:
        return call()
    except KeyError as e:
        # str() of a KeyError is the repr of its message; the message itself is args[0].
        raise HTTPException(404, str(e.args[0]) if e.args else "not found")
    except FileExistsError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


# ------------------------------------------------------------------ JSON shapes


def hit_json(number: int, hit) -> dict:
    chunk = hit.chunk
    return {"n": number, "source": chunk["source"], "page": chunk.get("page"),
            "heading_path": chunk["heading_path"], "text": chunk["text"],
            "start_char": chunk["start_char"], "end_char": chunk["end_char"],
            "score": round(hit.score, 4)}


def answer_json(result) -> dict:
    return {"text": result.text, "found": result.found, "refusal": result.refusal,
            "language": result.language, "cited": result.cited,
            "invalid_citations": result.invalid_citations,
            "timings": {k: (round(v, 2) if isinstance(v, float) else v)
                        for k, v in result.timings.items()}}


# ------------------------------------------------------------------- endpoints


@app.get("/api/status")
def status() -> dict:
    lib = library()
    return {"documents": len(lib.sources()), "chunks": len(lib.index.chunks),
            "folders": len(lib.folders()), "llm": state.llm is not None, "model": state.model,
            "max_upload_mb": MAX_UPLOAD_BYTES // 2**20}


@app.get("/api/folders")
def list_folders() -> list[dict]:
    return library().folders()


class NewFolder(BaseModel):
    name: str


@app.post("/api/folders", status_code=201)
def create_folder(body: NewFolder) -> dict:
    return {"name": errors(lambda: library().create_folder(body.name))}


@app.delete("/api/folders/{name}")
def delete_folder(name: str, with_documents: bool = False) -> dict:
    removed = errors(lambda: library().delete_folder(name, with_documents))
    return {"name": name, "documents_removed": removed}


@app.get("/api/documents")
def list_documents(folder: str | None = None) -> list[dict]:
    return library().documents(folder)


@app.post("/api/folders/{folder}/documents")
def upload(folder: str, files: list[UploadFile] = File(...), replace: bool = Form(False)) -> list[dict]:
    """Add one or more files. Each gets its own report, so one bad file does not stop
    the others."""
    reports = []
    for upload_file in files:
        # One byte past the limit is enough to know the file is too large.
        data = upload_file.file.read(MAX_UPLOAD_BYTES + 1)
        try:
            report = errors(lambda: library().add(folder, upload_file.filename or "", data,
                                                  replace=replace))
            reports.append({"file": upload_file.filename, "ok": True, **report})
        except HTTPException as e:
            if e.status_code == 404:   # the folder itself is missing
                raise
            reports.append({"file": upload_file.filename, "ok": False, "error": e.detail,
                            "status": e.status_code})
    return reports


@app.get("/api/files/{source:path}")
def document_file(source: str) -> FileResponse:
    """The original file, to be shown inline (PDFs in the browser's viewer).

    Anything but a PDF is served sandboxed: an uploaded HTML or SVG file must not run
    scripts with the app's origin, where it could call this API. Chrome refuses to show
    a PDF in a sandboxed document, and its PDF viewer runs no script of the page anyway.
    """
    path = errors(lambda: library().file_path(source))
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    headers = {"X-Content-Type-Options": "nosniff"}
    if media_type != "application/pdf":
        headers["Content-Security-Policy"] = "sandbox"
    return FileResponse(path, media_type=media_type, filename=path.name,
                        content_disposition_type="inline", headers=headers)


@app.get("/api/text/{source:path}")
def document_text(source: str) -> dict:
    """A document's extracted text, for formats the browser cannot show itself."""
    return errors(lambda: library().document_text(source))


@app.get("/api/sheets/{source:path}")
def document_sheets(source: str) -> dict:
    """A spreadsheet or CSV, sheet by sheet, as rows of cell text."""
    return errors(lambda: library().sheets(source))


@app.get("/api/topics")
def topics(folder: str | None = None, limit: int = 4) -> list[dict]:
    """Section headings from the library, offered as starting questions."""
    return errors(lambda: library().topics(folder, max(1, min(limit, 8))))


@app.delete("/api/documents/{source:path}")
def delete_document(source: str) -> dict:
    errors(lambda: library().remove(source))
    return {"removed": source}


class Query(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    folder: str | None = None
    top_n: int = Field(5, ge=1, le=10)


@app.post("/api/search")
def search(query: Query) -> list[dict]:
    hits = errors(lambda: library().search(query.question, folder=query.folder,
                                           limit=query.top_n))
    return [hit_json(n, h) for n, h in enumerate(hits, start=1)]


@app.post("/api/ask")
def ask(query: Query) -> StreamingResponse:
    """The answer as server-sent events: the sources, the text as it is written, then
    the parsed answer (see rag.generation.answer.answer_stream)."""
    if state.llm is None:
        raise HTTPException(503, "the server was started without the language model (--no-llm)")
    # Created before the stream starts, so an unknown folder is a plain 404.
    events = errors(lambda: library().ask_stream(query.question, state.llm,
                                                 folder=query.folder, top_n=query.top_n))

    def sse():
        with state.generating:
            try:
                yield from answer_events()
            except Cancelled:
                return   # the client left; ClosingStream cancelled the model's request
            finally:
                # Also closes the request to llama-server, which stops generating.
                events.close()

    def answer_events():
        for event in events:
            if event["event"] == "sources":
                payload = {"event": "sources", "language": event["language"],
                           "sources": [hit_json(n, h) for n, h in
                                       enumerate(event["sources"], start=1)]}
            elif event["event"] == "answer":
                payload = {"event": "answer", **answer_json(event["answer"])}
            else:
                payload = event
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return ClosingStream(sse(), media_type="text/event-stream",
                         headers={"Cache-Control": "no-cache"}, cancel=state.llm.cancel)


class ClosingStream(StreamingResponse):
    """A StreamingResponse that closes its generator when the request ends.

    Starlette does not: when the client leaves mid-answer (closed tab, folder switch),
    the generator stays suspended inside `with state.generating`, and every later
    question waits on that lock until the garbage collector runs. `cancel` stops the
    model's request first, so a client that leaves while the model is still reading
    the sources does not keep the lock for the tens of seconds before its first token.
    """

    def __init__(self, content, cancel=None, **kwargs) -> None:
        super().__init__(content, **kwargs)
        self.generator = content
        self.cancel = cancel

    async def __call__(self, scope, receive, send) -> None:
        # Starlette notices a client that left only when it next sends something, i.e.
        # at the model's first token. Watching the incoming messages catches the
        # disconnect at once, so the model's request is cancelled while it still reads
        # the sources. The messages are passed on, so Starlette sees them as before.
        messages: asyncio.Queue = asyncio.Queue()

        async def watch() -> None:
            while True:
                message = await receive()
                await messages.put(message)
                if message["type"] == "http.disconnect":
                    if self.cancel is not None and self.generator.gi_frame is not None:
                        await run_in_threadpool(self.cancel)
                    return

        watcher = asyncio.create_task(watch())
        try:
            await super().__call__(scope, messages.get, send)
        finally:
            watcher.cancel()
            await run_in_threadpool(self._close, self.generator, self.cancel)

    @staticmethod
    def _close(generator, cancel=None) -> None:
        if cancel is not None and generator.gi_frame is not None:
            cancel()   # the stream did not finish: the client left
        # A worker thread may still be inside next(), waiting for a token; close()
        # refuses a running generator, so wait for that step to finish.
        while True:
            try:
                generator.close()
                return
            except ValueError:   # "generator already executing"
                time.sleep(0.05)


# Mounted last, so the /api routes above match first.
if FRONTEND.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
else:
    @app.get("/")
    def no_frontend() -> dict:
        return {"detail": "the interface is not built: run `npm install && npm run build` in "
                          "frontend/, or `npm run dev` there and open http://localhost:5173"}
