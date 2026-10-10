"""The two indexes over the chunks: dense vectors in Qdrant and a BM25 keyword index.

An index directory holds:
    qdrant/         the vectors, with each chunk as payload (Qdrant in local mode)
    bm25.pkl        the BM25 index, rebuilt whenever the chunk list changes
    chunks.jsonl    the chunk list, the reference both indexes must match
    manifest.json   the embedding model used, written last when saving

`build()` creates an index from scratch; `rag.library` uses the smaller pieces
(`embed_chunks`, `upsert`, `delete_sources`, `save`) to add or remove one document.
Both indexes must hold exactly the same chunks, because RRF fuses their rankings.
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path

from rag.embedding import (EMBEDDING_DIM, EMBEDDING_MODEL, MAX_MODEL_TOKENS, PASSAGE_PREFIX,
                           embed_passages, get_tokenizer)

BATCH_SIZE = 32
COLLECTION = "chunks"
ID_NAMESPACE = uuid.UUID("6f9b1d9e-2f4a-4c7e-9f1b-3a7c5d8e0f21")  # arbitrary but fixed


def point_id(chunk_id: str) -> str:
    """The Qdrant point id of a chunk: a deterministic UUID, so re-indexing a chunk
    overwrites its point instead of adding a copy."""
    return str(uuid.uuid5(ID_NAMESPACE, chunk_id))


# ------------------------------------------------------------- sparse side


def fold(text: str) -> str:
    """Lowercase and strip accents, so "sedinta" matches "ședința".

    Romanian is often typed without diacritics; without this, such a question
    shares no terms with the documents.
    """
    return "".join(c for c in unicodedata.normalize("NFD", text.lower())
                   if not unicodedata.combining(c))


_STEMMERS = None
_STEMS: dict[str, str] = {}           # word -> stem
_TOKENS: dict[str, list[str]] = {}    # text -> tokens; BM25 is rebuilt on every upload


def stemmers():
    """The Romanian and English Snowball stemmers.

    Both are applied to every token, with no language detection: a stem only has to
    be identical on the question side and the document side, not linguistically
    correct. Without stemming, "credit" would never match "credite".
    """
    global _STEMMERS
    if _STEMMERS is None:
        import snowballstemmer
        _STEMMERS = (snowballstemmer.stemmer("romanian"), snowballstemmer.stemmer("english"))
    return _STEMMERS


def bm25_tokens(text: str) -> list[str]:
    """BM25 tokens of a text: folded, alphanumeric, stemmed. Used for documents and
    questions alike. The returned list is cached and must not be modified."""
    cached = _TOKENS.get(text)
    if cached is not None:
        return cached
    words = ["".join(ch for ch in word if ch.isalnum())
             for word in fold(text).split() if any(ch.isalnum() for ch in word)]
    unseen = list(dict.fromkeys(w for w in words if w not in _STEMS))
    if unseen:
        romanian, english = stemmers()
        _STEMS.update(zip(unseen, english.stemWords(romanian.stemWords(unseen))))
    tokens = [_STEMS[w] for w in words]
    _TOKENS[text] = tokens
    return tokens


# ----------------------------------------------------------------- language

# Function words: frequent in any text of their language and rare in the other.
# "in" is not in the English list because Romanian without diacritics writes "în" as "in".
STOPWORDS = {
    "ro": {"și", "si", "de", "la", "cu", "pe", "care", "este", "sunt", "pentru", "din",
           "sau", "nu", "al", "ale", "unei", "unui", "se", "prin", "privind"},
    "en": {"the", "and", "of", "to", "is", "are", "for", "with", "that", "this", "be",
           "by", "on", "as", "or", "an", "which", "from", "at", "it"},
}


def detect_language(text: str) -> str:
    """"ro" or "en": the language whose function words the text uses more."""
    words = text.lower().split()
    counts = {lang: sum(1 for w in words if w in stop) for lang, stop in STOPWORDS.items()}
    return max(counts, key=counts.get)


# --------------------------------------------------------------- the index


@dataclass
class Index:
    """Everything needed to search the chunks.

    `chunks` is the source of truth; the Qdrant points and the BM25 rows mirror it.
    Code that changes `chunks` calls `save()` afterwards.
    """
    chunks: list[dict]
    client: object
    bm25: object = None
    with_heading: bool = True
    collection: str = COLLECTION

    def __post_init__(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        """Recompute the lookups derived from the chunk list."""
        self._by_id = {c["chunk_id"]: c for c in self.chunks}
        # One language per document, decided on all of its text, since a single chunk
        # can be a table of numbers or an English abstract inside a Romanian thesis.
        text_by_source: dict[str, list[str]] = {}
        for c in self.chunks:
            text_by_source.setdefault(c["source"], []).append(c["text"])
        self.language = {s: detect_language(" ".join(t)) for s, t in text_by_source.items()}

    def by_id(self, chunk_id: str) -> dict | None:
        return self._by_id.get(chunk_id)

    def close(self) -> None:
        """Release the Qdrant files. In local mode Qdrant locks its directory, so only
        one open client (one process) can use an index at a time."""
        self.client.close()

    def __enter__(self) -> Index:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def embedding_text(chunk: dict, with_heading: bool = True) -> str:
    """The text that is embedded and indexed for a chunk: its heading path, then its text.

    The heading path tells the embedding which chapter a chunk belongs to, which
    a short passage often does not say itself. Headings already present in the
    text are not repeated.
    """
    if not with_heading or not chunk["heading_path"]:
        return chunk["text"]
    headings = {line.lstrip("#").strip() for line in chunk["text"].split("\n")
                if line.lstrip().startswith("#")}
    parts = [p for p in chunk["heading_path"].split(" > ") if p not in headings]
    return f"{' > '.join(parts)}\n\n{chunk['text']}" if parts else chunk["text"]


def read_chunks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_atomic(path: Path, data: str | bytes) -> None:
    """Write through a temporary file, so a crash never leaves a half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    if isinstance(data, bytes):
        tmp.write_bytes(data)
    else:
        tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)


# ------------------------------------------------------- the pieces of a build


def embed_chunks(chunks: list[dict], with_heading: bool = True, batch_size: int = BATCH_SIZE,
                 show_progress: bool = False):
    """Embedding vectors for these chunks, in order. This is the slow step on a CPU."""
    texts = [embedding_text(c, with_heading) for c in chunks]
    # The heading path is added after chunking, so the total can exceed the model's
    # window; the encoder would then drop the end of the chunk without a warning.
    tokenizer = get_tokenizer()
    for chunk, text in zip(chunks, texts):
        size = len(tokenizer(PASSAGE_PREFIX + text, add_special_tokens=True)["input_ids"])
        if size > MAX_MODEL_TOKENS:
            print(f"[warn] {chunk['source']}#{chunk['chunk_index']}: {size} tokens with "
                  f"its heading path, over the {MAX_MODEL_TOKENS} window; it will be truncated")
    return embed_passages(texts, batch_size=batch_size, show_progress=show_progress)


def create_collection(client) -> None:
    from qdrant_client import models

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE))


def upsert(client, chunks: list[dict], vectors) -> None:
    """Store chunks with their vectors. The whole chunk is the payload: `source` is what
    the folder filter matches on."""
    from qdrant_client import models

    client.upsert(collection_name=COLLECTION, points=[
        models.PointStruct(id=point_id(c["chunk_id"]), vector=v.tolist(), payload=c)
        for c, v in zip(chunks, vectors)])


def delete_sources(client, sources: list[str]) -> None:
    """Remove every point belonging to these documents."""
    from qdrant_client import models

    if sources:
        client.delete(collection_name=COLLECTION, points_selector=models.FilterSelector(
            filter=models.Filter(must=[models.FieldCondition(
                key="source", match=models.MatchAny(any=sources))])))


def build_bm25(chunks: list[dict], out_dir: Path, with_heading: bool = True):
    """Build and save the BM25 index over the same text the dense side embedded.

    rank_bm25 cannot be updated incrementally (every IDF changes when a chunk is
    added), so it is rebuilt; the token cache keeps that under a second. An empty
    library has no BM25 index.
    """
    from rank_bm25 import BM25Okapi

    path = out_dir / "bm25.pkl"
    if not chunks:
        path.unlink(missing_ok=True)
        return None
    bm25 = BM25Okapi([bm25_tokens(embedding_text(c, with_heading)) for c in chunks])
    write_atomic(path, pickle.dumps({"bm25": bm25, "chunk_ids": [c["chunk_id"] for c in chunks]}))
    return bm25


def save(index: Index, index_dir: Path) -> None:
    """Write the chunk list, rebuild BM25 over it, then write the manifest.

    Qdrant persists its own points. The manifest goes last, so an index with a
    manifest is one that was saved completely.
    """
    index.refresh()
    write_atomic(index_dir / "chunks.jsonl",
                 "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in index.chunks))
    index.bm25 = build_bm25(index.chunks, index_dir, index.with_heading)
    write_atomic(index_dir / "manifest.json", json.dumps(
        {"model": EMBEDDING_MODEL, "with_heading": index.with_heading,
         "count": len(index.chunks)}, indent=2))


def build(chunks: list[dict], out_dir: Path, with_heading: bool = True,
          batch_size: int = BATCH_SIZE) -> Index:
    """Embed and index every chunk, replacing whatever is in `out_dir`."""
    from qdrant_client import QdrantClient

    ids = [c["chunk_id"] for c in chunks]
    if len(set(ids)) != len(ids):
        # A repeated id would overwrite a point and leave the two indexes disagreeing.
        raise ValueError(f"{len(ids) - len(set(ids))} duplicate chunk_id(s) in the chunk file")

    vectors = embed_chunks(chunks, with_heading, batch_size, show_progress=True) if chunks else []
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    client = QdrantClient(path=str(out_dir / "qdrant"))
    create_collection(client)
    if chunks:
        upsert(client, chunks, vectors)
    index = Index(chunks=list(chunks), client=client, with_heading=with_heading)
    save(index, out_dir)
    return index


def check_in_sync(index: Index) -> list[str]:
    """Problems found when comparing Qdrant, BM25 and the chunk list; empty if they agree."""
    problems = []
    if index.chunks and index.bm25 is None:
        problems.append("no bm25 index")
    elif index.bm25 is not None and len(index.bm25.doc_freqs) != len(index.chunks):
        problems.append(f"bm25 has {len(index.bm25.doc_freqs)} documents, "
                        f"chunks file has {len(index.chunks)}")
    expected = {c["chunk_id"] for c in index.chunks}
    if len(expected) != len(index.chunks):
        problems.append(f"{len(index.chunks) - len(expected)} duplicate chunk_id(s)")
    stored, offset = set(), None
    while True:
        points, offset = index.client.scroll(collection_name=index.collection, limit=256,
                                             offset=offset, with_payload=["chunk_id"],
                                             with_vectors=False)
        stored.update(p.payload["chunk_id"] for p in points)
        if offset is None:
            break
    if stored != expected:
        problems.append(f"qdrant holds {len(stored)} ids, chunks file has {len(expected)}; "
                        f"missing {len(expected - stored)}, extra {len(stored - expected)}")
    return problems


def load(index_dir: Path) -> Index:
    """Open an existing index.

    Refuses an index built with a different embedding model: the vectors would not
    be comparable with the question's, and nothing else would raise an error.
    """
    from qdrant_client import QdrantClient

    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["model"] != EMBEDDING_MODEL:
        raise ValueError(f"{index_dir} was built with {manifest['model']}, but the code now "
                         f"uses {EMBEDDING_MODEL}. Rebuild the index before querying it.")
    chunks = read_chunks(index_dir / "chunks.jsonl")
    bm25 = None
    if (index_dir / "bm25.pkl").exists():
        with (index_dir / "bm25.pkl").open("rb") as f:
            stored = pickle.load(f)
        # BM25 rows are mapped back to chunks by position, so the order must match too.
        if stored["chunk_ids"] != [c["chunk_id"] for c in chunks]:
            raise ValueError(f"{index_dir}: bm25.pkl and chunks.jsonl list different chunks; "
                             f"rebuild the index")
        bm25 = stored["bm25"]
    return Index(chunks=chunks, client=QdrantClient(path=str(index_dir / "qdrant")), bm25=bm25,
                 with_heading=manifest.get("with_heading", True))


def open_or_create(index_dir: Path, with_heading: bool = True) -> Index:
    """The index in this directory, or a new empty one if there is none yet."""
    if (index_dir / "manifest.json").exists():
        return load(index_dir)
    return build([], index_dir, with_heading=with_heading)
