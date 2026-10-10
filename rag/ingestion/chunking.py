"""Splitting documents into chunks for retrieval.

Three strategies, compared in the evaluation:
    fixed       a sliding window of tokens (the baseline)
    structure   sections by markdown headings, split further only when too long (used)
    semantic    cuts where the topic changes between consecutive sentences

Sizes are counted in tokens of the embedding model's tokenizer, because its 512-token
window is the hard limit: anything longer is truncated silently when embedded.
"""
from __future__ import annotations

import json
import re
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
from pathlib import Path

from rag.embedding import MAX_MODEL_TOKENS, embed_queries, get_tokenizer
from rag.ingestion.loaders import page_at, sha

MAX_TOKENS = 400      # target chunk size
OVERLAP_TOKENS = 60   # 15% of MAX_TOKENS
MIN_TOKENS = 100      # smaller chunks carry too little to stand alone

# Leave room for the "passage: " prefix and the special tokens the encoder adds.
assert MAX_TOKENS + 16 <= MAX_MODEL_TOKENS, "chunks could be truncated when embedded"


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    chunk_index: int
    strategy: str
    text: str
    heading_path: str    # "Capitolul 2. Credite > Articolul 3", "" if there are no headings
    start_char: int      # offsets into Document.text
    end_char: int
    page: int | None     # 1-based, PDFs only
    num_chars: int
    num_tokens: int


# --------------------------------------------------------------------- tokens


class TokenIndex:
    """A document's token boundaries, tokenized once.

    Answers "how many tokens lie between these two character positions?" with a
    binary search, instead of re-tokenizing every candidate span.
    """

    def __init__(self, text: str, tokenizer=None):
        tok = tokenizer or get_tokenizer()
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        self.offsets: list[tuple[int, int]] = [tuple(o) for o in enc["offset_mapping"]]
        self._starts = [start for start, _ in self.offsets]

    def __len__(self) -> int:
        return len(self.offsets)

    def count(self, start: int, end: int) -> int:
        """Number of tokens starting inside [start, end)."""
        return bisect_left(self._starts, end) - bisect_left(self._starts, start)


# ------------------------------------------------------------------ splitting

# A period after these does not end a sentence ("art. 5", "alin. (2)", "e.g.").
ABBREVIATIONS = {
    "art.", "alin.", "lit.", "nr.", "pct.", "cap.", "par.", "pag.", "p.", "pp.",
    "dl.", "dna.", "prof.", "dr.", "ing.", "ec.", "etc.", "ex.", "cca.", "vs.",
    "resp.", "urm.", "n.b.", "ș.a.", "s.a.", "î.e.n.",
    "mr.", "mrs.", "ms.", "sr.", "jr.", "st.", "no.", "fig.", "eq.", "cf.",
    "e.g.", "i.e.", "al.", "approx.", "inc.", "ltd.", "co.", "dept.", "univ.",
    "vol.", "ch.",
}

SENT_END = re.compile(r"[.!?…]+[\"'”’)\]]*(\s+)")
LAST_WORD = re.compile(r"\S+$")
INITIAL = re.compile(r"^\w\.$")            # "J." in "J. R. R. Tolkien"
LIST_OR_ROW = re.compile(r"^(\||[-*+]\s|\d+[.)]\s|#{1,6}\s)")


def _trim(text: str, start: int, end: int) -> tuple[int, int] | None:
    """[start, end) without surrounding whitespace; None if nothing is left."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def _blocks(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Blank-line separated blocks (paragraphs) inside [start, end)."""
    spans, pos = [], start
    for m in re.finditer(r"\n[ \t]*\n", text[start:end]):
        spans.append((pos, start + m.start()))
        pos = start + m.end()
    spans.append((pos, end))
    return [t for t in (_trim(text, a, b) for a, b in spans) if t]


def _is_boundary(text: str, punct_end: int, next_start: int) -> bool:
    word = LAST_WORD.search(text[:punct_end])
    if word and (word.group().lower() in ABBREVIATIONS or INITIAL.match(word.group())):
        return False
    # A new sentence does not start in lowercase.
    return not (next_start < len(text) and text[next_start].islower())


def split_sentences(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Sentence spans inside [start, end).

    A regex and an abbreviation list, since common sentence splitters (NLTK, pysbd)
    have no Romanian model. Lists, tables and heading lines give one unit per line.
    """
    spans = []
    for b0, b1 in _blocks(text, start, end):
        lines = text[b0:b1].split("\n")
        if sum(1 for line in lines if LIST_OR_ROW.match(line.strip())) * 2 >= len(lines):
            pos = b0
            for line in lines:
                if line.strip():
                    a = pos + len(line) - len(line.lstrip())
                    spans.append((a, a + len(line.strip())))
                pos += len(line) + 1
            continue
        block, last = text[b0:b1], 0
        for m in SENT_END.finditer(block):
            cut, nxt = m.start(1), m.end()
            if not _is_boundary(block, cut, nxt):
                continue
            spans.append((b0 + last, b0 + cut))
            last = nxt
        if last < len(block):
            spans.append((b0 + last, b1))
    return [(a, b) for a, b in spans if text[a:b].strip()]


def find_sections(text: str) -> list[tuple[int, int, str]]:
    """Split markdown by its headings into (start, end, heading_path) sections.

    The loaders turn every format into markdown, so this one parser covers them all.
    The heading line stays inside its section, so a chunk opens with its own title.
    """
    lines = text.split("\n")
    line_start, pos = [], 0
    for line in lines:
        line_start.append(pos)
        pos += len(line) + 1

    heads, in_fence = [], False
    for i, line in enumerate(lines):
        if line.strip().startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:  # "#" inside a code block is not a heading
            continue
        m = re.match(r"(#{1,6})\s+(\S.*?)\s*$", line)
        if m:
            heads.append((i, len(m.group(1)), m.group(2)))

    sections, stack = [], []  # stack: (level, title)
    if not heads or heads[0][0] > 0:
        first = line_start[heads[0][0]] if heads else len(text)
        sections.append((0, first, ""))  # text before the first heading
    for idx, (line_no, level, title) in enumerate(heads):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        end = line_start[heads[idx + 1][0]] if idx + 1 < len(heads) else len(text)
        sections.append((line_start[line_no], end, " > ".join(t for _, t in stack)))
    return [(a, b, h) for a, b, h in sections if text[a:b].strip()]


def _parent(path: str) -> str:
    return path.rsplit(" > ", 1)[0] if " > " in path else ""


def _is_heading_only(text: str, start: int, end: int) -> bool:
    """True if the section is only its heading line, with no body."""
    body = text[start:end]
    return bool(re.match(r"#{1,6}\s", body)) and not body.partition("\n")[2].strip()


# ------------------------------------------------------------------- strategies


def _pack(units: list[tuple[int, int]], tokens: TokenIndex,
          max_tokens: int, overlap: int) -> list[tuple[int, int]]:
    """Group consecutive units (sentences, paragraphs) into spans of at most max_tokens.

    The last units of each group, up to `overlap` tokens, are repeated at the start of
    the next one, so a fact sitting on a boundary appears whole in at least one chunk.
    """
    groups, current = [], []
    for unit in units:
        size = tokens.count(unit[0], unit[1])
        used = tokens.count(current[0][0], current[-1][1]) if current else 0
        if current and used + size > max_tokens:
            groups.append((current[0][0], current[-1][1]))
            carry, carried = [], 0
            # Never carry the whole group, or a small group would be repeated verbatim.
            for prev in reversed(current[1:]):
                cost = tokens.count(prev[0], prev[1])
                if carried + cost > overlap:
                    break
                carry.insert(0, prev)
                carried += cost
            current = carry
        current.append(unit)
    if current:
        groups.append((current[0][0], current[-1][1]))
    return groups


def chunk_fixed(text: str, tokens: TokenIndex, max_tokens: int = MAX_TOKENS,
                overlap: int = OVERLAP_TOKENS, **_) -> list[tuple[int, int]]:
    """Baseline: a sliding window of `max_tokens`, ignoring the document's structure."""
    step = max(1, max_tokens - overlap)
    spans, i = [], 0
    while i < len(tokens):
        window = tokens.offsets[i:i + max_tokens]
        spans.append((window[0][0], window[-1][1]))
        if i + max_tokens >= len(tokens):
            break
        i += step
    # A short last window is merged into the previous one.
    if len(spans) > 1 and tokens.count(*spans[-1]) < min(MIN_TOKENS, max_tokens // 2):
        spans[-2] = (spans[-2][0], spans[-1][1])
        spans.pop()
    return spans


def chunk_structure(text: str, tokens: TokenIndex, max_tokens: int = MAX_TOKENS,
                    overlap: int = OVERLAP_TOKENS,
                    min_tokens: int = MIN_TOKENS, **_) -> list[tuple[int, int, str]]:
    """One chunk per section, following the document's headings.

    Small sections are merged with a neighbour; sections that are too long are
    split into paragraphs, then sentences, and as a last resort token windows.
    Returns (start, end, heading_path) spans.
    """
    # Entries are (start, end, label, last_path): `label` is the heading_path the chunk
    # will carry, `last_path` the real heading of the last section merged into it.
    merged: list[tuple[int, int, str, str]] = []
    for start, end, path in find_sections(text):
        if merged:
            prev_start, prev_end, prev_label, prev_path = merged[-1]
            # A small section that opens the next one is merged forward: a heading with no
            # body, a chapter title above its first article, or text before the first
            # heading. On its own it would be a chunk that holds no fact.
            opens_next = (not prev_path or path.startswith(prev_path + " > ")
                          or _is_heading_only(text, prev_start, prev_end))
            if opens_next and tokens.count(prev_start, prev_end) < min_tokens:
                merged[-1] = (prev_start, end, path, path)
                continue
            if tokens.count(prev_start, end) <= max_tokens:
                # Two short siblings under the same heading are merged and labelled with
                # that heading. Requiring a real shared parent keeps unrelated top-level
                # chapters apart.
                parent = _parent(path)
                if (parent and parent == _parent(prev_path)
                        and not _is_heading_only(text, start, end)
                        and tokens.count(prev_start, prev_end) < min_tokens):
                    merged[-1] = (prev_start, end, parent, path)
                    continue
        merged.append((start, end, path, path))

    out = []
    for start, end, path, _ in merged:
        if tokens.count(start, end) <= max_tokens:
            out.append((start, end, path))
            continue
        units = _blocks(text, start, end)
        if any(tokens.count(a, b) > max_tokens for a, b in units):
            units = split_sentences(text, start, end)
        for a, b in _pack(units, tokens, max_tokens, overlap):
            if tokens.count(a, b) <= max_tokens:
                out.append((a, b, path))
            else:  # a single sentence longer than the window
                sub = TokenIndex(text[a:b])
                out += [(a + s, a + e, path) for s, e in chunk_fixed(text[a:b], sub,
                                                                     max_tokens, overlap)]
    return out


def heading_at(sections, section_starts: list[int], pos: int) -> str:
    """heading_path of the section that contains a character position."""
    index = bisect_right(section_starts, pos) - 1
    return sections[index][2] if index >= 0 else ""


# ------------------------------------------------------------------- semantic

SEMANTIC_PERCENTILE = 90  # cut at the most dissimilar 10% of sentence boundaries
SEMANTIC_BUFFER = 1       # sentences of context on each side when embedding

def chunk_semantic(text: str, tokens: TokenIndex, max_tokens: int = MAX_TOKENS,
                   overlap: int = OVERLAP_TOKENS, min_tokens: int = MIN_TOKENS,
                   percentile: int = SEMANTIC_PERCENTILE,
                   buffer_size: int = SEMANTIC_BUFFER, **_) -> list[tuple[int, int]]:
    """Cut where consecutive sentences stop being about the same thing.

    Each sentence boundary is scored by the cosine distance between the text that
    ends there and the text that starts there (each with `buffer_size` neighbouring
    sentences, so short sentences get enough context). The windows on the two sides
    do not overlap, otherwise cuts land one sentence late. The cut threshold is a
    percentile of the document's own distances, since absolute cosine values vary
    by model and by subject.
    """
    import numpy as np

    sentences = split_sentences(text, 0, len(text))
    if len(sentences) < 3:  # too few boundaries to measure
        return chunk_fixed(text, tokens, max_tokens, overlap)

    def window(lo: int, hi: int) -> str:
        return " ".join(text[a:b] for a, b in sentences[max(0, lo):hi])

    # For the boundary before sentence i: the text ending at i-1, and the text starting at i.
    before = [window(i - buffer_size, i + 1) for i in range(len(sentences) - 1)]
    after = [window(i, i + buffer_size + 1) for i in range(1, len(sentences))]

    vectors = embed_queries(before + after, batch_size=16)
    left, right = vectors[:len(before)], vectors[len(before):]
    distances = 1.0 - (left * right).sum(axis=1)
    threshold = float(np.percentile(distances, percentile))

    cuts = {i + 1 for i, d in enumerate(distances) if d >= threshold}
    groups, current = [], []
    for i, span in enumerate(sentences):
        if i in cuts and current:
            groups.append(current)
            current = []
        current.append(span)
    if current:
        groups.append(current)

    # Topic boundaries ignore size, so the size limits are applied afterwards.
    merged = merge_small([(group[0][0], group[-1][1]) for group in groups], tokens, min_tokens)

    out = []
    for start, end in merged:
        if tokens.count(start, end) <= max_tokens:
            out.append((start, end))
        else:
            out += _pack(split_sentences(text, start, end), tokens, max_tokens, overlap)
    return out


def merge_small(spans: list[tuple[int, int]], tokens: TokenIndex,
                min_tokens: int = MIN_TOKENS) -> list[tuple[int, int]]:
    """Merge every span under `min_tokens` into the next one (a small last span, into
    the previous one). Oversized results are split afterwards by the caller."""
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and tokens.count(*merged[-1]) < min_tokens:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    if len(merged) > 1 and tokens.count(*merged[-1]) < min_tokens:
        merged[-2:] = [(merged[-2][0], merged[-1][1])]
    return merged


STRATEGIES = {"fixed": chunk_fixed, "structure": chunk_structure,
              "semantic": chunk_semantic}


# ---------------------------------------------------------------------- driver


def chunk_document(doc: dict, strategy: str, **kwargs) -> list[Chunk]:
    """Split one loaded document with the given strategy."""
    text = doc["text"]
    tokens = TokenIndex(text)
    spans = STRATEGIES[strategy](text, tokens, **kwargs)

    page_starts = doc.get("metadata", {}).get("page_starts")
    # Strategies that do not cut on headings still get a heading_path, from where the
    # chunk starts, so the strategies differ only in where they cut.
    sections = find_sections(text)
    section_starts = [start for start, _, _ in sections]

    chunks = []
    for index, span in enumerate(spans):
        start, end, path = span if len(span) == 3 else (*span, heading_at(
            sections, section_starts, span[0]))
        body = text[start:end].strip()
        if not body:
            continue
        chunks.append(Chunk(
            chunk_id=sha(f"{doc['doc_id']}|{strategy}|{start}|{end}"),
            doc_id=doc["doc_id"],
            source=doc["source"],
            chunk_index=index,
            strategy=strategy,
            text=body,
            heading_path=path,
            start_char=start,
            end_char=end,
            page=page_at(page_starts, start) if page_starts else None,
            num_chars=len(body),
            num_tokens=tokens.count(start, end),
        ))
    return chunks


def read_documents(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(chunks: list[Chunk], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
