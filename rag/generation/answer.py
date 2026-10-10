"""Question in, grounded answer with citations out: retrieve, prompt, generate, parse.

`answer_stream()` is the pipeline the web app serves; `answer()` runs the same code
without streaming, for the evaluation.
"""
from __future__ import annotations

import time
from typing import Iterator

from rag.generation.prompt import (CITE_REQUEST, NOT_FOUND, Answer, build_messages,
                                   not_found, parse_answer)
from rag.ingestion.indexing import Index
from rag.retrieval.pipeline import PER_LANGUAGE, retrieve

TOP_N = 5   # sources shown to the model


def answer(index: Index, llm, question: str, *, top_n: int = TOP_N, mode: str = "hybrid",
           per_language: bool = PER_LANGUAGE, use_reranker: bool = True,
           sources: list[str] | None = None,
           ask_for_citations: bool = True) -> Answer:
    """The final Answer of answer_stream(), with nothing streamed.

    `llm` is anything with `chat(messages) -> Reply`. With `ask_for_citations=False`
    a reply without citations is a refusal, with no follow-up request.
    """
    result = None
    for event in answer_stream(index, llm, question, top_n=top_n, mode=mode,
                               per_language=per_language, use_reranker=use_reranker,
                               sources=sources,
                               ask_for_citations=ask_for_citations, stream=False):
        if event["event"] == "answer":
            result = event["answer"]
    return result


def answer_stream(index: Index, llm, question: str, *, top_n: int = TOP_N,
                  mode: str = "hybrid", per_language: bool = PER_LANGUAGE,
                  use_reranker: bool = True,
                  sources: list[str] | None = None, ask_for_citations: bool = True,
                  stream: bool = True) -> Iterator[dict]:
    """The answer pipeline, as events in the order the work finishes:

        {"event": "sources", "sources": [Hit, ...], "language": "ro"}
        {"event": "delta", "text": "..."}        zero or more, while the model writes
        {"event": "retry"}                       only if the reply had no citations
        {"event": "answer", "answer": Answer}    parsed: citations, refusal, timings

    The sources are ready in about two seconds and the answer takes about a minute on a
    CPU, so a client can show both early. The final answer can differ from the streamed
    text (a refusal, or a rewrite with citations), so it replaces it.
    """
    start = time.perf_counter()
    hits = retrieve(index, question, top_n=top_n, use_reranker=use_reranker, mode=mode,
                    per_language=per_language, sources=sources)
    retrieval_seconds = time.perf_counter() - start
    messages, language = build_messages(question, hits)
    yield {"event": "sources", "sources": hits, "language": language}

    if not hits:
        # Nothing to ground an answer in, so the model is not asked at all.
        result = not_found(question, hits, language, "no_sources")
        result.timings = {"retrieval": retrieval_seconds, "generation": 0.0}
        yield {"event": "answer", "answer": result}
        return

    if stream:
        deltas = llm.stream(messages)
        while True:
            try:
                yield {"event": "delta", "text": next(deltas)}
            except StopIteration as done:
                reply = done.value   # llm.stream() returns the finished Reply
                break
    else:
        reply = llm.chat(messages)

    result = parse_answer(question, reply.text, hits, language)
    timings = {"retrieval": retrieval_seconds, "generation": reply.seconds,
               "prompt_tokens": reply.prompt_tokens,
               "completion_tokens": reply.completion_tokens}

    if ask_for_citations and result.refusal == "no_citations":
        yield {"event": "retry"}
        # The same conversation continued: llama-server reuses the sources from its
        # cache, so this request only costs the new turn.
        follow_up = messages + [
            {"role": "assistant", "content": reply.text},
            {"role": "user", "content": CITE_REQUEST.format(not_found=NOT_FOUND)}]
        second = llm.chat(follow_up)
        result = parse_answer(question, second.text, hits, language)
        result.first_raw = reply.text
        timings["generation"] += second.seconds
        timings["citation_retry"] = {"seconds": second.seconds,
                                     "prompt_tokens": second.prompt_tokens,
                                     "cached_tokens": second.cached_tokens}
    result.timings = timings
    yield {"event": "answer", "answer": result}
