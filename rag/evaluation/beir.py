"""Retrieval on a public benchmark: SciFact, from BEIR.

    python -m rag.evaluation beir [--dataset scifact] [--split test]

The UBB sets are small and written by the author, so retrieval is also measured on one
public set with relevance judged by others. SciFact (5,183 scientific abstracts, 300 test
claims) was chosen on 2026-10-09, before this run, and is reported whatever the outcome.
It measures retrieval only: SciFact has no reference answers for the generation step.
The configurations below were fixed before the run as well.

The corpus goes through the app's own code: each abstract becomes a markdown document
with its title as heading, is cut by the "structure" chunker, embedded with e5-base and
indexed in Qdrant + BM25 in data/index/beir-<dataset>. A document's rank is the rank of
its first chunk in the result list.

Every step is saved, so a stopped run resumes where it stopped:
    data/beir/<dataset>/            the downloaded set (corpus, queries, qrels)
    data/beir/<dataset>/vectors/    embeddings, one file per slice of chunks
    data/beir/<dataset>/runs/       one line per answered query, per configuration
Results: data/eval/results/beir-<dataset>-<split>.json
"""
from __future__ import annotations

import io
import json
import math
import shutil
import time
import urllib.request
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rag.config import DATA_DIR
from rag.evaluation.gold import read_jsonl
from rag.ingestion.chunking import chunk_document
from rag.ingestion.indexing import (Index, create_collection, embed_chunks, load, save,
                                    upsert)
from rag.retrieval.pipeline import CANDIDATES, retrieve
from rag.retrieval.search import search

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
BEIR = DATA_DIR / "beir"
SLICE = 1000          # chunks embedded per saved file
DEPTH = 100           # results per list for the single-method baselines
BOOTSTRAP = 10_000    # resamples for the confidence intervals
KS = (5, 10)

# name -> (description, how to run one query). Fixed before the run.
CONFIGS = {
    "bm25": "BM25 alone, top 100",
    "dense": "e5-base alone, top 100",
    "hybrid": "dense + BM25, 100 per list, RRF",
    "app-no-rerank": "the app's candidates (5 per list, RRF), fused order",
    "app": "the app's configuration: those candidates reranked by the cross-encoder",
}


def run_query(index: Index, name: str, query: str) -> list:
    if name in ("bm25", "dense"):
        return search(index, query, limit=DEPTH, mode="sparse" if name == "bm25" else "dense")
    if name == "hybrid":
        return search(index, query, limit=DEPTH, dense_k=DEPTH, sparse_k=DEPTH,
                      per_language=True)
    # The app's retrieve(), asked for all its candidates instead of the 5 the model reads.
    return retrieve(index, query, top_n=CANDIDATES, use_reranker=name == "app")


# ------------------------------------------------------------------ the data


def download(dataset: str) -> Path:
    folder = BEIR / dataset
    if (folder / "corpus.jsonl").exists():
        return folder
    print(f"downloading {dataset} from BEIR ...")
    data = urllib.request.urlopen(BEIR_URL.format(dataset), timeout=120).read()
    tmp = BEIR / f"{dataset}.partial"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for member in z.namelist():
            # Only the files we read, each written to a path we choose.
            name = member.split("/", 1)[-1]
            if name in ("corpus.jsonl", "queries.jsonl") or (
                    name.startswith("qrels/") and name.endswith(".tsv")):
                out = tmp / name
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(z.read(member))
    tmp.rename(folder)
    return folder


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    """query id -> {doc id: relevance}, from a BEIR tsv with a header line."""
    qrels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as f:
        next(f)
        for line in f:
            qid, did, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                qrels.setdefault(qid, {})[did] = int(score)
    return qrels


def as_document(dataset: str, doc: dict) -> dict:
    title, text = doc.get("title", "").strip(), doc.get("text", "").strip()
    return {"doc_id": f"{dataset}-{doc['_id']}", "source": f"{dataset}/{doc['_id']}",
            "text": f"# {title}\n\n{text}" if title else text}


# ------------------------------------------------------------------ the index


def build_index(dataset: str, folder: Path) -> Path:
    index_dir = DATA_DIR / "index" / f"beir-{dataset}"
    if (index_dir / "manifest.json").exists():
        return index_dir
    chunks = [asdict(c) for doc in read_jsonl(folder / "corpus.jsonl")
              for c in chunk_document(as_document(dataset, doc), "structure")]
    print(f"{len(chunks)} chunks")

    vectors_dir = folder / "vectors"
    vectors_dir.mkdir(exist_ok=True)
    parts = []
    for start in range(0, len(chunks), SLICE):
        path = vectors_dir / f"{start:06d}.npy"
        if not path.exists():
            started = time.time()
            np.save(path, embed_chunks(chunks[start:start + SLICE]))
            print(f"embedded {min(start + SLICE, len(chunks))}/{len(chunks)} "
                  f"({time.time() - started:.0f} s)", flush=True)
        parts.append(np.load(path))
    vectors = np.concatenate(parts)

    from qdrant_client import QdrantClient

    shutil.rmtree(index_dir, ignore_errors=True)
    index_dir.mkdir(parents=True)
    client = QdrantClient(path=str(index_dir / "qdrant"))
    create_collection(client)
    for start in range(0, len(chunks), SLICE):
        upsert(client, chunks[start:start + SLICE], vectors[start:start + SLICE])
    index = Index(chunks=chunks, client=client)
    save(index, index_dir)   # writes the manifest last
    index.close()
    return index_dir


# ------------------------------------------------------------------ metrics


def doc_ranking(hits) -> list[str]:
    """Document ids in the order their first chunk appears."""
    seen: dict[str, None] = {}
    for hit in hits:
        seen.setdefault(hit.source.split("/", 1)[1], None)
    return list(seen)


def ndcg(ranking: list[str], relevant: dict[str, int], k: int = 10) -> float:
    """nDCG@k with the relevance as gain, as trec_eval (and so BEIR) computes it."""
    dcg = sum(relevant.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranking[:k]))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    return dcg / sum(g / math.log2(i + 2) for i, g in enumerate(ideal))


def scores(ranking: list[str], relevant: dict[str, int]) -> dict[str, float]:
    out = {"ndcg@10": ndcg(ranking, relevant)}
    for k in KS:
        out[f"recall@{k}"] = len(set(ranking[:k]) & set(relevant)) / len(relevant)
    first = next((i for i, d in enumerate(ranking[:10], 1) if d in relevant), None)
    out["mrr@10"] = 1 / first if first else 0.0
    return out


def bootstrap_ci(values: np.ndarray, seed: int = 0) -> list[float]:
    """95% percentile bootstrap interval of the mean, resampling queries."""
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (BOOTSTRAP, len(values)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


# ------------------------------------------------------------------ the run


def run(dataset: str = "scifact", split: str = "test") -> dict:
    folder = download(dataset)
    queries = {q["_id"]: q["text"] for q in read_jsonl(folder / "queries.jsonl")}
    qrels = read_qrels(folder / "qrels" / f"{split}.tsv")
    qids = sorted(qrels, key=lambda q: (len(q), q))
    index_dir = build_index(dataset, folder)

    runs_dir = folder / "runs" / split
    runs_dir.mkdir(parents=True, exist_ok=True)
    with load(index_dir) as index:
        languages = sorted(set(index.language.values()))
        print(f"index: {len(index.chunks)} chunks, document languages {languages}")
        for name in CONFIGS:
            path = runs_dir / f"{name}.jsonl"
            done = {r["qid"] for r in read_jsonl(path)} if path.exists() else set()
            todo = [q for q in qids if q not in done]
            if not todo:
                continue
            started = time.time()
            with path.open("a", encoding="utf-8") as f:
                for n, qid in enumerate(todo, 1):
                    ranking = doc_ranking(run_query(index, name, queries[qid]))
                    f.write(json.dumps({"qid": qid, "ranking": ranking[:DEPTH]}) + "\n")
                    f.flush()
                    if n % 50 == 0 or n == len(todo):
                        print(f"{name}: {len(done) + n}/{len(qids)} "
                              f"({time.time() - started:.0f} s)", flush=True)

    result = {"dataset": dataset, "split": split, "queries": len(qids),
              "chunks": len(read_jsonl(index_dir / "chunks.jsonl")),
              "documents": len(read_jsonl(folder / "corpus.jsonl")),
              "bootstrap": BOOTSTRAP, "configs": {}, "differences": {}, "per_query": {}}
    per_metric: dict[str, dict[str, np.ndarray]] = {}
    for name, description in CONFIGS.items():
        rankings = {r["qid"]: r["ranking"] for r in read_jsonl(runs_dir / f"{name}.jsonl")}
        rows = [scores(rankings[q], qrels[q]) for q in qids]
        per_metric[name] = {m: np.array([r[m] for r in rows]) for m in rows[0]}
        result["configs"][name] = {
            "description": description,
            "returned_docs_median": float(np.median([len(rankings[q]) for q in qids])),
            **{m: {"value": float(v.mean()), "ci95": bootstrap_ci(v)}
               for m, v in per_metric[name].items()}}
        result["per_query"][name] = {q: {"ndcg@10": round(r["ndcg@10"], 4),
                                         "top10": rankings[q][:10]} for q, r in zip(qids, rows)}
    # Paired differences, on the same queries: what each step adds.
    for a, b in [("hybrid", "bm25"), ("hybrid", "dense"), ("app", "app-no-rerank"),
                 ("app", "bm25"), ("app", "hybrid")]:
        diff = per_metric[a]["ndcg@10"] - per_metric[b]["ndcg@10"]
        result["differences"][f"{a} - {b}"] = {"ndcg@10": float(diff.mean()),
                                               "ci95": bootstrap_ci(diff)}
    return result


def print_report(result: dict) -> None:
    print(f"\n{result['dataset']} / {result['split']}: {result['queries']} queries, "
          f"{result['documents']} documents, {result['chunks']} chunks\n")
    print(f"{'':16}{'nDCG@10':>22}{'R@5':>8}{'R@10':>8}{'MRR@10':>8}{'docs':>6}")
    for name, c in result["configs"].items():
        lo, hi = c["ndcg@10"]["ci95"]
        print(f"{name:16}{c['ndcg@10']['value']:>8.3f} [{lo:.3f}–{hi:.3f}]"
              f"{c['recall@5']['value']:>8.3f}{c['recall@10']['value']:>8.3f}"
              f"{c['mrr@10']['value']:>8.3f}{c['returned_docs_median']:>6.0f}")
    print("\npaired differences in nDCG@10:")
    for name, d in result["differences"].items():
        lo, hi = d["ci95"]
        print(f"  {name:24}{d['ndcg@10']:+.3f} [{lo:+.3f}, {hi:+.3f}]")
