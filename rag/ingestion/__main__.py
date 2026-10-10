"""Rebuild the index from the documents in data/raw/, run from the project root:

    python -m rag.ingestion [--strategy structure]

Three steps, each writing its output:
    load     data/raw/**            -> data/processed/documents.jsonl
    chunk    documents.jsonl        -> data/processed/chunks.<strategy>.jsonl
    index    chunks.<strategy>.jsonl -> data/index/<strategy>/   (replaced)

Documents uploaded through the web app are indexed directly and need none of this.
"""
import argparse
import sys

from rag.config import DATA_DIR
from rag.ingestion import chunking, indexing, loaders

sys.stdout.reconfigure(encoding="utf-8")
parser = argparse.ArgumentParser(prog="python -m rag.ingestion")
parser.add_argument("--strategy", default="structure", choices=list(chunking.STRATEGIES))
parser.add_argument("--no-heading", action="store_true",
                    help="do not prepend the heading path to the indexed text")
args = parser.parse_args()

documents_file = DATA_DIR / "processed" / "documents.jsonl"
chunks_file = DATA_DIR / "processed" / f"chunks.{args.strategy}.jsonl"
index_dir = DATA_DIR / "index" / args.strategy

documents = loaders.load_directory(DATA_DIR / "raw")
loaders.save_jsonl(documents, documents_file)
for d in documents:
    print(f"{d.metadata['format']:9} {d.metadata['num_chars']:>8} chars  {d.source}")
print(f"\n{len(documents)} documents -> {documents_file}\n")

chunks = []
for document in chunking.read_documents(documents_file):
    made = chunking.chunk_document(document, args.strategy)
    chunks += made
    sizes = [c.num_tokens for c in made]
    print(f"{document['source']:60} {len(made):>3} chunks  "
          f"min/avg/max tokens {min(sizes):>4}/{sum(sizes) // len(sizes):>4}/{max(sizes):>4}")
chunking.save_jsonl(chunks, chunks_file)
print(f"\n{len(chunks)} chunks ({args.strategy}) -> {chunks_file}\n")

with indexing.build(indexing.read_chunks(chunks_file), index_dir,
                    with_heading=not args.no_heading) as index:
    issues = indexing.check_in_sync(index)
    for issue in issues:
        print(f"[error] {issue}")
    print(f"\n{len(index.chunks)} chunks indexed -> {index_dir}"
          f"{'  (in sync)' if not issues else '  (OUT OF SYNC)'}")
sys.exit(1 if issues else 0)
