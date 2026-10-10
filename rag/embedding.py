"""The embedding model (multilingual-e5) and its tokenizer.

All text is embedded through this module because e5 expects a prefix: "query: " on
questions and "passage: " on indexed chunks. A wrong prefix raises no error, it only
lowers retrieval quality, so only the prefixed functions are exposed.
"""
from __future__ import annotations

from rag.config import EMBEDDING_MODEL

# Vector size of each supported model, so the Qdrant collection can be created
# without loading the model.
E5_SIZES = {
    "intfloat/multilingual-e5-small": 384,
    "intfloat/multilingual-e5-base": 768,
    "intfloat/multilingual-e5-large": 1024,
}

if EMBEDDING_MODEL not in E5_SIZES:
    raise ValueError(f"unknown embedding model {EMBEDDING_MODEL!r}; "
                     f"known: {', '.join(E5_SIZES)}")
EMBEDDING_DIM = E5_SIZES[EMBEDDING_MODEL]
MAX_MODEL_TOKENS = 512    # longer inputs are truncated silently by the encoder

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

_TOKENIZER = None
_EMBEDDER = None


def get_tokenizer(name: str = EMBEDDING_MODEL):
    """The model's tokenizer, loaded once."""
    global _TOKENIZER
    if _TOKENIZER is None:
        import transformers
        transformers.logging.set_verbosity_error()  # silence the ">512 tokens" notice
        _TOKENIZER = transformers.AutoTokenizer.from_pretrained(name)
    return _TOKENIZER


def get_embedder(name: str = EMBEDDING_MODEL):
    """The embedding model, loaded once, on CPU."""
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer(name, device="cpu")
    return _EMBEDDER


def _encode(texts: list[str], batch_size: int, show_progress: bool):
    # Unit-length vectors: a dot product is then the cosine similarity.
    return get_embedder().encode(texts, normalize_embeddings=True,
                                 batch_size=batch_size, show_progress_bar=show_progress)


def embed_passages(texts: list[str], batch_size: int = 32, show_progress: bool = False):
    """Vectors for chunks being indexed."""
    return _encode([PASSAGE_PREFIX + t for t in texts], batch_size, show_progress)


def embed_queries(texts: list[str], batch_size: int = 32, show_progress: bool = False):
    """Vectors for questions, and for comparing two passages with each other.

    The e5 model card asks for "query: " on both sides of a symmetric comparison,
    which is why semantic chunking uses this function.
    """
    return _encode([QUERY_PREFIX + t for t in texts], batch_size, show_progress)
