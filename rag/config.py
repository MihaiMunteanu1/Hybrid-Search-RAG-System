"""Machine-specific settings, read from the environment or from a `.env` file.

Paths are relative to the working directory, so commands are run from the project root.
See `.env.example` for what each variable means.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Loaded before any Hugging Face import, so HF_HUB_OFFLINE from .env takes effect.
load_dotenv(PROJECT_ROOT / ".env")


def _int_or_none(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    return int(value) if value else None


DATA_DIR = Path(os.environ.get("RAG_DATA", "data"))

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")

LLM_MODEL = Path(os.environ.get("RAG_MODEL", "models/Qwen3.5-4B-Q4_K_M.gguf"))
# A llama-server already running elsewhere (e.g. its own Docker container). When set, the
# app connects to it instead of starting one, and RAG_MODEL / LLAMA_SERVER are not used.
LLM_URL = os.environ.get("RAG_LLM_URL", "").strip() or None
LLAMA_SERVER = os.environ.get("LLAMA_SERVER", "llama-server")
LLAMA_THREADS = _int_or_none("LLAMA_THREADS")
LLAMA_THREADS_BATCH = _int_or_none("LLAMA_THREADS_BATCH")

API_HOST = os.environ.get("RAG_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("RAG_PORT", "8000"))
