"""Start the web app, run from the project root:

    python -m rag.api [--host 127.0.0.1] [--port 8000] [--no-llm]

--no-llm skips the language model: folders, upload and search only.
"""
import argparse
import os
import sys

import uvicorn

from rag.config import API_HOST, API_PORT

sys.stdout.reconfigure(encoding="utf-8")
parser = argparse.ArgumentParser(prog="python -m rag.api")
parser.add_argument("--host", default=API_HOST)
parser.add_argument("--port", type=int, default=API_PORT)
parser.add_argument("--no-llm", action="store_true",
                    help="skip the language model: folders, upload and search only")
args = parser.parse_args()

if args.no_llm:
    os.environ["RAG_NO_LLM"] = "1"
uvicorn.run("rag.api.app:app", host=args.host, port=args.port)
