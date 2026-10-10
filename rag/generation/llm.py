"""The language model, served by llama.cpp's `llama-server` and called over HTTP.

The model runs in its own process, so the same code works whether the server is
started here (`start_server`) or runs elsewhere, for example on a GPU machine.
llama-server speaks the OpenAI chat format (/v1/chat/completions); nothing leaves
this machine.

    with start_server() as llm:
        reply = llm.chat([{"role": "user", "content": "Salut"}])
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import httpx

from rag.config import DATA_DIR, LLAMA_SERVER, LLAMA_THREADS, LLAMA_THREADS_BATCH, LLM_MODEL

DEFAULT_MODEL = LLM_MODEL
MODEL_URL = "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf"
LOG_DIR = DATA_DIR / "logs"

CONTEXT_SIZE = 8192   # room for up to 10 chunks of ~400 tokens, the instructions and the answer
MAX_TOKENS = 512      # a grounded answer is a paragraph
# Prompt tokens read per step. llama-server notices a cancelled request only between
# steps: with its default of 2048 a request cancelled while the sources were being read
# kept the server busy ~21 s more, with 256 ~2 s. Reading speed was the same
# (55.6 tokens/s for both, alternating runs on the laptop CPU).
BATCH_SIZE = 256
TEMPERATURE = 0.0     # greedy decoding, so evaluation runs are reproducible


@dataclass
class Reply:
    text: str
    prompt_tokens: int
    completion_tokens: int
    prompt_per_second: float     # how fast the prompt (the sources) was read
    tokens_per_second: float     # how fast the answer was written
    seconds: float               # wall-clock time of the whole request
    cached_tokens: int = 0       # prompt tokens reused from the server's cache


class Cancelled(Exception):
    """The request was cancelled with LLM.cancel() while it was running."""


class LLM:
    """A client for one running llama-server."""

    def __init__(self, base_url: str, timeout: float = 600, http: httpx.Client | None = None):
        # Long timeout: on a CPU, reading the sources alone takes tens of seconds.
        self.base_url = base_url.rstrip("/")
        self.http = http or httpx.Client(timeout=timeout)
        self._active: set[httpx.Response] = set()
        self._cancelled: set[int] = set()
        self._lock = threading.Lock()

    @contextmanager
    def _request(self, request: dict) -> Iterator[httpx.Response]:
        """A chat request whose response cancel() can close from another thread."""
        with self.http.stream("POST", f"{self.base_url}/v1/chat/completions",
                              json=request) as response:
            with self._lock:
                self._active.add(response)
            try:
                response.raise_for_status()
                yield response
            except Exception as error:
                with self._lock:
                    cancelled = id(response) in self._cancelled
                if cancelled:
                    raise Cancelled() from error
                raise
            finally:
                with self._lock:
                    self._active.discard(response)
                    self._cancelled.discard(id(response))

    def cancel(self) -> None:
        """Stop every request in progress, by closing its connection.

        While the model reads the sources (tens of seconds on a CPU) the request waits
        for its first token; closing the connection ends the wait at once, and
        llama-server stops the work when the client goes away. The thread that made
        the request gets Cancelled.
        """
        with self._lock:
            active = list(self._active)
            self._cancelled.update(id(response) for response in active)
        for response in active:
            try:
                response.close()
            except Exception:
                pass

    def healthy(self) -> bool:
        try:
            return self.http.get(f"{self.base_url}/health").status_code == 200
        except httpx.HTTPError:
            return False

    def wait_until_healthy(self, timeout: float) -> None:
        """Block until the server answers /health; it may still be loading the model."""
        deadline = time.monotonic() + timeout
        while not self.healthy():
            if time.monotonic() > deadline:
                raise TimeoutError(f"llama-server at {self.base_url} not ready after {timeout}s")
            time.sleep(1)

    def model_name(self) -> str | None:
        """File name of the model the server has loaded."""
        try:
            models = self.http.get(f"{self.base_url}/v1/models").json()["data"]
            return Path(models[0]["id"]).name if models else None
        except (httpx.HTTPError, KeyError, ValueError):
            return None

    def chat(self, messages: list[dict], max_tokens: int = MAX_TOKENS,
             temperature: float = TEMPERATURE) -> Reply:
        """Send a conversation and wait for the whole reply."""
        start = time.perf_counter()
        request = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            # No "thinking" phase: it would cost minutes on a CPU, and the facts are
            # already in the prompt. Models without that mode ignore the flag.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        with self._request(request) as response:
            response.read()
            body = response.json()
        timings = body.get("timings", {})
        usage = body.get("usage", {})
        return Reply(
            text=body["choices"][0]["message"]["content"] or "",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            prompt_per_second=timings.get("prompt_per_second", 0.0),
            tokens_per_second=timings.get("predicted_per_second", 0.0),
            seconds=time.perf_counter() - start,
            cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        )

    def stream(self, messages: list[dict], max_tokens: int = MAX_TOKENS,
               temperature: float = TEMPERATURE) -> Iterator[str]:
        """The same request as chat(), yielding the answer text as it is written.

        When the stream ends, the generator *returns* the finished Reply (as
        StopIteration.value), with the same timings and token counts as chat().
        """
        start = time.perf_counter()
        request = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": True,
            "stream_options": {"include_usage": True},   # timings and token counts
        }
        parts: list[str] = []
        usage, timings = {}, {}
        with self._request(request) as response:
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: "):].strip()
                if payload == "[DONE]":
                    break
                body = json.loads(payload)
                usage = body.get("usage") or usage
                timings = body.get("timings") or timings
                for choice in body.get("choices", []):
                    delta = (choice.get("delta") or {}).get("content")
                    if delta:
                        parts.append(delta)
                        yield delta
        return Reply(
            text="".join(parts),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            prompt_per_second=timings.get("prompt_per_second", 0.0),
            tokens_per_second=timings.get("predicted_per_second", 0.0),
            seconds=time.perf_counter() - start,
            cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        )

    def close(self) -> None:
        self.http.close()


def free_port() -> int:
    """A port nothing listens on, chosen by the operating system."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@contextmanager
def start_server(model: Path = DEFAULT_MODEL, port: int | None = None,
                 threads: int | None = LLAMA_THREADS,
                 threads_batch: int | None = LLAMA_THREADS_BATCH, context: int = CONTEXT_SIZE,
                 batch: int | None = BATCH_SIZE, startup_timeout: float = 180):
    """Run llama-server for the duration of a `with` block and yield a client for it.

    `threads` is used for writing the answer and `threads_batch` for reading the
    prompt; None lets llama.cpp choose. The server log goes to data/logs/.
    """
    binary = shutil.which(LLAMA_SERVER)
    if binary is None:
        raise FileNotFoundError(f"llama-server not found at {LLAMA_SERVER!r}; set LLAMA_SERVER "
                                f"in .env to the llama.cpp binary")
    if not model.exists():
        hint = f"; download it from {MODEL_URL}" if model == DEFAULT_MODEL else ""
        raise FileNotFoundError(f"{model} not found{hint}")

    port = port or free_port()
    command = [binary, "--model", str(model), "--port", str(port),
               "--ctx-size", str(context), "--host", "127.0.0.1"]
    if threads:
        command += ["--threads", str(threads)]
    if threads_batch:
        command += ["--threads-batch", str(threads_batch)]
    if batch:
        command += ["--batch-size", str(batch), "--ubatch-size", str(min(batch, 512))]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = open(LOG_DIR / f"{model.stem}.server.log", "w", encoding="utf-8")
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
                               if os.name == "nt" else 0)
    client = LLM(f"http://127.0.0.1:{port}")
    try:
        deadline = time.monotonic() + startup_timeout
        while not client.healthy():
            if process.poll() is not None:
                raise RuntimeError(f"llama-server exited early; see {log.name}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"llama-server not ready after {startup_timeout}s")
            time.sleep(0.5)
        yield client
    finally:
        client.close()
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
        log.close()
