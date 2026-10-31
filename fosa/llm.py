"""The LLM transport: chat requests to any OpenAI-compatible server, with retries, JSON replies and a limit on the
requests in flight across jobs; and the log root that every run writes under.

Settings (environment): FOSA_LLM_ENDPOINT, FOSA_LLM_MODEL, FOSA_LLM_API_KEY (else OPENAI_API_KEY),
FOSA_LLM_MAX_CONCURRENCY, FOSA_LLM_TIMEOUT_S and FOSA_LOG_ROOT.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from fosa import REPO_ROOT

DEFAULT_ENDPOINT = "http://localhost:8000/v1/chat/completions"
DEFAULT_MODEL = "gemma-4-31B-it"  # the paper's LLM
DEFAULT_TIMEOUT_S = 300.0  # seconds to wait for one reply
CONTEXT_WINDOWS = {"glm-5.2": 256_000, "gemma-4-31b-it": 262_144, "gpt-oss-120b": 131_072}  # tokens
DEFAULT_CONTEXT_WINDOW = 8192
NO_REASONING_MODELS = {"glm-5.2", "gemma-4-31b-it", "gpt-oss-120b"}  # sent reasoning_effort "none"
OPTIONAL_FIELDS = ("response_format", "reasoning_effort")  # dropped, in this order, by a server that rejects them
RETRYABLE_HTTP_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
MAX_ATTEMPTS = 12  # for errors that persist; an overloaded server (HTTP 429) is waited out without using them
JSON_ATTEMPTS = 4
MAX_OUTPUT_TOKENS = 8192  # the token limit of a JSON reply doubles on each retry, up to this
COMPACT_JSON_REQUEST = (
    "Return the same required JSON much more compactly: omit optional prose and use counts instead of expanding "
    "repeated items."
)
LOCK_DIR = REPO_ROOT / "cache" / "llm_locks"


def model_log_name(model: str | None = None) -> str:
    """The model's name as a directory name."""
    selected = str(model or os.environ.get("FOSA_LLM_MODEL") or DEFAULT_MODEL)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", selected).strip("._-")


def model_log_root(model: str | None = None) -> Path:
    """Where runs write: FOSA_LOG_ROOT, else logs/<model> in the repository."""
    override = os.environ.get("FOSA_LOG_ROOT")
    return Path(override).expanduser().resolve() if override else REPO_ROOT / "logs" / model_log_name(model)


def default_api_key() -> str | None:
    return os.environ.get("FOSA_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or None


def default_timeout_s() -> float:
    return max(1.0, float(os.environ.get("FOSA_LLM_TIMEOUT_S", DEFAULT_TIMEOUT_S)))


@contextlib.contextmanager
def exclusive_file(path: Path):
    """Hold the lock of path, so one job at a time creates a shared cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write through a temporary file, so that no reader sees a partial file."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


# ----------------------------------------------------------------------------------------------------------------------
# Prompts


def compact_json(payload: Any) -> str:
    """payload as one-line JSON, without its empty values."""
    return json.dumps(prune_empty(payload), separators=(",", ":"), sort_keys=False)


def prune_empty(value: Any) -> Any:
    if isinstance(value, dict):
        pruned = {key: prune_empty(item) for key, item in value.items()}
        return {key: item for key, item in pruned.items() if item not in (None, [], {}, "")}
    if isinstance(value, list):
        return [prune_empty(item) for item in value]
    return value


def truncate_text(value: Any, limit: int = 240) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return (text or None) if len(text) <= limit else text[: limit - 1] + "…"


def call_json_within_context(
    build_messages: Callable[[int], list[dict[str, str]] | None],
    args: argparse.Namespace,
    *,
    max_tokens: int,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """A JSON reply to the most detailed prompt that fits the model's context window. build_messages(level) is the
    prompt at detail level 0, 1, ..., each shorter than the one before, and None past the last level."""
    window = CONTEXT_WINDOWS.get(args.model.lower(), DEFAULT_CONTEXT_WINDOW)
    budget = max(512, window - int(max_tokens) - 256)  # prompt tokens, leaving room for the reply
    level = 0
    while (messages := build_messages(level)) is not None:
        prompt_tokens = sum(len(message["content"]) for message in messages) // 3 + 8 * len(messages)
        if prompt_tokens <= budget:
            return chat_json_completion(
                endpoint=os.environ.get("FOSA_LLM_ENDPOINT", DEFAULT_ENDPOINT),
                model=args.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2,
                api_key=default_api_key(),
                timeout_s=default_timeout_s() if timeout_s is None else timeout_s,
            )
        level += 1
    raise RuntimeError("No prompt variant fits the model context window")


# ----------------------------------------------------------------------------------------------------------------------
# Requests


@contextlib.contextmanager
def transport_slot():
    """One of FOSA_LLM_MAX_CONCURRENCY request slots, shared through lock files by every job on the filesystem."""
    slots = max(1, int(os.environ.get("FOSA_LLM_MAX_CONCURRENCY", "8")))
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        for index in range(slots):
            with (LOCK_DIR / f"slot-{index}.lock").open("a+") as handle:  # closing the file releases the lock
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                yield
                return
        time.sleep(random.uniform(0.3, 0.8))


def retry_after_s(error: urllib.error.HTTPError) -> float:
    try:
        return float(error.headers.get("Retry-After", 0.0))
    except (TypeError, ValueError):  # an HTTP date
        return 0.0


def _chat_choice(
    endpoint: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    api_key: str | None,
    timeout_s: float,
    json_mode: bool = False,
) -> dict[str, Any]:
    """The first choice of one chat request. Network errors and transient HTTP errors are retried with backoff, up to
    MAX_ATTEMPTS times; an overloaded server (HTTP 429) is waited out; a rejected optional field is dropped."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max(1, int(max_tokens)),
        "temperature": float(temperature),
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if model.lower() in NO_REASONING_MODELS:
        payload["reasoning_effort"] = "none"
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {api_key}"} if api_key else {})}
    attempt = 0
    while True:
        request = urllib.request.Request(
            endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
        )
        try:
            with transport_slot(), urllib.request.urlopen(request, timeout=max(1.0, float(timeout_s))) as response:
                return json.loads(response.read().decode("utf-8"))["choices"][0]
        except OSError as error:  # urllib's HTTPError and URLError, and timeouts
            status = error.code if isinstance(error, urllib.error.HTTPError) else None
            rejected = [name for name in OPTIONAL_FIELDS if name in payload]
            if status in {400, 422} and rejected:
                payload.pop(rejected[0])
                continue
            if status is not None and status not in RETRYABLE_HTTP_CODES:
                raise
            wait = 0.0 if status is None else retry_after_s(error)
            if status == 429:
                # saturation, not refusal: every job shares one key's parallel-request cap, so wait for a turn
                # (walltime bounds it) and keep the attempts for errors that persist
                wait = max(wait, 15.0)
            else:
                attempt += 1
                if attempt >= MAX_ATTEMPTS:
                    raise
                wait = max(wait, min(60.0, 2.0**attempt))
        time.sleep(wait + random.uniform(0.0, 0.5))


def chat_completion(
    *,
    endpoint: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    api_key: str | None,
    timeout_s: float,
) -> str:
    """The text of one chat reply."""
    return _chat_choice(endpoint, model, messages, max_tokens, temperature, api_key, timeout_s)["message"]["content"]


def chat_json_completion(
    *,
    endpoint: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    api_key: str | None,
    timeout_s: float,
) -> dict[str, Any]:
    """One JSON object as the reply. A reply that does not parse (usually one cut off at the token limit) is asked for
    again, more compactly and with twice the token limit, up to JSON_ATTEMPTS times in all."""
    request_messages, token_limit = messages, max_tokens
    for attempt in range(JSON_ATTEMPTS):
        choice = _chat_choice(endpoint, model, request_messages, token_limit, temperature, api_key, timeout_s, True)
        content = choice["message"].get("content") or ""
        try:
            return json.loads(extract_json(content))
        except ValueError:
            if attempt == JSON_ATTEMPTS - 1:
                finish_reason = choice.get("finish_reason")
                raise ValueError(
                    f"Incomplete VLM JSON after {len(content)} characters (finish_reason={finish_reason!r})"
                )
        token_limit = min(2 * token_limit, MAX_OUTPUT_TOKENS)
        request_messages = [*messages, {"role": "user", "content": COMPACT_JSON_REQUEST}]


def extract_json(content: str) -> str:
    """The JSON object in a reply: the whole reply, a ```json block, or the outermost braces; "" if there is none."""
    stripped = content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if match:
        return match.group(1)
    start, end = stripped.find("{"), stripped.rfind("}")
    return stripped[start : end + 1] if 0 <= start < end else ""
