"""Small OpenAI-compatible JSON client with auditable usage accounting."""
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from . import __version__
from .context_manager import estimate_tokens
from .telemetry import ExecutionLedger


def first_json_object(content: Any) -> Any:
    """Decode the first JSON value in a model reply, ignoring surrounding text."""
    text = str(content or "")
    start = text.find("{")
    if start < 0:
        raise ValueError("model reply does not contain a JSON object")
    value, _end = json.JSONDecoder().raw_decode(text, start)
    return value


_slots_guard = threading.Lock()
_slots: Dict[int, threading.BoundedSemaphore] = {}


def request_slots(limit: int) -> Optional[threading.BoundedSemaphore]:
    """Process-wide cap on in-flight model requests, shared by every client.

    Parallel agents and replay workers queue here when an upstream only allows
    serial (or lightly concurrent) use; 0 means unlimited."""
    if limit <= 0:
        return None
    with _slots_guard:
        return _slots.setdefault(limit, threading.BoundedSemaphore(limit))


class JsonChatClient:
    def __init__(
        self, base_url: str, api_key: str, model: str,
        provider: str = "openai-compatible", timeout: int = 60,
        extra_headers: Optional[Dict[str, str]] = None,
        json_retries: int = 0,
        max_concurrency: Optional[int] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.timeout = timeout
        self.extra_headers = dict(extra_headers or {})
        self.json_retries = max(0, min(2, int(json_retries)))
        if max_concurrency is None:
            max_concurrency = int(os.environ.get("DIFFJURY_LLM_MAX_CONCURRENCY", "0") or 0)
        self.slots = request_slots(int(max_concurrency))

    @staticmethod
    def _mention_json(user: str) -> str:
        """Gateways that map json_object mode onto the Responses API require the
        word "json" in the input messages (system instructions do not count)."""
        return user if "json" in user.lower() else user + "\n\nReply with a json object."

    def complete_json(
        self, role: str, system: str, user: str,
        ledger: Optional[ExecutionLedger] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": self._mention_json(user)},
            ],
            "response_format": {"type": "json_object"},
        }
        if max_tokens:
            payload["max_tokens"] = int(max_tokens)
        headers = {
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            # Some gateways (Cloudflare error 1010) reject urllib's default agent.
            "User-Agent": "DiffJury/%s" % __version__,
        }
        headers.update(self.extra_headers)
        for attempt in range(self.json_retries + 1):
            if attempt:
                payload["messages"][0]["content"] = system + (
                    "\nYour last response was not valid JSON. Return one COMPLETE JSON object, no Markdown. "
                    "Keep explanation, fix and test text short; preserve required verbatim evidence. "
                    "Close every object and array within the output limit."
                )
                if ledger:
                    ledger.record_instruction(role, estimate_tokens(payload["messages"][0]["content"]))
            request = urllib.request.Request(
                self.base_url + "/chat/completions",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers=headers, method="POST",
            )
            if self.slots is not None:
                self.slots.acquire()
            # Latency excludes time spent queueing for a request slot.
            started = time.monotonic()
            usage, finish_reason, response_chars, can_retry = {}, "", 0, False
            try:
                try:
                    with urllib.request.urlopen(request, timeout=self.timeout) as response:
                        body = json.loads(response.read().decode("utf-8"))
                finally:
                    if self.slots is not None:
                        self.slots.release()
                choice = body["choices"][0]
                usage = body.get("usage") or {}
                finish_reason = str(choice.get("finish_reason") or "")
                content = choice["message"]["content"]
                response_chars = len(str(content or ""))
                can_retry = True
                result = first_json_object(content)
                if not isinstance(result, dict):
                    raise ValueError("model JSON root is not an object")
                if ledger:
                    ledger.record_model(role, self.provider, self.model, usage,
                                        int((time.monotonic() - started) * 1000), True)
                return result
            except urllib.error.HTTPError as exc:
                detail = exc.read(1000).decode("utf-8", errors="replace")
                message = "%s API returned HTTP %d: %s" % (self.provider, exc.code, detail)
            except (urllib.error.URLError, socket.timeout, ValueError, KeyError,
                    IndexError, TypeError, json.JSONDecodeError) as exc:
                message = "%s JSON request failed: %s" % (self.provider, exc)
                if finish_reason:
                    message += " (finish_reason=%s, response_chars=%d)" % (finish_reason, response_chars)
            if self.api_key:
                message = message.replace(self.api_key, "[redacted]")
            if ledger:
                ledger.record_model(role, self.provider, self.model, usage,
                                    int((time.monotonic() - started) * 1000), False, message)
            if not can_retry or attempt >= self.json_retries:
                raise RuntimeError(message)
