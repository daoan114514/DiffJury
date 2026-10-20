import io
import json
import urllib.error
from unittest.mock import patch

import pytest

from diffjury.llm import JsonChatClient
from diffjury.telemetry import ExecutionLedger


def response(content, reason="stop", tokens=20):
    return io.BytesIO(json.dumps({"choices": [{"message": {"content": content}, "finish_reason": reason}],
                                  "usage": {"prompt_tokens": 10, "completion_tokens": tokens}}).encode())


def test_invalid_json_retries_complete_object_and_accounts_for_failed_reply():
    requests = []
    replies = iter([response('{"findings":[', "length", 4000), response('{"findings":[]}')])

    def send(request, timeout):
        requests.append(json.loads(request.data))
        return next(replies)

    ledger = ExecutionLedger("test")
    client = JsonChatClient("https://example.invalid/v1", "private-sentinel", "fake", json_retries=1)
    with patch("urllib.request.urlopen", send):
        assert client.complete_json("correctness-reliability", "Return JSON", "review", ledger, max_tokens=4000) == {"findings": []}
    assert requests[0]["max_tokens"] == requests[1]["max_tokens"] == 4000
    assert "COMPLETE JSON object" in requests[1]["messages"][0]["content"]
    assert [call.ok for call in ledger.model_calls] == [False, True]
    assert ledger.model_calls[0].output_tokens == 4000
    assert "finish_reason=length" in ledger.model_calls[0].error


def test_max_concurrency_serialises_requests_across_clients():
    import threading
    import time

    active, peak, lock = [0], [0], threading.Lock()

    def send(request, timeout):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with lock:
            active[0] -= 1
        return response('{"ok": true}')

    clients = [JsonChatClient("https://example.invalid/v1", "k", "fake", max_concurrency=1) for _ in range(2)]
    with patch("urllib.request.urlopen", send):
        threads = [threading.Thread(target=clients[index % 2].complete_json, args=("r", "Return JSON", "x"))
                   for index in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert peak[0] == 1


def test_http_errors_are_not_retried_inline_and_credentials_are_redacted():
    client = JsonChatClient("https://example.invalid/v1", "private-sentinel", "fake", json_retries=1)
    ledger = ExecutionLedger("test")
    error = urllib.error.HTTPError("https://example.invalid", 401, "invalid", {}, io.BytesIO(b"private-sentinel"))
    with patch("urllib.request.urlopen", side_effect=error) as send:
        with pytest.raises(RuntimeError) as captured:
            client.complete_json("test", "Return JSON", "test", ledger)
    assert send.call_count == 1
    assert "private-sentinel" not in str(captured.value)
    assert "private-sentinel" not in ledger.model_calls[0].error
