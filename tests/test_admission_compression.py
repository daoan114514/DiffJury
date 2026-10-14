from diffjury.admission import AdmissionGate
from diffjury.context_manager import estimate_tokens
from diffjury.metrics import metrics


class CompressClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def complete_json(self, role, system, user, ledger=None, max_tokens=None):
        self.calls.append((role, system))
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return {"content": reply}


LONG = "Check the changed behaviour and all related conditions " * 20


def test_strict_retry_fits_and_uses_eighty_percent_target():
    client = CompressClient([LONG, "Check array shape before indexing."])
    gate = AdmissionGate(None, client, max_rule_tokens=60)
    before = metrics.counters["knowledge_admission_compress_retry_total"]
    assert gate._compress(LONG) == "Check array shape before indexing."
    assert "at most 48 tokens" in client.calls[1][1]
    assert "concrete identifiers" in client.calls[1][1]
    assert metrics.counters["knowledge_admission_compress_retry_total"] == before + 1


def test_successful_first_compression_does_not_retry():
    client = CompressClient(["Validate shapes."])
    assert AdmissionGate(None, client)._compress(LONG) == "Validate shapes."
    assert len(client.calls) == 1


def test_first_sentence_fallback_preserves_api_name_and_negation():
    first = "Never call np.reshape on incompatible shapes."
    client = CompressClient([LONG, first + " " + LONG])
    result = AdmissionGate(None, client)._compress(first + " " + LONG)
    assert result == first
    assert estimate_tokens(result) <= 60


def test_outage_can_use_original_first_sentence():
    first = "Check input dimensions."
    client = CompressClient([TimeoutError(), TimeoutError()])
    assert AdmissionGate(None, client)._compress(first + " " + LONG) == first


def test_unfit_sentence_is_not_cut_and_counts_failure():
    client = CompressClient([LONG, LONG])
    gate = AdmissionGate(None, client)
    before = metrics.counters["knowledge_admission_compress_failed_total"]
    assert gate._compress(LONG) == ""
    assert metrics.counters["knowledge_admission_compress_failed_total"] == before + 1
    decision = AdmissionGate(None, max_rule_tokens=60)._decide("default", {
        "content": LONG, "evidence": [{"task_id": "test"}],
    })
    assert decision.reason == "rule exceeds 60 tokens after compression"


def test_compression_cannot_reverse_a_negative_recommendation():
    negative = "Do not report invalid shapes because " + LONG
    client = CompressClient(["Report invalid shapes.", "Do not report invalid shapes."])
    assert AdmissionGate(None, client)._compress(negative) == "Do not report invalid shapes."
    assert len(client.calls) == 2
