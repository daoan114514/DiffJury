import json

import pytest

from diffjury.admission import AdmissionGate
from diffjury.knowledge import KnowledgeItem
from diffjury.metrics import metrics
from diffjury.similarity import fingerprint
from diffjury.store import TaskStore


class ConflictClient:
    def __init__(self, verdict="CONFLICT"):
        self.verdict = verdict
        self.calls = []

    def complete_json(self, role, system, user, ledger=None, max_tokens=None):
        payload = json.loads(user)
        self.calls.append(payload)
        assert "CONFLICT" in system
        return {"decision": self.verdict, "target_id": payload["similar_items"][0]["id"],
                "reason": "opposite actions in the same scenario"}


@pytest.mark.parametrize("negative", ["Do not report invalid shapes.", "Never report invalid shapes.", "Avoid reporting invalid shapes."])
def test_negative_similar_lesson_is_judged_and_rejected(tmp_path, negative):
    repo = TaskStore(str(tmp_path / "knowledge.db")).knowledge
    original = repo.create(KnowledgeItem("default", "rule", "Report invalid shapes.",
                                         status="active", evidence=[{"task_id": "original"}]))
    client = ConflictClient()
    gate = AdmissionGate(repo, client, similarity_threshold=0.1, dup_threshold=0.3)
    before = metrics.counters["knowledge_admission_conflict_total"]
    decision = gate.admit("default", {"content": negative, "evidence": [{"task_id": "opposite"}]})
    assert decision.action == "REJECT"
    assert decision.reason == "conflicts with %s" % original.id
    assert len(client.calls) == 1
    assert repo.get("default", original.id).evidence == [{"task_id": "original"}]
    assert repo.get("default", original.id).content == original.content
    assert any(event["event"] == "conflict" for event in repo.events("default", original.id))
    assert metrics.counters["knowledge_admission_conflict_total"] == before + 1


def test_fingerprint_ignores_negation_but_gate_does_not(tmp_path):
    assert fingerprint("Report invalid shapes.") == fingerprint("Do not report invalid shapes.")
    repo = TaskStore(str(tmp_path / "knowledge.db")).knowledge
    repo.create(KnowledgeItem("default", "rule", "Report invalid shapes.", status="active"))
    decision = AdmissionGate(repo).admit("default", {
        "content": "Do not report invalid shapes.", "evidence": [{"task_id": "test"}],
    })
    assert decision.action == "REJECT"
    assert "requires" in decision.reason


def test_invalid_judge_reply_cannot_upvote_opposite_advice(tmp_path):
    repo = TaskStore(str(tmp_path / "knowledge.db")).knowledge
    original = repo.create(KnowledgeItem("default", "rule", "Report invalid shapes.", status="active"))
    decision = AdmissionGate(repo, ConflictClient("")).admit("default", {
        "content": "Do not report invalid shapes.", "evidence": [{"task_id": "test"}],
    })
    assert decision.action == "REJECT"
    assert repo.get("default", original.id).evidence == []


def test_duplicate_from_another_repository_is_not_upvoted(tmp_path):
    repo = TaskStore(str(tmp_path / "knowledge.db")).knowledge
    repo.create(KnowledgeItem("default", "rule", "Report invalid shapes.", scope="org/first"))
    decision = AdmissionGate(repo).admit("default", {
        "content": "Report invalid shapes.", "scope": "org/second", "evidence": [{"task_id": "test"}],
    })
    assert decision.action == "ADD"
    assert repo.get("default", decision.item_id).scope == "org/second"
