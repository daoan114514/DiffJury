import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from diffjury.assembler import ContextAssembler, KnowledgeView, TaskSignals
from diffjury.evaluation_temporal import temporal_split
from diffjury.knowledge import KnowledgeItem
from diffjury.skills import AgentSkill
from diffjury.release_gate import ReleaseGate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_swr_rot_experiment import FakeSwrClient, SwrArm, review_train, scope_events, write_summary
import run_rot_stress_test as rot


def source_cases():
    return [{"id": "%s-%d" % (repo, i), "repository": repo, "pull_request": i,
             "created_at": "2025-01-%02dT12:00:00Z" % (i + 1), "split": "holdout"}
            for repo in ("org/one", "org/two") for i in range(10)][::-1]


def test_temporal_split_is_per_repository_chronological_and_keeps_source():
    source = source_cases()
    cases, summary = temporal_split(source, ("org/one", "org/two"))
    assert summary["by_split"] == {"train": 12, "validation": 2, "holdout": 6}
    assert all(case["split"] == "holdout" for case in source)
    assert len({(case["repository"], case["pull_request"]) for case in cases}) == 20
    assert summary["repositories"]["org/one"]["train"]["last_created_at"] == "2025-01-06T12:00:00Z"
    with pytest.raises(ValueError, match="more than once"):
        temporal_split(source + [source[0]], ("org/one", "org/two"))
    with pytest.raises(ValueError, match="created_at"):
        temporal_split([{**case, "created_at": ""} for case in source], ("org/one", "org/two"))


def skill_for(record):
    return AgentSkill(record.name, record.content, record.body, record.instruction_text(),
                       source="learned", version="1")


def test_repository_scope_filters_rules_catalog_and_explicit_skill_bodies():
    signals = TaskSignals.from_review("+shape = array.shape", ["app.py"], "org/one")
    one = KnowledgeItem("default", "skill", "Review shape", name="shape-one", body="Check shape.", scope="org/one")
    two = KnowledgeItem("default", "skill", "Review shape", name="shape-two", body="Check shape.", scope="org/two")
    universal = KnowledgeItem("default", "rule", "Check shape.", scope="*", pinned=True)
    foreign = KnowledgeItem("default", "rule", "Check shape.", scope="org/two", pinned=True)
    view = KnowledgeView(ContextAssembler(None), [one, two, universal, foreign], signals)
    skills = [skill_for(one), skill_for(two)]
    assert view.rules_for("security").ids == [universal.id]
    assert [skill.name for skill in view.skill_catalog(skills)] == [one.name]
    assert [skill.name for skill in view.skill_bodies("security", skills)["skills"]] == [one.name]
    # A foreign body with the same name cannot sneak through the scoped catalog.
    assert view.skill_bodies("security", [AgentSkill(one.name, one.content, "Foreign body", "Foreign body", source="learned")])["skills"] == []


def test_real_noise_and_contradiction_lessons_inherit_source_training_repository():
    train = [{"id": "one", "repository": "org/one", "created_at": "2025"},
             {"id": "two", "repository": "org/two", "created_at": "2026"}]
    events = [{"index": 0, "type": "missed", "case_id": "one", "lessons": [{}]},
              {"index": 1, "type": "missed", "case_id": "two", "lessons": [{}]},
              {"index": 2, "type": "noise", "lessons": [{}]},
              {"index": 3, "type": "contradiction", "target_index": 0, "lessons": [{}]}]
    scope_events(events, train)
    assert [event["lessons"][0]["scope"] for event in events] == ["org/one", "org/two", "org/two", "org/one"]


def test_temporal_boundaries_and_each_holdout_comparison_are_rendered(tmp_path):
    _, temporal = temporal_split(source_cases(), ("org/one", "org/two"))
    state = {"temporal_split": temporal, "holdout_by_scope": {"all": {}, "recent": {}},
             "comparisons_by_scope": {scope: {scope + "-pair": {"f1": {"delta": 0, "ci95": [-1, 1]}}}
                                       for scope in ("all", "recent")}}
    write_summary(str(tmp_path), state)
    text = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "org/one" in text and "org/two" in text
    assert "all-pair" in text and "recent-pair" in text
    write_summary(str(tmp_path), {"temporal_split": temporal})


def test_gate_retries_transport_failure_and_keeps_full_execution_requirement():
    attempts = []
    case = {"id": "one", "split": "validation", "repository": "org/one", "pull_request": 1,
            "diff": "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-old\n+new\n", "expected_findings": []}

    class Reviewer:
        name = "retry-test"

        def review_case_detailed(self, case, parsed):
            attempts.append(case["id"])
            if len(attempts) == 1:
                raise TimeoutError()
            return [], {}

    gate = ReleaseGate([case], lambda *args: Reviewer(), ContextAssembler(None), retry_attempts=1)
    result = gate.evaluate([], [])
    assert result["passed"]
    assert result["execution_success_rate"] == {"baseline": 1.0, "candidate": 1.0}
    assert len(attempts) == 2


def test_learning_resumes_each_lesson_once_and_preserves_naive_scope(tmp_path):
    settings = SimpleNamespace(
        knowledge_rule_budget_tokens=600, knowledge_skill_budget_tokens=3000,
        knowledge_skill_body_cap_tokens=1500, knowledge_skill_catalog_limit=8,
        knowledge_half_life_days=30, knowledge_rule_max_tokens=60,
        knowledge_similarity_threshold=0.3, knowledge_duplicate_threshold=0.85,
        knowledge_capacity_tokens_per_role=2400,
    )
    args = SimpleNamespace(promote_min_exposures=3, evict_min_exposures=12,
                           probation_max_exposures=16, workers=16)
    lesson = {"kind": "rule", "content": "Check shape before indexing.", "scope": "org/one", "role": "security"}
    event = {"index": 1, "type": "missed", "lessons": [lesson, dict(lesson)]}
    naive = SwrArm("naive", str(tmp_path), FakeSwrClient(), settings, [], [], args)
    governed = SwrArm("governed", str(tmp_path), FakeSwrClient(), settings, [], [], args)
    assert naive.learn(event) == ["APPEND", "APPEND"]
    assert naive.learn(event) == ["APPEND", "APPEND"]
    assert len(naive.repo.list("default")) == 2
    assert all(item.scope == "org/one" for item in naive.repo.list("default"))
    assert governed.learn(event) == ["ADD", "UPVOTE"]
    assert governed.learn(event) == ["ADD", "UPVOTE"]
    assert len([row for row in governed.repo.events("default") if row["event"] == "admission"]) == 2


def test_train_retries_transient_failure_and_never_silently_drops_feedback(tmp_path):
    calls = []
    cases = [{"id": str(i), "repository": "org/one", "expected_findings": [],
              "diff": "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-old\n+new\n"} for i in range(2)]

    class Reviewer:
        def review_case_detailed(self, case, parsed):
            calls.append(case["id"])
            if case["id"] == "0" and calls.count("0") == 1:
                raise TimeoutError()
            return [], {}

    values = review_train(cases, Reviewer(), str(tmp_path / "cache"), 1, retry_attempts=1)
    assert all(values.values())
    assert calls.count("0") == 2 and calls.count("1") == 1

    class Unavailable:
        def review_case_detailed(self, case, parsed):
            raise TimeoutError()

    with pytest.raises(rot.ModelUnavailable):
        review_train(cases, Unavailable(), str(tmp_path / "empty"), 1)
