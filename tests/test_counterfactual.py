import pytest

from diffjury.admission import AdmissionGate
from diffjury.assembler import ContextAssembler
from diffjury.knowledge import KnowledgeItem
from diffjury.knowledge_service import KnowledgeGovernor
from diffjury.lifecycle import KnowledgeLifecycle, LifecyclePolicy
from diffjury.store import TaskStore


def setup_governor(tmp_path, exposures=0, scope="*", minimum=3):
    repo = TaskStore(str(tmp_path / "knowledge.db")).knowledge
    stable = repo.create(KnowledgeItem("default", "rule", "Check array shape.", status="active",
                                       triggers=["shape"], pinned=True))
    trial = repo.create(KnowledgeItem("default", "rule", "Check shape before indexing.",
                                      status="probation", triggers=["shape"], exposures=exposures,
                                      scope=scope))
    governor = KnowledgeGovernor(repo, ContextAssembler(repo), AdmissionGate(repo), background=False,
                                  lifecycle=KnowledgeLifecycle(repo, LifecyclePolicy(promote_min_exposures=minimum)))
    return governor, stable, trial


def cases(count=4):
    return [{"id": "train-%d" % i, "split": "train", "repository": "org/repo",
             "diff": "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-old\n+shape = array.shape\n",
             "created_at": "2025-01-%02d" % (i + 1)} for i in range(count)]


def factory_for(trial, calls, helpful=True, fail=False, exposed=True):
    def factory(knowledge, name):
        class Reviewer:
            def review_case_detailed(self, case, parsed):
                with_trial = trial.id in {item.id for item in knowledge.items}
                calls.append((case["id"], with_trial, [item.id for item in knowledge.items]))
                if fail and with_trial:
                    raise TimeoutError()
                findings = ["hit"] if with_trial and helpful else ([] if helpful else ["fp"] if with_trial else [])
                return findings, {"knowledge": {"exposed": [trial.id] if with_trial and exposed else []}}
        return Reviewer()
    return factory


def classify(case, findings):
    return {"hits": [0] if "hit" in findings else [],
            "false_positives": [0] if "fp" in findings else []}


def test_pairs_only_training_cases_and_records_causal_outcomes(tmp_path):
    governor, stable, trial = setup_governor(tmp_path)
    calls = []
    result = governor.counterfactual_round("default", cases(), factory_for(trial, calls), classify,
                                           round_id="one", workers=2)
    assert result["pairs"] == 3
    assert result["replays"] == 6
    assert result["helpful"] == 3
    after = governor.repository.get("default", trial.id)
    assert (after.exposures, after.helpful_total, after.status) == (3, 3.0, "probation")
    assert governor.lifecycle.promotable(after, after.updated_at)
    assert all(stable.id in ids for _, _, ids in calls)
    assert governor.repository.get("default", stable.id).helpful_total == 0
    assert all(event["detail"]["source"] == "counterfactual"
               for event in governor.repository.events("default", trial.id))


def test_holdout_and_validation_are_rejected_before_any_review(tmp_path):
    governor, _, trial = setup_governor(tmp_path)
    calls = []
    for split in ("validation", "holdout"):
        with pytest.raises(ValueError, match="train cases only"):
            governor.counterfactual_round("default", [{**cases(1)[0], "split": split}],
                                           factory_for(trial, calls), classify)
    assert calls == []


def test_review_budget_counts_both_sides_and_respects_scope(tmp_path):
    governor, _, trial = setup_governor(tmp_path)
    calls = []
    result = governor.counterfactual_round("default", cases(), factory_for(trial, calls), classify,
                                           replay_limit=5)
    assert result["replays"] == 4
    other = [{**case, "repository": "org/other"} for case in cases()]
    trial.scope = "org/repo"
    governor.repository.save(trial)
    result = governor.counterfactual_round("default", other, factory_for(trial, calls), classify)
    assert result["replays"] == 0


def test_failed_or_unexposed_pair_does_not_earn_outcomes(tmp_path):
    governor, _, trial = setup_governor(tmp_path)
    calls = []
    failed = governor.counterfactual_round("default", cases(1), factory_for(trial, calls, fail=True), classify)
    assert failed["failed"] == 1
    skipped = governor.counterfactual_round("default", cases(1), factory_for(trial, calls, exposed=False), classify)
    assert skipped["not_exposed"] == 1
    after = governor.repository.get("default", trial.id)
    assert (after.exposures, after.helpful_total, after.harmful_total) == (0, 0, 0)


def test_harmful_pairs_and_resumption_do_not_duplicate_evidence(tmp_path):
    governor, _, trial = setup_governor(tmp_path, minimum=5)
    calls = []
    result = governor.counterfactual_round("default", cases(3), factory_for(trial, calls, helpful=False),
                                           classify, round_id="one")
    assert result["harmful"] == 3
    again = governor.counterfactual_round("default", cases(3), factory_for(trial, calls), classify,
                                          round_id="one")
    assert again["replays"] == 0
    after = governor.repository.get("default", trial.id)
    assert (after.exposures, after.harmful_total) == (3, 3.0)
    assert governor.lifecycle.eviction_reason(after, after.updated_at) == "harmful"


def test_outcome_and_source_event_are_atomically_idempotent(tmp_path):
    governor, _, trial = setup_governor(tmp_path)
    detail = {"source": "counterfactual", "applied": True, "idempotency_key": "round:item:case"}
    for _ in range(2):
        governor.repository.apply_outcome("default", trial.id, helpful=1, exposures=1,
                                           event="counterfactual", detail=detail)
    assert governor.repository.get("default", trial.id).exposures == 1
    assert governor.repository.get("default", trial.id).helpful_total == 1
    assert len(governor.repository.events("default", trial.id)) == 1
