import pytest

from diffjury.admission import AdmissionGate
from diffjury.assembler import ContextAssembler
from diffjury.evaluation_recurrence import finding_from_dict
from diffjury.evaluation_governance import read_governance
from diffjury.knowledge import KnowledgeItem
from diffjury.knowledge_service import KnowledgeGovernor
from diffjury.lifecycle import KnowledgeLifecycle, LifecyclePolicy
from diffjury.source_validation import source_judgment
from diffjury.store import TaskStore
from diffjury.evaluation_notes import governance_markdown


def finding(line, rule="LOGIC"):
    return finding_from_dict({"path": "app.py", "line": line, "rule_id": rule})


def label(line):
    return {"path": "app.py", "start_line": line, "end_line": line, "severity": "medium", "rule_id": "LOGIC"}


class Gate:
    def __init__(self, passed=True, split="validation"):
        self.passed, self.calls = passed, []
        self.cases = [{"split": split}]

    def evaluate(self, before, after, max_token_growth=0):
        self.calls.append(([item.id for item in before], [item.id for item in after], max_token_growth))
        return {"passed": self.passed, "failures": [] if self.passed else ["f1"]}


def setup(tmp_path, source="false_positive", minimum=100, gate=None, count=3):
    tmp_path.mkdir(parents=True, exist_ok=True)
    repo = TaskStore(str(tmp_path / "source.db")).knowledge
    item = repo.create(KnowledgeItem("default", "rule", "Check shape values." if source == "missed" else "Do not flag safe shape values.",
                                     source=source, role="critic" if source == "false_positive" else "correctness-reliability",
                                     scope="org/repo", triggers=["shape"], evidence=[{"event": 0, "case": "train-0", "type": source}]))
    lifecycle = KnowledgeLifecycle(repo, LifecyclePolicy(promote_min_exposures=minimum))
    governor = KnowledgeGovernor(repo, ContextAssembler(repo), AdmissionGate(repo), lifecycle=lifecycle,
                                  release_gate=gate, background=False)
    diff = "--- a/app.py\n+++ b/app.py\n@@ -0,0 +1,35 @@\n" + "".join("+shape = data.shape\n" for _ in range(35))
    cases = [{"id": "train-%d" % i, "split": "train", "repository": "org/repo", "diff": diff,
              "created_at": "2025-01-%02d" % (i + 1), "expected_findings": [label(10)] + ([label(20)] if source == "missed" and i == 0 else [])}
             for i in range(count)]
    target = finding(20).to_dict() if source == "missed" else finding(1, "FP-TARGET").to_dict()
    events = [{"index": 0, "type": source, "case_id": "train-0", "finding": target}]
    if source == "missed":
        events[0]["label_index"] = 1
    originals = {case["id"]: {"id": case["id"], "findings": [finding(10).to_dict()] + ([target] if source == "false_positive" and case["id"] == "train-0" else [])}
                 for case in cases}
    return governor, item, cases, events, originals


def factory_for(items, calls, source="false_positive", effective=True, exposed=True, fail=False,
                lose_original=False, side_loss=False, extra_fp=False):
    ids = {item.id for item in items}

    def factory(knowledge, name):
        active = ids.intersection(item.id for item in knowledge.items)

        class Reviewer:
            def review_case_detailed(self, case, parsed):
                trial = bool(active)
                calls.append((case["id"], trial))
                if trial and fail:
                    raise TimeoutError()
                results = [] if trial and (lose_original or side_loss and case["id"] != "train-0") else [finding(10)]
                if case["id"] == "train-0":
                    if source == "false_positive" and (not trial or not effective):
                        results.append(finding(1, "FP-TARGET"))
                    if source == "missed" and trial and effective:
                        results.append(finding(20))
                    if trial and extra_fp:
                        results.append(finding(30, "EXTRA-FP"))
                return results, {"knowledge": {"exposed": sorted(active) if exposed else []}}

        return Reviewer()

    return factory


def validate(governor, item, cases, events, originals, calls, **options):
    factory_options = options.pop("factory_options", {})
    return governor.source_pr_validation_round("default", cases, events, originals,
        factory_for([item], calls, source=item.source, **factory_options), workers=2, round_id="100", **options)


@pytest.mark.parametrize("source", ["false_positive", "missed"])
def test_three_source_pairs_other_prs_and_fast_promotion_still_require_gate(tmp_path, source):
    gate = Gate()
    governor, item, cases, events, originals = setup(tmp_path, source, gate=gate)
    calls = []
    result = validate(governor, item, cases, events, originals, calls)
    assert result["passed"] == 1 and result["round_replays"] == 10
    assert calls.count(("train-0", False)) == calls.count(("train-0", True)) == 3
    assert all(case_id in {case["id"] for case in cases} for case_id, _ in calls)
    after = governor.repository.get("default", item.id)
    assert after.status == "probation" and after.exposures == 5
    assert not governor.lifecycle.promotable(after, after.updated_at)
    summary = governor.maintain("default")
    assert summary["promotion"]["fast_promoted"] == [item.id]
    assert governor.repository.get("default", item.id).status == "active"
    assert gate.calls[-1] == ([], [item.id], None)
    assert len([row for row in governor.repository.events("default", item.id) if row["event"] == "source_validation_promotion"]) == 1


@pytest.mark.parametrize("passed", [False, None])
def test_rejected_or_missing_gate_cannot_publish_verified_item(tmp_path, passed):
    gate = Gate(False) if passed is False else None
    governor, item, cases, events, originals = setup(tmp_path, minimum=3, gate=gate)
    validate(governor, item, cases, events, originals, [])
    result = governor.maintain("default")
    assert governor.repository.get("default", item.id).status == "probation"
    if gate is None:
        assert result["source_validation"]["status"] == "waiting_for_validation_gate"
    else:
        assert result["promotion"]["status"] == "rejected"
    assert not any(row["event"] == "source_validation_promotion" for row in governor.repository.events("default"))


def test_source_and_side_hit_losses_block_fast_promotion(tmp_path):
    governor, item, cases, events, originals = setup(tmp_path, gate=Gate())
    result = validate(governor, item, cases, events, originals, [], factory_options={"side_loss": True})
    assert result["harmful"] == 1 and result["passed"] == 0
    assert governor._refresh_source_validation("default")[item.id]["side_effect"] == "lost annotation hit"
    governor.maintain("default")
    assert governor.repository.get("default", item.id).status != "active"


@pytest.mark.parametrize("source,options", [("false_positive", {"lose_original": True}), ("missed", {"extra_fp": True})])
def test_original_hit_loss_or_extra_false_positives_invalidate_source_benefit(tmp_path, source, options):
    governor, item, cases, events, originals = setup(tmp_path, source)
    result = validate(governor, item, cases, events, originals, [], factory_options=options)
    assert result["passed"] == 0 and result["harmful"] == 1


@pytest.mark.parametrize("options", [{"fail": True}, {"exposed": False}])
def test_failed_or_unexposed_sources_never_earn_outcomes(tmp_path, options):
    governor, item, cases, events, originals = setup(tmp_path)
    result = validate(governor, item, cases, events, originals, [], factory_options=options)
    after = governor.repository.get("default", item.id)
    assert result["passed"] == 0 and (after.exposures, after.helpful_total, after.harmful_total) == (0, 0, 0)
    assert result["round_replays"] == 2


def test_budget_and_resume_cover_both_arms_and_do_not_duplicate_credit(tmp_path):
    governor, item, cases, events, originals = setup(tmp_path)
    second = governor.repository.create(KnowledgeItem("default", "rule", "Do not report validated shape constants.",
        source="false_positive", role="critic", scope="org/repo", triggers=["shape"], evidence=item.evidence))
    calls = []
    factory = factory_for([item, second], calls)
    result = governor.source_pr_validation_round("default", cases, events, originals, factory,
                                                 replay_limit=10, side_cases=1, workers=2, round_id="100")
    assert result["passed"] == 1 and result["round_replays"] == 8
    exposures = sum(record.exposures for record in governor.repository.list("default"))
    again = governor.source_pr_validation_round("default", cases, events, originals, factory,
                                                 replay_limit=10, side_cases=1, workers=2, round_id="100")
    assert again["replays"] == 0 and len(calls) == 8
    assert sum(record.exposures for record in governor.repository.list("default")) == exposures == 4


def test_pending_reservation_survives_process_interruption_without_exceeding_cap(tmp_path):
    governor, item, cases, events, originals = setup(tmp_path)
    governor.repository.log_event("default", item.id, "source_validation_reservation",
        {"round_id": "100", "job_key": "unfinished", "reserved_replays": 8, "signature": "old"})
    calls = []
    result = validate(governor, item, cases, events, originals, calls, replay_limit=8, side_cases=1)
    assert result["replays"] == 0 and result["budget_consumed"] == 8 and calls == []


def test_holdout_source_evidence_and_nonvalidation_gate_are_rejected(tmp_path):
    governor, item, cases, events, originals = setup(tmp_path, gate=Gate(split="holdout"))
    calls = []
    with pytest.raises(ValueError, match="train cases only"):
        validate(governor, item, [{**cases[0], "split": "holdout"}], events, originals, calls)
    with pytest.raises(ValueError, match="outside the training"):
        validate(governor, item, cases[1:], events, originals, calls)
    assert calls == []
    validate(governor, item, cases, events, originals, calls)
    with pytest.raises(ValueError, match="validation-only"):
        governor.maintain("default")
    assert governor.release_gate.calls == []


def test_no_other_triggering_pr_cannot_pass_and_body_changes_invalidate_proof(tmp_path):
    governor, item, cases, events, originals = setup(tmp_path, gate=Gate(), count=1)
    calls = []
    result = validate(governor, item, cases, events, originals, calls)
    assert result["skipped"] == 1 and calls == []
    governor, item, cases, events, originals = setup(tmp_path / "other", gate=Gate())
    validate(governor, item, cases, events, originals, [])
    updated = governor.repository.get("default", item.id)
    updated.content = "Flag every shape value."
    governor.repository.save(updated)
    governor.maintain("default")
    assert governor.repository.get("default", item.id).status == "probation"


def test_capacity_drops_neutral_before_unverified_before_passed(tmp_path):
    governor, item, _, _, _ = setup(tmp_path)
    neutral = governor.repository.create(KnowledgeItem("default", "rule", "Check old shapes.", role="critic", source="missed"))
    passed = governor.repository.create(KnowledgeItem("default", "rule", "Check validated shapes.", role="critic", source="missed"))
    lifecycle = governor.lifecycle
    lifecycle.protect_pending_sources = True
    lifecycle.validation_status = {neutral.id: "neutral", passed.id: "passed"}
    lifecycle.policy.capacity_tokens_per_role = item.tokens + passed.tokens
    result = lifecycle.plan([item, neutral, passed])
    assert result.archived == {neutral.id: "capacity"}
    lifecycle.policy.capacity_tokens_per_role = passed.tokens
    result = lifecycle.plan([item, neutral, passed])
    assert result.archived == {neutral.id: "capacity", item.id: "capacity"}


def test_target_frequencies_are_aggregated_across_three_pairs():
    before = [{"hits": [0, 1] if hit else [0], "target_hits": hit, "retained_hits": [0], "false_positives": 0} for hit in (1, 0, 0)]
    after = [{"hits": [0, 1] if hit else [0], "target_hits": hit, "retained_hits": [0], "false_positives": 0} for hit in (0, 1, 1)]
    assert source_judgment("missed", before, after, 1)["passed"]


def test_validation_audit_reports_budgets_actual_reviews_and_gated_promotions(tmp_path):
    import shutil

    governor, item, cases, events, originals = setup(tmp_path, gate=Gate())
    validate(governor, item, cases, events, originals, [])
    governor.maintain("default")
    shutil.copyfile(tmp_path / "source.db", tmp_path / "governed.db")
    TaskStore(str(tmp_path / "naive.db"))
    result = read_governance(str(tmp_path))
    validation = result["source_validation"]
    assert validation["review_attempts"] == 10 and validation["completed_pairs"] == 5
    assert validation["passed_unique_items"] == validation["unique_fast_promoted_items"] == 1
    assert validation["unfinished_reservations"] == 0 and validation["train_only"]
    assert validation["rounds"]["100"] == {"replay_limit": 120, "budget_consumed": 10, "review_attempts": 10}
    text = "\n".join(governance_markdown(result))
    assert "来源 PR 验证与快速转正" in text and "不是显著性检验" in text
