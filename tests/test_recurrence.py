import pytest

from diffjury.evaluation_recurrence import (finding_from_dict, measure_recurrence,
    paired_recurrence_bootstrap, recurrence_rates, recurrence_targets, sample_feedback)


def label(line, path="app.py", **extra):
    return {"path": path, "start_line": line, "end_line": line, "severity": "medium", **extra}


def finding(line, path="app.py", rule="LOGIC"):
    return finding_from_dict({"path": path, "line": line, "rule_id": rule})


def row(case_id, numerator, denominator=1):
    return {"id": case_id, "execution_success": True,
            "metrics": {key: {"numerator": numerator, "denominator": denominator}
                        for key in ("false_positive_recurrence", "missed_recovery", "correct_retention")}}


def test_sampling_is_deterministic_bounded_and_train_only():
    cases = [{"id": str(i), "split": "train"} for i in range(100)]
    events = [{"type": kind, "case_id": str(i)} for i in range(100) for kind in ("missed", "false_positive")]
    selected = sample_feedback(events, cases, limit=80, seed=9)
    assert selected == sample_feedback(events, cases, limit=80, seed=9)
    assert all(len(ids) == 80 for ids in selected["by_type"].values())
    with pytest.raises(ValueError, match="train only"):
        sample_feedback(events, [{"id": "0", "split": "holdout"}])
    with pytest.raises(ValueError, match="outside"):
        sample_feedback(events, cases[:-1])


def test_recovery_and_retention_use_original_annotations_and_fp_uses_type():
    case = {"id": "one", "expected_findings": [label(10), label(30), label(50, neutral=True)]}
    original = {"findings": [finding(10).to_dict(), finding(80, rule="SEC-SQL").to_dict()]}
    events = [{"case_id": "one", "type": "missed", "label_index": 1, "finding": {"path": "app.py", "line": 30}},
              {"case_id": "one", "type": "false_positive", "finding": original["findings"][1]}]
    targets = recurrence_targets(case, events, original)
    result = measure_recurrence(targets, [finding(12), finding(29), finding(81, rule="SEC-SQL"), finding(50)])
    assert all(value == {"numerator": 1, "denominator": 1} for value in result.values())
    different = measure_recurrence(targets, [finding(81, rule="SEC-CMD")])
    assert different["false_positive_recurrence"]["numerator"] == 0


def test_original_feedback_without_label_index_and_nearby_duplicate_finds():
    case = {"id": "one", "expected_findings": [label(10)]}
    events = [{"case_id": "one", "type": "missed", "finding": {"path": "app.py", "line": 10}}]
    targets = recurrence_targets(case, events, {"findings": []})
    result = measure_recurrence(targets, [finding(10), finding(11)])
    assert result["missed_recovery"] == {"numerator": 1, "denominator": 1}
    assert result["correct_retention"]["denominator"] == 0


def test_clustered_paired_bootstrap_reports_signed_ci_and_missing_denominator():
    before = [row("one", 0), row("two", 0)]
    after = [row("one", 1), row("two", 1)]
    result = paired_recurrence_bootstrap(before, after, iterations=200)
    assert result["missed_recovery"]["delta"] == 1
    assert result["missed_recovery"]["ci95"] == [1, 1]
    assert result["false_positive_recurrence"]["higher_is_better"] is False
    assert recurrence_rates([row("one", 0, 0)])["correct_retention"]["rate"] is None
    assert paired_recurrence_bootstrap([row("one", 0, 0)], [row("one", 0, 0)])["correct_retention"]["ci95"] is None
    with pytest.raises(ValueError, match="identical ordered"):
        paired_recurrence_bootstrap(before, after[::-1])
    with pytest.raises(ValueError, match="failed"):
        paired_recurrence_bootstrap(before, [{**after[0], "execution_success": False}, after[1]])
