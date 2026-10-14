from pathlib import Path
import hashlib
import json
import sqlite3
import sys

import pytest

from diffjury.evaluation_governance import contamination, governance_statistics
from diffjury.evaluation_notes import build_notes, governance_markdown

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_swr_rot_experiment import refresh_reports
import run_swr_rot_experiment as experiment


def item(source, tokens, status="active", **extra):
    return {"id": source, "kind": "rule", "source": source, "tokens": tokens, "status": status,
            "archived_reason": "", **extra}


def admission(source, action, event, judge="llm"):
    return {"event": "admission", "item_id": "one", "detail": {
        "action": action, "judge": judge,
        "candidate": {"source": source, "evidence": [{"event": event}]}}}


def test_conflict_counts_are_exclusive_and_rates_keep_denominators():
    audit = [admission("missed", "ADD", 0), admission("noise", "REJECT", 1),
             admission("contradiction", "REJECT", 2, "llm-conflict"), admission("contradiction", "UPVOTE", 3)]
    items = {"naive": [item("missed", 80), item("noise", 20)],
             "governed": [item("disk", 200), item("missed", 80), item("contradiction", 20, "probation")]}
    events = [{"type": kind} for kind in ("missed", "noise", "contradiction", "contradiction")]
    result = governance_statistics(items, audit, events)
    counts = result["admission_by_source"]["contradiction"]
    assert counts["CONFLICT"] == 1 and counts["REJECT"] == 0 and counts["UPVOTE"] == 1
    assert result["interception"]["contradiction"]["candidate_rate"] == 0.5
    assert result["contamination"]["naive"]["item_rate"] == 0.5
    assert result["contamination"]["naive"]["token_rate"] == 0.2
    assert result["contamination"]["governed"]["tokens"] == 80
    assert result["contamination"]["governed"]["item_rate"] == 0
    assert result["contamination"]["governed_canary"]["token_rate"] == 0.2
    assert contamination([], ("active",))["item_rate"] is None
    assert "不适用" in "\n".join(governance_markdown(governance_statistics({"naive": [], "governed": []}, [], [])))


def test_counterfactual_statistics_include_failed_attempts_and_guard_train_only():
    rows = [{"event": "counterfactual", "item_id": "one", "detail": {
        "applied": True, "case_id": "pr", "split": "train", "replays": 2, "helpful": 1}},
        {"event": "counterfactual", "item_id": "one", "detail": {
            "applied": False, "split": "train", "replays": 1, "error_type": "TimeoutError"}},
        {"event": "promote", "item_id": "one", "detail": {}}]
    result = governance_statistics({"naive": [], "governed": []}, rows, [])
    cf = result["counterfactual"]
    assert (cf["pair_attempts"], cf["individual_replays"], cf["applied_pairs"], cf["failed"]) == (2, 3, 1, 1)
    assert cf["train_only"]
    rows[1]["detail"]["split"] = "holdout"
    assert not governance_statistics({"naive": [], "governed": []}, rows, [])["counterfactual"]["train_only"]


def test_archive_counts_distinguish_capacity_harm_and_merges():
    items = {"naive": [], "governed": [
        item("missed", 20, "archived", archived_reason="capacity"),
        item("noise", 20, "archived", archived_reason="harmful"),
        item("missed", 20, "archived", archived_reason="merged-into:K-one"),
    ]}
    result = governance_statistics(items, [], [])
    assert result["lifecycle"]["archived_categories"] == {"capacity": 1, "harmful": 1, "merged": 1}
    assert "容量淘汰 1、harmful 淘汰 1、归并 1" in "\n".join(governance_markdown(result))


def note_state():
    metrics = {"f1": 0.3, "precision": 0.3, "recall": 0.3,
               "average_instruction_tokens_per_call": 1000, "average_total_tokens_per_pr": 2000}
    comparisons = {key: {"delta": 0, "ci95": [-0.1, 0.1]} for key in metrics}
    return {"finished_at": "2026", "model": "fake", "seed": 1,
            "dataset": {"train": 6, "gate": 1, "holdout": 3, "holdout_executed_all": 3},
            "holdout_by_scope": {"executed_all": {name: dict(metrics) for name in ("baseline", "naive", "governed")}},
            "comparisons_by_scope": {"executed_all": {pair: dict(comparisons) for pair in
                ("governed_minus_baseline", "governed_minus_naive", "naive_minus_baseline")}},
            "governance": governance_statistics({"naive": [], "governed": []}, [], []),
            "recurrence": {"sample": {"selected_prs": 2, "paired_prs": 2, "by_event_type": {"missed": 2}},
                           "metrics": {}, "comparisons": {}}}


def test_notes_cannot_claim_f1_gain_when_ci_includes_zero():
    state = note_state()
    text = build_notes(state)
    assert "95% CI 包含 0，不能称F1提升" in text
    assert "不代表泛化能力" in text
    assert "公开历史 PR" in text
    assert "不适用" in text
    state["comparisons_by_scope"]["executed_all"]["governed_minus_baseline"]["f1"] = {"delta": 0.1, "ci95": [0.01, 0.2]}
    assert "可称本次设定下 F1 提升" in build_notes(state)
    state["comparisons_by_scope"]["executed_all"]["governed_minus_baseline"]["f1"] = {"delta": -0.1, "ci95": [-0.2, -0.01]}
    assert "在本次设定下 F1 下降" in build_notes(state)
    with pytest.raises(ValueError, match="finished experiment"):
        build_notes({})


def test_report_only_refuses_unfinished_run_without_creating_databases(tmp_path):
    with pytest.raises(ValueError, match="finished experiment"):
        refresh_reports(str(tmp_path))
    assert list(tmp_path.glob("*.db")) == []


def test_report_only_cli_refreshes_finished_utf8_reports_without_model_or_input_writes(tmp_path, monkeypatch):
    state = note_state()
    state.update({"measurements": [], "checkpoints": {}})
    (tmp_path / "state.json").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "events.json").write_text("[]\n", encoding="utf-8")
    (tmp_path / "run.log").write_text("已完成\n", encoding="utf-8")
    for name in ("naive", "governed"):
        with sqlite3.connect(tmp_path / (name + ".db")) as conn:
            conn.execute("CREATE TABLE knowledge_items (id TEXT,tenant_id TEXT,kind TEXT,source TEXT,status TEXT,tokens INTEGER,archived_reason TEXT)")
            conn.execute("CREATE TABLE knowledge_events (tenant_id TEXT,item_id TEXT,event TEXT,detail_json TEXT)")
            conn.execute("INSERT INTO knowledge_items VALUES (?,?,?,?,?,?,?)",
                         ("K-one", experiment.rot.TENANT, "rule", "missed", "active", 20, ""))
    inputs = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in tmp_path.iterdir()
              if path.name != "state.json"}

    def forbidden(*args, **kwargs):
        pytest.fail("report-only must not load model settings or create a client")

    monkeypatch.setattr(experiment.Settings, "from_env", forbidden)
    monkeypatch.setattr(experiment, "JsonChatClient", forbidden)
    monkeypatch.setattr(sys, "argv", ["run_swr_rot_experiment.py", "--out", str(tmp_path), "--report-only"])
    experiment.main()

    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in inputs.items())
    governance = json.loads((tmp_path / "governance.json").read_text(encoding="utf-8"))
    assert governance["contamination"]["governed"]["items"] == 1
    assert "知识治理统计" in (tmp_path / "summary.md").read_text(encoding="utf-8")
    notes = (tmp_path / "NOTES.md").read_text(encoding="utf-8")
    assert "假模型流程验证" in notes and "不代表泛化能力" in notes
    assert "\ufffd" not in notes
