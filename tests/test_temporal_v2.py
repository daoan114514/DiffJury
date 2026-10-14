import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from diffjury.assembler import ContextAssembler
from diffjury.release_gate import ReleaseGate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_swr_temporal_v2 import check_copied_inputs, copy_inputs, run_recurrence
import run_swr_temporal_v2 as v2


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_comparison_inputs_are_copied_and_never_overwritten_on_resume(tmp_path):
    source, diagnostic, out = (tmp_path / name for name in ("source", "diagnostic", "v2"))
    write(source / "state.json", {"finished_at": "done"})
    write(diagnostic / "state.json", {"finished_at": "done"})
    for name in ("events.json", "holdout_rows/baseline.json", "holdout_rows/final-naive.json", "recurrence_rows/targets.json",
                 "train_reviews/one.json", "recurrence_rows/baseline/one.json", "recurrence_rows/naive/one.json"):
        write(source / name, {"name": name})
    (source / "naive.db").write_bytes(b"frozen database")
    for name in ("holdout_rows/baseline_rerun.json", "diag.json", "manifest.json"):
        write(diagnostic / name, {"name": name})
    copy_inputs(source, diagnostic, out)
    check_copied_inputs(out)
    assert (out / "naive.db").read_bytes() == (source / "naive.db").read_bytes()
    assert not (out / "governed.db").exists()
    write(out / "holdout_rows/baseline.json", {"changed": True})
    with pytest.raises(ValueError, match="changed"):
        copy_inputs(source, diagnostic, out)
    assert json.loads((source / "holdout_rows/baseline.json").read_text(encoding="utf-8")) == {"name": "holdout_rows/baseline.json"}


def recurrence_row(case_id, successful=True):
    return {"id": case_id, "execution_success": successful,
            "metrics": {name: {"numerator": 0, "denominator": 1} for name in
                        ("false_positive_recurrence", "missed_recovery", "correct_retention")}}


def test_frozen_recurrence_failures_are_not_retried_and_only_new_arms_run(tmp_path, monkeypatch):
    train = [{"id": name, "split": "train"} for name in ("one", "two")]
    sample = {"case_ids": ["one", "two"], "selected_prs": 2, "paired_prs": 1, "by_event_type": {"missed": 2}}
    original = {"seed": 1, "recurrence": {"sample": sample, "notice": "feedback correction"}}
    for name in ("baseline", "naive"):
        for case in train:
            write(tmp_path / "recurrence_rows" / name / (case["id"] + ".json"),
                  recurrence_row(case["id"], name != "baseline" or case["id"] != "two"))
    targets = {case["id"]: {metric: [] for metric in ("false_positive_recurrence", "missed_recovery", "correct_retention")} for case in train}
    write(tmp_path / "recurrence_rows/targets.json", targets)
    before = (tmp_path / "recurrence_rows/baseline/two.json").read_bytes()
    calls = []

    def replay(reviewer, selected, directory, *args, **kwargs):
        calls.append(reviewer)
        return {"case_results": [{**recurrence_row(case["id"]), "findings": []} for case in selected]}

    monkeypatch.setattr(v2, "run_cached_replays", replay)
    result = run_recurrence(SimpleNamespace(workers=16, retry_workers=8, retry_failed=2), tmp_path, original,
        train, {name: name for name in ("baseline_rerun", "governed", "governed_canary")},
        {name: [] for name in ("baseline_rerun", "governed", "governed_canary")})
    assert calls == ["baseline_rerun", "governed", "governed_canary"]
    assert result["sample"]["paired_prs"] == 1 and result["failures"]["baseline"] == 1
    assert (tmp_path / "recurrence_rows/baseline/two.json").read_bytes() == before


def gate_case(name):
    return {"id": name, "repository": "org/repo", "pull_request": 1, "split": "validation",
            "diff": "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-old\n+new\n", "expected_findings": []}


def test_persistent_gate_cache_preserves_successes_and_retries_only_missing(tmp_path):
    calls = []

    class Reviewer:
        name = "persistent-gate"

        def review_case_detailed(self, case, parsed):
            calls.append(case["id"])
            if case["id"] == "two" and calls.count("two") == 1:
                raise TimeoutError()
            return [], {}

    cases = [gate_case("one"), gate_case("two")]
    cache = str(tmp_path / "cache")
    first = ReleaseGate(cases, lambda *args: Reviewer(), ContextAssembler(None), cache_dir=cache)
    assert first.replay([], "first")["metrics"]["execution_successes"] == 1
    second = ReleaseGate(cases, lambda *args: Reviewer(), ContextAssembler(None), cache_dir=cache)
    assert second.replay([], "resumed")["metrics"]["execution_successes"] == 2
    assert calls == ["one", "two", "two"]

    def forbidden(*args):
        pytest.fail("completed snapshot must be read from the persistent gate cache")

    third = ReleaseGate(cases, forbidden, ContextAssembler(None), cache_dir=cache)
    assert third.replay([], "again")["metrics"]["execution_successes"] == 2
