import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from diffjury.evaluation_replay import check_output, read_items, run_cached_replays
from diffjury.skills import AgentSkill
from diffjury.store import TaskStore
from diffjury.knowledge import KnowledgeItem

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from run_swr_temporal_diagnostics import diagnosis, write_report
import run_swr_temporal_diagnostics as diagnostics


def case():
    return {"id":"one","repository":"org/repo","pull_request":1,"split":"holdout","diff":"--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-old\n+new\n",
            "expected_findings":[]}


def test_second_handoff_protects_temporal_and_nested_output(tmp_path):
    for path in ("swr_rot_temporal","swr_rot_temporal/child","rot_stress_v2"):
        with pytest.raises(ValueError,match="protected"):
            check_output(tmp_path,tmp_path/"evaluation_results"/path)
    check_output(tmp_path,tmp_path/"evaluation_results/swr_rot_temporal_diag")


def test_diagnostic_snapshot_is_read_only_and_only_active(tmp_path):
    store=TaskStore(str(tmp_path/"governed.db"))
    store.knowledge.create(KnowledgeItem("default","rule","Check values.",status="active"))
    store.knowledge.create(KnowledgeItem("default","rule","Check paths.",status="probation"))
    before=hashlib.sha256((tmp_path/"governed.db").read_bytes()).hexdigest()
    assert len(read_items(tmp_path/"governed.db"))==1
    assert hashlib.sha256((tmp_path/"governed.db").read_bytes()).hexdigest()==before
    with pytest.raises(sqlite3.OperationalError):
        read_items(tmp_path/"absent.db")
    assert not (tmp_path/"absent.db").exists()


def test_replay_caches_each_pr_retries_failure_and_records_actual_worker_skill_bodies(tmp_path,monkeypatch):
    calls=[]
    skill=AgentSkill("check","Check code","Check changed values.","Check changed values.")

    class Reviewer:
        name="cached"

        def review_case_detailed(self,source,parsed):
            calls.append(source["id"])
            if len(calls)==1:
                raise TimeoutError("temporary")
            return [],{"execution":{"instruction_log":[{"role":"security","skills":["check"],"instruction_tokens":20}]}}

    report=run_cached_replays(Reviewer(),[case()],tmp_path/"rows",[skill],workers=1,retry_workers=1,log=lambda message:None)
    assert report["metrics"]["execution_successes"]==1
    assert report["case_results"][0]["skill_instructions"][0]["skill_body_tokens"]>0
    assert len(calls)==2
    run_cached_replays(Reviewer(),[case()],tmp_path/"rows",[skill],workers=1,retry_workers=1,log=lambda message:None)
    assert len(calls)==2
    reports={name:report for name in ("baseline","governed","baseline_rerun","governed_assembler_empty","governed_rerun")}
    audit={"catalog_drop_frequencies":{"check":1}}
    result=diagnosis(reports,audit,iterations=200)
    assert not result["decision"]["clipping_impact_confirmed"]
    assert not result["decision"]["knowledge_harm_confirmed"]
    write_report(tmp_path,result,audit,True)
    text=(tmp_path/"diag.md").read_text(encoding="utf-8")
    assert "假模型流程检查" in text and "暂不修改" in text
    assert result["knowledge_audit"]["governed_rerun"]["exposed_prs"]==0

    source=tmp_path/"source"
    (source/"holdout_rows").mkdir(parents=True)
    (source/"holdout_rows/final-governed.json").write_text(json.dumps(report),encoding="utf-8")
    (tmp_path/"holdout_rows").mkdir()
    (tmp_path/"holdout_rows/governed_rerun.json").write_text(json.dumps(report),encoding="utf-8")
    for name,value in (("diag",result),("skill_audit",audit),("state",{"fake":True})):
        (tmp_path/(name+".json")).write_text(json.dumps(value,ensure_ascii=False),encoding="utf-8")

    def forbidden():
        pytest.fail("report-only must not load model settings")

    monkeypatch.setattr(diagnostics.Settings,"from_env",forbidden)
    monkeypatch.setattr(sys,"argv",["diagnostics","--out",str(tmp_path),"--source",str(source),"--report-only"])
    diagnostics.main()
    assert "没有实际注入" in (tmp_path/"diag.md").read_text(encoding="utf-8")
