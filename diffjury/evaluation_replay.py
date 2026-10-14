"""Resumable, read-only-snapshot replays shared by SWR diagnostic experiments."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Sequence

from .assembler import ContextAssembler, StaticKnowledge, TaskSignals
from .context_manager import estimate_tokens
from .diff_parser import parse_unified_diff
from .evaluation_repos import GitSnapshots, SnapshotReviewer
from .evaluation_v2 import ProductionEvaluationHarness, ReplayReviewer
from .knowledge import KnowledgeItem, KnowledgeRepository


PROTECTED_RESULTS = ("swr_rot", "rot_stress", "rot_stress_v2", "swr_rot_temporal")


def check_output(root: Path, output: Path) -> None:
    output = output.resolve()
    for name in PROTECTED_RESULTS:
        protected = (root / "evaluation_results" / name).resolve()
        if output == protected or protected in output.parents:
            raise ValueError("output is a protected experiment directory")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_items(path: Path, statuses: Sequence[str] = ("active",)) -> List[KnowledgeItem]:
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        return [KnowledgeRepository._decode(row) for row in conn.execute(
            "SELECT * FROM knowledge_items WHERE tenant_id=?", ("default",)) if row["status"] in statuses]


def assembler_from_config(settings: dict) -> ContextAssembler:
    return ContextAssembler(
        None, rule_budget=settings["knowledge_rule_budget_tokens"],
        skill_budget=settings["knowledge_skill_budget_tokens"],
        skill_body_cap=settings["knowledge_skill_body_cap_tokens"],
        catalog_limit=settings["knowledge_skill_catalog_limit"],
        half_life_days=settings["knowledge_half_life_days"],
    )


def make_reviewer(client, config: dict, skills: list, snapshots: GitSnapshots, name: str,
                  assembler=None, items: Sequence[KnowledgeItem] = ()):
    knowledge = StaticKnowledge(assembler, items) if assembler is not None else None
    return SnapshotReviewer(ReplayReviewer(
        client, config["agent_token_budget"], config["agent_time_budget_seconds"],
        name=name, knowledge=knowledge, skill_provider=lambda _tenant: skills,
    ), snapshots)


class _CaptureReviewer:
    def __init__(self, delegate):
        self.delegate, self.name, self.summary, self.findings = delegate, delegate.name, {}, []

    def review_case_detailed(self, case, parsed):
        self.findings, self.summary = self.delegate.review_case_detailed(case, parsed)
        return self.findings, self.summary


class SkillReplayHarness(ProductionEvaluationHarness):
    def __init__(self, skills: list, **kwargs):
        super().__init__(**kwargs)
        self.skills = {skill.name: skill for skill in skills}

    def _run_case(self, reviewer, case):
        capture = _CaptureReviewer(reviewer)
        row = super()._run_case(capture, case)
        row["skill_instructions"] = [
            {**entry, "skill_body_tokens": sum(estimate_tokens(self.skills[name].instructions)
                                               for name in entry.get("skills", []) if name in self.skills)}
            for entry in (capture.summary.get("execution") or {}).get("instruction_log", [])
        ]
        row["knowledge"] = capture.summary.get("knowledge") or {}
        row["findings"] = [finding.to_dict() for finding in capture.findings]
        return row


def aggregate(rows: list) -> dict:
    totals = ProductionEvaluationHarness._empty_totals()
    for row in rows:
        ProductionEvaluationHarness._accumulate(totals, row)
    return {"case_results": rows, "metrics": ProductionEvaluationHarness._metrics(totals)}


def run_cached_replays(reviewer, cases: list, directory: Path, skills: list,
                       workers: int = 16, retry_workers: int = 8, retries: int = 2,
                       secret: str = "", log=print) -> dict:
    if not 1 <= workers <= 16 or not 1 <= retry_workers <= 8:
        raise ValueError("workers must be 1..16; retry-workers must be 1..8")
    directory.mkdir(parents=True, exist_ok=True)
    rows = {case["id"]: read_json(directory / (case["id"] + ".json")) for case in cases}
    harness = SkillReplayHarness(skills)

    def run(case):
        row = harness._run_case(reviewer, case)
        if secret and row.get("error"):
            row["error"] = str(row["error"]).replace(secret, "[REDACTED]")
        write_json(directory / (case["id"] + ".json"), row)
        return row

    for attempt in range(retries + 1):
        pending = [case for case in cases if not rows[case["id"]] or not rows[case["id"]].get("execution_success")]
        if not pending:
            break
        log("replay %s: %d PRs, attempt %d" % (reviewer.name, len(pending), attempt + 1))
        with ThreadPoolExecutor(max_workers=workers if attempt == 0 else retry_workers) as pool:
            for row in pool.map(run, pending):
                rows[row["id"]] = row
        result = aggregate([rows[case["id"]] for case in cases])
        write_json(directory.with_suffix(".json"), result)
        log("replay %s: completed=%d/%d, F1=%s" % (
            reviewer.name, result["metrics"]["execution_successes"], len(cases), result["metrics"]["f1"]))
    result = aggregate([rows[case["id"]] for case in cases])
    write_json(directory.with_suffix(".json"), result)
    if result["metrics"]["failure_rate"] > 0.10:
        raise RuntimeError("too many failed replays; resume from cached successful PRs")
    return result


def skill_audit(cases: list, skills: list, assembler: ContextAssembler) -> dict:
    rows, dropped_counts = [], {}
    for case in cases:
        signals = TaskSignals.from_review(case["diff"], parse_unified_diff(case["diff"]).files, case["repository"])
        catalog = assembler.skill_catalog(skills, signals)
        names = {skill.name for skill in catalog}
        dropped = [skill.name for skill in skills if skill.name not in names]
        for name in dropped:
            dropped_counts[name] = dropped_counts.get(name, 0) + 1
        fitted = assembler.skill_bodies(catalog)
        rows.append({"id": case["id"], "baseline_catalog": [skill.name for skill in skills],
                     "governed_catalog": [skill.name for skill in catalog], "catalog_dropped": dropped,
                     "baseline_all_body_tokens": sum(estimate_tokens(skill.instructions) for skill in skills),
                     "governed_all_body_tokens": fitted["tokens"], "body_dropped": fitted["dropped"]})
    return {"notice": "offline availability audit; selecting every catalog skill is a probe, not actual worker delegation",
            "skills": {skill.name: {"source": skill.source, "body_tokens": estimate_tokens(skill.instructions)} for skill in skills},
            "catalog_drop_frequencies": dropped_counts, "case_results": rows}


def background(script: Path, arguments: list, out: Path) -> None:
    command = [sys.executable, "-u", "-W", "ignore::SyntaxWarning", str(script)]
    command += [arg for arg in arguments if arg != "--background"]
    with (out / "run.log").open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=handle, cwd=script.parent.parent,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    write_json(out / "process.json", {"pid": process.pid, "command": command})
    print("background pid=%d; log=%s" % (process.pid, out / "run.log"))
