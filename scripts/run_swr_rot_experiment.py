"""Context-rot experiment on real pull requests (SWR-Bench).

Same question as run_rot_stress_test.py - does a governed knowledge base stay
small and keep review quality while a naive "append every lesson" prompt rots?
- but the learning signal comes from real review outcomes:

1. **Train reviews.** The knowledge-free reviewer reviews every train PR
   (scikit-learn, matplotlib, pytest, sphinx) inside a checkout of the PR head.
   Missed functional change points become "missed" events whose note is the
   human reviewer's own discussion summary; findings on code reviewers left
   alone become "false positive" events (at most one per PR).
2. **Stress.** Generic noise (20% of the stream) and contradictions of earlier
   lessons (10%) are mixed in, exactly as in the synthetic stress test.
3. **Replay.** Both arms consume the identical, time-ordered stream. The
   governed arm also earns attribution from periodic train reviews, and is
   maintained every ``--maintain-every`` events through the replay gate on a
   fixed validation subset (xarray, pylint, seaborn, flask).
4. **Holdout.** sympy, astropy, requests and django PRs are reviewed with no
   knowledge (baseline) and by both arms at the end; governed-vs-naive and
   each-arm-vs-baseline differences get paired-bootstrap 95% CIs, overall and
   for PRs created in 2024 or later (less likely to be in the model's training data).

The run is resumable and stops (instead of recording zeros) when the model
becomes unavailable, e.g. when the daily quota runs out.

    python scripts/run_swr_rot_experiment.py --out evaluation_results/swr_rot_improved
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_rot_stress_test as rot  # noqa: E402
from diffjury.agentic_core import AgenticReviewer  # noqa: E402
from diffjury.admission import opposite_polarity  # noqa: E402
from diffjury.assembler import StaticKnowledge  # noqa: E402
from diffjury.config import Settings  # noqa: E402
from diffjury.diff_parser import parse_unified_diff  # noqa: E402
from diffjury.evaluation_harness import dataset_fingerprint, load_jsonl, one_to_one_match  # noqa: E402
from diffjury.evaluation_repos import GitSnapshots, SnapshotReviewer  # noqa: E402
from diffjury.evaluation_recurrence import (  # noqa: E402
    RECURRENCE_METRICS, RECURRENCE_NOTICE, finding_from_dict, measure_recurrence,
    paired_recurrence_bootstrap, recurrence_rates, recurrence_targets, sample_feedback,
)
from diffjury.evaluation_governance import read_governance  # noqa: E402
from diffjury.evaluation_notes import build_notes, governance_markdown  # noqa: E402
from diffjury.evaluation_v2 import (  # noqa: E402
    ProductionEvaluationHarness, ReplayReviewer, _EvaluationTaskStore, paired_bootstrap_comparison,
)
from diffjury.knowledge_service import reflect_lessons  # noqa: E402
from diffjury.llm import JsonChatClient  # noqa: E402
from diffjury.skills import SkillRegistry  # noqa: E402
from diffjury.telemetry import ExecutionLedger  # noqa: E402


log, write_json, read_json = rot.log, rot.write_json, rot.read_json


def holdout_metrics(report: dict) -> Dict[str, Any]:
    """Single-pass holdout: no per-repeat F1 breakdown."""
    values = rot.holdout_metrics(report)
    values.pop("f1_runs", None)
    values.pop("f1_run_std", None)
    return values
DATASET = os.path.join(ROOT, "evaluation_data", "swrbench", "swr_bench_functional.jsonl")
REPOS = os.path.join(ROOT, "evaluation_data", "swrbench", "repos")
QUALITY = ("f1", "precision", "recall", "high_risk_recall", "clean_accuracy")
MAX_FP_EVENTS_PER_PR = 1


def functional(case: dict) -> List[dict]:
    return [label for label in case["expected_findings"] if not label.get("neutral")]


def neutral(case: dict) -> List[dict]:
    return [label for label in case["expected_findings"] if label.get("neutral")]


def classify(case: dict, findings: list) -> Dict[str, Any]:
    """Split one review's findings into hits, false positives and misses."""
    labels = functional(case)
    matches = one_to_one_match(labels, findings, 2)
    matched = {match.predicted_index for match in matches}
    leftover_index = [index for index in range(len(findings)) if index not in matched]
    leftover = [findings[index] for index in leftover_index]
    neutral_matches = one_to_one_match(neutral(case), leftover, 2) if neutral(case) else []
    neutral_hit = {leftover_index[match.predicted_index] for match in neutral_matches}
    return {
        "hits": sorted(matched),
        "false_positives": [index for index in leftover_index if index not in neutral_hit],
        "neutral": sorted(neutral_hit),
        "missed": [
            index for index in range(len(labels))
            if index not in {match.expected_index for match in matches}
        ],
    }


class FakeSwrClient(rot.FakeStressClient):
    """Offline stand-in: the security worker flags the first added line of a diff."""

    def complete_json(self, role, system, user, ledger=None, max_tokens=None):
        if role == "knowledge-reflection":
            if ledger:
                ledger.record_model(role, self.provider, self.model, {"prompt_tokens": 100, "completion_tokens": 20}, 1)
            payload = json.loads(user)
            path = str((payload.get("finding") or {}).get("path", ""))
            negative = payload.get("category") == "false_positive"
            return {"lessons": [{"role": "critic" if negative else "correctness-reliability",
                                 "content": "Do not flag constant values as defects." if negative else "Check changed values and function calls.",
                                 "triggers": [os.path.basename(path)] if path else []}]}
        if role == "knowledge-compress":
            if ledger:
                ledger.record_model(role, self.provider, self.model, {"prompt_tokens": 100, "completion_tokens": 20}, 1)
            lesson = str(json.loads(user).get("lesson", ""))
            return {"content": "Do not flag constant values as defects." if opposite_polarity(lesson, "")
                    else "Check changed values and function calls."}
        if role != "security":
            return super().complete_json(role, system, user, ledger, max_tokens)
        if ledger:
            ledger.record_model(role, self.provider, self.model, {"prompt_tokens": 100, "completion_tokens": 20}, 1)
        task = json.loads(json.loads(user)["task"])
        diff = str(task.get("diff", ""))
        path = (task.get("lead_assignment") or {}).get("files", [""])[0]
        match = re.search(r"^\+\s*(\d+) \| (.+)$", diff, re.M)
        if not match or not path:
            return {"action": "final", "findings": []}
        return {"action": "final", "findings": [{
            "rule_id": "LOGIC", "severity": "medium", "title": "Possible logic issue",
            "explanation": "Changed behaviour may be wrong.", "path": path,
            "line": int(match.group(1)), "evidence": match.group(2).strip(),
            "fix": "fix", "test": "test", "confidence": 0.8,
            "applied_rules": re.findall(r"\[(K-[0-9a-f]{8})\]", system)[:1],
        }]}


class SwrArm(rot.Arm):
    """An experiment arm whose replays run inside a checkout of each PR head."""

    snapshots: GitSnapshots = None

    def learn(self, event: dict) -> List[str]:
        """An interrupted event can resume without appending it a second time."""
        actions = []
        for position, lesson in enumerate(event.get("lessons") or []):
            if self.governor is None:
                existing = next((item for item in self.repo.list(rot.TENANT)
                                 if item.kind == lesson["kind"] and item.content == lesson["content"]
                                 and any(value.get("event") == event["index"] and
                                         value.get("lesson_index", 0) == position for value in item.evidence)), None)
                if existing:
                    actions.append("APPEND")
                    continue
            else:
                previous = next((row for row in self.repo.events(rot.TENANT, limit=1000000)
                                 if row["event"] == "admission"
                                 and row["detail"].get("candidate", {}).get("content") == lesson["content"]
                                 and any(value.get("event") == event["index"] and
                                         value.get("lesson_index", 0) == position for value in
                                         row["detail"].get("candidate", {}).get("evidence") or [])), None)
                if previous:
                    actions.append(previous["detail"]["action"])
                    continue
            actions.extend(super().learn({**event, "lessons": [lesson], "lesson_index": position}))
        return actions

    def _replay_factory(self, knowledge, name: str):
        skills = self.disk_skills + rot.skills_from_items(knowledge.items)
        return SnapshotReviewer(ReplayReviewer(
            self.client, self.settings.agent_token_budget,
            self.settings.agent_time_budget_seconds, name=name, knowledge=knowledge,
            skill_provider=lambda _tenant: skills,
        ), self.snapshots)

    def attribution_round(self, cases: List[dict], round_index: int) -> Dict[str, int]:
        counts = {"reviews": 0, "accepted": 0, "false_positive": 0, "failed": 0}
        lock = threading.Lock()

        def run(position_case):
            position, case = position_case
            lane = "stable" if position % 3 == 2 else "canary"
            task_id = "attr-%d-%d" % (round_index, position)
            try:
                with self.snapshots.checkout(case["repository"], case["head_commit"]) as root:
                    reviewer = AgenticReviewer(
                        _EvaluationTaskStore({"knowledge_lane": lane, "repository_root": root}),
                        self.client,
                        default_token_budget=self.settings.agent_token_budget,
                        default_time_budget=self.settings.agent_time_budget_seconds,
                        skill_provider=lambda _tenant: self.skills(self.repo.list(
                            rot.TENANT, statuses=("active", "probation") if lane == "canary" else ("active",))),
                        knowledge=self.governor,
                    )
                    findings = reviewer.review_with_context(
                        task_id, case["diff"], parse_unified_diff(case["diff"]),
                        repository=case["repository"],
                    )
            except Exception as exc:
                log("attribution review failed: %s" % str(exc)[:200])
                with lock:
                    counts["failed"] += 1
                return
            outcome = classify(case, findings)
            for index in outcome["hits"] + outcome["false_positives"]:
                category = "accepted" if index in outcome["hits"] else "false_positive"
                self.governor.record_feedback(
                    rot.TENANT, task_id, category, findings[index].to_dict(), "", learn=False,
                )
                with lock:
                    counts[category] += 1
            with lock:
                counts["reviews"] += 1

        with ThreadPoolExecutor(max_workers=self.args.workers) as pool:
            list(pool.map(run, list(enumerate(cases))))
        if counts["failed"] > len(cases) * rot.MAX_REPLAY_FAILURE_RATE:
            raise rot.ModelUnavailable("attribution round %d: %d of %d reviews failed" % (
                round_index, counts["failed"], len(cases),
            ))
        return counts


# ------------------------------------------------------------ train phase
def review_train(cases: List[dict], reviewer, cache_dir: str, workers: int,
                 borrow_cache: str = "", retry_attempts: int = 2,
                 retry_workers: int = 8) -> Dict[str, dict]:
    """Review each train PR once with no knowledge; cache outcomes per PR."""
    os.makedirs(cache_dir, exist_ok=True)
    borrowed = 0
    if borrow_cache:
        originals = {case["id"]: case for case in load_jsonl(DATASET)}
        for case in cases:
            target = os.path.join(cache_dir, case["id"] + ".json")
            source = os.path.join(borrow_cache, case["id"] + ".json")
            if os.path.exists(target) or not os.path.exists(source):
                continue
            original = originals.get(case["id"])
            if not original or any(original.get(key) != case.get(key) for key in
                                   ("head_commit", "diff", "expected_findings", "repository")):
                raise ValueError("borrowed cache input differs for %s" % case["id"])
            value = read_json(source, None)
            if not value or value.get("id") != case["id"]:
                raise ValueError("borrowed cache has wrong case id")
            findings = [finding_from_dict(item) for item in value["findings"]]
            with open(source, "rb") as handle:
                checksum = hashlib.sha256(handle.read()).hexdigest()
            write_json(target, {**value, "outcome": classify(case, findings),
                                "borrowed_from": os.path.abspath(source), "borrowed_sha256": checksum})
            borrowed += 1
    log("borrowed %d original knowledge-free train reviews" % borrowed)
    pending = [case for case in cases if not os.path.exists(os.path.join(cache_dir, case["id"] + ".json"))]
    log("train reviews: %d cached, %d to run" % (len(cases) - len(pending), len(pending)))
    failures = [0]
    lock = threading.Lock()

    def run(case):
        try:
            findings, summary = reviewer.review_case_detailed(case, parse_unified_diff(case["diff"]))
        except Exception as exc:
            with lock:
                failures[0] += 1
            log("train review %s failed: %s" % (case["id"], str(exc)[:200]))
            return
        execution = summary.get("execution") or {}
        write_json(os.path.join(cache_dir, case["id"] + ".json"), {
            "id": case["id"], "findings": [finding.to_dict() for finding in findings],
            "outcome": classify(case, findings),
            "tokens": execution.get("total_tokens", 0), "llm_calls": execution.get("llm_calls", 0),
        })

    batch = max(workers * 4, 1)
    for start in range(0, len(pending), batch):
        chunk = pending[start:start + batch]
        before = failures[0]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run, chunk))
        log("train reviews: %d/%d done" % (min(start + batch, len(pending)), len(pending)))
        if (failures[0] - before > len(chunk) * rot.MAX_REPLAY_FAILURE_RATE
                and len(chunk) >= max(16, workers)):
            raise rot.ModelUnavailable("train reviews failing wholesale (%d of %d)" % (failures[0] - before, len(chunk)))
    for attempt in range(1, retry_attempts + 1):
        missing = [case for case in cases if not os.path.exists(os.path.join(cache_dir, case["id"] + ".json"))]
        if not missing:
            break
        log("train reviews: retrying %d failed cases (attempt %d)" % (len(missing), attempt))
        with ThreadPoolExecutor(max_workers=max(1, min(8, retry_workers))) as pool:
            list(pool.map(run, missing))
    missing = [case for case in cases if not os.path.exists(os.path.join(cache_dir, case["id"] + ".json"))]
    if missing:
        raise rot.ModelUnavailable("train reviews: %d cases still failed; resume retries only missing cases" % len(missing))
    return {
        case["id"]: read_json(os.path.join(cache_dir, case["id"] + ".json"), None)
        for case in cases
    }


def finding_brief(value: dict) -> dict:
    return {key: value.get(key) for key in (
        "rule_id", "cwe", "severity", "title", "explanation", "path", "line", "evidence", "source",
    )}


def plan_events(cases: List[dict], reviews: Dict[str, dict], seed: int) -> List[dict]:
    """Real outcomes in PR order, with 20% noise and 10% contradictions mixed in."""
    rng = random.Random(seed)
    real = []
    for case in sorted(cases, key=lambda item: (item.get("created_at", ""), item["id"])):
        review = reviews.get(case["id"])
        if not review:
            continue
        labels = functional(case)
        for index in review["outcome"]["missed"]:
            label = labels[index]
            real.append({
                "type": "missed", "case_id": case["id"], "label_index": index, "procedure": rng.random() < 0.2,
                "finding": {
                    "path": label["path"], "line": label["start_line"], "start_line": label["start_line"],
                    "title": label["change_type"], "severity": label["severity"],
                    "rule_id": label["change_type"].split()[0], "explanation": label["description"],
                },
                "note": "Human reviewers raised this on the PR and it was fixed before merge: %s" % label["description"],
            })
        for index in review["outcome"]["false_positives"][:MAX_FP_EVENTS_PER_PR]:
            finding = review["findings"][index]
            real.append({
                "type": "false_positive", "case_id": case["id"], "finding": finding_brief(finding),
                "note": "Maintainers reviewed this PR and did not ask for this change; the finding "
                        "'%s' at %s:%s was not a real problem." % (
                            finding.get("title", ""), finding.get("path", ""), finding.get("line", 0)),
            })
    total = int(round(len(real) / 0.7))
    stressors = ["noise"] * int(round(total * 0.2)) + ["contradiction"] * (total - len(real) - int(round(total * 0.2)))
    events = list(real)
    for kind in stressors:
        events.insert(rng.randrange(len(events) // 10, len(events) + 1), {"type": kind})
    for index, event in enumerate(events):
        event["index"] = index
        if event["type"] == "noise":
            event["lessons"] = [{
                "kind": "rule", "role": rot.ROLES[rng.randrange(len(rot.ROLES))],
                "content": rot.NOISE_PREFIXES[rng.randrange(len(rot.NOISE_PREFIXES))]
                + rot.NOISE[rng.randrange(len(rot.NOISE))], "triggers": [],
            }]
    return events


def scope_events(events: List[dict], train: List[dict]) -> None:
    """Stressors inherit the source training repository, never a holdout scope."""
    by_case = {case["id"]: case for case in train}
    by_index = {event["index"]: event for event in events}
    current = min(train, key=lambda case: (case.get("created_at", ""), case["id"])) if train else None
    for event in events:
        source = by_case.get(event.get("case_id"))
        if event["type"] == "contradiction":
            target = by_index.get(event.get("target_index"), {})
            source = by_case.get(target.get("case_id"))
        if source:
            current = source
        if current is None:
            if event.get("lessons"):
                raise ValueError("scoped lesson has no training repository")
            continue
        event["repository"] = current["repository"]
        event["source_case_id"] = current["id"]
        for lesson in event.get("lessons") or []:
            lesson["scope"] = current["repository"]


def realize_events(events: List[dict], cases_by_id: Dict[str, dict], client, workers: int, seed: int) -> List[dict]:
    ledger = ExecutionLedger("swr-events")
    rng = random.Random(seed + 7)

    def call(role, system, payload, max_tokens):
        for attempt in range(3):
            try:
                return client.complete_json(role, system, json.dumps(payload, ensure_ascii=False), ledger, max_tokens=max_tokens)
            except Exception as exc:
                log("event generation retry %d for %s: %s" % (attempt + 1, role, str(exc)[:160]))
                time.sleep(5 * (attempt + 1))
        return {}

    def lessons(event):
        case = cases_by_id[event["case_id"]]
        finding = event["finding"]
        if event["type"] == "missed" and event.get("procedure"):
            reply = call("knowledge-procedure", rot.PROCEDURE_PROMPT, {
                "finding": finding, "reviewer_discussion": event["note"], "diff_excerpt": case["diff"][:4000],
            }, 900)
            name = re.sub(r"[^a-z0-9-]+", "-", str(reply.get("name", "")).lower()).strip("-")[:60]
            if name and reply.get("description") and reply.get("body"):
                return [{"kind": "skill", "role": "*", "name": name,
                         "content": str(reply["description"])[:600], "body": str(reply["body"])[:6000],
                         "triggers": []}]
        category = "missed_issue" if event["type"] == "missed" else "false_positive"
        return [{"kind": "rule", **lesson} for lesson in reflect_lessons(
            client, ledger, category, finding, event["note"], case["diff"][:6000],
        )]

    pending = [event for event in events if event["type"] in {"missed", "false_positive"} and "lessons" not in event]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for event, value in zip(pending, pool.map(lessons, pending)):
            event["lessons"] = value
    learned = [event for event in events if event["type"] == "missed" and event.get("lessons")
               and event["lessons"][0]["kind"] == "rule"]
    for event in events:
        if event["type"] != "contradiction" or "lessons" in event:
            continue
        earlier = [item for item in learned if item["index"] < event["index"]]
        if not earlier:
            event["lessons"] = []
            continue
        target = earlier[rng.randrange(len(earlier))]
        reply = call("knowledge-contradict", rot.CONTRADICT_PROMPT, {"lesson": target["lessons"][0]["content"]}, 200)
        content = " ".join(str(reply.get("content", "")).split())
        event["target_index"] = target["index"]
        event["lessons"] = [{
            "kind": "rule", "role": target["lessons"][0].get("role", "correctness-reliability"),
            "content": content, "triggers": list(target["lessons"][0].get("triggers") or []),
        }] if content else []
    return events


# ---------------------------------------------------------------- holdout
def holdout_report(rows_path: str) -> dict:
    return read_json(rows_path, None)


def subset_report(report: dict, ids: set) -> dict:
    rows = [row for row in report["case_results"] if row["id"] in ids]
    totals = ProductionEvaluationHarness._empty_totals()
    for row in rows:
        ProductionEvaluationHarness._accumulate(totals, row)
    return {"case_results": rows, "metrics": ProductionEvaluationHarness._metrics(totals)}


def compare(left: dict, right: dict) -> Dict[str, Any]:
    return paired_bootstrap_comparison(left, right, iterations=2000, metrics=rot.COMPARISON_METRICS)


def run_recurrence(args, train: List[dict], events: List[dict], arms: dict, base_factory) -> dict:
    """Four frozen reviewers revisit feedback PRs; no outcomes are learned here."""
    selection = sample_feedback(events, train, args.recurrence_limit, args.seed)
    by_id = {case["id"]: case for case in train}
    selected = [by_id[key] for key in selection["case_ids"]]
    directory = os.path.join(args.out, "recurrence_rows")
    os.makedirs(directory, exist_ok=True)
    items = {name: arm.live_items() for name, arm in arms.items()}
    items["governed_canary"] = arms["governed"].repo.list(rot.TENANT, statuses=("active", "probation"))
    configuration = {"selection": selection, "fake": args.fake,
                     "model": arms["governed"].client.model,
                     "snapshots": {name: [item.to_dict() for item in values] for name, values in items.items()}}
    fingerprint = hashlib.sha256(json.dumps(configuration, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    manifest_path = os.path.join(directory, "manifest.json")
    previous = read_json(manifest_path, None)
    if previous is not None and previous["fingerprint"] != fingerprint:
        raise ValueError("recurrence cache uses different feedback or knowledge snapshots")
    write_json(manifest_path, {"fingerprint": fingerprint, **configuration})
    targets = {}
    for case in selected:
        original = read_json(os.path.join(args.out, "train_reviews", case["id"] + ".json"), None)
        if not original:
            raise ValueError("recurrence requires the original train review for %s" % case["id"])
        targets[case["id"]] = recurrence_targets(case, selection["events"], original)
    write_json(os.path.join(directory, "targets.json"), targets)
    reviewers = {"baseline": base_factory("recurrence-baseline")}
    reviewers.update({name: arms["governed" if name == "governed_canary" else name]._replay_factory(
        StaticKnowledge(arms["governed" if name == "governed_canary" else name].assembler, values),
        "recurrence-%s" % name,
    ) for name, values in items.items()})
    reports = {}
    for name, reviewer in reviewers.items():
        cache = os.path.join(directory, name)
        os.makedirs(cache, exist_ok=True)
        rows = {case["id"]: read_json(os.path.join(cache, case["id"] + ".json"), None) for case in selected}

        def run(case):
            row = {"id": case["id"], "execution_success": False}
            try:
                findings, summary = reviewer.review_case_detailed(case, parse_unified_diff(case["diff"]))
                row.update({"execution_success": True, "metrics": measure_recurrence(targets[case["id"]], findings),
                            "findings": [finding.to_dict() for finding in findings],
                            "knowledge": summary.get("knowledge") or {}})
            except Exception as exc:
                row["error_type"] = type(exc).__name__
            write_json(os.path.join(cache, case["id"] + ".json"), row)
            return row

        for attempt in range(args.retry_failed + 1):
            pending = [case for case in selected if not rows[case["id"]] or not rows[case["id"]]["execution_success"]]
            if not pending:
                break
            log("recurrence %s: %d PRs, attempt %d" % (name, len(pending), attempt + 1))
            with ThreadPoolExecutor(max_workers=args.workers if attempt == 0 else args.retry_workers) as pool:
                for row in pool.map(run, pending):
                    rows[row["id"]] = row
        reports[name] = [rows[case["id"]] for case in selected]
    paired_ids = set.intersection(*(
        {row["id"] for row in rows if row["execution_success"]} for rows in reports.values()
    ))
    paired = {name: [row for row in rows if row["id"] in paired_ids] for name, rows in reports.items()}
    comparison = {}
    for left, right in (("baseline", "naive"), ("baseline", "governed"),
                        ("baseline", "governed_canary"), ("naive", "governed"),
                        ("governed", "governed_canary")):
        comparison["%s_minus_%s" % (right, left)] = paired_recurrence_bootstrap(
            paired[left], paired[right], seed=args.seed,
        )
    result = {"notice": RECURRENCE_NOTICE, "sampling_unit": "PR", "sample": {
        "selected_prs": len(selected), "paired_prs": len(paired_ids),
        "by_event_type": {name: len(ids) for name, ids in selection["by_type"].items()},
        "case_ids": selection["case_ids"], "paired_case_ids": sorted(paired_ids),
        "limit_per_event_type": args.recurrence_limit,
    }, "failures": {name: sum(not row["execution_success"] for row in rows) for name, rows in reports.items()},
              "metrics": {name: recurrence_rates(rows) for name, rows in paired.items()}, "comparisons": comparison}
    write_json(os.path.join(args.out, "recurrence.json"), result)
    if selected and len(paired_ids) < len(selected) * (1 - rot.MAX_REPLAY_FAILURE_RATE):
        raise rot.ModelUnavailable("recurrence: too many failed replays; cached successes can be resumed")
    return result


def refresh_reports(out_dir: str) -> None:
    """Refresh finished artifacts with no model calls, learning or DB writes."""
    state = read_json(os.path.join(out_dir, "state.json"), None)
    if not state or not state.get("finished_at"):
        raise ValueError("report-only requires a finished experiment")
    governance = read_governance(out_dir, rot.TENANT)
    if governance is None:
        raise ValueError("both experiment databases are required")
    state["governance"] = governance
    write_json(os.path.join(out_dir, "governance.json"), governance)
    write_json(os.path.join(out_dir, "state.json"), state)
    rot.write_reports(out_dir, state)
    write_summary(out_dir, state)
    if state.get("recurrence"):
        compression = read_json(os.path.join(ROOT, "evaluation_results", "admission_compress_check.json"), None)
        conflict = read_json(os.path.join(ROOT, "evaluation_results", "admission_conflict_check.json"), None)
        with open(os.path.join(out_dir, "NOTES.md"), "w", encoding="utf-8") as handle:
            handle.write(build_notes(state, compression, conflict))


# ------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--repos", default=REPOS)
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation_results", "swr_rot_improved"))
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--gate-cases", type=int, default=60)
    parser.add_argument("--holdout-limit", type=int, default=0, help="0 = all holdout cases")
    parser.add_argument("--train-limit", type=int, default=0, help="0 = all train cases")
    parser.add_argument("--train-cache", default="", help="read-only source of original knowledge-free train reviews")
    parser.add_argument("--repository-scoped", action="store_true", help="scope all lessons to their source training repository")
    parser.add_argument("--measure-every", type=int, default=25)
    parser.add_argument("--review-every", type=int, default=25)
    parser.add_argument("--review-batch", type=int, default=6)
    parser.add_argument("--maintain-every", type=int, default=200)
    parser.add_argument("--promote-min-exposures", type=int, default=3)
    parser.add_argument("--evict-min-exposures", type=int, default=12)
    parser.add_argument("--probation-max-exposures", type=int, default=16)
    parser.add_argument("--counterfactual-replay-limit", type=int, default=40,
                        help="maximum individual train replays before each maintenance (two per pair)")
    parser.add_argument("--counterfactual-cases-per-item", type=int, default=3)
    parser.add_argument("--source-validation", action="store_true", help="screen source feedback before fast gated promotion")
    parser.add_argument("--source-validation-replay-limit", type=int, default=120)
    parser.add_argument("--source-validation-side-cases", type=int, default=2)
    parser.add_argument("--recurrence", action="store_true", help="revisit previously missed/false-positive train feedback PRs")
    parser.add_argument("--recurrence-limit", type=int, default=80, help="maximum sampled PRs per feedback type (0..80)")
    parser.add_argument("--worktrees", default="C:/dj_wt/swr-new", help="short directory for PR checkouts")
    parser.add_argument("--retry-failed", type=int, default=2,
                        help="re-review holdout cases whose replay failed to execute, up to N times")
    parser.add_argument("--retry-workers", type=int, default=8)
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--background", action="store_true", help="run hidden and append stdout/stderr to OUT/run.log")
    parser.add_argument("--report-only", action="store_true", help="refresh finished summary/NOTES with no LLM calls")
    args = parser.parse_args()

    if not 1 <= args.workers <= 16 or not 1 <= args.retry_workers <= 8:
        parser.error("workers must be 1..16 and retry-workers 1..8")
    if args.counterfactual_replay_limit < 0 or not 0 <= args.counterfactual_cases_per_item <= 3:
        parser.error("counterfactual limit must be nonnegative and cases per item 0..3")
    if not 0 <= args.recurrence_limit <= 80:
        parser.error("recurrence limit must be 0..80")
    if args.source_validation_replay_limit < 0 or not 1 <= args.source_validation_side_cases <= 2:
        parser.error("source validation budget must be nonnegative and side cases 1..2")
    protected = [os.path.realpath(os.path.join(ROOT, "evaluation_results", name))
                 for name in ("swr_rot", "rot_stress", "rot_stress_v2", "swr_rot_temporal")]
    output = os.path.realpath(args.out)
    if any(output == path or output.startswith(path + os.sep) for path in protected):
        parser.error("output directory is a protected prior experiment; choose a new directory")

    os.makedirs(args.out, exist_ok=True)
    if args.report_only:
        refresh_reports(args.out)
        log("reports refreshed: %s" % args.out)
        return
    if args.background:
        command = [sys.executable, "-u", "-W", "ignore::SyntaxWarning", os.path.abspath(__file__)]
        command += [arg for arg in sys.argv[1:] if arg != "--background"]
        with open(os.path.join(args.out, "run.log"), "a", encoding="utf-8") as handle:
            process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=handle,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        write_json(os.path.join(args.out, "process.json"), {"pid": process.pid, "command": command})
        log("background pid=%d; log=%s" % (process.pid, os.path.join(args.out, "run.log")))
        return
    settings = Settings.from_env()
    if args.fake:
        client, model = FakeSwrClient(), "fake"
    else:
        config = settings.resolved_llm()
        client = JsonChatClient(
            str(config["base_url"]), str(config["api_key"]), str(config["model"]),
            str(config["provider"]), settings.timeout_seconds, dict(config.get("headers") or {}),
        )
        model = str(config["model"])

    cases = load_jsonl(args.dataset)
    temporal_path = os.path.splitext(args.dataset)[0] + "_summary.json"
    temporal = read_json(temporal_path, None)
    if temporal and temporal.get("protocol", "").startswith("within-repository"):
        with open(args.dataset, "rb") as handle:
            if hashlib.sha256(handle.read()).hexdigest() != temporal["dataset_sha256"]:
                raise ValueError("temporal dataset differs from its recorded boundaries")
        args.repository_scoped = True
    if args.train_cache and not args.fake:
        source_state = read_json(os.path.join(os.path.dirname(args.train_cache), "state.json"), {})
        if source_state.get("model") != model:
            raise ValueError("borrowed train reviews use a different model")
    by_split = {split: [case for case in cases if case["split"] == split] for split in ("train", "validation", "holdout")}
    train = sorted(by_split["train"], key=lambda case: case["id"])
    if args.train_limit:
        train = train[:args.train_limit]
    defects = sorted((case for case in by_split["validation"] if functional(case)), key=lambda case: case["id"])
    cleans = sorted((case for case in by_split["validation"] if not functional(case)), key=lambda case: case["id"])
    gate_count = min(args.gate_cases or len(by_split["validation"]), len(by_split["validation"]))
    half = gate_count // 2
    gate_cases = defects[:half] + cleans[:gate_count - min(half, len(defects))]
    taken = {case["id"] for case in gate_cases}
    gate_cases += [case for case in defects + cleans if case["id"] not in taken][:gate_count - len(gate_cases)]
    holdout = sorted(by_split["holdout"], key=lambda case: case["id"])
    if args.holdout_limit:
        holdout = holdout[:args.holdout_limit]
    recent_ids = {case["id"] for case in holdout if case.get("created_at", "") >= "2024"}
    holdout_by_id = {case["id"]: case for case in holdout}
    probes = [case for case in holdout if functional(case)][:5]
    cases_by_id = {case["id"]: case for case in cases}

    snapshots = GitSnapshots(args.repos, args.worktrees or os.path.join(args.out, "wt"))
    SwrArm.snapshots = snapshots
    registry = SkillRegistry(os.path.join(ROOT, "skills"))
    registry.reload()
    disk_skills = registry.agent_skills()

    state_path = os.path.join(args.out, "state.json")
    rows_dir = os.path.join(args.out, "holdout_rows")
    os.makedirs(rows_dir, exist_ok=True)
    state = read_json(state_path, {
        "next_event": 0, "measurements": [], "checkpoints": {}, "maintenance": [], "attribution": [],
        "admission_actions": {}, "comparisons": {}, "model": model, "seed": args.seed,
        "holdout_repeats": 1, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": {"train": len(train), "gate": len(gate_cases), "holdout": len(holdout),
                    "holdout_recent_2024": len(recent_ids)},
    })
    configuration = {"dataset_sha256": dataset_fingerprint(cases), "model": model, "fake": args.fake,
                     "seed": args.seed, "train_ids": [case["id"] for case in train],
                     "gate_ids": [case["id"] for case in gate_cases], "holdout_ids": [case["id"] for case in holdout],
                     "repository_scoped": args.repository_scoped,
                     "review_every": args.review_every, "review_batch": args.review_batch,
                     "maintain_every": args.maintain_every,
                     "promote_min_exposures": args.promote_min_exposures,
                     "evict_min_exposures": args.evict_min_exposures,
                     "probation_max_exposures": args.probation_max_exposures,
                     "counterfactual_replay_limit": args.counterfactual_replay_limit,
                     "counterfactual_cases_per_item": args.counterfactual_cases_per_item,
                     "source_validation": args.source_validation,
                     "source_validation_replay_limit": args.source_validation_replay_limit,
                     "source_validation_side_cases": args.source_validation_side_cases,
                     "recurrence": args.recurrence, "recurrence_limit": args.recurrence_limit,
                     "settings": {key: getattr(settings, key) for key in (
                         "agent_token_budget", "agent_time_budget_seconds", "knowledge_rule_budget_tokens",
                         "knowledge_skill_budget_tokens", "knowledge_skill_body_cap_tokens", "knowledge_skill_catalog_limit",
                         "knowledge_rule_max_tokens", "knowledge_similarity_threshold", "knowledge_duplicate_threshold",
                         "knowledge_half_life_days", "knowledge_gate_margin", "knowledge_gate_max_drop",
                         "knowledge_capacity_tokens_per_role")}}
    if state.get("run_configuration") and state["run_configuration"] != configuration:
        raise ValueError("resume configuration differs from this experiment; choose a new output directory")
    if state.get("model") != model:
        raise ValueError("resume model differs from the cached experiment")
    state["run_configuration"] = configuration
    state["training_transport"] = {"invalid_json_retries": 1, "evaluation_invalid_json_retries": 0,
                                   "failed_reply_usage_recorded": True}
    if temporal:
        state["temporal_split"] = temporal

    def save():
        governance = read_governance(args.out, rot.TENANT)
        if governance is not None:
            state["governance"] = governance
            write_json(os.path.join(args.out, "governance.json"), governance)
        state["runtime"] = {"python_version": sys.version.split()[0], "platform": sys.platform}
        write_json(state_path, state)
        rot.write_reports(args.out, state)
        write_summary(args.out, state)

    def base_reviewer(name):
        review_client = client
        if name == "train" and not args.fake:
            review_client = JsonChatClient(
                client.base_url, client.api_key, client.model, client.provider,
                client.timeout, client.extra_headers, json_retries=1,
            )
        return SnapshotReviewer(ReplayReviewer(
            review_client, settings.agent_token_budget, settings.agent_time_budget_seconds, name=name,
            skill_provider=lambda _tenant: disk_skills,
        ), snapshots)

    def retry_failed(reviewer, value: dict, label: str, path: str) -> dict:
        """Re-review cases whose replay did not execute (gateway timeouts, resets).

        Every arm gets the same treatment, rows keep their order for the paired
        bootstrap, and the pre-retry failure rate is kept for the report."""
        for attempt in range(1, args.retry_failed + 1):
            failed = [row["id"] for row in value["case_results"] if not row.get("execution_success")]
            if not failed:
                break
            log("holdout %s: retrying %d failed cases (attempt %d)" % (label, len(failed), attempt))
            report = ProductionEvaluationHarness(workers=args.retry_workers).run(
                reviewer, [holdout_by_id[case_id] for case_id in failed], name=label,
            )
            fresh = {row["id"]: row for row in report["case_results"] if row.get("execution_success")}
            rows = [fresh.get(row["id"], row) for row in value["case_results"]]
            retried = dict(value.get("retried") or {})
            for case_id in failed:
                retried[case_id] = retried.get(case_id, 0) + 1
            first_failure_rate = value.get("first_failure_rate", round(len(failed) / len(value["case_results"]), 4))
            value = {**subset_report({"case_results": rows}, {row["id"] for row in rows}),
                     "retried": retried, "first_failure_rate": first_failure_rate}
            write_json(path, value)
            log("holdout %s: after retry %d: %s" % (label, attempt, json.dumps(holdout_metrics(value))))
        return value

    def run_holdout(reviewer, label: str) -> dict:
        path = os.path.join(rows_dir, "%s.json" % label)
        cached = holdout_report(path)
        if cached:
            return retry_failed(reviewer, cached, label, path)
        log("holdout replay %s (%d cases)" % (label, len(holdout)))
        report = ProductionEvaluationHarness(workers=args.workers).run(reviewer, holdout, name=label)
        if report["metrics"]["failure_rate"] > rot.MAX_REPLAY_FAILURE_RATE:
            errors = sorted({str(row.get("error") or "")[:200] for row in report["case_results"] if row.get("error")})
            raise rot.ModelUnavailable("holdout %s failed for %.0f%% of cases: %s" % (
                label, 100 * report["metrics"]["failure_rate"], errors[:3]))
        value = {"case_results": report["case_results"], "metrics": report["metrics"]}
        write_json(path, value)
        log("holdout %s: %s" % (label, json.dumps(holdout_metrics(report))))
        return retry_failed(reviewer, value, label, path)

    save()
    # 1. Baseline: holdout with no learned knowledge (shared starting point).
    baseline = run_holdout(base_reviewer("baseline"), "baseline")
    state["checkpoints"]["0"] = {name: holdout_metrics(baseline) for name in ("naive", "governed")}
    save()

    # 2. Train reviews -> real outcomes -> event stream.
    events_path = os.path.join(args.out, "events.json")
    events = read_json(events_path, None)
    if events is None:
        reviews = review_train(train, base_reviewer("train"), os.path.join(args.out, "train_reviews"), args.workers,
                               args.train_cache, args.retry_failed, args.retry_workers)
        events = plan_events(train, reviews, args.seed)
        log("planned %d events: %s" % (len(events), json.dumps({
            kind: sum(event["type"] == kind for event in events)
            for kind in ("missed", "false_positive", "noise", "contradiction")
        })))
        events = realize_events(events, cases_by_id, client, args.workers, args.seed)
        if args.repository_scoped:
            scope_events(events, train)
        write_json(events_path, events)
    state["events_total"] = len(events)
    state["event_mix"] = {
        kind: sum(event["type"] == kind for event in events)
        for kind in ("missed", "false_positive", "noise", "contradiction")
    }

    # 3. Replay the stream into both arms.
    arms = {
        name: SwrArm(name, args.out, client, settings, disk_skills, gate_cases if name == "governed" else [], args)
        for name in ("naive", "governed")
    }
    def measure(count: int):
        for name, arm in arms.items():
            row = {"event": count, "arm": name, **arm.probe(probes)}
            state["measurements"] = [
                item for item in state["measurements"] if not (item["event"] == count and item["arm"] == name)
            ] + [row]
        log("measured @%d: %s" % (count, " | ".join(
            "%s %s tok/call, %s live tok" % (row["arm"], row["instruction_tokens_per_call"], row["live_knowledge_tokens"])
            for row in state["measurements"] if row["event"] == count
        )))

    def maintain(count: int):
        if any(row["event"] == count for row in state["maintenance"]):
            return
        log("maintenance @%d" % count)
        if args.source_validation:
            originals = {case["id"]: read_json(os.path.join(args.out, "train_reviews", case["id"] + ".json"), None) for case in train}
            validation = arms["governed"].governor.source_pr_validation_round(
                rot.TENANT, train, events, originals, arms["governed"]._replay_factory,
                replay_limit=args.source_validation_replay_limit, side_cases=args.source_validation_side_cases,
                workers=args.workers, round_id=str(count))
            state.setdefault("source_validation", []).append({"event": count, **validation})
            save()
            log("source validation @%d: %s" % (count, json.dumps(validation)))
        counterfactual = arms["governed"].governor.counterfactual_round(
            rot.TENANT, train, arms["governed"]._replay_factory, classify,
            replay_limit=args.counterfactual_replay_limit,
            max_cases_per_item=args.counterfactual_cases_per_item,
            workers=args.workers, round_id=str(count),
        )
        state.setdefault("counterfactual", []).append({"event": count, **counterfactual})
        save()
        log("counterfactual @%d: %s" % (count, json.dumps(counterfactual)))
        if counterfactual["failed"] > counterfactual["pairs"] * rot.MAX_REPLAY_FAILURE_RATE:
            raise rot.ModelUnavailable("counterfactual round %d: paired train replays failed" % count)
        summary = arms["governed"].governor.maintain(rot.TENANT, force_dream=True)
        promotion = summary.get("promotion") or {}
        dream_release = summary["dream"].get("release") or {}
        failures = set((dream_release.get("gate") or {}).get("failures") or []) | set(
            (promotion.get("gate") or {}).get("failures") or [])
        if "execution_success_rate" in failures:
            raise rot.ModelUnavailable("maintenance @%d: gate replays failed to execute" % count)
        compact = {
            "dream": {key: value for key, value in summary["dream"].items() if key not in {"merges", "drops"}},
            "promotion": {key: value for key, value in promotion.items() if key != "gate"},
            "promotion_gate_failures": (promotion.get("gate") or {}).get("failures"),
            "dream_release": dream_release.get("status"),
            "dream_gate_failures": (dream_release.get("gate") or {}).get("failures"),
            "live_tokens_before": summary["live_tokens_before"], "live_tokens_after": summary["live_tokens_after"],
        }
        state["maintenance"].append({"event": count, "summary": compact})
        log("maintenance @%d: %s" % (count, json.dumps(compact, ensure_ascii=False)[:1500]))

    if state["next_event"] == 0:
        measure(0)
        save()

    def event_stages(count: int):
        if (args.review_every and count % args.review_every == 0
                and not any(row["event"] == count for row in state["attribution"])):
            batch_rng = random.Random(args.seed + count)
            batch = batch_rng.sample(train, min(args.review_batch, len(train)))
            result = arms["governed"].attribution_round(batch, count)
            state["attribution"].append({"event": count, **result})
            save()
            log("attribution @%d: %s" % (count, json.dumps(result)))
        if args.maintain_every and count % args.maintain_every == 0:
            maintain(count)
        if args.measure_every and count % args.measure_every == 0:
            measure(count)

    if state.get("pending_event"):
        event_stages(state["pending_event"])
        state.pop("pending_event", None)
        save()
    while state["next_event"] < len(events):
        event = events[state["next_event"]]
        for name, arm in arms.items():
            for action in arm.learn(event):
                if name == "governed":
                    state["admission_actions"][action] = state["admission_actions"].get(action, 0) + 1
        count = state["next_event"] + 1
        state["next_event"] = count
        state["pending_event"] = count
        save()
        event_stages(count)
        state.pop("pending_event", None)
        save()
    if not any(item["event"] == len(events) for item in state["maintenance"]):
        maintain(len(events))
    measure(len(events))
    save()

    # 4. Final holdout for both arms and paired comparisons.
    final = str(len(events))
    reports = {"baseline": baseline}
    for name, arm in arms.items():
        reports[name] = run_holdout(arm.snapshot_reviewer("final"), "final-%s" % name)
    state["checkpoints"][final] = {name: holdout_metrics(reports[name]) for name in ("naive", "governed")}
    state["holdout_failures"] = {
        name: {"first": report.get("first_failure_rate", report["metrics"]["failure_rate"]),
               "after_retry": report["metrics"]["failure_rate"], "retried": len(report.get("retried") or {})}
        for name, report in reports.items()
    }
    # Sensitivity check: only PRs every arm actually reviewed.
    executed_ids = set.intersection(*(
        {row["id"] for row in report["case_results"] if row.get("execution_success")} for report in reports.values()
    ))
    state["dataset"]["holdout_executed_all"] = len(executed_ids)
    for scope, ids in (("all", None), ("recent_2024", recent_ids), ("executed_all", executed_ids)):
        pick = (lambda report: report) if ids is None else (lambda report, ids=ids: subset_report(report, ids))
        state.setdefault("comparisons_by_scope", {})[scope] = {
            "governed_minus_naive": compare(pick(reports["naive"]), pick(reports["governed"])),
            "naive_minus_baseline": compare(pick(reports["baseline"]), pick(reports["naive"])),
            "governed_minus_baseline": compare(pick(reports["baseline"]), pick(reports["governed"])),
        }
        state.setdefault("holdout_by_scope", {})[scope] = {
            name: {key: pick(report)["metrics"].get(key) for key in QUALITY + (
                "average_instruction_tokens_per_call", "average_total_tokens_per_pr",
            )} for name, report in reports.items()
        }
    state["comparisons"][final] = state["comparisons_by_scope"]["all"]["governed_minus_naive"]
    if args.recurrence:
        state["recurrence"] = run_recurrence(args, train, events, arms, base_reviewer)
    state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save()
    if args.recurrence:
        refresh_reports(args.out)
    log("done: %s" % os.path.join(args.out, "summary.md"))


def write_summary(out_dir: str, state: dict) -> None:
    lines = ["# SWR-Bench 知识治理评测", "",
             "- 实验完成时间：%s" % state.get("finished_at", "尚未完成"),
             "- 配置模型：%s；数据集：%s" % (state.get("model"), json.dumps(state.get("dataset"))),
             "- 学习事件：%s %s" % (state.get("events_total", "?"), json.dumps(state.get("event_mix", {}))),
             "- governed 写入动作：%s" % json.dumps(state.get("admission_actions", {})),
             "- holdout 执行失败率（首轮、补跑后）：%s" % json.dumps(state.get("holdout_failures", {})),
             "", "主要质量结论使用 executed_all（所有比较组共同成功的 PR）；all 保留完整样本结果，recent_2024 为敏感性检查。比较名 A_minus_B 表示 A 减 B。", ""]
    scopes = state.get("holdout_by_scope") or {}
    ordered_scopes = sorted(scopes, key=lambda scope: scope != "executed_all")
    for scope in ordered_scopes:
        values = scopes[scope]
        lines += ["## Holdout (%s)" % scope, "", "| arm | " + " | ".join(QUALITY) + " | instr. tokens/call | tokens/PR |",
                  "|---|" + "---:|" * (len(QUALITY) + 2)]
        for name, metrics in values.items():
            lines.append("| %s | %s | %s | %s |" % (
                name, " | ".join(str(metrics.get(key)) for key in QUALITY),
                metrics.get("average_instruction_tokens_per_call"), metrics.get("average_total_tokens_per_pr"),
            ))
        lines.append("")
        for pair, comparison in (state.get("comparisons_by_scope") or {}).get(scope, {}).items():
            lines.append("- %s: %s" % (pair, "; ".join(
                "%s %+.4f [%s, %s]" % (metric, value["delta"], value["ci95"][0], value["ci95"][1])
                for metric, value in comparison.items()
            )))
        lines.append("")
    if state.get("temporal_split"):
        lines += ["## 同仓库时间划分", "", "各仓库按 created_at 排序，前 60% 训练、接下来 10% 门禁、最后 30% holdout；PR 不跨划分。", "",
                  "| 仓库 | 划分 | PR 数 | 最早创建时间 | 最晚创建时间 |", "|---|---|---:|---|---|"]
        for repository, groups in state["temporal_split"]["repositories"].items():
            for split in ("train", "validation", "holdout"):
                values = groups[split]
                lines.append("| %s | %s | %s | %s | %s |" % (
                    repository, split, values["count"], values["first_created_at"], values["last_created_at"]))
        lines.append("")
    recurrence = state.get("recurrence")
    if recurrence:
        sample = recurrence["sample"]
        lines += ["## 已反馈问题纠正率", "", RECURRENCE_NOTICE, "",
                  "- 抽样 %d 个不同 PR，四组共同成功 %d 个；按反馈类型抽样：%s。完整 PR 清单见 [recurrence.json](recurrence.json)。" % (
                      sample["selected_prs"], sample["paired_prs"], json.dumps(sample["by_event_type"], ensure_ascii=False)),
                  "- 各通道最终失败 PR 数：%s。" % json.dumps(recurrence.get("failures", {})), "",
                  "| 通道 | 误报复发率 | 漏报找回率 | 正确发现保留率 |", "|---|---:|---:|---:|"]
        for name, values in recurrence["metrics"].items():
            lines.append("| %s | %s |" % (name, " | ".join(
                "%s (%d/%d)" % (values[metric]["rate"], values[metric]["numerator"], values[metric]["denominator"])
                for metric in RECURRENCE_METRICS)))
        lines += ["", "误报复发率越低越好，其余两项越高越好。配对 bootstrap 按 PR 重采样，95% CI：", ""]
        for pair, values in recurrence["comparisons"].items():
            lines.append("- %s：%s" % (pair, json.dumps(values, ensure_ascii=False)))
        lines.append("")
    if state.get("governance"):
        lines += governance_markdown(state["governance"])
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    try:
        main()
    except rot.ModelUnavailable as exc:
        log("stopped: model unavailable: %s" % exc)
        sys.exit(3)
