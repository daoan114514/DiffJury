"""Context-rot stress test: naive append vs. governed knowledge.

Both arms consume the *same* pre-generated stream of learning events built
from the train repositories (acme/service-01..06):

* missed issue (50%)  - a labelled defect the reviewer "missed"; an LLM
                        reflects it into a lesson (1 in 5 becomes a skill)
* false positive (20%)- a plausible wrong finding on a clean change; the lesson
                        teaches the critic when not to report
* noise (20%)         - generic, low-value advice of the kind agents write
* contradiction (10%) - a plausible but wrong lesson contradicting an earlier one

The naive arm appends every lesson to its prompt (and every procedure as a new
skill). The governed arm runs the product pipeline: admission gate, budgeted
assembly, outcome attribution from periodic train reviews, lifecycle eviction,
Dream consolidation and the replay release gate (validation repos 07-08).
Prompt size is probed every ``--measure-every`` events; review quality is
measured on the holdout repos (09-10) at each checkpoint.

The run is resumable: rerun the same command and it continues from state.json.

    python scripts/run_rot_stress_test.py --out evaluation_results/rot_stress
    python scripts/run_rot_stress_test.py --fake --events 60 --out /tmp/rot  # smoke
"""
import argparse
import csv
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from diffjury.admission import AdmissionGate  # noqa: E402
from diffjury.agentic_core import AgenticReviewer, CRITIC_PROMPT, LEAD_PROMPT, RELIABILITY_PROMPT, SECURITY_PROMPT  # noqa: E402
from diffjury.assembler import ContextAssembler, StaticKnowledge, TaskSignals  # noqa: E402
from diffjury.config import Settings  # noqa: E402
from diffjury.context_manager import estimate_tokens  # noqa: E402
from diffjury.diff_parser import parse_unified_diff  # noqa: E402
from diffjury.evaluation_harness import load_replay_cases, one_to_one_match  # noqa: E402
from diffjury.evaluation_v2 import (  # noqa: E402
    ProductionEvaluationHarness, ReplayReviewer, _EvaluationTaskStore, paired_bootstrap_comparison,
)
from diffjury.knowledge import KnowledgeItem  # noqa: E402
from diffjury.knowledge_service import KnowledgeGovernor, reflect_lessons  # noqa: E402
from diffjury.lifecycle import KnowledgeLifecycle, LifecyclePolicy  # noqa: E402
from diffjury.llm import JsonChatClient  # noqa: E402
from diffjury.release_gate import ReleaseGate  # noqa: E402
from diffjury.skills import AgentSkill, SkillRegistry  # noqa: E402
from diffjury.store import TaskStore  # noqa: E402
from diffjury.telemetry import ExecutionLedger  # noqa: E402


TENANT = "default"
TRAIN_REPOS = {"acme/service-%02d" % index for index in range(1, 7)}
VALIDATION_REPOS = {"acme/service-07", "acme/service-08"}
HOLDOUT_REPOS = {"acme/service-09", "acme/service-10"}
EVENT_MIX = (("missed", 0.5), ("false_positive", 0.2), ("noise", 0.2), ("contradiction", 0.1))
WRONG_CLAIMS = (
    ("CWE-89", "SQL injection", "query text is built from request data"),
    ("CWE-78", "OS command injection", "a shell command is built from input"),
    ("CWE-798", "Hard-coded credential", "a literal looks like a secret"),
    ("CWE-400", "Unbounded resource use", "the new limit could be bypassed"),
    ("CWE-703", "Unchecked error", "an exception path is not handled"),
)
NOISE = (
    "Always review the code carefully before reporting.",
    "Make sure to consider security in every change.",
    "Be thorough and check all edge cases in the diff.",
    "Prefer high quality findings over low quality findings.",
    "Remember that correctness matters for production code.",
    "Look closely at every added line for possible problems.",
    "Consider performance, reliability and maintainability together.",
    "Think step by step before deciding on a finding.",
    "Report issues clearly with helpful explanations.",
    "Double-check your reasoning before you finish the review.",
    "Pay attention to naming, style and readability as well.",
    "Keep the user experience and product goals in mind.",
)
NOISE_PREFIXES = ("", "Important: ", "Note: ", "Reminder: ", "Tip: ")
ROLES = ("security", "correctness-reliability", "critic", "lead")

PROCEDURE_PROMPT = """Turn one missed code-review defect into a reusable Agent Skill for an
automated reviewer. Return JSON only: {"name":"kebab-case-name","description":"one sentence:
what it reviews and when to use it","body":"markdown procedure of 120-250 words: what to
look for, how to confirm it from the diff, how to report it"}"""

CONTRADICT_PROMPT = """You are generating adversarial test data for a knowledge base.
Write one plausible-sounding but WRONG code-review lesson (one sentence, at most 40 words)
that contradicts the given lesson, e.g. by claiming the pattern is safe or should not be
reported. Return JSON only: {"content":"..."}"""


# --------------------------------------------------------------------- utils
def write_json(path: str, value: Any) -> None:
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
    os.replace(temporary, path)


def read_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


class ModelUnavailable(SystemExit):
    """Replays are failing wholesale (quota, outage); stop rather than record zeros."""


MAX_REPLAY_FAILURE_RATE = 0.5


def log(message: str) -> None:
    print("[%s] %s" % (time.strftime("%H:%M:%S"), message), flush=True)


def added_line(case: dict, path: str, line: int) -> str:
    for item in parse_unified_diff(case["diff"]).added_lines:
        if item.path == path and item.line == line:
            return item.content.strip()
    return ""


# ------------------------------------------------------------------ fake LLM
class FakeStressClient:
    """Deterministic offline stand-in so the full pipeline can be smoke-tested."""

    provider, model = "fake", "fake-stress"
    RISKY = {
        "eval(": ("CWE-95", "SEC-EVAL"), "pickle.loads(": ("CWE-502", "SEC-DESERIALIZE"),
        "shell=True": ("CWE-78", "SEC-CMD"), "execute(f": ("CWE-89", "SEC-SQL"),
    }

    def complete_json(self, role, system, user, ledger=None, max_tokens=None):
        if ledger:
            ledger.record_model(role, self.provider, self.model, {
                "prompt_tokens": estimate_tokens(system) + estimate_tokens(user),
                "completion_tokens": 20,
            }, 1)
        payload = json.loads(user)
        if role == "knowledge-reflection":
            finding = payload.get("finding") or {}
            role_name = "critic" if payload["category"] == "false_positive" else "security"
            return {"lessons": [{"role": role_name, "content": payload["reviewer_note"][:200],
                                 "triggers": ["cwe:%s" % finding.get("cwe", "")]}]}
        if role == "knowledge-admission":
            similar = payload.get("similar_items") or []
            content = payload["candidate"]["content"].lower()
            if any(phrase.lower().rstrip(".") in content for phrase in NOISE):
                return {"decision": "REJECT", "reason": "generic"}
            if similar and similar[0]["similarity"] > 0.6:
                return {"decision": "UPVOTE", "target_id": similar[0]["id"]}
            return {"decision": "ADD"}
        if role in {"knowledge-dream", "knowledge-compress"}:
            return {}
        if role == "knowledge-procedure":
            finding = payload["finding"]
            slug = re.sub(r"[^a-z0-9]+", "-", str(finding.get("rule_id", "x")).lower()).strip("-")
            return {"name": "review-%s" % slug, "description": "Review %s." % finding.get("cwe"),
                    "body": "Look for %s on added lines and report it." % finding.get("cwe")}
        if role == "knowledge-contradict":
            return {"content": "Do not report: " + payload["lesson"][:120]}
        task = json.loads(payload["task"])
        if role == "lead":
            if task.get("phase") == "delegate":
                return {"action": "final", "delegations": [], "risk_level": "normal"}
            return {"action": "final", "accepted_finding_indices": list(
                range(len(task.get("candidate_findings") or []))
            )}
        if role == "critic":
            return {"action": "final", "decisions": [
                {"finding_index": index, "accepted": True}
                for index in range(len(task.get("candidates") or []))
            ]}
        findings = []
        if role == "security":
            for line in str(task.get("diff", "")).splitlines():
                match = re.match(r"\+\s*(\d+) \| (.*)", line)
                if not match:
                    continue
                for marker, (cwe, rule_id) in self.RISKY.items():
                    if marker in match.group(2):
                        findings.append({
                            "cwe": cwe, "rule_id": rule_id, "severity": "high",
                            "title": rule_id, "explanation": "risky call", "path": task["lead_assignment"]["files"][0],
                            "line": int(match.group(1)), "evidence": match.group(2).strip(),
                            "call_chain": [{"path": "x", "line": 1, "symbol": "x"}],
                            "fix": "fix", "test": "test", "confidence": 0.9,
                            "applied_rules": re.findall(r"\[(K-[0-9a-f]{8})\]", system)[:1],
                        })
        return {"action": "final", "findings": findings}


# ------------------------------------------------------------------- events
class EventGenerator:
    def __init__(self, client, cases: List[dict], seed: int, workers: int):
        self.client = client
        self.cases = cases
        self.rng = random.Random(seed)
        self.workers = workers
        self.ledger = ExecutionLedger("rot-stress-events")

    def plan(self, count: int) -> List[dict]:
        risky = [
            (case, finding) for case in self.cases for finding in case["expected_findings"]
        ]
        clean = [case for case in self.cases if not case["expected_findings"]]
        kinds = [kind for kind, _weight in EVENT_MIX]
        weights = [weight for _kind, weight in EVENT_MIX]
        plan = []
        for index in range(count):
            kind = self.rng.choices(kinds, weights)[0]
            event: Dict[str, Any] = {"index": index, "type": kind}
            if kind == "missed":
                case, finding = risky[self.rng.randrange(len(risky))]
                event.update(case_id=case["id"], finding=dict(finding),
                             procedure=self.rng.random() < 0.2)
            elif kind == "false_positive":
                case = clean[self.rng.randrange(len(clean))]
                lines = parse_unified_diff(case["diff"]).added_lines
                line = lines[self.rng.randrange(len(lines))]
                cwe, title, reason = WRONG_CLAIMS[self.rng.randrange(len(WRONG_CLAIMS))]
                event.update(case_id=case["id"], finding={
                    "path": line.path, "start_line": line.line, "cwe": cwe,
                    "rule_id": "CLAIM-" + cwe, "title": title, "severity": "high",
                    "evidence": line.content.strip(), "explanation": reason,
                })
            elif kind == "noise":
                event["lessons"] = [{
                    "kind": "rule", "role": ROLES[self.rng.randrange(len(ROLES))],
                    "content": NOISE_PREFIXES[self.rng.randrange(len(NOISE_PREFIXES))]
                    + NOISE[self.rng.randrange(len(NOISE))], "triggers": [],
                }]
            plan.append(event)
        return plan

    def realize(self, events: List[dict]) -> List[dict]:
        by_case = {case["id"]: case for case in self.cases}
        pending = [event for event in events if "lessons" not in event and event["type"] != "contradiction"]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for event, lessons in zip(pending, pool.map(
                lambda item: self._lessons(item, by_case[item["case_id"]]), pending,
            )):
                event["lessons"] = lessons
        learned = []
        for event in events:
            if event["type"] == "contradiction":
                event["target_index"] = learned[self.rng.randrange(len(learned))]["index"] if learned else None
            elif event["type"] == "missed" and event.get("lessons"):
                learned.append(event)
        contradictions = [event for event in events if event["type"] == "contradiction"]
        by_index = {event["index"]: event for event in events}
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for event, lessons in zip(contradictions, pool.map(
                lambda item: self._contradiction(item, by_index), contradictions,
            )):
                event["lessons"] = lessons
        return events

    def _lessons(self, event: dict, case: dict) -> List[dict]:
        finding = dict(event["finding"])
        finding["line"] = finding.get("start_line", finding.get("line", 0))
        code = added_line(case, finding["path"], finding["line"]) or finding.get("evidence", "")
        if event["type"] == "missed" and event.get("procedure"):
            reply = self._call("knowledge-procedure", PROCEDURE_PROMPT, {
                "finding": finding, "code": code, "diff_excerpt": case["diff"][:4000],
            }, 900)
            name = re.sub(r"[^a-z0-9-]+", "-", str(reply.get("name", "")).lower()).strip("-")[:60]
            if name and reply.get("description") and reply.get("body"):
                return [{
                    "kind": "skill", "role": "*", "name": name,
                    "content": str(reply["description"])[:600], "body": str(reply["body"])[:6000],
                    "triggers": ["cwe:%s" % finding.get("cwe", "")],
                }]
        if event["type"] == "missed":
            category = "missed_issue"
            note = "The review missed %s (%s, %s) at %s:%s: `%s`" % (
                finding.get("cwe"), finding.get("rule_id"), finding.get("severity"),
                finding["path"], finding["line"], code,
            )
        else:
            category = "false_positive"
            note = "False positive: the review reported %s at %s:%s on `%s`, but this added line is safe." % (
                finding.get("title"), finding["path"], finding["line"], code,
            )
        lessons = reflect_lessons(self.client, self.ledger, category, finding, note, case["diff"][:6000])
        return [{"kind": "rule", **lesson} for lesson in lessons]

    def _contradiction(self, event: dict, by_index: Dict[int, dict]) -> List[dict]:
        target = by_index.get(event.get("target_index"))
        if not target or not target.get("lessons"):
            return []
        lesson = target["lessons"][0]
        reply = self._call("knowledge-contradict", CONTRADICT_PROMPT, {
            "lesson": lesson["content"],
        }, 200)
        content = " ".join(str(reply.get("content", "")).split())
        if not content:
            return []
        role = lesson.get("role", "security") if lesson.get("kind") == "rule" else "security"
        return [{"kind": "rule", "role": role, "content": content,
                 "triggers": list(lesson.get("triggers") or [])}]

    def _call(self, role: str, system: str, payload: dict, max_tokens: int) -> dict:
        for attempt in range(3):
            try:
                return self.client.complete_json(
                    role, system, json.dumps(payload, ensure_ascii=False), self.ledger,
                    max_tokens=max_tokens,
                )
            except Exception as exc:  # transient relay errors
                log("event generation retry %d for %s: %s" % (attempt + 1, role, str(exc)[:200]))
                time.sleep(5 * (attempt + 1))
        return {}


# --------------------------------------------------------------------- arms
def skills_from_items(items: List[KnowledgeItem]) -> List[AgentSkill]:
    return [
        AgentSkill(
            name=item.name, description=item.content, instructions=item.body,
            content=item.instruction_text(), source="learned", version=str(item.version),
        )
        for item in items
        if item.kind == "skill" and item.source != "disk" and item.status != "archived" and item.body
    ]


class Arm:
    def __init__(self, name: str, out_dir: str, client, settings: Settings, disk_skills,
                 gate_cases: List[dict], args):
        self.name = name
        self.client = client
        self.settings = settings
        self.disk_skills = list(disk_skills)
        self.args = args
        self.store = TaskStore(os.path.join(out_dir, "%s.db" % name))
        self.repo = self.store.knowledge
        if name == "naive":
            self.assembler = ContextAssembler(
                self.repo, rule_budget=10 ** 9, skill_budget=10 ** 9, skill_body_cap=10 ** 9,
                catalog_limit=10 ** 6, min_relevance=-1.0,
            )
            self.governor = None
        else:
            self.assembler = ContextAssembler(
                self.repo, rule_budget=settings.knowledge_rule_budget_tokens,
                skill_budget=settings.knowledge_skill_budget_tokens,
                skill_body_cap=settings.knowledge_skill_body_cap_tokens,
                catalog_limit=settings.knowledge_skill_catalog_limit,
                half_life_days=settings.knowledge_half_life_days,
            )
            admission = AdmissionGate(
                self.repo, client, max_rule_tokens=settings.knowledge_rule_max_tokens,
                max_skill_tokens=settings.knowledge_skill_body_cap_tokens,
                similarity_threshold=settings.knowledge_similarity_threshold,
                dup_threshold=settings.knowledge_duplicate_threshold,
            )
            lifecycle = KnowledgeLifecycle(self.repo, LifecyclePolicy(
                capacity_tokens_per_role=settings.knowledge_capacity_tokens_per_role,
                half_life_days=settings.knowledge_half_life_days,
                promote_min_exposures=args.promote_min_exposures,
                evict_min_exposures=args.evict_min_exposures,
                probation_max_exposures=args.probation_max_exposures,
            ))
            gate = ReleaseGate(
                gate_cases, self._replay_factory, self.assembler,
                margin=settings.knowledge_gate_margin, workers=args.workers,
                max_point_drop=settings.knowledge_gate_max_drop, bootstrap_iterations=1000,
                retry_attempts=getattr(args, "retry_failed", 0), retry_workers=getattr(args, "retry_workers", 8),
            ) if gate_cases else None
            self.governor = KnowledgeGovernor(
                self.repo, self.assembler, admission, client, background=False,
                lifecycle=lifecycle, release_gate=gate, maintain_every_reviews=0,
                dream_token_threshold=0,
                source_validation_enabled=getattr(args, "source_validation", False),
            )

    # Naive appends; governed admits.
    def learn(self, event: dict) -> List[str]:
        actions = []
        for position, lesson in enumerate(event.get("lessons") or []):
            evidence = [{"event": event["index"], "type": event["type"], "case": event.get("case_id", ""),
                         "lesson_index": event.get("lesson_index", position)}]
            if self.governor is None:
                name = lesson.get("name", "")
                if lesson["kind"] == "skill":
                    taken = {item.name for item in self.repo.list(TENANT, kind="skill")}
                    base, suffix = name, 2
                    while name in taken:
                        name, suffix = "%s-%d" % (base, suffix), suffix + 1
                self.repo.create(KnowledgeItem(
                    tenant_id=TENANT, kind=lesson["kind"], role=lesson.get("role", "*"),
                    scope=str(lesson.get("scope") or "*"),
                    name=name, content=lesson["content"], body=lesson.get("body", ""),
                    triggers=lesson.get("triggers") or [], evidence=evidence,
                    status="active", source=event["type"],
                ))
                actions.append("APPEND")
            else:
                decision = self.governor.admit(TENANT, {**lesson, "evidence": evidence, "source": event["type"]})
                actions.append(decision.action)
        return actions

    def live_items(self) -> List[KnowledgeItem]:
        statuses = ("active", "probation") if self.governor is None else ("active",)
        return self.repo.list(TENANT, statuses=statuses)

    def skills(self, items: Optional[List[KnowledgeItem]] = None) -> List[AgentSkill]:
        return self.disk_skills + skills_from_items(self.live_items() if items is None else items)

    def _replay_factory(self, knowledge: StaticKnowledge, name: str):
        skills = self.disk_skills + skills_from_items(knowledge.items)
        return ReplayReviewer(
            self.client, self.settings.agent_token_budget,
            self.settings.agent_time_budget_seconds, name=name, knowledge=knowledge,
            skill_provider=lambda _tenant: skills,
        )

    def snapshot_reviewer(self, label: str):
        items = self.live_items()
        return self._replay_factory(StaticKnowledge(self.assembler, items), "%s-%s" % (self.name, label))

    # Probe: prompt size each role would pay on a fixed set of holdout diffs (no LLM).
    def probe(self, probes: List[dict]) -> Dict[str, Any]:
        items = self.live_items()
        knowledge = StaticKnowledge(self.assembler, items)
        bases = {
            "lead": LEAD_PROMPT, "security": SECURITY_PROMPT, "critic": CRITIC_PROMPT,
            "correctness-reliability": RELIABILITY_PROMPT,
        }
        per_call, catalog_tokens = [], []
        skills = self.skills(items)
        for case in probes:
            parsed = parse_unified_diff(case["diff"])
            view = knowledge.prepare(TENANT, case["repository"], "stable", TaskSignals.from_review(
                case["diff"], parsed.files, case["repository"],
            ))
            for role, base in bases.items():
                per_call.append(estimate_tokens(base) + view.rules_for(role).tokens)
            catalog = view.skill_catalog(skills)
            catalog_tokens.append(estimate_tokens([skill.catalog_entry() for skill in catalog]))
        rules = [item for item in items if item.kind == "rule"]
        learned = [item for item in items if item.kind == "skill" and item.source != "disk"]
        stats = self.repo.stats(TENANT)
        return {
            "live_rules": len(rules), "live_learned_skills": len(learned),
            "live_knowledge_tokens": sum(item.tokens for item in rules + learned),
            "probation": stats["by_status"]["probation"], "archived": stats["by_status"]["archived"],
            "instruction_tokens_per_call": round(sum(per_call) / max(1, len(per_call)), 1),
            "skill_catalog_tokens": round(sum(catalog_tokens) / max(1, len(catalog_tokens)), 1),
        }

    # Governed arm only: review train cases so cited items earn or lose credit.
    def attribution_round(self, cases: List[dict], round_index: int) -> Dict[str, int]:
        counts = {"reviews": 0, "accepted": 0, "false_positive": 0, "failed": 0}
        lock = threading.Lock()

        def run(position_case):
            position, case = position_case
            # Two in three reviews run in the canary lane so probation items get exposure.
            lane = "stable" if position % 3 == 2 else "canary"
            reviewer = AgenticReviewer(
                _EvaluationTaskStore({"knowledge_lane": lane}), self.client,
                default_token_budget=self.settings.agent_token_budget,
                default_time_budget=self.settings.agent_time_budget_seconds,
                skill_provider=lambda _tenant: self.skills(),
                knowledge=self.governor,
            )
            task_id = "attr-%d-%d" % (round_index, position)
            try:
                findings = reviewer.review_with_context(
                    task_id, case["diff"], parse_unified_diff(case["diff"]),
                    repository=case["repository"],
                )
            except Exception as exc:
                log("attribution review failed: %s" % str(exc)[:200])
                with lock:
                    counts["failed"] += 1
                return
            matched = {item.predicted_index for item in one_to_one_match(case["expected_findings"], findings)}
            for index, finding in enumerate(findings):
                category = "accepted" if index in matched else "false_positive"
                self.governor.record_feedback(
                    TENANT, task_id, category, finding.to_dict(), "", learn=False,
                )
                with lock:
                    counts[category] += 1
            with lock:
                counts["reviews"] += 1

        with ThreadPoolExecutor(max_workers=self.args.workers) as pool:
            list(pool.map(run, list(enumerate(cases))))
        return counts


# ------------------------------------------------------------------ report
COMPARISON_METRICS = (
    "f1", "precision", "recall", "high_risk_recall", "clean_accuracy",
    "average_instruction_tokens_per_call", "average_total_tokens_per_pr",
)


def repeated_cases(cases: List[dict], repeats: int) -> List[dict]:
    """The holdout replayed ``repeats`` times as distinct, pairable cases."""
    return [
        {**case, "id": "%s#%d" % (case["id"], repeat)}
        for repeat in range(max(1, repeats)) for case in cases
    ]


def per_repeat_f1(report: dict) -> List[float]:
    groups: Dict[str, dict] = {}
    for row in report["case_results"]:
        repeat = str(row["id"]).rsplit("#", 1)[-1]
        totals = groups.setdefault(repeat, ProductionEvaluationHarness._empty_totals())
        ProductionEvaluationHarness._accumulate(totals, row)
    return [
        ProductionEvaluationHarness._metrics(groups[key])["f1"]
        for key in sorted(groups, key=lambda value: int(value) if value.isdigit() else 0)
    ]


def holdout_metrics(report: dict) -> Dict[str, Any]:
    metrics = report["metrics"]
    keys = (
        "f1", "precision", "recall", "high_risk_recall", "clean_accuracy",
        "average_instruction_tokens_per_call", "average_total_tokens_per_pr",
        "average_latency_ms_per_pr", "failure_rate",
    )
    values = {key: metrics.get(key) for key in keys}
    runs = per_repeat_f1(report)
    values["f1_runs"] = runs
    if len(runs) > 1:
        mean = sum(runs) / len(runs)
        values["f1_run_std"] = round((sum((run - mean) ** 2 for run in runs) / (len(runs) - 1)) ** 0.5, 4)
    return values


def svg_lines(title: str, series: Dict[str, List[tuple]], y_label: str) -> str:
    width, height, pad = 760, 360, 56
    points = [point for values in series.values() for point in values]
    if not points:
        return ""
    max_x = max(x for x, _ in points) or 1
    max_y = max(y for _, y in points) or 1
    colors = {"naive": "#d9534f", "governed": "#2b7bb9"}

    def sx(x):
        return pad + (width - 2 * pad) * x / max_x

    def sy(y):
        return height - pad - (height - 2 * pad) * y / max_y

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" font-family="sans-serif" font-size="12">' % (width, height),
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="%d" y="24" font-size="15" font-weight="bold">%s</text>' % (pad, title),
        '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#333"/>' % (pad, height - pad, width - pad, height - pad),
        '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#333"/>' % (pad, pad, pad, height - pad),
        '<text x="%d" y="%d">events</text>' % (width - pad - 30, height - pad + 32),
        '<text x="8" y="%d">%s</text>' % (pad - 12, y_label),
        '<text x="%d" y="%d">%d</text>' % (pad - 40, pad + 4, max_y),
        '<text x="%d" y="%d">%d</text>' % (width - pad - 10, height - pad + 16, max_x),
    ]
    for offset, (name, values) in enumerate(series.items()):
        path = " ".join("%.1f,%.1f" % (sx(x), sy(y)) for x, y in values)
        color = colors.get(name, "#555")
        parts.append('<polyline fill="none" stroke="%s" stroke-width="2.5" points="%s"/>' % (color, path))
        parts.append('<text x="%d" y="%d" fill="%s">%s</text>' % (width - pad - 120, pad + 16 * offset, color, name))
    parts.append("</svg>")
    return "\n".join(parts)


def write_reports(out_dir: str, state: dict) -> None:
    measurements = state["measurements"]
    with open(os.path.join(out_dir, "measurements.csv"), "w", newline="", encoding="utf-8") as handle:
        if measurements:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in measurements for key in row}))
            writer.writeheader()
            writer.writerows(measurements)
    for metric, label in (
        ("instruction_tokens_per_call", "tokens/call"), ("live_knowledge_tokens", "tokens"),
    ):
        series = {
            arm: [(row["event"], row[metric]) for row in measurements if row["arm"] == arm]
            for arm in ("naive", "governed")
        }
        with open(os.path.join(out_dir, "%s.svg" % metric), "w", encoding="utf-8") as handle:
            handle.write(svg_lines(metric.replace("_", " "), series, label))
    checkpoints = state["checkpoints"]
    f1_series = {
        arm: [(int(event), values[arm]["f1"]) for event, values in sorted(checkpoints.items(), key=lambda kv: int(kv[0])) if arm in values]
        for arm in ("naive", "governed")
    }
    with open(os.path.join(out_dir, "holdout_f1.svg"), "w", encoding="utf-8") as handle:
        handle.write(svg_lines("holdout F1 (x100)", {
            arm: [(x, y * 100) for x, y in values] for arm, values in f1_series.items()
        }, "F1x100"))
    lines = [
        "# Context-rot stress test", "",
        "- events: %s (mix %s)" % (state.get("events_total", "?"), json.dumps(state["event_mix"]) if state.get("event_mix")
                                    else ", ".join("%s %d%%" % (k, w * 100) for k, w in EVENT_MIX)),
        "- model: %s" % state.get("model", ""), "- seed: %s" % state.get("seed"),
        "- admission actions (governed): %s" % json.dumps(state.get("admission_actions", {})),
        "", "## Prompt size (probe, mean instruction tokens per model call)", "",
        "| event | naive tokens/call | governed tokens/call | naive live tokens | governed live tokens | naive rules | governed rules |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by_event: Dict[int, Dict[str, dict]] = {}
    for row in measurements:
        by_event.setdefault(row["event"], {})[row["arm"]] = row
    for event in sorted(by_event):
        naive, governed = by_event[event].get("naive", {}), by_event[event].get("governed", {})
        lines.append("| %d | %s | %s | %s | %s | %s | %s |" % (
            event, naive.get("instruction_tokens_per_call", ""), governed.get("instruction_tokens_per_call", ""),
            naive.get("live_knowledge_tokens", ""), governed.get("live_knowledge_tokens", ""),
            naive.get("live_rules", ""), governed.get("live_rules", ""),
        ))
    holdout_scope = "within-repository temporal" if state.get("temporal_split") else "repos 09-10"
    lines += ["", "## Holdout quality (%s, %s repeat(s) pooled)" % (holdout_scope, state.get("holdout_repeats", 1)), "",
              "| event | arm | F1 | F1 per repeat | precision | recall | high-risk recall | clean acc. | instr. tokens/call | tokens/PR | failure |",
              "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for event in sorted(checkpoints, key=int):
        for arm, values in sorted(checkpoints[event].items()):
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                event, arm, values["f1"], " / ".join(str(item) for item in values.get("f1_runs", [])),
                values["precision"], values["recall"],
                values["high_risk_recall"], values["clean_accuracy"],
                values["average_instruction_tokens_per_call"], values["average_total_tokens_per_pr"],
                values["failure_rate"],
            ))
    comparisons = state.get("comparisons", {})
    if comparisons:
        lines += ["", "## Governed minus naive (paired bootstrap, 95% CI)", "",
                  "| event | metric | delta | 95% CI |", "|---:|---|---:|---|"]
        for event in sorted(comparisons, key=int):
            for metric, value in comparisons[event].items():
                lines.append("| %s | %s | %s | [%s, %s] |" % (
                    event, metric, value["delta"], value["ci95"][0], value["ci95"][1],
                ))
    lines += ["", "## Maintenance (governed)", ""]
    for item in state.get("maintenance", []):
        lines.append("- event %s: %s" % (item["event"], json.dumps(item["summary"], ensure_ascii=False)))
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    write_json(os.path.join(out_dir, "report.json"), state)


# --------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--dataset", default="")
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation_results", "rot_stress"))
    parser.add_argument("--events", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--checkpoints", default="0,100,250,500")
    parser.add_argument("--measure-every", type=int, default=25)
    parser.add_argument("--review-every", type=int, default=25)
    parser.add_argument("--review-batch", type=int, default=6)
    parser.add_argument("--maintain-every", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gate-cases", type=int, default=20)
    parser.add_argument("--holdout-cases", type=int, default=20)
    parser.add_argument("--promote-min-exposures", type=int, default=3)
    parser.add_argument("--evict-min-exposures", type=int, default=12)
    parser.add_argument("--probation-max-exposures", type=int, default=16)
    parser.add_argument("--holdout-repeats", type=int, default=3,
                        help="replay the holdout this many times per arm and pool the cases")
    parser.add_argument("--fake", action="store_true", help="offline deterministic client")
    parser.add_argument("--maintain-now", default="",
                        help="on resume, run one maintenance cycle first; the text is recorded as its note")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    settings = Settings.from_env()
    if args.fake:
        client, model = FakeStressClient(), "fake"
    else:
        config = settings.resolved_llm()
        if not config:
            raise SystemExit("configure an LLM in .env or pass --fake")
        client = JsonChatClient(
            str(config["base_url"]), str(config["api_key"]), str(config["model"]),
            str(config["provider"]), settings.timeout_seconds, dict(config.get("headers") or {}),
        )
        model = str(config["model"])
    cases = load_replay_cases(args.dataset)
    train = [case for case in cases if case["repository"] in TRAIN_REPOS]
    gate_cases = [case for case in cases if case["repository"] in VALIDATION_REPOS][:args.gate_cases]
    holdout = [case for case in cases if case["repository"] in HOLDOUT_REPOS][:args.holdout_cases]
    probes = [case for case in holdout if case["expected_findings"]][:5] or holdout[:5]
    registry = SkillRegistry(os.path.join(ROOT, "skills"))
    registry.reload()
    disk_skills = registry.agent_skills()

    events_path = os.path.join(args.out, "events.json")
    events = read_json(events_path, None)
    if events is None or len(events) != args.events:
        log("generating %d events with %s" % (args.events, model))
        generator = EventGenerator(client, train, args.seed, args.workers)
        events = generator.realize(generator.plan(args.events))
        write_json(events_path, events)
        log("events ready: %s" % json.dumps({
            kind: sum(event["type"] == kind for event in events) for kind, _ in EVENT_MIX
        }))

    state_path = os.path.join(args.out, "state.json")
    state = read_json(state_path, {
        "next_event": 0, "measurements": [], "checkpoints": {}, "maintenance": [],
        "attribution": [], "admission_actions": {}, "events_total": args.events, "comparisons": {},
        "model": model, "seed": args.seed, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    arms = {
        name: Arm(name, args.out, client, settings, disk_skills, gate_cases if name == "governed" else [], args)
        for name in ("naive", "governed")
    }
    checkpoints = sorted({int(item) for item in args.checkpoints.split(",") if item.strip()})
    rows_dir = os.path.join(args.out, "holdout_rows")
    os.makedirs(rows_dir, exist_ok=True)
    state.setdefault("comparisons", {})
    state.setdefault("holdout_repeats", args.holdout_repeats)
    attribution_rng = random.Random(args.seed + 1)

    def save():
        write_json(state_path, state)
        write_reports(args.out, state)

    def measure(event_count: int):
        for name, arm in arms.items():
            row = {"event": event_count, "arm": name, **arm.probe(probes)}
            state["measurements"] = [
                item for item in state["measurements"]
                if not (item["event"] == event_count and item["arm"] == name)
            ] + [row]
        log("measured @%d: %s" % (event_count, " | ".join(
            "%s %s tok/call, %s live tok" % (row["arm"], row["instruction_tokens_per_call"], row["live_knowledge_tokens"])
            for row in state["measurements"] if row["event"] == event_count
        )))

    def checkpoint(event_count: int):
        key = str(event_count)
        done = state["checkpoints"].get(key, {})
        for name, arm in arms.items():
            if name in done:
                continue
            if event_count == 0 and done:
                done[name] = dict(next(iter(done.values())))  # both arms start empty
                continue
            cases = repeated_cases(holdout, args.holdout_repeats)
            log("holdout replay for %s @%d (%d cases x %d repeats)" % (
                name, event_count, len(holdout), args.holdout_repeats,
            ))
            report = ProductionEvaluationHarness(workers=args.workers).run(
                arm.snapshot_reviewer("holdout-%d" % event_count), cases,
                name="%s-%d" % (name, event_count),
            )
            if report["metrics"]["failure_rate"] > MAX_REPLAY_FAILURE_RATE:
                errors = sorted({str(row.get("error", ""))[:160] for row in report["case_results"] if row.get("error")})
                raise ModelUnavailable("holdout replay for %s @%d failed for %.0f%% of cases: %s" % (
                    name, event_count, 100 * report["metrics"]["failure_rate"], errors[:3],
                ))
            write_json(os.path.join(rows_dir, "%d-%s.json" % (event_count, name)), {
                "case_results": report["case_results"], "metrics": report["metrics"],
            })
            done[name] = holdout_metrics(report)
            state["checkpoints"][key] = done
            save()
            log("holdout %s @%d: %s" % (name, event_count, json.dumps(done[name])))
        compare(event_count)

    def compare(event_count: int):
        """Paired bootstrap of governed minus naive over the same pooled holdout cases."""
        key = str(event_count)
        if key in state["comparisons"] or event_count == 0:
            return
        paths = [os.path.join(rows_dir, "%d-%s.json" % (event_count, name)) for name in ("naive", "governed")]
        if not all(os.path.exists(path) for path in paths):
            return
        naive, governed = (read_json(path, {}) for path in paths)
        state["comparisons"][key] = paired_bootstrap_comparison(
            naive, governed, iterations=2000, metrics=COMPARISON_METRICS,
        )
        save()
        log("governed - naive @%d: %s" % (event_count, json.dumps({
            metric: state["comparisons"][key][metric] for metric in ("f1", "high_risk_recall", "clean_accuracy")
        })))

    def maintain(event_count: int, note: str = ""):
        log("maintenance @%d%s" % (event_count, " (%s)" % note if note else ""))
        summary = arms["governed"].governor.maintain(TENANT, force_dream=True)
        promotion = summary.get("promotion") or {}
        compact = {
            "dream": {key: value for key, value in summary["dream"].items() if key not in {"merges", "drops"}},
            "merges": len(summary["dream"].get("merges", [])),
            "drops": len(summary["dream"].get("drops", [])),
            "promotion": {key: value for key, value in promotion.items() if key != "gate"},
            "promotion_gate_failures": (promotion.get("gate") or {}).get("failures"),
            "dream_release": (summary["dream"].get("release") or {}).get("status"),
            "dream_gate_failures": ((summary["dream"].get("release") or {}).get("gate") or {}).get("failures"),
            "live_tokens_before": summary["live_tokens_before"],
            "live_tokens_after": summary["live_tokens_after"],
        }
        failures = set(compact["dream_gate_failures"] or []) | set(compact["promotion_gate_failures"] or [])
        if "execution_success_rate" in failures:
            # The gate already rejected and nothing was written; leave the cycle unrecorded
            # so a resumed run (--maintain-now) repeats it once the model is back.
            raise ModelUnavailable("maintenance @%d: gate replays failed to execute: %s" % (
                event_count, json.dumps(compact, ensure_ascii=False)[:600],
            ))
        entry = {"event": event_count, "summary": compact}
        if note:
            entry["note"] = note
        state["maintenance"].append(entry)
        save()
        log("maintenance @%d: %s" % (event_count, json.dumps(compact, ensure_ascii=False)))

    if args.maintain_now and state["next_event"]:
        maintain(state["next_event"], args.maintain_now)
        if args.measure_every:
            measure(state["next_event"])
        save()
    # A run stopped mid-checkpoint (or after a quota outage) redoes what is missing.
    for pending in checkpoints:
        if 0 < pending <= state["next_event"] and len(state["checkpoints"].get(str(pending), {})) < len(arms):
            checkpoint(pending)
            save()
    if state["next_event"] == 0:
        measure(0)
        if 0 in checkpoints:
            checkpoint(0)
        save()
    while state["next_event"] < len(events):
        index = state["next_event"]
        event = events[index]
        for name, arm in arms.items():
            actions = arm.learn(event)
            if name == "governed":
                for action in actions:
                    state["admission_actions"][action] = state["admission_actions"].get(action, 0) + 1
        count = index + 1
        governed = arms["governed"]
        if args.review_every and count % args.review_every == 0:
            batch = attribution_rng.sample(train, min(args.review_batch, len(train)))
            result = governed.attribution_round(batch, count)
            state["attribution"].append({"event": count, **result})
            log("attribution @%d: %s" % (count, json.dumps(result)))
        if args.maintain_every and count % args.maintain_every == 0:
            maintain(count)
        if args.measure_every and count % args.measure_every == 0:
            measure(count)
        state["next_event"] = count
        save()
        if count in checkpoints:
            checkpoint(count)
            save()
    state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save()
    log("done: %s" % os.path.join(args.out, "report.md"))


if __name__ == "__main__":
    main()
