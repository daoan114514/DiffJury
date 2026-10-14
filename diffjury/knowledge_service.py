"""Knowledge governance: one place that owns how lessons enter, earn and leave.

* ``prepare``        - per-task budgeted view (read side, see assembler.py)
* ``record_review``  - attribution: exposures, citations, helpful/harmful credit
* ``record_feedback``- human signals; blame/credit cited items, reflect lessons
* ``maintain``       - lifecycle + Dream + promotion, each change set released
                       only through the replay gate, plus memory governance
* ``rollback``       - restore the live set of any earlier knowledge release
"""
import json
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .admission import AdmissionDecision, AdmissionGate
from .assembler import ContextAssembler, KnowledgeView, StaticKnowledge, TaskSignals, finding_matches, scope_allows, trigger_score
from .diff_parser import parse_unified_diff
from .dream import DreamConsolidator
from .finding_identity import canonical_identity
from .knowledge import ROLES, KnowledgeItem, KnowledgeRepository, utc_now
from .lifecycle import KnowledgeLifecycle
from .metrics import metrics
from .release_gate import apply_snapshot_changes
from .skills import AgentSkill
from .source_validation import SOURCES, TERMINAL, can_inject, observation, pair_credit, source_judgment, targets_for, validation_signature
from .telemetry import ExecutionLedger


LOGGER = logging.getLogger(__name__)

# Outcome weights. A published finding is weak evidence a lesson helped; a
# finding the pipeline itself rejected is weak blame; humans are stronger.
HELPFUL_PUBLISHED = 1.0
# A published finding that matches an exposed item's cwe/keyword trigger but
# does not cite it: the model applied the lesson without naming it.
HELPFUL_IMPLICIT = 0.5
HARMFUL_REJECTED = 0.5
HARMFUL_FALSE_POSITIVE = 2.0
HELPFUL_ACCEPTED = 1.0

SECURITY_CWES = {
    "CWE-20", "CWE-22", "CWE-78", "CWE-79", "CWE-89", "CWE-94", "CWE-95",
    "CWE-200", "CWE-259", "CWE-284", "CWE-285", "CWE-287", "CWE-295",
    "CWE-327", "CWE-352", "CWE-502", "CWE-532", "CWE-611", "CWE-798",
    "CWE-862", "CWE-863", "CWE-918",
}

REFLECT_PROMPT = """You turn one piece of code-review feedback into at most two reusable
lessons for an automated reviewer. A lesson is one imperative sentence of at most 40
words that names a concrete trigger (API, pattern, file type or CWE) and what to do.
Do not restate generic best practice; capture only what this feedback teaches.
Roles: "security" and "correctness-reliability" find defects; "critic" rejects false
positives; "lead" plans the review.
For a false positive, write a critic lesson that says when NOT to report.
For a missed issue, write a lesson for the worker that should have found it.
Return JSON only: {"lessons":[{"role":"...","content":"...","triggers":["cwe:CWE-89",
"lang:py","keyword"]}]}"""


def default_role(category: str, finding: Dict[str, Any]) -> str:
    if category == "false_positive":
        return "critic"
    cwe = str(finding.get("cwe") or "").upper()
    source = str(finding.get("source") or "")
    if cwe in SECURITY_CWES or source == "security":
        return "security"
    return "correctness-reliability"


def reflect_lessons(
    client, ledger, category: str, finding: Dict[str, Any], note: str, diff: str = "",
) -> List[Dict[str, Any]]:
    """Turn one feedback signal into at most two candidate lessons.

    With a model the lessons are distilled by REFLECT_PROMPT; without one (or
    when the call fails) the reviewer's own note becomes the lesson.
    """
    fallback_role = default_role(category, finding)
    if client is not None:
        try:
            reply = client.complete_json(
                "knowledge-reflection", REFLECT_PROMPT, json.dumps({
                    "category": category, "finding": {
                        key: finding.get(key) for key in (
                            "rule_id", "cwe", "severity", "title", "explanation",
                            "path", "line", "evidence",
                        )
                    }, "reviewer_note": note[:2000], "diff_excerpt": diff[:6000],
                }, ensure_ascii=False), ledger, max_tokens=500,
            )
            lessons = []
            for item in reply.get("lessons") or []:
                if not isinstance(item, dict) or not str(item.get("content") or "").strip():
                    continue
                role = str(item.get("role") or "")
                lessons.append({
                    "role": role if role in ROLES else fallback_role,
                    "content": str(item["content"]).strip(),
                    "triggers": [str(value) for value in item.get("triggers") or []][:8],
                })
            if lessons:
                return lessons[:2]
        except Exception as exc:
            LOGGER.warning("knowledge reflection fell back to the reviewer note: %s", exc)
    content = " ".join(str(note or "").split())
    if not content:
        return []
    triggers = []
    if finding.get("cwe"):
        triggers.append("cwe:%s" % str(finding["cwe"]).upper())
    path = str(finding.get("path") or "")
    if "." in path:
        triggers.append("lang:%s" % path.rsplit(".", 1)[-1].lower())
    return [{"role": fallback_role, "content": content, "triggers": triggers}]


def _finding_key(value: Dict[str, Any]) -> tuple:
    return (
        str(value.get("path", "")), int(value.get("line", 0) or 0),
        canonical_identity(str(value.get("rule_id", "")), value.get("cwe")),
    )


class KnowledgeGovernor:
    def __init__(
        self, repository: KnowledgeRepository, assembler: ContextAssembler,
        admission: AdmissionGate, client=None, half_life_days: float = 30.0,
        pin_disk_skills: bool = True, background: bool = True,
        lifecycle: Optional[KnowledgeLifecycle] = None,
        dream: Optional[DreamConsolidator] = None, release_gate=None,
        memory=None, maintain_every_reviews: int = 50,
        dream_token_threshold: int = 4000, memory_policy: Optional[Dict[str, Any]] = None,
        source_validation_enabled: bool = False,
    ):
        self.repository = repository
        self.assembler = assembler
        self.admission = admission
        self.client = client
        self.half_life_days = half_life_days
        self.pin_disk_skills = pin_disk_skills
        self.lifecycle = lifecycle or KnowledgeLifecycle(repository)
        self.dream = dream or DreamConsolidator(
            repository, self.lifecycle, client,
            admission.similarity_threshold, admission.dup_threshold,
            admission.max_rule_tokens,
        )
        self.release_gate = release_gate
        self.memory = memory
        self.maintain_every_reviews = max(0, int(maintain_every_reviews))
        self.dream_token_threshold = max(0, int(dream_token_threshold))
        self.memory_policy = dict(memory_policy or {})
        self.ledger = ExecutionLedger("knowledge-reflection")
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="knowledge") if background else None
        self._synced: Dict[str, str] = {}
        self._sync_lock = threading.Lock()
        self._reviews_since_maintenance: Dict[str, int] = {}
        self._maintenance_lock = threading.Lock()
        self.source_validation_enabled = source_validation_enabled
        self._source_validation_lock = threading.Lock()

    def _refresh_source_validation(self, tenant_id: str) -> Dict[str, dict]:
        proofs = {}
        items = {item.id: item for item in self.repository.list(tenant_id)}
        for row in self.repository.events(tenant_id, limit=1000000):
            detail, item = row["detail"], items.get(row["item_id"])
            if (row["event"] != "source_validation" or item is None
                    or detail.get("signature") != validation_signature(item)):
                continue
            previous = proofs.get(item.id)
            if previous is None or (previous.get("status") not in TERMINAL and detail.get("status") in TERMINAL):
                proofs[item.id] = detail
        self.lifecycle.protect_pending_sources = self.source_validation_enabled
        self.lifecycle.validation_status = {key: value.get("status", "unverified") for key, value in proofs.items()}
        return proofs

    # ----------------------------------------------------------- read side
    def prepare(
        self, tenant_id: str, repository: str, lane: str, signals: TaskSignals,
    ) -> KnowledgeView:
        return KnowledgeView(
            self.assembler, self.assembler.live_items(tenant_id, lane), signals, lane,
        )

    def learned_skills(self, tenant_id: str, lane: str = "stable") -> List[AgentSkill]:
        """Skills that were learned (not shipped on disk) and are live for this lane."""
        statuses = ["active"] + (["probation"] if lane == "canary" else [])
        values = []
        for item in self.repository.list(tenant_id, statuses=statuses, kind="skill"):
            if item.source == "disk" or not item.body:
                continue
            values.append(AgentSkill(
                name=item.name, description=item.content, instructions=item.body,
                content=item.instruction_text(), source="learned", version=str(item.version),
            ))
        return values

    def sync_disk_skills(self, tenant_id: str, skills: Sequence[AgentSkill]) -> None:
        """Track shipped skills as knowledge items so their usefulness is measured."""
        with self._sync_lock:
            pending = [
                skill for skill in skills
                if self._synced.get("%s:%s" % (tenant_id, skill.name)) != skill.content_sha256
            ]
            if not pending:
                return
            existing = {
                item.name: item
                for item in self.repository.list(tenant_id, kind="skill")
                if item.source == "disk"
            }
            for skill in pending:
                key = "%s:%s" % (tenant_id, skill.name)
                record = existing.get(skill.name)
                if record is None:
                    self.repository.create(KnowledgeItem(
                        tenant_id=tenant_id, kind="skill", name=skill.name,
                        content=skill.description, body=skill.instructions,
                        status="active", pinned=self.pin_disk_skills, source="disk",
                        evidence=[{"source": "disk", "sha256": skill.content_sha256}],
                    ))
                elif record.content != skill.description or record.body != skill.instructions:
                    record.content, record.body = skill.description, skill.instructions
                    record.version += 1
                    self.repository.save(record)
                self._synced[key] = skill.content_sha256

    # --------------------------------------------------------- attribution
    def record_review(
        self, tenant_id: str, task_id: str, view: Optional[KnowledgeView],
        candidates: Iterable[Any], published: Iterable[Any],
    ) -> Dict[str, Any]:
        """Credit cited items for published findings and blame them for rejected ones."""
        if view is None:
            return {}
        published_keys = {_finding_key(item.to_dict()) for item in published}
        deltas: Dict[str, Dict[str, float]] = {
            item_id: {"exposures": 1, "citations": 0, "helpful": 0.0, "harmful": 0.0}
            for item_id in view.exposures
        }
        exposed_items = {item.id: item for item in view.items if item.id in deltas}
        implicit_seen = set()
        for finding in candidates:
            accepted = _finding_key(finding.to_dict()) in published_keys
            cited = [item for item in getattr(finding, "applied_rules", []) or [] if item in deltas]
            for item_id in cited:
                deltas[item_id]["citations"] += 1
                if accepted:
                    deltas[item_id]["helpful"] += HELPFUL_PUBLISHED
                else:
                    deltas[item_id]["harmful"] += HARMFUL_REJECTED
            if not accepted:
                continue
            # The worker that reported it, and the critic/lead that judged it.
            handlers = {str(getattr(finding, "source", "")), "critic", "lead"}
            for item_id, item in exposed_items.items():
                key = (item_id, _finding_key(finding.to_dict()))
                if (
                    item_id in cited or key in implicit_seen
                    or not handlers.intersection(view.exposures.get(item_id, ()))
                    or not finding_matches(item, finding)
                ):
                    continue
                implicit_seen.add(key)
                deltas[item_id]["citations"] += 1
                deltas[item_id]["helpful"] += HELPFUL_IMPLICIT
                deltas[item_id]["implicit"] = deltas[item_id].get("implicit", 0) + 1
        now = utc_now()
        for item_id, delta in deltas.items():
            self.repository.apply_outcome(
                tenant_id, item_id, delta["helpful"], delta["harmful"],
                int(delta["exposures"]), int(delta["citations"]), now, self.half_life_days,
            )
        cited = {key: value for key, value in deltas.items() if value["citations"]}
        if cited:
            self.repository.log_event(tenant_id, "", "attribution", {
                "task_id": task_id, "items": cited,
            })
        self._count_review(tenant_id)
        return {"exposed": len(deltas), "cited": len(cited)}

    def _count_review(self, tenant_id: str) -> None:
        if not self.maintain_every_reviews or self._executor is None:
            return
        with self._maintenance_lock:
            count = self._reviews_since_maintenance.get(tenant_id, 0) + 1
            due = count >= self.maintain_every_reviews
            self._reviews_since_maintenance[tenant_id] = 0 if due else count
        if due:
            self._executor.submit(self._guard, lambda: self.maintain(tenant_id))

    # ------------------------------------------------------------ feedback
    def record_feedback(
        self, tenant_id: str, task_id: str, category: str,
        finding: Optional[Dict[str, Any]], note: str, diff: str = "",
        repository: str = "", learn: bool = True,
    ) -> Dict[str, Any]:
        finding = dict(finding or {})
        cited = [str(item) for item in finding.get("applied_rules") or []]
        weight = {
            "false_positive": ("harmful", HARMFUL_FALSE_POSITIVE),
            "accepted": ("helpful", HELPFUL_ACCEPTED),
        }.get(category)
        credited = []
        if weight:
            for item_id in cited:
                updated = self.repository.apply_outcome(
                    tenant_id, item_id,
                    helpful=weight[1] if weight[0] == "helpful" else 0.0,
                    harmful=weight[1] if weight[0] == "harmful" else 0.0,
                    half_life_days=self.half_life_days,
                )
                if updated is not None:
                    credited.append(item_id)
        result: Dict[str, Any] = {"credited": credited, "lessons": "none"}
        if learn and category in {"false_positive", "missed_issue"} and (note.strip() or finding):
            job = lambda: self.reflect_and_admit(  # noqa: E731
                tenant_id, task_id, category, finding, note, diff, repository,
            )
            if self._executor is not None:
                self._executor.submit(self._guard, job)
                result["lessons"] = "queued"
            else:
                result["lessons"] = [item.to_dict() for item in job()]
        return result

    @staticmethod
    def _guard(job) -> None:
        try:
            job()
        except Exception:  # background learning must never crash the worker thread
            LOGGER.exception("knowledge reflection failed")

    def reflect_and_admit(
        self, tenant_id: str, task_id: str, category: str,
        finding: Dict[str, Any], note: str, diff: str = "", repository: str = "",
    ) -> List[AdmissionDecision]:
        evidence = {
            "task_id": task_id, "category": category,
            "path": finding.get("path", ""), "line": finding.get("line", 0),
            "rule_id": finding.get("rule_id", ""), "cwe": finding.get("cwe", ""),
            "note": note[:300],
        }
        decisions = []
        for lesson in self.reflect(category, finding, note, diff):
            decisions.append(self.admission.admit(tenant_id, {
                "kind": "rule", "role": lesson["role"], "content": lesson["content"],
                "triggers": lesson.get("triggers") or [], "evidence": [evidence],
                "source": category, "scope": "*",
            }))
        return decisions

    def reflect(
        self, category: str, finding: Dict[str, Any], note: str, diff: str = "",
    ) -> List[Dict[str, Any]]:
        return reflect_lessons(self.client, self.ledger, category, finding, note, diff)

    def admit(self, tenant_id: str, candidate: Dict[str, Any]) -> AdmissionDecision:
        return self.admission.admit(tenant_id, candidate)

    def counterfactual_round(
        self, tenant_id: str, train_cases: Sequence[dict],
        reviewer_factory: Callable[[StaticKnowledge, str], Any],
        classify: Callable[[dict, list], Dict[str, Any]],
        replay_limit: int = 40, max_cases_per_item: int = 3,
        workers: int = 16, round_id: str = "",
    ) -> Dict[str, Any]:
        """Measure each underexposed trial against stable knowledge on TRAIN only.

        The cap counts individual reviews, so one paired comparison costs two.
        Existing lifecycle thresholds and release gates remain responsible for
        promotion/eviction. Only complete, actually exposed pairs earn outcomes.
        """
        if any(case.get("split") != "train" for case in train_cases):
            raise ValueError("counterfactual attribution accepts train cases only")
        baseline = self.repository.list(tenant_id, statuses=("active",))
        trials = [item for item in self.repository.list(tenant_id, statuses=("probation",))
                  if item.exposures < self.lifecycle.policy.promote_min_exposures]
        completed = {(event["item_id"], event["detail"].get("case_id"))
                     for event in self.repository.events(tenant_id, limit=1000000)
                     if event["event"] == "counterfactual" and event["detail"].get("applied")
                     and event["detail"].get("round_id") == round_id}
        signals = {case["id"]: TaskSignals.from_review(
            case["diff"], parse_unified_diff(case["diff"]).files, case["repository"],
        ) for case in train_cases}
        jobs, unmatched = [], 0
        for item in sorted(trials, key=lambda record: (record.exposures, record.created_at, record.id)):
            if len(jobs) >= max(0, replay_limit) // 2:
                break
            pool = baseline + [item]
            eligible = []
            for case in train_cases:
                task = signals[case["id"]]
                if (item.id, case["id"]) in completed or item.scope not in {"*", task.repository}:
                    continue
                if item.kind == "rule":
                    injects = any(item.id in self.assembler.rules_for(role, task, pool).ids
                                  for role in ROLES if role != "*")
                else:
                    skills = [AgentSkill(
                        name=record.name, description=record.content, instructions=record.body,
                        content=record.instruction_text(), source="learned", version=str(record.version),
                    ) for record in pool if record.kind == "skill" and record.body
                              and record.scope in {"*", task.repository} and scope_allows(record, task)]
                    chosen = self.assembler.skill_catalog(skills, task)
                    fitted = self.assembler.skill_bodies(chosen)["skills"]
                    injects = any(skill.name == item.name for skill in fitted)
                if injects:
                    eligible.append(case)
            eligible.sort(key=lambda case: (-trigger_score(item, signals[case["id"]]),
                                            case.get("created_at", ""), case["id"]))
            if not eligible:
                unmatched += 1
            missing = self.lifecycle.policy.promote_min_exposures - item.exposures
            for case in eligible[:min(max(0, max_cases_per_item), missing)]:
                if len(jobs) >= max(0, replay_limit) // 2:
                    break
                jobs.append((item, case))

        def run(job):
            item, case = job
            detail = {"source": "counterfactual", "round_id": round_id,
                      "case_id": case["id"], "split": "train", "repository": case["repository"],
                      "applied": False, "replays": 0}
            try:
                outcomes = []
                for label, items in (("stable", baseline), ("trial", baseline + [item])):
                    reviewer = reviewer_factory(StaticKnowledge(self.assembler, items),
                                                "counterfactual-%s-%s-%s" % (round_id, item.id, label))
                    detail["replays"] += 1
                    findings, summary = reviewer.review_case_detailed(case, parse_unified_diff(case["diff"]))
                    if label == "trial" and item.id not in (summary.get("knowledge") or {}).get("exposed", []):
                        detail["skipped"] = "trial item was not exposed"
                        return detail
                    outcomes.append(classify(case, findings))
                before, after = outcomes
                hits_before, hits_after = len(before["hits"]), len(after["hits"])
                fp_before, fp_after = len(before["false_positives"]), len(after["false_positives"])
                worsened = hits_after < hits_before or fp_after > fp_before
                improved = hits_after > hits_before or fp_after < fp_before
                helpful, harmful = int(improved and not worsened), int(worsened)
                detail.update({"applied": True, "helpful": helpful, "harmful": harmful,
                               "neutral": int(not helpful and not harmful),
                               "before": {"hits": hits_before, "false_positives": fp_before},
                               "after": {"hits": hits_after, "false_positives": fp_after}})
                detail["idempotency_key"] = "%s:%s:%s" % (round_id, item.id, case["id"])
                self.repository.apply_outcome(
                    tenant_id, item.id, helpful=helpful, harmful=harmful,
                    exposures=1, citations=int(bool(helpful or harmful)),
                    half_life_days=self.half_life_days, event="counterfactual", detail=detail,
                )
            except Exception as exc:
                detail["applied"] = False
                detail["error_type"] = type(exc).__name__
            finally:
                if not detail["applied"]:
                    self.repository.log_event(tenant_id, item.id, "counterfactual", detail)
            return detail

        with ThreadPoolExecutor(max_workers=max(1, min(16, workers))) as pool:
            rows = list(pool.map(run, jobs))
        return {"trials_considered": len(trials), "no_training_match": unmatched,
                "pairs": len(rows), "replay_limit": max(0, replay_limit),
                "replays": sum(row["replays"] for row in rows),
                "applied": sum(row["applied"] for row in rows),
                "helpful": sum(row.get("helpful", 0) for row in rows),
                "harmful": sum(row.get("harmful", 0) for row in rows),
                "neutral": sum(row.get("neutral", 0) for row in rows),
                "failed": sum("error_type" in row for row in rows),
                "not_exposed": sum("skipped" in row for row in rows)}

    def source_pr_validation_round(
        self, tenant_id: str, train_cases: Sequence[dict], feedback_events: Sequence[dict],
        original_reviews: Dict[str, dict], reviewer_factory: Callable[[StaticKnowledge, str], Any],
        replay_limit: int = 120, side_cases: int = 2, workers: int = 16, round_id: str = "",
    ) -> Dict[str, Any]:
        """Screen source feedback with three pairs, then one/two other TRAIN PRs.

        Reservations survive an interrupted process. Started review events count
        actual attempts; incomplete reservations consume their full allowance so
        a resumed round cannot exceed its cap or credit an unfinished comparison.
        """
        if any(case.get("split") != "train" for case in train_cases):
            raise ValueError("source validation accepts train cases only")
        if replay_limit < 0 or not 1 <= side_cases <= 2 or not 1 <= workers <= 16:
            raise ValueError("invalid source validation budget, side cases or workers")
        by_id = {case["id"]: case for case in train_cases}
        events = {event["index"]: event for event in feedback_events if event["type"] in SOURCES}
        if any(event.get("case_id") not in by_id for event in events.values()):
            raise ValueError("source feedback points outside the training set")
        trials = self.repository.list(tenant_id, statuses=("probation",))
        for item in trials:
            if item.source not in SOURCES:
                continue
            for evidence in item.evidence:
                case_id = evidence.get("case") or evidence.get("case_id")
                if case_id and case_id not in by_id:
                    raise ValueError("source evidence points outside the training set")
        with self._source_validation_lock:
            self.source_validation_enabled = True
            stable = self.repository.list(tenant_id, statuses=("active",))
            disk_skills = getattr(getattr(reviewer_factory, "__self__", None), "disk_skills", ())
            all_rows = self.repository.events(tenant_id, limit=1000000)
            initial_started = sum(row["event"] == "source_validation_replay" and row["detail"].get("round_id") == round_id for row in all_rows)
            results = []

            def spent(rows):
                reservations = {row["detail"]["job_key"]: row["detail"] for row in rows
                                if row["event"] == "source_validation_reservation" and row["detail"].get("round_id") == round_id}
                completed = {row["detail"]["job_key"]: row["detail"] for row in rows
                             if row["event"] == "source_validation" and row["detail"].get("round_id") == round_id}
                return sum(completed[key]["replays"] if key in completed else value["reserved_replays"]
                           for key, value in reservations.items())

            def source_event(item):
                choices = []
                for evidence in item.evidence:
                    event = events.get(evidence.get("event"))
                    case_id = evidence.get("case") or evidence.get("case_id")
                    if event is None and case_id:
                        event = next((value for value in events.values() if value["case_id"] == case_id and value["type"] == item.source), None)
                    if event is not None and event["type"] == item.source:
                        if case_id and case_id != event["case_id"]:
                            raise ValueError("source evidence and feedback PR differ")
                        if item.scope not in {"*", by_id[event["case_id"]]["repository"]}:
                            raise ValueError("source item scope differs from its training PR")
                        choices.append(event)
                return min(choices, key=lambda event: event["index"]) if choices else None

            def run(job):
                item, event, source_case, others, detail = job
                before, after, credits = [], [], []
                try:
                    targets = targets_for(source_case, event, original_reviews[source_case["id"]])

                    def replay(case, label, repeat, kind):
                        review_detail = {"source": "source_validation", "round_id": round_id,
                                         "job_key": detail["job_key"], "case_id": case["id"], "split": "train",
                                         "repository": case["repository"], "kind": kind, "label": label, "repeat": repeat}
                        self.repository.log_event(tenant_id, item.id, "source_validation_replay", review_detail)
                        detail["replays"] += 1
                        metrics.inc("knowledge_source_validation_reviews_total")
                        pool = stable if label == "stable" else stable + [item]
                        reviewer = reviewer_factory(StaticKnowledge(self.assembler, pool),
                                                    "source-%s-%s-%s-%s-%d" % (round_id, item.id, kind, label, repeat))
                        findings, summary = reviewer.review_case_detailed(case, parse_unified_diff(case["diff"]))
                        if label == "trial" and item.id not in (summary.get("knowledge") or {}).get("exposed", []):
                            detail["not_exposed"] = True
                            raise LookupError("source trial item was not exposed")
                        return observation(case, findings, targets if kind == "source" else None, item.source)

                    for repeat in range(3):
                        left = replay(source_case, "stable", repeat, "source")
                        right = replay(source_case, "trial", repeat, "source")
                        before.append(left); after.append(right)
                        credits.append(pair_credit(left, right, item.source))
                    detail.update(source_judgment(item.source, before, after, len(targets["correct_retention"])))
                    if detail["passed"]:
                        for repeat, case in enumerate(others):
                            left, right = replay(case, "stable", repeat, "side"), replay(case, "trial", repeat, "side")
                            detail["side_results"].append({"case_id": case["id"], "split": "train", "repository": case["repository"],
                                                           "before": left, "after": right})
                            credits.append(pair_credit(left, right))
                            if not set(left["hits"]).issubset(right["hits"]):
                                detail.update({"passed": False, "status": "harmful", "side_effect": "lost annotation hit"})
                    detail.update({"applied": True, "source_before": before, "source_after": after,
                                   "completed_pairs": len(credits),
                                   "helpful": sum(value[0] for value in credits), "harmful": sum(value[1] for value in credits),
                                   "neutral": sum(value[2] for value in credits)})
                    self.repository.apply_outcome(
                        tenant_id, item.id, helpful=detail["helpful"], harmful=detail["harmful"],
                        exposures=len(credits), citations=detail["helpful"] + detail["harmful"],
                        half_life_days=self.half_life_days, event="source_validation", detail=detail,
                    )
                except Exception as exc:
                    detail.update({"applied": False, "passed": False, "status": "unverified", "error_type": type(exc).__name__,
                                   "source_before": before, "source_after": after, "completed_pairs": len(credits)})
                    self.repository.log_event(tenant_id, item.id, "source_validation", detail)
                metrics.inc("knowledge_source_validation_%s_total" % detail["status"])
                return detail

            while True:
                all_rows = self.repository.events(tenant_id, limit=1000000)
                proofs = self._refresh_source_validation(tenant_id)
                seen = {row["item_id"] for row in all_rows if row["event"] in {"source_validation", "source_validation_reservation"}
                        and row["detail"].get("round_id") == round_id}
                attempts = {}
                for row in all_rows:
                    if row["event"] == "source_validation_reservation":
                        key = (row["item_id"], row["detail"].get("signature"))
                        attempts[key] = attempts.get(key, 0) + 1
                pending = [item for item in self.repository.list(tenant_id, statuses=("probation",)) if item.source in SOURCES
                           and item.id not in seen and proofs.get(item.id, {}).get("status") not in TERMINAL]
                pending.sort(key=lambda item: (attempts.get((item.id, validation_signature(item)), 0),
                                               min((value["event"] for value in item.evidence if isinstance(value.get("event"), int)), default=1000000), item.id))
                available = max(0, replay_limit - spent(all_rows))
                jobs = []
                for item in pending:
                    event = source_event(item)
                    signature = validation_signature(item)
                    detail = {"source": "source_validation", "round_id": round_id,
                              "job_key": "%s:%s:%s" % (round_id, item.id, signature), "signature": signature,
                              "idempotency_key": "source:%s:%s:%s" % (round_id, item.id, signature),
                              "source_type": item.source, "split": "train", "repeats": 3,
                              "replay_limit": replay_limit,
                              "applied": False, "passed": False, "status": "unverified", "replays": 0, "side_results": []}
                    if event is None or event["case_id"] not in original_reviews:
                        detail["skipped"] = "no source training feedback/review"
                    else:
                        source_case = by_id[event["case_id"]]
                        detail.update({"case_id": source_case["id"], "repository": source_case["repository"], "event_index": event["index"]})
                        others = [case for case in train_cases if case["id"] != source_case["id"]
                                  and case["repository"] == source_case["repository"]
                                  and can_inject(self.assembler, item, case, stable, disk_skills)]
                        others.sort(key=lambda case: (-trigger_score(item, TaskSignals.from_review(
                            case["diff"], parse_unified_diff(case["diff"]).files, case["repository"])), case.get("created_at", ""), case["id"]))
                        others = others[:side_cases]
                        if not others:
                            detail["skipped"] = "no other triggering training PR"
                    if "skipped" in detail:
                        self.repository.log_event(tenant_id, item.id, "source_validation", detail)
                        metrics.inc("knowledge_source_validation_unverified_total")
                        results.append(detail)
                        continue
                    reservation = 6 + 2 * len(others)
                    if reservation > available:
                        continue
                    available -= reservation
                    detail["reserved_replays"] = reservation
                    detail["side_case_ids"] = [case["id"] for case in others]
                    self.repository.log_event(tenant_id, item.id, "source_validation_reservation", dict(detail))
                    metrics.inc("knowledge_source_validation_attempts_total")
                    jobs.append((item, event, source_case, others, detail))
                    if len(jobs) == workers:
                        break
                if not jobs:
                    break
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    results.extend(pool.map(run, jobs))
            self._refresh_source_validation(tenant_id)
            all_rows = self.repository.events(tenant_id, limit=1000000)
            started = sum(row["event"] == "source_validation_replay" and row["detail"].get("round_id") == round_id for row in all_rows)
            return {"round_id": round_id, "replay_limit": replay_limit, "replays": started - initial_started,
                    "round_replays": started, "budget_consumed": spent(all_rows), "entries": len(results),
                    "passed": sum(row.get("status") == "passed" for row in results),
                    "neutral": sum(row.get("status") == "neutral" for row in results),
                    "harmful": sum(row.get("status") == "harmful" for row in results),
                    "failed": sum(row.get("error_type") not in {None, "LookupError"} for row in results),
                    "not_exposed": sum(bool(row.get("not_exposed")) for row in results),
                    "skipped": sum("skipped" in row for row in results)}

    # --------------------------------------------------------- maintenance
    def maintain(self, tenant_id: str, force_dream: bool = False, now: str = "") -> Dict[str, Any]:
        """One maintenance cycle: Dream (incl. eviction), then promotion, then memory."""
        now = now or utc_now()
        if self.source_validation_enabled:
            if self.release_gate is not None and any(case.get("split") != "validation" for case in getattr(self.release_gate, "cases", [])):
                raise ValueError("fast promotion requires a validation-only release gate")
            self._refresh_source_validation(tenant_id)
        live_tokens = self.repository.stats(tenant_id)["live_tokens"]
        result: Dict[str, Any] = {"tenant_id": tenant_id, "live_tokens_before": live_tokens}
        if not self.repository.list_releases(tenant_id, 1):
            self.repository.save_release(tenant_id, "baseline", "initial", {})
        # Eviction runs every cycle; LLM consolidation only once the live set is
        # large enough to be worth it (or on demand).
        consolidate = force_dream or live_tokens >= self.dream_token_threshold
        proposal = self.dream.propose(tenant_id, now, consolidate=consolidate)
        result["dream"] = proposal.summary()
        result["dream"]["consolidated"] = consolidate
        if proposal.changed:
            result["dream"]["release"] = self._release(
                tenant_id, proposal.baseline, proposal.candidate,
                "dream", max_token_growth=0.0,
            )
        promotable = self.lifecycle.plan(
            self.repository.list(tenant_id, statuses=("active", "probation")), now,
        ).promotable
        fast_ids = []
        if self.source_validation_enabled:
            proofs = self._refresh_source_validation(tenant_id)
            eligible = {item.id for item in self.repository.list(tenant_id, statuses=("probation",))
                        if proofs.get(item.id, {}).get("status") == "passed"}
            promotable = [item_id for item_id in promotable if item_id not in eligible]
            if self.release_gate is not None:
                fast_ids = sorted(eligible)
                promotable = sorted(set(promotable) | eligible)
            elif eligible:
                result["source_validation"] = {"status": "waiting_for_validation_gate", "items": sorted(eligible)}
        if promotable:
            baseline = self.repository.list(tenant_id, statuses=("active",))
            probation = {
                item.id: item for item in self.repository.list(tenant_id, statuses=("probation",))
            }
            promoted = []
            for item_id in promotable:
                item = probation.get(item_id)
                if item is not None:
                    item.status = "active"
                    promoted.append(item)
            result["promotion"] = self._release(
                tenant_id, baseline, baseline + promoted, "promotion",
                max_token_growth=None, persist_ids=[item.id for item in promoted],
            )
            published = (result["promotion"].get("changes") or {}).get("promoted", [])
            result["promotion"]["fast_promoted"] = [item_id for item_id in published if item_id in fast_ids]
            for item_id in result["promotion"]["fast_promoted"]:
                self.repository.log_event(tenant_id, item_id, "source_validation_promotion", {
                    "source": "source_validation", "signature": proofs[item_id]["signature"],
                    "round_id": proofs[item_id]["round_id"], "gate_version": result["promotion"]["version"]})
                metrics.inc("knowledge_source_validation_promotions_total")
        if self.memory is not None:
            result["memory"] = self.memory.govern(tenant_id, **self.memory_policy)
        result["live_tokens_after"] = self.repository.stats(tenant_id)["live_tokens"]
        self.repository.log_event(tenant_id, "", "maintenance", {
            key: value for key, value in result.items() if key != "dream"
        } | {"dream": {
            key: value for key, value in result["dream"].items()
            if key not in {"merges", "drops"}
        }})
        return result

    def _release(
        self, tenant_id: str, baseline: List[KnowledgeItem], candidate: List[KnowledgeItem],
        reason: str, max_token_growth: Optional[float], persist_ids: Iterable[str] = (),
    ) -> Dict[str, Any]:
        """Gate a candidate snapshot; persist it and record a release only if it passes."""
        live_candidate = [item for item in candidate if item.status != "archived"]
        if self.release_gate is not None:
            gate = self.release_gate.evaluate(
                [item for item in baseline if item.status != "archived"], live_candidate,
                max_token_growth=max_token_growth,
            )
        else:
            gate = {"passed": True, "ungated": True}
        if not gate["passed"]:
            metrics.inc("knowledge_releases_rejected_total")
            release = self.repository.save_release(
                tenant_id, "rejected", reason, gate, items=live_candidate,
            )
            return {"status": "rejected", "version": release["version"], "gate": gate}
        if reason == "promotion":
            changed = {"promoted": self.lifecycle.promote(
                tenant_id, list(persist_ids), "release gate passed",
            )}
        else:
            changed = apply_snapshot_changes(
                self.repository, tenant_id, baseline, candidate, reason,
            )
            for item_id in changed.get("archived", []):
                self.repository.log_event(tenant_id, item_id, "evict", {"reason": reason})
        release = self.repository.save_release(tenant_id, "released", reason, gate)
        metrics.inc("knowledge_releases_released_total")
        metrics.inc("knowledge_evictions_total", len(changed.get("archived", [])))
        return {
            "status": "released", "version": release["version"], "gate": gate,
            "changes": changed,
        }

    def rollback(self, tenant_id: str, version: int, actor: str = "") -> Dict[str, Any]:
        """Make the live set match a past release (content and status; counters stay)."""
        release = next((
            item for item in self.repository.list_releases(tenant_id, 100000)
            if int(item["version"]) == int(version)
        ), None)
        if release is None or release["status"] == "rejected":
            raise ValueError("knowledge release %s is not a restorable release" % version)
        snapshot = self.repository.release_snapshot(tenant_id, version) or []
        target = {item.id: item for item in snapshot}
        restored, archived = [], []
        for item in self.repository.list(tenant_id):
            wanted = target.get(item.id)
            if wanted is None:
                if item.status != "archived":
                    self.repository.set_status(tenant_id, item.id, "archived", "rollback")
                    archived.append(item.id)
                continue
            fields = ("content", "body", "triggers", "status", "pinned", "role", "scope", "name")
            if any(getattr(item, name) != getattr(wanted, name) for name in fields):
                for name in fields:
                    setattr(item, name, getattr(wanted, name))
                item.archived_reason = ""
                item.version += 1
                self.repository.save(item)
                restored.append(item.id)
        release = self.repository.save_release(
            tenant_id, "rollback", "rollback-to-v%d" % int(version),
            {"actor": actor, "restored": restored, "archived": archived},
        )
        return {"version": release["version"], "restored": restored, "archived": archived}

    def submit(self, job) -> bool:
        """Run ``job`` on the knowledge worker (inline when there is none)."""
        if self._executor is None:
            job()
            return False
        self._executor.submit(self._guard, job)
        return True

    def drain(self, timeout: float = 600.0) -> None:
        """Wait for queued background learning (tests and graceful shutdown)."""
        if self._executor is None:
            return
        marker: Future = self._executor.submit(lambda: None)
        marker.result(timeout=timeout)

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
