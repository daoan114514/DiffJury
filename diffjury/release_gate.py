"""One gate for every knowledge change: replay before release.

Promotions out of probation and Dream consolidations both produce a
*candidate* knowledge snapshot. The gate replays a fixed labelled set through
the real AgenticReviewer twice - once with the current (baseline) snapshot and
once with the candidate - and accepts the candidate only if it is Pareto
no-worse:

* no quality metric regresses confidently: the upper bound of its paired
  bootstrap 95% CI must be >= -margin. On a small replay set a single finding
  moves F1 by several points, so a bare point-estimate drop is not treated as
  evidence; only a drop the whole interval agrees on is;
* no quality metric drops by more than ``max_point_drop`` (or one case's
  worth of that metric, whichever is larger) even when the interval is wide,
  which bounds the risk of an underpowered replay, and
* the mean instruction tokens per model call grows by at most
  ``max_token_growth`` (0 for consolidation: Dream may not make prompts longer).

Baseline replays are cached by (dataset, snapshot) fingerprint, so a run of
gate decisions against an unchanged baseline pays for it once.
"""
import hashlib
import json
from pathlib import Path
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence

from .assembler import ContextAssembler, StaticKnowledge
from .evaluation_harness import dataset_fingerprint
from .evaluation_v2 import ProductionEvaluationHarness, ReplayReviewer, paired_bootstrap_comparison
from .knowledge import KnowledgeItem


QUALITY_METRICS = ("f1", "high_risk_recall", "clean_accuracy")
TOKEN_METRIC = "average_instruction_tokens_per_call"


def snapshot_fingerprint(items: Sequence[KnowledgeItem]) -> str:
    rendered = sorted(
        (item.id, item.version, item.status, item.content, item.body, item.role, item.scope,
         item.pinned, item.triggers, item.helpful, item.harmful, item.exposures, item.citations)
        for item in items if item.status != "archived"
    )
    return hashlib.sha256(json.dumps(rendered, ensure_ascii=False).encode("utf-8")).hexdigest()[:24]


class ReleaseGate:
    def __init__(
        self, cases: Sequence[dict], reviewer_factory: Callable[[StaticKnowledge, str], Any],
        assembler: ContextAssembler, margin: float = 0.02, workers: int = 4,
        bootstrap_iterations: int = 1000, metrics: Sequence[str] = QUALITY_METRICS,
        max_point_drop: float = 0.10,
        retry_attempts: int = 0, retry_workers: int = 8,
        cache_dir: Optional[str] = None,
    ):
        if not cases:
            raise ValueError("the release gate needs at least one labelled case")
        self.cases = list(cases)
        self.reviewer_factory = reviewer_factory
        self.assembler = assembler
        self.margin = float(margin)
        self.max_point_drop = float(max_point_drop)
        self.workers = max(1, int(workers))
        self.bootstrap_iterations = int(bootstrap_iterations)
        self.metrics = tuple(metrics)
        self.retry_attempts = max(0, int(retry_attempts))
        self.retry_workers = max(1, min(8, int(retry_workers)))
        self.dataset = dataset_fingerprint(self.cases)
        self._cache: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self.cache_dir = cache_dir

    @classmethod
    def for_client(
        cls, cases: Sequence[dict], client, assembler: ContextAssembler,
        token_budget: int = 8000, time_budget_seconds: int = 60, **options,
    ) -> "ReleaseGate":
        def factory(knowledge: StaticKnowledge, name: str):
            return ReplayReviewer(
                client, token_budget, time_budget_seconds, name=name, knowledge=knowledge,
            )
        return cls(cases, factory, assembler, **options)

    def replay(self, items: Sequence[KnowledgeItem], name: str) -> dict:
        key = "%s:%s" % (self.dataset, snapshot_fingerprint(items))
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached
        path = Path(self.cache_dir) / (hashlib.sha256(key.encode("utf-8")).hexdigest() + ".json") if self.cache_dir else None
        disk = json.loads(path.read_text(encoding="utf-8")) if path and path.exists() else None
        if disk is not None:
            if [row["id"] for row in disk["case_results"]] != [case["id"] for case in self.cases]:
                raise ValueError("release replay cache has different case ids")
            if all(row.get("execution_success") for row in disk["case_results"]):
                with self._lock:
                    self._cache[key] = disk
                return disk

        def save(report):
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                temporary.replace(path)

        reviewer = self.reviewer_factory(StaticKnowledge(self.assembler, items), name)
        pending = self.cases if disk is None else [case for case, row in zip(self.cases, disk["case_results"]) if not row.get("execution_success")]
        report = ProductionEvaluationHarness(workers=self.workers).run(reviewer, pending, name=name)
        if disk is not None:
            fresh = {row["id"]: row for row in report["case_results"] if row.get("execution_success")}
            report = disk
            report["case_results"] = [fresh.get(row["id"], row) for row in report["case_results"]]
            totals = ProductionEvaluationHarness._empty_totals()
            for row in report["case_results"]:
                ProductionEvaluationHarness._accumulate(totals, row)
            report["metrics"] = ProductionEvaluationHarness._metrics(totals)
        by_id = {case["id"]: case for case in self.cases}
        report.setdefault("first_execution_success_rate", report["metrics"].get("execution_success_rate", 1.0))
        save(report)
        for _ in range(self.retry_attempts):
            failed = [row["id"] for row in report["case_results"] if not row.get("execution_success")]
            if not failed:
                break
            retry = ProductionEvaluationHarness(workers=self.retry_workers).run(
                reviewer, [by_id[key] for key in failed], name=name,
            )
            fresh = {row["id"]: row for row in retry["case_results"] if row.get("execution_success")}
            report["case_results"] = [fresh.get(row["id"], row) for row in report["case_results"]]
            totals = ProductionEvaluationHarness._empty_totals()
            for row in report["case_results"]:
                ProductionEvaluationHarness._accumulate(totals, row)
            report["metrics"] = ProductionEvaluationHarness._metrics(totals)
            save(report)
        with self._lock:
            if all(row.get("execution_success") for row in report["case_results"]):
                self._cache[key] = report
        return report

    def point_drop_tolerance(self, metric: str, totals: Dict[str, Any]) -> float:
        """``max_point_drop``, but never less than one case's worth of the metric.

        With four high-risk findings in the replay set, missing one moves
        high-risk recall by 0.25; a fixed 0.10 guard would then block every
        change on a single miss.
        """
        denominator = {
            "high_risk_recall": totals.get("high_total"),
            "clean_accuracy": totals.get("clean_cases"),
            "recall": (totals.get("tp") or 0) + (totals.get("fn") or 0),
        }.get(metric)
        if not denominator:
            return self.max_point_drop
        return max(self.max_point_drop, 1.0 / float(denominator) + 1e-9)

    def evaluate(
        self, baseline: Sequence[KnowledgeItem], candidate: Sequence[KnowledgeItem],
        max_token_growth: Optional[float] = 0.0,
    ) -> Dict[str, Any]:
        before = self.replay(baseline, "knowledge-baseline")
        after = self.replay(candidate, "knowledge-candidate")
        comparison = paired_bootstrap_comparison(
            before, after, iterations=self.bootstrap_iterations,
            metrics=self.metrics + (TOKEN_METRIC,),
        )
        checks = {}
        for metric in self.metrics:
            delta = comparison[metric]
            tolerance = self.point_drop_tolerance(metric, before["metrics"])
            checks[metric] = {
                "baseline": before["metrics"][metric], "candidate": after["metrics"][metric],
                "delta": delta["delta"], "ci95": delta["ci95"],
                "max_point_drop": round(tolerance, 4),
                "passed": delta["ci95"][1] >= -self.margin
                and delta["delta"] >= -tolerance,
            }
        tokens_before = float(before["metrics"][TOKEN_METRIC] or 0.0)
        tokens_after = float(after["metrics"][TOKEN_METRIC] or 0.0)
        token_check = {
            "baseline": tokens_before, "candidate": tokens_after,
            "max_growth": max_token_growth,
            "passed": max_token_growth is None
            or tokens_after <= tokens_before * (1.0 + max_token_growth) + 1.0,
        }
        failures = [name for name, check in checks.items() if not check["passed"]]
        if not token_check["passed"]:
            failures.append(TOKEN_METRIC)
        executions = (
            before["metrics"].get("execution_success_rate", 1.0),
            after["metrics"].get("execution_success_rate", 1.0),
        )
        if min(executions) < 1.0:
            failures.append("execution_success_rate")
        return {
            "passed": not failures, "failures": failures, "margin": self.margin,
            "max_point_drop": self.max_point_drop,
            "dataset_fingerprint": self.dataset, "cases": len(self.cases),
            "metrics": checks, "instruction_tokens_per_call": token_check,
            "execution_success_rate": {"baseline": executions[0], "candidate": executions[1]},
            "baseline_snapshot": snapshot_fingerprint(baseline),
            "candidate_snapshot": snapshot_fingerprint(candidate),
        }


def apply_snapshot_changes(
    repository, tenant_id: str, baseline: List[KnowledgeItem], candidate: List[KnowledgeItem],
    reason: str,
) -> Dict[str, List[str]]:
    """Persist the difference between two snapshots of one tenant's live items."""
    before = {item.id: item for item in baseline}
    after = {item.id: item for item in candidate}
    changed: Dict[str, List[str]] = {"updated": [], "archived": [], "created": []}
    for item_id, item in after.items():
        previous = before.get(item_id)
        if previous is None:
            existing = repository.get(tenant_id, item_id)
            if existing is None:
                repository.create(item)
                changed["created"].append(item_id)
                continue
            previous = existing
        if item.to_dict() != previous.to_dict():
            repository.save(item)
            changed["archived" if item.status == "archived" else "updated"].append(item_id)
    for item_id in before:
        if item_id not in after:
            repository.set_status(tenant_id, item_id, "archived", reason)
            changed["archived"].append(item_id)
    return changed
