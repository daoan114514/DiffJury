"""Product-backed agentic replay evaluation for labelled PR datasets."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import random
import threading
import time
from typing import Any, Dict, List, Optional

from .agentic_core import AgenticReviewer
from .evaluation_harness import EndToEndEvaluationHarness, dataset_fingerprint, one_to_one_match
from .reviewer import Reviewer


AGENTIC_ROLES = ("lead", "security", "correctness-reliability", "critic")


def validate_real_dataset(cases: List[dict], minimum_cases: int = 300) -> Dict[str, Any]:
    repositories_by_split = {}
    cases_by_split = Counter()
    source_kinds = set()
    for case in cases:
        split = str(case.get("split", ""))
        if split not in {"train", "validation", "holdout"}:
            raise ValueError("every case must use train, validation or holdout split")
        repositories_by_split.setdefault(split, set()).add(str(case.get("repository", "")))
        cases_by_split[split] += 1
        source_kinds.add(str((case.get("source") or {}).get("kind", "unknown")))
        if not isinstance(case.get("expected_findings"), list):
            raise ValueError("every real PR must include human expected_findings")
        for finding in case["expected_findings"]:
            if "should_comment" not in finding:
                raise ValueError("human labels must include should_comment")
            if not all(key in finding for key in ("severity", "path")) or not (
                "line" in finding or "start_line" in finding
            ):
                raise ValueError(
                    "human labels require severity, path and line/start_line"
                )
    overlaps = {}
    splits = sorted(repositories_by_split)
    for index, left in enumerate(splits):
        for right in splits[index + 1:]:
            shared = repositories_by_split[left].intersection(repositories_by_split[right])
            if shared:
                overlaps["%s:%s" % (left, right)] = sorted(shared)
    public_or_historical = source_kinds.issubset({
        "public-github-pr", "private-historical-pr",
    }) and bool(source_kinds)
    gates = {
        "minimum_300_cases": len(cases) >= minimum_cases,
        "real_provenance": public_or_historical,
        "repository_isolation": not overlaps,
        "train_present": bool(repositories_by_split.get("train")),
        "validation_present": bool(repositories_by_split.get("validation")),
        "hidden_holdout_present": bool(repositories_by_split.get("holdout")),
    }
    return {
        "ready": all(gates.values()), "gates": gates, "cases": len(cases),
        "repositories": len({str(case.get("repository")) for case in cases}),
        "repositories_by_split": {
            key: len(value) for key, value in repositories_by_split.items()
        },
        "cases_by_split": {
            key: int(cases_by_split.get(key, 0))
            for key in ("train", "validation", "holdout")
        },
        "repository_overlap": overlaps, "source_kinds": sorted(source_kinds),
        "dataset_sha256": dataset_fingerprint(cases),
    }


class _EvaluationTaskStore:
    """Minimal task input provider used by AgenticReviewer during replay."""

    def __init__(self, task_input: dict):
        self.task_input = dict(task_input)
        self._per_task: Dict[str, dict] = {}
        self._lock = threading.Lock()

    def bind(self, task_id: str, values: dict) -> None:
        with self._lock:
            self._per_task[task_id] = dict(values)

    def release(self, task_id: str) -> None:
        with self._lock:
            self._per_task.pop(task_id, None)

    def get(self, task_id: str, _tenant_id: Optional[str] = None) -> dict:
        with self._lock:
            extra = dict(self._per_task.get(task_id) or {})
        return {"input": {**self.task_input, **extra}}


class ReplayReviewer(Reviewer):
    """Replay labelled cases through the product AgenticReviewer.

    Replays never write live state: there is no checkpoint store, no memory and
    no knowledge outcome recording, so evaluation cannot leak into production.
    """

    def __init__(
        self, client, token_budget: int = 8000, time_budget_seconds: int = 60,
        name: str = "agentic-replay", **reviewer_options,
    ):
        self.name = name
        self.store = _EvaluationTaskStore({
            "mode": "agentic", "enabled_agents": list(AGENTIC_ROLES),
        })
        self.agentic = AgenticReviewer(
            self.store, client,
            default_token_budget=token_budget,
            default_time_budget=time_budget_seconds,
            enabled_roles=set(AGENTIC_ROLES),
            **reviewer_options,
        )
        self._sequence = 0
        self._lock = threading.Lock()

    def review(self, diff: str, parsed) -> list:
        return self.review_case({"diff": diff, "repository": ""}, parsed)

    def review_case(self, case: dict, parsed) -> list:
        return self.review_case_detailed(case, parsed)[0]

    def review_case_detailed(self, case: dict, parsed):
        with self._lock:
            self._sequence += 1
            task_id = "replay:%s:%d" % (self.name, self._sequence)
        # A case may carry a checkout of the reviewed revision so repository
        # tools (read_file, symbol, ...) see real project context.
        if case.get("repository_root"):
            self.store.bind(task_id, {"repository_root": str(case["repository_root"])})
        try:
            findings = self.agentic.review_with_context(
                task_id, case["diff"], parsed,
                repository=str(case.get("repository") or ""),
            )
        finally:
            self.store.release(task_id)
        summary = self.agentic.collaboration_summary(task_id)
        self.agentic.release_summary(task_id)
        self._validate_execution(summary)
        return findings, summary

    @staticmethod
    def _validate_execution(summary: dict) -> None:
        execution = summary.get("execution") or {}
        actual = Counter(
            str(item.get("role")) for item in execution.get("model_call_log") or []
            if bool(item.get("ok", True))
        )
        required = set(AGENTIC_ROLES)
        collaboration = summary.get("collaboration") or {}
        if not int(collaboration.get("candidate_findings_before_critic", 0) or 0):
            required.discard("critic")
        missing = sorted(role for role in required if actual[role] < 1)
        if missing:
            errors = {}
            for item in execution.get("model_call_log") or []:
                if str(item.get("role")) in missing and item.get("error"):
                    errors.setdefault(str(item.get("role")), str(item["error"])[:200])
            raise RuntimeError(
                "replay completed without successful LLM role(s): %s%s" % (
                    ", ".join(missing),
                    "".join(" | %s: %s" % pair for pair in sorted(errors.items())),
                )
            )


class _CaseRecorder:
    """Capture one case's findings and execution summary, safe under parallel replay."""

    def __init__(self, delegate):
        self.delegate = delegate
        self.name = delegate.name
        self.findings = []
        self.summary: Dict[str, Any] = {}

    def review(self, diff, parsed):
        return self.review_case({"diff": diff, "repository": ""}, parsed)

    def review_case(self, case, parsed):
        detailed = getattr(self.delegate, "review_case_detailed", None)
        if detailed:
            self.findings, self.summary = detailed(case, parsed)
        else:
            method = getattr(self.delegate, "review_case", None)
            self.findings = (
                method(case, parsed) if method else self.delegate.review(case["diff"], parsed)
            )
        return self.findings


class ProductionEvaluationHarness(EndToEndEvaluationHarness):
    def __init__(self, line_tolerance: int = 2, workers: int = 1):
        super().__init__(line_tolerance)
        self.workers = max(1, int(workers))

    def run(self, reviewer, cases: List[dict], name: str = "") -> Dict[str, Any]:
        started = time.monotonic()
        if self.workers == 1:
            case_results = [self._run_case(reviewer, case) for case in cases]
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                case_results = list(pool.map(
                    lambda case: self._run_case(reviewer, case), cases
                ))
        totals = self._empty_totals()
        for result in case_results:
            self._accumulate(totals, result)
        return self._report(reviewer, cases, name, totals, case_results, started)

    def _run_case(self, reviewer, case):
        recorder = _CaseRecorder(reviewer)
        started = time.monotonic()
        result = super()._run_case(recorder, case)
        result.update({
            "invalid_comments": result["fp"],
            "exact_location_hits": 0,
            "evidence_hits": 0,
            "accepted_comments": int(case.get("accepted_comments", 0) or 0),
            "closed_comments": int(case.get("closed_comments", 0) or 0),
            "cost_usd": 0.0,
            "latency_ms": int((time.monotonic() - started) * 1000),
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "instruction_calls": 0,
            "instruction_tokens": 0,
            "model_roles": {},
            "critic_accepted": 0,
            "critic_rejected": 0,
            "revision_requests": 0,
            "revision_results": 0,
            "cited_rules": {},
        })
        if not result["execution_success"]:
            return result
        expected = [
            item for item in case["expected_findings"]
            if bool(item.get("should_comment", True)) and not item.get("neutral")
        ]
        findings = recorder.findings
        matches = one_to_one_match(expected, findings, self.line_tolerance)
        matched = {match.predicted_index for match in matches}
        for match in matches:
            finding = findings[match.predicted_index]
            result["exact_location_hits"] += int(match.location_distance == 0)
            result["evidence_hits"] += int(bool(
                finding.evidence_refs or finding.call_chain or finding.evidence.strip()
            ))
        cited: Dict[str, Dict[str, int]] = {}
        for index, finding in enumerate(findings):
            outcome = "tp" if index in matched else "fp"
            for rule_id in getattr(finding, "applied_rules", None) or []:
                cited.setdefault(rule_id, {"tp": 0, "fp": 0})[outcome] += 1
        result["cited_rules"] = cited
        execution = recorder.summary.get("execution") or {}
        result["cost_usd"] = float(execution.get("cost_usd", 0) or 0)
        result["latency_ms"] = int(execution.get("duration_ms", result["latency_ms"]))
        result["llm_calls"] = int(execution.get("llm_calls", 0) or 0)
        result["input_tokens"] = int(execution.get("input_tokens", 0) or 0)
        result["output_tokens"] = int(execution.get("output_tokens", 0) or 0)
        result["total_tokens"] = int(execution.get("total_tokens", 0) or 0)
        instructions = execution.get("instruction_log") or []
        result["instruction_calls"] = len(instructions)
        result["instruction_tokens"] = sum(int(item.get("instruction_tokens", 0)) for item in instructions)
        result["model_roles"] = dict(Counter(
            str(item.get("role")) for item in execution.get("model_call_log") or []
        ))
        collaboration = recorder.summary.get("collaboration") or {}
        decisions = (
            list(collaboration.get("critic_decisions") or [])
            if "critic" in set(collaboration.get("roles") or []) else []
        )
        result["critic_accepted"] = sum(bool(item.get("accepted")) for item in decisions)
        result["critic_rejected"] = sum(not bool(item.get("accepted")) for item in decisions)
        result["revision_requests"] = sum(
            len(item.get("revision_requests") or [])
            for item in (collaboration.get("lead") or {}).get("assessments") or []
        )
        result["revision_results"] = len(collaboration.get("revision_results") or [])
        return result

    @staticmethod
    def _empty_totals():
        values = EndToEndEvaluationHarness._empty_totals()
        values.update({
            "invalid_comments": 0, "exact_location_hits": 0, "evidence_hits": 0,
            "accepted_comments": 0, "closed_comments": 0,
            "latency_ms": 0, "cost_microusd": 0,
            "llm_calls": 0, "input_tokens": 0,
            "output_tokens": 0, "total_tokens": 0,
            "instruction_calls": 0, "instruction_tokens": 0,
            "critic_accepted": 0, "critic_rejected": 0,
            "revision_requests": 0, "revision_results": 0,
        })
        return values

    @staticmethod
    def _accumulate(totals, result):
        EndToEndEvaluationHarness._accumulate(totals, result)
        for field in (
            "invalid_comments", "exact_location_hits", "evidence_hits",
            "accepted_comments", "closed_comments", "latency_ms",
            "llm_calls", "input_tokens", "output_tokens", "total_tokens",
            "instruction_calls", "instruction_tokens",
            "critic_accepted", "critic_rejected",
            "revision_requests", "revision_results",
        ):
            totals[field] += int(result.get(field, 0))
        totals["cost_microusd"] += int(float(result.get("cost_usd", 0)) * 1_000_000)

    @staticmethod
    def _metrics(totals):
        values = EndToEndEvaluationHarness._metrics(totals)
        cases = totals["cases"] or 1
        tp = totals["tp"] or 1
        commented = totals["accepted_comments"] + totals["closed_comments"]
        values.update({
            "invalid_comments_per_pr": round(totals["invalid_comments"] / cases, 4),
            "exact_line_accuracy": round(totals["exact_location_hits"] / tp, 4),
            "evidence_accuracy": round(totals["evidence_hits"] / tp, 4),
            "comment_acceptance_rate": round(
                totals["accepted_comments"] / commented, 4
            ) if commented else None,
            "average_cost_usd_per_pr": round(
                totals["cost_microusd"] / 1_000_000 / cases, 8
            ),
            "average_latency_ms_per_pr": round(totals["latency_ms"] / cases, 2),
            "average_llm_calls_per_pr": round(totals["llm_calls"] / cases, 4),
            "average_input_tokens_per_pr": round(totals["input_tokens"] / cases, 2),
            "average_output_tokens_per_pr": round(totals["output_tokens"] / cases, 2),
            "average_total_tokens_per_pr": round(totals["total_tokens"] / cases, 2),
            "average_instruction_tokens_per_call": round(
                totals["instruction_tokens"] / totals["instruction_calls"], 2
            ) if totals["instruction_calls"] else 0.0,
            "failure_rate": round(
                1 - totals["execution_successes"] / cases, 4
            ),
            "critic_acceptance_rate": round(
                totals["critic_accepted"]
                / (totals["critic_accepted"] + totals["critic_rejected"]), 4
            ) if totals["critic_accepted"] + totals["critic_rejected"] else None,
            "critic_accepted_per_pr": round(totals["critic_accepted"] / cases, 4),
            "critic_rejected_per_pr": round(totals["critic_rejected"] / cases, 4),
            "revision_requests_per_pr": round(totals["revision_requests"] / cases, 4),
            "revision_results_per_pr": round(totals["revision_results"] / cases, 4),
        })
        return values


DEFAULT_COMPARISON_METRICS = (
    "f1", "precision", "recall", "high_risk_recall",
    "severity_accuracy", "clean_accuracy", "exact_line_accuracy",
    "evidence_accuracy", "invalid_comments_per_pr",
    "average_total_tokens_per_pr", "average_latency_ms_per_pr",
    "average_cost_usd_per_pr", "average_instruction_tokens_per_call", "failure_rate",
)


def paired_bootstrap_comparison(
    left: dict, right: dict, iterations: int = 2000, seed: int = 20260819,
    metrics=DEFAULT_COMPARISON_METRICS,
) -> dict:
    """Paired case bootstrap for two reports evaluated in identical order."""
    left_cases = left["case_results"]
    right_cases = right["case_results"]
    if [item["id"] for item in left_cases] != [item["id"] for item in right_cases]:
        raise ValueError("paired comparison requires identical ordered case ids")
    count = len(left_cases)
    iterations = max(200, int(iterations)) if count else 0
    output = {}
    for metric_index, metric in enumerate(metrics):
        if not count:
            output[metric] = {"delta": 0.0, "ci95": [0.0, 0.0], "iterations": 0}
            continue
        rng = random.Random(int(seed) + metric_index)
        deltas = []
        for _ in range(iterations):
            left_totals = ProductionEvaluationHarness._empty_totals()
            right_totals = ProductionEvaluationHarness._empty_totals()
            for _sample in range(count):
                index = rng.randrange(count)
                ProductionEvaluationHarness._accumulate(left_totals, left_cases[index])
                ProductionEvaluationHarness._accumulate(right_totals, right_cases[index])
            left_value = ProductionEvaluationHarness._metrics(left_totals)[metric]
            right_value = ProductionEvaluationHarness._metrics(right_totals)[metric]
            deltas.append(float(right_value or 0) - float(left_value or 0))
        deltas.sort()
        lower = deltas[int((len(deltas) - 1) * 0.025)]
        upper = deltas[int((len(deltas) - 1) * 0.975)]
        point = float(right["metrics"][metric] or 0) - float(left["metrics"][metric] or 0)
        output[metric] = {
            "delta": round(point, 4),
            "ci95": [round(lower, 4), round(upper, 4)],
            "iterations": iterations,
        }
    return output
