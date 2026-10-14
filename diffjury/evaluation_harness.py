"""End-to-end PR diff evaluation with reproducible one-to-one matching."""
import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .diff_parser import parse_unified_diff
from .finding_identity import canonical_identity
from .models import Finding
from .reviewer import Reviewer


SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}
DEFAULT_REPLAY_DATASET = os.path.abspath(os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "evaluation_data", "pr_diff_100.jsonl",
))


@dataclass
class Match:
    expected_index: int
    predicted_index: int
    location_distance: int


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def dataset_fingerprint(cases: Iterable[dict]) -> str:
    """Fingerprint exactly what is scored, independent of JSONL formatting."""
    digest = hashlib.sha256()
    for case in sorted(cases, key=lambda item: str(item["id"])):
        digest.update(_canonical_json(case).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_jsonl(path: str) -> List[dict]:
    cases = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                case = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError("invalid JSON on line %d: %s" % (line_number, exc)) from exc
            validate_case(case, line_number)
            cases.append(case)
    ids = [str(case["id"]) for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("dataset contains duplicate case ids")
    return cases


def load_replay_cases(path: str = "", splits: Optional[Iterable[str]] = None) -> List[dict]:
    """Load the labelled replay dataset, optionally restricted to some splits."""
    cases = load_jsonl(path or DEFAULT_REPLAY_DATASET)
    if splits is None:
        return cases
    wanted = set(splits)
    return [case for case in cases if case["split"] in wanted]


def validate_case(case: dict, line_number: int = 0) -> None:
    prefix = "dataset line %d" % line_number if line_number else "evaluation case"
    for field in ("id", "repository", "pull_request", "split", "diff", "expected_findings"):
        if field not in case:
            raise ValueError("%s is missing %s" % (prefix, field))
    if case["split"] not in {"train", "validation", "holdout"}:
        raise ValueError("%s has invalid split" % prefix)
    parsed = parse_unified_diff(str(case["diff"]))
    if not parsed.files or not parsed.added_lines:
        raise ValueError("%s does not contain a scoreable unified diff" % prefix)
    if not isinstance(case["expected_findings"], list):
        raise ValueError("%s expected_findings must be an array" % prefix)
    added_locations = {
        (_normalized_path(item.path), int(item.line)) for item in parsed.added_lines
    }
    for expected in case["expected_findings"]:
        # cwe is optional: labels without one are matched by location only.
        for field in ("path", "start_line", "end_line", "severity"):
            if field not in expected:
                raise ValueError("%s finding is missing %s" % (prefix, field))
        if str(expected["severity"]).lower() not in SEVERITY_RANK:
            raise ValueError("%s finding has invalid severity" % prefix)
        if int(expected["start_line"]) > int(expected["end_line"]):
            raise ValueError("%s finding has an inverted line range" % prefix)
        expected_path = _normalized_path(str(expected["path"]))
        if not any(
            path == expected_path
            and int(expected["start_line"]) <= line <= int(expected["end_line"])
            for path, line in added_locations
        ):
            raise ValueError("%s finding does not cover an added line" % prefix)


def _normalized_path(path: str) -> str:
    value = path.replace("\\", "/").strip()
    return value[2:] if value.startswith(("a/", "b/")) else value


def _candidate_edges(
    expected: List[dict], predicted: List[Finding], line_tolerance: int,
) -> Dict[int, List[Tuple[int, int]]]:
    edges: Dict[int, List[Tuple[int, int]]] = {}
    for expected_index, truth in enumerate(expected):
        start = int(truth["start_line"])
        end = int(truth["end_line"])
        truth_path = _normalized_path(str(truth["path"]))
        truth_identity = canonical_identity(
            truth.get("rule_id", ""), truth.get("cwe", "")
        )
        options = []
        for predicted_index, finding in enumerate(predicted):
            if _normalized_path(finding.path) != truth_path:
                continue
            if truth_identity and canonical_identity(
                finding.rule_id, getattr(finding, "cwe", "")
            ) != truth_identity:
                continue
            if start <= finding.line <= end:
                distance = 0
            else:
                distance = min(abs(finding.line - start), abs(finding.line - end))
            if distance <= line_tolerance:
                options.append((predicted_index, distance))
        edges[expected_index] = sorted(options, key=lambda item: (item[1], item[0]))
    return edges


def one_to_one_match(
    expected: List[dict], predicted: List[Finding], line_tolerance: int = 2,
) -> List[Match]:
    """Maximum-cardinality bipartite matching with deterministic edge ordering."""
    edges = _candidate_edges(expected, predicted, line_tolerance)
    prediction_owner: Dict[int, int] = {}

    def assign(expected_index: int, visited: set) -> bool:
        for predicted_index, _distance in edges.get(expected_index, []):
            if predicted_index in visited:
                continue
            visited.add(predicted_index)
            previous = prediction_owner.get(predicted_index)
            if previous is None or assign(previous, visited):
                prediction_owner[predicted_index] = expected_index
                return True
        return False

    # Constrained truths go first so flexible ranges do not consume their only edge.
    order = sorted(range(len(expected)), key=lambda index: (len(edges[index]), index))
    for expected_index in order:
        assign(expected_index, set())

    matches = []
    for predicted_index, expected_index in prediction_owner.items():
        distance = next(
            distance for index, distance in edges[expected_index]
            if index == predicted_index
        )
        matches.append(Match(expected_index, predicted_index, distance))
    return sorted(matches, key=lambda item: (item.expected_index, item.predicted_index))


class EndToEndEvaluationHarness:
    def __init__(self, line_tolerance: int = 2):
        self.line_tolerance = line_tolerance

    def run(self, reviewer: Reviewer, cases: List[dict], name: str = "") -> Dict[str, Any]:
        started = time.monotonic()
        totals = self._empty_totals()
        case_results = []
        for case in cases:
            result = self._run_case(reviewer, case)
            case_results.append(result)
            self._accumulate(totals, result)
        return self._report(reviewer, cases, name, totals, case_results, started)

    def _report(
        self, reviewer: Reviewer, cases: List[dict], name: str, totals: Dict[str, int],
        case_results: List[dict], started: float,
    ) -> Dict[str, Any]:
        by_split = {}
        for split in ("train", "validation", "holdout"):
            split_totals = self._empty_totals()
            for item in case_results:
                if item["split"] == split:
                    self._accumulate(split_totals, item)
            by_split[split] = self._metrics(split_totals)
        source_kinds = sorted({
            str((case.get("source") or {}).get("kind", "unknown")) for case in cases
        })
        return {
            "schema_version": 2,
            "name": name or reviewer.name,
            "reviewer": reviewer.name,
            "dataset": {
                "cases": len(cases),
                "repositories": len({case["repository"] for case in cases}),
                "risk_cases": sum(bool(case["expected_findings"]) for case in cases),
                "clean_cases": sum(not case["expected_findings"] for case in cases),
                "source_kinds": source_kinds,
                "sha256": dataset_fingerprint(cases),
            },
            "metrics": self._metrics(totals),
            "by_split": by_split,
            "duration_seconds": round(time.monotonic() - started, 4),
            "case_results": case_results,
        }

    def _run_case(self, reviewer: Reviewer, case: dict) -> Dict[str, Any]:
        expected = [
            item for item in case["expected_findings"]
            if bool(item.get("should_comment", True)) and not item.get("neutral")
        ]
        # Neutral labels (e.g. style-only review points when scoring defects)
        # are neither required nor penalised: a finding on one is not an FP.
        neutral = [item for item in case["expected_findings"] if item.get("neutral")]
        result = {
            "id": case["id"],
            "repository": case["repository"],
            "pull_request": case["pull_request"],
            "split": case["split"],
            "expected": len(expected),
            "predicted": 0,
            "tp": 0,
            "fp": 0,
            "fn": len(expected),
            "severity_hits": 0,
            "high_total": sum(
                str(item["severity"]).lower() in {"high", "critical"} for item in expected
            ),
            "high_hits": 0,
            "clean_hit": False,
            "neutral_hits": 0,
            "execution_success": False,
            "matches": [],
            "error": None,
        }
        try:
            parsed = parse_unified_diff(case["diff"])
            review_case = getattr(reviewer, "review_case", None)
            findings = (
                review_case(case, parsed)
                if review_case else reviewer.review(case["diff"], parsed)
            )
            matches = one_to_one_match(expected, findings, self.line_tolerance)
            matched = {match.predicted_index for match in matches}
            leftover = [finding for index, finding in enumerate(findings) if index not in matched]
            neutral_hits = len(one_to_one_match(neutral, leftover, self.line_tolerance)) if neutral else 0
            result["predicted"] = len(findings)
            result["tp"] = len(matches)
            result["neutral_hits"] = neutral_hits
            result["fp"] = len(findings) - len(matches) - neutral_hits
            result["fn"] = len(expected) - len(matches)
            result["clean_hit"] = not expected and result["fp"] == 0
            result["execution_success"] = True
            for match in matches:
                truth = expected[match.expected_index]
                finding = findings[match.predicted_index]
                severity_hit = finding.severity.value == str(truth["severity"]).lower()
                high = str(truth["severity"]).lower() in {"high", "critical"}
                result["severity_hits"] += int(severity_hit)
                result["high_hits"] += int(high)
                result["matches"].append({
                    "expected_index": match.expected_index,
                    "predicted_index": match.predicted_index,
                    "path": finding.path,
                    "line": finding.line,
                    "cwe": canonical_identity(
                        finding.rule_id, getattr(finding, "cwe", "")
                    ),
                    "rule_id": finding.rule_id,
                    "expected_severity": truth["severity"],
                    "predicted_severity": finding.severity.value,
                    "severity_hit": severity_hit,
                    "location_distance": match.location_distance,
                })
        except Exception as exc:
            result["error"] = str(exc)[:1000]
        return result

    @staticmethod
    def _empty_totals() -> Dict[str, int]:
        return {
            "cases": 0, "risk_cases": 0, "clean_cases": 0, "tp": 0, "fp": 0,
            "fn": 0, "severity_hits": 0, "high_total": 0, "high_hits": 0,
            "clean_hits": 0, "execution_successes": 0,
        }

    @staticmethod
    def _accumulate(totals: Dict[str, int], result: dict) -> None:
        totals["cases"] += 1
        totals["risk_cases"] += int(result["expected"] > 0)
        totals["clean_cases"] += int(result["expected"] == 0)
        for field in ("tp", "fp", "fn", "severity_hits", "high_total", "high_hits"):
            totals[field] += int(result[field])
        totals["clean_hits"] += int(result["clean_hit"])
        totals["execution_successes"] += int(result["execution_success"])

    @staticmethod
    def _metrics(totals: Dict[str, int]) -> Dict[str, Any]:
        def ratio(numerator: int, denominator: int, empty: float = 1.0) -> float:
            return round(numerator / denominator, 4) if denominator else empty

        precision = ratio(totals["tp"], totals["tp"] + totals["fp"], 0.0)
        recall = ratio(totals["tp"], totals["tp"] + totals["fn"], 1.0)
        f1 = round(
            2 * precision * recall / (precision + recall), 4
        ) if precision + recall else 0.0
        return {
            **totals,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "severity_accuracy": ratio(totals["severity_hits"], totals["tp"]),
            "high_risk_recall": ratio(totals["high_hits"], totals["high_total"]),
            "clean_accuracy": ratio(totals["clean_hits"], totals["clean_cases"]),
            "execution_success_rate": ratio(
                totals["execution_successes"], totals["cases"], 0.0
            ),
        }
