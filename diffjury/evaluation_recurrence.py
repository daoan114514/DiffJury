"""Paired, PR-clustered correction checks for previously supplied feedback."""
import random
from typing import Any, Dict, List, Sequence

from .evaluation_harness import one_to_one_match
from .models import Finding, Severity


RECURRENCE_METRICS = ("false_positive_recurrence", "missed_recovery", "correct_retention")
RECURRENCE_NOTICE = "这是对已反馈问题的纠正（同一批 PR），不代表泛化能力。"


def finding_from_dict(value: dict) -> Finding:
    fields = {key: value[key] for key in Finding.__dataclass_fields__ if key in value}
    for key in ("rule_id", "title", "explanation", "path", "evidence", "fix", "test"):
        fields.setdefault(key, "")
    fields["line"] = int(value.get("line", value.get("start_line", 0)) or 0)
    try:
        fields["severity"] = Severity(str(value.get("severity") or "medium"))
    except ValueError:
        fields["severity"] = Severity.MEDIUM
    return Finding(**fields)


def sample_feedback(events: Sequence[dict], cases: Sequence[dict], limit: int = 80,
                    seed: int = 20261013) -> Dict[str, Any]:
    by_id = {case["id"]: case for case in cases}
    if any(case.get("split") != "train" for case in cases):
        raise ValueError("recurrence source cases must be train only")
    chosen, by_type = set(), {}
    for offset, kind in enumerate(("missed", "false_positive")):
        ids = sorted({event["case_id"] for event in events if event["type"] == kind})
        if any(case_id not in by_id for case_id in ids):
            raise ValueError("feedback points outside the training set")
        selected = sorted(random.Random(seed + offset).sample(ids, min(max(0, limit), len(ids))))
        chosen.update(selected)
        by_type[kind] = selected
    selected_events = [event for event in events if event["type"] in by_type
                       and event.get("case_id") in by_type[event["type"]]]
    return {"case_ids": sorted(chosen), "by_type": by_type, "events": selected_events,
            "limit_per_type": limit, "seed": seed}


def recurrence_targets(case: dict, events: Sequence[dict], original_review: dict) -> dict:
    labels = [label for label in case["expected_findings"]
              if not label.get("neutral") and label.get("should_comment", True)]
    old_findings = [finding_from_dict(value) for value in original_review["findings"]]
    retained = sorted({match.expected_index for match in one_to_one_match(labels, old_findings, 2)})
    missed, false_positives = set(), {}
    for event in events:
        if event.get("case_id") != case["id"]:
            continue
        value = event["finding"]
        if event["type"] == "missed":
            index = event.get("label_index")
            if index is None:
                matches = one_to_one_match(labels, [finding_from_dict(value)], 2)
                index = matches[0].expected_index if matches else None
            if index is None or not 0 <= index < len(labels):
                raise ValueError("missed feedback cannot be resolved to an original annotation")
            missed.add(index)
        else:
            line = int(value.get("line", 0) or 0)
            key = (value.get("path", ""), line, value.get("rule_id", ""), value.get("cwe"))
            false_positives[key] = {"path": key[0], "start_line": line, "end_line": line,
                                    "rule_id": key[2], "cwe": key[3]}
    return {"false_positive_recurrence": list(false_positives.values()),
            "missed_recovery": [labels[index] for index in sorted(missed)],
            "correct_retention": [labels[index] for index in retained]}


def measure_recurrence(targets: dict, findings: List[Finding]) -> dict:
    return {metric: {"numerator": len(one_to_one_match(targets[metric], findings, 2)),
                     "denominator": len(targets[metric])} for metric in RECURRENCE_METRICS}


def recurrence_rates(rows: Sequence[dict]) -> dict:
    result = {}
    for metric in RECURRENCE_METRICS:
        numerator = sum(row["metrics"][metric]["numerator"] for row in rows)
        denominator = sum(row["metrics"][metric]["denominator"] for row in rows)
        result[metric] = {"numerator": numerator, "denominator": denominator,
                          "rate": round(numerator / denominator, 6) if denominator else None}
    return result


def paired_recurrence_bootstrap(left: Sequence[dict], right: Sequence[dict],
                                iterations: int = 2000, seed: int = 20261013) -> dict:
    if [row["id"] for row in left] != [row["id"] for row in right]:
        raise ValueError("paired recurrence requires identical ordered PR ids")
    if any(not row.get("execution_success", True) for row in list(left) + list(right)):
        raise ValueError("exclude failed replays before paired recurrence comparison")
    before, after = recurrence_rates(left), recurrence_rates(right)
    output = {}
    for offset, metric in enumerate(RECURRENCE_METRICS):
        if before[metric]["rate"] is None or after[metric]["rate"] is None:
            output[metric] = {"delta": None, "ci95": None, "iterations": 0, "paired_prs": len(left)}
            continue
        rng, samples = random.Random(seed + offset), []
        counts = [(row["metrics"][metric]["numerator"], row["metrics"][metric]["denominator"],
                   right[index]["metrics"][metric]["numerator"], right[index]["metrics"][metric]["denominator"])
                  for index, row in enumerate(left)]
        for _ in range(max(200, iterations)):
            a, b, c, d = 0, 0, 0, 0
            for _ in counts:
                row = counts[rng.randrange(len(counts))]
                a += row[0]
                b += row[1]
                c += row[2]
                d += row[3]
            if b and d:
                samples.append(c / d - a / b)
        samples.sort()
        output[metric] = {"delta": round(after[metric]["rate"] - before[metric]["rate"], 6),
                          "ci95": [round(samples[int((len(samples) - 1) * q)], 6) for q in (0.025, 0.975)],
                          "iterations": len(samples), "paired_prs": len(left),
                          "sampling_unit": "PR", "higher_is_better": metric != "false_positive_recurrence"}
    return output
