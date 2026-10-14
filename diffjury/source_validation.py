"""Source-PR screening judgments; targets always come from training feedback."""
import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence

from .assembler import ContextAssembler, StaticKnowledge, TaskSignals
from .diff_parser import parse_unified_diff
from .evaluation_harness import one_to_one_match
from .evaluation_recurrence import recurrence_targets
from .knowledge import KnowledgeItem, ROLES
from .skills import AgentSkill


SOURCES = {"false_positive", "missed"}
TERMINAL = {"passed", "neutral", "harmful"}


def validation_signature(item: KnowledgeItem) -> str:
    fields = {name: getattr(item, name) for name in ("kind", "source", "content", "body", "role", "scope", "triggers")}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def can_inject(assembler: ContextAssembler, item: KnowledgeItem, case: dict,
               stable: Sequence[KnowledgeItem], disk_skills: Sequence[Any] = ()) -> bool:
    if item.scope not in {"*", case["repository"]}:
        return False
    signals = TaskSignals.from_review(case["diff"], parse_unified_diff(case["diff"]).files, case["repository"])
    view = StaticKnowledge(assembler, list(stable) + [item]).prepare("default", case["repository"], "stable", signals)
    if item.kind == "rule":
        return any(item.id in view.rules_for(role).ids for role in ROLES if role != "*")
    skills = list(disk_skills) + [AgentSkill(record.name, record.content, record.body, record.instruction_text(), source="learned")
                                 for record in list(stable) + [item] if record.kind == "skill" and record.body]
    catalog = view.skill_catalog(skills)
    selected = [skill for skill in catalog if skill.name == item.name and skill.content == item.instruction_text()]
    return bool(selected and view.skill_bodies("source-validation", selected)["skills"])


def targets_for(case: dict, event: dict, original_review: dict) -> dict:
    if original_review.get("id", case["id"]) != case["id"]:
        raise ValueError("source review id differs from the training PR")
    return recurrence_targets(case, [event], original_review)


def observation(case: dict, findings: list, targets: Optional[dict] = None, source: str = "") -> Dict[str, Any]:
    labels = [label for label in case["expected_findings"] if not label.get("neutral") and label.get("should_comment", True)]
    matches = one_to_one_match(labels, findings, 2)
    matched = {match.predicted_index for match in matches}
    leftovers = [finding for index, finding in enumerate(findings) if index not in matched]
    neutral = [label for label in case["expected_findings"] if label.get("neutral")]
    fp = len(leftovers) - len(one_to_one_match(neutral, leftovers, 2))
    result = {"hits": sorted({match.expected_index for match in matches}), "false_positives": fp}
    if targets is not None:
        key = "false_positive_recurrence" if source == "false_positive" else "missed_recovery"
        result.update({"target_hits": len(one_to_one_match(targets[key], findings, 2)),
                       "retained_hits": sorted({match.expected_index for match in one_to_one_match(targets["correct_retention"], findings, 2)})})
    return result


def pair_worsened(before: dict, after: dict, source: str = "") -> bool:
    if not set(before["hits"]).issubset(after["hits"]):
        return True
    if source == "false_positive":
        return after["target_hits"] > before["target_hits"]
    if source == "missed":
        return after["false_positives"] > before["false_positives"] or after["target_hits"] < before["target_hits"]
    return False


def source_judgment(source: str, before: List[dict], after: List[dict], retained_count: int) -> dict:
    if len(before) != 3 or len(after) != 3:
        raise ValueError("source validation requires exactly three complete pairs")
    target_before = sum(row["target_hits"] for row in before)
    target_after = sum(row["target_hits"] for row in after)
    no_loss = sum(len(row["hits"]) for row in after) >= sum(len(row["hits"]) for row in before)
    retained = all(len(row["retained_hits"]) == retained_count for row in after)
    if source == "false_positive":
        improves = target_before >= 2 and target_after <= 1
    elif source == "missed":
        improves = target_after > target_before and sum(row["false_positives"] for row in after) <= sum(row["false_positives"] for row in before)
    else:
        raise ValueError("unsupported source validation category")
    harmful = (not no_loss or sum(len(row["retained_hits"]) for row in after) < sum(len(row["retained_hits"]) for row in before)
               or (target_after > target_before if source == "false_positive" else
                   target_after < target_before or sum(row["false_positives"] for row in after) > sum(row["false_positives"] for row in before)))
    passed = improves and no_loss and retained
    return {"passed": passed, "target_before": target_before, "target_after": target_after,
            "original_hits_retained": retained, "source_hits_not_lost": no_loss,
            "status": "passed" if passed else "harmful" if harmful else "neutral"}


def pair_credit(before: dict, after: dict, source: str = "") -> tuple:
    harmful = int(pair_worsened(before, after, source))
    better = (after["target_hits"] < before["target_hits"] if source == "false_positive" else
              after["target_hits"] > before["target_hits"] if source == "missed" else len(after["hits"]) > len(before["hits"]))
    helpful = int(better and not harmful)
    return helpful, harmful, int(not helpful and not harmful)
