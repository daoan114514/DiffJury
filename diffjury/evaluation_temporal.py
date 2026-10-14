"""Chronological within-repository SWR splits, with explicit leakage checks."""
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence, Tuple


TEMPORAL_REPOSITORIES = ("scikit-learn/scikit-learn", "sympy/sympy",
                         "matplotlib/matplotlib", "astropy/astropy")


def _time(case: dict) -> datetime:
    value = str(case.get("created_at") or "")
    if not value:
        raise ValueError("temporal cases require created_at")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def temporal_split(cases: Sequence[dict], repositories: Sequence[str] = TEMPORAL_REPOSITORIES
                   ) -> Tuple[List[dict], Dict[str, Any]]:
    selected = [case for case in cases if case["repository"] in repositories]
    ids, prs = set(), set()
    for case in selected:
        pr = (case["repository"], case["pull_request"])
        if case["id"] in ids or pr in prs:
            raise ValueError("a PR or case appears more than once in the temporal input")
        ids.add(case["id"])
        prs.add(pr)
    output, boundaries = [], {}
    for repository in repositories:
        rows = sorted((case for case in selected if case["repository"] == repository),
                      key=lambda case: (_time(case), str(case["id"])))
        if len(rows) < 10:
            raise ValueError("temporal repository needs at least 10 PRs: %s" % repository)
        first, second = int(len(rows) * 0.6), int(len(rows) * 0.7)
        groups = {"train": rows[:first], "validation": rows[first:second], "holdout": rows[second:]}
        if not (_time(groups["train"][-1]) <= _time(groups["validation"][0])
                and _time(groups["validation"][-1]) <= _time(groups["holdout"][0])):
            raise ValueError("chronological split overlaps")
        boundaries[repository] = {split: {"count": len(values),
                                          "first_created_at": values[0]["created_at"],
                                          "last_created_at": values[-1]["created_at"]}
                                  for split, values in groups.items()}
        boundaries[repository]["tied_boundary_timestamps"] = {
            "train_validation": _time(groups["train"][-1]) == _time(groups["validation"][0]),
            "validation_holdout": _time(groups["validation"][-1]) == _time(groups["holdout"][0]),
        }
        for split, values in groups.items():
            output.extend({**case, "split": split} for case in values)
    counts = dict(Counter(case["split"] for case in output))
    return output, {"protocol": "within-repository created_at order; floor(0.6*N), floor(0.7*N)",
                    "cases": len(output), "by_split": counts, "repositories": boundaries,
                    "case_disjoint": True, "pr_disjoint": True, "chronological": True}
