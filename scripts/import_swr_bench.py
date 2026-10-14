"""Convert SWR-Bench (arXiv 2509.01494) into DiffJury evaluation cases.

Each SWR-Bench instance is a real GitHub PR reviewed as it stood before its
first human review (the last of ``pr_commits``). Ground truth are the
"change points" a reviewer raised that led to a code change. This importer

1. rebuilds the reviewed diff from local clones: ``git diff base...head``
   (merge-base semantics), fetching ``pull/<n>/head`` when a commit is missing;
2. locates every change point on added lines of that diff by fuzzy-matching
   its (LLM-extracted, sometimes abridged) code snippet;
3. splits by repository so no repository's PRs appear in two splits.

Functional change points (F.*) are scored; evolvability points (E.*) are
written as ``neutral`` labels in the functional dataset (a finding on them is
neither a hit nor a false positive) and as scored labels in the "all" dataset.

    python scripts/import_swr_bench.py --raw <swr_datasets_d5c5.jsonl> \\
        --repos evaluation_data/swrbench/repos --out evaluation_data/swrbench
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from diffjury.diff_parser import parse_unified_diff  # noqa: E402
from diffjury.evaluation_harness import validate_case  # noqa: E402


SPLITS = {
    "scikit-learn/scikit-learn": "train", "matplotlib/matplotlib": "train",
    "pytest-dev/pytest": "train", "sphinx-doc/sphinx": "train",
    "pydata/xarray": "validation", "pylint-dev/pylint": "validation",
    "mwaskom/seaborn": "validation", "pallets/flask": "validation",
    "sympy/sympy": "holdout", "astropy/astropy": "holdout",
    "psf/requests": "holdout", "django/django": "holdout",
}
# Defects that break behaviour count as high risk; other functional points medium.
HIGH_RISK_TYPES = ("F.2", "F.3", "F.6")
MIN_MATCH_CHARS = 6
MAX_GAP = 12


def git(repo: str, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", repo, *args], capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if check and result.returncode != 0:
        raise RuntimeError("git %s failed: %s" % (" ".join(args), result.stderr.strip()[:300]))
    return result.stdout


def has_commit(repo: str, sha: str) -> bool:
    # In a partial clone a missing object triggers a lazy fetch with "noop"
    # negotiation, which re-downloads the whole history behind the commit.
    # Check locally only; ensure_commit fetches with normal negotiation.
    env = {**os.environ, "GIT_NO_LAZY_FETCH": "1"}
    return subprocess.run(
        ["git", "-C", repo, "cat-file", "-e", sha + "^{commit}"],
        capture_output=True, env=env,
    ).returncode == 0


def ensure_commit(repo: str, sha: str, pull_request: int) -> None:
    if has_commit(repo, sha):
        return
    git(repo, "fetch", "--quiet", "--filter=blob:none", "origin",
        "+refs/pull/%d/head:refs/remotes/origin/pr/%d" % (pull_request, pull_request))
    if not has_commit(repo, sha):
        git(repo, "fetch", "--quiet", "--filter=blob:none", "origin", sha, check=False)
    if not has_commit(repo, sha):
        raise RuntimeError("commit %s is not reachable from origin" % sha[:12])


def normalize(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def snippet_lines(snippet: str) -> Tuple[str, List[str], List[str]]:
    """Path (if the snippet starts with one), its added lines and its other code lines."""
    lines = snippet.replace("\r\n", "\n").split("\n")
    path = ""
    if len(lines) > 1 and lines[1].startswith("@@") and not lines[0][:1] in "+- @":
        path, lines = lines[0].strip(), lines[1:]
    added, other = [], []
    for line in lines:
        if line.startswith("@@"):
            continue
        marker, body = (line[:1], line[1:]) if line[:1] in "+- " else ("", line)
        text = normalize(body)
        if len(text) < MIN_MATCH_CHARS or text.startswith("#") or "..." in text:
            continue
        if marker == "+":
            added.append(text)
        elif marker != "-":
            other.append(text)
    return path, added, other


def locate(snippet: str, added_lines: List[Tuple[str, int, str]]) -> Optional[dict]:
    """Map a change-point snippet to the densest run of matching added lines."""
    path, plus, context = snippet_lines(snippet)
    index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for file_path, number, content in added_lines:
        index[normalize(content)].append((file_path, number))
    files = {file_path for file_path, _number, _content in added_lines}
    for candidates in (plus, context):
        hits: Dict[str, List[int]] = defaultdict(list)
        for text in candidates:
            for file_path, number in index.get(text, []):
                if not path or file_path == path or path not in files:
                    hits[file_path].append(number)
        if not hits:
            continue
        best_file = max(hits, key=lambda key: (len(set(hits[key])), key == path))
        numbers = sorted(set(hits[best_file]))
        runs, run = [], [numbers[0]]
        for number in numbers[1:]:
            if number - run[-1] <= MAX_GAP:
                run.append(number)
            else:
                runs.append(run)
                run = [number]
        runs.append(run)
        best = max(runs, key=lambda values: (len(values), -values[0]))
        return {"path": best_file, "start_line": best[0], "end_line": best[-1], "matched_lines": len(best)}
    return None


def build_case(row: dict, repos_dir: str) -> Tuple[Optional[dict], str]:
    repository = row["repo"]
    pull_request = int(row["instance_id"].rsplit("-", 1)[-1])
    repo = os.path.join(repos_dir, repository.replace("/", "__"))
    head = row["pr_commits"][-1]["sha"]
    base = row["base_commit"]
    try:
        ensure_commit(repo, head, pull_request)
        ensure_commit(repo, base, pull_request)
        merge_base = git(repo, "merge-base", base, head).strip()
        diff = git(repo, "diff", "--no-color", "--no-ext-diff", "--no-renames", "-U3",
                   "%s...%s" % (base, head))
    except RuntimeError as exc:
        return None, "git: %s" % exc
    parsed = parse_unified_diff(diff)
    if not parsed.added_lines:
        return None, "no added lines"
    added = [(item.path, item.line, item.content) for item in parsed.added_lines]
    labels, unlocated, unlocated_functional = [], 0, 0
    for change in row["changes"]:
        change_type = str(change.get("change_type", ""))
        functional = change_type.startswith("F")
        where = locate(change["change_introducing"].get("code_snippet", ""), added)
        if where is None:
            unlocated += 1
            unlocated_functional += int(functional)
            continue
        discussion = change.get("change_discussion") or {}
        labels.append({
            **{key: where[key] for key in ("path", "start_line", "end_line")},
            "severity": (
                "high" if change_type.startswith(HIGH_RISK_TYPES)
                else "medium" if functional else "low"
            ),
            "change_type": change_type,
            "category": "functional" if functional else "evolvability",
            "description": str(discussion.get("discussion_summary", ""))[:1500],
            "should_comment": True,
            "matched_lines": where["matched_lines"],
        })
    case = {
        "id": row["instance_id"], "repository": repository, "pull_request": pull_request,
        "split": SPLITS[repository], "created_at": row.get("created_at", ""),
        "base_commit": merge_base, "head_commit": head,
        "title": str(row.get("pr_title", ""))[:300],
        "description": str(row.get("pr_statement", ""))[:4000],
        "diff": diff,
        "change_introduced": bool(row.get("change_introduced")),
        "change_points": len(row["changes"]), "unlocated_change_points": unlocated,
        "unlocated_functional_points": unlocated_functional,
        "expected_findings": labels,
        "source": {
            "kind": "swr-bench", "dataset": "SWR-Bench (arXiv 2509.01494)",
            "public_url": "https://github.com/%s/pull/%d" % (repository, pull_request),
        },
    }
    return case, ""


def variant(case: dict, functional_only: bool) -> dict:
    labels = []
    for label in case["expected_findings"]:
        label = dict(label)
        if functional_only and label["category"] != "functional":
            label["neutral"] = True
            label["should_comment"] = False
        labels.append(label)
    return {**case, "expected_findings": labels}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--raw", required=True)
    parser.add_argument("--repos", default=os.path.join(ROOT, "evaluation_data", "swrbench", "repos"))
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation_data", "swrbench"))
    parser.add_argument("--max-diff-bytes", type=int, default=400_000)
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.raw, encoding="utf-8") if line.strip()]
    os.makedirs(args.out, exist_ok=True)
    cases, skipped = [], Counter()
    stats = Counter()
    # Repositories are independent; one worker per repository keeps git's
    # fetch/lock traffic serial within a clone while lazy blob fetches overlap.
    by_repository: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_repository[row["repo"]].append(row)
    lock = threading.Lock()
    built: Dict[str, dict] = {}
    done = [0]

    def run(repository_rows: List[dict]) -> None:
        for row in repository_rows:
            case, reason = build_case(row, args.repos)
            with lock:
                done[0] += 1
                if case is None:
                    skipped[reason.split(":")[0]] += 1
                    print("skip %s: %s" % (row["instance_id"], reason), flush=True)
                elif len(case["diff"].encode("utf-8")) > args.max_diff_bytes:
                    skipped["diff too large"] += 1
                else:
                    built[row["instance_id"]] = case
                if done[0] % 50 == 0:
                    print("%d/%d processed, %d kept" % (done[0], len(rows), len(built)), flush=True)

    with ThreadPoolExecutor(max_workers=len(by_repository)) as pool:
        list(pool.map(run, by_repository.values()))
    # Keep the dataset's original order so files are reproducible.
    cases = [built[row["instance_id"]] for row in rows if row["instance_id"] in built]
    for case in cases:
        stats["change_points"] += case["change_points"]
        stats["unlocated"] += case["unlocated_change_points"]

    for name, functional_only in (("swr_bench_functional.jsonl", True), ("swr_bench_all.jsonl", False)):
        with open(os.path.join(args.out, name), "w", encoding="utf-8", newline="\n") as handle:
            for case in cases:
                # A defect we could not place on a line would turn its PR into a
                # false "clean" case and penalise the reviewer for finding it.
                if functional_only and case["unlocated_functional_points"]:
                    continue
                value = variant(case, functional_only)
                validate_case(value)
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")

    summary = {
        "kept": len(cases), "skipped": dict(skipped), **stats,
        "excluded_from_functional": sum(bool(case["unlocated_functional_points"]) for case in cases),
        "splits": {},
    }
    for split in ("train", "validation", "holdout"):
        subset = [case for case in cases if case["split"] == split]
        scored = [case for case in subset if not case["unlocated_functional_points"]]
        summary["splits"][split] = {
            "cases": len(subset),
            "functional_dataset_cases": len(scored),
            "functional_dataset_defect_prs": sum(
                any(label["category"] == "functional" for label in case["expected_findings"])
                for case in scored
            ),
            "with_change_points": sum(case["change_introduced"] for case in subset),
            "functional_labels": sum(
                label["category"] == "functional" for case in subset for label in case["expected_findings"]
            ),
            "high_risk_labels": sum(
                label["severity"] == "high" for case in subset for label in case["expected_findings"]
            ),
            "evolvability_labels": sum(
                label["category"] == "evolvability" for case in subset for label in case["expected_findings"]
            ),
            "created_2024_or_later": sum(case["created_at"] >= "2024" for case in subset),
        }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
