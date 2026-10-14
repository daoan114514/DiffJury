"""Refresh finished task 9 reports and verify frozen inputs without model calls."""
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from diffjury.config import Settings
from diffjury.evaluation_replay import read_json, write_json
from monitor_swr_rot_experiment import process_alive
from run_swr_temporal_v2 import refresh_reports


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    out = ROOT / "evaluation_results/swr_rot_temporal_v2"
    process = read_json(out / "process.json")
    state = read_json(out / "state.json")
    if process_alive(process["pid"]) or not state.get("finished_at"):
        raise ValueError("finalization requires the task process to exit successfully")
    refresh_reports(out)
    state = read_json(out / "state.json")
    protected = read_json(ROOT / "evaluation_results/handoff_2_protected_checksums.json")
    frozen = read_json(out / "copied_inputs.json")
    protected_bad = [name for name, expected in protected.items()
                     if not (ROOT / name).is_file() or digest(ROOT / name) != expected]
    frozen_bad = [name for name, row in frozen.items() if digest(out / name) != row["sha256"]]
    split = read_json(ROOT / "evaluation_data/swrbench/swr_bench_temporal_summary.json")
    data_hashes = {name: digest(ROOT / "evaluation_data/swrbench" / name) for name in
                   ("swr_bench_functional.jsonl", "swr_bench_temporal.jsonl")}
    expected_data = [split["source_sha256"], split["dataset_sha256"]]
    data_ok = list(data_hashes.values()) == expected_data
    cases = {row["id"]: row for row in (json.loads(line) for line in
             (ROOT / "evaluation_data/swrbench/swr_bench_temporal.jsonl").read_text(encoding="utf-8").splitlines())}
    with sqlite3.connect((out / "governed.db").as_uri() + "?mode=ro", uri=True) as conn:
        events = [{"item_id": row[0], "event": row[1], "detail": json.loads(row[2]), "created_at": row[3]}
                  for row in conn.execute("SELECT item_id,event,detail_json,created_at FROM knowledge_events ORDER BY created_at")]
        scopes = dict(conn.execute("SELECT id,scope FROM knowledge_items"))
        releases = [{"status": row[0], "gate": json.loads(row[1])}
                    for row in conn.execute("SELECT status,gate_json FROM knowledge_releases")]
    wrong_training = []
    for row in events:
        if row["event"] not in {"counterfactual", "source_validation_replay"}:
            continue
        detail = row["detail"]
        case = cases.get(detail.get("case_id"))
        if (not case or case["split"] != "train" or detail.get("split") != "train"
                or case["repository"] != detail.get("repository")
                or scopes[row["item_id"]] not in {case["repository"], "*"}):
            wrong_training.append(row["item_id"])
    source_rounds = state["governance"]["source_validation"]["rounds"]
    budgets_ok = all(row["review_attempts"] <= row["budget_consumed"] <= row["replay_limit"]
                     for row in source_rounds.values())
    released_gates_ok = all(row["gate"].get("passed") and row["gate"].get("cases") == 73
                            for row in releases if row["status"] == "released")
    audit = read_json(out / "resume_audit.json")
    cutoff = datetime.fromisoformat(audit["snapshots"][0]["captured_at"])
    new_events = [row for row in events if datetime.fromisoformat(row["created_at"]) > cutoff]
    cf_rounds = defaultdict(lambda: {"pairs": 0, "review_attempts": 0, "applied": 0, "failed": 0, "not_exposed": 0})
    for row in events:
        if row["event"] == "counterfactual":
            detail = row["detail"]
            record = cf_rounds[detail["round_id"]]
            record["pairs"] += 1
            record["review_attempts"] += detail.get("replays", 0)
            record["applied"] += bool(detail.get("applied"))
            record["failed"] += "error_type" in detail
            record["not_exposed"] += "skipped" in detail
    observations = [json.loads(line) for path in out.glob("gate_cache_observations_*.jsonl")
                    for line in path.read_text(encoding="utf-8").splitlines()]
    replacements = [row for row in observations if row["kind"] == "cache_replaced"]
    new_cf = [row for row in new_events if row["event"] == "counterfactual"]
    audit["final"] = {"finished_at": state["finished_at"], "event_counts": dict(Counter(row["event"] for row in events)),
        "new_event_counts": dict(Counter(row["event"] for row in new_events)),
        "new_counterfactual_review_attempts": sum(row["detail"].get("replays", 0) for row in new_cf),
        "counterfactual_rounds": dict(cf_rounds),
        "observed_gate_review_attempts_lower_bound": sum(row["review_attempts_lower_bound"] for row in replacements),
        "observed_gate_cache_replacements": len(replacements),
        "gate_cache_snapshots": len(list((out / "gate_cache").glob("*.json"))),
        "gate_attempt_notice": "Cache observations give a lower bound only; historical failed retries and unobserved replacements are not recoverable.",
        "holdout_failures": state["holdout_failures"], "recurrence_failures": state["recurrence"]["failures"],
        "stop_log_entries": sum("stopped:" in line for line in (out / "run.log").read_text(encoding="utf-8").splitlines())}
    write_json(out / "resume_audit.json", audit)
    final = audit["final"]
    notes = (out / "NOTES.md").read_text(encoding="utf-8")
    notes += "\n## 续跑的实际账目\n\n"
    notes += ("原 PID 28840 在第 200 轮 validation 门禁连接失败后退出，当时事件进度 200/243，来源审查 226 次，已快速转正 7 条。"
              "本次按原参数从检查点续跑，已有转正和来源验证结果保留；没有重新执行原 baseline/naive 或已完成的任务 7 诊断。\n\n")
    notes += "日志累计有 %d 次显式 stopped；最终完成时间 %s。\n\n" % (final["stop_log_entries"], state["finished_at"])
    notes += ("续跑后新增来源审查 %d 次、新增反事实比较 %d 对（%d 次审查），新增快速转正 %d 次。"
              "这些数字包括后续第 243 轮正常维护，不能全部称为失败重试。\n\n") % (
        final["new_event_counts"].get("source_validation_replay", 0), len(new_cf),
        final["new_counterfactual_review_attempts"], final["new_event_counts"].get("source_validation_promotion", 0))
    notes += "| 反事实维护轮次 | 累计比较 | 累计审查尝试 | 记账比较 | 失败 | 未注入 |\n|---|---:|---:|---:|---:|---:|\n"
    for round_id, row in sorted(cf_rounds.items(), key=lambda pair: int(pair[0])):
        notes += "| %s | %d | %d | %d | %d | %d |\n" % (round_id, row["pairs"], row["review_attempts"], row["applied"], row["failed"], row["not_exposed"])
    notes += ("\n第 200 轮来源验证在续跑中新增 0 次审查，仍按持久化预算累计；原反事实阶段只对已应用的条目/PR 组合去重，"
              "重新进入同一轮仍可能增加比较，因此不能声称所有训练回放均被完整复用。state 中阶段摘要可能被续跑摘要覆盖，本报告使用数据库累计账目。\n\n")
    notes += ("续跑门禁观察器记录的审查尝试下界为 %d；最终缓存保存 %d 个不同快照，每快照 73 个 validation PR。"
              "缓存槽位数不等于模型调用数。首次运行全部失败重试次数及未观测到的缓存替换无法精确追溯，故不报告门禁累计调用的精确总数。\n\n") % (
        final["observed_gate_review_attempts_lower_bound"], final["gate_cache_snapshots"])
    notes += "最终 holdout 失败比例与冻结标记：`%s`；纠正率失败 PR 数：`%s`。失败记录保留，共同成功样本见前表。\n\n" % (
        json.dumps(final["holdout_failures"], ensure_ascii=False), json.dumps(final["recurrence_failures"], ensure_ascii=False))
    notes += "详细审计：[resume_audit.json](resume_audit.json)；门禁观察记录：`gate_cache_observations_*.jsonl`。\n"
    (out / "NOTES.md").write_text(notes, encoding="utf-8")
    text_paths = list(out.rglob("*.json")) + list(out.rglob("*.jsonl")) + list(out.rglob("*.md")) + list(out.rglob("*.log"))
    text_paths += [ROOT / "evaluation_results" / name for name in ("CODEX_PROGRESS.md", "CODEX_PROGRESS_2.md")]
    encoding_bad, replacement_bad, credential_bad = [], [], []
    secret = str(Settings.from_env().resolved_llm().get("api_key") or "")
    for path in text_paths:
        try:
            content = path.read_text(encoding="utf-8", errors="strict")
        except UnicodeDecodeError:
            encoding_bad.append(str(path.relative_to(ROOT))); continue
        if "\ufffd" in content or "????" in content:
            replacement_bad.append(str(path.relative_to(ROOT)))
        if secret and secret in content:
            credential_bad.append(str(path.relative_to(ROOT)))
    registered = []
    for repo in (ROOT / "evaluation_data/swrbench/repos").iterdir():
        if not (repo / ".git").exists():
            continue
        result = subprocess.run(["git", "-C", str(repo), "worktree", "list", "--porcelain"],
                                capture_output=True, text=True, encoding="utf-8", check=True)
        registered += [line for line in result.stdout.splitlines() if line.startswith("worktree ")
                       and line[9:].replace("\\", "/").lower().startswith("c:/dj_wt/t9/")]
    worktree_root = Path("C:/dj_wt/t9")
    remaining = list(worktree_root.iterdir()) if worktree_root.exists() else []
    qa = {"protected_files": len(protected), "protected_mismatches": protected_bad,
          "frozen_inputs": len(frozen), "frozen_mismatches": frozen_bad, "dataset_hashes_ok": data_ok,
          "training_scope_errors": wrong_training, "source_validation_budgets_ok": budgets_ok,
          "released_gates_use_73_validation_prs_and_pass": released_gates_ok,
          "text_files_checked": len(text_paths), "non_utf8": encoding_bad, "replacement_text": replacement_bad,
          "credential_match_count": len(credential_bad), "credential_match_paths": credential_bad,
          "t9_registered_worktrees": registered, "t9_remaining_entries": len(remaining),
          "notes_chinese_characters": sum("\u4e00" <= char <= "\u9fff" for char in notes)}
    qa["passed"] = not any((protected_bad, frozen_bad, wrong_training, encoding_bad, replacement_bad,
                             credential_bad, registered, remaining)) and data_ok and budgets_ok and released_gates_ok
    write_json(out / "final_qa.json", qa)
    print(json.dumps(qa, ensure_ascii=False))
    if not qa["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
