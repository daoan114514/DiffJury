"""Replay only governed learning, keeping copied comparison arms immutable."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from diffjury.assembler import StaticKnowledge
from diffjury.config import Settings
from diffjury.evaluation_governance import read_governance
from diffjury.evaluation_harness import dataset_fingerprint, load_jsonl
from diffjury.evaluation_notes import build_notes, interval
from diffjury.evaluation_recurrence import finding_from_dict, measure_recurrence, paired_recurrence_bootstrap, recurrence_rates
from diffjury.evaluation_replay import (aggregate, background, check_output, checksum, make_reviewer,
    read_json, run_cached_replays, write_json)
from diffjury.evaluation_repos import GitSnapshots
from diffjury.evaluation_v2 import paired_bootstrap_comparison
from diffjury.llm import JsonChatClient
from diffjury.skills import SkillRegistry
from monitor_swr_rot_experiment import process_alive
from run_swr_rot_experiment import FakeSwrClient, SwrArm, classify, functional, write_summary
import run_rot_stress_test as rot


def log(message: str) -> None:
    print("[%s] %s" % (time.strftime("%H:%M:%S"), message), flush=True)


def copy_inputs(source: Path, diagnostic: Path, out: Path) -> dict:
    original, diag_state = read_json(source / "state.json"), read_json(diagnostic / "state.json")
    if not original or not original.get("finished_at") or not diag_state or not diag_state.get("finished_at"):
        raise ValueError("source experiment and diagnostics must be finished")
    wanted = ["events.json", "naive.db", "holdout_rows/baseline.json", "holdout_rows/final-naive.json",
              "recurrence_rows/targets.json"]
    wanted += [str(path.relative_to(source)).replace("\\", "/") for name in
               ("train_reviews", "recurrence_rows/baseline", "recurrence_rows/naive") for path in (source / name).glob("*.json")]
    mapping = {name: source / name for name in wanted}
    mapping["holdout_rows/baseline_rerun.json"] = diagnostic / "holdout_rows/baseline_rerun.json"
    mapping["source_state.json"] = source / "state.json"
    mapping["diagnostic.json"] = diagnostic / "diag.json"
    mapping["diagnostic_manifest.json"] = diagnostic / "manifest.json"
    expected = {name: {"source": str(path.resolve()), "sha256": checksum(path)} for name, path in mapping.items()}
    manifest_path = out / "copied_inputs.json"
    previous = read_json(manifest_path)
    if previous is not None and previous != expected:
        raise ValueError("frozen comparison inputs differ from the copied manifest")
    for name, path in mapping.items():
        target = out / name
        if target.exists():
            if checksum(target) != expected[name]["sha256"]:
                raise ValueError("copied comparison input was changed: %s" % name)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    write_json(manifest_path, expected)
    return original


def check_copied_inputs(out: Path) -> None:
    manifest = read_json(out / "copied_inputs.json")
    if not manifest or any(checksum(out / name) != row["sha256"] for name, row in manifest.items()):
        raise ValueError("frozen comparison input checksum mismatch")


def compare_holdout(reports: dict, log_fn=log) -> dict:
    paired_ids = set.intersection(*({row["id"] for row in value["case_results"] if row["execution_success"]} for value in reports.values()))
    if not paired_ids:
        raise ValueError("holdout has no jointly successful PRs")
    subsets = {name: aggregate(sorted((row for row in report["case_results"] if row["id"] in paired_ids), key=lambda row: row["id"]))
               for name, report in reports.items()}
    comparisons = {}
    pairs = [(reference, governed) for reference in ("baseline", "naive", "baseline_rerun")
             for governed in ("governed", "governed_canary")]
    pairs += [("governed", "governed_canary"), ("baseline", "naive"), ("baseline_rerun", "baseline")]
    for left, right in pairs:
        name = "%s_minus_%s" % (right, left)
        log_fn("holdout paired bootstrap: %s" % name)
        comparisons[name] = paired_bootstrap_comparison(subsets[left], subsets[right], iterations=2000,
            metrics=("f1", "precision", "recall", "high_risk_recall", "clean_accuracy",
                     "average_instruction_tokens_per_call", "average_total_tokens_per_pr"))
    return {"paired_case_ids": sorted(paired_ids), "comparisons": comparisons,
            "scores": {name: report["metrics"] for name, report in subsets.items()}}


def run_recurrence(args, out: Path, original: dict, train: list, reviewers: dict, skills: dict, secret: str = "") -> dict:
    sample = original["recurrence"]["sample"]
    selected = [case for case in train if case["id"] in set(sample["case_ids"])]
    if len(selected) != sample["selected_prs"] or any(case["split"] != "train" for case in selected):
        raise ValueError("copied recurrence selection is not the original training cohort")
    targets = read_json(out / "recurrence_rows/targets.json")
    reports = {}
    for name in ("baseline", "naive"):
        reports[name] = [read_json(out / "recurrence_rows" / name / (case["id"] + ".json")) for case in selected]
        if any(row is None for row in reports[name]):
            raise ValueError("missing frozen recurrence result")
    for name in ("baseline_rerun", "governed", "governed_canary"):
        directory = out / "recurrence_rows" / name
        report = run_cached_replays(reviewers[name], selected, directory, skills[name],
                                   args.workers, args.retry_workers, args.retry_failed,
                                   secret=secret, log=log)
        for row in report["case_results"]:
            if row["execution_success"]:
                row["metrics"] = measure_recurrence(targets[row["id"]], [finding_from_dict(value) for value in row["findings"]])
                write_json(directory / (row["id"] + ".json"), row)
        reports[name] = report["case_results"]
    paired_ids = set.intersection(*({row["id"] for row in rows if row["execution_success"]} for rows in reports.values()))
    if not paired_ids:
        raise ValueError("recurrence has no jointly successful PRs")
    paired = {name: sorted((row for row in rows if row["id"] in paired_ids), key=lambda row: row["id"]) for name, rows in reports.items()}
    comparisons = {}
    pairs = [(reference, governed) for reference in ("baseline", "naive", "baseline_rerun")
             for governed in ("governed", "governed_canary")]
    pairs += [("governed", "governed_canary"), ("baseline", "naive"), ("baseline_rerun", "baseline")]
    for left, right in pairs:
        name = "%s_minus_%s" % (right, left)
        log("recurrence paired bootstrap: %s" % name)
        comparisons[name] = paired_recurrence_bootstrap(paired[left], paired[right], seed=original["seed"])
    result = {"notice": original["recurrence"]["notice"], "sampling_unit": "PR",
              "sample": {**sample, "paired_prs": len(paired_ids), "paired_case_ids": sorted(paired_ids)},
              "failures": {name: sum(not row["execution_success"] for row in rows) for name, rows in reports.items()},
              "metrics": {name: recurrence_rates(rows) for name, rows in paired.items()}, "comparisons": comparisons}
    write_json(out / "recurrence.json", result)
    return result


def refresh_reports(out: Path) -> None:
    state = read_json(out / "state.json")
    if not state or not state.get("finished_at") or not state.get("recurrence"):
        raise ValueError("v2 report-only requires a finished experiment and recurrence")
    check_copied_inputs(out)
    state["governance"] = read_governance(str(out), rot.TENANT)
    write_json(out / "governance.json", state["governance"])
    write_json(out / "state.json", state)
    write_summary(str(out), state)
    notes = build_notes(state, read_json(ROOT / "evaluation_results/admission_compress_check.json"),
                        read_json(ROOT / "evaluation_results/admission_conflict_check.json"))
    aa = state["comparisons_by_scope"]["executed_all"]["baseline_minus_baseline_rerun"]["f1"]
    diagnosis = read_json(out / "diagnostic.json")
    validation = state["governance"]["source_validation"]
    extra = ["", "## 第二轮评测的证据边界", "",
             "本轮只从头重放 governed 事件流。训练初审、反馈教训、naive 数据库、原基线和 naive 的 holdout/纠正率结果均按文件哈希复制；复制来的失败记录保留，未重新调用这两组。任务 7 的 baseline_rerun holdout 也直接复制。",
             "任务 7 只重审了 holdout，因此本轮另外测量 baseline_rerun 在训练来源 PR 上的纠正率，供同批问题的 A/A 对照。governed 正式与正式+试用均重新评测。",
             "快速转正用了来源 PR 做验证，所以纠正率评测与转正依据有重叠，只能表述为“对已反馈问题的纠正”。不能用这些重叠样本证明泛化收益。",
             "holdout 是本轮唯一干净的泛化评估。这里“干净”指未用于教训写入、来源验证或转正门禁；该集合已用于任务 7 的诊断，不是全新、从未查看过的独立测试集。validation 只用于发布门禁。",
             "任务 7 未确认 Skill 裁剪造成质量下降，因此没有修改预置 Skill 的裁剪规则；预算和九个预置 Skill 沿用原设定。",
             "在本轮共同成功 cohort 上，A/A（原基线减重复基线）的 F1 差为 %s。任务 7 的 A/A 为 %s。两者的样本排除可能不同，不能把其中一次波动当成模型长期波动的上界。" % (
                 interval(aa), interval(diagnosis["comparisons"]["baseline_minus_baseline_rerun"]["f1"])),
             "即使某项 PR bootstrap CI 完全位于有利方向，也只能限定到本批样本与比较对象；该 CI 没有覆盖多次模型运行的随机性。CI 包含 0 就不能说提升，需同时对照 A/A 的点差与区间。",
             "来源验证使用 k=3 的目标频率筛选和其他训练 PR 的命中保留检查，这是筛选条件，不是显著性检验。副作用检查只要求标注命中不减少，不能据此保证在其他 PR 上误报不会增加。", "",
             "## 本轮可写与不能写的简历表述", "",
             "- 实现仅重放治理组的可续跑评测，保留原比较组结果；在 %d 个 holdout PR 上评测正式/灰度通道，与两组旧结果及新基线做配对 bootstrap 比较。" % state["dataset"]["holdout"],
             "- 实现每轮最多 120 次审查的来源 PR 验证，实际发起 %d 次训练审查，%d 个不同条目通过筛选，%d 个通过原 validation 门禁后快速转正。验证通过数不能直接称为有效知识或泛化提升。" % (
                 validation["review_attempts"], validation["passed_unique_items"], validation["unique_fast_promoted_items"]),
             "- 可以报告本次拦截率、按 source 定义的污染率、实际转正/归档和训练验证账目，须保留分母、比较对象与归档原因。",
             "- 不能写“证明自进化提高泛化质量”“消除模型随机波动”“所有知识均因果有效”，也不能将中性/失败验证或容量归档计为有益知识与有害知识清除数。", ""]
    (out / "NOTES.md").write_text(notes + "\n".join(extra), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="evaluation_results/swr_rot_temporal")
    parser.add_argument("--diagnostic", default="evaluation_results/swr_rot_temporal_diag")
    parser.add_argument("--dataset", default="evaluation_data/swrbench/swr_bench_temporal.jsonl")
    parser.add_argument("--out", default="evaluation_results/swr_rot_temporal_v2")
    parser.add_argument("--worktrees", default="C:/dj_wt/t9")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retry-workers", type=int, default=8)
    parser.add_argument("--retry-failed", type=int, default=2)
    parser.add_argument("--source-validation-replay-limit", type=int, default=120)
    parser.add_argument("--source-validation-side-cases", type=int, default=2)
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    check_output(ROOT, out)
    if not 1 <= args.workers <= 16 or not 1 <= args.retry_workers <= 8:
        parser.error("workers must be 1..16, retry-workers 1..8")
    if args.source_validation_replay_limit < 0 or not 1 <= args.source_validation_side_cases <= 2:
        parser.error("invalid source verification configuration")
    out.mkdir(parents=True, exist_ok=True)
    if args.background:
        previous = read_json(out / "process.json")
        if previous and process_alive(previous["pid"]):
            raise RuntimeError("v2 process is already running; do not start a second writer")
        background(Path(__file__).resolve(), sys.argv[1:], out)
        return
    if args.report_only:
        refresh_reports(out)
        return
    original = copy_inputs(Path(args.source), Path(args.diagnostic), out)
    config = original["run_configuration"]
    if args.fake != bool(config["fake"]):
        raise ValueError("v2 must use the source model mode")
    cases = load_jsonl(args.dataset)
    diagnostic_config = read_json(out / "diagnostic_manifest.json")
    if (dataset_fingerprint(cases) != config["dataset_sha256"]
            or checksum(Path(args.dataset)) != diagnostic_config["dataset_sha256"]):
        raise ValueError("dataset differs from the frozen experiment or diagnostics")
    if diagnostic_config["fake"] != args.fake or diagnostic_config["model"] != original["model"]:
        raise ValueError("diagnostic baseline uses a different model")
    by_id = {case["id"]: case for case in cases}
    train = [by_id[key] for key in config["train_ids"]]
    gate_cases = [by_id[key] for key in config["gate_ids"]]
    holdout = [by_id[key] for key in config["holdout_ids"]]
    if not train or not gate_cases or not holdout:
        raise ValueError("v2 requires nonempty train, validation and holdout cohorts")
    if any(case["split"] != split for values, split in ((train,"train"),(gate_cases,"validation"),(holdout,"holdout")) for case in values):
        raise ValueError("source cohort split mismatch")
    repeated = read_json(out / "holdout_rows/baseline_rerun.json")
    if {row["id"] for row in repeated["case_results"]} != set(config["holdout_ids"]):
        raise ValueError("diagnostic baseline does not cover the identical holdout")
    settings = replace(Settings.from_env(), **config["settings"])
    connection = settings.resolved_llm()
    if not args.fake and connection["model"] != original["model"]:
        raise ValueError("v2 requires the original model")
    client = FakeSwrClient() if args.fake else JsonChatClient(str(connection["base_url"]), str(connection["api_key"]),
        str(connection["model"]), str(connection["provider"]), settings.timeout_seconds, dict(connection.get("headers") or {}))
    registry = SkillRegistry(str(ROOT / "skills")); registry.reload()
    disk = registry.agent_skills()
    if {skill.name: skill.content_sha256 for skill in disk} != read_json(out / "diagnostic_manifest.json")["skills_sha256"]:
        raise ValueError("shipped skills differ from task 7")
    configuration = {"source_configuration": config, "dataset_sha256": checksum(Path(args.dataset)),
                     "model": original["model"], "fake": args.fake, "source_validation_replay_limit": args.source_validation_replay_limit,
                     "source_validation_side_cases": args.source_validation_side_cases,
                     "code_sha256": {name: checksum(ROOT / name) for name in ("diffjury/assembler.py", "diffjury/source_validation.py", "diffjury/knowledge_service.py", "diffjury/lifecycle.py", "diffjury/agentic_core.py", "diffjury/release_gate.py", "diffjury/evaluation_replay.py", "scripts/run_swr_temporal_v2.py")}}
    state = read_json(out / "state.json")
    if state is None:
        if (out / "governed.db").exists():
            raise ValueError("governed database exists without v2 progress state")
        state = {"mode": "governed_only_v2", "next_event": 0, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "model": original["model"], "seed": original["seed"], "run_configuration": configuration,
                 "dataset": {"train": len(train), "gate": len(gate_cases), "holdout": len(holdout)},
                 "temporal_split": original.get("temporal_split"), "maintenance": [], "attribution": [],
                 "source_validation": [], "counterfactual": [], "holdout_by_scope": {}, "comparisons_by_scope": {}}
        write_json(out / "state.json", state)
    if state["run_configuration"] != configuration:
        raise ValueError("v2 resume configuration differs")
    if state.get("finished_at"):
        refresh_reports(out)
        return
    for name in ("promote_min_exposures", "evict_min_exposures", "probation_max_exposures"):
        setattr(args, name, config[name])
    args.source_validation = True
    SwrArm.snapshots = GitSnapshots(str(ROOT / "evaluation_data/swrbench/repos"), args.worktrees)
    governed = SwrArm("governed", str(out), client, settings, disk, gate_cases, args)
    governed.governor.release_gate.cache_dir = str(out / "gate_cache")
    events = read_json(out / "events.json")
    state["events_total"] = len(events)
    state["event_mix"] = original["event_mix"]
    originals = {case["id"]: read_json(out / "train_reviews" / (case["id"] + ".json")) for case in train}

    def save():
        state["governance"] = read_governance(str(out), rot.TENANT)
        state["admission_actions"] = {}
        for row in governed.repo.events(rot.TENANT, limit=1000000):
            if row["event"] == "admission":
                action = row["detail"]["action"]
                state["admission_actions"][action] = state["admission_actions"].get(action, 0) + 1
        write_json(out / "state.json", state)
        write_json(out / "governance.json", state["governance"])
        write_summary(str(out), state)

    def maintain(count):
        if any(row["event"] == count for row in state["maintenance"]):
            return
        log("source validation @%d" % count)
        verification = governed.governor.source_pr_validation_round(rot.TENANT, train, events, originals,
            governed._replay_factory, replay_limit=args.source_validation_replay_limit,
            side_cases=args.source_validation_side_cases, workers=args.workers, round_id=str(count))
        state["source_validation"] = [row for row in state["source_validation"] if row["event"] != count] + [{"event": count, **verification}]
        save(); log("source validation @%d: %s" % (count, json.dumps(verification)))
        cf = governed.governor.counterfactual_round(rot.TENANT, train, governed._replay_factory, classify,
            replay_limit=config["counterfactual_replay_limit"], max_cases_per_item=config["counterfactual_cases_per_item"],
            workers=args.workers, round_id=str(count))
        state["counterfactual"] = [row for row in state["counterfactual"] if row["event"] != count] + [{"event": count, **cf}]
        save(); log("counterfactual @%d: %s" % (count, json.dumps(cf)))
        if cf["failed"] > cf["pairs"] * rot.MAX_REPLAY_FAILURE_RATE:
            raise rot.ModelUnavailable("counterfactual transport failure; successful outcomes are cached")
        log("maintenance @%d: validation gate and release" % count)
        result = governed.governor.maintain(rot.TENANT, force_dream=True)
        dream_gate = ((result["dream"].get("release") or {}).get("gate") or {})
        failures = set(dream_gate.get("failures") or [])
        failures.update((result.get("promotion", {}).get("gate") or {}).get("failures") or [])
        if "execution_success_rate" in failures:
            save()
            raise rot.ModelUnavailable("validation gate transport failure; resume this checkpoint")
        state["maintenance"].append({"event": count, "summary": result})
        save(); log("maintenance @%d complete" % count)

    def stages(count):
        if config["review_every"] and count % config["review_every"] == 0 and not any(row["event"] == count for row in state["attribution"]):
            batch = random.Random(original["seed"] + count).sample(train, min(config["review_batch"], len(train)))
            result = governed.attribution_round(batch, count)
            state["attribution"].append({"event": count, **result})
            save(); log("attribution @%d: %s" % (count, json.dumps(result)))
        if config["maintain_every"] and count % config["maintain_every"] == 0:
            maintain(count)

    save()
    if state.get("pending_event"):
        stages(state["pending_event"])
        state.pop("pending_event", None); save()
    while state["next_event"] < len(events):
        event = events[state["next_event"]]
        governed.learn(event)
        state["next_event"] += 1
        state["pending_event"] = state["next_event"]
        save(); stages(state["next_event"])
        state.pop("pending_event", None); save()
    maintain(len(events))
    save()
    frozen = {"governed": governed.repo.list(rot.TENANT, statuses=("active",)),
              "governed_canary": governed.repo.list(rot.TENANT, statuses=("active", "probation"))}
    reviewers = {name: governed._replay_factory(StaticKnowledge(governed.assembler, items), "v2-%s" % name) for name, items in frozen.items()}
    skills = {name: disk + rot.skills_from_items(items) for name, items in frozen.items()}
    reviewers["baseline_rerun"] = make_reviewer(client, config["settings"], disk, SwrArm.snapshots, "v2-recurrence-baseline-rerun")
    skills["baseline_rerun"] = disk
    reports = {name: read_json(out / "holdout_rows" / filename) for name, filename in
               (("baseline", "baseline.json"), ("naive", "final-naive.json"), ("baseline_rerun", "baseline_rerun.json"))}
    for name in ("governed", "governed_canary"):
        reports[name] = run_cached_replays(reviewers[name], holdout, out / "holdout_rows" / name, skills[name],
            args.workers, args.retry_workers, args.retry_failed, secret=str(connection.get("api_key") or ""), log=log)
    paired = compare_holdout(reports)
    state["dataset"]["holdout_executed_all"] = len(paired["paired_case_ids"])
    state["holdout_paired_case_ids"] = paired["paired_case_ids"]
    state["comparisons_by_scope"]["executed_all"] = paired["comparisons"]
    state["holdout_by_scope"] = {"executed_all": paired["scores"], "all": {name: value["metrics"] for name, value in reports.items()}}
    state["holdout_failures"] = {name: {"after_retry": value["metrics"]["failure_rate"], "frozen": name in {"baseline", "naive", "baseline_rerun"}} for name, value in reports.items()}
    save()
    state["recurrence"] = run_recurrence(args, out, original, train, reviewers, skills, str(connection.get("api_key") or ""))
    check_copied_inputs(out)
    state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save(); refresh_reports(out)
    log("done: %s" % (out / "NOTES.md"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    try:
        main()
    except rot.ModelUnavailable as exc:
        log("stopped: %s" % exc)
        sys.exit(3)
