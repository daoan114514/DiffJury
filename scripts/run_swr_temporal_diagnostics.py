"""Diagnose model repeatability, shipped skill clipping and learned knowledge."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from diffjury.config import Settings
from diffjury.evaluation_harness import load_jsonl
from diffjury.evaluation_replay import (aggregate, assembler_from_config, background, check_output,
    checksum, make_reviewer, read_items, read_json, run_cached_replays, skill_audit, write_json)
from diffjury.evaluation_repos import GitSnapshots
from diffjury.evaluation_v2 import paired_bootstrap_comparison
from diffjury.llm import JsonChatClient
from diffjury.skills import SkillRegistry
from run_swr_rot_experiment import FakeSwrClient


def log(message: str) -> None:
    print("[%s] %s" % (time.strftime("%H:%M:%S"), message), flush=True)


def exposure_statistics(reports: dict) -> dict:
    return {name: {"exposed_prs": sum(bool(row.get("knowledge", {}).get("exposed")) for row in report["case_results"])
                                   if all("knowledge" in row for row in report["case_results"]) else None,
                   "cited_prs": sum(bool(row.get("cited_rules")) for row in report["case_results"])}
            for name,report in reports.items() if name in {"governed","governed_rerun"}}


def diagnosis(reports: dict, audit: dict, iterations: int = 2000) -> dict:
    ids = set.intersection(*({row["id"] for row in report["case_results"] if row.get("execution_success")}
                            for report in reports.values()))
    if not ids:
        raise ValueError("diagnosis requires jointly successful PRs")
    paired = {name: aggregate(sorted((row for row in report["case_results"] if row["id"] in ids), key=lambda row: row["id"]))
              for name, report in reports.items()}
    comparisons = {}
    for left, right in (("baseline_rerun", "baseline"), ("baseline_rerun", "governed_assembler_empty"),
                        ("governed_assembler_empty", "governed_rerun"), ("baseline_rerun", "governed_rerun"),
                        ("baseline", "governed"), ("governed", "governed_rerun")):
        comparisons["%s_minus_%s" % (right, left)] = paired_bootstrap_comparison(
            paired[left], paired[right], iterations=iterations,
            metrics=("f1", "precision", "recall", "fp", "average_instruction_tokens_per_call"))
    aa = comparisons["baseline_minus_baseline_rerun"]["f1"]
    empty = comparisons["governed_assembler_empty_minus_baseline_rerun"]["f1"]
    learned = comparisons["governed_rerun_minus_governed_assembler_empty"]["f1"]
    clipping = bool(audit["catalog_drop_frequencies"]) and empty["ci95"][1] < 0
    worker_costs = {}
    for name, report in paired.items():
        entries = [entry for row in report["case_results"] for entry in row.get("skill_instructions", [])
                   if entry["role"] not in {"lead", "critic"}]
        worker_costs[name] = {"recorded_worker_calls": len(entries),
                              "body_tokens": sum(entry["skill_body_tokens"] for entry in entries),
                              "mean_body_tokens_per_worker_call": round(sum(entry["skill_body_tokens"] for entry in entries)/len(entries), 2) if entries else None,
                              "skill_call_frequencies": dict(Counter(skill for entry in entries for skill in entry["skills"]))}
    knowledge_audit = exposure_statistics(paired)
    return {"jointly_successful_prs": len(ids), "paired_case_ids": sorted(ids),
            "metrics": {name: report["metrics"] for name, report in paired.items()},
            "all_pr_metrics": {name: report["metrics"] for name, report in reports.items()},
            "comparisons": comparisons, "worker_skills": worker_costs, "knowledge_audit": knowledge_audit,
            "decision": {"clipping_impact_confirmed": clipping,
                         "knowledge_harm_confirmed": learned["ci95"][1] < 0 and bool(knowledge_audit["governed_rerun"]["exposed_prs"]),
                         "aa_f1_delta": aa["delta"], "aa_f1_ci95": aa["ci95"],
                         "criterion": "catalog omission exists and empty-assembler minus fresh-baseline F1 CI is wholly below zero"}}


def write_report(out: Path, result: dict, audit: dict, fake: bool) -> None:
    decision = result["decision"]
    lines = ["# 同仓库时间外推诊断", ""]
    if fake:
        lines += ["本文件为假模型流程检查，不能用作真实诊断结论。", ""]
    lines += ["## 方法与样本", "", "沿用原实验 holdout PR、head 快照、diff、标注、模型和预算；仅增加三组回放，原目录和数据库只读。",
              "主要配对比较使用五组共同成功的 %d 个 PR。原实验 baseline/governed 的正文日志未保存，正文 token 对比使用本次三组逐调用记录，不填补旧日志。" % result["jointly_successful_prs"],
              "配对 bootstrap 按 PR 重采样 2,000 次，seed=20260819。A/A 是 baseline 减 baseline_rerun，其他比较名 A_minus_B 表示 A 减 B。", "",
              "## 共同成功样本的质量", "", "| 组 | F1 | 精确率 | 召回率 | 命中 | 误报 | 漏报 | 指令 token/调用 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name,m in result["metrics"].items():
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (name,m["f1"],m["precision"],m["recall"],m["tp"],m["fp"],m["fn"],m["average_instruction_tokens_per_call"]))
    lines += ["", "### 完整样本与执行失败", "", "下表保留全量 PR 口径，执行失败可能被记作未命中，不能单独用来归因质量差异；上表与以下 CI 均使用共同成功样本。", "",
              "| 组 | PR 数 | F1 | 精确率 | 召回率 | 误报 | 执行失败 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name,m in result["all_pr_metrics"].items():
        lines.append("| %s | %d | %s | %s | %s | %d | %d |" % (name,m["cases"],m["f1"],m["precision"],m["recall"],m["fp"],m["cases"]-m["execution_successes"]))
    lines += ["", "完整结果及共同成功 PR 清单见 [diag.json](diag.json)。", "", "## 配对差值与 95% CI", "", "| 比较 | 指标 | 差值 | 95% CI |", "|---|---|---:|---|"]
    for pair,metrics in result["comparisons"].items():
        for metric,row in metrics.items():
            lines.append("| %s | %s | %+.4f | [%s, %s] |" % (pair,metric,row["delta"],*row["ci95"]))
    lines += ["", "## 预置 Skill 裁剪与正文", "", "以下频率按 PR 计算。逐 PR 被裁名单与正文可用量见 [skill_audit.json](skill_audit.json)。离线探针假设全选目录，仅说明可用量；实际 worker 正文另按调用记录统计。", "",
              "| 被裁预置 Skill | PR 次数 |", "|---|---:|"]
    for name,count in audit["catalog_drop_frequencies"].items():
        lines.append("| %s | %d |" % (name,count))
    lines += ["", "| 组 | 已记录 worker 调用 | 正文 token 总数 | 正文 token/worker 调用 |", "|---|---:|---:|---:|"]
    for name,row in result["worker_skills"].items():
        if row["recorded_worker_calls"]:
            lines.append("| %s | %d | %d | %s |" % (name,row["recorded_worker_calls"],row["body_tokens"],row["mean_body_tokens_per_worker_call"]))
    lines += ["", "正文 token 用项目统一估算，不等于完整 system prompt 或网关 input token。worker 选择还受模型委派波动影响，不能把观察到的所有正文差值都归因于裁剪。", "",
              "## 诊断结论", "", "A/A 的 F1 差值为 %+.4f，95%% CI [%s, %s]。这是一对重复运行的观察值，不是模型长期波动的上界。" % (decision["aa_f1_delta"],*decision["aa_f1_ci95"])]
    if decision["clipping_impact_confirmed"]:
        lines += ["空知识装配器相对新基线的 F1 CI 完全低于 0，且离线确认预置 Skill 被裁。该对照支持装配裁剪/预算对质量有不利影响，按交接要求修复预置 Skill 的预算豁免。"]
    else:
        lines += ["空知识装配器相对新基线的 F1 CI 未完全低于 0；当前诊断未确认裁剪造成质量下降，暂不修改预置 Skill 的裁剪规则。目录裁剪本身已被离线检查，但其质量影响与模型波动不能据此等同。"]
    if decision["knowledge_harm_confirmed"]:
        lines += ["带正式知识相对空知识装配器的 F1 CI 完全低于 0，本次对照支持正式知识存在不利影响。"]
    elif result.get("knowledge_audit",{}).get("governed_rerun",{}).get("exposed_prs") == 0:
        lines += ["正式条目在本次共同成功样本中没有实际注入，不能把两组质量差异归因于学习知识。该结论来自曝光记录，不能用零引用替代。"]
    else:
        lines += ["带正式知识相对空知识装配器的 F1 CI 未完全低于 0，不能确认原下降由这两条知识导致。"]
    rerun = result["comparisons"]["governed_rerun_minus_baseline_rerun"]["f1"]
    original = result["comparisons"]["governed_minus_baseline"]["f1"]
    lines += ["原 governed 相对原基线的 F1 差为 %+.4f，重跑两组的差为 %+.4f [%s, %s]；A/A 点差的绝对幅度为 %.4f。" % (
        original["delta"],rerun["delta"],*rerun["ci95"],abs(decision["aa_f1_delta"]))]
    if rerun["ci95"][1] < 0:
        lines += ["governed 相对基线的下降在本次重跑中复现，CI 完全低于 0。"]
    elif rerun["delta"] >= 0:
        lines += ["本次重跑的点估计未复现原下降；这仍不足以把原下降全部归为随机波动。"]
    else:
        lines += ["本次重跑仍有下降点估计，但 CI 包含 0，不能确认下降稳定复现。"]
    lines += ["正式知识曝光与引用统计：%s。原实验未保存曝光日志，其曝光为 0 的计数不能作证据，引用计数仍可复核。" % json.dumps(result.get("knowledge_audit",{}),ensure_ascii=False)]
    lines += ["单次 A/A 与每组一次独立回放不足以精确分解随机波动、裁剪和知识的贡献。CI 包含 0 时不称提升；PR bootstrap 未覆盖多次模型运行的波动。结果只适用于本次四仓库配置。", ""]
    (out/"diag.md").write_text("\n".join(lines),encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",default="evaluation_results/swr_rot_temporal")
    parser.add_argument("--dataset",default="evaluation_data/swrbench/swr_bench_temporal.jsonl")
    parser.add_argument("--out",default="evaluation_results/swr_rot_temporal_diag")
    parser.add_argument("--worktrees",default="C:/dj_wt/t7")
    parser.add_argument("--workers",type=int,default=16)
    parser.add_argument("--retry-workers",type=int,default=8)
    parser.add_argument("--retry-failed",type=int,default=2)
    parser.add_argument("--holdout-limit",type=int,default=0)
    parser.add_argument("--fake",action="store_true")
    parser.add_argument("--background",action="store_true")
    parser.add_argument("--report-only",action="store_true")
    args = parser.parse_args()
    out,source = Path(args.out),Path(args.source)
    check_output(ROOT,out)
    if not 1<=args.workers<=16 or not 1<=args.retry_workers<=8:
        parser.error("workers must be 1..16; retry-workers must be 1..8")
    out.mkdir(parents=True,exist_ok=True)
    if args.background:
        background(Path(__file__).resolve(),sys.argv[1:],out)
        return
    if args.report_only:
        result = read_json(out/"diag.json")
        paired_ids = set(result["paired_case_ids"])
        result["knowledge_audit"] = exposure_statistics({name:{"case_results":[row for row in read_json(path)["case_results"] if row["id"] in paired_ids]}
            for name,path in (("governed",source/"holdout_rows/final-governed.json"),("governed_rerun",out/"holdout_rows/governed_rerun.json"))})
        result["decision"]["knowledge_harm_confirmed"] = (
            result["comparisons"]["governed_rerun_minus_governed_assembler_empty"]["f1"]["ci95"][1] < 0
            and bool(result["knowledge_audit"]["governed_rerun"]["exposed_prs"]))
        write_json(out/"diag.json",result)
        write_report(out,result,read_json(out/"skill_audit.json"),read_json(out/"state.json")["fake"])
        return
    previous = read_json(source/"state.json")
    if not previous or not previous.get("finished_at"):
        raise ValueError("diagnostics require a finished source experiment")
    config = previous["run_configuration"]["settings"]
    ids = set(previous["run_configuration"]["holdout_ids"])
    cases = sorted([case for case in load_jsonl(args.dataset) if case["id"] in ids],key=lambda case:case["id"])
    if args.holdout_limit:
        cases = cases[:args.holdout_limit]
    assert cases and all(case["split"]=="holdout" for case in cases)
    assert len(cases)==(min(len(ids),args.holdout_limit) if args.holdout_limit else len(ids))
    registry = SkillRegistry(str(ROOT/"skills")); registry.reload()
    skills = registry.agent_skills()
    assembler = assembler_from_config(config)
    manifest = {"fake":args.fake,"model": "fake" if args.fake else previous["model"],
                "dataset_sha256":checksum(Path(args.dataset)),"case_ids":[case["id"] for case in cases],
                "source_governed_sha256":checksum(source/"governed.db"),
                "source_baseline_sha256":checksum(source/"holdout_rows/baseline.json"),
                "source_holdout_governed_sha256":checksum(source/"holdout_rows/final-governed.json"),
                "assembler_sha256":checksum(ROOT/"diffjury/assembler.py"),"settings":config,
                "skills_sha256":{skill.name:skill.content_sha256 for skill in skills}}
    saved = read_json(out/"manifest.json")
    if saved is not None and saved != manifest:
        raise ValueError("diagnostic resume configuration differs")
    write_json(out/"manifest.json",manifest)
    state = read_json(out/"state.json",{"started_at":time.strftime("%Y-%m-%dT%H:%M:%S"),"fake":args.fake,"completed_arms":[]})
    write_json(out/"state.json",state)
    audit = skill_audit(cases,skills,assembler)
    write_json(out/"skill_audit.json",audit)
    settings = Settings.from_env()
    connection = settings.resolved_llm()
    if not args.fake and connection["model"] != previous["model"]:
        raise ValueError("diagnostics require the original model")
    client = FakeSwrClient() if args.fake else JsonChatClient(str(connection["base_url"]),str(connection["api_key"]),str(connection["model"]),str(connection["provider"]),settings.timeout_seconds,dict(connection.get("headers") or {}))
    snapshots = GitSnapshots(str(ROOT/"evaluation_data/swrbench/repos"),args.worktrees)
    items = read_items(source/"governed.db")
    reports = {name:aggregate([row for row in read_json(source/"holdout_rows"/filename)["case_results"] if row["id"] in {case["id"] for case in cases}])
               for name,filename in (("baseline","baseline.json"),("governed","final-governed.json"))}
    for name,pool in (("baseline_rerun",None),("governed_assembler_empty",[]),("governed_rerun",items)):
        reviewer = make_reviewer(client,config,skills,snapshots,name,None if pool is None else assembler,pool or [])
        reports[name] = run_cached_replays(reviewer,cases,out/"holdout_rows"/name,skills,args.workers,args.retry_workers,args.retry_failed,secret=str(connection.get("api_key") or ""),log=log)
        if name not in state["completed_arms"]:
            state["completed_arms"].append(name)
        write_json(out/"state.json",state)
    result = diagnosis(reports,audit)
    write_json(out/"diag.json",result)
    write_report(out,result,audit,args.fake)
    state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write_json(out/"state.json",state)
    log("done: %s" % (out/"diag.md"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
