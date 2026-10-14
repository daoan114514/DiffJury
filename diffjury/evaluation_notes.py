"""Chinese report text generated from measured experiment artifacts."""
from typing import Any, Dict, List, Optional

from .evaluation_governance import DECISIONS
from .evaluation_recurrence import RECURRENCE_METRICS, RECURRENCE_NOTICE


SOURCE_NAMES = {"missed": "漏报反馈", "false_positive": "误报反馈", "noise": "噪声", "contradiction": "矛盾"}
ARM_NAMES = {"baseline": "无学习知识基线", "baseline_rerun": "无学习知识重复基线", "naive": "naive", "governed": "governed 正式", "governed_canary": "governed 正式+试用"}
METRIC_NAMES = {"f1": "F1", "precision": "精确率", "recall": "召回率", "high_risk_recall": "高风险召回率",
                "clean_accuracy": "无缺陷 PR 正确率", "average_instruction_tokens_per_call": "指令 token/调用",
                "average_total_tokens_per_pr": "总 token/PR", "false_positive_recurrence": "误报复发率",
                "missed_recovery": "漏报找回率", "correct_retention": "正确发现保留率"}


def percentage(value: Optional[float]) -> str:
    return "不适用" if value is None else "%.2f%%" % (value * 100)


def interval(value: dict) -> str:
    if value.get("delta") is None or value.get("ci95") is None:
        return "不适用（无目标标注）"
    return "%+.6f [%+.6f, %+.6f]" % (value["delta"], value["ci95"][0], value["ci95"][1])


def governance_markdown(governance: dict) -> List[str]:
    lines = ["## 知识治理统计", "", "写入决策按候选条目计数，共 %d 条；CONFLICT 单独展示，其持久化动作仍为 REJECT。" % governance["admission_candidates"], "",
             "| 来源 | " + " | ".join(DECISIONS) + " |", "|---|" + "---:|" * len(DECISIONS)]
    for source, counts in sorted(governance["admission_by_source"].items()):
        lines.append("| %s | %s |" % (SOURCE_NAMES.get(source, source), " | ".join(str(counts.get(key, 0)) for key in DECISIONS)))
    lines += ["", "### 噪声与矛盾拦截", "", "| 来源 | 拒绝候选/候选总数 | 候选拦截率 | 全部候选均被拒的事件/有候选事件 | 事件拦截率 |",
              "|---|---:|---:|---:|---:|"]
    for source, row in governance["interception"].items():
        lines.append("| %s | %d/%d | %s | %d/%d | %s |" % (
            SOURCE_NAMES.get(source, source), row["rejected_candidates"], row["candidate_count"], percentage(row["candidate_rate"]),
            row["blocked_events"], row["events_with_candidates"], percentage(row["event_rate"])))
    lines += ["", "### 最终在用学习知识的污染率", "",
              "污染定义为 source 属于 noise 或 contradiction。分母包含在用规则和学习技能，排除预置 disk 技能；正式通道只计 active，灰度通道计 active+probation。分母为 0 时不报告污染率。", "",
              "| 通道 | 污染条目/在用条目 | 条目污染率 | 污染 token/在用 token | token 污染率 |", "|---|---:|---:|---:|---:|"]
    for name, row in governance["contamination"].items():
        lines.append("| %s | %d/%d | %s | %d/%d | %s |" % (
            ARM_NAMES[name], row["polluted_items"], row["items"], percentage(row["item_rate"]),
            row["polluted_tokens"], row["tokens"], percentage(row["token_rate"])))
    lifecycle, cf = governance["lifecycle"], governance["counterfactual"]
    categories = lifecycle.get("archived_categories", {})
    lines += ["", "### 转正、归档与反事实归因", "",
              "- 转正动作 %d 次，涉及 %d 个不同条目；归档动作 %d 次，涉及 %d 个不同条目。归档包含淘汰和归并。" % (
                  lifecycle["promotion_events"], lifecycle["unique_promoted_items"], lifecycle["eviction_events"], lifecycle["unique_evicted_items"]),
              "- 当前归档 %d 个条目，其中容量淘汰 %d、harmful 淘汰 %d、归并 %d；完整原因计数见 governance.json。容量淘汰不能表述为识别并清除有害知识。" % (
                  lifecycle["currently_archived"], categories.get("capacity", 0), categories.get("harmful", 0), categories.get("merged", 0)),
              "- 反事实比较尝试 %d 对，发起审查 %d 次（包含失败和未注入试用条目的回放），完整且实际注入后记账 %d 对；有益 %d、有害 %d、中性 %d、失败 %d、未实际注入 %d。" % (
                  cf["pair_attempts"], cf["individual_replays"], cf["applied_pairs"], cf["helpful"], cf["harmful"], cf["neutral"], cf["failed"], cf["not_exposed"]),
              "- 归因涉及 %d 个条目、%d 个不同训练 PR；训练集隔离检查：%s。" % (
                  cf["unique_items"], cf["unique_prs"], "通过" if cf["train_only"] else "失败"), ""]
    validation = governance.get("source_validation") or {}
    if validation.get("result_events") or validation.get("entry_attempts"):
        lines += ["### 来源 PR 验证与快速转正", "",
                  "- 验证尝试 %d 个条目批次，发起 %d 次训练审查，完成记账 %d 个条目、%d 对比较；有益 %d、有害 %d、中性 %d。" % (
                      validation["entry_attempts"], validation["review_attempts"], validation["applied_entries"], validation["completed_pairs"],
                      validation["helpful"], validation["harmful"], validation["neutral"]),
                  "- 通过来源筛选 %d 个不同条目；通过原 validation 门禁后快速转正 %d 个不同条目（%d 次动作）。结果状态：%s。" % (
                      validation["passed_unique_items"], validation["unique_fast_promoted_items"], validation["fast_promotion_events"], str(validation["statuses"])),
                  "- 失败 %d、未实际注入 %d、跳过 %d、尚无结果的预算预留 %d；来源 PR %d 个，副作用检查 PR %d 个；仅训练集：%s。" % (
                      validation["failed"], validation["not_exposed"], validation["skipped"], validation["unfinished_reservations"],
                      validation["unique_source_prs"], validation["unique_side_prs"], "通过" if validation["train_only"] else "无实际审查" if validation["train_only"] is None else "失败"),
                  "- 已归档的漏报/误报条目按当前有效验证状态计数：%s。" % str(validation["archived_validation_status"]),
                  "- k=3 是筛选条件，不是显著性检验；来源 PR 与纠正率评测重叠，不能将验证通过数表述为泛化收益。", "",
                  "| 维护轮次 | 审查上限 | 实际发起审查 | 已消耗预算（含未完成预留） |", "|---|---:|---:|---:|"]
        for round_id, row in validation["rounds"].items():
            lines.append("| %s | %s | %d | %d |" % (round_id, row["replay_limit"], row["review_attempts"], row["budget_consumed"]))
        lines.append("")
    return lines


def build_notes(state: Dict[str, Any], compression: Optional[dict] = None,
                conflict: Optional[dict] = None) -> str:
    if not state.get("finished_at") or not state.get("recurrence") or not state.get("governance"):
        raise ValueError("final notes require a finished experiment, recurrence and governance statistics")
    scope = "executed_all" if "executed_all" in state["holdout_by_scope"] else "all"
    scores = state["holdout_by_scope"][scope]
    comparisons = state["comparisons_by_scope"][scope]
    primary = comparisons["governed_minus_baseline"]["f1"]
    dataset, governance, recurrence = state["dataset"], state["governance"], state["recurrence"]
    lines = ["# 同仓库时间外推评测记录", "", "## 结论", "",
             "实验于 %s 完成。报告由已保存的结果和只读治理账目生成。" % state["finished_at"],
             "本实验比较反馈写入后的代码审查质量、指令成本和知识污染，证据范围为所选四个仓库、配置模型 %s 和本次事件流。" % state["model"],
             "主要质量结论使用 %d 组都成功执行的 %d 个 holdout PR；完整 holdout 共 %d 个 PR，失败与补跑情况保留在 summary.md。" % (
                 len(scores), dataset.get("holdout_executed_all", dataset["holdout"]), dataset["holdout"])]
    if state.get("model") == "fake" or state.get("run_configuration", {}).get("fake"):
        lines.insert(2, "本文件为假模型流程验证，不能作为实测性能证据。\n")
    ci = primary["ci95"]
    if ci[0] > 0:
        lines.append("在这批时间外推 PR 上，governed 正式通道相对基线的 F1 增量为 %s，95%% CI 完全高于 0。可称本次设定下 F1 提升。" % interval(primary))
    elif ci[1] < 0:
        lines.append("governed 正式通道相对基线的 F1 差值为 %s，95%% CI 完全低于 0；在本次设定下 F1 下降。" % interval(primary))
    else:
        lines.append("governed 正式通道相对基线的 F1 增量为 %s，95%% CI 包含 0，不能称F1提升。" % interval(primary))
    naive_tokens = scores["naive"]["average_instruction_tokens_per_call"]
    governed_tokens = scores["governed"]["average_instruction_tokens_per_call"]
    token_delta = comparisons["governed_minus_naive"]["average_instruction_tokens_per_call"]
    reduction = 1 - governed_tokens / naive_tokens if naive_tokens else None
    cost_direction = "下降" if reduction is not None and reduction >= 0 else "上升"
    lines += ["", "governed 相比 naive 的每次调用指令 token 为 %s 对 %s，%s点估计 %s；绝对差及配对 95%% CI 为 %s。" % (
        governed_tokens, naive_tokens, cost_direction, percentage(abs(reduction) if reduction is not None else None), interval(token_delta))]
    naive_total = scores["naive"]["average_total_tokens_per_pr"]
    total_reduction = 1 - scores["governed"]["average_total_tokens_per_pr"] / naive_total if naive_total else None
    total_direction = "下降" if total_reduction is not None and total_reduction >= 0 else "上升"
    lines += ["每个 PR 的总 token 相比 naive %s点估计 %s；绝对差及配对 95%% CI 为 %s。这里的成本仅指 token 用量。" % (
        total_direction, percentage(abs(total_reduction) if total_reduction is not None else None), interval(comparisons["governed_minus_naive"]["average_total_tokens_per_pr"])), ""]
    blocked = governance["interception"]
    formal = governance["contamination"]["governed"]
    canary = governance["contamination"]["governed_canary"]
    lines += ["噪声拒绝 %d/%d，矛盾拒绝 %d/%d。正式通道污染条目 %d/%d，正式+试用通道污染条目 %d/%d；正式通道污染率为 0 不能说明所有矛盾都被拦截，也不能说明审查质量改善。" % (
        blocked["noise"]["rejected_candidates"], blocked["noise"]["candidate_count"],
        blocked["contradiction"]["rejected_candidates"], blocked["contradiction"]["candidate_count"],
        formal["polluted_items"], formal["items"], canary["polluted_items"], canary["items"]),
        "", "## 评测口径与数据", "",
        "四仓库各自按 created_at 排序，以 floor(0.6N)、floor(0.7N) 为边界。训练 %d、validation 门禁 %d、holdout %d；PR 不跨划分，时间边界见 summary.md。" % (
            dataset["train"], dataset["gate"], dataset["holdout"]),
        "教训按来源仓库设置 scope，装配只使用同仓库或 scope='*' 的条目。训练初审不带学习知识；同一快照、diff 和标注的缓存可复用。" + (
            "本轮只按时间顺序重放 governed 事件流，naive 结果使用冻结副本。" if state.get("mode") == "governed_only_v2" else "事件按时间顺序送入两组。"),
        "反事实归因只选训练 PR，每条每次调用最多 3 个 PR，每次调用最多 40 次审查；维护检查点续跑可能再次调用，因此同一维护轮次的累计审查可能超过 40，须以数据库账目为准。正式知识与正式知识加单条试用知识各回放一次；无退化且命中增加或误报减少记有益，任一项退化记有害，均不变记中性。只有完整且实际注入的比较才记账。转正、淘汰和发布门禁沿用现有规则。",
        "指令 token 使用项目统一估算，总 token 来自网关 usage。holdout 与纠正率均按同一 PR 配对重采样 2,000 次，报告 95%% CI；holdout bootstrap seed=20260819，纠正率 seed=%s。" % state["seed"],
        "数据与分析来源：[state.json](state.json)、[governance.json](governance.json)、[recurrence.json](recurrence.json) 和 [holdout_rows](holdout_rows/)。", "",
        "## 时间外推结果与配对 CI", "", "| 通道 | F1 | 精确率 | 召回率 | 指令 token/调用 | 总 token/PR |", "|---|---:|---:|---:|---:|---:|"]
    for name, row in scores.items():
        lines.append("| %s | %s | %s | %s | %s | %s |" % (ARM_NAMES[name], row["f1"], row["precision"], row["recall"],
                                                               row["average_instruction_tokens_per_call"], row["average_total_tokens_per_pr"]))
    lines += ["", "比较名 A_minus_B 表示 A 减 B；指令和总 token 的负差表示成本下降。", "",
              "| 比较 | 指标 | 差值 [95% CI] |", "|---|---|---|"]
    for pair, values in comparisons.items():
        for metric in ("f1", "precision", "recall", "average_instruction_tokens_per_call", "average_total_tokens_per_pr"):
            lines.append("| %s | %s | %s |" % (pair, METRIC_NAMES[metric], interval(values[metric])))
    lines += [""] + governance_markdown(governance)
    lines += ["## 已反馈问题的纠正", "", RECURRENCE_NOTICE, "",
              "抽样 %d 个不同来源 PR，%d 个通道共同成功执行 %d 个；按反馈类型抽样数量：%s。" % (
                  recurrence["sample"]["selected_prs"], len(recurrence["metrics"]), recurrence["sample"]["paired_prs"], str(recurrence["sample"]["by_event_type"])), "",
              "| 通道 | 误报复发率 | 漏报找回率 | 正确发现保留率 |", "|---|---:|---:|---:|"]
    for name, values in recurrence["metrics"].items():
        cells = ["%s (%d/%d)" % (percentage(values[metric]["rate"]), values[metric]["numerator"], values[metric]["denominator"])
                 for metric in RECURRENCE_METRICS]
        lines.append("| %s | %s |" % (ARM_NAMES[name], " | ".join(cells)))
    lines += ["", "误报复发率越低越好，其余两项越高越好。", "", "| 比较 | 指标 | 差值 [95% CI] |", "|---|---|---|"]
    for pair, values in recurrence["comparisons"].items():
        for metric in RECURRENCE_METRICS:
            lines.append("| %s | %s | %s |" % (pair, METRIC_NAMES[metric], interval(values[metric])))
    correction = recurrence["comparisons"].get("governed_minus_baseline", {})
    if correction:
        lines += ["", "governed 正式通道相对基线：误报复发率差 %s，漏报找回率差 %s。若对应 CI 包含 0，不能称该项纠正能力提升。正确发现保留率的目标分母仅 %d，不足以说明不会过度压制。" % (
            interval(correction["false_positive_recurrence"]), interval(correction["missed_recovery"]),
            recurrence["metrics"]["governed"]["correct_retention"]["denominator"])]
    lines += ["", "## 能说和不能说的话", "",
              "能报告本次样本的分数、成本、候选拦截率、最终污染率、转正归档数量和归因账目。某一项配对 CI 完全位于有利方向时，可以限定比较对象和样本范围，称该项改善；CI 包含 0 时不能称该项提升。",
              "纠正率只说明同一批已反馈 PR 的再审结果，不能当作泛化能力。时间外推结果也只覆盖这四个仓库和本次模型设定，不能推出跨仓库或其他模型的表现。",
              "误报反馈以未匹配功能标注、且未命中 neutral 标注的发现为代理；原基准标注未必覆盖所有真实问题。匹配使用同文件、行距 ±2 和规范化类别，未额外进行人工语义复核。公开历史 PR 也不能保证未被模型预训练见过；2024 子集仅作敏感性检查。",
              "bootstrap CI 描述本批 PR 重采样的不确定性，未覆盖多次训练、不同事件流或模型调用波动。污染率按条目 source 统计；UPVOTE 只补证据，source 不随之改变，因此须同时阅读写入决策表。", "",
              "## " + ("示例数字（假模型，不作性能证据）" if state.get("model") == "fake" else "可用于简历的事实数字"), "",
              "- 搭建四仓库共 %d 个真实 PR 的时间外推评测：训练 %d、发布门禁 %d、holdout %d；输出配对 bootstrap 95%% CI 和 %d 通道纠正率。" % (
                  dataset["train"] + dataset["gate"] + dataset["holdout"],
                  dataset["train"], dataset["gate"], dataset["holdout"], len(recurrence["metrics"]))]
    if compression:
        lines.append("- 修复短规则压缩，对 %d 条历史超长候选做真实模型只读回放，%d 条通过，比例 %s。" % (
            compression["candidates"], compression["fits"], percentage(compression["fit_rate"])))
    if conflict:
        lines.append("- 在 %d 条受控矛盾样本中识别 %d 条 CONFLICT，拒绝 %d 条；该数字来自与原教训配对的离线检查。" % (
            conflict["candidates"], conflict["conflicts"], conflict["rejected"]))
    cf, lifecycle = governance["counterfactual"], governance["lifecycle"]
    lines.append("- 实现定向反事实归因，发起 %d 对训练比较、%d 次审查，其中 %d 对完整且实际注入后记账；转正 %d 个不同条目，执行 %d 次归档动作。归档原因须按治理统计说明。" % (
        cf["pair_attempts"], cf["individual_replays"], cf["applied_pairs"], lifecycle["unique_promoted_items"], lifecycle["eviction_events"]))
    if reduction is not None and reduction > 0 and token_delta["ci95"][1] < 0:
        lines.append("- 在共同成功执行的 holdout 样本上，governed 比 naive 的单次调用指令 token 下降 %s；绝对差 95%% CI 为 [%s, %s]。" % (
            percentage(reduction), token_delta["ci95"][0], token_delta["ci95"][1]))
    return "\n".join(lines) + "\n"
