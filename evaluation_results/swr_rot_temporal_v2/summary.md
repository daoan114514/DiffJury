# SWR-Bench 知识治理评测

- 实验完成时间：2026-10-19T15:12:06
- 配置模型：gpt-6.1-sol；数据集：{"train": 434, "gate": 73, "holdout": 219, "holdout_executed_all": 217}
- 学习事件：243 {"missed": 89, "false_positive": 81, "noise": 49, "contradiction": 24}
- governed 写入动作：{"ADD": 154, "REJECT": 84, "UPVOTE": 6}
- holdout 执行失败率（首轮、补跑后）：{"baseline": {"after_retry": 0.0, "frozen": true}, "naive": {"after_retry": 0.0091, "frozen": true}, "baseline_rerun": {"after_retry": 0.0046, "frozen": true}, "governed": {"after_retry": 0.0, "frozen": false}, "governed_canary": {"after_retry": 0.0, "frozen": false}}

主要质量结论使用 executed_all（所有比较组共同成功的 PR）；all 保留完整样本结果，recent_2024 为敏感性检查。比较名 A_minus_B 表示 A 减 B。

## Holdout (executed_all)

| arm | f1 | precision | recall | high_risk_recall | clean_accuracy | instr. tokens/call | tokens/PR |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.3455 | 0.3393 | 0.3519 | 0.375 | 0.8623 | 1056.12 | 24180.62 |
| naive | 0.3119 | 0.3091 | 0.3148 | 0.375 | 0.8383 | 1840.59 | 27864.18 |
| baseline_rerun | 0.3091 | 0.3036 | 0.3148 | 0.375 | 0.8563 | 1050.14 | 23770.71 |
| governed | 0.2931 | 0.2742 | 0.3148 | 0.375 | 0.8204 | 871.75 | 22818.1 |
| governed_canary | 0.2909 | 0.2857 | 0.2963 | 0.375 | 0.8383 | 994.52 | 22886.65 |

- governed_minus_baseline: f1 -0.0524 [-0.1334, 0.0287]; precision -0.0651 [-0.1522, 0.0121]; recall -0.0371 [-0.1316, 0.0477]; high_risk_recall +0.0000 [-0.12, 0.12]; clean_accuracy -0.0419 [-0.0839, 0.0]; average_instruction_tokens_per_call -184.3700 [-238.31, -133.79]; average_total_tokens_per_pr -1362.5200 [-2731.58, -205.96]
- governed_canary_minus_baseline: f1 -0.0546 [-0.1409, 0.029]; precision -0.0536 [-0.1397, 0.0265]; recall -0.0556 [-0.1591, 0.0408]; high_risk_recall +0.0000 [-0.12, 0.12]; clean_accuracy -0.0240 [-0.0685, 0.0187]; average_instruction_tokens_per_call -61.6000 [-121.48, -0.47]; average_total_tokens_per_pr -1293.9700 [-2724.7, -36.62]
- governed_minus_naive: f1 -0.0188 [-0.0938, 0.0601]; precision -0.0349 [-0.1184, 0.0378]; recall +0.0000 [-0.0909, 0.0847]; high_risk_recall +0.0000 [-0.1111, 0.1177]; clean_accuracy -0.0179 [-0.0585, 0.0245]; average_instruction_tokens_per_call -968.8400 [-1051.63, -885.93]; average_total_tokens_per_pr -5046.0800 [-6799.16, -3759.99]
- governed_canary_minus_naive: f1 -0.0210 [-0.1061, 0.0707]; precision -0.0234 [-0.1089, 0.0602]; recall -0.0185 [-0.1154, 0.0755]; high_risk_recall +0.0000 [-0.1111, 0.1177]; clean_accuracy +0.0000 [-0.0414, 0.0412]; average_instruction_tokens_per_call -846.0700 [-920.01, -770.51]; average_total_tokens_per_pr -4977.5300 [-6891.4, -3569.87]
- governed_minus_baseline_rerun: f1 -0.0160 [-0.0904, 0.066]; precision -0.0294 [-0.1052, 0.0433]; recall +0.0000 [-0.0943, 0.0877]; high_risk_recall +0.0000 [-0.125, 0.125]; clean_accuracy -0.0359 [-0.0783, 0.006]; average_instruction_tokens_per_call -178.3900 [-236.11, -122.85]; average_total_tokens_per_pr -952.6100 [-2152.98, 157.81]
- governed_canary_minus_baseline_rerun: f1 -0.0182 [-0.1017, 0.0689]; precision -0.0179 [-0.1026, 0.0663]; recall -0.0185 [-0.1191, 0.0784]; high_risk_recall +0.0000 [-0.125, 0.125]; clean_accuracy -0.0180 [-0.0647, 0.0289]; average_instruction_tokens_per_call -55.6200 [-113.97, 6.54]; average_total_tokens_per_pr -884.0600 [-1916.27, 60.29]
- governed_canary_minus_governed: f1 -0.0022 [-0.0756, 0.0691]; precision +0.0115 [-0.0573, 0.0882]; recall -0.0185 [-0.0968, 0.0625]; high_risk_recall +0.0000 [0.0, 0.0]; clean_accuracy +0.0179 [-0.0243, 0.0596]; average_instruction_tokens_per_call +122.7700 [91.28, 156.53]; average_total_tokens_per_pr +68.5500 [-838.72, 925.74]
- naive_minus_baseline: f1 -0.0336 [-0.1182, 0.0467]; precision -0.0302 [-0.1188, 0.0489]; recall -0.0371 [-0.1305, 0.05]; high_risk_recall +0.0000 [-0.12, 0.125]; clean_accuracy -0.0240 [-0.0585, 0.0061]; average_instruction_tokens_per_call +784.4700 [709.96, 858.69]; average_total_tokens_per_pr +3683.5600 [2025.74, 5664.6]
- baseline_minus_baseline_rerun: f1 +0.0364 [-0.0288, 0.103]; precision +0.0357 [-0.036, 0.1098]; recall +0.0371 [-0.0339, 0.1132]; high_risk_recall +0.0000 [-0.125, 0.1176]; clean_accuracy +0.0060 [-0.0297, 0.0426]; average_instruction_tokens_per_call +5.9800 [-32.47, 49.45]; average_total_tokens_per_pr +409.9100 [-440.52, 1407.33]

## Holdout (all)

| arm | f1 | precision | recall | high_risk_recall | clean_accuracy | instr. tokens/call | tokens/PR |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.3508 | 0.3448 | 0.3571 | 0.36 | 0.8623 | 1056.79 | 24267.33 |
| naive | 0.3063 | 0.3091 | 0.3036 | 0.36 | 0.8383 | 1840.59 | 27609.71 |
| baseline_rerun | 0.3036 | 0.3036 | 0.3036 | 0.36 | 0.8563 | 1057.95 | 23962.02 |
| governed | 0.2975 | 0.2769 | 0.3214 | 0.36 | 0.8204 | 871.29 | 22934.73 |
| governed_canary | 0.3103 | 0.3 | 0.3214 | 0.4 | 0.8383 | 994.56 | 23026.59 |


## 同仓库时间划分

各仓库按 created_at 排序，前 60% 训练、接下来 10% 门禁、最后 30% holdout；PR 不跨划分。

| 仓库 | 划分 | PR 数 | 最早创建时间 | 最晚创建时间 |
|---|---|---:|---|---|
| scikit-learn/scikit-learn | train | 145 | 2011-05-21T16:45:53+00:00 | 2021-09-01T18:35:50+00:00 |
| scikit-learn/scikit-learn | validation | 25 | 2021-09-02T14:20:13+00:00 | 2022-06-30T20:57:58+00:00 |
| scikit-learn/scikit-learn | holdout | 73 | 2022-07-16T10:14:55+00:00 | 2025-04-15T11:53:30+00:00 |
| sympy/sympy | train | 135 | 2011-04-18T04:35:42+00:00 | 2019-12-28T13:56:09+00:00 |
| sympy/sympy | validation | 22 | 2019-12-31T15:45:24+00:00 | 2020-08-09T15:46:17+00:00 |
| sympy/sympy | holdout | 68 | 2020-08-29T22:35:12+00:00 | 2025-03-14T13:53:42+00:00 |
| matplotlib/matplotlib | train | 87 | 2011-08-14T21:39:06+00:00 | 2018-02-10T23:16:38+00:00 |
| matplotlib/matplotlib | validation | 14 | 2018-04-15T22:41:12+00:00 | 2020-03-08T09:42:41+00:00 |
| matplotlib/matplotlib | holdout | 44 | 2020-03-20T06:01:32+00:00 | 2025-03-10T16:34:04+00:00 |
| astropy/astropy | train | 67 | 2011-12-13T14:56:36+00:00 | 2018-03-28T20:59:46+00:00 |
| astropy/astropy | validation | 12 | 2018-05-29T11:06:28+00:00 | 2019-12-17T21:58:57+00:00 |
| astropy/astropy | holdout | 34 | 2020-04-08T15:46:38+00:00 | 2025-03-31T16:22:41+00:00 |

## 已反馈问题纠正率

这是对已反馈问题的纠正（同一批 PR），不代表泛化能力。

- 抽样 150 个不同 PR，四组共同成功 149 个；按反馈类型抽样：{"missed": 80, "false_positive": 80}。完整 PR 清单见 [recurrence.json](recurrence.json)。
- 各通道最终失败 PR 数：{"baseline": 1, "naive": 0, "baseline_rerun": 0, "governed": 0, "governed_canary": 0}。

| 通道 | 误报复发率 | 漏报找回率 | 正确发现保留率 |
|---|---:|---:|---:|
| baseline | 0.544304 (43/79) | 0.091954 (8/87) | 0.909091 (10/11) |
| naive | 0.227848 (18/79) | 0.091954 (8/87) | 0.909091 (10/11) |
| baseline_rerun | 0.531646 (42/79) | 0.137931 (12/87) | 1.0 (11/11) |
| governed | 0.468354 (37/79) | 0.114943 (10/87) | 0.909091 (10/11) |
| governed_canary | 0.291139 (23/79) | 0.137931 (12/87) | 0.818182 (9/11) |

误报复发率越低越好，其余两项越高越好。配对 bootstrap 按 PR 重采样，95% CI：

- governed_minus_baseline：{"false_positive_recurrence": {"delta": -0.07595, "ci95": [-0.188406, 0.0375], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": false}, "missed_recovery": {"delta": 0.022989, "ci95": [-0.028571, 0.081081], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": true}, "correct_retention": {"delta": 0.0, "ci95": [0.0, 0.0], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": true}}
- governed_canary_minus_baseline：{"false_positive_recurrence": {"delta": -0.253165, "ci95": [-0.381579, -0.135135], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": false}, "missed_recovery": {"delta": 0.045977, "ci95": [-0.025, 0.122222], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": true}, "correct_retention": {"delta": -0.090909, "ci95": [-0.3, 0.0], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": true}}
- governed_minus_naive：{"false_positive_recurrence": {"delta": 0.240506, "ci95": [0.125, 0.355263], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": false}, "missed_recovery": {"delta": 0.022989, "ci95": [-0.035294, 0.081081], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": true}, "correct_retention": {"delta": 0.0, "ci95": [0.0, 0.0], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": true}}
- governed_canary_minus_naive：{"false_positive_recurrence": {"delta": 0.063291, "ci95": [-0.026316, 0.152941], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": false}, "missed_recovery": {"delta": 0.045977, "ci95": [-0.012658, 0.10989], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": true}, "correct_retention": {"delta": -0.090909, "ci95": [-0.3, 0.0], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": true}}
- governed_minus_baseline_rerun：{"false_positive_recurrence": {"delta": -0.063292, "ci95": [-0.170732, 0.041096], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": false}, "missed_recovery": {"delta": -0.022988, "ci95": [-0.1, 0.047619], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": true}, "correct_retention": {"delta": -0.090909, "ci95": [-0.285714, 0.0], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": true}}
- governed_canary_minus_baseline_rerun：{"false_positive_recurrence": {"delta": -0.240507, "ci95": [-0.358974, -0.121622], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": false}, "missed_recovery": {"delta": 0.0, "ci95": [-0.079545, 0.079545], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": true}, "correct_retention": {"delta": -0.181818, "ci95": [-0.444444, 0.0], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": true}}
- governed_canary_minus_governed：{"false_positive_recurrence": {"delta": -0.177215, "ci95": [-0.292135, -0.068966], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": false}, "missed_recovery": {"delta": 0.022988, "ci95": [-0.0375, 0.08642], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": true}, "correct_retention": {"delta": -0.090909, "ci95": [-0.3, 0.0], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": true}}
- naive_minus_baseline：{"false_positive_recurrence": {"delta": -0.316456, "ci95": [-0.426829, -0.2125], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": false}, "missed_recovery": {"delta": 0.0, "ci95": [-0.054348, 0.055556], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": true}, "correct_retention": {"delta": 0.0, "ci95": [0.0, 0.0], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": true}}
- baseline_minus_baseline_rerun：{"false_positive_recurrence": {"delta": 0.012658, "ci95": [-0.094118, 0.123288], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": false}, "missed_recovery": {"delta": -0.045977, "ci95": [-0.11236, 0.012346], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": true}, "correct_retention": {"delta": -0.090909, "ci95": [-0.285714, 0.0], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": true}}

## 知识治理统计

写入决策按候选条目计数，共 244 条；CONFLICT 单独展示，其持久化动作仍为 REJECT。

| 来源 | ADD | REJECT | UPVOTE | CONFLICT | MERGE | RESTORE |
|---|---:|---:|---:|---:|---:|---:|
| 矛盾 | 2 | 6 | 0 | 16 | 0 | 0 |
| 误报反馈 | 75 | 5 | 2 | 0 | 0 | 0 |
| 漏报反馈 | 77 | 8 | 4 | 0 | 0 | 0 |
| 噪声 | 0 | 49 | 0 | 0 | 0 | 0 |

### 噪声与矛盾拦截

| 来源 | 拒绝候选/候选总数 | 候选拦截率 | 全部候选均被拒的事件/有候选事件 | 事件拦截率 |
|---|---:|---:|---:|---:|
| 矛盾 | 22/24 | 91.67% | 22/24 | 91.67% |
| 噪声 | 49/49 | 100.00% | 49/49 | 100.00% |

### 最终在用学习知识的污染率

污染定义为 source 属于 noise 或 contradiction。分母包含在用规则和学习技能，排除预置 disk 技能；正式通道只计 active，灰度通道计 active+probation。分母为 0 时不报告污染率。

| 通道 | 污染条目/在用条目 | 条目污染率 | 污染 token/在用 token | token 污染率 |
|---|---:|---:|---:|---:|
| naive | 73/244 | 29.92% | 2816/23862 | 11.80% |
| governed 正式 | 0/9 | 0.00% | 0/477 | 0.00% |
| governed 正式+试用 | 0/124 | 0.00% | 0/14236 | 0.00% |

### 转正、归档与反事实归因

- 转正动作 9 次，涉及 9 个不同条目；归档动作 30 次，涉及 30 个不同条目。归档包含淘汰和归并。
- 当前归档 30 个条目，其中容量淘汰 30、harmful 淘汰 0、归并 0；完整原因计数见 governance.json。容量淘汰不能表述为识别并清除有害知识。
- 反事实比较尝试 80 对，发起审查 158 次（包含失败和未注入试用条目的回放），完整且实际注入后记账 44 对；有益 4、有害 0、中性 40、失败 3、未实际注入 33。
- 归因涉及 17 个条目、34 个不同训练 PR；训练集隔离检查：通过。

### 来源 PR 验证与快速转正

- 验证尝试 57 个条目批次，发起 340 次训练审查，完成记账 45 个条目、153 对比较；有益 26、有害 15、中性 112。
- 通过来源筛选 9 个不同条目；通过原 validation 门禁后快速转正 9 个不同条目（9 次动作）。结果状态：{'unverified': 18, 'neutral': 27, 'harmful': 9, 'passed': 9}。
- 失败 11、未实际注入 1、跳过 6、尚无结果的预算预留 0；来源 PR 56 个，副作用检查 PR 15 个；仅训练集：通过。
- 已归档的漏报/误报条目按当前有效验证状态计数：{'neutral': 16, 'unverified': 9, 'harmful': 3}。
- k=3 是筛选条件，不是显著性检验；来源 PR 与纠正率评测重叠，不能将验证通过数表述为泛化收益。

| 维护轮次 | 审查上限 | 实际发起审查 | 已消耗预算（含未完成预留） |
|---|---:|---:|---:|
| 100 | 120 | 111 | 111 |
| 200 | 120 | 115 | 115 |
| 243 | 120 | 114 | 114 |

