# SWR-Bench 知识治理评测

- 实验完成时间：2026-10-13T23:58:23
- 配置模型：gpt-6.1-sol；数据集：{"train": 434, "gate": 73, "holdout": 219, "holdout_recent_2024": 66, "holdout_executed_all": 217}
- 学习事件：243 {"missed": 89, "false_positive": 81, "noise": 49, "contradiction": 24}
- governed 写入动作：{"ADD": 158, "REJECT": 80, "UPVOTE": 6}
- holdout 执行失败率（首轮、补跑后）：{"baseline": {"first": 0.032, "after_retry": 0.0, "retried": 7}, "naive": {"first": 0.0228, "after_retry": 0.0091, "retried": 5}, "governed": {"first": 0.0183, "after_retry": 0.0046, "retried": 4}}

主要质量结论使用 executed_all（三组共同成功的 PR）；all 保留完整样本结果，recent_2024 为敏感性检查。比较名 A_minus_B 表示 A 减 B。

## Holdout (executed_all)

| arm | f1 | precision | recall | high_risk_recall | clean_accuracy | instr. tokens/call | tokens/PR |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.3455 | 0.3393 | 0.3519 | 0.375 | 0.8623 | 1056.12 | 24180.62 |
| naive | 0.3119 | 0.3091 | 0.3148 | 0.375 | 0.8383 | 1840.59 | 27864.18 |
| governed | 0.2543 | 0.2344 | 0.2778 | 0.3333 | 0.8204 | 837.28 | 22315.63 |

- governed_minus_naive: f1 -0.0576 [-0.1392, 0.0239]; precision -0.0747 [-0.1629, 0.0026]; recall -0.0370 [-0.125, 0.0535]; high_risk_recall -0.0417 [-0.1852, 0.0953]; clean_accuracy -0.0179 [-0.0602, 0.0229]; average_instruction_tokens_per_call -1003.3100 [-1087.94, -914.7]; average_total_tokens_per_pr -5548.5500 [-7615.53, -4006.59]
- naive_minus_baseline: f1 -0.0336 [-0.1182, 0.0467]; precision -0.0302 [-0.1188, 0.0489]; recall -0.0371 [-0.1305, 0.05]; high_risk_recall +0.0000 [-0.12, 0.125]; clean_accuracy -0.0240 [-0.0585, 0.0061]; average_instruction_tokens_per_call +784.4700 [709.96, 858.69]; average_total_tokens_per_pr +3683.5600 [2025.74, 5664.6]
- governed_minus_baseline: f1 -0.0912 [-0.162, -0.0335]; precision -0.1049 [-0.1881, -0.0436]; recall -0.0741 [-0.1539, -0.0167]; high_risk_recall -0.0417 [-0.1333, 0.0]; clean_accuracy -0.0419 [-0.0823, -0.0059]; average_instruction_tokens_per_call -218.8400 [-273.56, -165.69]; average_total_tokens_per_pr -1864.9900 [-3245.01, -722.01]

## Holdout (all)

| arm | f1 | precision | recall | high_risk_recall | clean_accuracy | instr. tokens/call | tokens/PR |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.3508 | 0.3448 | 0.3571 | 0.36 | 0.8623 | 1056.79 | 24267.33 |
| naive | 0.3063 | 0.3091 | 0.3036 | 0.36 | 0.8383 | 1840.59 | 27609.71 |
| governed | 0.2623 | 0.2424 | 0.2857 | 0.32 | 0.8204 | 837.12 | 22312.59 |

- governed_minus_naive: f1 -0.0440 [-0.1276, 0.0377]; precision -0.0667 [-0.1569, 0.0101]; recall -0.0179 [-0.1111, 0.0747]; high_risk_recall -0.0400 [-0.1818, 0.0909]; clean_accuracy -0.0179 [-0.0614, 0.0227]; average_instruction_tokens_per_call -1003.4700 [-1089.61, -916.92]; average_total_tokens_per_pr -5297.1200 [-7259.97, -3681.83]
- naive_minus_baseline: f1 -0.0445 [-0.1302, 0.0378]; precision -0.0357 [-0.125, 0.0467]; recall -0.0535 [-0.151, 0.0378]; high_risk_recall +0.0000 [-0.1111, 0.12]; clean_accuracy -0.0240 [-0.0578, 0.0063]; average_instruction_tokens_per_call +783.8000 [706.01, 858.72]; average_total_tokens_per_pr +3342.3800 [1535.67, 5275.46]
- governed_minus_baseline: f1 -0.0885 [-0.1591, -0.0316]; precision -0.1024 [-0.1787, -0.0358]; recall -0.0714 [-0.1463, -0.0154]; high_risk_recall -0.0400 [-0.125, 0.0]; clean_accuracy -0.0419 [-0.0828, -0.006]; average_instruction_tokens_per_call -219.6700 [-272.89, -168.68]; average_total_tokens_per_pr -1954.7400 [-3369.62, -802.2]

## Holdout (recent_2024)

| arm | f1 | precision | recall | high_risk_recall | clean_accuracy | instr. tokens/call | tokens/PR |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.1176 | 0.1538 | 0.0952 | 0.1 | 0.8696 | 1101.4 | 25605.23 |
| naive | 0.1819 | 0.25 | 0.1429 | 0.1 | 0.8696 | 1724.9 | 26485.8 |
| governed | 0.125 | 0.1818 | 0.0952 | 0.1 | 0.8696 | 874.57 | 23157.45 |

- governed_minus_naive: f1 -0.0569 [-0.1892, 0.0077]; precision -0.0682 [-0.3, 0.0238]; recall -0.0477 [-0.16, 0.0]; high_risk_recall +0.0000 [0.0, 0.0]; clean_accuracy +0.0000 [0.0, 0.0]; average_instruction_tokens_per_call -850.3300 [-1005.43, -710.09]; average_total_tokens_per_pr -3328.3500 [-4516.62, -2192.81]
- naive_minus_baseline: f1 +0.0643 [-0.0077, 0.1929]; precision +0.0962 [-0.0238, 0.3778]; recall +0.0477 [0.0, 0.1579]; high_risk_recall +0.0000 [0.0, 0.0]; clean_accuracy +0.0000 [-0.0612, 0.0638]; average_instruction_tokens_per_call +623.5000 [470.83, 760.43]; average_total_tokens_per_pr +880.5700 [-1757.29, 2786.18]
- governed_minus_baseline: f1 +0.0074 [-0.0079, 0.0278]; precision +0.0280 [-0.041, 0.1515]; recall +0.0000 [0.0, 0.0]; high_risk_recall +0.0000 [0.0, 0.0]; clean_accuracy +0.0000 [-0.0612, 0.0638]; average_instruction_tokens_per_call -226.8300 [-355.09, -118.48]; average_total_tokens_per_pr -2447.7800 [-5003.86, -759.63]

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
- 各通道最终失败 PR 数：{"baseline": 1, "naive": 0, "governed": 0, "governed_canary": 0}。

| 通道 | 误报复发率 | 漏报找回率 | 正确发现保留率 |
|---|---:|---:|---:|
| baseline | 0.544304 (43/79) | 0.091954 (8/87) | 0.909091 (10/11) |
| naive | 0.227848 (18/79) | 0.091954 (8/87) | 0.909091 (10/11) |
| governed | 0.455696 (36/79) | 0.091954 (8/87) | 0.909091 (10/11) |
| governed_canary | 0.455696 (36/79) | 0.103448 (9/87) | 0.818182 (9/11) |

误报复发率越低越好，其余两项越高越好。配对 bootstrap 按 PR 重采样，95% CI：

- naive_minus_baseline：{"false_positive_recurrence": {"delta": -0.316456, "ci95": [-0.426829, -0.2125], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": false}, "missed_recovery": {"delta": 0.0, "ci95": [-0.054348, 0.055556], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": true}, "correct_retention": {"delta": 0.0, "ci95": [0.0, 0.0], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": true}}
- governed_minus_baseline：{"false_positive_recurrence": {"delta": -0.088608, "ci95": [-0.2, 0.023529], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": false}, "missed_recovery": {"delta": 0.0, "ci95": [-0.053333, 0.057471], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": true}, "correct_retention": {"delta": 0.0, "ci95": [0.0, 0.0], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": true}}
- governed_canary_minus_baseline：{"false_positive_recurrence": {"delta": -0.088608, "ci95": [-0.216216, 0.035714], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": false}, "missed_recovery": {"delta": 0.011494, "ci95": [-0.036145, 0.060976], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": true}, "correct_retention": {"delta": -0.090909, "ci95": [-0.3, 0.0], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": true}}
- governed_minus_naive：{"false_positive_recurrence": {"delta": 0.227848, "ci95": [0.116883, 0.341176], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": false}, "missed_recovery": {"delta": 0.0, "ci95": [-0.043011, 0.042254], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": true}, "correct_retention": {"delta": 0.0, "ci95": [0.0, 0.0], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": true}}
- governed_canary_minus_governed：{"false_positive_recurrence": {"delta": 0.0, "ci95": [-0.098765, 0.1], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": false}, "missed_recovery": {"delta": 0.011494, "ci95": [-0.048193, 0.070588], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": true}, "correct_retention": {"delta": -0.090909, "ci95": [-0.3, 0.0], "iterations": 2000, "paired_prs": 149, "sampling_unit": "PR", "higher_is_better": true}}

## 知识治理统计

写入决策按候选条目计数，共 244 条；CONFLICT 单独展示，其持久化动作仍为 REJECT。

| 来源 | ADD | REJECT | UPVOTE | CONFLICT | MERGE | RESTORE |
|---|---:|---:|---:|---:|---:|---:|
| 矛盾 | 3 | 5 | 1 | 15 | 0 | 0 |
| 误报反馈 | 79 | 2 | 1 | 0 | 0 | 0 |
| 漏报反馈 | 76 | 9 | 4 | 0 | 0 | 0 |
| 噪声 | 0 | 49 | 0 | 0 | 0 | 0 |

### 噪声与矛盾拦截

| 来源 | 拒绝候选/候选总数 | 候选拦截率 | 全部候选均被拒的事件/有候选事件 | 事件拦截率 |
|---|---:|---:|---:|---:|
| 矛盾 | 20/24 | 83.33% | 20/24 | 83.33% |
| 噪声 | 49/49 | 100.00% | 49/49 | 100.00% |

### 最终在用学习知识的污染率

污染定义为 source 属于 noise 或 contradiction。分母包含在用规则和学习技能，排除预置 disk 技能；正式通道只计 active，灰度通道计 active+probation。分母为 0 时不报告污染率。

| 通道 | 污染条目/在用条目 | 条目污染率 | 污染 token/在用 token | token 污染率 |
|---|---:|---:|---:|---:|
| naive | 73/244 | 29.92% | 2816/23862 | 11.80% |
| governed 正式 | 0/2 | 0.00% | 0/116 | 0.00% |
| governed 正式+试用 | 2/97 | 2.06% | 100/12229 | 0.82% |

### 转正、归档与反事实归因

- 转正动作 2 次，涉及 2 个不同条目；归档动作 61 次，涉及 61 个不同条目。归档包含淘汰和归并。
- 当前归档 61 个条目，其中容量淘汰 61、harmful 淘汰 0、归并 0；完整原因计数见 governance.json。容量淘汰不能表述为识别并清除有害知识。
- 反事实比较尝试 60 对，发起审查 119 次（包含失败和未注入试用条目的回放），完整且实际注入后记账 30 对；有益 4、有害 1、中性 25、失败 2、未实际注入 28。
- 归因涉及 12 个条目、23 个不同训练 PR；训练集隔离检查：通过。

