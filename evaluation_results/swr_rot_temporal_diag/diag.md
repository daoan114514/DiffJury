# 同仓库时间外推诊断

## 方法与样本

沿用原实验 holdout PR、head 快照、diff、标注、模型和预算；仅增加三组回放，原目录和数据库只读。
主要配对比较使用五组共同成功的 217 个 PR。原实验 baseline/governed 的正文日志未保存，正文 token 对比使用本次三组逐调用记录，不填补旧日志。
配对 bootstrap 按 PR 重采样 2,000 次，seed=20260819。A/A 是 baseline 减 baseline_rerun，其他比较名 A_minus_B 表示 A 减 B。

## 共同成功样本的质量

| 组 | F1 | 精确率 | 召回率 | 命中 | 误报 | 漏报 | 指令 token/调用 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.3571 | 0.3509 | 0.3636 | 20 | 37 | 35 | 1056.42 |
| governed | 0.2667 | 0.2462 | 0.2909 | 16 | 49 | 39 | 837.28 |
| baseline_rerun | 0.3063 | 0.3036 | 0.3091 | 17 | 39 | 38 | 1057.28 |
| governed_assembler_empty | 0.2807 | 0.2712 | 0.2909 | 16 | 43 | 39 | 818.74 |
| governed_rerun | 0.3214 | 0.3158 | 0.3273 | 18 | 39 | 37 | 845.03 |

### 完整样本与执行失败

下表保留全量 PR 口径，执行失败可能被记作未命中，不能单独用来归因质量差异；上表与以下 CI 均使用共同成功样本。

| 组 | PR 数 | F1 | 精确率 | 召回率 | 误报 | 执行失败 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 219 | 0.3508 | 0.3448 | 0.3571 | 38 | 0 |
| governed | 219 | 0.2623 | 0.2424 | 0.2857 | 50 | 1 |
| baseline_rerun | 219 | 0.3036 | 0.3036 | 0.3036 | 39 | 1 |
| governed_assembler_empty | 219 | 0.2783 | 0.2712 | 0.2857 | 43 | 2 |
| governed_rerun | 219 | 0.3158 | 0.3103 | 0.3214 | 40 | 1 |

完整结果及共同成功 PR 清单见 [diag.json](diag.json)。

## 配对差值与 95% CI

| 比较 | 指标 | 差值 | 95% CI |
|---|---|---:|---|
| baseline_minus_baseline_rerun | f1 | +0.0508 | [-0.0188, 0.1226] |
| baseline_minus_baseline_rerun | precision | +0.0473 | [-0.0245, 0.1268] |
| baseline_minus_baseline_rerun | recall | +0.0545 | [-0.0192, 0.1395] |
| baseline_minus_baseline_rerun | fp | -2.0000 | [-12.0, 7.0] |
| baseline_minus_baseline_rerun | average_instruction_tokens_per_call | -0.8600 | [-44.48, 43.44] |
| governed_assembler_empty_minus_baseline_rerun | f1 | -0.0256 | [-0.1299, 0.0762] |
| governed_assembler_empty_minus_baseline_rerun | precision | -0.0324 | [-0.1291, 0.0625] |
| governed_assembler_empty_minus_baseline_rerun | recall | -0.0182 | [-0.1344, 0.0981] |
| governed_assembler_empty_minus_baseline_rerun | fp | +4.0000 | [-6.0, 13.0] |
| governed_assembler_empty_minus_baseline_rerun | average_instruction_tokens_per_call | -238.5400 | [-305.76, -175.71] |
| governed_rerun_minus_governed_assembler_empty | f1 | +0.0407 | [-0.0442, 0.1324] |
| governed_rerun_minus_governed_assembler_empty | precision | +0.0446 | [-0.0439, 0.1311] |
| governed_rerun_minus_governed_assembler_empty | recall | +0.0364 | [-0.06, 0.1428] |
| governed_rerun_minus_governed_assembler_empty | fp | -4.0000 | [-13.0, 6.0] |
| governed_rerun_minus_governed_assembler_empty | average_instruction_tokens_per_call | +26.2900 | [4.68, 50.29] |
| governed_rerun_minus_baseline_rerun | f1 | +0.0151 | [-0.0573, 0.0909] |
| governed_rerun_minus_baseline_rerun | precision | +0.0122 | [-0.0622, 0.0859] |
| governed_rerun_minus_baseline_rerun | recall | +0.0182 | [-0.0589, 0.1] |
| governed_rerun_minus_baseline_rerun | fp | +0.0000 | [-10.0, 9.0] |
| governed_rerun_minus_baseline_rerun | average_instruction_tokens_per_call | -212.2500 | [-276.23, -153.73] |
| governed_minus_baseline | f1 | -0.0904 | [-0.1598, -0.0328] |
| governed_minus_baseline | precision | -0.1047 | [-0.1848, -0.0419] |
| governed_minus_baseline | recall | -0.0727 | [-0.15, -0.0164] |
| governed_minus_baseline | fp | +12.0000 | [3.0, 22.0] |
| governed_minus_baseline | average_instruction_tokens_per_call | -219.1400 | [-276.37, -167.31] |
| governed_rerun_minus_governed | f1 | +0.0547 | [-0.0214, 0.1332] |
| governed_rerun_minus_governed | precision | +0.0696 | [0.0073, 0.1404] |
| governed_rerun_minus_governed | recall | +0.0364 | [-0.0492, 0.125] |
| governed_rerun_minus_governed | fp | -10.0000 | [-19.0, -1.0] |
| governed_rerun_minus_governed | average_instruction_tokens_per_call | +7.7500 | [-17.6, 32.68] |

## 预置 Skill 裁剪与正文

以下频率按 PR 计算。逐 PR 被裁名单与正文可用量见 [skill_audit.json](skill_audit.json)。离线探针假设全选目录，仅说明可用量；实际 worker 正文另按调用记录统计。

| 被裁预置 Skill | PR 次数 |
|---|---:|
| performance-review | 12 |
| test-quality | 89 |
| code-quality | 5 |
| api-compatibility | 6 |
| security-review | 94 |
| database-review | 5 |
| reliability-review | 8 |

| 组 | 已记录 worker 调用 | 正文 token 总数 | 正文 token/worker 调用 |
|---|---:|---:|---:|
| baseline_rerun | 537 | 829525 | 1544.74 |
| governed_assembler_empty | 515 | 533006 | 1034.96 |
| governed_rerun | 527 | 567696 | 1077.22 |

正文 token 用项目统一估算，不等于完整 system prompt 或网关 input token。worker 选择还受模型委派波动影响，不能把观察到的所有正文差值都归因于裁剪。

## 诊断结论

A/A 的 F1 差值为 +0.0508，95% CI [-0.0188, 0.1226]。这是一对重复运行的观察值，不是模型长期波动的上界。
空知识装配器相对新基线的 F1 CI 未完全低于 0；当前诊断未确认裁剪造成质量下降，暂不修改预置 Skill 的裁剪规则。目录裁剪本身已被离线检查，但其质量影响与模型波动不能据此等同。
带正式知识相对空知识装配器的 F1 CI 未完全低于 0，不能确认原下降由这两条知识导致。
原 governed 相对原基线的 F1 差为 -0.0904，重跑两组的差为 +0.0151 [-0.0573, 0.0909]；A/A 点差的绝对幅度为 0.0508。
本次重跑的点估计未复现原下降；这仍不足以把原下降全部归为随机波动。
正式知识曝光与引用统计：{"governed": {"exposed_prs": null, "cited_prs": 0}, "governed_rerun": {"exposed_prs": 28, "cited_prs": 0}}。原实验未保存曝光日志，其曝光为 0 的计数不能作证据，引用计数仍可复核。
单次 A/A 与每组一次独立回放不足以精确分解随机波动、裁剪和知识的贡献。CI 包含 0 时不称提升；PR bootstrap 未覆盖多次模型运行的波动。结果只适用于本次四仓库配置。
