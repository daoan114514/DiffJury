# SWR-Bench context-rot experiment

- model: gpt-6.1-sol; dataset: {"train": 442, "gate": 60, "holdout": 376, "holdout_recent_2024": 25, "holdout_executed_all": 372}
- events: 253 {"missed": 106, "false_positive": 71, "noise": 51, "contradiction": 25}
- admission (governed): {"REJECT": 195, "ADD": 54, "UPVOTE": 4}
- holdout failure rate (first pass -> after retries): {"baseline": {"first": 0.016, "after_retry": 0.0, "retried": 6}, "naive": {"first": 0.0106, "after_retry": 0.0, "retried": 4}, "governed": {"first": 0.0904, "after_retry": 0.0106, "retried": 34}}

## Holdout (all)

| arm | f1 | precision | recall | high_risk_recall | clean_accuracy | instr. tokens/call | tokens/PR |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.3009 | 0.2537 | 0.3696 | 0.4681 | 0.8028 | 1198.11 | 27755.32 |
| naive | 0.2708 | 0.2263 | 0.337 | 0.4255 | 0.8131 | 4178.8 | 41702.29 |
| governed | 0.2844 | 0.246 | 0.337 | 0.4255 | 0.7924 | 967.97 | 26337.11 |

- governed_minus_naive: f1 +0.0136 [-0.0392, 0.0675]; precision +0.0197 [-0.0252, 0.0668]; recall +0.0000 [-0.0704, 0.0736]; high_risk_recall +0.0000 [-0.1042, 0.1042]; clean_accuracy -0.0207 [-0.0561, 0.0138]; average_instruction_tokens_per_call -3210.8300 [-3330.42, -3079.92]; average_total_tokens_per_pr -15365.1800 [-17568.39, -13386.42]
- naive_minus_baseline: f1 -0.0301 [-0.0828, 0.0212]; precision -0.0274 [-0.0739, 0.0144]; recall -0.0326 [-0.1136, 0.0435]; high_risk_recall -0.0426 [-0.15, 0.0638]; clean_accuracy +0.0103 [-0.0209, 0.0441]; average_instruction_tokens_per_call +2980.6900 [2867.86, 3086.8]; average_total_tokens_per_pr +13946.9700 [12281.23, 15844.59]
- governed_minus_baseline: f1 -0.0165 [-0.0678, 0.0315]; precision -0.0077 [-0.0532, 0.0384]; recall -0.0326 [-0.0975, 0.0326]; high_risk_recall -0.0426 [-0.1315, 0.0392]; clean_accuracy -0.0104 [-0.0412, 0.0207]; average_instruction_tokens_per_call -230.1400 [-269.35, -187.52]; average_total_tokens_per_pr -1418.2100 [-2562.56, -314.05]

## Holdout (recent_2024)

| arm | f1 | precision | recall | high_risk_recall | clean_accuracy | instr. tokens/call | tokens/PR |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.3 | 0.3 | 0.3 | 0.5 | 0.8125 | 1006.1 | 19643.48 |
| naive | 0.2857 | 0.2727 | 0.3 | 0.3333 | 0.875 | 3639.67 | 33039.64 |
| governed | 0.2353 | 0.2857 | 0.2 | 0.3333 | 0.8125 | 877.12 | 16695.16 |

- governed_minus_naive: f1 -0.0504 [-0.304, 0.0866]; precision +0.0130 [-0.3333, 0.35]; recall -0.1000 [-0.3334, 0.0]; high_risk_recall +0.0000 [0.0, 0.0]; clean_accuracy -0.0625 [-0.2, 0.0]; average_instruction_tokens_per_call -2762.5500 [-3337.16, -2167.3]; average_total_tokens_per_pr -16344.4800 [-25251.12, -9710.0]
- naive_minus_baseline: f1 -0.0143 [-0.2566, 0.2476]; precision -0.0273 [-0.2273, 0.2667]; recall +0.0000 [-0.2857, 0.3333]; high_risk_recall -0.1667 [-0.5, 0.0]; clean_accuracy +0.0625 [0.0, 0.2]; average_instruction_tokens_per_call +2633.5700 [2124.93, 3089.65]; average_total_tokens_per_pr +13396.1600 [8306.48, 21447.76]
- governed_minus_baseline: f1 -0.0647 [-0.3, 0.0763]; precision -0.0143 [-0.3333, 0.4091]; recall -0.1000 [-0.3334, 0.0]; high_risk_recall -0.1667 [-0.5, 0.0]; clean_accuracy +0.0000 [-0.1818, 0.1765]; average_instruction_tokens_per_call -128.9800 [-288.25, -14.6]; average_total_tokens_per_pr -2948.3200 [-6111.92, -641.52]

## Holdout (executed_all)

| arm | f1 | precision | recall | high_risk_recall | clean_accuracy | instr. tokens/call | tokens/PR |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.3162 | 0.2698 | 0.382 | 0.5 | 0.8056 | 1199.87 | 27669.33 |
| naive | 0.2791 | 0.2381 | 0.3371 | 0.4318 | 0.816 | 4180.08 | 41679.36 |
| governed | 0.2883 | 0.246 | 0.3483 | 0.4545 | 0.7951 | 967.97 | 26620.3 |

- governed_minus_naive: f1 +0.0092 [-0.0458, 0.0638]; precision +0.0079 [-0.0386, 0.0521]; recall +0.0112 [-0.0632, 0.0897]; high_risk_recall +0.0227 [-0.0769, 0.122]; clean_accuracy -0.0209 [-0.0588, 0.0143]; average_instruction_tokens_per_call -3212.1100 [-3331.98, -3088.12]; average_total_tokens_per_pr -15059.0600 [-17301.77, -13110.02]
- naive_minus_baseline: f1 -0.0371 [-0.094, 0.0184]; precision -0.0317 [-0.0797, 0.0138]; recall -0.0449 [-0.1274, 0.0283]; high_risk_recall -0.0682 [-0.175, 0.0256]; clean_accuracy +0.0104 [-0.0237, 0.0445]; average_instruction_tokens_per_call +2980.2100 [2870.15, 3088.54]; average_total_tokens_per_pr +14010.0300 [12344.24, 15867.99]
- governed_minus_baseline: f1 -0.0279 [-0.0826, 0.0214]; precision -0.0238 [-0.0688, 0.0172]; recall -0.0337 [-0.101, 0.0358]; high_risk_recall -0.0455 [-0.1395, 0.0385]; clean_accuracy -0.0105 [-0.0435, 0.0238]; average_instruction_tokens_per_call -231.9000 [-270.14, -190.74]; average_total_tokens_per_pr -1049.0300 [-2189.63, 37.34]

