# SWR-Bench context-rot experiment

- model: gpt-6.1-sol; dataset: {"train": 442, "gate": 60, "holdout": 376, "holdout_recent_2024": 25}
- events: 253 {"missed": 106, "false_positive": 71, "noise": 51, "contradiction": 25}
- admission (governed): {"REJECT": 195, "ADD": 54, "UPVOTE": 4}

## Holdout (all)

| arm | f1 | precision | recall | high_risk_recall | clean_accuracy | instr. tokens/call | tokens/PR |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.3 | 0.2578 | 0.3587 | 0.4681 | 0.8028 | 1195.52 | 27140.91 |
| naive | 0.2806 | 0.2403 | 0.337 | 0.4255 | 0.8131 | 4179.36 | 41171.86 |
| governed | 0.2802 | 0.2522 | 0.3152 | 0.4043 | 0.7301 | 961.02 | 23706.38 |

- governed_minus_naive: f1 -0.0004 [-0.0625, 0.0576]; precision +0.0119 [-0.0423, 0.066]; recall -0.0218 [-0.0968, 0.0555]; high_risk_recall -0.0212 [-0.1333, 0.087]; clean_accuracy -0.0830 [-0.1273, -0.0456]; average_instruction_tokens_per_call -3218.3400 [-3338.73, -3086.11]; average_total_tokens_per_pr -17465.4800 [-20187.18, -15034.92]
- naive_minus_baseline: f1 -0.0194 [-0.078, 0.034]; precision -0.0175 [-0.0687, 0.0299]; recall -0.0217 [-0.1022, 0.0562]; high_risk_recall -0.0426 [-0.15, 0.0638]; clean_accuracy +0.0103 [-0.0209, 0.0441]; average_instruction_tokens_per_call +2983.8400 [2869.04, 3091.83]; average_total_tokens_per_pr +14030.9500 [12181.66, 16039.24]
- governed_minus_baseline: f1 -0.0198 [-0.0772, 0.033]; precision -0.0056 [-0.0579, 0.0454]; recall -0.0435 [-0.1101, 0.023]; high_risk_recall -0.0638 [-0.1667, 0.0238]; clean_accuracy -0.0727 [-0.114, -0.0356]; average_instruction_tokens_per_call -234.5000 [-277.53, -186.93]; average_total_tokens_per_pr -3434.5300 [-5267.36, -1611.12]

## Holdout (recent_2024)

| arm | f1 | precision | recall | high_risk_recall | clean_accuracy | instr. tokens/call | tokens/PR |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.3333 | 0.375 | 0.3 | 0.5 | 0.8125 | 1006.82 | 18582.44 |
| naive | 0.4 | 0.6 | 0.3 | 0.3333 | 0.875 | 3607.44 | 27654.0 |
| governed | 0.2353 | 0.2857 | 0.2 | 0.3333 | 0.8125 | 877.12 | 16695.16 |

- governed_minus_naive: f1 -0.1647 [-0.4305, 0.0238]; precision -0.3143 [-1.0, 0.1]; recall -0.1000 [-0.3334, 0.0]; high_risk_recall +0.0000 [0.0, 0.0]; clean_accuracy -0.0625 [-0.2, 0.0]; average_instruction_tokens_per_call -2730.3200 [-3351.74, -2088.69]; average_total_tokens_per_pr -10958.8400 [-22274.76, 925.32]
- naive_minus_baseline: f1 +0.0667 [-0.2518, 0.3778]; precision +0.2250 [-0.2667, 0.8571]; recall +0.0000 [-0.2857, 0.3333]; high_risk_recall -0.1667 [-0.5, 0.0]; clean_accuracy +0.0625 [0.0, 0.2]; average_instruction_tokens_per_call +2600.6200 [2039.15, 3090.11]; average_total_tokens_per_pr +9071.5600 [-3009.96, 20061.04]
- governed_minus_baseline: f1 -0.0980 [-0.3333, 0.0357]; precision -0.0893 [-0.5, 0.2]; recall -0.1000 [-0.3334, 0.0]; high_risk_recall -0.1667 [-0.5, 0.0]; clean_accuracy +0.0000 [-0.1818, 0.1765]; average_instruction_tokens_per_call -129.7000 [-297.09, -13.75]; average_total_tokens_per_pr -1887.2800 [-4181.88, -436.72]

