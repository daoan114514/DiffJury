# Context-rot stress test

- events: 253 (mix {"missed": 106, "false_positive": 71, "noise": 51, "contradiction": 25})
- model: gpt-6.1-sol
- seed: 20261013
- admission actions (governed): {"REJECT": 195, "ADD": 54, "UPVOTE": 4}

## Prompt size (probe, mean instruction tokens per model call)

| event | naive tokens/call | governed tokens/call | naive live tokens | governed live tokens | naive rules | governed rules |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 291.8 | 291.8 | 0 | 0 | 0 | 0 |
| 25 | 541.6 | 291.8 | 2818 | 0 | 23 | 0 |
| 50 | 756.6 | 291.8 | 5910 | 0 | 45 | 0 |
| 75 | 984.1 | 291.8 | 7630 | 0 | 70 | 0 |
| 100 | 1185.9 | 291.8 | 9140 | 0 | 95 | 0 |
| 125 | 1383.7 | 291.8 | 12285 | 0 | 117 | 0 |
| 150 | 1546.6 | 291.8 | 14731 | 0 | 140 | 0 |
| 175 | 1755.5 | 291.8 | 18246 | 0 | 161 | 0 |
| 200 | 1953.8 | 291.8 | 20953 | 0 | 184 | 0 |
| 225 | 2202.7 | 291.8 | 22497 | 0 | 209 | 0 |
| 250 | 2379.8 | 291.8 | 24051 | 0 | 234 | 0 |
| 253 | 2404.7 | 303.4 | 24299 | 60 | 237 | 1 |

## Holdout quality (repos 09-10, 1 repeat(s) pooled)

| event | arm | F1 | F1 per repeat | precision | recall | high-risk recall | clean acc. | instr. tokens/call | tokens/PR | failure |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | governed | 0.3009 |  | 0.2537 | 0.3696 | 0.4681 | 0.8028 | 1198.11 | 27755.32 | 0.0 |
| 0 | naive | 0.3009 |  | 0.2537 | 0.3696 | 0.4681 | 0.8028 | 1198.11 | 27755.32 | 0.0 |
| 253 | governed | 0.2844 |  | 0.246 | 0.337 | 0.4255 | 0.7924 | 967.97 | 26337.11 | 0.0106 |
| 253 | naive | 0.2708 |  | 0.2263 | 0.337 | 0.4255 | 0.8131 | 4178.8 | 41702.29 | 0.0 |

## Governed minus naive (paired bootstrap, 95% CI)

| event | metric | delta | 95% CI |
|---:|---|---:|---|
| 253 | f1 | 0.0136 | [-0.0392, 0.0675] |
| 253 | precision | 0.0197 | [-0.0252, 0.0668] |
| 253 | recall | 0.0 | [-0.0704, 0.0736] |
| 253 | high_risk_recall | 0.0 | [-0.1042, 0.1042] |
| 253 | clean_accuracy | -0.0207 | [-0.0561, 0.0138] |
| 253 | average_instruction_tokens_per_call | -3210.83 | [-3330.42, -3079.92] |
| 253 | average_total_tokens_per_pr | -15365.18 | [-17568.39, -13386.42] |

## Maintenance (governed)

- event 200: {"dream": {"clusters": 0, "evictions": {"K-25e6be30": "probation-failed", "K-b1217978": "probation-failed", "K-21eeedbc": "probation-failed"}, "live_items_before": 42, "live_items_after": 39, "live_tokens_before": 9294, "live_tokens_after": 9127, "consolidated": true, "release": {"status": "released", "version": 2, "gate": {"passed": true, "failures": [], "margin": 0.02, "max_point_drop": 0.1, "dataset_fingerprint": "15e2ece9f9c7e2634ca04989d68f9d1ebfbdbe471f7f7eefb76e29eb21c1b4dc", "cases": 60, "metrics": {"f1": {"baseline": 0.3509, "candidate": 0.3934, "delta": 0.0425, "ci95": [-0.0598, 0.1491], "max_point_drop": 0.1, "passed": true}, "high_risk_recall": {"baseline": 0.3333, "candidate": 0.4286, "delta": 0.0953, "ci95": [0.0, 0.2174], "max_point_drop": 0.1, "passed": true}, "clean_accuracy": {"baseline": 0.7333, "candidate": 0.7, "delta": -0.0333, "ci95": [-0.1143, 0.0], "max_point_drop": 0.1, "passed": true}}, "instruction_tokens_per_call": {"baseline": 1049.78, "candidate": 1025.08, "max_growth": 0.0, "passed": true}, "execution_success_rate": {"baseline": 1.0, "candidate": 1.0}, "baseline_snapshot": "ddbb7698a9654e349bb9f513", "candidate_snapshot": "be5155ea027867fea5ae02b5"}, "changes": {"updated": [], "archived": ["K-25e6be30", "K-b1217978", "K-21eeedbc"], "created": []}}}, "promotion": {}, "promotion_gate_failures": null, "dream_release": "released", "dream_gate_failures": [], "live_tokens_before": 9294, "live_tokens_after": 9127}
- event 253: {"dream": {"clusters": 0, "evictions": {"K-76f4b913": "probation-failed", "K-7963039d": "probation-failed"}, "live_items_before": 51, "live_items_after": 49, "live_tokens_before": 9755, "live_tokens_after": 9645, "consolidated": true, "release": {"status": "released", "version": 3, "gate": {"passed": true, "failures": [], "margin": 0.02, "max_point_drop": 0.1, "dataset_fingerprint": "15e2ece9f9c7e2634ca04989d68f9d1ebfbdbe471f7f7eefb76e29eb21c1b4dc", "cases": 60, "metrics": {"f1": {"baseline": 0.3666, "candidate": 0.4193, "delta": 0.0527, "ci95": [-0.0529, 0.1681], "max_point_drop": 0.1, "passed": true}, "high_risk_recall": {"baseline": 0.381, "candidate": 0.381, "delta": 0.0, "ci95": [-0.1429, 0.15], "max_point_drop": 0.1, "passed": true}, "clean_accuracy": {"baseline": 0.7667, "candidate": 0.7333, "delta": -0.0334, "ci95": [-0.1143, 0.0], "max_point_drop": 0.1, "passed": true}}, "instruction_tokens_per_call": {"baseline": 1071.41, "candidate": 1013.84, "max_growth": 0.0, "passed": true}, "execution_success_rate": {"baseline": 1.0, "candidate": 1.0}, "baseline_snapshot": "63ac5ef63acb7a52333be666", "candidate_snapshot": "264bdeb0dfd8f3ea9ccbd0cd"}, "changes": {"updated": [], "archived": ["K-76f4b913", "K-7963039d"], "created": []}}}, "promotion": {"status": "released", "version": 4, "changes": {"promoted": ["K-fb99060d"]}}, "promotion_gate_failures": [], "dream_release": "released", "dream_gate_failures": [], "live_tokens_before": 9755, "live_tokens_after": 9645}
