# Context-rot stress test

- events: 500 (mix missed 50%, false_positive 20%, noise 20%, contradiction 10%)
- model: gpt-6.1-sol
- seed: 20261010
- admission actions (governed): {"ADD": 58, "REJECT": 163, "MERGE": 44, "UPVOTE": 235}

## Prompt size (probe, mean instruction tokens per model call)

| event | naive tokens/call | governed tokens/call | naive live tokens | governed live tokens | naive rules | governed rules |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 291.8 | 291.8 | 0 | 0 | 0 | 0 |
| 25 | 609.0 | 291.8 | 2580 | 0 | 22 | 0 |
| 50 | 841.2 | 291.8 | 5675 | 0 | 43 | 0 |
| 75 | 1156.0 | 291.8 | 7480 | 0 | 67 | 0 |
| 100 | 1437.0 | 291.8 | 9635 | 0 | 90 | 0 |
| 125 | 1682.0 | 291.8 | 13296 | 0 | 110 | 0 |
| 128 | 1725.2 | 291.8 | 13951 | 0 | 113 | 0 |
| 150 | 2008.5 | 291.8 | 15091 | 0 | 135 | 0 |
| 175 | 2283.8 | 291.8 | 17242 | 0 | 158 | 0 |
| 200 | 2550.2 | 291.8 | 19873 | 0 | 180 | 0 |
| 225 | 2789.5 | 291.8 | 22979 | 0 | 201 | 0 |
| 250 | 3039.8 | 468.8 | 25527 | 597 | 223 | 11 |
| 275 | 3270.5 | 468.8 | 29551 | 597 | 242 | 11 |
| 300 | 3578.5 | 469.8 | 32384 | 601 | 264 | 11 |
| 325 | 3861.2 | 469.8 | 34553 | 601 | 287 | 11 |
| 350 | 4200.0 | 471.2 | 36445 | 607 | 311 | 11 |
| 375 | 4428.5 | 486.0 | 40509 | 723 | 330 | 13 |
| 400 | 4718.2 | 485.8 | 42692 | 722 | 353 | 13 |
| 425 | 5026.5 | 485.8 | 44956 | 722 | 376 | 13 |
| 450 | 5331.8 | 485.8 | 47269 | 722 | 399 | 13 |
| 475 | 5647.0 | 485.2 | 48534 | 720 | 424 | 13 |
| 500 | 5896.2 | 485.2 | 51132 | 720 | 446 | 13 |

## Holdout quality (repos 09-10, 3 repeat(s) pooled)

| event | arm | F1 | F1 per repeat | precision | recall | high-risk recall | clean acc. | instr. tokens/call | tokens/PR | failure |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | governed | 0.3529 | 0.3704 / 0.5 / 0.1904 | 0.2727 | 0.5 | 0.8889 | 0.6111 | 1703.67 | 44823.2 | 0.0 |
| 0 | naive | 0.3529 | 0.3704 / 0.5 / 0.1904 | 0.2727 | 0.5 | 0.8889 | 0.6111 | 1703.67 | 44823.2 | 0.0 |
| 100 | governed | 0.3243 | 0.3636 / 0.24 / 0.3704 | 0.24 | 0.5 | 0.7778 | 0.3889 | 1378.3 | 41614.15 | 0.0 |
| 100 | naive | 0.3421 | 0.32 / 0.3479 / 0.3571 | 0.25 | 0.5417 | 0.8889 | 0.4722 | 3092.44 | 62848.97 | 0.0 |
| 250 | governed | 0.1918 | 0.1379 / 0.25 / 0.2 | 0.1429 | 0.2917 | 0.5556 | 0.4722 | 1449.27 | 40990.8 | 0.0 |
| 250 | naive | 0.2632 | 0.2857 / 0.2727 / 0.2308 | 0.1923 | 0.4167 | 0.6667 | 0.3889 | 4531.48 | 77163.32 | 0.0 |
| 500 | governed | 0.225 | 0.2666 / 0.2143 / 0.1819 | 0.1607 | 0.375 | 0.7778 | 0.3889 | 1485.9 | 43515.02 | 0.0 |
| 500 | naive | 0.2727 | 0.2667 / 0.3479 / 0.2143 | 0.2143 | 0.375 | 0.7778 | 0.5556 | 6926.87 | 100287.48 | 0.0 |

## Governed minus naive (paired bootstrap, 95% CI)

| event | metric | delta | 95% CI |
|---:|---|---:|---|
| 100 | f1 | -0.0178 | [-0.1455, 0.1155] |
| 100 | precision | -0.01 | [-0.1222, 0.0945] |
| 100 | recall | -0.0417 | [-0.2222, 0.1428] |
| 100 | high_risk_recall | -0.1111 | [-0.5, 0.2857] |
| 100 | clean_accuracy | -0.0833 | [-0.3142, 0.1463] |
| 100 | average_instruction_tokens_per_call | -1714.14 | [-1812.15, -1616.61] |
| 100 | average_total_tokens_per_pr | -21234.82 | [-25331.56, -16754.69] |
| 250 | f1 | -0.0714 | [-0.2043, 0.0563] |
| 250 | precision | -0.0494 | [-0.1522, 0.0548] |
| 250 | recall | -0.125 | [-0.3334, 0.08] |
| 250 | high_risk_recall | -0.1111 | [-0.5, 0.2857] |
| 250 | clean_accuracy | 0.0833 | [-0.0937, 0.2581] |
| 250 | average_instruction_tokens_per_call | -3082.21 | [-3224.62, -2945.57] |
| 250 | average_total_tokens_per_pr | -36172.52 | [-40177.76, -32041.18] |
| 500 | f1 | -0.0477 | [-0.1605, 0.0643] |
| 500 | precision | -0.0536 | [-0.1651, 0.0353] |
| 500 | recall | 0.0 | [-0.1739, 0.1666] |
| 500 | high_risk_recall | 0.0 | [-0.3, 0.3333] |
| 500 | clean_accuracy | -0.1667 | [-0.4118, 0.0882] |
| 500 | average_instruction_tokens_per_call | -5440.97 | [-5642.06, -5244.92] |
| 500 | average_total_tokens_per_pr | -56772.46 | [-62581.23, -50853.1] |

## Maintenance (governed)

- event 125: {"dream": {"clusters": 3, "evictions": {"K-53a01cd0": "probation-failed", "K-720fc79c": "probation-failed", "K-de6c9ca9": "probation-failed", "K-217d6f75": "probation-failed"}, "live_items_before": 30, "live_items_after": 26, "live_tokens_before": 4502, "live_tokens_after": 4291, "consolidated": true, "release": {"status": "rejected", "version": 2, "gate": {"passed": false, "failures": ["high_risk_recall"], "margin": 0.02, "max_point_drop": 0.1, "dataset_fingerprint": "fc06a9f11fd0ecde481f2eb1c4baf1d98416de1e08755d6624b4f7fffd9bf70c", "cases": 20, "metrics": {"f1": {"baseline": 0.3448, "candidate": 0.3448, "delta": 0.0, "ci95": [-0.1618, 0.1354], "passed": true}, "high_risk_recall": {"baseline": 0.75, "candidate": 0.5, "delta": -0.25, "ci95": [-0.75, 0.0], "passed": false}, "clean_accuracy": {"baseline": 0.3333, "candidate": 0.25, "delta": -0.0833, "ci95": [-0.4445, 0.3], "passed": true}}, "instruction_tokens_per_call": {"baseline": 1561.93, "candidate": 1533.55, "max_growth": 0.0, "passed": true}, "execution_success_rate": {"baseline": 1.0, "candidate": 1.0}, "baseline_snapshot": "c0c0eae61cb1f4f7d6b8f963", "candidate_snapshot": "5ea1c8fe43b01f38915103be"}}}, "merges": 0, "drops": 0, "promotion": {"status": "rejected", "version": 3}, "dream_release": "rejected", "dream_gate_failures": ["high_risk_recall"], "live_tokens_before": 4502, "live_tokens_after": 4502}
- event 128: {"dream": {"clusters": 3, "evictions": {"K-53a01cd0": "probation-failed", "K-720fc79c": "probation-failed", "K-de6c9ca9": "probation-failed", "K-217d6f75": "probation-failed"}, "live_items_before": 30, "live_items_after": 26, "live_tokens_before": 4538, "live_tokens_after": 4327, "consolidated": true, "release": {"status": "released", "version": 4, "gate": {"passed": true, "failures": [], "margin": 0.02, "max_point_drop": 0.1, "dataset_fingerprint": "fc06a9f11fd0ecde481f2eb1c4baf1d98416de1e08755d6624b4f7fffd9bf70c", "cases": 20, "metrics": {"f1": {"baseline": 0.3704, "candidate": 0.3479, "delta": -0.0225, "ci95": [-0.3129, 0.2909], "max_point_drop": 0.1, "passed": true}, "high_risk_recall": {"baseline": 0.75, "candidate": 1.0, "delta": 0.25, "ci95": [0.0, 0.75], "max_point_drop": 0.25, "passed": true}, "clean_accuracy": {"baseline": 0.25, "candidate": 0.5833, "delta": 0.3333, "ci95": [-0.1538, 0.75], "max_point_drop": 0.1, "passed": true}}, "instruction_tokens_per_call": {"baseline": 1552.18, "candidate": 1533.78, "max_growth": 0.0, "passed": true}, "execution_success_rate": {"baseline": 1.0, "candidate": 1.0}, "baseline_snapshot": "b10cdddf763a176c046827e2", "candidate_snapshot": "465df4c6a152083e539b700a"}, "changes": {"updated": [], "archived": ["K-53a01cd0", "K-720fc79c", "K-de6c9ca9", "K-217d6f75"], "created": []}}}, "merges": 0, "drops": 0, "promotion": {"status": "rejected", "version": 5}, "promotion_gate_failures": ["clean_accuracy"], "dream_release": "released", "dream_gate_failures": [], "live_tokens_before": 4538, "live_tokens_after": 4327}
- event 250: {"dream": {"clusters": 3, "evictions": {"K-183d0956": "probation-failed", "K-168c51c8": "probation-failed", "K-9ffd0be4": "probation-failed", "K-513eaddc": "probation-failed", "K-64cf5a3b": "probation-failed", "K-b4d8838a": "probation-failed", "K-0549a911": "probation-failed", "K-f511254d": "probation-failed"}, "live_items_before": 39, "live_items_after": 30, "live_tokens_before": 6497, "live_tokens_after": 6006, "consolidated": true, "release": {"status": "rejected", "version": 8, "gate": {"passed": false, "failures": ["clean_accuracy"], "margin": 0.02, "max_point_drop": 0.1, "dataset_fingerprint": "fc06a9f11fd0ecde481f2eb1c4baf1d98416de1e08755d6624b4f7fffd9bf70c", "cases": 20, "metrics": {"f1": {"baseline": 0.2778, "candidate": 0.2581, "delta": -0.0197, "ci95": [-0.141, 0.1026], "max_point_drop": 0.1, "passed": true}, "high_risk_recall": {"baseline": 0.75, "candidate": 0.5, "delta": -0.25, "ci95": [-0.75, 0.0], "max_point_drop": 0.25, "passed": true}, "clean_accuracy": {"baseline": 0.3333, "candidate": 0.1667, "delta": -0.1666, "ci95": [-0.5556, 0.2308], "max_point_drop": 0.1, "passed": false}}, "instruction_tokens_per_call": {"baseline": 1546.82, "candidate": 1398.52, "max_growth": 0.0, "passed": true}, "execution_success_rate": {"baseline": 1.0, "candidate": 1.0}, "baseline_snapshot": "1ad5c4511f98a70861460e85", "candidate_snapshot": "ee30ace52421cac77919b162"}}}, "merges": 1, "drops": 0, "promotion": {"status": "released", "version": 9, "changes": {"promoted": ["K-b94557cc", "K-b06b769c", "K-e8cdf771", "K-8ada0ec0", "K-cb3c0600", "K-624b9627", "K-94b803f0", "K-cbc475c5", "K-610ca850", "K-6a4bbb46", "K-684f0141"]}}, "promotion_gate_failures": [], "dream_release": "rejected", "dream_gate_failures": ["clean_accuracy"], "live_tokens_before": 6497, "live_tokens_after": 6497}
- event 375: {"dream": {"clusters": 5, "evictions": {"K-183d0956": "probation-failed", "K-9ffd0be4": "probation-failed", "K-513eaddc": "probation-failed", "K-64cf5a3b": "probation-failed", "K-b4d8838a": "probation-failed", "K-0549a911": "probation-failed", "K-f511254d": "probation-failed", "K-f12c8aed": "probation-failed"}, "live_items_before": 43, "live_items_after": 34, "live_tokens_before": 6812, "live_tokens_after": 6301, "consolidated": true, "release": {"status": "released", "version": 10, "gate": {"passed": true, "failures": [], "margin": 0.02, "max_point_drop": 0.1, "dataset_fingerprint": "fc06a9f11fd0ecde481f2eb1c4baf1d98416de1e08755d6624b4f7fffd9bf70c", "cases": 20, "metrics": {"f1": {"baseline": 0.4348, "candidate": 0.4445, "delta": 0.0097, "ci95": [-0.1923, 0.2096], "max_point_drop": 0.1, "passed": true}, "high_risk_recall": {"baseline": 0.75, "candidate": 0.75, "delta": 0.0, "ci95": [0.0, 0.0], "max_point_drop": 0.25, "passed": true}, "clean_accuracy": {"baseline": 0.5833, "candidate": 0.5, "delta": -0.0833, "ci95": [-0.4167, 0.2728], "max_point_drop": 0.1, "passed": true}}, "instruction_tokens_per_call": {"baseline": 1530.96, "candidate": 1465.2, "max_growth": 0.0, "passed": true}, "execution_success_rate": {"baseline": 1.0, "candidate": 1.0}, "baseline_snapshot": "660b46c3ff335f3e8c9b0fb7", "candidate_snapshot": "f8767c1e96d9c909b12c09ab"}, "changes": {"updated": [], "archived": ["K-183d0956", "K-9ffd0be4", "K-513eaddc", "K-64cf5a3b", "K-b4d8838a", "K-8408310a", "K-0549a911", "K-f511254d", "K-f12c8aed"], "created": []}}}, "merges": 1, "drops": 0, "promotion": {"status": "released", "version": 11, "changes": {"promoted": ["K-168c51c8", "K-9f78cac5"]}}, "promotion_gate_failures": [], "dream_release": "released", "dream_gate_failures": [], "live_tokens_before": 6812, "live_tokens_after": 6301}
- event 500: {"dream": {"clusters": 3, "evictions": {"K-cff5d1de": "probation-failed", "K-c90a8009": "probation-failed", "K-dc7da626": "probation-failed"}, "live_items_before": 45, "live_items_after": 39, "live_tokens_before": 7427, "live_tokens_after": 7104, "consolidated": true, "release": {"status": "rejected", "version": 12, "gate": {"passed": false, "failures": ["f1"], "margin": 0.02, "max_point_drop": 0.1, "dataset_fingerprint": "fc06a9f11fd0ecde481f2eb1c4baf1d98416de1e08755d6624b4f7fffd9bf70c", "cases": 20, "metrics": {"f1": {"baseline": 0.4445, "candidate": 0.3333, "delta": -0.1112, "ci95": [-0.3309, 0.0929], "max_point_drop": 0.1, "passed": false}, "high_risk_recall": {"baseline": 1.0, "candidate": 0.75, "delta": -0.25, "ci95": [-0.75, 0.0], "max_point_drop": 0.25, "passed": true}, "clean_accuracy": {"baseline": 0.3333, "candidate": 0.6667, "delta": 0.3334, "ci95": [-0.1111, 0.75], "max_point_drop": 0.1, "passed": true}}, "instruction_tokens_per_call": {"baseline": 1549.1, "candidate": 1547.85, "max_growth": 0.0, "passed": true}, "execution_success_rate": {"baseline": 1.0, "candidate": 1.0}, "baseline_snapshot": "b7551d1b5d393d966c2c97eb", "candidate_snapshot": "c9861f8fd1af0a1e8cdab481"}}}, "merges": 2, "drops": 0, "promotion": {"status": "rejected", "version": 13}, "promotion_gate_failures": ["clean_accuracy"], "dream_release": "rejected", "dream_gate_failures": ["f1"], "live_tokens_before": 7427, "live_tokens_after": 7427}
