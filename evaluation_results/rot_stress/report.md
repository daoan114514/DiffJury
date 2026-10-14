# Context-rot stress test

- events: 500 (mix missed 50%, false_positive 20%, noise 20%, contradiction 10%)
- model: gpt-6.1-sol
- seed: 20261010
- admission actions (governed): {"ADD": 57, "REJECT": 160, "UPVOTE": 242, "MERGE": 41}

## Prompt size (probe, mean instruction tokens per model call)

| event | naive tokens/call | governed tokens/call | naive live tokens | governed live tokens | naive rules | governed rules |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 291.8 | 291.8 | 0 | 0 | 0 | 0 |
| 25 | 609.0 | 291.8 | 2580 | 0 | 22 | 0 |
| 50 | 841.2 | 291.8 | 5675 | 0 | 43 | 0 |
| 75 | 1156.0 | 291.8 | 7480 | 0 | 67 | 0 |
| 100 | 1437.0 | 291.8 | 9635 | 0 | 90 | 0 |
| 125 | 1682.0 | 291.8 | 13296 | 0 | 110 | 0 |
| 150 | 1995.0 | 291.8 | 15037 | 0 | 134 | 0 |
| 175 | 2270.2 | 291.8 | 17188 | 0 | 157 | 0 |
| 200 | 2536.8 | 291.8 | 19819 | 0 | 179 | 0 |
| 225 | 2776.0 | 291.8 | 22925 | 0 | 200 | 0 |
| 250 | 3026.2 | 320.2 | 25473 | 117 | 222 | 2 |
| 275 | 3257.0 | 320.2 | 29497 | 117 | 241 | 2 |
| 300 | 3565.0 | 320.2 | 32330 | 117 | 263 | 2 |
| 325 | 3847.8 | 320.2 | 34499 | 117 | 286 | 2 |
| 350 | 4186.5 | 320.2 | 36391 | 117 | 310 | 2 |
| 375 | 4415.0 | 348.8 | 40455 | 174 | 329 | 3 |
| 400 | 4704.8 | 348.8 | 42638 | 174 | 352 | 3 |
| 425 | 5013.0 | 348.8 | 44902 | 174 | 375 | 3 |
| 450 | 5318.2 | 348.8 | 47215 | 174 | 398 | 3 |
| 475 | 5633.5 | 348.8 | 48480 | 174 | 423 | 3 |
| 500 | 5882.8 | 348.8 | 51078 | 174 | 445 | 3 |

## Holdout quality (repos 09-10)

| event | arm | F1 | precision | recall | high-risk recall | clean acc. | instr. tokens/call | tokens/PR | failure |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | governed | 0.3158 | 0.2727 | 0.375 | 0.6667 | 0.6667 | 1670.86 | 43925.7 | 0.0 |
| 0 | naive | 0.3158 | 0.2727 | 0.375 | 0.6667 | 0.6667 | 1670.86 | 43925.7 | 0.0 |
| 100 | governed | 0.25 | 0.1875 | 0.375 | 1.0 | 0.3333 | 1397.98 | 41016.6 | 0.0 |
| 100 | naive | 0.3077 | 0.2222 | 0.5 | 1.0 | 0.4167 | 2917.56 | 58514.5 | 0.0 |
| 250 | governed | 0.4348 | 0.3333 | 0.625 | 1.0 | 0.75 | 1333.93 | 38420.7 | 0.0 |
| 250 | naive | 0.421 | 0.3636 | 0.5 | 1.0 | 0.6667 | 4545.65 | 77832.15 | 0.0 |
| 500 | governed | 0.24 | 0.1765 | 0.375 | 0.6667 | 0.5833 | 1441.52 | 41684.55 | 0.0 |
| 500 | naive | 0.2353 | 0.2222 | 0.25 | 0.3333 | 0.6667 | 7076.4 | 106311.7 | 0.0 |

## Maintenance (governed)

- event 125: {"dream": {"clusters": 2, "evictions": {"K-607c7124": "probation-failed", "K-baed409a": "probation-failed", "K-e41ddf3d": "probation-failed", "K-1963e5a5": "probation-failed", "K-2183c84d": "probation-failed", "K-dfb5141a": "probation-failed", "K-ef9bbb68": "probation-failed", "K-30da9f30": "probation-failed", "K-01624d47": "probation-failed", "K-022c3187": "probation-failed", "K-27704d2d": "probation-failed", "K-3d9ee47d": "probation-failed"}, "live_items_before": 28, "live_items_after": 16, "live_tokens_before": 4401, "live_tokens_after": 3743, "consolidated": true, "release": {"status": "released", "version": 2, "gate": {"passed": true, "failures": [], "margin": 0.02, "dataset_fingerprint": "fc06a9f11fd0ecde481f2eb1c4baf1d98416de1e08755d6624b4f7fffd9bf70c", "cases": 20, "metrics": {"f1": {"baseline": 0.2308, "candidate": 0.4138, "delta": 0.183, "ci95": [0.0133, 0.3704], "passed": true}, "high_risk_recall": {"baseline": 0.5, "candidate": 0.75, "delta": 0.25, "ci95": [0.0, 0.75], "passed": true}, "clean_accuracy": {"baseline": 0.1667, "candidate": 0.4167, "delta": 0.25, "ci95": [0.0, 0.5385], "passed": true}}, "instruction_tokens_per_call": {"baseline": 1445.44, "candidate": 1385.82, "max_growth": 0.0, "passed": true}, "execution_success_rate": {"baseline": 1.0, "candidate": 1.0}, "baseline_snapshot": "b90bc569c51040b6ffdc689f", "candidate_snapshot": "df908850fb52acf0eba45adf"}, "changes": {"updated": [], "archived": ["K-607c7124", "K-baed409a", "K-e41ddf3d", "K-1963e5a5", "K-2183c84d", "K-dfb5141a", "K-ef9bbb68", "K-30da9f30", "K-01624d47", "K-022c3187", "K-27704d2d", "K-3d9ee47d"], "created": []}}}, "merges": 0, "drops": 0, "promotion": {"status": "rejected", "version": 3}, "dream_release": "released", "dream_gate_failures": [], "live_tokens_before": 4401, "live_tokens_after": 3743}
- event 250: {"dream": {"clusters": 3, "evictions": {"K-151dd8a5": "probation-failed", "K-c338cda4": "probation-failed", "K-aeb93438": "probation-failed", "K-5fc77b80": "probation-failed", "K-fc58aaac": "probation-failed", "K-2292231d": "probation-failed", "K-f14ffd08": "probation-failed", "K-bb7903d3": "probation-failed", "K-80ff85b6": "probation-failed", "K-36682595": "probation-failed", "K-c4cc05ec": "probation-failed", "K-a0f0b7ee": "probation-failed", "K-9138f340": "probation-failed", "K-895f38fc": "probation-failed"}, "live_items_before": 35, "live_items_after": 21, "live_tokens_before": 6304, "live_tokens_after": 5550, "consolidated": true, "release": {"status": "rejected", "version": 4, "gate": {"passed": false, "failures": ["f1", "clean_accuracy"], "margin": 0.02, "dataset_fingerprint": "fc06a9f11fd0ecde481f2eb1c4baf1d98416de1e08755d6624b4f7fffd9bf70c", "cases": 20, "metrics": {"f1": {"baseline": 0.4445, "candidate": 0.2857, "delta": -0.1588, "ci95": [-0.2892, -0.032], "passed": false}, "high_risk_recall": {"baseline": 0.75, "candidate": 0.75, "delta": 0.0, "ci95": [0.0, 0.0], "passed": true}, "clean_accuracy": {"baseline": 0.5, "candidate": 0.4167, "delta": -0.0833, "ci95": [-0.4445, 0.3], "passed": false}}, "instruction_tokens_per_call": {"baseline": 1523.65, "candidate": 1372.52, "max_growth": 0.0, "passed": true}, "execution_success_rate": {"baseline": 1.0, "candidate": 1.0}, "baseline_snapshot": "fe3b21991d65c18a1dff0ead", "candidate_snapshot": "b0cc229a081d0b50efacf30c"}}}, "merges": 0, "drops": 0, "promotion": {"status": "released", "version": 5, "changes": {"promoted": ["K-43f46d5f", "K-8d68b64e"]}}, "dream_release": "rejected", "dream_gate_failures": ["f1", "clean_accuracy"], "live_tokens_before": 6304, "live_tokens_after": 6304}
- event 375: {"dream": {"clusters": 3, "evictions": {"K-151dd8a5": "probation-failed", "K-03464c33": "probation-failed", "K-c338cda4": "probation-failed", "K-aeb93438": "probation-failed", "K-5fc77b80": "probation-failed", "K-2292231d": "probation-failed", "K-f14ffd08": "probation-failed", "K-bb7903d3": "probation-failed", "K-80ff85b6": "probation-failed", "K-36682595": "probation-failed", "K-c4cc05ec": "probation-failed", "K-a0f0b7ee": "probation-failed", "K-9138f340": "probation-failed", "K-66e07948": "probation-failed", "K-895f38fc": "probation-failed", "K-b96f55bd": "probation-failed", "K-31f9a4ac": "probation-failed", "K-5d33a428": "probation-failed", "K-2f9ce973": "probation-failed", "K-421d06dc": "probation-failed"}, "live_items_before": 39, "live_items_after": 19, "live_tokens_before": 6714, "live_tokens_after": 5611, "consolidated": true, "release": {"status": "rejected", "version": 6, "gate": {"passed": false, "failures": ["clean_accuracy"], "margin": 0.02, "dataset_fingerprint": "fc06a9f11fd0ecde481f2eb1c4baf1d98416de1e08755d6624b4f7fffd9bf70c", "cases": 20, "metrics": {"f1": {"baseline": 0.4167, "candidate": 0.4138, "delta": -0.0029, "ci95": [-0.1964, 0.207], "passed": true}, "high_risk_recall": {"baseline": 0.75, "candidate": 1.0, "delta": 0.25, "ci95": [0.0, 0.75], "passed": true}, "clean_accuracy": {"baseline": 0.5, "candidate": 0.4167, "delta": -0.0833, "ci95": [-0.4546, 0.3], "passed": false}}, "instruction_tokens_per_call": {"baseline": 1534.57, "candidate": 1454.01, "max_growth": 0.0, "passed": true}, "execution_success_rate": {"baseline": 1.0, "candidate": 1.0}, "baseline_snapshot": "be806231fb78423ad8a6364b", "candidate_snapshot": "9b2a7c7b9a94eab942cd84ec"}}}, "merges": 0, "drops": 0, "promotion": {"status": "released", "version": 7, "changes": {"promoted": ["K-fc58aaac"]}}, "dream_release": "rejected", "dream_gate_failures": ["clean_accuracy"], "live_tokens_before": 6714, "live_tokens_after": 6714}
- event 500: {"dream": {"clusters": 2, "evictions": {"K-151dd8a5": "probation-failed", "K-03464c33": "probation-failed", "K-c338cda4": "probation-failed", "K-aeb93438": "probation-failed", "K-5fc77b80": "probation-failed", "K-2292231d": "probation-failed", "K-f14ffd08": "probation-failed", "K-bb7903d3": "probation-failed", "K-80ff85b6": "probation-failed", "K-36682595": "probation-failed", "K-c4cc05ec": "probation-failed", "K-a0f0b7ee": "probation-failed", "K-9138f340": "probation-failed", "K-66e07948": "probation-failed", "K-895f38fc": "probation-failed", "K-b96f55bd": "probation-failed", "K-31f9a4ac": "probation-failed", "K-5d33a428": "probation-failed", "K-2f9ce973": "probation-failed", "K-421d06dc": "probation-failed"}, "live_items_before": 45, "live_items_after": 25, "live_tokens_before": 7993, "live_tokens_after": 6890, "consolidated": true, "release": {"status": "released", "version": 8, "gate": {"passed": true, "failures": [], "margin": 0.02, "dataset_fingerprint": "fc06a9f11fd0ecde481f2eb1c4baf1d98416de1e08755d6624b4f7fffd9bf70c", "cases": 20, "metrics": {"f1": {"baseline": 0.4167, "candidate": 0.5, "delta": 0.0833, "ci95": [-0.24, 0.3175], "passed": true}, "high_risk_recall": {"baseline": 0.75, "candidate": 0.75, "delta": 0.0, "ci95": [0.0, 0.0], "passed": true}, "clean_accuracy": {"baseline": 0.5, "candidate": 0.9167, "delta": 0.4167, "ci95": [0.1111, 0.6923], "passed": true}}, "instruction_tokens_per_call": {"baseline": 1581.45, "candidate": 1433.92, "max_growth": 0.0, "passed": true}, "execution_success_rate": {"baseline": 1.0, "candidate": 1.0}, "baseline_snapshot": "eb8c6f5a49d77bb3fe9489d0", "candidate_snapshot": "7359182fb4e011bea560f53e"}, "changes": {"updated": [], "archived": ["K-151dd8a5", "K-03464c33", "K-c338cda4", "K-aeb93438", "K-5fc77b80", "K-2292231d", "K-f14ffd08", "K-bb7903d3", "K-80ff85b6", "K-36682595", "K-c4cc05ec", "K-a0f0b7ee", "K-9138f340", "K-66e07948", "K-895f38fc", "K-b96f55bd", "K-31f9a4ac", "K-5d33a428", "K-2f9ce973", "K-421d06dc"], "created": []}}}, "merges": 0, "drops": 0, "promotion": {}, "dream_release": "released", "dream_gate_failures": [], "live_tokens_before": 7993, "live_tokens_after": 6890}
