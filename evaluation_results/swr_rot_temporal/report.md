# Context-rot stress test

- events: 243 (mix {"missed": 89, "false_positive": 81, "noise": 49, "contradiction": 24})
- model: gpt-6.1-sol
- seed: 20261013
- admission actions (governed): {"ADD": 158, "REJECT": 80, "UPVOTE": 6}

## Prompt size (probe, mean instruction tokens per model call)

| event | naive tokens/call | governed tokens/call | naive live tokens | governed live tokens | naive rules | governed rules |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 291.8 | 291.8 | 0 | 0 | 0 | 0 |
| 25 | 315.2 | 291.8 | 2383 | 0 | 24 | 0 |
| 50 | 334.6 | 291.8 | 5690 | 0 | 46 | 0 |
| 75 | 350.1 | 291.8 | 7881 | 0 | 70 | 0 |
| 100 | 350.1 | 291.8 | 10334 | 59 | 94 | 1 |
| 125 | 395.9 | 291.8 | 12710 | 59 | 117 | 1 |
| 150 | 440.0 | 291.8 | 15242 | 59 | 140 | 1 |
| 175 | 453.1 | 291.8 | 17988 | 59 | 163 | 1 |
| 200 | 486.5 | 291.8 | 20679 | 116 | 186 | 2 |
| 225 | 521.8 | 291.8 | 22699 | 116 | 210 | 2 |
| 243 | 521.8 | 291.8 | 23862 | 116 | 228 | 2 |

## Holdout quality (within-repository temporal, 1 repeat(s) pooled)

| event | arm | F1 | F1 per repeat | precision | recall | high-risk recall | clean acc. | instr. tokens/call | tokens/PR | failure |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | governed | 0.3508 |  | 0.3448 | 0.3571 | 0.36 | 0.8623 | 1056.79 | 24267.33 | 0.0 |
| 0 | naive | 0.3508 |  | 0.3448 | 0.3571 | 0.36 | 0.8623 | 1056.79 | 24267.33 | 0.0 |
| 243 | governed | 0.2623 |  | 0.2424 | 0.2857 | 0.32 | 0.8204 | 837.12 | 22312.59 | 0.0046 |
| 243 | naive | 0.3063 |  | 0.3091 | 0.3036 | 0.36 | 0.8383 | 1840.59 | 27609.71 | 0.0091 |

## Governed minus naive (paired bootstrap, 95% CI)

| event | metric | delta | 95% CI |
|---:|---|---:|---|
| 243 | f1 | -0.044 | [-0.1276, 0.0377] |
| 243 | precision | -0.0667 | [-0.1569, 0.0101] |
| 243 | recall | -0.0179 | [-0.1111, 0.0747] |
| 243 | high_risk_recall | -0.04 | [-0.1818, 0.0909] |
| 243 | clean_accuracy | -0.0179 | [-0.0614, 0.0227] |
| 243 | average_instruction_tokens_per_call | -1003.47 | [-1089.61, -916.92] |
| 243 | average_total_tokens_per_pr | -5297.12 | [-7259.97, -3681.83] |

## Maintenance (governed)

- event 100: {"dream": {"clusters": 0, "evictions": {}, "live_items_before": 72, "live_items_after": 72, "live_tokens_before": 6112, "live_tokens_after": 6112, "consolidated": true}, "promotion": {"status": "released", "version": 2, "changes": {"promoted": ["K-5af1f6aa"]}}, "promotion_gate_failures": [], "dream_release": null, "dream_gate_failures": null, "live_tokens_before": 6112, "live_tokens_after": 6112}
- event 200: {"dream": {"clusters": 0, "evictions": {"K-228513a9": "capacity", "K-b3f745e1": "capacity", "K-247df0a6": "capacity", "K-962fa994": "capacity", "K-f5a0a35d": "capacity", "K-fb44df5d": "capacity", "K-7c243108": "capacity", "K-7a9a1930": "capacity", "K-0f95dfb8": "capacity", "K-54f1e256": "capacity", "K-dc2e079e": "capacity", "K-04021aab": "capacity", "K-aa608edc": "capacity", "K-ba56012a": "capacity", "K-f58a1d75": "capacity", "K-5fb9f846": "capacity", "K-c6e9ad99": "capacity", "K-27784f63": "capacity", "K-ab4c0b87": "capacity", "K-be8a4120": "capacity", "K-40499fa5": "capacity", "K-688a2ee8": "capacity", "K-8c16615a": "capacity", "K-1279fd3a": "capacity", "K-291f29fa": "capacity", "K-6dd81d91": "capacity", "K-c565eb03": "capacity", "K-68ba849b": "capacity", "K-1ecc859b": "capacity", "K-1cc93d05": "capacity", "K-f89219da": "capacity", "K-f07163af": "capacity"}, "live_items_before": 130, "live_items_after": 98, "live_tokens_before": 13489, "live_tokens_after": 11700, "consolidated": true, "release": {"status": "released", "version": 3, "gate": {"passed": true, "failures": [], "margin": 0.02, "max_point_drop": 0.1, "dataset_fingerprint": "cf28cf398969a8e5ac09670a5d6d936f7669e4239c1d8551130a26a7bfd2442a", "cases": 73, "metrics": {"f1": {"baseline": 0.3704, "candidate": 0.3333, "delta": -0.0371, "ci95": [-0.1713, 0.089], "max_point_drop": 0.1, "passed": true}, "high_risk_recall": {"baseline": 0.4118, "candidate": 0.3529, "delta": -0.0589, "ci95": [-0.2308, 0.0], "max_point_drop": 0.1, "passed": true}, "clean_accuracy": {"baseline": 0.8542, "candidate": 0.8333, "delta": -0.0209, "ci95": [-0.1277, 0.0851], "max_point_drop": 0.1, "passed": true}}, "instruction_tokens_per_call": {"baseline": 983.3, "candidate": 965.93, "max_growth": 0.0, "passed": true}, "execution_success_rate": {"baseline": 1.0, "candidate": 1.0}, "baseline_snapshot": "dc2b86e593c4d2dcae5ce9da", "candidate_snapshot": "9f95738b51db93f58664fce7"}, "changes": {"updated": [], "archived": ["K-228513a9", "K-40499fa5", "K-f58a1d75", "K-8c16615a", "K-ba56012a", "K-7a9a1930", "K-c565eb03", "K-68ba849b", "K-1ecc859b", "K-27784f63", "K-b3f745e1", "K-0f95dfb8", "K-1279fd3a", "K-247df0a6", "K-54f1e256", "K-5fb9f846", "K-c6e9ad99", "K-291f29fa", "K-688a2ee8", "K-1cc93d05", "K-f5a0a35d", "K-dc2e079e", "K-f89219da", "K-04021aab", "K-f07163af", "K-962fa994", "K-aa608edc", "K-ab4c0b87", "K-be8a4120", "K-fb44df5d", "K-7c243108", "K-6dd81d91"], "created": []}}}, "promotion": {"status": "released", "version": 4, "changes": {"promoted": ["K-db90d837"]}}, "promotion_gate_failures": [], "dream_release": "released", "dream_gate_failures": [], "live_tokens_before": 13489, "live_tokens_after": 11700}
- event 243: {"dream": {"clusters": 0, "evictions": {"K-f220a712": "capacity", "K-73e92176": "capacity", "K-341a013b": "capacity", "K-6a50e024": "capacity", "K-a00f8264": "capacity", "K-e983d24c": "capacity", "K-15692fcc": "capacity", "K-d10cf422": "capacity", "K-56ec12ea": "capacity", "K-d4adcd7d": "capacity", "K-bffe54d2": "capacity", "K-b333b35f": "capacity", "K-af7f43aa": "capacity", "K-b12fef51": "capacity", "K-02d87541": "capacity", "K-a27114ac": "capacity", "K-d7f733bb": "capacity", "K-4e359cec": "capacity", "K-0a0eedc4": "capacity", "K-d6b8dfc7": "capacity", "K-3121b7a4": "capacity", "K-f1180922": "capacity", "K-89c76163": "capacity", "K-d664cad7": "capacity", "K-4c122d71": "capacity", "K-81f70a1a": "capacity", "K-cf06fa10": "capacity", "K-aae39b62": "capacity", "K-e1b122e4": "capacity"}, "live_items_before": 126, "live_items_after": 97, "live_tokens_before": 13814, "live_tokens_after": 12229, "consolidated": true, "release": {"status": "released", "version": 5, "gate": {"passed": true, "failures": [], "margin": 0.02, "max_point_drop": 0.1, "dataset_fingerprint": "cf28cf398969a8e5ac09670a5d6d936f7669e4239c1d8551130a26a7bfd2442a", "cases": 73, "metrics": {"f1": {"baseline": 0.2692, "candidate": 0.339, "delta": 0.0698, "ci95": [-0.0242, 0.1923], "max_point_drop": 0.1, "passed": true}, "high_risk_recall": {"baseline": 0.3529, "candidate": 0.3529, "delta": 0.0, "ci95": [0.0, 0.0], "max_point_drop": 0.1, "passed": true}, "clean_accuracy": {"baseline": 0.8333, "candidate": 0.7917, "delta": -0.0416, "ci95": [-0.1225, 0.0378], "max_point_drop": 0.1, "passed": true}}, "instruction_tokens_per_call": {"baseline": 982.19, "candidate": 924.56, "max_growth": 0.0, "passed": true}, "execution_success_rate": {"baseline": 1.0, "candidate": 1.0}, "baseline_snapshot": "0d5bf44c1aba0982eb99807b", "candidate_snapshot": "f9a57f9c16e2e46b286316c8"}, "changes": {"updated": [], "archived": ["K-e983d24c", "K-d664cad7", "K-4c122d71", "K-73e92176", "K-81f70a1a", "K-a00f8264", "K-4e359cec", "K-3121b7a4", "K-341a013b", "K-cf06fa10", "K-15692fcc", "K-f1180922", "K-aae39b62", "K-d10cf422", "K-d4adcd7d", "K-bffe54d2", "K-0a0eedc4", "K-56ec12ea", "K-e1b122e4", "K-b333b35f", "K-d6b8dfc7", "K-af7f43aa", "K-f220a712", "K-b12fef51", "K-02d87541", "K-89c76163", "K-6a50e024", "K-a27114ac", "K-d7f733bb"], "created": []}}}, "promotion": {}, "promotion_gate_failures": null, "dream_release": "released", "dream_gate_failures": [], "live_tokens_before": 13814, "live_tokens_after": 12229}
