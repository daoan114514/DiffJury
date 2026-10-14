import json
import os
import tempfile
import unittest

from diffjury.evaluation_harness import (
    EndToEndEvaluationHarness,
    dataset_fingerprint,
    load_jsonl,
    load_replay_cases,
    one_to_one_match,
)
from diffjury.models import Finding, Severity


class EndToEndEvaluationTests(unittest.TestCase):
    def test_replay_dataset_has_repository_level_split_and_expected_counts(self):
        cases = load_replay_cases()
        self.assertEqual(100, len(cases))
        self.assertEqual(40, sum(bool(item["expected_findings"]) for item in cases))
        self.assertEqual(60, sum(not item["expected_findings"] for item in cases))
        validation_repos = {
            item["repository"] for item in cases if item["split"] == "validation"
        }
        holdout_repos = {
            item["repository"] for item in cases if item["split"] == "holdout"
        }
        self.assertEqual(8, len(validation_repos))
        self.assertEqual(2, len(holdout_repos))
        self.assertFalse(validation_repos & holdout_repos)
        self.assertEqual(20, len(load_replay_cases(splits=["holdout"])))

    def test_one_to_one_matching_counts_duplicate_prediction_once(self):
        expected = [{
            "path": "src/a.py", "start_line": 10, "end_line": 12,
            "cwe": "CWE-95", "severity": "critical",
        }]
        predicted = [
            Finding(
                "SEC-EVAL", Severity.CRITICAL, "a", "long enough explanation",
                "src/a.py", line, "eval(x)", "replace eval safely",
                "add malicious input test", 0.9,
            )
            for line in (10, 11)
        ]
        matches = one_to_one_match(expected, predicted)
        self.assertEqual(1, len(matches))

    def test_one_to_one_matching_uses_explicit_cwe_not_random_rule_id(self):
        expected = [{
            "path": "src/a.py", "start_line": 10, "end_line": 10,
            "cwe": "CWE-95", "severity": "critical",
        }]
        predicted = [Finding(
            "MODEL-GENERATED-LABEL", Severity.CRITICAL, "a",
            "long enough explanation", "src/a.py", 10, "eval(x)",
            "replace eval safely", "add malicious input test", 0.9,
            cwe="CWE-95",
        )]
        self.assertEqual(1, len(one_to_one_match(expected, predicted)))

    def test_one_to_one_matching_rejects_unknown_rule_without_cwe(self):
        expected = [{
            "path": "src/a.py", "start_line": 10, "end_line": 10,
            "cwe": "CWE-95", "severity": "critical",
        }]
        predicted = [Finding(
            "MODEL-GENERATED-LABEL", Severity.CRITICAL, "a",
            "long enough explanation", "src/a.py", 10, "eval(x)",
            "replace eval safely", "add malicious input test", 0.9,
        )]
        self.assertEqual([], one_to_one_match(expected, predicted))

    def test_harness_scores_canonical_cwe_as_true_positive(self):
        case = {
            "id": "canonical-cwe", "repository": "demo/api", "pull_request": 1,
            "split": "validation",
            "diff": "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+eval(data)\n",
            "expected_findings": [{
                "path": "a.py", "start_line": 1, "end_line": 1,
                "cwe": "CWE-95", "severity": "critical",
            }],
        }

        class Reviewer:
            name = "test"

            def review(self, _diff, parsed):
                line = parsed.added_lines[0]
                return [Finding(
                    "MODEL-LABEL", Severity.CRITICAL, "x", "x",
                    line.path, line.line, line.content, "fix", "test", 0.9,
                    cwe="CWE-95",
                )]

        report = EndToEndEvaluationHarness().run(Reviewer(), [case])
        self.assertEqual(1, report["case_results"][0]["tp"])
        self.assertEqual(1.0, report["metrics"]["f1"])

    def test_neutral_labels_are_neither_required_nor_false_positives(self):
        diff = "--- a/a.py\n+++ b/a.py\n@@ -1 +1,3 @@\n-old\n+x = compute()\n+name = 'tmp'\n+eval(data)\n"
        case = {
            "id": "neutral", "repository": "demo/api", "pull_request": 1, "split": "holdout",
            "diff": diff,
            "expected_findings": [
                {"path": "a.py", "start_line": 3, "end_line": 3, "severity": "high"},
                {"path": "a.py", "start_line": 2, "end_line": 2, "severity": "low",
                 "neutral": True, "should_comment": False},
            ],
        }

        class Reviewer:
            name = "test"

            def review(self, _diff, parsed):
                return [Finding(
                    "R", Severity.HIGH, "x", "x", "a.py", line, "x", "fix", "test", 0.9,
                ) for line in (2, 3)]

        row = EndToEndEvaluationHarness().run(Reviewer(), [case])["case_results"][0]
        self.assertEqual((1, 0, 0, 1), (row["tp"], row["fp"], row["fn"], row["neutral_hits"]))

    def test_dataset_round_trip_has_stable_fingerprint(self):
        cases = load_replay_cases()
        handle, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(handle)
        try:
            with open(path, "w", encoding="utf-8") as output:
                for case in cases:
                    output.write(json.dumps(case, ensure_ascii=False) + "\n")
            loaded = load_jsonl(path)
            self.assertEqual(dataset_fingerprint(cases), dataset_fingerprint(loaded))
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
