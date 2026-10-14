import os
import tempfile
import time
import unittest

from diffjury.config import Settings
from diffjury.fixer import SafeFixer
from diffjury.service import ReviewService
from diffjury.store import TaskStore
from agentic_fake import enable_agentic_service


def settings(path):
    return Settings(
        host="127.0.0.1", port=8080, db_path=path, max_diff_bytes=10000,
        max_steps=8, timeout_seconds=10, llm_base_url="", llm_api_key="", llm_model="",
        github_webhook_secret="", github_token="", auto_post_review=False,
        skills_dir="skills",
    )


class AdvancedFeatureTests(unittest.TestCase):
    def setUp(self):
        handle, self.path = tempfile.mkstemp(suffix=".db")
        os.close(handle)

    def tearDown(self):
        os.unlink(self.path)

    def test_safe_fixer_changes_only_supported_rules(self):
        content = 'password = "secret-value"\nresult = eval(user_input)\nprint(result)\n'
        findings = [
            {"path": "app.py", "line": 1, "rule_id": "SEC-HARDCODED-SECRET"},
            {"path": "app.py", "line": 2, "rule_id": "SEC-EVAL"},
            {"path": "app.py", "line": 3, "rule_id": "REL-DEBUG-PRINT"},
        ]
        result = SafeFixer().apply(content, findings, "app.py")
        self.assertIn("import os", result["content"])
        self.assertRegex(result["content"], r"password = os\.environ\[['\"]PASSWORD['\"]\]")
        self.assertIn("eval(user_input)", result["content"])
        self.assertNotIn("print(result)", result["content"])
        self.assertEqual({"SEC-HARDCODED-SECRET", "REL-DEBUG-PRINT"}, set(result["rules"]))

    def test_deepseek_and_free_openrouter_provider_presets(self):
        deepseek = settings(self.path)
        deepseek = deepseek.__class__(
            **{**deepseek.__dict__, "llm_provider": "deepseek", "deepseek_api_key": "test-key"}
        )
        self.assertEqual("https://api.deepseek.com", deepseek.resolved_llm()["base_url"])
        self.assertEqual("deepseek-v4-flash", deepseek.resolved_llm()["model"])

        free = settings(self.path)
        free = free.__class__(
            **{
                **free.__dict__,
                "llm_provider": "openrouter-deepseek-free",
                "openrouter_api_key": "test-key",
            }
        )
        self.assertTrue(str(free.resolved_llm()["model"]).endswith(":free"))

    def test_evaluation_cases_are_immutable_and_idempotent(self):
        store = TaskStore(self.path)
        diff = "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+eval(data)\n"
        expected = [{"path": "a.py", "line": 1, "min_severity": "high"}]
        first = store.save_evaluation_case(
            "stable-case-v1", "validation", diff, expected, "test"
        )
        repeated = store.save_evaluation_case(
            "stable-case-v1", "validation", diff, expected, "test"
        )
        self.assertEqual(first["id"], repeated["id"])
        with self.assertRaisesRegex(ValueError, "immutable"):
            store.save_evaluation_case(
                "stable-case-v1", "validation", diff, [], "test"
            )

    def test_async_multi_agent_review(self):
        service = enable_agentic_service(ReviewService(settings(self.path)))
        diff = "--- a/a.py\n+++ b/a.py\n@@ -1 +1,2 @@\n-old\n+eval(data)\n+# TODO finish validation\n"
        result = service.enqueue_review("org/repo", diff, 2)
        task = None
        for _ in range(50):
            task = service.store.get(result["task_id"])
            if task["state"] in {"SUCCESS", "FAILED"}:
                break
            time.sleep(0.02)
        service.queue.close()
        self.assertEqual("SUCCESS", task["state"])
        self.assertEqual({"SEC-EVAL", "QUALITY-UNFINISHED"},
                         {item["rule_id"] for item in task["report"]["findings"]})


if __name__ == "__main__":
    unittest.main()
