import os
import tempfile
import unittest

from diffjury.agentic_core import BoundedRole
from diffjury.config import Settings
from diffjury.knowledge import KnowledgeItem
from diffjury.metrics import Metrics, metrics
from diffjury.runtime import RuntimeCancelled, ToolRegistry
from diffjury.service import ReviewService
from diffjury.telemetry import ExecutionLedger


class OperationsTests(unittest.TestCase):
    def test_bounded_role_stops_before_the_next_model_call_when_cancelled(self):
        calls = []

        class Client:
            def complete_json(self, *args, **kwargs):
                calls.append(args)
                return {"action": "final"}

        role = BoundedRole(
            "security", "prompt", Client(), 4000, 60, cancel_check=lambda: True,
        )
        ledger = ExecutionLedger("agentic")
        with self.assertRaises(RuntimeCancelled):
            role.run("{}", ToolRegistry(), ledger)
        self.assertEqual([], calls)
        self.assertEqual("cancelled", ledger.agent_traces["security"][-1]["event"])

    def test_gauges_render_with_labels(self):
        registry = Metrics()
        registry.gauge("knowledge_items", 3, status="active", tenant="t")
        registry.gauge("knowledge_items", 1, status="archived", tenant="t")
        text = registry.prometheus()
        self.assertEqual(1, text.count("# TYPE diffjury_knowledge_items gauge"))
        self.assertIn('diffjury_knowledge_items{status="active",tenant="t"} 3.0', text)

    def test_service_publishes_knowledge_rot_gauges(self):
        handle, path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        try:
            service = ReviewService(Settings(
                host="127.0.0.1", port=8080, db_path=path, max_diff_bytes=10000,
                max_steps=8, timeout_seconds=10, llm_base_url="", llm_api_key="",
                llm_model="", github_webhook_secret="", github_token="",
                auto_post_review=False,
            ))
            service.store.knowledge.create(KnowledgeItem(
                tenant_id="default", kind="rule", role="security", status="active",
                content="Bind SQL parameters.", evidence=[{"task_id": "t"}],
            ))
            service.publish_knowledge_metrics()
            text = metrics.prometheus()
            self.assertIn('diffjury_knowledge_items{status="active",tenant="default"} 1.0', text)
            self.assertIn('diffjury_knowledge_active_rule_tokens{role="security",tenant="default"}', text)
            service.close()
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
