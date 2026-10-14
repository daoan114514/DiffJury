import json
import os
import tempfile
import unittest

from diffjury.config import Settings
from diffjury.diff_parser import parse_unified_diff
from diffjury.service import ReviewService
from diffjury.evaluation_v2 import ReplayReviewer
from diffjury.knowledge import KnowledgeItem
from diffjury.skills import AgentSkill, SkillRegistry
from diffjury.store import TaskStore


RISK_DIFF = "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+dangerous_call(data)\n"
CLEAN_DIFF = "--- a/b.py\n+++ b/b.py\n@@ -1 +1 @@\n-old\n+safe_call(data)\n"


def skill_markdown(name="review-dangerous-calls", include_rule=True):
    guidance = (
        "\n## Confirmed SEC-DANGEROUS-CALL guidance\n\n"
        "Inspect added behavior equivalent to `dangerous_call(data)`.\n"
        "Report `SEC-DANGEROUS-CALL` at `high` severity when context confirms it.\n"
        if include_rule else ""
    )
    return """---
name: %s
description: Review added code for confirmed project-specific dangerous calls.
---

# Review dangerous calls

Use changed-line evidence and report only actionable defects.%s
""" % (name, guidance)


def skill(name="review-dangerous-calls", include_rule=True):
    return AgentSkill.from_markdown(skill_markdown(name, include_rule), expected_name=name)


class SkillAwareClient:
    provider = "fake"
    model = "skill-aware"

    def complete_json(self, role, system, user, ledger=None, max_tokens=None):
        if ledger:
            ledger.record_model(role, self.provider, self.model, {
                "prompt_tokens": 10, "completion_tokens": 5,
            }, 1)
        managed = json.loads(user)
        task = json.loads(managed["task"])
        if role == "lead":
            if task["phase"] == "delegate":
                requested = task.get("requested_agent_skills") or []
                selected = requested or [
                    item["name"] for item in task.get("available_agent_skills") or []
                ]
                return {"action": "final", "delegations": [{
                    "assignment_id": "security-1", "worker": "security",
                    "objective": "Review project-specific dangerous calls",
                    "skills": selected,
                }], "risk_level": "normal"}
            if task["phase"] == "assess-workers":
                return {"action": "final", "revision_requests": [], "critic_objective": "Verify"}
            if task["phase"] == "finalize":
                return {
                    "action": "final",
                    "accepted_finding_indices": list(range(len(task["candidate_findings"]))),
                    "confidence_adjustments": [],
                }
        if role == "security":
            # Selected skill bodies are delivered in the system prompt.
            instructions = system.split("Active Agent Skills", 1)[-1] if (
                "Active Agent Skills" in system
            ) else ""
            rendered_diff = json.dumps(task["diff"], ensure_ascii=False)
            if "dangerous_call(data)" in instructions and "dangerous_call(data)" in rendered_diff:
                return {"action": "final", "findings": [{
                    "rule_id": "SEC-DANGEROUS-CALL", "severity": "high",
                    "title": "Dangerous call", "explanation": "Confirmed unsafe API was added.",
                    "path": "a.py", "line": 1, "evidence": "dangerous_call(data)",
                    "fix": "Use safe_call instead.", "test": "Add a regression test.",
                    "confidence": .9,
                    "call_chain": [{"path": "a.py", "line": 1, "symbol": "dangerous_call"}],
                }]}
            return {"action": "final", "findings": []}
        if role == "correctness-reliability":
            return {"action": "final", "findings": []}
        if role == "critic":
            return {"action": "final", "decisions": [{
                "finding_index": index, "accepted": True, "objections": [],
                "confidence_adjustment": 0.0,
            } for index, _item in enumerate(task["candidates"])]}
        raise AssertionError(role)


class AgentSkillTests(unittest.TestCase):
    def setUp(self):
        handle, self.path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        self.store = TaskStore(self.path)
        self.client = SkillAwareClient()

    def tearDown(self):
        os.unlink(self.path)

    def test_registry_loads_standard_skill_and_resources_without_python(self):
        with tempfile.TemporaryDirectory() as root:
            directory = os.path.join(root, "review-dangerous-calls")
            os.makedirs(os.path.join(directory, "references"))
            with open(os.path.join(directory, "SKILL.md"), "w", encoding="utf-8") as handle:
                handle.write(skill_markdown())
            with open(os.path.join(directory, "references", "policy.md"), "w", encoding="utf-8") as handle:
                handle.write("Use safe_call.\n")
            with open(os.path.join(directory, "skill.py"), "w", encoding="utf-8") as handle:
                handle.write("raise RuntimeError('must never be imported')\n")
            registry = SkillRegistry(root)
            registry.reload()
            loaded = registry.get_agent_skill("review-dangerous-calls")
            self.assertIsNotNone(loaded)
            self.assertEqual("Use safe_call.\n", loaded.read_resource("references/policy.md"))
            self.assertEqual("agent-skill", registry.list()[0]["kind"])
            os.unlink(os.path.join(directory, "SKILL.md"))
            registry.reload()
            self.assertIsNone(registry.get_agent_skill("review-dangerous-calls"))

    def test_bundled_agent_skills_are_discoverable(self):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills"))
        registry = SkillRegistry(root)
        registry.reload()
        self.assertEqual({
            "api-compatibility", "code-quality", "correctness-review",
            "database-review", "observability-review", "performance-review",
            "reliability-review", "security-review", "test-quality",
        }, {item["name"] for item in registry.catalog()})
        for item in registry.list():
            self.assertEqual("agent-skill", item["kind"])
            self.assertTrue(item["description"])

    def test_skill_md_requires_frontmatter_and_matching_directory_name(self):
        with self.assertRaisesRegex(ValueError, "frontmatter"):
            AgentSkill.from_markdown("# Missing metadata")
        with self.assertRaisesRegex(ValueError, "match"):
            AgentSkill.from_markdown(
                skill_markdown("review-dangerous-calls"),
                expected_name="different-name",
            )

    def test_agent_skill_body_is_selected_and_runs_through_worker_graph(self):
        reviewer = ReplayReviewer(
            self.client, name="skill-replay", skill_provider=lambda _tenant: [skill()],
        )
        findings, summary = reviewer.review_case_detailed(
            {"diff": RISK_DIFF, "repository": ""}, parse_unified_diff(RISK_DIFF),
        )
        self.assertEqual(["SEC-DANGEROUS-CALL"], [item.rule_id for item in findings])
        self.assertEqual(["review-dangerous-calls"], summary["collaboration"]["agent_skills"])
        security_calls = [
            item for item in summary["execution"]["instruction_log"]
            if item["role"] == "security"
        ]
        self.assertEqual(["review-dangerous-calls"], security_calls[0]["skills"])

    def test_learned_skills_are_tenant_isolated_and_used_in_review(self):
        learned = skill()
        with tempfile.TemporaryDirectory() as skills_dir:
            settings = Settings(
                host="127.0.0.1", port=8080, db_path=self.path,
                max_diff_bytes=10000, max_steps=8, timeout_seconds=10,
                llm_base_url="", llm_api_key="", llm_model="",
                github_webhook_secret="", github_token="", auto_post_review=False,
                skills_dir=skills_dir,
            )
            service = ReviewService(settings)
            service.store.knowledge.create(KnowledgeItem(
                tenant_id="tenant-a", kind="skill", name=learned.name,
                content=learned.description, body=learned.instructions,
                status="active", source="feedback", evidence=[{"task_id": "t"}],
            ))
            service.chat_client = self.client
            service.reviewer = service._build_agentic_reviewer()
            service.harness.reviewer = service.reviewer
            try:
                self.assertIn(
                    "review-dangerous-calls", {item["name"] for item in service.list_skills("tenant-a")}
                )
                self.assertNotIn(
                    "review-dangerous-calls", {item["name"] for item in service.list_skills("tenant-b")}
                )
                report = service.create_review(
                    "org/repo", RISK_DIFF, tenant_id="tenant-a",
                    enabled_skills=["review-dangerous-calls"],
                )["report"]
                self.assertIn(
                    "SEC-DANGEROUS-CALL", [item["rule_id"] for item in report["findings"]]
                )
            finally:
                service.queue.close()
                service.knowledge.close()

if __name__ == "__main__":
    unittest.main()
