import json
import os
import tempfile
import unittest

from diffjury.admission import AdmissionGate
from diffjury.agentic_core import AgenticReviewer
from diffjury.assembler import ContextAssembler, KnowledgeView, StaticKnowledge, TaskSignals
from diffjury.diff_parser import parse_unified_diff
from diffjury.knowledge import KnowledgeItem, utility
from diffjury.knowledge_service import KnowledgeGovernor
from diffjury.models import Finding, Severity
from diffjury.similarity import TfidfIndex, fingerprint, tokenize
from diffjury.store import TaskStore


EVIDENCE = [{"task_id": "t-1", "category": "missed_issue"}]
SQL_DIFF = (
    "--- a/app/db.py\n+++ b/app/db.py\n@@ -1 +1,2 @@\n-old\n"
    "+query = f\"SELECT * FROM users WHERE id={user_id}\"\n+cursor.execute(query)\n"
)


def rule(content, role="security", status="active", **extra):
    return KnowledgeItem(
        tenant_id="default", kind="rule", role=role, content=content,
        status=status, evidence=list(EVIDENCE), **extra,
    )


class JudgeClient:
    """Scripted admission judge returning one fixed verdict."""

    provider = "fake"
    model = "judge"

    def __init__(self, verdict):
        self.verdict = verdict
        self.calls = []

    def complete_json(self, role, system, user, ledger=None, max_tokens=None):
        self.calls.append((role, json.loads(user)))
        if ledger:
            ledger.record_model(role, self.provider, self.model, {}, 1)
        verdict = dict(self.verdict)
        if verdict.get("target_id") == "$first":
            verdict["target_id"] = json.loads(user)["similar_items"][0]["id"]
        return verdict


class KnowledgeTestCase(unittest.TestCase):
    def setUp(self):
        handle, self.path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        self.store = TaskStore(self.path)
        self.repo = self.store.knowledge

    def tearDown(self):
        os.unlink(self.path)


class SimilarityTests(unittest.TestCase):
    def test_tokenizer_splits_identifiers_and_drops_stopwords(self):
        self.assertEqual(
            ["parameterize", "sql", "cursor", "execute", "user", "id"],
            tokenize("Parameterize SQL in cursor.execute with userId"),
        )

    def test_fingerprint_ignores_word_order_and_case(self):
        self.assertEqual(
            fingerprint("Validate redirect URL hosts"),
            fingerprint("hosts URL redirect validate"),
        )

    def test_tfidf_ranks_the_closest_lesson_first(self):
        index = TfidfIndex({
            "sql": "Parameterize SQL queries passed to cursor.execute",
            "log": "Never log access tokens or passwords",
        })
        self.assertEqual("sql", index.top_k("build SQL with f-string for cursor.execute")[0][0])


class AdmissionTests(KnowledgeTestCase):
    def test_new_lesson_enters_probation(self):
        decision = AdmissionGate(self.repo).admit("default", {
            "role": "security", "content": "Parameterize SQL built with f-strings.",
            "evidence": EVIDENCE,
        })
        self.assertEqual("ADD", decision.action)
        self.assertEqual("probation", self.repo.get("default", decision.item_id).status)

    def test_lesson_without_evidence_is_rejected(self):
        decision = AdmissionGate(self.repo).admit("default", {
            "role": "security", "content": "Be careful with SQL.", "evidence": [],
        })
        self.assertEqual("REJECT", decision.action)
        self.assertEqual([], self.repo.list("default"))

    def test_over_budget_rule_is_rejected_without_a_compressor(self):
        decision = AdmissionGate(self.repo, max_rule_tokens=20).admit("default", {
            "role": "security", "content": " ".join(["Check every SQL call site"] * 10),
            "evidence": EVIDENCE,
        })
        self.assertEqual("REJECT", decision.action)
        self.assertIn("exceeds 20 tokens", decision.reason)

    def test_exact_duplicate_upvotes_and_merges_evidence(self):
        gate = AdmissionGate(self.repo)
        first = gate.admit("default", {
            "role": "security", "content": "Parameterize SQL built with f-strings.",
            "evidence": EVIDENCE,
        })
        second = gate.admit("default", {
            "role": "security", "content": "parameterize SQL built with f-strings",
            "evidence": [{"task_id": "t-2"}],
        })
        self.assertEqual(("UPVOTE", first.item_id), (second.action, second.item_id))
        self.assertEqual(1, len(self.repo.list("default")))
        self.assertEqual(2, len(self.repo.get("default", first.item_id).evidence))

    def test_judge_merge_rewrites_target_within_cap(self):
        gate = AdmissionGate(self.repo, client=JudgeClient({
            "decision": "MERGE", "target_id": "$first",
            "merged_content": "Parameterize SQL built with f-strings or % formatting.",
        }))
        base = gate.admit("default", {
            "role": "security", "content": "Parameterize SQL built with f-strings.",
            "evidence": EVIDENCE,
        })
        merged = gate.admit("default", {
            "role": "security", "content": "Parameterize SQL assembled with % formatting.",
            "evidence": [{"task_id": "t-2"}],
        })
        item = self.repo.get("default", base.item_id)
        self.assertEqual(("MERGE", base.item_id), (merged.action, merged.item_id))
        self.assertEqual("Parameterize SQL built with f-strings or % formatting.", item.content)
        self.assertEqual(2, item.version)
        self.assertEqual(1, len(self.repo.list("default")))

    def test_merge_that_would_bloat_degrades_to_upvote(self):
        gate = AdmissionGate(self.repo, max_rule_tokens=30, client=JudgeClient({
            "decision": "MERGE", "target_id": "$first",
            "merged_content": "Parameterize SQL " + "and also check everything " * 20,
        }))
        base = gate.admit("default", {
            "role": "security", "content": "Parameterize SQL built with f-strings.",
            "evidence": EVIDENCE,
        })
        decision = gate.admit("default", {
            "role": "security", "content": "Parameterize SQL assembled with % formatting.",
            "evidence": [{"task_id": "t-2"}],
        })
        self.assertEqual("UPVOTE", decision.action)
        self.assertEqual(
            "Parameterize SQL built with f-strings.",
            self.repo.get("default", base.item_id).content,
        )

    def test_judge_screens_lessons_without_neighbours(self):
        client = JudgeClient({"decision": "REJECT", "reason": "generic advice"})
        decision = AdmissionGate(self.repo, client=client).admit("default", {
            "role": "security", "content": "Always review code carefully.",
            "evidence": EVIDENCE,
        })
        self.assertEqual("REJECT", decision.action)
        self.assertEqual([], client.calls[0][1]["similar_items"])
        self.assertEqual([], self.repo.list("default"))

    def test_relearning_an_archived_lesson_restores_it(self):
        archived = self.repo.create(rule(
            "Parameterize SQL built with f-strings in cursor.execute.", status="archived",
        ))
        decision = AdmissionGate(self.repo).admit("default", {
            "role": "security",
            "content": "Parameterize SQL built with f-strings in cursor.execute calls.",
            "evidence": [{"task_id": "t-9"}],
        })
        self.assertEqual(("RESTORE", archived.id), (decision.action, decision.item_id))
        self.assertEqual("probation", self.repo.get("default", archived.id).status)


class AssemblyTests(KnowledgeTestCase):
    def signals(self):
        return TaskSignals.from_review(SQL_DIFF, ["app/db.py"], "org/repo")

    def test_rules_fit_the_budget_and_prefer_relevant_items(self):
        relevant = self.repo.create(rule(
            "Parameterize SQL passed to cursor.execute.", triggers=["cursor.execute"],
        ))
        self.repo.create(rule("Pin container images by digest in Dockerfiles."))
        filler = [
            self.repo.create(rule("Parameterize SQL query %d before cursor.execute runs." % index))
            for index in range(30)
        ]
        assembler = ContextAssembler(self.repo, rule_budget=80)
        assembly = assembler.rules_for("security", self.signals(), assembler.live_items("default"))
        self.assertLessEqual(sum(item.tokens for item in assembly.items), 80)
        self.assertIn(relevant.id, assembly.ids)
        self.assertNotIn("Dockerfiles", assembly.text)
        self.assertTrue(set(assembly.dropped) & {item.id for item in filler})

    def test_pinned_items_are_kept_and_role_scope_is_respected(self):
        pinned = self.repo.create(rule("Report hard-coded credentials.", pinned=True))
        critic_only = self.repo.create(rule(
            "Reject SQL findings on constant strings.", role="critic", triggers=["cwe:CWE-89"],
        ))
        assembler = ContextAssembler(self.repo, rule_budget=600)
        items = assembler.live_items("default")
        security = assembler.rules_for("security", self.signals(), items)
        view = KnowledgeView(assembler, items, self.signals())
        self.assertEqual([], view.rules_for("lead").ids[1:])
        view.extend_signals("SEC-SQL CWE-89 SQL injection")
        critic = view.rules_for("critic")
        self.assertEqual(pinned.id, security.ids[0])
        self.assertNotIn(critic_only.id, security.ids)
        self.assertIn(critic_only.id, critic.ids)

    def test_language_triggers_filter_but_do_not_make_items_relevant(self):
        sql = self.repo.create(rule("Parameterize SQL passed to cursor.execute.", triggers=["lang:py"]))
        docker = self.repo.create(rule("Pin container base images by digest.", triggers=["lang:py"]))
        java_sql = self.repo.create(rule(
            "Parameterize SQL passed to cursor.execute.", triggers=["lang:java", "cursor.execute"],
        ))
        assembler = ContextAssembler(self.repo)
        ids = assembler.rules_for("security", self.signals(), assembler.live_items("default")).ids
        self.assertIn(sql.id, ids)
        self.assertNotIn(docker.id, ids)
        self.assertNotIn(java_sql.id, ids)

    def test_probation_items_are_only_visible_in_the_canary_lane(self):
        probation = self.repo.create(rule("Parameterize SQL passed to cursor.execute.", status="probation"))
        assembler = ContextAssembler(self.repo)
        self.assertNotIn(probation.id, [item.id for item in assembler.live_items("default")])
        self.assertIn(probation.id, [item.id for item in assembler.live_items("default", "canary")])

    def test_view_resolves_only_exposed_citations(self):
        item = self.repo.create(rule("Parameterize SQL passed to cursor.execute."))
        assembler = ContextAssembler(self.repo)
        view = KnowledgeView(assembler, assembler.live_items("default"), self.signals())
        view.rules_for("security")
        self.assertEqual([item.id], view.resolve_citations([item.id, "K-deadbeef"], "security"))
        self.assertEqual([], view.resolve_citations([item.id], "critic"))


class AttributionTests(KnowledgeTestCase):
    def governor(self):
        assembler = ContextAssembler(self.repo)
        return KnowledgeGovernor(
            self.repo, assembler, AdmissionGate(self.repo), background=False,
        )

    def finding(self, line, applied):
        return Finding(
            "SEC-SQL", Severity.HIGH, "SQL injection", "explained", "app/db.py", line,
            "cursor.execute(query)", "fix", "test", 0.9, cwe="CWE-89",
            applied_rules=list(applied),
        )

    def test_published_citation_is_helpful_and_rejected_is_harmful(self):
        helpful = self.repo.create(rule("Parameterize SQL passed to cursor.execute."))
        harmful = self.repo.create(rule("Report every query passed to cursor.execute as injection."))
        idle = self.repo.create(rule("Check user_id values in SQL queries are integers."))
        governor = self.governor()
        view = governor.prepare("default", "org/repo", "stable", TaskSignals.from_review(
            SQL_DIFF, ["app/db.py"], "org/repo",
        ))
        view.rules_for("security")
        published = self.finding(2, [helpful.id])
        rejected = self.finding(1, [harmful.id])
        governor.record_review("default", "task-1", view, [published, rejected], [published])
        helpful_after = self.repo.get("default", helpful.id)
        harmful_after = self.repo.get("default", harmful.id)
        idle_after = self.repo.get("default", idle.id)
        self.assertEqual((1.0, 0.0, 1, 1), (
            helpful_after.helpful, helpful_after.harmful,
            helpful_after.exposures, helpful_after.citations,
        ))
        self.assertEqual((0.0, 0.5), (harmful_after.helpful, harmful_after.harmful))
        self.assertEqual((1, 0), (idle_after.exposures, idle_after.citations))
        self.assertGreater(utility(helpful_after), utility(harmful_after))

    def test_uncited_but_matching_items_earn_implicit_credit(self):
        matching = self.repo.create(rule(
            "Parameterize SQL passed to cursor.execute.", triggers=["cwe:CWE-89"],
        ))
        untriggered = self.repo.create(rule("Parameterize SQL passed to cursor.execute calls."))
        governor = self.governor()
        view = governor.prepare("default", "org/repo", "stable", TaskSignals.from_review(
            SQL_DIFF, ["app/db.py"], "org/repo",
        ))
        view.rules_for("security")
        published = self.finding(2, [])
        published.source = "security"
        rejected = self.finding(1, [])
        rejected.source = "security"
        governor.record_review("default", "task-1", view, [published, rejected], [published])
        matched = self.repo.get("default", matching.id)
        self.assertEqual((0.5, 0.0, 1), (matched.helpful, matched.harmful, matched.citations))
        plain = self.repo.get("default", untriggered.id)
        self.assertEqual((0.0, 0, 1), (plain.helpful, plain.citations, plain.exposures))

    def test_human_false_positive_blames_cited_items_and_teaches_the_critic(self):
        cited = self.repo.create(rule("Report every SQL string as injection."))
        governor = self.governor()
        result = governor.record_feedback(
            "default", "task-1", "false_positive",
            {"path": "app/db.py", "line": 2, "cwe": "CWE-89", "applied_rules": [cited.id]},
            "The query only interpolates a constant table name.",
        )
        self.assertEqual([cited.id], result["credited"])
        self.assertEqual(2.0, self.repo.get("default", cited.id).harmful)
        lessons = [item for item in self.repo.list("default") if item.id != cited.id]
        self.assertEqual(["critic"], [item.role for item in lessons])
        self.assertIn("cwe:cwe-89", lessons[0].triggers)

    def test_utility_decays_old_evidence(self):
        item = rule("Parameterize SQL.", helpful=8.0, decayed_at="2026-01-01T00:00:00+00:00")
        fresh = utility(item, "2026-01-01T00:00:00+00:00", half_life_days=30)
        stale = utility(item, "2026-04-01T00:00:00+00:00", half_life_days=30)
        self.assertGreater(fresh, stale)


class AgenticKnowledgeTests(unittest.TestCase):
    def test_knowledge_reaches_prompts_within_budget_and_citations_are_kept(self):
        item = rule("Parameterize SQL passed to cursor.execute.")
        seen = {}

        class Client:
            provider, model = "fake", "fake"

            def complete_json(self, role, system, user, ledger=None, max_tokens=None):
                if ledger:
                    ledger.record_model(role, "fake", "fake", {}, 1)
                seen.setdefault(role, system)
                task = json.loads(json.loads(user)["task"])
                if role == "lead":
                    if task["phase"] == "delegate":
                        return {"action": "final", "delegations": [], "risk_level": "normal"}
                    return {"action": "final", "accepted_finding_indices": [0]}
                if role == "security":
                    return {"action": "final", "findings": [{
                        "cwe": "CWE-89", "rule_id": "SEC-SQL", "severity": "medium",
                        "title": "SQL injection", "explanation": "User id reaches SQL.",
                        "path": "app/db.py", "line": 1,
                        "evidence": "query = f\"SELECT * FROM users WHERE id={user_id}\"",
                        "fix": "Bind parameters.", "test": "Inject a quote.",
                        "confidence": 0.9, "applied_rules": [item.id, "K-00000000"],
                    }]}
                if role == "critic":
                    return {"action": "final", "decisions": [
                        {"finding_index": 0, "accepted": True}
                    ]}
                return {"action": "final", "findings": []}

        class Store:
            def get(self, *_args):
                return {"input": {}}

        knowledge = StaticKnowledge(ContextAssembler(None, rule_budget=200), [item])
        reviewer = AgenticReviewer(Store(), Client(), knowledge=knowledge)
        findings = reviewer.review_with_context(
            "task", SQL_DIFF, parse_unified_diff(SQL_DIFF), repository="org/repo",
        )
        summary = reviewer.collaboration_summary("task")
        self.assertIn("[%s]" % item.id, seen["security"])
        self.assertEqual([[item.id]], [finding.applied_rules for finding in findings])
        log = summary["execution"]["instruction_log"]
        self.assertTrue(all(entry["instruction_tokens"] > 0 for entry in log))
        self.assertIn(item.id, summary["knowledge"]["exposed"])


if __name__ == "__main__":
    unittest.main()
