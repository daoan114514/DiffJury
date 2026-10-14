import os
import tempfile
import unittest

from diffjury.admission import AdmissionGate
from diffjury.assembler import ContextAssembler
from diffjury.dream import DreamConsolidator
from diffjury.knowledge import KnowledgeItem
from diffjury.knowledge_service import KnowledgeGovernor
from diffjury.lifecycle import KnowledgeLifecycle, LifecyclePolicy
from diffjury.memory import MemoryManager
from diffjury.release_gate import ReleaseGate, snapshot_fingerprint
from diffjury.store import TaskStore


NOW = "2026-06-01T00:00:00+00:00"
EVIDENCE = [{"task_id": "t-1"}]


def rule(content, role="security", status="active", **extra):
    extra.setdefault("created_at", "2026-05-01T00:00:00+00:00")
    extra.setdefault("decayed_at", NOW)
    return KnowledgeItem(
        tenant_id="default", kind="rule", role=role, content=content,
        status=status, evidence=list(EVIDENCE), **extra,
    )


class FakeGate:
    def __init__(self, passed=True):
        self.passed = passed
        self.calls = []

    def evaluate(self, baseline, candidate, max_token_growth=0.0):
        self.calls.append((len(baseline), len(candidate), max_token_growth))
        return {"passed": self.passed, "failures": [] if self.passed else ["f1"]}


class LifecycleTestCase(unittest.TestCase):
    def setUp(self):
        handle, self.path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        self.store = TaskStore(self.path)
        self.repo = self.store.knowledge

    def tearDown(self):
        os.unlink(self.path)

    def governor(self, gate=None, client=None, **options):
        admission = AdmissionGate(self.repo, client)
        return KnowledgeGovernor(
            self.repo, ContextAssembler(self.repo), admission, client,
            background=False, release_gate=gate, **options,
        )


class LifecycleTests(LifecycleTestCase):
    def test_eviction_reasons(self):
        lifecycle = KnowledgeLifecycle(self.repo)
        useless = rule("Check things carefully.", exposures=40)
        harmful = rule("Flag every SQL string.", harmful=4.0, exposures=8, citations=4)
        idle = rule("Pin images.", exposures=12, citations=2, helpful=2.0,
                    last_cited_at="2026-01-01T00:00:00+00:00")
        useful = rule("Bind SQL parameters.", exposures=40, citations=12, helpful=10.0,
                      last_cited_at=NOW)
        pinned = rule("Report secrets.", exposures=40, pinned=True)
        self.assertEqual("low-utility", lifecycle.eviction_reason(useless, NOW))
        self.assertEqual("harmful", lifecycle.eviction_reason(harmful, NOW))
        self.assertEqual("idle", lifecycle.eviction_reason(idle, NOW))
        self.assertEqual("", lifecycle.eviction_reason(useful, NOW))
        self.assertEqual("", lifecycle.eviction_reason(pinned, NOW))

    def test_probation_promotes_on_evidence_and_fails_after_long_trial(self):
        lifecycle = KnowledgeLifecycle(self.repo)
        good = rule("Bind SQL parameters.", status="probation", exposures=6,
                    citations=3, helpful=3.0)
        silent = rule("Validate redirect hosts.", status="probation", exposures=30)
        self.assertTrue(lifecycle.promotable(good, NOW))
        self.assertFalse(lifecycle.promotable(silent, NOW))
        self.assertEqual("probation-failed", lifecycle.eviction_reason(silent, NOW))

    def test_capacity_archives_least_valuable_rules_first(self):
        policy = LifecyclePolicy(capacity_tokens_per_role=28)
        lifecycle = KnowledgeLifecycle(self.repo, policy)
        best = rule("Bind SQL parameters.", exposures=10, citations=8, helpful=8.0)
        weak = rule("Consider maybe reviewing SQL.", exposures=10)
        trial = rule("Escape LIKE wildcards.", status="probation")
        report = lifecycle.plan([best, weak, trial], NOW)
        self.assertNotIn(best.id, report.archived)
        self.assertEqual("capacity", report.archived.get(trial.id))
        self.assertLessEqual(report.live_tokens_by_role["security"], 28)


class DreamTests(LifecycleTestCase):
    def test_deterministic_dream_merges_only_near_identical_items(self):
        first = self.repo.create(rule("Parameterize SQL built with f-strings in execute.",
                                      helpful=3.0, citations=3, exposures=5))
        second = self.repo.create(rule("Parameterize SQL built with f-strings in execute calls.",
                                       helpful=1.0, citations=1, exposures=4))
        distinct = self.repo.create(rule("Never log OAuth refresh tokens."))
        dream = DreamConsolidator(self.repo, KnowledgeLifecycle(self.repo))
        proposal = dream.propose("default", NOW)
        by_id = {item.id: item for item in proposal.candidate}
        self.assertEqual(1, len(proposal.merges))
        self.assertEqual("active", by_id[first.id].status)
        self.assertEqual("merged-into:%s" % first.id, by_id[second.id].archived_reason)
        self.assertEqual(4.0, by_id[first.id].helpful)
        self.assertEqual(9, by_id[first.id].exposures)
        self.assertEqual("active", by_id[distinct.id].status)
        # A proposal is not persisted until released.
        self.assertEqual("active", self.repo.get("default", second.id).status)

    def test_llm_dream_respects_token_cap_and_pins(self):
        pinned = self.repo.create(rule("Report hard-coded secrets in config.", pinned=True))
        loose = self.repo.create(rule("Report secrets hard-coded in config files."))

        class Client:
            def complete_json(self, role, system, user, ledger=None, max_tokens=None):
                return {
                    "merges": [{"keep_id": loose.id, "absorb_ids": [pinned.id],
                                "content": "x " * 200}],
                    "drop": [{"id": pinned.id, "reason": "duplicate"}],
                }

        dream = DreamConsolidator(self.repo, KnowledgeLifecycle(self.repo), Client())
        proposal = dream.propose("default", NOW)
        self.assertEqual([], proposal.merges)
        self.assertEqual([], proposal.drops)
        self.assertFalse(proposal.changed)


class GovernorMaintenanceTests(LifecycleTestCase):
    def seed_duplicates(self):
        keep = self.repo.create(rule("Parameterize SQL built with f-strings in the execute call.",
                                     helpful=2.0, citations=2, exposures=3))
        dup = self.repo.create(rule("Parameterize SQL built using f-strings in execute call."))
        return keep, dup

    def test_gate_rejection_leaves_knowledge_untouched(self):
        keep, dup = self.seed_duplicates()
        result = self.governor(gate=FakeGate(passed=False)).maintain(
            "default", force_dream=True, now=NOW,
        )
        self.assertEqual("rejected", result["dream"]["release"]["status"])
        self.assertEqual("active", self.repo.get("default", dup.id).status)
        statuses = [item["status"] for item in self.repo.list_releases("default")]
        self.assertEqual(["rejected", "baseline"], statuses)

    def test_passed_dream_is_released_and_can_be_rolled_back(self):
        keep, dup = self.seed_duplicates()
        gate = FakeGate(passed=True)
        governor = self.governor(gate=gate)
        result = governor.maintain("default", force_dream=True, now=NOW)
        self.assertEqual("released", result["dream"]["release"]["status"])
        self.assertEqual(0.0, gate.calls[0][2])
        self.assertEqual("archived", self.repo.get("default", dup.id).status)
        self.assertLess(result["live_tokens_after"], result["live_tokens_before"])
        rolled = governor.rollback("default", 1)
        self.assertIn(dup.id, rolled["restored"])
        self.assertEqual("active", self.repo.get("default", dup.id).status)
        self.assertEqual(
            "Parameterize SQL built with f-strings in the execute call.",
            self.repo.get("default", keep.id).content,
        )

    def test_promotion_goes_through_the_gate_with_token_growth_allowed(self):
        trial = self.repo.create(rule("Escape LIKE wildcards in search queries.",
                                      status="probation", exposures=6, citations=3,
                                      helpful=3.0))
        gate = FakeGate(passed=True)
        result = self.governor(gate=gate).maintain("default", now=NOW)
        self.assertEqual([trial.id], result["promotion"]["changes"]["promoted"])
        self.assertIsNone(gate.calls[-1][2])
        self.assertEqual("active", self.repo.get("default", trial.id).status)

    def test_rejected_release_cannot_be_restored(self):
        self.seed_duplicates()
        governor = self.governor(gate=FakeGate(passed=False))
        governor.maintain("default", force_dream=True, now=NOW)
        with self.assertRaisesRegex(ValueError, "not a restorable"):
            governor.rollback("default", 2)

    def test_memory_governance_caps_episodes_per_repository(self):
        memory = MemoryManager(self.store)
        for index in range(5):
            memory.remember(
                "default", "org/repo", "episodic", "task_summary",
                "episode %d" % index, importance=index / 10,
            )
        result = memory.govern("default", episodic_ttl_days=0, episodic_cap=2)
        self.assertEqual(3, result["episodic"])
        remaining = self.store.list_agent_memories("default", "org/repo", ("episodic",), 10)
        self.assertEqual(["episode 4", "episode 3"], [item["content"] for item in remaining])


class ReleaseGateTests(unittest.TestCase):
    def test_gate_blocks_quality_regression_and_prompt_growth(self):
        cases = [{"id": "c%d" % index, "repository": "r", "pull_request": index,
                  "split": "validation", "diff": "", "expected_findings": []}
                 for index in range(20)]

        class Harnessless(ReleaseGate):
            def __init__(self, reports):
                super().__init__(cases, lambda *_: None, ContextAssembler(None),
                                 bootstrap_iterations=200)
                self.reports = reports

            def replay(self, items, name):
                return self.reports[name]

        def report(f1_hits, tokens, size=4):
            rows = [{
                "id": "c%d" % index, "tp": int(index < f1_hits), "fp": 0,
                "fn": int(index >= f1_hits), "expected": 1, "severity_hits": 0,
                "high_total": 0, "high_hits": 0, "clean_hit": False,
                "execution_success": True,
                "instruction_calls": 1, "instruction_tokens": tokens,
            } for index in range(size)]
            from diffjury.evaluation_v2 import ProductionEvaluationHarness
            totals = ProductionEvaluationHarness._empty_totals()
            for row in rows:
                ProductionEvaluationHarness._accumulate(totals, row)
            return {"case_results": rows, "metrics": ProductionEvaluationHarness._metrics(totals)}

        same = Harnessless({"knowledge-baseline": report(4, 100), "knowledge-candidate": report(4, 100)})
        self.assertTrue(same.evaluate([], [])["passed"])
        worse = Harnessless({"knowledge-baseline": report(4, 100), "knowledge-candidate": report(2, 100)})
        self.assertIn("f1", worse.evaluate([], [])["failures"])
        # A small drop with a wide interval is noise on a small replay set.
        noisy = Harnessless({
            "knowledge-baseline": report(10, 100, 20), "knowledge-candidate": report(9, 100, 20),
        })
        verdict = noisy.evaluate([], [])
        self.assertLess(verdict["metrics"]["f1"]["delta"], -0.02)
        self.assertTrue(verdict["passed"])
        longer = Harnessless({"knowledge-baseline": report(4, 100), "knowledge-candidate": report(4, 150)})
        self.assertIn("average_instruction_tokens_per_call", longer.evaluate([], [])["failures"])
        self.assertTrue(longer.evaluate([], [], max_token_growth=None)["passed"])

    def test_point_drop_tolerance_allows_one_case_on_small_denominators(self):
        cases = [{"id": "c", "repository": "r", "pull_request": 1, "split": "validation",
                  "diff": "", "expected_findings": []}]
        gate = ReleaseGate(cases, lambda *_: None, ContextAssembler(None), max_point_drop=0.10)
        self.assertAlmostEqual(0.25, gate.point_drop_tolerance("high_risk_recall", {"high_total": 4}), 6)
        self.assertAlmostEqual(0.10, gate.point_drop_tolerance("clean_accuracy", {"clean_cases": 12}), 6)
        self.assertAlmostEqual(0.10, gate.point_drop_tolerance("f1", {"tp": 3, "fn": 5}), 6)

    def test_snapshot_fingerprint_ignores_archived_items(self):
        live = rule("Bind SQL parameters.")
        archived = rule("Old lesson.", status="archived")
        self.assertEqual(snapshot_fingerprint([live]), snapshot_fingerprint([live, archived]))


if __name__ == "__main__":
    unittest.main()
