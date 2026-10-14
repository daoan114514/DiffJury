import hashlib
import uuid
from typing import Any, Dict, Optional

from .admission import AdmissionGate
from .agentic_core import AgenticReviewer
from .assembler import ContextAssembler
from .auth import AuthManager
from .config import Settings
from .context_manager import ContextManager
from .evaluation_harness import validate_case
from .evaluation_harness import load_replay_cases
from .knowledge import utility
from .knowledge_service import KnowledgeGovernor
from .lifecycle import KnowledgeLifecycle, LifecyclePolicy
from .release_gate import ReleaseGate
from .patching import SuggestionOnlyFixer, VerifiedPatchFixer
from .github import GitHubAppAuthenticator, GitHubClient
from .harness import ReviewHarness
from .metrics import metrics
from .llm import JsonChatClient
from .modes import RunMode
from .memory import MemoryManager
from .models import TaskState, TraceEvent
from .observability import AlertManager, Observability
from .postgres_store import create_store
from .report import to_markdown
from .reviewer import ReliabilityRuleReviewer, SecurityRuleReviewer
from .skills import SkillRegistry
from .store import utc_now
from .task_queue import PermanentTaskError, TaskQueue
from .rollout import ReleaseManager
from .verifier import RepairVerifier


# Release-lane name for knowledge rollouts: canary tasks also see probation items.
KNOWLEDGE_DEPLOYMENT = "knowledge"


class ReviewService:
    def __init__(self, settings: Settings):
        self.settings = settings
        settings.validate()
        self.llm_config = settings.resolved_llm()
        self.store = create_store(settings.database_url, settings.db_path)
        self.memory = MemoryManager(
            self.store, settings.memory_enabled, settings.memory_recall_limit,
            settings.memory_working_ttl_seconds,
        )
        self.context_manager = ContextManager(
            settings.agent_context_window_tokens,
            settings.agent_context_input_tokens,
            settings.context_diff_token_budget,
            settings.context_observation_token_budget,
            settings.context_recent_observations,
            settings.context_map_chunk_tokens,
        )
        self.observability = Observability(settings.otel_service_name, settings.otel_endpoint)
        self.registry = SkillRegistry(settings.skills_dir)
        self.registry.register(
            "security-review", SecurityRuleReviewer(),
            "1.0.0", "Security, injection and secret detection",
        )
        self.registry.register(
            "reliability-review", ReliabilityRuleReviewer(),
            "1.0.0", "Reliability and observability review",
        )
        self.registry.reload()
        self.chat_client = (
            JsonChatClient(
                str(self.llm_config["base_url"]), str(self.llm_config["api_key"]),
                str(self.llm_config["model"]), str(self.llm_config["provider"]),
                settings.timeout_seconds, dict(self.llm_config.get("headers") or {}),
            ) if self.llm_config else None
        )
        self.knowledge = self._build_knowledge()
        self.reviewer = self._build_agentic_reviewer()
        self.harness = ReviewHarness(
            self.store, self.reviewer, settings.max_steps, settings.timeout_seconds,
            observability=self.observability,
        )
        self.github = GitHubClient(settings.github_token)
        repair_verifier = RepairVerifier(
            settings.repair_test_command, settings.repair_verify_timeout_seconds
        )
        self.fixer = (
            VerifiedPatchFixer(self.chat_client, repair_verifier)
            if self.chat_client else SuggestionOnlyFixer()
        )
        self.auth = AuthManager(
            self.store, settings.auth_secret, settings.session_ttl_seconds,
            settings.bootstrap_admin_username, settings.bootstrap_admin_password,
            settings.default_tenant_id,
        )
        self.releases = ReleaseManager(self.store)
        self.alerts = AlertManager(
            self.store, settings.alert_failure_rate, settings.alert_min_samples
        )
        self.queue = TaskQueue(
            self._process_queued, settings.async_workers, settings.redis_url,
            settings.queue_max_attempts, settings.queue_lease_seconds,
            self._on_dead_letter,
        )

    def close(self) -> None:
        self.queue.close()
        self.knowledge.close()

    def _build_knowledge(self, client=None) -> KnowledgeGovernor:
        settings = self.settings
        client = client if client is not None else self.chat_client
        assembler = ContextAssembler(
            self.store.knowledge,
            rule_budget=settings.knowledge_rule_budget_tokens,
            skill_budget=settings.knowledge_skill_budget_tokens,
            skill_body_cap=settings.knowledge_skill_body_cap_tokens,
            catalog_limit=settings.knowledge_skill_catalog_limit,
            half_life_days=settings.knowledge_half_life_days,
        )
        admission = AdmissionGate(
            self.store.knowledge, client,
            max_rule_tokens=settings.knowledge_rule_max_tokens,
            max_skill_tokens=settings.knowledge_skill_body_cap_tokens,
            similarity_threshold=settings.knowledge_similarity_threshold,
            dup_threshold=settings.knowledge_duplicate_threshold,
        )
        lifecycle = KnowledgeLifecycle(self.store.knowledge, LifecyclePolicy(
            capacity_tokens_per_role=settings.knowledge_capacity_tokens_per_role,
            half_life_days=settings.knowledge_half_life_days,
        ))
        return KnowledgeGovernor(
            self.store.knowledge, assembler, admission, client,
            half_life_days=settings.knowledge_half_life_days,
            pin_disk_skills=settings.knowledge_pin_disk_skills,
            lifecycle=lifecycle,
            release_gate=self._build_release_gate(client, assembler),
            memory=self.memory,
            maintain_every_reviews=settings.knowledge_maintain_every_reviews,
            dream_token_threshold=settings.knowledge_dream_token_threshold,
            memory_policy={
                "episodic_ttl_days": settings.memory_episodic_ttl_days,
                "episodic_cap": settings.memory_episodic_cap,
                "semantic_cap": settings.memory_semantic_cap,
            },
        )

    def _build_release_gate(self, client, assembler):
        """Replay gate over labelled cases; disabled without a model or cases."""
        settings = self.settings
        if client is None or not settings.knowledge_gate_max_cases:
            return None
        splits = [item.strip() for item in settings.knowledge_gate_splits.split(",") if item.strip()]
        try:
            cases = load_replay_cases(settings.knowledge_gate_dataset, splits)
        except (OSError, ValueError):
            return None
        if not cases:
            return None
        # A deterministic, repository-stratified subset keeps each gate run bounded.
        cases = sorted(cases, key=lambda case: (case["id"]))[::max(
            1, len(cases) // settings.knowledge_gate_max_cases
        )][:settings.knowledge_gate_max_cases]
        return ReleaseGate.for_client(
            cases, client, assembler,
            token_budget=settings.agent_token_budget,
            time_budget_seconds=settings.agent_time_budget_seconds,
            margin=settings.knowledge_gate_margin,
            max_point_drop=settings.knowledge_gate_max_drop,
            workers=settings.knowledge_gate_workers,
        )

    def _knowledge_lane(self, task_id: str) -> str:
        """Deterministic share of tasks that also see probation knowledge."""
        bucket = int(hashlib.sha256(("knowledge:%s" % task_id).encode("utf-8")).hexdigest()[:8], 16) % 100
        return "canary" if bucket < self.settings.knowledge_canary_percent else "stable"

    def _build_agentic_reviewer(self) -> AgenticReviewer:
        enabled = {
            item.strip() for item in self.settings.enabled_agents.split(",") if item.strip()
        }
        unknown = enabled.difference({
            "lead", "security", "correctness-reliability", "critic"
        })
        if unknown:
            raise ValueError("unsupported enabled Agent role(s): %s" % ", ".join(sorted(unknown)))
        return AgenticReviewer(
            self.store, self.chat_client,
            self.settings.agent_token_budget,
            self.settings.agent_time_budget_seconds,
            self.settings.llm_input_cost_per_million,
            self.settings.llm_output_cost_per_million,
            enabled,
            self.registry.reviewers(),
            None,
            self.settings.repair_test_command,
            self.memory,
            self.context_manager,
            self._active_agent_skills,
            self.knowledge if self.settings.knowledge_enabled else None,
        )

    def _run_review(
        self, task_id: str, repository: str, pull_request: Optional[int],
        diff: str, tenant_id: str,
    ):
        return self.harness.run(task_id, repository, pull_request, diff, tenant_id)

    def reload_skills(self) -> list:
        self.registry.reload()
        skills = self.registry.list()
        self.reviewer = self._build_agentic_reviewer()
        self.harness = ReviewHarness(
            self.store, self.reviewer, self.settings.max_steps, self.settings.timeout_seconds,
            observability=self.observability,
        )
        return skills

    def _active_agent_skills(self, tenant_id: str) -> list:
        """Shipped skills plus learned skills that are live for the stable lane."""
        disk = self.registry.agent_skills()
        values = {skill.name: skill for skill in disk}
        if self.settings.knowledge_enabled:
            self.knowledge.sync_disk_skills(tenant_id, disk)
            for skill in self.knowledge.learned_skills(tenant_id):
                values.setdefault(skill.name, skill)
        return [values[name] for name in sorted(values)]

    def list_skills(self, tenant_id: str) -> list:
        scanners = [item for item in self.registry.list() if item.get("kind") == "scanner"]
        return scanners + [{
            "name": skill.name, "version": skill.version,
            "description": skill.description, "source": skill.source,
            "kind": "agent-skill", "sandboxed": False,
            "permissions": list(skill.allowed_tools),
            "content_sha256": skill.content_sha256,
            "resources": list(skill.resource_paths),
        } for skill in self._active_agent_skills(tenant_id)]

    def _validate_review(self, repository: str, diff: str) -> None:
        if not repository or len(repository) > 250:
            raise ValueError("repository is required and must be at most 250 characters")
        size = len(diff.encode("utf-8"))
        if size == 0:
            raise ValueError("diff is required")
        if size > self.settings.max_diff_bytes:
            raise ValueError("diff exceeds maximum size of %d bytes" % self.settings.max_diff_bytes)

    def _require_agentic_model(self) -> None:
        if self.chat_client is None:
            raise RuntimeError("agentic review requires a configured model")

    def _create_task(
        self, repository: str, diff: str, pull_request: Optional[int], source: str,
        tenant_id: str = "default", repository_root: str = "",
        enabled_agents: Optional[list] = None,
        enabled_skills: Optional[list] = None,
    ) -> str:
        task_id = str(uuid.uuid4())
        encoded = diff.encode("utf-8")
        assignment = self.releases.assignment(tenant_id, KNOWLEDGE_DEPLOYMENT, task_id)
        self.store.create(task_id, repository, pull_request, {
            "source": source, "diff_bytes": len(encoded), "diff_sha256": hashlib.sha256(encoded).hexdigest(),
            "release_lane": assignment["lane"], "shadow": assignment["shadow"],
            "knowledge_lane": self._knowledge_lane(task_id),
            "mode": RunMode.AGENTIC.value,
            "repository_root": repository_root,
            "enabled_agents": enabled_agents or [],
            "enabled_skills": enabled_skills or [],
        }, tenant_id)
        self.store.save_task_payload(task_id, diff)
        return task_id

    def _create_deferred_task(
        self, repository: str, pull_request: Optional[int], source: str,
        tenant_id: str, payload: Dict[str, Any],
    ) -> str:
        task_id = str(uuid.uuid4())
        assignment = self.releases.assignment(tenant_id, KNOWLEDGE_DEPLOYMENT, task_id)
        self.store.create(task_id, repository, pull_request, {
            "source": source, "diff_pending": True,
            "release_lane": assignment["lane"], "shadow": assignment["shadow"],
            "knowledge_lane": self._knowledge_lane(task_id),
            **payload,
            "mode": RunMode.AGENTIC.value,
        }, tenant_id)
        return task_id

    def create_review(
        self, repository: str, diff: str, pull_request: Optional[int] = None,
        source: str = "api", tenant_id: str = "default", mode: str = "",
        repository_root: str = "", enabled_agents: Optional[list] = None,
        enabled_skills: Optional[list] = None,
    ) -> Dict[str, Any]:
        self._validate_review(repository, diff)
        RunMode.parse(mode) if mode else None
        self._require_agentic_model()
        self._validate_repository_root(repository_root)
        self._validate_enabled_agents(enabled_agents)
        self._validate_enabled_skills(enabled_skills, tenant_id)
        self._authorize_repository(tenant_id, repository)
        task_id = self._create_task(
            repository, diff, pull_request, source, tenant_id,
            repository_root, enabled_agents, enabled_skills,
        )
        try:
            with self.observability.span(
                "review", task_id, task_id=task_id, tenant_id=tenant_id,
                repository=repository,
            ), metrics.timer("review_duration"):
                report = self._run_review(
                    task_id, repository, pull_request, diff, tenant_id
                )
            metrics.inc("reviews_total")
            lane = (self.store.get(task_id, tenant_id).get("input") or {}).get(
                "release_lane", "stable"
            )
            self.releases.observe(tenant_id, KNOWLEDGE_DEPLOYMENT, False, lane)
            return {"task_id": task_id, "state": "SUCCESS", "report": report.to_dict()}
        except Exception:
            task = self.store.get(task_id, tenant_id) or {}
            lane = (task.get("input") or {}).get("release_lane", "stable")
            self.releases.observe(tenant_id, KNOWLEDGE_DEPLOYMENT, True, lane)
            self.alerts.evaluate(tenant_id)
            raise

    def enqueue_review(
        self, repository: str, diff: str, pull_request: Optional[int] = None,
        source: str = "api", github_issue_url: str = "", installation_id: Optional[int] = None,
        tenant_id: str = "default", mode: str = "", repository_root: str = "",
        enabled_agents: Optional[list] = None,
        enabled_skills: Optional[list] = None,
    ) -> Dict[str, Any]:
        self._validate_review(repository, diff)
        RunMode.parse(mode) if mode else None
        self._require_agentic_model()
        self._validate_repository_root(repository_root)
        self._validate_enabled_agents(enabled_agents)
        self._validate_enabled_skills(enabled_skills, tenant_id)
        self._authorize_repository(tenant_id, repository)
        task_id = self._create_task(
            repository, diff, pull_request, source, tenant_id,
            repository_root, enabled_agents, enabled_skills,
        )
        self.queue.submit({
            "task_id": task_id, "repository": repository, "pull_request": pull_request,
            "github_issue_url": github_issue_url, "installation_id": installation_id,
            "tenant_id": tenant_id,
        }, message_id=task_id)
        metrics.inc("reviews_enqueued_total")
        return {"task_id": task_id, "state": "PENDING", "queue": self.queue.backend}

    def _process_queued(self, payload: Dict[str, Any]) -> None:
        task_id = payload["task_id"]
        task = self.store.get(task_id)
        if not task:
            raise PermanentTaskError("task record no longer exists")
        tenant_id = payload.get("tenant_id") or task.get("tenant_id") or "default"
        diff = self.store.get_task_payload(task_id)
        if diff is None and payload.get("diff_url"):
            client = (
                self.github_client_for_installation(payload.get("installation_id"))
                if payload.get("installation_id") else self.github
            )
            client.ensure_repository_access(payload["repository"])
            diff = client.fetch_diff(payload["diff_url"])
            self._validate_review(payload["repository"], diff)
            encoded = diff.encode("utf-8")
            self.store.save_task_payload(task_id, diff)
            self.store.update_task_input(task_id, {
                "diff_pending": False, "diff_bytes": len(encoded),
                "diff_sha256": hashlib.sha256(encoded).hexdigest(),
            })
        if diff is None:
            raise PermanentTaskError("task payload no longer exists")
        try:
            with self.observability.span(
                "review.async", task_id, task_id=task_id, tenant_id=tenant_id,
            ), metrics.timer("review_duration"):
                report = self._run_review(
                    task_id, payload["repository"], payload.get("pull_request"), diff,
                    tenant_id,
                )
            metrics.inc("reviews_total")
            lane = (task.get("input") or {}).get("release_lane", "stable")
            self.releases.observe(tenant_id, KNOWLEDGE_DEPLOYMENT, False, lane)
            if payload.get("github_issue_url") and self.settings.auto_post_review:
                client = self.github_client_for_installation(payload.get("installation_id"))
                client.upsert_comment(
                    payload["github_issue_url"], to_markdown(report.to_dict()),
                    "<!-- diffjury-review:%s -->" % task_id,
                )
        except Exception:
            metrics.inc("reviews_failed_total")
            lane = (task.get("input") or {}).get("release_lane", "stable")
            self.releases.observe(tenant_id, KNOWLEDGE_DEPLOYMENT, True, lane)
            self.alerts.evaluate(tenant_id)
            raise

    def _on_dead_letter(self, payload: Dict[str, Any], error: str) -> None:
        task_id = payload.get("task_id", "")
        tenant_id = payload.get("tenant_id", "default")
        task = self.store.get(task_id, tenant_id) if task_id else None
        if task and task.get("state") not in {
            TaskState.SUCCESS.value, TaskState.FAILED.value, TaskState.CANCELLED.value,
        }:
            step = max(
                [int(item.get("step", 0)) for item in task.get("trace", [])] or [0]
            ) + 1
            self.store.fail(
                task_id, error,
                TraceEvent(
                    step, TaskState.FAILED,
                    "Task entered the dead-letter queue: %s" % error, utc_now(),
                ),
            )
        self.store.create_alert(
            tenant_id, "dlq:%s" % (task_id or "unknown"), "critical",
            "Task %s entered the dead-letter queue: %s" % (task_id, error),
        )
        metrics.inc("dead_letters_total")

    def handle_github_pull_request(
        self, payload: Dict[str, Any], delivery_id: str,
        payload_sha256: str, tenant_id: str = "",
    ) -> Dict[str, Any]:
        installation_id = (payload.get("installation") or {}).get("id")
        tenant_id = tenant_id or (
            self.store.installation_tenant(installation_id) if installation_id else None
        ) or self.settings.default_tenant_id
        if not self.store.claim_webhook(
            delivery_id, tenant_id, "pull_request", payload_sha256
        ):
            existing = self.store.get_webhook(delivery_id) or {}
            return {
                "duplicate": True, "task_id": existing.get("task_id"),
                "state": "PENDING" if existing.get("task_id") else "ACCEPTED",
            }
        action = payload.get("action")
        if action not in {"opened", "reopened", "synchronize"}:
            self.store.complete_webhook(delivery_id, None)
            return {"ignored": True, "reason": "unsupported pull_request action: %s" % action}
        self._require_agentic_model()
        pull = payload.get("pull_request") or {}
        repository = (payload.get("repository") or {}).get("full_name", "")
        number = payload.get("number")
        diff_url = pull.get("diff_url")
        if not repository or not isinstance(number, int) or not diff_url:
            raise ValueError("invalid GitHub pull_request payload")
        self._authorize_repository(tenant_id, repository)
        task_id = self._create_deferred_task(
            repository, number, "github-webhook", tenant_id,
            {"diff_url": diff_url},
        )
        self.queue.submit({
            "task_id": task_id, "repository": repository, "pull_request": number,
            "github_issue_url": pull.get("issue_url", ""),
            "installation_id": installation_id, "tenant_id": tenant_id,
            "diff_url": diff_url,
        }, message_id=task_id)
        metrics.inc("reviews_enqueued_total")
        result = {"task_id": task_id, "state": "PENDING", "queue": self.queue.backend}
        self.store.complete_webhook(delivery_id, result["task_id"])
        result["will_post_to_github"] = self.settings.auto_post_review
        return result

    def github_client_for_installation(self, installation_id: Optional[int] = None) -> GitHubClient:
        if installation_id is None:
            return self.github
        if not self.settings.github_app_id or not self.settings.github_private_key_path:
            raise ValueError("GitHub App credentials are not configured")
        token = GitHubAppAuthenticator(
            self.settings.github_app_id, self.settings.github_private_key_path
        ).installation_token(installation_id)
        return GitHubClient(token)

    def create_fix(
        self, task_id: str, installation_id: Optional[int] = None,
        tenant_id: Optional[str] = None,
    ) -> dict:
        task = self.store.get(task_id, tenant_id)
        if not task or not task.get("report"):
            raise ValueError("completed task not found")
        if task.get("pull_request") is None:
            raise ValueError("fix commits require a GitHub pull request task")
        actual_tenant = task.get("tenant_id") or tenant_id or "default"
        if not self.store.repository_allowed(actual_tenant, task["repository"], True):
            raise PermissionError("automatic repair is not enabled for this repository")
        result = self.fixer.create_fix_commits(
            self.github_client_for_installation(installation_id),
            task["repository"], task["pull_request"], task["report"],
        )
        metrics.inc("fix_runs_total")
        return result

    def record_feedback(
        self, task_id: str, category: str, finding: Optional[dict], note: str,
        tenant_id: Optional[str] = None,
    ) -> dict:
        task = self.store.get(task_id, tenant_id)
        if not task:
            raise ValueError("task not found")
        if task.get("state") != "SUCCESS" or not task.get("report"):
            raise ValueError("feedback requires a completed review task")
        if category not in {"false_positive", "missed_issue", "bad_fix", "accepted"}:
            raise ValueError("unsupported feedback category")
        actual_tenant = task.get("tenant_id") or tenant_id or "default"
        self.store.record_failure_case(task_id, category, {"finding": finding, "note": note[:2000]})
        self.memory.remember_feedback(
            actual_tenant, task["repository"], task_id, category, finding, note[:2000],
        )
        knowledge = {}
        if self.settings.knowledge_enabled:
            knowledge = self.knowledge.record_feedback(
                actual_tenant, task_id, category, finding, note[:2000],
                self.store.get_task_payload(task_id) or "", task["repository"],
            )
        metrics.inc("feedback_total")
        return {"recorded": True, "category": category, "knowledge": knowledge}

    def add_evaluation_case(
        self, name: str, diff: str, expected: list, split: str = "validation",
    ) -> dict:
        name, split = name.strip(), split.strip().lower()
        if not name or len(name) > 120:
            raise ValueError("evaluation case name is required and must be at most 120 characters")
        validate_case({
            "id": name, "repository": "api", "pull_request": 0, "split": split,
            "diff": diff, "expected_findings": expected,
        })
        return self.store.save_evaluation_case(name, split, diff, expected, "api", True)

    def maintain_knowledge(self, tenant_id: str, force_dream: bool = False) -> dict:
        """Queue one maintenance cycle; the replay gate can take minutes."""
        return {"queued": self.knowledge.submit(
            lambda: self.knowledge.maintain(tenant_id, force_dream=force_dream)
        )}

    def knowledge_stats(self, tenant_id: str) -> dict:
        settings = self.settings
        stats = self.store.knowledge.stats(tenant_id)
        live_by_role: Dict[str, int] = {}
        for item in self.store.knowledge.list(tenant_id, statuses=("active",), kind="rule"):
            live_by_role[item.role] = live_by_role.get(item.role, 0) + item.tokens
        stats["active_rule_tokens_by_role"] = live_by_role
        stats["budgets"] = {
            "rule_budget_tokens": settings.knowledge_rule_budget_tokens,
            "skill_budget_tokens": settings.knowledge_skill_budget_tokens,
            "skill_body_cap_tokens": settings.knowledge_skill_body_cap_tokens,
            "skill_catalog_limit": settings.knowledge_skill_catalog_limit,
            "rule_max_tokens": settings.knowledge_rule_max_tokens,
            "capacity_tokens_per_role": settings.knowledge_capacity_tokens_per_role,
        }
        return stats

    def publish_knowledge_metrics(self, tenant_id: str = "") -> None:
        """Refresh knowledge-rot gauges for a Prometheus scrape."""
        tenant_id = tenant_id or self.settings.default_tenant_id
        stats = self.knowledge_stats(tenant_id)
        for status, count in stats["by_status"].items():
            metrics.gauge("knowledge_items", count, tenant=tenant_id, status=status)
        metrics.gauge("knowledge_live_tokens", stats["live_tokens"], tenant=tenant_id)
        for role, tokens in stats["active_rule_tokens_by_role"].items():
            metrics.gauge("knowledge_active_rule_tokens", tokens, tenant=tenant_id, role=role)
        metrics.gauge(
            "knowledge_rule_budget_tokens", stats["budgets"]["rule_budget_tokens"], tenant=tenant_id,
        )

    def knowledge_items(self, tenant_id: str, statuses=(), kind: str = "") -> list:
        now = utc_now()
        return [
            {**item.to_dict(), "utility": utility(item, now, self.settings.knowledge_half_life_days)}
            for item in self.store.knowledge.list(tenant_id, statuses=statuses, kind=kind)
        ]

    def update_knowledge_item(
        self, tenant_id: str, item_id: str, action: str, actor: str = "",
    ):
        repository = self.store.knowledge
        item = repository.get(tenant_id, item_id)
        if item is None:
            return None
        if action in {"pin", "unpin"}:
            item.pinned = action == "pin"
            item = repository.save(item)
        elif action == "archive":
            item = repository.set_status(tenant_id, item_id, "archived", "manual")
        elif action == "restore":
            item = repository.set_status(tenant_id, item_id, "probation")
        else:
            raise ValueError("unsupported knowledge action")
        repository.log_event(tenant_id, item_id, action, {"actor": actor})
        self.store.audit(tenant_id, actor or "system", "knowledge.%s" % action, item_id, {})
        return item

    def resume_task(self, task_id: str, tenant_id: Optional[str] = None) -> dict:
        task = self.store.get(task_id, tenant_id)
        if not task:
            raise ValueError("task not found")
        if task["state"] == "SUCCESS":
            return {"task_id": task_id, "state": "SUCCESS", "report": task["report"]}
        diff = self.store.get_task_payload(task_id)
        if diff is None:
            raise ValueError("task payload is no longer available")
        self.queue.submit({
            "task_id": task_id, "repository": task["repository"],
            "pull_request": task.get("pull_request"),
            "tenant_id": task.get("tenant_id", "default"),
        }, message_id=task_id)
        return {"task_id": task_id, "state": "PENDING", "resumed": True}

    def cancel_task(self, task_id: str, tenant_id: Optional[str] = None) -> bool:
        return self.store.request_cancel(task_id, tenant_id)

    def _authorize_repository(self, tenant_id: str, repository: str) -> None:
        if not self.store.repository_allowed(tenant_id, repository):
            raise PermissionError("repository is not authorized for this tenant")

    @staticmethod
    def _validate_repository_root(repository_root: str) -> None:
        if not repository_root:
            return
        import os
        if not os.path.isabs(repository_root) or not os.path.isdir(repository_root):
            raise ValueError("repository_root must be an existing absolute directory")

    @staticmethod
    def _validate_enabled_agents(enabled_agents: Optional[list]) -> None:
        if enabled_agents is None:
            return
        allowed = {"lead", "security", "correctness-reliability", "critic"}
        unknown = set(enabled_agents).difference(allowed)
        if unknown:
            raise ValueError("unsupported enabled Agent role(s): %s" % ", ".join(sorted(unknown)))

    def _validate_enabled_skills(
        self, enabled_skills: Optional[list], tenant_id: str,
    ) -> None:
        if enabled_skills is None:
            return
        if not all(isinstance(item, str) for item in enabled_skills):
            raise ValueError("enabled_skills must contain Agent Skill names")
        available = {skill.name for skill in self._active_agent_skills(tenant_id)}
        unknown = set(enabled_skills).difference(available)
        if unknown:
            raise ValueError(
                "unknown enabled Agent Skill(s): %s" % ", ".join(sorted(unknown))
            )
