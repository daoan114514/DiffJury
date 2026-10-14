"""Knowledge lifecycle: promotion out of probation and eviction to a cold tier.

Eviction is utility-driven rather than plain LRU: "recently used" for a
lesson means *recently cited in a finding*, not merely injected, and an item
is only judged after it has been exposed often enough for its counters to
mean something. Archived items leave every prompt but stay restorable (the
admission gate restores them when the same lesson is learned again), and a
hard per-role token capacity keeps the live set bounded no matter how much
feedback arrives.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .knowledge import KnowledgeItem, KnowledgeRepository, days_between, decay_factor, utc_now, utility


@dataclass
class LifecyclePolicy:
    promote_min_exposures: int = 5
    promote_min_utility: float = 0.45
    probation_max_exposures: int = 25
    evict_min_exposures: int = 30
    evict_max_utility: float = 0.30
    harmful_min: float = 3.0
    harmful_max_precision: float = 0.35
    idle_days: float = 60.0
    idle_min_exposures: int = 10
    capacity_tokens_per_role: int = 2400
    half_life_days: float = 30.0


@dataclass
class LifecycleReport:
    promotable: List[str] = field(default_factory=list)
    archived: Dict[str, str] = field(default_factory=dict)
    live_tokens_by_role: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "promotable": list(self.promotable), "archived": dict(self.archived),
            "live_tokens_by_role": dict(self.live_tokens_by_role),
        }


def precision(item: KnowledgeItem, now: str, half_life_days: float) -> float:
    factor = decay_factor(days_between(item.decayed_at, now), half_life_days)
    helpful, harmful = item.helpful * factor, item.harmful * factor
    return (helpful + 1.0) / (helpful + harmful + 2.0)


class KnowledgeLifecycle:
    def __init__(self, repository: KnowledgeRepository, policy: Optional[LifecyclePolicy] = None):
        self.repository = repository
        self.policy = policy or LifecyclePolicy()
        self.protect_pending_sources = False
        self.validation_status: Dict[str, str] = {}

    def eviction_reason(self, item: KnowledgeItem, now: str) -> str:
        """Why an item should leave the live set, or "" to keep it."""
        policy = self.policy
        if item.pinned or item.status == "archived":
            return ""
        factor = decay_factor(days_between(item.decayed_at, now), policy.half_life_days)
        if (
            item.harmful * factor >= policy.harmful_min
            and precision(item, now, policy.half_life_days) < policy.harmful_max_precision
        ):
            return "harmful"
        value = utility(item, now, policy.half_life_days)
        if item.status == "probation":
            if item.exposures >= policy.probation_max_exposures and value < policy.promote_min_utility:
                return "probation-failed"
            return ""
        if item.exposures >= policy.evict_min_exposures and value < policy.evict_max_utility:
            return "low-utility"
        last_signal = item.last_cited_at or item.created_at
        if (
            item.exposures >= policy.idle_min_exposures
            and days_between(last_signal, now) > policy.idle_days
        ):
            return "idle"
        return ""

    def promotable(self, item: KnowledgeItem, now: str) -> bool:
        policy = self.policy
        if (self.protect_pending_sources and item.source in {"false_positive", "missed"}
                and self.validation_status.get(item.id) != "passed"):
            return False
        return (
            item.status == "probation"
            and item.exposures >= policy.promote_min_exposures
            and utility(item, now, policy.half_life_days) >= policy.promote_min_utility
            and item.helpful >= item.harmful
        )

    def evaluate(self, tenant_id: str, now: str = "", apply: bool = True) -> LifecycleReport:
        """Archive what has rotted and list probation items ready for promotion."""
        now = now or utc_now()
        report = self.plan(self.repository.list(tenant_id, statuses=("active", "probation")), now)
        if apply:
            for item_id, reason in report.archived.items():
                self.repository.set_status(tenant_id, item_id, "archived", reason)
                self.repository.log_event(tenant_id, item_id, "evict", {"reason": reason})
        return report

    def plan(self, live: List[KnowledgeItem], now: str = "") -> LifecycleReport:
        """Pure decision over a list of live items; nothing is persisted."""
        now = now or utc_now()
        report = LifecycleReport()
        survivors = []
        for item in live:
            reason = self.eviction_reason(item, now)
            if reason:
                report.archived[item.id] = reason
            else:
                survivors.append(item)
                if self.promotable(item, now):
                    report.promotable.append(item.id)
        self._enforce_capacity(survivors, now, report)
        report.promotable = [
            item_id for item_id in report.promotable if item_id not in report.archived
        ]
        return report

    def _enforce_capacity(
        self, survivors: List[KnowledgeItem], now: str, report: LifecycleReport,
    ) -> None:
        """Bound live rule tokens per role; drop the least valuable first."""
        cap = self.policy.capacity_tokens_per_role
        by_role: Dict[str, List[KnowledgeItem]] = {}
        for item in survivors:
            if item.kind == "rule":
                by_role.setdefault(item.role, []).append(item)
        for role, items in by_role.items():
            total = sum(item.tokens for item in items)
            # Probation first, then lowest utility, then least recently cited.
            order = sorted(items, key=lambda item: (
                item.pinned, item.status == "active", self.validation_priority(item),
                utility(item, now, self.policy.half_life_days),
                item.last_cited_at or item.created_at,
            ))
            for item in order:
                if total <= cap or item.pinned:
                    break
                report.archived[item.id] = "capacity"
                total -= item.tokens
            report.live_tokens_by_role[role] = total

    def validation_priority(self, item: KnowledgeItem) -> int:
        """Pending source lessons outlive invalid/neutral trials under capacity pressure."""
        if not self.protect_pending_sources or item.source not in {"false_positive", "missed"}:
            return 0
        status = self.validation_status.get(item.id)
        return 2 if status == "passed" else 0 if status in {"neutral", "harmful"} else 1

    def promote(self, tenant_id: str, item_ids: List[str], reason: str = "") -> List[str]:
        promoted = []
        for item_id in item_ids:
            item = self.repository.get(tenant_id, item_id)
            if item is None or item.status != "probation":
                continue
            self.repository.set_status(tenant_id, item_id, "active")
            self.repository.log_event(tenant_id, item_id, "promote", {"reason": reason})
            promoted.append(item_id)
        return promoted
