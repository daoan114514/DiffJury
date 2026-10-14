"""Read-side budget: decide which knowledge each role actually sees.

The knowledge base may grow, but a role's instructions may not. For every
model call the assembler ranks live items by

    score = relevance(task) x utility(outcomes) / tokens

and packs them greedily into a fixed token budget (pinned items first).
Skills get the same treatment: the Lead sees a capped catalog of the most
relevant skills, and a worker's selected skill bodies share one budget.
Items that do not fit are reported as ``dropped`` so the rot dashboard can
show budget pressure instead of silently losing knowledge.
"""
import fnmatch
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .context_manager import estimate_tokens
from .knowledge import KnowledgeItem, KnowledgeRepository, utc_now, utility
from .similarity import TfidfIndex


CWE = re.compile(r"CWE-\d+", re.IGNORECASE)
RULE_HEADER = (
    "Learned review knowledge (verified lessons from past reviews; each is a hint, "
    "not proof). When a lesson shapes a finding, list its id in that finding's "
    "applied_rules."
)


@dataclass
class TaskSignals:
    """What the current review is about, extracted once per task."""

    text: str
    files: List[str] = field(default_factory=list)
    extensions: List[str] = field(default_factory=list)
    cwes: List[str] = field(default_factory=list)
    repository: str = ""

    @classmethod
    def from_review(
        cls, diff: str, files: Sequence[str], repository: str = "",
        extra: Iterable[str] = (),
    ) -> "TaskSignals":
        added = [line[1:] for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")]
        text = "\n".join(list(files) + added[:400] + list(extra))
        return cls(
            text=text[:40000], files=list(files),
            extensions=sorted({
                os.path.splitext(path)[1].lower().lstrip(".")
                for path in files if os.path.splitext(path)[1]
            }),
            cwes=sorted({item.upper() for item in CWE.findall(text)}),
            repository=repository,
        )


@dataclass
class Assembly:
    role: str
    items: List[KnowledgeItem]
    text: str
    tokens: int
    budget: int
    dropped: List[str] = field(default_factory=list)

    @property
    def ids(self) -> List[str]:
        return [item.id for item in self.items]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role, "ids": self.ids, "tokens": self.tokens,
            "budget": self.budget, "dropped": list(self.dropped),
        }


FILTER_PREFIXES = ("lang", "path")


def scope_allows(item: KnowledgeItem, signals: TaskSignals) -> bool:
    """``lang:``/``path:`` triggers narrow where an item applies; they never make it relevant."""
    filters = [trigger.partition(":") for trigger in item.triggers if trigger.partition(":")[0] in FILTER_PREFIXES]
    if not filters:
        return True
    for prefix, _, value in filters:
        if prefix == "lang" and value in signals.extensions:
            return True
        if prefix == "path" and any(fnmatch.fnmatch(path.lower(), value) for path in signals.files):
            return True
    return False


def finding_matches(item: KnowledgeItem, finding: Any) -> bool:
    """Whether a finding is about what an item's content triggers describe.

    Used for implicit attribution: models often apply a lesson without citing
    its id. Items without content triggers never match, so generic lessons
    cannot collect credit this way.
    """
    cwe = str(getattr(finding, "cwe", "") or "").upper()
    text = " ".join(str(getattr(finding, name, "") or "") for name in (
        "rule_id", "title", "explanation", "evidence",
    )).lower()
    for trigger in item.triggers:
        prefix, _, value = trigger.partition(":")
        if prefix in FILTER_PREFIXES:
            continue
        if prefix == "cwe":
            if value and value.upper() == cwe:
                return True
        elif len(trigger) >= 4 and trigger in text:
            return True
    return False


def trigger_score(item: KnowledgeItem, signals: TaskSignals) -> float:
    """1.0 when a content trigger (``cwe:`` or a keyword) fires, else 0.0."""
    lowered = signals.text.lower()
    for trigger in item.triggers:
        prefix, _, value = trigger.partition(":")
        if prefix in FILTER_PREFIXES:
            continue
        if prefix == "cwe":
            if value.upper() in signals.cwes:
                return 1.0
        elif trigger in lowered:
            return 1.0
    return 0.0


class ContextAssembler:
    def __init__(
        self, repository: Optional[KnowledgeRepository],
        rule_budget: int = 600, skill_budget: int = 3000,
        skill_body_cap: int = 1500, catalog_limit: int = 8,
        half_life_days: float = 30.0, include_probation: bool = False,
        min_relevance: float = 0.05,
    ):
        self.repository = repository
        self.rule_budget = max(0, int(rule_budget))
        self.skill_budget = max(0, int(skill_budget))
        self.skill_body_cap = max(64, int(skill_body_cap))
        self.catalog_limit = max(1, int(catalog_limit))
        self.half_life_days = half_life_days
        self.include_probation = include_probation
        self.min_relevance = min_relevance

    # ---------------------------------------------------------------- rules
    def live_items(self, tenant_id: str, lane: str = "stable") -> List[KnowledgeItem]:
        if self.repository is None:
            return []
        statuses = ["active"]
        if lane == "canary" or self.include_probation:
            statuses.append("probation")
        return self.repository.list(tenant_id, statuses=statuses)

    def rules_for(
        self, role: str, signals: TaskSignals, items: Sequence[KnowledgeItem],
        budget: Optional[int] = None,
    ) -> Assembly:
        budget = self.rule_budget if budget is None else max(0, int(budget))
        pool = [
            item for item in items
            if item.kind == "rule" and item.role in {"*", role}
            and item.scope in {"*", signals.repository}
        ]
        ranked = self._rank(pool, signals)
        chosen, dropped, used = [], [], 0
        for item, _score in ranked:
            if used + item.tokens <= budget:
                chosen.append(item)
                used += item.tokens
            else:
                dropped.append(item.id)
        if not chosen:
            return Assembly(role, [], "", 0, budget, dropped)
        text = RULE_HEADER + "\n" + "\n".join(
            "[%s] %s" % (item.id, item.content) for item in chosen
        )
        return Assembly(role, chosen, text, estimate_tokens(text), budget, dropped)

    def _rank(self, pool: Sequence[KnowledgeItem], signals: TaskSignals):
        if not pool:
            return []
        index = TfidfIndex({
            item.id: " ".join([item.name, item.content] + [
                trigger for trigger in item.triggers
                if trigger.partition(":")[0] not in FILTER_PREFIXES
            ])
            for item in pool
        })
        query = index.query(signals.text)
        now = utc_now()
        ranked = []
        for item in pool:
            lexical = sum(
                weight * index.vectors[item.id].get(term, 0.0)
                for term, weight in query.items()
            )
            if not item.pinned and not scope_allows(item, signals):
                continue
            relevance = max(trigger_score(item, signals), min(1.0, lexical * 2.0))
            if relevance < self.min_relevance and not item.pinned:
                continue
            value = utility(item, now, self.half_life_days)
            score = relevance * value / max(10, item.tokens)
            ranked.append((item, (0 if item.pinned else 1, -score, item.tokens, item.id)))
        ranked.sort(key=lambda pair: pair[1])
        return [(item, -key[1]) for item, key in ranked]

    # --------------------------------------------------------------- skills
    def skill_catalog(
        self, skills: Sequence[Any], signals: TaskSignals,
        stats: Optional[Dict[str, KnowledgeItem]] = None, requested: Iterable[str] = (),
    ) -> List[Any]:
        """Most relevant skills for the Lead's catalog; requested ones always stay."""
        requested = set(requested)
        if len(skills) <= self.catalog_limit:
            return list(skills)
        index = TfidfIndex({skill.name: "%s %s" % (skill.name, skill.description) for skill in skills})
        query = index.query(signals.text)
        now = utc_now()
        scored = []
        for skill in skills:
            lexical = sum(
                weight * index.vectors[skill.name].get(term, 0.0)
                for term, weight in query.items()
            )
            record = (stats or {}).get(skill.name)
            value = utility(record, now, self.half_life_days) if record else 0.375
            scored.append((skill.name in requested, lexical * value, skill.name, skill))
        scored.sort(key=lambda item: (not item[0], -item[1], item[2]))
        limit = max(self.catalog_limit, len(requested))
        return [item[3] for item in scored[:limit]]

    def skill_bodies(self, skills: Sequence[Any]) -> Dict[str, Any]:
        """Fit selected skill bodies into the worker's skill budget, in order."""
        chosen, dropped, used = [], [], 0
        for skill in skills:
            instructions = skill.instructions
            tokens = estimate_tokens(instructions)
            if tokens > self.skill_body_cap:
                dropped.append(skill.name)
                continue
            if used + tokens > self.skill_budget:
                dropped.append(skill.name)
                continue
            chosen.append(skill)
            used += tokens
        return {"skills": chosen, "tokens": used, "dropped": dropped}


class KnowledgeView:
    """One task's frozen view of knowledge; remembers what it exposed to whom."""

    def __init__(
        self, assembler: ContextAssembler, items: Sequence[KnowledgeItem],
        signals: TaskSignals, lane: str = "stable",
    ):
        self.assembler = assembler
        self.items = list(items)
        self.signals = signals
        self.lane = lane
        self.skill_stats = {item.name: item for item in self.items if item.kind == "skill"
                            and item.scope in {"*", signals.repository} and scope_allows(item, signals)}
        self.exposures: Dict[str, set] = {}
        self.assemblies: List[Dict[str, Any]] = []
        self._by_role: Dict[str, Assembly] = {}
        self._lock = threading.Lock()

    def rules_for(self, role: str) -> Assembly:
        """Same role, same task, same rules: assembled once and reused."""
        with self._lock:
            assembly = self._by_role.get(role)
            if assembly is None:
                assembly = self.assembler.rules_for(role, self.signals, self.items)
                self._by_role[role] = assembly
                self.assemblies.append(assembly.to_dict())
                for item_id in assembly.ids:
                    self.exposures.setdefault(item_id, set()).add(role)
            return assembly

    def extend_signals(self, text: str) -> None:
        """Add later-phase context (e.g. candidate findings) for roles not yet assembled."""
        with self._lock:
            self.signals.text = (self.signals.text + "\n" + text)[:60000]
            self.signals.cwes = sorted(set(self.signals.cwes) | {
                item.upper() for item in CWE.findall(text)
            })

    def skill_catalog(self, skills: Sequence[Any], requested: Iterable[str] = ()) -> List[Any]:
        allowed = [skill for skill in skills if self._allows_skill(skill)]
        return self.assembler.skill_catalog(allowed, self.signals, self.skill_stats, requested)

    def _allows_skill(self, skill: Any) -> bool:
        if getattr(skill, "source", "") != "learned":
            return True
        record = self.skill_stats.get(skill.name)
        return record is not None and skill.content == record.instruction_text()

    def skill_bodies(self, role: str, skills: Sequence[Any]) -> Dict[str, Any]:
        fitted = self.assembler.skill_bodies([skill for skill in skills if self._allows_skill(skill)])
        with self._lock:
            for skill in fitted["skills"]:
                record = self.skill_stats.get(skill.name)
                if record is not None:
                    self.exposures.setdefault(record.id, set()).add(role)
        return fitted

    def resolve_citations(self, values: Iterable[Any], role: str = "") -> List[str]:
        """Map cited rule ids or skill names to exposed item ids, dropping inventions."""
        resolved = []
        for value in values or []:
            key = str(value).strip()
            record = self.skill_stats.get(key)
            item_id = record.id if record is not None else key
            roles = self.exposures.get(item_id)
            if roles and (not role or role in roles) and item_id not in resolved:
                resolved.append(item_id)
        return resolved[:10]

    def summary(self) -> Dict[str, Any]:
        return {
            "lane": self.lane,
            "exposed": sorted(self.exposures),
            "assemblies": list(self.assemblies),
        }


class StaticKnowledge:
    """Fixed knowledge snapshot for replays and release gates: never records outcomes."""

    def __init__(self, assembler: ContextAssembler, items: Sequence[KnowledgeItem]):
        self.assembler = assembler
        self.items = [item for item in items if item.status != "archived"]

    def prepare(self, tenant_id: str, repository: str, lane: str, signals: TaskSignals) -> KnowledgeView:
        return KnowledgeView(self.assembler, self.items, signals, lane)

    def record_review(self, *_args, **_kwargs) -> None:
        return None
