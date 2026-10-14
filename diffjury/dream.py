"""Dream: offline consolidation of the knowledge base.

Admission keeps each *new* lesson small; Dream fixes what accumulates anyway.
It clusters live items by lexical similarity (per kind and role) and, for each
cluster, merges near-duplicates into one item that inherits their evidence and
outcome counters, drops contradicted or generic items, and then applies the
lifecycle plan (rot eviction and per-role capacity). The result is a
*proposal* - a candidate snapshot - that the release gate must accept before
anything is written. Without a model, only near-identical items are merged,
deterministically, which can never lose a distinct lesson (the ACE "context
collapse" failure is an LLM rewriting a whole playbook at once; Dream only
ever rewrites one small cluster, under a token cap).
"""
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .context_manager import estimate_tokens
from .knowledge import KnowledgeItem, KnowledgeRepository, utc_now, utility
from .lifecycle import KnowledgeLifecycle
from .similarity import pairwise_duplicates
from .telemetry import ExecutionLedger


LOGGER = logging.getLogger(__name__)
MAX_EVIDENCE = 20

CONSOLIDATE_PROMPT = """You maintain a code-review knowledge base that must stay small and
correct. Below is one cluster of similar lessons for the same reviewer role, with their
outcome statistics. Consolidate it:
- merges: lessons that teach the same thing become one; keep_id is the best-supported
  item and content is one sentence of at most {cap} tokens covering all of them.
- drop: lessons that are contradicted by a better-supported lesson, are generic advice,
  or have clearly hurt reviews (high harmful counts).
Leave unrelated lessons untouched by omitting them. Never invent new advice.
Return JSON only: {{"merges":[{{"keep_id":"K-...","absorb_ids":["K-..."],
"content":"..."}}],"drop":[{{"id":"K-...","reason":"..."}}]}}"""


@dataclass
class DreamProposal:
    baseline: List[KnowledgeItem]
    candidate: List[KnowledgeItem]
    merges: List[Dict[str, Any]] = field(default_factory=list)
    drops: List[Dict[str, Any]] = field(default_factory=list)
    evictions: Dict[str, str] = field(default_factory=dict)
    clusters: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.merges or self.drops or self.evictions)

    def live_tokens(self, items: Optional[List[KnowledgeItem]] = None) -> int:
        return sum(item.tokens for item in (self.candidate if items is None else items) if item.status != "archived")

    def summary(self) -> Dict[str, Any]:
        return {
            "clusters": self.clusters, "merges": self.merges, "drops": self.drops,
            "evictions": self.evictions,
            "live_items_before": sum(item.status != "archived" for item in self.baseline),
            "live_items_after": sum(item.status != "archived" for item in self.candidate),
            "live_tokens_before": self.live_tokens(self.baseline),
            "live_tokens_after": self.live_tokens(),
        }


def absorb(keep: KnowledgeItem, others: List[KnowledgeItem], content: str = "") -> None:
    """Fold duplicates into ``keep``: one item, all of their evidence and outcomes."""
    for other in others:
        keep.helpful = round(keep.helpful + other.helpful, 6)
        keep.harmful = round(keep.harmful + other.harmful, 6)
        keep.helpful_total = round(keep.helpful_total + other.helpful_total, 6)
        keep.harmful_total = round(keep.harmful_total + other.harmful_total, 6)
        keep.exposures += other.exposures
        keep.citations += other.citations
        keep.last_cited_at = max(keep.last_cited_at, other.last_cited_at)
        keep.last_used_at = max(keep.last_used_at, other.last_used_at)
        for value in other.evidence:
            if value not in keep.evidence:
                keep.evidence.append(value)
        for value in other.triggers:
            if value not in keep.triggers:
                keep.triggers.append(value)
        keep.parent_ids.append(other.id)
        keep.pinned = keep.pinned or other.pinned
        other.status = "archived"
        other.archived_reason = "merged-into:%s" % keep.id
    keep.evidence = keep.evidence[-MAX_EVIDENCE:]
    if content and content != keep.content:
        keep.content = content
    keep.version += 1
    keep.refresh_derived()


class DreamConsolidator:
    def __init__(
        self, repository: KnowledgeRepository, lifecycle: KnowledgeLifecycle,
        client=None, similarity_threshold: float = 0.3, dup_threshold: float = 0.85,
        max_rule_tokens: int = 60, max_cluster: int = 8,
    ):
        self.repository = repository
        self.lifecycle = lifecycle
        self.client = client
        self.similarity_threshold = similarity_threshold
        self.dup_threshold = dup_threshold
        self.max_rule_tokens = max_rule_tokens
        self.max_cluster = max(2, int(max_cluster))
        self.ledger = ExecutionLedger("knowledge-dream")

    def propose(self, tenant_id: str, now: str = "", consolidate: bool = True) -> DreamProposal:
        now = now or utc_now()
        baseline = self.repository.list(tenant_id, statuses=("active", "probation"))
        candidate = [copy.deepcopy(item) for item in baseline]
        proposal = DreamProposal(baseline, candidate)
        by_id = {item.id: item for item in candidate}
        groups: Dict[tuple, List[KnowledgeItem]] = {}
        for item in candidate if consolidate else []:
            if item.kind == "rule":  # skills consolidate through admission merges
                groups.setdefault((item.kind, item.role, item.scope), []).append(item)
        for items in groups.values():
            for cluster in self._clusters(items):
                proposal.clusters += 1
                self._consolidate(cluster, by_id, proposal, now)
        live = [item for item in candidate if item.status != "archived"]
        plan = self.lifecycle.plan(live, now)
        for item_id, reason in plan.archived.items():
            by_id[item_id].status = "archived"
            by_id[item_id].archived_reason = reason
            proposal.evictions[item_id] = reason
        return proposal

    def _clusters(self, items: List[KnowledgeItem]) -> List[List[KnowledgeItem]]:
        if len(items) < 2:
            return []
        documents = {item.id: item.content + " " + " ".join(item.triggers) for item in items}
        parent = {item.id: item.id for item in items}

        def find(key):
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key

        threshold = self.similarity_threshold if self.client is not None else self.dup_threshold
        for left, right, _score in pairwise_duplicates(documents, threshold):
            parent[find(left)] = find(right)
        members: Dict[str, List[KnowledgeItem]] = {}
        for item in items:
            members.setdefault(find(item.id), []).append(item)
        clusters = []
        for values in members.values():
            if len(values) < 2:
                continue
            values.sort(key=lambda item: (-utility(item), item.created_at))
            for start in range(0, len(values), self.max_cluster):
                chunk = values[start:start + self.max_cluster]
                if len(chunk) > 1:
                    clusters.append(chunk)
        return clusters

    def _consolidate(self, cluster, by_id, proposal: DreamProposal, now: str) -> None:
        ids = {item.id for item in cluster}
        if self.client is None:
            keep, others = cluster[0], cluster[1:]
            absorb(keep, others)
            proposal.merges.append({
                "keep_id": keep.id, "absorb_ids": [item.id for item in others],
                "content": keep.content, "judge": "threshold",
            })
            return
        payload = {"items": [
            {
                "id": item.id, "content": item.content, "triggers": item.triggers,
                "status": item.status, "pinned": item.pinned,
                "helpful": round(item.helpful, 2), "harmful": round(item.harmful, 2),
                "exposures": item.exposures, "citations": item.citations,
                "utility": utility(item, now), "evidence_count": len(item.evidence),
            }
            for item in cluster
        ]}
        try:
            reply = self.client.complete_json(
                "knowledge-dream", CONSOLIDATE_PROMPT.format(cap=self.max_rule_tokens),
                json.dumps(payload, ensure_ascii=False), self.ledger, max_tokens=900,
            )
        except Exception as exc:
            LOGGER.warning("dream consolidation skipped a cluster: %s", exc)
            return
        consumed = set()
        for merge in reply.get("merges") or []:
            if not isinstance(merge, dict):
                continue
            keep_id = str(merge.get("keep_id", ""))
            absorb_ids = [
                str(value) for value in merge.get("absorb_ids") or []
                if str(value) in ids and str(value) != keep_id
            ]
            content = " ".join(str(merge.get("content", "")).split())
            if (
                keep_id not in ids or not absorb_ids or keep_id in consumed
                or consumed.intersection(absorb_ids)
                or (content and estimate_tokens(content) > self.max_rule_tokens)
            ):
                continue
            absorb(by_id[keep_id], [by_id[value] for value in absorb_ids], content)
            consumed.update([keep_id, *absorb_ids])
            proposal.merges.append({
                "keep_id": keep_id, "absorb_ids": absorb_ids,
                "content": by_id[keep_id].content, "judge": "llm",
            })
        for drop in reply.get("drop") or []:
            if not isinstance(drop, dict):
                continue
            item = by_id.get(str(drop.get("id", "")))
            if item is None or item.id not in ids or item.pinned or item.id in consumed:
                continue
            item.status = "archived"
            item.archived_reason = "dream:%s" % str(drop.get("reason", "dropped"))[:120]
            consumed.add(item.id)
            proposal.drops.append({"id": item.id, "reason": item.archived_reason})
