"""Write-side gate that keeps learned knowledge from bloating.

Every candidate lesson is checked before it touches the knowledge base:

1. validation - non-empty, evidence-backed, within its token cap (an LLM may
   compress an over-long rule, retry strictly, then try its first sentence);
2. exact duplicate (content fingerprint) - UPVOTE the existing item;
3. near duplicates - TF-IDF top-k over live items of the same kind and role.
   With a model, a judge picks ADD / MERGE / UPVOTE / REJECT; a MERGE must fit
   the same cap, so merging can never grow an item past its budget. Without a
   model, cosine >= ``dup_threshold`` is an UPVOTE and anything else an ADD;
4. a candidate that closely matches an archived item RESTOREs it to probation
   instead of creating a second copy.

New items always start in ``probation`` and only reach every review after the
canary lane and the release gate have seen them.
"""
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from .context_manager import estimate_tokens
from .knowledge import ROLES, KnowledgeItem, KnowledgeRepository, utc_now, utility
from .similarity import TfidfIndex
from .metrics import metrics
from .telemetry import ExecutionLedger


SKILL_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
MAX_EVIDENCE = 20

JUDGE_PROMPT = """You curate a code-review knowledge base that must stay small.
Decide what to do with one candidate lesson given its most similar existing items.
- UPVOTE: an existing item already says the same thing; the candidate adds only evidence.
- MERGE: the candidate refines one existing item; write merged_content that keeps both
  lessons in at most {cap} tokens. Never concatenate; generalize and cut redundancy.
- ADD: the candidate is a genuinely new, specific, actionable lesson.
- CONFLICT: the candidate and an existing item apply to the SAME scenario but
  recommend opposite actions (e.g. "Require X" versus "Do not report X").
  Give the existing item's target_id. Never UPVOTE or MERGE contradictory advice.
- REJECT: vague, generic advice the model already follows, wrong, or contradicted by a
  better-supported existing item.
Prefer UPVOTE or MERGE over ADD whenever the lessons overlap.
Return JSON only: {{"decision":"ADD|MERGE|UPVOTE|REJECT|CONFLICT","target_id":"K-...",
"merged_content":"...","reason":"..."}}"""

COMPRESS_PROMPT = """Rewrite one code-review lesson as a single imperative sentence of at
most {cap} tokens AND {chars} characters, including spaces and punctuation.
Keep the concrete trigger (API, pattern, CWE) and the action; drop
examples, rationale and filler. Preserve the ORIGINAL recommendation and negation,
even if you disagree with its facts. Do not correct, reverse, or improve its advice.
If it says not to flag something, keep that instruction explicitly negative.
Return JSON only: {{"content":"..."}}"""

STRICT_COMPRESS_PROMPT = """Compress this code-review lesson to at most {cap} tokens
AND {chars} characters, including spaces and punctuation. This is a HARD limit;
use fewer than 20 words and count characters before returning the sentence.
Keep only ONE short imperative sentence with the general trigger and action.
Remove ALL examples, concrete identifiers, file paths, variable/function names,
explanations and rationale. Preserve negation and the intended recommendation.
Do not correct the lesson's facts or reverse its advice, even if it is wrong.
Return JSON only: {{"content":"..."}}"""


@dataclass
class AdmissionDecision:
    action: str
    item_id: str = ""
    reason: str = ""
    judge: str = ""
    similar: List[Tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action, "item_id": self.item_id, "reason": self.reason,
            "judge": self.judge,
            "similar": [{"id": key, "score": score} for key, score in self.similar],
        }


def _merge_unique(values: List[Any], extra: List[Any], limit: int) -> List[Any]:
    merged = list(values)
    for item in extra:
        if item not in merged:
            merged.append(item)
    return merged[-limit:]


def opposite_polarity(left: str, right: str) -> bool:
    """A similarity shortcut must not discard a recommendation's negation."""
    negative = re.compile(r"\b(?:do\s+not|don't|does\s+not|should\s+not|must\s+not|never|avoid)\b", re.I)
    return bool(negative.search(left)) != bool(negative.search(right))


class AdmissionGate:
    def __init__(
        self, repository: KnowledgeRepository, client=None,
        max_rule_tokens: int = 60, max_skill_tokens: int = 1500,
        similarity_threshold: float = 0.3, dup_threshold: float = 0.85,
        top_k: int = 5, require_evidence: bool = True, screen_new: bool = True,
    ):
        self.repository = repository
        self.client = client
        self.max_rule_tokens = max(8, int(max_rule_tokens))
        self.max_skill_tokens = max(64, int(max_skill_tokens))
        self.similarity_threshold = float(similarity_threshold)
        self.dup_threshold = float(dup_threshold)
        self.top_k = max(1, int(top_k))
        self.require_evidence = require_evidence
        # With a model, even lessons without neighbours are screened for
        # generic or wrong advice before they may enter probation.
        self.screen_new = screen_new
        self.ledger = ExecutionLedger("knowledge-admission")

    # --------------------------------------------------------------- public
    def admit(self, tenant_id: str, candidate: Dict[str, Any]) -> AdmissionDecision:
        decision = self._decide(tenant_id, dict(candidate))
        metrics.inc("knowledge_admission_%s_total" % decision.action.lower())
        self.repository.log_event(tenant_id, decision.item_id, "admission", {
            **decision.to_dict(),
            "candidate": {
                key: candidate.get(key) for key in (
                    "kind", "role", "scope", "name", "content", "source", "triggers", "evidence",
                )
            },
        })
        return decision

    def cap(self, kind: str) -> int:
        return self.max_skill_tokens if kind == "skill" else self.max_rule_tokens

    # ------------------------------------------------------------- pipeline
    def _decide(self, tenant_id: str, candidate: Dict[str, Any]) -> AdmissionDecision:
        kind = str(candidate.get("kind") or "rule")
        role = str(candidate.get("role") or "*")
        evidence = [item for item in candidate.get("evidence") or [] if item]
        content = " ".join(str(candidate.get("content") or "").split())
        if kind not in {"rule", "skill"} or role not in ROLES:
            return AdmissionDecision("REJECT", reason="invalid kind or role", judge="validation")
        if not content:
            return AdmissionDecision("REJECT", reason="empty content", judge="validation")
        if self.require_evidence and not evidence:
            return AdmissionDecision(
                "REJECT", reason="a lesson must cite the review evidence it came from",
                judge="validation",
            )
        name = str(candidate.get("name") or "").strip().lower()
        body = str(candidate.get("body") or "").strip()
        if kind == "skill" and not SKILL_NAME.fullmatch(name):
            return AdmissionDecision("REJECT", reason="invalid skill name", judge="validation")
        if kind == "rule":
            name, body = "", ""
            if estimate_tokens(content) > self.max_rule_tokens:
                content = self._compress(content)
                if not content or estimate_tokens(content) > self.max_rule_tokens:
                    return AdmissionDecision(
                        "REJECT", reason="rule exceeds %d tokens after compression" % self.max_rule_tokens,
                        judge="validation",
                    )
        try:
            item = KnowledgeItem(
                tenant_id=tenant_id, kind=kind, role=role,
                scope=str(candidate.get("scope") or "*"), name=name, content=content,
                body=body, triggers=list(candidate.get("triggers") or []),
                evidence=evidence[:MAX_EVIDENCE],
                source=str(candidate.get("source") or "feedback"), status="probation",
            )
        except ValueError as exc:
            return AdmissionDecision("REJECT", reason=str(exc), judge="validation")
        if item.tokens > self.cap(kind):
            return AdmissionDecision(
                "REJECT", reason="%s exceeds %d tokens" % (kind, self.cap(kind)),
                judge="validation",
            )

        duplicate = next((existing for existing in self.repository.list(tenant_id, kind=kind)
                          if existing.fingerprint == item.fingerprint
                          and existing.status != "archived"
                          and existing.scope in {item.scope, "*"}
                          and (existing.role in {item.role, "*"} or item.role == "*")), None)
        if duplicate is not None and not opposite_polarity(duplicate.content, item.content):
            self._upvote(duplicate, item)
            return AdmissionDecision(
                "UPVOTE", duplicate.id, "identical lesson already exists", "fingerprint",
                [(duplicate.id, 1.0)],
            )

        live, archived = self._neighbors(tenant_id, item)
        if duplicate is not None and opposite_polarity(duplicate.content, item.content):
            live = [(duplicate, 1.0)] + [(neighbor, score) for neighbor, score in live
                                        if neighbor.id != duplicate.id]
        if (archived and archived[0][1] >= self.dup_threshold
                and opposite_polarity(archived[0][0].content, item.content)
                and archived[0][0].id not in {neighbor.id for neighbor, _ in live}):
            live.append(archived[0])
        if (archived and archived[0][1] >= self.dup_threshold
                and not opposite_polarity(archived[0][0].content, item.content)):
            restored = archived[0][0]
            restored.status = "probation"
            restored.archived_reason = ""
            restored.evidence = _merge_unique(restored.evidence, item.evidence, MAX_EVIDENCE)
            self.repository.save(restored)
            self.repository.log_event(tenant_id, restored.id, "restore", {
                "reason": "an archived lesson was learned again", "score": archived[0][1],
            })
            return AdmissionDecision(
                "RESTORE", restored.id, "matches an archived lesson", "threshold",
                [(restored.id, archived[0][1])],
            )
        similar = [(neighbor.id, score) for neighbor, score in live]
        if not live and (self.client is None or not self.screen_new):
            self.repository.create(item)
            return AdmissionDecision("ADD", item.id, "no similar lesson", "threshold")
        if self.client is None:
            best, score = live[0]
            if score >= self.dup_threshold and opposite_polarity(best.content, item.content):
                return AdmissionDecision(
                    "REJECT", reason="opposite advice requires an admission judge",
                    judge="validation", similar=similar,
                )
            if score >= self.dup_threshold:
                self._upvote(best, item)
                return AdmissionDecision(
                    "UPVOTE", best.id, "near-duplicate (cosine %.2f)" % score,
                    "threshold", similar,
                )
            self.repository.create(item)
            return AdmissionDecision(
                "ADD", item.id, "below duplicate threshold", "threshold", similar,
            )
        return self._judge(item, live, similar)

    def _neighbors(self, tenant_id: str, item: KnowledgeItem):
        pool = [
            existing for existing in self.repository.list(tenant_id, kind=item.kind)
            if (existing.role in {item.role, "*"} or item.role == "*")
            and existing.scope in {item.scope, "*"}
        ]
        if not pool:
            return [], []
        documents = {existing.id: self._text(existing) for existing in pool}
        documents["__candidate__"] = self._text(item)
        index = TfidfIndex(documents)
        by_id = {existing.id: existing for existing in pool}
        ranked = index.top_k(documents["__candidate__"], k=len(pool) + 1, exclude=["__candidate__"])
        live = [
            (by_id[key], score) for key, score in ranked
            if by_id[key].status != "archived" and score >= self.similarity_threshold
        ][:self.top_k]
        archived = [
            (by_id[key], score) for key, score in ranked
            if by_id[key].status == "archived"
        ][:1]
        return live, archived

    @staticmethod
    def _text(item: KnowledgeItem) -> str:
        return " ".join([item.name, item.content, " ".join(item.triggers)])

    def _upvote(self, target: KnowledgeItem, candidate: KnowledgeItem) -> None:
        target.evidence = _merge_unique(target.evidence, candidate.evidence, MAX_EVIDENCE)
        target.triggers = _merge_unique(target.triggers, candidate.triggers, 24)
        self.repository.save(target)

    def _judge(self, item: KnowledgeItem, live, similar) -> AdmissionDecision:
        cap = self.cap(item.kind)
        now = utc_now()
        payload = {
            "candidate": {
                "kind": item.kind, "role": item.role, "name": item.name,
                "content": item.content, "triggers": item.triggers, "scope": item.scope,
            },
            "similar_items": [
                {
                    "id": neighbor.id, "content": neighbor.content, "name": neighbor.name,
                    "similarity": score, "status": neighbor.status,
                    "scope": neighbor.scope, "triggers": neighbor.triggers,
                    "utility": utility(neighbor, now), "evidence_count": len(neighbor.evidence),
                }
                for neighbor, score in live
            ],
        }
        try:
            verdict = self.client.complete_json(
                "knowledge-admission", JUDGE_PROMPT.format(cap=cap),
                json.dumps(payload, ensure_ascii=False), self.ledger, max_tokens=600,
            )
        except Exception as exc:  # judge outages must not block learning
            verdict = {"decision": "", "reason": "judge failed: %s" % str(exc)[:200]}
        action = str(verdict.get("decision", "")).strip().upper()
        reason = str(verdict.get("reason", ""))[:500]
        targets = {neighbor.id: neighbor for neighbor, _score in live}
        target = targets.get(str(verdict.get("target_id", "")))
        if action == "CONFLICT":
            target = target or (live[0][0] if live else None)
            if target is None:
                return AdmissionDecision("REJECT", reason="conflict reply has no existing target", judge="llm", similar=similar)
            self.repository.log_event(item.tenant_id, target.id, "conflict", {
                "candidate": item.content, "source": item.source,
                "scope": item.scope, "reason": reason,
            })
            metrics.inc("knowledge_admission_conflict_total")
            return AdmissionDecision("REJECT", target.id, "conflicts with %s" % target.id, "llm-conflict", similar)
        if not live and action not in {"ADD", "REJECT"}:
            action = "ADD"
        if action in {"UPVOTE", "MERGE"} and target is None:
            target = live[0][0]
        if action == "UPVOTE":
            self._upvote(target, item)
            return AdmissionDecision("UPVOTE", target.id, reason, "llm", similar)
        if action == "MERGE":
            merged = " ".join(str(verdict.get("merged_content", "")).split())
            if item.kind == "skill":
                merged = merged or target.content
            if merged and estimate_tokens(merged) <= cap:
                before = target.content
                target.content = merged
                if item.kind == "skill" and item.body and estimate_tokens(
                    "%s: %s\n%s" % (target.name, merged, item.body)
                ) <= cap and len(item.body) > len(target.body):
                    target.body = item.body
                target.triggers = _merge_unique(target.triggers, item.triggers, 24)
                target.evidence = _merge_unique(target.evidence, item.evidence, MAX_EVIDENCE)
                target.version += 1
                target.parent_ids = _merge_unique(
                    target.parent_ids, ["candidate:%s" % item.fingerprint], 50,
                )
                self.repository.save(target)
                self.repository.log_event(item.tenant_id, target.id, "merge", {
                    "before": before, "after": merged, "candidate": item.content,
                })
                return AdmissionDecision("MERGE", target.id, reason, "llm", similar)
            # An over-budget merge degrades to an upvote rather than bloating the item.
            self._upvote(target, item)
            return AdmissionDecision(
                "UPVOTE", target.id, "merge exceeded %d tokens; kept existing text" % cap,
                "llm", similar,
            )
        if action == "REJECT":
            return AdmissionDecision("REJECT", "", reason, "llm", similar)
        if action != "ADD" and live:
            best, score = live[0]
            if score >= self.dup_threshold and opposite_polarity(best.content, item.content):
                return AdmissionDecision("REJECT", reason=reason or "opposite advice requires a valid judge reply", judge="validation", similar=similar)
            if score >= self.dup_threshold:
                self._upvote(best, item)
                return AdmissionDecision(
                    "UPVOTE", best.id, reason or "invalid judge reply", "threshold", similar,
                )
        self.repository.create(item)
        return AdmissionDecision("ADD", item.id, reason, "llm" if action == "ADD" else "threshold", similar)

    def _compress(self, content: str) -> str:
        """Fit a rule without clipping a sentence or reversing its advice."""
        if estimate_tokens(content) <= self.max_rule_tokens:
            return content
        compressed = self._compression_request(content, strict=False)
        if (compressed and estimate_tokens(compressed) <= self.max_rule_tokens
                and not opposite_polarity(content, compressed)):
            return compressed
        metrics.inc("knowledge_admission_compress_retry_total")
        retried = self._compression_request(content, strict=True)
        if (retried and estimate_tokens(retried) <= self.max_rule_tokens
                and not opposite_polarity(content, retried)):
            return retried
        for value in (retried, compressed, content):
            # A dot inside an API name is not a sentence boundary.
            first = re.split(r"(?<=[.!?])\s+|(?<=[。！？])", value.strip(), maxsplit=1)[0]
            if (first and estimate_tokens(first) <= self.max_rule_tokens
                    and not opposite_polarity(content, first)):
                return first
        metrics.inc("knowledge_admission_compress_failed_total")
        return ""

    def _compression_request(self, content: str, strict: bool) -> str:
        if self.client is None:
            return ""
        try:
            prompt = STRICT_COMPRESS_PROMPT if strict else COMPRESS_PROMPT
            cap = max(1, int(self.max_rule_tokens * 0.8)) if strict else self.max_rule_tokens
            reply = self.client.complete_json(
                "knowledge-compress", prompt.format(cap=cap, chars=max(1, int((cap - 4) * 3.2))),
                json.dumps({"lesson": content[:4000]}, ensure_ascii=False),
                self.ledger, max_tokens=600,
            )
        except Exception:
            return ""
        return " ".join(str(reply.get("content", "")).split())
