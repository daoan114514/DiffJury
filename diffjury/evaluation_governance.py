"""Audit-derived governance statistics; no model calls or database writes."""
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .knowledge import KnowledgeItem
from .source_validation import TERMINAL, validation_signature


POLLUTED_SOURCES = {"noise", "contradiction"}
DECISIONS = ("ADD", "REJECT", "UPVOTE", "CONFLICT", "MERGE", "RESTORE")


def contamination(items: Sequence[dict], statuses: Sequence[str]) -> dict:
    live = [item for item in items if item["status"] in statuses and item.get("source") != "disk"]
    polluted = [item for item in live if item.get("source") in POLLUTED_SOURCES]
    total_tokens = sum(item["tokens"] for item in live)
    polluted_tokens = sum(item["tokens"] for item in polluted)
    return {"items": len(live), "polluted_items": len(polluted),
            "item_rate": round(len(polluted) / len(live), 6) if live else None,
            "tokens": total_tokens, "polluted_tokens": polluted_tokens,
            "token_rate": round(polluted_tokens / total_tokens, 6) if total_tokens else None,
            "by_source": dict(Counter(item["source"] for item in live)),
            "definition": "live rules and learned skills; excludes shipped disk skills; pollution uses item.source"}


def governance_statistics(items: Dict[str, List[dict]], audit: Sequence[dict],
                          events: Sequence[dict]) -> Dict[str, Any]:
    by_source, admissions = {}, []
    grouped = defaultdict(list)
    for row in audit:
        if row["event"] != "admission":
            continue
        detail = row["detail"]
        candidate = detail.get("candidate") or {}
        source = candidate.get("source") or "unknown"
        action = detail.get("action", "UNKNOWN")
        decision = "CONFLICT" if detail.get("judge") == "llm-conflict" else action
        counts = by_source.setdefault(source, {key: 0 for key in DECISIONS})
        counts[decision] = counts.get(decision, 0) + 1
        admissions.append((source, action, decision))
        for evidence in candidate.get("evidence") or []:
            if "event" in evidence:
                grouped[(source, evidence["event"])].append(action)
    interception = {}
    for source in sorted(POLLUTED_SOURCES):
        candidates = [row for row in admissions if row[0] == source]
        rejected = sum(row[1] == "REJECT" for row in candidates)
        source_events = [row for key, row in grouped.items() if key[0] == source]
        blocked_events = sum(all(action == "REJECT" for action in row) for row in source_events)
        interception[source] = {
            "planned_events": sum(event["type"] == source for event in events),
            "candidate_count": len(candidates), "rejected_candidates": rejected,
            "candidate_rate": round(rejected / len(candidates), 6) if candidates else None,
            "events_with_candidates": len(source_events), "blocked_events": blocked_events,
            "event_rate": round(blocked_events / len(source_events), 6) if source_events else None,
        }
    promotions = [row for row in audit if row["event"] == "promote"]
    evictions = [row for row in audit if row["event"] == "evict"]
    counterfactual = [row for row in audit if row["event"] == "counterfactual"]
    applied = [row for row in counterfactual if row["detail"].get("applied")]
    cf = {"pair_attempts": len(counterfactual),
          "individual_replays": sum(row["detail"].get("replays", 0) for row in counterfactual),
          "replay_definition": "review attempts, including failures and unexposed trial reviews",
          "applied_pairs": len(applied),
          "helpful": sum(row["detail"].get("helpful", 0) for row in applied),
          "harmful": sum(row["detail"].get("harmful", 0) for row in applied),
          "neutral": sum(row["detail"].get("neutral", 0) for row in applied),
          "failed": sum("error_type" in row["detail"] for row in counterfactual),
          "not_exposed": sum("skipped" in row["detail"] for row in counterfactual),
          "unique_items": len({row["item_id"] for row in applied}),
          "unique_prs": len({row["detail"]["case_id"] for row in applied}),
          "train_only": all(row["detail"].get("split") == "train" for row in counterfactual)}
    archived = [item for item in items["governed"] if item["status"] == "archived"]
    validation = [row for row in audit if row["event"] == "source_validation"]
    reservations = [row for row in audit if row["event"] == "source_validation_reservation"]
    started = [row for row in audit if row["event"] == "source_validation_replay"]
    validated = [row for row in validation if row["detail"].get("applied")]
    fast_promotions = [row for row in audit if row["event"] == "source_validation_promotion"]
    completed_jobs = {row["detail"].get("job_key"): row["detail"] for row in validation}
    validation_rounds = {}
    for row in reservations:
        detail = row["detail"]
        record = validation_rounds.setdefault(detail["round_id"], {"replay_limit": detail.get("replay_limit"), "budget_consumed": 0, "review_attempts": 0})
        record["budget_consumed"] += completed_jobs.get(detail["job_key"], {}).get("replays", detail["reserved_replays"])
    for row in started:
        record = validation_rounds.setdefault(row["detail"]["round_id"], {"replay_limit": None, "budget_consumed": 0, "review_attempts": 0})
        record["review_attempts"] += 1
    proof_status = {}
    by_id = {item["id"]: item for item in items["governed"]}
    for row in validation:
        item, detail = by_id.get(row["item_id"]), row["detail"]
        if item and all(key in item for key in ("content", "body", "role", "scope", "triggers")):
            if detail.get("signature") != validation_signature(KnowledgeItem.from_dict({"tenant_id": "default", **item})):
                continue
            previous = proof_status.get(item["id"])
            if previous is None or previous not in TERMINAL and detail.get("status") in TERMINAL:
                proof_status[item["id"]] = detail.get("status", "unverified")
    source_validation = {
        "entry_attempts": len(reservations), "result_events": len(validation),
        "review_attempts": len(started), "applied_entries": len(validated),
        "completed_pairs": sum(row["detail"].get("completed_pairs", 0) for row in validated),
        "helpful": sum(row["detail"].get("helpful", 0) for row in validated),
        "harmful": sum(row["detail"].get("harmful", 0) for row in validated),
        "neutral": sum(row["detail"].get("neutral", 0) for row in validated),
        "statuses": dict(Counter(row["detail"].get("status", "unverified") for row in validation)),
        "passed_unique_items": len({row["item_id"] for row in validation if row["detail"].get("status") == "passed"}),
        "failed": sum(row["detail"].get("error_type") not in {None, "LookupError"} for row in validation),
        "not_exposed": sum(bool(row["detail"].get("not_exposed")) for row in validation),
        "skipped": sum("skipped" in row["detail"] for row in validation),
        "unfinished_reservations": sum(row["detail"]["job_key"] not in completed_jobs for row in reservations),
        "unique_source_prs": len({row["detail"]["case_id"] for row in validation if row["detail"].get("case_id")}),
        "unique_side_prs": len({case["case_id"] for row in validation for case in row["detail"].get("side_results", [])}),
        "train_only": all(row["detail"].get("split") == "train" for row in started) if started else None,
        "fast_promotion_events": len(fast_promotions), "unique_fast_promoted_items": len({row["item_id"] for row in fast_promotions}),
        "archived_validation_status": dict(Counter(proof_status.get(item["id"], "unverified") for item in archived if item["source"] in {"missed", "false_positive"})),
        "rounds": validation_rounds,
        "notice": "source screening overlaps recurrence feedback PRs; k=3 is a screening condition, not a significance test",
    }
    archive_categories = Counter()
    for item in archived:
        reason = item.get("archived_reason") or "unspecified"
        category = "merged" if reason.startswith("merged-into:") else reason.split(":", 1)[0]
        archive_categories[category] += 1
    return {"admission_by_source": by_source, "admission_candidates": len(admissions),
            "decision_definition": "CONFLICT is an exclusive display category; its stored action is REJECT",
            "interception": interception,
            "contamination": {"naive": contamination(items["naive"], ("active",)),
                              "governed": contamination(items["governed"], ("active",)),
                              "governed_canary": contamination(items["governed"], ("active", "probation"))},
            "lifecycle": {"promotion_events": len(promotions),
                          "unique_promoted_items": len({row["item_id"] for row in promotions}),
                          "eviction_events": len(evictions),
                          "unique_evicted_items": len({row["item_id"] for row in evictions}),
                          "currently_archived": len(archived),
                          "archived_categories": dict(archive_categories),
                          "archived_reasons": dict(Counter(item.get("archived_reason") or "unspecified" for item in archived))},
            "counterfactual": cf, "source_validation": source_validation}


def read_governance(out_dir: str, tenant_id: str = "default") -> Optional[Dict[str, Any]]:
    root = Path(out_dir)
    if not all((root / (name + ".db")).exists() for name in ("naive", "governed")):
        return None
    items, audit = {}, []
    for name in ("naive", "governed"):
        with sqlite3.connect((root / (name + ".db")).resolve().as_uri() + "?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            items[name] = [dict(row) for row in conn.execute(
                "SELECT * FROM knowledge_items WHERE tenant_id=?", (tenant_id,))]
            for item in items[name]:
                if "triggers_json" in item:
                    item["triggers"] = json.loads(item.pop("triggers_json"))
            if name == "governed":
                for row in conn.execute("SELECT item_id,event,detail_json FROM knowledge_events WHERE tenant_id=?", (tenant_id,)):
                    audit.append({"item_id": row["item_id"], "event": row["event"], "detail": json.loads(row["detail_json"])})
    path = root / "events.json"
    events = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    return governance_statistics(items, audit, events)
