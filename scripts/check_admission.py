"""Read-only replay checks against the original SWR experiment."""
import argparse
import json
import os
import re
import sqlite3
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffjury.admission import AdmissionGate  # noqa: E402
from diffjury.config import Settings  # noqa: E402
from diffjury.context_manager import estimate_tokens  # noqa: E402
from diffjury.llm import JsonChatClient  # noqa: E402
from diffjury.knowledge import KnowledgeItem  # noqa: E402
from diffjury.metrics import metrics  # noqa: E402
from diffjury.store import TaskStore  # noqa: E402


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def make_client(fake: bool = False):
    if fake:
        class Client:
            model = "fake"

            def complete_json(self, role, system, user, *args, **kwargs):
                if role == "knowledge-admission":
                    payload = json.loads(user)
                    return {"decision": "CONFLICT", "target_id": payload["similar_items"][0]["id"]}
                return {"content": "Check changed behaviour before reporting a defect."}
        return Client()
    config = Settings.from_env().resolved_llm()
    if not config:
        raise ValueError("No model configured in .env")
    return JsonChatClient(config["base_url"], config["api_key"], config["model"],
                          provider=config["provider"], timeout=115,
                          extra_headers=config.get("headers"))


def compress_check(args) -> None:
    source = ROOT / "evaluation_results/swr_rot/governed.db"
    with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT id, detail_json FROM knowledge_events WHERE event='admission' ORDER BY id"
        ).fetchall()
    candidates = []
    for key, detail in rows:
        event = json.loads(detail)
        if event.get("action") == "REJECT" and event.get("reason", "").startswith("rule exceeds"):
            candidates.append((key, event["candidate"]))
    if args.limit:
        candidates = candidates[:args.limit]
    output = Path(args.out)
    previous = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    client = make_client(args.fake)
    if previous and (previous.get("model") != client.model or previous.get("fake") != args.fake):
        raise ValueError("Existing check uses a different model or fake mode")
    results = {row["event_id"]: row for row in previous.get("results", [])}

    def run(pair):
        key, candidate = pair
        calls = []

        class AuditedClient:
            def complete_json(self, *values, **kwargs):
                try:
                    reply = client.complete_json(*values, **kwargs)
                except Exception as exc:
                    calls.append({"ok": False, "error_type": type(exc).__name__,
                                  "http_status": re.findall(r"HTTP (\d+)", str(exc))})
                    raise
                text = str(reply.get("content") or "")
                calls.append({"ok": True, "returned_tokens": estimate_tokens(text)})
                return reply

        gate = AdmissionGate(None, AuditedClient(), max_rule_tokens=args.cap)
        content = str(candidate.get("content") or "")
        compressed = gate._compress(content)
        return {"event_id": key, "source": candidate.get("source"),
                "original_tokens": estimate_tokens(content), "compressed_tokens": estimate_tokens(compressed),
                "fits": bool(compressed) and estimate_tokens(compressed) <= args.cap,
                "content": compressed, "compression_calls": calls}

    def save():
        ordered = [results[key] for key, _ in candidates if key in results]
        fits = sum(row["fits"] for row in ordered)
        write_json(output, {"model": client.model, "fake": args.fake, "cap": args.cap,
                            "source": str(source.relative_to(ROOT)), "candidates": len(candidates),
                            "completed": len(ordered), "fits": fits,
                            "fit_rate": fits / len(ordered) if ordered else 0,
                            "target_met": bool(ordered) and fits / len(ordered) >= 0.8,
                            "compression_only": True, "source_database_read_only": True,
                            "metrics": dict(metrics.counters), "results": ordered})

    pending = [pair for pair in candidates if pair[0] not in results or
               (args.retry_failed and not results[pair[0]]["fits"])]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run, pair) for pair in pending]):
            row = future.result()
            results[row["event_id"]] = row
            save()
            print("compression completed %d/%d; fits %d" % (
                len(results), len(candidates), sum(row["fits"] for row in results.values())), flush=True)
    save()


def conflict_check(args) -> None:
    source = ROOT / "evaluation_results/swr_rot/events.json"
    events = json.loads(source.read_text(encoding="utf-8"))
    by_index = {event["index"]: event for event in events}
    candidates = [event for event in events if event["type"] == "contradiction"]
    if args.limit:
        candidates = candidates[:args.limit]
    client = make_client(args.fake)
    output = Path(args.out)
    previous = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    if previous and (previous.get("model") != client.model or previous.get("fake") != args.fake):
        raise ValueError("Existing check uses a different model or fake mode")
    results = {row["event_index"]: row for row in previous.get("results", [])}

    def run(event):
        target = by_index.get(event.get("target_index"))
        if not target or not target.get("lessons") or not event.get("lessons"):
            return {"event_index": event["index"], "action": "SKIP", "conflict": False,
                    "reason": "missing original lesson or contradiction"}
        with tempfile.TemporaryDirectory(prefix="dj-conflict-") as temporary:
            repo = TaskStore(str(Path(temporary) / "check.db")).knowledge
            original = target["lessons"][0]
            item = repo.create(KnowledgeItem(
                "default", original["kind"], original["content"],
                role=original.get("role", "*"), triggers=original.get("triggers") or [],
                status="active", evidence=[{"event": target["index"]}], source="missed",
            ))
            gate = AdmissionGate(repo, client, max_rule_tokens=args.cap)
            decision = gate.admit("default", {
                **event["lessons"][0], "source": "contradiction",
                "evidence": [{"event": event["index"]}],
            })
            return {"event_index": event["index"], "target_index": event["target_index"],
                    **decision.to_dict(), "conflict": decision.judge == "llm-conflict",
                    "original_unchanged": repo.get("default", item.id).content == original["content"]
                    and repo.get("default", item.id).evidence == [{"event": target["index"]}]}

    def save():
        rows = [results[event["index"]] for event in candidates if event["index"] in results]
        blocked = sum(row["action"] == "REJECT" for row in rows)
        write_json(output, {"model": client.model, "fake": args.fake,
                            "source": str(source.relative_to(ROOT)), "candidates": len(candidates),
                            "completed": len(rows), "conflicts": sum(row["conflict"] for row in rows),
                            "rejected": blocked, "blocked_rate": blocked / len(rows) if rows else 0,
                            "protocol": "each contradiction judged against its original lesson in an isolated temporary database",
                            "source_unchanged": True, "results": rows})

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = [event for event in candidates if event["index"] not in results or
                   (args.retry_failed and results[event["index"]]["action"] != "REJECT")]
        for future in as_completed([pool.submit(run, event) for event in pending]):
            row = future.result()
            results[row["event_index"]] = row
            save()
            print("conflict completed %d/%d" % (len(results), len(candidates)), flush=True)
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("compress", "conflict"), default="compress")
    parser.add_argument("--out", default="")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--cap", type=int, default=60)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be between 1 and 16")
    if not args.out:
        args.out = str(ROOT / ("evaluation_results/admission_%s_check.json" % args.mode))
    (compress_check if args.mode == "compress" else conflict_check)(args)


if __name__ == "__main__":
    main()
