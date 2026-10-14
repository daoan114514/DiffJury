"""Print read-only progress for the task-owned governed-only experiment."""
import argparse
import json
from pathlib import Path
import sqlite3

from monitor_swr_rot_experiment import process_alive


def progress(out: Path) -> dict:
    process = json.loads((out / "process.json").read_text(encoding="utf-8"))
    state = json.loads((out / "state.json").read_text(encoding="utf-8")) if (out / "state.json").exists() else {}
    lines = (out / "run.log").read_text(encoding="utf-8").splitlines()
    phase = "starting"
    for line in lines:
        for marker in ("attribution @", "source validation @", "counterfactual @", "maintenance @", "replay ",
                       "holdout paired bootstrap:", "recurrence paired bootstrap:", "stopped:", "done:"):
            if marker in line:
                phase = line.split("] ", 1)[-1]
    pending = state.get("pending_event")
    original_config = (state.get("run_configuration") or {}).get("source_configuration") or {}
    interval = original_config.get("review_every", 0)
    if (pending and interval and pending % interval == 0
            and not any(row["event"] == pending for row in state.get("attribution", []))):
        phase = "attribution @%s in progress" % pending
    elif phase == "starting" and state.get("events_total"):
        phase = "governed learning %s/%s" % (state.get("next_event"), state["events_total"])
    caches = {}
    for name in ("governed", "governed_canary", "baseline_rerun"):
        for kind in ("holdout_rows", "recurrence_rows"):
            paths = list((out / kind / name).glob("*.json"))
            rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
            caches["%s/%s" % (kind, name)] = {"cached": len(rows), "successful": sum(bool(row["execution_success"]) for row in rows)}
    validation = (state.get("governance") or {}).get("source_validation") or {}
    live = {}
    database = out / "governed.db"
    if database.exists():
        with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as conn:
            live["event_counts"] = dict(conn.execute("SELECT event,COUNT(*) FROM knowledge_events GROUP BY event"))
            live["item_statuses"] = dict(conn.execute("SELECT status,COUNT(*) FROM knowledge_items GROUP BY status"))
    return {"pid": process["pid"], "alive": process_alive(process["pid"]), "phase": phase,
            "next_event": state.get("next_event"), "events_total": state.get("events_total"),
            "completed_maintenance": [row["event"] for row in state.get("maintenance", [])],
            "source_validation_reviews": validation.get("review_attempts"),
            "fast_promoted": validation.get("unique_fast_promoted_items"), "caches": caches,
            "live_governance": live, "finished_at": state.get("finished_at")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="evaluation_results/swr_rot_temporal_v2")
    args = parser.parse_args()
    print(json.dumps(progress(Path(args.out))))
