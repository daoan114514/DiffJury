"""Read-only observer of gate cache replacements during this resume."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from monitor_swr_rot_experiment import process_alive


def snapshot(path):
    stat = path.stat()
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = report["case_results"]
    return {"mtime_ns": stat.st_mtime_ns, "rows": len(rows),
            "failed": sum(not row["execution_success"] for row in rows)}


def main():
    out = Path(__file__).resolve().parent
    pid = int(sys.argv[1])
    previous = {path.name: snapshot(path) for path in (out / "gate_cache").glob("*.json")}
    destination = out / ("gate_cache_observations_%s.jsonl" % pid)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"kind": "observer_start", "pid": pid,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "baseline": previous,
            "notice": "Observed cache replacements imply a lower bound on review attempts; unobserved writes and first-run retries cannot be reconstructed."}) + "\n")
        handle.flush()
        while True:
            for path in (out / "gate_cache").glob("*.json"):
                try:
                    current = snapshot(path)
                except (OSError, ValueError):
                    continue
                old = previous.get(path.name)
                if old and old["mtime_ns"] == current["mtime_ns"]:
                    continue
                handle.write(json.dumps({"kind": "cache_replaced", "cache": path.name,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "previous": old, "current": current,
                    "review_attempts_lower_bound": old["failed"] if old else current["rows"]}) + "\n")
                handle.flush()
                previous[path.name] = current
            if not process_alive(pid):
                handle.write(json.dumps({"kind": "observer_end", "pid": pid,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
                break
            time.sleep(0.5)


if __name__ == "__main__":
    main()
