"""Read safe progress metadata for the diagnostic background process."""
import argparse
import json
from pathlib import Path

from monitor_swr_rot_experiment import process_alive


def progress(directory: Path) -> dict:
    process = json.loads((directory / "process.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    state_path = directory / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    counts = {}
    for name in ("baseline_rerun", "governed_assembler_empty", "governed_rerun"):
        paths = list((directory / "holdout_rows" / name).glob("*.json"))
        rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        counts[name] = {"cached": len(rows), "successful": sum(bool(row["execution_success"]) for row in rows)}
    return {"pid": process["pid"], "alive": process_alive(process["pid"]),
            "holdout_total": len(manifest["case_ids"]), "arms": counts,
            "completed_arms": state.get("completed_arms", []), "finished_at": state.get("finished_at")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="evaluation_results/swr_rot_temporal_diag")
    args = parser.parse_args()
    print(json.dumps(progress(Path(args.out))))
