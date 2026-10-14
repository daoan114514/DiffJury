"""Print safe progress metadata for a task-owned SWR background process."""
import argparse
import ctypes
import json
import re
from pathlib import Path
from ctypes import wintypes


def process_alive(pid: int) -> bool:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    code = wintypes.DWORD()
    try:
        return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
    finally:
        kernel.CloseHandle(handle)


def progress(out: Path) -> dict:
    state_path = out / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    process = json.loads((out / "process.json").read_text(encoding="utf-8"))
    lines = (out / "run.log").read_text(encoding="utf-8", errors="replace").splitlines()
    phase = "starting"
    for line in lines:
        if "holdout replay " in line:
            phase = line.split("holdout replay ", 1)[1].split(" (")[0]
        elif re.search(r"holdout (baseline|final-naive|final-governed):", line):
            phase = re.search(r"holdout (baseline|final-naive|final-governed):", line).group(1)
        elif "train reviews:" in line:
            phase = "train"
        elif "planned " in line and "events:" in line:
            phase = "events"
        elif "maintenance @" in line:
            phase = "maintenance"
        elif "recurrence " in line:
            phase = "recurrence"
        elif "stopped:" in line:
            phase = "stopped"
        elif "done:" in line:
            phase = "done"
    command = process["command"]
    pending = state.get("pending_event")
    interval = state.get("run_configuration", {}).get("review_every", 0)
    if pending and interval and pending % interval == 0 and not any(
            row["event"] == pending for row in state.get("attribution", [])):
        phase = "attribution"
    if phase == "maintenance" and any(row["event"] == state.get("pending_event", state.get("next_event"))
                                       for row in state.get("maintenance", [])):
        phase = "events"
    if phase == "maintenance" and not state.get("pending_event") and state.get("next_event", 0) < state.get("events_total", 0):
        phase = "events"
    root = Path(__file__).resolve().parents[1]
    configuration = state.get("run_configuration") or {}
    dataset_arg = command[command.index("--dataset") + 1]
    dataset = Path(dataset_arg) if Path(dataset_arg).is_absolute() else root / dataset_arg
    cases = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    held = sorted((case for case in cases if case["id"] in configuration.get("holdout_ids", [])), key=lambda case: case["id"])
    heads = {case["head_commit"]: position for position, case in enumerate(held)}
    worktrees = Path(command[command.index("--worktrees") + 1])
    positions, active = [], 0
    clones = (root / "evaluation_data/swrbench/repos").resolve()
    for directory in worktrees.glob("*"):
        if not directory.is_dir():
            continue
        active += 1
        try:
            marker = (directory / ".git").read_text(encoding="utf-8").strip()
            registry = Path(marker.partition(": ")[2]).resolve()
            if not registry.is_relative_to(clones):
                continue
            head = (registry / "HEAD").read_text(encoding="utf-8").strip()
            if head in heads:
                positions.append(heads[head])
        except OSError:
            pass
    return {"pid": process["pid"], "alive": process_alive(process["pid"]), "phase": phase,
            "train_cached": len(list((out / "train_reviews").glob("*.json"))),
            "train_total": state.get("dataset", {}).get("train"),
            "holdout_total": len(held), "holdout_started_at_least": max(positions, default=-1) + 1,
            "active_worktrees": active, "next_event": state.get("next_event"),
            "events_total": state.get("events_total"), "finished_at": state.get("finished_at"),
            "recurrence_cached": {name: len(list((out / "recurrence_rows" / name).glob("*.json")))
                                  for name in ("baseline", "naive", "governed", "governed_canary")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="evaluation_results/swr_rot_temporal")
    args = parser.parse_args()
    print(json.dumps(progress(Path(args.out))))
