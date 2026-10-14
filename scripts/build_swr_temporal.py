"""Create the temporal dataset without modifying the original JSONL."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffjury.evaluation_harness import load_jsonl  # noqa: E402
from diffjury.evaluation_temporal import temporal_split  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(ROOT / "evaluation_data/swrbench/swr_bench_functional.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "evaluation_data/swrbench/swr_bench_temporal.jsonl"))
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.out)
    if source.resolve() == output.resolve():
        parser.error("temporal output must differ from the original dataset")
    rows, summary = temporal_split(load_jsonl(str(source)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary.update({"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "dataset_sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
    output.with_name(output.stem + "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"cases": summary["cases"], "by_split": summary["by_split"]}))


if __name__ == "__main__":
    main()
