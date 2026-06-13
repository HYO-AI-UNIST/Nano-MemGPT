from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep only completed teacher trajectories approved by the DMR answer judge."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--judge-jsonl", required=True)
    parser.add_argument("--output-jsonl", default="data/trajectories/gpt41_memgpt_approved_rows.jsonl")
    return parser.parse_args()


def read_latest_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    latest = {row["dataset_index"]: row for row in rows}
    return [latest[index] for index in sorted(latest)]


def main() -> None:
    args = parse_args()
    input_rows = read_latest_rows(Path(args.input_jsonl))
    judge_by_index = {
        row["dataset_index"]: row for row in read_latest_rows(Path(args.judge_jsonl))
    }
    approved = [
        row
        for row in input_rows
        if row.get("status") == "ok"
        and judge_by_index.get(row["dataset_index"], {}).get("correct") is True
    ]
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for row in approved:
            output.write(json.dumps(row, ensure_ascii=True) + "\n")
    summary = {
        "num_input_rows": len(input_rows),
        "num_judged_rows": len(judge_by_index),
        "num_approved_rows": len(approved),
        "num_rejected_or_unjudged_rows": len(input_rows) - len(approved),
        "result_file": str(output_path),
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
