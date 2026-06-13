from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract replayable teacher tool-call sequences from DMR evaluation JSONL."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", default="data/trajectories/dmr_teacher_oracle_traces.jsonl")
    return parser.parse_args()


def read_latest_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    latest = {row["dataset_index"]: row for row in rows}
    return [latest[index] for index in sorted(latest)]


def extract_steps(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    returns_by_id = {
        message["tool_call_id"]: message
        for message in messages
        if message.get("message_type") == "tool_return_message" and message.get("tool_call_id")
    }
    steps = []
    for message in messages:
        if message.get("message_type") != "tool_call_message":
            continue
        call = message.get("tool_call")
        if not call:
            continue
        arguments = call.get("arguments")
        try:
            parsed_arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError:
            parsed_arguments = {"_raw": arguments}
        returned = returns_by_id.get(call.get("tool_call_id"), {})
        steps.append(
            {
                "name": call.get("name"),
                "arguments": parsed_arguments,
                "tool_call_id": call.get("tool_call_id"),
                "status": returned.get("status"),
                "tool_return": returned.get("tool_return"),
            }
        )
    return steps


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    completed = [row for row in rows if row["status"] == "ok"]
    search_counts = [
        sum(step["name"] == "conversation_search" for step in row["teacher_steps"])
        for row in completed
    ]
    summary = {
        "num_results": len(rows),
        "num_completed": len(completed),
        "num_errors": len(rows) - len(completed),
        "num_with_conversation_search": sum(count > 0 for count in search_counts),
        "mean_conversation_search_calls": statistics.fmean(search_counts) if search_counts else None,
        "result_file": str(path),
    }
    path.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with output_path.open("w", encoding="utf-8") as output:
        for result in read_latest_rows(input_path):
            row = {
                "dataset_index": result["dataset_index"],
                "teacher_model": result["model"],
                "teacher_protocol": result["protocol"],
                "probe": result["probe"],
                "reference": result["reference"],
                "status": result["status"],
                "teacher_answer": result.get("answer", ""),
                "teacher_steps": extract_steps(result.get("raw_messages", [])),
            }
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=True) + "\n")
    write_summary(output_path, rows)


if __name__ == "__main__":
    main()
