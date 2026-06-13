from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nanomemgpt.trajectory.schema import FunctionCall, TrajectoryStep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export context-complete SFT trajectory steps from captured teacher provider traces."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", default="data/trajectories/gpt4_memgpt_train.jsonl")
    parser.add_argument("--split", default="train")
    return parser.parse_args()


def read_latest_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    latest = {row["dataset_index"]: row for row in rows}
    return [latest[index] for index in sorted(latest)]


def parse_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str):
        return {"_raw": value}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {"_raw": value}
    return parsed if isinstance(parsed, dict) else {"_raw": parsed}


def tool_returns_by_id(messages: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        message["tool_call_id"]: {
            "status": message.get("status"),
            "tool_return": message.get("tool_return"),
        }
        for message in messages
        if message.get("message_type") == "tool_return_message" and message.get("tool_call_id")
    }


def iter_steps(row: dict[str, Any], split: str) -> list[TrajectoryStep]:
    returns_by_id = tool_returns_by_id(row.get("raw_messages", []))
    steps = []
    for step_index, trace in enumerate(row.get("provider_traces", [])):
        request = trace.get("request_json") or {}
        response = trace.get("response_json") or {}
        choices = response.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        for call_index, call in enumerate(message.get("tool_calls") or []):
            function = call.get("function") or {}
            arguments = parse_arguments(function.get("arguments"))
            request_heartbeat = bool(arguments.pop("request_heartbeat", False))
            steps.append(
                TrajectoryStep(
                    sample_id=f"dmr-{row['dataset_index']}-step-{step_index}-call-{call_index}",
                    split=split,
                    step_index=step_index,
                    context=request.get("messages") or [],
                    teacher_action=FunctionCall(
                        name=function.get("name") or "",
                        arguments=arguments,
                        request_heartbeat=request_heartbeat,
                    ),
                    function_output=returns_by_id.get(call.get("id")),
                    source="teacher",
                )
            )
    return steps


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = read_latest_rows(Path(args.input_jsonl))
    steps = [
        step
        for row in results
        if row.get("status") == "ok"
        for step in iter_steps(row, args.split)
    ]
    with output_path.open("w", encoding="utf-8") as output:
        for step in steps:
            output.write(step.model_dump_json() + "\n")
    summary = {
        "num_input_rows": len(results),
        "num_completed_rows": sum(row.get("status") == "ok" for row in results),
        "num_trajectory_steps": len(steps),
        "result_file": str(output_path),
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
