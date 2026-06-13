from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from nanomemgpt.training.formatting import iter_chat_sft_records
from nanomemgpt.trajectory.schema import TrajectoryStep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit teacher trajectory token lengths after student chat-template rendering."
    )
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--output-json")
    return parser.parse_args()


def load_steps(path: Path) -> list[TrajectoryStep]:
    return [
        TrajectoryStep.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def percentile(values: list[int], ratio: float) -> int:
    if not values:
        return 0
    return sorted(values)[round((len(values) - 1) * ratio)]


def summarize(values: list[int]) -> dict[str, float | int]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def main() -> None:
    args = parse_args()
    steps = load_steps(Path(args.trajectories))
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=not args.allow_download,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    action_counts = Counter(
        step.teacher_action.name if step.teacher_action else "target_text"
        for step in steps
    )
    prompt_lengths = []
    completion_lengths = []
    total_lengths = []
    longest: list[dict[str, Any]] = []
    for record in iter_chat_sft_records(steps, tokenizer):
        prompt_tokens = tokenizer(record["prompt"], add_special_tokens=False)["input_ids"]
        completion_tokens = tokenizer(record["completion"], add_special_tokens=False)["input_ids"]
        prompt_length = len(prompt_tokens)
        completion_length = len(completion_tokens)
        total_length = prompt_length + completion_length
        prompt_lengths.append(prompt_length)
        completion_lengths.append(completion_length)
        total_lengths.append(total_length)
        longest.append(
            {
                "sample_id": record["sample_id"],
                "prompt_tokens": prompt_length,
                "completion_tokens": completion_length,
                "total_tokens": total_length,
            }
        )

    summary = {
        "trajectory_file": args.trajectories,
        "model": args.model,
        "max_length": args.max_length,
        "num_steps": len(steps),
        "action_counts": dict(sorted(action_counts.items())),
        "prompt_tokens": summarize(prompt_lengths),
        "completion_tokens": summarize(completion_lengths),
        "total_tokens": summarize(total_lengths),
        "num_over_max_length": sum(length > args.max_length for length in total_lengths),
        "longest_samples": sorted(longest, key=lambda row: row["total_tokens"], reverse=True)[:10],
        "longest_sample_within_max_length": max(
            (row for row in longest if row["total_tokens"] <= args.max_length),
            key=lambda row: row["total_tokens"],
            default=None,
        ),
    }
    rendered = json.dumps(summary, ensure_ascii=True, indent=2)
    print(rendered)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"{rendered}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
