from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any

from openai import OpenAI


SYSTEM_PROMPT = """You are grading a deep-memory retrieval answer.
Decide whether the candidate answer correctly answers the probe using the expected answer.
Allow paraphrases and extra harmless text. Reject answers that merely repeat the probe,
answer a different question, or contradict the expected answer.
Return JSON only: {"correct": true or false, "reason": "brief explanation"}."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add OpenAI judge labels to DMR JSONL results.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl")
    parser.add_argument("--model", default=os.getenv("OPENAI_JUDGE_MODEL"))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_latest_rows(path: Path, key: str) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    latest = {row[key]: row for row in rows}
    return [latest[value] for value in sorted(latest)]


def judge_answer(client: OpenAI, model: str, result: dict[str, Any]) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Probe: {result['probe']}\n"
                    f"Expected answer: {result['reference']}\n"
                    f"Candidate answer: {result['answer']}"
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    payload = json.loads(response.choices[0].message.content or "")
    if not isinstance(payload.get("correct"), bool):
        raise ValueError(f"Judge response is missing boolean correct: {payload!r}")
    return {
        "dataset_index": result["dataset_index"],
        "judge_model": model,
        "correct": payload["correct"],
        "reason": str(payload.get("reason", "")),
    }


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    summary = {
        "num_judged": len(rows),
        "llm_judge_accuracy": statistics.fmean(float(row["correct"]) for row in rows) if rows else None,
        "judge_model": rows[0]["judge_model"] if rows else None,
        "result_file": str(path),
    }
    path.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required to run the DMR judge.")
    if not args.model:
        raise SystemExit("Set OPENAI_JUDGE_MODEL or pass --model.")
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl) if args.output_jsonl else input_path.with_suffix(".judged.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_rows = [
        row for row in read_latest_rows(input_path, "dataset_index") if row["status"] == "ok"
    ]
    judged_rows = read_latest_rows(output_path, "dataset_index") if args.resume and output_path.exists() else []
    judged_by_index = {row["dataset_index"]: row for row in judged_rows}
    client = OpenAI(timeout=60.0)
    output_mode = "a" if args.resume else "w"
    with output_path.open(output_mode, encoding="utf-8") as output:
        for result in source_rows:
            index = result["dataset_index"]
            if index in judged_by_index:
                continue
            judged = judge_answer(client, args.model, result)
            judged_by_index[index] = judged
            output.write(json.dumps(judged, ensure_ascii=True) + "\n")
            output.flush()
            print(f"[{len(judged_by_index)}/{len(source_rows)}] index={index} correct={judged['correct']}")
    write_summary(output_path, [judged_by_index[index] for index in sorted(judged_by_index)])


if __name__ == "__main__":
    main()
