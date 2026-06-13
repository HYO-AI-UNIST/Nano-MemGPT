from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any

from datasets import load_from_disk
from openai import OpenAI

from nanomemgpt.eval.metrics import contains_reference, normalize_answer, rouge_l_recall


SYSTEM_PROMPT = """Answer the user's memory question as Speaker 1.
Use only the provided oracle evidence and persona facts. Return a short direct answer.
Do not describe the evidence, memory system, or your reasoning. Do not call tools."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DMR answers with injected oracle evidence.")
    parser.add_argument("--model", required=True, help="Direct vLLM served model ID.")
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://llama-vllm:8000/v1"))
    parser.add_argument("--dataset-dir", default="data/raw/msc_dmr")
    parser.add_argument("--output-dir", default="data/evaluation/oracle_dmr")
    parser.add_argument("--oracle-mode", choices=["full_history", "teacher_trace"], required=True)
    parser.add_argument("--teacher-traces")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def extract_turns(session: Any) -> list[dict[str, str]]:
    dialog = session["dialog"] if isinstance(session, dict) else session
    turns = []
    for index, utterance in enumerate(dialog):
        speaker_id = utterance.get("id")
        is_speaker_one = speaker_id == "Speaker 1" if speaker_id else index % 2 == 0
        turns.append(
            {
                "speaker": "Speaker 1" if is_speaker_one else "Speaker 2",
                "content": utterance["text"].strip(),
            }
        )
    return turns


def render_personas(row: dict[str, Any]) -> str:
    sections = []
    for speaker, facts in zip(("Speaker 1", "Speaker 2"), row["init_personas"]):
        sections.append(f"{speaker} persona:\n" + "\n".join(f"- {fact}" for fact in facts))
    return "\n\n".join(sections)


def render_full_history(row: dict[str, Any]) -> str:
    sessions = [*row["previous_dialogs"], row["dialog"]]
    blocks = []
    for session_number, session in enumerate(sessions, start=1):
        lines = [
            f"{turn['speaker']}: {turn['content']}"
            for turn in extract_turns(session)
        ]
        blocks.append(f"Historical session {session_number}:\n" + "\n".join(lines))
    return "\n\n".join(blocks)


def read_teacher_traces(path: str | None) -> dict[int, dict[str, Any]]:
    if not path:
        return {}
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {row["dataset_index"]: row for row in rows}


def render_teacher_trace(trace: dict[str, Any]) -> str:
    blocks = []
    for step_number, step in enumerate(trace["teacher_steps"], start=1):
        blocks.append(
            f"Oracle step {step_number}: {step['name']}\n"
            f"Arguments: {json.dumps(step['arguments'], ensure_ascii=False)}\n"
            f"Output: {step.get('tool_return')}"
        )
    return "\n\n".join(blocks)


def build_prompt(
    row: dict[str, Any],
    oracle_mode: str,
    teacher_trace: dict[str, Any] | None,
) -> str:
    if oracle_mode == "full_history":
        evidence = render_full_history(row)
        evidence_label = "Complete historical conversation oracle"
    else:
        if teacher_trace is None:
            raise ValueError("Teacher trace is missing for this dataset index.")
        if teacher_trace["status"] != "ok":
            raise ValueError("Teacher trace was not completed successfully.")
        evidence = render_teacher_trace(teacher_trace)
        evidence_label = "GPT teacher MemGPT call/output oracle"
    return (
        f"{render_personas(row)}\n\n"
        f"{evidence_label}:\n{evidence}\n\n"
        f"User probe: {row['self_instruct']['B']}"
    )


def evaluate_case(
    client: OpenAI,
    args: argparse.Namespace,
    index: int,
    row: dict[str, Any],
    teacher_trace: dict[str, Any] | None,
) -> dict[str, Any]:
    started_at = time.time()
    result: dict[str, Any] = {
        "protocol": f"msc_dmr_oracle_{args.oracle_mode}_v1",
        "dataset_index": index,
        "model": args.model,
        "oracle_mode": args.oracle_mode,
        "probe": row["self_instruct"]["B"],
        "reference": row["self_instruct"]["A"],
    }
    try:
        prompt = build_prompt(row, args.oracle_mode, teacher_trace)
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=args.max_tokens,
        )
        answer = response.choices[0].message.content or ""
        reference = row["self_instruct"]["A"]
        result.update(
            {
                "status": "ok",
                "answer": answer,
                "normalized_answer": normalize_answer(answer),
                "rouge_l_recall": rouge_l_recall(answer, reference),
                "contains_reference": contains_reference(answer, reference),
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    result["elapsed_seconds"] = round(time.time() - started_at, 3)
    return result


def write_summary(path: Path, results: list[dict[str, Any]], *, emit: bool = True) -> None:
    completed = [result for result in results if result["status"] == "ok"]
    summary = {
        "protocol": results[0]["protocol"] if results else None,
        "num_results": len(results),
        "num_completed": len(completed),
        "num_errors": len(results) - len(completed),
        "mean_rouge_l_recall": (
            statistics.fmean(result["rouge_l_recall"] for result in completed) if completed else None
        ),
        "contains_reference_accuracy": (
            statistics.fmean(float(result["contains_reference"]) for result in completed) if completed else None
        ),
        "result_file": str(path),
    }
    path.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n")
    if emit:
        print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    if args.oracle_mode == "teacher_trace" and not args.teacher_traces:
        raise SystemExit("--teacher-traces is required for teacher_trace mode.")
    dataset = load_from_disk(args.dataset_dir)["train"]
    end = min(args.offset + args.limit, len(dataset))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / (
        f"{slugify(args.model)}-{args.oracle_mode}-offset-{args.offset}-limit-{end - args.offset}.jsonl"
    )
    results = []
    if args.resume and result_path.exists():
        results = [
            json.loads(line)
            for line in result_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    results_by_index = {result["dataset_index"]: result for result in results}
    completed_indices = {
        index for index, result in results_by_index.items() if result["status"] == "ok"
    }
    teacher_traces = read_teacher_traces(args.teacher_traces)
    candidate_indices = (
        [
            index
            for index in sorted(teacher_traces)
            if args.offset <= index < end
        ]
        if args.oracle_mode == "teacher_trace"
        else list(range(args.offset, end))
    )
    client = OpenAI(base_url=args.base_url, api_key=os.getenv("VLLM_API_KEY", "EMPTY"), max_retries=0)
    output_mode = "a" if args.resume else "w"
    with result_path.open(output_mode, encoding="utf-8") as output:
        pending_indices = [
            index for index in candidate_indices if index not in completed_indices
        ]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    evaluate_case,
                    client,
                    args,
                    index,
                    dataset[index],
                    teacher_traces.get(index),
                ): index
                for index in pending_indices
            }
            for future in as_completed(futures):
                index = futures[future]
                result = future.result()
                results_by_index[index] = result
                output.write(json.dumps(result, ensure_ascii=True) + "\n")
                output.flush()
                ordered = [results_by_index[key] for key in sorted(results_by_index)]
                write_summary(result_path, ordered, emit=False)
                print(
                    f"[{len(results_by_index)}/{len(candidate_indices)}] index={index} "
                    f"status={result['status']} answer={result.get('answer', '')!r}"
                )
    write_summary(result_path, [results_by_index[key] for key in sorted(results_by_index)])


if __name__ == "__main__":
    main()
