from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import uuid
from pathlib import Path
from typing import Any

from datasets import load_from_disk
from openai import OpenAI

from eval_phase_routed_dmr import (
    dedupe_retrieved,
    history_messages,
    local_conversation_search,
    render_search_results,
    run_answer_phase,
    slugify,
)
from nanomemgpt.eval.metrics import contains_reference, normalize_answer, rouge_l_recall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay GPT teacher conversation_search queries through the local skeleton recall path."
    )
    parser.add_argument("--answer-model", default="nano-memgpt-llama3-r16")
    parser.add_argument("--base-url", default="http://llama-vllm:8000/v1")
    parser.add_argument("--dataset-dir", default="data/raw/msc_dmr")
    parser.add_argument("--teacher-traces", default="data/trajectories/gpt41_paper_substring_scaled_approved_oracle.jsonl")
    parser.add_argument("--output-dir", default="data/evaluation/teacher_query_skeleton_dmr")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-searches", type=int)
    parser.add_argument("--search-limit", type=int, default=10)
    parser.add_argument("--answer-max-tokens", type=int, default=96)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_teacher_traces(path: str) -> dict[int, dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {row["dataset_index"]: row for row in rows}


def search_steps(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        step for step in trace.get("teacher_steps", [])
        if step.get("name") == "conversation_search"
        and isinstance(step.get("arguments"), dict)
        and isinstance(step["arguments"].get("query"), str)
        and step["arguments"]["query"].strip()
    ]


def run_teacher_search(
    args: argparse.Namespace,
    trace: dict[str, Any],
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    replay_trace: list[dict[str, Any]] = []
    retrieved: list[dict[str, Any]] = []
    steps = search_steps(trace)
    if args.max_searches is not None:
        steps = steps[: args.max_searches]
    for step in steps:
        arguments = step["arguments"]
        query = arguments["query"].strip()
        roles = arguments.get("roles")
        if not isinstance(roles, list) or not all(isinstance(role, str) for role in roles):
            roles = ["assistant", "user"]
        limit = arguments.get("limit")
        if not isinstance(limit, int) or isinstance(limit, bool):
            limit = args.search_limit
        limit = max(1, min(limit, args.search_limit))
        results = local_conversation_search(messages, query, roles=roles, limit=limit)
        tool_output = render_search_results(results)
        replay_step = {
            "tool_call_id": step.get("tool_call_id") or f"call_teacher_{uuid.uuid4().hex[:12]}",
            "arguments": {
                "query": query,
                "roles": roles,
                "limit": limit,
                "request_heartbeat": bool(arguments.get("request_heartbeat", True)),
            },
            "teacher_tool_return": step.get("tool_return"),
            "tool_output": tool_output,
            "num_results": len(results),
        }
        replay_trace.append(replay_step)
        retrieved.extend(results)
    return replay_trace, retrieved


def evaluate_case(
    client: OpenAI,
    args: argparse.Namespace,
    index: int,
    row: dict[str, Any],
    teacher_trace: dict[str, Any],
) -> dict[str, Any]:
    started_at = time.time()
    reference = row["self_instruct"]["A"]
    result: dict[str, Any] = {
        "protocol": "msc_dmr_teacher_query_skeleton_answer_model_v1",
        "dataset_index": index,
        "teacher_model": teacher_trace.get("teacher_model"),
        "answer_model": args.answer_model,
        "probe": row["self_instruct"]["B"],
        "reference": reference,
        "teacher_status": teacher_trace.get("status"),
        "teacher_had_search": bool(search_steps(teacher_trace)),
    }
    try:
        messages = history_messages(row)
        trace, retrieved = run_teacher_search(args, teacher_trace, messages)
        retrieved = dedupe_retrieved(retrieved)
        answer = run_answer_phase(client, args, row, trace, retrieved)
        result.update(
            {
                "status": "ok",
                "answer": answer,
                "normalized_answer": normalize_answer(answer),
                "rouge_l_recall": rouge_l_recall(answer, reference),
                "contains_reference": contains_reference(answer, reference),
                "search_trace": trace,
                "num_searches": len(trace),
                "num_retrieved": len(retrieved),
                "retrieved_contains_reference": any(
                    normalize_answer(reference) in normalize_answer(item["content"])
                    for item in retrieved
                ),
                "retrieved_evidence": retrieved,
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


def summarize_group(results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [result for result in results if result["status"] == "ok"]
    return {
        "num_results": len(results),
        "num_completed": len(completed),
        "num_errors": len(results) - len(completed),
        "mean_rouge_l_recall": statistics.fmean(result["rouge_l_recall"] for result in completed) if completed else None,
        "contains_reference_accuracy": (
            statistics.fmean(float(result["contains_reference"]) for result in completed) if completed else None
        ),
        "retrieved_contains_reference_rate": (
            statistics.fmean(float(result["retrieved_contains_reference"]) for result in completed) if completed else None
        ),
        "mean_num_searches": statistics.fmean(result["num_searches"] for result in completed) if completed else None,
        "mean_num_retrieved": statistics.fmean(result["num_retrieved"] for result in completed) if completed else None,
    }


def write_summary(path: Path, results: list[dict[str, Any]], *, emit: bool = True) -> None:
    with_search = [result for result in results if result.get("teacher_had_search")]
    no_search = [result for result in results if not result.get("teacher_had_search")]
    summary = {
        "protocol": results[0]["protocol"] if results else None,
        "all_approved": summarize_group(results),
        "teacher_search_subset": summarize_group(with_search),
        "teacher_no_search_subset": summarize_group(no_search),
        "result_file": str(path),
    }
    path.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n")
    if emit:
        print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    dataset = load_from_disk(args.dataset_dir)["train"]
    teacher_traces = read_teacher_traces(args.teacher_traces)
    end = min(args.offset + args.limit, len(dataset))
    candidate_indices = [
        index for index in sorted(teacher_traces)
        if args.offset <= index < end
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / (
        f"teacher-query-to-{slugify(args.answer_model)}-"
        f"skeleton-offset-{args.offset}-limit-{end - args.offset}.jsonl"
    )
    results: list[dict[str, Any]] = []
    if args.resume and result_path.exists():
        results = [
            json.loads(line)
            for line in result_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    results_by_index = {result["dataset_index"]: result for result in results}
    completed_indices = {index for index, result in results_by_index.items() if result["status"] == "ok"}
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", max_retries=0)
    output_mode = "a" if args.resume else "w"
    with result_path.open(output_mode, encoding="utf-8") as output:
        for index in candidate_indices:
            if index in completed_indices:
                continue
            result = evaluate_case(client, args, index, dataset[index], teacher_traces[index])
            results_by_index[index] = result
            output.write(json.dumps(result, ensure_ascii=True) + "\n")
            output.flush()
            ordered = [results_by_index[key] for key in sorted(results_by_index)]
            write_summary(result_path, ordered, emit=False)
            print(
                f"[{len(results_by_index)}/{len(candidate_indices)}] index={index} "
                f"teacher_search={result.get('teacher_had_search')} status={result['status']} "
                f"searches={result.get('num_searches')} retrieved={result.get('num_retrieved')} "
                f"contains={result.get('contains_reference')} answer={result.get('answer', '')!r}"
            )
    write_summary(result_path, [results_by_index[key] for key in sorted(results_by_index)])


if __name__ == "__main__":
    main()
