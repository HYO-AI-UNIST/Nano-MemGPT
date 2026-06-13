from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
import uuid
from pathlib import Path
from typing import Any

from datasets import load_from_disk
from openai import OpenAI

from eval_phase_routed_dmr import (
    ANSWER_SYSTEM,
    dedupe_retrieved,
    history_messages,
    local_conversation_search,
    render_answer_prompt,
    render_search_results,
    run_answer_phase,
    slugify,
)
from nanomemgpt.eval.metrics import contains_reference, normalize_answer, rouge_l_recall


QUERY_SKELETON_SYSTEM = """You are the query generator inside a memory-search controller.
Generate exactly one short literal substring query for conversation_search.
The query should be words likely to appear verbatim in prior conversation memory.
Do not answer the user. Do not explain. Do not write JSON unless unavoidable.
Return only the query text."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DMR with deterministic tool skeleton: model emits query text only."
    )
    parser.add_argument("--query-model", default="nano-memgpt-llama3-query-only-r16")
    parser.add_argument("--answer-model", default="nano-memgpt-llama3-r16")
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://llama-vllm:8000/v1"))
    parser.add_argument("--dataset-dir", default="data/raw/msc_dmr")
    parser.add_argument("--output-dir", default="data/evaluation/query_skeleton_dmr")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-searches", type=int, default=3)
    parser.add_argument("--search-limit", type=int, default=10)
    parser.add_argument("--query-max-tokens", type=int, default=64)
    parser.add_argument("--answer-max-tokens", type=int, default=96)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def strip_code_fence(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|text)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def find_json_query(value: Any) -> str | None:
    if isinstance(value, dict):
        query = value.get("query")
        if isinstance(query, str) and query.strip():
            return query.strip()
        arguments = value.get("arguments")
        if isinstance(arguments, dict):
            query = arguments.get("query")
            if isinstance(query, str) and query.strip():
                return query.strip()
    return None


def parse_query_text(content: str | None) -> str | None:
    if not isinstance(content, str):
        return None
    text = strip_code_fence(content)
    if not text:
        return None

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    query = find_json_query(parsed)
    if query:
        return sanitize_query(query)

    match = re.search(r'"query"\s*:\s*"([^"]+)"', text)
    if match:
        return sanitize_query(match.group(1))

    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    first_line = re.sub(r"^(?:query|search query|conversation_search)\s*[:=]\s*", "", first_line, flags=re.IGNORECASE)
    return sanitize_query(first_line)


def sanitize_query(query: str) -> str | None:
    cleaned = query.strip().strip("'\"`")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None
    if len(cleaned) > 120:
        cleaned = cleaned[:120].rsplit(" ", 1)[0].strip() or cleaned[:120].strip()
    return cleaned


def build_query_prompt(probe: str, trace: list[dict[str, Any]]) -> str:
    if trace:
        previous = []
        for step in trace:
            previous.append(
                f"- query: {step['arguments']['query']}\n"
                f"  results: {step['num_results']}\n"
                f"  output: {step['tool_output'][:600]}"
            )
        previous_block = "\n".join(previous)
    else:
        previous_block = "No previous searches."
    return (
        f"User memory question:\n{probe}\n\n"
        f"Previous searches:\n{previous_block}\n\n"
        "Generate the next single literal substring query. "
        "Prefer concrete names, objects, foods, places, jobs, music terms, hobbies, or quoted phrases. "
        "If a previous broad query failed, choose a different narrower phrase."
    )


def run_skeleton_search_phase(
    client: OpenAI,
    args: argparse.Namespace,
    row: dict[str, Any],
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trace: list[dict[str, Any]] = []
    retrieved: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    probe = row["self_instruct"]["B"]
    for _ in range(args.max_searches):
        response = client.chat.completions.create(
            model=args.query_model,
            messages=[
                {"role": "system", "content": QUERY_SKELETON_SYSTEM},
                {"role": "user", "content": build_query_prompt(probe, trace)},
            ],
            temperature=0,
            max_tokens=args.query_max_tokens,
        )
        raw_query = response.choices[0].message.content or ""
        query = parse_query_text(raw_query)
        if not query:
            break
        query_key = query.casefold()
        results = local_conversation_search(messages, query, roles=["assistant", "user"], limit=args.search_limit)
        tool_output = render_search_results(results)
        step = {
            "tool_call_id": f"call_skeleton_{uuid.uuid4().hex[:12]}",
            "raw_query_output": raw_query,
            "arguments": {
                "query": query,
                "roles": ["assistant", "user"],
                "limit": args.search_limit,
                "request_heartbeat": True,
            },
            "tool_output": tool_output,
            "num_results": len(results),
        }
        trace.append(step)
        retrieved.extend(results)
        if query_key in seen_queries:
            break
        seen_queries.add(query_key)
    return trace, retrieved


def evaluate_case(client: OpenAI, args: argparse.Namespace, index: int, row: dict[str, Any]) -> dict[str, Any]:
    started_at = time.time()
    reference = row["self_instruct"]["A"]
    result: dict[str, Any] = {
        "protocol": "msc_dmr_query_skeleton_search_answer_model_v1",
        "dataset_index": index,
        "query_model": args.query_model,
        "answer_model": args.answer_model,
        "probe": row["self_instruct"]["B"],
        "reference": reference,
        "max_searches": args.max_searches,
    }
    try:
        messages = history_messages(row)
        trace, retrieved = run_skeleton_search_phase(client, args, row, messages)
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


def write_summary(path: Path, results: list[dict[str, Any]], *, emit: bool = True) -> None:
    completed = [result for result in results if result["status"] == "ok"]
    summary = {
        "protocol": results[0]["protocol"] if results else None,
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
        "result_file": str(path),
    }
    path.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n")
    if emit:
        print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    dataset = load_from_disk(args.dataset_dir)["train"]
    end = min(args.offset + args.limit, len(dataset))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / (
        f"{slugify(args.query_model)}-to-{slugify(args.answer_model)}-"
        f"skeleton-searches-{args.max_searches}-offset-{args.offset}-limit-{end - args.offset}.jsonl"
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
    client = OpenAI(base_url=args.base_url, api_key=os.getenv("VLLM_API_KEY", "EMPTY"), max_retries=0)
    output_mode = "a" if args.resume else "w"
    with result_path.open(output_mode, encoding="utf-8") as output:
        for index in range(args.offset, end):
            if index in completed_indices:
                continue
            result = evaluate_case(client, args, index, dataset[index])
            results_by_index[index] = result
            output.write(json.dumps(result, ensure_ascii=True) + "\n")
            output.flush()
            results = [results_by_index[key] for key in sorted(results_by_index)]
            write_summary(result_path, results, emit=False)
            print(
                f"[{len(results_by_index)}/{end - args.offset}] index={index} "
                f"status={result['status']} searches={result.get('num_searches')} "
                f"retrieved={result.get('num_retrieved')} contains={result.get('contains_reference')} "
                f"answer={result.get('answer', '')!r}"
            )
    write_summary(result_path, [results_by_index[key] for key in sorted(results_by_index)])


if __name__ == "__main__":
    main()
