from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from datasets import load_from_disk
from openai import OpenAI

from nanomemgpt.eval.metrics import contains_reference, normalize_answer, rouge_l_recall

sys.path.append(str(Path(__file__).resolve().parent))
from vllm_tool_rescue_proxy import rescue_tool_call


QUERY_SYSTEM = """You are the search phase of a memory agent.
Your only job is to call conversation_search with one short literal query likely to occur
verbatim in prior conversation memory. Do not answer the user."""

ANSWER_SYSTEM = """Answer the user's memory question as Speaker 1.
Use only the provided retrieved conversation evidence. Return a short direct answer.
Do not describe the search process, evidence, memory system, or your reasoning. Do not call tools.
If the retrieved evidence is empty or insufficient, return UNKNOWN."""

CONVERSATION_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "conversation_search",
        "description": "Search prior conversation history using case-insensitive substring matching.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Short literal word or phrase to match as a substring.",
                },
                "roles": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["assistant", "user"]},
                },
                "limit": {"type": "integer"},
                "request_heartbeat": {"type": "boolean"},
                "thinking": {"type": "string"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate phase-routed DMR: query-only search phase + answer model."
    )
    parser.add_argument("--query-model", default="nano-memgpt-llama3-query-only-r16")
    parser.add_argument("--answer-model", default="nano-memgpt-llama3-r16")
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://llama-vllm:8000/v1"))
    parser.add_argument("--dataset-dir", default="data/raw/msc_dmr")
    parser.add_argument("--output-dir", default="data/evaluation/phase_routed_dmr")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-searches", type=int, default=3)
    parser.add_argument("--search-limit", type=int, default=10)
    parser.add_argument("--query-max-tokens", type=int, default=192)
    parser.add_argument("--answer-max-tokens", type=int, default=96)
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
                "role": "assistant" if is_speaker_one else "user",
                "speaker": "Speaker 1" if is_speaker_one else "Speaker 2",
                "content": utterance["text"].strip(),
            }
        )
    return turns


def history_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = []
    for session_number, session in enumerate([*row["previous_dialogs"], row["dialog"]], start=1):
        for turn_number, turn in enumerate(extract_turns(session), start=1):
            messages.append(
                {
                    "session": session_number,
                    "turn": turn_number,
                    **turn,
                }
            )
    return messages


def render_personas(row: dict[str, Any]) -> str:
    speaker_1 = "\n".join(f"- {fact}" for fact in row["init_personas"][0])
    speaker_2 = "\n".join(f"- {fact}" for fact in row["init_personas"][1])
    return f"Speaker 1 persona:\n{speaker_1}\n\nSpeaker 2 persona:\n{speaker_2}"


def local_conversation_search(
    messages: list[dict[str, Any]],
    query: str,
    *,
    roles: list[str] | None,
    limit: int,
) -> list[dict[str, Any]]:
    normalized_query = query.casefold().strip()
    if not normalized_query:
        return []
    allowed_roles = set(roles or [])
    results = []
    for message in messages:
        if allowed_roles and message["role"] not in allowed_roles:
            continue
        if normalized_query in message["content"].casefold():
            results.append(message)
    return results[-limit:]


def render_search_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No results found."
    lines = [f"Showing {len(results)} results:"]
    for index, result in enumerate(results, start=1):
        lines.append(
            f"[{index}] session={result['session']} turn={result['turn']} "
            f"role={result['role']} content={result['content']}"
        )
    return "\n".join(lines)


def parse_query_response(response: Any, request_body: dict[str, Any]) -> dict[str, Any] | None:
    message = response.choices[0].message
    tool_calls = message.tool_calls or []
    if tool_calls:
        call = tool_calls[0]
        if call.function.name != "conversation_search":
            return None
        try:
            arguments = json.loads(call.function.arguments)
        except json.JSONDecodeError:
            return None
        return arguments if isinstance(arguments, dict) else None

    content = message.content
    if not isinstance(content, str):
        return None
    rescued = rescue_tool_call(content, request_body)
    if rescued is None or rescued["function"]["name"] != "conversation_search":
        return None
    try:
        arguments = json.loads(rescued["function"]["arguments"])
    except json.JSONDecodeError:
        return None
    return arguments if isinstance(arguments, dict) else None


def build_query_messages(probe: str, query_trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": QUERY_SYSTEM},
        {
            "role": "user",
            "content": (
                "User memory question:\n"
                f"{probe}\n\n"
                "Call conversation_search with a short literal query. Prefer concrete words that may occur verbatim."
            ),
        },
    ]
    for step in query_trace:
        tool_call_id = step["tool_call_id"]
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": "conversation_search",
                            "arguments": json.dumps(step["arguments"], ensure_ascii=False),
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": "conversation_search",
                "content": step["tool_output"],
            }
        )
    return messages


def run_search_phase(
    client: OpenAI,
    args: argparse.Namespace,
    row: dict[str, Any],
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trace = []
    retrieved = []
    seen_queries = set()
    probe = row["self_instruct"]["B"]
    for _ in range(args.max_searches):
        request_messages = build_query_messages(probe, trace)
        request_body = {"tools": [CONVERSATION_SEARCH_TOOL]}
        response = client.chat.completions.create(
            model=args.query_model,
            messages=request_messages,
            tools=[CONVERSATION_SEARCH_TOOL],
            tool_choice={"type": "function", "function": {"name": "conversation_search"}},
            temperature=0,
            max_tokens=args.query_max_tokens,
        )
        arguments = parse_query_response(response, request_body)
        if not arguments:
            break
        query = str(arguments.get("query", "")).strip()
        if not query:
            break
        query_key = query.casefold()
        roles = arguments.get("roles")
        if not isinstance(roles, list) or not all(isinstance(role, str) for role in roles):
            roles = None
        limit = arguments.get("limit")
        if not isinstance(limit, int) or isinstance(limit, bool):
            limit = args.search_limit
        limit = max(1, min(limit, args.search_limit))
        results = local_conversation_search(messages, query, roles=roles, limit=limit)
        tool_output = render_search_results(results)
        step = {
            "tool_call_id": f"call_phase_{uuid.uuid4().hex[:12]}",
            "arguments": {
                "query": query,
                "roles": roles,
                "limit": limit,
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


def dedupe_retrieved(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for result in results:
        key = (result["session"], result["turn"], result["role"], result["content"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(result)
    return deduped


def render_answer_prompt(row: dict[str, Any], trace: list[dict[str, Any]], retrieved: list[dict[str, Any]]) -> str:
    if retrieved:
        evidence = "\n".join(
            f"[{index}] session={item['session']} turn={item['turn']} "
            f"{item['speaker']}: {item['content']}"
            for index, item in enumerate(retrieved, start=1)
        )
    else:
        evidence = "No retrieved evidence."
    queries = "\n".join(
        f"- {step['arguments']['query']} ({step['num_results']} results)"
        for step in trace
    ) or "- no query"
    return (
        f"User probe:\n{row['self_instruct']['B']}\n\n"
        f"Search queries issued:\n{queries}\n\n"
        f"Retrieved conversation evidence:\n{evidence}"
    )


def run_answer_phase(
    client: OpenAI,
    args: argparse.Namespace,
    row: dict[str, Any],
    trace: list[dict[str, Any]],
    retrieved: list[dict[str, Any]],
) -> str:
    response = client.chat.completions.create(
        model=args.answer_model,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": render_answer_prompt(row, trace, retrieved)},
        ],
        temperature=0,
        max_tokens=args.answer_max_tokens,
    )
    return response.choices[0].message.content or ""


def evaluate_case(client: OpenAI, args: argparse.Namespace, index: int, row: dict[str, Any]) -> dict[str, Any]:
    started_at = time.time()
    reference = row["self_instruct"]["A"]
    result: dict[str, Any] = {
        "protocol": "msc_dmr_phase_routed_query_only_search_answer_model_v1",
        "dataset_index": index,
        "query_model": args.query_model,
        "answer_model": args.answer_model,
        "probe": row["self_instruct"]["B"],
        "reference": reference,
        "max_searches": args.max_searches,
    }
    try:
        messages = history_messages(row)
        trace, retrieved = run_search_phase(client, args, row, messages)
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
        f"searches-{args.max_searches}-offset-{args.offset}-limit-{end - args.offset}.jsonl"
    )
    results = []
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
