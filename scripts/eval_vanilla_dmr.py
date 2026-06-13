from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import psycopg
import requests
from datasets import load_from_disk
from letta_client import Letta

from nanomemgpt.eval.letta_contracts import configure_dmr_recall_contract
from nanomemgpt.eval.metrics import contains_reference, normalize_answer, rouge_l_recall


DMR_PERSONA_PREFIX = """The following is information about myself. My task is to completely
immerse myself in this role. I should never say that I am an AI, and should reply as if I am
playing this role. If the user asks me a question, I should reply with a best guess using the
information in core memory and conversation search."""
RECALL_COUNT_PATTERN = re.compile(
    r"- (?P<count>\d+) previous messages between you and the user are stored in recall memory"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Vanilla MemGPT model on the MSC deep-memory retrieval task."
    )
    parser.add_argument("--model", required=True, help="Letta model handle, for example vllm/org/model.")
    parser.add_argument("--model-source-note", default="", help="Provenance note stored with every result.")
    parser.add_argument("--embedding", default="letta/letta-free")
    parser.add_argument("--letta-base-url", default=os.getenv("LETTA_BASE_URL", "http://letta-server:8283"))
    parser.add_argument("--dataset-dir", default="data/raw/msc_dmr")
    parser.add_argument("--output-dir", default="data/evaluation/vanilla_dmr")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--infrastructure-retries", type=int, default=2)
    parser.add_argument(
        "--infrastructure-retry-delay-seconds",
        type=float,
        default=1,
        help="Base delay before retrying infrastructure errors. Use at least 65 for low-TPM API tiers.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--row-delay-seconds",
        type=float,
        default=0,
        help="Delay between sequential rows. Use at least 65 for low-TPM API tiers.",
    )
    parser.add_argument("--keep-agents", action="store_true")
    parser.add_argument(
        "--recall-search-contract",
        choices=["paper_substring"],
        default="paper_substring",
        help="Tool schema exposed to the model. paper_substring matches the paper-era DMR recall implementation.",
    )
    parser.add_argument(
        "--capture-provider-traces",
        action="store_true",
        help="Checkpoint full LLM request/response traces for teacher-trajectory export.",
    )
    parser.add_argument(
        "--teacher-query-hints",
        help="Approved teacher oracle JSONL. When set, teacher conversation_search queries are added as hints.",
    )
    parser.add_argument(
        "--teacher-query-policy",
        choices=["none", "first", "all"],
        default="none",
        help="How many teacher conversation_search queries to expose as hints in the probe message.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def extract_turns(session: Any) -> list[dict[str, str]]:
    dialog = session["dialog"] if isinstance(session, dict) else session
    turns = []
    for index, utterance in enumerate(dialog):
        text = utterance["text"].strip()
        speaker_id = utterance.get("id")
        is_speaker_one = speaker_id == "Speaker 1" if speaker_id else index % 2 == 0
        turns.append({"role": "assistant" if is_speaker_one else "user", "content": text})
    return turns


def dump_message(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        return message.model_dump(mode="json")
    if isinstance(message, dict):
        return message
    return {"repr": repr(message)}


def tool_trace(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trace = []
    for message in messages:
        tool_call = message.get("tool_call")
        if not tool_call:
            continue
        trace.append(
            {
                "name": tool_call.get("name"),
                "arguments": tool_call.get("arguments"),
                "tool_call_id": tool_call.get("tool_call_id"),
            }
        )
    return trace


def extract_answer(messages: list[dict[str, Any]]) -> str:
    answers = [
        message.get("content", "")
        for message in messages
        if message.get("message_type") == "assistant_message" and isinstance(message.get("content"), str)
    ]
    if answers:
        return answers[-1]
    for call in reversed(tool_trace(messages)):
        if call["name"] != "send_message":
            continue
        try:
            arguments = json.loads(call["arguments"])
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(arguments.get("message"), str):
            return arguments["message"]
    return ""


def classify_failure(answer: str, reference: str, trace: list[dict[str, Any]]) -> list[str]:
    labels = []
    names = [call["name"] for call in trace]
    searched = "conversation_search" in names
    if not searched:
        labels.append("retrieval_miss_candidate")
    if searched and not answer:
        labels.append("chain_failure_candidate")
    for call in trace:
        if call["name"] != "conversation_search":
            continue
        try:
            arguments = json.loads(call["arguments"])
        except (TypeError, json.JSONDecodeError):
            labels.append("retrieval_hallucination_candidate")
            break
        if not arguments.get("query"):
            labels.append("retrieval_hallucination_candidate")
            break
    if answer and not contains_reference(answer, reference):
        labels.append("incorrect_answer")
    return labels


def fetch_provider_traces(agent_id: str) -> list[dict[str, Any]]:
    pg_uri = os.getenv("LETTA_PG_URI", "postgresql://nanomemgpt:nanomemgpt@pgvector:5432/nanomemgpt")
    with psycopg.connect(pg_uri) as connection:
        rows = connection.execute(
            """
            SELECT request_json, response_json, step_id, run_id, call_type, source, created_at
            FROM provider_traces
            WHERE agent_id = %s
            ORDER BY created_at
            """,
            (agent_id,),
        ).fetchall()
    return [
        {
            "request_json": row[0],
            "response_json": row[1],
            "step_id": row[2],
            "run_id": row[3],
            "call_type": row[4],
            "source": row[5],
            "created_at": row[6].isoformat(),
        }
        for row in rows
    ]


def fetch_provider_responses(agent_id: str) -> list[dict[str, Any]]:
    return [
        trace["response_json"]
        for trace in fetch_provider_traces(agent_id)
        if isinstance(trace.get("response_json"), dict)
    ]


def extract_provider_contents(provider_responses: list[dict[str, Any]]) -> list[str]:
    contents = []
    for response in provider_responses:
        for choice in response.get("choices", []):
            content = choice.get("message", {}).get("content")
            if isinstance(content, str):
                contents.append(content)
    return contents


def classify_rejected_response(error: str, provider_contents: list[str]) -> list[str]:
    labels = []
    if "No tool calls found in response" in error:
        labels.append("tool_call_format_failure")
    if any(
        "(send_message," in content
        or "(core_memory_search," in content
        or "<function name=" in content
        for content in provider_contents
    ):
        labels.append("textual_tool_imitation")
    if any("core_memory_search" in content for content in provider_contents):
        labels.append("unknown_tool_name_candidate")
    return labels


def read_teacher_query_hints(path: str | None, policy: str) -> dict[int, list[str]]:
    if not path or policy == "none":
        return {}
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    hints: dict[int, list[str]] = {}
    for row in rows:
        queries = []
        for step in row.get("teacher_steps", []):
            if step.get("name") != "conversation_search":
                continue
            arguments = step.get("arguments") or {}
            query = arguments.get("query")
            if isinstance(query, str) and query.strip():
                queries.append(query.strip())
        if policy == "first":
            queries = queries[:1]
        hints[int(row["dataset_index"])] = queries
    return hints


def render_teacher_query_probe(probe: str, queries: list[str]) -> str:
    if not queries:
        return probe
    lines = [
        "[Research ablation: teacher search-query hints]",
        "Use the following literal strings only as conversation_search queries.",
        "They are not answers. Search memory first, then answer only from retrieved conversation evidence.",
        "Suggested search queries:",
        *[f"{index}. {query}" for index, query in enumerate(queries, start=1)],
        "",
        f"User probe: {probe}",
    ]
    return "\n".join(lines)


def capture_exchange(
    session: requests.Session,
    letta_base_url: str,
    agent_id: str,
    model: str,
    user_messages: list[str],
    assistant_message: str,
) -> None:
    response = session.post(
        f"{letta_base_url}/v1/agents/{agent_id}/messages/capture",
        json={
            "provider": "dataset",
            "model": model,
            "request_messages": [{"role": "user", "content": text} for text in user_messages],
            "response_dict": {"content": assistant_message},
        },
        timeout=60,
    )
    response.raise_for_status()


def capture_history(
    session: requests.Session,
    letta_base_url: str,
    agent_id: str,
    model: str,
    sessions: list[Any],
) -> int:
    capture_count = 0
    for session_number, historical_session in enumerate(sessions, start=1):
        pending_user_messages = []
        for turn in extract_turns(historical_session):
            if turn["role"] == "user":
                pending_user_messages.append(turn["content"])
                continue
            if not pending_user_messages:
                pending_user_messages.append(f"[Historical session {session_number} begins.]")
            capture_exchange(
                session,
                letta_base_url,
                agent_id,
                model,
                pending_user_messages,
                turn["content"],
            )
            capture_count += 1
            pending_user_messages = []
        if pending_user_messages:
            capture_exchange(
                session,
                letta_base_url,
                agent_id,
                model,
                pending_user_messages,
                f"[Historical session {session_number} ended.]",
            )
            capture_count += 1
    return capture_count


def reset_and_recompile_recall_metadata(
    session: requests.Session,
    letta_base_url: str,
    agent_id: str,
) -> int:
    response = session.patch(
        f"{letta_base_url}/v1/agents/{agent_id}/reset-messages",
        json={"add_default_initial_messages": False},
        timeout=60,
    )
    response.raise_for_status()
    response = session.post(
        f"{letta_base_url}/v1/agents/{agent_id}/recompile",
        timeout=60,
    )
    response.raise_for_status()
    match = RECALL_COUNT_PATTERN.search(response.json())
    if match is None:
        raise RuntimeError("Recompiled system prompt is missing the recall-memory count.")
    count = int(match.group("count"))
    if count <= 0:
        raise RuntimeError("Recompiled system prompt reports no persisted recall messages.")
    return count


def render_memory_block(title: str, facts: list[str]) -> str:
    return f"{title}\n" + "\n".join(f"- {fact}" for fact in facts)


def evaluate_row(
    client: Letta | None,
    http_session: requests.Session,
    args: argparse.Namespace,
    index: int,
    row: dict[str, Any],
) -> dict[str, Any]:
    started_at = time.time()
    agent = None
    result: dict[str, Any] = {
        "dataset_index": index,
        "model": args.model,
        "model_source_note": args.model_source_note,
        "agent_type": "memgpt_agent",
        "protocol": "msc_dmr_recall_capture_reset_recompile_v2",
        "recall_search_contract": args.recall_search_contract,
        "probe": row["self_instruct"]["B"],
        "reference": row["self_instruct"]["A"],
    }
    teacher_query_hints = getattr(args, "teacher_query_hints_by_index", {}).get(index, [])
    if teacher_query_hints:
        result["teacher_query_hints"] = teacher_query_hints
        result["teacher_query_policy"] = args.teacher_query_policy
        result["protocol"] = "msc_dmr_recall_teacher_query_hint_v1"
    try:
        if args.dry_run:
            result["status"] = "dry_run"
            result["history_sessions"] = len(row["previous_dialogs"]) + 1
            return result

        assert client is not None
        agent = client.agents.create(
            name=f"vanilla-dmr-{slugify(args.model)[-42:]}-{index}-{uuid.uuid4().hex[:8]}",
            agent_type="memgpt_agent",
            model=args.model,
            embedding=args.embedding,
            max_tokens=args.max_tokens,
            initial_message_sequence=[],
            memory_blocks=[
                {
                    "label": "human",
                    "value": render_memory_block("The user is Speaker 2.", row["init_personas"][1]),
                },
                {
                    "label": "persona",
                    "value": f"{DMR_PERSONA_PREFIX}\n\n"
                    + render_memory_block("I am Speaker 1.", row["init_personas"][0]),
                },
            ],
        )
        result["agent_id"] = agent.id
        result["captured_exchanges"] = capture_history(
            http_session,
            args.letta_base_url,
            agent.id,
            args.model,
            [*row["previous_dialogs"], row["dialog"]],
        )
        result["rendered_recall_message_count"] = reset_and_recompile_recall_metadata(
            http_session,
            args.letta_base_url,
            agent.id,
        )
        # Letta refreshes core tools while creating agents. Re-apply the paper-era
        # substring-search contract immediately before the first model request.
        configure_dmr_recall_contract(
            args.letta_base_url,
            args.recall_search_contract,
        )
        user_probe = render_teacher_query_probe(row["self_instruct"]["B"], teacher_query_hints)
        response = client.agents.messages.create(
            agent_id=agent.id,
            messages=[{"role": "user", "content": user_probe}],
            max_steps=args.max_steps,
        )
        raw_messages = [dump_message(message) for message in response.messages]
        trace = tool_trace(raw_messages)
        answer = extract_answer(raw_messages)
        reference = row["self_instruct"]["A"]
        result.update(
            {
                "status": "ok",
                "stop_reason": dump_message(response.stop_reason) if response.stop_reason else None,
                "answer": answer,
                "normalized_answer": normalize_answer(answer),
                "rouge_l_recall": rouge_l_recall(answer, reference),
                "contains_reference": contains_reference(answer, reference),
                "tool_trace": trace,
                "failure_candidates": classify_failure(answer, reference, trace),
                "raw_messages": raw_messages,
            }
        )
        if args.capture_provider_traces:
            result["provider_traces"] = fetch_provider_traces(agent.id)
    except Exception as exc:  # Preserve failed trajectories for diagnosis.
        error = str(exc)
        provider_traces = fetch_provider_traces(agent.id) if agent is not None else []
        provider_responses = [
            trace["response_json"]
            for trace in provider_traces
            if isinstance(trace.get("response_json"), dict)
        ]
        provider_contents = extract_provider_contents(provider_responses)
        failure_candidates = classify_rejected_response(error, provider_contents)
        result.update(
            {
                "status": "behavioral_failure" if failure_candidates else "infrastructure_error",
                "error_type": type(exc).__name__,
                "error": error,
                "failure_candidates": failure_candidates,
                "raw_provider_responses": provider_responses,
                "raw_provider_contents": provider_contents,
            }
        )
        if args.capture_provider_traces:
            result["provider_traces"] = provider_traces
    finally:
        result["elapsed_seconds"] = round(time.time() - started_at, 3)
        if agent is not None and not args.keep_agents:
            try:
                client.agents.delete(agent.id)
            except Exception as exc:
                result["cleanup_error"] = f"{type(exc).__name__}: {exc}"
    return result


def evaluate_row_with_retries(
    client: Letta | None,
    http_session: requests.Session,
    args: argparse.Namespace,
    index: int,
    row: dict[str, Any],
) -> dict[str, Any]:
    for attempt in range(args.infrastructure_retries + 1):
        result = evaluate_row(client, http_session, args, index, row)
        result["infrastructure_attempts"] = attempt + 1
        if result["status"] != "infrastructure_error":
            return result
        if attempt < args.infrastructure_retries:
            print(
                f"[retry] index={index} infrastructure_error="
                f"{result.get('error_type')}: {result.get('error')}"
            )
            time.sleep(args.infrastructure_retry_delay_seconds * (attempt + 1))
    return result


def evaluate_row_isolated(
    args: argparse.Namespace,
    index: int,
    row: dict[str, Any],
) -> dict[str, Any]:
    client = None if args.dry_run else Letta(
        base_url=args.letta_base_url,
        api_key=os.getenv("LETTA_API_KEY") or "EMPTY",
    )
    with requests.Session() as http_session:
        return evaluate_row_with_retries(client, http_session, args, index, row)


def write_summary(path: Path, results: list[dict[str, Any]], *, emit: bool = True) -> None:
    completed = [result for result in results if result["status"] == "ok"]
    dry_runs = [result for result in results if result["status"] == "dry_run"]
    behavioral_failures = [result for result in results if result["status"] == "behavioral_failure"]
    failure_counts = Counter(
        label for result in results for label in result.get("failure_candidates", [])
    )
    provider_attempts = [
        len(result.get("raw_provider_responses", [])) for result in behavioral_failures
    ]
    recall_counts = [
        result["rendered_recall_message_count"]
        for result in results
        if isinstance(result.get("rendered_recall_message_count"), int)
    ]
    summary = {
        "num_results": len(results),
        "num_completed": len(completed),
        "num_dry_runs": len(dry_runs),
        "num_behavioral_failures": len(behavioral_failures),
        "num_errors": len(results) - len(completed) - len(dry_runs) - len(behavioral_failures),
        "mean_rouge_l_recall": statistics.fmean(result["rouge_l_recall"] for result in completed) if completed else None,
        "contains_reference_accuracy": (
            statistics.fmean(float(result["contains_reference"]) for result in completed) if completed else None
        ),
        "search_rate": (
            statistics.fmean(
                float(any(call["name"] == "conversation_search" for call in result["tool_trace"]))
                for result in completed
            )
            if completed
            else None
        ),
        "failure_candidate_counts": dict(sorted(failure_counts.items())),
        "mean_provider_attempts_per_behavioral_failure": (
            statistics.fmean(provider_attempts) if provider_attempts else None
        ),
        "min_rendered_recall_message_count": min(recall_counts) if recall_counts else None,
        "max_rendered_recall_message_count": max(recall_counts) if recall_counts else None,
        "result_file": str(path),
    }
    path.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n")
    if emit:
        print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    args.teacher_query_hints_by_index = read_teacher_query_hints(
        args.teacher_query_hints,
        args.teacher_query_policy,
    )
    if not args.dry_run:
        contract = configure_dmr_recall_contract(
            args.letta_base_url,
            args.recall_search_contract,
        )
        print(f"[contract] {json.dumps(contract, ensure_ascii=True)}")
    dataset = load_from_disk(args.dataset_dir)["train"]
    end = min(args.offset + args.limit, len(dataset))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name_parts = [slugify(args.model)]
    if args.teacher_query_policy != "none":
        name_parts.append(f"teacher-query-{args.teacher_query_policy}")
    name_parts.append(f"offset-{args.offset}")
    name_parts.append(f"limit-{end - args.offset}")
    result_path = output_dir / ("-".join(name_parts) + ".jsonl")
    loaded_results = []
    if args.resume and result_path.exists():
        loaded_results = [
            json.loads(line)
            for line in result_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    results_by_index = {result["dataset_index"]: result for result in loaded_results}
    completed_indices = {
        index
        for index, result in results_by_index.items()
        if result["status"] != "infrastructure_error"
    }
    output_mode = "a" if args.resume else "w"
    candidate_indices = (
        [
            index
            for index in sorted(args.teacher_query_hints_by_index)
            if args.offset <= index < end
        ]
        if args.teacher_query_policy != "none" and args.teacher_query_hints_by_index
        else list(range(args.offset, end))
    )
    pending_indices = [index for index in candidate_indices if index not in completed_indices]

    def persist_result(output: Any, index: int, result: dict[str, Any]) -> None:
        results_by_index[index] = result
        completed_indices.add(index)
        output.write(json.dumps(result, ensure_ascii=True) + "\n")
        output.flush()
        results = [results_by_index[key] for key in sorted(results_by_index)]
        write_summary(result_path, results, emit=False)
        print(
                    f"[{len(completed_indices)}/{len(candidate_indices)}] index={index} "
            f"status={result['status']} rouge_l={result.get('rouge_l_recall')} "
            f"answer={result.get('answer', '')!r}"
        )

    with result_path.open(output_mode, encoding="utf-8") as output:
        if args.workers <= 1:
            for position, index in enumerate(pending_indices):
                persist_result(output, index, evaluate_row_isolated(args, index, dataset[index]))
                if args.row_delay_seconds > 0 and position < len(pending_indices) - 1:
                    print(f"[cooldown] sleeping={args.row_delay_seconds}s")
                    time.sleep(args.row_delay_seconds)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(evaluate_row_isolated, args, index, dataset[index]): index
                    for index in pending_indices
                }
                for future in as_completed(futures):
                    index = futures[future]
                    persist_result(output, index, future.result())
    results = [results_by_index[key] for key in sorted(results_by_index)]
    write_summary(result_path, results)


if __name__ == "__main__":
    main()
