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
    dedupe_retrieved,
    history_messages,
    local_conversation_search,
    render_search_results,
    run_answer_phase,
    slugify,
)
from eval_query_skeleton_dmr import QUERY_SKELETON_SYSTEM, sanitize_query
from nanomemgpt.eval.metrics import contains_reference, normalize_answer, rouge_l_recall


STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "any",
    "are",
    "because",
    "been",
    "before",
    "being",
    "but",
    "can",
    "did",
    "does",
    "doing",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "into",
    "just",
    "like",
    "mentioned",
    "more",
    "much",
    "not",
    "now",
    "other",
    "our",
    "out",
    "said",
    "say",
    "should",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "they",
    "this",
    "told",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "why",
    "with",
    "would",
    "you",
    "your",
}


CANDIDATE_QUERY_SYSTEM = """You are the query generator inside a memory-search controller.
Generate several short literal substring queries for conversation_search.
Each query should be words likely to appear verbatim in prior conversation memory.
Do not answer the user. Do not explain. Do not write JSON.
Return one query per line."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate DMR with candidate query generation and non-oracle local retrieval reranking."
        )
    )
    parser.add_argument("--query-model", default="nano-memgpt-llama3-query-only-r16")
    parser.add_argument("--answer-model", default="nano-memgpt-llama3-r16")
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://llama-vllm:8000/v1"))
    parser.add_argument("--dataset-dir", default="data/raw/msc_dmr")
    parser.add_argument("--output-dir", default="data/evaluation/query_candidate_rerank_dmr")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-searches", type=int, default=3)
    parser.add_argument("--num-candidates", type=int, default=5)
    parser.add_argument("--target-results", type=int, default=3)
    parser.add_argument("--search-limit", type=int, default=10)
    parser.add_argument("--query-max-tokens", type=int, default=160)
    parser.add_argument("--answer-max-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument(
        "--scoring-mode",
        choices=["count", "lexical"],
        default="count",
        help="count preserves the original result-count scorer; lexical adds probe/evidence overlap penalties.",
    )
    parser.add_argument("--evidence-overlap-weight", type=float, default=5.0)
    parser.add_argument("--query-overlap-weight", type=float, default=1.0)
    parser.add_argument("--broad-result-penalty", type=float, default=1.5)
    parser.add_argument("--repeat-overlap-penalty", type=float, default=3.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def build_candidate_prompt(probe: str, trace: list[dict[str, Any]], num_candidates: int) -> str:
    if trace:
        previous = []
        for step in trace:
            previous.append(
                f"- selected query: {step['arguments']['query']}\n"
                f"  results: {step['num_results']}\n"
                f"  output: {step['tool_output'][:600]}"
            )
        previous_block = "\n".join(previous)
    else:
        previous_block = "No previous searches."
    return (
        f"User memory question:\n{probe}\n\n"
        f"Previous searches:\n{previous_block}\n\n"
        f"Generate {num_candidates} different candidate literal substring queries, one per line. "
        "Prefer concrete names, objects, foods, places, jobs, music terms, hobbies, or quoted phrases. "
        "Include both narrow entity-like phrases and one broader fallback phrase. "
        "Avoid repeating previous queries."
    )


def parse_candidate_lines(content: str | None, limit: int) -> list[str]:
    if not isinstance(content, str):
        return []
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|text)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    raw_lines = []
    for line in text.splitlines():
        line = line.strip()
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line)
        line = re.sub(r"^(?:query|candidate|search query)\s*\d*\s*[:=]\s*", "", line, flags=re.IGNORECASE)
        if line:
            raw_lines.append(line)

    if len(raw_lines) == 1 and ("," in raw_lines[0] or ";" in raw_lines[0]):
        raw_lines = [part.strip() for part in re.split(r"[,;]", raw_lines[0]) if part.strip()]

    candidates: list[str] = []
    seen: set[str] = set()
    for line in raw_lines:
        query = sanitize_query(line)
        if not query:
            continue
        key = query.casefold()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(query)
        if len(candidates) >= limit:
            break
    return candidates


def query_specificity_score(query: str) -> float:
    tokens = [tok for tok in re.findall(r"[A-Za-z0-9']+", query) if len(tok) > 1]
    if not tokens:
        return -2.0
    word_count = len(tokens)
    score = min(word_count, 5) * 0.25
    if word_count == 1:
        score -= 0.25
    if len(query) > 80:
        score -= 1.0
    return score


def keyword_tokens(text: str) -> set[str]:
    tokens = set()
    for token in re.findall(r"[A-Za-z0-9']+", text.casefold()):
        token = token.strip("'")
        if len(token) < 3:
            continue
        if token in STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def result_text(results: list[dict[str, Any]]) -> str:
    return "\n".join(str(item.get("content", "")) for item in results)


def lexical_score_details(
    query: str,
    probe: str,
    results: list[dict[str, Any]],
    seen_queries: set[str],
    args: argparse.Namespace,
) -> dict[str, float]:
    query_tokens = keyword_tokens(query)
    probe_tokens = keyword_tokens(probe)
    evidence_tokens = keyword_tokens(result_text(results))
    evidence_overlap = (
        len(probe_tokens & evidence_tokens) / len(probe_tokens)
        if probe_tokens
        else 0.0
    )
    query_probe_overlap = jaccard(query_tokens, probe_tokens)
    repeat_overlap = max(
        (jaccard(query_tokens, keyword_tokens(seen_query)) for seen_query in seen_queries),
        default=0.0,
    )
    broad_penalty = args.broad_result_penalty if len(results) >= args.search_limit else 0.0
    over_target_penalty = max(0, len(results) - (args.target_results * 2)) * 0.25
    return {
        "evidence_overlap": evidence_overlap,
        "query_probe_overlap": query_probe_overlap,
        "repeat_overlap": repeat_overlap,
        "broad_penalty": broad_penalty,
        "over_target_penalty": over_target_penalty,
        "lexical_bonus": (
            args.evidence_overlap_weight * evidence_overlap
            + args.query_overlap_weight * query_probe_overlap
        ),
        "lexical_penalty": (
            args.repeat_overlap_penalty * repeat_overlap
            + broad_penalty
            + over_target_penalty
        ),
    }


def result_count_score(num_results: int, target_results: int) -> float:
    if num_results <= 0:
        return -50.0
    return 10.0 - abs(num_results - target_results)


def select_candidate(
    candidates: list[str],
    messages: list[dict[str, Any]],
    probe: str,
    seen_queries: set[str],
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    scored: list[dict[str, Any]] = []
    for query in candidates:
        key = query.casefold()
        results = local_conversation_search(
            messages,
            query,
            roles=["assistant", "user"],
            limit=args.search_limit,
        )
        num_results = len(results)
        details = {
            "count_score": result_count_score(num_results, args.target_results),
            "specificity_score": query_specificity_score(query),
        }
        score = details["count_score"] + details["specificity_score"]
        if args.scoring_mode == "lexical":
            lexical_details = lexical_score_details(query, probe, results, seen_queries, args)
            details.update(lexical_details)
            score += lexical_details["lexical_bonus"] - lexical_details["lexical_penalty"]
        if key in seen_queries:
            score -= 100.0
            details["exact_repeat_penalty"] = 100.0
        scored.append(
            {
                "query": query,
                "score": round(score, 4),
                "num_results": num_results,
                "score_details": {key: round(value, 4) for key, value in details.items()},
                "tool_output": render_search_results(results),
                "results": results,
            }
        )
    if not scored:
        return None, []
    scored.sort(key=lambda item: (item["score"], -abs(item["num_results"] - args.target_results)), reverse=True)
    return scored[0], scored


def run_candidate_search_phase(
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
                {"role": "system", "content": CANDIDATE_QUERY_SYSTEM},
                {"role": "user", "content": build_candidate_prompt(probe, trace, args.num_candidates)},
            ],
            temperature=args.temperature,
            max_tokens=args.query_max_tokens,
        )
        raw_output = response.choices[0].message.content or ""
        candidates = parse_candidate_lines(raw_output, args.num_candidates)
        selected, scored = select_candidate(candidates, messages, probe, seen_queries, args)
        if selected is None:
            break
        query = selected["query"]
        query_key = query.casefold()
        step = {
            "tool_call_id": f"call_candidate_{uuid.uuid4().hex[:12]}",
            "raw_query_output": raw_output,
            "candidate_queries": [
                {
                    "query": item["query"],
                    "score": item["score"],
                    "num_results": item["num_results"],
                    "score_details": item.get("score_details", {}),
                    "tool_output_preview": item["tool_output"][:600],
                }
                for item in scored
            ],
            "arguments": {
                "query": query,
                "roles": ["assistant", "user"],
                "limit": args.search_limit,
                "request_heartbeat": True,
            },
            "tool_output": selected["tool_output"],
            "num_results": selected["num_results"],
            "selection_score": selected["score"],
        }
        trace.append(step)
        retrieved.extend(selected["results"])
        if query_key in seen_queries:
            break
        seen_queries.add(query_key)
    return trace, retrieved


def evaluate_case(client: OpenAI, args: argparse.Namespace, index: int, row: dict[str, Any]) -> dict[str, Any]:
    started_at = time.time()
    reference = row["self_instruct"]["A"]
    result: dict[str, Any] = {
        "protocol": "msc_dmr_query_candidate_rerank_v1",
        "dataset_index": index,
        "query_model": args.query_model,
        "answer_model": args.answer_model,
        "probe": row["self_instruct"]["B"],
        "reference": reference,
        "max_searches": args.max_searches,
        "num_candidates": args.num_candidates,
        "target_results": args.target_results,
        "scoring_mode": args.scoring_mode,
    }
    try:
        messages = history_messages(row)
        trace, retrieved = run_candidate_search_phase(client, args, row, messages)
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


def summarize(results: list[dict[str, Any]], result_file: Path) -> dict[str, Any]:
    completed = [row for row in results if row.get("status") == "ok"]
    return {
        "protocol": "msc_dmr_query_candidate_rerank_v1",
        "num_results": len(results),
        "num_completed": len(completed),
        "num_errors": len(results) - len(completed),
        "mean_rouge_l_recall": statistics.mean([row["rouge_l_recall"] for row in completed]) if completed else 0.0,
        "contains_reference_accuracy": statistics.mean([row["contains_reference"] for row in completed])
        if completed
        else 0.0,
        "retrieved_contains_reference_rate": statistics.mean(
            [row["retrieved_contains_reference"] for row in completed]
        )
        if completed
        else 0.0,
        "mean_num_searches": statistics.mean([row["num_searches"] for row in completed]) if completed else 0.0,
        "mean_num_retrieved": statistics.mean([row["num_retrieved"] for row in completed]) if completed else 0.0,
        "result_file": str(result_file),
    }


def main() -> None:
    args = parse_args()
    dataset = load_from_disk(args.dataset_dir)["train"]
    end = min(args.offset + args.limit, len(dataset))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / (
        f"{slugify(args.query_model)}-to-{slugify(args.answer_model)}-"
        f"candidate-rerank-{args.scoring_mode}-c{args.num_candidates}-target-{args.target_results}-"
        f"offset-{args.offset}-limit-{end - args.offset}.jsonl"
    )
    summary_file = result_file.with_suffix(".summary.json")

    existing: dict[int, dict[str, Any]] = {}
    if args.resume and result_file.exists():
        with result_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    existing[int(row["dataset_index"])] = row

    client = OpenAI(base_url=args.base_url, api_key=os.getenv("VLLM_API_KEY", "EMPTY"), max_retries=0)
    results: list[dict[str, Any]] = []
    with result_file.open("a" if args.resume else "w", encoding="utf-8") as f:
        for position, index in enumerate(range(args.offset, end), start=1):
            if index in existing:
                result = existing[index]
            else:
                result = evaluate_case(client, args, index, dataset[index])
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
            results.append(result)
            print(
                f"[{position}/{end - args.offset}] index={index} status={result.get('status')} "
                f"searches={result.get('num_searches')} retrieved={result.get('num_retrieved')} "
                f"contains={result.get('contains_reference')} answer={result.get('answer', '')[:120]!r}",
                flush=True,
            )

    if args.resume and existing:
        merged = list(existing.values()) + [row for row in results if int(row["dataset_index"]) not in existing]
        result_by_index = {int(row["dataset_index"]): row for row in merged}
        results = [result_by_index[index] for index in sorted(result_by_index)]
    summary = summarize(results, result_file)
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
