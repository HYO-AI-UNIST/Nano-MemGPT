from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from eval_phase_routed_dmr import run_answer_phase, slugify
from eval_query_candidate_rerank_dmr import jaccard, keyword_tokens
from nanomemgpt.eval.metrics import contains_reference, normalize_answer, rouge_l_recall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay an existing DMR retrieval result file after non-oracle evidence filtering."
        )
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--answer-model", default="nano-memgpt-llama3-r16")
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://llama-vllm:8000/v1"))
    parser.add_argument("--output-dir", default="data/evaluation/evidence_filter_dmr")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--filter-mode",
        choices=["none", "first", "lexical"],
        default="lexical",
        help="none keeps all evidence, first keeps the first top-k messages, lexical ranks evidence non-oracle.",
    )
    parser.add_argument("--answer-max-tokens", type=int, default=96)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def selected_queries(trace: list[dict[str, Any]]) -> list[str]:
    queries = []
    for step in trace:
        arguments = step.get("arguments", {})
        query = arguments.get("query")
        if isinstance(query, str) and query.strip():
            queries.append(query.strip())
    return queries


def content_length_penalty(content: str) -> float:
    token_count = len(re.findall(r"[A-Za-z0-9']+", content))
    if token_count <= 0:
        return 2.0
    if token_count > 80:
        return min(2.0, (token_count - 80) / 80)
    return 0.0


def evidence_score(
    evidence: dict[str, Any],
    probe: str,
    queries: list[str],
) -> dict[str, float]:
    content = str(evidence.get("content", ""))
    evidence_tokens = keyword_tokens(content)
    probe_tokens = keyword_tokens(probe)
    query_token_sets = [keyword_tokens(query) for query in queries]
    probe_recall = (
        len(evidence_tokens & probe_tokens) / len(probe_tokens)
        if probe_tokens
        else 0.0
    )
    max_query_overlap = max(
        (jaccard(evidence_tokens, query_tokens) for query_tokens in query_token_sets),
        default=0.0,
    )
    exact_query_hits = 0
    folded = content.casefold()
    for query in queries:
        if query.casefold() in folded:
            exact_query_hits += 1
    speaker_bonus = 0.2 if evidence.get("role") == "assistant" else 0.0
    length_penalty = content_length_penalty(content)
    score = (
        4.0 * probe_recall
        + 3.0 * max_query_overlap
        + 0.75 * exact_query_hits
        + speaker_bonus
        - length_penalty
    )
    return {
        "score": score,
        "probe_recall": probe_recall,
        "max_query_overlap": max_query_overlap,
        "exact_query_hits": float(exact_query_hits),
        "speaker_bonus": speaker_bonus,
        "length_penalty": length_penalty,
    }


def filter_evidence(
    row: dict[str, Any],
    *,
    top_k: int,
    mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evidence = list(row.get("retrieved_evidence") or [])
    if mode == "none" or top_k <= 0:
        return evidence, [
            {"rank": index, "original_index": index, "score_details": {"score": 0.0}, **item}
            for index, item in enumerate(evidence)
        ]
    if mode == "first":
        selected = evidence[:top_k]
        return selected, [
            {"rank": index, "original_index": index, "score_details": {"score": 0.0}, **item}
            for index, item in enumerate(selected)
        ]

    probe = str(row.get("probe", ""))
    queries = selected_queries(row.get("search_trace") or [])
    scored = []
    for index, item in enumerate(evidence):
        details = evidence_score(item, probe, queries)
        scored.append(
            {
                "rank": 0,
                "original_index": index,
                "score_details": {key: round(value, 4) for key, value in details.items()},
                **item,
            }
        )
    scored.sort(
        key=lambda item: (
            item["score_details"]["score"],
            item["score_details"]["exact_query_hits"],
            -item["original_index"],
        ),
        reverse=True,
    )
    selected_scored = scored[:top_k]
    for rank, item in enumerate(selected_scored, start=1):
        item["rank"] = rank
    selected = [
        {key: value for key, value in item.items() if key not in {"rank", "original_index", "score_details"}}
        for item in selected_scored
    ]
    return selected, selected_scored


def reference_in_evidence(reference: str, evidence: list[dict[str, Any]]) -> bool:
    normalized_reference = normalize_answer(reference)
    return any(normalized_reference in normalize_answer(str(item.get("content", ""))) for item in evidence)


def evaluate_case(client: OpenAI, args: argparse.Namespace, source: dict[str, Any]) -> dict[str, Any]:
    started_at = time.time()
    reference = str(source.get("reference", ""))
    filtered, scored = filter_evidence(source, top_k=args.top_k, mode=args.filter_mode)
    answer_row = {
        "self_instruct": {
            "A": reference,
            "B": str(source.get("probe", "")),
        }
    }
    answer = run_answer_phase(client, args, answer_row, source.get("search_trace") or [], filtered)
    return {
        "protocol": "msc_dmr_evidence_filter_v1",
        "dataset_index": source.get("dataset_index"),
        "source_protocol": source.get("protocol"),
        "source_query_model": source.get("query_model"),
        "answer_model": args.answer_model,
        "filter_mode": args.filter_mode,
        "top_k": args.top_k,
        "probe": source.get("probe"),
        "reference": reference,
        "status": "ok",
        "answer": answer,
        "normalized_answer": normalize_answer(answer),
        "rouge_l_recall": rouge_l_recall(answer, reference),
        "contains_reference": contains_reference(answer, reference),
        "source_contains_reference": bool(source.get("contains_reference")),
        "source_retrieved_contains_reference": bool(source.get("retrieved_contains_reference")),
        "filtered_retrieved_contains_reference": reference_in_evidence(reference, filtered),
        "source_num_retrieved": len(source.get("retrieved_evidence") or []),
        "num_retrieved": len(filtered),
        "num_searches": source.get("num_searches", len(source.get("search_trace") or [])),
        "search_trace": source.get("search_trace") or [],
        "retrieved_evidence": filtered,
        "filtered_evidence_scores": scored,
        "elapsed_seconds": round(time.time() - started_at, 3),
    }


def summarize(results: list[dict[str, Any]], result_file: Path) -> dict[str, Any]:
    completed = [row for row in results if row.get("status") == "ok"]
    return {
        "protocol": "msc_dmr_evidence_filter_v1",
        "num_results": len(results),
        "num_completed": len(completed),
        "num_errors": len(results) - len(completed),
        "filter_mode": completed[0].get("filter_mode") if completed else None,
        "top_k": completed[0].get("top_k") if completed else None,
        "mean_rouge_l_recall": statistics.fmean(row["rouge_l_recall"] for row in completed)
        if completed
        else 0.0,
        "contains_reference_accuracy": statistics.fmean(float(row["contains_reference"]) for row in completed)
        if completed
        else 0.0,
        "source_contains_reference_accuracy": statistics.fmean(
            float(row["source_contains_reference"]) for row in completed
        )
        if completed
        else 0.0,
        "source_retrieved_contains_reference_rate": statistics.fmean(
            float(row["source_retrieved_contains_reference"]) for row in completed
        )
        if completed
        else 0.0,
        "filtered_retrieved_contains_reference_rate": statistics.fmean(
            float(row["filtered_retrieved_contains_reference"]) for row in completed
        )
        if completed
        else 0.0,
        "mean_source_num_retrieved": statistics.fmean(row["source_num_retrieved"] for row in completed)
        if completed
        else 0.0,
        "mean_num_retrieved": statistics.fmean(row["num_retrieved"] for row in completed)
        if completed
        else 0.0,
        "result_file": str(result_file),
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    source_rows = load_jsonl(input_path)
    end = len(source_rows) if args.limit is None else min(args.offset + args.limit, len(source_rows))
    selected_sources = source_rows[args.offset:end]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / (
        f"{slugify(args.answer_model)}-evidence-filter-{args.filter_mode}-k{args.top_k}-"
        f"offset-{args.offset}-limit-{len(selected_sources)}.jsonl"
    )
    summary_file = result_file.with_suffix(".summary.json")

    existing: dict[int, dict[str, Any]] = {}
    if args.resume and result_file.exists():
        for row in load_jsonl(result_file):
            existing[int(row["dataset_index"])] = row

    client = OpenAI(base_url=args.base_url, api_key=os.getenv("VLLM_API_KEY", "EMPTY"), max_retries=0)
    results: list[dict[str, Any]] = []
    with result_file.open("a" if args.resume else "w", encoding="utf-8") as f:
        for position, source in enumerate(selected_sources, start=1):
            index = int(source["dataset_index"])
            if index in existing:
                result = existing[index]
            else:
                try:
                    result = evaluate_case(client, args, source)
                except Exception as exc:
                    result = {
                        "protocol": "msc_dmr_evidence_filter_v1",
                        "dataset_index": index,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
            results.append(result)
            print(
                f"[{position}/{len(selected_sources)}] index={index} status={result.get('status')} "
                f"source_retrieved={result.get('source_num_retrieved')} filtered={result.get('num_retrieved')} "
                f"source_hit={result.get('source_retrieved_contains_reference')} "
                f"filtered_hit={result.get('filtered_retrieved_contains_reference')} "
                f"contains={result.get('contains_reference')} answer={result.get('answer', '')[:120]!r}",
                flush=True,
            )

    summary = summarize(results, result_file)
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
