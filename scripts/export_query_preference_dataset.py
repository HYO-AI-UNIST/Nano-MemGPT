#!/usr/bin/env python3
"""Export retrieval-supervised query preference and SFT datasets."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


QUERY_SKELETON_SYSTEM = """You are the query generator inside a memory-search controller.
Generate exactly one short literal substring query for conversation_search.
The query should be words likely to appear verbatim in prior conversation memory.
Do not answer the user. Do not explain. Do not write JSON unless unavoidable.
Return only the query text."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build hard-negative query datasets from teacher-query skeleton replay "
            "and query-only skeleton replay."
        )
    )
    parser.add_argument(
        "--teacher",
        default=(
            "data/evaluation/teacher_query_skeleton_dmr_approved500_max3/"
            "teacher-query-to-nano-memgpt-llama3-r16-skeleton-offset-0-limit-500.jsonl"
        ),
    )
    parser.add_argument(
        "--student",
        default=(
            "data/evaluation/query_skeleton_dmr_evidence_only500/"
            "nano-memgpt-llama3-query-only-r16-to-nano-memgpt-llama3-r16-"
            "skeleton-searches-3-offset-0-limit-500.jsonl"
        ),
    )
    parser.add_argument(
        "--preference-output",
        default="data/trajectories/query_hard_negative_preferences.jsonl",
    )
    parser.add_argument(
        "--sft-output",
        default="data/trajectories/query_hard_positive_sft.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        default="data/trajectories/query_hard_negative_preferences.summary.json",
    )
    parser.add_argument(
        "--subset",
        choices=["all", "teacher_search"],
        default="teacher_search",
    )
    parser.add_argument(
        "--category",
        choices=["teacher_only", "teacher_positive", "all"],
        default="teacher_only",
        help=(
            "teacher_only: teacher retrieved reference and student did not; "
            "teacher_positive: teacher retrieved reference; all: all joined rows."
        ),
    )
    parser.add_argument(
        "--positive-strategy",
        choices=["first_hit", "last_hit", "all_hits"],
        default="first_hit",
    )
    parser.add_argument(
        "--negative-strategy",
        choices=["aligned_nonhit", "zero_result_preferred", "zero_result_only"],
        default="zero_result_preferred",
        help=(
            "How to select the rejected student query. zero_result_preferred picks an "
            "empty-result student query when available, reducing noisy negatives."
        ),
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize(text: Any) -> str:
    lowered = str(text or "").casefold()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def trace_query(step: dict[str, Any]) -> str:
    arguments = step.get("arguments") or {}
    query = arguments.get("query")
    return str(query or "").strip()


def step_hits_reference(step: dict[str, Any], reference: str) -> bool:
    norm_ref = normalize(reference)
    if not norm_ref:
        return False
    return norm_ref in normalize(step.get("tool_output", ""))


def query_hits_reference(row: dict[str, Any], reference: str) -> list[bool]:
    return [step_hits_reference(step, reference) for step in row.get("search_trace") or []]


def build_query_prompt(probe: str, trace: list[dict[str, Any]]) -> str:
    if trace:
        previous = []
        for step in trace:
            arguments = step.get("arguments") or {}
            previous.append(
                f"- query: {arguments.get('query', '')}\n"
                f"  results: {step.get('num_results', 0)}\n"
                f"  output: {str(step.get('tool_output', ''))[:600]}"
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


def prompt_messages(probe: str, previous_trace: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": QUERY_SKELETON_SYSTEM},
        {"role": "user", "content": build_query_prompt(probe, previous_trace)},
    ]


def plain_prompt(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(f"{msg['role'].upper()}:\n{msg['content']}" for msg in messages)


def selected_positive_indices(
    teacher: dict[str, Any],
    strategy: str,
) -> list[int]:
    trace = teacher.get("search_trace") or []
    hits = query_hits_reference(teacher, str(teacher.get("reference", "")))
    hit_indices = [idx for idx, hit in enumerate(hits) if hit and trace_query(trace[idx])]
    if not hit_indices:
        return []
    if strategy == "all_hits":
        return hit_indices
    if strategy == "last_hit":
        return [hit_indices[-1]]
    return [hit_indices[0]]


def negative_query_for_step(
    student: dict[str, Any],
    step_index: int,
    chosen_query: str,
    strategy: str,
) -> tuple[str, int | None, dict[str, Any] | None]:
    trace = student.get("search_trace") or []
    hits = query_hits_reference(student, str(student.get("reference", "")))
    chosen_key = chosen_query.casefold()

    candidate_order = list(range(len(trace)))
    if 0 <= step_index < len(trace):
        candidate_order.remove(step_index)
        candidate_order.insert(0, step_index)

    valid_candidates: list[tuple[str, int, dict[str, Any]]] = []
    for idx in candidate_order:
        query = trace_query(trace[idx])
        if not query or query.casefold() == chosen_key:
            continue
        if idx < len(hits) and hits[idx]:
            continue
        valid_candidates.append((query, idx, trace[idx]))

    if strategy in {"zero_result_preferred", "zero_result_only"}:
        for query, idx, step in valid_candidates:
            if int(step.get("num_results") or 0) == 0:
                return query, idx, step
        if strategy == "zero_result_only":
            return "", None, None

    if valid_candidates:
        return valid_candidates[0]
    return "", None, None


def include_row(
    teacher: dict[str, Any],
    student: dict[str, Any],
    subset: str,
    category: str,
) -> bool:
    if subset == "teacher_search" and not teacher.get("teacher_had_search"):
        return False
    teacher_hit = bool(teacher.get("retrieved_contains_reference"))
    student_hit = bool(student.get("retrieved_contains_reference"))
    if category == "teacher_only":
        return teacher_hit and not student_hit
    if category == "teacher_positive":
        return teacher_hit
    return True


def make_sft_step(
    sample_id: str,
    messages: list[dict[str, str]],
    target_query: str,
) -> dict[str, Any]:
    return {
        "sample_id": f"sft-{sample_id}",
        "split": "train",
        "step_index": 0,
        "context": messages,
        "teacher_action": None,
        "function_output": None,
        "target_text": target_query,
        "source": "teacher",
    }


def main() -> None:
    args = parse_args()
    teacher_rows = {int(row["dataset_index"]): row for row in load_jsonl(Path(args.teacher))}
    student_rows = {int(row["dataset_index"]): row for row in load_jsonl(Path(args.student))}

    preference_records: list[dict[str, Any]] = []
    sft_steps: list[dict[str, Any]] = []
    skipped = Counter()

    for dataset_index in sorted(set(teacher_rows) & set(student_rows)):
        teacher = teacher_rows[dataset_index]
        student = student_rows[dataset_index]
        if not include_row(teacher, student, args.subset, args.category):
            skipped["filtered_row"] += 1
            continue
        trace = teacher.get("search_trace") or []
        positive_indices = selected_positive_indices(teacher, args.positive_strategy)
        if not positive_indices:
            skipped["no_positive_hit_step"] += 1
            continue
        for positive_index in positive_indices:
            chosen = trace_query(trace[positive_index])
            rejected, rejected_index, rejected_step = negative_query_for_step(
                student,
                positive_index,
                chosen,
                args.negative_strategy,
            )
            if not chosen:
                skipped["empty_chosen"] += 1
                continue
            if not rejected:
                skipped["empty_rejected"] += 1
                continue
            messages = prompt_messages(str(teacher.get("probe", "")), trace[:positive_index])
            sample_id = f"query-pref-dmr-{dataset_index}-step-{positive_index}"
            metadata = {
                "dataset_index": dataset_index,
                "positive_step_index": positive_index,
                "negative_step_index": rejected_index,
                "reference": teacher.get("reference", ""),
                "teacher_queries": [trace_query(step) for step in teacher.get("search_trace") or []],
                "student_queries": [trace_query(step) for step in student.get("search_trace") or []],
                "teacher_num_retrieved": teacher.get("num_retrieved", 0),
                "student_num_retrieved": student.get("num_retrieved", 0),
                "teacher_contains_reference": bool(teacher.get("contains_reference")),
                "student_contains_reference": bool(student.get("contains_reference")),
                "rejected_num_results": (
                    int(rejected_step.get("num_results") or 0) if rejected_step else None
                ),
                "rejected_tool_output_preview": (
                    str(rejected_step.get("tool_output", ""))[:600] if rejected_step else ""
                ),
            }
            preference_records.append(
                {
                    "sample_id": sample_id,
                    "prompt": plain_prompt(messages),
                    "prompt_messages": messages,
                    "chosen": chosen,
                    "rejected": rejected,
                    "chosen_messages": [{"role": "assistant", "content": chosen}],
                    "rejected_messages": [{"role": "assistant", "content": rejected}],
                    "metadata": metadata,
                }
            )
            sft_steps.append(make_sft_step(sample_id, messages, chosen))

    preference_path = Path(args.preference_output)
    sft_path = Path(args.sft_output)
    summary_path = Path(args.summary_output)
    preference_path.parent.mkdir(parents=True, exist_ok=True)
    sft_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with preference_path.open("w", encoding="utf-8") as f:
        for record in preference_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with sft_path.open("w", encoding="utf-8") as f:
        for step in sft_steps:
            f.write(json.dumps(step, ensure_ascii=False) + "\n")

    summary = {
        "dataset_type": "retrieval_supervised_query_preferences",
        "teacher_file": args.teacher,
        "student_file": args.student,
        "subset": args.subset,
        "category": args.category,
        "positive_strategy": args.positive_strategy,
        "negative_strategy": args.negative_strategy,
        "num_teacher_rows": len(teacher_rows),
        "num_student_rows": len(student_rows),
        "num_joined_rows": len(set(teacher_rows) & set(student_rows)),
        "num_preference_records": len(preference_records),
        "num_sft_steps": len(sft_steps),
        "skipped": dict(sorted(skipped.items())),
        "preference_output": str(preference_path),
        "sft_output": str(sft_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
