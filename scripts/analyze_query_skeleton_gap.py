#!/usr/bin/env python3
"""Compare teacher-query skeleton replay with student query-only skeleton rows."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def query_list(row: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for step in row.get("search_trace") or []:
        args = step.get("arguments") or {}
        query = args.get("query")
        if isinstance(query, str) and query.strip():
            out.append(query.strip())
    return out


def preview(text: str | None, limit: int = 220) -> str:
    if not text:
        return ""
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def hit_category(teacher: dict[str, Any], student: dict[str, Any], key: str) -> str:
    t = bool(teacher.get(key))
    s = bool(student.get(key))
    if t and s:
        return "both"
    if t and not s:
        return "teacher_only"
    if s and not t:
        return "student_only"
    return "neither"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teacher",
        required=True,
        help="Teacher query skeleton JSONL, usually max-3 for fair comparison.",
    )
    parser.add_argument(
        "--student",
        required=True,
        help="Query-only skeleton JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/analysis/query_skeleton_gap",
    )
    parser.add_argument(
        "--subset",
        choices=["all", "teacher_search"],
        default="all",
        help="Rows to compare after joining on dataset_index.",
    )
    parser.add_argument("--max-examples", type=int, default=50)
    args = parser.parse_args()

    teacher_rows = {int(r["dataset_index"]): r for r in load_jsonl(Path(args.teacher))}
    student_rows = {int(r["dataset_index"]): r for r in load_jsonl(Path(args.student))}
    common_indices = sorted(set(teacher_rows) & set(student_rows))

    compared: list[dict[str, Any]] = []
    for idx in common_indices:
        teacher = teacher_rows[idx]
        student = student_rows[idx]
        if args.subset == "teacher_search" and not teacher.get("teacher_had_search"):
            continue
        compared.append(
            {
                "dataset_index": idx,
                "teacher_had_search": bool(teacher.get("teacher_had_search")),
                "probe": teacher.get("probe") or student.get("probe"),
                "reference": teacher.get("reference") or student.get("reference"),
                "teacher_retrieved_contains_reference": bool(
                    teacher.get("retrieved_contains_reference")
                ),
                "student_retrieved_contains_reference": bool(
                    student.get("retrieved_contains_reference")
                ),
                "teacher_contains_reference": bool(teacher.get("contains_reference")),
                "student_contains_reference": bool(student.get("contains_reference")),
                "retrieval_category": hit_category(
                    teacher, student, "retrieved_contains_reference"
                ),
                "containment_category": hit_category(
                    teacher, student, "contains_reference"
                ),
                "teacher_num_searches": int(teacher.get("num_searches") or 0),
                "student_num_searches": int(student.get("num_searches") or 0),
                "teacher_num_retrieved": int(teacher.get("num_retrieved") or 0),
                "student_num_retrieved": int(student.get("num_retrieved") or 0),
                "teacher_queries": " | ".join(query_list(teacher)),
                "student_queries": " | ".join(query_list(student)),
                "teacher_answer": preview(teacher.get("answer")),
                "student_answer": preview(student.get("answer")),
            }
        )

    retrieval_counts = Counter(r["retrieval_category"] for r in compared)
    containment_counts = Counter(r["containment_category"] for r in compared)

    def rate(count: int) -> float:
        return count / len(compared) if compared else 0.0

    summary = {
        "teacher_file": args.teacher,
        "student_file": args.student,
        "subset": args.subset,
        "num_teacher_rows": len(teacher_rows),
        "num_student_rows": len(student_rows),
        "num_common_rows": len(common_indices),
        "num_compared_rows": len(compared),
        "retrieval_counts": dict(retrieval_counts),
        "retrieval_rates": {k: rate(v) for k, v in sorted(retrieval_counts.items())},
        "containment_counts": dict(containment_counts),
        "containment_rates": {k: rate(v) for k, v in sorted(containment_counts.items())},
        "teacher_retrieved_reference_rate": rate(
            sum(r["teacher_retrieved_contains_reference"] for r in compared)
        ),
        "student_retrieved_reference_rate": rate(
            sum(r["student_retrieved_contains_reference"] for r in compared)
        ),
        "teacher_containment": rate(sum(r["teacher_contains_reference"] for r in compared)),
        "student_containment": rate(sum(r["student_contains_reference"] for r in compared)),
        "mean_teacher_num_retrieved": (
            sum(r["teacher_num_retrieved"] for r in compared) / len(compared)
            if compared
            else 0.0
        ),
        "mean_student_num_retrieved": (
            sum(r["student_num_retrieved"] for r in compared) / len(compared)
            if compared
            else 0.0
        ),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"query_skeleton_gap_{args.subset}.csv"
    jsonl_path = out_dir / f"query_skeleton_gap_{args.subset}.jsonl"
    summary_path = out_dir / f"query_skeleton_gap_{args.subset}.summary.json"
    markdown_path = out_dir / f"query_skeleton_gap_{args.subset}.md"

    fieldnames = list(compared[0].keys()) if compared else []
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(compared)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in compared:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    priority = [
        row
        for row in compared
        if row["retrieval_category"] == "teacher_only"
        or row["containment_category"] == "teacher_only"
    ][: args.max_examples]
    with markdown_path.open("w", encoding="utf-8") as f:
        f.write("# Query Skeleton Gap Analysis\n\n")
        f.write(f"- subset: `{args.subset}`\n")
        f.write(f"- compared rows: `{len(compared)}`\n")
        f.write(
            f"- teacher/student retrieved-reference: "
            f"`{summary['teacher_retrieved_reference_rate']:.3f}` / "
            f"`{summary['student_retrieved_reference_rate']:.3f}`\n"
        )
        f.write(
            f"- teacher/student containment: "
            f"`{summary['teacher_containment']:.3f}` / "
            f"`{summary['student_containment']:.3f}`\n\n"
        )
        f.write("## Retrieval Category Counts\n\n")
        for key in ["both", "teacher_only", "student_only", "neither"]:
            f.write(f"- `{key}`: `{retrieval_counts.get(key, 0)}`\n")
        f.write("\n## Containment Category Counts\n\n")
        for key in ["both", "teacher_only", "student_only", "neither"]:
            f.write(f"- `{key}`: `{containment_counts.get(key, 0)}`\n")
        f.write("\n## Teacher-Only Priority Examples\n\n")
        for row in priority:
            f.write(f"### dataset_index `{row['dataset_index']}`\n\n")
            f.write(f"- reference: {row['reference']}\n")
            f.write(f"- probe: {row['probe']}\n")
            f.write(f"- teacher queries: `{row['teacher_queries']}`\n")
            f.write(f"- student queries: `{row['student_queries']}`\n")
            f.write(
                f"- retrieval category: `{row['retrieval_category']}`, "
                f"containment category: `{row['containment_category']}`\n"
            )
            f.write(f"- teacher answer: {row['teacher_answer']}\n")
            f.write(f"- student answer: {row['student_answer']}\n\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
