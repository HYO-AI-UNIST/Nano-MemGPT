from __future__ import annotations

import argparse
import ast
import collections
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any

LEAK_PATTERNS = [
    re.compile(r"\bthinking\b", re.IGNORECASE),
    re.compile(r"\blet me (check|search|look)\b", re.IGNORECASE),
    re.compile(r"\bi('ll| will) (check|search|look)\b", re.IGNORECASE),
]


def normalize_answer(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def contains_reference(prediction: str, reference: str) -> bool:
    normalized_prediction = normalize_answer(prediction)
    normalized_reference = normalize_answer(reference)
    return bool(normalized_reference and normalized_reference in normalized_prediction)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit post-LoRA DMR failures by separating search, evidence, and answer errors."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME:RESULT_JSONL:JUDGE_JSONL",
        help="Run spec. Example: r16:data/eval/result.jsonl:data/eval/result.judged.gpt41.jsonl",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--examples-per-type", type=int, default=8)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def latest_by_index(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["dataset_index"]): row for row in rows}


def safe_json_or_repr(payload: str | None) -> Any:
    if not payload:
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(payload)
        except (SyntaxError, ValueError):
            return payload


def extract_query(arguments: str | None) -> str:
    parsed = safe_json_or_repr(arguments)
    if isinstance(parsed, dict):
        for key in ("query", "search"):
            value = parsed.get(key)
            if isinstance(value, str):
                return value
    return ""


def iter_tool_returns(row: dict[str, Any]) -> list[dict[str, Any]]:
    returns: list[dict[str, Any]] = []
    for message in row.get("raw_messages", []):
        if message.get("message_type") != "tool_return_message":
            continue
        payloads = message.get("tool_returns") or [message]
        for payload in payloads:
            tool_return = payload.get("tool_return") if isinstance(payload, dict) else None
            parsed = safe_json_or_repr(tool_return)
            returns.append(
                {
                    "tool_call_id": payload.get("tool_call_id") if isinstance(payload, dict) else None,
                    "status": payload.get("status") if isinstance(payload, dict) else message.get("status"),
                    "parsed": parsed,
                    "raw": tool_return,
                }
            )
    return returns


def flatten_tool_return_texts(tool_returns: list[dict[str, Any]]) -> list[str]:
    texts: list[str] = []
    for item in tool_returns:
        parsed = item.get("parsed")
        if isinstance(parsed, dict):
            message = parsed.get("message")
            if isinstance(message, str):
                texts.append(message)
            results = parsed.get("results")
            if isinstance(results, list):
                for result in results:
                    if isinstance(result, dict) and isinstance(result.get("content"), str):
                        texts.append(result["content"])
        elif isinstance(parsed, str):
            texts.append(parsed)
    return texts


def has_surface_leak(answer: str) -> bool:
    return any(pattern.search(answer) for pattern in LEAK_PATTERNS)


def row_features(row: dict[str, Any], judge_by_index: dict[int, dict[str, Any]]) -> dict[str, Any]:
    index = int(row["dataset_index"])
    judge = judge_by_index.get(index)
    trace = row.get("tool_trace", [])
    search_calls = [call for call in trace if call.get("name") == "conversation_search"]
    queries = [extract_query(call.get("arguments")) for call in search_calls]
    tool_returns = iter_tool_returns(row)
    evidence_texts = flatten_tool_return_texts(tool_returns)
    evidence_blob = "\n".join(evidence_texts)
    answer = row.get("answer") or ""
    reference = row.get("reference") or ""
    semantic_correct = judge.get("correct") if judge else None
    lexical_correct = bool(row.get("contains_reference"))
    evidence_contains_ref = contains_reference(evidence_blob, reference)
    answer_contains_evidence = False
    norm_answer = normalize_answer(answer)
    for text in evidence_texts:
        norm_text = normalize_answer(text)
        if norm_text and norm_text[:40] in norm_answer:
            answer_contains_evidence = True
            break

    if row.get("status") != "ok":
        failure_type = "tool_call_format_failure"
    elif semantic_correct is True:
        failure_type = "correct_semantic"
    elif not search_calls:
        failure_type = "no_search"
    elif not evidence_texts:
        failure_type = "empty_tool_result"
    elif not evidence_contains_ref:
        failure_type = "searched_wrong_or_insufficient_evidence"
    elif evidence_contains_ref:
        failure_type = "evidence_found_but_not_used"
    else:
        failure_type = "wrong_answer_other"

    if row.get("status") == "ok" and semantic_correct is False and has_surface_leak(answer):
        surface_issue = "reasoning_or_search_intent_leak"
    else:
        surface_issue = ""

    return {
        "dataset_index": index,
        "status": row.get("status"),
        "semantic_correct": semantic_correct,
        "lexical_correct": lexical_correct,
        "failure_type": failure_type,
        "surface_issue": surface_issue,
        "searched": bool(search_calls),
        "num_search_calls": len(search_calls),
        "queries": queries,
        "num_evidence_texts": len(evidence_texts),
        "evidence_contains_reference": evidence_contains_ref,
        "answer_contains_evidence_prefix": answer_contains_evidence,
        "probe": row.get("probe", ""),
        "reference": reference,
        "answer": answer,
        "evidence_preview": evidence_texts[:5],
        "failure_candidates": row.get("failure_candidates", []),
    }


def summarize(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in rows if row["status"] == "ok"]
    judged = [row for row in rows if row["semantic_correct"] is not None]
    wrong = [row for row in judged if row["semantic_correct"] is False]
    search_counts = [row["num_search_calls"] for row in ok]
    type_counts = collections.Counter(row["failure_type"] for row in rows)
    surface_counts = collections.Counter(row["surface_issue"] for row in rows if row["surface_issue"])
    wrong_type_counts = collections.Counter(row["failure_type"] for row in wrong)
    return {
        "run": name,
        "num_rows": len(rows),
        "num_ok": len(ok),
        "num_judged": len(judged),
        "semantic_accuracy": (
            statistics.fmean(float(row["semantic_correct"]) for row in judged) if judged else None
        ),
        "lexical_accuracy": statistics.fmean(float(row["lexical_correct"]) for row in ok) if ok else None,
        "search_rate": statistics.fmean(float(row["searched"]) for row in ok) if ok else None,
        "mean_search_calls_ok": statistics.fmean(search_counts) if search_counts else None,
        "evidence_contains_reference_rate_ok": (
            statistics.fmean(float(row["evidence_contains_reference"]) for row in ok) if ok else None
        ),
        "evidence_contains_reference_rate_wrong": (
            statistics.fmean(float(row["evidence_contains_reference"]) for row in wrong) if wrong else None
        ),
        "failure_type_counts": dict(sorted(type_counts.items())),
        "wrong_only_failure_type_counts": dict(sorted(wrong_type_counts.items())),
        "surface_issue_counts": dict(sorted(surface_counts.items())),
    }


def write_markdown(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Post-LoRA Failure Audit",
        "",
        "## Summary",
        "",
        "| Run | Rows | OK | Judge acc | Lexical acc | Search rate | Evidence-hit rate | Evidence-hit among wrong | Mean searches |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            "| {run} | {num_rows} | {num_ok} | {semantic_accuracy:.4f} | {lexical_accuracy:.4f} | "
            "{search_rate:.4f} | {evidence_contains_reference_rate_ok:.4f} | "
            "{evidence_contains_reference_rate_wrong:.4f} | {mean_search_calls_ok:.4f} |".format(
                **summary
            )
        )
    lines.extend(["", "## Failure Type Counts", ""])
    for summary in summaries:
        lines.extend([f"### {summary['run']}", "", "| Type | Count |", "| --- | ---: |"])
        for key, value in summary["failure_type_counts"].items():
            lines.append(f"| `{key}` | `{value}` |")
        lines.extend(["", "Wrong judged rows only:", "", "| Type | Count |", "| --- | ---: |"])
        for key, value in summary["wrong_only_failure_type_counts"].items():
            lines.append(f"| `{key}` | `{value}` |")
        if summary["surface_issue_counts"]:
            lines.extend(["", "Surface issues:", "", "| Issue | Count |", "| --- | ---: |"])
            for key, value in summary["surface_issue_counts"].items():
                lines.append(f"| `{key}` | `{value}` |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    all_rows_by_run: dict[str, list[dict[str, Any]]] = {}
    for spec in args.run:
        parts = spec.split(":", 2)
        if len(parts) != 3:
            raise SystemExit(f"Invalid --run spec: {spec}")
        name, result_path, judge_path = parts
        results = read_jsonl(Path(result_path))
        judge_by_index = latest_by_index(read_jsonl(Path(judge_path)))
        features = [row_features(row, judge_by_index) for row in results]
        all_rows_by_run[name] = features
        summaries.append(summarize(name, features))

        with (output_dir / f"{name}.rows.jsonl").open("w", encoding="utf-8") as output:
            for row in features:
                output.write(json.dumps(row, ensure_ascii=True) + "\n")
        with (output_dir / f"{name}.rows.csv").open("w", encoding="utf-8", newline="") as output:
            fieldnames = [
                "dataset_index",
                "status",
                "semantic_correct",
                "lexical_correct",
                "failure_type",
                "surface_issue",
                "searched",
                "num_search_calls",
                "queries",
                "num_evidence_texts",
                "evidence_contains_reference",
                "failure_candidates",
            ]
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            for row in features:
                writer.writerow({key: row[key] for key in fieldnames})

    (output_dir / "summary.json").write_text(
        json.dumps({"runs": summaries}, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(output_dir / "summary.md", summaries)

    examples_per_type = args.examples_per_type
    for name, rows in all_rows_by_run.items():
        grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in rows:
            if row["semantic_correct"] is False or row["status"] != "ok":
                grouped[row["failure_type"]].append(row)
        with (output_dir / f"{name}.examples.jsonl").open("w", encoding="utf-8") as output:
            for failure_type in sorted(grouped):
                for row in grouped[failure_type][:examples_per_type]:
                    output.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(json.dumps({"runs": summaries, "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
