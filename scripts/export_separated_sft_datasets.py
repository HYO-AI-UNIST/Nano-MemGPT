from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nanomemgpt.trajectory.schema import TrajectoryStep


ANSWER_SYSTEM_PROMPT = """Answer the user's memory question as Speaker 1.
Use the provided MemGPT search evidence when it is present.
Return only the short final answer. Do not mention search, tools, evidence, or reasoning."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export separated query-chain and evidence-grounded answer SFT datasets."
    )
    parser.add_argument(
        "--approved-sft",
        default="data/trajectories/gpt41_paper_substring_scaled_approved_sft.jsonl",
    )
    parser.add_argument(
        "--approved-oracle",
        default="data/trajectories/gpt41_paper_substring_scaled_approved_oracle.jsonl",
    )
    parser.add_argument(
        "--query-output",
        default="data/trajectories/gpt41_paper_substring_scaled_query_chain_sft.jsonl",
    )
    parser.add_argument(
        "--answer-output",
        default="data/trajectories/gpt41_paper_substring_scaled_evidence_answer_sft.jsonl",
    )
    parser.add_argument(
        "--combined-output",
        default="data/trajectories/gpt41_paper_substring_scaled_query_answer_sft.jsonl",
    )
    parser.add_argument("--split", default="train")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_steps(path: Path, steps: list[TrajectoryStep]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for step in steps:
            output.write(step.model_dump_json() + "\n")


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def export_query_chain(approved_sft_path: Path, output_path: Path, split: str) -> list[TrajectoryStep]:
    steps = [
        TrajectoryStep.model_validate_json(line)
        for line in approved_sft_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    query_steps = []
    for step in steps:
        if step.teacher_action is None or step.teacher_action.name != "conversation_search":
            continue
        query_steps.append(
            step.model_copy(
                update={
                    "sample_id": f"query-{step.sample_id}",
                    "split": split,
                }
            )
        )
    write_steps(output_path, query_steps)
    write_summary(
        output_path,
        {
            "dataset_type": "query_chain_sft",
            "source_file": str(approved_sft_path),
            "num_source_steps": len(steps),
            "num_query_steps": len(query_steps),
            "result_file": str(output_path),
        },
    )
    return query_steps


def render_teacher_evidence(row: dict[str, Any]) -> str:
    steps = row.get("teacher_steps") or []
    if not steps:
        return "No teacher search calls were used for this approved row."
    blocks = []
    for step_number, step in enumerate(steps, start=1):
        if step.get("name") != "conversation_search":
            continue
        arguments = step.get("arguments") or {}
        blocks.append(
            f"Search {step_number}\n"
            f"Query: {arguments.get('query', '')}\n"
            f"Tool output: {step.get('tool_return', '')}"
        )
    return "\n\n".join(blocks) or "No conversation_search evidence was available."


def build_answer_context(row: dict[str, Any]) -> list[dict[str, str]]:
    content = (
        f"User probe:\n{row['probe']}\n\n"
        f"MemGPT search evidence:\n{render_teacher_evidence(row)}"
    )
    return [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def export_evidence_answers(approved_oracle_path: Path, output_path: Path, split: str) -> list[TrajectoryStep]:
    rows = read_jsonl(approved_oracle_path)
    steps = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        reference = str(row.get("reference", "")).strip()
        if not reference:
            continue
        steps.append(
            TrajectoryStep(
                sample_id=f"answer-dmr-{row['dataset_index']}",
                split=split,
                step_index=0,
                context=build_answer_context(row),
                teacher_action=None,
                function_output=None,
                target_text=reference,
                source="teacher",
            )
        )
    write_steps(output_path, steps)
    write_summary(
        output_path,
        {
            "dataset_type": "evidence_grounded_answer_sft",
            "source_file": str(approved_oracle_path),
            "num_source_rows": len(rows),
            "num_answer_steps": len(steps),
            "num_with_teacher_search_evidence": sum(bool(row.get("teacher_steps")) for row in rows),
            "result_file": str(output_path),
        },
    )
    return steps


def main() -> None:
    args = parse_args()
    query_steps = export_query_chain(Path(args.approved_sft), Path(args.query_output), args.split)
    answer_steps = export_evidence_answers(Path(args.approved_oracle), Path(args.answer_output), args.split)
    combined_steps = [*query_steps, *answer_steps]
    combined_path = Path(args.combined_output)
    write_steps(combined_path, combined_steps)
    write_summary(
        combined_path,
        {
            "dataset_type": "query_chain_plus_evidence_answer_sft",
            "query_file": args.query_output,
            "answer_file": args.answer_output,
            "num_query_steps": len(query_steps),
            "num_answer_steps": len(answer_steps),
            "num_combined_steps": len(combined_steps),
            "result_file": str(combined_path),
        },
    )


if __name__ == "__main__":
    main()
