from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any

from datasets import load_from_disk
from openai import OpenAI

from nanomemgpt.eval.metrics import normalize_answer


PROTOCOL = "nq_open_local_context_pack_proxy_v1"
SYSTEM_PROMPT = """Answer the factoid question using only the provided Wikipedia passages.
Return only the shortest answer string. Do not explain your answer. If the passages do not
contain the answer, return UNKNOWN."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate document-QA answer extraction with local NQ context packs."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://llama-vllm:8000/v1"))
    parser.add_argument("--dataset-dir", default="data/raw/nq_open")
    parser.add_argument("--output-dir", default="data/evaluation/document_qa_context_pack")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--retrieved-k", nargs="+", type=int, default=[5, 10, 20, 40])
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["gold_plus_dpr", "dpr_only"],
        default=["gold_plus_dpr", "dpr_only"],
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def select_contexts(row: dict[str, Any], mode: str, requested_k: int) -> list[dict[str, Any]]:
    contexts = row["ctxs"]
    if mode == "dpr_only":
        contexts = sorted(
            (context for context in contexts if not context["isgold"]),
            key=lambda context: context["original_retrieval_index"],
        )
    return contexts[:requested_k]


def render_prompt(question: str, contexts: list[dict[str, Any]]) -> str:
    passages = "\n\n".join(
        f"[{index}] {context['title']}\n{context['text']}"
        for index, context in enumerate(contexts, start=1)
    )
    return f"Question: {question}\n\nWikipedia passages:\n{passages}"


def matches_alias(answer: str, aliases: list[str], *, containment: bool) -> bool:
    normalized_answer = normalize_answer(answer)
    normalized_aliases = [normalize_answer(alias) for alias in aliases]
    if containment:
        return any(alias and alias in normalized_answer for alias in normalized_aliases)
    return normalized_answer in normalized_aliases


def evaluate_case(
    client: OpenAI,
    args: argparse.Namespace,
    index: int,
    row: dict[str, Any],
    mode: str,
    requested_k: int,
) -> dict[str, Any]:
    started_at = time.time()
    contexts = select_contexts(row, mode, requested_k)
    result: dict[str, Any] = {
        "protocol": PROTOCOL,
        "protocol_note": (
            "Proxy only: local MemGPT/qa_data contexts replace the unavailable public "
            "MemGPT/wikipedia_embeddings 20M-passage index."
        ),
        "dataset_index": index,
        "model": args.model,
        "mode": mode,
        "requested_k": requested_k,
        "effective_k": len(contexts),
        "question": row["question"],
        "references": row["answers"],
        "contains_annotated_gold_context": any(context["isgold"] for context in contexts),
        "contains_answer_labeled_context": any(context["hasanswer"] for context in contexts),
    }
    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": render_prompt(row["question"], contexts)},
            ],
            temperature=0,
            max_tokens=args.max_tokens,
        )
        answer = response.choices[0].message.content or ""
        result.update(
            {
                "status": "ok",
                "answer": answer,
                "normalized_answer": normalize_answer(answer),
                "exact_match": matches_alias(answer, row["answers"], containment=False),
                "contains_reference": matches_alias(answer, row["answers"], containment=True),
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
    groups: dict[str, dict[str, Any]] = {}
    for mode in sorted({result["mode"] for result in results}):
        for requested_k in sorted({result["requested_k"] for result in results}):
            selected = [
                result
                for result in results
                if result["mode"] == mode and result["requested_k"] == requested_k
            ]
            completed = [result for result in selected if result["status"] == "ok"]
            key = f"{mode}/k={requested_k}"
            groups[key] = {
                "num_results": len(selected),
                "num_completed": len(completed),
                "num_errors": len(selected) - len(completed),
                "min_effective_k": min((result["effective_k"] for result in selected), default=None),
                "max_effective_k": max((result["effective_k"] for result in selected), default=None),
                "answer_labeled_context_rate": (
                    statistics.fmean(float(result["contains_answer_labeled_context"]) for result in completed)
                    if completed
                    else None
                ),
                "exact_match_accuracy": (
                    statistics.fmean(float(result["exact_match"]) for result in completed) if completed else None
                ),
                "contains_reference_accuracy": (
                    statistics.fmean(float(result["contains_reference"]) for result in completed)
                    if completed
                    else None
                ),
            }
    summary = {
        "protocol": PROTOCOL,
        "num_results": len(results),
        "groups": groups,
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
        f"{slugify(args.model)}-offset-{args.offset}-limit-{end - args.offset}.jsonl"
    )
    results = []
    if args.resume and result_path.exists():
        results = [
            json.loads(line)
            for line in result_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    completed_keys = {
        (result["dataset_index"], result["mode"], result["requested_k"]) for result in results
    }
    client = OpenAI(base_url=args.base_url, api_key=os.getenv("VLLM_API_KEY", "EMPTY"), max_retries=0)
    output_mode = "a" if args.resume else "w"
    total = (end - args.offset) * len(args.modes) * len(args.retrieved_k)
    with result_path.open(output_mode, encoding="utf-8") as output:
        for index in range(args.offset, end):
            row = dataset[index]
            for mode in args.modes:
                for requested_k in args.retrieved_k:
                    key = (index, mode, requested_k)
                    if key in completed_keys:
                        continue
                    result = evaluate_case(client, args, index, row, mode, requested_k)
                    results.append(result)
                    completed_keys.add(key)
                    output.write(json.dumps(result, ensure_ascii=True) + "\n")
                    output.flush()
                    write_summary(result_path, results, emit=False)
                    print(
                        f"[{len(completed_keys)}/{total}] index={index} mode={mode} "
                        f"k={requested_k} status={result['status']} "
                        f"answer={result.get('answer', '')!r}"
                    )
    write_summary(result_path, results)


if __name__ == "__main__":
    main()
