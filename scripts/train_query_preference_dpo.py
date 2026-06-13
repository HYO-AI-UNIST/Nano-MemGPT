from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import DPOConfig, DPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a query-generator LoRA with DPO on teacher/student query preferences."
    )
    parser.add_argument(
        "--preferences",
        default="data/trajectories/query_hard_negative_preferences_zero_only.jsonl",
    )
    parser.add_argument("--model", default="NousResearch/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--output-dir", default="outputs/lora_query_preference_zero_dpo_r16")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--eval-ratio", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=5.0e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument(
        "--loss-type",
        choices=["sigmoid", "hinge", "ipo", "exo_pair", "nca_pair", "robust"],
        default="sigmoid",
    )
    parser.add_argument(
        "--truncation-mode",
        choices=["keep_start", "keep_end"],
        default="keep_start",
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=20)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row_key(sample_id: str) -> str:
    return sample_id.split("-step-", maxsplit=1)[0]


def split_row_keys(
    records: list[dict[str, Any]],
    eval_ratio: float,
    seed: int,
) -> set[str]:
    keys = sorted({row_key(record["sample_id"]) for record in records})
    random.Random(seed).shuffle(keys)
    num_eval = max(1, round(len(keys) * eval_ratio)) if eval_ratio else 0
    return set(keys[:num_eval])


def token_ids_for_messages(
    tokenizer: Any,
    messages: list[dict[str, str]],
    add_generation_prompt: bool = False,
) -> list[int]:
    result = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if hasattr(result, "keys") and "input_ids" in result:
        result = result["input_ids"]
    if result and isinstance(result[0], list):
        result = result[0]
    return result


def prepare_records(
    raw_records: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
    max_samples: int | None,
    selected_sample_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for raw in raw_records:
        if selected_sample_ids and raw["sample_id"] not in selected_sample_ids:
            continue

        prompt_messages = raw.get("prompt_messages")
        chosen_messages = raw.get("chosen_messages")
        rejected_messages = raw.get("rejected_messages")
        if not prompt_messages or not chosen_messages or not rejected_messages:
            prompt_messages = [{"role": "user", "content": str(raw["prompt"])}]
            chosen_messages = [{"role": "assistant", "content": str(raw["chosen"])}]
            rejected_messages = [{"role": "assistant", "content": str(raw["rejected"])}]

        prompt_ids = token_ids_for_messages(tokenizer, prompt_messages, add_generation_prompt=True)
        chosen_total_ids = token_ids_for_messages(tokenizer, prompt_messages + chosen_messages)
        rejected_total_ids = token_ids_for_messages(tokenizer, prompt_messages + rejected_messages)
        total_tokens = max(len(chosen_total_ids), len(rejected_total_ids))

        enriched = {
            "sample_id": raw["sample_id"],
            "prompt": prompt_messages,
            "chosen": chosen_messages,
            "rejected": rejected_messages,
            "chosen_text": str(raw["chosen"]).strip(),
            "rejected_text": str(raw["rejected"]).strip(),
            "prompt_tokens": len(prompt_ids),
            "chosen_total_tokens": len(chosen_total_ids),
            "rejected_total_tokens": len(rejected_total_ids),
            "total_tokens": total_tokens,
            "metadata": raw.get("metadata") or {},
        }
        if total_tokens > max_length:
            dropped.append(enriched)
            continue
        records.append(enriched)
        if max_samples and len(records) >= max_samples:
            break
    return records, dropped


def write_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    dropped: list[dict[str, Any]],
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rejected_counts = Counter(
        str((record.get("metadata") or {}).get("rejected_num_results"))
        for record in records
    )
    manifest = {
        "model": args.model,
        "preferences": args.preferences,
        "output_dir": str(output_dir),
        "max_length": args.max_length,
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "loss_type": args.loss_type,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_records": len(records),
        "num_train_records": len(train_records),
        "num_eval_records": len(eval_records),
        "num_dropped_overlength": len(dropped),
        "rejected_num_results_counts": dict(sorted(rejected_counts.items())),
        "token_stats": {
            "max_total_tokens": max((record["total_tokens"] for record in records), default=0),
            "max_prompt_tokens": max((record["prompt_tokens"] for record in records), default=0),
        },
        "dropped_overlength": [
            {"sample_id": record["sample_id"], "total_tokens": record["total_tokens"]}
            for record in dropped
        ],
    }
    rendered = json.dumps(manifest, ensure_ascii=True, indent=2)
    (output_dir / "run_manifest.json").write_text(f"{rendered}\n", encoding="utf-8")
    print(rendered)


def to_dpo_dataset(records: list[dict[str, Any]]) -> Dataset:
    return Dataset.from_list(
        [
            {
                "prompt": record["prompt"],
                "chosen": record["chosen"],
                "rejected": record["rejected"],
            }
            for record in records
        ]
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=not args.allow_download,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_records = load_jsonl(Path(args.preferences))
    records, dropped = prepare_records(
        raw_records,
        tokenizer,
        args.max_length,
        args.max_samples,
        set(args.sample_id),
    )
    eval_keys = split_row_keys(records, args.eval_ratio, args.seed)
    train_records = [record for record in records if row_key(record["sample_id"]) not in eval_keys]
    eval_records = [record for record in records if row_key(record["sample_id"]) in eval_keys]
    write_manifest(output_dir, args, records, dropped, train_records, eval_records)
    if args.prepare_only:
        return

    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    training_args = DPOConfig(
        output_dir=str(output_dir),
        max_length=args.max_length,
        truncation_mode=args.truncation_mode,
        beta=args.beta,
        loss_type=[args.loss_type],
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps" if eval_records else "no",
        eval_steps=args.save_steps if eval_records else None,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        model_init_kwargs={
            "dtype": torch.bfloat16,
            "local_files_only": not args.allow_download,
        },
    )
    trainer = DPOTrainer(
        model=args.model,
        args=training_args,
        train_dataset=to_dpo_dataset(train_records),
        eval_dataset=to_dpo_dataset(eval_records) if eval_records else None,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir / "final_adapter"))
    tokenizer.save_pretrained(str(output_dir / "final_adapter"))


if __name__ == "__main__":
    main()
