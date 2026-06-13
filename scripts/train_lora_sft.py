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
from trl import SFTConfig, SFTTrainer

from nanomemgpt.training.formatting import iter_chat_sft_records
from nanomemgpt.trajectory.schema import TrajectoryStep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a LoRA adapter on approved MemGPT teacher function-call trajectories."
    )
    parser.add_argument(
        "--trajectories",
        default="data/trajectories/gpt41_paper_substring_scaled_approved_sft.jsonl",
    )
    parser.add_argument("--model", default="NousResearch/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--output-dir", default="outputs/lora_student_r16")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--loss-type", choices=["nll", "chunked_nll"], default="chunked_nll")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def load_steps(path: Path) -> list[TrajectoryStep]:
    return [
        TrajectoryStep.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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


def prepare_records(
    steps: list[TrajectoryStep],
    tokenizer: Any,
    max_length: int,
    max_samples: int | None,
    selected_sample_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = []
    dropped = []
    for step, record in zip(steps, iter_chat_sft_records(steps, tokenizer), strict=True):
        if selected_sample_ids and record["sample_id"] not in selected_sample_ids:
            continue
        total_tokens = len(
            tokenizer(
                f"{record['prompt']}{record['completion']}",
                add_special_tokens=False,
            )["input_ids"]
        )
        enriched = {
            **record,
            "action_name": step.teacher_action.name if step.teacher_action else "target_text",
            "total_tokens": total_tokens,
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
    manifest = {
        "model": args.model,
        "trajectories": args.trajectories,
        "output_dir": str(output_dir),
        "max_length": args.max_length,
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "loss_type": args.loss_type,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_records": len(records),
        "num_train_records": len(train_records),
        "num_eval_records": len(eval_records),
        "num_dropped_overlength": len(dropped),
        "action_counts": dict(sorted(Counter(record["action_name"] for record in records).items())),
        "dropped_overlength": [
            {"sample_id": record["sample_id"], "total_tokens": record["total_tokens"]}
            for record in dropped
        ],
    }
    rendered = json.dumps(manifest, ensure_ascii=True, indent=2)
    (output_dir / "run_manifest.json").write_text(f"{rendered}\n", encoding="utf-8")
    print(rendered)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=not args.allow_download,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    steps = load_steps(Path(args.trajectories))
    records, dropped = prepare_records(
        steps,
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

    dataset_columns = ["prompt", "completion"]
    train_dataset = Dataset.from_list(
        [{key: record[key] for key in dataset_columns} for record in train_records]
    )
    eval_dataset = (
        Dataset.from_list(
            [{key: record[key] for key in dataset_columns} for record in eval_records]
        )
        if eval_records
        else None
    )
    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    training_args = SFTConfig(
        output_dir=str(output_dir),
        max_length=args.max_length,
        completion_only_loss=True,
        loss_type=args.loss_type,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.save_steps if eval_dataset is not None else None,
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
    trainer = SFTTrainer(
        model=args.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir / "final_adapter"))
    tokenizer.save_pretrained(str(output_dir / "final_adapter"))


if __name__ == "__main__":
    main()
