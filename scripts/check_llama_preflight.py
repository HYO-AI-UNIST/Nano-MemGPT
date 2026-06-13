from __future__ import annotations

import os
from pathlib import Path

from nanomemgpt.config import load_config


def hf_cache_dir(model_id: str) -> Path:
    hf_home = Path(os.getenv("HF_HOME", "/root/.cache/huggingface"))
    return hf_home / "hub" / f"models--{model_id.replace('/', '--')}"


def main() -> None:
    config = load_config()
    student = config["models"]["student"]
    model_id = student["model_id"]
    cache_dir = hf_cache_dir(model_id)
    snapshots = list((cache_dir / "snapshots").glob("*")) if cache_dir.exists() else []
    has_token = bool(os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"))

    print(f"student_model={model_id}")
    print(f"dtype={student['dtype']}")
    print(f"quantization={student['quantization']}")
    print(f"cache_dir={cache_dir}")
    print(f"cached_snapshots={len(snapshots)}")
    print(f"hf_token_configured={'yes' if has_token else 'no'}")
    print("bf16_weight_estimate_gib=14.90")
    print("fp32_weight_estimate_gib=29.80")

    if snapshots:
        print("status=ready_to_start_vllm")
    elif has_token:
        print("status=ready_to_download_gated_model")
    else:
        print("status=blocked_missing_hf_token_and_model_cache")
        print("next=accept the Meta Llama license and set HF_TOKEN in .env")


if __name__ == "__main__":
    main()
