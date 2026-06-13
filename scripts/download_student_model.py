from __future__ import annotations

import os

from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError

from nanomemgpt.config import load_config


def main() -> None:
    model_id = load_config()["models"]["student"]["model_id"]
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    try:
        path = snapshot_download(repo_id=model_id, token=token)
    except GatedRepoError:
        raise SystemExit(
            f"Access denied for gated repository {model_id}. Request access on Hugging Face "
            "or set STUDENT_MODEL_ID to an accessible checkpoint mirror."
        )
    print(f"downloaded {model_id} -> {path}")


if __name__ == "__main__":
    main()
