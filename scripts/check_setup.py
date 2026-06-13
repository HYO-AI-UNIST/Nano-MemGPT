from __future__ import annotations

import importlib
from pathlib import Path

from nanomemgpt.config import load_config


REQUIRED_IMPORTS = [
    "accelerate",
    "datasets",
    "openai",
    "peft",
    "pgvector",
    "rouge_score",
    "torch",
    "transformers",
    "trl",
]

EXTERNAL_REPOS = [
    "external/repos/letta",
    "external/repos/peft",
    "external/repos/trl",
    "external/repos/vllm",
    "external/repos/lighteval",
]


def main() -> None:
    config = load_config()
    print(f"Loaded project config: {config['project']['name']}")
    for module_name in REQUIRED_IMPORTS:
        importlib.import_module(module_name)
        print(f"ok: import {module_name}")
    missing_external = []
    for path in EXTERNAL_REPOS:
        exists = Path(path).exists()
        print(f"{'ok' if exists else 'missing'}: {path}")
        if not exists:
            missing_external.append(path)
    if missing_external:
        print("Run `python scripts/bootstrap_external_repos.py` to clone pinned references.")


if __name__ == "__main__":
    main()
