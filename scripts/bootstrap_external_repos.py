from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "external" / "repos"

REPOS = [
    {
        "name": "memgpt-original",
        "url": "https://github.com/cpacker/MemGPT.git",
        "commit": "1131535716e8a31c9a437f8695e25ac98f203a24",
    },
    {
        "name": "letta",
        "url": "https://github.com/letta-ai/letta.git",
        "commit": "1131535716e8a31c9a437f8695e25ac98f203a24",
    },
    {
        "name": "peft",
        "url": "https://github.com/huggingface/peft.git",
        "commit": "a106ff4c7061dd9e59609f88724e4770c3b37293",
    },
    {
        "name": "trl",
        "url": "https://github.com/huggingface/trl.git",
        "commit": "2ffaabd5472c49d5bfec713898f0af4ace4d39c7",
    },
    {
        "name": "vllm",
        "url": "https://github.com/vllm-project/vllm.git",
        "commit": "7b546902447c695c3a555a81352719710d4f1783",
    },
    {
        "name": "lighteval",
        "url": "https://github.com/huggingface/lighteval.git",
        "commit": "89f9395b2af4a4b765e540b9062d6d4ed883119c",
    },
]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    for repo in REPOS:
        dest = EXTERNAL / repo["name"]
        if not dest.exists():
            run(["git", "clone", repo["url"], str(dest)])
        else:
            print(f"ok: {dest} already exists")
        run(["git", "fetch", "--all", "--tags"], cwd=dest)
        run(["git", "checkout", repo["commit"]], cwd=dest)
    print("External repositories are ready.")


if __name__ == "__main__":
    main()
