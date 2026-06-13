from __future__ import annotations

from pathlib import Path

from datasets import load_dataset

from nanomemgpt.config import load_config


def save_dataset(name: str, output_dir: Path) -> None:
    dataset = load_dataset(name)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))
    print(f"saved {name} -> {output_dir}")


def main() -> None:
    config = load_config()
    save_dataset(config["data"]["msc_dmr"]["hf_dataset"], Path(config["data"]["msc_dmr"]["raw_dir"]))
    save_dataset(config["data"]["nq_open"]["hf_dataset"], Path(config["data"]["nq_open"]["raw_dir"]))
    print(
        "Skipped MemGPT/wikipedia_embeddings by default because the proposal's 20M-passage "
        "index is large. Fetch it explicitly once storage and time budget are confirmed."
    )


if __name__ == "__main__":
    main()
