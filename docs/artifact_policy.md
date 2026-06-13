# Artifact Policy

This repository should contain source code, configuration, documentation, and paper sources.
Large or regenerated artifacts should stay outside Git and be reproduced from scripts.

## Tracked

- `src/`: reusable Nano-MemGPT package code
- `scripts/`: experiment, export, training, and evaluation entry points
- `configs/`: experiment and model configuration
- `docker*.yaml`, `docker/`: reproducible local services
- `docs/`: research reports and paper source
- `external/repos/COMMITS.md`: exact upstream reference commits

## Ignored

- `data/raw/`, `data/processed/`, `data/trajectories/`, `data/evaluation/`, `data/analysis/`
- `outputs/` LoRA adapters and checkpoints
- `logs/`
- cloned upstream repositories under `external/repos/`
- Python caches and LaTeX build products
- model weight files such as `.safetensors`, `.bin`, `.pt`, `.pth`, `.ckpt`, `.gguf`

## Reconstructing Local Artifacts

Clone the external reference repositories:

```bash
python scripts/bootstrap_external_repos.py
```

Prepare public datasets:

```bash
docker compose exec nano-memgpt-dev python scripts/prepare_memgpt_datasets.py
```

Evaluation outputs and training adapters are intentionally not tracked. For publication,
upload model adapters and large JSONL artifacts to a model or artifact registry, then link
them from the README or release notes.
