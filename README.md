# Nano-MemGPT

Scaffold for the proposal **Diagnosing and Recovering MemGPT's Function-Calling Failures in
Small Open-Source LLMs via Knowledge Distillation**.

The setup is organized around four research loops:

1. Run Letta/MemGPT on DMR and document QA.
2. Collect GPT-4 teacher trajectories: `(context, function call, function output)`.
3. Diagnose small-model failures with oracle function calls and failure-type labels.
4. Distill teacher behavior into a small model with full SFT or LoRA.

## What is already set up

- Docker-based GPU dev container using the same NVIDIA PyTorch base family as `AI_dev/docker`.
- PostgreSQL + pgvector service for persisted recall messages and vector-backed archival
  memory experiments.
- Optional Letta server service.
- Python package skeleton under `src/nanomemgpt`.
- Configs for DMR, NQ-Open, trajectories, diagnostics, and LoRA.
- Exact upstream reference commits recorded in `external/repos/COMMITS.md`:
  - `cpacker/MemGPT`
  - `letta-ai/letta`
  - `huggingface/peft`
  - `huggingface/trl`
  - `vllm-project/vllm`
  - `huggingface/lighteval`

`vllm` is cloned for reference and optional local serving, but it is not installed in the
main image by default because its pip wheel can replace the NVIDIA PyTorch base stack. Use
`requirements-vllm.txt` only after checking CUDA/torch compatibility.

## Quick start

```bash
cd /home/hserver/workspace/AI_dev/NLP/Nano-MemGPT
cp .env.example .env
python scripts/bootstrap_external_repos.py
docker compose up -d --build
docker compose exec nano-memgpt-dev python scripts/check_setup.py
```

Optional vLLM install inside the dev container:

```bash
python -m pip install -r requirements-vllm.txt
```

Fetch the small public MemGPT datasets:

```bash
docker compose exec nano-memgpt-dev python scripts/prepare_memgpt_datasets.py
```

The proposal references `MemGPT/wikipedia_embeddings`, but the public Hugging Face dataset
currently contains only `.gitattributes` and reports zero stored bytes. The local
context-pack document-QA proxy is documented in `docs/document_qa_proxy.md`; it must not be
reported as the paper-faithful 20M-passage retrieval condition.

## Repository hygiene

Large generated artifacts are intentionally ignored by Git: raw datasets, evaluation JSONL
outputs, logs, LoRA checkpoints, local upstream clones, and LaTeX build products. The source
tree keeps scripts, configs, Docker files, research reports, and paper sources. See
`docs/artifact_policy.md` for the tracked/ignored split and artifact reconstruction notes.

## Vanilla Llama MemGPT baseline

The baseline uses `meta-llama/Meta-Llama-3-8B-Instruct` in BF16 without quantization. Accept
the Meta Llama license on Hugging Face and set `HF_TOKEN` in `.env`, then run:

```bash
docker compose up -d pgvector nano-memgpt-dev letta-server
docker compose exec nano-memgpt-dev python scripts/check_llama_preflight.py
docker compose exec nano-memgpt-dev python scripts/download_student_model.py
docker compose --profile llama up -d llama-vllm
docker compose exec nano-memgpt-dev python scripts/smoke_vllm_tools.py
```

The Llama vLLM service uses one RTX PRO 4000 Blackwell GPU and host port `8001`. It enables
the `llama3_json` parser so tool-calling behavior can be checked before running Letta agent
trajectories.

If official Meta access is unavailable, the current reproducible fallback is the public
`NousResearch/Meta-Llama-3-8B-Instruct` mirror pinned in `configs/vanilla_models.yaml`.

If Meta Llama access is still pending, the proposal-listed Mistral baseline can use the
already-cached BF16 model:

```bash
STUDENT_MODEL_ID=mistralai/Mistral-7B-Instruct-v0.3 \
docker compose -f docker-compose.yaml -f docker-compose.mistral.yaml \
  --profile llama up -d --force-recreate llama-vllm
```

The Mistral override uses vLLM's Hugging Face tokenizer path and the upstream parallel tool
chat template while retaining automatic config and weight loading. This avoids the stricter
`mistral-common` role validator when Letta chains memory tools and an NVIDIA vLLM `0.15.1`
`head_dim=None` failure triggered by Hugging Face config loading.

## Vanilla DMR evaluation

The DMR harness captures the five historical MSC sessions directly into Letta recall memory,
then sends only the session-6 probe to an OG `memgpt_agent`. Start with a one-row pilot:

```bash
docker compose exec nano-memgpt-dev python scripts/eval_vanilla_dmr.py \
  --model vllm/mistralai/Mistral-7B-Instruct-v0.3 \
  --model-source-note "Official Mistral BF16 checkpoint" \
  --limit 1
```

Results are written under `data/evaluation/vanilla_dmr`. Each row preserves the answer,
ROUGE-L recall, tool trace, diagnostic failure candidates, and rejected provider responses.
The initial compatibility-pilot findings are recorded in `docs/vanilla_memgpt_pilot.md`.

The paper-era DMR recall tool uses case-insensitive substring matching. The maintained
Letta server otherwise describes `conversation_search` as hybrid semantic search even when
its local PostgreSQL execution falls back to substring matching. `docker-compose.yaml`
mounts a narrow docstring override so the schema exposed to the model matches the local
DMR execution semantics.

For the scaled resumable run:

```bash
docker compose exec nano-memgpt-dev python scripts/eval_vanilla_dmr.py \
  --model vllm/NousResearch/Meta-Llama-3-8B-Instruct \
  --model-source-note "NousResearch mirror BF16 strict parser reset-recompile baseline" \
  --output-dir data/evaluation/experiment_1/dmr \
  --limit 500 \
  --workers 4 \
  --resume
```

After the run, add the proposal's LLM-as-judge accuracy labels when `OPENAI_API_KEY` is
available:

```bash
docker compose exec nano-memgpt-dev python scripts/judge_dmr_answers.py \
  --input-jsonl data/evaluation/experiment_1/dmr/vllm-nousresearch-meta-llama-3-8b-instruct-offset-0-limit-500.jsonl \
  --resume
```

## Template-aligned adapter evaluation

The strict adapter condition keeps model weights frozen and only repackages a single,
explicit, schema-valid tool call. It rejects unknown tools, missing required arguments,
ambiguous multiple calls, and natural-language descriptions of intended calls.

Run the Llama condition:

```bash
STUDENT_MODEL_ID=NousResearch/Meta-Llama-3-8B-Instruct \
docker compose -f docker-compose.yaml -f docker-compose.strict.yaml \
  --profile llama up -d --force-recreate llama-vllm
```

Run the Mistral condition:

```bash
STUDENT_MODEL_ID=mistralai/Mistral-7B-Instruct-v0.3 \
docker compose -f docker-compose.yaml -f docker-compose.mistral-strict.yaml \
  --profile llama up -d --force-recreate llama-vllm
```

When switching a previously served vLLM model back on, Letta `0.16.8` can leave its
provider-model row soft-deleted after a unique-key conflict. Restore only the intended
vLLM handle if it is missing from `/v1/models/`:

```bash
docker compose exec nano-memgpt-dev python scripts/restore_vllm_provider_model.py \
  --handle vllm/mistralai/Mistral-7B-Instruct-v0.3
```

## Oracle replay

The proposal's primary Oracle condition first collects GPT-4 Turbo MemGPT trajectories,
then injects the teacher's exact function call outputs into each student model. A separate
full-history condition is available as a no-key upper-bound diagnostic. See
`docs/oracle_experiment.md` for the distinction and reproducible commands. Current local
results are recorded in `docs/oracle_experiment_report.md`. The original paper used the
retired `gpt-4-1106-preview`; the modern reproducible teacher is pinned as
`gpt-4.1-2025-04-14`.

## Notes

The current Letta project is the maintained successor of the original MemGPT codebase. For
paper-faithful experiments, use the cloned Letta source as the reference implementation and
record exact commit hashes in experiment logs.
