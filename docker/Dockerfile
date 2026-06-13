FROM nvcr.io/nvidia/pytorch:26.02-py3

RUN apt-get update && apt-get install -y \
    git curl wget vim tmux htop tree ripgrep ffmpeg libgl1 \
    postgresql-client libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/AI_dev/NLP/Nano-MemGPT

COPY requirements.txt /tmp/nano-memgpt-requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/nano-memgpt-requirements.txt

COPY pyproject.toml README.md /tmp/nano-memgpt/
COPY src /tmp/nano-memgpt/src
RUN python -m pip install -e /tmp/nano-memgpt

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface \
    TORCH_HOME=/root/.cache/torch
