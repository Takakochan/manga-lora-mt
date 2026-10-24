#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! nvidia-smi >/dev/null 2>&1; then
    echo "no NVIDIA GPU found; use a GPU instance with the Deep Learning AMI" >&2
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv sync --python 3.12 --extra finetune
uv run python -c "import torch; assert torch.cuda.is_available(), 'torch cannot see the GPU'"

if [ ! -f out/mt/pages.jsonl ]; then
    uv run scripts/data.py
fi

uv run scripts/lora.py train --holdout balloon_dream --epochs 0.1 --run smoke
echo "smoke test passed; adapter in out/mt/adapters/smoke/balloon_dream"
