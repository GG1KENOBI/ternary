#!/usr/bin/env bash
# Linux / WSL2, отдельное окружение. Не собирает llama.cpp и не запускает обучение.
set -euo pipefail
cd "$(dirname "$0")"
if [ "$(uname -s)" != Linux ]; then
  echo 'Откройте Ubuntu в WSL2 (или Linux), см. BLACKWELL_RU.md.' >&2
  exit 1
fi
if ! command -v uv >/dev/null; then
  echo 'Нужен uv. Установка: curl -LsSf https://astral.sh/uv/install.sh | sh' >&2
  echo 'Затем: export PATH="$HOME/.local/bin:$PATH"' >&2
  exit 1
fi
uv venv --python 3.12 --allow-existing .venv-blackwell
PY=.venv-blackwell/bin/python
# Фиксируем CUDA-сборку с поддержкой sm_120; CPU-only wheel здесь не подходит.
uv pip install --python "$PY" 'torch==2.9.1' --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$PY" 'torch==2.9.1' 'transformers==5.19.0' \
  accelerate datasets numpy safetensors sentencepiece flash-linear-attention
mkdir -p blackwell-results
uv pip freeze --python "$PY" > blackwell-results/packages.txt
"$PY" check_blackwell.py --out blackwell-results/hardware.json
echo 'Окружение готово. Короткий тест: bash pilot_blackwell.sh smoke'
