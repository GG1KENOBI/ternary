#!/usr/bin/env bash
# Установка окружения на Linux-сервере с видеокартами NVIDIA (драйвер с поддержкой CUDA 12.8+).
# Использование: bash setup.sh   (создаёт .venv рядом со скриптами и собирает llama.cpp в ~/llama.cpp)
set -euo pipefail
cd "$(dirname "$0")"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python "transformers>=5.0" accelerate datasets numpy safetensors sentencepiece \
    flash-linear-attention   # нужно для Qwen3.5 (слои Gated DeltaNet)
if [ ! -d "$HOME/llama.cpp" ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$HOME/llama.cpp"
fi
cmake -S "$HOME/llama.cpp" -B "$HOME/llama.cpp/build" -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$HOME/llama.cpp/build" -j --target llama-quantize llama-bench llama-perplexity llama-cli
uv pip install --python .venv/bin/python -r "$HOME/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt" || true
.venv/bin/python -c "import torch; print('torch', torch.__version__, 'видеокарт:', torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
echo "Готово. Дальше: source .venv/bin/activate и см. README_RU.md"
