#!/bin/bash
# Живая демонстрация: сервер-роутер со всеми моделями из models.ini (в памяти одна, грузится по выбору на странице).
# Запуск: bash start.sh, затем откройте http://127.0.0.1:8090 (остановить: Ctrl+C)
set -euo pipefail
cd "$(dirname "$0")"
echo "Откройте http://127.0.0.1:8090  (остановить: Ctrl+C)"
exec ~/llama.cpp-cpu/build-lut/bin/llama-server --models-preset "$PWD/models.ini" --models-max 1 \
  --host 127.0.0.1 --port 8090 --path "$PWD/web"
