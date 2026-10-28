#!/usr/bin/env bash
# Скорость и качество на CPU: тернарная модель против исходной в Q4_K_M (обычное 4-битное сжатие).
# Использование: bash bench_cpu.sh <тернарная .gguf> <исходная модель bf16 .gguf> <текст для перплексии .txt> [потоки]
# Исходную bf16 .gguf получают так: python ~/llama.cpp/convert_hf_to_gguf.py <папка модели> --outtype bf16 --outfile base-bf16.gguf
set -euo pipefail
TERN=${1:?}; BASE=${2:?}; TXT=${3:?}; T=${4:-$(nproc 2>/dev/null || sysctl -n hw.perflevel0.physicalcpu)}
LC=${LLAMA_CPP:-$HOME/llama.cpp}/build/bin
Q4=${BASE%-bf16.gguf}-Q4_K_M.gguf
[ -f "$Q4" ] || "$LC/llama-quantize" "$BASE" "$Q4" Q4_K_M
echo "== размеры"; ls -lh "$TERN" "$Q4"
echo "== скорость (prompt 512 токенов, генерация 128), потоков $T"
"$LC/llama-bench" -m "$TERN" -m "$Q4" -t "$T" -p 512 -n 128 -ngl 0
echo "== перплексия на $TXT (меньше — лучше)"
for m in "$TERN" "$Q4"; do
  echo "-- $m"; "$LC/llama-perplexity" -m "$m" -f "$TXT" -t "$T" -ngl 0 -c 1024 --chunks 40 2>&1 | grep -E "Final estimate|PPL ="| tail -1
done
