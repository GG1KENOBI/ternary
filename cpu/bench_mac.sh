#!/bin/bash
# Замеры скорости и качества тернарных моделей на CPU Mac (без видеочипа: -ngl 0) в свежем llama.cpp.
# ЗАПУСКАТЬ ТОЛЬКО С РАЗРЕШЕНИЯ ПОЛЬЗОВАТЕЛЯ (Mac занят другими задачами). Нагрузка: сборка ~3–5 мин на всех ядрах,
# затем модели по одной (до ~1,5 ГБ памяти каждая), никакого обучения.
# Нужные файлы в ./models (скопировать с сервера):
#   e1-TQ2_0.gguf (0.8B, блоки, без поворота), e2p-TQ2_0.gguf (0.8B, блоки + поворот), val_ru.txt, val_chat_ru.txt
#   по желанию: tern4b-TQ2_0.gguf (наша 4B), base 0.8B/4B в Q4_K_M для сравнения
set -euo pipefail
cd "$(dirname "$0")"
LC=${LC:-$HOME/llama.cpp-cpu}
if [ ! -x "$LC/build/bin/llama-bench" ]; then
  [ -d "$LC" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LC"
  cmake -S "$LC" -B "$LC/build" -DGGML_METAL=OFF -DGGML_BLAS=OFF -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
  cmake --build "$LC/build" -j --target llama-bench llama-perplexity llama-cli llama-completion
fi
P=$(sysctl -n hw.perflevel0.physicalcpu)   # производительные ядра
echo "== $(sysctl -n machdep.cpu.brand_string), производительных ядер $P, llama.cpp $(git -C "$LC" log -1 --format=%h)"
for m in models/*.gguf; do
  echo "-- $m ($(du -h "$m" | cut -f1))"
  for t in $((P/2)) $P; do
    "$LC/build/bin/llama-bench" -m "$m" -ngl 0 -t $t -p 512 -n 128 -r 3 2>/dev/null | grep -E "pp512|tg128" | awk -F"|" -v t=$t '{print "   потоков " t ": " $7 " " $8}'
  done
  for l in ru chat_ru; do
    [ -f models/val_$l.txt ] && echo "   ppl $l: $("$LC/build/bin/llama-perplexity" -m "$m" -f models/val_$l.txt -c 512 --chunks 30 -ngl 0 -t $P 2>&1 | grep -o 'Final estimate: PPL = [0-9.]*')"
  done
done
