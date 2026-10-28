#!/usr/bin/env bash
# Перевод тернарной HF-модели в GGUF для llama.cpp: тернарные матрицы -> TQ2_0 (~2,06 бита на вес),
# эмбеддинги и выходной слой -> Q6_K (HEAD=q8_0 для точности: +0,1% качества, −3% скорости). Всё на CPU.
# Использование: bash export_gguf.sh <папка hf_ternary> <префикс итоговых файлов> [путь к llama.cpp]
# Переменные: QTYPE (TQ2_0 по умолчанию; PTQ1_0/PQ2_0 в форке Prism), HEAD (q6_K), LC (путь к llama.cpp).
set -euo pipefail
HF=${1:?папка HF-модели}; OUT=${2:?префикс итога}; LC=${3:-${LC:-$HOME/llama.cpp}}
PY=${PYTHON:-python}
# слой MTP (предсказание нескольких токенов) наш экспорт не сохраняет; форк Prism без этого флага модель не загрузит
HAS_MTP=$($PY -c "import json,sys; c=json.load(open(sys.argv[1])); t=c.get('text_config', c); print(int((t.get('mtp_num_hidden_layers') or 0) > 0))" "$HF/config.json")
NOMTP=$([ "$HAS_MTP" = 1 ] && grep -q -- '"--no-mtp"' "$LC/convert_hf_to_gguf.py" && echo --no-mtp || true)
$PY "$LC/convert_hf_to_gguf.py" "$HF" $NOMTP --outtype bf16 --outfile "$OUT-bf16.gguf"
HID=$($PY -c "import json,sys; c=json.load(open(sys.argv[1])); print(c.get('text_config', c)['hidden_size'])" "$HF/config.json")
HEAD=${HEAD:-$([ $((HID % 256)) -eq 0 ] && echo q6_K || echo q8_0)}   # Q6_K требует ширину, кратную 256
Q=${QTYPE:-TQ2_0}   # TQ2_0 — обычный llama.cpp; PTQ1_0 / PQ2_0 — форк Prism (масштаб на 128, меньше файл)
# шлюзы линейного внимания Qwen3.5 (ssm_alpha/ssm_beta) не тернарные — оставляем в bf16
"$LC/build/bin/llama-quantize" --token-embedding-type $HEAD --output-tensor-type $HEAD \
    --tensor-type ssm_alpha=bf16 --tensor-type ssm_beta=bf16 \
    "$OUT-bf16.gguf" "$OUT-$Q.gguf" $Q 2>&1 | tee "$OUT-quantize.log" | grep -v "^warning: blk" | tail -5
if grep -q "required fallback quantization" "$OUT-quantize.log"; then
  echo "ВНИМАНИЕ: у части матриц число столбцов не кратно 256, TQ2_0 к ним неприменим, и llama.cpp сохранил их"
  echo "в Q4_0 с небольшой потерей точности (список в $OUT-quantize.log). У Qwen3.5/Qwen3/Llama такого обычно нет."
fi
ls -lh "$OUT-bf16.gguf" "$OUT-$Q.gguf"
echo "Промежуточный $OUT-bf16.gguf можно удалить."
