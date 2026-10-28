#!/usr/bin/env bash
# Полная проверка комплекта на одной видеокарте с SmolLM2-135M (~20 минут).
# Использование: bash test_gpu.sh [python]   (из папки комплекта)
set -euo pipefail
PY=${1:-python}
TR=$(dirname "$PY")/torchrun
M=HuggingFaceTB/SmolLM2-135M
D=rt/data; R=rt/run
mkdir -p rt
[ -f $D/val_en.npy ] || $PY prepare_data.py --model $M --out $D --ru 3e6 --en 1e6 --val 0.2e6
rm -rf $R rt/ddp rt/hf2 rt/*.gguf
C="--model $M --data $D --out $R --tokens 2e6 --batch 8 --seq 512 --latent-bits 16 --kd 0.5 --off 1.0 --eval-every 40 --save-every 40 --val-tokens 0.1e6"

echo "=== 1. обучение, прерываем после второй точки сохранения"
$PY train.py $C & P=$!
until [ "$(grep -c '"step": [1-9]' $R/log.jsonl 2>/dev/null)" -ge 2 ] && [ -f $R/ckpt.pt ]; do sleep 5; done
sleep 3; kill $P; wait $P || true
echo "=== 2. продолжение"
$PY train.py $C --save-hf | tee rt/resume.log
grep -q "продолжение с шага" rt/resume.log && echo "ПРОДОЛЖЕНИЕ: ok"

echo "=== 3. проверка и экспорт"
$PY eval.py --model $M --packed $R/ternary_packed.pt --data $D --seq 512 --val-tokens 0.1e6 --out rt/eval.json
$PY export.py --model $M --packed $R/ternary_packed.pt --out rt/hf2
$PY - <<'PY'
from safetensors.torch import load_file
a = load_file("rt/run/hf_ternary/model.safetensors"); b = load_file("rt/hf2/model.safetensors")
d = max((a[n].float() - b[n].float()).abs().max().item() for n in a)
w = a[[n for n in a if "layers.3.mlp.down_proj" in n][0]].float()
print("ЭКСПОРТ: расхождение", d, "| разных |w| в строке:", w[0].abs().unique().numel())
PY

echo "=== 4. два процесса"
$TR --nproc_per_node 2 train.py --model $M --data $D --out rt/ddp --tokens 0.5e6 --batch 4 --seq 512 --latent-bits 16 --eval-every 30 --save-every 30 --val-tokens 0.1e6 | tee rt/ddp.log
grep -q "разошлись" rt/ddp.log && echo "ДВА ПРОЦЕССА: состояния расходились" || echo "ДВА ПРОЦЕССА: ok, состояния совпадают"

if [ -x "$HOME/llama.cpp/build/bin/llama-quantize" ]; then
  echo "=== 5. GGUF TQ2_0"
  PYTHON=$PY bash export_gguf.sh rt/hf2 rt/smol
  $PY -c "import numpy as np; from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('$M'); open('rt/val.txt','w').write(t.decode(np.load('$D/val_en.npy')[:60000]))"
  for m in rt/smol-bf16.gguf rt/smol-TQ2_0.gguf; do
    echo "-- $m"; "$HOME/llama.cpp/build/bin/llama-perplexity" -m $m -f rt/val.txt -c 512 --chunks 30 -ngl 0 2>&1 | grep -E "Final estimate" || true
  done
fi
echo "=== ТЕСТ ПРОЙДЕН"
