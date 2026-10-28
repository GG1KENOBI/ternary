#!/bin/bash
# Шаг 1б: модель E2 (блок 256 + поворот 256) в форке Prism. Сверка с эквивалентной обычной моделью и скорость.
set -x
cd ~/ternary/release
PY=~/jev/.venv/bin/python; M=Qwen/Qwen3.5-0.8B; S=~/s1; W=$S/gguf; mkdir -p $W
PL=~/prism-llama.cpp; SL=~/llama.cpp
export HF_HUB_OFFLINE=1
~/.local/bin/cmake --build $PL/build -j 8 --target llama-quantize 2>&1 | tail -1
$PY -c "
import numpy as np; from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('$M')
for l in ('ru', 'chat_ru'): open('$W/val_' + l + '.txt', 'w').write(t.decode(np.load('$S/data/val_' + l + '.npy')[:120000]))
"
ppl() { echo "   $3 $2: $($1/build/bin/llama-perplexity -m $4 -f $W/val_$2.txt -c 512 --chunks 30 -t 8 2>&1 | grep -o 'Final estimate: PPL = [0-9.]* +/- [0-9.]*')"; }
speed() { $1/build/bin/llama-bench -m $2 -t 8 -p 256 -n 64 -r 3 2>&1 | grep -E "pp256|tg64" | awk -F"|" "{print \"   \" \$7 \": \" \$8}"; }

echo "== 1. E2 в форке Prism (тернарные веса в повёрнутом базисе + поворот в среде)"
$PY export.py --model $M --packed $S/E2_g256_had/ternary_packed.pt --out $W/hf_e2_prism --prism 2>&1 | grep -vE "Loading|Warning"
$PY $PL/convert_hf_to_gguf.py $W/hf_e2_prism --outtype bf16 --outfile $W/e2p-bf16.gguf 2>&1 | grep -E "Hadamard|Error|error|exported" | tail -4
rm -rf $W/hf_e2_prism
for Q in PTQ1_0 TQ2_0; do
  $PL/build/bin/llama-quantize --token-embedding-type q6_K --output-tensor-type q6_K --tensor-type ssm_alpha=bf16 --tensor-type ssm_beta=bf16 \
     $W/e2p-bf16.gguf $W/e2p-$Q.gguf $Q 2>&1 | grep -E "quant size|fallback"
done
rm -f $W/e2p-bf16.gguf
ls -lh $W
for Q in PTQ1_0 TQ2_0; do for l in ru chat_ru; do ppl $PL $l "E2 форк $Q" $W/e2p-$Q.gguf; done; done
echo "   скорость E2 форк PTQ1_0:"; speed $PL $W/e2p-PTQ1_0.gguf
echo "   скорость E2 форк TQ2_0:";  speed $PL $W/e2p-TQ2_0.gguf

echo "== 2. E2 эквивалентная обычная (bf16, обычный llama.cpp) — эталон для сверки"
$PY export.py --model $M --packed $S/E2_g256_had/ternary_packed.pt --out $W/hf_e2_dense 2>&1 | grep -vE "Loading|Warning|ВНИМАНИЕ|но её"
$PY $SL/convert_hf_to_gguf.py $W/hf_e2_dense --outtype bf16 --outfile $W/e2d-bf16.gguf 2>&1 | tail -1
rm -rf $W/hf_e2_dense
for l in ru chat_ru; do ppl $SL $l "E2 обычная bf16" $W/e2d-bf16.gguf; done
rm -f $W/e2d-bf16.gguf

echo "== 3. E1 (блок 256, без поворота) в обычном llama.cpp TQ2_0"
$PY export.py --model $M --packed $S/E1_g256/ternary_packed.pt --out $W/hf_e1 2>&1 | grep -vE "Loading|Warning"
LC=$SL PYTHON=$PY bash export_gguf.sh $W/hf_e1 $W/e1 2>&1 | grep -E "quant size|ВНИМАНИЕ|TQ2_0.gguf"
rm -rf $W/hf_e1 $W/e1-bf16.gguf
for l in ru chat_ru; do ppl $SL $l "E1 TQ2_0" $W/e1-TQ2_0.gguf; done
echo "   скорость E1 TQ2_0:"; speed $SL $W/e1-TQ2_0.gguf
df -h / | tail -1
echo "=== 1Б ГОТОВ"
