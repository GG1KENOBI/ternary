#!/bin/bash
# После обучения: проверка в PyTorch, затем форматы llama.cpp по очереди (замер -> удаление), чтобы влезть в диск.
cd ~/ternary/release
PY=~/jev/.venv/bin/python; LC=~/llama.cpp/build/bin; W=~/q4b/bench; R=~/q4b/run8
export HF_HUB_OFFLINE=1
until grep -q "^EXIT" ~/q4b/run8.log; do sleep 60; done
grep -q "^EXIT 0" ~/q4b/run8.log || { echo "обучение завершилось с ошибкой"; exit 1; }
mkdir -p $W
free() { df -BG --output=avail / | tail -1 | tr -dc 0-9; }
echo "== свободно $(free) ГБ"

echo "== 1. PyTorch: исходная против тернарной на отложенных текстах + примеры"
$PY eval.py --model Qwen/Qwen3.5-4B --packed $R/ternary_packed.pt --data ~/q4b/data --seq 1024 --val-tokens 0.3e6 \
  --prompts "Столица России — город" "Чтобы испечь хлеб, нужно" "Кратко объясни, что такое инфляция:" "The main idea of the theory of relativity is" \
  --out $W/eval.json 2>&1 | grep -vE "Loading|Warning|warn|causal_conv1d"

$PY -c "
import numpy as np; from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained(\"Qwen/Qwen3.5-4B\")
for l in (\"ru\", \"en\"): open(\"$W/val_\" + l + \".txt\", \"w\").write(t.decode(np.load(\"/home/ubuntu/q4b/data/val_\" + l + \".npy\")[:150000]))
"
measure() {  # $1 — файл модели, $2 — подпись
  echo "-- $2: $(du -h $1 | cut -f1)"
  for l in ru en; do echo "   ppl $l: $($LC/llama-perplexity -m $1 -f $W/val_$l.txt -c 512 --chunks 25 -t 8 2>&1 | grep -o "Final estimate: PPL = [0-9.]* +/- [0-9.]*")"; done
  $LC/llama-bench -m $1 -t 8 -p 256 -n 64 -r 3 2>&1 | grep -E "pp256|tg64" | awk -F"|" "{print \"   \" \$7 \": \" \$8}"
}

echo "== 2. тернарная"
$PY ~/llama.cpp/convert_hf_to_gguf.py $R/hf_ternary --outtype q8_0 --outfile $W/tern-q8.gguf 2>&1 | tail -1
rm -rf $R/hf_ternary   # восстанавливается из ternary_packed.pt через export.py
for head in q8_0 q6_K q4_K; do
  $LC/llama-quantize --allow-requantize --token-embedding-type $head --output-tensor-type $head $W/tern-q8.gguf $W/tern-TQ2_0-head_$head.gguf TQ2_0 2>&1 | grep -E "quant size|fallback"
  measure $W/tern-TQ2_0-head_$head.gguf "тернарная TQ2_0, выходной слой $head"
done
rm -f $W/tern-q8.gguf
echo "== свободно $(free) ГБ"

echo "== 3. исходная Qwen3.5-4B в обычных форматах"
SNAP=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*)
$PY ~/llama.cpp/convert_hf_to_gguf.py $SNAP --outtype q8_0 --outfile $W/base-Q8_0.gguf 2>&1 | tail -1
measure $W/base-Q8_0.gguf "исходная Q8_0"; rm -f $W/base-Q8_0.gguf
$PY ~/llama.cpp/convert_hf_to_gguf.py $SNAP --outtype bf16 --outfile $W/base-bf16.gguf 2>&1 | tail -1
echo "== свободно $(free) ГБ"
for q in Q4_K_M Q3_K_M Q2_K; do
  $LC/llama-quantize $W/base-bf16.gguf $W/base-$q.gguf $q 2>&1 | grep "quant size"
  measure $W/base-$q.gguf "исходная $q"; rm -f $W/base-$q.gguf
done
rm -f $W/base-bf16.gguf
echo "== свободно $(free) ГБ"
echo "=== ЗАМЕРЫ ГОТОВЫ"
