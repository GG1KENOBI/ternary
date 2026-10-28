#!/bin/bash
# Сравнение «тот же объём памяти»: Qwen3.5-2B в обычных форматах на тех же отложенных текстах.
PY=~/jev/.venv/bin/python; LC=~/llama.cpp/build/bin; W=~/q4b/bench
HF_HUB_OFFLINE=0 $PY -c "from huggingface_hub import snapshot_download; print(snapshot_download(\"Qwen/Qwen3.5-2B\"))" 2>&1 | tail -1
SNAP=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/*)
$PY -c "
from transformers import AutoTokenizer
a=AutoTokenizer.from_pretrained(\"Qwen/Qwen3.5-4B\"); b=AutoTokenizer.from_pretrained(\"$SNAP\")
s=open(\"$W/val_ru.txt\").read()[:20000]; print(\"токенизатор тот же:\", a(s)[\"input_ids\"]==b(s)[\"input_ids\"], len(a), len(b))"
measure() {
  echo "-- $2: $(du -h $1 | cut -f1)"
  for l in ru en; do echo "   ppl $l: $($LC/llama-perplexity -m $1 -f $W/val_$l.txt -c 512 --chunks 25 -t 8 2>&1 | grep -o "Final estimate: PPL = [0-9.]* +/- [0-9.]*")"; done
  $LC/llama-bench -m $1 -t 8 -p 256 -n 64 -r 3 2>&1 | grep -E "pp256|tg64" | awk -F"|" "{print \"   \" \$7 \": \" \$8}"
}
$PY ~/llama.cpp/convert_hf_to_gguf.py $SNAP --outtype bf16 --outfile $W/b2-bf16.gguf 2>&1 | tail -1
for q in Q8_0 Q6_K Q4_K_M; do
  $LC/llama-quantize $W/b2-bf16.gguf $W/b2-$q.gguf $q 2>&1 | grep "quant size"
  measure $W/b2-$q.gguf "Qwen3.5-2B $q"; rm -f $W/b2-$q.gguf
done
rm -f $W/b2-bf16.gguf
echo "-- повтор тернарной 4B (голова Q6_K) для контроля скорости в том же сеансе"
$LC/llama-bench -m $W/tern-TQ2_0-head_q6_K.gguf -t 8 -p 256 -n 64 -r 3 2>&1 | grep -E "pp256|tg64" | awk -F"|" "{print \"   \" \$7 \": \" \$8}"
echo "=== 2B ГОТОВО"
