#!/bin/bash
B=~/llama.cpp-cpu/build/bin; D=~/tern-mac/models; M=/Volumes/KINGSTON/miniBrain/ios/Models
for m in $D/base08-Q8_0 $D/base08-Q4_K_M $D/e1-TQ2_0 $D/e2-TQ2_0 $D/tern-TQ2_0-head_q6_K $M/Ternary-Bonsai-2-2B-Q2_0 ~/tern-mac/bonsai2b-TQ2_0 $M/Qwen3.5-2B-Q3_K_M; do
  r="$(basename $m) $(du -m $m.gguf | cut -f1)МБ"
  for l in ru chat_ru; do r="$r | ppl_$l $($B/llama-perplexity -m $m.gguf -f $D/val_$l.txt -c 512 --chunks 30 -ngl 0 -t 8 2>&1 | grep -oE 'PPL = [0-9.]+' | tail -1)"; done
  for t in 6 8; do r="$r | t$t $($B/llama-bench -m $m.gguf -ngl 0 -t $t -p 512 -n 128 -r 3 2>&1 | grep -E 'pp512|tg128' | awk -F'|' '{printf "%s:%s ", $7, $8}')"; done
  echo "$r"
done
echo ALLDONE
