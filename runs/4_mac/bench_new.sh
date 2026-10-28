#!/bin/bash
B=~/llama.cpp-cpu/build-lut/bin; D=~/tern-mac/models; M=/Volumes/KINGSTON/miniBrain/ios/Models
for m in $D/e2-TQ2_0 $D/tern-TQ2_0-head_q6_K ~/tern-mac/bonsai2b-TQ2_0 ~/tern-mac/bonsai2b-Q4_0 $M/Ternary-Bonsai-2-2B-Q2_0 $D/base08-Q8_0 $D/base08-Q4_K_M $M/Qwen3.5-2B-Q3_K_M; do
  for r in 1 0; do
  echo "$(basename $m) repack=$r $($B/llama-bench -m $m.gguf -ngl 0 -t 8 -p 512 -n 128 -r 3 --repack $r 2>&1 | grep -E 'pp512|tg128' | awk -F'|' '{printf "%s:%s ", $(NF-2), $(NF-1)}')"
  done
done
echo ALLDONE
