#!/bin/bash
# Bonsai 2B (родной Q2_0 от Prism) -> TQ2_0 и Q4_0 обычным llama-quantize. Поворот (prism.hadamard) сохраняется в метаданных.
B=~/llama.cpp-cpu/build/bin; M=/Volumes/KINGSTON/miniBrain/ios/Models; W=~/tern-mac; mkdir -p $W
for q in TQ2_0 Q4_0; do
  $B/llama-quantize --allow-requantize $M/Ternary-Bonsai-2-2B-Q2_0.gguf $W/bonsai2b-$q.gguf $q 8
done
