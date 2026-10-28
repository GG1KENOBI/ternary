#!/bin/bash
# E2 (0.8B, g256 + поворот) -> HF с поворотом -> GGUF bf16 свежим конвертером основной ветки; база 0.8B -> GGUF bf16
set -x
export HF_HUB_OFFLINE=1
PY=~/jev/.venv/bin/python; LC=~/llama.cpp-new; O=~/s1/mac; mkdir -p $O
cd ~/ternary/release
$PY export.py --model Qwen/Qwen3.5-0.8B --packed ~/s1/E2_g256_had/ternary_packed.pt --out $O/hf_e2 --prism
$PY $LC/convert_hf_to_gguf.py $O/hf_e2 --outtype bf16 --outfile $O/e2-bf16.gguf 2>&1 | grep -iE "prism|hadamard|error|Traceback|wrote|Model successfully" | head -20
B=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*/ | head -1)
$PY $LC/convert_hf_to_gguf.py $B --outtype bf16 --outfile $O/base08-bf16.gguf 2>&1 | grep -iE "error|Traceback|Model successfully" | head
ls -la $O
echo === ГОТОВО
