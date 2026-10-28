#!/bin/bash
# Qwen3.5-4B: TED-8 + дистилляция по выходу + OFF, 30 млн токенов на RTX 4090. Повторный запуск продолжает с ckpt.pt.
cd ~/ternary/release
export HF_HUB_OFFLINE=1
~/jev/.venv/bin/python train.py --model Qwen/Qwen3.5-4B --data ~/q4b/data --out ~/q4b/run8 --latent-bits 8 \
  --tokens 30e6 --batch 4 --seq 1024 --kd 0.5 --off 1.0 --eval-every 500 --save-every 500 --val-tokens 0.1e6 --save-hf
echo "EXIT $?"
