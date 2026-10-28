#!/bin/bash
# Повтор с другим зерном на 30 млн токенов: STE против 16-битного TED.
cd ~/ternary
PY=~/eg2/.venv/bin/python
until grep -q "проверки готовы" results/diag4.log && grep -q "long_l8_msr_b95" results/diag5.log 2>/dev/null; do sleep 30; done
run() { name=$1; shift; $PY train_qat.py --name "$name" "$@" > "results/qat_$name.log" 2>&1; echo "=== $name -> $(grep -E "шаг .*проверка" results/qat_$name.log | tail -1 | sed "s/, обучение.*//") ($(date +%H:%M))"; }
echo "старт $(date +%H:%M)"
run long_ste_s2 --method ste --lr 1e-3 --tokens 30e6 --eval-every 500 --seed 2 &
sleep 30
run long_l16_msr_s2 --method ted --lr 1e-3 --latent-int8 --latent-bits 16 --m-int8 --m-round stochastic --v-mode factored --tokens 30e6 --eval-every 500 --seed 2 &
wait
echo "=== зёрна готовы ($(date +%H:%M))"
