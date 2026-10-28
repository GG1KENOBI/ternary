#!/bin/bash
# Фаза 1б на 4090: прогоны строго по одному, по 10 млн токенов.
cd ~/ternary
PY=~/eg2/.venv/bin/python
run() { name=$1; shift; echo "=== $name: $* ($(date +%H:%M))"; $PY train_qat.py --name "$name" "$@" > "results/qat_$name.log" 2>&1; grep -E "исходная|шаг .*проверка|Traceback|Error" "results/qat_$name.log" | tail -2; }
run ste_lr3e-4 --method ste --lr 3e-4
run snf_k0.5 --method snf --kappa 0.5
run bop_t2 --method bop --tau 2.0
run ste_lr1e-3 --method ste --lr 1e-3
run snf_int8 --method snf --kappa 0.5 --int8
echo "=== готово ($(date +%H:%M))"
