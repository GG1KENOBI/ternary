#!/bin/bash
# Диагностика фазы 1б: ждём конца основной серии, затем по одному прогону.
cd ~/ternary
PY=~/eg2/.venv/bin/python
while pgrep -f run_server.sh >/dev/null; do sleep 20; done
run() { name=$1; shift; echo "=== $name: $* ($(date +%H:%M))"; $PY train_qat.py --name "$name" "$@" > "results/qat_$name.log" 2>&1; grep -E "исходная|шаг .*проверка|Traceback|Error" "results/qat_$name.log" | tail -2; }
run trial_anchor --method snf --anchor-mu 1.0 --tokens 0.3e6 --eval-every 25
run d3_snf_anchor --method snf --kappa 0.5 --anchor-mu 1.0
run d1_snf_lam8_k1 --method snf --lam 8 --kappa 1.0
run d2_snf_lrs5e-4 --method snf --kappa 0.5 --lr-s 5e-4
run d4_ste_bf16 --method ste --lr 3e-4 --ste-bf16
echo "=== диагностика готова ($(date +%H:%M))"
