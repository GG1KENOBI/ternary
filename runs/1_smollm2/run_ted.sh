#!/bin/bash
# Проверка остатка внутри ячейки: 4 прогона по 10 млн токенов, парами на одной 4090.
cd ~/ternary
PY=~/eg2/.venv/bin/python
run() { name=$1; shift; $PY train_qat.py --name "$name" "$@" > "results/qat_$name.log" 2>&1; echo "=== $name: $* -> $(grep -E "шаг .*проверка" results/qat_$name.log | tail -1 | sed "s/, видеопамять.*//")"; }
echo "старт $(date +%H:%M)"
run t1_ted_rfp32 --method ted --lr 1e-3 --r-fp32 &
run t2_ted_int8_sr --method ted --lr 1e-3 &
wait
run t3_ted_int8_nearest --method ted --lr 1e-3 --round nearest &
run t4_ted_lowmem --method ted --lr 1e-3 --m-int8 --v-mode factored &
wait
echo "=== проверка ted готова ($(date +%H:%M))"
