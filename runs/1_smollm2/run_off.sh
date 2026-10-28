#!/bin/bash
# OFF (TernaryLLM) поверх 16-битного TED, 10 млн токенов: без дистилляции / по выходу / по выходу + OFF.
cd ~/ternary
PY=~/eg2/.venv/bin/python
until grep -q "зёрна готовы" results/seeds30.log 2>/dev/null; do sleep 30; done
run() { name=$1; shift; $PY train_qat.py --name "$name" "$@" > "results/qat_$name.log" 2>&1; echo "=== $name -> $(grep -E "шаг .*проверка" results/qat_$name.log | tail -1 | sed "s/, обучение.*//") ($(date +%H:%M))"; grep -m1 -A2 Traceback results/qat_$name.log; }
C="--method ted --lr 1e-3 --latent-int8 --latent-bits 16 --m-int8 --m-round stochastic --v-mode factored --tokens 10e6"
echo "старт $(date +%H:%M)"
run off_l16_kdoff $C --kd 0.5 --off 1.0 &
sleep 30
run off_l16_kd $C --kd 0.5 &
wait
run off_l16_base $C
echo "=== OFF готово ($(date +%H:%M))"
