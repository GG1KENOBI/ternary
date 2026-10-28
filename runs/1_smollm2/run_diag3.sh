#!/bin/bash
# Неравномерная int8-сетка (густо у порогов) — после запуска очереди diag2, не больше двух прогонов одновременно.
cd ~/ternary
PY=~/eg2/.venv/bin/python
until [ -f results/qat_long_l16_msr.log ]; do sleep 30; done
sleep 90
while [ "$(pgrep -fc "train_qat.py --name")" -ge 2 ]; do sleep 30; done
name=long_grid_msr
echo "$(date +%H:%M) старт $name"
$PY train_qat.py --name $name --method ted --lr 1e-3 --latent-int8 --grid-band 0.15 --grid-n 90 --m-int8 --m-round stochastic --v-mode factored --tokens 30e6 --eval-every 500 > results/qat_$name.log 2>&1
echo "=== $name -> $(grep -E "шаг .*проверка" results/qat_$name.log | tail -1 | sed "s/, обучение.*//") ($(date +%H:%M))"
