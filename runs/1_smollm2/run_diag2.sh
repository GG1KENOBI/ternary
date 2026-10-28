#!/bin/bash
# Диагностика отставания TED на 30 млн токенов: не больше двух прогонов одновременно.
cd ~/ternary
PY=~/eg2/.venv/bin/python
launch() { name=$1; shift
  while [ "$(pgrep -fc "train_qat.py --name")" -ge 2 ]; do sleep 30; done
  echo "$(date +%H:%M) старт $name: $*"
  ( $PY train_qat.py --name "$name" "$@" > "results/qat_$name.log" 2>&1; echo "=== $name -> $(grep -E "шаг .*проверка" results/qat_$name.log | tail -1 | sed "s/, обучение.*//") ($(date +%H:%M))" ) &
  sleep 60
}
launch long_l8_msr --method ted --lr 1e-3 --latent-int8 --m-int8 --m-round stochastic --v-mode factored --tokens 30e6 --eval-every 500
launch long_l16_msr --method ted --lr 1e-3 --latent-int8 --latent-bits 16 --m-int8 --m-round stochastic --v-mode factored --tokens 30e6 --eval-every 500
wait
echo "=== диагностика 2 готова"
