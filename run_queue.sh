#!/bin/bash
# Очередь фазы 1б: строго один процесс; ждём, пока нет других тяжёлых ML-процессов и свободно >= 8 ГБ.
cd "$(dirname "$0")"
PY=~/.venvs/eg2/bin/python
ready() {
  pgrep -f "extract_posts_mlx|mlx_lm|train_reader|train_stream" >/dev/null && return 1
  avail=$($PY -c "import psutil; print(int(psutil.virtual_memory().available/1e8))")
  [ "$avail" -ge 80 ]
}
wait_ready() {
  for i in $(seq 1 2880); do ready && return 0; sleep 30; done
  echo "не дождался свободной памяти за сутки"; exit 1
}
run() {  # имя, затем аргументы
  name=$1; shift
  wait_ready
  echo "=== $name: $* ($(date +%H:%M))"
  $PY train_qat.py --name "$name" "$@" > "results/qat_$name.log" 2>&1
  grep -E "исходная|шаг .*проверка|ОСТАНОВКА|Traceback|Error" "results/qat_$name.log" | tail -3
}
mkdir -p results
run trial_ste --method ste --tokens 0.2e6 --eval-every 20
run trial_snf --method snf --tokens 0.2e6 --eval-every 20
run ste_lr3e-4 --method ste --lr 3e-4
run snf_k0.5 --method snf --kappa 0.5
run bop_t2 --method bop --tau 2.0
run ste_lr1e-3 --method ste --lr 1e-3
run snf_int8 --method snf --kappa 0.5 --int8
echo "=== очередь закончена ($(date +%H:%M))"
