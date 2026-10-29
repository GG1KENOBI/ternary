#!/usr/bin/env bash
# Только короткие прогоны. Все результаты в уникальной папке; существующие не удаляются.
set -euo pipefail
cd "$(dirname "$0")"
MODE=${1:-smoke}
case "$MODE" in
  check|smoke|pilot4b|ddp-smoke) ;;
  *) echo 'Режимы: check | smoke | pilot4b | ddp-smoke' >&2; exit 2 ;;
esac
PY=${TED_PYTHON:-.venv-blackwell/bin/python}
if [ ! -x "$PY" ]; then
  echo 'Сначала выполните bash setup_blackwell.sh' >&2; exit 1
fi
export PYTHONUNBUFFERED=1
# mktemp в папке результатов исключает случайное продолжение старой точки сохранения.
mkdir -p blackwell-results
OUT=$(mktemp -d "blackwell-results/${MODE}-$(date +%Y%m%d-%H%M%S)-XXXXXXXX")
exec > >(tee "$OUT/console.log") 2>&1
trap 'rc=$?; echo "Код завершения: $rc; результаты: $OUT"; exit "$rc"' EXIT
echo "Режим $MODE; результаты: $OUT"
git rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || true
"$PY" -c 'from importlib.metadata import distributions; print("\n".join(sorted("{}=={}".format(d.metadata["Name"], d.version) for d in distributions())))' > "$OUT/packages.txt"
"$PY" check_blackwell.py --out "$OUT/hardware.json"
[ "$MODE" != check ] || exit 0
NPROC=1
if [ "$MODE" = ddp-smoke ]; then
  NPROC=${TED_NPROC:-2}
fi
# Большой пилот пока только на одной карте; выбор задайте CUDA_VISIBLE_DEVICES.
if [ "$MODE" = pilot4b ]; then
  MODEL=Qwen/Qwen3.5-4B; TOKENS=2000000; SEQ=512; BATCH=2
  RU=800000; EN=300000; CHAT_RU=700000; CHAT_EN=200000; NEED_GB=50
else
  MODEL=HuggingFaceTB/SmolLM2-135M; TOKENS=131072; SEQ=256; BATCH=2
  RU=100000; EN=100000; CHAT_RU=0; CHAT_EN=0; NEED_GB=5
fi
"$PY" - "$OUT" "$NEED_GB" "$NPROC" <<'PY'
import shutil, sys, torch
free = shutil.disk_usage(sys.argv[1]).free / 10**9
if free < int(sys.argv[2]):
    raise SystemExit(f'Недостаточно места: свободно {free:.1f} ГБ, нужно минимум {sys.argv[2]} ГБ.')
nproc = int(sys.argv[3])
if nproc < 1 or nproc > torch.cuda.device_count():
    raise SystemExit('TED_NPROC должен быть от 1 до числа видимых GPU.')
PY
"$PY" prepare_data.py --model "$MODEL" --out "$OUT/data" \
  --ru "$RU" --en "$EN" --chat-ru "$CHAT_RU" --chat-en "$CHAT_EN" --val 4096
TRAIN=(train.py --model "$MODEL" --data "$OUT/data" --out "$OUT/train" \
  --tokens "$TOKENS" --seq "$SEQ" --batch "$BATCH" --latent-bits 16 \
  --group 256 --hadamard 256 --kd 0.5 --off 1.0 \
  --eval-every 32 --save-every 64 --val-tokens 4096)
if [ "$MODE" = ddp-smoke ]; then
  "$PY" -m torch.distributed.run --standalone --nproc_per_node "$NPROC" "${TRAIN[@]}" 2>&1 | tee "$OUT/distributed.log"
else
  "$PY" "${TRAIN[@]}"
fi
# В новой папке checkpoint не мог существовать: расхождение значит, что DDP ещё не готов.
if [ "$MODE" = ddp-smoke ] && grep -q 'состояния карт разошлись' "$OUT/distributed.log"; then
  echo 'DDP: состояния карт расходились. Длинное обучение пока не запускать.' >&2
  exit 1
fi
"$PY" eval.py --model "$MODEL" --packed "$OUT/train/ternary_packed.pt" \
  --data "$OUT/data" --seq "$SEQ" --batch "$BATCH" --val-tokens 4096 --out "$OUT/eval.json"
echo "Готово. Пришлите папку $OUT (кроме data/, ckpt.pt и ternary_packed.pt)."
