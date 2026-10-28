#!/bin/bash
# Шаг 1 (приёмы Bonsai + диалоги) на Qwen3.5-0.8B, RTX 4090.
#   E0: масштаб на строку (как раньше)          E1: масштаб на блок 256          E2: блок 256 + поворот Адамара 256
# Все: TED-16 + дистилляция по выходу 0.5, 10 млн токенов, данные: тексты + диалоги ru/en.
# Запуск на сервере: bash ~/ternary/step1_qwen08.sh > ~/s1/step1.log 2>&1
set -x
cd ~/ternary/release
PY=~/jev/.venv/bin/python
M=Qwen/Qwen3.5-0.8B
S=~/s1; mkdir -p $S

# 1. модель и данные
HF_HUB_OFFLINE=0 $PY -c "from huggingface_hub import snapshot_download; snapshot_download('$M', allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja'])"
[ -f $S/data/val_chat_en.npy ] || HF_HUB_OFFLINE=0 $PY prepare_data.py --model $M --out $S/data \
    --ru 8e6 --en 3e6 --chat-ru 8e6 --chat-en 3e6 --val 0.2e6
export HF_HUB_OFFLINE=1

# 2. ошибка приближения на настоящих весах (без обучения)
$PY - <<'PY'
import torch, ted
from transformers import AutoModelForCausalLM
for mid in ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-4B"]:
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16)
    lins = [(n, x) for n, x in m.named_modules() if isinstance(x, torch.nn.Linear) and ".layers." in n
            and not any(k in n for k in ("visual", "mtp"))]
    lins = lins[:: max(1, len(lins) // 24)]  # ~24 матрицы по всей глубине
    for g, h in [(0, 0), (256, 0), (256, 256)]:
        ted.set_format(g, h)
        num = den = 0.0
        for n, x in lins:
            W = x.weight.detach().float().cuda()
            lin = torch.nn.Linear(W.shape[1], W.shape[0], bias=False, device="cuda"); lin.weight.data.copy_(W)
            t = ted.TedLinear(lin, momentum=False)
            num += float((t.weight_dense(torch.float32) - W).pow(2).sum()); den += float(W.pow(2).sum())
            del t, lin, W
        print(f"{mid}: группа={g} поворот={h}: относительная ошибка {(num / den) ** 0.5:.4f}", flush=True)
    del m
PY

# 3. три обучения: по два одновременно
C="--model $M --data $S/data --tokens 10e6 --batch 8 --seq 1024 --kd 0.5 --latent-bits 16 --eval-every 300 --save-every 100000 --val-tokens 0.1e6"
run() { name=$1; shift; $PY train.py $C --out $S/$name "$@" > $S/$name.log 2>&1; rm -f $S/$name/ckpt.pt;
        echo "=== $name: $(grep -E 'шаг [0-9]+/' $S/$name.log | tail -1 | sed 's/, обучение.*//')"; }
run E0_row --group 0 &
sleep 120
run E1_g256 --group 256 &
wait
run E2_g256_had --group 256 --hadamard 256

# 4. проверка с ответами в формате чата
for e in E0_row E1_g256 E2_g256_had; do
  echo "=== проверка $e"
  $PY eval.py --model $M --packed $S/$e/ternary_packed.pt --data $S/data --seq 1024 --val-tokens 0.1e6 --out $S/$e/eval.json 2>&1 \
    | grep -vE "Loading|Warning|warn|causal_conv1d"
done
echo "=== ШАГ 1 ГОТОВ"
