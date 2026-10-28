"""Тернаризация модели целиком с экономной памятью (ted.py) на одной видеокарте.

Все линейные слои внутри блоков становятся тернарными с остатком внутри ячейки в int8;
эмбеддинги и нормировки заморожены. Шаг оптимизатора делается в обратном проходе каждого слоя,
активации пересчитываются (gradient checkpointing), потери считаются кусками без хранения логитов.

Запуск:
  python train_big.py --model HuggingFaceTB/SmolLM2-135M --data ~/.cache/ternary/fineweb_edu_smollm2.npy --tokens 10e6 --name big_smol
  python train_big.py --model Qwen/Qwen3.5-4B --data ~/.cache/ternary/qwen_ru_en.npy --tokens 30e6 --name qwen4b
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

from ted import Hyper, ternarize  # noqa: E402

DEV = "cuda"
OUT = Path(__file__).parent / "results"
torch.backends.cuda.matmul.allow_tf32 = True


def lm_loss(model, x, chunk=1024):
    """Средняя кросс-энтропия без хранения всех логитов: голова и CE считаются кусками с пересчётом."""
    base = model.get_decoder() if hasattr(model, "get_decoder") else model.model
    h = base(input_ids=x[:, :-1]).last_hidden_state
    y = x[:, 1:]
    h, y = h.reshape(-1, h.shape[-1]), y.reshape(-1)
    head = model.get_output_embeddings()

    def part(hc, yc):
        return F.cross_entropy(head(hc).float(), yc, reduction="sum")

    total = 0.0
    for i in range(0, h.shape[0], chunk):
        total = total + checkpoint(part, h[i:i + chunk], y[i:i + chunk], use_reentrant=False)
    return total / y.numel()


@torch.no_grad()
def evaluate(model, val, batch):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(val), batch):
        x = torch.from_numpy(val[i:i + batch].astype(np.int64)).to(DEV)
        tot += float(lm_loss(model, x)) * (x.shape[0] * (x.shape[1] - 1))
        n += x.shape[0] * (x.shape[1] - 1)
    model.train()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokens", type=float, default=10e6)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-s", type=float, default=5e-5)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--val-seqs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--no-momentum", action="store_true", help="без момента: ~1 байт на вес")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM
    torch.manual_seed(a.seed)
    data = np.load(os.path.expanduser(a.data), mmap_mode="r")
    n_val = a.val_seqs * (a.seq + 1)
    val = np.asarray(data[-n_val:]).reshape(-1, a.seq + 1)
    train = data[:-n_val]
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(DEV)
    model.config.use_cache = False
    fp_loss = evaluate(model, val, a.batch)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    layers = ternarize(model, momentum=not a.no_momentum)
    Hyper.momentum = not a.no_momentum
    torch.cuda.empty_cache()
    n_tern = sum(l.L.numel() for l in layers)
    state_gb = sum(l.state_bytes() for l in layers) / 1e9
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    start_loss = evaluate(model, val, a.batch)
    steps = int(a.tokens // (a.batch * a.seq))
    warm = min(100, steps // 10)
    factor = lambda t: (t + 1) / warm if t < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (t - warm) / max(1, steps - warm)))
    OUT.mkdir(exist_ok=True)
    log = open(OUT / f"big_{a.name}.jsonl", "w")
    head = {"step": 0, "val_loss": start_loss, "fp_loss": fp_loss, "ternary_params": n_tern, "state_gb": state_gb,
            "ternarize_s": time.time() - t0, "args": vars(a)}
    log.write(json.dumps(head) + "\n")
    print(f"[{a.name}] исходная {fp_loss:.4f}, после тернаризации {start_loss:.4f}, тернарных весов {n_tern / 1e6:.0f} млн, "
          f"состояние {state_gb:.2f} ГБ ({state_gb * 1e9 / n_tern:.2f} Б/вес), шагов {steps}", flush=True)
    rng = np.random.default_rng(a.seed)
    torch.cuda.reset_peak_memory_stats()
    t0, run, Hyper.flips = time.time(), 0.0, 0
    Hyper.lr_s = a.lr_s
    for step in range(1, steps + 1):
        Hyper.t, Hyper.lr = step, a.lr * factor(step - 1)
        idx = rng.integers(0, len(train) - a.seq - 1, a.batch)
        x = torch.from_numpy(np.stack([np.asarray(train[i:i + a.seq + 1]) for i in idx]).astype(np.int64)).to(DEV)
        loss = lm_loss(model, x)
        loss.backward()  # шаги оптимизатора выполняются внутри обратного прохода слоёв
        for l in layers:
            l.dummy.grad = None
        run += float(loss)
        if step % a.eval_every == 0 or step == steps:
            vl = evaluate(model, val, a.batch)
            dt = time.time() - t0
            rec = {"step": step, "tokens": step * a.batch * a.seq, "val_loss": vl, "train_loss": run / a.eval_every,
                   "flips_pct_per_step": 100 * Hyper.flips / a.eval_every / n_tern, "tok_per_s": step * a.batch * a.seq / dt,
                   "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"  шаг {step}/{steps}: проверка {vl:.4f}, обучение {run / a.eval_every:.4f}, переключений "
                  f"{rec['flips_pct_per_step']:.4f}%/шаг, {rec['tok_per_s']:.0f} ток/с, пик видеопамяти {rec['peak_gpu_gb']:.1f} ГБ",
                  flush=True)
            run, Hyper.flips = 0.0, 0
    print("готово", flush=True)


if __name__ == "__main__":
    main()
