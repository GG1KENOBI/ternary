"""Фаза 1а: механизм тернарного оптимизатора на одном линейном слое Qwen3-0.6B.

Задача: подобрать W_hat = s * Q, Q in {-1, 0, +1} (масштаб s на строку), чтобы выход слоя
на реальных русских и английских текстах как можно меньше отличался от исходного:
    ошибка = ||X (W - W_hat)^T||^2 / ||X W^T||^2.
Веса подбираются на одних текстах (X_train), ошибка мерится на других (X_eval).

Методы:
  absmean     — округление BitNet b1.58 (без обучения);
  twn         — лучший порог |w| > delta на строку (без обучения);
  gptq        — GPTQ с тернарной сеткой: компенсация ошибки через гессиан X^T X (сильный PTQ);
  ste_adam    — обычное обучение с теневыми fp32-весами: STE + Adam (как BitNet при QAT);
  bop         — тернарный Bop без теневых весов: накопленный градиент > порога -> шаг по сетке;
  snf         — наш: шаг по сетке, если предсказанный выигрыш второго порядка больше шума градиента;
  snf_noh     — то же без второго порядка (проверка, нужен ли он);
  snf_int8    — snf, но накопители градиента хранятся в 8 битах (экономия памяти).
Запуск: ~/.venvs/eg2/bin/python layer_bench.py
"""
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch  # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
CACHE = Path.home() / ".cache" / "ternary"
OUT = Path(__file__).parent / "results"
SEED = 13
STEPS, BATCH = 2000, 4096  # одинаковые для всех методов с обучением


# ---------- данные: входы выбранных слоёв на русских и английских текстах ----------
@torch.no_grad()
def collect_inputs(targets, n_train=128, n_eval=32, seq=512):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16).to(DEV).eval()
    texts = json.load(open(CACHE / "calib_texts.json", encoding="utf-8"))

    def seqs(lang, n, offset):
        ids, out = [], []
        for t in texts[lang][offset:]:
            ids += tok(t)["input_ids"] + [tok.eos_token_id]
            while len(ids) >= seq and len(out) < n:
                out.append(ids[:seq])
                ids = ids[seq:]
            if len(out) >= n:
                break
        return out

    # обучение и проверка — на разных текстах (первая и вторая половины выборки)
    train = seqs("ru", n_train // 2, 0) + seqs("en", n_train // 2, 0)
    evals = seqs("ru", n_eval // 2, 1500) + seqs("en", n_eval // 2, 1500)
    mods = dict(model.named_modules())
    store = {name: [] for name in targets}
    hooks = [mods[name].register_forward_pre_hook(lambda m, a, name=name: store[name].append(a[0].reshape(-1, a[0].shape[-1]).float().cpu()))
             for name in targets]
    res = {}
    for split, batch in (("train", train), ("eval", evals)):
        for name in targets:
            store[name].clear()
        for i in range(0, len(batch), 8):
            model(torch.tensor(batch[i:i + 8], device=DEV))
        for name in targets:
            res.setdefault(name, {})[split] = torch.cat(store[name])
    for h in hooks:
        h.remove()
    weights = {name: mods[name].weight.detach().float().cpu() for name in targets}
    del model
    if DEV == "mps":
        torch.mps.empty_cache()
    return res, weights


# ---------- общие функции ----------
def rel_err(W, What, X):
    D = (X @ (W - What).T)
    return float((D * D).sum() / ((X @ W.T) ** 2).sum())


def refit_scale(W, Q, H):
    """Лучший масштаб на строку при фиксированных кодах: s_r = (q H w) / (q H q)."""
    QH = Q @ H
    num, den = (QH * W).sum(1), (QH * Q).sum(1)
    return torch.where(den > 0, num / den.clamp_min(1e-12), torch.zeros_like(num))


def absmean(W):
    s = W.abs().mean(1).clamp_min(1e-8)
    return (W / s[:, None]).round().clamp(-1, 1), s


def twn(W, grid=40):
    """Порог delta на строку, минимизирующий ||w - s q||^2 (Ternary Weight Networks)."""
    a = W.abs()
    best_err = torch.full((W.shape[0],), float("inf"), device=W.device)
    best_Q, best_s = torch.zeros_like(W), torch.zeros(W.shape[0], device=W.device)
    for qt in torch.linspace(0.05, 0.95, grid):
        delta = torch.quantile(a, qt.item(), dim=1, keepdim=True)
        mask = (a > delta).float()
        s = (a * mask).sum(1) / mask.sum(1).clamp_min(1)
        Q = torch.sign(W) * mask
        err = ((W - s[:, None] * Q) ** 2).sum(1)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_Q[better], best_s[better] = Q[better], s[better]
    return best_Q, best_s


def gptq(W, H, s, block=128, damp=0.01):
    """GPTQ (Frantar et al.) с тернарной сеткой и фиксированным масштабом строк s."""
    W = W.clone()
    n = W.shape[1]
    H = H.clone()
    H += damp * H.diag().mean() * torch.eye(n, device=H.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    for i1 in range(0, n, block):
        i2 = min(i1 + block, n)
        W1, Hi = W[:, i1:i2].clone(), Hinv[i1:i2, i1:i2]
        E = torch.zeros_like(W1)
        for i in range(i2 - i1):
            w, d = W1[:, i], Hi[i, i]
            q = (w / s).round().clamp(-1, 1)
            Q[:, i1 + i] = q
            err = (w - s * q) / d
            W1[:, i:] -= err[:, None] * Hi[i, i:][None, :]
            E[:, i] = err
        W[:, i2:] -= E @ Hinv[i1:i2, i2:]
    return Q


def batches(X, steps, bs, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(steps):
        yield X[torch.randint(0, X.shape[0], (bs,), generator=g)]


def grad_what(W, What, Xb):
    """Градиент относительной ошибки по W_hat на минибатче."""
    Y = Xb @ W.T
    D = Xb @ (What - W).T
    return 2 * D.T @ Xb / (Y * Y).sum()


def ste_adam(W, X, H, lr, steps=STEPS, bs=BATCH, seed=SEED):
    Q0, s0 = twn(W)
    lat = W.clone().requires_grad_(True)
    logs = s0.clamp_min(1e-8).log().clone().requires_grad_(True)
    opt = torch.optim.Adam([lat, logs], lr=lr)
    for Xb in batches(X, steps, bs, seed):
        s = logs.exp()
        q = lat / s[:, None]
        q = q + (q.round().clamp(-1, 1) - q).detach()  # STE
        What = s[:, None] * q
        Y = Xb @ W.T
        loss = ((Xb @ (W - What).T) ** 2).sum() / (Y * Y).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        s = logs.exp()
        return (lat / s[:, None]).round().clamp(-1, 1), s, {"bytes_per_weight": 12}  # fp32 вес + 2 момента


def quant8(x, scale):
    """Хранение накопителя в int8 с общим масштабом строки (эмуляция)."""
    return (x / scale).round().clamp(-127, 127) * scale


def flip_opt(W, X, H, kind, kappa=1.0, tau=None, lr_ema=0.05, beta2=0.99, max_frac=0.01, int8=False,
             refit_every=100, steps=STEPS, bs=BATCH, seed=SEED, init=None):
    """Оптимизаторы без теневых весов: состояние = коды Q + накопители градиента.

    bop:     m = EMA(g); шаг по сетке против знака m, если |m| > tau; после шага m = 0.
    snf:     выигрыш шага dq in {-1, +1}:  -dL = -(g dw + 0.5 h dw^2),  dw = s dq,
             h — диагональ гессиана (для этого слоя точная: 2 s^2 H_jj / ||XW||^2);
             шаг принимается, если -dL > kappa * sigma * |dw|, где sigma — оценка шума среднего
             градиента по EMA(g) и EMA(g^2); после шага g <- g + h dw (предсказанный новый градиент).
    snf_noh: то же с h = 0 (только порог по шуму).
    Не больше max_frac весов меняется за шаг (диагональное приближение не учитывает взаимное влияние).
    """
    Q, s = init if init is not None else twn(W)
    g = torch.zeros_like(W)
    v = torch.zeros_like(W)
    n_eff = (2 - lr_ema) / lr_ema
    Y2 = ((X @ W.T) ** 2).sum() / X.shape[0]  # на один токен, как в минибатче
    hdiag = 2 * H.diag()[None, :] / X.shape[0] / Y2  # кривизна по dW_hat на один токен батча
    flips = 0
    for step, Xb in enumerate(batches(X, steps, bs, seed)):
        What = s[:, None] * Q
        gr = grad_what(W, What, Xb)
        g.mul_(1 - lr_ema).add_(gr, alpha=lr_ema)
        v.mul_(beta2).add_(gr * gr, alpha=1 - beta2)
        if int8:
            g = quant8(g, g.abs().amax(1, keepdim=True).clamp_min(1e-30) / 127)
            v = quant8(v, v.abs().amax(1, keepdim=True).clamp_min(1e-30) / 127).clamp_min(0)
        if kind == "bop":
            score = g.abs() - tau
            dq = -torch.sign(g)
            gain = score
        else:
            h = torch.zeros_like(W) if kind == "snf_noh" else hdiag.expand_as(W)
            sigma = ((v - g * g).clamp_min(0) / n_eff).sqrt()
            gains = []
            for d in (-1.0, 1.0):
                dw = s[:, None] * d
                gains.append(-(g * dw + 0.5 * h * dw * dw) - kappa * sigma * dw.abs())
            gain = torch.maximum(gains[0], gains[1])
            dq = torch.where(gains[1] > gains[0], 1.0, -1.0)
        newQ = (Q + dq).clamp(-1, 1)
        ok = (gain > 0) & (newQ != Q)
        k = int(max_frac * W.numel())
        if ok.sum() > k:  # самые выгодные шаги
            thr = torch.topk(torch.where(ok, gain, torch.full_like(gain, -float("inf"))).flatten(), k).values[-1]
            ok &= gain >= thr
        if ok.any():
            dw = s[:, None] * (newQ - Q)
            if kind == "bop":
                g = torch.where(ok, torch.zeros_like(g), g)
            else:
                g = torch.where(ok, g + h * dw, g)
            Q = torch.where(ok, newQ, Q)
            flips += int(ok.sum())
        if refit_every and (step + 1) % refit_every == 0:
            s = refit_scale(W, Q, H).clamp_min(1e-8)
    bpw = 0.25 + (1 if int8 else 4) * (1 if kind == "bop" else 2)  # коды 2 бита + накопители
    return Q, s, {"flips_total": flips, "flips_per_step_pct": 100 * flips / steps / W.numel(), "bytes_per_weight": bpw}


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    targets = sys.argv[1:] or ["model.layers.12.mlp.down_proj", "model.layers.12.self_attn.o_proj"]
    t0 = time.time()
    data, weights = collect_inputs(targets)
    print(f"входы собраны за {time.time() - t0:.0f} с", flush=True)
    OUT.mkdir(exist_ok=True)
    report = {}
    for name in targets:
        W = weights[name].to(DEV)
        X, Xe = data[name]["train"].to(DEV), data[name]["eval"].to(DEV)
        H = X.T @ X
        print(f"\n== {name}: W {tuple(W.shape)}, токенов обучение {X.shape[0]}, проверка {Xe.shape[0]}", flush=True)
        rows = {}

        def add(method, Q, s, extra=None, t=None):
            r = {"eval_err": rel_err(W, s[:, None] * Q, Xe),
                 "eval_err_refit": rel_err(W, refit_scale(W, Q, H)[:, None] * Q, Xe),
                 "train_err": rel_err(W, s[:, None] * Q, X), "zeros_pct": float((Q == 0).float().mean() * 100),
                 "sec": round(time.time() - t, 1) if t else None, **(extra or {})}
            rows[method] = r
            print(f"  {method:22s} ошибка {r['eval_err']:.4f} (с подгонкой масштаба {r['eval_err_refit']:.4f}), "
                  f"нулей {r['zeros_pct']:.0f}%" + (f", {r['sec']} с" if r["sec"] else "")
                  + (f", перевернулось за шаг {r['flips_per_step_pct']:.3f}%" if "flips_per_step_pct" in r else ""),
                  flush=True)

        t = time.time(); Q, s = absmean(W); add("absmean", Q, s, t=t)
        t = time.time(); Q, s = twn(W); add("twn", Q, s, t=t)
        t = time.time(); _, s_twn = twn(W); Q = gptq(W, H, s_twn); add("gptq", Q, refit_scale(W, Q, H), t=t)
        Qg = Q.clone()
        sg = refit_scale(W, Qg, H).clamp_min(1e-8)
        for lr in (1e-4, 2e-4, 3e-4, 5e-4):
            t = time.time(); Q, s, ex = ste_adam(W, X, H, lr); add(f"ste_adam lr={lr}", Q, s, ex, t)
        g_scale = float(grad_what(W, twn(W)[1][:, None] * twn(W)[0], X[:BATCH]).abs().mean())
        for mult in (1.0, 2.0, 4.0):
            t = time.time(); Q, s, ex = flip_opt(W, X, H, "bop", tau=mult * g_scale); add(f"bop tau={mult}x", Q, s, ex, t)
        for kappa in (0.0, 0.5, 1.0, 2.0):
            t = time.time(); Q, s, ex = flip_opt(W, X, H, "snf", kappa=kappa); add(f"snf kappa={kappa}", Q, s, ex, t)
        t = time.time(); Q, s, ex = flip_opt(W, X, H, "snf_noh", kappa=1.0); add("snf_noh kappa=1.0", Q, s, ex, t)
        t = time.time(); Q, s, ex = flip_opt(W, X, H, "snf", kappa=0.5, int8=True); add("snf_int8 kappa=0.5", Q, s, ex, t)
        # старт с GPTQ: может ли переключающий оптимизатор улучшить сильное решение после обучения
        t = time.time(); Q, s, ex = flip_opt(W, X, H, "snf", kappa=0.5, init=(Qg, sg)); add("gptq+snf kappa=0.5", Q, s, ex, t)
        t = time.time(); Q, s, ex = flip_opt(W, X, H, "bop", tau=2.0 * g_scale, init=(Qg, sg)); add("gptq+bop tau=2.0x", Q, s, ex, t)
        report[name] = rows
    json.dump(report, open(OUT / "layer_bench_v2.json", "w"), ensure_ascii=False, indent=1)
    print("\nсохранено:", OUT / "layer_bench.json")


if __name__ == "__main__":
    main()
