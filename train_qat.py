"""Фаза 1б: тернаризация целой модели (SmolLM2-135M) при равном числе токенов.

Все линейные слои блоков становятся тернарными: W_hat = s * Q, Q in {-1, 0, +1}, масштаб s на строку.
Эмбеддинги, нормировки и выходной слой заморожены одинаково для всех методов.

Методы:
  ste — теневые fp32-веса, квантование absmean на каждом шаге, STE + AdamW (как BitNet b1.58 при QAT);
  bop — без теневых весов: накопленный градиент по W_hat > tau -> шаг по сетке (тернарный Bop);
  snf — без теневых весов: шаг по сетке, если выигрыш по квадратичной модели
        -(g dw + 0.5 h dw^2) - kappa * sigma * |dw| > 0, где кривизна h_rj = lam * T^2 * E[go_r^2] * E[x_j^2]
        (диагональ Гаусса-Ньютона в факторизованном виде, как в K-FAC), sigma — шум среднего градиента;
        после шага g <- g + h dw. Масштабы s у bop и snf учатся Adam.
Запуск: ~/.venvs/eg2/bin/python train_qat.py --method snf --tokens 10e6 --name snf_base
"""
import argparse
import json
import math
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.2")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.4")

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
if DEV == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
CACHE = Path.home() / ".cache" / "ternary"
OUT = Path(__file__).parent / "results"
MODEL = "HuggingFaceTB/SmolLM2-135M"
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def guard(reserve_gb=3.0):
    """Сторож памяти Mac: аварийная остановка, если системе остаётся меньше reserve_gb."""
    if DEV != "mps":
        return

    def watch():
        while True:
            if psutil.virtual_memory().available / 1e9 < reserve_gb:
                print("\n[guard] АВАРИЙНАЯ ОСТАНОВКА: мало свободной памяти", flush=True)
                os._exit(3)
            time.sleep(0.5)
    threading.Thread(target=watch, daemon=True).start()


def twn(W, grid=40):
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


class _TernaryFn(torch.autograd.Function):
    """y = x (s*Q)^T; в обратном проходе слой получает градиент по W_hat и статистику кривизны."""

    @staticmethod
    def forward(ctx, x, s, layer):
        W = s[:, None] * layer.Q.float()
        ctx.save_for_backward(x, s)
        ctx.layer = layer
        return x @ W.T

    @staticmethod
    def backward(ctx, go):
        x, s = ctx.saved_tensors
        L = ctx.layer
        Qf = L.Q.float()
        gx = go @ (s[:, None] * Qf)
        x2, go2 = x.reshape(-1, x.shape[-1]), go.reshape(-1, go.shape[-1])
        gW = go2.T @ x2
        L.gW = gW
        L.tokens = x2.shape[0]
        r = L.stat_rate
        L.a.mul_(1 - r).add_((x2 * x2).mean(0), alpha=r)
        L.b.mul_(1 - r).add_((go2 * go2).mean(0), alpha=r)
        return gx, (gW * Qf).sum(1), None


class TernaryLinear(nn.Module):
    def __init__(self, lin, method, anchor=False, ste_bf16=False):
        super().__init__()
        W = lin.weight.detach().float()
        self.method = method
        if method == "ste":
            self.weight = nn.Parameter(W.clone().to(torch.bfloat16 if ste_bf16 else torch.float32))
        else:
            Q, s = twn(W)
            self.register_buffer("Q", Q.to(torch.int8))
            self.s = nn.Parameter(s.clamp_min(1e-8))
            out, inp = W.shape
            self.register_buffer("g", torch.zeros_like(W))
            self.register_buffer("v", torch.zeros_like(W))
            self.register_buffer("a", torch.zeros(inp))
            self.register_buffer("b", torch.zeros(out))
            self.stat_rate = 0.05
            self.gW, self.tokens = None, 0
            self.anchor = anchor
            if anchor:  # исходные веса в 8 битах: 1 байт на вес + масштаб строки
                sc = W.abs().amax(1, keepdim=True).clamp_min(1e-12) / 127
                self.register_buffer("W0q", (W / sc).round().clamp(-127, 127).to(torch.int8))
                self.register_buffer("W0s", sc)

    def weight_hat(self):
        if self.method == "ste":
            W = self.weight
            s = W.abs().mean(1, keepdim=True).clamp_min(1e-8).detach()
            return s * (W / s).round().clamp(-1, 1)
        return self.s[:, None] * self.Q.float()

    def forward(self, x):
        if self.method == "ste":
            W = self.weight.float()
            s = W.abs().mean(1, keepdim=True).clamp_min(1e-8).detach()
            Wq = s * (W / s).round().clamp(-1, 1)
            return F.linear(x, W + (Wq - W).detach())
        return _TernaryFn.apply(x, self.s, self)

    @torch.no_grad()
    def flip_step(self, cfg, step):
        if self.gW is None:
            return 0
        gr = self.gW
        be = cfg.ema
        self.g.mul_(1 - be).add_(gr, alpha=be)
        self.v.mul_(cfg.beta2).add_(gr * gr, alpha=1 - cfg.beta2)
        if cfg.int8:
            for name in ("g", "v"):
                t = getattr(self, name)
                sc = t.abs().amax(1, keepdim=True).clamp_min(1e-30) / 127
                t.copy_((t / sc).round().clamp(-127, 127) * sc)
        if step < cfg.calib_step:  # сначала накопить градиент и статистику кривизны
            self.gW = None
            return 0
        s = self.s.detach()[:, None]
        Q = self.Q.float()
        if self.method == "bop":
            if not hasattr(self, "tau"):  # порог задаётся относительно типичного накопленного градиента
                self.tau = cfg.tau * float(self.g.abs().mean())
            gain = self.g.abs() - self.tau
            dq = -torch.sign(self.g)
            h = None
        else:
            T = self.tokens
            h = cfg.lam * (T * T) * self.b[:, None] * self.a[None, :]
            n_eff = (2 - be) / be
            sigma = ((self.v - self.g * self.g).clamp_min(0) / n_eff).sqrt()
            gains = [-(self.g * (s * d) + 0.5 * h * (s * d) ** 2) - cfg.kappa * sigma * s.abs() for d in (-1.0, 1.0)]
            if self.anchor and cfg.anchor_mu > 0:
                # стоимость ухода от исходного веса w0: 0.5 * mu * h * ((w_new - w0)^2 - (w_old - w0)^2)
                w0 = self.W0q.float() * self.W0s
                w_old = s * Q
                for i, d in enumerate((-1.0, 1.0)):
                    w_new = s * (Q + d).clamp(-1, 1)
                    gains[i] = gains[i] - 0.5 * cfg.anchor_mu * h * ((w_new - w0) ** 2 - (w_old - w0) ** 2)
            gain = torch.maximum(gains[0], gains[1])
            dq = torch.where(gains[1] > gains[0], 1.0, -1.0)
        newQ = (Q + dq).clamp(-1, 1)
        ok = (gain > 0) & (newQ != Q)
        k = max(1, int(cfg.max_frac * Q.numel()))
        if int(ok.sum()) > k:
            thr = torch.topk(torch.where(ok, gain, torch.full_like(gain, -float("inf"))).flatten(), k).values[-1]
            ok &= gain >= thr
        n = int(ok.sum())
        if n:
            if h is None:
                self.g.masked_fill_(ok, 0.0)
            else:
                self.g.add_(torch.where(ok, h * s * (newQ - Q), torch.zeros_like(self.g)))
            self.Q.copy_(torch.where(ok, newQ, Q).to(torch.int8))
        self.gW = None
        return n


class TedLinear(nn.Module):
    """Тернарный слой с остатком внутри ячейки сетки (ternary error diffusion).

    Вместо теневой копии веса хранится Q in {-1,0,+1} и остаток r — положение веса внутри
    ячейки в единицах шага сетки s. Обновление Adam прибавляется к r; когда r выходит за
    половину ячейки, вес переходит на соседнее значение, а r сдвигается на 1. В начале
    r = W0/s - Q, т.е. положение исходного веса — память об исходной модели сохраняется.
    r хранится в int8 (шаг 1/127, диапазон [-1, 1]) с вероятностным или обычным округлением;
    момент Adam — fp32 или int8; второй момент — полный или факторизованный (как в Adafactor).
    """

    def __init__(self, lin, cfg):
        super().__init__()
        W = lin.weight.detach().float()
        Q, s = twn(W)
        out, inp = W.shape
        self.cfg = cfg
        self.register_buffer("Q", Q.to(torch.int8))
        self.s = nn.Parameter(s.clamp_min(1e-8))
        r = (W / s[:, None] - Q).clamp(-1, 1)
        self.register_buffer("r", r.clone())
        self.register_buffer("m", torch.zeros_like(W))
        if cfg.v_mode == "full":
            self.register_buffer("v", torch.zeros_like(W))
        else:
            self.register_buffer("vr", torch.zeros(out))
            self.register_buffer("vc", torch.zeros(inp))
        self.register_buffer("a", torch.zeros(inp))
        self.register_buffer("b", torch.zeros(out))
        self.stat_rate = 0.05
        self.gW, self.tokens = None, 0
        self._store_r()

    def _store_r(self):
        if self.cfg.r_fp32:
            return
        x = self.r * 127
        x = torch.floor(x + torch.rand_like(x)) if self.cfg.round == "stochastic" else x.round()
        self.r.copy_(x.clamp(-127, 127) / 127)

    def forward(self, x):
        return _TernaryFn.apply(x, self.s, self)

    @torch.no_grad()
    def ted_step(self, t, lr):
        if self.gW is None:
            return 0
        cfg, g = self.cfg, self.gW
        b1, b2, eps = (0.0 if cfg.no_momentum else cfg.b1), 0.95, 1e-8
        self.m.mul_(b1).add_(g, alpha=1 - b1)
        if cfg.m_int8:
            sc = self.m.abs().amax(1, keepdim=True).clamp_min(1e-30) / 127
            mq = self.m / sc
            mq = torch.floor(mq + torch.rand_like(mq)) if cfg.m_round == "stochastic" else mq.round()
            self.m.copy_(mq.clamp(-127, 127) * sc)
        if cfg.v_mode == "full":
            self.v.mul_(b2).add_(g * g, alpha=1 - b2)
            v = self.v
        else:  # Adafactor: v ~ (строки x столбцы) / среднее, память O(out + in)
            g2 = g * g + 1e-30
            self.vr.mul_(b2).add_(g2.mean(1), alpha=1 - b2)
            self.vc.mul_(b2).add_(g2.mean(0), alpha=1 - b2)
            v = self.vr[:, None] * self.vc[None, :] / self.vr.mean().clamp_min(1e-30)
        mh = self.m / (1 - b1 ** t)
        vh = v / (1 - b2 ** t)
        u = mh / (vh.sqrt() + eps)
        if cfg.no_momentum:  # ограничение RMS нормированного обновления строки (как в Adafactor)
            u = u / (u.pow(2).mean(1, keepdim=True).sqrt() / 1.0).clamp_min(1.0)
        delta = -lr * u
        if cfg.latent_int8:  # одно int8-положение L = Q + r в диапазоне [-1.5, 1.5], шаг 1.5/127
            L = (self.Q.float() + self.r + delta / self.s.detach()[:, None]).clamp(-1.5, 1.5)
            if cfg.grid_band > 0:  # неравномерная сетка: густо у порогов ±0.5, редко вдали
                L = grid_sround(L, GRID[0].to(L.device)) if cfg.round == "stochastic" else L
                Qn = torch.sign(L) * (L.abs() >= 0.5 - 1e-6)
            else:
                lim = 2 ** (cfg.latent_bits - 1) - 1
                step_ = 1.5 / lim
                x = L / step_
                x = torch.floor(x + torch.rand_like(x)) if cfg.round == "stochastic" else x.round()
                L = x.clamp(-lim, lim) * step_
                Qn = L.round().clamp(-1, 1)
            n = int((Qn != self.Q.float()).sum())
            self.Q.copy_(Qn.to(torch.int8))
            self.r.copy_(L - Qn)
            self.gW = None
            return n
        r = self.r + delta / self.s.detach()[:, None]
        Q = self.Q.float()
        up = (r > 0.5) & (Q < 1)
        dn = (r < -0.5) & (Q > -1)
        Q = Q + up.float() - dn.float()
        r = r - up.float() + dn.float()
        self.Q.copy_(Q.to(torch.int8))
        self.r.copy_(r.clamp(-1, 1))
        self._store_r()
        self.gW = None
        return int(up.sum() + dn.sum())


GRID = [None]


def make_grid(band, n_band, lim=127, top=1.5):
    """Неравномерная сетка из 2*lim+1 положений: n_band делений в полосе 0.5±band, остальные поровну
    по ширине на [0, 0.5-band] и [0.5+band, top]. Точка 0.5 входит в сетку."""
    rest = lim - n_band
    wa, wc = 0.5 - band, top - 0.5 - band
    na = max(1, round(rest * wa / (wa + wc)))
    nc = rest - na
    a = torch.linspace(0, wa, na + 1)
    b = torch.linspace(wa, 0.5 + band, n_band + 1)[1:]
    c = torch.linspace(0.5 + band, top, nc + 1)[1:]
    pos = torch.cat([a, b, c])
    return torch.cat([-pos.flip(0)[:-1], pos])


def grid_sround(x, g):
    """Вероятностное округление к соседним точкам сетки g (несмещённое)."""
    i = (torch.bucketize(x, g, right=True) - 1).clamp(0, g.numel() - 2)
    lo, hi = g[i], g[i + 1]
    p = (x - lo) / (hi - lo)
    return torch.where(torch.rand_like(x) < p, hi, lo)


def ternarize(model, method, anchor=False, ste_bf16=False, cfg=None):
    layers = []
    for name, mod in list(model.named_modules()):
        for t in TARGETS:
            if hasattr(mod, t) and isinstance(getattr(mod, t), nn.Linear):
                tl = (TedLinear(getattr(mod, t), cfg) if method == "ted" else
                      TernaryLinear(getattr(mod, t), method, anchor, ste_bf16)).to(DEV)
                setattr(mod, t, tl)
                layers.append(tl)
    return layers


@torch.no_grad()
def evaluate(model, val, seq):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(val), 8):
        x = torch.from_numpy(val[i:i + 8].astype(np.int64)).to(DEV)
        out = model(x[:, :-1])
        loss = F.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]).float(), x[:, 1:].reshape(-1), reduction="sum")
        tot += float(loss)
        n += x[:, 1:].numel()
    model.train()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["fp", "ste", "bop", "snf", "ted"], required=True)
    ap.add_argument("--r-fp32", action="store_true", help="ted: остаток в fp32 (проверка, должно совпасть с ste)")
    ap.add_argument("--round", choices=["stochastic", "nearest"], default="stochastic", help="ted: округление остатка")
    ap.add_argument("--m-int8", action="store_true", help="ted: момент Adam в int8")
    ap.add_argument("--v-mode", choices=["full", "factored"], default="full", help="ted: второй момент Adam")
    ap.add_argument("--latent-int8", action="store_true",
                    help="ted: код и остаток слиты в одно int8-положение в диапазоне [-1.5, 1.5] шага сетки")
    ap.add_argument("--kd", type=float, default=0.0, help="доля дистилляции по выходу (KL к исходной модели)")
    ap.add_argument("--off", type=float, default=0.0, help="вес дистилляции признаков OFF (косинус скрытых состояний)")
    ap.add_argument("--off-layers", type=int, default=15, help="сколько первых слоёв в OFF")
    ap.add_argument("--b1", type=float, default=0.9, help="ted: коэффициент момента")
    ap.add_argument("--m-round", choices=["nearest", "stochastic"], default="nearest", help="ted --m-int8: округление момента")
    ap.add_argument("--grid-band", type=float, default=0.0, help="ted --latent-int8: полуширина густой полосы у порога (0 — сетка равномерная)")
    ap.add_argument("--grid-n", type=int, default=90, help="сколько из 127 делений положить в густую полосу")
    ap.add_argument("--latent-bits", type=int, default=8, help="ted --latent-int8: разрядность положения (8 или 16)")
    ap.add_argument("--no-momentum", action="store_true",
                    help="ted: без момента (как Adafactor), с ограничением RMS обновления строки до 1")
    ap.add_argument("--tokens", type=float, default=10e6)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4, help="AdamW для теневых весов (ste)")
    ap.add_argument("--lr-s", type=float, default=5e-5, help="Adam для масштабов строк (bop, snf)")
    ap.add_argument("--tau", type=float, default=2.0, help="порог bop в единицах среднего |g|")
    ap.add_argument("--kappa", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=1.0, help="множитель оценки кривизны")
    ap.add_argument("--ema", type=float, default=0.05)
    ap.add_argument("--beta2", type=float, default=0.99)
    ap.add_argument("--max-frac", type=float, default=0.002, help="не больше этой доли весов слоя за шаг")
    ap.add_argument("--int8", action="store_true", help="накопители g и v в 8 битах")
    ap.add_argument("--anchor-mu", type=float, default=0.0,
                    help="якорь: штраф за уход от исходного веса (хранится в int8), вес по той же кривизне")
    ap.add_argument("--ste-bf16", action="store_true", help="теневые веса и состояния AdamW в bf16 (экономия памяти)")
    ap.add_argument("--calib-step", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--ste-sr16", action="store_true",
                    help="ste: после каждого шага теневые веса округляются до точности bf16 вероятностно (известный приём)")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    if a.grid_band > 0:
        GRID[0] = make_grid(a.grid_band, a.grid_n)

    from transformers import AutoModelForCausalLM
    guard()
    torch.manual_seed(a.seed)
    data = np.load(CACHE / "fineweb_edu_smollm2.npy")
    n_val = 120 * (a.seq + 1)
    val = data[-n_val:].reshape(-1, a.seq + 1)
    train = data[:-n_val]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV)
    fp_loss = evaluate(model, val, a.seq)
    for p in model.parameters():
        p.requires_grad_(False)
    teacher = None
    if a.kd > 0 or a.off > 0:  # учитель — копия исходной модели до тернаризации
        import copy
        teacher = copy.deepcopy(model).eval()
    layers = ternarize(model, a.method, a.anchor_mu > 0, a.ste_bf16, a) if a.method != "fp" else []
    if a.method == "ste":
        params = [l.weight for l in layers]
    else:
        params = [l.s for l in layers]
    for p in params:
        p.requires_grad_(True)
    n_tern = sum(l.weight.numel() if a.method == "ste" else l.Q.numel() for l in layers)
    lr = a.lr if a.method == "ste" else a.lr_s
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0) if params else None
    steps = int(a.tokens // (a.batch * a.seq))
    warm = min(100, steps // 10)
    lr_factor = lambda t: (t + 1) / warm if t < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (t - warm) / max(1, steps - warm)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor) if opt else None
    OUT.mkdir(exist_ok=True)
    log = open(OUT / f"qat_{a.name}.jsonl", "w")
    start_loss = evaluate(model, val, a.seq)
    rec = {"step": 0, "tokens": 0, "val_loss": start_loss, "fp_loss": fp_loss, "ternary_params": n_tern, "args": vars(a)}
    log.write(json.dumps(rec) + "\n")
    print(f"[{a.name}] исходная модель {fp_loss:.4f}, после тернаризации {start_loss:.4f}, тернарных весов "
          f"{n_tern / 1e6:.1f} млн, шагов {steps}", flush=True)
    rng = np.random.default_rng(a.seed)
    t0, flips_acc, run_loss = time.time(), 0, 0.0
    for step in range(1, steps + 1):
        idx = rng.integers(0, len(train) - a.seq - 1, a.batch)
        x = torch.from_numpy(np.stack([train[i:i + a.seq + 1] for i in idx]).astype(np.int64)).to(DEV)
        out = model(x[:, :-1], output_hidden_states=a.off > 0)
        logits = out.logits.reshape(-1, out.logits.shape[-1])
        loss = F.cross_entropy(logits, x[:, 1:].reshape(-1))
        if teacher is not None:
            with torch.no_grad():
                tout = teacher(x[:, :-1], output_hidden_states=a.off > 0)
            if a.kd > 0:  # дистилляция по выходу: KL(учитель || ученик)
                pt = tout.logits.reshape(-1, logits.shape[-1]).float().softmax(-1)
                kl = (pt * (pt.clamp_min(1e-30).log() - logits.float().log_softmax(-1))).sum(-1).mean()
                loss = (1 - a.kd) * loss + a.kd * kl
            if a.off > 0:  # OFF (TernaryLLM): 1 - косинус скрытых состояний по токенам, первые off_layers слоёв
                feat = sum((1 - F.cosine_similarity(out.hidden_states[l].float(), tout.hidden_states[l].float(), dim=-1)).mean()
                           for l in range(1, a.off_layers + 1))
                loss = loss + a.off * feat
        if opt:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        if opt:
            opt.step()
            sched.step()
        if a.method == "ste" and a.ste_sr16:  # вероятностное округление теневых весов до bf16
            with torch.no_grad():
                for l in layers:
                    w = l.weight.data
                    bits = w.view(torch.int32)
                    noise = torch.randint_like(bits, 0, 1 << 16)
                    l.weight.data.copy_(((bits + noise) & ~0xFFFF).view(torch.float32))
        if a.method == "ted":
            with torch.no_grad():
                for l in layers:
                    l.s.clamp_(min=1e-6)
            flips_acc += sum(l.ted_step(step, a.lr * lr_factor(step - 1)) for l in layers)
        if a.method in ("bop", "snf"):
            with torch.no_grad():
                for l in layers:
                    l.s.clamp_(min=1e-6)
            flips_acc += sum(l.flip_step(a, step) for l in layers)
        run_loss += float(loss)
        if step % a.eval_every == 0 or step == steps:
            vl = evaluate(model, val, a.seq)
            dt = time.time() - t0
            rec = {"step": step, "tokens": step * a.batch * a.seq, "val_loss": vl, "train_loss": run_loss / a.eval_every,
                   "flips_pct_per_step": 100 * flips_acc / a.eval_every / max(n_tern, 1), "tok_per_s": step * a.batch * a.seq / dt,
                   "mps_gb": (torch.mps.driver_allocated_memory() / 1e9 if DEV == "mps" else
                              torch.cuda.max_memory_allocated() / 1e9 if DEV == "cuda" else None),
                   "free_gb": psutil.virtual_memory().available / 1e9}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"  шаг {step}/{steps}: проверка {vl:.4f}, обучение {run_loss / a.eval_every:.4f}, "
                  f"переключений {rec['flips_pct_per_step']:.4f}%/шаг, {rec['tok_per_s']:.0f} ток/с, "
                  f"видеопамять {rec['mps_gb'] or 0:.1f} ГБ, свободно {rec['free_gb']:.1f} ГБ", flush=True)
            flips_acc, run_loss = 0, 0.0
    print("готово", flush=True)


if __name__ == "__main__":
    main()
