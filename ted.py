"""Экономная тернаризация при обучении: одно int8-положение на вес (ternary-aligned fixed-point latent).

Для каждого линейного слоя хранится:
  L  int8  — положение веса в единицах шага сетки s, диапазон [-1.5, 1.5], шаг 1.5/127;
             тернарный код вычисляется на лету: Q = round(L) in {-1, 0, +1};
  m  int8  — момент Adam с масштабом на строку (можно отключить: momentum=False);
  vr, vc   — второй момент в факторизованном виде (строки и столбцы, как в Adafactor);
  s        — масштаб строки (fp32) со своим маленьким Adam.
Итого ~2 байта на вес с моментом и ~1 байт без него — против ~16 у обычного обучения с теневыми
fp32-весами и ~4 у теневых bf16 с вероятностным округлением и 8-битным Adam.

Сдвиги оптимизатора прибавляются к L с вероятностным округлением (иначе мелкие сдвиги теряются).
В прямом проходе всегда точные тернарные веса s*Q — обучается ровно та модель, что будет работать.
В начале L = W0/s, то есть положение исходного веса сохраняется. Шаг оптимизатора делается прямо в
обратном проходе слоя, поэтому градиенты всех слоёв одновременно не хранятся.
Близкая работа: DQT (arXiv 2412.04787) — 8-битные веса с вероятностным округлением при обучении с нуля.
"""
import torch
import torch.nn as nn

STEP = 1.5 / 127  # шаг int8-положения в единицах шага сетки


class Hyper:
    """Общие для всех слоёв параметры шага; t и lr обновляет цикл обучения."""
    t = 0
    lr = 1e-3
    lr_s = 5e-5
    b1, b2, eps = 0.9, 0.95, 1e-8
    momentum = True
    flips = 0


def _sround(x):
    """Вероятностное округление к целому (несмещённое)."""
    return torch.floor(x + torch.rand_like(x))


def ternary_scale(W, grid=40):
    """Масштаб строки по лучшему порогу |w| > delta (TWN)."""
    a = W.abs()
    best_err = torch.full((W.shape[0],), float("inf"), device=W.device)
    best_s = torch.zeros(W.shape[0], device=W.device)
    for qt in torch.linspace(0.05, 0.95, grid):
        delta = torch.quantile(a, qt.item(), dim=1, keepdim=True)
        mask = a > delta
        s = (a * mask).sum(1) / mask.sum(1).clamp_min(1)
        err = ((W - s[:, None] * torch.sign(W) * mask) ** 2).sum(1)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_s = torch.where(better, s, best_s)
    return best_s.clamp_min(1e-8)


class _TedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dummy, layer):
        ctx.save_for_backward(x)
        ctx.layer = layer
        return x @ layer.weight_hat(x.dtype).T

    @staticmethod
    def backward(ctx, go):
        (x,) = ctx.saved_tensors
        L = ctx.layer
        gx = go @ L.weight_hat(go.dtype)
        gW = go.reshape(-1, go.shape[-1]).T.float() @ x.reshape(-1, x.shape[-1]).float()
        L.update(gW)
        return gx, None, None


class TedLinear(nn.Module):
    def __init__(self, lin, momentum=True):
        super().__init__()
        W = lin.weight.detach().float()
        out, inp = W.shape
        s = ternary_scale(W)
        self.register_buffer("s", s)
        self.register_buffer("L", _sround((W / s[:, None]).clamp(-1.5, 1.5) / STEP).clamp(-127, 127).to(torch.int8))
        if momentum:
            self.register_buffer("m", torch.zeros(out, inp, dtype=torch.int8, device=W.device))
            self.register_buffer("m_sc", torch.zeros(out, 1, device=W.device))
        else:
            self.m = None
        self.register_buffer("vr", torch.zeros(out, device=W.device))
        self.register_buffer("vc", torch.zeros(inp, device=W.device))
        self.register_buffer("s_m", torch.zeros(out, device=W.device))
        self.register_buffer("s_v", torch.zeros(out, device=W.device))
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.detach(), requires_grad=False)
        # autograd.Function выдаёт градиент, только если хоть один вход его требует
        self.dummy = nn.Parameter(torch.zeros((), device=W.device), requires_grad=True)

    def codes(self):
        return (self.L.float() * STEP).round().clamp(-1, 1)

    def weight_hat(self, dtype):
        return (self.s[:, None] * self.codes()).to(dtype)

    def forward(self, x):
        y = _TedFn.apply(x, self.dummy, self)
        return y if self.bias is None else y + self.bias

    @torch.no_grad()
    def update(self, gW):
        h = Hyper
        t, b2, eps = max(h.t, 1), h.b2, h.eps
        b1 = h.b1 if self.m is not None else 0.0
        Q = self.codes()
        # масштаб строки: маленький Adam в fp32
        gs = (gW * Q).sum(1)
        self.s_m.mul_(h.b1).add_(gs, alpha=1 - h.b1)
        self.s_v.mul_(b2).add_(gs * gs, alpha=1 - b2)
        self.s.sub_(h.lr_s * (self.s_m / (1 - h.b1 ** t)) / ((self.s_v / (1 - b2 ** t)).sqrt() + eps)).clamp_(min=1e-6)
        # второй момент: строки x столбцы / среднее
        g2 = gW * gW + 1e-30
        self.vr.mul_(b2).add_(g2.mean(1), alpha=1 - b2)
        self.vc.mul_(b2).add_(g2.mean(0), alpha=1 - b2)
        denom = (self.vr[:, None] * self.vc[None, :] / self.vr.mean().clamp_min(1e-30) / (1 - b2 ** t)).sqrt_().add_(eps)
        if self.m is not None:  # момент в int8 с масштабом строки, вероятностное округление
            m = self.m.float().mul_(self.m_sc).mul_(b1).add_(gW, alpha=1 - b1)
            sc = m.abs().amax(1, keepdim=True).clamp_min(1e-30) / 127
            self.m.copy_(_sround(m / sc).clamp_(-127, 127).to(torch.int8))
            self.m_sc.copy_(sc)
            u = m.div_(1 - b1 ** t).div_(denom)
        else:  # без момента: ограничение RMS нормированного обновления строки до 1 (как в Adafactor)
            u = gW / denom
            u.div_(u.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1.0))
        # новое положение: сдвиг в единицах шага сетки, вероятностное округление к int8
        pos = self.L.float().mul_(STEP).sub_(u.mul_(h.lr).div_(self.s[:, None])).clamp_(-1.5, 1.5)
        newL = _sround(pos.div_(STEP)).clamp_(-127, 127)
        Hyper.flips += int(((newL * STEP).round().clamp(-1, 1) != Q).sum())
        self.L.copy_(newL.to(torch.int8))

    def state_bytes(self):
        bufs = [self.L, self.vr, self.vc, self.s, self.s_m, self.s_v] + ([self.m, self.m_sc] if self.m is not None else [])
        return sum(b.numel() * b.element_size() for b in bufs)


def ternarize(model, momentum=True, skip=("lm_head", "embed", "visual", "mtp")):
    """Заменить все nn.Linear внутри слоёв модели на TedLinear; остальное заморозить."""
    for p in model.parameters():
        p.requires_grad_(False)
    layers = []
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            if isinstance(child, nn.Linear) and ".layers." in f".{full}." and not any(k in full for k in skip):
                setattr(mod, cname, TedLinear(child, momentum))
                layers.append(getattr(mod, cname))
    return layers
