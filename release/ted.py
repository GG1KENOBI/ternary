"""TED: тернаризация языковых моделей при обучении без fp32-копии весов.

Идея. Обычное обучение тернарной модели (BitNet b1.58 и его QAT-варианты) хранит рядом с каждым
тернарным весом точную fp32-копию и два состояния Adam: ~16 байт на вес. Здесь вместо копии хранится
одно целое число L (16 или 8 бит) — положение веса в единицах шага сетки s внутри диапазона [-1.5, 1.5].
Тернарный код берётся на лету: Q = round(L * step) in {-1, 0, +1}, вес в прямом проходе — s*Q.

Шаг обучения: L <- SR(clip(L - lr * u / s)), где u — нормированное обновление Adam, а SR —
вероятностное округление к сетке. Так как E[SR(x)] = x, положение в среднем движется так же, как точная
копия, и мелкие сдвиги не теряются (при обычном округлении сдвиг меньше половины деления обнуляется —
обучение «залипает»). Цена — шум: его дисперсия примерно равна шагу сетки, умноженному на полный путь
веса. Поэтому 16-битное положение (шаг в 258 раз мельче) заметно лучше 8-битного, а момент Adam,
сглаживающий путь, важен. Начальное положение L = W0/s сохраняет, где исходный вес лежал внутри ячейки.

Память на вес: L (2 или 1 байт) + момент Adam в int8 (1 байт, можно отключить) + второй момент Adam в
факторизованном виде (строки и столбцы, как в Adafactor — O(n+m) на матрицу).
Итого ~3 байта (16 бит) или ~2 байта (8 бит) против ~16 у обычного QAT. Шаг оптимизатора делается прямо
в обратном проходе слоя, поэтому градиенты всех слоёв одновременно не хранятся.

Близкая работа: DQT (arXiv 2412.04787) — int8-веса с вероятностным округлением при обучении с нуля,
прямой проход в int8 и масштаб на матрицу; здесь — тернарный прямой проход, старт из обученной
модели, масштаб на строку и сжатый оптимизатор.
"""
import torch
import torch.distributed as dist
import torch.nn as nn

class Latent:
    """Разрядность положения L: 8 бит (по умолчанию, ~2 байта на вес вместе с моментом) или 16 бит
    (~3 байта на вес, меньше шума округления). Положение = L * step в диапазоне [-1.5, 1.5];
    код Q = ±1 при |L| >= thr, иначе 0 (то же, что round(L * step), проверено для всех значений)."""
    bits, lim, step, thr, dtype = 8, 127, 1.5 / 127, 43, torch.int8


def set_latent_bits(bits=8):
    assert bits in (8, 16), "поддерживаются 8 и 16 бит"
    lim = 2 ** (bits - 1) - 1
    Latent.bits, Latent.lim, Latent.step = bits, lim, 1.5 / lim
    Latent.thr = int(0.5 / Latent.step) + 1  # 43 для 8 бит, 10923 для 16 бит
    Latent.dtype = torch.int8 if bits == 8 else torch.int16


def _to_index(pos):
    """Положение -> целочисленный индекс с вероятностным (несмещённым) округлением."""
    return _sround(pos / Latent.step).clamp_(-Latent.lim, Latent.lim)


def _to_pos(L):
    return L.float().mul_(Latent.step)


class Hyper:
    """Общие для всех слоёв параметры шага; t и lr обновляет цикл обучения."""
    t = 0
    lr = 1e-3
    lr_s = 5e-5
    b1, b2, eps = 0.9, 0.95, 1e-8


def _sround(x):
    """Вероятностное округление к целому (несмещённое)."""
    return torch.floor(x + torch.rand_like(x))


def ternary_scale(W, grid=40):
    """Масштаб строки по лучшему порогу |w| > delta (Ternary Weight Networks)."""
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


class Fmt:
    """Формат тернарного слоя.
    group: сколько подряд идущих весов строки делят один масштаб (0 — вся строка). 256 совпадает с блоком
           TQ2_0 в llama.cpp, поэтому экспорт остаётся точным, а масштабов в inp/256 раз больше — точнее.
    hadamard: размер блока поворота Адамара по входу (0 — без поворота). Слой считает W·(R·x), где
           R = H·D / sqrt(b) поблочно (H — матрица Адамара, D — случайные знаки). Поворот размазывает выбросы,
           и тернарная сетка ложится точнее (как в Bonsai). Для запуска нужна среда, умеющая поворачивать входы."""
    group = 0
    hadamard = 0
    seed = 1234


def set_format(group=0, hadamard=0):
    Fmt.group, Fmt.hadamard = int(group), int(hadamard)
    _HCACHE.clear()


_HCACHE = {}


def _hadamard(b, dev, dtype):
    key = (b, str(dev), dtype)
    if key not in _HCACHE:
        H = torch.ones(1, 1)
        while H.shape[0] < b:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        _HCACHE[key] = (H / b ** 0.5).to(dev, dtype)
    return _HCACHE[key]


def _signs(n, dev):
    g = torch.Generator().manual_seed(Fmt.seed + n)
    return (torch.randint(0, 2, (n,), generator=g) * 2 - 1).to(torch.int8).to(dev)


def rot(v, signs, b):
    """R·v по последней оси: знаки, затем поблочный Адамар (ортогонально, R^-1 = R^T)."""
    sh = v.shape
    w = (v * signs.to(v.dtype)).reshape(*sh[:-1], sh[-1] // b, b)
    return (w @ _hadamard(b, v.device, v.dtype)).reshape(sh)


def rot_t(v, signs, b):
    """R^T·v: поблочный Адамар (он симметричен), затем знаки."""
    sh = v.shape
    w = v.reshape(*sh[:-1], sh[-1] // b, b) @ _hadamard(b, v.device, v.dtype)
    return w.reshape(sh) * signs.to(v.dtype)


class _TedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dummy, layer):
        if layer.hb:
            x = rot(x, layer.hs, layer.hb)
        ctx.save_for_backward(x)
        ctx.layer = layer
        return x @ layer.weight_hat(x.dtype).T

    @staticmethod
    def backward(ctx, go):
        (x,) = ctx.saved_tensors
        L = ctx.layer
        gx = go @ L.weight_hat(go.dtype)
        gW = (go.reshape(-1, go.shape[-1]).T @ x.reshape(-1, x.shape[-1])).float()  # матричное умножение в bf16
        L.update(gW)
        if L.hb:
            gx = rot_t(gx, L.hs, L.hb)
        return gx, None, None


class TedLinear(nn.Module):
    def __init__(self, lin, momentum=True):
        super().__init__()
        W = lin.weight.detach().float()
        out, inp = W.shape
        dev = W.device
        self.hb = Fmt.hadamard if Fmt.hadamard and inp % Fmt.hadamard == 0 else 0
        self.register_buffer("hs", _signs(inp, dev) if self.hb else torch.ones(1, dtype=torch.int8, device=dev))
        if self.hb:
            W = rot(W, self.hs, self.hb)  # храним тернарную матрицу в повёрнутом базисе
        self.g = Fmt.group if Fmt.group and inp % Fmt.group == 0 else inp
        G = inp // self.g
        s = ternary_scale(W.reshape(out * G, self.g)).reshape(out, G)
        self.register_buffer("s", s)
        self.register_buffer("L", _to_index((W.reshape(out, G, self.g) / s[:, :, None]).clamp(-1.5, 1.5))
                             .reshape(out, inp).to(Latent.dtype))
        del W
        if momentum:
            self.register_buffer("m", torch.zeros(out, inp, dtype=torch.int8, device=dev))
            self.register_buffer("m_sc", torch.zeros(out, 1, device=dev))
        else:
            self.m = None
        self.register_buffer("vr", torch.zeros(out, device=dev))
        self.register_buffer("vc", torch.zeros(inp, device=dev))
        self.register_buffer("s_m", torch.zeros(out, G, device=dev))
        self.register_buffer("s_v", torch.zeros(out, G, device=dev))
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.detach(), requires_grad=False)
        # autograd.Function выдаёт градиент, только если хоть один вход его требует
        self.dummy = nn.Parameter(torch.zeros((), device=dev), requires_grad=True)
        self.nflip = torch.zeros((), dtype=torch.long, device=dev)

    def codes(self, dtype=torch.float32):
        return ((self.L >= Latent.thr).to(dtype) - (self.L <= -Latent.thr).to(dtype))

    def weight_hat(self, dtype):
        """Тернарная матрица s*Q (в повёрнутом базисе, если включён поворот)."""
        out, inp = self.L.shape
        return (self.codes(dtype).view(out, -1, self.g) * self.s.to(dtype)[:, :, None]).view(out, inp)

    def weight_dense(self, dtype):
        """Эквивалентная обычная матрица: W такая, что W·x = s*Q·(R·x). Без поворота совпадает с weight_hat."""
        w = self.weight_hat(torch.float32)
        return (rot_t(w, self.hs, self.hb) if self.hb else w).to(dtype)

    def forward(self, x):
        y = _TedFn.apply(x, self.dummy, self)
        return y if self.bias is None else y + self.bias

    @torch.no_grad()
    def update(self, gW):
        h = Hyper
        t, b2, eps = max(h.t, 1), h.b2, h.eps
        b1 = h.b1 if self.m is not None else 0.0
        gW = gW.to(self.s.device)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            # несколько видеокарт, у каждой своя часть данных: усредняем градиент перед шагом.
            # Состояние и случайность округления на всех картах одинаковы, поэтому шаги совпадают.
            dist.all_reduce(gW)
            gW.div_(dist.get_world_size())
        out, inp = gW.shape
        Q = self.codes()
        # масштабы групп: маленький Adam в fp32
        gs = (gW * Q).view(out, -1, self.g).sum(-1)
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
        u.mul_(h.lr).view(out, -1, self.g).div_(self.s[:, :, None])
        pos = _to_pos(self.L).sub_(u).clamp_(-1.5, 1.5)
        newL = _to_index(pos)
        self.L.copy_(newL.to(Latent.dtype))
        self.nflip += (self.codes() != Q).sum()  # без синхронизации с CPU на каждом слое

    def state_bytes(self):
        bufs = [self.L, self.vr, self.vc, self.s, self.s_m, self.s_v] + ([self.m, self.m_sc] if self.m is not None else [])
        return sum(b.numel() * b.element_size() for b in bufs)


def pop_flips(layers):
    """Сколько тернарных кодов сменилось с прошлого вызова (по всем слоям)."""
    n = sum(int(l.nflip) for l in layers.values())
    for l in layers.values():
        l.nflip.zero_()
    return n


# in_proj_a/in_proj_b — крошечные матрицы шлюзов линейного внимания Qwen3.5 (0,04% параметров): их оставляем
# в полной точности, как Bonsai; среда с поворотом (форк Prism) для них поворот и не поддерживает.
SKIP = ("lm_head", "embed", "visual", "vision", "mtp", "audio", "in_proj_a", "in_proj_b")


def ternarize(model, momentum=True, skip=SKIP):
    """Заменить все nn.Linear внутри декодерных слоёв на TedLinear; остальное заморозить.
    Формат (группа масштабов, поворот) берётся из Fmt — задайте его через set_format до вызова.
    Буферы создаются на том же устройстве, где лежал слой (работает и при разнесении по видеокартам)."""
    for p in model.parameters():
        p.requires_grad_(False)
    # сначала только имена: если держать ссылки на старые слои, их bf16-веса не освободятся до конца цикла
    targets = [(name, cname) for name, mod in model.named_modules() for cname, child in mod.named_children()
               if isinstance(child, nn.Linear) and ".layers." in f".{name}.{cname}." and
               not any(k in f"{name}.{cname}" for k in skip)]
    layers = {}
    for name, cname in targets:
        parent = model.get_submodule(name) if name else model
        full = f"{name}.{cname}" if name else cname
        layers[full] = TedLinear(getattr(parent, cname), momentum)
        setattr(parent, cname, layers[full])  # старый nn.Linear больше ни на что не ссылается и освобождается
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return layers


@torch.no_grad()
def sync_state(layers, src=0):
    """Выровнять состояние всех карт по карте src (страховка от расхождения; дёшево, раз в N шагов)."""
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return 0
    diff = 0
    for l in layers.values():
        for k, v in l.state_dict().items():
            if k == "dummy":
                continue
            # пересылаем как байты: int16 не поддерживают ни NCCL, ни gloo
            mine = v.detach().contiguous().reshape(-1).view(torch.uint8)
            ref = mine.clone()
            dist.broadcast(ref, src)
            diff += int((ref != mine).sum())
            v.copy_(ref.view(v.dtype).view(v.shape))
    return diff


def save_state(layers, path, meta):
    """Точка сохранения: только состояние тернарных слоёв (int8 + маленькие fp32-векторы) и метаданные."""
    torch.save({"meta": meta, "layers": {n: {k: v.cpu() for k, v in l.state_dict().items() if k != "dummy"}
                                         for n, l in layers.items()}}, path)


def load_state(layers, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for n, l in layers.items():
        st = ck["layers"][n]
        for k, v in st.items():
            buf = getattr(l, k)
            buf.copy_(v.to(buf.device).view(buf.shape))  # view: старые точки хранили масштаб строки как [out]
    return ck["meta"]


@torch.no_grad()
def export_ternary(layers, path):
    """Компактный итог: коды по 2 бита (4 веса в байте) + масштабы групп (+ знаки поворота, если он был)."""
    out = {}
    for n, l in layers.items():
        q = (l.codes(torch.int8) + 1).to(torch.uint8).flatten()  # {0, 1, 2}
        pad = (-q.numel()) % 4
        if pad:
            q = torch.cat([q, torch.zeros(pad, dtype=torch.uint8, device=q.device)])
        q = q.view(-1, 4)
        packed = q[:, 0] | (q[:, 1] << 2) | (q[:, 2] << 4) | (q[:, 3] << 6)
        out[n] = {"packed": packed.cpu(), "shape": tuple(l.L.shape), "s": l.s.cpu(), "group": l.g,
                  "hadamard": l.hb, "signs": l.hs.cpu() if l.hb else None,
                  "bias": None if l.bias is None else l.bias.detach().cpu()}
    torch.save(out, path)


@torch.no_grad()
def materialize(model, layers, dtype=torch.bfloat16):
    """Заменить TedLinear обратно на обычные nn.Linear (для save_pretrained и llama.cpp).
    С поворотом веса получаются эквивалентными, но уже не тернарными."""
    for full, l in layers.items():
        parent_name, _, cname = full.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        out, inp = l.L.shape
        lin = nn.Linear(inp, out, bias=l.bias is not None, device=l.L.device, dtype=dtype)
        lin.weight.copy_(l.weight_dense(dtype))
        if l.bias is not None:
            lin.bias.copy_(l.bias)
        setattr(parent, cname, lin)
    return model


@torch.no_grad()
def load_packed(model, path, dtype=torch.bfloat16, keep_rotated=False):
    """Подставить в исходную модель тернарные веса из export_ternary (для проверки и сохранения).
    Если в файле был поворот: по умолчанию веса пересчитываются в эквивалентную обычную матрицу (не тернарную,
    для проверки в PyTorch); с keep_rotated=True остаются тернарными в повёрнутом базисе — для среды,
    которая сама поворачивает входы (форк llama.cpp от Prism, см. export.py --prism)."""
    packed = torch.load(path, map_location="cpu", weights_only=False)
    for full, rec in packed.items():
        lin = model.get_submodule(full)
        dev = lin.weight.device
        out, inp = rec["shape"]
        p = rec["packed"].to(dev)
        q = torch.stack([(p >> k) & 3 for k in (0, 2, 4, 6)], 1).flatten()[: out * inp]
        Q = (q.to(torch.int8) - 1).float().view(out, inp)
        s = rec["s"].to(dev).float()
        s = s.view(out, -1)  # старые файлы: масштаб на строку [out] -> [out, 1]
        g = inp // s.shape[1]
        W = (Q.view(out, -1, g) * s[:, :, None]).view(out, inp)
        if rec.get("hadamard") and not (keep_rotated and prism_foldable(full)):
            W = rot_t(W, rec["signs"].to(dev), rec["hadamard"])
        lin.weight.copy_(W.to(dtype))
        if rec["bias"] is not None:
            lin.bias.copy_(rec["bias"])
    return model


def packed_is_rotated(path):
    return any(r.get("hadamard") for r in torch.load(path, map_location="cpu", weights_only=False).values())


PRISM_NOT_FOLDABLE = ("in_proj_a", "in_proj_b")  # шлюзы линейного внимания: форк Prism их не поворачивает


def prism_foldable(name):
    return not any(k in name for k in PRISM_NOT_FOLDABLE)


def prism_manifest(path):
    """Описание поворота для конвертера форка Prism (hadamard_packing.json): какие матрицы хранятся повёрнутыми,
    блок и знаки. Их среда считает W·(H·(signs*x)) — тот же порядок, что rot() здесь."""
    packed = torch.load(path, map_location="cpu", weights_only=False)
    rotated = {n: r for n, r in packed.items() if r.get("hadamard") and prism_foldable(n)}
    blocks = {r["hadamard"] for r in rotated.values()}
    assert len(blocks) == 1, f"в одной модели нужен один размер блока поворота, а не {blocks}"
    signs = {}
    for r in rotated.values():
        signs.setdefault(str(r["shape"][1]), [int(v) for v in r["signs"].tolist()])
    return {
        "schema_version": 1, "kind": "hadamard-weight-fold", "status": "requires-matching-runtime",
        "transform": {"name": "normalized-signed-sylvester-walsh-hadamard", "block_size": blocks.pop(),
                      "sign_mode": "explicit"},
        "signs": signs, "tied_output": False,
        "tensors": [{"name": f"{n}.weight", "axis": -1, "role": "fold-before-matmul"} for n in sorted(rotated)],
    }
