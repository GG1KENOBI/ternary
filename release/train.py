"""Тернаризация модели целиком методом TED (ted.py), с дистилляцией, сохранением и продолжением.

Все nn.Linear внутри декодерных слоёв становятся тернарными с одним int8-положением на вес;
эмбеддинги, нормировки и выходной слой заморожены. Шаг оптимизатора — внутри обратного прохода слоя,
активации пересчитываются (gradient checkpointing), потери считаются кусками без хранения всех логитов.

Дистилляция (--kd 0.5): к обычной кросс-энтропии добавляется расхождение с исходной моделью
(учитель — та же модель до тернаризации). Сильно ускоряет восстановление качества, но держит
учителя в памяти (bf16, ~2 байта на параметр).

Сохранение каждые --save-every шагов в <out>/ckpt.pt; повторный запуск с той же --out продолжит
с последней точки. В конце: <out>/ternary_packed.pt (2 бита на вес + масштабы) и, с --save-hf,
обычная HF-модель с весами s*Q для замеров и перевода в llama.cpp.

Примеры:
  # одна видеокарта 24–48 ГБ, модель до ~4–8B
  python train.py --model Qwen/Qwen3.5-4B --data data/qwen35 --out runs/q4b --tokens 100e6 --kd 0.5
  # три видеокарты, у каждой полная копия модели и своя часть данных (модели до ~8B на картах по 48 ГБ)
  torchrun --nproc_per_node 3 train.py --model Qwen/Qwen3-8B --data data/q8 --out runs/q8 --tokens 300e6 --kd 0.5
  # модель не помещается в одну карту: раскладывается по картам слоями (работает одна карта за раз)
  python train.py --model Qwen/Qwen3-32B --data data/q32 --out runs/q32 --tokens 200e6 --device-map auto \
      --max-gpu-memory 44GiB --no-momentum
"""
import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

from ted import Hyper, export_ternary, load_state, materialize, pop_flips, save_state, set_format, set_latent_bits, sync_state, ternarize  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True


def decoder(model):
    return model.get_decoder() if hasattr(model, "get_decoder") else model.model


def tied(model):
    cfg = model.config
    return bool(getattr(cfg, "tie_word_embeddings", False) or getattr(getattr(cfg, "text_config", None), "tie_word_embeddings", False))


def first_device(model):
    return model.get_input_embeddings().weight.device


def loss_fn(student, x, teacher=None, kd=0.0, temp=1.0, off=0.0, off_layers=0, chunk=256):
    """(1 - kd) * кросс-энтропия + kd * T^2 * KL(учитель || ученик) + off * OFF, кусками по chunk токенов.
    OFF (TernaryLLM, arXiv 2406.07177): сумма по первым off_layers слоям среднего (1 - косинус) между
    скрытыми состояниями ученика и учителя; косинус не чувствителен к выбросам в признаках."""
    use_off = teacher is not None and off > 0
    so = decoder(student)(input_ids=x[:, :-1], output_hidden_states=use_off)
    hs = so.last_hidden_state
    y = x[:, 1:].reshape(-1).to(hs.device)
    hs = hs.reshape(-1, hs.shape[-1])
    ht, feat = None, 0.0
    if teacher is not None and (kd > 0 or use_off):
        with torch.no_grad():
            to = decoder(teacher)(input_ids=x[:, :-1], output_hidden_states=use_off)
            ht = to.last_hidden_state.reshape(-1, hs.shape[-1]).to(hs.device) if kd > 0 else None
        if use_off:
            for l in range(1, off_layers + 1):
                a_, b_ = so.hidden_states[l], to.hidden_states[l].to(so.hidden_states[l].device)
                feat = feat + (1 - F.cosine_similarity(a_.float(), b_.float(), dim=-1)).mean().to(hs.device)
    head_s = student.get_output_embeddings()
    head_t = teacher.get_output_embeddings() if ht is not None else None

    def part(hc, yc, tc):
        ls = head_s(hc).float()
        out = F.cross_entropy(ls, yc, reduction="sum")
        if tc is None:
            return out
        with torch.no_grad():
            lt = head_t(tc.to(head_t.weight.device)).float().to(ls.device) / temp
            pt = lt.softmax(-1)
            lpt = lt.log_softmax(-1)
        kl = (pt * (lpt - (ls / temp).log_softmax(-1))).sum() * temp * temp
        return (1 - kd) * out + kd * kl

    total = 0.0
    for i in range(0, hs.shape[0], chunk):
        tc = None if ht is None else ht[i:i + chunk]
        total = total + checkpoint(part, hs[i:i + chunk], y[i:i + chunk], tc, use_reentrant=False)
    return total / y.numel() + off * feat


@torch.no_grad()
def evaluate(model, val, seq, batch):
    """Средняя кросс-энтропия (нат/токен) на проверочном наборе."""
    if val is None:
        return None
    model.eval()
    n_win = len(val) // (seq + 1)
    wins = val[: n_win * (seq + 1)].reshape(n_win, seq + 1)
    tot, n = 0.0, 0
    dev = first_device(model)
    for i in range(0, len(wins), batch):
        x = torch.from_numpy(wins[i:i + batch].astype(np.int64)).to(dev)
        tot += float(loss_fn(model, x)) * x[:, 1:].numel()
        n += x[:, 1:].numel()
    model.train()
    return tot / n


def load(model_id, device_map, max_mem):
    from transformers import AutoModelForCausalLM
    kw = {"dtype": torch.bfloat16}
    if device_map == "auto":
        n = torch.cuda.device_count()
        kw.update(device_map="auto", max_memory={i: max_mem for i in range(n)} if max_mem else None)
        m = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    else:
        m = AutoModelForCausalLM.from_pretrained(model_id, **kw).to(device_map)
    m.config.use_cache = False
    return m


def main():
    ap = argparse.ArgumentParser(description="Тернаризация модели методом TED")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="папка из prepare_data.py (train.npy, val_ru.npy, val_en.npy)")
    ap.add_argument("--out", required=True, help="папка прогона: лог, точки сохранения, итог")
    ap.add_argument("--tokens", type=float, default=20e6, help="сколько токенов обучения всего")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4, help="последовательностей за шаг (одна видеокарта 24 ГБ: 4)")
    ap.add_argument("--lr", default="auto",
                    help="шаг обучения или auto: 1e-3 * (медиана s / 0.23) — тот же относительный шаг по сетке, что "
                         "подобран на SmolLM2 (у неё медиана масштаба строк 0.23, у Qwen3.5-4B 0.0146)")
    ap.add_argument("--lr-s", default="auto", help="шаг для масштабов строк или auto: 5e-5 * (медиана s / 0.23)")
    ap.add_argument("--no-momentum", action="store_true", help="~1 байт на вес вместо ~2 (качество чуть ниже)")
    ap.add_argument("--latent-bits", type=int, default=16, choices=[8, 16],
                    help="разрядность положения веса: 16 (~3 байта на вес, лучше обычного QAT) или 8 (~2 байта, чуть хуже)")
    ap.add_argument("--group", type=int, default=256,
                    help="сколько весов строки делят один масштаб: 256 совпадает с блоком TQ2_0 (точный экспорт), 0 — вся строка")
    ap.add_argument("--hadamard", type=int, default=0,
                    help="поворот Адамара по входу блоками этого размера (например 256), 0 — без поворота. "
                         "Модель с поворотом не переводится в обычный llama.cpp (нужна среда с поворотом входов)")
    ap.add_argument("--kd", type=float, default=0.0, help="доля дистилляции от исходной модели (0 — выключена)")
    ap.add_argument("--kd-temp", type=float, default=1.0)
    ap.add_argument("--off", type=float, default=0.0,
                    help="вес дистилляции признаков OFF (косинус скрытых состояний учителя и ученика), например 1.0")
    ap.add_argument("--off-layers", type=int, default=0, help="сколько первых слоёв в OFF (0 — половина слоёв)")
    ap.add_argument("--device-map", default="cuda:0", help="cuda:0, auto (разложить по всем видеокартам) или cpu (только проверка)")
    ap.add_argument("--max-gpu-memory", default=None, help="для auto: предел на видеокарту, например 44GiB")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--val-tokens", type=float, default=0.2e6, help="токенов на каждый проверочный язык")
    ap.add_argument("--save-hf", action="store_true", help="в конце сохранить обычную HF-модель с весами s*Q")
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()

    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world > 1:  # запуск через torchrun: каждая карта держит всю модель и берёт свою часть данных
        assert a.device_map != "auto", "torchrun и --device-map auto вместе не используются"
        local = int(os.environ.get("LOCAL_RANK", 0))
        backend = "gloo" if a.device_map == "cpu" else "nccl"
        if a.device_map != "cpu":  # cpu — только для проверки скриптов
            n = torch.cuda.device_count()
            if n < world:  # процессов больше, чем карт (только для проверки): делят карты, NCCL так не умеет
                print(f"внимание: процессов {world}, видеокарт {n} — проверочный режим через gloo", flush=True)
                backend = "gloo"
            torch.cuda.set_device(local % n)
            a.device_map = f"cuda:{local % n}"
        dist.init_process_group(os.environ.get("TED_BACKEND", backend))
    main_proc = rank == 0
    say = (lambda *x: print(*x, flush=True)) if main_proc else (lambda *x: None)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(a.data)
    train = np.load(data / "train.npy", mmap_mode="r")
    vals = {}
    for lang in ("ru", "en", "chat_ru", "chat_en"):
        p = data / f"val_{lang}.npy"
        vals[lang] = np.asarray(np.load(p, mmap_mode="r")[: int(a.val_tokens)]) if p.exists() else None
    ev = lambda m: {k: evaluate(m, v, a.seq, a.batch) for k, v in vals.items()} if main_proc else {}

    teacher = None
    student = load(a.model, a.device_map, a.max_gpu_memory)
    fp = ev(student)
    if a.off > 0 and a.off_layers == 0:
        a.off_layers = student.config.get_text_config().num_hidden_layers // 2
    if (a.kd > 0 or a.off > 0) and a.device_map != "auto":
        # одна карта на копию модели: учитель — копия исходной модели; эмбеддинги общие (они заморожены)
        teacher = copy.deepcopy(student).eval()
        teacher.set_input_embeddings(student.get_input_embeddings())
        if tied(student):
            teacher.get_output_embeddings().weight = student.get_input_embeddings().weight
        for p in teacher.parameters():
            p.requires_grad_(False)
    set_latent_bits(a.latent_bits)
    set_format(a.group, a.hadamard)
    t0 = time.time()
    torch.manual_seed(a.seed)  # одинаковая случайность округления на всех картах
    layers = ternarize(student, momentum=not a.no_momentum)
    n_tern = sum(l.L.numel() for l in layers.values())
    s_med = torch.tensor([float(l.s.median()) for l in layers.values()]).median().item()
    a.lr = 1e-3 * s_med / 0.23 if a.lr == "auto" else float(a.lr)
    a.lr_s = 5e-5 * s_med / 0.23 if a.lr_s == "auto" else float(a.lr_s)
    say(f"медиана масштаба строк {s_med:.4f}: lr {a.lr:.2e}, lr_s {a.lr_s:.2e}")
    state_gb = sum(l.state_bytes() for l in layers.values()) / 1e9
    if (a.kd > 0 or a.off > 0) and a.device_map == "auto":
        # модель разложена по картам: учитель загружается отдельно в оставшуюся свободную память
        teacher = load(a.model, "auto", None).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.train()

    per_step = a.batch * a.seq * world
    steps = int(a.tokens // per_step)
    warm = min(200, max(1, steps // 20))
    factor = lambda t: (t + 1) / warm if t < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (t - warm) / max(1, steps - warm)))
    start = 0
    ck = out / "ckpt.pt"
    if ck.exists():
        meta = load_state(layers, ck)
        for k in ("latent_bits", "group", "hadamard"):
            assert meta.get(k, {"group": 0, "hadamard": 0}.get(k)) == getattr(a, k), f"точка сохранения сделана с другим --{k}"
        start = meta["step"]
        say(f"продолжение с шага {start}")
    # данные у каждой карты свои; округление одинаковое на всех картах (зерно не зависит от rank)
    rng = np.random.default_rng([a.seed, rank, start])
    torch.manual_seed(a.seed * 1_000_003 + start)
    log = open(out / "log.jsonl", "a") if main_proc else None
    if start == 0 and main_proc:
        st = ev(student)
        head = {"step": 0, "fp_loss": fp, "start_loss": st, "ternary_params": n_tern, "state_gb": state_gb,
                "bytes_per_weight": state_gb * 1e9 / n_tern, "ternarize_s": time.time() - t0, "gpus": world, "args": vars(a)}
        log.write(json.dumps(head, ensure_ascii=False) + "\n")
        log.flush()
        say(f"исходная модель {fp}, после тернаризации {st}; тернарных весов {n_tern / 1e6:.0f} млн, "
            f"состояние обучения {state_gb:.2f} ГБ ({state_gb * 1e9 / n_tern:.2f} байта на вес), видеокарт {world}, шагов {steps}")

    Hyper.lr_s = a.lr_s
    dev = first_device(student)
    n_gpu = torch.cuda.device_count() if dev.type == "cuda" else 0
    for i in range(n_gpu):
        torch.cuda.reset_peak_memory_stats(i)
    t0, run, done = time.time(), 0.0, 0
    pop_flips(layers)
    for step in range(start + 1, steps + 1):
        Hyper.t, Hyper.lr = step, a.lr * factor(step - 1)
        idx = rng.integers(0, len(train) - a.seq - 1, a.batch)
        x = torch.from_numpy(np.stack([np.asarray(train[i:i + a.seq + 1]) for i in idx]).astype(np.int64)).to(dev)
        loss = loss_fn(student, x, teacher, a.kd, a.kd_temp, a.off, a.off_layers)
        loss.backward()  # шаги оптимизатора выполняются внутри обратного прохода слоёв
        for l in layers.values():
            l.dummy.grad = None
        run += float(loss.detach())
        done += 1
        if step % a.eval_every == 0 or step == steps:
            dt = time.time() - t0
            if main_proc:
                vl = ev(student)
                rec = {"step": step, "tokens": step * per_step, "val_loss": vl, "train_loss": run / done,
                       "flips_pct_per_step": 100 * pop_flips(layers) / done / n_tern, "tok_per_s": done * per_step / dt,
                       "peak_gpu_gb": max([torch.cuda.max_memory_allocated(i) for i in range(n_gpu)] or [0]) / 1e9}
                log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                log.flush()
                vs = ", ".join(f"{k} {v:.4f}" for k, v in vl.items() if v is not None)
                say(f"  шаг {step}/{steps}: проверка [{vs}], обучение {run / done:.4f}, "
                    f"{rec['tok_per_s']:.0f} ток/с, пик видеопамяти {rec['peak_gpu_gb']:.1f} ГБ")
            if world > 1:
                dist.barrier()
            run, done, t0 = 0.0, 0, time.time()
        if step % a.save_every == 0 or step == steps:
            diff = sync_state(layers)
            if diff:
                say(f"  внимание: состояния карт разошлись в {diff} значениях, выровнено по карте 0")
            if main_proc:
                tmp = out / "ckpt.pt.tmp"
                save_state(layers, tmp, {"step": step, "latent_bits": a.latent_bits, "group": a.group, "hadamard": a.hadamard})
                os.replace(tmp, ck)
            if world > 1:
                dist.barrier()
    if main_proc:
        export_ternary(layers, out / "ternary_packed.pt")
        say(f"итог: {out / 'ternary_packed.pt'}")
        if a.save_hf:
            del teacher
            torch.cuda.empty_cache()
            materialize(student, layers)
            student.save_pretrained(out / "hf_ternary", safe_serialization=True)
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained(a.model).save_pretrained(out / "hf_ternary")
            say(f"HF-модель: {out / 'hf_ternary'}")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
