"""Проверка тернарной модели: потери на отдельных русских и английских текстах + пример генерации.

Сравнивает исходную модель и тернарную (из ternary_packed.pt) на одних и тех же текстах.

Пример:
  python eval.py --model Qwen/Qwen3.5-4B --packed runs/q4b/ternary_packed.pt --data data/qwen35
"""
import argparse
import json
import math

import numpy as np
import torch

from ted import load_packed
from train import evaluate, load


CHAT = "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


@torch.no_grad()
def sample(model, tok, prompt, n=120, chat=False):
    dev = model.get_input_embeddings().weight.device
    x = tok(CHAT.format(q=prompt) if chat else prompt, return_tensors="pt").to(dev)
    stop = [tok.eos_token_id] + ([tok.convert_tokens_to_ids("<|im_end|>")] if chat else [])
    model.config.use_cache = True
    y = model.generate(**x, max_new_tokens=n, do_sample=False, repetition_penalty=1.1, eos_token_id=stop)
    model.config.use_cache = False
    return tok.decode(y[0, x["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--packed", required=True, help="ternary_packed.pt из train.py")
    ap.add_argument("--data", required=True)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--val-tokens", type=float, default=0.5e6)
    ap.add_argument("--device-map", default="cuda:0")
    ap.add_argument("--prompts", nargs="*", default=["Столица России — город", "Чтобы испечь хлеб, нужно",
                                                     "The main idea of the theory of relativity is"])
    ap.add_argument("--chat", nargs="*", default=["Столица России — какой город? Ответь одним предложением.",
                                                  "Кратко объясни, что такое инфляция.",
                                                  "Напиши короткое деловое письмо клиенту о переносе встречи на пятницу."],
                    help="вопросы в формате чата (проверка, что модель осталась ассистентом)")
    ap.add_argument("--out", default=None, help="куда записать итог в JSON")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    vals = {}
    for lang in ("ru", "en", "chat_ru", "chat_en"):
        try:
            vals[lang] = np.asarray(np.load(f"{a.data}/val_{lang}.npy", mmap_mode="r")[: int(a.val_tokens)])
        except FileNotFoundError:
            pass
    model = load(a.model, a.device_map, None)
    res = {}
    for name in ("исходная", "тернарная"):
        if name == "тернарная":
            load_packed(model, a.packed)
        model.eval()
        loss = {k: evaluate(model, v, a.seq, a.batch) for k, v in vals.items()}
        res[name] = {"loss": loss, "ppl": {k: math.exp(v) for k, v in loss.items()},
                     "samples": {p: sample(model, tok, p) for p in a.prompts},
                     "chat": {q: sample(model, tok, q, chat=True) for q in a.chat}}
        print(f"\n== {name}: " + ", ".join(f"{k}: потери {v:.4f}, перплексия {math.exp(v):.2f}" for k, v in loss.items()))
        for p, s in list(res[name]["samples"].items()) + list(res[name]["chat"].items()):
            print(f"  [{p}] {s.strip()[:400]}")
    if a.out:
        json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
