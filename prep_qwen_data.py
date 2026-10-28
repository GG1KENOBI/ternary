"""Тексты для тернаризации Qwen: русский FineWeb-2 и английский FineWeb-Edu в токенах Qwen3.5.

Первые 3000 документов каждого набора пропускаются (они уже были калибровкой слоя в фазе 1а).
Сохраняется один массив uint32: сначала обучение (перемешанные по документам ru/en), в конце — проверка.

Запуск: python prep_qwen_data.py --ru 30e6 --en 10e6
"""
import argparse
import itertools
import os
import time

import numpy as np

os.environ["HF_HUB_OFFLINE"] = "0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ru", type=float, default=30e6)
    ap.add_argument("--en", type=float, default=10e6)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--out", default=os.path.expanduser("~/.cache/ternary/qwen_ru_en.npy"))
    a = ap.parse_args()
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    eos = tok.eos_token_id
    docs = {"ru": [], "en": []}
    for lang, repo, cfg, budget in (("ru", "HuggingFaceFW/fineweb-2", "rus_Cyrl", a.ru), ("en", "HuggingFaceFW/fineweb-edu", "sample-10BT", a.en)):
        t, n = time.time(), 0
        ds = load_dataset(repo, cfg, split="train", streaming=True)
        for r in itertools.islice(ds, 3000, None):
            ids = tok(r["text"])["input_ids"] + [eos]
            docs[lang].append(np.array(ids, dtype=np.uint32))
            n += len(ids)
            if n >= budget:
                break
        print(f"{lang}: {len(docs[lang])} документов, {n / 1e6:.1f} млн токенов, {time.time() - t:.0f} с", flush=True)
    allx = docs["ru"] + docs["en"]
    rng = np.random.default_rng(13)
    rng.shuffle(allx)
    arr = np.concatenate(allx)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.save(a.out, arr)
    print(f"сохранено {a.out}: {len(arr) / 1e6:.1f} млн токенов, {arr.nbytes / 1e6:.0f} МБ", flush=True)


if __name__ == "__main__":
    main()
