"""Подготовка данных в токенах выбранной модели: тексты (русский FineWeb-2 + английский FineWeb-Edu)
и диалоги (русский GrandMaster-PRO-MAX + английский smoltalk) в формате чата Qwen.

Создаёт в папке --out:
  train.npy   — обучающие токены (все источники перемешаны по документам), uint32;
  val_ru.npy, val_en.npy, val_chat_ru.npy, val_chat_en.npy — отдельные проверочные наборы (их нет в обучении).
Диалоги нужны, чтобы модель после тернаризации осталась ассистентом: дистилляция только на веб-текстах
превращает её в «продолжатель текста».
Лицензии: FineWeb-2 и FineWeb-Edu — ODC-By 1.0; GrandMaster-PRO-MAX и smoltalk (smol-magpie-ultra) — Apache 2.0.

Пример:
  python prepare_data.py --model Qwen/Qwen3.5-9B --out data/q9 --ru 100e6 --en 50e6 --chat-ru 100e6 --chat-en 50e6
"""
import argparse
import itertools
import os
import time

import numpy as np


def chat_text(messages):
    """Диалог в формате чата Qwen без рассуждений (пустой блок <think>), как отвечает instruct-режим."""
    out = []
    for m in messages:
        role, content = m.get("role"), (m.get("content") or "").strip()
        if role not in ("system", "user", "assistant") or not content:
            continue
        if role == "assistant":
            content = "<think>\n\n</think>\n\n" + content
        out.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(out)


SOURCES = {  # имя: (репозиторий, конфигурация, функция «запись -> текст»)
    "ru": ("HuggingFaceFW/fineweb-2", "rus_Cyrl", lambda r: r["text"]),
    "en": ("HuggingFaceFW/fineweb-edu", "sample-10BT", lambda r: r["text"]),
    "chat_ru": ("Vikhrmodels/GrandMaster-PRO-MAX", None, lambda r: chat_text(r["conversation"])),
    "chat_en": ("HuggingFaceTB/smoltalk", "smol-magpie-ultra", lambda r: chat_text(r["messages"])),
}


def stream_docs(name, tok, budget, skip=0):
    from datasets import load_dataset
    repo, cfg, to_text = SOURCES[name]
    ds = load_dataset(repo, cfg, split="train", streaming=True)
    if name.startswith("chat"):
        ds = ds.shuffle(seed=13, buffer_size=10000)  # диалоги в наборах сгруппированы по источникам
    docs, n = [], 0
    eos = [] if name.startswith("chat") else [tok.eos_token_id]  # у диалога конец — <|im_end|>
    for r in itertools.islice(ds, skip, None):
        text = to_text(r)
        if not text:
            continue
        ids = tok(text)["input_ids"] + eos
        docs.append(np.array(ids, dtype=np.uint32))
        n += len(ids)
        if n >= budget:
            break
    return docs, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="модель, чьим токенизатором режем текст")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ru", type=float, default=30e6, help="токенов русского для обучения")
    ap.add_argument("--en", type=float, default=10e6, help="токенов английского для обучения")
    ap.add_argument("--chat-ru", type=float, default=0, help="токенов русских диалогов для обучения")
    ap.add_argument("--chat-en", type=float, default=0, help="токенов английских диалогов для обучения")
    ap.add_argument("--val", type=float, default=0.5e6, help="токенов на каждый проверочный набор")
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    os.makedirs(a.out, exist_ok=True)
    train = []
    for name, budget in (("ru", a.ru), ("en", a.en), ("chat_ru", a.chat_ru), ("chat_en", a.chat_en)):
        if budget <= 0:
            continue
        t = time.time()
        val, nv = stream_docs(name, tok, a.val)  # первые документы — на проверку
        np.save(os.path.join(a.out, f"val_{name}.npy"), np.concatenate(val))
        docs, n = stream_docs(name, tok, budget, skip=len(val))
        train += docs
        print(f"{name}: проверка {nv / 1e6:.2f} млн токенов, обучение {n / 1e6:.1f} млн токенов ({len(docs)} документов), "
              f"{time.time() - t:.0f} с", flush=True)
    np.random.default_rng(a.seed).shuffle(train)
    arr = np.concatenate(train)
    np.save(os.path.join(a.out, "train.npy"), arr)
    print(f"готово: {a.out}/train.npy — {len(arr) / 1e6:.1f} млн токенов, {arr.nbytes / 1e9:.2f} ГБ")


if __name__ == "__main__":
    main()
    os._exit(0)  # потоковый режим datasets иногда падает при штатном выходе интерпретатора
