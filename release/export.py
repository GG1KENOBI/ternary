"""Сборка обычной HF-модели с тернарными весами s*Q (для llama.cpp и любых HF-инструментов).

Пример:
  python export.py --model Qwen/Qwen3.5-4B --packed runs/q4b/ternary_packed.pt --out runs/q4b/hf_ternary
Затем: bash export_gguf.sh runs/q4b/hf_ternary runs/q4b/model

Модель с поворотом Адамара (--hadamard при обучении): добавьте --prism. Веса сохраняются тернарными в повёрнутом
базисе вместе с hadamard_packing.json (формат описания от Prism). Такую модель переводит в GGUF и запускает
свежий llama.cpp (поддержка prism.hadamard есть в основной ветке с октября 2026, проверено на Mac 28.10.2026
на Bonsai 2B с поворотом, коммит a7af149; нашу модель там ещё не запускали) или форк Prism (проверено: совпадает с PyTorch). Формат TQ2_0 на CPU быстрее их PTQ1_0 (замер: 65 против 39 ток/с генерации):
  python export.py --model ... --packed ... --out runs/x/hf_rot --prism
  bash export_gguf.sh runs/x/hf_rot runs/x/model          # свежий ~/llama.cpp, TQ2_0
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import json
from pathlib import Path

from ted import load_packed, packed_is_rotated, prism_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--packed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prism", action="store_true",
                    help="для модели с поворотом: тернарные веса в повёрнутом базисе + hadamard_packing.json (свежий llama.cpp или форк Prism)")
    a = ap.parse_args()
    rotated = packed_is_rotated(a.packed)
    if a.prism and not rotated:
        raise SystemExit("--prism нужен только для модели с поворотом (--hadamard при обучении)")
    if rotated and not a.prism:
        print("ВНИМАНИЕ: модель обучена с поворотом Адамара. Сохраняю эквивалентную обычную модель (для проверки в PyTorch),\n"
              "но её веса уже не тернарные, и для llama.cpp TQ2_0 она не годится. Для llama.cpp используйте --prism.")
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)  # на CPU, видеокарта не нужна
    load_packed(model, a.packed, keep_rotated=a.prism)
    model.save_pretrained(a.out, safe_serialization=True)
    if a.prism:
        (Path(a.out) / "hadamard_packing.json").write_text(json.dumps(prism_manifest(a.packed), ensure_ascii=False))
        print("записано описание поворота: hadamard_packing.json (переводить в GGUF свежим llama.cpp: bash export_gguf.sh ...)")
    AutoTokenizer.from_pretrained(a.model).save_pretrained(a.out)
    print(f"готово: {a.out}")


if __name__ == "__main__":
    main()
