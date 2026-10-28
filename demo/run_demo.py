"""Демонстрация тернарных моделей на Mac (только процессор): одни и те же задачи для нескольких моделей.

Модели запускаются по очереди (одна в памяти), через llama-server из ~/llama.cpp-cpu/build-lut (наше ядро TQ2_0).
Итог пишется в web/results.json, его показывает web/index.html.

Пример:
  python3 run_demo.py                     # все модели и задачи
  python3 run_demo.py --models tern4b     # одна модель (результаты остальных сохраняются)
"""
import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOME = Path.home()
SERVER = HOME / "llama.cpp-cpu/build-lut/bin/llama-server"
PORT = 8091

MODELS = {
    "tern4b": {"name": "Наша 4B, тернарная", "note": "TED-16, учили на текстах без диалогов",
               "path": HOME / "tern-mac/models/tern-TQ2_0-head_q6_K.gguf", "ours": True},
    "tern4b_text": {"name": "Наша 4B, режим «продолжи текст»", "note": "та же модель, задание без формата чата",
                    "path": HOME / "tern-mac/models/tern-TQ2_0-head_q6_K.gguf", "ours": True, "mode": "text"},
    "tern08": {"name": "Наша 0.8B, тернарная", "note": "блоки 256 + поворот, с диалогами",
               "path": HOME / "tern-mac/models/e2-TQ2_0.gguf", "ours": True},
    "bonsai2b": {"name": "Bonsai 2B (Prism)", "note": "тернарная, родной формат Q2_0",
                 "path": Path("/Volumes/KINGSTON/miniBrain/ios/Models/Ternary-Bonsai-2-2B-Q2_0.gguf")},
    "bonsai2b_tq": {"name": "Bonsai 2B в нашем формате", "note": "те же веса, переупакованы в TQ2_0, наше ядро",
                    "path": HOME / "tern-mac/bonsai2b-TQ2_0.gguf", "ours": True},
    "qwen2b": {"name": "Qwen3.5-2B, обычная", "note": "3 бита, Q3_K_M",
               "path": Path("/Volumes/KINGSTON/miniBrain/ios/Models/Qwen3.5-2B-Q3_K_M.gguf")},
}

CHAT = "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def task_text(t):
    if t.get("text_file"):
        return (HERE / t["text_file"]).read_text(encoding="utf-8")
    return t.get("text", "")


def prompt(model, t):
    text = task_text(t)
    if model.get("mode") == "text":
        return (f"Текст:\n{text}\n\n" if text else "") + f"Задание: {t['instruction']}\n\nОтвет:\n"
    return CHAT.format(q=t["instruction"] + (f"\n\nТекст:\n{text}" if text else ""))


def post(path, body, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(proc, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise SystemExit("llama-server завершился при загрузке модели")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return
        except Exception:
            pass
        time.sleep(1)
    raise SystemExit("llama-server не поднялся")


def save(res, out_path):
    js = json.dumps(res, ensure_ascii=False, indent=1)
    out_path.write_text(js, encoding="utf-8")
    # тот же итог для страницы, открытой прямо с диска (file:// не даёт читать json через fetch)
    out_path.with_suffix(".js").write_text("window.RESULTS = " + js + ";\n", encoding="utf-8")


def rss_mb(pid):
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return int(out) // 1024 if out else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    tasks = json.loads((HERE / "tasks.json").read_text(encoding="utf-8"))
    if a.tasks:
        tasks = [t for t in tasks if t["id"] in a.tasks]
    out_path = HERE / "web/results.json"
    res = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {"results": {}}
    res["tasks"] = [{k: t[k] for k in ("id", "group", "title", "instruction")} | {"text": task_text(t)}
                    for t in json.loads((HERE / "tasks.json").read_text(encoding="utf-8"))]
    old_models = res.get("models", {})
    res["models"] = {k: {"name": m["name"], "note": m["note"], "ours": m.get("ours", False),
                         "size_mb": round(m["path"].stat().st_size / 2**20)}
                        | ({"ram_mb": old_models[k]["ram_mb"]} if old_models.get(k, {}).get("ram_mb") else {})
                     for k, m in MODELS.items()}
    res["machine"] = "Apple M4 Pro, только процессор (8 потоков), llama.cpp + наше ядро TQ2_0"

    for mid in a.models:
        m = MODELS[mid]
        print(f"== {m['name']}", flush=True)
        proc = subprocess.Popen([str(SERVER), "-m", str(m["path"]), "-c", "16384", "-t", str(a.threads), "-ngl", "0",
                                 "-np", "1", "--host", "127.0.0.1", "--port", str(PORT)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            wait_ready(proc)
            r = res["results"].setdefault(mid, {})
            peak = rss_mb(proc.pid)
            for t in tasks:
                t0 = time.time()
                j = post("/completion", {"prompt": prompt(m, t), "n_predict": t["max_tokens"], "temperature": 0,
                                         "repeat_penalty": 1.1, "cache_prompt": False,
                                         "stop": ["<|im_end|>", "<|endoftext|>", "\nЗадание:", "\nТекст:"]})
                tm = j.get("timings", {})
                r[t["id"]] = {"answer": j["content"].strip(), "seconds": round(time.time() - t0, 1),
                              "prompt_tokens": tm.get("prompt_n"), "prompt_tps": round(tm.get("prompt_per_second", 0)),
                              "gen_tokens": tm.get("predicted_n"), "gen_tps": round(tm.get("predicted_per_second", 0))}
                peak = max(peak or 0, rss_mb(proc.pid) or 0)
                print(f"   {t['id']}: {r[t['id']]['prompt_tokens']} ток. запроса за {r[t['id']]['seconds']} с, "
                      f"чтение {r[t['id']]['prompt_tps']} ток/с, генерация {r[t['id']]['gen_tps']} ток/с", flush=True)
                save(res, out_path)
            res["models"][mid]["ram_mb"] = peak
            save(res, out_path)
        finally:
            proc.terminate()
            proc.wait()
    print("ГОТОВО", flush=True)


if __name__ == "__main__":
    main()
