"""Последовательное сравнение Q8 MoE в Windows, без Python-пакетов и WSL."""
import argparse
import csv
import contextlib
import hashlib
import io
import json
import os
import platform
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
GIB = 2**30
KIT_VERSION = "2026-10-30.2"


def configure_output():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, target, size, digest, attempts=3):
    """Возобновление только при корректном HTTP Range; проверка до использования файла."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"  Проверяю сохранённый файл: {target.name}", flush=True)
        if target.stat().st_size == size and sha256(target) == digest:
            return target
        raise RuntimeError(f"Неверный существующий файл: {target}. Не перезаписываю его.")
    part = target.with_name(target.name + ".part")
    for attempt in range(attempts):
        try:
            offset = part.stat().st_size if part.exists() else 0
            if offset > size:
                raise RuntimeError(f"Частичный файл больше ожидаемого: {part}")
            if offset < size:
                headers = {"User-Agent": "ternary-windows-moe/1", "Accept-Encoding": "identity"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=60) as response:
                    resumed = response.status == 206
                    if resumed:
                        value = response.headers.get("Content-Range", "")
                        if not value.startswith(f"bytes {offset}-"):
                            raise RuntimeError("Сервер прислал неверный Content-Range")
                    else:
                        offset = 0
                    with part.open("ab" if resumed else "wb") as f:
                        done, last = offset, time.monotonic()
                        for chunk in iter(lambda: response.read(8 * 1024**2), b""):
                            f.write(chunk)
                            done += len(chunk)
                            if time.monotonic() - last > 5:
                                print(f"  {target.name}: {done / GIB:.2f}/{size / GIB:.2f} ГиБ", flush=True)
                                last = time.monotonic()
            print(f"  Проверяю SHA256: {target.name}", flush=True)
            if part.stat().st_size < size:
                raise OSError("Загрузка оборвалась раньше ожидаемого размера, продолжу с .part")
            if part.stat().st_size != size or sha256(part) != digest:
                # Удаляется только .part, созданный загрузчиком в служебной папке.
                part.unlink()
                raise RuntimeError("Размер или SHA256 скачанного файла не совпал")
            os.replace(part, target)
            return target
        except (OSError, urllib.error.URLError, RuntimeError) as exc:
            if attempt + 1 == attempts:
                raise RuntimeError(f"Не удалось скачать {target.name}: {exc}") from exc
            print(f"  Сбой скачивания, повтор {attempt + 2}/{attempts}: {exc}", flush=True)
            time.sleep(2**attempt)


def extract_runtime(archive, dest):
    """Только EXE/DLL, без путей из архива и без исполнения до проверки SHA256."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            normalized = info.filename.replace("\\", "/")
            if ".." in normalized.split("/") or normalized.startswith("/"):
                raise RuntimeError("Небезопасный путь в архиве программы")
            name = normalized.rsplit("/", 1)[-1]
            if name.lower() == "llama-server.exe" or name.lower().endswith(".dll"):
                with z.open(info) as src, (dest / name).open("wb") as out:
                    shutil.copyfileobj(src, out)


def setup_runtime(manifest):
    runtime = manifest["runtime"]
    dest = HERE / "_runtime" / runtime["release"]
    # Каждый архив проверяется и распаковывается заново: локальная подмена DLL не кешируется.
    for asset in runtime["assets"]:
        archive = download(asset["url"], HERE / "_downloads" / asset["name"],
                           asset["size"], asset["sha256"])
        extract_runtime(archive, dest)
    server = dest / "llama-server.exe"
    if not server.is_file():
        raise RuntimeError("В архиве не найден llama-server.exe")
    return server


def run_text(command, env=None, timeout=30):
    result = subprocess.run(command, env=env, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=timeout,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    text = result.stdout + result.stderr
    if result.returncode:
        raise RuntimeError(f"{Path(command[0]).name}: код {result.returncode}\n{text}")
    return text


def inspect_gpu(gpu, minimum_driver):
    command = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,driver_version,uuid",
               "--format=csv,noheader,nounits"]
    text = run_text(command)
    cards = []
    for row in csv.reader(io.StringIO(text)):
        if len(row) == 6:
            i, name, total, free, driver, uuid = (v.strip() for v in row)
            cards.append(dict(index=int(i), name=name, total_mib=float(total),
                              free_mib=float(free), driver=driver, uuid=uuid))
    chosen = next((c for c in cards if c["index"] == gpu), None)
    if chosen is None:
        raise RuntimeError(f"Нет карты {gpu}. Доступны: {cards}. Укажите --gpu НОМЕР.")
    if int(chosen["driver"].split(".")[0]) < minimum_driver:
        raise RuntimeError(f"Драйвер {chosen['driver']} старее ветки {minimum_driver}, нужной комплекту CUDA 13.x. "
                           "Сохраните диагностику; обновление драйвера автоматически не выполняется.")
    return {"cards": cards, "selected": chosen, "nvidia_smi": text}


def request_json(port, key, path, body=None, timeout=20):
    payload = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=payload,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def wait_ready(proc, port, key, timeout=600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server завершился с кодом {proc.returncode}; см. server.log")
        try:
            if request_json(port, key, "/health", timeout=2).get("status") == "ok":
                return
        except (OSError, ValueError):
            pass
        time.sleep(1)
    raise RuntimeError("Загрузка модели превысила 10 минут; см. server.log")


def stop_process(proc):
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)


@contextlib.contextmanager
def run_lock(path):
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Другой запуск уже использует комплект ({path}). "
                           "После принудительного закрытия окна удалите этот lock только убедившись, "
                           "что прошлый compare.py и его сервер остановлены.") from exc
    try:
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))
        yield
    finally:
        Path(path).unlink(missing_ok=True)


class MemorySampler:
    """Выборка общей занятой памяти выбранной GPU, включая возможные чужие процессы."""
    def __init__(self, gpu):
        self.gpu, self.peak, self.samples = gpu, None, 0
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.done.is_set():
            try:
                text = run_text(["nvidia-smi", "-i", str(self.gpu), "--query-gpu=memory.used",
                                 "--format=csv,noheader,nounits"], timeout=5)
                value = float(text.strip())
                self.peak = max(self.peak or 0, value)
                self.samples += 1
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired):
                pass
            self.done.wait(2)

    def stop(self):
        self.done.set()
        self.thread.join(timeout=7)


def load_tasks(ids=None):
    source = HERE.parent / "demo"
    tasks = json.loads((source / "tasks.json").read_text(encoding="utf-8"))
    if ids and set(ids) - {t["id"] for t in tasks}:
        raise ValueError("Неизвестный --tasks")
    result = []
    for item in tasks:
        if ids and item["id"] not in ids:
            continue
        task = dict(item)
        if task.get("text_file"):
            task["text"] = (source / task["text_file"]).read_text(encoding="utf-8")
        result.append(task)
    return result


def chat_request(task):
    content = task["instruction"]
    if task.get("text"):
        content += "\n\nТекст:\n" + task["text"]
    return {"model": "candidate", "messages": [{"role": "user", "content": content}],
            "temperature": 0, "top_p": 1, "seed": 13, "max_tokens": task["max_tokens"],
            "stream": False, "cache_prompt": False, "repeat_penalty": 1.1,
            "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}


def assert_offloaded(log):
    matches = re.findall(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers to GPU", log)
    if not matches or not all(int(a) == int(b) and int(a) > 0 for a, b in matches):
        raise RuntimeError("Не подтверждено полное размещение слоёв на GPU. CPU-замер не засчитывается; см. server.log")
    return {"offloaded_layers": int(matches[-1][0]), "total_layers": int(matches[-1][1])}


def server_command(server, weights, args, port, key):
    # В b11960 библиотечные INFO (включая offloaded N/N) скрыты при обычном уровне 3.
    return [str(server), "-m", str(weights), "--alias", "candidate", "--host", "127.0.0.1",
            "--port", str(port), "--api-key", key, "--device", "CUDA0", "--split-mode", "none",
            "-ngl", "999", "-c", str(args.context), "-np", "1", "-b", "512", "-ub", "128",
            "-t", str(args.threads), "--jinja", "--reasoning", "off", "--reasoning-budget", "0",
            "--no-context-shift", "--no-webui", "--log-verbosity", "4", "--log-colors", "off"]


def wait_offloaded(log_path, timeout=3):
    # common_log пишет из фонового потока; /health может ответить до завершения записи.
    deadline = time.monotonic() + timeout
    while True:
        log = log_path.read_text(encoding="utf-8", errors="replace")
        if re.search(r"offloaded\s+\d+\s*/\s*\d+\s+layers", log) or time.monotonic() >= deadline:
            return assert_offloaded(log)
        time.sleep(0.1)


def model_path(model):
    # Зарезервирована только эта папка. Ни кеш HF, ни файлы вне комплекта не удаляются.
    return HERE / "_models" / model["id"] / model["file"]


def compare_model(server, model, tasks, args, folder, state, report_path):
    weights = model_path(model)
    record = state["results"][model["id"]] = {"model": model, "tasks": {}, "status": "downloading"}
    save_json(report_path, state)
    free = shutil.disk_usage(HERE).free
    part = weights.with_name(weights.name + ".part")
    existing = weights.stat().st_size if weights.exists() else (part.stat().st_size if part.exists() else 0)
    if free < model["size"] - existing + 3 * GIB:
        raise RuntimeError(f"Мало места: нужно ещё {(model['size'] - existing) / GIB + 3:.1f} ГиБ")
    # Ошибка скачивания завершает весь прогон, .part остаётся для следующего запуска.
    url = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{model['file']}"
    download(url, weights, model["size"], model["sha256"])
    proc = None
    sampler = MemorySampler(args.gpu)
    folder.mkdir(parents=True, exist_ok=True)
    log_path = folder / "server.log"
    env = {k: v for k, v in os.environ.items() if not k.startswith("LLAMA_ARG_")}
    env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=getattr(args, "visible_device", str(args.gpu)))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    key = secrets.token_urlsafe(24)
    command = server_command(server, weights, args, port, key)
    record["command"] = ["<session-key>" if x == key else x for x in command]
    record["status"] = "loading"
    sampler.thread.start()
    try:
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            wait_ready(proc, port, key)
            record["gpu_offload"] = wait_offloaded(log_path)
            # GGUF содержит шаблон именно этой модели: Qwen-шаблон не навязывается Gemma/GigaChat.
            record["server_properties"] = request_json(port, key, "/props")
            record["status"] = "running"
            save_json(report_path, state)
            for task in tasks:
                print(f"  {model['id']}: {task['id']}", flush=True)
                started = time.monotonic()
                try:
                    raw = request_json(port, key, "/v1/chat/completions", chat_request(task), timeout=args.timeout)
                    choice = raw["choices"][0]
                    answer = choice["message"].get("content") or ""
                    result = {"status": "ok" if answer.strip() else "empty_answer", "answer": answer,
                              "seconds": round(time.monotonic() - started, 3), "raw": raw}
                    if choice.get("finish_reason") == "length":
                        result["warning"] = "Достигнут предел длины ответа"
                    record["tasks"][task["id"]] = result
                except Exception as exc:
                    record["tasks"][task["id"]] = {"status": "error", "error": str(exc)}
                    # После тайм-аута сервер ещё может считать: не создаём очередь новых запросов.
                    raise RuntimeError(f"Ошибка задания {task['id']}: {exc}") from exc
                save_json(report_path, state)
            record["status"] = "complete" if all(t["status"] == "ok" for t in record["tasks"].values()) else "needs_review"
    except Exception as exc:
        record.update(status="error", error=str(exc))
        print(f"  ОШИБКА {model['id']}: {exc}", flush=True)
        # Системный сбой не должен запускать ещё два больших скачивания.
        raise
    except KeyboardInterrupt:
        record.update(status="interrupted", error="Остановлено пользователем")
        raise
    finally:
        # Нельзя удалять модель и запускать следующую, пока наш процесс не остановился.
        stop_process(proc)
        sampler.stop()
        record.update(max_sampled_gpu_used_mib=sampler.peak, memory_samples=sampler.samples)
        if not args.keep_models and record["status"] in ("complete", "needs_review"):
            weights.unlink(missing_ok=True)
            record["weights_deleted"] = True
        else:
            record["weights_deleted"] = False
            record["weights_retained_reason"] = "--keep-models" if args.keep_models else "Ошибка или прерывание: оставлены для повторной попытки"
        save_json(report_path, state)


def write_report(folder, state):
    lines = ["# Сравнение MoE Q8 на Windows", "", "Это проверка готовых моделей, не тернаризация TED.",
             "GPU-скорость не является оценкой скорости на i3.", "",
             "Температура 0; без рассуждений; штатный шаблон чата каждой модели; кеш запросов выключен.", ""]
    for mid, model in state.get("results", {}).items():
        lines += [f"## {model['model']['name']}", "", f"Статус: {model['status']}", ""]
        if model.get("error"):
            lines += [model["error"], ""]
        for tid, result in model["tasks"].items():
            lines += [f"### {tid}", "", f"Статус: {result['status']}; время: {result.get('seconds', '—')} с", ""]
            timings = result.get("raw", {}).get("timings", {})
            lines += [f"Чтение: {timings.get('prompt_per_second', 'нет данных')} ток/с; "
                      f"генерация: {timings.get('predicted_per_second', 'нет данных')} ток/с.", "",
                      result.get("answer") or result.get("error", ""), ""]
            if result.get("warning"):
                lines += ["Примечание: " + result["warning"], ""]
    (folder / "answers.md").write_text("\n".join(lines), encoding="utf-8")
    # В архив попадают только результаты и логи, папка весов находится вне folder.
    shutil.make_archive(str(folder), "zip", root_dir=folder)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", type=int, default=1, help="индекс из nvidia-smi (по умолчанию 1)")
    ap.add_argument("--models", nargs="+", default=["giga", "gemma", "qwen"], choices=["giga", "gemma", "qwen"])
    ap.add_argument("--tasks", nargs="+")
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=900, help="предел секунд на одно задание")
    ap.add_argument("--check", action="store_true", help="только оборудование и программа, без весов")
    ap.add_argument("--plan", action="store_true", help="показать план без скачиваний и запуска")
    ap.add_argument("--server", type=Path, help="готовый llama-server.exe вместо автоматического скачивания")
    ap.add_argument("--keep-models", action="store_true", help="не удалять веса после проверки")
    args = ap.parse_args(argv)
    manifest = json.loads((HERE / "models.json").read_text(encoding="utf-8"))
    selected = [next(m for m in manifest["models"] if m["id"] == mid) for mid in dict.fromkeys(args.models)]
    tasks = load_tasks(args.tasks)
    if args.plan:
        print(json.dumps({"kit_version": KIT_VERSION, "gpu": args.gpu, "models": selected, "tasks": [t["id"] for t in tasks],
                          "runtime": manifest["runtime"], "context": args.context}, ensure_ascii=False, indent=2))
        return 0
    if os.name != "nt":
        ap.error("Запускайте комплект в Windows. Для проверки плана доступен --plan.")
    if args.context < 512 or args.timeout < 1 or args.threads < 1:
        ap.error("Некорректный контекст, тайм-аут или число потоков")
    with run_lock(HERE / "_run.lock"):
        return execute(args, manifest, selected, tasks)


def execute(args, manifest, selected, tasks):
    folder = HERE / "results" / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3))
    folder.mkdir(parents=True)
    state = {"kit_version": KIT_VERSION, "started": datetime.now().astimezone().isoformat(), "platform": platform.platform(),
             "python": sys.version, "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
             "manifest": manifest, "tasks": tasks, "results": {}}
    report_path = folder / "results.json"
    exit_code = 0
    try:
        state["hardware"] = inspect_gpu(args.gpu, 0 if args.server else manifest["runtime"]["minimum_driver_major"])
        # Индексы CUDA и nvidia-smi могут различаться: выбираем физическую карту по UUID.
        args.visible_device = state["hardware"]["selected"]["uuid"]
        save_json(report_path, state)
        server = args.server.resolve() if args.server else setup_runtime(manifest)
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=args.visible_device)
        devices = run_text([str(server), "--list-devices"], env=env)
        (folder / "devices.txt").write_text(devices, encoding="utf-8")
        if "CUDA0" not in devices:
            raise RuntimeError("llama.cpp не видит CUDA0. CPU-режим не используется. См. devices.txt")
        state["devices"] = devices
        state["server_version"] = run_text([str(server), "--version"], env=env)
        if not args.check:
            for model in selected:
                free = inspect_gpu(args.gpu, 0)["selected"]["free_mib"] * 1024**2
                if free < model["size"] + 6 * GIB:
                    raise RuntimeError(f"Недостаточно свободной VRAM для {model['name']}; закройте другие GPU-задачи")
                print(f"\n=== {model['name']} Q8, {model['size'] / GIB:.1f} ГиБ ===", flush=True)
                compare_model(server, model, tasks, args, folder / model["id"], state, report_path)
            if any(r["status"] != "complete" for r in state["results"].values()):
                exit_code = 1
    except KeyboardInterrupt:
        state["error"] = "Остановлено пользователем; незавершённая загрузка может быть продолжена повторным запуском"
        exit_code = 130
    except Exception as exc:
        state["error"] = str(exc)
        print(f"ОШИБКА: {exc}", flush=True)
        exit_code = 1
    finally:
        state["exit_code"] = exit_code
        save_json(report_path, state)
        write_report(folder, state)
        print(f"\nРезультаты: {folder}\nПришлите архив: {folder}.zip", flush=True)
    return exit_code


if __name__ == "__main__":
    configure_output()
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"ОШИБКА: {error}", file=sys.stderr)
        raise SystemExit(1)
