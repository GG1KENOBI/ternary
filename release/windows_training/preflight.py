"""Диагностика готовности Windows-сервера к HF-обучению GigaChat MoE.

Только стандартная библиотека. Ничего не устанавливает, не ходит в сеть, не читает
веса и не запускает обучение. torch/transformers проверяются, если уже установлены.
Запуск: py -3 preflight.py --gpu 1 --out training-preflight.json
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import importlib
import importlib.metadata as metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

PACKAGES = ("torch", "transformers", "accelerate", "safetensors")
NOT_TESTED = [
    "fp8_checkpoint_load",
    "gigachat_training_or_convergence",
    "quality_after_ternarization",
    "gguf_export",
]
GPU_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,name,uuid,memory.total,memory.free,driver_version",
    "--format=csv,noheader,nounits",
]
PROCESS_QUERY = [
    "nvidia-smi",
    "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
    "--format=csv,noheader,nounits",
]


class PreflightError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class Gpu:
    index: int
    name: str
    uuid: str
    memory_total_mib: Optional[int]
    memory_free_mib: Optional[int]
    driver_version: str


def _to_mib(value: str) -> Optional[int]:
    text = (value or "").strip()
    if not text or text.lower() in {"n/a", "[n/a]", "not supported"}:
        return None
    for suffix in ("MiB", "MB", "Mib"):
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
            break
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_nvidia_smi_csv(text: str) -> list[Gpu]:
    """Разобрать `nvidia-smi --query-gpu ... --format=csv,noheader,nounits`."""
    gpus: list[Gpu] = []
    for row in csv.reader(line for line in text.splitlines() if line.strip()):
        if not row or row[0].strip().lower() in {"index", "gpu"}:
            continue
        if len(row) < 6:
            raise PreflightError(
                "nvidia_smi_parse",
                f"ожидалось 6 колонок nvidia-smi, получено {len(row)}",
            )
        try:
            index = int(row[0].strip())
        except ValueError as exc:
            raise PreflightError("nvidia_smi_parse", f"некорректный index GPU: {row[0]!r}") from exc
        gpus.append(
            Gpu(
                index=index,
                name=row[1].strip(),
                uuid=row[2].strip(),
                memory_total_mib=_to_mib(row[3]),
                memory_free_mib=_to_mib(row[4]),
                driver_version=row[5].strip(),
            )
        )
    if not gpus:
        raise PreflightError("nvidia_smi_empty", "nvidia-smi не вернул ни одной GPU")
    return gpus


def parse_compute_apps_csv(text: str) -> set[str]:
    """UUID GPU, на которых есть compute-процессы."""
    busy: set[str] = set()
    for row in csv.reader(line for line in text.splitlines() if line.strip()):
        if not row:
            continue
        value = row[0].strip()
        if not value or value.lower() in {"gpu_uuid", "uuid"}:
            continue
        busy.add(value)
    return busy


def select_gpu(gpus: Iterable[Gpu], index: int = 1, busy_uuids: Optional[set[str]] = None) -> Gpu:
    """Выбрать физическую GPU строго по index. Никакого молчаливого fallback."""
    gpus = list(gpus)
    busy_uuids = busy_uuids or set()
    matches = [g for g in gpus if g.index == index]
    if not matches:
        available = ", ".join(str(g.index) for g in gpus) or "нет"
        raise PreflightError(
            "gpu_missing",
            f"GPU {index} не найдена в nvidia-smi (доступны: {available}); "
            "другая карта не выбирается",
        )
    if len(matches) > 1:
        raise PreflightError("gpu_ambiguous", f"nvidia-smi вернул несколько GPU с index={index}")
    gpu = matches[0]
    if gpu.uuid in busy_uuids:
        raise PreflightError(
            "gpu_busy",
            f"GPU {index} ({gpu.uuid}) занята compute-процессами; не трогаем её",
        )
    return gpu


def run_command(cmd: list[str], timeout: int = 15) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PreflightError("command_failed", f"{cmd[0]}: {type(exc).__name__}: {exc}") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().replace("\n", " ")[:300]
        raise PreflightError("command_failed", f"{cmd[0]} завершился с кодом {proc.returncode}: {detail}")
    return proc.stdout


def query_gpus(runner: Callable[[list[str]], str] = run_command) -> list[Gpu]:
    return parse_nvidia_smi_csv(runner(GPU_QUERY))


def query_busy_uuids(runner: Callable[[list[str]], str] = run_command) -> set[str]:
    return parse_compute_apps_csv(runner(PROCESS_QUERY))


def collect_packages(version_fn: Callable[[str], str] = metadata.version) -> dict[str, Optional[str]]:
    versions: dict[str, Optional[str]] = {}
    for name in PACKAGES:
        try:
            versions[name] = version_fn(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
        except Exception:
            versions[name] = None
    return versions


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def system_memory() -> dict:
    if os.name == "nt":
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        try:
            ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        except Exception as exc:
            return {"total_bytes": None, "free_bytes": None, "source": "GlobalMemoryStatusEx", "error": str(exc)}
        if not ok:
            return {"total_bytes": None, "free_bytes": None, "source": "GlobalMemoryStatusEx", "error": "вызов не удался"}
        return {
            "total_bytes": int(status.ullTotalPhys),
            "free_bytes": int(status.ullAvailPhys),
            "source": "GlobalMemoryStatusEx",
        }
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        total = os.sysconf("SC_PHYS_PAGES") * page
        free = os.sysconf("SC_AVPHYS_PAGES") * page
        return {"total_bytes": int(total), "free_bytes": int(free), "source": "sysconf"}
    except (AttributeError, OSError, ValueError):
        return {"total_bytes": None, "free_bytes": None, "source": "unavailable"}


def disk_info(path: Path) -> dict:
    probe = path.resolve().parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    return {"path": str(path.resolve()), "probe_path": str(probe),
            "total_bytes": int(usage.total), "free_bytes": int(usage.free)}


def import_torch():
    return importlib.import_module("torch")


def bf16_smoke(torch) -> dict:
    """Крошечный BF16 forward/backward. Не доказательство обучения GigaChat."""
    if not getattr(torch, "cuda", None) or not torch.cuda.is_available():
        raise PreflightError("cuda_unavailable", "torch установлен, но CUDA недоступна")
    if int(torch.cuda.device_count()) < 1:
        raise PreflightError("cuda_unavailable", "torch не видит ни одной CUDA-карты")
    device = torch.device("cuda:0")  # выбранная физическая GPU видна как локальная 0
    a = b = y = loss = None
    try:
        a = torch.randn((128, 128), device=device, dtype=torch.bfloat16, requires_grad=True)
        b = torch.randn((128, 128), device=device, dtype=torch.bfloat16, requires_grad=True)
        y = a @ b
        loss = y.float().square().mean()
        loss.backward()
        torch.cuda.synchronize()
        loss_finite = bool(torch.isfinite(loss).item())
        grad_finite = bool(torch.isfinite(a.grad).all().item())
        if not (loss_finite and grad_finite):
            raise PreflightError("bf16_test_failed", "BF16 forward/backward дал NaN/Inf")
        return {
            "attempted": True,
            "ok": True,
            "shape": [128, 128],
            "dtype": "bfloat16",
            "loss_finite": loss_finite,
            "grad_finite": grad_finite,
            "loss_abs": float(loss.detach().float().abs().item()),
        }
    finally:
        del a, b, y, loss
        if getattr(torch, "cuda", None) and hasattr(torch.cuda, "empty_cache"):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def _write_json(report: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(tmp, out)


def _error(code: str, message: str) -> dict:
    return {"code": code, "message": message}


def configure_stdio(stdout=None, stderr=None) -> None:
    """UTF-8 с заменой на выходе CLI: русский текст не должен падать на CP1252."""
    for stream in (sys.stdout if stdout is None else stdout,
                   sys.stderr if stderr is None else stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass


def run_preflight(
    gpu_index: int = 1,
    out_path: str | Path = "training-preflight.json",
    *,
    gpu_runner: Callable[[list[str]], str] = run_command,
    busy_runner: Callable[[list[str]], str] = run_command,
    torch_importer: Callable[[], object] = import_torch,
    version_fn: Callable[[str], str] = metadata.version,
    environ: Optional[dict] = None,
    timestamp: Optional[str] = None,
) -> dict:
    """Собрать отчёт, всегда попытаться сохранить его и вернуть dict."""
    out = Path(out_path).expanduser().resolve()
    report: dict = {
        "schema_version": 1,
        "generated_at": timestamp or datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "error",
        "ok": False,
        "error": None,
        "report_saved": False,
        "requested_gpu_index": gpu_index,
        "python": {
            "version": platform.python_version(),
            "bits": 64 if sys.maxsize > 2**32 else 32,
            "implementation": platform.python_implementation(),
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "memory": None,
        "disk": None,
        "packages": collect_packages(version_fn),
        "nvidia_smi": {"available": False, "gpus": [], "error": None},
        "gpu": None,
        "torch": {
            "imported": False,
            "cuda_build": None,
            "cuda_available": None,
            "device_count": None,
            "bf16_test": {"attempted": False, "ok": False},
        },
        "not_tested": list(NOT_TESTED),
        "notes": [
            "CUDA_VISIBLE_DEVICES выставляется по UUID выбранной физ. GPU; внутри процесса это cuda:0.",
            "BF16-тест 128x128 проверяет только окружение, а не обучение или качество GigaChat.",
        ],
    }
    try:
        report["disk"] = disk_info(out)
    except Exception as exc:
        report["disk"] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        report["memory"] = system_memory()
    except Exception as exc:
        report["memory"] = {"total_bytes": None, "free_bytes": None,
                            "source": "unavailable", "error": f"{type(exc).__name__}: {exc}"}

    status = "error"
    error: Optional[dict] = None
    torch = None
    try:
        gpus = query_gpus(gpu_runner)
        report["nvidia_smi"] = {"available": True, "gpus": [asdict(g) for g in gpus], "error": None}
        try:
            busy = query_busy_uuids(busy_runner)
            report["nvidia_smi"]["busy_gpu_uuids"] = sorted(busy)
        except Exception as exc:
            busy = set()
            report["nvidia_smi"]["busy_query_error"] = f"{type(exc).__name__}: {exc}"
            report["notes"].append("Не удалось получить compute-процессы; занятость GPU не подтверждена.")

        gpu = select_gpu(gpus, gpu_index, busy)
        report["gpu"] = {
            "requested_index": gpu_index,
            "selected_index": gpu.index,
            "name": gpu.name,
            "uuid": gpu.uuid,
            "cuda_visible_devices": gpu.uuid,
            "logical_device": "cuda:0",
        }
        env = os.environ if environ is None else environ
        env["CUDA_VISIBLE_DEVICES"] = gpu.uuid  # строго до импорта torch

        try:
            torch = torch_importer()
        except ImportError as exc:
            raise PreflightError("missing_dependency", f"torch не установлен или не импортируется: {exc}") from exc
        except Exception as exc:
            raise PreflightError("torch_import_failed", f"ошибка импорта torch: {type(exc).__name__}: {exc}") from exc
        if torch is None:
            raise PreflightError("missing_dependency", "импорт torch вернул None")

        version = getattr(torch, "version", None)
        cuda_build = getattr(version, "cuda", None)
        report["torch"]["imported"] = True
        report["torch"]["cuda_build"] = cuda_build
        try:
            available = bool(torch.cuda.is_available())
        except Exception as exc:
            raise PreflightError("cuda_unavailable", f"torch.cuda.is_available() упал: {exc}") from exc
        report["torch"]["cuda_available"] = available
        report["torch"]["device_count"] = int(torch.cuda.device_count()) if available else 0
        if not available:
            raise PreflightError("cuda_unavailable", "torch установлен, но CUDA недоступна")
        if report["torch"]["device_count"] < 1:
            raise PreflightError("cuda_unavailable", "torch не видит ни одной CUDA-карты")

        try:
            smoke = bf16_smoke(torch)
        except PreflightError as exc:
            report["torch"]["bf16_test"] = {"attempted": True, "ok": False, "error": exc.message}
            raise
        except Exception as exc:
            report["torch"]["bf16_test"] = {"attempted": True, "ok": False,
                                             "error": f"{type(exc).__name__}: {exc}"}
            raise PreflightError("bf16_test_failed", f"BF16-тест упал: {type(exc).__name__}: {exc}") from exc
        report["torch"]["bf16_test"] = smoke
        status = "ok"
    except PreflightError as exc:
        status = exc.code
        error = _error(exc.code, exc.message)
    except Exception as exc:
        status = "error"
        error = _error("unexpected_error", f"{type(exc).__name__}: {exc}")

    report["status"] = status
    report["ok"] = status == "ok" and bool(report["torch"]["bf16_test"].get("ok"))
    report["error"] = error
    try:
        # В файл пишем копию с report_saved=True: успешная запись должна быть
        # правдивой уже в самом документе, а не только в памяти после неё.
        _write_json({**report, "report_saved": True}, out)
        report["report_saved"] = True
    except Exception as exc:
        report["report_saved"] = False
        report["report_write_error"] = f"{type(exc).__name__}: {exc}"
        report["diagnostic_status"] = report["status"]
        if report["status"] == "ok":
            report["status"] = "report_write_failed"
        report["ok"] = False
    return report


def main(argv: Optional[list[str]] = None, *, runner: Callable = run_preflight) -> int:
    configure_stdio()
    parser = argparse.ArgumentParser(description="Преflight окружения для HF-обучения GigaChat")
    parser.add_argument("--gpu", type=int, default=1, help="физический index GPU из nvidia-smi (по умолчанию 1)")
    parser.add_argument("--out", default="training-preflight.json", help="путь к JSON-отчёту")
    args = parser.parse_args(argv)
    out = Path(args.out)
    try:
        report = runner(args.gpu, out)
    except Exception as exc:  # последняя страховка: отчёт всё равно пробуем сохранить
        report = {
            "schema_version": 1,
            "status": "error",
            "ok": False,
            "error": _error("fatal", f"{type(exc).__name__}: {exc}"),
            "requested_gpu_index": args.gpu,
            "not_tested": list(NOT_TESTED),
            "report_saved": False,
        }
        try:
            _write_json({**report, "report_saved": True}, out)
            report["report_saved"] = True
        except Exception as write_exc:
            report["report_write_error"] = f"{type(write_exc).__name__}: {write_exc}"
    status = report.get("status", "error")
    saved = bool(report.get("report_saved"))
    line = f"preflight: {status} gpu={args.gpu}"
    if saved:
        print(f"{line} report={out}")
    else:
        detail = report.get("report_write_error")
        suffix = f": {detail}" if detail else ""
        print(f"{line} report={out} НЕ СОХРАНЁН{suffix}", file=sys.stderr)
    err = report.get("error")
    if err:
        print(f"preflight: {err.get('code')}: {err.get('message')}", file=sys.stderr)
    return 0 if (report.get("ok") and saved and not report.get("report_write_error")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
