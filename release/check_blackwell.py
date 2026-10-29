"""Проверка CUDA и bf16 forward/backward на видимых GPU, без скачивания моделей."""
import argparse
import json
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="blackwell-results/hardware.json")
    args = ap.parse_args()
    report = {"python": sys.version, "platform": platform.platform(),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpus": []}
    code = 0
    try:
        import torch
        report.update(torch=torch.__version__, cuda_build=torch.version.cuda,
                      compiled_architectures=torch.cuda.get_arch_list())
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA недоступна. Проверьте nvidia-smi внутри WSL2 и Windows-драйвер.")
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                prop = torch.cuda.get_device_properties(i)
                row = {"visible_index": i, "name": prop.name,
                       "capability": list(torch.cuda.get_device_capability(i)),
                       "total_gib": prop.total_memory / 2**30,
                       "free_gib": torch.cuda.mem_get_info()[0] / 2**30}
                report["gpus"].append(row)
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError(f"GPU {i}: bf16 не поддерживается.")
                x = torch.randn(256, 256, dtype=torch.bfloat16, device=f"cuda:{i}", requires_grad=True)
                (x @ x.T).float().square().mean().backward()
                torch.cuda.synchronize()
                if not torch.isfinite(x.grad).all().item():
                    raise RuntimeError(f"GPU {i}: неконечный градиент в bf16-тесте.")
                row["bf16_forward_backward"] = "ok"
                del x
                torch.cuda.empty_cache()
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                                     "--format=csv"], capture_output=True, text=True, timeout=20)
            report["nvidia_smi"] = result.stdout or result.stderr
        except (OSError, subprocess.TimeoutExpired) as exc:
            report["nvidia_smi"] = str(exc)
        report["ok"] = True
    except Exception as exc:
        code = 1
        report.update(ok=False, error=str(exc), traceback=traceback.format_exc())
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
