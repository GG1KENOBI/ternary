"""CPU-тесты preflight.py: только stdlib, без nvidia-smi, torch и сети."""
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

PREFLIGHT = Path(__file__).resolve().parents[1] / "windows_training" / "preflight.py"
spec = importlib.util.spec_from_file_location("training_preflight", PREFLIGHT)
pf = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pf  # dataclass ищет модуль в sys.modules
spec.loader.exec_module(pf)

SMI = """0, NVIDIA RTX PRO 6000 Blackwell, GPU-aaaa, 97887 MiB, 1024 MiB, 580.65
1, NVIDIA RTX PRO 6000 Blackwell, GPU-bbbb, 97887 MiB, 97000 MiB, 580.65
2, NVIDIA RTX PRO 6000 Blackwell, GPU-cccc, 97887 MiB, 96000 MiB, 580.65
"""

SMOKE = {
    "attempted": True,
    "ok": True,
    "shape": [128, 128],
    "dtype": "bfloat16",
    "loss_finite": True,
    "grad_finite": True,
    "loss_abs": 1.0,
}


def fake_torch():
    return SimpleNamespace(
        version=SimpleNamespace(cuda="12.8"),
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1),
    )


class StdioTest(unittest.TestCase):
    def test_configure_stdio_is_utf8_replace_on_cp1252(self):
        out_raw, err_raw = io.BytesIO(), io.BytesIO()
        out = io.TextIOWrapper(out_raw, encoding="cp1252", errors="strict")
        err = io.TextIOWrapper(err_raw, encoding="cp1252", errors="strict")
        pf.configure_stdio(out, err)
        out.write("ошибка вывода\n")
        err.write("сбой диагностики\n")
        out.flush()
        err.flush()
        self.assertEqual(out.encoding.lower(), "utf-8")
        self.assertEqual(out.errors, "replace")
        self.assertEqual(err.encoding.lower(), "utf-8")
        self.assertEqual(err.errors, "replace")
        self.assertEqual(out_raw.getvalue().decode("utf-8"), "ошибка вывода\n")
        self.assertEqual(err_raw.getvalue().decode("utf-8"), "сбой диагностики\n")


class ParseAndSelectTest(unittest.TestCase):
    def test_parse_nvidia_smi_csv(self):
        gpus = pf.parse_nvidia_smi_csv(SMI)
        self.assertEqual([g.index for g in gpus], [0, 1, 2])
        self.assertEqual(gpus[1].name, "NVIDIA RTX PRO 6000 Blackwell")
        self.assertEqual(gpus[1].uuid, "GPU-bbbb")
        self.assertEqual(gpus[1].memory_total_mib, 97887)
        self.assertEqual(gpus[1].memory_free_mib, 97000)
        self.assertEqual(gpus[1].driver_version, "580.65")

    def test_parse_compute_apps(self):
        busy = pf.parse_compute_apps_csv("GPU-bbbb, 1234, python.exe, 1000\n")
        self.assertEqual(busy, {"GPU-bbbb"})

    def test_select_gpu_by_index_not_position(self):
        gpus = pf.parse_nvidia_smi_csv(
            "2, GPU C, GPU-cccc, 10, 9, 1\n0, GPU A, GPU-aaaa, 10, 9, 1\n1, GPU B, GPU-bbbb, 10, 9, 1\n"
        )
        gpu = pf.select_gpu(gpus, 2)
        self.assertEqual((gpu.index, gpu.uuid), (2, "GPU-cccc"))
        gpu = pf.select_gpu(gpus, 1)
        self.assertEqual((gpu.index, gpu.uuid), (1, "GPU-bbbb"))

    def test_missing_gpu_does_not_fallback(self):
        gpus = pf.parse_nvidia_smi_csv(
            "0, GPU A, GPU-aaaa, 10, 9, 1\n2, GPU C, GPU-cccc, 10, 9, 1\n"
        )
        with self.assertRaises(pf.PreflightError) as cm:
            pf.select_gpu(gpus, 1)
        self.assertEqual(cm.exception.code, "gpu_missing")
        self.assertIn("доступны: 0, 2", str(cm.exception))

    def test_busy_gpu_rejected(self):
        gpus = pf.parse_nvidia_smi_csv(SMI)
        busy = pf.parse_compute_apps_csv("GPU-bbbb, 1234, python.exe, 1000\n")
        with self.assertRaises(pf.PreflightError) as cm:
            pf.select_gpu(gpus, 1, busy)
        self.assertEqual(cm.exception.code, "gpu_busy")
        self.assertIn("GPU-bbbb", str(cm.exception))


class ReportTest(unittest.TestCase):
    def test_missing_torch_sets_env_before_import_and_writes_json(self):
        env = {}
        seen = []

        def importer():
            seen.append(env.get("CUDA_VISIBLE_DEVICES"))
            raise ImportError("torch отсутствует")

        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "training-preflight.json"
            report = pf.run_preflight(
                1,
                out,
                gpu_runner=lambda _cmd: SMI,
                busy_runner=lambda _cmd: "",
                torch_importer=importer,
                version_fn=lambda name: None if name == "torch" else "1.0",
                environ=env,
            )
            data = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(report["status"], "missing_dependency")
        self.assertFalse(report["ok"])
        self.assertEqual(seen, ["GPU-bbbb"])  # CVD выставлен до импорта torch
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "GPU-bbbb")
        self.assertEqual(data["status"], "missing_dependency")
        self.assertTrue(data["report_saved"])
        self.assertFalse(data["torch"]["imported"])
        self.assertIsNone(data["packages"]["torch"])
        self.assertEqual(data["gpu"]["selected_index"], 1)
        self.assertEqual(data["gpu"]["logical_device"], "cuda:0")
        self.assertEqual(data["not_tested"], pf.NOT_TESTED)

    def test_nvidia_smi_failure_still_writes_json(self):
        def boom(_cmd):
            raise pf.PreflightError("command_failed", "nvidia-smi не найден")

        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "training-preflight.json"
            report = pf.run_preflight(
                1, out, gpu_runner=boom, busy_runner=boom,
                torch_importer=lambda: self.fail("torch не должен импортироваться"),
                version_fn=lambda _name: None,
            )
            data = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(report["status"], "command_failed")
        self.assertEqual(data["error"]["code"], "command_failed")
        self.assertTrue(data["report_saved"])
        self.assertIsNone(data["gpu"])
        self.assertFalse(data["nvidia_smi"]["available"])
        self.assertIn("fp8_checkpoint_load", data["not_tested"])

    def test_busy_gpu_stops_before_torch_and_writes_json(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "training-preflight.json"
            report = pf.run_preflight(
                1, out, gpu_runner=lambda _cmd: SMI,
                busy_runner=lambda _cmd: "GPU-bbbb, 42, python.exe, 1000\n",
                torch_importer=lambda: self.fail("занятую GPU нельзя импортировать/touch"),
                version_fn=lambda _name: None,
            )
            data = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(report["status"], "gpu_busy")
        self.assertEqual(data["error"]["code"], "gpu_busy")
        self.assertTrue(data["report_saved"])
        self.assertIsNone(data["gpu"])
        self.assertFalse(data["torch"]["imported"])

    def test_missing_gpu_stops_before_torch_and_writes_json(self):
        only0 = "0, GPU A, GPU-aaaa, 10, 9, 1\n"
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "training-preflight.json"
            report = pf.run_preflight(
                1, out, gpu_runner=lambda _cmd: only0,
                busy_runner=lambda _cmd: "",
                torch_importer=lambda: self.fail("нельзя молча брать GPU0"),
                version_fn=lambda _name: None,
            )
            data = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(report["status"], "gpu_missing")
        self.assertEqual(data["error"]["code"], "gpu_missing")
        self.assertTrue(data["report_saved"])
        self.assertIsNone(data["gpu"])


class SuccessAndWriteFailureTest(unittest.TestCase):
    def test_fake_torch_success_checks_uuid_import_order_and_not_tested(self):
        env, seen = {}, []

        def importer():
            seen.append(env.get("CUDA_VISIBLE_DEVICES"))
            return fake_torch()

        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "training-preflight.json"
            with mock.patch.object(pf, "bf16_smoke", return_value=dict(SMOKE)) as smoke:
                report = pf.run_preflight(
                    1, out, gpu_runner=lambda _cmd: SMI,
                    busy_runner=lambda _cmd: "",
                    torch_importer=importer,
                    version_fn=lambda _name: "1.0",
                    environ=env,
                )
            data = json.loads(out.read_text(encoding="utf-8"))
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                rc = pf.main(["--gpu", "1", "--out", str(out)],
                             runner=lambda gpu, o: report)

        self.assertEqual(rc, 0)
        self.assertEqual(report["status"], "ok")
        self.assertTrue(report["ok"])
        self.assertTrue(report["report_saved"])
        self.assertEqual(seen, ["GPU-bbbb"])  # UUID выставлен до импорта torch
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "GPU-bbbb")
        self.assertEqual(report["gpu"]["logical_device"], "cuda:0")
        smoke.assert_called_once()
        self.assertTrue(data["report_saved"])
        self.assertNotIn("report_write_error", data)
        self.assertTrue(data["torch"]["bf16_test"]["ok"])
        self.assertEqual(data["torch"]["cuda_build"], "12.8")
        self.assertTrue(data["torch"]["cuda_available"])
        self.assertEqual(data["torch"]["device_count"], 1)
        for item in ("fp8_checkpoint_load", "gigachat_training_or_convergence",
                     "quality_after_ternarization", "gguf_export"):
            self.assertIn(item, data["not_tested"])

    def test_write_failure_gives_nonzero_and_honest_message(self):
        env = {}

        def importer():
            return fake_torch()

        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "training-preflight.json"
            with mock.patch.object(pf, "bf16_smoke", return_value=dict(SMOKE)), \
                    mock.patch.object(pf, "_write_json", side_effect=OSError("disk full")):
                report = pf.run_preflight(
                    1, out, gpu_runner=lambda _cmd: SMI,
                    busy_runner=lambda _cmd: "",
                    torch_importer=importer,
                    version_fn=lambda _name: "1.0",
                    environ=env,
                )
                stdout, stderr = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    rc = pf.main(["--gpu", "1", "--out", str(out)],
                                 runner=lambda gpu, o: report)

        self.assertEqual(rc, 1)
        self.assertEqual(report["status"], "report_write_failed")
        self.assertEqual(report["diagnostic_status"], "ok")
        self.assertTrue(report["torch"]["bf16_test"]["ok"])
        self.assertFalse(report["ok"])
        self.assertFalse(report["report_saved"])
        self.assertIn("disk full", report["report_write_error"])
        self.assertIn("НЕ СОХРАНЁН", stderr.getvalue())
        self.assertIn("disk full", stderr.getvalue())
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
