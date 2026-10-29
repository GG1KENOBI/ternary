import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import tempfile
import threading
import unittest
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock, patch

KIT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("compare", KIT / "compare.py")
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)
compare.configure_output()


@contextlib.contextmanager
def http_fixture(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class Tests(unittest.TestCase):
    def test_download_resume_and_ignored_range(self):
        payload = b"GGUF-test" * 100
        for supports_range in (True, False):
            requests = []
            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):
                    value = self.headers.get("Range")
                    requests.append(value)
                    offset = int(value.split("=")[1].split("-")[0]) if value and supports_range else 0
                    self.send_response(206 if offset else 200)
                    if offset:
                        self.send_header("Content-Range", f"bytes {offset}-{len(payload)-1}/{len(payload)}")
                    self.end_headers()
                    self.wfile.write(payload[offset:])
                def log_message(self, *args):
                    pass
            with tempfile.TemporaryDirectory() as tmp, http_fixture(Handler) as port:
                target = Path(tmp) / "model.gguf"
                target.with_suffix(".gguf.part").write_bytes(payload[:21])
                compare.download(f"http://127.0.0.1:{port}/file", target, len(payload), hashlib.sha256(payload).hexdigest())
                self.assertEqual(target.read_bytes(), payload)
                self.assertEqual(requests, ["bytes=21-"])
                # Повторное использование проверяет хеш без новой сети.
                compare.download("http://unused", target, len(payload), hashlib.sha256(payload).hexdigest())
                self.assertFalse(target.with_suffix(".gguf.part").exists())

    def test_download_rejects_wrong_hash_and_preserves_existing_file(self):
        response = Mock(status=200, headers={})
        response.read.side_effect = [b"wrong", b""]
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        with tempfile.TemporaryDirectory() as tmp, patch.object(compare.urllib.request, "urlopen", return_value=response):
            target = Path(tmp) / "model"
            with self.assertRaises(RuntimeError):
                compare.download("http://invalid", target, 5, "0" * 64, attempts=1)
            self.assertFalse(target.exists())
            self.assertFalse(target.with_name("model.part").exists())
            target.write_bytes(b"old")
            with self.assertRaises(RuntimeError):
                compare.download("http://invalid", target, 5, "0" * 64, attempts=1)
            self.assertEqual(target.read_bytes(), b"old")

    def test_partial_eof_keeps_bytes_for_next_attempt(self):
        response = Mock(status=200, headers={})
        response.read.side_effect = [b"abc", b""]
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        with tempfile.TemporaryDirectory() as tmp, patch.object(compare.urllib.request, "urlopen", return_value=response):
            target = Path(tmp) / "model"
            with self.assertRaises(RuntimeError):
                compare.download("http://invalid", target, 10, "0" * 64, attempts=1)
            self.assertEqual(target.with_name("model.part").read_bytes(), b"abc")

    def test_chat_uses_messages_and_no_model_specific_template(self):
        body = compare.chat_request({"instruction": "Сделай JSON", "text": "текст", "max_tokens": 80})
        self.assertEqual(body["messages"], [{"role": "user", "content": "Сделай JSON\n\nТекст:\nтекст"}])
        self.assertNotIn("prompt", body)
        self.assertFalse(body["chat_template_kwargs"]["enable_thinking"])
        self.assertFalse(body["cache_prompt"])
        self.assertEqual(len(compare.load_tasks()), 10)
        self.assertGreater(len(compare.load_tasks(["sum_xlong"])[0]["text"]), 30000)

    def test_partial_cpu_offload_is_not_accepted(self):
        compare.assert_offloaded("load_tensors: offloaded 41/41 layers to GPU")
        for log in ("offloaded 0/41 layers to GPU", "offloaded 40/41 layers to GPU", "CPU only"):
            with self.assertRaises(RuntimeError):
                compare.assert_offloaded(log)

    def test_default_quiet_log_does_not_masquerade_as_cpu_offload(self):
        args = argparse.Namespace(context=32768, threads=8)
        command = compare.server_command("server.exe", "model.gguf", args, 1234, "key")
        self.assertEqual(command[command.index("--log-verbosity") + 1], "4")
        self.assertEqual(command[command.index("--log-colors") + 1], "off")
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "server.log"
            log.write_text("verbosity = 3\nmodel loaded\nlistening on localhost\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                compare.wait_offloaded(log, timeout=0)

    def test_gpu_index_maps_to_physical_uuid_and_driver_is_checked(self):
        text = '0, Display GPU, 8000, 6000, 582.10, GPU-screen\n1, RTX PRO 6000, 97887, 97000, 582.10, GPU-test\n'
        with patch.object(compare, "run_text", return_value=text):
            self.assertEqual(compare.inspect_gpu(1, 580)["selected"]["uuid"], "GPU-test")
            with self.assertRaises(RuntimeError):
                compare.inspect_gpu(2, 580)
            with self.assertRaises(RuntimeError):
                compare.inspect_gpu(1, 600)

    def test_stop_only_own_child_and_kill_on_timeout(self):
        process = Mock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("fake", 15), 0]
        compare.stop_process(process)
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        self.assertEqual(process.wait.call_count, 2)

    def test_lock_blocks_second_runner_and_releases_on_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lock"
            with self.assertRaises(ValueError):
                with compare.run_lock(path):
                    with self.assertRaises(RuntimeError):
                        with compare.run_lock(path):
                            self.fail("second runner admitted")
                    raise ValueError("stop")
            self.assertFalse(path.exists())

    def test_zip_extraction_does_not_traverse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("../evil.dll", b"x")
            with self.assertRaises(RuntimeError):
                compare.extract_runtime(archive, root / "bin")
            self.assertFalse((root / "evil.dll").exists())

    def test_report_includes_error_and_never_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "results"
            folder.mkdir()
            (Path(tmp) / "model.gguf").write_bytes(b"big")
            state = {"results": {"giga": {"model": {"name": "GigaChat"}, "status": "error", "error": "unsupported", "tasks": {}}}}
            compare.save_json(folder / "results.json", state)
            compare.write_report(folder, state)
            with zipfile.ZipFile(str(folder) + ".zip") as z:
                self.assertEqual(set(z.namelist()), {"answers.md", "results.json"})
                self.assertIn("unsupported", z.read("answers.md").decode())

    def test_model_lifecycle_success_and_task_failure(self):
        for fail in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                weights = root / "owned.gguf"
                unrelated = root / "old-user.gguf"
                unrelated.write_bytes(b"keep")
                state = {"results": {}}
                args = argparse.Namespace(gpu=1, context=32768, threads=8, timeout=3, keep_models=False)
                model = dict(id="test", name="test", repo="public/repo", revision="a", file="owned.gguf", size=3, sha256="a")
                process = Mock()
                process.poll.return_value = None
                def download(*unused):
                    weights.write_bytes(b"123")
                def popen(command, **kwargs):
                    self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "1")
                    kwargs["stdout"].write("offloaded 41/41 layers to GPU\n")
                    kwargs["stdout"].flush()
                    return process
                def terminate():
                    self.assertTrue(weights.exists(), "weights deleted before server stop")
                process.terminate.side_effect = terminate
                def request(port, key, path, body=None, timeout=0):
                    if path == "/props":
                        return {"chat_template": "native-template"}
                    if fail:
                        raise TimeoutError("task timeout")
                    return {"choices": [{"message": {"content": "ответ"}, "finish_reason": "stop"}], "timings": {"predicted_per_second": 42}}
                with patch.object(compare, "model_path", return_value=weights), \
                     patch.object(compare, "download", side_effect=download), \
                     patch.object(compare.subprocess, "Popen", side_effect=popen), \
                     patch.object(compare, "wait_ready"), \
                     patch.object(compare, "MemorySampler") as sampler, \
                     patch.object(compare, "request_json", side_effect=request):
                    sampler.return_value.peak, sampler.return_value.samples = 1200, 3
                    task_list = [{"id": "short", "instruction": "Ответь", "text": "", "max_tokens": 50}]
                    if fail:
                        with self.assertRaises(RuntimeError):
                            compare.compare_model(root / "server.exe", model, task_list, args,
                                                  root / "logs", state, root / "results.json")
                    else:
                        compare.compare_model(root / "server.exe", model, task_list, args,
                                              root / "logs", state, root / "results.json")
                self.assertEqual(weights.exists(), fail)
                self.assertTrue(unrelated.exists())
                process.terminate.assert_called_once()
                row = json.loads((root / "results.json").read_text())["results"]["test"]
                self.assertEqual(row["status"], "error" if fail else "complete")
                self.assertEqual(row["weights_deleted"], not fail)
                self.assertIn("<session-key>", row["command"])

    def test_failed_first_model_stops_before_second_download(self):
        args = argparse.Namespace(gpu=1, server=None, check=False, keep_models=False,
                                  context=32768, threads=8, timeout=3)
        with tempfile.TemporaryDirectory() as tmp, patch.object(compare, "HERE", Path(tmp)), \
             patch.object(compare, "inspect_gpu", return_value={"selected": {"uuid": "GPU-test", "free_mib": 1e8}}), \
             patch.object(compare, "setup_runtime", return_value=Path("server.exe")), \
             patch.object(compare, "run_text", return_value="CUDA0"), \
             patch.object(compare, "compare_model", side_effect=RuntimeError("missing offload log")) as model_run:
            code = compare.execute(args, {"runtime": {"minimum_driver_major": 580}},
                                   [{"id": "first", "name": "first", "size": 1},
                                    {"id": "second", "name": "second", "size": 1}], [])
            self.assertEqual(code, 1)
            self.assertEqual(model_run.call_count, 1)
            self.assertEqual(len(list((Path(tmp) / "results").glob("*.zip"))), 1)

    def test_real_local_http_auth_and_unicode(self):
        captured = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                captured.append((self.headers["Authorization"], json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"answer": "Москва"}).encode())
            def log_message(self, *args):
                pass
        with http_fixture(Handler) as port:
            result = compare.request_json(port, "session-key", "/chat", {"text": "Привет"})
            self.assertEqual(result["answer"], "Москва")
            self.assertEqual(captured, [("Bearer session-key", {"text": "Привет"})])

    @unittest.skipUnless(os.name == "nt", "BAT launcher is tested on Windows CI")
    def test_windows_bat_launcher_and_exit_code(self):
        for arguments, expected in (("--plan --models giga --tasks sum_short", 0), ("--unknown-option", 2)):
            result = subprocess.run(["cmd.exe", "/c", "run.bat " + arguments], cwd=KIT,
                                    input="\n", capture_output=True, text=True, encoding="utf-8",
                                    errors="replace", timeout=30)
            self.assertEqual(result.returncode, expected, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
