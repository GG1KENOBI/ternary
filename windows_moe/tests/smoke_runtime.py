"""Сквозной тест закреплённого Windows-сервера на модели 260K (~1,2 МБ), CPU CI."""
import argparse
import json
import secrets
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compare


def main():
    compare.configure_output()
    manifest = json.loads((compare.HERE / "models.json").read_text(encoding="utf-8"))
    server = compare.setup_runtime(manifest)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        weights = compare.download(
            "https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf",
            root / "tiny.gguf", 1185376,
            "270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d")
        for level in (3, 4):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            key = secrets.token_hex(16)
            command = compare.server_command(server, weights,
                                            argparse.Namespace(context=512, threads=2), port, key)
            # Меняем только устройство и размер задачи: GPU на CI нет.
            for option, value in (("--device", "none"), ("-ngl", "0"), ("--log-verbosity", str(level))):
                command[command.index(option) + 1] = value
            command += ["--chat-template", "chatml"]
            log_path = root / f"server-{level}.log"
            proc = None
            try:
                with log_path.open("w", encoding="utf-8") as log:
                    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    compare.wait_ready(proc, port, key, timeout=60)
                    response = compare.request_json(port, key, "/v1/chat/completions",
                        compare.chat_request({"instruction": "Hello", "max_tokens": 8}), timeout=30)
                    assert response.get("choices"), response
                    assert "timings" in response, response
            finally:
                compare.stop_process(proc)
            text = log_path.read_text(encoding="utf-8", errors="replace")
            has_library_info = "model buffer size" in text
            assert has_library_info == (level == 4), text
            print(f"llama b11960: level={level}, library INFO visible={has_library_info}; chat/timings: PASS")


if __name__ == "__main__":
    main()
