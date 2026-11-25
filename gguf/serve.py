"""Routed steering on stock llama.cpp: one OpenAI-style endpoint that picks the topic and adds its control vector.

Run on Windows with the ROCm venv (it has torch + transformers):
    C:/Users/mhnda/.venvs/ai-training-rocm-windows/Scripts/python.exe gguf/serve.py
Then point any OpenAI client at http://127.0.0.1:8091/v1 (model name: anything).

How it works:
1. Starts llama-server in router mode with one preset per topic. Every preset is the same base GGUF; the topic presets
   add their control vector (gguf/export_vectors.py). llama-server loads and swaps them on demand.
2. For each chat request, the trained router reads the last user message through layers 0..read_layer of the
   frozen model (on the GPU, the same code as system.py) and picks one topic or "none".
3. The request goes to that preset. The reply carries the pick in X-Steer-Route / X-Steer-Confidence headers, and
   in a "steer" field for non-streamed replies. Send "steer": "<topic>" or "none" in the body to skip the router.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("STEERING_PROFILE", "qwen4b")

BUNDLE = Path("C:/llm/models/qwen3-4b-steer")
LLAMA = Path("C:/llm/tq-vulkan/build/bin/llama-server.exe")
NAME = "qwen3-4b-steer"
NONE = "none"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bundle", type=Path, default=BUNDLE, help="folder with the GGUF, vectors/ and hf/")
    p.add_argument("--gguf", default="Qwen3-4B-Instruct-2507-F16.gguf", help="base model file inside --bundle")
    p.add_argument("--llama", type=Path, default=LLAMA)
    p.add_argument("--port", type=int, default=8091, help="this proxy")
    p.add_argument("--llama-port", type=int, default=8090)
    p.add_argument("--device", default="Vulkan1", help="llama.cpp device for the model")
    p.add_argument("--ctx", type=int, default=16384, help="total context per loaded topic, shared by --parallel")
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--models-max", type=int, default=1, help="topics kept loaded at once (F16 = 8 GB each)")
    p.add_argument("--router", type=Path, default=None, help="router checkpoint (default: the profile's)")
    p.add_argument("--router-device", default="cuda:0",
                   help="cuda:0 = 23 ms per prompt; cpu works but took 30 s per prompt on this PC")
    return p.parse_args(argv)


def topics_in(bundle: Path) -> list[str]:
    return sorted(path.stem.removeprefix("steer-") for path in (bundle / "vectors").glob("steer-*.gguf"))


def write_presets(args, topics: list[str]) -> Path:
    """One llama-server preset per topic; all share the base GGUF, so only the control vector differs."""
    shared = [f"model = {(args.bundle / args.gguf).as_posix()}", "n-gpu-layers = 999", f"device = {args.device}",
              f"ctx-size = {args.ctx}", f"parallel = {args.parallel}", "jinja = true"]
    lines = ["version = 1", ""]
    for topic in [NONE, *topics]:
        lines += [f"[steer-{topic}]", *shared]
        if topic != NONE:
            lines.append(f"control-vector = {(args.bundle / 'vectors' / f'steer-{topic}.gguf').as_posix()}")
        lines.append("load-on-startup = " + ("true" if topic == NONE else "false"))
        lines.append("")
    path = args.bundle / "presets.ini"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


class Router:
    """The trained router on the first read_layer + 1 layers of the frozen model."""

    def __init__(self, hf: Path, checkpoint: Path, device: str):
        import torch
        from transformers import AutoModel, AutoTokenizer

        from router.pipeline import chat_parts
        from system import load_router

        self.head, _ = load_router(checkpoint, device)
        dtype = torch.float32 if device == "cpu" else torch.float16
        self.base = AutoModel.from_pretrained(hf, dtype=dtype, num_hidden_layers=self.head.read_layer + 1)
        self.base = self.base.to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(hf)
        self.parts = chat_parts(str(hf))
        self.lock = threading.Lock()

    def route(self, text: str):
        from system import pick_routes, read_prompts

        with self.lock:
            features = read_prompts(self.base, self.tokenizer, self.parts, self.head.read_layer, [text])
            return pick_routes(self.head, features)[0]


def last_user_text(messages: list[dict]) -> str:
    """The router was trained on single prompts, so it reads the latest user turn."""
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content") or ""
            if isinstance(content, list):  # OpenAI content parts.
                content = "\n".join(part.get("text", "") for part in content if part.get("type") == "text")
            return content
    return ""


def make_handler(router: Router, topics: list[str], upstream: str):
    allowed = {NONE, *topics}

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status: int, data: dict, headers: dict | None = None) -> None:
            raw = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                self.send_json(200, {"object": "list", "data": [{"id": NAME, "object": "model", "owned_by": "local"}]})
            elif self.path == "/health":
                self.send_json(200, {"status": "ok", "topics": topics})
            else:
                self.send_json(404, {"error": f"unknown path {self.path}"})

        def do_POST(self):
            if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
                return self.send_json(404, {"error": "only /v1/chat/completions is routed"})
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            choice, start = body.pop("steer", "auto"), time.time()
            if choice == "auto":
                route = router.route(last_user_text(body.get("messages", [])))
                name, confidence = route.name, route.confidence
            elif choice in allowed:
                name, confidence = choice, 1.0
            else:
                return self.send_json(400, {"error": f"steer must be auto or one of {sorted(allowed)}"})
            route_ms = round((time.time() - start) * 1000)
            print(json.dumps({"route": name, "confidence": round(confidence, 3), "route_ms": route_ms}), flush=True)
            body["model"] = f"steer-{name}"
            tags = {"X-Steer-Route": name, "X-Steer-Confidence": f"{confidence:.3f}"}
            self.forward(body, tags, {"route": name, "confidence": round(confidence, 3)})

        def forward(self, body: dict, tags: dict, steer: dict) -> None:
            request = urllib.request.Request(f"{upstream}/v1/chat/completions", json.dumps(body).encode(),
                                             {"Content-Type": "application/json"})
            try:
                reply = urllib.request.urlopen(request, timeout=3600)
            except urllib.error.HTTPError as error:
                return self.send_json(error.code, {"error": error.read().decode(errors="replace")}, tags)
            if not body.get("stream"):
                data = json.loads(reply.read())
                return self.send_json(200, {**data, "model": NAME, "steer": steer}, tags)
            self.send_response(200)  # HTTP/1.0: the stream ends when the connection closes.
            self.send_header("Content-Type", "text/event-stream")
            for key, value in tags.items():
                self.send_header(key, value)
            self.end_headers()
            while chunk := reply.read1(8192):
                self.wfile.write(chunk)
                self.wfile.flush()

        def log_message(self, *_):
            pass  # One JSON line per request is printed in do_POST.

    return Handler


def wait_ready(url: str, process, seconds: int = 300) -> None:
    for _ in range(seconds):
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited with code {process.returncode}; see its log above")
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise TimeoutError(f"{url} did not answer in {seconds} s")


def main(argv=None):
    from model_profiles import get_profile

    args = parse_args(argv)
    topics = topics_in(args.bundle)
    if not topics:
        raise FileNotFoundError(f"No vectors in {args.bundle / 'vectors'}; run gguf/export_vectors.py first")
    presets = write_presets(args, topics)
    # Router mode also lists the Hugging Face cache; WSL-made links there stop it, so give it an empty cache.
    empty = args.bundle / "empty-cache"
    empty.mkdir(exist_ok=True)
    env = {**os.environ, "HF_HUB_CACHE": str(empty), "HF_HOME": str(empty), "LLAMA_CACHE": str(empty)}
    llama = subprocess.Popen([str(args.llama), "--models-preset", str(presets), "--models-max", str(args.models_max),
                              "--host", "127.0.0.1", "--port", str(args.llama_port)], env=env)
    try:
        router = Router(args.bundle / "hf", args.router or get_profile().router_checkpoint, args.router_device)
        router.route("warm up")  # The first GPU call compiles kernels (about 8 s); do it before taking requests.
        upstream = f"http://127.0.0.1:{args.llama_port}"
        wait_ready(f"{upstream}/health", llama)
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(router, topics, upstream))
        print(json.dumps({"ready": f"http://127.0.0.1:{args.port}/v1", "topics": topics}), flush=True)
        server.serve_forever()
    finally:
        llama.terminate()
        llama.wait()


if __name__ == "__main__":
    main()
