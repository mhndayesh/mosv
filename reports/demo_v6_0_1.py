"""Use the shipped v6.0.1 GGUF the way a user would: llama-server with its chat template, plain chat requests.

Run (gguf-tools or ai-training venv, repo root, both cards free):
    python reports/demo_v6_0_1.py --model C:/llm/models/gemma4-12b/v6_0_1/MoSV-Gemma-4-12B-v6.0.1-Q4_K_M.gguf
The base GGUF runs on one card and the trained GGUF on the other, at the same time, with the same hand-written
prompts (none of them from the datasets), the model's recommended sampling, thinking off and a fixed seed.
Writes results/v6_0_1_demo.md: each prompt with both answers, one after the other.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = "C:/llm/tq-vulkan/build/bin/llama-server.exe"
BASE = Path("C:/llm/models/gemma4-12b/gemma-4-12b-it-Q4_K_M.gguf")
SAMPLING = {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "min_p": 0.0, "seed": 7}  # Gemma's own settings
PROMPTS = [  # (what it tests, prompt): written by hand for this demo
    ("teaching (trained)", "Teach fractions (Mathematics) to a Primary 3 student."),
    ("debugging (trained)", "Here is a bug report from my app: 'After upgrading to version 2.4, the export button saves an "
                            "empty CSV file on Windows, but it works on Mac. No error is shown.' Where does it go wrong and how do I fix it?"),
    ("story (trained)", "Write a short story about a lighthouse keeper who finds a message in a bottle addressed to her."),
    ("everyday (not trained)", "What's a good way to keep fresh basil from wilting for a week?"),
    ("engineering (not trained)", "Write a Python function that returns the n-th Fibonacci number without recursion."),
]


def start(model: Path, card: str, port: int) -> subprocess.Popen:
    cmd = [SERVER, "-m", str(model), "-dev", card, "-ngl", "99", "--port", str(port), "-fa", "on", "-c", "8192",
           "-np", "1", "--jinja", "--no-webui", "-ctk", "q8_0", "-ctv", "q8_0"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(600):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            return proc
        except Exception:
            time.sleep(0.3)
    proc.terminate()
    raise SystemExit(f"llama-server for {model.name} did not start")


def chat(port: int, prompt: str) -> dict:
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 2048,
            "chat_template_kwargs": {"enable_thinking": False}, **SAMPLING}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    out = json.loads(urllib.request.urlopen(req, timeout=900).read())
    return {"text": out["choices"][0]["message"]["content"], "tokens": out["usage"]["completion_tokens"]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, type=Path)
    args = p.parse_args(argv)
    servers = [start(BASE, "Vulkan0", 8301), start(args.model, "Vulkan1", 8302)]
    try:
        with ThreadPoolExecutor(2) as pool:
            pairs = [list(pool.map(lambda port: chat(port, prompt), (8301, 8302))) for _, prompt in PROMPTS]
    finally:
        for s in servers:
            s.terminate()
        for s in servers:
            s.wait()
    lines = [f"# v6.0.1 demo: base Gemma 4 12B vs {args.model.name}", "",
             "llama-server `--jinja`, plain chat requests, temperature 1.0, top_k 64, top_p 0.95, min_p 0, seed 7, "
             "thinking off. Prompts written by hand for this demo.", ""]
    for (what, prompt), (base, trained) in zip(PROMPTS, pairs):
        lines += [f"## {what}", "", f"**Prompt:** {prompt}", "",
                  f"### Base ({base['tokens']} tokens)", "", base["text"], "",
                  f"### MoSV v6.0.1 ({trained['tokens']} tokens)", "", trained["text"], ""]
        print(f"{what:26s} tokens base {base['tokens']:5d} | trained {trained['tokens']:5d} | identical: {base['text'] == trained['text']}")
    out = ROOT / "results" / "v6_0_1_demo.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
