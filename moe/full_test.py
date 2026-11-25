"""v4 step 4: full-length answers from the v4 MoE GGUF in stock llama-server, in the full test's format.

Run on Windows (any Python 3.10+, no packages): python moe/full_test.py <v4.gguf>
One llama-server per card (each holds a full copy; never tensor parallel), prompts split between them, several
slots per server. The normal answers are reused from vllm_full_test.json, so score_full.py compares v4 against
the same baseline v3 was scored against:
    python reports/score_full.py --answers data_review/reports/v4_full_test.json --out data_review/reports/v4_scores.json
"""

import argparse
import json
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "data_review" / "reports"
LLAMA = Path("C:/llm/tq-vulkan/build/bin/llama-server.exe")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("gguf", type=Path)
    p.add_argument("--out", type=Path, default=REPORTS / "v4_full_test.json")
    p.add_argument("--cards", default="Vulkan0,Vulkan1")
    p.add_argument("--slots", type=int, default=6, help="parallel requests per card")
    p.add_argument("--slot-ctx", type=int, default=12288, help="context per request (prompt + answer)")
    p.add_argument("--max-new", type=int, default=8192)
    return p.parse_args(argv)


def start(args, card: str, port: int, log: Path):
    command = [str(LLAMA), "-m", str(args.gguf), "-dev", card, "-ngl", "999", "--port", str(port),
               "-np", str(args.slots), "-c", str(args.slots * args.slot_ctx), "-fa", "on",
               "-ctk", "q8_0", "-ctv", "q8_0", "--jinja"]
    return subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=log.open("w"))


def wait_ready(port: int, process) -> None:
    for _ in range(300):
        if process.poll() is not None:
            raise RuntimeError(f"llama-server on port {port} exited ({process.returncode})")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise TimeoutError(f"port {port} not ready")


def answer(port: int, prompt: str, max_new: int) -> dict:
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "temperature": 0,
                       "max_tokens": max_new}).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                     {"Content-Type": "application/json"})
    reply = json.loads(urllib.request.urlopen(request, timeout=7200).read())
    return {"text": reply["choices"][0]["message"]["content"].strip(), "end": reply["choices"][0]["finish_reason"],
            "tokens": reply["usage"]["completion_tokens"]}


def main(argv=None):
    args = parse_args(argv)
    rows = json.loads((REPORTS / "vllm_full_test.json").read_text(encoding="utf-8"))
    cards = args.cards.split(",")
    ports = [8100 + i for i in range(len(cards))]
    servers = [start(args, card, port, args.out.with_suffix(f".{card}.log")) for card, port in zip(cards, ports)]
    try:
        for port, server in zip(ports, servers):
            wait_ready(port, server)
        start_time = time.time()
        jobs = [(ports[i % len(ports)], row) for i, row in enumerate(rows)]
        with ThreadPoolExecutor(len(ports) * args.slots) as pool:
            done = list(pool.map(lambda job: answer(job[0], job[1]["prompt"], args.max_new), jobs))
        seconds = time.time() - start_time
    finally:
        for server in servers:
            server.terminate()
            server.wait()
    out = [{**row, "steered": d["text"], "steered_tokens": d["tokens"], "steered_end": d["end"]}
           for row, d in zip(rows, done)]
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    made = sum(d["tokens"] for d in done)
    print(json.dumps({"answers": len(out), "tokens": made, "seconds": round(seconds), "tok_per_s": round(made / seconds),
                      "hit_limit": sum(d["end"] == "length" for d in done)}))


if __name__ == "__main__":
    main()
