"""Answer a locked v5 test split with a GGUF model on both cards (one llama-server per card) and record everything.

Run (Windows, gguf-tools venv):
    python reports/answer_v5.py --topic engineering --model C:/llm/models/bonsai-4b/Bonsai-4B-Q1_0.gguf --tag bonsai_normal
Greedy, f16 KV cache. --max-new 0 = no output cap: an answer ends at its own stop token, or only if it fills its
slot's whole context (--slot-ctx, default 16,384 tokens; the row's finish is then "length"). Writes data_review/v5_runs/<topic>/<tag>.jsonl with
id, source, prompt, answer, tokens, finish reason and seconds, plus a settings header line. Same prompts, same
order and same settings for every model, so before/after answers line up by id.
"""

import argparse
import json
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = "C:/llm/tq-vulkan/build/bin/llama-server.exe"
CARDS = (("Vulkan0", 8101), ("Vulkan1", 8102))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--topic", required=True)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--tag", required=True, help="output name, e.g. bonsai_normal")
    p.add_argument("--max-new", type=int, default=0, help="0 = no cap (stop token or full slot context)")
    p.add_argument("--slot-ctx", type=int, default=16384, help="context tokens per parallel answer")
    p.add_argument("--slots", type=int, default=5, help="parallel answers per card (5 x 16k f16 fits each card)")
    return p.parse_args(argv)


def locked_prompts(topic: str) -> list[dict]:
    ids = set(json.loads((ROOT / "results" / "locked_v5" / f"{topic}.json").read_text(encoding="utf-8"))["test_ids"])
    path = ROOT / "data_review" / "clean_v5" / topic / "test.jsonl"
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    if {r["id"] for r in rows} != ids:
        raise SystemExit(f"{path} does not match its lock")
    return rows


def start(model: Path, card: str, port: int, args) -> subprocess.Popen:
    ctx = args.slots * args.slot_ctx
    cmd = [SERVER, "-m", str(model), "-dev", card, "-ngl", "99", "--port", str(port), "-fa", "on", "-c", str(ctx),
           "-np", str(args.slots), "-b", "4096", "-ub", "1024"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(180):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return proc
        except Exception:
            time.sleep(1)
    proc.terminate()
    raise SystemExit(f"server on {card} did not start")


def answer(row: dict, port: int, max_new: int) -> dict:
    request = {"messages": [{"role": "user", "content": row["prompt"]}], "temperature": 0}
    if max_new:
        request["max_tokens"] = max_new
    body = json.dumps(request).encode()
    t = time.time()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type": "application/json"})
    out = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    choice = out["choices"][0]
    return {"id": row["id"], "source": row["source"], "prompt": row["prompt"], "answer": choice["message"]["content"],
            "tokens": out["usage"]["completion_tokens"], "finish": choice["finish_reason"],
            "seconds": round(time.time() - t, 2)}


def main(argv=None):
    args = parse_args(argv)
    rows = locked_prompts(args.topic)
    servers = [start(args.model, card, port, args) for card, port in CARDS]
    t = time.time()
    try:
        with ThreadPoolExecutor(args.slots * len(CARDS)) as pool:
            jobs = [pool.submit(answer, row, CARDS[i % len(CARDS)][1], args.max_new) for i, row in enumerate(rows)]
            results = [j.result() for j in jobs]
    finally:
        for s in servers:
            s.terminate()
    took = time.time() - t
    out = ROOT / "data_review" / "v5_runs" / args.topic / f"{args.tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    header = {"settings": {**{k: str(v) for k, v in vars(args).items()}, "kv": "f16", "temperature": 0},
              "answers": len(results), "seconds": round(took), "tokens": sum(r["tokens"] for r in results),
              "stopped_at_cap": sum(r["finish"] == "length" for r in results)}
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in [header, *results]), encoding="utf-8")
    print(json.dumps({**header, "tok_s": round(header["tokens"] / took), "file": str(out)}))


if __name__ == "__main__":
    main()
