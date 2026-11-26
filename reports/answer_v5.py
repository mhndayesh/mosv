"""Answer a locked v5 test split with a GGUF model on both cards (one llama-server per card) and record everything.

Run (Windows, gguf-tools venv):
    python reports/answer_v5.py --topic engineering --model C:/llm/models/bonsai-4b/Bonsai-4B-Q1_0.gguf --tag bonsai_normal
Greedy, f16 KV cache, separate slots: each parallel answer owns --slot-ctx tokens (prompt + answer), so attention
reads only its own text. (A shared --kv-unified pool was tried: every answer then scans the whole pool, measured
~4 tok/s per answer at 80 parallel, so it is off.) Stop rules, identical for every model:
  - the model's own stop token (finish "stop");
  - a loop: the last LOOP_WINDOW characters already appear LOOP_REPEATS times earlier (finish "loop", text kept);
  - --max-new tokens per answer, 2,048 by default, or the slot filling up (finish "length").
Each answer is appended to data_review/v5_runs/<topic>/<tag>.jsonl the moment it ends (a settings header first),
so a stop never loses finished work; a rerun skips ids already in the file. Progress prints every 10 answers.
"""

import argparse
import json
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = "C:/llm/tq-vulkan/build/bin/llama-server.exe"
CARDS = (("Vulkan0", 8101), ("Vulkan1", 8102))
LOOP_WINDOW, LOOP_REPEATS, LOOP_CHECK_EVERY = 400, 4, 200  # chars, earlier copies, chars between checks


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--topic", required=True, help="a v5 topic (e.g. engineering), or v6 = the locked 7-topic test")
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--tag", required=True, help="output name, e.g. bonsai_normal")
    p.add_argument("--max-new", type=int, default=2048, help="tokens per answer at most")
    p.add_argument("--slot-ctx", type=int, default=3072, help="tokens per slot: prompt (up to ~1,000) + answer")
    p.add_argument("--slots", default="23,32", help="parallel answers per card, card0,card1 (~144 KB per token: "
                   "23 x 3,072 fits card0 next to the display, 32 x 3,072 fits card1)")
    p.add_argument("--sampling", default="greedy", help='greedy, or JSON like {"temperature":1.0,"top_k":64,"top_p":0.95}'
                   " (the model's own recommended settings, recorded in the file header)")
    p.add_argument("--seed", type=int, default=0, help="per-answer seed base, so sampled runs repeat exactly")
    p.add_argument("--thinking", choices=["off", "on", "template"], default="off",
                   help="model thinking mode via chat_template_kwargs.enable_thinking (template = leave as the template decides)")
    p.add_argument("--kv", default="f16", help="llama.cpp KV cache type for both K and V (f16, q8_0, q4_0)")
    p.add_argument("--guide", choices=["none", "rubric"], default="none",
                   help="rubric = tell the model its topic's behavior rules in a system message (the ceiling test: "
                        "how good is the model when told what to do). The saved prompt stays the plain one.")
    return p.parse_args(argv)


def guide_text(topic: str) -> str:
    """The topic's rubric as instructions to the model (never shown to the judge as part of the prompt)."""
    import sys
    sys.path.insert(0, str(ROOT))
    from rl.judge_rubrics import RUBRICS
    rules = RUBRICS[topic].replace("Score lower if:", "Avoid:")
    return "Follow these rules when you write your answer. Do not mention the rules.\n" + rules


def locked_v6() -> list[dict]:
    lock = json.loads((ROOT / "results" / "locked_v6.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (ROOT / "data_review" / "v6" / "test.jsonl").open(encoding="utf-8")]
    if [[r["id"], r["behavior"]] for r in rows] != lock["ids"]:
        raise SystemExit("data_review/v6/test.jsonl does not match results/locked_v6.json")
    return [{**r, "source": r["behavior"]} for r in rows]


V6_TOPICS = ("story", "teaching", "debugging", "math_solving", "explanation", "professional_email", "engineering")


def v6_dev(per_topic: int = 20) -> list[dict]:
    """First `per_topic` dev prompts of each v6 topic: for choosing a checkpoint, never the locked test."""
    rows = []
    for t in V6_TOPICS:
        path = ROOT / "data_review" / "v6" / t / "dev.jsonl"
        rows += [{**json.loads(l), "source": t} for l in path.open(encoding="utf-8")][:per_topic]
    return rows


def locked_prompts(topic: str) -> list[dict]:
    if topic == "v6":
        return locked_v6()
    if topic == "v6dev":
        return v6_dev()
    ids = set(json.loads((ROOT / "results" / "locked_v5" / f"{topic}.json").read_text(encoding="utf-8"))["test_ids"])
    path = ROOT / "data_review" / "clean_v5" / topic / "test.jsonl"
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    if {r["id"] for r in rows} != ids:
        raise SystemExit(f"{path} does not match its lock")
    return rows


def per_card(value: str) -> list[int]:
    return [int(v) for v in value.split(",")]


def start(model: Path, card: str, port: int, slot_ctx: int, slots: int, kv: str = "f16") -> subprocess.Popen:
    cmd = [SERVER, "-m", str(model), "-dev", card, "-ngl", "99", "--port", str(port), "-fa", "on",
           "-c", str(slot_ctx * slots), "-np", str(slots), "-b", "4096", "-ub", "1024", "-ctk", kv, "-ctv", kv]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(180):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return proc
        except Exception:
            time.sleep(1)
    proc.terminate()
    raise SystemExit(f"server on {card} did not start")


def is_loop(text: str) -> bool:
    tail = text[-LOOP_WINDOW:]
    return len(text) > LOOP_WINDOW * (LOOP_REPEATS + 1) and text[:-LOOP_WINDOW].count(tail) >= LOOP_REPEATS


def sampling_fields(sampling: str, seed: int) -> dict:
    if sampling == "greedy":
        return {"temperature": 0}
    return {**json.loads(sampling), "seed": seed}


def answer(row: dict, port: int, max_new: int, sampling: str = "greedy", seed: int = 0, thinking: str = "off",
           guide: str = "") -> dict:
    """Stream one answer; closing the stream on a loop makes llama-server stop that slot."""
    system = [{"role": "system", "content": guide}] if guide else []
    request = {"messages": [*system, {"role": "user", "content": row["prompt"]}], **sampling_fields(sampling, seed),
               "max_tokens": max_new, "stream": True, "stream_options": {"include_usage": True}}
    if thinking != "template":
        request["chat_template_kwargs"] = {"enable_thinking": thinking == "on"}
    body = json.dumps(request).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type": "application/json"})
    t, parts, size, checked, finish, tokens, thought = time.time(), [], 0, 0, None, None, []
    with urllib.request.urlopen(req, timeout=7200) as stream:
        for line in stream:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                tokens = chunk["usage"]["completion_tokens"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                thought.append(delta.get("reasoning_content") or "")
                text = delta.get("content") or ""
                parts.append(text)
                size += len(text)
                finish = choice.get("finish_reason") or finish
            if size - checked >= LOOP_CHECK_EVERY:
                checked = size
                if is_loop("".join(parts)):
                    finish = "loop"
                    break
    text = "".join(parts)
    return {"id": row["id"], "source": row["source"], "prompt": row["prompt"], "answer": text,
            "tokens": tokens, "chars": len(text), "finish": finish, "seconds": round(time.time() - t, 2),
            "thinking": "".join(thought)}


def main(argv=None):
    args = parse_args(argv)
    rows = locked_prompts(args.topic)
    out = ROOT / "data_review" / "v5_runs" / ("v6" if args.topic == "v6dev" else args.topic) / f"{args.tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l).get("id") for l in out.open(encoding="utf-8")} - {None}
    else:
        settings = {**{k: str(v) for k, v in vars(args).items()}, "kv": f"{args.kv}, separate slots",
                    "loop_rule": f"last {LOOP_WINDOW} chars seen {LOOP_REPEATS}x earlier"}
        out.write_text(json.dumps({"settings": settings}) + "\n", encoding="utf-8")
    todo = [r for r in rows if r["id"] not in done]
    lock, count, t = threading.Lock(), [len(done)], time.time()

    slots = per_card(args.slots)
    ports = [port for (_, port), n in zip(CARDS, slots) for _ in range(n)]  # jobs spread in proportion to slots

    def run(i_row):
        i, row = i_row
        seed = args.seed * 100003 + int(row["id"][:8], 16) % 100003  # fixed per prompt, so runs repeat
        guide = guide_text(row["source"]) if args.guide == "rubric" else ""
        result = answer(row, ports[i % len(ports)], args.max_new, args.sampling, seed, args.thinking, guide)
        with lock:
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            count[0] += 1
            if count[0] % 10 == 0 or count[0] == len(rows):
                print(f"{count[0]}/{len(rows)} done, {round(time.time() - t)} s", flush=True)

    servers = [start(args.model, card, port, args.slot_ctx, n, args.kv) for (card, port), n in zip(CARDS, slots)]
    try:
        with ThreadPoolExecutor(sum(slots)) as pool:
            list(pool.map(run, enumerate(todo)))
    finally:
        for s in servers:
            s.terminate()
    results = [json.loads(l) for l in out.open(encoding="utf-8")][1:]
    finishes = {f: sum(r["finish"] == f for r in results) for f in ("stop", "loop", "length")}
    print(json.dumps({"answers": len(results), "seconds": round(time.time() - t), "finish": finishes, "file": str(out)}))


if __name__ == "__main__":
    main()
