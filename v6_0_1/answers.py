"""Write answers with llama.cpp on both cards, in one of two ways:

  plain  the user's prompt alone: how the model answers by itself, and how the trained model is tested;
  told   the same prompt with the topic's rules (behaviors.guide) in a system message: the teacher's answers.

Settings are the locked baseline's, on purpose, so every number in v6.0.1 compares with the 3.43 baseline:
the model's recommended sampling with every sampler written out, thinking off, a q8_0 KV cache, 2,048 new tokens
at most, 13 parallel answers on the display card and 16 on the other. The server start, the streaming request and
the loop stop rule are reused from reports/answer_v5.py, not copied.

Other stages call record(). Every answer is appended the moment it ends; a rerun skips what is already written.
"""

from __future__ import annotations

import json
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from reports.answer_v5 import CARDS, answer, start  # the two cards with their ports, one streamed answer, server start
from v6.train import recommended_sampling          # reads the model's own generation_config.json (critical rule)
from v6_0_1.behaviors import BEHAVIORS, guide
from v6_0_1.files import append_row, read_rows

SLOTS = (13, 16)   # parallel answers on Vulkan0 (it also drives the display) and on Vulkan1
SLOT_CTX = 3072    # tokens one answer may use in total: prompt + rules + answer
KV = "q8_0"        # KV cache type, the same as the baseline run
MAX_NEW = 2048     # answer tokens at most (the baseline's cap)


def job_seed(seed: int, row_id: str, variant: str, copy: int) -> int:
    """A fixed sampling seed for one answer, so a rerun writes the same text. crc32 works for any id format."""
    return seed * 100003 + zlib.crc32(f"{row_id}|{variant}|{copy}".encode())


def record(rows: list[dict], plan: dict[str, int], model: Path, out: Path, seed: int = 6) -> int:
    """Answer every prompt in `rows` as the plan says and append the answers to `out`. Returns how many failed.

    plan maps a variant to a number of copies, e.g. {"plain": 1, "told": 2} = one plain answer and two told answers
    per prompt. A topic with no rules (none, math_solving) cannot be told, so it only gets its plain answers.
    """
    done = {(r["id"], r["variant"], r["copy"]) for r in read_rows(out)}
    todo = [(row, variant, copy) for row in rows for variant, copies in plan.items()
            if variant == "plain" or row["topic"] in BEHAVIORS
            for copy in range(copies) if (row["id"], variant, copy) not in done]
    if not todo:
        return 0
    sampling = json.dumps(recommended_sampling())  # answer() takes the sampler settings as a JSON string
    ports = [port for (_, port), n in zip(CARDS, SLOTS) for _ in range(n)]  # one entry per slot: jobs spread by slot count
    lock, failed, t0, count = threading.Lock(), [], time.time(), [0]

    def work(numbered: tuple) -> None:
        n, (row, variant, copy) = numbered
        ask = {"id": row["id"], "source": row["topic"], "prompt": row["prompt"]}  # the row shape answer() expects
        rules = guide(row["topic"]) if variant == "told" else ""
        try:
            got = answer(ask, ports[n % len(ports)], MAX_NEW, sampling, job_seed(seed, row["id"], variant, copy), "off", rules)
        except Exception as error:  # noqa: BLE001 - one failed answer must not stop a long run: report it, a rerun retries it
            with lock:
                failed.append((row["id"], variant, copy))
                print(f"answer failed {row['id']} {variant} {copy}: {type(error).__name__}: {error}", flush=True)
            return
        with lock:  # one writer at a time keeps the lines of the file whole
            append_row(out, {"id": row["id"], "topic": row["topic"], "variant": variant, "copy": copy,
                             "prompt": row["prompt"], "answer": got["answer"], "tokens": got["tokens"] or 0,
                             "finish": got["finish"]})
            count[0] += 1
            if count[0] % 25 == 0 or count[0] == len(todo):
                print(f"answers {count[0]}/{len(todo)}, {round(time.time() - t0)} s", flush=True)

    servers = []
    try:
        for (card, port), n in zip(CARDS, SLOTS):  # one llama-server per card, each with its own separate slots
            servers.append(start(model, card, port, SLOT_CTX, n, KV))
        with ThreadPoolExecutor(len(ports)) as pool:  # one thread per slot: every slot always has an answer to write
            list(pool.map(work, enumerate(todo)))
    finally:
        for s in servers:
            s.terminate()
        for s in servers:
            s.wait()  # wait until the VRAM is really free before the next stage starts its own servers
    print(f"answers done: {len(todo) - len(failed)} written, {len(failed)} failed, {round(time.time() - t0)} s", flush=True)
    return len(failed)
