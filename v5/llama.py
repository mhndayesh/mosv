"""llama-server pool for v5 training: one server per card on the live GGUF, restarted after each patch.

Prompts go in as token ids (the HF chat template, so PyTorch and llama.cpp see identical prompts) and the sampled
token ids come back, so the learning step scores exactly the tokens that were sampled.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

SERVER = "C:/llm/tq-vulkan/build/bin/llama-server.exe"


@dataclass
class Card:
    device: str  # llama.cpp name, e.g. "Vulkan0"
    port: int
    slots: int
    slot_ctx: int


class Pool:
    def __init__(self, model: Path, cards: list[Card], log_dir: Path):
        self.model, self.cards, self.log_dir, self.procs = model, cards, log_dir, []

    def start(self) -> float:
        t = time.time()
        self.procs = []
        for card in self.cards:
            cmd = [SERVER, "-m", str(self.model), "-dev", card.device, "-ngl", "99", "--port", str(card.port),
                   "-fa", "on", "-c", str(card.slots * card.slot_ctx), "-np", str(card.slots), "-b", "4096",
                   "-ub", "1024", "--no-webui"]
            log = (self.log_dir / f"llama_{card.device}.log").open("a", encoding="utf-8")
            self.procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))
        for card in self.cards:
            self._wait(card)
        return time.time() - t

    def _wait(self, card: Card) -> None:
        for _ in range(600):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{card.port}/health", timeout=1)
                return
            except Exception:
                time.sleep(0.1)
        raise RuntimeError(f"llama-server on {card.device} did not start; see {self.log_dir}")

    def stop(self) -> None:
        for p in self.procs:
            p.terminate()
        for p in self.procs:
            p.wait()
        self.procs = []

    def restart(self) -> float:
        self.stop()
        return self.start()

    def _complete(self, port: int, prompt: list[int], max_new: int, greedy: bool, seed: int) -> dict:
        body = {"prompt": prompt, "n_predict": max_new, "return_tokens": True, "cache_prompt": True, "seed": seed,
                "temperature": 0.0 if greedy else 1.0, "top_k": 1 if greedy else 0, "top_p": 1.0, "min_p": 0.0,
                "typical_p": 1.0, "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        out = json.loads(urllib.request.urlopen(req, timeout=1800).read())
        return {"tokens": out.get("tokens") or [], "stop": out.get("stop_type") or ""}

    def generate(self, prompts: list[list[int]], copies: int, max_new: int, greedy: bool = False,
                 seed: int = 0) -> list[list[dict]]:
        """copies answers per prompt; each answer is {"tokens": [...], "stop": "eos" | "limit" | ...}."""
        jobs = [(i, c) for i in range(len(prompts)) for c in range(copies)]
        ports = [card.port for card in self.cards for _ in range(card.slots)]
        out = [[None] * copies for _ in prompts]
        with ThreadPoolExecutor(max(1, min(len(jobs), len(ports)))) as pool:
            futures = {pool.submit(self._complete, ports[n % len(ports)], prompts[i], max_new, greedy,
                                   seed * 100003 + i * 101 + c): (i, c) for n, (i, c) in enumerate(jobs)}
            for future, (i, c) in futures.items():
                out[i][c] = future.result()
        return out
