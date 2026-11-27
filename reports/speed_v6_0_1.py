"""Speed test: base Gemma 4 12B vs the shipped MoSV v6.0.1 GGUF, under the same controlled conditions.

Run (gguf-tools or ai-training venv, repo root, both cards free):
    python reports/speed_v6_0_1.py

Controls (the same for both models):
- the same llama.cpp binary, flags, context size and KV-cache type; one model loaded at a time
- two card setups, in this order: DUAL (Vulkan0 + Vulkan1, layer split, never tensor parallel), then SINGLE (Vulkan1)
- before each load, free VRAM is checked: the run stops if a card has less than the model needs plus a margin
- the same 10 hand-written prompts; Gemma's recommended sampling with EVERY sampler set explicitly
  (llama.cpp adds min_p 0.05 and others by default); fixed seed; thinking off; no prompt cache
- a warm-up request before measuring, which is not counted
- FIXED-LENGTH runs: exactly 512 new tokens per prompt (end-of-answer ignored), so both models do the same work.
  Order base, MoSV, MoSV, base (ABBA), so heat or background drift hits both models equally.
- NATURAL runs (first session of each model): each model stops where it wants (max 2048); answers recorded
- llama-bench: the standard prompt-processing (512 tokens) and generation (128 tokens) test, 5 repeats each

Writes into results/speed_v6_0_1/: speed.json (every request), README.md (tables), outputs.md (all natural
answers), *_server.log and *_bench.log (raw llama.cpp logs, as a reference).
"""

from __future__ import annotations

import json
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from demo_v6_0_1 import BASE, SERVER, SERVER_FLAGS, start  # same server setup as the demo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "speed_v6_0_1"
MOSV = Path("C:/llm/models/gemma4-12b/v6_0_1/MoSV-Gemma-4-12B-v6.0.1-Q4_K_M.gguf")
BENCH = Path(SERVER).with_name("llama-bench.exe")
PORT = 8311
MODELS = {"base": BASE, "mosv": MOSV}
SETUPS = [("dual", "Vulkan0,Vulkan1", "Vulkan0/Vulkan1"), ("single", "Vulkan1", "Vulkan1")]  # (name, server, bench)
VRAM_MARGIN_MIB = 3000  # KV cache (8192 ctx, q8_0) + compute buffers + room for the display
FIXED_TOKENS, NATURAL_MAX = 512, 2048
SAMPLING = {  # Gemma 4 recommended: temperature 1.0, top_k 64, top_p 0.95, min_p 0; every other sampler off
    "temperature": 1.0, "top_k": 64, "top_p": 0.95, "min_p": 0.0, "typical_p": 1.0, "repeat_penalty": 1.0,
    "presence_penalty": 0.0, "frequency_penalty": 0.0, "dry_multiplier": 0.0, "xtc_probability": 0.0,
    "top_n_sigma": -1.0, "mirostat": 0, "seed": 7}
PROMPTS = [  # (kind, prompt): written by hand, none from the datasets
    ("teaching", "Teach fractions (Mathematics) to a Primary 3 student."),
    ("teaching", "Teach the water cycle (Science) to a Primary 5 student."),
    ("debugging", "Bug report: 'After upgrading to version 2.4, the export button saves an empty CSV file on Windows, "
                  "but it works on Mac. No error is shown.' Where does it go wrong and how do I fix it?"),
    ("debugging", "Bug report: 'Users get logged out at random, roughly every 10 to 40 minutes, only on the mobile app. "
                  "The web app is fine.' What is likely going on?"),
    ("story", "Write a short story about a lighthouse keeper who finds a message in a bottle addressed to her."),
    ("story", "Write a short story about a boy who repairs old radios and hears a voice from the future."),
    ("everyday", "What's a good way to keep fresh basil from wilting for a week?"),
    ("explanation", "Explain why the sky is blue."),
    ("engineering", "Write a Python function that returns the n-th Fibonacci number without recursion."),
    ("email", "Write a short email to my landlord asking to fix a leaking kitchen tap this week."),
]


def free_vram() -> dict[str, int]:
    """Free MiB per card. llama.cpp's own "free" figure on Vulkan/Windows ignores other programs (it missed a model
    LM Studio had loaded), so the used amount comes from Windows' GPU counters. Which counter belongs to which
    Vulkan card is unknown, so every card is given the worst case: total minus the busiest card's usage."""
    out = subprocess.run([SERVER, "--list-devices"], capture_output=True, text=True, errors="replace")
    total = {d: int(t) for d, t in re.findall(r"(Vulkan\d+): .*?\((\d+) MiB,", out.stdout + out.stderr)}
    ps = ('(Get-Counter "\\GPU Adapter Memory(*)\\Dedicated Usage").CounterSamples | '
          'ForEach-Object { [int]($_.CookedValue / 1MB) }')
    used = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout.split()
    busiest = max((int(u) for u in used), default=0)
    return {d: t - busiest for d, t in total.items()}


def check_vram(model: Path, cards: str):
    """Stop before loading if the cards cannot hold the model (split evenly over the cards) plus the margin."""
    cards_list = cards.split(",")
    need = model.stat().st_size // 2**20 // len(cards_list) + VRAM_MARGIN_MIB
    free = free_vram()
    for c in cards_list:
        if free.get(c, 0) < need:
            raise SystemExit(f"{c} has {free.get(c, 0)} MiB free, needs about {need} MiB for {model.name}; stopped")
    print(f"VRAM ok: need ~{need} MiB per card, free {free}", flush=True)


def chat(prompt: str, max_tokens: int, fixed: bool) -> dict:
    """One chat request; returns llama.cpp's own timings, the wall time and the answer."""
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "cache_prompt": False,
            "ignore_eos": fixed, "chat_template_kwargs": {"enable_thinking": False}, **SAMPLING}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    out = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    wall = time.perf_counter() - t0
    t = out["timings"]
    return {"prompt_tokens": t["prompt_n"], "prompt_ms": t["prompt_ms"], "prompt_tps": t["prompt_per_second"],
            "gen_tokens": t["predicted_n"], "gen_ms": t["predicted_ms"], "gen_tps": t["predicted_per_second"],
            "wall_s": round(wall, 3), "text": out["choices"][0]["message"]["content"]}


def session(setup: str, cards: str, name: str, label: str, natural: bool) -> list[dict]:
    """Load one model, warm up, run the fixed-length pass (and the natural pass), then unload it."""
    check_vram(MODELS[name], cards)
    server = start(MODELS[name], cards, PORT, OUT / f"{setup}_{label}_server.log")
    rows = []
    try:
        chat("Say hello.", 32, True)  # warm-up, not counted
        passes = [("fixed", FIXED_TOKENS, True)] + ([("natural", NATURAL_MAX, False)] if natural else [])
        for mode, max_tokens, fixed in passes:
            for i, (kind, prompt) in enumerate(PROMPTS):
                r = chat(prompt, max_tokens, fixed)
                rows.append({"setup": setup, "model": name, "session": label, "mode": mode, "prompt_id": i,
                             "kind": kind, **r})
                print(f"{setup:6s} {label:6s} {mode:7s} #{i} {kind:11s} gen {r['gen_tokens']:4d} tok "
                      f"{r['gen_tps']:6.1f} tok/s | prompt {r['prompt_tps']:7.1f} tok/s | wall {r['wall_s']:6.1f}s",
                      flush=True)
    finally:
        server.terminate()
        server.wait()
    return rows


def bench(setup: str, devs: str, name: str) -> list[dict]:
    """llama-bench pp512 and tg128 with the same cards, split mode and flash attention; raw log kept."""
    check_vram(MODELS[name], devs.replace("/", ","))
    cmd = [str(BENCH), "-m", str(MODELS[name]), "-dev", devs, "-sm", "layer", "-ngl", "99", "-fa", "1",
           "-ctk", "q8_0", "-ctv", "q8_0", "-p", "512", "-n", "128", "-r", "5", "-o", "json"]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    log = OUT / f"{setup}_{name}_bench.log"
    log.write_text(" ".join(cmd) + "\n\n" + res.stderr + "\n" + res.stdout, encoding="utf-8")
    if res.returncode:
        raise SystemExit(f"llama-bench failed for {setup} {name}; see {log}")
    return [{"setup": setup, "model": name, "test": f"pp{r['n_prompt']}" if r["n_gen"] == 0 else f"tg{r['n_gen']}",
             "tps": r["avg_ts"], "sd": r["stddev_ts"]} for r in json.loads(res.stdout)]


def mean(rows, key):
    return statistics.mean(r[key] for r in rows)


def setup_tables(rows: list[dict], benches: list[dict], setup: str) -> list[str]:
    """Summary and per-prompt tables for one card setup."""
    pick = lambda m, mode: [r for r in rows if r["setup"] == setup and r["model"] == m and r["mode"] == mode]
    lines = [f"## {setup} card(s)", "", "### Fixed length: 512 tokens each, 2 sessions (same work for both)", "",
             "| Model | Requests | Generation tok/s | Prompt tok/s | Seconds per answer |", "|---|---|---|---|---|"]
    for m in MODELS:
        f = pick(m, "fixed")
        lines.append(f"| {m} | {len(f)} | {mean(f, 'gen_tps'):.2f} | {mean(f, 'prompt_tps'):.1f} | {mean(f, 'wall_s'):.2f} |")
    b, s = mean(pick("base", "fixed"), "gen_tps"), mean(pick("mosv", "fixed"), "gen_tps")
    lines += ["", f"MoSV generation speed vs base: **{100 * (s / b - 1):+.1f}%**.", "",
              "### Natural answers (each model stops where it wants)", "",
              "| Model | Avg answer tokens | Generation tok/s | Avg seconds per answer |", "|---|---|---|---|"]
    for m in MODELS:
        n = pick(m, "natural")
        lines.append(f"| {m} | {mean(n, 'gen_tokens'):.0f} | {mean(n, 'gen_tps'):.2f} | {mean(n, 'wall_s'):.2f} |")
    lines += ["", "### llama-bench (5 repeats)", "", "| Model | Test | tok/s | ± sd |", "|---|---|---|---|"]
    lines += [f"| {x['model']} | {x['test']} | {x['tps']:.2f} | {x['sd']:.2f} |" for x in benches if x["setup"] == setup]
    lines += ["", "### Per prompt, fixed length (gen tok/s, mean of 2 sessions)", "",
              "| # | Kind | Base | MoSV | Diff |", "|---|---|---|---|---|"]
    for i, (kind, _) in enumerate(PROMPTS):
        bi = mean([r for r in pick("base", "fixed") if r["prompt_id"] == i], "gen_tps")
        si = mean([r for r in pick("mosv", "fixed") if r["prompt_id"] == i], "gen_tps")
        lines.append(f"| {i} | {kind} | {bi:.2f} | {si:.2f} | {100 * (si / bi - 1):+.1f}% |")
    lines += ["", "### Per prompt, natural answers", "",
              "| # | Kind | Base tokens | MoSV tokens | Base s | MoSV s |", "|---|---|---|---|---|---|"]
    for i, (kind, _) in enumerate(PROMPTS):
        bn, sn = [next(r for r in pick(m, "natural") if r["prompt_id"] == i) for m in MODELS]
        lines.append(f"| {i} | {kind} | {bn['gen_tokens']} | {sn['gen_tokens']} | {bn['wall_s']:.1f} | {sn['wall_s']:.1f} |")
    return lines + [""]


def tables(rows: list[dict], benches: list[dict]) -> str:
    lines = ["# Speed: base Gemma 4 12B vs MoSV v6.0.1", "",
             f"Setups: dual (Vulkan0 + Vulkan1, layer split), then single (Vulkan1); RX 9070 XT. One model at a time. "
             f"llama-server flags `{' '.join(SERVER_FLAGS)}`. Sampling `{json.dumps(SAMPLING)}`. Thinking off, "
             f"no prompt cache, warm-up not counted. Fixed runs: {FIXED_TOKENS} new tokens each (end ignored), "
             "order base, MoSV, MoSV, base. Raw logs: `*_server.log`, `*_bench.log`. Every request: `speed.json`. "
             "Answers: `outputs.md`.", ""]
    natural = [r for r in rows if r["mode"] == "natural"]
    cut = sum(r["gen_tokens"] >= NATURAL_MAX for r in natural)
    lines += [f"Natural answers cut at the {NATURAL_MAX}-token limit: **{cut} of {len(natural)}** "
              f"(longest: {max(r['gen_tokens'] for r in natural)} tokens); every other answer ended on its own.", ""]
    for setup, _, _ in SETUPS:
        lines += setup_tables(rows, benches, setup)
    return "\n".join(lines)


def outputs(rows: list[dict]) -> str:
    """Every natural answer, base and MoSV side by side, for both setups."""
    lines = ["# Natural answers recorded during the speed test", ""]
    for setup, _, _ in SETUPS:
        for i, (kind, prompt) in enumerate(PROMPTS):
            lines += [f"## {setup}: {i}. {kind}", "", f"**Prompt:** {prompt}", ""]
            for m in MODELS:
                r = next(r for r in rows if r["setup"] == setup and r["model"] == m
                         and r["mode"] == "natural" and r["prompt_id"] == i)
                lines += [f"### {m} ({r['gen_tokens']} tokens, {r['wall_s']:.1f} s)", "", r["text"], ""]
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, benches = [], []
    for setup, cards, devs in SETUPS:
        rows += session(setup, cards, "base", "base_1", True) + session(setup, cards, "mosv", "mosv_1", True)
        rows += session(setup, cards, "mosv", "mosv_2", False) + session(setup, cards, "base", "base_2", False)
        benches += bench(setup, devs, "base") + bench(setup, devs, "mosv")
        # saved after each setup, so a crash in the second keeps the first
        info = {"server": SERVER, "flags": SERVER_FLAGS, "sampling": SAMPLING, "setups": SETUPS,
                "fixed_tokens": FIXED_TOKENS, "natural_max": NATURAL_MAX, "prompts": PROMPTS,
                "models": {m: {"path": str(p), "bytes": p.stat().st_size} for m, p in MODELS.items()},
                "date": time.strftime("%Y-%m-%d %H:%M:%S")}
        (OUT / "speed.json").write_text(json.dumps({"setup": info, "requests": rows, "bench": benches}, indent=1),
                                        encoding="utf-8")
    (OUT / "README.md").write_text(tables(rows, benches), encoding="utf-8")
    (OUT / "outputs.md").write_text(outputs(rows), encoding="utf-8")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
