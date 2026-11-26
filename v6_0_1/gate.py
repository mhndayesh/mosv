"""The headroom gate: measure, BEFORE any training, whether each topic's behavior can be taught at all.

v6 trained for hours on a target with no room: on the 1-5 overall score Gemma got 3.38, Gemma told the rubric 3.46
and Haiku itself 3.71. This stage asks the three questions that would have shown that in half an hour:

  1. Told gain: does the model follow the behavior when it is TOLD to? (told pass rate minus plain pass rate)
     Experts can only move the model toward what it can already do when asked; they add no ability.
  2. Guard: does telling it leave correctness intact? (guard pass rate, told vs plain)
  3. Repeat: does the checker give the same yes/no twice on the same answer? A reward that does not repeat is noise.

It also records what a strong model (Haiku) does with the plain prompt, as a reference column: if Haiku shows a
behavior less often than plain Gemma does, the behavior may not be worth wanting.

Run (ai-training-rocm-windows venv, repo root, both cards free; about 25 minutes for 60 prompts per topic):
    python -m v6_0_1.gate
Writes results/v6_0_1_gate.json. build_data.py and train.py use only the topics marked "train" there.
Uses dev prompts only: never the locked test, never the training prompts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict

from rl.api_judge import MODEL, load_api_key
from v6.train import BASE_GGUF
from v6_0_1 import answers, checks, variety
from v6_0_1.behaviors import BEHAVIORS, fingerprint
from v6_0_1.files import DATA, RESULTS, append_row, claim, prompts, read_rows

ANSWERS = DATA / "gate" / "answers.jsonl"  # plain, told and reference answers to the dev prompts
CHECKS = DATA / "gate" / "checks.jsonl"    # the yes/no verdicts, two rounds
REPORT = RESULTS / "v6_0_1_gate.json"

# Thresholds. They are judgment calls: each comment says what the number protects against.
# They only mean something with enough prompts. With 20 prompts per topic a single answer moves the guard by 5
# points, and the plain score of the same 20 prompts moved by up to 13 points between two runs (first two gate
# runs, 2026-11-26). Hence the default of 60 prompts per topic: one answer is then 1.7 points.
MIN_TOLD_GAIN = 0.15   # told must pass at least 15 points more of the behavior checks than plain: something to teach
MAX_GUARD_DROP = 0.10  # told may lose at most 10 points on the correctness guard: telling must not break the answer
MIN_REPEAT = 0.85      # two rounds must agree on at least 85% of the yes/no answers: the reward must repeat
DEFAULT_PROMPTS = 60   # dev prompts per topic


def verdict(plain: float, told: float, plain_guard: float, told_guard: float, repeat: float) -> tuple[bool, str]:
    """(train?, reason) for one topic, from its behavior pass rates, guard pass rates and checker repeat rate."""
    gain, drop = round(told - plain, 3), round(plain_guard - told_guard, 3)  # rounded, so 0.15 is not read as 0.1499
    if repeat < MIN_REPEAT:
        return False, f"skip: the checker repeats itself only {repeat:.0%} of the time (needs {MIN_REPEAT:.0%})"
    if gain < MIN_TOLD_GAIN:
        return False, f"skip: telling the model adds only {gain:+.0%} (needs {MIN_TOLD_GAIN:+.0%})"
    if drop > MAX_GUARD_DROP:
        return False, f"skip: telling the model costs {drop:.0%} on correctness (limit {MAX_GUARD_DROP:.0%})"
    return True, f"train: told passes {told:.0%} of the checks, plain {plain:.0%}"


async def _reference(rows: list[dict], concurrency: int = 8) -> list[dict]:
    """Haiku answers each plain prompt once (same 2,048-token cap, thinking off, like the local model)."""
    import anthropic
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=load_api_key(), max_retries=checks.RETRIES, timeout=180)
    gate = asyncio.Semaphore(concurrency)

    async def one(row: dict) -> dict:
        try:
            async with gate:
                message = await client.messages.create(model=MODEL, max_tokens=answers.MAX_NEW, thinking={"type": "disabled"},
                                                       messages=[{"role": "user", "content": row["prompt"]}])
        except anthropic.APIError as error:  # status and network errors alike: report it, a rerun retries this prompt
            return {"error": f"{type(error).__name__}: {error}"}
        if message.stop_reason == "refusal":  # no answer to check
            return {"error": "refusal"}
        text = "".join(block.text for block in message.content if block.type == "text")
        return {"answer": text, "tokens": message.usage.output_tokens,
                "finish": "stop" if message.stop_reason == "end_turn" else "length"}

    try:
        return await asyncio.gather(*(one(row) for row in rows))
    finally:
        await client.close()


def reference_answers(rows: list[dict]) -> None:
    """Add the strong model's plain answers to ANSWERS as variant "reference" (skips prompts already answered)."""
    done = {(r["id"], r["variant"]) for r in read_rows(ANSWERS)}
    todo = [r for r in rows if (r["id"], "reference") not in done]
    for row, got in zip(todo, asyncio.run(_reference(todo)) if todo else []):
        if "error" in got:
            print(f"reference failed {row['id']}: {got['error']}", flush=True)
            continue
        append_row(ANSWERS, {"id": row["id"], "topic": row["topic"], "variant": "reference", "copy": 0,
                             "prompt": row["prompt"], **got})


def rates(answer_rows: list[dict], check_rows: list[dict]) -> dict:
    """{topic: {variant: {"n", "behavior", "guard", "tokens", "checks": {question: pass rate}}}} from round-1 verdicts."""
    tokens = {checks.key(a): a["tokens"] for a in answer_rows}
    groups = defaultdict(list)
    for row in check_rows:
        groups[(row["topic"], row["variant"])].append(row)
    out = defaultdict(dict)
    for (topic, variant), rows in sorted(groups.items()):
        both = [checks.shares(r["pass"]) for r in rows]
        out[topic][variant] = {
            "n": len(rows),
            "behavior": round(sum(b for b, _ in both) / len(rows), 3),
            "guard": round(sum(g for _, g in both) / len(rows), 3),
            "tokens": round(sum(tokens[checks.key(r)] for r in rows) / len(rows)),
            "checks": {name: round(sum(r["pass"][name] for r in rows) / len(rows), 3) for name in rows[0]["pass"]}}
    return out


def repeat_rate(first: list[dict], second: list[dict]) -> dict[str, float]:
    """Per topic: the share of yes/no answers that came out the same in both rounds on the same answer."""
    again = {checks.key(r): r["pass"] for r in second}
    same, total = defaultdict(int), defaultdict(int)
    for row in first:
        for name, value in again.get(checks.key(row), {}).items():
            same[row["topic"]] += value == row["pass"][name]
            total[row["topic"]] += 1
    return {topic: round(same[topic] / total[topic], 3) for topic in total}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--per-topic", type=int, default=DEFAULT_PROMPTS, help="dev prompts per topic")
    p.add_argument("--fresh", action="store_true", help="delete the cached gate answers first (needed after editing behaviors.py)")
    args = p.parse_args(argv)

    claim(ANSWERS.parent, fingerprint(), args.fresh)  # refuses to reuse answers written under older rules
    rows = [row for topic in BEHAVIORS for row in prompts(topic, "dev", args.per_topic)]
    failed = answers.record(rows, {"plain": 1, "told": 1}, BASE_GGUF, ANSWERS)  # both cards, about 5 minutes
    reference_answers(rows)
    written = read_rows(ANSWERS)
    first = checks.run(written, CHECKS, round_no=1)
    second = checks.run([a for a in written if a["variant"] != "reference"], CHECKS, round_no=2)  # the repeat test

    table, repeat = rates(written, first), repeat_rate(first, second)
    report = {"settings": {"per_topic": args.per_topic, "failed_answers": failed, "min_told_gain": MIN_TOLD_GAIN,
                           "max_guard_drop": MAX_GUARD_DROP, "min_repeat": MIN_REPEAT, "judge": MODEL},
              "topics": {}}
    for topic, by_variant in table.items():
        plain, told = by_variant["plain"], by_variant["told"]
        train, reason = verdict(plain["behavior"], told["behavior"], plain["guard"], told["guard"], repeat[topic])
        report["topics"][topic] = {**by_variant, "repeat": repeat[topic], "train": train, "reason": reason,
                                   "variety": {v: variety.report([a["answer"] for a in written if a["topic"] == topic and a["variant"] == v])
                                               for v in ("plain", "told")}}  # speech openings and repeated names
        reference = by_variant.get("reference", {}).get("behavior")
        print(f"{topic:19s} plain {plain['behavior']:.2f} | told {told['behavior']:.2f} | reference {reference} | "
              f"guard {plain['guard']:.2f} -> {told['guard']:.2f} | repeat {repeat[topic]:.2f} | "
              f"tokens {plain['tokens']} -> {told['tokens']} | {reason}", flush=True)
    REPORT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"saved {REPORT}", flush=True)


if __name__ == "__main__":
    main()
