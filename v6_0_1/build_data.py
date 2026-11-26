"""Build the training set: tell the model the rules, check every answer, keep only the answers that pass.

For each training prompt of a topic the gate marked "train":
  1. the model writes one plain answer and --copies told answers (answers.py, both cards);
  2. Haiku checks every one of them (checks.py);
  3. choose() keeps the best told answer ONLY if it is complete, passes every guard, passes at least KEEP_MIN of
     the behavior checks, and passes more of them than the plain answer to the same prompt.
The kept (plain prompt -> told answer) pairs are what train.py teaches. This is what makes the learning signal
strong: v6 hoped that one of four random answers would happen to be better; here the better answer is produced on
purpose and then verified. A prompt that fails a rule goes to rejected.jsonl with the reason, so what the filter
threw away can be read.

The stage also records plain answers for prompts the router must learn to leave alone: everyday prompts, and every
v6 topic that is not being trained (math_solving, plus any topic the gate skipped).

Run (ai-training-rocm-windows venv, repo root, both cards free; about 1.5 hours for 300 prompts of 3-4 topics,
estimated from 0.6 answers per second):
    python -m v6_0_1.build_data --per-topic 300
Reads results/v6_0_1_gate.json. Writes data_review/v6_0_1/{train,router,rejected}.jsonl and results/v6_0_1_data.json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from v6.train import BASE_GGUF
from v6.train import TOPICS as V6_TOPICS  # all seven v6 topics, including the ones v6.0.1 does not train
from v6_0_1 import answers, checks
from v6_0_1.behaviors import fingerprint
from v6_0_1.files import DATA, RESULTS, claim, prompts, read_rows

ANSWERS = DATA / "build" / "answers.jsonl"  # every plain and told answer written by this stage
CHECKS = DATA / "build" / "checks.jsonl"    # their yes/no verdicts
TRAIN = DATA / "train.jsonl"                # the kept pairs: what train.py teaches
ROUTER = DATA / "router.jsonl"              # labeled texts for fitting the router
REJECTED = DATA / "rejected.jsonl"          # dropped prompts with the reason, for reading
GATE = RESULTS / "v6_0_1_gate.json"
REPORT = RESULTS / "v6_0_1_data.json"

KEEP_MIN = 0.75  # a kept answer passes at least 3 of its 4 behavior checks


def choose(plain_behavior: float, told: list[dict]) -> tuple[dict | None, str]:
    """Pick the told answer to keep for one prompt, or (None, why it was dropped).

    Each told item is its answer row plus "behavior" and "guard" (the shares from checks.shares).
    """
    complete = [t for t in told if t["finish"] == "stop"]  # a cut-off answer would teach the model to stop mid-way
    if not complete:
        return None, "cut off"
    safe = [t for t in complete if t["guard"] == 1.0]      # never teach an answer that failed a correctness guard
    if not safe:
        return None, "guard failed"
    best = max(safe, key=lambda t: (t["behavior"], -t["tokens"]))  # most checks passed; on a tie, the shorter answer
    if best["behavior"] < KEEP_MIN:
        return None, "too few checks passed"
    if best["behavior"] <= plain_behavior:                 # the plain answer was already as good: nothing to teach
        return None, "not better than plain"
    return best, "kept"


def write_rows(path: Path, rows: list[dict]) -> None:
    """Write a whole JSON-lines file at once (the outputs of this stage are rebuilt from scratch on every run)."""
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--per-topic", type=int, default=300, help="training prompts per trained topic")
    p.add_argument("--copies", type=int, default=2, help="told answers per prompt (the best one that passes is kept)")
    p.add_argument("--router-per-class", type=int, default=150, help="prompts per class the router must leave alone")
    p.add_argument("--fresh", action="store_true", help="delete the cached build answers first (needed after editing behaviors.py)")
    args = p.parse_args(argv)

    claim(ANSWERS.parent, fingerprint(), args.fresh)  # refuses to reuse answers written under older rules
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    trained = [topic for topic, row in gate["topics"].items() if row["train"]]
    if not trained:
        raise SystemExit(f"{GATE} marks no topic for training: nothing to build")
    others = ["none", *[t for t in V6_TOPICS if t not in trained]]  # the router sends all of these to "none"
    teach = [row for t in trained for row in prompts(t, "train", args.per_topic)]
    leave = [row for t in others for row in prompts(t, "train", args.router_per_class)]

    failed = answers.record(teach, {"plain": 1, "told": args.copies}, BASE_GGUF, ANSWERS)
    failed += answers.record(leave, {"plain": 1}, BASE_GGUF, ANSWERS)  # plain only: nothing is taught for these

    wanted, left = {r["id"] for r in teach}, {r["id"] for r in leave}
    written = read_rows(ANSWERS)
    verdicts = {checks.key(r): r["pass"] for r in checks.run([a for a in written if a["id"] in wanted], CHECKS)}
    plain, told = {}, defaultdict(list)  # per prompt id: its checked plain answer, and its checked told answers
    for a in written:
        if a["id"] in wanted and checks.key(a) in verdicts:
            behavior, guard = checks.shares(verdicts[checks.key(a)])
            scored = {**a, "behavior": behavior, "guard": guard}
            if a["variant"] == "plain":
                plain[a["id"]] = scored
            else:
                told[a["id"]].append(scored)

    kept, rejected, reasons = [], [], defaultdict(Counter)
    for row in teach:
        if row["id"] not in plain or not told[row["id"]]:  # an answer or a check failed: reported above, rerun to retry
            reasons[row["topic"]]["not answered or not checked"] += 1
            continue
        base = plain[row["id"]]
        best, reason = choose(base["behavior"], told[row["id"]])
        reasons[row["topic"]][reason] += 1
        if best is None:
            rejected.append({"id": row["id"], "topic": row["topic"], "reason": reason, "plain_behavior": base["behavior"],
                             "told": [{k: t[k] for k in ("behavior", "guard", "finish", "tokens", "answer")} for t in told[row["id"]]]})
        else:
            kept.append({"id": row["id"], "topic": row["topic"], "prompt": row["prompt"], "answer": best["answer"],
                         "behavior": best["behavior"], "base_behavior": base["behavior"],
                         "tokens": best["tokens"], "base_tokens": base["tokens"]})

    # Router texts. "none" = everything to leave alone. A trained topic gets both its plain answers and its kept
    # told answers, because after training the model's own words will look like the told ones.
    router = [{"label": "none", "prompt": a["prompt"], "answer": a["answer"]} for a in written
              if a["id"] in left and a["variant"] == "plain"]
    router += [{"label": a["topic"], "prompt": a["prompt"], "answer": a["answer"]} for a in plain.values()]
    router += [{"label": k["topic"], "prompt": k["prompt"], "answer": k["answer"]} for k in kept]
    write_rows(TRAIN, kept)
    write_rows(ROUTER, router)
    write_rows(REJECTED, rejected)

    report = {"settings": {**vars(args), "keep_min": KEEP_MIN, "failed_answers": failed}, "trained": trained,
              "router_texts": dict(Counter(r["label"] for r in router)), "topics": {}}
    for topic in trained:
        mine = [k for k in kept if k["topic"] == topic]
        base = [a for a in plain.values() if a["topic"] == topic]
        report["topics"][topic] = {
            "prompts": sum(r["topic"] == topic for r in teach), "kept": len(mine), "outcomes": dict(reasons[topic]),
            "plain_behavior": mean([a["behavior"] for a in base]), "kept_behavior": mean([k["behavior"] for k in mine]),
            "plain_tokens": round(mean([a["tokens"] for a in base])), "kept_tokens": round(mean([k["tokens"] for k in mine]))}
        print(topic, report["topics"][topic], flush=True)
    REPORT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"kept {len(kept)} of {len(teach)} prompts -> {TRAIN}; rejected {len(rejected)} -> {REJECTED}", flush=True)


if __name__ == "__main__":
    main()
