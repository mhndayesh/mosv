"""Test a trained checkpoint: plain prompts answered by the built GGUF, behavior checks, compared with the base model.

This is the real result of a run. No rules are given at test time: the question is whether the model now behaves
as told WITHOUT being told. For every topic it reports, base model vs trained model on the same prompts:
  behavior  share of the behavior checks passed (what was taught);
  guard     share of the correctness guards passed (must not drop);
  tokens    mean answer length;
  wins      how often the trained answer passes more behavior checks than the base answer (a tie counts half).

Run (ai-training-rocm-windows venv, repo root, both cards free):
    python -m v6_0_1.evaluate --run gemma_v601 --checkpoint steer_0100.pt --split dev
    python -m v6_0_1.evaluate --run gemma_v601 --checkpoint steer_0100.pt --split test      (the locked 300, once)
Pick the checkpoint on dev; run the locked test only for the one that was picked.
Writes results/v6_0_1_eval_<split>_<run>_<checkpoint>.json. Base answers of a split are recorded once and reused.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from v6.train import BASE_GGUF, GGUF_PY
from v6.train import TOPICS as V6_TOPICS
from v6_0_1 import answers, checks
from v6_0_1.behaviors import BEHAVIORS, fingerprint
from v6_0_1.files import DATA, GGUF_DIR, RESULTS, RUNS, claim, prompts, read_rows

EVAL = DATA / "eval"  # answers and verdicts of every evaluation


def build_gguf(checkpoint: Path, out: Path) -> list[str]:
    """Write the shippable model: the base GGUF plus this checkpoint's router and experts. Returns its classes.

    Runs on the CPU. Base tensors are copied unchanged, so the file works in stock llama.cpp and LM Studio.
    """
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "-1")  # keep PyTorch off the cards: llama.cpp needs their memory
    sys.path.insert(0, str(GGUF_PY))
    import torch

    from moe import gemma_host
    from v6.steer import GemmaSteer

    saved = torch.load(checkpoint, weights_only=False)["steer"]
    layers = tuple(saved["layers"])
    width = saved["gate"][layers[0]].shape[1]  # the model's hidden size, read from the router matrix
    steer = GemmaSteer(saved["classes"], width, layers, "cpu", ff=saved["ff"])
    steer.load(saved)
    out.parent.mkdir(parents=True, exist_ok=True)
    gemma_host.build(BASE_GGUF, out, steer.gguf_layers(), saved["classes"], steer.ff)
    return saved["classes"]


def summary(answer_rows: list[dict], verdicts: list[dict]) -> dict:
    """Per topic: mean behavior, guard and tokens, each question's pass rate, and the per-prompt behavior share."""
    tokens = {r["id"]: r["tokens"] for r in answer_rows}
    out = {}
    for topic in BEHAVIORS:
        rows = [v for v in verdicts if v["topic"] == topic]
        if not rows:
            continue
        both = {v["id"]: checks.shares(v["pass"]) for v in rows}
        out[topic] = {"n": len(rows),
                      "behavior": round(sum(b for b, _ in both.values()) / len(rows), 3),
                      "guard": round(sum(g for _, g in both.values()) / len(rows), 3),
                      "tokens": round(sum(tokens[v["id"]] for v in rows) / len(rows)),
                      "checks": {name: round(sum(v["pass"][name] for v in rows) / len(rows), 3) for name in rows[0]["pass"]},
                      "by_prompt": {i: b for i, (b, _) in both.items()}}
    return out


def compare(base: dict, new: dict) -> dict:
    """The report: per topic the base and trained numbers side by side, plus how often the trained answer wins."""
    report = {}
    for topic in new:
        if topic not in base:
            continue
        shared = [i for i in new[topic]["by_prompt"] if i in base[topic]["by_prompt"]]
        wins = sum((new[topic]["by_prompt"][i] > base[topic]["by_prompt"][i])
                   + 0.5 * (new[topic]["by_prompt"][i] == base[topic]["by_prompt"][i]) for i in shared)
        keep = lambda row: {k: v for k, v in row.items() if k != "by_prompt"}  # noqa: E731 - per-prompt detail stays out of the report
        report[topic] = {"base": keep(base[topic]), "trained": keep(new[topic]),
                         "prompts": len(shared), "trained_wins_pct": round(100 * wins / max(len(shared), 1), 1)}
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True)
    p.add_argument("--checkpoint", default="steer.pt", help="a file inside the run folder, e.g. steer_0100.pt")
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--per-topic", type=int, default=60, help="dev prompts per topic: enough that one answer cannot flip the result "
                   "(the locked test always uses all of its prompts)")
    p.add_argument("--fresh", action="store_true", help="delete every cached evaluation answer and verdict first (needed after editing behaviors.py)")
    args = p.parse_args(argv)

    claim(EVAL, fingerprint(), args.fresh)  # verdicts written under older questions must not be mixed with new ones

    count = 10**6 if args.split == "test" else args.per_topic
    topics = V6_TOPICS if args.split == "test" else tuple(BEHAVIORS)  # the locked test keeps its math prompts
    rows = [row for t in topics for row in prompts(t, args.split, count)]
    tag = f"{args.split}_{args.run}_{Path(args.checkpoint).stem}"
    gguf = GGUF_DIR / f"{args.run}_{Path(args.checkpoint).stem}.gguf"
    classes = build_gguf(RUNS / args.run / args.checkpoint, gguf)

    plain = {"plain": 1}  # plain prompts only: no rules are given at test time
    base_file, new_file = EVAL / f"{args.split}_base.jsonl", EVAL / f"{tag}.jsonl"
    failed = answers.record(rows, plain, BASE_GGUF, base_file) + answers.record(rows, plain, gguf, new_file)
    wanted = {r["id"] for r in rows}
    base_answers = [a for a in read_rows(base_file) if a["id"] in wanted]
    new_answers = [a for a in read_rows(new_file) if a["id"] in wanted]
    base = summary(base_answers, checks.run(base_answers, EVAL / f"{args.split}_base_checks.jsonl"))
    new = summary(new_answers, checks.run(new_answers, EVAL / f"{tag}_checks.jsonl"))

    report = {"run": args.run, "checkpoint": args.checkpoint, "split": args.split, "classes": classes, "gguf": str(gguf),
              "failed_answers": failed, "topics": compare(base, new)}
    for topic, row in report["topics"].items():
        note = "" if topic in classes else " (not trained: routed to none)"
        print(f"{topic:19s} behavior {row['base']['behavior']:.2f} -> {row['trained']['behavior']:.2f} | "
              f"guard {row['base']['guard']:.2f} -> {row['trained']['guard']:.2f} | "
              f"tokens {row['base']['tokens']} -> {row['trained']['tokens']} | trained wins {row['trained_wins_pct']}%{note}", flush=True)
    out = RESULTS / f"v6_0_1_eval_{tag}.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
