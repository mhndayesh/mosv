"""Held-out check of the whole path: router + vectors against the oracle expert and the unsteered model.

Run: python evaluate.py --run pilot_final [--model PATH] [--limit 20] [--out report.json]
Cases: data_review/rl/<behavior>_test.jsonl plus `none` prompts (split `test`) from the router prompts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from behaviors.rubrics import score_response
from rl.data import BEHAVIORS, read_split
from system import MAX_NEW_TOKENS, NONE, load_system

ROOT = Path(__file__).parent
ROUTER_PROMPTS = ROOT / "data_review" / "router" / "prompts.jsonl"


def load_cases(limit: int | None) -> list[dict]:
    cases = [{"label": b, "prompt": row["prompt"]} for b in BEHAVIORS for row in read_split(b, "test")[:limit]]
    with ROUTER_PROMPTS.open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file]
    cases += [{"label": NONE, "prompt": r["text"]} for r in rows if r["split"] == "test" and r["label"] == NONE][:limit]
    return cases


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def mean_scores(label: str, cases: list[dict], texts: list[str]) -> float | None:
    return mean([score_response(label, t, c["prompt"]).score for c, t in zip(cases, texts)])


def evaluate_behavior(system, label: str, cases: list[dict], max_new: int) -> dict:
    prompts = [c["prompt"] for c in cases]
    routed = system.generate(prompts, max_new)
    oracle = system.generate_fixed(prompts, label, max_new)
    unsteered = system.generate_fixed(prompts, None, max_new)
    routes = [a.route for a in routed]
    return {
        "n": len(cases),
        "routed_score": mean_scores(label, cases, [a.text for a in routed]),
        "oracle_score": mean_scores(label, cases, oracle),
        "unsteered_score": mean_scores(label, cases, unsteered),
        "misses": sum(r == NONE for r in routes),
        "wrong_expert": sum(r not in (label, NONE) for r in routes),
    }


def evaluate_none(system, cases: list[dict]) -> dict:
    routes = system.route([c["prompt"] for c in cases])
    return {"n": len(cases), "false_steers": sum(r.name != NONE for r in routes)}


def summarize(per_behavior: dict, none_result: dict) -> dict:
    scored = [per_behavior[b] for b in BEHAVIORS if per_behavior[b]["n"]]
    total = sum(r["n"] for r in scored)

    def pooled(key):
        return round(sum(r[key] * r["n"] for r in scored) / total, 4) if total else None

    return {
        "routed_score": pooled("routed_score"), "oracle_score": pooled("oracle_score"),
        "unsteered_score": pooled("unsteered_score"),
        "misses": sum(r["misses"] for r in scored),
        "wrong_steers": sum(r["wrong_expert"] for r in scored) + none_result["false_steers"],
        "wrong_expert": sum(r["wrong_expert"] for r in scored),
        "false_steers_on_none": none_result["false_steers"],
        "behavior_prompts": total, "none_prompts": none_result["n"],
    }


def main(args) -> None:
    system = load_system(args.model, args.run, max_new_tokens=args.max_new_tokens)
    cases = load_cases(args.limit)
    by_label = {b: [c for c in cases if c["label"] == b] for b in (*BEHAVIORS, NONE)}
    per_behavior = {}
    for behavior in BEHAVIORS:
        per_behavior[behavior] = evaluate_behavior(system, behavior, by_label[behavior], args.max_new_tokens)
        print(behavior, json.dumps(per_behavior[behavior]), flush=True)
    none_result = evaluate_none(system, by_label[NONE])
    report = {"run": args.run, "per_behavior": per_behavior, "none": none_result,
              "overall": summarize(per_behavior, none_result)}
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="pilot_final")
    ap.add_argument("--model", default=None, help="Qwen3-0.6B folder (else $QWEN_MODEL_PATH, the WSL path, or the hub cache)")
    ap.add_argument("--limit", type=int, default=None, help="max prompts per behavior and for none")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--out", default=None, help="also write the report as JSON here")
    main(ap.parse_args())
