import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from behaviors.rubrics import score_response
from evaluate import load_cases
from model_profiles import get_profile
from system import NONE, load_system

RUN = "qwen4b_haiku_v3_test"  # Picked vectors; the router choice does not depend on them.
SEED = 7
TRAINED = ("engineering", "story", "teaching", "debugging", "math_solving", "explanation", "professional_email",
           "how_to_guide")
PER_TOPIC = 10
PER_NONE = 6
MAX_PROMPT_TOKENS = 512
BATCH = 8
VRAM_LIMIT_BYTES = 13.5e9
OUT = ROOT / "data_review" / "reports" / "compare_4b.json"


def fits(system, prompt):
    return len(system.tokenizer(system._chat(prompt), add_special_tokens=False)["input_ids"]) <= MAX_PROMPT_TOKENS


def pick(system, cases):
    rng, chosen = random.Random(SEED), []
    for label in (*TRAINED, NONE):
        rows = [c for c in cases if c["label"] == label and fits(system, c["prompt"])]
        chosen += rng.sample(rows, PER_NONE if label == NONE else PER_TOPIC)
    return chosen


def score(label, text, prompt):
    return None if label == NONE else round(score_response(label, text, prompt).score, 3)


def answer_group(system, group, ids, max_new, batch):
    """All prompts in one call: the system batches them by route, so each card does few, full batches."""
    prompts = [c["prompt"] for c in group]
    routed = system.generate(prompts, max_new, batch)
    normal = system.generate_fixed(prompts, None, max_new, batch)
    return [{"id": i, "type": c["label"], "prompt": c["prompt"], "route": a.route,
             "confidence": round(a.confidence, 3), "normal": n, "steered": a.text,
             "normal_score": score(c["label"], n, c["prompt"]),
             "steered_score": score(c["label"], a.text, c["prompt"])}
            for i, c, a, n in zip(ids, group, routed, normal)]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Routed vs normal answers, side by side")
    parser.add_argument("--run", default=RUN, help="vector folder under checkpoints/rl_pilot")
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--labels", default=",".join((*TRAINED, NONE)), help="this card's share of the prompt types")
    parser.add_argument("--merge", type=Path, nargs="+", help="join these part files into --out, then stop")
    parser.add_argument("--out", type=Path, default=OUT)
    return parser.parse_args(argv)


def merge(parts, out):
    rows = sorted((r for p in parts for r in json.loads(p.read_text(encoding="utf-8"))), key=lambda r: r["id"])
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("The part files share prompt ids")
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"merged": len(rows), "routes": [r["route"] for r in rows]}), flush=True)


def main():
    args = parse_args()
    if args.merge:
        return merge(args.merge, args.out)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, VRAM_LIMIT_BYTES / total), 0)
    profile = get_profile("qwen4b")
    system = load_system(None, args.run, profile="qwen4b")
    chosen = pick(system, load_cases(None, TRAINED, profile.router_folder / "prompts.jsonl"))
    labels = set(args.labels.split(","))
    mine = [(i + 1, c) for i, c in enumerate(chosen) if c["label"] in labels]  # Ids match across cards.
    print(json.dumps({"picked": len(chosen), "this_card": len(mine)}), flush=True)
    start = time.time()
    rows = answer_group(system, [c for _, c in mine], [i for i, _ in mine], args.max_new, args.batch)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"finished": len(rows), "seconds": round(time.time() - start),
                      "peak_gb": round(torch.cuda.max_memory_allocated(0) / 1e9, 2),
                      "routes": [r["route"] for r in rows]}), flush=True)


if __name__ == "__main__":
    main()
