"""Lock the 200-prompt test set used to compare every model (Qwen, Bonsai, before and after MoSV training).

25 prompts per topic x 8 topics, all from the router's held-out "test" split (never used to train v3 or v4).
Picked at random per topic with seed 11; Qwen and Bonsai are all re-answered on this same set.

Writes:
  results/test200_ids.json         ids + sha256 of the prompt texts (in git; no prompt text, the sources are research-only)
  data_review/test200/prompts.jsonl  the full prompts (gitignored)
Run again and it refuses to change an existing lock unless --force.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "data_review" / "router_v2" / "prompts.jsonl"
LOCK = ROOT / "results" / "test200_ids.json"
FULL = ROOT / "data_review" / "test200" / "prompts.jsonl"
TOPICS = ("engineering", "story", "teaching", "debugging", "math_solving", "explanation", "professional_email",
          "how_to_guide")
PER_TOPIC = 25
SEED = 11


def digest(rows):
    return hashlib.sha256("\n".join(r["prompt"] for r in rows).encode("utf-8")).hexdigest()


def pick(test_rows):
    rng, chosen = random.Random(SEED), []
    for topic in TOPICS:
        pool = sorted((r for r in test_rows if r["label"] == topic), key=lambda r: r["id"])
        chosen += rng.sample(pool, PER_TOPIC)
    return [{"id": r["id"], "label": r["label"], "prompt": r["text"]} for r in chosen]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--force", action="store_true", help="overwrite an existing lock")
    args = p.parse_args(argv)
    if LOCK.exists() and not args.force:
        sys.exit(f"{LOCK} already exists; the test set is locked (use --force to rebuild)")
    rows = [json.loads(line) for line in PROMPTS.open(encoding="utf-8")]
    test_rows = [r for r in rows if r["split"] == "test"]
    chosen = pick(test_rows)
    FULL.parent.mkdir(parents=True, exist_ok=True)
    FULL.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chosen), encoding="utf-8")
    lock = {"seed": SEED, "per_topic": PER_TOPIC, "source": "data_review/router_v2/prompts.jsonl split=test",
            "sha256_prompts": digest(chosen), "ids": [[r["id"], r["label"]] for r in chosen]}
    LOCK.write_text(json.dumps(lock, indent=1) + "\n", encoding="utf-8")
    print(f"{len(chosen)} prompts, sha256 {lock['sha256_prompts'][:16]}")


if __name__ == "__main__":
    main()
