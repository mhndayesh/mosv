"""v6 prompt set (Gemma 4 12B): the v4 topics minus engineering and how-to, plus the v5 engineering data.

Run: python -m behaviors.prepare_v6
Topics: story, teaching, debugging, math_solving, explanation, professional_email (from data_review/clean_v2, the
v3/v4 data) + engineering (from data_review/clean_v5, behavior-teaching coding prompts). how_to_guide and the old
engineering set are dropped. Splits are kept as built (train/dev/test never mix).

Writes data_review/v6/<topic>/{train,dev}.jsonl and data_review/v6/test.jsonl: a LOCKED test set of TEST_SIZE
prompts spread evenly over the 7 topics, drawn only from each topic's test split (v5 engineering: from its locked
test split). results/locked_v6.json records ids + sha256; a rebuild that would change it stops (--force to replace).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys

from behaviors.prepare_data import DATA, write_jsonl

ROOT = DATA.parent
OLD = DATA / "clean_v2"
NEW = DATA / "clean_v5"
OUT = DATA / "v6"
LOCK = ROOT / "results" / "locked_v6.json"
OLD_TOPICS = ("story", "teaching", "debugging", "math_solving", "explanation", "professional_email")
TOPICS = OLD_TOPICS + ("engineering",)
TEST_SIZE = 300
SEED = 6


def read(path):
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def topic_rows(topic: str, split: str) -> list[dict]:
    folder = NEW if topic == "engineering" else OLD
    return [{**r, "behavior": topic} for r in read(folder / topic / f"{split}.jsonl")]


def pick_test(rng: random.Random) -> list[dict]:
    """TEST_SIZE // 7 per topic, remainder to the first topics; ordered by topic then id."""
    base, extra = divmod(TEST_SIZE, len(TOPICS))
    chosen = []
    for i, topic in enumerate(TOPICS):
        pool = sorted(topic_rows(topic, "test"), key=lambda r: r["id"])
        chosen += sorted(rng.sample(pool, base + (i < extra)), key=lambda r: r["id"])
    return chosen


def digest(rows):
    return hashlib.sha256("\n".join(r["prompt"] for r in rows).encode("utf-8")).hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    test = pick_test(random.Random(SEED))
    lock = {"seed": SEED, "topics": TOPICS, "size": len(test), "sha256": digest(test),
            "ids": [[r["id"], r["behavior"]] for r in test]}
    if LOCK.exists() and not args.force and json.loads(LOCK.read_text(encoding="utf-8"))["sha256"] != lock["sha256"]:
        sys.exit(f"{LOCK} is locked and the new test set differs; use --force to replace it on purpose")
    for topic in TOPICS:
        for split in ("train", "dev"):
            write_jsonl(OUT / topic / f"{split}.jsonl", topic_rows(topic, split))
    write_jsonl(OUT / "test.jsonl", test)
    LOCK.write_text(json.dumps(lock, indent=1) + "\n", encoding="utf-8")
    counts = {t: {s: len(topic_rows(t, s)) for s in ("train", "dev")} | {"test_locked": sum(r["behavior"] == t for r in test)}
              for t in TOPICS}
    print(json.dumps({"test": len(test), "sha256": lock["sha256"][:16], "per_topic": counts}, indent=1))


if __name__ == "__main__":
    main()
