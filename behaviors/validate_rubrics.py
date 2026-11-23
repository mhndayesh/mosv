"""Prove the checklists can fail: damaged answers must score lower than real ones.

Run: python -m behaviors.validate_rubrics
"""

from __future__ import annotations

import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean

from behaviors.gold_examples import GOLD
from behaviors.rubrics import score_response, sentences

CLEAN = Path(__file__).parent.parent / "data_review" / "clean"
REFERENCE_SAMPLE = 3000
MIN_GOLD_SCORE = 0.8
MIN_SHUFFLE_DROP = 0.10
MAX_OFF_TOPIC = 0.30
MAX_WRONG_FORMAT = 0.25


def pieces(behavior: str, text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()] if behavior == "engineering" else sentences(text)


def join(behavior: str, parts: list[str]) -> str:
    return ("\n" if behavior == "engineering" else " ").join(parts)


def shuffled(behavior: str, text: str, seed: int) -> str:
    parts = pieces(behavior, text)
    random.Random(seed).shuffle(parts)
    return join(behavior, parts)


def without_middle(behavior: str, text: str) -> str:
    parts = pieces(behavior, text)
    return join(behavior, parts[:int(len(parts) * 0.35)] + parts[int(len(parts) * 0.7):])


def first_half(behavior: str, text: str) -> str:
    parts = pieces(behavior, text)
    return join(behavior, parts[:len(parts) // 2])


def score_one(job: tuple[str, str, str | None]) -> float:
    return score_response(job[0], job[1], job[2]).score


def run_jobs(pool, jobs: list[tuple]) -> float:
    return round(mean(pool.map(score_one, jobs, chunksize=64)), 4)


def control_scores(pool, behavior: str, items: list[tuple[str | None, str]]) -> dict[str, float]:
    other = [b for b in GOLD if b != behavior]
    result = {
        "real": run_jobs(pool, [(behavior, text, prompt) for prompt, text in items]),
        "shuffled": run_jobs(pool, [(behavior, shuffled(behavior, text, i), prompt)
                                    for i, (prompt, text) in enumerate(items)]),
        "no_middle": run_jobs(pool, [(behavior, without_middle(behavior, text), prompt) for prompt, text in items]),
        "first_half": run_jobs(pool, [(behavior, first_half(behavior, text), prompt) for prompt, text in items]),
    }
    for name in other:
        result[f"scored_as_{name}"] = run_jobs(pool, [(name, text, None) for _, text in items])
    return result


def off_topic(pool, behavior: str) -> float:
    items = GOLD[behavior]
    return run_jobs(pool, [(behavior, text, items[(i + 1) % len(items)][0]) for i, (_, text) in enumerate(items)])


def read_reference(behavior: str, source: str) -> list[tuple[None, str]]:
    with (CLEAN / behavior / "reference.jsonl").open(encoding="utf-8") as file:
        texts = [row["text"] for row in map(json.loads, file) if row["source"] == source]
    random.Random(11).shuffle(texts)
    return [(None, text) for text in texts[:REFERENCE_SAMPLE]]


def verdict(scores: dict[str, float], gold: bool) -> dict[str, bool]:
    passed = {"shuffle_drops": scores["real"] - scores["shuffled"] >= MIN_SHUFFLE_DROP,
              "damage_drops": scores["no_middle"] < scores["real"] and scores["first_half"] < scores["real"]}
    if gold:
        passed["gold_scores_high"] = scores["real"] >= MIN_GOLD_SCORE
        passed["off_topic_low"] = scores["off_topic"] <= MAX_OFF_TOPIC
    return passed


def main() -> None:
    report = {}
    with ProcessPoolExecutor(max_workers=max(2, (os.cpu_count() or 4) - 2)) as pool:
        for behavior, items in GOLD.items():
            scores = control_scores(pool, behavior, items)
            scores["off_topic"] = off_topic(pool, behavior)
            report[f"gold/{behavior}"] = {"n": len(items), **scores, "pass": verdict(scores, True)}
        for behavior, source in (("story", "simplestories"), ("teaching", "kindergarten")):
            items = read_reference(behavior, source)
            scores = control_scores(pool, behavior, items)
            report[f"real/{source}"] = {"n": len(items), **scores, "pass": verdict(scores, False)}
        # Teacher planning documents are not a lesson delivered to a child, so they should score low.
        items = read_reference("teaching", "lesson_plan")
        scores = control_scores(pool, "teaching", items)
        report["wrong_format/lesson_plan"] = {"n": len(items), **scores,
                                              "pass": {"scores_low": scores["real"] <= MAX_WRONG_FORMAT}}
    (CLEAN / "rubric_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
