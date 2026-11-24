"""Live test of the Haiku judge on 20 random prompts across all 8 topics.

Each prompt gets four answers: normal, steered, normal with shuffled sentences, and an answer to another prompt.
A useful judge must rank the normal answer above the two broken controls.
Run in WSL: python reports/judge_smoke.py
"""

import json
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl.api_judge import ApiJudge, Group

ANSWERS = ROOT / "data_review" / "reports" / "compare_4b.json"
OUT = ROOT / "data_review" / "reports" / "judge_smoke"
SEED, TOTAL, BUDGET_USD = 7, 20, 0.25
ROLES = ("normal", "steered", "shuffled", "wrong_prompt")


def pick_rows(rows, rng):
    topics = sorted({row["type"] for row in rows if row["type"] != "none"})
    by_topic = {topic: [row for row in rows if row["type"] == topic and row["route"] == topic and row["steered"] != row["normal"]]
                for topic in topics}
    picked = [row for topic in topics for row in rng.sample(by_topic[topic], 2)]
    left = [row for topic in topics for row in by_topic[topic] if row not in picked]
    return picked + rng.sample(left, TOTAL - len(picked))


def shuffled_sentences(text, rng):
    parts = [part for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]
    if len(parts) < 3:
        words = text.split()
        rng.shuffle(words)
        return " ".join(words)
    original = list(parts)
    while parts == original:
        rng.shuffle(parts)
    return " ".join(parts)


def build_groups(rows, all_rows, rng):
    groups = []
    for row in rows:
        other = rng.choice([item for item in all_rows if item["type"] not in (row["type"], "none")])
        answers = (row["normal"], row["steered"], shuffled_sentences(row["normal"], rng), other["normal"])
        groups.append(Group(f"id{row['id']}", row["type"], row["prompt"], answers))
    return groups


def win_rate(rewards, first, second):
    pairs = [(reward[ROLES.index(first)], reward[ROLES.index(second)]) for reward in rewards]
    wins = sum(a > b for a, b in pairs) + 0.5 * sum(a == b for a, b in pairs)
    return round(100 * wins / len(pairs), 1)


def order_agreement(log_rows):
    by_group = defaultdict(dict)
    for row in log_rows:
        by_group[row["group"]][row["shuffle"]] = [row["labels"][label] for label in row["ranking"]]
    pairs = [value for value in by_group.values() if len(value) == 2]
    return {"both_orders": len(pairs), "same_best_pct": round(100 * sum(p[0][0] == p[1][0] for p in pairs) / max(len(pairs), 1), 1),
            "same_full_rank_pct": round(100 * sum(p[0] == p[1] for p in pairs) / max(len(pairs), 1), 1)}


def main():
    rng = random.Random(SEED)
    rows = json.loads(ANSWERS.read_text(encoding="utf-8"))
    groups = build_groups(pick_rows(rows, rng), rows, rng)
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "calls.jsonl"
    log.unlink(missing_ok=True)
    judge = ApiJudge(log, run="smoke", budget_usd=BUDGET_USD, seed=SEED)
    try:
        rewards, stats = judge.judge(groups, 0)
    finally:
        judge.close()
    log_rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    calls = [row for row in log_rows if row["type"] == "call" and row["status"] == "ok"]
    scored = [(group, reward) for group, reward in zip(groups, rewards) if reward is not None]
    only = [reward for _, reward in scored]
    report = {"prompts": len(groups), "topics": dict(sorted({g.topic: sum(x.topic == g.topic for x in groups) for g in groups}.items())),
              "batch": stats.summary(), "scored_prompts": len(scored),
              "cached_calls": sum(row["tokens"]["cache_read"] > 0 for row in calls),
              "mean_reward": {role: round(statistics.mean(r[i] for r in only), 3) for i, role in enumerate(ROLES)},
              "normal_beats_shuffled_pct": win_rate(only, "normal", "shuffled"),
              "normal_beats_wrong_prompt_pct": win_rate(only, "normal", "wrong_prompt"),
              "steered_beats_normal_pct": win_rate(only, "steered", "normal"),
              "order_agreement": order_agreement(calls),
              "by_topic": {topic: {role: round(statistics.mean(r[i] for g, r in scored if g.topic == topic), 2)
                                   for i, role in enumerate(ROLES)} for topic in sorted({g.topic for g, _ in scored})}}
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
