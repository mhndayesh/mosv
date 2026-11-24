"""Build and check the training data for the Haiku-judged run: all 8 topics, with the full story mix.

Run: python -m rl.prepare_v4
Story comes from data_review/rl_v3 and the other topics from data_review/rl_v2. Sexual-themed prompts are
removed from every topic and never printed. The run stops if a split check fails.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from behaviors.prepare_story_mix import SEXUAL, is_blocked
from rl.data import check_separate, read_split, write_jsonl
from router import resolve_topics

ROOT = Path(__file__).parent.parent
OUT = ROOT / "data_review" / "rl_v4"
SOURCES = {"story": ROOT / "data_review" / "rl_v3"}
DEFAULT_SOURCE = ROOT / "data_review" / "rl_v2"
SPLITS = ("routed", "dev", "test")
PROMPTS_PER_UPDATE = 16
EVERYDAY_WORDS = {"strip", "strips", "stripped", "stripping", "moana"}
# A repeated fan-fiction template implies nudity without using the sexual keywords, sometimes with underage characters.
NUDITY_TEMPLATE = re.compile(r"\b(?:in the buff|au naturel|skinny[- ]dip\w*|butts?|buttocks|bare[- ]?(?:ass|butt)|"
                             r"(?:bathe|bathes|bathing|bath|wash\w*|relax\w*) in (?:a|the) pond)\b", re.I)


def has_sexual_theme(text: str) -> bool:
    """Keyword hits, ignoring everyday words such as stripping a string or the film Moana."""
    return any(match.group(0).lower() not in EVERYDAY_WORDS for match in SEXUAL.finditer(text))


def keep_prompt(topic: str, text: str) -> bool:
    if is_blocked(text) or has_sexual_theme(text):
        return False
    return not (topic == "story" and NUDITY_TEMPLATE.search(text))


def problems_in(parts: dict[str, list[dict]]) -> list[str]:
    found = []
    rows = [row for part in parts.values() for row in part]
    if any(not row["prompt"].strip() for row in rows):
        found.append("blank prompt")
    if len({row["id"] for row in rows}) != len(rows):
        found.append("duplicate id")
    names = list(parts)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            if {row["prompt"] for row in parts[first]} & {row["prompt"] for row in parts[second]}:
                found.append(f"same prompt in {first} and {second}")
    return found


def prepare_topic(topic: str, out: Path) -> tuple[dict, list[str]]:
    folder = SOURCES.get(topic, DEFAULT_SOURCE)
    loaded = {name: read_split(topic, name, folder) for name in SPLITS}
    parts = {name: [row for row in rows if keep_prompt(topic, row["prompt"])] for name, rows in loaded.items()}
    check_separate(parts, topic)
    found = problems_in(parts)
    sexual = sum(has_sexual_theme(row["prompt"]) for part in parts.values() for row in part)
    if sexual:
        found.append(f"{sexual} sexual-themed prompts left")
    for name, rows in parts.items():
        write_jsonl(out / f"{topic}_{name}.jsonl", rows)
    counts = {name: len(rows) for name, rows in parts.items()}
    return {**counts, "removed_for_safety": sum(len(loaded[n]) - counts[n] for n in SPLITS), "sexual_keyword_hits": sexual,
            "updates_one_pass": math.ceil(counts["routed"] / PROMPTS_PER_UPDATE)}, found


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report, failures = {}, Counter()
    for topic in resolve_topics("v2"):
        report[topic], found = prepare_topic(topic, OUT)
        failures.update({f"{topic}: {item}": 1 for item in found})
    totals = {key: sum(row[key] for row in report.values()) for key in ("routed", "dev", "test", "updates_one_pass")}
    summary = {"topics": report, "totals": totals, "problems": sorted(failures)}
    (OUT / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
