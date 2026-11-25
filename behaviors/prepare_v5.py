"""v5 prompt sets (Bonsai): build behavior-teaching prompts per topic into train/dev/test. RESEARCH USE ONLY.

Run: python -m behaviors.prepare_v5 --topic engineering
Steering teaches HOW to answer (plan, simple design, clean code, tests), not facts. So prompts are tasks to DO
("write / implement / refactor / review this code"), never questions to look up ("what is", "explain", "difference").

Raw downloads: data_review/raw_v5 (see SOURCES). Output: data_review/clean_v5/<topic>/{train,dev,test}.jsonl,
report.json and removed_samples.jsonl. Splits come from a hash of each row's group (near-duplicates share a group),
so a test prompt can never land in train or dev. results/locked_v5/<topic>.json records the test ids + sha256;
a rebuild that would change a locked test split stops instead (--force to rebuild on purpose).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
import re
import sys

import pyarrow.parquet as pq

from behaviors.prepare_data import DATA, Audit, norm_key, write_jsonl
from behaviors.prepare_topics import balanced_cap, common_reason, make_record, pick_order, split_rows

ROOT = DATA.parent
RAW = DATA / "raw_v5"
OUT = DATA / "clean_v5"
LOCKS = ROOT / "results" / "locked_v5"
CAPS = {"train": 3000, "dev": 300, "test": 300}
REVIEW_PER_TOPIC = 25  # the fixed prompts in the before/after comparison set

DO_TASK = re.compile(r"^\s*(?:please\s+)?(?:write|implement|create|build|develop|design|refactor|optimi[sz]e|fix|"
                     r"convert|add|make|generate|define|construct|code|program|modify|extend|rewrite)\b", re.I)
LOOKUP = re.compile(r"^\s*(?:what|why|explain|describe|when|which|is|are|does|do you|difference|compare)\b|"
                    r"\b(?:difference between|what is|what are|explain (?:the|how|what))\b", re.I)
WEB_OR_GAME = re.compile(r"\b(?:landing page|website|web ?page|web app|html|css|game|three\.js|animation)\b", re.I)
MISSING_SNIPPET = re.compile(r"\b(?:given|following|this|provided|above|below) (?:code|snippet|script|function)\b", re.I)
# Ordinary code words the shared adult/violence filter would wrongly hit (std = standard deviation, sex = a field).
CODE_WORDS = re.compile(r"\b(?:std|stds|sex|kill|killed|strip\w*|execut\w*|meth)\b", re.I)
WORDS = {"self_oss": (12, 300), "magpie_coder": (8, 300), "codereview": (25, 450)}
MAX_CHARS = 2500


def tidy_code(text: str) -> str:
    """Keep newlines and indentation (code); only unescape HTML and trim trailing spaces."""
    lines = html.unescape(text or "").replace("\r\n", "\n").split("\n")
    return "\n".join(line.rstrip() for line in lines).strip()


def reason(text: str, source: str) -> str | None:
    if len(text) > MAX_CHARS:
        return "too_long_chars"
    base = common_reason(CODE_WORDS.sub(" ", text), WORDS[source])
    if base:
        return base
    if WEB_OR_GAME.search(text):
        return "web_or_game"
    if source != "codereview" and LOOKUP.search(text) and not DO_TASK.search(text):
        return "lookup_question"
    if source != "codereview" and MISSING_SNIPPET.search(text) and "```" not in text and "\n    " not in text:
        return "missing_snippet"
    return None


def load_self_oss() -> list[tuple[str, str]]:
    table = pq.read_table(sorted((RAW / "self_oss" / "data").glob("*.parquet")), columns=["instruction", "seed"])
    return [(r["instruction"], r["seed"] or r["instruction"]) for r in table.to_pylist()]


def load_magpie() -> list[tuple[str, str]]:
    cols = ["instruction", "task_category", "difficulty", "input_quality", "language", "uuid", "min_similar_uuid"]
    rows = []
    for path in sorted((RAW / "magpie_coder" / "data").glob("*.parquet")):
        for r in pq.read_table(path, columns=cols).to_pylist():
            if (r["task_category"] == "Coding & Debugging" and r["difficulty"] in ("medium", "hard")
                    and r["input_quality"] in ("good", "excellent") and r["language"] == "EN"
                    and DO_TASK.search(r["instruction"] or "")):
                rows.append((r["instruction"], r["min_similar_uuid"] or r["uuid"]))
    return rows


def load_codereview() -> list[tuple[str, str]]:
    with gzip.open(RAW / "codereview" / "codereview.stackexchange.com.jsonl.gz", "rt", encoding="utf-8") as f:
        return [(r[0], r[0][:120]) for r in map(json.loads, f)]  # each line: [title_body, best answer]


SOURCES = {"engineering": {"self_oss": load_self_oss, "magpie_coder": load_magpie, "codereview": load_codereview}}


def clean_source(topic: str, source: str, loader, audit: Audit, seen: set) -> list[dict]:
    out = []
    items = loader()
    audit.counts[source]["raw"] = len(items)
    for raw, group in items:
        text = tidy_code(raw)
        why = reason(text, source)
        if not why and norm_key(text) in seen:
            why = "duplicate"
        if why:
            audit.drop(source, why, text)
            continue
        seen.add(norm_key(text))
        audit.keep(source)
        out.append(make_record(topic, source, f"{source}|{group}", text))
    return out


def digest(rows: list[dict]) -> str:
    return hashlib.sha256("\n".join(r["prompt"] for r in rows).encode("utf-8")).hexdigest()


def lock_test(topic: str, test: list[dict], force: bool) -> None:
    """Record the test split; refuse to silently change it once locked."""
    path = LOCKS / f"{topic}.json"
    review = sorted(test, key=lambda r: pick_order(r["group"]))[:REVIEW_PER_TOPIC]
    lock = {"topic": topic, "test_ids": [r["id"] for r in test], "test_sha256": digest(test),
            "review_ids": [r["id"] for r in review], "review_sha256": digest(review)}
    if path.exists() and not force:
        old = json.loads(path.read_text(encoding="utf-8"))
        if old["test_sha256"] != lock["test_sha256"]:
            sys.exit(f"{path} is locked and the new test split differs; use --force to replace it on purpose")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock, indent=1) + "\n", encoding="utf-8")


def build(topic: str, force: bool) -> dict:
    audit, seen = Audit(), set()
    by_source = {name: clean_source(topic, name, loader, audit, seen) for name, loader in SOURCES[topic].items()}
    parts = {k: [] for k in CAPS}
    for rows in by_source.values():
        for split, part in split_rows(rows).items():
            parts[split] += part
    final = {}
    for split, rows in parts.items():
        grouped = {name: [r for r in rows if r["source"] == name] for name in by_source}
        final[split] = sorted(balanced_cap(grouped, CAPS[split]), key=lambda r: r["id"])
    lock_test(topic, final["test"], force)
    for split, rows in final.items():
        write_jsonl(OUT / topic / f"{split}.jsonl", rows)
    write_jsonl(OUT / topic / "removed_samples.jsonl", audit.samples)
    report = {"topic": topic, "per_source": {k: dict(v) for k, v in audit.counts.items()},
              "final": {s: {n: sum(r["source"] == n for r in rows) for n in by_source} for s, rows in final.items()}}
    (OUT / topic / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--topic", required=True, choices=sorted(SOURCES))
    p.add_argument("--force", action="store_true", help="replace a locked test split")
    args = p.parse_args(argv)
    print(json.dumps(build(args.topic, args.force), indent=1))


if __name__ == "__main__":
    main()
