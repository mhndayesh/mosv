"""Clean the raw datasets into train/test prompts. RESEARCH USE ONLY.

Run: python -m behaviors.prepare_data
Removes adult content, blanks, duplicates and non-primary lessons, and saves a
sample of everything removed so the filters can be checked by eye.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

DATA = Path(__file__).parent.parent / "data_review"
RAW = DATA / "raw"
CLEAN = DATA / "clean"
TEST_PERCENT = 10
SAMPLES_PER_REASON = 25
MIN_PROMPT_WORDS, MAX_PROMPT_WORDS = 5, 70
MIN_ASCII_SHARE = 0.95

ADULT = re.compile(
    r"\b(?:sex\w*|porn\w*|nude\w*|naked|rap(?:e|ed|es|ing|ist|ists)|incest|p(?:a|ae)?edophil\w*"
    r"|molest\w*|erotic\w*|orgasm\w*|penis\w*|vagina\w*|genital\w*|boobs?|breasts?|hookers?"
    r"|prostitut\w*|masturbat\w*|fetish\w*|nsfw|whores?|sluts?|fuck\w*|tortur\w*|gore"
    r"|dismember\w*|decapitat\w*|mutilat\w*|suicid\w*|self-harm|cannibal\w*)\b",
    re.IGNORECASE,
)
PRIMARY = re.compile(r"^PRIMARY\s*([1-6])?$", re.IGNORECASE)
WP_TAG = re.compile(r"^\s*[\[(]\s*(\w+)\s*[\])]\s*")
CURLY_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "—": "-", "–": "-", "…": "..."})


class Audit:
    """Counts what was kept and removed, and keeps samples of the removed text."""

    def __init__(self) -> None:
        self.counts: dict[str, Counter] = defaultdict(Counter)
        self.samples: list[dict] = []
        self._sampled: Counter = Counter()

    def keep(self, source: str) -> None:
        self.counts[source]["kept"] += 1

    def drop(self, source: str, reason: str, text: str) -> None:
        self.counts[source][reason] += 1
        if self._sampled[(source, reason)] < SAMPLES_PER_REASON:
            self._sampled[(source, reason)] += 1
            self.samples.append({"source": source, "reason": reason, "text": text[:300]})


def norm_key(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def detokenize(text: str) -> str:
    """Undo the spaced-out punctuation in WritingPrompts, e.g. `did n't` and `word ,`."""
    text = text.translate(CURLY_QUOTES).replace("``", '"').replace("''", '"')
    text = re.sub(r"\s+(n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1", text)
    text = re.sub(r"\s+([,.;:!?)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    return " ".join(text.split())


def is_test(group: str) -> bool:
    """Stable split by group, so the same prompt or topic never lands on both sides."""
    digest = hashlib.sha256(norm_key(group).encode()).digest()
    return int.from_bytes(digest[:8], "big") % 100 < TEST_PERCENT


def record(behavior: str, source: str, group: str, prompt: str) -> dict:
    uid = hashlib.sha256(f"{source}|{norm_key(prompt)}".encode()).hexdigest()[:16]
    return {"id": uid, "behavior": behavior, "source": source, "group": norm_key(group), "prompt": prompt}


def read_parquet(folder: Path, columns: list[str] | None = None) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(folder.rglob("*.parquet")):
        rows.extend(pq.read_table(path, columns=columns).to_pylist())
    return rows


def clean_story_prompt(raw: str | None) -> tuple[str | None, str | None]:
    """Return (prompt, None) when kept, or (None, reason) when removed."""
    text = (raw or "").strip()
    tag = WP_TAG.match(text)
    if not text:
        return None, "blank"
    if not tag or tag.group(1).upper() != "WP":
        return None, "not_wp_tag"
    text = detokenize(text[tag.end():])
    count = len(text.split())
    if ADULT.search(text):
        return None, "adult"
    if not MIN_PROMPT_WORDS <= count <= MAX_PROMPT_WORDS:
        return None, "bad_length"
    if sum(c.isascii() for c in text) / len(text) < MIN_ASCII_SHARE:
        return None, "not_english"
    return text, None


def story_prompts(audit: Audit) -> list[dict]:
    source, seen, out = "writingprompts", set(), []
    for row in read_parquet(RAW / "story_writingprompts", ["prompt"]):
        text, reason = clean_story_prompt(row["prompt"])
        if reason:
            audit.drop(source, reason, row["prompt"] or "")
        elif norm_key(text) in seen:
            audit.drop(source, "duplicate", text)
        else:
            seen.add(norm_key(text))
            audit.keep(source)
            out.append(record("story", source, text, f"Write a short story about this idea: {text}"))
    return out


def srdd_prompts(audit: Audit, seen: set[str]) -> list[dict]:
    source, out = "srdd", []
    path = RAW / "engineering_srdd" / "data_attribute_format.csv"
    with path.open(encoding="utf-8", errors="replace", newline="") as file:
        for row in csv.DictReader(file):
            name = (row["Name"] or "").replace("_", " ").strip()
            text = " ".join((row["Description"] or "").split())
            if not name or not text:
                audit.drop(source, "blank", f"{name}: {text}")
            elif ADULT.search(f"{name} {text}"):
                audit.drop(source, "adult", text)
            elif not text.endswith((".", "!", "?")):
                audit.drop(source, "cut_off", text)
            elif norm_key(name) in seen or norm_key(text) in seen:
                audit.drop(source, "duplicate", text)
            else:
                seen.update((norm_key(name), norm_key(text)))
                audit.keep(source)
                out.append(record("engineering", source, name, f"Create a new software project for {name}. {text}"))
    return out


def app_idea_summary(markdown: str) -> tuple[str, str]:
    title = re.search(r"^#\s+(.+)$", markdown, flags=re.MULTILINE)
    body = re.sub(r"^#.*$|^\*\*Tier:\*\*.*$", "", markdown, flags=re.MULTILINE)
    body = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", body)
    paragraph = next((p for p in re.split(r"\n\s*\n", body) if p.strip()), "")
    return (title.group(1).strip() if title else ""), " ".join(paragraph.split())


def app_idea_prompts(audit: Audit, seen: set[str]) -> list[dict]:
    source, out = "app-ideas", []
    for path in sorted((RAW / "engineering_app_ideas").glob("*.md")):
        name, text = app_idea_summary(path.read_text(encoding="utf-8", errors="replace"))
        if not name or not text:
            audit.drop(source, "blank", path.name)
        elif ADULT.search(f"{name} {text}"):
            audit.drop(source, "adult", text)
        elif norm_key(name) in seen:
            audit.drop(source, "duplicate", name)
        else:
            seen.add(norm_key(name))
            audit.keep(source)
            item = record("engineering", source, name, f"Create a new software project for {name}. {text}")
            out.append({**item, "force_test": True})
    return out


def lesson_fields(input_text: str | None) -> tuple[str, str, str]:
    text = input_text or ""
    subject = re.search(r"Subject:\s*([^,]*)", text)
    topic = re.search(r"Topic:\s*(.*?),\s*Class:", text)
    grade = re.search(r"Class:\s*(.*)$", text)
    return tuple(" ".join(m.group(1).split()) if m else "" for m in (subject, topic, grade))


def tidy_case(text: str) -> str:
    text = text.translate(CURLY_QUOTES).strip(" -:")
    return text.title() if text.isupper() else text


def lesson_reason(subject: str, topic: str, grade: str) -> str | None:
    if not subject or not topic or not grade:
        return "blank"
    if not PRIMARY.match(grade):
        return "non_primary"
    if ADULT.search(f"{subject} {topic}"):
        return "adult"
    if "�" in topic:
        return "broken_text"
    return None


def lesson_key(subject: str, topic: str, grade: str) -> str:
    """`PRIMARY 4` and `PRIMARY4` are the same class."""
    return norm_key(f"{subject} {topic} {grade.replace(' ', '')}")


def teaching_prompts(audit: Audit) -> tuple[list[dict], list[dict]]:
    source, seen, prompts, references = "lesson_plan", set(), [], []
    for row in read_parquet(RAW / "teaching_lesson_plan"):
        subject, topic, grade = lesson_fields(row["input"])
        output = row["output"] or ""
        reason = lesson_reason(subject, topic, grade)
        key = lesson_key(subject, topic, grade)
        if reason:
            audit.drop(source, reason, row["instruction"] or "")
        elif key in seen:
            audit.drop(source, "duplicate", row["instruction"])
        else:
            seen.add(key)
            audit.keep(source)
            number = PRIMARY.match(grade).group(1)
            level = f"Primary {number}" if number else "primary school"
            topic_text, subject_text = tidy_case(topic), tidy_case(subject)
            prompts.append(record("teaching", source, topic,
                                  f"Teach {topic_text} ({subject_text}) to a {level} student."))
            if "�" not in output and not ADULT.search(output):
                references.append({"source": source, "text": output})
    return prompts, references


def story_references(audit: Audit) -> list[dict]:
    source, out = "simplestories", []
    for row in read_parquet(RAW / "story_simplestories" / "test", ["story"]):
        text = (row["story"] or "").strip()
        if not text:
            audit.drop(source, "blank", "")
        elif ADULT.search(text):
            audit.drop(source, "adult", text)
        else:
            audit.keep(source)
            out.append({"source": source, "text": text})
    return out


def kindergarten_references(audit: Audit) -> list[dict]:
    source, out = "kindergarten", []
    for path in sorted((RAW / "teaching_kindergarten" / "kindergarten_K1_lessons").rglob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            audit.drop(source, "blank", path.name)
        elif ADULT.search(text):
            audit.drop(source, "adult", text)
        else:
            audit.keep(source)
            out.append({"source": source, "text": text})
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_behavior(behavior: str, prompts: list[dict], references: list[dict]) -> dict:
    train, test = [], []
    for item in prompts:
        forced = item.get("force_test", False)
        clean = {k: v for k, v in item.items() if k != "force_test"}
        (test if forced or is_test(clean["group"]) else train).append(clean)
    overlap = {r["group"] for r in train} & {r["group"] for r in test}
    if overlap:
        raise ValueError(f"{behavior}: {len(overlap)} groups are in both train and test")
    write_jsonl(CLEAN / behavior / "train.jsonl", train)
    write_jsonl(CLEAN / behavior / "test.jsonl", test)
    write_jsonl(CLEAN / behavior / "reference.jsonl", references)
    return {"train": len(train), "test": len(test), "reference": len(references)}


def main() -> None:
    audit, seen = Audit(), set()
    engineering = srdd_prompts(audit, seen) + app_idea_prompts(audit, seen)
    story = story_prompts(audit)
    story_refs = story_references(audit)
    lessons, lesson_refs = teaching_prompts(audit)
    teaching_refs = lesson_refs + kindergarten_references(audit)
    sizes = {
        "engineering": write_behavior("engineering", engineering, []),
        "story": write_behavior("story", story, story_refs),
        "teaching": write_behavior("teaching", lessons, teaching_refs),
    }
    report = {"research_only": True, "sizes": sizes,
              "sources": {name: dict(counts) for name, counts in audit.counts.items()}}
    CLEAN.mkdir(parents=True, exist_ok=True)
    (CLEAN / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_jsonl(CLEAN / "removed_samples.jsonl", audit.samples)
    (CLEAN / "RESEARCH_ONLY.md").write_text(
        "Research use only. Some sources are non-commercial or have unclear rights. "
        "Do not publish or redistribute this data.\n", encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
