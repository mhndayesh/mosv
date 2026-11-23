"""Clean real prompts for the eight topics into train/dev/test files. RESEARCH USE ONLY.

Run: python -m behaviors.prepare_topics [--out data_review/clean_v2] [--topics v2] [--cap 12000]
Raw downloads live in data_review/raw_v2 (see SOURCES). The older 0.6B files in
data_review/clean are only read, never changed. Every rejection is counted by reason and a
sample of each reason is saved so the filters can be checked by eye. Nothing is invented:
a topic with fewer real rows than the cap keeps all of them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from behaviors.prepare_data import ADULT, CLEAN, DATA, Audit, norm_key, write_jsonl
from router import resolve_topics

RAW2 = DATA / "raw_v2"
OUT = DATA / "clean_v2"
CAP = 12_000
TEST_PERCENT, DEV_PERCENT = 10, 10
MAX_NON_ASCII_LETTERS = 0.02  # share of letters; symbols like the minus sign or a degree mark do not count
SAMPLE_PROMPTS = 3
ADULT_EXTRA = re.compile(
    r"\b(?:sexy|seduc\w*|virgins?|virginity|lingerie|strip(?:per|pers|tease)|erection\w*|arous\w*|orgy|bdsm"
    r"|kink\w*|condoms?|horny|hickeys?|escorts?|cleavage|aphrodisiac\w*|threesome|lust\w*|ejaculat\w*|semen|sperm"
    r"|testicl\w*|scrot\w*|vulva|clitor\w*|anus|nipples?|boner|stds?(?!::)|stis?|hiv|syphilis|gonorrh\w*"
    r"|chlamydia|herpes|murder\w*|massacre\w*|genocide|holocaust|nazis?|terroris\w*|cocaine|heroin|overdos\w*"
    r"|drug dealers?|meth|lsd)\b",
    re.IGNORECASE,
)
PUNCT = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "‐": "-", "‑": "-",
                       "–": "-", "—": "-", "…": "...", " ": " "})
BROKEN = re.compile(r"[�âãÃÂ]")  # replacement char and typical mojibake lead letters

# Words allowed per prompt, per source family.
LIMITS = {"existing": (5, 300), "build": (6, 250), "bug": (8, 250), "math": (3, 150), "email": (20, 120),
          "question": (4, 40), "title": (3, 20)}


# ---------- shared checks ----------

def common_reason(text: str, limits: tuple[int, int]) -> str | None:
    """The first reason a prompt is unusable, or None. Applies to every topic."""
    if not text.strip():
        return "blank"
    if BROKEN.search(text):
        return "broken_text"
    if ADULT.search(text) or ADULT_EXTRA.search(text):
        return "adult"
    letters = [c for c in text if c.isalpha()]
    if letters and sum(not c.isascii() for c in letters) / len(letters) > MAX_NON_ASCII_LETTERS:
        return "not_english"
    count = len(text.split())
    if count < limits[0]:
        return "too_short"
    return "too_long" if count > limits[1] else None


def keep(prompt: str, group: str) -> tuple:
    return ("keep", prompt, group)


def drop(reason: str, text: str) -> tuple:
    return ("drop", reason, str(text)[:300])


def checked(prompt: str, group: str, limits_name: str) -> tuple:
    reason = common_reason(prompt, LIMITS[limits_name])
    return drop(reason, prompt) if reason else keep(prompt, group)


def tidy(text: str | None) -> str:
    return " ".join((text or "").translate(PUNCT).split())


def stable_bucket(group: str) -> int:
    """0-99 from the group text. The old 0.6B test split is bucket < 10, so it stays test here."""
    digest = hashlib.sha256(norm_key(group).encode()).digest()
    return int.from_bytes(digest[:8], "big") % 100


def split_of(group: str) -> str:
    bucket = stable_bucket(group)
    return "test" if bucket < TEST_PERCENT else "dev" if bucket < TEST_PERCENT + DEV_PERCENT else "train"


# ---------- cleaners: one per source, each takes one raw item ----------

def clean_existing(row: dict) -> tuple:
    """A prompt that the 0.6B pipeline already cleaned. Re-checked, never rewritten."""
    return checked(row["prompt"], row["group"], "existing")


BUILD_NOUN = re.compile(
    r"\b(?:apps?|applications?|web ?sites?|web ?apps?|sites?|pages?|dashboards?|games?|simulations?|tools?"
    r"|platforms?|portals?|systems?|software|programs?|apis?|bots?|editors?|trackers?|calculators?|generators?"
    r"|extensions?|plugins?|widgets?|projects?|prototypes?|animations?|visuali[sz]ations?|components?"
    r"|interfaces?|clones?|stores?|shops?|managers?|players?|planners?|viewers?|converters?|scripts?"
    r"|crawlers?|scrapers?|servers?|clients?|libraries|library|cli|chat)\b", re.IGNORECASE)
NOT_PROJECT = re.compile(r"^(?:please |can you |could you )?(?:create|make|generate|draw|write|design)\s+(?:me )?"
                         r"(?:an? |the )?(?:[\w-]+\s+){0,3}(?:svg|logo|icon|poem|essay|story|image|picture|drawing"
                         r"|wallpaper)\b", re.IGNORECASE)


VERBS = r"(?:build|create|make|design|develop|implement|generate|code|write|program|set up)"
BUILD_START = re.compile(  # "Build a ...", "Can you create ...", "I want you to make ...", "Let's build ..."
    r"^(?:please |hey,? |hi,? )?(?:(?:can|could|would) you (?:please )?|(?:i|we) (?:need|want|would like) (?:you )?to "
    r"|let'?s |help me )?" + VERBS + r"\b", re.IGNORECASE)
NEED_A_THING = re.compile(r"^(?:please )?(?:i|we) (?:need|want|would like) (?:an?|the|my|some) ", re.IGNORECASE)
VERB_EARLY = re.compile(r"^(?:\W*\w+){0,7}?\W+(?:build|create|make|develop|code)\b", re.IGNORECASE)
PASTED_CODE = re.compile(r"^\s*(?:<!?[a-z]|```|import |function |const |#include)", re.IGNORECASE)


def clean_arena(row: tuple) -> tuple:
    language, prompt = row
    text = (prompt or "").replace("\r\n", "\n").strip()
    if language != "en":
        return drop("not_english", text)
    if PASTED_CODE.match(text):
        return drop("pasted_code", text)
    reason = common_reason(text, LIMITS["build"])
    if reason:
        return drop(reason, text)
    asks = BUILD_START.match(text) or ((NEED_A_THING.match(text) or VERB_EARLY.match(text)) and BUILD_NOUN.search(text))
    if not asks or NOT_PROJECT.match(text):
        return drop("not_a_build_request", text)
    return keep(text, text)


def clean_webdev(instruction: str) -> tuple:
    text = (instruction or "").replace("\r\n", "\n").strip()
    return checked(text, text, "build")


FEATURE_START = re.compile(
    r"^(?:\[[^\]]*\]\s*)*(?:support|add|allow|provide|enable|introduce|implement|improve|offer|extend|make|create"
    r"|enhance|consider|expose|new|feature|enhancement|rfe|request|proposal|please|need|want|suggest\w*)\b",
    re.IGNORECASE)
FEATURE_TEXT = re.compile(r"\b(?:feature request|enhancement|would be (?:nice|great|useful|good|helpful|cool)"
                          r"|it would be|i would like|i'd like|we would like|should be (?:added|possible)"
                          r"|suggestion|proposal|propose|wish|nice to have|ability to)\b", re.IGNORECASE)
BUG_TEXT = re.compile(  # substring match on purpose: `ClassCastException`, `NullPointerException`
    r"exception|error|crash|fail|npe\b|nullpointer|deadlock|hang|freez|frozen|leak|stack ?trace|corrupt"
    r"|regression|unexpected|incorrect|wrong|broken|invalid|abnormal|garbled|flicker|blank|ignor|n't|\bnot\b"
    r"|cannot|can ?not|unable|no longer|missing|mix(?:ed)? up|stuck|throws?|\bbug\b|problem|disappear|abort|\blost\b",
    re.IGNORECASE)
FIX_LEAK = re.compile(r"\b(?:fix(?:ed)? (?:in|by)|fix committed|committed|patch\w*|duplicates? of|dups? of"
                      r"|resolved (?:in|by)|released to|workaround|proposed fix|suggested fix)\b", re.IGNORECASE)
BUG_PREFIX = re.compile(r"^Bug\s+\d+\s*[-:]?\s*", re.IGNORECASE)
DEBUG_WRAPPERS = (
    "Debug this {project} bug report. What is causing it and how would you fix it?\n\nTitle: {title}\n\n{body}",
    "Here is a bug report from {project}. Work out the likely root cause and tell me how to fix it.\n\n"
    "Title: {title}\n\nReport: {body}",
    "Can you debug this issue reported against {project}? Explain the cause and the fix.\n\n{title}\n\n{body}",
    "I need help debugging this {project} bug. Find the problem and suggest a fix.\n\nSummary: {title}\n\n"
    "Details: {body}",
    "Please investigate this {project} defect, find the most likely cause, and propose a fix.\n\n"
    "Title: {title}\n\nDescription: {body}",
    "Debug the following bug report ({project}). Where does it go wrong and how do I fix it?\n\n{title}\n\n{body}",
)


def debug_prompt(project: str, title: str, body: str) -> str:
    index = int(hashlib.sha256(norm_key(title).encode()).hexdigest(), 16) % len(DEBUG_WRAPPERS)
    return DEBUG_WRAPPERS[index].format(project=project.replace("_", " "), title=title, body=body)


def clean_bug(row: tuple) -> tuple:
    project, summary, description = row
    title, body = tidy(BUG_PREFIX.sub("", summary or "")), tidy(description)
    if not title or body.lower() in ("", "none", "null"):
        return drop("empty_description", f"{title} | {body}")
    if FEATURE_START.match(title) or re.search(r"\bshould\b", title, re.I) or FEATURE_TEXT.search(f"{title} {body}"):
        return drop("feature_request", f"{title} | {body}")
    if not BUG_TEXT.search(f"{title} {body}"):
        return drop("no_bug_evidence", f"{title} | {body}")
    if FIX_LEAK.search(f"{title} {body}"):
        return drop("mentions_fix", f"{title} | {body}")
    reason = common_reason(f"{title} {body}", LIMITS["bug"])
    if reason:
        return drop(reason, f"{title} | {body}")
    return keep(debug_prompt(project, title, body), title)


ANSWER_GIVEN = re.compile(r"####|\\boxed|\bsolution\s*:|\banswer\s*(?:is|=|:)\s*[-$\d]", re.IGNORECASE)


def clean_math(row: tuple) -> tuple:
    """Question text only. The solution column is never read."""
    source, question = row
    text = " ".join((question or "").split())
    if "[asy]" in text:
        return drop("diagram_code", text)
    if ANSWER_GIVEN.search(text):
        return drop("contains_answer", text)
    return checked(text, text, "math")


def clean_email(row: tuple) -> tuple:
    """Instruction + input only. The `output` email is never read."""
    instruction, context = tidy(row[0]), (row[1] or "").strip()
    if not instruction or not context:
        return drop("blank", f"{instruction} | {context}")
    return checked(f"{instruction}\n\n{context}", context, "email")


ELI5_TAIL = re.compile(r"\.?\s*\[(?:removed|deleted)\]\s*$", re.IGNORECASE)
ELI5_PREFIX = re.compile(r"^\s*eli5\s*[:\-]?\s*", re.IGNORECASE)
ELI5_CONTEXT = re.compile(r"_URL_|\bedit\b|https?://|\breddit\b|\bsub(?:reddit)?\b|\bOP\b|\bupvote\w*|\bthanks\b",
                          re.IGNORECASE)
QUESTION_START = re.compile(r"^(?:why|how|what|when|where|who|whom|whose|which|is|are|was|were|do|does|did|can"
                            r"|could|would|will|should|if|in|has|have|am)\b", re.IGNORECASE)


def clean_eli5(raw: str) -> tuple:
    """Keep the question title only. The stored text is `title. body`, so cut at the first `?. `."""
    text = ELI5_PREFIX.sub("", ELI5_TAIL.sub("", (raw or "").strip()))
    cut = text.find("?. ")
    text = text[:cut + 1] if cut >= 0 else text
    if "\n" in text:
        return drop("has_body_text", raw)
    text = tidy(text).lstrip("-*> ").strip()
    if ELI5_CONTEXT.search(text) or "[removed]" in text:
        return drop("reddit_context", raw)
    reason = common_reason(text, LIMITS["question"])
    if reason:
        return drop(reason, text)
    if re.match(r"how to\b", text, re.IGNORECASE):
        return drop("how_to_not_explanation", text)
    if not (text.endswith("?") and QUESTION_START.match(text)):
        return drop("not_a_question", text)  # fragments like `Apple vs. Samsung lawsuits` have nothing to explain
    return keep(text, text)


def clean_wikihow(row: tuple) -> tuple:
    raw, title = row
    text = tidy(title)
    if not text.lower().startswith("how to "):
        return drop("not_how_to", raw)
    return checked(text, text, "title")


CLEANERS = {name[6:]: fn for name, fn in list(globals().items()) if name.startswith("clean_")}


def run_cleaner(name: str, raw) -> tuple:
    return CLEANERS[name](raw)


# ---------- loaders: raw files to plain items ----------

def read_jsonl_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def existing_rows(topic: str) -> list[dict]:
    return [row for part in ("train", "test") for row in read_jsonl_rows(CLEAN / topic / f"{part}.jsonl")]


def parquet_column(path: Path, column: str) -> list:
    return pq.read_table(path, columns=[column]).column(column).to_pylist()


def load_arena() -> list[tuple]:
    table = pq.read_table(RAW2 / "lmarena-ai__arena-webdev-preferences/data/preference.parquet",
                          columns=["language", "prompt"])
    return list(zip(table.column("language").to_pylist(), table.column("prompt").to_pylist()))


def load_webdev() -> list[str]:
    data = json.loads((RAW2 / "ameeq-ul-haq__WebDev-Dataset/dataset.json").read_text(encoding="utf-8"))
    return [row["instruction"] for row in data]


def load_bugs() -> list[tuple]:
    table = pq.read_table(RAW2 / "Partha117__apache_bug_reports/data/train-00000-of-00001.parquet",
                          columns=["project_name", "summary", "description"])
    return list(zip(*(table.column(c).to_pylist() for c in ("project_name", "summary", "description"))))


def load_math(folder: str, column: str, pattern: str) -> list[tuple]:
    paths = sorted((RAW2 / folder).glob(pattern))
    return [(folder, text) for path in paths for text in parquet_column(path, column)]


def load_email() -> list[tuple]:
    path = RAW2 / "wardacoder__business-email-dataset/alpaca_email_dataset_complete.jsonl"
    return [(row["instruction"], row["input"]) for row in read_jsonl_rows(path)]


def load_eli5() -> list[str]:
    path = RAW2 / "vincentmin__eli5_rlhf_explainlikeim5/data/finetune/train-00000-of-00001.parquet"
    return sorted(set(parquet_column(path, "question")))  # one row per answer pair, so dedupe the questions


def strip_footnotes(titles: list[str]) -> list[str]:
    """WikiHow titles carry a part number glued to the end (`...Artist2`). Remove it only when the
    same base title shows a 1..n series, so `Vitamin B12` and `NBA 2K14` keep their digits."""
    series: dict[str, set[int]] = defaultdict(set)
    options: dict[str, list[tuple[str, int]]] = {}
    for title in titles:
        run = re.search(r"\d+$", title)
        found = []
        if run:
            for k in range(1, min(len(run.group()), 3) + 1):
                tail, base = title[-k:], title[:-k]
                if tail[0] != "0" and base and not base.endswith(" "):
                    series[base].add(int(tail))
                    found.append((base, int(tail)))
        options[title] = found
    cleaned = []
    for title in titles:
        valid = [(base, n) for base, n in options[title] if set(range(1, n + 1)) <= series[base]]
        if valid:
            cleaned.append(max(valid, key=lambda item: len(series[item[0]]))[0])
        else:  # a lone part number glued to a lowercase letter or closing mark, e.g. `...Solid3`
            cleaned.append(re.sub(r"(?<=[a-z)\"'?!.])\d{1,2}$", "", title))
    return cleaned


def load_wikihow() -> list[tuple]:
    table = pacsv.read_csv(
        RAW2 / "Lajvi__wikihowAll/wikihowAll.csv", read_options=pacsv.ReadOptions(block_size=1 << 26),
        parse_options=pacsv.ParseOptions(newlines_in_values=True, invalid_row_handler=lambda row: "skip"),
        convert_options=pacsv.ConvertOptions(include_columns=["title"], strings_can_be_null=True))
    titles = [t for t in table.column("title").to_pylist() if t]
    return list(zip(titles, strip_footnotes(titles)))


# (topic, source name, cleaner name, loader)
SOURCES = (
    ("engineering", "existing_engineering", "existing", lambda: existing_rows("engineering")),
    ("engineering", "arena_webdev", "arena", load_arena),
    ("engineering", "webdev_100", "webdev", load_webdev),
    ("story", "writingprompts", "existing", lambda: existing_rows("story")),
    ("teaching", "lesson_plan", "existing", lambda: existing_rows("teaching")),
    ("debugging", "apache_bug_reports", "bug", load_bugs),
    ("math_solving", "gsm8k", "math", lambda: load_math("openai__gsm8k", "question", "main/*.parquet")),
    ("math_solving", "hendrycks_math", "math", lambda: load_math("EleutherAI__hendrycks_math", "problem", "*/*.parquet")),
    ("math_solving", "orca_math", "math", lambda: load_math("microsoft__orca-math-word-problems-200k", "question", "data/*.parquet")),
    ("explanation", "eli5", "eli5", load_eli5),
    ("professional_email", "business_email", "email", load_email),
    ("how_to_guide", "wikihow", "wikihow", load_wikihow),
)


# ---------- assemble ----------

def pick_order(group: str) -> str:
    return hashlib.sha256(f"pick|{norm_key(group)}".encode()).hexdigest()


def balanced_cap(by_source: dict[str, list[dict]], cap: int) -> list[dict]:
    """Take at most `cap` rows, sharing the room evenly across sources (small sources keep everything)."""
    if sum(map(len, by_source.values())) <= cap:
        return [row for rows in by_source.values() for row in rows]
    quotas, room = {}, cap
    for index, name in enumerate(sorted(by_source, key=lambda n: len(by_source[n]))):
        quotas[name] = min(len(by_source[name]), room // (len(by_source) - index))
        room -= quotas[name]
    return [row for name, rows in by_source.items() for row in sorted(rows, key=lambda r: pick_order(r["group"]))[:quotas[name]]]


def make_record(topic: str, source: str, group: str, prompt: str) -> dict:
    uid = hashlib.sha256(f"{source}|{norm_key(prompt)}".encode()).hexdigest()[:16]
    return {"id": uid, "behavior": topic, "source": source, "group": norm_key(group), "prompt": prompt}


def run_source(pool, audit: Audit, seen: dict[str, set], topic: str, source: str, cleaner: str, loader) -> list[dict]:
    items = loader()
    results = pool.map(run_cleaner, [cleaner] * len(items), items, chunksize=2000)
    out, raw = [], len(items)
    for item, result in zip(items, results):
        if result[0] == "drop":
            audit.drop(source, result[1], result[2])
            continue
        _, prompt, group = result
        key = norm_key(prompt)  # exact repeats are dropped; look-alikes that share a group stay in one split
        if key in seen[topic]:
            audit.drop(source, "duplicate", prompt)
            continue
        seen[topic].add(key)
        audit.keep(source)
        out.append(make_record(topic, source, group, prompt))
    audit.counts[source]["raw"] = raw
    return out


def split_rows(rows: list[dict]) -> dict[str, list[dict]]:
    parts: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for row in rows:
        parts[split_of(row["group"])].append(row)
    groups = [{r["group"] for r in part} for part in parts.values()]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("a group landed in two splits")
    return parts


def topic_report(topic: str, audit: Audit, sources: list[str], kept: list[dict], final: list[dict],
                 parts: dict[str, list[dict]]) -> dict:
    rejections: Counter = Counter()
    for name in sources:
        rejections.update({k: v for k, v in audit.counts[name].items() if k not in ("kept", "raw")})
    samples = sorted(final, key=lambda r: pick_order(r["group"]))[:SAMPLE_PROMPTS]
    return {
        "sources": sources,
        "raw_rows": sum(audit.counts[name]["raw"] for name in sources),
        "kept_rows": len(kept),
        "rejected_by_reason": dict(rejections),
        "over_cap_left_out": len(kept) - len(final),
        "final_rows": len(final),
        "final_by_source": dict(Counter(r["source"] for r in final)),
        "splits": {name: len(part) for name, part in parts.items()},
        "sample_prompts": [r["prompt"][:500] for r in samples],
    }


def build(topics: tuple[str, ...], out: Path, cap: int) -> dict:
    audit, seen = Audit(), defaultdict(set)
    chosen = [s for s in SOURCES if s[0] in topics]
    workers = max(2, (os.cpu_count() or 4) - 2)
    report = {"research_only": True, "cap_per_topic": cap, "split_percent": {
        "train": 100 - TEST_PERCENT - DEV_PERCENT, "dev": DEV_PERCENT, "test": TEST_PERCENT}, "topics": {}}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        by_topic: dict[str, dict[str, list[dict]]] = {t: {} for t in topics}
        for topic, source, cleaner, loader in chosen:
            by_topic[topic][source] = run_source(pool, audit, seen, topic, source, cleaner, loader)
    for topic in topics:
        kept = [r for rows in by_topic[topic].values() for r in rows]
        final = balanced_cap(by_topic[topic], cap)
        parts = split_rows(final)
        for name, part in parts.items():
            write_jsonl(out / topic / f"{name}.jsonl", sorted(part, key=lambda r: r["id"]))
        report["topics"][topic] = topic_report(topic, audit, list(by_topic[topic]), kept, final, parts)
    report["sources"] = {name: dict(counts) for name, counts in audit.counts.items()}
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_jsonl(out / "removed_samples.jsonl", audit.samples)
    (out / "RESEARCH_ONLY.md").write_text(
        "Research use only. Some sources are non-commercial or have unclear rights. "
        "Do not publish or redistribute this data.\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--topics", default="v2", help="v1, v2 or a comma list of topic names")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--cap", type=int, default=CAP)
    args = parser.parse_args()
    report = build(resolve_topics(args.topics), args.out, args.cap)
    print(json.dumps({t: {k: v for k, v in info.items() if k in ("raw_rows", "kept_rows", "final_rows", "splits")}
                      for t, info in report["topics"].items()}, indent=1))


if __name__ == "__main__":
    main()
