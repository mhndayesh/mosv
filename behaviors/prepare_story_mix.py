"""Build one large, mixed story dataset from real user and human prompts. RESEARCH USE ONLY.

Run: python -m behaviors.prepare_story_mix
Reads data_review/raw_v3 and writes data_review/clean_v3/story. All sexual-themed prompts are removed from
every source, and WildChat's moderation flags are applied too. Removed sexual text is never saved.
Prompts already in the current story set are left out, so the old story test stays a fair check.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

from behaviors.prepare_data import DATA, norm_key, write_jsonl
from behaviors.prepare_topics import BROKEN, MAX_NON_ASCII_LETTERS, make_record, pick_order, split_of, tidy

RAW = DATA / "raw_v3"
OUT = DATA / "clean_v3" / "story"
PROSE_OUT = DATA / "clean_v3_prose" / "story"
CURRENT = DATA / "clean_v2" / "story"
TOPIC = "story"
MIN_WORDS, MAX_WORDS = 4, 400
SOURCE_SHARE = 0.30  # No source may exceed this share. At 0.25 the small sources limited the set to 21,076.
PREFIX_SHARE, MIN_PREFIX_CAP = 0.015, 25  # Cap on prompts sharing their first 4 words, per source.
GROUP_CAP = 3  # Prompts sharing their first 8 words.
SAMPLES_PER_REASON = 5
WORKERS = 12
MINOR_FLAG, MINOR_SCORE_LIMIT = "sexual/minors", 0.01
MODERATION_BLOCKED = "__moderation_blocked__"

CREATIVE = (r"(?:stor(?:y|ies)|fanfic\w*|fan[- ]fiction|fiction|novel(?:la)?|chapters?|scenes?|screenplay|episode|"
            r"poems?|poetry|haiku|sonnet|limerick|ballad|lyrics|songs?|rap|dialogues?|monologue|fairy ?tale|fable|"
            r"myth|legend|backstory|sitcom|sketch|script|worldbuilding|narrative|tale)")
REQUEST = re.compile(r"\b(?:write|writing|rewrite|continue|compose|tell|create|generate|draft|imagine|finish|"
                     r"make|give me|come up with|develop)\b[^.?!]{0,70}?\b" + CREATIVE + r"\b", re.I | re.S)
CODE = re.compile(r"```|\b(?:python|javascript|typescript|c\+\+|c#|sql|html|css|lua|bash|powershell|php|golang|regex|"
                  r"json|xml|docker|kubernetes|autohotkey|vba|source code|shell script|batch script|script (?:that|to)|"
                  r"(?:write|generate|create|fix) (?:me )?(?:a |an |the |some )?(?:code|function|program|algorithm|api)|"
                  r"code (?:for|to|that))\b", re.I)
NOT_STORY = re.compile(r"\b(?:midjourney|prompt generator|image prompts?|stable diffusion|product description|seo|"
                       r"cover letter|(?:my|a) resume|essay|blog post|press release|e-?mail|linkedin|instagram|tweet|"
                       r"youtube (?:title|description)|ad copy|marketing|business plan|translate|sales|lesson plan|"
                       r"research paper|user stor(?:y|ies)|acceptance criteria|product backlog|agile|thesis|"
                       r"dissertation|articles?|news|list of|recommend\w*|which ones)\b|^summar", re.I)
QUESTION = re.compile(r"^(?:what|who|when|where|why|which|how|is|are|does|did|do)\b", re.I)
WP_TAG = re.compile(r"^\s*\[\s*([a-z]{2,4})\s*\]\s*", re.I)
WP_KEEP = {"wp", "eu", "tt", "cw", "ff", "ip", "rf", "sp"}
OLD_WRAPPER = re.compile(r"^write a short story about this idea:\s*", re.I)

STORY_WORD = re.compile(r"\b(?:stor(?:y|ies)|tale|fanfic\w*|fan[- ]fiction)\b", re.I)
OTHER_FORMAT = re.compile(r"\b(?:poems?|poetry|haiku|sonnet|limerick|ballad|lyrics|songs?|rap|verses?|stanzas?|scripts?|"
                          r"screenplay|episode|sitcom|sketch|dialogues?|monologue|scenes?|outline|backstory)\b", re.I)
ALWAYS_BLOCK = re.compile(r"\b(?:loli\w*|shota\w*|cp|pedo\w*|paedo\w*)\b", re.I)
SEXUAL = re.compile(r"\b(?:nsfw|smut|hentai|sex\w*|nude\w*|naked|erotic\w*|porn\w*|fetish\w*|orgasm\w*|penis\w*|cock|dick|vagina\w*|pussy|"
                    r"breasts?|boobs?|tits|nipples?|lewd|horny|kink\w*|incest|rap(?:e|ed|es|ing)|moan\w*|arous\w*|"
                    r"seduc\w*|lingerie|masturbat\w*|foreplay|blowjob|handjob|cum|ejaculat\w*|grop\w*|molest\w*|"
                    r"undress\w*|strip(?:s|ped|ping|per|tease)?|panties|thong|lust\w*|spank\w*|bdsm|bondage)\b", re.I)
MINOR = re.compile(r"\b(?:underage\w*|minors?|child(?:ren)?|kids?|toddlers?|infants?|bab(?:y|ies)|preteens?|pre-teens?|"
                   r"tweens?|teen\w*|schoolgirls?|schoolboys?|high ?school\w*|middle ?school\w*|elementary|"
                   r"(?:little|young) (?:girl|boy)s?|daughters?|sons?|step-?daughters?|step-?sons?|niece|nephew|"
                   r"(?:[1-9]|1[0-7])[- ]?(?:years?|yrs?|yo|y/o)(?:[- ]old)?|(?:aged?) (?:[1-9]|1[0-7]))\b", re.I)


def is_prose_request(text: str) -> bool:
    """The story checklist scores a prose arc, so another requested format would be rewarded for being ignored."""
    return bool(STORY_WORD.search(text)) or not OTHER_FORMAT.search(text)


def is_blocked(text: str) -> bool:
    """True for sexual content combined with any sign of a minor, including ambiguous cases."""
    return bool(ALWAYS_BLOCK.search(text) or (SEXUAL.search(text) and MINOR.search(text)))


def basic_reason(text: str) -> str | None:
    if not text.strip():
        return "blank"
    if BROKEN.search(text):
        return "broken_text"
    letters = [c for c in text if c.isalpha()]
    if not letters or sum(not c.isascii() for c in letters) / len(letters) > MAX_NON_ASCII_LETTERS:
        return "not_english"
    count = len(text.split())
    return "too_short" if count < MIN_WORDS else "too_long" if count > MAX_WORDS else None


def request_reason(text: str) -> str | None:
    head = text[:500]
    if CODE.search(head):
        return "code"
    if NOT_STORY.search(head):
        return "not_story"
    if QUESTION.match(head) or not REQUEST.search(head):
        return "no_creative_request"
    return None


def clean_wp(text: str) -> tuple[str | None, str | None]:
    """Remove the Reddit tag and its spaced-out punctuation. Returns (prompt, drop reason)."""
    tag = WP_TAG.match(text or "")
    if tag and tag.group(1).lower() not in WP_KEEP:
        return None, "off_topic_tag"
    body = text[tag.end():] if tag else (text or "")
    body = body.replace("``", '"').replace("''", '"')
    body = re.sub(r"\s+([,.!?;:%)])", r"\1", body)
    body = re.sub(r"\s+(n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1", body)
    return tidy(body.replace("( ", "(")), None


def clean_one(item: tuple[str, str]) -> tuple:
    source, raw = item
    if raw == MODERATION_BLOCKED:
        return ("drop", "safety", "")
    if source == "writingprompts":
        prompt, reason = clean_wp(raw)
        if reason:
            return ("drop", reason, str(raw)[:300])
    else:
        prompt = tidy(raw)
    reason = basic_reason(prompt)
    if reason:
        return ("drop", reason, prompt[:300])
    if is_blocked(prompt):
        return ("drop", "safety", "")  # The text is never stored.
    if SEXUAL.search(prompt):
        return ("drop", "sexual_theme", "")
    if source != "writingprompts":
        reason = request_reason(prompt)
        if reason:
            return ("drop", reason, prompt[:300])
    return ("keep", prompt, " ".join(norm_key(prompt).split()[:8]))


# ---------- loaders: each returns the first user message of the relevant rows ----------

def parquet_rows(folder: str, columns: list[str], pattern: str = "**/*.parquet"):
    for path in sorted((RAW / folder).glob(pattern)):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2000, columns=columns):
            yield from batch.to_pylist()


def first_text(turns, role_key: str, text_key: str, role: str):
    for turn in turns or []:
        if turn.get(role_key) == role:
            content = turn.get(text_key)
            if isinstance(content, list):
                return " ".join(part.get("text") or "" for part in content if isinstance(part, dict))
            return content
    return None


def load_writingprompts() -> list:
    return [row["prompt"] for row in parquet_rows("euclaise__writingprompts", ["prompt"])]


def moderation_flagged(moderation) -> bool:
    """WildChat's own score for sexual content involving minors, on the first user message."""
    first = (moderation or [{}])[0] or {}
    scores = first.get("category_scores") or {}
    return bool((first.get("categories") or {}).get(MINOR_FLAG)) or (scores.get(MINOR_FLAG) or 0) >= MINOR_SCORE_LIMIT


def load_wildchat() -> list:
    columns = ["conversation", "language", "openai_moderation"]
    return [MODERATION_BLOCKED if moderation_flagged(row["openai_moderation"])
            else first_text(row["conversation"], "role", "content", "user")
            for row in parquet_rows("allenai__WildChat-1M", columns) if row["language"] == "English"]


def load_lmsys() -> list:
    wanted = {"creative writing", "text completion", "specific format writing"}
    return [first_text(row["conversations"], "from", "value", "human")
            for row in parquet_rows("OpenLeecher__lmsys_chat_1m_clean", ["conversations", "category"])
            if row["category"] in wanted]


def load_arena() -> list:
    columns = ["conversation_a", "category_tag", "language", "is_code"]
    return [first_text(row["conversation_a"], "role", "content", "user")
            for row in parquet_rows("lmarena-ai__arena-human-preference-140k", columns)
            if row["language"] == "en" and not row["is_code"]
            and ((row["category_tag"] or {}).get("creative_writing_v0.1") or {}).get("creative_writing")]


def load_sharegpt() -> list:
    path = RAW / "learnanything__sharegpt_v3_unfiltered_cleaned_split" / "ShareGPT_V3_unfiltered_cleaned_split.json"
    with path.open(encoding="utf-8") as file:
        return [first_text(row.get("conversations"), "from", "value", "human") for row in json.load(file)]


def load_no_robots() -> list:
    return [row["prompt"] for row in parquet_rows("HuggingFaceH4__no_robots", ["prompt", "category"])
            if row["category"] in {"Generation", "Rewrite", "Brainstorm"}]


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def load_small_human() -> list:
    dolly = load_jsonl(RAW / "databricks__databricks-dolly-15k" / "databricks-dolly-15k.jsonl")
    out = [row["instruction"] for row in dolly if row["category"] == "creative_writing"]
    out += [row["prompt"] for row in load_jsonl(RAW / "google__IFEval" / "ifeval_input_data.jsonl")]
    out += [row["conversation_input"][0]["content"] for row in parquet_rows("allenai__WildBench", ["conversation_input", "primary_tag"])
            if row["primary_tag"] == "Creative Writing"]
    for folder in ("OpenAssistant__oasst1", "OpenAssistant__oasst2"):
        out += [row["text"] for row in parquet_rows(folder, ["text", "role", "parent_id", "lang"], "data/*.parquet")
                if row["role"] == "prompter" and row["parent_id"] is None and row["lang"] == "en"]
    return out


SOURCES = (("writingprompts", load_writingprompts), ("wildchat", load_wildchat), ("lmsys", load_lmsys),
           ("arena", load_arena), ("sharegpt", load_sharegpt), ("no_robots", load_no_robots),
           ("small_human", load_small_human))


# ---------- assemble ----------

def current_keys() -> set[str]:
    keys = set()
    for name in ("train", "dev", "test"):
        keys |= {norm_key(OLD_WRAPPER.sub("", row["prompt"])) for row in load_jsonl(CURRENT / f"{name}.jsonl")}
    return keys


def cap_prefixes(rows: list[dict]) -> tuple[list[dict], int]:
    """Limit prompts that start with the same 4 words, so one template cannot dominate a source."""
    limit, used, kept = max(MIN_PREFIX_CAP, int(len(rows) * PREFIX_SHARE)), Counter(), []
    for row in sorted(rows, key=lambda r: pick_order(r["group"])):
        prefix = " ".join(norm_key(row["prompt"]).split()[:4])
        used[prefix] += 1
        if used[prefix] <= limit:
            kept.append(row)
    return kept, len(rows) - len(kept)


def balance_sources(sizes: dict[str, int], share: float = SOURCE_SHARE) -> dict[str, int]:
    """The largest per-source count so that no source exceeds `share` of the final total."""
    cap = max(sizes.values())
    for _ in range(100):
        total = sum(min(size, cap) for size in sizes.values())
        next_cap = int(total * share)
        if next_cap >= cap:
            break
        cap = next_cap
    return {name: min(size, cap) for name, size in sizes.items()}


def clean_source(pool, name: str, loader, seen: set, counts: Counter, samples: list) -> list[dict]:
    raw = [text for text in loader() if isinstance(text, str)]
    counts["raw"] = len(raw)
    rows = []
    for result in pool.map(clean_one, [(name, text) for text in raw], chunksize=2000):
        if result[0] == "drop":
            counts[result[1]] += 1
            if result[1] != "safety" and sum(s["reason"] == result[1] and s["source"] == name for s in samples) < SAMPLES_PER_REASON:
                samples.append({"source": name, "reason": result[1], "text": result[2]})
            continue
        key = norm_key(result[1])
        if key in seen:
            counts["duplicate_or_current_set"] += 1
            continue
        seen.add(key)
        rows.append(make_record(TOPIC, name, result[2], result[1]))
    rows, capped = cap_prefixes(rows)
    counts["repeated_template"] = capped
    return rows


def limit_groups(rows: list[dict]) -> list[dict]:
    used, kept = Counter(), []
    for row in sorted(rows, key=lambda r: r["id"]):
        used[row["group"]] += 1
        if used[row["group"]] <= GROUP_CAP:
            kept.append(row)
    return kept


def build(out: Path = OUT, prose_only: bool = False) -> dict:
    seen, by_source, audit, samples = current_keys(), {}, defaultdict(Counter), []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for name, loader in SOURCES:
            rows = limit_groups(clean_source(pool, name, loader, seen, audit[name], samples))
            by_source[name] = [row for row in rows if is_prose_request(row["prompt"])] if prose_only else rows
            audit[name]["other_format"] = len(rows) - len(by_source[name])
            print(json.dumps({"source": name, "kept": len(by_source[name]), **audit[name]}), flush=True)
    quotas = balance_sources({name: len(rows) for name, rows in by_source.items()})
    final = [row for name, rows in by_source.items()
             for row in sorted(rows, key=lambda r: pick_order(r["group"]))[:quotas[name]]]
    parts = {"train": [], "dev": [], "test": []}
    for row in final:
        parts[split_of(row["group"])].append(row)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        write_jsonl(out / f"{name}.jsonl", sorted(part, key=lambda r: r["id"]))
    write_jsonl(out / "removed_samples.jsonl", samples)
    report = {"research_only": True, "final_rows": len(final), "splits": {k: len(v) for k, v in parts.items()},
              "final_by_source": dict(Counter(r["source"] for r in final)), "source_audit": {k: dict(v) for k, v in audit.items()}}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prose", action="store_true", help="keep prose story requests, in data_review/clean_v3_prose")
    args = parser.parse_args()
    print(json.dumps(build(PROSE_OUT if args.prose else OUT, args.prose), indent=1))
