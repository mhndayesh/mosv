"""Build balanced router prompts with separate wording for test. RESEARCH USE ONLY.

Run on Windows: python -m router.dataset
Test prompts use unseen topics AND unseen phrasing, so the router cannot pass by
memorising one dataset's template.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

from behaviors.prepare_data import ADULT, CLEAN, RAW, is_test, norm_key
from router import CLASSES

OUT = CLEAN.parent / "router" / "prompts.jsonl"
REVIEW = CLEAN.parent / "router" / "alpaca_review.json"  # Blind labels for every Alpaca task prompt.
PER_CLASS_TRAIN = 800
PER_CLASS_TEST = 200
VAL_PERCENT = 15
SEED = 13

TASK_WORDS = re.compile(
    r"\b(story|stories|tale|poem|narrative|fiction|teach\w*|lesson\w*|student\w*|pupil\w*|child\w*"
    r"|kids?|class|software|program\w*|code|coding|app|apps|application\w*|repo\w*|project\w*"
    r"|script\w*|function\w*|website\w*|algorithm\w*|web|webpage\w*|html|css|sql|database\w*|schema\w*"
    r"|apis?|regex|crawler\w*|python|javascript|java|server\w*|backend|frontend|chatbot\w*)\b",
    re.IGNORECASE,
)
# Could be a software build, so it is neither a clean `none` nor a clean engineering example.
UNCLEAR_BUILD = re.compile(
    r"^(?:please )?(?:design|develop|create|build|construct|implement|generate|write|make)\b.*"
    r"\b(?:system|model|form|assistant|bot|platform|dashboard|plugin|extension|interface|tool|game|network)\b",
    re.IGNORECASE,
)
ENGINEERING = re.compile(r"^Create a new software project for (.+?)\. (.*)$", re.DOTALL)
STORY_PREFIX = "Write a short story about this idea: "
TEACHING = re.compile(r"^Teach (.+) to a (.+) student\.$", re.DOTALL)

TEMPLATES = {
    "engineering": {
        "train": ["Create a new software project for <X>", "Set up a repository for this program: <X>",
                  "Build a simple application: <X>", "I need a well organized codebase for <X>",
                  "Plan and scaffold this tool for me: <X>", "Can you help me start a new coding project? <X>",
                  "How should I structure a repo for this app? <X>", "Please organize a new program for: <X>",
                  "Start a project from scratch: <X>", "What would the project setup look like for <X>",
                  "Develop this app step by step: <X>", "Make me a small program. <X>",
                  "Would you bootstrap a clean repo for this idea? <X>",
                  "My team needs this program engineered properly: <X>",
                  "Get this software project off the ground: <X>"],
        "val": ["Could you lay out the code project for this? <X>", "We want to build this software: <X>",
                "Draft the project layout for <X>"],
        "test": ["Spin up a fresh repository for this concept: <X>",
                 "Our startup wants this built the right way: <X>",
                 "Lay the groundwork for this software: <X>"],
    },
    "story": {
        "train": ["Write a short story about this idea: <X>", "Tell me a story based on this: <X>",
                  "Write fiction about the following: <X>", "Compose a tale from this premise: <X>",
                  "Can you narrate something inspired by this? <X>", "Could you tell a story where this happens? <X>",
                  "Story idea: <X>", "Please turn this into a short story: <X>",
                  "I want a bedtime story about this: <X>", "How would this play out as a story? <X>",
                  "Create a narrative around this: <X>", "Spin a yarn about this: <X>",
                  "Would you craft a fictional piece around this? <X>",
                  "Here's a premise, turn it into a story please: <X>", "What story could you tell from this? <X>"],
        "val": ["Can you write me a story from this prompt? <X>", "Make a short fiction piece: <X>",
                "Imagine this and tell it as a tale: <X>"],
        "test": ["Weave a fictional account of this: <X>",
                 "Got a premise for you, make it into a narrative: <X>",
                 "Could this become a story? Please try: <X>"],
    },
    "teaching": {
        "train": ["Teach <X> to a <L> student.", "Give a lesson on <X> for a <L> pupil.",
                  "Prepare a class about <X> for <L> children.", "Help a <L> child understand <X>.",
                  "How would you explain <X> to a young <L> learner?", "Can you teach my <L> kid about <X>?",
                  "Explain <X> to a <L> student step by step.", "Please make a lesson about <X> for <L>.",
                  "My <L> child is learning <X>. Can you teach it?", "Walk a <L> pupil through <X>.",
                  "What is the best way to teach <X> to <L> kids?", "Lesson request: <X> for <L>.",
                  "Could you tutor a <L> child on <X>?", "Run a short class for <L> kids covering <X>.",
                  "What would a good <L> lesson on <X> look like?"],
        "val": ["Could you show a <L> student how <X> works?", "I teach <L>. Plan today's lesson on <X>.",
                "How do I help <L> children learn <X>?"],
        "test": ["Coach a <L> youngster on <X>.", "Deliver a short <L> classroom session on <X>.",
                 "If you were a <L> teacher, how would you cover <X>?"],
    },
}

OPENERS = ("", "Please ", "Can you ", "Could you ", "I need you to ", "I'd like you to ", "Would you please ",
           "Hey, ", "Quick request: ", "For me, ", "Help me ", "Today, ")
REQUESTS = {
    "engineering": (
        "create a project for <X>", "set up a codebase for <X>", "build an app for <X>", "build this program: <X>",
        "start a repo for <X>", "scaffold a project for <X>", "organize a software project for <X>",
        "structure the files for this app: <X>", "write a simple program for <X>", "develop a small tool for <X>",
        "plan the architecture of <X>", "design the project structure for <X>", "initialize a repository for <X>",
        "put together a new application: <X>", "make a working prototype of <X>",
        "prepare a clean project skeleton for <X>", "code this up as a project: <X>",
        "create the folder structure for <X>", "turn this idea into software: <X>", "implement this application: <X>"),
    "story": (
        "write a story about <X>", "tell a story about <X>", "write a short story on this: <X>",
        "create a story from this: <X>", "compose a story based on <X>", "narrate a tale about <X>",
        "make up a story about <X>", "invent a tale from this idea: <X>", "write a fairy tale inspired by <X>",
        "write a fable about <X>", "turn this into fiction: <X>", "tell me a tale of <X>",
        "generate a short story about <X>", "write a creative story using this: <X>",
        "develop this into a story: <X>", "write the story of <X>", "create a fictional story about <X>",
        "come up with a story for <X>", "tell a bedtime tale about <X>", "write a little story around <X>"),
    "teaching": (
        "teach <X> to a <L> student", "explain <X> to a <L> student", "give a <L> class a lesson on <X>",
        "create a lesson on <X> for <L>", "write a lesson plan on <X> for <L>", "teach my <L> child about <X>",
        "help a <L> pupil learn <X>", "show <L> children how <X> works", "prepare a <L> lesson about <X>",
        "break down <X> for a <L> learner", "guide a <L> student through <X>", "make <X> easy for a <L> child",
        "plan a lesson about <X> for <L> pupils", "introduce <X> to <L> kids", "teach <L> students the basics of <X>",
        "explain <X> simply to a <L> kid", "build a <L> lesson around <X>", "design a class on <X> for <L>",
        "go over <X> with a <L> student", "develop a lesson for <L> about <X>"),
}


def expand_wording(label: str) -> list[str]:
    """Every opener with every request, for training only, never val or test wording."""
    blocked = set(TEMPLATES[label]["val"]) | set(TEMPLATES[label]["test"])
    made = {(opener + request)[0].upper() + (opener + request)[1:] for opener in OPENERS for request in REQUESTS[label]}
    return sorted((set(TEMPLATES[label]["train"]) | made) - blocked)


for _label in TEMPLATES:
    TEMPLATES[_label]["train"] = expand_wording(_label)

ASK = r"^(?:please |can you |could you )?"
STORY_ASK = re.compile(  # The story must be the thing requested, not "a title for a story".
    ASK + r"(?:write|compose|create|generate|tell|make up|craft|invent|narrate)\s+(?:me\s+)?"
    r"(?:(?!for\b|of\b|about\b|to\b|that\b|in\b|which\b|with\b)[\w'-]+\s+){0,5}"
    r"(?:story|stories|tale|fable|fairy tale)\b", re.IGNORECASE)
NOT_STORY = re.compile(r"opening line|first line|\btitle\b|story idea|summar|\breview\b|\boutline\b|\bscript\b"
                       r"|headline|news story|\bpoem\b|new character|\bscenario\b|story of why", re.I)
ENGINEERING_ASK = re.compile(
    ASK + r"(?:build|create|develop|design|implement|make|write)\s+an?\s+"
    r"(?:(?!to\b|for\b|about\b|of\b|on\b|with\b)[\w+#-]+\s+){0,4}"
    r"(?:program|app|application|website|chatbot|software|tool|api)\b", re.IGNORECASE)
NOT_ENGINEERING = re.compile(r"\b(?:marketing|question|logo|slogan|plan|survey|ad|advertisement|tagline|homepage"
                             r"|report|children|kids|lesson|story|loyalty|rewards?|customers?|fitness|workout"
                             r"|exercise|training|mentor\w*|volunteer|education|study|diet|recycling"
                             r"|job|form|flow|menu|endpoint)\b", re.IGNORECASE)
TEACHING_ASK = re.compile(
    ASK + r"(?:(?:explain|teach)\b.*\bto (?:an? )?(?:child|children|kids?|\d+[- ]year[- ]old|young|elementary|primary)\b"
    r"|(?:create|design|write|develop|prepare|plan)\s+an?\s+(?:[\w-]+\s+){0,2}lesson(?: plan)?\b)", re.IGNORECASE)
NEAR_MISS = re.compile(r"^(?:edit|rewrite|summarize|translate|classify|identify|find|what|who|why|when|where"
                       r"|how many|name|list|define|compare|correct|is|are|does)\b", re.IGNORECASE)
MAX_ALPACA_TRAIN = 250
MAX_ALPACA_HELD_OUT = 100
NEAR_MISS_TRAIN = 400
NEAR_MISS_TEST = 40
TUNE_NONE = 2000
TUNE_NEAR_MISS = 400
USED_TEST_NONE = 160  # Earlier test runs already looked at these, so they are skipped.

HARD_NEGATIVES = (
    "Find the bug in this line: for i in range(len(x)) print(x[i])",
    "What does the Python len function return?",
    "Convert 45 degrees Celsius to Fahrenheit.",
    "Summarize the plot of Romeo and Juliet in one sentence.",
    "Who wrote the novel Pride and Prejudice?",
    "Is The Old Man and the Sea a good book?",
    "Explain photosynthesis to a biology professor.",
    "What is the capital of Australia?",
    "Translate 'good morning' into Spanish.",
    "How many students attend Harvard each year?",
    "What is the average salary of a school teacher in Canada?",
    "List three popular programming languages.",
    "What year was GitHub founded?",
    "Fix the grammar: she dont like apples.",
    "What is the difference between a virus and a bacterium?",
    "Give me a recipe for lentil soup.",
    "Write a polite email asking for a day off.",
    "What is 17 times 23?",
    "Recommend a laptop for video editing.",
    "What causes the seasons on Earth?",
    "Rewrite this sentence to sound formal: gimme the report.",
    "Which is faster, a list or a set lookup in Python?",
    "Name the main characters in Harry Potter.",
    "Compare renting and buying a house.",
    "What does HTTP status code 404 mean?",
    "Define the word resilience.",
    "How do vaccines work?",
    "What is a good stretch for lower back pain?",
    "Who was the first person on the Moon?",
    "Suggest a name for a coffee shop.",
)

# Written once, before seeing results. Never used for training, tuning or picking settings.
FINAL_TEST = {
    "engineering": (
        "I want to make a little expense tracker in Python, how do I set up the whole project?",
        "new side project: a CLI that renames my photos by date. get me started properly",
        "Help me kick off a small Flask app for tracking my gym sessions.",
        "we're building a todo app, what should the repo look like from day one",
        "Can you set up a tiny weather dashboard project with tests and a README?",
        "I have an idea for a recipe manager app. Walk me through creating it as a real project.",
        "Start me a Node project for a URL shortener.",
        "Make a clean project for a command line pomodoro timer.",
        "need a starter repo for a markdown note taking tool",
        "How do I organize the code for a simple inventory program for my shop?",
        "Put together a basic chat server project I can extend later.",
        "Set up a Rust project for a file backup utility.",
        "I'd like to build a flashcard app from scratch, with proper folders and tests.",
        "Create the initial codebase for a budget planner.",
        "let's build a small bookmark manager, begin with the project setup",
    ),
    "story": (
        "Tell me about a lighthouse keeper who finds a message in a bottle, as a story.",
        "write me something fictional about two rival bakers in a tiny town",
        "I'd love a short story where a dog gets lost at the fair and finds its way home.",
        "Once upon a time there was a clockmaker... please continue this into a full story.",
        "Make up a tale about a girl who can talk to the wind.",
        "short story please: an astronaut wakes up alone on the ship",
        "Can I get a bedtime story about a shy dragon?",
        "Write a story where a thief steals the wrong suitcase.",
        "give me a spooky tale about an old elevator",
        "Tell a story about grandfather's last fishing trip.",
        "Write a little adventure story set on a pirate ship.",
        "I need a fictional story about a robot who wants to be a chef.",
        "Invent a fairy tale about a king who lost his shadow.",
        "Story time! A kid discovers a door in the back of the library.",
        "Could you write a story about two sisters reuniting after the war?",
    ),
    "teaching": (
        "My 7 year old doesn't get fractions, can you teach her?",
        "Teach my son how to tell time, he's in first grade.",
        "Explain why the moon changes shape to a second grader.",
        "I'm a primary teacher, build me a lesson on the water cycle for 8 year olds.",
        "how would you teach a 6 year old to add two numbers",
        "Give my daughter a simple lesson on nouns and verbs, she is nine.",
        "Help a young child learn the days of the week.",
        "Make a class for third graders about how plants grow.",
        "Can you explain magnets to a little kid in a lesson?",
        "Teach kindergarten kids about shapes.",
        "Lesson for grade 2: counting money.",
        "Show a five year old how to spell their name, step by step.",
        "I homeschool my kids, please teach them what a volcano is, they're seven.",
        "Prepare a primary school lesson about healthy food.",
        "Walk a fourth grader through long division.",
    ),
    "none": (
        "What's the boiling point of water at high altitude?",
        "Write a haiku about autumn rain.",
        "Draft an email to my landlord about a broken heater.",
        "Write a 5-paragraph essay on the causes of World War I.",
        "Create a dialogue between a customer and a waiter.",
        "How do I remove a coffee stain from a shirt?",
        "Write a poem about my mother for her birthday.",
        "Give me five tips for a job interview.",
        "Summarize the French Revolution in three sentences.",
        "What does this error mean: list index out of range?",
        "Make a slide outline about renewable energy for my boss.",
        "Explain inflation to an economist.",
        "Which movie won best picture in 1998?",
        "Write a cover letter for a nursing job.",
        "Plan a three day trip to Rome.",
    ),
}
OTHER_WRITING = re.compile(
    r"^(?:please )?(?:write|create|compose|generate|draft|construct|design|make|produce)\b.*"
    r"\b(?:poems?|haikus?|limericks?|sonnets?|essays?|dialogues?|dialogs?|conversations?|emails?|letters?|speech"
    r"|presentations?|powerpoint|blog|articles?|songs?|lyrics|tweets?|slogans?|summary|recipes?|reports?|exercises?"
    r"|taxonomy|scenarios?|outline|survey|questionnaire)\b", re.IGNORECASE)
UNCLEAR_STORY = re.compile(r"\b(?:allegor\w*|fanfic\w*|prologue|epilogue|scene|plot|myth|legend|anecdote|parable"
                           r"|screenplay|novel|character\w*|narrat\w*)\b", re.IGNORECASE)
OTHER_WRITING_TRAIN = 300
TUNE_OTHER_WRITING = 500


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def body_fields(label: str, prompt: str) -> tuple[str, str] | None:
    """Remove the dataset's fixed wording, keeping only the task content."""
    if label == "engineering":
        match = ENGINEERING.match(prompt)
        return (f"{match.group(1)}. {match.group(2)}", "") if match else None
    if label == "story":
        return (prompt[len(STORY_PREFIX):], "") if prompt.startswith(STORY_PREFIX) else None
    match = TEACHING.match(prompt)
    return (match.group(1), match.group(2)) if match else None


def render(label: str, split: str, body: str, level: str, key: str) -> str:
    pool = TEMPLATES[label][split]
    index = int(hashlib.sha256(key.encode()).hexdigest(), 16) % len(pool)
    return pool[index].replace("<L>", level).replace("<X>", body)


def is_val(group: str) -> bool:
    digest = hashlib.sha256(f"val|{norm_key(group)}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 100 < VAL_PERCENT


def make_row(label: str, split: str, group: str, text: str) -> dict:
    uid = hashlib.sha256(f"{label}|{text}".encode()).hexdigest()[:16]
    return {"id": uid, "split": split, "label": label, "group": norm_key(group), "text": text}


def alpaca_label(text: str) -> str | None:
    """Return a task label only when the request is clear. Unclear prompts are dropped."""
    hits = []
    if STORY_ASK.search(text) and not NOT_STORY.search(text):
        hits.append("story")
    if ENGINEERING_ASK.search(text) and not NOT_ENGINEERING.search(text):
        hits.append("engineering")
    if TEACHING_ASK.search(text):
        hits.append("teaching")
    return hits[0] if len(hits) == 1 else None


def is_near_miss(text: str) -> bool:
    """Mentions a task word but only asks a question or a small edit, so it is `none`."""
    return bool(TASK_WORDS.search(text) and NEAR_MISS.match(text)) and alpaca_label(text) is None


def alpaca_texts() -> list[str]:
    final = [text for texts in FINAL_TEST.values() for text in texts]
    seen, out = {norm_key(text) for text in (*HARD_NEGATIVES, *final)}, []
    for path in sorted((RAW / "none_alpaca").glob("*.parquet")):
        for row in pq.read_table(path, columns=["instruction", "input"]).to_pylist():
            text = " ".join((row["instruction"] or "").split())
            key = norm_key(text)
            if (row["input"] or "").strip() or len(text.split()) < 3 or key in seen or ADULT.search(text):
                continue
            seen.add(key)
            out.append(text)
    return out


def classify_one(text: str) -> tuple[str | None, bool]:
    return alpaca_label(text), is_near_miss(text)


def classify_all(texts: list[str]) -> dict[str, tuple[str | None, bool]]:
    """Sort every prompt once, across all CPU cores."""
    workers = max(2, (os.cpu_count() or 4) - 2)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tags = dict(zip(texts, pool.map(classify_one, texts, chunksize=500)))
    return apply_review(tags, json.loads(REVIEW.read_text(encoding="utf-8")))


def apply_review(tags: dict, review: dict[str, str]) -> dict:
    """A rule-labeled task prompt stays only if the blind reviewer gave it the same label."""
    return {text: (None, False) if label and review.get(text) != label else (label, near)
            for text, (label, near) in tags.items()}


def alpaca_task_rows(label: str, texts: list[str], tags: dict) -> list[dict]:
    """Natural requests: most train the router, held-out ones get their own score."""
    items = [text for text in texts if tags[text][0] == label]
    random.Random(f"{SEED}-alpaca-{label}").shuffle(items)
    fit = [text for text in items if not is_test(text)][:MAX_ALPACA_TRAIN]
    held_out = [text for text in items if is_test(text)][:MAX_ALPACA_HELD_OUT]
    rows = [make_row(label, "val" if is_val(text) else "train", text, text) for text in fit]
    return rows + [make_row(label, "natural", text, text) for text in held_out]


def task_rows(label: str, natural_fit: int = 0) -> list[dict]:
    rng, rows = random.Random(f"{SEED}-{label}"), []
    for source, limit in (("train", PER_CLASS_TRAIN - natural_fit), ("test", PER_CLASS_TEST)):
        items = read_jsonl(CLEAN / label / f"{source}.jsonl")
        rng.shuffle(items)
        # Use a fresh slice of test topics wherever the dataset has enough of them.
        start = limit if source == "test" and len(items) >= 2 * limit else 0
        for item in items[start:start + limit]:
            fields = body_fields(label, item["prompt"])
            if fields is None:
                raise ValueError(f"Unparsed {label} prompt: {item['prompt'][:80]}")
            split = "test" if source == "test" else "val" if is_val(item["group"]) else "train"
            text = render(label, split, fields[0], fields[1], item["id"])
            rows.append(make_row(label, split, item["group"], text))
    return rows


def is_clean_none(text: str) -> bool:
    """Leave out anything that might be a software build or a piece of fiction."""
    return not (TASK_WORDS.search(text) or UNCLEAR_BUILD.search(text) or UNCLEAR_STORY.search(text))


def none_rows(texts: list[str], tags: dict) -> list[dict]:
    items = [text for text in texts if is_clean_none(text)]
    near = [text for text in texts if tags[text][1]]
    random.Random(f"{SEED}-none").shuffle(items)
    random.Random(f"{SEED}-near").shuffle(near)
    near_fit, near_held = [t for t in near if not is_test(t)], [t for t in near if is_test(t)]
    plain_fit, plain_held = [t for t in items if not is_test(t)], [t for t in items if is_test(t)]
    writing_fit = [t for t in plain_fit if OTHER_WRITING.search(t)]
    near_train = near_fit[:NEAR_MISS_TRAIN]
    hard_train = near_train + writing_fit[:OTHER_WRITING_TRAIN]
    chosen = set(hard_train)
    train = hard_train + [t for t in plain_fit if t not in chosen][:PER_CLASS_TRAIN - len(hard_train)]
    used = set(train)
    tune_near = [t for t in near_fit if is_val(t) and t not in used][:TUNE_NEAR_MISS]
    tune_writing = [t for t in writing_fit if is_val(t) and t not in used][:TUNE_OTHER_WRITING]
    tune_hard = tune_near + tune_writing
    taken = used | set(tune_hard)
    tune = tune_hard + [t for t in plain_fit if is_val(t) and t not in taken][:TUNE_NONE - len(tune_hard)]
    test = near_held[NEAR_MISS_TEST:2 * NEAR_MISS_TEST]
    test += plain_held[USED_TEST_NONE:USED_TEST_NONE + PER_CLASS_TEST - len(test)]
    rows = [make_row("none", "val" if is_val(t) else "train", t, t) for t in train]
    rows += [make_row("none", "tune", t, t) for t in tune]
    return rows + [make_row("none", "test", t, t) for t in test]


def check_rows(rows: list[dict]) -> None:
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate prompt ids")
    for label in CLASSES:
        groups: dict[str, set] = {}
        for row in rows:
            if row["label"] == label and row["split"] not in ("hard", "final"):
                groups.setdefault(row["split"], set()).add(row["group"])
        splits = list(groups)
        for i, first in enumerate(splits):
            for second in splits[i + 1:]:
                if groups[first] & groups[second]:
                    raise ValueError(f"{label}: topic shared by {first} and {second}")


def main() -> None:
    texts, rows = alpaca_texts(), []
    tags = classify_all(texts)
    for label in ("engineering", "story", "teaching"):
        natural = alpaca_task_rows(label, texts, tags)
        rows += natural + task_rows(label, sum(row["split"] != "natural" for row in natural))
    rows += none_rows(texts, tags)
    rows += [make_row("none", "hard", text, text) for text in HARD_NEGATIVES]
    rows += [make_row(label, "final", text, text) for label, items in FINAL_TEST.items() for text in items]
    check_rows(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = Counter((row["split"], row["label"]) for row in rows)
    print(json.dumps({f"{s}/{l}": n for (s, l), n in sorted(counts.items())}, indent=1))
    print("total", len(rows))


if __name__ == "__main__":
    main()
