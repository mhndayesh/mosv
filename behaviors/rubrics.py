"""Rule-based process checklists: did a response follow the structure?

They judge structure and order only, never quality or correctness. Credit needs
real sentences in the right place, and an answer that ignores the prompt loses
its score, so one memorized template cannot earn reward on every task. Word lists
are still a proxy: compare with an independent judge before trusting a high score.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

MIN_WORDS = 40
MIN_UNIQUE_WORD_SHARE = 0.2
MAX_SENTENCE_REPEATS = 3
MIN_SENTENCE_WORDS = 4
MAX_TREE_LINE_CHARS = 70
MAX_TREE_LINE_WORDS = 4
MIN_TREE_LINES = 5
MAX_WORDS_PER_SENTENCE = 16
MAX_LONG_WORD_SHARE = 0.15
LONG_WORD_LETTERS = 9
MIN_TOPIC_OVERLAP = 0.25

WORD = re.compile(r"[A-Za-z']+")
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")
STOPWORDS = frozenset(
    "about after again also because been before being could every from have into just like "
    "made make many more most much must only other over said some still such than that their "
    "them then there these they this through very want wanted were what when where which "
    "while will with would your".split()
)
REQUEST_WORDS = frozenset(
    "write tell story stories short idea create project software program application simple build teach "
    "lesson student students primary explain help please give need want based following tale fiction "
    "repository repo code codebase class pupil child children learner young setup start small".split()
)


def rx(body: str, flags: int = re.IGNORECASE) -> re.Pattern:
    return re.compile(rf"\b(?:{body})\b", flags)


REQUIREMENTS_HEAD = re.compile(r"^(?:requirements?|goals?|scope|user stor(?:y|ies)|features?|objectives?)\b", re.I)
DESIGN_HEAD = re.compile(r"^(?:design|architecture|components?|modules?|data flow|technical design)\b", re.I)
SETUP_HEAD = re.compile(r"^(?:setup|set up|getting started|installation|install|how to run|run|usage|steps)\b", re.I)
FILE_TOKEN = re.compile(
    r"(?<![\w.-])(?:[\w.-]+/)*(?:[\w-]+\.(?:py|js|ts|jsx|tsx|java|go|rs|rb|cpp|h|md|txt|toml|json"
    r"|ya?ml|cfg|ini|html|css|sh|lock|mod)|\.gitignore|\.env\.example|Dockerfile|Makefile|LICENSE|Gemfile"
    r"|[\w-]+/)(?!\w)",
    re.IGNORECASE,
)
CODE_FILE = re.compile(r"(?<![\w-])([\w-]+)\.(?:py|js|ts|jsx|tsx|go|rs|java|cpp|rb)\b", re.I)
TEST_FILE = re.compile(r"(?<![\w-])(?:test_([\w-]+)|([\w-]+)[._](?:test|spec))\.\w+", re.I)
SOURCE_DIR = re.compile(r"(?<![\w-])(?:src|app|lib|pkg|cmd)/", re.I)
TEST_DIR = re.compile(r"(?<![\w-])(?:tests?|spec)/", re.I)
README = re.compile(r"\bREADME(?:\.\w+)?\b", re.I)
GITIGNORE = re.compile(r"\.gitignore", re.I)
STEP_LINE = re.compile(r"^\s*(?:step\s*)?\d+[.):]", re.IGNORECASE | re.MULTILINE)
SETUP_VERB = rx(r"install|create|init(?:ialize)?|configure|run|test|clone|set up|add")
DEPENDENCY_TOOLS = {
    "requirements.txt": rx("pip|uv"), "pyproject.toml": rx("pip|uv|poetry"), "package.json": rx("npm|yarn|pnpm"),
    "cargo.toml": rx("cargo"), "go.mod": rx("go"), "pom.xml": rx("mvn|maven"), "gemfile": rx("bundle|bundler"),
}
TEST_COMMAND = rx(r"pytest|unittest|npm test|yarn test|cargo test|go test|mvn test|rspec|run the tests?")

PLACE = rx(r"village|town|city|forest|house|home|school|ship|room|kingdom|castle|garden|farm|river"
           r"|mountain|island|street|park|planet|station|beach|desert|sea|lake|woods?|land|world")
CHARACTER = rx(r"he|she|they|i|girl|boy|man|woman|king|queen|child|friend|mother|father|named")
PROBLEM = rx(r"problem|trouble|danger|threat\w*|lost|missing|stolen|broke\w*|afraid|fear\w*|wanted"
             r"|needed|had to|could not|couldn't|cannot|can't|worried|attack\w*|trapped|sick"
             r"|curse\w*|storm|challenge")
ATTEMPT = rx(r"tried|try|tries|attempt\w*|again|failed|fail|however|but|instead|searched|decided"
             r"|set out|climbed|fought|another")
CLIMAX = rx(r"suddenly|finally|at last|at that moment|just then|all at once|faced?|confront\w*"
            r"|battle|fight|leap\w*|scream\w*|roar\w*|crash\w*|desperate\w*|last chance")
RESOLUTION = rx(r"saved|solved|safe|returned|home|peace|forgave|forgiven|together|learned|realized"
                r"|understood|happy|relieved|found|ended|no longer|from then on|after that")
CLOSING = rx(r"in the end|finally|at last|from then on|from that day|ever after|never again|no longer"
             r"|together|learned|realized|happy|peace|stood")

OBJECTIVE = rx(r"today,? we|we will learn|we are going to learn|you will learn|our goal|objective"
               r"|by the end|learn about")
PRIOR = rx(r"remember when|do you remember|you already know|you know|have you ever|think about"
           r"|last time|yesterday|like when")
FIRST_WORD = rx(r"first|step 1|to start")
NEXT_WORD = rx(r"next|second|then|after that|step 2")
LAST_WORD = rx(r"then|finally|last|lastly|third|step 3")
EXAMPLE = rx(r"for example|for instance|such as|imagine|like this|here is an example|let's say")
MODELING = rx(r"watch me|let me show|i will show|i'll show|see how i|look at how|here is how|show you")
PRACTICE = rx(r"let's try|lets try|try it|your turn|now you|you try|can you try|let's do|practice")
CHECK_WORD = rx(r"what|why|how|which|who|where|can you|do you")
RECAP = rx(r"today we learned|we learned|to recap|in summary|remember that|let's review|review"
           r"|what we learned|so now you know")


@dataclass(frozen=True)
class Score:
    behavior: str
    score: float
    checks: dict[str, bool]
    degenerate: bool
    topic: float = 1.0


def words(text: str) -> list[str]:
    return [word.lower() for word in WORD.findall(text)]


def sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_BREAK.split(text) if part.strip()]


def real_sentences(text: str) -> list[str]:
    """Fragments of a few words cannot earn credit for a story or lesson stage."""
    return [part for part in sentences(text) if len(words(part)) >= MIN_SENTENCE_WORDS]


def content_words(text: str) -> set[str]:
    return {word for word in words(text) if len(word) >= 4 and word not in STOPWORDS}


def is_degenerate(text: str) -> bool:
    """Blocks empty, looping or keyword-stuffed output from earning any reward."""
    tokens = words(text)
    if len(tokens) < MIN_WORDS or len(set(tokens)) / len(tokens) < MIN_UNIQUE_WORD_SHARE:
        return True
    repeats = Counter(" ".join(words(line)) for line in sentences(text) if len(words(line)) >= 3)
    return bool(repeats) and repeats.most_common(1)[0][1] > MAX_SENTENCE_REPEATS


def first(pattern: re.Pattern, text: str, start: int = 0) -> int | None:
    match = pattern.search(text, start)
    return match.start() if match else None


def matching(pattern: re.Pattern, items: list[str]) -> list[int]:
    return [index for index, item in enumerate(items) if pattern.search(item)]


def in_order(*positions: int | None) -> bool:
    if any(position is None for position in positions):
        return False
    return all(a < b for a, b in zip(positions, positions[1:]))


def next_after(indices: list[int], position: int | None) -> int | None:
    if position is None:
        return None
    return next((index for index in indices if index > position), None)


def section_at(text: str, heading: re.Pattern) -> int | None:
    """Find a real heading line, not a word used in passing."""
    offset = 0
    for line in text.splitlines(keepends=True):
        clean = line.strip().lstrip("#*-_0123456789. ").rstrip(":*_ ")
        if heading.match(clean) and (len(clean.split()) <= 6 or ":" in line[:40]):
            return offset
        offset += len(line)
    return None


def tree_lines(text: str) -> list[str]:
    return [line for line in text.splitlines()
            if len(line) <= MAX_TREE_LINE_CHARS and len(line.split()) <= MAX_TREE_LINE_WORDS
            and FILE_TOKEN.search(line) and not STEP_LINE.match(line)]


def tests_match_source(tree_text: str) -> bool:
    tested = {(a or b).lower() for a, b in TEST_FILE.findall(tree_text)}
    code = {name.lower() for name in CODE_FILE.findall(tree_text)
            if not name.lower().startswith("test_") and not re.search(r"[._](?:test|spec)$", name, re.I)}
    return bool(tested & code)


def setup_steps(text: str, tree_at: int | None) -> list[str]:
    if tree_at is None:
        return []
    return [line for line in text[tree_at:].splitlines() if STEP_LINE.match(line)]


def commands_fit_files(steps: list[str], tree_text: str) -> bool:
    joined, lowered = "\n".join(steps), tree_text.lower()
    tools = [tool for name, tool in DEPENDENCY_TOOLS.items() if name in lowered]
    return bool(tools) and any(tool.search(joined) for tool in tools) and bool(TEST_COMMAND.search(joined))


def has_body(text: str, start: int | None, end: int | None) -> bool:
    return start is not None and end is not None and len(words(text[start:end])) >= 6


def engineering_checks(text: str) -> dict[str, bool]:
    tree = tree_lines(text)
    tree_text = "\n".join(tree)
    tree_at = text.find(tree[0]) if tree else None
    requirements, design = section_at(text, REQUIREMENTS_HEAD), section_at(text, DESIGN_HEAD)
    setup, steps = section_at(text, SETUP_HEAD), setup_steps(text, tree_at)
    return {
        "requirements_section": has_body(text, requirements, design if in_order(requirements, design) else tree_at),
        "design_section": in_order(requirements, design) and has_body(text, design, tree_at),
        "file_tree": len(tree) >= MIN_TREE_LINES,
        "source_and_tests_separate": bool(SOURCE_DIR.search(tree_text) and TEST_DIR.search(tree_text)),
        "tests_match_source": tests_match_source(tree_text),
        "readme": bool(README.search(tree_text)),
        "dependency_file": any(name in tree_text.lower() for name in DEPENDENCY_TOOLS),
        "gitignore": bool(GITIGNORE.search(tree_text)),
        "setup_steps": len(steps) >= 3 and sum(bool(SETUP_VERB.search(line)) for line in steps) >= 2,
        "commands_fit_files": commands_fit_files(steps, tree_text),
        "sections_in_order": in_order(requirements, design, tree_at, setup),
    }


def ending_echoes_problem(parts: list[str], problem: int | None) -> bool:
    if problem is None:
        return False
    start = max(problem + 1, len(parts) - math.ceil(len(parts) * 0.2))
    return len(content_words(parts[problem]) & set(words(" ".join(parts[start:])))) >= 2


def story_order_ok(setting, problem, attempt, climax, resolution) -> bool:
    if setting is None or problem is None or setting > problem:
        return False
    return in_order(problem, attempt, climax, resolution)


def closes_at_end(parts: list[str]) -> bool:
    hits = matching(CLOSING, parts)
    return bool(hits) and hits[-1] / len(parts) >= 0.75


def story_checks(text: str) -> dict[str, bool]:
    parts = real_sentences(text)
    count = max(len(parts), 1)
    opening = " ".join(parts[:max(1, math.ceil(count * 0.2))])
    problems = matching(PROBLEM, parts)
    problem = problems[0] if problems else None
    climax = next((i for i in matching(CLIMAX, parts) if 0.6 <= i / count <= 0.9), None)
    attempts = [i for i in matching(ATTEMPT, parts)
                if problem is not None and i > problem and (climax is None or i < climax)]
    resolution = next_after(matching(RESOLUTION, parts), climax)
    setting = next(iter(matching(PLACE, parts)), None)
    return {
        "opening_character_and_setting": bool(PLACE.search(opening) and CHARACTER.search(opening)),
        "early_problem": problem is not None and problem / count <= 0.33,
        "rising_attempts": len(attempts) >= 2,
        "climax_in_late_middle": climax is not None,
        "resolution_after_climax": resolution is not None,
        "ending_addresses_problem": ending_echoes_problem(parts, problem),
        "stages_in_order": story_order_ok(setting, problem, attempts[0] if attempts else None, climax, resolution),
        "closes_at_end": closes_at_end(parts),
    }


def simple_language(text: str, parts: list[str]) -> bool:
    tokens = words(text)
    long_share = sum(len(word) >= LONG_WORD_LETTERS for word in tokens) / max(len(tokens), 1)
    return len(tokens) / max(len(parts), 1) <= MAX_WORDS_PER_SENTENCE and long_share <= MAX_LONG_WORD_SHARE


def markers_in_order(text: str) -> bool:
    start = first(FIRST_WORD, text)
    middle = first(NEXT_WORD, text, start + 1) if start is not None else None
    end = first(LAST_WORD, text, middle + 1) if middle is not None else None
    return in_order(start, middle, end)


def question_after(text: str, position: int | None) -> int | None:
    if position is None:
        return None
    offset = position
    for part in sentences(text[position:]):
        found = text.find(part, offset)
        if part.endswith("?") and CHECK_WORD.search(part):
            return found
        offset = found + len(part)
    return None


def teaching_checks(text: str) -> dict[str, bool]:
    parts = real_sentences(text)
    objective, example, modeling = first(OBJECTIVE, text), first(EXAMPLE, text), first(MODELING, text)
    practice = first(PRACTICE, text, modeling or 0)
    check = question_after(text, practice)
    recap = first(RECAP, text, int(len(text) * 0.75))
    prior = first(PRIOR, text)
    return {
        "states_objective": objective is not None and objective <= len(text) * 0.25,
        "connects_prior_knowledge": prior is not None and prior <= len(text) * 0.5,
        "small_steps_in_order": len(STEP_LINE.findall(text)) >= 3 or markers_in_order(text),
        "simple_language": simple_language(text, parts),
        "concrete_example": example is not None,
        "models_before_practice": in_order(modeling, practice),
        "guided_practice": practice is not None,
        "checks_after_practice": check is not None,
        "recap_at_end": recap is not None,
        "stages_in_order": in_order(objective, example, practice, recap),
    }


def topic_match(prompt: str | None, text: str) -> float:
    """1.0 when the answer uses the prompt's subject words, falling to 0 when it ignores them."""
    if not prompt:
        return 1.0
    wanted = content_words(prompt) - REQUEST_WORDS
    if not wanted:
        return 1.0
    return min(1.0, (len(wanted & content_words(text)) / len(wanted)) / MIN_TOPIC_OVERLAP)


CHECKLISTS = {"engineering": engineering_checks, "story": story_checks, "teaching": teaching_checks}


def score_response(behavior: str, text: str, prompt: str | None = None) -> Score:
    """Share of checks passed, in [0, 1], reduced when the answer ignores the prompt."""
    if behavior not in CHECKLISTS:
        raise ValueError(f"Unknown behavior: {behavior}")
    if is_degenerate(text):
        return Score(behavior, 0.0, {}, True, 0.0)
    checks = CHECKLISTS[behavior](text)
    topic = topic_match(prompt, text)
    return Score(behavior, topic * sum(checks.values()) / len(checks), checks, False, topic)
