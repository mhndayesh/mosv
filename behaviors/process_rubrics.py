"""Process checklists for five everyday topics: debugging, math, explanation, email, how-to guides.

Same rules as rubrics.py: structure and order only, no model call, never truth. A stage needs a real sentence
(not a label), in order after the stage before it. Repeated sentences count once, and a sentence that only
announces a stage ("First I will explain the cause") earns nothing. Word lists are a proxy: compare with an
independent judge before trusting a high score.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from behaviors.rubrics import (
    CHECKLISTS, LESSON_SPLIT, REQUEST_WORDS, content_words, real_sentences, rx, simple_language, words,
)

STAGE_WORDS = 6
MIN_CALCULATIONS = 3
MAX_ANSWER_WORDS = 25
MAX_HEADING_WORDS = 5  # a line shorter than this is a label, not a sentence
MAX_RECAP_WORDS = 30
MAX_EMAIL_WORDS = 220
MIN_STEP_WORDS = 5
VERB_SHARE = 0.75

FENCE = re.compile(r"```.*?```", re.S)
LEAD = re.compile(r"^(?:[>#]+\s*|[-*•–—]\s+|\*{1,2}|_{2}|\d+[.)]\s+|step\s*\d+\s*[.):-]?\s*)+", re.I)
STEP_NUMBER = re.compile(r"^[\s>*#_-]*(?:step\s*)?(\d+)\s*[.):]\*{0,2}\s+(?=\S)", re.I)
CODE_ONLY = re.compile(r"`[^`]*`[.,;:]?")
CODEISH = re.compile(r"`[^`]+`|\b\w+\([^)]*\)|\w+\s*[=<>!]=?\s*\w+")
ANNOUNCE = rx(r"(?:i|we)(?:'ll| will| am going to|'m going to) (?:now |first |next |then )?(?:restate|reproduce"
               r"|explain|identify|verify|summari[sz]e|walk (?:you )?through|describe|outline|show|cover|discuss)")
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
OPERATOR = r"[-+−×x*/÷^]"
TERM = r"\$?\d[\d,.]*(?:\s*[A-Za-z%]+)?"  # a number, with its unit or variable: "12 boxes", "3x"
CALC = re.compile(rf"{TERM}\s*{OPERATOR}\s*{TERM}(?:\s*{OPERATOR}\s*{TERM})*\s*=\s*[-−]?\$?(\d[\d,]*(?:\.\d+)?)")


Test = Callable[[list[str], int], bool]


@dataclass(frozen=True)
class Stage:
    test: Test
    lo: float = 0.0  # earliest and latest place, as a share of the answer, where the stage may sit
    hi: float = 1.0


def parse(text: str) -> tuple[list[str], dict[int, int]]:
    """Sentences in reading order, markup removed, repeats dropped, plus the step number each numbered line opens."""
    text = FENCE.sub(lambda m: "`" + " ".join(m.group().strip("`").split()) + "`", text)
    items: list[str] = []
    steps: dict[int, int] = {}
    seen: set[str] = set()
    for line in text.splitlines():
        number = STEP_NUMBER.match(line)
        clean = LEAD.sub("", line.strip()).replace("*", "").replace("__", "").strip()
        opens_step = number is not None
        for part in LESSON_SPLIT.split(clean):
            key = " ".join(part.lower().split())
            if not key or key in seen or ANNOUNCE.search(part):
                continue
            seen.add(key)
            if items and not number and CODE_ONLY.fullmatch(part):
                items[-1] += " " + part  # a code line belongs to the sentence that introduces it
                continue
            if opens_step:
                steps[len(items)] = int(number.group(1))
                opens_step = False
            items.append(part)
    return items, steps


def chain(items: list[str], stages: list[Stage]) -> list[int | None]:
    """Index of each stage's sentence. A stage must come after the last stage that was found."""
    found: list[int | None] = []
    at = -1
    for stage in stages:
        index = next((i for i in range(at + 1, len(items))
                      if stage.lo <= (i + 0.5) / len(items) <= stage.hi and stage.test(items, i)), None)
        found.append(index)
        at = at if index is None else index
    return found


def cue(*needs: re.Pattern, min_words: int = STAGE_WORDS) -> Test:
    """A sentence of real length that matches every pattern."""
    return lambda items, i: len(words(items[i])) >= min_words and all(need.search(items[i]) for need in needs)


def both(*tests: Test) -> Test:
    return lambda items, i: all(test(items, i) for test in tests)


def never(*patterns: re.Pattern) -> Test:
    """A sentence that matches none of the patterns, so one sentence cannot serve two different stages."""
    return lambda items, i: not any(pattern.search(items[i]) for pattern in patterns)


def echo(prompt: str | None, need: int = 2) -> Test:
    """The sentence reuses the prompt's own subject words, so a stock sentence cannot earn it. Passes with no prompt."""
    wanted = content_words(prompt) - REQUEST_WORDS if prompt else set()
    return lambda items, i: not wanted or len(wanted & content_words(items[i])) >= min(need, len(wanted))


def shares_words(a: str, b: str, need: int = 2) -> bool:
    return len(content_words(a) & content_words(b)) >= need


def numbers(text: str) -> set[str]:
    return {number.replace(",", "") for number in NUMBER.findall(text)}


# --- debugging -------------------------------------------------------------------------------------------------

FAILURE = rx(r"error|exception|traceback|crash\w*|fail\w*|bug|broken|wrong|incorrect|unexpected|rais\w+|throw\w*"
             r"|hang\w*|freez\w*|segfault|stuck|not working|doesn't work|does not work|instead of"
             r"|returns? (?:none|null|nan|nothing|an empty|the wrong)|never (?:ends|finishes|stops)")
REPRO_WORD = rx(r"reproduc\w+|repro|recreate|steps? to (?:trigger|see|recreate)")
REPRO_WHEN = rx(r"every time|each time|whenever|when you|if you|as soon as|consistently|always")
ACTION = rx(r"run|call|pass|open|click|enter|send|start|type|use|load|import|execute|submit|set|give|feed|try|invoke")
CAUSE = rx(r"root cause|the cause|caused by|the reason|due to|because|happens when|happens because|results? from"
           r"|stems from|comes from|the problem is|the issue is|the bug is|which means")
REASON = rx(r"because|so|which means|therefore|that is why|that's why|as a result|this is why|so that|leads? to"
            r"|results? in|means that")
FIX = rx(r"fix|change|replace|add|remove|rename|use|set|move|wrap|update|patch|swap|should be|instead|initiali[sz]e"
         r"|convert|cast|guard|catch|return|increase|decrease|call|start|stop")
VERIFY = rx(r"re-?run|run (?:it|the \w+|them|that|this) again|run (?:the )?(?:tests?|script|code|program|suite)"
            r"|confirm\w*|verif\w+|check that|should (?:now )?(?:print|return|pass|work|show|output|succeed)"
            r"|no longer|passes|passing|works? (?:as expected|correctly)")
REGRESSION = rx(r"regression tests?|add an? (?:unit |new )?test|write an? (?:unit |new )?test|test case|assert\w*"
                r"|pytest|unittest|a test (?:that|for)")


def has_code(items: list[str], i: int) -> bool:
    return bool(CODEISH.search(items[i]))


def debugging_checks(text: str, prompt: str | None = None) -> dict[str, bool]:
    items, _ = parse(text)
    stages = [
        Stage(both(cue(FAILURE), echo(prompt)), hi=0.4),
        Stage(lambda it, i: cue(REPRO_WORD, ACTION)(it, i) or cue(REPRO_WHEN, ACTION, CODEISH)(it, i)),
        Stage(cue(CAUSE, min_words=8)),
        Stage(both(cue(FIX), has_code, never(VERIFY, REGRESSION))),
        Stage(cue(VERIFY), lo=0.4),
    ]
    failure, repro, cause, fix, verify = found = chain(items, stages)
    reasoning = sum(bool(REASON.search(item)) for item in items[(failure or 0) + 1:fix]) if None not in (failure, fix) else 0
    return {
        "restates_failure": failure is not None,
        "reproduction_steps": repro is not None,
        "root_cause_reasoning": cause is not None,
        "cause_is_reasoned": cause is not None and reasoning >= 2,
        "concrete_fix": fix is not None,
        "fix_addresses_cause": None not in (cause, fix) and shares_words(items[cause], items[fix]),
        "verification": verify is not None,
        "regression_test": fix is not None and any(cue(REGRESSION)(items, i) for i in range(fix + 1, len(items))),
        "stages_in_order": None not in found,
    }


# --- math ------------------------------------------------------------------------------------------------------

RESTATE = rx(r"(?:we|you|i) (?:are|were|have been) (?:asked|given|told)|(?:we|you) (?:need|have|want) to "
             r"(?:find|calculate|determine|compute|figure out|work out|solve)|the (?:question|problem) "
             r"(?:asks|is asking|wants|says|gives|states)|given that|we (?:know|are given)|asked to|to find"
             r"|how (?:many|much|long|far|old)")
PLAN_CUE = rx(r"plan|approach|strategy|we will|we'll|we can|we need to|we should|let's|let us|start by|begin by"
              r"|to solve|the idea|first,? we|step by step")
OPERATION = rx(r"multipl\w+|divid\w+|add\w*|subtract\w*|find\w*|set up|isolate|us\w+|compute|calculate|convert"
               r"|solve|combine|compare|total|sum|average|equation|formula")
FINAL = re.compile(r"^(?:(?:so|thus|therefore|hence),? )?(?:the )?(?:final )?answer\b|^(?:therefore|thus|hence|so)\b"
                   r"|^final answer", re.I)
CHECK = rx(r"check\w*|verif\w+|plug\w*|substitut\w+|sanity|makes sense|double-?check|reasonable|confirm\w*|going back")
AGREE = rx(r"matches|same|equals?|correct|works|true|right|reasonable|agrees?|consistent")


def calc_results(items: list[str]) -> set[str]:
    return {match.group(1).replace(",", "") for item in items for match in CALC.finditer(item)}


def is_calc(items: list[str], i: int) -> bool:
    return bool(CALC.search(items[i]))


def restated(prompt: str | None) -> Test:
    """Names the quantities from the question: shares its numbers or subject words, or at least shows a number."""
    wanted = numbers(prompt) if prompt else set()
    return lambda items, i: (bool(wanted & numbers(items[i])) if wanted else bool(NUMBER.search(items[i]))) \
        or echo(prompt)(items, i) and bool(prompt)


def is_final(items: list[str], i: int) -> bool:
    return bool(FINAL.search(items[i])) and bool(NUMBER.search(items[i])) and len(words(items[i])) <= MAX_ANSWER_WORDS


def is_check(items: list[str], i: int) -> bool:
    item = items[i]
    return len(words(item)) >= STAGE_WORDS and bool(CHECK.search(item)) and (bool(CALC.search(item)) or
                                                                           bool(NUMBER.search(item) and AGREE.search(item)))


def numbered_run(steps: dict[int, int]) -> int:
    """How many steps count 1, 2, 3 ... in a row from the start."""
    run = 0
    for number in steps.values():
        if number != run + 1:
            break
        run += 1
    return run


def math_checks(text: str, prompt: str | None = None) -> dict[str, bool]:
    items, steps = parse(text)
    stages = [
        Stage(both(cue(RESTATE), restated(prompt)), hi=0.35),
        Stage(both(cue(PLAN_CUE, OPERATION), lambda it, i: not CALC.search(it[i])), hi=0.6),
        Stage(is_final, lo=0.55),
        Stage(is_check, lo=0.6),
    ]
    restate, plan, final, check = found = chain(items, stages)
    work = [i for i in range(len(items)) if is_calc(items, i)]
    between = [i for i in work if None not in (plan, final) and plan < i < final]
    return {
        "restates_question": restate is not None,
        "states_plan": plan is not None,
        "numbered_steps": numbered_run(steps) >= 3,
        "calculations_shown": len(work) >= MIN_CALCULATIONS,
        "work_between_plan_and_answer": len(between) >= 2,
        "final_answer_marked": final is not None,
        "answer_matches_work": final is not None and bool(numbers(items[final]) & calc_results(items[:final])),
        "checks_the_answer": check is not None,
        "stages_in_order": None not in found,
    }


# --- explanation -----------------------------------------------------------------------------------------------

OPENER = rx(r"great question|good question|sure|certainly|of course|absolutely|let's dive|let me explain|here's|here is"
            r"|happy to")
WHY = rx(r"because|that's why|that is why|this is why|which is why|works by|happens when|happens because|so that"
         r"|as a result|the reason|this happens|that happens")
ANALOGY = rx(r"it'?s like|it is like|is like (?:a|an|the|your)|are like|think of|imagine|similar to|just like"
             r"|picture (?:a|an|the|your)|acts like|works like|as if|compare it to")
EXAMPLE = rx(r"for example|for instance|such as|e\.g\.|say you|suppose|take (?:a|an|the|your)|let's say|in real life")
RECAP = rx(r"in short|in summary|to sum up|the key (?:idea|point|thing)|remember|to recap|in a nutshell|the takeaway"
           r"|bottom line|put simply|simply put|all in all|so now")
EVERYDAY = re.compile(
    r"\b(?:kitchen|water|pizza|bik?e|bicycle|car|school|ball|cup|toy|garden|phone|door|bus|bread|cake|cookie|river"
    r"|road|rain|sun|tree|house|room|bed|boat|bag|box|book|pencil|chair|table|window|lamp|shower|bath|cat|dog|bird"
    r"|fish|milk|egg|soup|tea|coffee|sponge|balloon|bucket|hose|pipe|traffic|library|mirror|blanket|oven|fridge"
    r"|kettle|train|playground|swing|park|shop|store|market|bank|mail|letter|map|key|lock|clock|shoe|sock|coat"
    r"|hat|spoon|bowl|plate|glass|bottle|candle|fire|ice|snow|wind|kite|ladder|bridge|wall|fence|team|family"
    r"|friend|classroom|teacher|party|game|race|stairs|elevator|highway|parcel|backpack|bakery|farm|zoo|beach"
    r"|sand|wave|cloud|apple|orange|banana|rope|spring|trampoline|swimming pool)(?:s|es)?\b", re.I)
YOUR = re.compile(r"\byour \w+", re.I)
CAPITAL_WORD = re.compile(r"(?<=\s)[A-Z][a-z]{2,}")


def is_answer(prompt: str | None) -> Test:
    """One short plain sentence that answers the question: not a greeting, a heading or a question."""
    plain = both(lambda items, i: MAX_HEADING_WORDS <= len(words(items[i])) <= MAX_ANSWER_WORDS, echo(prompt, 1))
    return lambda items, i: plain(items, i) and not OPENER.search(items[i]) and not items[i].rstrip().endswith("?")


def concrete(items: list[str], i: int) -> bool:
    item = items[i]
    return bool(NUMBER.search(item) or EVERYDAY.search(item) or CAPITAL_WORD.search(item))


def explanation_checks(text: str, prompt: str | None = None) -> dict[str, bool]:
    items, _ = parse(text)
    stages = [
        Stage(is_answer(prompt), hi=0.25),
        Stage(cue(WHY), hi=0.75),
        Stage(both(cue(ANALOGY, min_words=8), lambda it, i: bool(EVERYDAY.search(it[i]) or YOUR.search(it[i]))),
              lo=0.15, hi=0.9),
        Stage(both(cue(EXAMPLE, min_words=8), concrete), lo=0.2),
        Stage(cue(RECAP), lo=0.6),
    ]
    answer, why, analogy, example, recap = found = chain(items, stages)
    explained = None not in (answer, why) and shares_words(items[answer], items[why], 1)  # the why is about the answer
    first = explained and all(len(words(item)) < MAX_HEADING_WORDS for item in items[:answer])  # nothing before it
    return {
        "plain_answer_first": first,
        "explains_why_or_how": explained,
        "everyday_analogy": analogy is not None,
        "concrete_example": example is not None,
        "short_recap_echoes_answer": (recap is not None and answer is not None and len(words(items[recap])) <= MAX_RECAP_WORDS
                                     and shares_words(items[recap], items[answer])),
        "simple_language": explained and simple_language(text, real_sentences(text)),  # plain wording of an explanation
        "stages_in_order": None not in found,
    }


# --- professional email ----------------------------------------------------------------------------------------

SUBJECT = re.compile(r"^subject\s*:\s*(.+)$", re.I)
GREETING = re.compile(r"^(?:dear|hello|hi|good (?:morning|afternoon|evening)|greetings)\b[\s,]*[\w.\[\]' -]*[,:]?$", re.I)
SIGNOFF = re.compile(r"^(?:best regards|kind regards|warm regards|best wishes|regards|sincerely|yours sincerely"
                     r"|yours faithfully|respectfully|best|thanks|thank you|many thanks|cordially)[,.!]?$", re.I)
SIGNER = re.compile(r"^(?:\[[^\]]+\]|[A-Z][\w.'-]*(?:[ ,]+[A-Z][\w.'-]*){0,4})$")
PURPOSE = rx(r"i am writing|i'm writing|i am reaching out|i'm reaching out|i wanted to|i would like to|i'd like to"
             r"|following up|i am contacting|writing to|reaching out|i am emailing|this email is")
DETAIL = re.compile(r"\d|\$|#\w|\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march"
                    r"|april|may|june|july|august|september|october|november|december|tomorrow|next week"
                    r"|next month|this week)\b", re.I)
ASK = rx(r"could you|can you|would you|will you|please \w+|i would appreciate|i'd appreciate|i would be grateful"
         r"|kindly|i am requesting|i'd like you to|i need you to|let me know|your approval")
NEXT_STEP = rx(r"by (?:\w+day|tomorrow|end of|noon|\d)|before|no later than|within \d+|next steps?|until|this week"
               r"|as soon as|i will|i'll|we will|we'll")
CLOSE = rx(r"thank you|thanks for|i appreciate|appreciate your|i look forward|looking forward|please feel free"
           r"|feel free to|don't hesitate|happy to (?:discuss|answer|help|clarify)|if you have any questions")
INFORMAL = rx(r"hey|yo|lol|omg|gonna|wanna|gotta|dunno|thx|pls|plz|u|ur|btw|dude|bro|guys|awesome|kinda|sorta|yeah"
              r"|yep|nope")
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]|:\\)|:\\(")
FORMAL = rx(r"please|thank|appreciate|regards|sincerely|kindly|grateful|look forward|would|could")


def email_lines(text: str) -> list[str]:
    return [line.strip().replace("*", "").lstrip("#> ").strip() for line in text.splitlines() if line.strip()]


def sign_off_at(lines: list[str]) -> int | None:
    for i in range(max(len(lines) - 4, 0), len(lines) - 1):
        if SIGNOFF.match(lines[i]) and SIGNER.match(lines[i + 1]):
            return i
    return None


def body_units(lines: list[str]) -> list[str]:
    return [part for line in lines for part in LESSON_SPLIT.split(line) if part]


def professional_tone(text: str) -> bool:
    formal = {match.lower() for match in FORMAL.findall(text)}
    return not INFORMAL.search(text) and not EMOJI.search(text) and text.count("!") <= 1 and len(formal) >= 2


def email_checks(text: str, prompt: str | None = None) -> dict[str, bool]:
    lines = email_lines(text)
    subject = SUBJECT.match(lines[0]) if lines else None
    greeting = 1 if len(lines) > 1 and GREETING.match(lines[1]) and len(words(lines[1])) >= 2 else None
    sign = sign_off_at(lines)
    middle = lines[(greeting or 0) + 1:sign]
    items = body_units(middle) if greeting is not None else []
    stages = [
        Stage(both(cue(PURPOSE, min_words=8), echo(prompt)), hi=0.4),
        Stage(cue(DETAIL)),
        Stage(cue(ASK)),
        Stage(cue(CLOSE, min_words=5)),
    ]
    purpose, detail, ask, close = found = chain(items, stages)
    details = [i for i in range(len(items)) if purpose is not None and ask is not None
               and purpose < i < ask and cue(DETAIL)(items, i)]
    return {
        "subject_line": bool(subject) and 3 <= len(words(subject.group(1))) <= 12,
        "greeting_with_name": greeting is not None,
        "purpose_early": purpose is not None,
        "specific_details": len(details) >= 2,
        "clear_request": ask is not None,
        "deadline_or_next_step": ask is not None and any(cue(NEXT_STEP, min_words=4)(items, i)
                                                         for i in range(ask, len(items))),
        "polite_close": close is not None,
        "sign_off_with_name": sign is not None,
        "professional_tone": professional_tone(text),
        "concise": len(words(text)) <= MAX_EMAIL_WORDS,
        "stages_in_order": bool(subject) and greeting is not None and sign is not None and None not in found,
    }


# --- how-to guide ----------------------------------------------------------------------------------------------

GOAL = rx(r"goal|how to|this guide|you will (?:learn|end up|have|be able)|you'll (?:learn|end up|have)|by the end"
          r"|will show you|walks? you through|aim|objective|we will|we'll")
MATERIALS = rx(r"you(?:'ll| will)? need|materials?|prerequisites?|requirements?|before you (?:start|begin)"
               r"|what you need|ingredients|tools|supplies|equipment|make sure you have|you should have|you'll want")
WARNING = rx(r"warning|caution|careful|avoid|don't|do not|never|common mistake|mistake|watch out|be sure not"
             r"|note that|danger|otherwise|if you skip|beware|be aware|pitfall")
SUCCESS = rx(r"you(?:'ll| will)? (?:know|see|hear|notice|be able)|you should (?:now )?(?:see|hear|be able|have|notice|feel)"
             r"|to (?:confirm|check|verify|test)|it worked|success\w*|is working|test it|you(?:'re| are) done"
             r"|if (?:it|everything|the \w+) (?:works|is|looks|shows)")
NOT_VERBS = frozenset(
    "the a an you your it its this that these those there then next first finally also now once when if after "
    "before while as and but or so we they he she i my our their his her is are was were be been being do does did "
    "has have had can could will would should may might step each every all any some one two three four five six "
    "seven eight nine ten".split())
NOT_VERB_ENDINGS = ("ing", "ed", "ly", "tion", "ment", "ness", "ity", "ous", "ful")
OBJECT_STARTS = frozenset(
    "the a an your it each all any both to into on in up out off over down with until for every one two three four "
    "five six seven eight nine ten this that these those them".split())
VERBS = frozenset(
    "open close click press tap select choose enter type add remove delete mix stir pour heat cook bake boil cut slice "
    "chop wash rinse dry place put set turn plug unplug connect disconnect attach insert tighten loosen check make "
    "take use install download run save copy paste move drag write read measure wait let leave keep fill empty clean "
    "wipe apply sand paint cover wrap fold roll hold lift push pull screw drill mark trim test restart update sign "
    "log go find search scroll whisk preheat grate peel season serve sprinkle spread drain dig plant water gather "
    "prepare start stop repeat finish replace pick".split())


def imperative(step: str) -> bool:
    tokens = words(step)
    if len(tokens) < 2:
        return False
    head, after = tokens[0], tokens[1]
    if head in VERBS:
        return True
    return (head not in NOT_VERBS and not head.endswith(NOT_VERB_ENDINGS)
            and (after in OBJECT_STARTS or after.endswith("ly")))


def listed(items: list[str], i: int) -> bool:
    """The materials line is followed by real items: several short lines, or a comma list in the line itself."""
    after = items[i].split(":", 1)[-1]
    if len(re.split(r",| and ", after)) >= 3:
        return True
    short = 0
    for item in items[i + 1:]:
        if len(words(item)) > 8:
            break
        short += 1
    return short >= 2


def how_to_checks(text: str, prompt: str | None = None) -> dict[str, bool]:
    items, steps = parse(text)
    last_step = max(steps, default=-1)
    stages = [
        Stage(both(cue(GOAL), echo(prompt)), hi=0.3),
        Stage(both(lambda it, i: bool(MATERIALS.search(it[i])), listed), hi=0.5),
        Stage(lambda it, i: steps.get(i) == 1),
        Stage(cue(WARNING)),
        Stage(both(cue(SUCCESS), lambda it, i: i > last_step), lo=0.5),
    ]
    goal, materials, first_step, warning, success = found = chain(items, stages)
    step_texts = [items[i] for i in steps]
    return {
        "states_goal": goal is not None,
        "lists_materials": materials is not None,
        "numbered_steps": numbered_run(steps) >= 3,
        "steps_start_with_verbs": bool(step_texts) and sum(map(imperative, step_texts)) / len(step_texts) >= VERB_SHARE,
        "steps_have_substance": sum(len(words(s)) >= MIN_STEP_WORDS and len(content_words(s)) >= 2 for s in step_texts) >= 3,
        "warning_or_common_mistake": warning is not None and first_step is not None,  # a warning about steps
        "confirms_success": success is not None and first_step is not None,
        "stages_in_order": None not in found,
    }


CHECKLISTS.update({
    "debugging": debugging_checks, "math_solving": math_checks, "explanation": explanation_checks,
    "professional_email": email_checks, "how_to_guide": how_to_checks,
})
