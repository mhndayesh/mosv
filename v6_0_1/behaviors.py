"""The behaviors v6.0.1 teaches: per topic, the rules told to the model and the yes/no questions that check them.

This file is the one place that says what "good behavior" means. Three stages read it:
  answers.py  tells the model the rules (the "guide") when it writes a "told" answer;
  checks.py   asks Haiku the yes/no questions about an answer;
  gate.py     compares plain and told answers question by question.

How the lists were written (owner rule: behavior, not knowledge):
  - every rule and question is about HOW the answer is written, something the model can do when asked;
  - every question can be answered yes or no from the request and the answer alone, and "yes" is always good;
  - "checks" are the behaviors we want to teach;
  - "guards" are correctness questions that must not get worse while the behavior improves (an explanation that
    became plainer but false is not an improvement).
Written from the 2026-11-26 dev answers (plain vs told) and their per-line scores; see docs/V6_GEMMA.md.

math_solving has no entry on purpose. Its score depends on getting the answer right (ability, not behavior), and
telling Gemma rules made it worse (4.10 -> 3.80). Its prompts are routed to "none", so they are answered unchanged.
"""

from __future__ import annotations

import json
import zlib

HEADER ="Follow these rules when you write your answer. Do not mention the rules."  # first line of every guide

BEHAVIORS = {
    # Bug reports arrive without the code. The base model invents class names and states guesses as facts
    # (line "honest about what is guessed": 1.4 of 5). Told to be honest it reached 2.8.
    "debugging": {
        "guide": """- You only have the bug report, not the code. Start with what the report itself shows: symptom, platform, version, error text.
- Then give the most likely cause first and one or two alternatives. Tie each cause to a clue in the report.
- Say plainly which parts are guesses. Never state a class, method, file, line number or setting as fact unless it appears in the report.
- For each cause, say how to check it: what to log, run, reproduce or compare.
- Keep the fix short and conditional on the cause being confirmed.""",
        "checks": [
            "Does the answer clearly separate what the report states from what the writer is guessing?",
            "Is every class, method, file, line number or setting that the answer states as fact also present in the report? (Names offered as explicit guesses are fine.)",
            "Is each proposed cause tied to a specific clue from the report, such as the symptom, platform, version or error text?",
            "Does the answer say how to check or confirm at least one cause: what to log, run, reproduce or compare?",
        ],
        "guards": ["Does the answer stay on this bug report and propose at least one concrete cause or fix, not only generic advice?"],
    },
    # "Why / how" questions. The base model writes headed lists of facts in technical words (line "plain words":
    # 2.4). Told, it reached 4.0 but also cut true content, so the guard asks whether the main cause is still right.
    "explanation": {
        # Second version. The first one said only "answer in one or two sentences": told answers shrank to ~100
        # tokens and the correctness guard fell from 0.70 to 0.60, so the gate skipped the topic.
        "guide": """- Start with the direct answer in one or two sentences.
- Then explain it step by step in one or two short paragraphs. Include every step of the real cause: a complete, correct answer matters more than a short one.
- Use short sentences and everyday words. Explain any technical term you must use.
- Give one concrete example or comparison that matches how the thing really works.
- Link the ideas by cause and effect. Do not list side facts or use section headings.
- Say nothing false. If you are not sure of a detail, leave it out or say that you are not sure.""",
        "checks": [
            "Do the first two sentences directly answer the question that was asked?",
            "Is the answer written in short sentences and everyday words, with any technical term explained?",
            "Does it give at least one concrete example or comparison?",
            "Is the explanation one connected cause-and-effect chain, not a list of separate facts or headed sections?",
        ],
        "guards": ["Is the main cause given in the answer factually correct? (Simplified is fine; false is not.)"],
    },
    # "Teach X to a Primary 2 student." The base model writes a lesson plan for a teacher. The behavior wanted is
    # to teach the child directly, in small steps, and to check understanding (that line: 3.1 -> 3.8 when told).
    "teaching": {
        "guide": """- Speak to the child directly, in words and sentence lengths right for the stated grade. Do not write a lesson plan for a teacher.
- Start from something the child already knows or sees every day.
- Go in small steps, and show one worked example before asking the child to try.
- Ask the child at least two questions, or give one short task, to check understanding.
- Keep it short enough for one lesson.""",
        "checks": [
            "Does the answer speak to the child directly, instead of describing a lesson plan to a teacher?",
            "Is the language right for the stated grade: words and sentence lengths a child of that grade can follow?",
            "Does it connect the idea to something from the child's everyday life?",
            "Does it ask the child questions or give a short task to check understanding?",
        ],
        "guards": ["Are the facts taught correct for this subject?"],
    },
    # Code tasks and code reviews. The base model opens with paragraphs of plan and adds extras. Told, answers were
    # half as long and scored +0.40. The last check keeps a short "why", because v5 showed that code with no
    # explanation loses points.
    "engineering": {
        # Second version. The first one made answers a third as long and the correctness guard fell from 0.70 to
        # 0.50, so "meet every requirement" now comes before "keep it simple".
        "guide": """- Start with the code or the first concrete finding. Write at most two sentences before it.
- Do exactly what was asked: keep the given names, signature and language, and meet every stated requirement, edge case and example.
- Choose the simplest design that meets all of that. Add no extra features and no alternative versions. Simple does not mean incomplete.
- Do not restart or second-guess yourself.
- After the code, say briefly why the key parts work. For a review, point at specific lines or names and say what was wrong.""",
        "checks": [
            "Does the code, or the first concrete finding, start within the first three sentences?",
            "Does the answer keep to what was asked (given names, signature, language) without adding features or alternative versions nobody asked for?",
            "Is the answer free of self-corrections and restarts, such as 'Wait', 'Actually' or a second full rewrite?",
            "Does it briefly explain why: the key reasoning for the code, or for a review what was wrong at specific lines or names?",
        ],
        "guards": ["Does the code or review look correct for the stated task, with no claim that contradicts the code shown?"],
    },
    # Stories. The base model opens on mood or weather and tells feelings in abstract words (lines "believable
    # characters" 2.8, "fresh detail" 3.0). Third version of the rules: the second ("first sentence = an action or a
    # line of speech") made nearly every story open with a quote and reuse names like Elias. variety.py also caps
    # speech openings and repeated names among the kept answers.
    "story": {
        "guide": """- Write the piece in the form that was asked: story, poem or script.
- Open on something specific: a character doing something, a line of speech, or one unusual concrete detail. Choose the opening that suits this story; do not default to dialogue. Never open with weather, light, sound, the time of day or a general mood.
- Give characters names that fit the story's setting, and avoid stock names such as Elias, Silas, Sarah or Kael.
- Use specific concrete details instead of stock phrases.
- Show feelings through what characters do and say, not by naming the feeling.
- Do not explain a moral or lesson at the end.""",
        "checks": [
            "Does the opening start with an action, a voice or a specific image, instead of weather, light or general mood-setting?",
            "Does the piece use specific concrete details (particular objects, names, sensory facts) rather than generic phrases?",
            "Are feelings shown through actions or dialogue rather than named abstractly?",
            "Does the ending avoid stating a moral or lesson outright?",
        ],
        "guards": ["Does the piece fit the request and its form, and is it coherent?"],
    },
    # Emails built from a few fields. The key points are bare labels ("Compliance status"), so a writer must either
    # invent facts or stay vague. The behavior wanted is: use only given facts, and write a bracketed placeholder
    # where a needed fact is missing. The first guide (the old rubric) gave no gain; this one says what to do instead.
    "professional_email": {
        # Second version. The first one made emails half as long; "covers the key points" fell from 1.00 to 0.75,
        # so covering every key point is now an explicit rule.
        "guide": """- State the purpose of the email in its first two sentences.
- Cover every key point from the request: give each one a short paragraph that says what the reader needs to know or do about it.
- Use only the facts given in the request. Do not invent names, dates, numbers, results or attachments.
- Where a needed fact was not given, write a short bracketed placeholder such as [date] or [figure] instead of making one up.
- End with a clear ask or next step. Match the requested tone. No filler sentences.""",
        "checks": [
            "Is the purpose of the email clear within the first two sentences of the body?",
            "Is every specific fact in the email (names, dates, numbers, results, attachments) either given in the request or written as a bracketed placeholder?",
            "Does the email end with a clear ask or next step?",
            "Is it free of filler and template padding, with each paragraph carrying one needed idea?",
        ],
        "guards": ["Does it cover the requested purpose and key points, with the right recipient, sender and tone?"],
    },
}


def fingerprint() -> str:
    """A short code that changes whenever any rule or question changes.

    Stages store it next to their cached answers (files.claim), so answers written under old rules are never mixed
    with answers written under new ones.
    """
    return f"{zlib.crc32(json.dumps(BEHAVIORS, sort_keys=True).encode()):08x}"


def guide(topic: str) -> str:
    """The system message that tells the model this topic's rules (used only to write "told" answers)."""
    return f"{HEADER}\n{BEHAVIORS[topic]['guide']}"


def questions(topic: str) -> dict[str, str]:
    """The topic's yes/no questions keyed b1..bN (behavior checks) then g1..gM (guards), in file order."""
    entry = BEHAVIORS[topic]
    asked = {f"b{i}": text for i, text in enumerate(entry["checks"], start=1)}
    asked.update({f"g{i}": text for i, text in enumerate(entry["guards"], start=1)})
    return asked
