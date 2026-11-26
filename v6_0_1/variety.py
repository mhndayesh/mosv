"""Variety limits for the training set: stop one opening style or one character name from taking over a topic.

Why: the first story rule ("first sentence = an action or a line of speech") made the model open nearly every kept
story with a quoted line plus "X said", and reuse the same names (Elias in two of three samples). Training on that
would teach a new habit, not the behavior. These limits are applied to the kept answers of a topic, best answers
first, and the overflow is dropped with a reason (build_data.py). gate.py reports the same numbers for told answers.
"""

from __future__ import annotations

import re
from collections import Counter

OPENING_QUOTES = ('"', "“", "'", "‘")  # straight and curly quote marks that start a line of speech
MAX_QUOTE_OPENINGS = 0.40  # at most 40% of a topic's kept answers may open with a line of speech
MAX_NAME_SHARE = 0.05      # one character name may appear in at most 5% of a topic's kept answers...
MIN_NAME_ALLOWANCE = 2     # ...but always at least 2 answers (small sets would otherwise allow none)
NAME_WINDOW = 600          # characters read from the start of an answer when looking for names
# Capitalized words that are not character names but often appear mid-sentence.
NOT_NAMES = {"The", "And", "But", "His", "Her", "Its", "Old", "New", "World", "God", "Mom", "Dad", "Mr", "Mrs", "Dr",
             "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
LIMITED_TOPICS = ("story",)  # topics whose kept answers get these limits


def opens_with_speech(answer: str) -> bool:
    """True when the answer's first character is a quote mark, i.e. it opens with a line of speech."""
    return answer.lstrip().startswith(OPENING_QUOTES)


def names(answer: str) -> set[str]:
    """Likely character names near the start: capitalized words that are not the first word of a sentence."""
    found = set()
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer[:NAME_WINDOW]):
        words = re.findall(r"[A-Za-z']+", sentence)
        found.update(w for w in words[1:] if re.fullmatch(r"[A-Z][a-z]{2,}", w) and w not in NOT_NAMES)
    return found


def limit(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """(kept, dropped) for one topic's kept answers; rows must already be sorted best first.

    A row is dropped when keeping it would push the share of speech openings over MAX_QUOTE_OPENINGS, or the count
    of any one name over its allowance.
    """
    max_quotes = int(MAX_QUOTE_OPENINGS * len(rows))
    max_name = max(MIN_NAME_ALLOWANCE, int(MAX_NAME_SHARE * len(rows)))
    kept, dropped, quotes, seen = [], [], 0, Counter()
    for row in rows:
        speech, who = opens_with_speech(row["answer"]), names(row["answer"])
        if (speech and quotes >= max_quotes) or any(seen[n] >= max_name for n in who):
            dropped.append(row)
            continue
        kept.append(row)
        quotes += speech
        seen.update(who)
    return kept, dropped


def report(answers: list[str]) -> dict:
    """Variety numbers for a set of answers: share opening with speech, and the most repeated names."""
    counts = Counter(n for a in answers for n in names(a))
    return {"speech_openings": round(sum(map(opens_with_speech, answers)) / max(len(answers), 1), 3),
            "top_names": [[n, round(c / max(len(answers), 1), 3)] for n, c in counts.most_common(3)]}
