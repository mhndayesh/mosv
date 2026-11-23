"""Router: picks which steering expert to use, or none, once per prompt.

The topic list is the one place the topic names live. The default is the original three
(the 0.6B data). Set STEERING_TOPICS=v2 (or a comma list) to route the eight-topic set.
"""

import os

TOPICS_V1 = ("engineering", "story", "teaching")
TOPICS_V2 = ("engineering", "story", "teaching", "debugging", "math_solving", "explanation",
             "professional_email", "how_to_guide")
NONE_LABEL = "none"


def resolve_topics(spec: str | None) -> tuple[str, ...]:
    """`v1`, `v2` or a comma list of topic names. Empty means the default (v1)."""
    spec = (spec or "v1").strip()
    named = {"v1": TOPICS_V1, "v2": TOPICS_V2}
    if spec in named:
        return named[spec]
    topics = tuple(part.strip() for part in spec.split(",") if part.strip())
    unknown = [t for t in topics if t not in TOPICS_V2]
    if not topics or unknown or len(set(topics)) != len(topics):
        raise ValueError(f"Bad topic list {spec!r}; use v1, v2 or names from {TOPICS_V2}")
    return topics


def classes_for(topics: tuple[str, ...]) -> tuple[str, ...]:
    return (*topics, NONE_LABEL)


CLASSES = classes_for(resolve_topics(os.environ.get("STEERING_TOPICS")))
NONE_INDEX = CLASSES.index(NONE_LABEL)
