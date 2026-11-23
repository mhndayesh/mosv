"""Prove the checklists can fail: damaged, stuffed, looped and off-topic answers must score lower than real ones.

Run: python -m behaviors.validate_rubrics
"""

from __future__ import annotations

import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean

from behaviors.gold_examples import GOLD
from behaviors.rubrics import FUNCTION_WORDS, is_meta_story, real_sentences, score_response, sentences

ROOT = Path(__file__).parent.parent
CLEAN = ROOT / "data_review" / "clean"
RL_STORY = ROOT / "data_review" / "rl" / "vllm_full" / "story.json"
REFERENCE_SAMPLE = 3000
MIN_GOLD_SCORE = 0.8
MIN_SHUFFLE_DROP = 0.10
MAX_OFF_TOPIC = 0.30
MAX_STUFFED = 0.25
MAX_LOOPED = 0.25
MAX_EXPLOIT = 0.25
MAX_WRONG_FORMAT = 0.25
LINE_BEHAVIORS = {"engineering", "professional_email", "how_to_guide", "math_solving", "debugging"}
STUFF_SENTENCE_WORDS = 9
STUFF_SEEDS = 20
# Checklists written before the stuffing control. Plain stuffing must fail them; the harder stuffing (grammatical salad
# of cue, filler and function words) is reported for them but is a known weakness, so it does not gate.
OLDER_TOPICS = {"engineering", "story", "teaching"}

# Words each checklist looks for. A model that only repeats these must not earn the reward.
CUES = {
    "engineering": "requirements design README tests setup install pytest src app tree module steps",
    "story": "village problem tried again suddenly finally saved together happy learned home never",
    "teaching": "today learn remember first next then example imagine show try practice review recap",
    "debugging": "error crash fails reproduce because root cause fix replace change add confirm rerun regression "
                 "test assert returns raises when function list",
    "math_solving": "asked plan multiply divide add subtract equals final answer check verify 12 48 240 times total",
    "explanation": "because like imagine example for instance short summary remember simple water sun ball sky",
    "professional_email": "subject dear writing request please could thank you regards sincerely Friday March 14 meeting",
    "how_to_guide": "goal need materials tools open press add mix warning avoid never mistake confirm success see know",
}
FILLER = ("the river window garden paper yellow silver market bridge carpet engine forest letter button candle "
          "ticket planet anchor basket cotton dragon eleven flower guitar hammer island jacket kitchen ladder "
          "magnet needle orange pillow rabbit saddle tunnel umbrella valley walnut yogurt zipper almond bottle "
          "castle donkey feather ginger harbor insect jungle kettle lemon monkey nickel oyster pencil quiet "
          "ribbon spoon tomato velvet whistle").split()
# Answers that talk about a story instead of telling one (the known exploit), one with a real prompt.
STORY_PROMPT = 'Could this become a story? Please try: " This too shall pass "'
STORY_ABOUT_A_STORY = (
    "Yes, this could become a story. The main conflict is a hero and a villain in a small town. "
    "The villain wanted to attack the town, and the people were afraid. "
    "In this story, the civilians are not alone, because the heroes live in the same town. "
    "The hero tried to save them, but the villain fought again and again. "
    "The story would involve a lot of emotional conflict. "
    "The hero decided to face the villain in a last battle. "
    "To make it a story, the heroes and villains would have to face a lot of challenges. "
    "Suddenly the villain was defeated and the people were safe. "
    "Together they found peace in the town, and they stood strong. "
    "So, the question is: Can the story be made?"
)


def pieces(behavior: str, text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()] if behavior in LINE_BEHAVIORS else sentences(text)


def join(behavior: str, parts: list[str]) -> str:
    return ("\n" if behavior in LINE_BEHAVIORS else " ").join(parts)


def shuffled(behavior: str, text: str, seed: int) -> str:
    parts = pieces(behavior, text)
    random.Random(seed).shuffle(parts)
    return join(behavior, parts)


def without_middle(behavior: str, text: str) -> str:
    parts = pieces(behavior, text)
    return join(behavior, parts[:int(len(parts) * 0.35)] + parts[int(len(parts) * 0.7):])


def first_half(behavior: str, text: str) -> str:
    parts = pieces(behavior, text)
    return join(behavior, parts[:len(parts) // 2])


def stuffed(behavior: str, text: str, seed: int, mixed: bool = False) -> str:
    """As many words as the real answer, drawn at random from the checklist's own vocabulary.

    Plain stuffing repeats the few cue words and is caught by the loop and prose gates. Mixed stuffing adds filler and
    function words ("the", "and") so it gets past both gates and only the checks themselves can reject it.
    """
    vocabulary = CUES[behavior].split() + (FILLER + sorted(FUNCTION_WORDS) if mixed else [])
    rng = random.Random(seed)
    count = max(len(text.split()), 60)
    lines = [" ".join(rng.choices(vocabulary, k=STUFF_SENTENCE_WORDS)).capitalize() + "."
             for _ in range(count // STUFF_SENTENCE_WORDS)]
    return join(behavior, lines)


def looped(behavior: str, text: str) -> str:
    return join(behavior, pieces(behavior, text)[:3] * 12)


def score_one(job: tuple[str, str, str | None]) -> float:
    return score_response(job[0], job[1], job[2]).score


def run_jobs(pool, jobs: list[tuple]) -> float:
    return round(mean(pool.map(score_one, jobs, chunksize=64)), 4)


def control_scores(pool, behavior: str, items: list[tuple[str | None, str]]) -> dict[str, float]:
    other = [b for b in GOLD if b != behavior]
    seeds = STUFF_SEEDS if len(items) < 100 else 1  # a few hand-written answers need several random stuffings
    result = {
        "real": run_jobs(pool, [(behavior, text, prompt) for prompt, text in items]),
        "shuffled": run_jobs(pool, [(behavior, shuffled(behavior, text, i), prompt)
                                    for i, (prompt, text) in enumerate(items)]),
        "no_middle": run_jobs(pool, [(behavior, without_middle(behavior, text), prompt) for prompt, text in items]),
        "first_half": run_jobs(pool, [(behavior, first_half(behavior, text), prompt) for prompt, text in items]),
        "cue_words_repeated": run_jobs(pool, [(behavior, stuffed(behavior, text, seed), None)
                                              for seed in range(seeds) for _, text in items]),
        "keyword_stuffed": run_jobs(pool, [(behavior, stuffed(behavior, text, seed, mixed=True), None)
                                           for seed in range(seeds) for _, text in items]),
        "looped": run_jobs(pool, [(behavior, looped(behavior, text), prompt) for prompt, text in items]),
    }
    for name in other:
        result[f"scored_as_{name}"] = run_jobs(pool, [(name, text, None) for _, text in items])
    return result


def off_topic(pool, behavior: str) -> float:
    items = GOLD[behavior]
    return run_jobs(pool, [(behavior, text, items[(i + 1) % len(items)][0]) for i, (_, text) in enumerate(items)])


def read_reference(behavior: str, source: str) -> list[tuple[None, str]]:
    with (CLEAN / behavior / "reference.jsonl").open(encoding="utf-8") as file:
        texts = [row["text"] for row in map(json.loads, file) if row["source"] == source]
    random.Random(11).shuffle(texts)
    return [(None, text) for text in texts[:REFERENCE_SAMPLE]]


def verdict(scores: dict[str, float], gold: bool, behavior: str = "") -> dict[str, bool]:
    passed = {"shuffle_drops": scores["real"] - scores["shuffled"] >= MIN_SHUFFLE_DROP,
              "damage_drops": scores["no_middle"] < scores["real"] and scores["first_half"] < scores["real"]}
    if gold:
        passed["gold_scores_high"] = scores["real"] >= MIN_GOLD_SCORE
        passed["off_topic_low"] = scores["off_topic"] <= MAX_OFF_TOPIC
        stuffing = [scores["cue_words_repeated"]] + ([] if behavior in OLDER_TOPICS else [scores["keyword_stuffed"]])
        passed["keyword_stuffing_low"] = max(stuffing) <= MAX_STUFFED
        passed["looping_low"] = scores["looped"] <= MAX_LOOPED
    return passed


def flagged_share(items: list[tuple[None, str]]) -> float:
    """Share of real stories with at least one sentence that talks about a story: should be close to zero."""
    return round(mean(any(is_meta_story(part) for part in real_sentences(text)) for _, text in items), 4)


def rl_story_scores() -> dict[str, float]:
    """The story answers saved from the RL run (cut to 700 characters), scored with their prompts."""
    if not RL_STORY.exists():
        return {}
    groups = json.loads(RL_STORY.read_text(encoding="utf-8"))["test"]
    return {name: round(mean(score_response("story", s["answer"], s["prompt"]).score for s in group["samples"]), 4)
            for name, group in groups.items()}


def main() -> None:
    report = {}
    with ProcessPoolExecutor(max_workers=max(2, (os.cpu_count() or 4) - 2)) as pool:
        for behavior, items in GOLD.items():
            scores = control_scores(pool, behavior, items)
            scores["off_topic"] = off_topic(pool, behavior)
            report[f"gold/{behavior}"] = {"n": len(items), **scores, "pass": verdict(scores, True, behavior)}
        exploit = score_response("story", STORY_ABOUT_A_STORY, STORY_PROMPT).score
        report["exploit/story_about_a_story"] = {"score": round(exploit, 4), "pass": {"scores_low": exploit <= MAX_EXPLOIT}}
        for behavior, source in (("story", "simplestories"), ("teaching", "kindergarten")):
            items = read_reference(behavior, source)
            scores = control_scores(pool, behavior, items)
            extra = {"stories_flagged_as_talk": flagged_share(items)} if behavior == "story" else {}
            report[f"real/{source}"] = {"n": len(items), **scores, **extra, "pass": verdict(scores, False)}
        # Teacher planning documents are not a lesson delivered to a child, so they should score low.
        items = read_reference("teaching", "lesson_plan")
        scores = control_scores(pool, "teaching", items)
        report["wrong_format/lesson_plan"] = {"n": len(items), **scores,
                                              "pass": {"scores_low": scores["real"] <= MAX_WRONG_FORMAT}}
    report["rl_story_samples_mean_score"] = rl_story_scores()
    (CLEAN / "rubric_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
