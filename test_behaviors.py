"""Run: python -m unittest test_behaviors.py"""

import unittest

from behaviors.prepare_data import (
    app_idea_summary, clean_story_prompt, detokenize, is_test, lesson_fields, lesson_key, lesson_reason,
    tidy_case,
)
from behaviors.gold_examples import GOLD
from behaviors.process_rubrics import ANNOUNCE, imperative, numbers, parse
from behaviors.rubrics import (
    FREE_REPETITION, is_degenerate, is_meta_story, meta_factor, meta_story_share, repetition, score_response,
    sentences, unique_tree, tree_lines,
)
from behaviors.validate_rubrics import FILLER, looped, stuffed

NEW_TOPICS = ("debugging", "math_solving", "explanation", "professional_email", "how_to_guide")

ENGINEERING = """Requirements: the app must let a user add, list and delete short notes from the terminal.
Design: a storage module saves notes as JSON, and a command module handles user input.

notes-app/
  README.md
  requirements.txt
  .gitignore
  src/app.py
  src/storage.py
  tests/test_app.py

Setup:
1. Create a virtual environment.
2. Install the dependencies with pip install -r requirements.txt.
3. Run the tests with pytest.
"""

STORY = (
    "Mira lived in a small village by the river with her father. "
    "One morning the old bridge broke and the village lost its only road. "
    "Mira tried to carry wood across the river, but the water was too fast. "
    "She tried again with a rope, but it snapped. "
    "Then she asked the farmers for help and they decided to build a raft. "
    "The rain grew heavier each hour. "
    "Suddenly the raft tore loose with the last of the wood. "
    "Mira jumped in and held on through the current. "
    "She saved the wood, and together they rebuilt the bridge. "
    "The village road was no longer broken, and the bridge stood strong."
)

LESSON = (
    "Today we will learn about plants. Do you remember the seed we planted? "
    "First, a plant needs water. Next, it needs sun. Then it grows new leaves. "
    "For example, imagine a bean in a cup. Let me show you how I water it. "
    "Now it is your turn to try it with your cup. What does a plant need to grow? "
    "Today we learned that plants need water and sun."
)

PLAIN = (
    "This program is a tool that people can use on a computer every day for many things. "
    "It works well and it is easy for anyone to open it and use it when they need it, "
    "and it can be changed later on by anyone who would like to do that for their work."
)

# Real pilot output: the same entries over and over, and a lesson that loops "Let's practice".
LOOPED_TREE = """Requirements: the app must let a user start, stop and reset a stopwatch from the terminal.
Design: a timer module counts the seconds, and a command module handles what the user types.

stopwatch-app/
├── README.md
├── requirements.txt
├── package.json
├── app/
│   ├── README.md
│   ├── package.json
│   ├── package.json
│   ├── README.md
│   ├── package.json
├── .gitignore
├── README.md
├── README.md
"""

LOOPED_LESSON = (
    "Hello, let's learn about Adjectives! Today, we will practice using adjectives in a fun way. "
    "Let's start by learning about them.\n\n"
    "First, I need to practice using adjectives. Let's see... How to use adjectives? For example, if I say "
    "\"The cat is happy,\" that is an adverb. Let's try to practice.\n\n"
    "Okay, let's start. I will practice using adjectives. Let's practice. Once I learn, I can practice. "
    "Let's do it together.\n\n"
    "Now, let's practice. Let's think. How to use adjectives? Let's practice. Once I learn, I can practice. "
    "Let's do it. Then, I will practice. Let's check. Then, I will learn. Let's practice. Then, I will answer. "
    "Let's do it. Once I finish, I will review. Let's check. Let's do it. Let's practice. Then, I will review."
)

TAUGHT_LESSON = (
    "Today we will learn about adjectives. Do you remember the word big? "
    "First, an adjective tells us more about a noun. Next, it can say what a thing looks like. "
    "Then it can say how many there are. For example, imagine a red ball in the grass. "
    "Let me show you how I find the adjective. Now it is your turn to try it with a blue cup. "
    "Which word tells us the color of the cup? Today we learned that adjectives describe nouns."
)


class TestRubrics(unittest.TestCase):
    def test_full_structure_scores_one(self):
        for behavior, text in (("engineering", ENGINEERING), ("story", STORY), ("teaching", LESSON)):
            result = score_response(behavior, text)
            failed = [name for name, passed in result.checks.items() if not passed]
            self.assertEqual(result.score, 1.0, f"{behavior} failed: {failed}")

    def test_unstructured_text_scores_low(self):
        for behavior in ("engineering", "story", "teaching"):
            self.assertLessEqual(score_response(behavior, PLAIN).score, 0.25, behavior)

    def test_reversed_order_loses_points(self):
        for behavior, text in (("story", STORY), ("teaching", LESSON)):
            backwards = " ".join(reversed(sentences(text)))
            self.assertLess(score_response(behavior, backwards).score, 1.0, behavior)

    def test_files_before_requirements_loses_points(self):
        tree, plan = ENGINEERING.split("Setup:")[0].split("notes-app/")
        flipped = f"notes-app/{plan}\n{tree}\nSetup:{ENGINEERING.split('Setup:')[1]}"
        checks = score_response("engineering", flipped).checks
        self.assertFalse(checks["sections_in_order"])

    def test_keyword_stuffing_earns_nothing(self):
        stuffed = "requirements design README tests recap climax " * 30
        result = score_response("engineering", stuffed)
        self.assertTrue(result.degenerate)
        self.assertEqual(result.score, 0.0)

    def test_ignoring_the_prompt_loses_the_score(self):
        on_topic = score_response("story", STORY, "Write a story about a village bridge and a river.")
        off_topic = score_response("story", STORY, "Write a story about astronauts orbiting Jupiter.")
        self.assertEqual(on_topic.score, 1.0)
        self.assertEqual(off_topic.score, 0.0)

    def test_gold_examples_score_high(self):
        for behavior, items in GOLD.items():
            for prompt, text in items:
                result = score_response(behavior, text, prompt)
                failed = [name for name, passed in result.checks.items() if not passed]
                self.assertGreaterEqual(result.score, 0.8, f"{behavior}: {prompt} failed {failed}")

    def test_engineering_needs_matching_tests_and_commands(self):
        broken = ENGINEERING.replace("tests/test_app.py", "tests/test_other.py").replace("pip install", "brew install")
        checks = score_response("engineering", broken).checks
        self.assertFalse(checks["tests_match_source"])
        self.assertFalse(checks["commands_fit_files"])

    def test_short_output_earns_nothing(self):
        self.assertEqual(score_response("story", "He lost it. Suddenly he found it.").score, 0.0)

    def test_unknown_behavior_raises(self):
        with self.assertRaises(ValueError):
            score_response("cooking", STORY)

    def test_normal_text_is_not_repetitive(self):
        for text in (ENGINEERING, STORY, LESSON, TAUGHT_LESSON, *(t for items in GOLD.values() for _, t in items)):
            self.assertLessEqual(repetition(text), FREE_REPETITION)

    def test_loops_are_degenerate(self):
        loop = "Let's practice this together now. Then we will check it. " * 12
        self.assertGreater(repetition(loop), 0.8)
        self.assertTrue(score_response("teaching", loop).degenerate)

    def test_repetition_lowers_the_score_smoothly(self):
        padded = STORY + " " + " ".join(sentences(STORY)[:3])
        mild = score_response("story", padded).score
        heavy = score_response("story", padded + " " + " ".join(sentences(STORY)[:3] * 2)).score
        self.assertLess(mild, 1.0)
        self.assertLess(heavy, mild)

    def test_repeated_file_entries_count_once(self):
        listed = tree_lines(LOOPED_TREE)
        self.assertEqual(len(listed), 13)
        self.assertEqual(len(unique_tree(listed)), 8)  # README.md and package.json also exist under app/
        checks = score_response("engineering", LOOPED_TREE).checks
        self.assertTrue(checks["file_tree"])  # 8 real paths
        self.assertFalse(checks["no_duplicate_files"])

    def test_endless_file_loop_is_degenerate(self):
        text = LOOPED_TREE + "├── README.md\n" * 20
        self.assertTrue(score_response("engineering", text).degenerate)

    def test_one_real_entry_repeated_is_not_a_tree(self):
        tree = "demo/\n" + "├── README.md\n" * 4
        text = ENGINEERING.split("notes-app/")[0] + tree + "Setup:" + ENGINEERING.split("Setup:")[1]
        checks = score_response("engineering", text).checks
        self.assertFalse(checks["file_tree"])
        self.assertFalse(checks["no_duplicate_files"])
        self.assertLess(score_response("engineering", text).score, score_response("engineering", ENGINEERING).score)

    def test_same_name_in_another_folder_is_a_new_file(self):
        paths = unique_tree(["pkg/", "  __init__.py", "  a.py", "tests/", "  __init__.py"])
        self.assertEqual(len(paths), 5)

    def test_looped_engineering_tree_scores_low(self):
        self.assertLess(score_response("engineering", LOOPED_TREE).score, 0.4)

    def test_looped_lesson_earns_no_practice_or_check_points(self):
        result = score_response("teaching", LOOPED_LESSON)
        self.assertFalse(result.checks.get("models_before_practice", False))
        self.assertFalse(result.checks.get("stages_in_order", False))
        self.assertLessEqual(result.score, 0.2)

    def test_empty_stage_phrases_do_not_count(self):
        hollow = LESSON.replace("Now it is your turn to try it with your cup.", "Let's practice.")
        self.assertFalse(score_response("teaching", hollow).checks["guided_practice"])

    def test_learner_voice_is_penalised(self):
        learner = TAUGHT_LESSON.replace("Which word tells", "First, I need to practice. Which word tells")
        self.assertEqual(score_response("teaching", TAUGHT_LESSON).score, 1.0)
        self.assertLess(score_response("teaching", learner).score, 0.9)


# Real pilot output: the answer talks about a story instead of telling one, and still earned about 0.9 before the fix.
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
STORY_PROMPT = 'Could this become a story? Please try: " This too shall pass "'


class TestStoryTalkExploit(unittest.TestCase):
    def test_talk_about_a_story_earns_almost_nothing(self):
        result = score_response("story", STORY_ABOUT_A_STORY, STORY_PROMPT)
        self.assertLessEqual(result.score, 0.1)
        self.assertGreaterEqual(meta_story_share(STORY_ABOUT_A_STORY), 1 / 3)

    def test_every_talk_frame_is_caught(self):
        frames = [
            "The story would involve a lot of emotional conflict.",           # the story as subject of a plan
            "Yes, this could become a story.",                                # could become a story
            "That could make a great story.",
            "To make it a story, the heroes would have to face challenges.",  # an instruction
            "Can the story be made?",
            "In this story, the civilians are not alone.",
            "Sure, this is a lovely idea for a story.",                       # assistant acknowledgement
            "Here's a possible story based on that line.",
            "Let me know if you would like more.",
            "The main conflict is a hero and a villain.",                     # a summary, not scenes
            "This line could be used to explore guilt.",
            "The story is about a girl named Lila and her family.",
        ]
        for frame in frames:
            self.assertTrue(is_meta_story(frame), frame)

    def test_narrative_sentences_are_not_caught(self):
        narrative = [
            "Mira told the old story to the children by the fire.",
            '"Your story will join the others here," she said.',
            "She learned that stories could bring joy to anyone.",
            "The village story was told again each winter.",
            "The hero decided to face the villain in a last battle.",
            "The story was about a hidden treasure, buried deep in the ground.",  # a character's own story
        ]
        for sentence in narrative:
            self.assertFalse(is_meta_story(sentence), sentence)

    def test_real_narrative_scores_unchanged(self):
        self.assertEqual(meta_story_share(STORY), 0.0)
        self.assertEqual(score_response("story", STORY).score, 1.0)
        for _, text in GOLD["story"]:
            self.assertEqual(meta_factor("story", text), 1.0)

    def test_credit_falls_with_the_share_of_talk(self):
        preface = "Here's a possible story based on that line. " + STORY
        mild = score_response("story", preface).score
        heavy = score_response("story", "To make it a story, it would need a hero. " * 3 + STORY).score
        self.assertLess(mild, 1.0)
        self.assertGreater(mild, 0.5)  # one stray sentence is a small cost
        self.assertLess(heavy, mild)
        self.assertEqual(meta_factor("teaching", STORY_ABOUT_A_STORY), 1.0)  # only stories are judged this way

    def test_talk_sentences_cannot_earn_story_stages(self):
        only_talk = "The story would involve a village. " + STORY_ABOUT_A_STORY
        self.assertLess(score_response("story", only_talk).score, 0.1)


FILLERS = [
    "Programming is a skill that people build slowly with patience and time.",
    "The weather stayed calm while the town went about its quiet day.",
    "Every morning the market opens early and the streets fill with people.",
    "Old buildings along the harbor have stood for more than two hundred years.",
    "Music often travels between countries long before the words are translated.",
    "Many families keep photographs on the shelves of their living rooms.",
    "Quiet evenings give writers plenty of space to think about their work.",
    "Trains across the valley run on a timetable posted near the station.",
]


def knock(behavior: str, index: int, *lines: int, swap: tuple[int, int] | None = None, edit=None) -> dict:
    """Checks of a gold answer after its given lines are replaced by cue-free filler (or two lines swapped)."""
    prompt, text = GOLD[behavior][index]
    parts = text.split("\n")
    for number, line in enumerate(lines):
        parts[line] = FILLERS[number]
    if swap:
        parts[swap[0]], parts[swap[1]] = parts[swap[1]], parts[swap[0]]
    text = "\n".join(parts)
    return score_response(behavior, edit(text) if edit else text, prompt).checks


class TestProcessTopics(unittest.TestCase):
    def assertFlips(self, behavior, index, check, *lines, **options):
        self.assertTrue(score_response(behavior, GOLD[behavior][index][1], GOLD[behavior][index][0]).checks[check], check)
        self.assertFalse(knock(behavior, index, *lines, **options)[check], f"{behavior}/{check} should fail")

    def test_gold_passes_every_check(self):
        for behavior in NEW_TOPICS:
            for prompt, text in GOLD[behavior]:
                result = score_response(behavior, text, prompt)
                self.assertEqual(result.score, 1.0, f"{behavior} failed {[k for k, v in result.checks.items() if not v]}")

    def test_debugging_each_stage_is_required(self):
        self.assertFlips("debugging", 0, "restates_failure", 0, 1, 2)
        self.assertFlips("debugging", 0, "reproduction_steps", 1)
        self.assertFlips("debugging", 0, "root_cause_reasoning", 2, 3)
        self.assertFlips("debugging", 0, "cause_is_reasoned", 3)
        self.assertFlips("debugging", 0, "concrete_fix", 4, 5, 6)
        self.assertFlips("debugging", 0, "verification", 5)
        self.assertFlips("debugging", 0, "regression_test", 6)
        self.assertFlips("debugging", 0, "stages_in_order", swap=(0, 6))

    def test_debugging_fix_must_be_about_the_cause(self):
        elsewhere = "The fix is to wrap everything in a retry loop with `time.sleep(1)` after each attempt."
        text = GOLD["debugging"][0][1].replace(GOLD["debugging"][0][1].split("\n")[4], elsewhere)
        checks = score_response("debugging", text, GOLD["debugging"][0][0]).checks
        self.assertTrue(checks["concrete_fix"])
        self.assertFalse(checks["fix_addresses_cause"])

    def test_debugging_fix_needs_code_not_just_advice(self):
        advice = "The fix is to return early when the list is empty, so nothing gets divided by zero."
        text = GOLD["debugging"][0][1].replace(GOLD["debugging"][0][1].split("\n")[4], advice)
        self.assertFalse(score_response("debugging", text).checks["concrete_fix"])

    def test_announcing_a_stage_earns_nothing(self):
        announce = ["First I will explain the root cause because the function divides by zero for an empty list.",
                    "Then I will describe the reason, which means the crash is not random at all."]
        parts = GOLD["debugging"][0][1].split("\n")
        text = "\n".join(parts[:2] + announce + parts[4:])
        self.assertTrue(all(ANNOUNCE.search(line) for line in announce))
        self.assertEqual(len(parse(text)[0]), 5)
        self.assertFalse(score_response("debugging", text).checks["root_cause_reasoning"])

    def test_math_each_stage_is_required(self):
        self.assertFlips("math_solving", 0, "restates_question", 0)
        self.assertFlips("math_solving", 0, "states_plan", 1)
        self.assertFlips("math_solving", 0, "calculations_shown", 2, 3, 4)
        self.assertFlips("math_solving", 0, "work_between_plan_and_answer", 3, 4)
        self.assertFlips("math_solving", 0, "final_answer_marked", 5)
        self.assertFlips("math_solving", 0, "checks_the_answer", 6)
        self.assertFlips("math_solving", 0, "stages_in_order", swap=(5, 6))

    def test_math_steps_must_be_numbered_in_a_row(self):
        text = GOLD["math_solving"][0][1].replace("2. Muffins", "5. Muffins")
        self.assertFalse(score_response("math_solving", text).checks["numbered_steps"])

    def test_math_answer_must_come_from_the_work(self):
        text = GOLD["math_solving"][0][1].replace("sells 240 muffins in 5 days", "sells 999 muffins in 5 days")
        checks = score_response("math_solving", text, GOLD["math_solving"][0][0]).checks
        self.assertTrue(checks["final_answer_marked"])
        self.assertFalse(checks["answer_matches_work"])

    def test_math_check_needs_a_real_recalculation(self):
        text = GOLD["math_solving"][0][1].replace("Check: 240 muffins ÷ 5 days = 48 muffins a day, which matches the first step.",
                                                  "Check: this answer looks fine to me and I am happy with it now.")
        self.assertFalse(score_response("math_solving", text).checks["checks_the_answer"])

    def test_explanation_each_stage_is_required(self):
        self.assertFlips("explanation", 0, "plain_answer_first", 0)
        self.assertFlips("explanation", 0, "explains_why_or_how", 1)
        self.assertFlips("explanation", 0, "everyday_analogy", 2)
        self.assertFlips("explanation", 0, "short_recap_echoes_answer", 4)
        self.assertFlips("explanation", 0, "stages_in_order", swap=(2, 3))

    def test_explanation_example_must_be_concrete(self):
        vague = "For example, some people say it is true."
        text = GOLD["explanation"][0][1].replace(GOLD["explanation"][0][1].split("\n")[3], vague)
        self.assertFalse(score_response("explanation", text).checks["concrete_example"])

    def test_explanation_recap_must_repeat_the_answer(self):
        other = "In short, remember that learning new things takes some time and effort for everyone."
        text = GOLD["explanation"][0][1].replace(GOLD["explanation"][0][1].split("\n")[4], other)
        self.assertFalse(score_response("explanation", text).checks["short_recap_echoes_answer"])

    def test_explanation_hard_words_lose_simple_language(self):
        jargon = (" Electromagnetic radiation, interacting with atmospheric constituents, produces characteristic "
                  "wavelength-dependent scattering phenomena, notably Rayleigh scattering, depending on polarization "
                  "characteristics, atmospheric composition, observational circumstances.")
        text = GOLD["explanation"][0][1].replace("other colors.", "other colors." + jargon)
        self.assertFalse(score_response("explanation", text).checks["simple_language"])

    def test_explanation_opener_is_not_an_answer(self):
        text = "Great question, let me explain this to you today.\n" + GOLD["explanation"][0][1]
        self.assertFalse(score_response("explanation", text).checks["plain_answer_first"])

    def test_email_each_part_is_required(self):
        self.assertFlips("professional_email", 0, "subject_line", 0)
        self.assertFlips("professional_email", 0, "greeting_with_name", 1)
        self.assertFlips("professional_email", 0, "purpose_early", 2)
        self.assertFlips("professional_email", 0, "specific_details", 4)
        self.assertFlips("professional_email", 0, "clear_request", 5)
        self.assertFlips("professional_email", 0, "polite_close", 6)
        self.assertFlips("professional_email", 0, "sign_off_with_name", 7)
        self.assertFlips("professional_email", 0, "stages_in_order", swap=(5, 6))

    def test_email_request_needs_a_deadline_or_next_step(self):
        text = GOLD["professional_email"][0][1].replace(" by Wednesday at noon", "")
        checks = score_response("professional_email", text).checks
        self.assertTrue(checks["clear_request"])
        self.assertFalse(checks["deadline_or_next_step"])

    def test_email_tone_and_length(self):
        base = GOLD["professional_email"][0][1]
        self.assertFalse(score_response("professional_email", base.replace("Thank you", "Hey, thanks lol")).checks["professional_tone"])
        self.assertFalse(score_response("professional_email", base.replace("notice.", "notice!!!")).checks["professional_tone"])
        rambling = "\n".join(text for name in ("story", "teaching") for _, text in GOLD[name])  # 400 different words
        long = base.replace("Kind regards,", rambling + "\nKind regards,")
        self.assertFalse(score_response("professional_email", long).checks["concise"])

    def test_email_subject_must_open_the_email(self):
        text = "Here is your email.\n" + GOLD["professional_email"][0][1]
        self.assertFalse(score_response("professional_email", text).checks["subject_line"])

    def test_how_to_each_part_is_required(self):
        self.assertFlips("how_to_guide", 0, "states_goal", 0)
        self.assertFlips("how_to_guide", 0, "warning_or_common_mistake", 11)
        self.assertFlips("how_to_guide", 0, "confirms_success", 12)
        self.assertFlips("how_to_guide", 0, "stages_in_order", swap=(0, 12))

    def test_how_to_materials_need_items(self):
        self.assertFlips("how_to_guide", 0, "lists_materials", 2, 3, 4)

    def test_how_to_steps_numbered_verbs_and_substance(self):
        steps = GOLD["how_to_guide"][0][1].split("\n")
        skipped = "\n".join(steps).replace("2. Let", "9. Let")
        self.assertFalse(score_response("how_to_guide", skipped).checks["numbered_steps"])
        nouns = ["The bike is flipped over and the wheel comes off the frame.", "The air is let out of the tire.",
                 "The old tube is pulled out of the tire.", "A new tube goes inside the tire."]
        for n, noun in enumerate(nouns):
            steps[5 + n] = f"{n + 1}. {noun}"
        self.assertFalse(score_response("how_to_guide", "\n".join(steps)).checks["steps_start_with_verbs"])
        short = [f"{n + 1}. Flip it." if n == 0 else f"{n + 1}. Do it." for n in range(6)]
        steps[5:11] = short
        self.assertFalse(score_response("how_to_guide", "\n".join(steps)).checks["steps_have_substance"])

    def test_imperative_verbs(self):
        for step in ("Preheat the oven.", "Click Save.", "Stir gently until smooth.", "Fit the wheel back on."):
            self.assertTrue(imperative(step), step)
        for step in ("The oven is hot.", "You should stir it.", "Cooking is fun.", "Spaghetti is ready."):
            self.assertFalse(imperative(step), step)

    def test_parse_drops_repeats_keeps_step_numbers_and_joins_code(self):
        items, steps = parse("1. Open the box now.\n2. Close the box now.\n2. Close the box now.\nUse this:\n`x = 1`")
        self.assertEqual(steps, {0: 1, 1: 2})
        self.assertEqual(items, ["Open the box now.", "Close the box now.", "Use this: `x = 1`"])
        self.assertEqual(numbers("12 boxes cost $3,400.50"), {"12", "3400.50"})

    def test_typical_plain_answers_score_low(self):
        plain = {
            "debugging": ("Sure! A ZeroDivisionError happens when you divide by zero. You can fix this by checking whether "
                          "the list is empty before dividing. Here is an updated version of your function:\n```python\n"
                          "def average(numbers):\n    if len(numbers) == 0:\n        return 0\n    return sum(numbers) / "
                          "len(numbers)\n```\nLet me know if you have more questions about your Python code."),
            "math_solving": ("The bakery sells 12 boxes with 4 muffins each day, so that is 48 muffins every day. "
                             "Over 5 days that comes to 240 muffins in total, which is the number you wanted to know."),
            "explanation": ("The sky is blue because of Rayleigh scattering, where shorter wavelengths of sunlight are "
                            "scattered by molecules in the atmosphere much more strongly than longer wavelengths are."),
            "professional_email": ("Hi, I cannot come to the meeting on Thursday because I have a dentist appointment. "
                                   "Can we move it to another day? Thanks a lot for understanding and see you soon, "
                                   "I hope that works for everybody on the team and that nobody minds the change."),
            "how_to_guide": ("To change a flat tire you take the wheel off, remove the old tube, put in a new tube and "
                             "put the wheel back on. It takes about fifteen minutes once you have done it a few times, "
                             "and a pump helps a lot with the last part of the job at the end."),
        }
        for behavior, text in plain.items():
            result = score_response(behavior, text, GOLD[behavior][0][0])
            self.assertLessEqual(result.score, 0.5, f"{behavior} scored {result.score}")

    def test_ignoring_the_prompt_loses_the_score(self):
        for behavior in NEW_TOPICS:
            (_, text), (other, _) = GOLD[behavior][0], GOLD[behavior][1]
            wrong = score_response(behavior, text, other).score
            self.assertLess(wrong, 0.5, behavior)

    def test_repeating_cue_words_earns_nothing(self):
        for behavior in NEW_TOPICS:
            text = GOLD[behavior][0][1]
            for seed in range(3):
                self.assertTrue(score_response(behavior, stuffed(behavior, text, seed)).degenerate, behavior)
                self.assertLessEqual(score_response(behavior, stuffed(behavior, text, seed, mixed=True)).score, 0.5, behavior)

    def test_loops_and_word_salad_are_degenerate(self):
        for behavior in NEW_TOPICS:
            self.assertTrue(score_response(behavior, looped(behavior, GOLD[behavior][0][1])).degenerate, behavior)
        salad = (" ".join(FILLER) + " requirements design tests climax recap").lower()  # 60 different topic words
        self.assertTrue(is_degenerate(salad))  # no "the", "and", "to": not prose
        self.assertFalse(is_degenerate(salad + " " + GOLD["story"][0][1]))

    def test_shuffled_text_loses_points(self):
        for behavior in NEW_TOPICS:
            prompt, text = GOLD[behavior][0]
            parts = text.split("\n")
            backwards = "\n".join(reversed(parts))
            self.assertLess(score_response(behavior, backwards, prompt).score, 0.8, behavior)


class TestCleaning(unittest.TestCase):
    def test_detokenize(self):
        self.assertEqual(detokenize("He did n't go , did he ?"), "He didn't go, did he?")

    def test_story_prompt_filters(self):
        self.assertEqual(clean_story_prompt("")[1], "blank")
        self.assertEqual(clean_story_prompt("[ IP ] A quiet lake at dawn today")[1], "not_wp_tag")
        self.assertEqual(clean_story_prompt("[ WP ] A nude man walks into town")[1], "adult")
        self.assertEqual(clean_story_prompt("[ WP ] Stuck")[1], "bad_length")

    def test_curly_quotes_are_not_foreign_text(self):
        text, reason = clean_story_prompt("[ WP ] “ I hate you. ” “ Then why hug me ? ”")
        self.assertIsNone(reason)
        self.assertIn('"', text)

    def test_lesson_filters(self):
        fields = lesson_fields("Subject: History, Topic: Rivers, Class: PRIMARY 3")
        self.assertEqual(fields, ("History", "Rivers", "PRIMARY 3"))
        self.assertIsNone(lesson_reason(*fields))
        self.assertEqual(lesson_reason("History", "Rivers", "SENIOR SECONDARY SCHOOL"), "non_primary")
        self.assertEqual(lesson_reason("History", "", "PRIMARY 3"), "blank")

    def test_grade_spacing_is_the_same_lesson(self):
        self.assertEqual(lesson_key("Art", "Drawing", "PRIMARY 4"), lesson_key("Art", "Drawing", "PRIMARY4"))
        self.assertNotEqual(lesson_key("Art", "Drawing", "PRIMARY 4"), lesson_key("Art", "Drawing", "PRIMARY 5"))

    def test_topic_tidy(self):
        self.assertEqual(tidy_case("- Plants"), "Plants")
        self.assertEqual(tidy_case("ACIDS"), "Acids")

    def test_split_ignores_case_and_punctuation(self):
        self.assertEqual(is_test("Hello, World!"), is_test("hello world"))

    def test_app_idea_summary_drops_links(self):
        name, text = app_idea_summary("# Timer\n\n**Tier:** 1-Beginner\n\nA [simple](http://x.y) timer.\n\n## Stories")
        self.assertEqual((name, text), ("Timer", "A simple timer."))


if __name__ == "__main__":
    unittest.main()
