"""Run: python -m unittest test_behaviors.py"""

import unittest

from behaviors.prepare_data import (
    app_idea_summary, clean_story_prompt, detokenize, is_test, lesson_fields, lesson_key, lesson_reason,
    tidy_case,
)
from behaviors.gold_examples import GOLD
from behaviors.rubrics import FREE_REPETITION, repetition, score_response, sentences, unique_tree, tree_lines

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
