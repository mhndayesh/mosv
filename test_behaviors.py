"""Run: python -m unittest test_behaviors.py"""

import unittest

from behaviors.prepare_data import (
    app_idea_summary, clean_story_prompt, detokenize, is_test, lesson_fields, lesson_key, lesson_reason,
    tidy_case,
)
from behaviors.gold_examples import GOLD
from behaviors.rubrics import score_response, sentences

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
