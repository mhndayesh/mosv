"""Run: python -m unittest test_router.py"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from router import CLASSES, NONE_INDEX
from router.dataset import (
    FINAL_TEST, OTHER_WRITING, TASK_WORDS, TEMPLATES, UNCLEAR_BUILD, alpaca_label, apply_review, body_fields,
    check_rows, is_clean_none, is_near_miss, make_row, render,
)
from router.pipeline import MAX_BODY_TOKENS, check_result, encode, load_audited, plan_batches, save_chunk
from router.train import decide, false_steer_rate, pick_threshold, safety, safety_pass


def rows(count=10):
    return [{"id": f"id{i:03d}", "split": "train", "label": CLASSES[i % 4], "group": f"g{i}", "text": "x" * (i + 1)}
            for i in range(count)]


def result(batch_id, items):
    return {"batch_id": batch_id, "ids": [r["id"] for r in items],
            "labels": [CLASSES.index(r["label"]) for r in items], "splits": [r["split"] for r in items],
            "features": np.ones((len(items), 1024), dtype=np.float32)}


class TestPlan(unittest.TestCase):
    def test_every_prompt_in_exactly_one_pipeline(self):
        plans = plan_batches(rows(101), size=8)
        first = {i for task in plans["main"] for i in task["ids"]}
        second = {i for task in plans["second"] for i in task["ids"]}
        self.assertFalse(first & second)
        self.assertEqual(len(first | second), 101)

    def test_labels_stay_with_their_ids(self):
        source = {r["id"]: r for r in rows(40)}
        for plan in plan_batches(list(source.values()), size=7).values():
            for task in plan:
                for uid, label in zip(task["ids"], task["labels"]):
                    self.assertEqual(CLASSES[label], source[uid]["label"])


class FakeTokenizer:
    def encode_batch(self, texts, add_special_tokens=False):
        return [type("Item", (), {"ids": [len(text)] * len(text)})() for text in texts]


class TestEncode(unittest.TestCase):
    def test_left_pad_keeps_assistant_marker_last(self):
        batch = encode(FakeTokenizer(), ["ab", "abcd"], [7], [8, 9], 0)
        self.assertEqual(batch["input_ids"].tolist(), [[0, 0, 7, 2, 2, 8, 9], [7, 4, 4, 4, 4, 8, 9]])
        self.assertEqual(batch["attention_mask"].tolist(), [[0, 0, 1, 1, 1, 1, 1], [1] * 7])

    def test_long_text_trims_body_only(self):
        batch = encode(FakeTokenizer(), ["x" * (MAX_BODY_TOKENS + 50)], [7], [8, 9], 0)
        self.assertEqual(batch["input_ids"].shape[1], MAX_BODY_TOKENS + 3)
        self.assertEqual(batch["input_ids"][0, -2:].tolist(), [8, 9])


class TestAudit(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.cache = Path(self.folder.name)
        self.rows = rows(8)

    def tearDown(self):
        self.folder.cleanup()

    def test_complete_set_passes(self):
        save_chunk(result("a", self.rows[:4]), self.cache)
        save_chunk(result("b", self.rows[4:]), self.cache)
        self.assertEqual(load_audited(self.rows, self.cache)["train"][0].shape, (8, 1024))

    def test_missing_prompt_fails(self):
        save_chunk(result("a", self.rows[:7]), self.cache)
        with self.assertRaisesRegex(ValueError, "missing"):
            load_audited(self.rows, self.cache)

    def test_duplicate_prompt_fails(self):
        save_chunk(result("a", self.rows), self.cache)
        save_chunk(result("b", self.rows[:1]), self.cache)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_audited(self.rows, self.cache)

    def test_wrong_label_fails(self):
        bad = result("a", self.rows)
        bad["labels"][0] = (bad["labels"][0] + 1) % 4
        save_chunk(bad, self.cache)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            load_audited(self.rows, self.cache)

    def test_bad_features_rejected(self):
        bad = result("a", self.rows)
        bad["features"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            check_result(bad)


class TestDecision(unittest.TestCase):
    def test_low_confidence_becomes_none(self):
        probabilities = np.array([[0.9, 0.05, 0.03, 0.02], [0.4, 0.3, 0.2, 0.1]])
        self.assertEqual(decide(probabilities, 0.6).tolist(), [0, NONE_INDEX])

    def test_false_steer_counts_only_none_prompts(self):
        self.assertEqual(false_steer_rate(np.array([0, 3, 1]), np.array([0, 3, 3])), 0.5)

    def test_safety_counts_miss_separately_from_wrong(self):
        labels = np.array([0, 0, 1, 2, NONE_INDEX, NONE_INDEX])
        choices = np.array([0, NONE_INDEX, 2, 2, NONE_INDEX, 1])
        result = safety(choices, labels)
        self.assertEqual((result["correct_steers"], result["missed_left_unchanged"]), (2, 1))
        self.assertEqual((result["wrong_expert"], result["false_steer"]), (1, 1))
        self.assertEqual(result["precision"], 0.5)

    def test_misses_pass_but_wrong_expert_fails(self):
        labels = np.array([0] * 90 + [1] * 10 + [NONE_INDEX] * 100)
        missed = labels.copy()
        missed[:10] = NONE_INDEX
        self.assertTrue(all(safety_pass(safety(missed, labels), 0.0, 0.0).values()))
        wrong = labels.copy()
        wrong[:10] = 1
        self.assertFalse(safety_pass(safety(wrong, labels), 0.0, 0.0)["wrong_expert"])

    def test_threshold_blocks_confident_wrong_steer(self):
        probabilities = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.7], [0.95, 0.03, 0.01, 0.01]])
        labels = np.array([NONE_INDEX, NONE_INDEX, 0])
        threshold = pick_threshold(probabilities, labels)
        self.assertEqual(false_steer_rate(decide(probabilities, threshold), labels), 0.0)
        self.assertEqual(decide(probabilities, threshold)[2], 0)


class TestDataset(unittest.TestCase):
    def test_body_removes_dataset_wording(self):
        self.assertEqual(body_fields("story", "Write a short story about this idea: A cat flies."), ("A cat flies.", ""))
        self.assertEqual(body_fields("teaching", "Teach Plants (Science) to a Primary 2 student."),
                         ("Plants (Science)", "Primary 2"))

    def test_test_split_uses_unseen_wording(self):
        text = render("story", "test", "A cat flies.", "", "key")
        self.assertFalse(text.startswith(("Write a short story", "Tell me a story", "Write fiction")))

    def test_wording_never_shared_between_splits(self):
        for label, pools in TEMPLATES.items():
            self.assertFalse(set(pools["train"]) & set(pools["val"]), label)
            self.assertFalse(set(pools["train"]) & set(pools["test"]), label)
            self.assertFalse(set(pools["val"]) & set(pools["test"]), label)

    def test_training_has_hundreds_of_wordings(self):
        for label, pools in TEMPLATES.items():
            self.assertGreaterEqual(len(pools["train"]), 200, label)

    def test_alpaca_clear_requests_get_a_label(self):
        self.assertEqual(alpaca_label("Write a story about a robot who is trying to learn"), "story")
        self.assertEqual(alpaca_label("Create an app that allows user to book a taxi."), "engineering")
        self.assertEqual(alpaca_label("Explain the process of photosynthesis to a child."), "teaching")
        self.assertEqual(alpaca_label("Create a lesson plan that teaches students to use a ruler."), "teaching")

    def test_alpaca_look_alikes_are_rejected(self):
        for text in ("Create a marketing plan to promote a website.", "Write a question about website design",
                     "Come up with a creative opening line for a story.", "Design a homepage for a new website.",
                     "Write a script of a conversation between two people.",
                     "Design a customer loyalty program that provides points to customers.",
                     "Create five short headlines for a news story about a movie star.",
                     "Design a job application form", "Write an effective job application for an engineer.",
                     "Design a chatbot conversational flow to make a reservation.",
                     "Write a poem that tells a romantic love story.", "Generate a new character for a fairy-tale.",
                     "Create a website navigation menu that contains four options.",
                     "Create a scenario for a story set in Italy.", "Tell the story of why the sky is blue",
                     "Generate two possible titles for a story about a robot.",
                     "Create a prompt that can be used to generate a story.",
                     "Generate a creative opening sentence for a story.", "Compose a song that tells a story",
                     "Create a new setting for a story.", "Create a unique character in a fairy tale."):
            self.assertIsNone(alpaca_label(text), text)

    def test_software_builds_cannot_enter_plain_none(self):
        for text in ("Write a web crawler in Python.", "Design a database schema for storing customer orders.",
                     "Create a standard HTML page with a table.", "Create a regex pattern that matches emails.",
                     "Design a system to help manage support tickets.", "Create a form for a newsletter.",
                     "Design an AI-powered chatbot."):
            self.assertTrue(TASK_WORDS.search(text) or UNCLEAR_BUILD.search(text), text)
        for text in ("Give three tips for staying healthy.", "Create a blog post about supply chains."):
            self.assertFalse(TASK_WORDS.search(text) or UNCLEAR_BUILD.search(text), text)

    def test_other_writing_is_none_but_fiction_is_unclear(self):
        self.assertTrue(OTHER_WRITING.search("Construct a 5-paragraph essay about aerosols."))
        self.assertTrue(is_clean_none("Write a haiku about the sea."))
        self.assertFalse(is_clean_none("Create a short allegory about knowledge."))
        self.assertFalse(is_clean_none("Write a prologue for a fanfiction about heroes."))

    def test_final_test_is_balanced_and_unique(self):
        items = [text for texts in FINAL_TEST.values() for text in texts]
        self.assertEqual({len(texts) for texts in FINAL_TEST.values()}, {15})
        self.assertEqual(len(items), len(set(items)))

    def test_review_keeps_only_confirmed_task_labels(self):
        tags = {"ok": ("story", False), "wrong": ("story", False), "missing": ("engineering", False),
                "question": (None, True)}
        result = apply_review(tags, {"ok": "story", "wrong": "none"})
        self.assertEqual(result["ok"], ("story", False))
        self.assertEqual(result["wrong"], (None, False))
        self.assertEqual(result["missing"], (None, False))
        self.assertEqual(result["question"], (None, True))

    def test_near_miss_is_a_small_task_question(self):
        self.assertTrue(is_near_miss("What is a function in programming?"))
        self.assertFalse(is_near_miss("Write a story about a dragon."))

    def test_shared_topic_across_splits_raises(self):
        with self.assertRaisesRegex(ValueError, "topic shared"):
            check_rows([make_row("story", "train", "same", "one"), make_row("story", "test", "same", "two")])


if __name__ == "__main__":
    unittest.main()
