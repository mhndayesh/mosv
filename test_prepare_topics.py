"""Run: python -m unittest test_prepare_topics.py"""

import json
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

os.environ.setdefault("HIP_VISIBLE_DEVICES", "-1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from behaviors import prepare_topics as pt
from router import CLASSES, TOPICS_V1, TOPICS_V2, classes_for, resolve_topics
from router import dataset as rd
from rl import data as rl_data


CONTEXT = ("Purpose: {i}\nRecipient: Ann Lee (Chief Financial Officer)\nSender: Raj Patel (Analyst)\n"
           "Company: Acme Corp\nKey Points: Costs, Timeline, Next steps")


def kept(result):
    return result[0] == "keep"


class TestCommonChecks(unittest.TestCase):
    def test_each_unusable_prompt_gets_its_reason(self):
        limits = (3, 10)
        cases = {"   ": "blank", "Why is the � sky blue": "broken_text", "How to have sex tonight": "adult",
                 "一个新员工入职的 RPG game": "not_english", "Two words": "too_short",
                 "one two three four five six seven eight nine ten eleven": "too_long"}
        for text, reason in cases.items():
            self.assertEqual(pt.common_reason(text, limits), reason, text)

    def test_words_that_only_contain_a_bad_word_are_kept(self):
        for text in ("Jim drove from Virginia to Texas and back again", "Pour water out of the jug into a cup",
                     "Can you make out of twelve mangoes how many lassis"):
            self.assertIsNone(pt.common_reason(text, (3, 30)), text)

    def test_math_symbols_do_not_count_as_another_language(self):
        self.assertIsNone(pt.common_reason("550 − ( 104 ÷ 20.8 ) = ? Find the value", (3, 30)))

    def test_mojibake_is_rejected(self):
        self.assertEqual(pt.common_reason("the banker â € ™ s gain is rs . 180", (3, 30)), "broken_text")


class TestMath(unittest.TestCase):
    def test_question_kept_and_leaked_answers_dropped(self):
        self.assertTrue(kept(pt.clean_math(("gsm8k", "Tom has 3 apples and buys 4 more. How many apples does he have?"))))
        self.assertEqual(pt.clean_math(("gsm8k", "Tom has 3 apples. #### 7 apples"))[1], "contains_answer")
        self.assertEqual(pt.clean_math(("math", "Find x. The answer is 12 for sure.")) [1], "contains_answer")
        self.assertEqual(pt.clean_math(("math", "Solve $x^2=4$. The solution: x = 2"))[1], "contains_answer")

    def test_asking_for_the_answer_is_not_a_leak(self):
        self.assertTrue(kept(pt.clean_math(("math", "Find x if 2x = 8. Express your answer as an integer."))))

    def test_diagram_code_dropped(self):
        self.assertEqual(pt.clean_math(("math", "Find the angle shown. [asy] draw(circle((0,0),1)); [/asy]"))[1],
                         "diagram_code")


class TestEmail(unittest.TestCase):
    def test_prompt_is_instruction_plus_input_and_groups_on_input(self):
        context = CONTEXT.format(i="Budget review")
        result = pt.clean_email(("Write a formal email.", context))
        self.assertTrue(kept(result))
        self.assertEqual(result[1], "Write a formal email.\n\n" + context)
        self.assertEqual(result[2], context)

    def test_blank_input_dropped(self):
        self.assertEqual(pt.clean_email(("Write a formal email.", "  "))[1], "blank")


class TestExplanation(unittest.TestCase):
    def test_title_is_cut_from_the_body_and_removed_marker(self):
        self.assertEqual(pt.clean_eli5("Why does gravity affect time?. [removed]")[1], "Why does gravity affect time?")
        text = "If space is so dark, why are there stars?. fyi, i am not high\n\nthanks"
        self.assertEqual(pt.clean_eli5(text)[1], "If space is so dark, why are there stars?")

    def test_non_questions_and_how_to_and_context_dropped(self):
        self.assertEqual(pt.clean_eli5("Apple vs. Samsung lawsuits and the like")[1], "not_a_question")
        self.assertEqual(pt.clean_eli5("How to read poetry like a pro?")[1], "how_to_not_explanation")
        self.assertEqual(pt.clean_eli5("Why is it so? Edit: thanks for the answers")[1], "reddit_context")
        self.assertEqual(pt.clean_eli5("Hydrogen is a gas. I read this somewhere\nwhat?")[1], "has_body_text")

    def test_adult_question_dropped(self):
        self.assertEqual(pt.clean_eli5("How easy is it to contract an STD from someone?")[1], "adult")


class TestHowTo(unittest.TestCase):
    def test_part_numbers_removed_only_when_they_form_a_series(self):
        titles = ["How to Be an Organized Artist1", "How to Be an Organized Artist2", "How to Take Vitamin B12",
                  "How to Tell Your Parents You Move Out when You Turn 181", "How to Tell Your Parents You Move Out when You Turn 182",
                  "How to Make Cat Toys10"] + [f"How to Make Cat Toys{i}" for i in range(1, 10)]
        cleaned = dict(zip(titles, pt.strip_footnotes(titles)))
        self.assertEqual(cleaned["How to Be an Organized Artist2"], "How to Be an Organized Artist")
        self.assertEqual(cleaned["How to Take Vitamin B12"], "How to Take Vitamin B12")
        self.assertEqual(cleaned["How to Tell Your Parents You Move Out when You Turn 182"],
                         "How to Tell Your Parents You Move Out when You Turn 18")
        self.assertEqual(cleaned["How to Make Cat Toys10"], "How to Make Cat Toys")

    def test_lone_part_number_after_a_letter_removed_but_real_numbers_stay(self):
        cleaned = pt.strip_footnotes(["How to Dunk on NBA 2K14", "How to Grow Roses3", "How to Install Windows 10"])
        self.assertEqual(cleaned, ["How to Dunk on NBA 2K14", "How to Grow Roses", "How to Install Windows 10"])

    def test_only_how_to_titles_kept(self):
        self.assertTrue(kept(pt.clean_wikihow(("How to Grow Roses1", "How to Grow Roses"))))
        self.assertEqual(pt.clean_wikihow(("Roses", "Roses is a flower"))[1], "not_how_to")
        self.assertEqual(pt.clean_wikihow(("How to Seduce Your Friend", "How to Seduce Your Friend"))[1], "adult")


class TestBugs(unittest.TestCase):
    def test_real_bug_becomes_a_debug_request_without_fix_details(self):
        result = pt.clean_bug(("Eclipse_Platform_UI", "Bug 123 NPE when closing the editor",
                               "Open a file, close it, and a NullPointerException appears in the log."))
        self.assertTrue(kept(result))
        self.assertIn("Eclipse Platform UI", result[1])
        self.assertIn("NPE when closing the editor", result[1])
        self.assertNotIn("Bug 123", result[1])

    def test_features_empty_and_fixed_reports_dropped(self):
        cases = {("JDT", "Bug 1 Add support for records", "Please add this to the formatter and the editor view."): "feature_request",
                 ("JDT", "Bug 2 Crash on save", "None"): "empty_description",
                 ("JDT", "Bug 3 Crash on save", "Saving crashes the editor. Fixed in HEAD by the nightly build."): "mentions_fix",
                 ("JDT", "Bug 4 Label colors", "The label colors are shown in the dialog on screen today."): "no_bug_evidence"}
        for row, reason in cases.items():
            self.assertEqual(pt.clean_bug(row)[1], reason, row)

    def test_wrapper_is_stable_for_the_same_title(self):
        self.assertEqual(pt.debug_prompt("Birt", "A title", "body text"), pt.debug_prompt("Birt", "A title", "body text"))


class TestEngineering(unittest.TestCase):
    def test_build_requests_kept_and_other_things_dropped(self):
        ok = "Build a fast web app for tracking gym sessions with charts and login."
        self.assertTrue(kept(pt.clean_arena(("en", ok))))
        self.assertEqual(pt.clean_arena(("zh", ok))[1], "not_english")
        self.assertEqual(pt.clean_arena(("en", "<!DOCTYPE html><html><body>hi there you</body></html>"))[1], "pasted_code")
        self.assertEqual(pt.clean_arena(("en", "Tell me about the weather in Paris this week please"))[1], "not_a_build_request")
        self.assertEqual(pt.clean_arena(("en", "Create a logo for my bakery in blue and gold tones"))[1], "not_a_build_request")

    def test_existing_prompts_pass_through_unchanged(self):
        row = {"prompt": "Create a new software project for Mystic Maze. A 3D game.", "group": "mystic maze"}
        self.assertEqual(pt.clean_existing(row), ("keep", row["prompt"], "mystic maze"))


class TestCapAndSplit(unittest.TestCase):
    def rows(self, source, count):
        return [{"group": f"{source}{i}", "source": source} for i in range(count)]

    def test_cap_shares_room_and_small_sources_keep_everything(self):
        out = pt.balanced_cap({"a": self.rows("a", 100), "b": self.rows("b", 100), "c": self.rows("c", 10)}, 60)
        self.assertEqual(Counter(r["source"] for r in out), {"a": 25, "b": 25, "c": 10})

    def test_below_cap_nothing_is_removed_and_choice_is_stable(self):
        small = {"a": self.rows("a", 5)}
        self.assertEqual(len(pt.balanced_cap(small, 50)), 5)
        big = {"a": self.rows("a", 100)}
        self.assertEqual(pt.balanced_cap(big, 20), pt.balanced_cap(big, 20))

    def test_split_is_stable_near_80_10_10_and_never_shares_a_group(self):
        rows = [{"group": f"topic number {i}"} for i in range(5000)]
        parts = pt.split_rows(rows)
        sizes = {k: len(v) for k, v in parts.items()}
        self.assertEqual(sum(sizes.values()), 5000)
        self.assertTrue(3800 < sizes["train"] < 4200 and 350 < sizes["dev"] < 650 and 350 < sizes["test"] < 650, sizes)
        self.assertEqual(pt.split_of("Topic  Number 7!"), pt.split_of("topic number 7"))

    def test_rows_sharing_a_group_stay_together(self):
        rows = [{"group": "same lesson topic", "prompt": f"Teach it to Primary {n}"} for n in range(1, 7)]
        parts = pt.split_rows(rows)
        self.assertEqual(sorted(len(v) for v in parts.values()), [0, 0, 6])

    def test_old_test_split_stays_test(self):
        from behaviors.prepare_data import is_test
        for group in (f"idea {i}" for i in range(300)):
            if is_test(group):
                self.assertEqual(pt.split_of(group), "test")


class TestBuild(unittest.TestCase):
    """End to end on tiny made-up sources, with the real multi-process cleaning."""

    def setUp(self):
        questions = [f"Why does thing number {i} happen in the world around us?" for i in range(40)]
        bad = ["Apple versus Samsung lawsuits are everywhere these days","Why is it sexy to see this ever?", f"Why does thing number {3} happen in the world around us?"]
        self.sources = (
            ("explanation", "eli5", "eli5", lambda: questions + bad),
            ("professional_email", "mail", "email", lambda: [("Write an email.", CONTEXT.format(i=f"item {i}"))
                                                          for i in range(30)] + [("Write an email.", "")]),
        )

    def run_build(self, cap):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(pt, "SOURCES", self.sources):
            out = Path(folder)
            report = pt.build(("explanation", "professional_email"), out, cap)
            files = {p.relative_to(out).as_posix(): [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
                     for p in out.rglob("*.jsonl")}
            saved = json.loads((out / "report.json").read_text(encoding="utf-8"))
        return report, files, saved

    def test_report_counts_reasons_splits_and_samples(self):
        report, files, saved = self.run_build(cap=1000)
        info = report["topics"]["explanation"]
        self.assertEqual((info["raw_rows"], info["kept_rows"]), (43, 40))
        self.assertEqual(info["rejected_by_reason"], {"not_a_question": 1, "adult": 1, "duplicate": 1})
        self.assertEqual(sum(info["splits"].values()), 40)
        self.assertEqual(len(info["sample_prompts"]), 3)
        self.assertEqual(report["topics"]["professional_email"]["rejected_by_reason"], {"blank": 1})
        self.assertEqual(saved["topics"].keys(), report["topics"].keys())
        total = sum(len(v) for k, v in files.items() if k.startswith("explanation/"))
        self.assertEqual(total, 40)

    def test_cap_limits_the_topic_and_reports_what_was_left_out(self):
        report, files, _ = self.run_build(cap=25)
        info = report["topics"]["explanation"]
        self.assertEqual((info["final_rows"], info["over_cap_left_out"]), (25, 15))
        split_groups = [{r["group"] for r in files[f"explanation/{n}.jsonl"]} for n in ("train", "dev", "test")]
        self.assertFalse(split_groups[0] & split_groups[1] or split_groups[0] & split_groups[2] or split_groups[1] & split_groups[2])


class TestConfigurableTopics(unittest.TestCase):
    def test_default_is_the_original_three_topics(self):
        self.assertEqual(resolve_topics(None), TOPICS_V1)
        self.assertEqual(classes_for(TOPICS_V1), ("engineering", "story", "teaching", "none"))
        if "STEERING_TOPICS" not in os.environ:
            self.assertEqual(CLASSES, classes_for(TOPICS_V1))
            self.assertEqual(rl_data.BEHAVIORS, TOPICS_V1)

    def test_v2_names_and_lists(self):
        self.assertEqual(len(TOPICS_V2), 8)
        self.assertEqual(resolve_topics("v2"), TOPICS_V2)
        self.assertEqual(resolve_topics("story, debugging"), ("story", "debugging"))
        for bad in ("nonsense", "story,story", ","):
            with self.assertRaises(ValueError):
                resolve_topics(bad)

    def test_output_folder_follows_the_topic_set(self):
        self.assertEqual(rl_data.default_out(TOPICS_V1).name, "rl")
        self.assertEqual(rl_data.default_out(TOPICS_V2).name, "rl_v2")


class TestRlDataFromCleanFiles(unittest.TestCase):
    def make_clean(self, folder, groups_by_split):
        for split, groups in groups_by_split.items():
            path = Path(folder) / "math_solving"
            path.mkdir(parents=True, exist_ok=True)
            rows = [{"id": f"{split}{g}", "group": g, "prompt": f"Question {g}?", "source": "x", "behavior": "math_solving"}
                    for g in groups]
            (path / f"{split}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    def test_v2_build_writes_train_dev_test_with_the_same_fields_as_v1(self):
        with tempfile.TemporaryDirectory() as clean, tempfile.TemporaryDirectory() as out:
            self.make_clean(clean, {"train": ["a", "b", "c"], "dev": ["d"], "test": ["e", "f"]})
            counts = rl_data.build(("math_solving",), Path(out), "clean", rl_data.PROMPTS, Path(clean))
            self.assertEqual(counts, {"math_solving": {"train": 3, "dev": 1, "test": 2}})
            first = rl_data.read_split("math_solving", "train", Path(out))[0]
            self.assertEqual(set(first), {"id", "group", "prompt"})

    def test_a_group_in_two_splits_is_refused(self):
        with tempfile.TemporaryDirectory() as clean:
            self.make_clean(clean, {"train": ["a"], "dev": ["a"], "test": ["e"]})
            with self.assertRaisesRegex(ValueError, "share a topic"):
                rl_data.clean_parts("math_solving", Path(clean))


class TestRouterV2(unittest.TestCase):
    def test_lookalikes_are_found(self):
        for text in ("Write an email to my boss.", "Fix this bug: the app crashes.", "What is 12 x 7?",
                     "Why is the sky blue?", "How to bake bread", "Explain gravity", "Compute the area of a circle."):
            self.assertTrue(rd.looks_like_topic(text), text)
        for text in ("Name three colors.", "Translate hello into French.", "Give two ways to conserve water"):
            self.assertFalse(rd.looks_like_topic(text), text)

    def test_old_topics_keep_their_template_wording_and_new_ones_stay_as_written(self):
        story = {"id": "i1", "prompt": "Write a short story about this idea: A cat flies."}
        self.assertTrue(rd.topic_text("story", "test", story).endswith("A cat flies."))
        self.assertNotEqual(rd.topic_text("story", "test", story), story["prompt"])
        email = {"id": "i2", "prompt": "Write an email.\n\nPurpose: Budget"}
        self.assertEqual(rd.topic_text("professional_email", "test", email), email["prompt"])
        arena = {"id": "i3", "prompt": "Build a quiz app for my class with scores."}
        self.assertEqual(rd.topic_text("engineering", "train", arena), arena["prompt"])

    def test_none_pool_drops_lookalikes_from_train_and_hard_set(self):
        texts = [f"Name {i} animals that live in the sea." for i in range(300)] + ["Write an email to the landlord."]
        texts += [f"List {i} facts about trees and forests." for i in range(300)] + ["How do vaccines work?"]
        tags = {t: (None, False) for t in texts}
        rows = rd.none_rows_v2(texts, tags)
        used = " ".join(r["text"] for r in rows)
        self.assertNotIn("email", used)
        self.assertNotIn("vaccines", used)
        self.assertTrue(any(r["split"] == "hard" for r in rows))
        self.assertTrue(all(r["label"] == "none" for r in rows))

    def test_one_text_with_two_labels_is_refused(self):
        rows = [rd.make_row("story", "train", "g1", "Same prompt here"), rd.make_row("none", "test", "g2", "Same prompt here")]
        with self.assertRaisesRegex(ValueError, "labeled"):
            rd.check_labels_disjoint(rows)

    def test_topic_rows_read_the_split_files_and_cap_each_split(self):
        with tempfile.TemporaryDirectory() as clean:
            folder = Path(clean) / "debugging"
            folder.mkdir()
            for name, count in (("train", 1300), ("dev", 50), ("test", 400)):
                rows = [{"id": f"{name}{i}", "group": f"{name} group {i}", "prompt": f"Debug item {name} {i}"}
                        for i in range(count)]
                (folder / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            rows = rd.topic_rows_v2("debugging", Path(clean))
        self.assertEqual(Counter(r["split"] for r in rows), {"train": 1200, "val": 50, "test": 300})
        rd.check_rows(rows, ("debugging",))


if __name__ == "__main__":
    unittest.main()
