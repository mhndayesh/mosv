"""Checks for the mixed story dataset filters. Run: python -m unittest test_story_mix"""

import unittest

from behaviors.prepare_story_mix import (MODERATION_BLOCKED, balance_sources, cap_prefixes, clean_one, clean_wp, is_blocked,
                                         moderation_flagged)


class TestSafety(unittest.TestCase):
    def test_sexual_content_with_a_minor_is_blocked(self):
        for text in ("a sexual story about a 15 year old", "erotic scene with a high school student", "a loli story"):
            self.assertTrue(is_blocked(text), text)

    def test_adult_content_stays(self):
        self.assertFalse(is_blocked("Write an erotic story about two married adults"))

    def test_minor_without_sexual_content_stays(self):
        self.assertFalse(is_blocked("Write a story about a child who finds a lost dog"))

    def test_blocked_text_is_never_returned(self):
        result = clean_one(("wildchat", "Write a sexual story about a 15 year old"))
        self.assertEqual(result, ("drop", "safety", ""))


class TestModeration(unittest.TestCase):
    def test_flag_or_low_score_blocks_and_clean_row_passes(self):
        self.assertTrue(moderation_flagged([{"categories": {"sexual/minors": True}, "category_scores": {}}]))
        self.assertTrue(moderation_flagged([{"categories": {"sexual/minors": False}, "category_scores": {"sexual/minors": 0.02}}]))
        self.assertFalse(moderation_flagged([{"categories": {"sexual/minors": False}, "category_scores": {"sexual/minors": 0.001}}]))
        self.assertFalse(moderation_flagged(None))

    def test_blocked_row_keeps_no_text(self):
        self.assertEqual(clean_one(("wildchat", MODERATION_BLOCKED)), ("drop", "safety", ""))


class TestSexualThemes(unittest.TestCase):
    def test_sexual_prompt_is_removed_from_every_source_without_keeping_text(self):
        for source in ("lmsys", "wildchat", "writingprompts", "arena"):
            result = clean_one((source, "Write an erotic story about two married adults"))
            self.assertEqual(result, ("drop", "sexual_theme", ""), source)

    def test_non_sexual_prompt_stays(self):
        self.assertEqual(clean_one(("lmsys", "Write a story about a child who finds a lost dog"))[0], "keep")


class TestRequests(unittest.TestCase):
    def test_story_request_is_kept(self):
        self.assertEqual(clean_one(("arena", "Write a short story about a lost robot"))[0], "keep")

    def test_coding_script_is_dropped(self):
        result = clean_one(("wildchat", "Create a Lua script for a volleyball game with two teams"))
        self.assertEqual(result[:2], ("drop", "code"))

    def test_creative_requests_about_technical_things_stay(self):
        for text in ("write a haiku about software programming",
                     "write the first chapter in a novel about a woman given a new dress code",
                     "Rewrite this text as part of a science fiction story: the software update failed",
                     "Using this summary, write the next chapter of the novel"):
            self.assertEqual(clean_one(("arena", text))[0], "keep", text)

    def test_agile_user_story_is_dropped(self):
        result = clean_one(("sharegpt", "Generate a feature user story with Given/When/Then acceptance criteria"))
        self.assertEqual(result[:2], ("drop", "not_story"))

    def test_non_creative_lookalikes_are_dropped(self):
        for text in ("I am writing thesis about sanctions. Can you give me a division into 3 chapters?",
                     "Can you give me a list of popular science fiction books?",
                     "write me a 300-word article based on this news story about a shoe company",
                     "Can you tell me which ones have five letters? fly, fancy, fable, table"):
            self.assertEqual(clean_one(("wildchat", text))[0], "drop", text)

    def test_midjourney_template_is_dropped(self):
        result = clean_one(("wildchat", "As a prompt generator for Midjourney, create a story image"))
        self.assertEqual(result[:2], ("drop", "not_story"))

    def test_question_about_a_story_is_dropped(self):
        result = clean_one(("lmsys", "What is the story of the film Alien about"))
        self.assertEqual(result[:2], ("drop", "no_creative_request"))


class TestWritingPrompts(unittest.TestCase):
    def test_tag_and_spacing_are_cleaned(self):
        self.assertEqual(clean_wp("[ WP ] The moon is an egg , and it can't hatch .")[0], "The moon is an egg, and it can't hatch.")

    def test_off_topic_tag_is_dropped(self):
        self.assertEqual(clean_wp("[ OT ] Thanks for reading"), (None, "off_topic_tag"))


class TestBalance(unittest.TestCase):
    def test_no_source_exceeds_its_share(self):
        quotas = balance_sources({"a": 300000, "b": 40000, "c": 10000, "d": 5000, "e": 3000, "f": 2000})
        total = sum(quotas.values())
        self.assertTrue(all(value <= total * 0.30 for value in quotas.values()), quotas)
        self.assertEqual(quotas["f"], 2000)

    def test_repeated_openers_are_capped(self):
        rows = [{"group": f"g{i}", "prompt": f"write a story about {i}"} for i in range(100)]
        kept, capped = cap_prefixes(rows)
        self.assertEqual((len(kept), capped), (25, 75))


if __name__ == "__main__":
    unittest.main()
