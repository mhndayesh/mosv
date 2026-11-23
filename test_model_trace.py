"""Run: python -m unittest test_model_trace.py"""

import unittest

import torch

from model_trace import match_vector, summarize

NAMES = ("engineering", "story")
VECTORS = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])


def step(vector, added=1.0, activation=10.0, match=1.0):
    return {"layers": {"7": {"activation": activation, "added": added, "vector": vector, "match": match}}}


class TestModelTrace(unittest.TestCase):
    def test_nothing_added_means_no_vector(self):
        self.assertEqual(match_vector(NAMES, VECTORS, torch.zeros(3)), ("none", 1.0))

    def test_added_push_matches_the_right_vector(self):
        self.assertEqual(match_vector(NAMES, VECTORS, torch.tensor([0.0, 0.5, 0.0])), ("story", 1.0))

    def test_summary_exposes_a_vector_switch_mid_answer(self):
        result = summarize([step("story"), step("engineering", match=0.9)])["7"]
        self.assertEqual(result["vectors"], ["engineering", "story"])
        self.assertEqual((result["push_percent"], result["lowest_match"]), (10.0, 0.9))

    def test_any_number_of_experts_and_any_layer_index(self):
        names = tuple(f"topic{i}" for i in range(8))
        vectors = torch.eye(8)
        self.assertEqual(match_vector(names, vectors, vectors[6] * 3), ("topic6", 1.0))
        row = {"layers": {"27": {"activation": 10.0, "added": 1.0, "vector": "topic6", "match": 1.0}}}
        self.assertEqual(list(summarize([row])), ["27"])

    def test_empty_answer_has_no_layers(self):
        self.assertEqual(summarize([]), {})


if __name__ == "__main__":
    unittest.main()
