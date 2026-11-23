"""Run: python -m unittest test_rl_pilot.py"""

import queue
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from rl.data import check_separate, split_rows
from rl.pilot import next_expert, parse_list
from rl.policy import (
    NormalizedMomentum, ShrinkableAdam, backward_policy_loss, build_policy_batch, kl_divergence, token_chunks,
    trim_completion,
)


class TinyCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(12, 6)
        self.vector = nn.Parameter(torch.randn(6))

    def forward(self, input_ids, attention_mask, use_cache=False):
        hidden = torch.tanh(self.embed(input_ids).cumsum(1) + self.vector)
        return SimpleNamespace(last_hidden_state=hidden)


def direct_gradient(core, head, batch, advantages, norm):
    core.zero_grad()
    hidden = core(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
    log_probs = torch.log_softmax(head(hidden[batch["rows"], batch["cols"]]), -1)
    picked = log_probs.gather(1, batch["targets"].unsqueeze(1)).squeeze(1)
    loss = -(advantages[batch["owners"]] / norm * picked).sum()
    loss.backward()
    return float(loss), core.vector.grad.clone()


class TestPolicy(unittest.TestCase):
    def test_trim_keeps_first_end_token(self):
        self.assertEqual(trim_completion([4, 5, 9, 7, 9], {9}), [4, 5, 9])
        self.assertEqual(trim_completion([4, 5], {9}), [4, 5])

    def test_chunks_keep_every_answer_once_under_budget(self):
        lengths = [300, 20, 280, 50, 310, 40, 900]
        chunks = token_chunks(list(range(len(lengths))), lengths, 620)
        self.assertEqual(sorted(i for chunk in chunks for i in chunk), list(range(len(lengths))))
        for chunk in chunks:
            self.assertTrue(len(chunk) == 1 or max(lengths[i] for i in chunk) * len(chunk) <= 620)

    def test_batch_scores_only_answer_tokens(self):
        batch = build_policy_batch(torch, [[1, 2], [3]], [[4, 5], [6, 7, 8]], 0, "cpu")
        self.assertEqual(batch["input_ids"].tolist(), [[1, 2, 4], [3, 6, 7]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1], [1, 1, 1]])
        self.assertEqual(batch["rows"].tolist(), [0, 0, 1, 1, 1])
        self.assertEqual(batch["cols"].tolist(), [1, 2, 0, 1, 2])
        self.assertEqual(batch["targets"].tolist(), [4, 5, 6, 7, 8])

    def test_chunked_gradient_matches_direct_gradient(self):
        torch.manual_seed(5)
        core, head = TinyCore(), nn.Linear(6, 12, bias=False)
        head.requires_grad_(False)
        batch = build_policy_batch(torch, [[1, 2, 3], [4]], [[5, 6, 7], [8, 9]], 0, "cpu")
        advantages = torch.tensor([0.5, -0.5])
        expected_loss, expected = direct_gradient(core, head, batch, advantages, 4.0)
        core.zero_grad()
        loss = backward_policy_loss(torch, core, head, batch, advantages, 4.0)
        self.assertAlmostEqual(loss, expected_loss, places=5)
        self.assertLess(float((core.vector.grad - expected).abs().max()), 1e-6)

    def test_positive_advantage_raises_answer_probability(self):
        torch.manual_seed(6)
        core, head = TinyCore(), nn.Linear(6, 12, bias=False)
        head.requires_grad_(False)
        batch = build_policy_batch(torch, [[1, 2]], [[5, 6]], 0, "cpu")

        def log_p():
            with torch.no_grad():
                hidden = core(batch["input_ids"], batch["attention_mask"]).last_hidden_state
                out = torch.log_softmax(head(hidden[batch["rows"], batch["cols"]]), -1)
                return float(out.gather(1, batch["targets"].unsqueeze(1)).sum())

        before = log_p()
        core.zero_grad()
        backward_policy_loss(torch, core, head, batch, torch.tensor([1.0]), 1.0)
        with torch.no_grad():
            core.vector -= 0.01 * core.vector.grad
        self.assertGreater(log_p(), before)


class TestOptimizers(unittest.TestCase):
    def make(self, momentum=0.9):
        vector = nn.Parameter(torch.zeros(2, 4))
        return vector, NormalizedMomentum(torch, vector, torch.tensor([1.0, 3.0]), momentum)

    def test_step_size_ignores_gradient_size_and_fits_each_layer(self):
        for size in (1e-6, 1e3):
            vector, optimizer = self.make(momentum=0.0)
            vector.grad = torch.tensor([[size, 0, 0, 0], [0, size, 0, 0]])
            optimizer.step()
            self.assertAlmostEqual(float(vector[0].norm()), 1.0, places=5)
            self.assertAlmostEqual(float(vector[1].norm()), 3.0, places=5)

    def test_step_moves_against_gradient(self):
        vector, optimizer = self.make(momentum=0.0)
        vector.grad = torch.ones(2, 4)
        optimizer.step()
        self.assertTrue((vector < 0).all())

    def test_disagreeing_noise_moves_less_than_agreement(self):
        same, first = self.make()
        noisy, second = self.make()
        for sign in (1, -1, 1, -1):
            same.grad = torch.ones(2, 4)
            first.step()
            noisy.grad = sign * torch.ones(2, 4)
            second.step()
        self.assertLess(float(noisy.norm()), float(same.norm()) * 0.5)

    def test_shrink_halves_later_steps(self):
        vector, optimizer = self.make(momentum=0.0)
        optimizer.shrink()
        vector.grad = torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]])
        optimizer.step()
        self.assertAlmostEqual(float(vector[0].norm()), 0.5, places=5)

    def test_adam_shrink_halves_learning_rate(self):
        optimizer = ShrinkableAdam(torch, nn.Parameter(torch.zeros(3)), 0.2)
        optimizer.shrink()
        self.assertAlmostEqual(optimizer.inner.param_groups[0]["lr"], 0.1)

    def test_kl_is_zero_for_same_and_positive_for_changed(self):
        before = torch.log_softmax(torch.tensor([[1.0, 2.0, 3.0]]), -1)
        after = torch.log_softmax(torch.tensor([[3.0, 2.0, 1.0]]), -1)
        self.assertAlmostEqual(kl_divergence(torch, before, before), 0.0, places=6)
        self.assertGreater(kl_divergence(torch, before, after), 0.1)


class TestPilotData(unittest.TestCase):
    def test_split_maps_validation_to_dev(self):
        rows = [{"id": "a", "group": "x", "text": "t1", "label": "story", "split": "train"},
                {"id": "b", "group": "y", "text": "t2", "label": "story", "split": "val"},
                {"id": "c", "group": "z", "text": "t3", "label": "story", "split": "test"},
                {"id": "d", "group": "q", "text": "t4", "label": "none", "split": "train"}]
        parts = split_rows(rows, "story")
        self.assertEqual({k: len(v) for k, v in parts.items()}, {"train": 1, "dev": 1, "test": 1})

    def test_shared_topic_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "share a topic"):
            check_separate({"train": [{"group": "x"}], "test": [{"group": "x"}]}, "story")

    def test_lists_reject_unknown_and_repeated_names(self):
        self.assertEqual(parse_list("main,second", ("main", "second")), ["main", "second"])
        for bad in ("third", "main,main", ""):
            with self.assertRaises(ValueError):
                parse_list(bad, ("main", "second"))

    def test_free_card_takes_each_behavior_once(self):
        class Jobs:
            def __init__(self):
                self.items = ["engineering", "story", "teaching"]

            def get(self, timeout):
                if not self.items:
                    raise queue.Empty
                return self.items.pop(0)

        jobs, taken = Jobs(), []
        while (job := next_expert(jobs)) is not None:
            taken.append(job)
        self.assertEqual(taken, ["engineering", "story", "teaching"])


if __name__ == "__main__":
    unittest.main()
