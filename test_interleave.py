"""Checks for shuffled multi-topic training. These use no GPU and no API."""

import unittest
from collections import Counter
from concurrent.futures import Future
from itertools import groupby
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from rl import interleave, pilot


def topic(name, value):
    return interleave.TopicState(name, [{"prompt": f"{name}{i}"} for i in range(5)], [], [], 3, torch.full((3, 4), value), mock.Mock(),
                                 torch.ones(3), Path("x.pt"))


class TestSchedule(unittest.TestCase):
    COUNTS = {"story": 300, "engineering": 40, "teaching": 12, "math_solving": 150}

    def test_every_topic_gets_its_exact_number_of_updates(self):
        self.assertEqual(Counter(interleave.mixed_schedule(self.COUNTS, 7)), Counter(self.COUNTS))

    def test_order_is_repeatable_and_mixed(self):
        order = interleave.mixed_schedule(self.COUNTS, 7)
        self.assertEqual(order, interleave.mixed_schedule(self.COUNTS, 7))
        self.assertNotEqual(order, interleave.mixed_schedule(self.COUNTS, 8))
        self.assertEqual(set(order[:60]), set(self.COUNTS))

    def test_same_topic_does_not_repeat_while_another_is_available(self):
        order = interleave.mixed_schedule({"a": 10, "b": 10, "c": 10}, 3)
        self.assertLessEqual(sum(x == y for x, y in zip(order, order[1:])), 1)

    def test_resume_runs_exactly_the_updates_left(self):
        order = interleave.resume_schedule({"a": 10, "b": 6, "c": 3}, {"a": 4, "b": 6, "c": 1}, seed=7)
        self.assertEqual(Counter(order), Counter({"a": 6, "c": 2}))

    def test_resume_with_a_new_topic_set_keeps_rotating(self):
        """The real 2026-11-25 case: resuming 3 of 8 topics gave 17 teaching updates in a row and no debugging."""
        totals = {"engineering": 342, "debugging": 1069, "teaching": 91}
        done = {"engineering": 160, "debugging": 497, "teaching": 39}
        order = interleave.resume_schedule(totals, done, seed=0)
        self.assertEqual(Counter(order), Counter({t: totals[t] - done[t] for t in totals}))
        self.assertIn("debugging", order[:10])
        self.assertLessEqual(max(len(list(run)) for _, run in groupby(order)), 6)  # Debugging is 79% of what is left.
        for part in (order[:100], order[-100:]):  # Every topic is still rotating at the start and at the end.
            self.assertTrue(all(Counter(part)[t] > 0 for t in totals))


class TestTopicIsolation(unittest.TestCase):
    def test_step_changes_only_the_current_topic(self):
        bank = SimpleNamespace(vectors=[torch.nn.Parameter(torch.zeros(3, 4))])
        run = SimpleNamespace(torch=SimpleNamespace(no_grad=torch.no_grad, cuda=SimpleNamespace(reset_peak_memory_stats=lambda: None)),
                              turns=SimpleNamespace(swap_seconds=0.0), judge_update=0)
        story, teaching = topic("story", 1.0), topic("teaching", 7.0)
        done = Future()
        done.set_result(([], 1.5))
        job = pilot.WriteJob(story.train[:1], [[1]], story.vector.cpu(), done)

        def fake_learn(_run, active_bank, optimizer, *_):
            self.assertTrue((active_bank.vectors[0] == 1.0).all())
            self.assertIs(optimizer, story.optimizer)
            with torch.no_grad():
                active_bank.vectors[0].add_(0.5)
            return {"reward": 0.5}

        with mock.patch.object(pilot, "written_rollouts", return_value=()), mock.patch.object(pilot, "learn_step", fake_learn):
            interleave.mixed_step(run, bank, story, job, lambda: None)
        self.assertTrue((story.vector == 1.5).all())
        self.assertTrue((teaching.vector == 7.0).all())

    def test_next_batch_starts_before_learning(self):
        events = []
        bank = SimpleNamespace(vectors=[torch.nn.Parameter(torch.zeros(3, 4))])
        run = SimpleNamespace(torch=SimpleNamespace(no_grad=torch.no_grad, cuda=SimpleNamespace(reset_peak_memory_stats=lambda: None)),
                              turns=SimpleNamespace(swap_seconds=0.0), judge_update=0)
        story = topic("story", 1.0)
        done = Future()
        done.set_result(([], 1.0))
        job = pilot.WriteJob(story.train[:1], [[1]], story.vector.cpu(), done)
        with mock.patch.object(pilot, "written_rollouts", return_value=()), \
                mock.patch.object(pilot, "learn_step", lambda *_: events.append("learn") or {}):
            interleave.mixed_step(run, bank, story, job, lambda: events.append("next"))
        self.assertEqual(events, ["next", "learn"])

    def test_rows_wrap_around_the_topic_data(self):
        state = topic("story", 0.0)
        self.assertEqual([r["prompt"] for r in interleave.rows_for(state, 2, 2)], ["story4", "story0"])


class TestSnapshots(unittest.TestCase):
    def test_dev_check_keeps_a_numbered_copy(self):
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            state = topic("story", 0.0)
            state.path, state.step = Path(folder) / "story.pt", 50
            state.path.write_bytes(b"vector")
            interleave.keep_snapshot(state)
            self.assertEqual((Path(folder) / "snapshots" / "story_step00050.pt").read_bytes(), b"vector")


class TestFlags(unittest.TestCase):
    def test_interleave_needs_overlap_and_haiku(self):
        args = SimpleNamespace(vllm_swap="separate", sampler="vllm", cards="main", overlap=False, interleave=True, judge="haiku", group=4)
        with self.assertRaisesRegex(ValueError, "--interleave"):
            pilot.check_separate_cards(args)


if __name__ == "__main__":
    unittest.main()
