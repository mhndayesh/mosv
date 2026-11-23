"""Run: python -m unittest test_rl_pilot.py"""

import os
import queue
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from rl.data import check_separate, split_rows
from rl import pilot
from rl.pilot import next_expert, parse_list
from rl.policy import (
    NormalizedMomentum, ShrinkableAdam, backward_policy_loss, build_policy_batch, kl_divergence, token_chunks,
    trim_completion,
)
from rl.vllm_sampler import MAX_MODEL_LEN, VllmSampler
from rl_math import group_advantages


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

EOS = 99


class FakeTokenizer:
    """A prompt of n letters becomes the n tokens 1..n."""

    def apply_chat_template(self, messages, **_):
        return messages[0]["content"]

    def __call__(self, texts, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [list(range(1, len(text) + 1)) for text in texts]}

    def decode(self, tokens, skip_special_tokens):
        return " ".join(str(t) for t in tokens)


class FakeLLM:
    """Answer j of a prompt is [first token, j] and then an end token followed by junk."""

    def __init__(self, events, tail=(EOS, 7, 8)):
        self.events, self.tail, self.pushed, self.prompts = events, tail, [], []

    def apply_model(self, func):
        self.events.append("push")
        self.pushed.append(func)

    def reset_prefix_cache(self):
        self.events.append("reset")

    def generate(self, prompts, params, use_tqdm):
        self.events.append("generate")
        self.prompts += [p["prompt_token_ids"] for p in prompts]
        outputs = [SimpleNamespace(index=j, token_ids=[p["prompt_token_ids"][0], j, *self.tail])
                   for p in prompts for j in range(params.group)]
        size = params.group
        # Engines may list a prompt's answers in any order; the sampler must sort them by index.
        return [SimpleNamespace(outputs=outputs[i * size:(i + 1) * size][::-1]) for i in range(len(prompts))]


def fake_model():
    layers = [SimpleNamespace(steering_delta=torch.zeros(4, dtype=torch.float16)) for _ in range(22)]
    return SimpleNamespace(model=SimpleNamespace(layers=layers))


def vllm_run(group=3, tail=(EOS, 7, 8), max_new=20):
    events = []
    llm = FakeLLM(events, tail)
    sampler = VllmSampler(llm, SimpleNamespace(group=group), group, max_new, {EOS})
    run = SimpleNamespace(tokenizer=FakeTokenizer(), args=SimpleNamespace(group=group), sampler=sampler)
    return run, llm, events


def rollouts(run, rows, vectors=None):
    bank = SimpleNamespace(vectors=[torch.ones(3, 4) if vectors is None else vectors])
    with mock.patch.object(pilot, "score_response", lambda expert, text, prompt: SimpleNamespace(score=len(text))):
        return pilot.sample_rollouts(run, bank, rows, "story")


class TestVllmRollouts(unittest.TestCase):
    ROWS = [{"prompt": "ab"}, {"prompt": "abcd"}]

    def test_one_aligned_entry_per_answer_in_prompt_order(self):
        run, llm, _ = vllm_run(group=3)
        prompts, completions, rewards, advantages = rollouts(run, self.ROWS)
        self.assertEqual(llm.prompts, [[1, 2], [1, 2, 3, 4]])
        self.assertEqual([len(x) for x in (prompts, completions, rewards, advantages)], [6] * 4)
        self.assertEqual(prompts, [[1, 2]] * 3 + [[1, 2, 3, 4]] * 3)
        self.assertEqual(completions, [[1, j, EOS] for j in range(3)] * 2)
        self.assertEqual(rewards, [len(" ".join(map(str, c))) for c in completions])
        for start in (0, 3):
            self.assertEqual(advantages[start:start + 3], group_advantages(rewards[start:start + 3]))

    def test_group_size_sets_answers_per_prompt(self):
        for group in (2, 4, 5):
            run, _, _ = vllm_run(group=group)
            prompts, completions, _, _ = rollouts(run, self.ROWS)
            self.assertEqual((len(prompts), len(completions)), (2 * group, 2 * group))

    def test_answers_stop_at_first_end_token_and_unfinished_answers_stay_whole(self):
        run, _, _ = vllm_run(group=2)
        _, completions, _, _ = rollouts(run, self.ROWS)
        self.assertTrue(all(c[-1] == EOS and c.count(EOS) == 1 for c in completions))
        run, _, _ = vllm_run(group=2, tail=(5, 6))
        _, completions, _, _ = rollouts(run, self.ROWS)
        self.assertEqual(completions[0], [1, 0, 5, 6])

    def test_vector_is_pushed_before_every_sampling_call(self):
        run, llm, events = vllm_run()
        rollouts(run, self.ROWS, torch.full((3, 4), 0.5))
        rollouts(run, self.ROWS, torch.full((3, 4), 2.0))
        self.assertEqual(events, ["push", "reset", "generate", "push", "reset", "generate"])
        model = fake_model()
        llm.pushed[1](model)
        for index in (7, 14, 21):
            self.assertEqual(model.model.layers[index].steering_delta.tolist(), [2.0] * 4)
        self.assertEqual(model.model.layers[0].steering_delta.tolist(), [0.0] * 4)

    def test_scoring_uses_the_chosen_vector_and_clears_old_prompt_work(self):
        run, llm, events = vllm_run(group=1)
        run.sampler.greedy = SimpleNamespace(group=1)
        bank = SimpleNamespace(vectors=[torch.full((3, 4), 2.0)])
        answers = pilot.eval_answers(run, bank, 0, self.ROWS)
        self.assertEqual(answers, [[1, 0, EOS], [1, 0, EOS]])
        self.assertEqual(events, ["push", "reset", "generate"])
        pilot.eval_answers(run, bank, None, self.ROWS)
        model = fake_model()
        llm.pushed[0](model)
        self.assertEqual(model.model.layers[7].steering_delta.tolist(), [2.0] * 4)
        llm.pushed[1](model)
        self.assertEqual(model.model.layers[7].steering_delta.tolist(), [0.0] * 4)

    def test_prompt_too_long_for_the_engine_is_refused(self):
        run, _, _ = vllm_run(max_new=100)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            rollouts(run, [{"prompt": "a" * (MAX_MODEL_LEN - 99)}])


class TestCardCpus(unittest.TestCase):
    def test_engine_and_learner_split_each_cards_cpus(self):
        for name, spec in pilot.PIPELINES.items():
            engine, learner = set(pilot.ENGINE_CPUS[name]), set(pilot.learner_cpus(name))
            self.assertFalse(engine & learner)
            self.assertEqual(engine | learner, set(spec["build"] + spec["gpu"]))

    def test_cards_share_no_cpus(self):
        main = set(pilot.ENGINE_CPUS["main"] + pilot.learner_cpus("main"))
        second = set(pilot.ENGINE_CPUS["second"] + pilot.learner_cpus("second"))
        self.assertFalse(main & second)

    def test_pin_moves_every_thread_and_skips_ended_ones(self):
        calls = []

        def pin(thread, cpus):
            calls.append((thread, cpus))
            if thread == 12:
                raise ProcessLookupError

        with mock.patch.object(pilot.os, "listdir", return_value=["11", "12", "13"]), \
                mock.patch.object(pilot.os, "sched_setaffinity", pin, create=True):
            pilot.pin_process([4, 5])
        self.assertEqual(calls, [(11, [4, 5]), (12, [4, 5]), (13, [4, 5])])

    def test_close_stops_engine_children_then_exits(self):
        stuck, clean = mock.Mock(), mock.Mock()
        stuck.is_alive.return_value, clean.is_alive.return_value = True, False
        with mock.patch.object(pilot.mp, "active_children", return_value=[stuck, clean]), \
                mock.patch.object(pilot.os, "_exit") as leave:
            pilot.close_worker()
        stuck.terminate.assert_called_once()
        clean.terminate.assert_called_once()
        stuck.kill.assert_called_once()
        clean.kill.assert_not_called()
        leave.assert_called_once_with(0)

    def test_vllm_starts_on_engine_cpus_then_learner_moves(self):
        order, fake_torch = [], SimpleNamespace(set_num_threads=lambda n: order.append(("threads", n)))
        args = SimpleNamespace(group=4, max_new=256, kv_cache_dtype="auto")
        with mock.patch.object(pilot, "pin_process", lambda cpus: order.append(("pin", cpus))), \
                mock.patch.object(pilot.VllmSampler, "start", lambda *a: order.append(("start",)) or "engine"):
            self.assertEqual(pilot.start_sampler("main", args, {EOS}, fake_torch), "engine")
        learner = pilot.learner_cpus("main")
        self.assertEqual(order, [("pin", pilot.ENGINE_CPUS["main"]), ("start",), ("pin", learner),
                                 ("threads", len(learner))])


class TestCapAndBestVector(unittest.TestCase):
    def test_huge_first_token_inflates_average_but_not_typical_size(self):
        tensor = torch.tensor([[[8200.0, 0.0], [25.0, 0.0], [26.0, 0.0], [27.0, 0.0], [28.0, 0.0]]])
        average, typical = pilot.layer_sizes(tensor, torch.ones(1, 5))
        self.assertGreater(average, 1000)
        self.assertEqual(typical, 27.0)  # The middle of 25, 26, 27, 28, 8200.

    def test_padding_is_not_measured(self):
        tensor = torch.tensor([[[9999.0, 0.0], [4.0, 0.0]]])
        self.assertEqual(pilot.layer_sizes(tensor, torch.tensor([[0, 1]])), (4.0, 4.0))

    def test_default_cap_is_half_a_typical_token(self):
        self.assertEqual(pilot.make_parser().parse_args([]).cap_fraction, 0.50)

    def test_finished_file_holds_the_best_vector_not_the_last(self):
        bank = SimpleNamespace(vectors=[torch.nn.Parameter(torch.full((3, 4), 9.0))])
        optimizer = SimpleNamespace(state_dict=lambda: {})
        saved = {}
        with mock.patch.object(pilot, "save_atomic", lambda _t, state, _p: saved.update(state)):
            pilot.save_best(SimpleNamespace(torch=torch), bank, torch.full((3, 4), 2.0), optimizer, 86, 40, "story", "x")
        self.assertTrue((saved["bank"]["vectors.0"] == 2.0).all())
        self.assertEqual((saved["step"], saved["best_step"]), (86, 40))


class TestSamplerChoice(unittest.TestCase):
    def test_hugging_face_is_the_default(self):
        args = pilot.make_parser().parse_args([])
        self.assertEqual((args.sampler, args.kv_cache_dtype), ("hf", "auto"))
        self.assertEqual(pilot.make_parser().parse_args(["--sampler", "vllm"]).sampler, "vllm")

    def test_no_sampler_means_hugging_face_generation(self):
        with mock.patch.object(pilot, "sample_rollouts_hf", return_value="hf") as hf, \
                mock.patch.object(pilot, "sample_rollouts_vllm") as vllm:
            self.assertEqual(pilot.sample_rollouts(SimpleNamespace(sampler=None), None, [], "story"), "hf")
        hf.assert_called_once()
        vllm.assert_not_called()

    def test_engine_gets_the_requested_settings(self):
        seen = {}
        fake = SimpleNamespace(LLM=lambda **kw: seen.setdefault("llm", kw) and "engine",
                               SamplingParams=lambda **kw: seen.setdefault("params", kw))
        with mock.patch.dict(sys.modules, {"vllm": fake}), mock.patch.dict(os.environ), \
                mock.patch.object(sys, "path", list(sys.path)):
            sampler = VllmSampler.start("model-path", 4, 256, {EOS, 5}, kv_cache_dtype="fp8")
        llm, params = seen["llm"], seen["params"]
        self.assertEqual((llm["dtype"], llm["kv_cache_dtype"], llm["max_num_seqs"]), ("float16", "fp8", 256))
        self.assertEqual((llm["max_num_batched_tokens"], llm["max_model_len"]), (8192, MAX_MODEL_LEN))
        self.assertEqual(llm["hf_overrides"], {"architectures": ["SteeredQwen3ForCausalLM"]})
        self.assertTrue(llm["disable_log_stats"] and 0 < llm["gpu_memory_utilization"] < 0.5)
        self.assertEqual({k: params[k] for k in ("n", "temperature", "top_p", "top_k", "repetition_penalty", "max_tokens")},
                         {"n": 4, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0,
                          "max_tokens": 256})
        self.assertEqual(params["stop_token_ids"], [5, EOS])
        self.assertEqual(sampler.group, 4)


if __name__ == "__main__":
    unittest.main()
