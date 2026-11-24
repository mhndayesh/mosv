"""Run: python -m unittest test_rl_pilot.py"""

import os
import queue
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from rl.data import check_separate, length_batches, split_rows
from rl import pilot
from rl.pilot import next_expert, parse_list
from model_profiles import DEFAULT_PROFILE, PROFILES, get_profile, model_width, profile_for_model
from rl.policy import (
    NormalizedMomentum, ShrinkableAdam, backward_policy_loss, build_policy_batch, checkpointed, compiled_forward,
    kl_divergence, prepare_layers, token_chunks, trim_completion, without_gradients,
)
from rl.vllm_sampler import MAX_MODEL_LEN, CardTurns, NoTurns, VllmSampler, WeightShuttle, engine_settings
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
    return SimpleNamespace(model=SimpleNamespace(layers=layers), config=SimpleNamespace(steering_layers=[7, 14, 21]))


def vllm_run(group=3, tail=(EOS, 7, 8), max_new=20):
    events = []
    llm = FakeLLM(events, tail)
    sampler = VllmSampler(llm, SimpleNamespace(group=group), group, max_new, {EOS})
    run = SimpleNamespace(tokenizer=FakeTokenizer(), args=SimpleNamespace(group=group), sampler=sampler,
                          turns=NoTurns())
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


class TestCardMemoryBudgets(unittest.TestCase):
    def test_4b_stays_under_the_stop_on_each_card(self):
        profile = PROFILES["qwen4b"]
        for card in ("main", "second"):
            self.assertLess(profile.torch_vram_gb[card] + profile.other_vram_gb[card], profile.hard_stop_gb)
        self.assertLess(profile.cells("main"), profile.cells("second"))

    def test_06b_keeps_its_old_budget(self):
        profile = PROFILES["qwen06b"]
        self.assertEqual((profile.cells("main"), profile.eval_batch, dict(profile.torch_vram_gb)), (22000, 32, {}))

    def test_limit_is_a_share_of_the_real_card(self):
        cuda = mock.Mock()
        cuda.get_device_properties.return_value = SimpleNamespace(total_memory=16e9)
        pilot.limit_torch_vram(SimpleNamespace(cuda=cuda), PROFILES["qwen4b"], "main")
        cuda.set_per_process_memory_fraction.assert_called_once_with(11.0 / 16, 0)

    def test_free_gpu_collects_cycles_before_emptying_the_cache(self):
        order = []
        fake = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: order.append("empty")))
        with mock.patch.object(pilot.gc, "collect", lambda: order.append("collect")):
            pilot.free_gpu(fake)
        self.assertEqual(order, ["collect", "empty"])

    def test_no_budget_sets_no_limit(self):
        cuda = mock.Mock()
        pilot.limit_torch_vram(SimpleNamespace(cuda=cuda), PROFILES["qwen06b"], "main")
        cuda.set_per_process_memory_fraction.assert_not_called()


class TestFittingRows(unittest.TestCase):
    def test_prompts_without_room_for_a_full_answer_are_dropped(self):
        run = SimpleNamespace(tokenizer=FakeTokenizer(), args=SimpleNamespace(max_new=256))
        room = MAX_MODEL_LEN - 256
        rows = [{"prompt": "a" * room}, {"prompt": "a" * (room + 1)}, {"prompt": "short"}]
        self.assertEqual(pilot.fitting_rows(run, rows), [rows[0], rows[2]])


class TestSeparateCards(unittest.TestCase):
    def test_engine_starts_on_main_card_and_learner_keeps_second(self):
        seen, pins = {}, []

        def fake_start(*_args, around_build, card, swap, **_kwargs):
            seen.update(card=card, swap=swap)
            around_build(lambda: seen.update(engine_card=os.environ["HIP_VISIBLE_DEVICES"]))
            return "engine"

        cuda = SimpleNamespace(get_device_properties=lambda _i: SimpleNamespace(total_memory=16e9))
        fake_torch = SimpleNamespace(cuda=cuda, set_num_threads=lambda _n: None)
        args = SimpleNamespace(group=4, max_new=256, kv_cache_dtype="auto", vllm_swap="separate")
        with mock.patch.dict(os.environ, {"HIP_VISIBLE_DEVICES": "0", "CUDA_VISIBLE_DEVICES": "0"}), \
                mock.patch.object(pilot, "pin_process", pins.append), \
                mock.patch.object(pilot.VllmSampler, "start", fake_start):
            self.assertEqual(pilot.start_sampler("second", args, {EOS}, fake_torch, PROFILES["qwen4b"]), "engine")
            self.assertEqual(os.environ["HIP_VISIBLE_DEVICES"], "0")
        self.assertEqual(seen, {"card": "main", "swap": "separate", "engine_card": "1"})
        self.assertEqual(pins[0], pilot.ENGINE_CPUS["main"])
        self.assertEqual(pins[1], sorted(pilot.PIPELINES["second"]["build"] + pilot.PIPELINES["second"]["gpu"]))
        self.assertFalse(set(pins[0]) & set(pins[1]))

    def test_separate_mode_never_enables_sleep(self):
        settings = engine_settings(PROFILES["qwen4b"], "main", int(16e9), "separate")
        self.assertNotIn("enable_sleep_mode", settings)

    def test_separate_mode_needs_vllm_on_the_second_card(self):
        good = SimpleNamespace(vllm_swap="separate", sampler="vllm", cards="second")
        pilot.check_separate_cards(good)
        for bad in (SimpleNamespace(vllm_swap="separate", sampler="hf", cards="second"),
                    SimpleNamespace(vllm_swap="separate", sampler="vllm", cards="main,second")):
            with self.assertRaises(ValueError):
                pilot.check_separate_cards(bad)


class TestRouteBatches(unittest.TestCase):
    def test_every_prompt_is_routed_once_in_similar_lengths(self):
        texts = ["a" * n for n in (9, 1, 5, 3, 7)]
        batches = length_batches(texts, 2)
        self.assertEqual(batches, [[1, 3], [2, 4], [0]])
        self.assertEqual(sorted(i for part in batches for i in part), list(range(5)))


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
        args = SimpleNamespace(group=4, max_new=256, kv_cache_dtype="auto", vllm_swap="sleep")

        def start(*_args, around_build, **_kwargs):
            return around_build(lambda: order.append(("start",)) or "engine")

        with mock.patch.object(pilot, "pin_process", lambda cpus: order.append(("pin", cpus))), \
                mock.patch.object(pilot.VllmSampler, "start", start):
            self.assertEqual(pilot.start_sampler("main", args, {EOS}, fake_torch, PROFILES["qwen06b"]), "engine")
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
            pilot.save_best(SimpleNamespace(torch=torch, profile=PROFILES["qwen4b"]), bank, torch.full((3, 4), 2.0),
                            optimizer, 86, 40, "story", "x")
        self.assertTrue((saved["bank"]["vectors.0"] == 2.0).all())
        self.assertEqual((saved["layers"], saved["profile"]), ([9, 18, 27], "qwen4b"))
        self.assertEqual((saved["step"], saved["best_step"]), (86, 40))

    def dev_setup(self):
        bank = SimpleNamespace(vectors=[torch.full((3, 4), 9.0)])
        optimizer = mock.Mock()
        run = SimpleNamespace(card="second", torch=torch)
        return run, bank, optimizer, (torch.full((3, 4), 2.0), 0.5, 10)

    def test_dev_score_saves_the_vector_that_was_scored(self):
        run, bank, optimizer, best = self.dev_setup()
        scored = torch.full((3, 4), 4.0)
        result = {"score": 0.8, "degenerate": 0.0, "on_topic": 1.0}
        with mock.patch.object(pilot, "say"):
            vector, score, step = pilot.record_dev(run, bank, optimizer, [], result, scored, 15, {"score": 0.1, "degenerate": 0.0},
                                                   "story", best)
        self.assertTrue((vector == 4.0).all())
        self.assertEqual((score, step), (0.8, 15))
        self.assertTrue((bank.vectors[0] == 9.0).all())

    def test_broken_answers_roll_back_to_the_best_vector(self):
        run, bank, optimizer, best = self.dev_setup()
        result = {"score": 0.9, "degenerate": 1.0, "on_topic": 1.0}
        with mock.patch.object(pilot, "say"):
            kept = pilot.record_dev(run, bank, optimizer, [], result, torch.zeros(3, 4), 15, {"score": 0.1, "degenerate": 0.0},
                                    "story", best)
        self.assertIs(kept, best)
        self.assertTrue((bank.vectors[0] == 2.0).all())
        optimizer.shrink.assert_called_once()

    def test_one_pass_covers_every_training_prompt(self):
        self.assertEqual(pilot.one_pass_updates(32, 16), 2)
        self.assertEqual(pilot.one_pass_updates(33, 16), 3)

    def test_judged_rollouts_leave_out_prompts_without_a_verdict(self):
        judge = mock.Mock()
        judge.judge.return_value = ([[1.0, 0.0], None], SimpleNamespace(summary=lambda: {"seconds": 1.0}))
        run = SimpleNamespace(judge=judge, judge_update=4, last_judge=None,
                              tokenizer=SimpleNamespace(decode=lambda tokens, skip_special_tokens: str(tokens)))
        out = pilot.judged_rollouts(run, [{"prompt": "a"}, {"prompt": "b"}], [[1], [2]], [[[5], [6]], [[7], [8]]], "story")
        self.assertEqual(out[0], [[1], [1]])
        self.assertEqual(out[1], [[5], [6]])
        self.assertEqual(out[3], [0.5, -0.5])

    def judged_run(self, rewards):
        judge = mock.Mock()
        judge.judge.return_value = (rewards, SimpleNamespace(retried=0, skipped=0, cost_usd=0.0, seconds=0.0))
        return SimpleNamespace(judge=judge, judge_update=1, card="second",
                               tokenizer=SimpleNamespace(decode=lambda tokens, skip_special_tokens: str(tokens)))

    def test_final_comparison_reports_each_answer_set(self):
        run = self.judged_run([[0.2, 0.8, 0.5], [0.4, 0.4, 0.1]])
        sets = {"base": [[1], [2]], "steered": [[3], [4]], "random": [[5], [6]]}
        with mock.patch.object(pilot, "say"):
            out = pilot.compare_with_judge(run, [{"prompt": "a"}, {"prompt": "b"}], sets, "story", "final_test")
        self.assertEqual(out["steered"], {"reward": 0.6, "beats_base_pct": 75.0})
        self.assertEqual(out["random"]["beats_base_pct"], 50.0)
        self.assertEqual(run.judge.judge.call_args.args[0][1].answers, ("[2]", "[4]", "[6]"))

    def test_too_few_judged_prompts_give_no_comparison(self):
        run = self.judged_run([[0.2, 0.8], None, None])
        sets = {"base": [[1], [2], [3]], "steered": [[4], [5], [6]]}
        with mock.patch.object(pilot, "say"):
            self.assertIsNone(pilot.compare_with_judge(run, [{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}], sets, "story", "dev"))

    def test_resume_keeps_the_best_update_of_a_finished_file(self):
        self.assertEqual(pilot.finished_best_step({"step": 86, "best_step": 65}, 86), 65)

    def test_resume_of_an_unfinished_file_has_no_saved_best(self):
        self.assertIsNone(pilot.finished_best_step({"step": 40, "best_step": 30}, 86))
        self.assertIsNone(pilot.finished_best_step({"step": 86}, 86))  # Stopped before the best was saved.


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
        self.assertEqual(llm["hf_overrides"], {"architectures": ["SteeredQwen3ForCausalLM"],
                                               "steering_layers": [7, 14, 21]})
        self.assertNotIn("enable_sleep_mode", llm)
        self.assertTrue(llm["disable_log_stats"] and 0 < llm["gpu_memory_utilization"] < 0.5)
        self.assertEqual({k: params[k] for k in ("n", "temperature", "top_p", "top_k", "repetition_penalty", "max_tokens")},
                         {"n": 4, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0,
                          "max_tokens": 256})
        self.assertEqual(params["stop_token_ids"], [5, EOS])
        self.assertEqual(sampler.group, 4)


class FakeCuda:
    def __init__(self, events, allocated=0):
        self.events, self.allocated = events, allocated

    def synchronize(self):
        self.events.append("sync")

    def empty_cache(self):
        self.events.append("empty")

    def memory_allocated(self):
        return self.allocated


class FakeSide:
    """Records moves and fails if both sides would hold the card."""

    def __init__(self, events, card, name):
        self.events, self.card, self.name, self.on_gpu = events, card, name, False
        self.bytes = 8_000_000_000

    def enter(self, label):
        if self.card["holder"] is not None:
            raise AssertionError(f"{self.name} loaded while {self.card['holder']} holds the card")
        self.card["holder"] = self.name
        self.events.append(label)

    def leave(self, label):
        self.card["holder"] = None
        self.events.append(label)


class FakeEngine(FakeSide):
    def rest(self):
        self.leave("vllm_off")

    def wake(self):
        self.enter("vllm_on")


class FakeWeights(FakeSide):
    def to_gpu(self):
        self.enter("torch_on")
        self.on_gpu = True

    def to_cpu(self):
        self.leave("torch_off")
        self.on_gpu = False


def make_turns(allocated=0):
    events, card = [], {"holder": "vllm"}
    engine, weights = FakeEngine(events, card, "vllm"), FakeWeights(events, card, "torch")
    return CardTurns(SimpleNamespace(cuda=FakeCuda(events, allocated)), engine, weights), events, card, weights


class TestCardTurns(unittest.TestCase):
    def test_vllm_leaves_before_torch_loads(self):
        turns, events, card, _ = make_turns()
        turns.for_torch()
        self.assertEqual(events, ["vllm_off", "torch_on"])
        self.assertEqual((turns.owner, card["holder"]), ("torch", "torch"))

    def test_torch_leaves_and_cache_is_freed_before_vllm_wakes(self):
        turns, events, card, _ = make_turns()
        turns.for_torch()
        events.clear()
        turns.for_vllm()
        self.assertEqual(events, ["torch_off", "sync", "empty", "vllm_on"])
        self.assertEqual((turns.owner, card["holder"]), ("vllm", "vllm"))

    def test_repeat_requests_do_not_swap(self):
        turns, events, _, _ = make_turns()
        turns.for_vllm()
        self.assertEqual(events, [])
        turns.for_torch()
        turns.for_torch()
        self.assertEqual((events.count("torch_on"), turns.swaps), (1, 1))

    def test_many_cycles_never_overlap(self):
        turns, events, _, _ = make_turns()
        for _ in range(5):
            turns.for_torch()
            turns.for_vllm()
        self.assertEqual((events.count("vllm_off"), events.count("vllm_on"), turns.swaps), (5, 5, 10))

    def test_vllm_stays_asleep_if_torch_memory_was_not_freed(self):
        turns, events, card, _ = make_turns(allocated=5_000_000_000)
        turns.for_torch()
        with self.assertRaisesRegex(RuntimeError, "still holds"):
            turns.for_vllm()
        self.assertNotIn("vllm_on", events)
        self.assertIsNone(card["holder"])

    def test_failed_upload_does_not_claim_vllm_holds_the_card(self):
        turns, _, _, weights = make_turns()
        weights.to_gpu = mock.Mock(side_effect=RuntimeError("out of memory"))
        with self.assertRaises(RuntimeError):
            turns.for_torch()
        self.assertEqual(turns.owner, "moving")

    def test_weights_on_the_card_at_start_are_refused(self):
        _, events, card, _ = make_turns()
        weights = FakeWeights(events, card, "torch")
        weights.on_gpu = True
        with self.assertRaisesRegex(RuntimeError, "off the card"):
            CardTurns(SimpleNamespace(cuda=FakeCuda(events)), FakeEngine(events, card, "vllm"), weights)


class TestEngineSleep(unittest.TestCase):
    def sampler(self, swap):
        llm = mock.Mock()
        rebuilt = mock.Mock()
        return VllmSampler(llm, SimpleNamespace(group=1), 1, 20, {EOS}, build=mock.Mock(return_value=rebuilt),
                           swap=swap), llm, rebuilt

    def test_sleep_mode_uses_level_one_and_wakes(self):
        sampler, llm, _ = self.sampler("sleep")
        sampler.rest()
        llm.sleep.assert_called_once_with(level=1)
        sampler.wake()
        llm.wake_up.assert_called_once_with()
        self.assertTrue(sampler.awake)

    def test_restart_mode_stops_then_builds_a_new_engine(self):
        sampler, llm, rebuilt = self.sampler("restart")
        sampler.rest()
        llm.llm_engine.engine_core.shutdown.assert_called_once_with()
        self.assertIsNone(sampler.llm)
        sampler.wake()
        self.assertIs(sampler.llm, rebuilt)

    def test_a_sleeping_engine_refuses_work(self):
        sampler, _, _ = self.sampler("sleep")
        sampler.rest()
        with self.assertRaisesRegex(RuntimeError, "asleep"):
            sampler.push(None)
        with self.assertRaisesRegex(RuntimeError, "asleep"):
            sampler.sample([[1, 2]])

    def test_only_turn_taking_models_enable_sleep(self):
        from rl.vllm_sampler import add_plugin_path

        add_plugin_path()
        small = engine_settings(PROFILES["qwen06b"], "main", None, "sleep")
        large = engine_settings(PROFILES["qwen4b"], "second", 16_000_000_000, "sleep")
        restart = engine_settings(PROFILES["qwen4b"], "second", 16_000_000_000, "restart")
        self.assertNotIn("enable_sleep_mode", small)
        self.assertEqual(small["gpu_memory_utilization"], 0.25)
        self.assertTrue(large["enable_sleep_mode"])
        self.assertNotIn("enable_sleep_mode", restart)
        self.assertEqual(large["hf_overrides"]["steering_layers"], [9, 18, 27])
        self.assertAlmostEqual(large["gpu_memory_utilization"], 13.0 / 16.0)


class TestWeightShuttle(unittest.TestCase):
    def test_leaving_the_card_points_back_at_the_file_view_without_a_copy(self):
        model = nn.Linear(3, 2, dtype=torch.float16)
        views = {name: torch.ones_like(w, dtype=torch.bfloat16) for name, w in model.named_parameters()}
        shuttle = WeightShuttle(model, views, torch.float16, "cpu")
        shuttle.to_gpu()
        self.assertEqual(model.weight.dtype, torch.float16)
        self.assertTrue(shuttle.on_gpu)
        shuttle.to_cpu()
        self.assertEqual(model.weight.data.data_ptr(), views["weight"].data_ptr())
        self.assertFalse(shuttle.on_gpu)

    def test_tied_weights_move_once(self):
        first, second = nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False)
        second.weight = first.weight
        model = nn.ModuleDict({"a": first, "b": second})
        shuttle = WeightShuttle(model, {"a.weight": torch.ones(2, 2)}, torch.float16, "cpu")
        self.assertEqual(len(shuttle.pairs), 1)
        shuttle.to_gpu()
        self.assertIs(model["a"].weight, model["b"].weight)

    def test_missing_or_wrong_file_tensor_is_refused(self):
        model = nn.Linear(3, 2, bias=False)
        with self.assertRaisesRegex(ValueError, "no matching"):
            WeightShuttle(model, {}, torch.float16, "cpu")
        with self.assertRaisesRegex(ValueError, "no matching"):
            WeightShuttle(model, {"weight": torch.zeros(9)}, torch.float16, "cpu")


def tiny_qwen():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(3)
    config = Qwen3Config(vocab_size=40, hidden_size=32, intermediate_size=64, num_hidden_layers=6,
                         num_attention_heads=4, num_key_value_heads=2, max_window_layers=6,
                         attn_implementation="eager", use_cache=False)
    return Qwen3ForCausalLM(config).eval().requires_grad_(False)


def steered_gradient(prepare=None, expert_off=False):
    """Gradient of a loss on three steering vectors, on a real tiny Qwen3 model."""
    from steering import SteeringBank

    model = tiny_qwen()
    bank = SteeringBank(["a"], (1, 3, 4), 32)
    with torch.no_grad():
        bank.vectors[0].copy_(torch.randn(3, 32) * 0.1)
    bank.attach(model)
    if prepare:
        prepare(model)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6], [0, 0, 7, 8, 9, 10]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]])
    with bank.choose(None if expert_off else 0):
        hidden = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        if expert_off:
            return hidden
        # A fixed direction, because the final norm layer makes a plain squared loss almost constant.
        target = torch.randn(hidden.shape, generator=torch.Generator().manual_seed(9))
        (hidden * target).sum().backward()
    return bank.vectors[0].grad.clone()


class TestGradientMemory(unittest.TestCase):
    def assert_same_gradient(self, prepare):
        expected = steered_gradient()
        self.assertGreater(float(expected[0].abs().max()), 0, "the first steered vector must get a gradient")
        got = steered_gradient(prepare)
        self.assertLess(float(((got - expected).norm() / expected.norm())), 1e-5)

    def test_no_grad_prefix_gives_the_full_pass_gradient(self):
        self.assert_same_gradient(lambda m: prepare_layers(torch, m.model.layers, 1, False, False))

    def test_checkpointing_gives_the_full_pass_gradient(self):
        self.assert_same_gradient(lambda m: prepare_layers(torch, m.model.layers, 1, False, True))

    def test_steering_hook_on_the_first_layer_keeps_its_gradient(self):
        got = steered_gradient(lambda m: prepare_layers(torch, m.model.layers, 1, False, True))
        self.assertTrue((got.norm(dim=-1) > 0).all())

    def test_prefix_body_builds_no_graph(self):
        weight = nn.Parameter(torch.ones(2))
        out = without_gradients(torch, lambda x: x * weight)(torch.ones(2))
        self.assertFalse(out.requires_grad)

    def test_checkpoint_is_skipped_without_a_gradient(self):
        with mock.patch("torch.utils.checkpoint.checkpoint") as saved:
            run = checkpointed(torch, lambda x, **_: x * 2)
            run(torch.ones(2))
            run(torch.ones(2, requires_grad=True), past_key_values=object())
            with torch.no_grad():
                run(torch.ones(2, requires_grad=True))
            saved.assert_not_called()
            run(torch.ones(2, requires_grad=True))
            saved.assert_called_once()

    def test_compiled_path_is_used_only_without_a_cache(self):
        eager, compiled = mock.Mock(return_value="eager"), mock.Mock(return_value="compiled")
        fake = SimpleNamespace(compile=lambda _fn, dynamic: compiled)
        forward = compiled_forward(fake, eager)
        self.assertEqual(forward(1), "compiled")
        self.assertEqual(forward(1, past_key_values=object()), "eager")

    def test_prefix_output_matches_a_plain_forward(self):
        plain = steered_gradient(expert_off=True)
        wrapped = steered_gradient(lambda m: prepare_layers(torch, m.model.layers, 1, False, True), expert_off=True)
        self.assertTrue(torch.equal(plain, wrapped))

    def test_smaller_head_chunks_do_not_change_the_gradient(self):
        torch.manual_seed(5)
        core, head = TinyCore(), nn.Linear(6, 12, bias=False)
        head.requires_grad_(False)
        batch = build_policy_batch(torch, [[1, 2, 3], [4]], [[5, 6, 7], [8, 9]], 0, "cpu")
        advantages = torch.tensor([0.5, -0.5])
        backward_policy_loss(torch, core, head, batch, advantages, 4.0)
        whole = core.vector.grad.clone()
        core.zero_grad()
        backward_policy_loss(torch, core, head, batch, advantages, 4.0, head_chunk=2)
        self.assertLess(float((core.vector.grad - whole).abs().max()), 1e-6)


class TestProfiles(unittest.TestCase):
    def test_default_stays_the_small_model(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("STEERING_PROFILE", None)
            self.assertEqual(get_profile().name, DEFAULT_PROFILE)
            self.assertEqual(get_profile().steering_layers, (7, 14, 21))

    def test_name_beats_environment(self):
        with mock.patch.dict(os.environ, {"STEERING_PROFILE": "qwen4b"}):
            self.assertEqual(get_profile().name, "qwen4b")
            self.assertEqual(get_profile("qwen06b").name, "qwen06b")

    def test_unknown_profile_is_refused(self):
        with self.assertRaisesRegex(ValueError, "Unknown model profile"):
            get_profile("nope")

    def test_4b_facts(self):
        profile = PROFILES["qwen4b"]
        self.assertEqual((profile.num_layers, profile.steering_layers, profile.read_layer), (36, (9, 18, 27), 9))
        self.assertTrue(profile.time_share and profile.gradient_checkpointing)
        self.assertIs(profile_for_model("Qwen/Qwen3-4B-Instruct-2507"), profile)

    def test_vllm_budgets_stay_under_the_hard_stop(self):
        profile = PROFILES["qwen4b"]
        for card, budget in profile.vllm_memory_gb.items():
            self.assertLess(budget + profile.other_vram_gb[card], profile.hard_stop_gb)

    def test_share_needs_the_card_size_only_for_gb_budgets(self):
        self.assertEqual(PROFILES["qwen06b"].vllm_share("main", None), 0.25)
        with self.assertRaisesRegex(ValueError, "card size"):
            PROFILES["qwen4b"].vllm_share("main", None)

    def test_chunk_budget_depends_on_card_and_sampler(self):
        self.assertEqual(PROFILES["qwen06b"].chunk_tokens("main", "vllm"), 2000)
        self.assertEqual(PROFILES["qwen06b"].chunk_tokens("main", "hf"), 4000)
        self.assertEqual(PROFILES["qwen4b"].chunk_tokens("main", "vllm"), 4000)
        self.assertEqual(PROFILES["qwen4b"].chunk_tokens("second", "vllm"), 8000)

    def test_width_comes_from_config_and_layer_count_is_checked(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "config.json"
            config.write_text(json.dumps({"num_hidden_layers": 36, "hidden_size": 2560}), encoding="utf-8")
            self.assertEqual(model_width(PROFILES["qwen4b"], folder), 2560)
            with self.assertRaisesRegex(ValueError, "expects 28 layers"):
                model_width(PROFILES["qwen06b"], folder)

    def test_engine_layers_come_from_the_config(self):
        from rl.vllm_sampler import add_plugin_path

        add_plugin_path()
        from qwen_steering import set_steering, steering_layers

        self.assertEqual(steering_layers(SimpleNamespace(steering_layers=[9, 18, 27])), (9, 18, 27))
        with self.assertRaisesRegex(ValueError, "no steering_layers"):
            steering_layers(SimpleNamespace())
        layers = [SimpleNamespace(steering_delta=torch.zeros(4, dtype=torch.float16)) for _ in range(36)]
        model = SimpleNamespace(model=SimpleNamespace(layers=layers), config=SimpleNamespace(steering_layers=[9, 18, 27]))
        set_steering(model, torch.ones(3, 4).numpy())
        self.assertEqual([i for i, l in enumerate(layers) if l.steering_delta.sum() > 0], [9, 18, 27])
        with self.assertRaisesRegex(ValueError, "2 vectors for 3"):
            set_steering(model, torch.ones(2, 4).numpy())

    def test_resume_refuses_vectors_from_another_model(self):
        old = {"layers": [7, 14, 21]}
        with self.assertRaisesRegex(ValueError, "Use a new --run"):
            pilot.check_resume(old, PROFILES["qwen4b"], "x.pt")
        pilot.check_resume(old, PROFILES["qwen06b"], "x.pt")  # Older files without a profile are 0.6B.
        pilot.check_resume({"layers": [9, 18, 27], "profile": "qwen4b"}, PROFILES["qwen4b"], "x.pt")

    def test_vram_guard(self):
        def run(gb, card="main", name="qwen4b"):
            cuda = SimpleNamespace(max_memory_reserved=lambda: gb * 1e9)
            return SimpleNamespace(torch=SimpleNamespace(cuda=cuda), profile=PROFILES[name], card=card)

        pilot.check_vram(run(11.2))  # 11.2 + 2.7 desktop = 13.9.
        with self.assertRaisesRegex(RuntimeError, "over the 14"):
            pilot.check_vram(run(11.4))
        pilot.check_vram(run(13.9, "second"))
        pilot.check_vram(run(99, name="qwen06b"))  # The 0.6B runs keep their old behavior.


if __name__ == "__main__":
    unittest.main()
