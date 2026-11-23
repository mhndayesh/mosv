"""Tests for the routed inference path with a tiny fake model, tokenizer and router.

Run: python -m unittest test_system.py
"""

from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

try:
    import torch
    from torch import nn
    from steering import SteeringBank
    from model_profiles import PROFILES
    from system import RouterHead, System, check_parts, load_vectors
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("PyTorch is needed for the routed inference tests") from exc

WIDTH, VOCAB, PAD = 8, 41, 0
EXPERTS = ("engineering", "story", "teaching")
PROMPTS = {
    "engineering": "build a parser",
    "story": "tell a tale of a fox",
    "teaching": "explain fractions to kids",
    "none": "hi",
}
READ_LAYER = 1


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(WIDTH, WIDTH)

    def forward(self, x):
        return x + torch.tanh(self.lin(x))


class FakeDecoder(nn.Module):
    """Per-token layers plus a mask-aware prompt average, so padding or wrong positions change the result."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, WIDTH)
        self.pos = nn.Embedding(64, WIDTH)
        self.layers = nn.ModuleList([Block() for _ in range(3)])

    def forward(self, input_ids, attention_mask, position_ids, use_cache=False):
        e = self.embed(input_ids)
        m = attention_mask.unsqueeze(-1).to(e.dtype)
        x = e + self.pos(position_ids) + (e * m).sum(1, keepdim=True) / m.sum(1, keepdim=True)
        for layer in self.layers:
            x = layer(x)
        return x


class FakeLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=WIDTH)
        self.model = FakeDecoder()
        self.lm_head = nn.Linear(WIDTH, VOCAB)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, max_new_tokens, **_):
        for _step in range(max_new_tokens):
            positions = (attention_mask.cumsum(1) - 1).clamp(min=0)
            hidden = self.model(input_ids, attention_mask, positions)
            token = self.lm_head(hidden[:, -1]).argmax(-1, keepdim=True)
            input_ids = torch.cat([input_ids, token], 1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(token)], 1)
        return input_ids


class Batch(dict):
    def to(self, _device):
        return self


class FakeBackend:
    def encode_batch(self, texts, add_special_tokens=False):
        return [SimpleNamespace(ids=ids(text)) for text in texts]


def ids(text: str) -> list[int]:
    return [ord(c) % 40 + 1 for c in text]


class FakeTokenizer:
    padding_side = "right"
    backend_tokenizer = FakeBackend()

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        return "U:" + messages[0]["content"] + ":A"

    def __call__(self, texts, padding, return_tensors, add_special_tokens):
        assert self.padding_side == "left"
        rows = [ids(t) for t in texts]
        width = max(map(len, rows))
        input_ids = torch.tensor([[PAD] * (width - len(r)) + r for r in rows])
        mask = torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in rows])
        return Batch(input_ids=input_ids, attention_mask=mask)

    def decode(self, row, skip_special_tokens):
        return " ".join(str(t) for t in row.tolist() if t != PAD)


def make_model() -> FakeLM:
    torch.manual_seed(5)
    return FakeLM().eval()


def make_bank() -> SteeringBank:
    torch.manual_seed(6)
    bank = SteeringBank(EXPERTS, (0, 1, 2), WIDTH)
    with torch.no_grad():
        for vector in bank.vectors:
            vector.copy_(torch.randn_like(vector) * 3)
    return bank


def make_system(model=None, bank=None, attach=True, temperature=1.0, threshold=0.5) -> System:
    model, bank = model or make_model(), bank or make_bank()
    if attach:
        bank.attach(model)
    parts = (ids("U:"), ids(":A"), PAD)
    router = RouterHead(nn.Linear(WIDTH, 4), torch.zeros(WIDTH), torch.ones(WIDTH), temperature, threshold,
                        (*EXPERTS, "none"), READ_LAYER)
    system = System(model, FakeTokenizer(), bank, router, parts, max_new_tokens=6)
    train_router_on_prototypes(system)
    return system


def train_router_on_prototypes(system: System) -> None:
    """Point each class at the features of its own prototype prompt, so routing is exact and known."""
    features = system._read(list(PROMPTS.values()))
    with torch.no_grad():
        system.router.linear.weight.copy_(torch.linalg.pinv(features).T * 50)  # Logits become 50 * identity.
        system.router.linear.bias.zero_()


class TestSystem(unittest.TestCase):
    def test_router_picks_each_prototype(self):
        system = make_system()
        routes = system.route(list(PROMPTS.values()))
        self.assertEqual([r.name for r in routes], list(PROMPTS))
        self.assertTrue(all(r.confidence > 0.5 for r in routes))

    def test_none_is_identical_to_the_original_model(self):
        plain = make_model()
        original = make_system(plain, make_bank(), attach=False)  # No hooks anywhere.
        system = make_system()
        answers = system.generate([PROMPTS["none"], "hi"])
        reference = original.generate_fixed([PROMPTS["none"], "hi"], None)
        self.assertEqual([a.route for a in answers], ["none", "none"])
        self.assertEqual([a.text for a in answers], reference)

    def test_correct_expert_is_applied(self):
        system = make_system()
        unsteered = system.generate_fixed([PROMPTS["story"]], None)[0]
        for name in EXPERTS:
            answer = system.generate([PROMPTS[name]])[0]
            self.assertEqual(answer.route, name)
            self.assertEqual(answer.text, system.generate_fixed([PROMPTS[name]], name)[0])
        solo = {n: system.generate_fixed([PROMPTS["story"]], n)[0] for n in EXPERTS}
        self.assertEqual(len(set(solo.values()) | {unsteered}), 4, "experts must change the answer, or this test proves nothing")

    def test_mixed_batch_does_not_leak_an_expert(self):
        system = make_system()
        names = ["story", "none", "engineering", "teaching", "story", "none", "engineering"]
        prompts = [PROMPTS[n] for n in names]
        for batch_size in (16, 2, 1):
            answers = system.generate(prompts, batch_size=batch_size)
            for name, prompt, answer in zip(names, prompts, answers):
                expert = None if name == "none" else name
                self.assertEqual(answer.route, name)
                self.assertEqual(answer.text, system.generate_fixed([prompt], expert)[0], (batch_size, name))
        self.assertIsNone(system.bank.active_expert)

    def test_left_padding_is_correct_for_unequal_prompts(self):
        system = make_system()
        short, long = PROMPTS["none"], PROMPTS["teaching"] + " with a very long extra tail"
        together = system.generate_fixed([short, long], "story", batch_size=2)
        alone = [system.generate_fixed([p], "story")[0] for p in (short, long)]
        self.assertEqual(together, alone)

    def test_router_reads_with_steering_off(self):
        system = make_system()
        seen = []
        layer = system.model.model.layers[READ_LAYER]
        layer.register_forward_hook(lambda *_: seen.append(system.bank.active_expert))
        clean = system._read(list(PROMPTS.values()))
        with system.bank.choose(2):  # A caller left an expert on: the read must still be unsteered.
            routes = system.route(list(PROMPTS.values()))
            self.assertEqual(system.bank.active_expert, 2)
        self.assertEqual([r.name for r in routes], list(PROMPTS))
        self.assertTrue(seen and all(value is None for value in seen))
        with system.bank.choose(2):
            self.assertTrue(torch.equal(system._read(list(PROMPTS.values())), clean))

    def test_low_confidence_becomes_none(self):
        system = make_system(temperature=1e6)  # Probabilities all near 0.25, under the 0.5 threshold.
        routes = system.route(list(PROMPTS.values()))
        self.assertEqual([r.name for r in routes], ["none"] * 4)
        self.assertTrue(all(r.confidence < 0.5 for r in routes))
        answers = system.generate([PROMPTS["story"]])
        self.assertEqual(answers[0].route, "none")
        self.assertEqual(answers[0].text, system.generate_fixed([PROMPTS["story"]], None)[0])

    def test_class_mismatch_is_rejected(self):
        system = make_system()
        wrong = RouterHead(system.router.linear, system.router.mean, system.router.std, 1.0, 0.5,
                           ("story", "engineering", "teaching", "none"), READ_LAYER)
        with self.assertRaises(ValueError):
            System(system.model, FakeTokenizer(), system.bank, wrong, (ids("U:"), ids(":A"), PAD))


class TestNExpertsAndModelChecks(unittest.TestCase):
    def test_eight_experts_route_and_steer(self):
        names = tuple(f"topic{i}" for i in range(8))
        model, bank = make_model(), SteeringBank(names, (0, 1, 2), WIDTH)
        with torch.no_grad():
            for vector in bank.vectors:
                vector.copy_(torch.randn_like(vector) * 3)
        bank.attach(model)
        router = RouterHead(nn.Linear(WIDTH, 9), torch.zeros(WIDTH), torch.ones(WIDTH), 1.0, 0.0, (*names, "none"),
                            READ_LAYER)
        system = System(model, FakeTokenizer(), bank, router, (ids("U:"), ids(":A"), PAD), max_new_tokens=3)
        answer = system.generate(["hello there"])[0]
        self.assertIn(answer.route, router.classes)
        self.assertEqual(answer.text, system.generate_fixed(["hello there"], None if answer.route == "none"
                                                            else answer.route)[0])

    def saved(self, profile, width):
        return {"model_id": profile.model_id, "read_layer": profile.read_layer, "mean": torch.zeros(width)}

    def test_parts_from_the_right_model_pass(self):
        profile = PROFILES["qwen4b"]
        check_parts(profile, self.saved(profile, 2560), 2560, (9, 18, 27), [torch.zeros(3, 2560)])

    def test_old_router_or_vectors_on_the_new_model_are_refused(self):
        new, old = PROFILES["qwen4b"], PROFILES["qwen06b"]
        with self.assertRaisesRegex(ValueError, "Router was trained"):
            check_parts(new, self.saved(old, 2560), 2560, (9, 18, 27), [torch.zeros(3, 2560)])
        with self.assertRaisesRegex(ValueError, "layers"):
            check_parts(new, self.saved(new, 2560), 2560, (7, 14, 21), [torch.zeros(3, 2560)])
        with self.assertRaisesRegex(ValueError, "width"):
            check_parts(new, self.saved(new, 2560), 2560, (9, 18, 27), [torch.zeros(3, 1024)])


if __name__ == "__main__":
    unittest.main()
