"""Tiny internal PyTorch assertions. Run: python -m unittest test_steering.py"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

try:
    import torch
    from torch import nn
    from steering import SteeringBank, install_steering, resolve_layer_indices
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("PyTorch is needed for internal transformer tests") from exc


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(3)])

    def forward(self, x):
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return x


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.model = TinyDecoder()

    def forward(self, x):
        return self.model(x)


class TestSteering(unittest.TestCase):
    def test_inside_model_and_router(self):
        torch.manual_seed(11)
        model = TinyLM()
        x = torch.randn(2, 3, 4)
        baseline = model(x).detach()
        base_copies = [p.detach().clone() for p in model.parameters()]
        bank, router = install_steering(model, ["skill_a", "skill_b"], [0, 2])
        self.assertIs(model.steering_bank, bank)
        self.assertIs(model.steering_router, router)
        self.assertTrue(all(not p.requires_grad for p in model.model.parameters()))
        self.assertTrue(bank.vectors[0].requires_grad)

        with bank.choose(None):
            self.assertTrue(torch.equal(model(x), baseline))

        with torch.no_grad():
            bank.vectors[0][0, 0] = 1.0
            bank.vectors[1][0, 0] = -1.0
        with bank.choose(0):
            chosen_output = model(x)
        self.assertGreater(float((chosen_output - baseline).abs().max()), 0)

        # Changing an unselected expert, even massively, does NOTHING.
        with torch.no_grad():
            bank.vectors[1].fill_(999.0)
        with bank.choose(0):
            self.assertTrue(torch.equal(model(x), chosen_output))

        for p in bank.parameters():
            p.grad = None
        with bank.choose(0):
            loss = model(x).square().mean()
            loss.backward()
        self.assertIsNotNone(bank.vectors[0].grad)
        self.assertIsNone(bank.vectors[1].grad)
        self.assertTrue(all(torch.equal(p, old) for p, old in zip(model.model.parameters(), base_copies)))


class TestLargeModelLayers(unittest.TestCase):
    def test_auto_layers_for_36_layers_are_9_18_27(self):
        model = SimpleNamespace(model=SimpleNamespace(layers=nn.ModuleList(nn.Identity() for _ in range(36))))
        self.assertEqual(resolve_layer_indices(model, "auto"), [9, 18, 27])

    def test_bank_holds_any_number_of_experts(self):
        names = [f"topic{i}" for i in range(8)]
        bank = SteeringBank(names, (9, 18, 27), 2560)
        self.assertEqual((len(bank.vectors), bank.vectors[7].shape), (8, torch.Size([3, 2560])))
        with self.assertRaises(ValueError), bank.choose(8):
            pass


if __name__ == "__main__":
    unittest.main()
