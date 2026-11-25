"""Light tests for v5/ (Bonsai trainer): the steer layout matches the GGUF export and starts as a no-op. CPU only.

Run: python -m unittest test_v5   (skips when torch is missing)
"""

import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "needs torch")
class TestSteer(unittest.TestCase):
    def make(self):
        import torch

        from v5.steer import LAYERS, Steer

        steer = Steer(["none", "engineering"], width=8, device="cpu", ff=4)
        mlps = {l: torch.nn.Linear(8, 8, bias=False).half() for l in LAYERS}
        model = type("M", (), {})()
        model.model = type("C", (), {})()
        model.model.layers = {l: type("L", (), {"mlp": mlps[l]})() for l in LAYERS}
        steer.attach(model)
        return steer, mlps

    def test_to_moe_matches_export_layout_and_expert_zero_is_none(self):
        steer, _ = self.make()
        part = steer.to_moe()["layers"][9]
        self.assertEqual(part["gate_inp"].shape, (2, 8))
        self.assertEqual(part["gate_exps"].shape, (2, 4, 8))
        self.assertEqual(part["down_exps"].shape, (2, 8, 4))
        self.assertEqual(str(part["up_exps"].dtype), "float16")
        self.assertEqual(float(abs(part["gate_exps"][0]).sum()), 0.0)

    def test_fresh_steer_changes_nothing_then_routes_and_adds(self):
        import torch

        steer, mlps = self.make()
        mlp = mlps[9]
        torch.manual_seed(0)
        x = torch.randn(1, 5, 8).half()
        base = mlp(x)  # hooked, but down = 0 and the all-zero gate picks expert 0
        self.assertTrue(torch.equal(base, torch.nn.functional.linear(x, mlp.weight)))
        with torch.no_grad():
            steer.gate[9][1] = 1.0  # tokens whose features sum above zero now pick the topic expert
            steer.experts[9]["down"].fill_(0.5)
            steer.experts[9]["gate"].fill_(0.5)
            steer.experts[9]["up"].fill_(0.5)
        out = mlp(x)
        picked = (x.float().sum(-1) > 0)[0]
        changed = (out != base).any(-1)[0]
        self.assertTrue(torch.equal(picked, changed))
        self.assertAlmostEqual(sum(steer.routed_share()[9].values()), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
