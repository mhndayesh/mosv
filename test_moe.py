"""Light tests for v4 (moe/): GGUF export layout and the training settings. No GPU.

Run: python -m pytest test_moe.py   (export test skips when torch or gguf-py is missing)
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "moe"))

HAS_GGUF = importlib.util.find_spec("gguf") is not None and importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_GGUF, "needs torch and gguf-py")
class TestMoeExport(unittest.TestCase):
    def test_expert_zero_is_none_and_steer_mlps_become_the_shared_expert(self):
        import torch

        import export

        n, width, ff = 2, 4, 3
        saved = {"classes": ["none", "a", "b"], "layers": [9], "ff": ff, "gate": {9: torch.ones(n + 1, width)},
                 "experts": {9: {"gate": torch.ones(n, ff, width), "up": torch.ones(n, ff, width),
                                 "down": torch.ones(n, width, ff)}},
                 "settings": {"seed": 0, "run": "x"}}
        with tempfile.TemporaryDirectory() as tmp:
            torch.save(saved, Path(tmp) / "moe.pt")
            moe = export.load_moe(Path(tmp) / "moe.pt")
        parts = moe["layers"][9]
        self.assertEqual(parts["gate_exps"].shape, (n + 1, ff, width))
        self.assertEqual(float(abs(parts["down_exps"][0]).sum()), 0.0)  # Expert 0 adds nothing.
        self.assertEqual(float(parts["down_exps"][1].sum()), width * ff)
        fake = SimpleNamespace(tensors=[SimpleNamespace(name=f"blk.{l}.ffn_{p}.weight") for l in (8, 9) for p in ("gate", "up", "down")])
        names = [name for name, _ in export.tensor_list(fake, moe)]
        self.assertIn("blk.9.ffn_down_shexp.weight", names)
        self.assertNotIn("blk.9.ffn_down.weight", names)
        self.assertIn("blk.8.ffn_down.weight", names)  # Other layers stay dense.
        self.assertIn("blk.9.ffn_gate_inp.weight", names)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "needs torch")
class TestTrainSettings(unittest.TestCase):
    def test_seed_defaults_to_zero_and_can_be_set(self):
        import train

        self.assertEqual(train.parse_args([]).seed, 0)
        self.assertEqual(train.parse_args(["--seed", "3"]).seed, 3)


class TestFullTestSettings(unittest.TestCase):
    def test_llama_server_path_can_be_chosen(self):
        import full_test

        self.assertEqual(full_test.parse_args(["m.gguf"]).llama, full_test.LLAMA)
        self.assertEqual(full_test.parse_args(["m.gguf", "--llama", "x.exe"]).llama, Path("x.exe"))


if __name__ == "__main__":
    unittest.main()
