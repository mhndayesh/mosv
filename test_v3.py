"""Light tests for v3 (control vectors + routing proxy, folder v3/): vector export layout and the routing proxy's request handling. No GPU.

Run: python -m pytest test_v3.py   (export test skips when gguf-py is missing)
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "v3"))

import serve  # noqa: E402

HAS_GGUF = importlib.util.find_spec("gguf") is not None and importlib.util.find_spec("torch") is not None


class TestServe(unittest.TestCase):
    def test_router_reads_the_latest_user_turn_and_text_parts(self):
        messages = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "a"},
                    {"role": "user", "content": [{"type": "text", "text": "new"}, {"type": "image_url"}]}]
        self.assertEqual(serve.last_user_text(messages), "new")
        self.assertEqual(serve.last_user_text([{"role": "system", "content": "s"}]), "")

    def test_presets_give_each_topic_its_vector_and_none_gets_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(bundle=Path(tmp), gguf="base.gguf", device="Vulkan1", ctx=4096, parallel=2)
            text = serve.write_presets(args, ["debugging", "story"]).read_text(encoding="utf-8")
        sections = {part.split("]")[0]: part for part in text.split("[")[1:]}
        self.assertEqual(set(sections), {"steer-none", "steer-debugging", "steer-story"})
        self.assertNotIn("control-vector", sections["steer-none"])
        self.assertIn("steer-debugging.gguf", sections["steer-debugging"])
        self.assertNotIn("steer-story.gguf", sections["steer-debugging"])
        self.assertIn("load-on-startup = true", sections["steer-none"])


@unittest.skipUnless(HAS_GGUF, "needs torch and gguf-py")
class TestExportVectors(unittest.TestCase):
    def test_each_layer_gets_its_own_direction_tensor(self):
        import gguf
        import torch

        import export_vectors

        vectors = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4)
        found = export_vectors.directions({"bank": {"vectors.0": vectors}, "layers": (9, 18, 27)})
        self.assertEqual(sorted(found), [9, 18, 27])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "steer-x.gguf"
            export_vectors.write(path, "x", found)
            reader = gguf.GGUFReader(path)
            tensors = {t.name: t.data.tolist() for t in reader.tensors}
            del reader
        self.assertEqual(tensors["direction.18"], vectors[1].tolist())
        with self.assertRaises(ValueError):
            export_vectors.directions({"bank": {"vectors.0": vectors}, "layers": (0, 18, 27)})


if __name__ == "__main__":
    unittest.main()
