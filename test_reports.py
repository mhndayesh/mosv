"""Light tests for the full-length test, the blind scorer and the PDF builder. No GPU, no model, no API calls.

Run: python -m pytest test_reports.py   (PDF tests skip when reportlab is missing)
"""

import asyncio
import importlib.util
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "reports"))

from reports import score_full as sf  # noqa: E402

HAS_REPORTLAB = importlib.util.find_spec("reportlab") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


def pair(topic="debugging", steered="S answer", normal="N answer", pid=1):
    return {"id": pid, "type": topic, "prompt": "Why does it crash?", "steered": steered, "normal": normal}


class FakeClient:
    """Answers like Haiku: answer_1 gets `first`, answer_2 gets `second`, after `bad` broken replies."""

    def __init__(self, first=4, second=2, bad=0):
        self.first, self.second, self.bad, self.requests = first, second, bad, []
        self.messages = self

    async def create(self, **request):
        self.requests.append(request)
        keys = [f"c{i}" for i in range(1, 6)]
        if self.bad:
            self.bad -= 1
            text = '{"answer_1": {"c1": 9}}'
        else:
            text = json.dumps({"answer_1": dict.fromkeys(keys, self.first), "answer_2": dict.fromkeys(keys, self.second)})
        usage = SimpleNamespace(input_tokens=100, output_tokens=20, cache_read_input_tokens=0, cache_creation_input_tokens=0)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], usage=usage)


class TestScoreFull(unittest.TestCase):
    def test_every_topic_has_five_criteria_and_none_gets_the_general_rubric(self):
        for topic in list(sf.RUBRICS) + ["none"]:
            self.assertEqual(len(sf.criteria(topic)), 5, topic)
        self.assertEqual(sf.rubric("none"), sf.GENERAL)

    def test_request_is_blind_and_strict(self):
        req = sf.request(pair(steered="AAA", normal="BBB"), "AAA", "BBB")
        text = json.dumps(req).lower()
        self.assertNotIn("steer", text)
        self.assertIn('label=\\"1\\"', json.dumps(req))
        fmt = req["output_config"]["format"]
        self.assertEqual(fmt["type"], "json_schema")
        self.assertFalse(fmt["schema"]["additionalProperties"])
        self.assertEqual(fmt["schema"]["properties"]["answer_1"]["required"], [f"c{i}" for i in range(1, 6)])

    def test_parse_accepts_only_full_valid_scores(self):
        good = json.dumps({"answer_1": {f"c{i}": 3 for i in range(1, 6)}, "answer_2": {f"c{i}": 5 for i in range(1, 6)}})
        self.assertEqual(sf.parse(good, 5), {"answer_1": [3] * 5, "answer_2": [5] * 5})
        for bad in ('{"answer_1": {"c1": 3}, "answer_2": {}}',
                    json.dumps({"answer_1": {f"c{i}": 6 for i in range(1, 6)}, "answer_2": {f"c{i}": 1 for i in range(1, 6)}}),
                    json.dumps({"answer_1": {f"c{i}": 3.0 for i in range(1, 6)}, "answer_2": {f"c{i}": 1 for i in range(1, 6)}}),
                    "not json"):
            with self.assertRaises(Exception):
                sf.parse(bad, 5)

    def test_judge_maps_labels_back_to_the_right_side_in_both_orders(self):
        async def both():
            client, gate = FakeClient(first=4, second=2), asyncio.Semaphore(2)
            return await asyncio.gather(sf.judge(client, gate, pair(), "steered", []),
                                        sf.judge(client, gate, pair(), "normal", []))

        calls = asyncio.run(both())
        self.assertEqual(calls[0]["steered"], [4] * 5)  # Steered shown first gets answer_1's score.
        self.assertEqual(calls[1]["normal"], [4] * 5)  # Normal shown first gets answer_1's score.
        summary = sf.summarize(pair(), calls)
        self.assertEqual((summary["steered_total"], summary["normal_total"]), (3.0, 3.0))  # Pure position effect cancels.

    def test_broken_reply_is_retried_and_a_dead_call_is_never_scored(self):
        ok = asyncio.run(sf.judge(FakeClient(bad=1), asyncio.Semaphore(1), pair(), "steered", []))
        self.assertEqual(ok["attempts"], 2)
        saved, sf.ATTEMPTS = sf.ATTEMPTS, 1
        try:
            dead = asyncio.run(sf.judge(FakeClient(bad=5), asyncio.Semaphore(1), pair(), "steered", []))
        finally:
            sf.ATTEMPTS = saved
        self.assertIn("error", dead)
        self.assertIsNone(sf.summarize(pair(), [dead, dead])["steered_total"])


@unittest.skipUnless(HAS_TORCH, "vllm_full_test imports the project's torch modules")
class TestVllmFullTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from reports import vllm_full_test
        cls.vt = vllm_full_test

    def test_loop_check_catches_short_and_long_loops_but_not_text(self):
        rng = random.Random(0)

        def loop(period, n):
            return ([rng.randrange(9999) for _ in range(period)] * (n // period + 1))[:n]

        self.assertTrue(self.vt.is_looping(loop(300, 2048)))
        self.assertTrue(self.vt.is_looping(loop(2500, 6016)))
        self.assertFalse(self.vt.is_looping([rng.randrange(150000) for _ in range(6016)]))
        self.assertFalse(self.vt.is_looping(loop(300, 1920)))  # Too short to judge.
        self.assertFalse(self.vt.is_looping(loop(300, 2049)))  # Checked only every 128 tokens.

    def test_manual_stop_keeps_text_and_only_hits_listed_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            abort = Path(tmp) / "abort.txt"
            abort.write_text("3\n", encoding="utf-8")
            old, self.vt.ABORT_FILE = self.vt.ABORT_FILE, abort
            stopped = []
            engine = SimpleNamespace(abort_request=stopped.extend)
            latest, done = {"3": [1, 2, 3], "4": [5]}, {}
            try:
                self.vt.manual_stops(engine, latest, done)
            finally:
                self.vt.ABORT_FILE = old
        self.assertEqual(stopped, ["3"])
        self.assertEqual(done, {3: {"tokens": [1, 2, 3], "reason": "manual"}})
        self.assertEqual(list(latest), ["4"])

    def test_progress_round_trips_and_is_keyed_by_context_and_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(out=Path(tmp) / "t.json", max_model_len=81920, run="r1")
            self.assertEqual(self.vt.load_progress(args), {})
            saved = {"normal": [[{"tokens": [1], "reason": "stop"}], {"seconds": 1}], "steered": {}}
            self.vt.save_progress(args, saved)
            self.assertEqual(self.vt.load_progress(args), saved)
            self.assertEqual(self.vt.load_progress(SimpleNamespace(out=args.out, max_model_len=131072, run="r1")), {})


@unittest.skipUnless(HAS_REPORTLAB, "needs reportlab")
class TestBuildPdfFull(unittest.TestCase):
    def test_pdf_has_a_bookmark_per_section_topic_and_prompt(self):
        import re

        import build_pdf_full as b

        rows, scores = [], []
        for i, topic in enumerate(["debugging", "debugging", "story", "none"], start=1):
            rows.append({"id": i, "type": topic, "prompt": f"prompt {i}", "route": topic, "confidence": 0.9,
                         "normal": "line\n" * 300 if i == 1 else "n", "steered": "s", "normal_tokens": 9,
                         "steered_tokens": 9, "normal_end": "stop", "steered_end": "repeat"})
            scores.append({"id": i, "topic": topic, "criteria": sf.criteria(topic), "normal": [2.0] * 5,
                           "steered": [4.0] * 5, "normal_total": 2.0, "steered_total": 4.0,
                           "calls": [{"answer_1_is": "steered"}, {"answer_1_is": "normal"}]})
        stats = {"capacity": {"max_model_len": 81920}, "normal_pass": {"requests": 4, "answer_tokens": 36, "seconds": 1.0}}
        with tempfile.TemporaryDirectory() as tmp:
            for name, data in (("vllm_full_test.json", rows), ("full_scores.json", scores),
                               ("vllm_full_test.stats.json", stats)):
                (Path(tmp) / name).write_text(json.dumps(data), encoding="utf-8")
            old_here, old_argv = b.HERE, sys.argv
            b.HERE, sys.argv = Path(tmp), ["x", str(Path(tmp) / "t.pdf")]
            try:
                b.main()
            finally:
                b.HERE, sys.argv = old_here, old_argv
            pdf = (Path(tmp) / "t.pdf").read_bytes()
        titles = re.findall(rb"/Title \(([^)]*)\)", pdf)[1:]  # The first title is the document's own.
        self.assertEqual(len(titles), 3 + 3 + 4)  # Abstract, criteria, index + 3 topics + 4 prompts.
        self.assertIn(b"/UseOutlines", pdf)
        self.assertGreater(len(re.findall(rb"/Subtype /Link", pdf)), 4)


if __name__ == "__main__":
    unittest.main()
