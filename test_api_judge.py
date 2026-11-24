"""Checks for the Haiku judge. These use a fake client and never call the API."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rl.judge_rubrics import RUBRICS, system_prompt
from rl.api_judge import (ApiJudge, BadVerdict, Group, JudgeFatal, build_jobs, combine_rewards, label_rewards, load_api_key,
                          parse_verdict, unique_answers)

FAKE_KEY = "sk-ant-" + "x" * 30


class StatusError(Exception):
    def __init__(self, status, body=None, retry_after="0"):
        self.status_code, self.body = status, body
        self.response = SimpleNamespace(headers={"retry-after": retry_after})


def reply(labels, stop="end_turn"):
    ranking = sorted(labels)
    scores = {label: 5 - place for place, label in enumerate(ranking)}
    usage = SimpleNamespace(input_tokens=100, output_tokens=20, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    message = SimpleNamespace(stop_reason=stop, usage=usage, _request_id="req_test",
                              content=[SimpleNamespace(type="text", text=json.dumps({"scores": scores, "ranking": ranking}))])
    return SimpleNamespace(parse=lambda: message, headers={"anthropic-ratelimit-requests-remaining": "9"})


class FakeClient:
    def __init__(self, errors=()):
        self.errors, self.calls = list(errors), []
        self.messages = SimpleNamespace(with_raw_response=SimpleNamespace(create=self.create))

    def with_options(self, **_):
        return self

    async def create(self, **request):
        self.calls.append(request)
        if self.errors:
            error = self.errors.pop(0)
            if error:
                raise error
        return reply(request["output_config"]["format"]["schema"]["properties"]["scores"]["required"])


def group(name="g1", answers=("one", "two", "three", "four")):
    return Group(name, "story", "Write a story", tuple(answers))


class JudgeCase(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.log = Path(self.folder.name) / "judge.jsonl"

    def tearDown(self):
        self.folder.cleanup()

    def judge(self, client, groups=None, **options):
        judge = ApiJudge(self.log, client=client, **options)
        try:
            return judge.judge(groups or [group()], 1)
        finally:
            judge.close()

    def rows(self, kind="call"):
        return [row for row in map(json.loads, self.log.read_text(encoding="utf-8").splitlines()) if row["type"] == kind]


class TestRequests(unittest.TestCase):
    def test_two_orders_are_exact_reverses(self):
        jobs, _ = build_jobs([group()], 3)
        self.assertEqual(len(jobs), 2)
        first, second = [list(job.labels.values()) for job in jobs]
        self.assertEqual(first, second[::-1])

    def test_identical_answers_are_sent_once(self):
        self.assertEqual(unique_answers(("a", "b", "a")), (["a", "b"], [0, 1, 0]))
        jobs, _ = build_jobs([group(answers=("a", "b", "a", "b"))], 1)
        self.assertEqual(len(jobs[0].labels), 2)

    def test_identical_group_needs_no_call(self):
        jobs, owners = build_jobs([group(answers=("a", "a"))], 1)
        self.assertEqual(jobs, [])
        self.assertEqual(combine_rewards(group(answers=("a", "a")), owners[0], []), [0.5, 0.5])


class TestVerdicts(unittest.TestCase):
    def test_valid_verdict(self):
        verdict = parse_verdict('{"scores":{"A":5,"B":1},"ranking":["A","B"]}', ["A", "B"])
        self.assertEqual(label_rewards(verdict), {"A": 1.0, "B": 0.0})

    def test_bad_verdicts_are_rejected(self):
        for text in ("no json", '{"scores":{"A":6,"B":1},"ranking":["A","B"]}', '{"scores":{"A":5},"ranking":["A","B"]}',
                     '{"scores":{"A":5,"B":1},"ranking":["A","A"]}', '{"scores":{"A":true,"B":1},"ranking":["A","B"]}'):
            with self.assertRaises(BadVerdict, msg=text):
                parse_verdict(text, ["A", "B"])

    def test_letters_map_back_to_the_right_answers(self):
        jobs, owners = build_jobs([group(answers=("best", "worst"))], 1)
        results = []
        for job in jobs:
            best = next(label for label, index in job.labels.items() if index == 0)
            worst = next(label for label, index in job.labels.items() if index == 1)
            results.append((job, {"scores": {best: 5, worst: 1}, "ranking": [best, worst]}))
        self.assertEqual(combine_rewards(group(answers=("best", "worst")), owners[0], results), [1.0, 0.0])

    def test_no_verdict_means_no_reward(self):
        self.assertIsNone(combine_rewards(group(), [0, 1, 2, 3], []))


class TestFailures(JudgeCase):
    def test_error_is_retried_once_then_succeeds(self):
        rewards, stats = self.judge(FakeClient([StatusError(500)]))
        self.assertIsNotNone(rewards[0])
        self.assertEqual((stats.ok, stats.retried, stats.skipped), (2, 1, 0))

    def test_second_error_skips_only_that_call(self):
        rewards, stats = self.judge(FakeClient([StatusError(500), StatusError(500)]))
        self.assertIsNotNone(rewards[0])  # The other label order still gives a reward.
        self.assertEqual((stats.ok, stats.skipped), (1, 1))

    def test_both_orders_failing_skips_the_group_without_a_low_score(self):
        client = FakeClient([StatusError(500)] * 4)
        rewards, stats = self.judge(client)
        self.assertIsNone(rewards[0])
        self.assertEqual((len(client.calls), stats.skipped, stats.groups_scored), (4, 2, 0))

    def test_bad_request_is_not_retried(self):
        client = FakeClient([StatusError(400)])
        _, stats = self.judge(client)
        self.assertEqual((len(client.calls), stats.skipped), (2, 1))

    def test_bad_key_stops_the_run(self):
        with self.assertRaises(JudgeFatal):
            self.judge(FakeClient([StatusError(401)]))

    def test_spend_cap_stops_the_run(self):
        body = {"error": {"details": {"error_code": "enforced_spend_limit_reached"}}}
        with self.assertRaises(JudgeFatal):
            self.judge(FakeClient([StatusError(429, body=body)]))

    def test_budget_stops_the_run(self):
        with self.assertRaises(JudgeFatal):
            self.judge(FakeClient(), budget_usd=0)


class TestLogging(JudgeCase):
    def test_log_has_tokens_but_no_key_or_answer_text(self):
        self.judge(FakeClient())
        text = self.log.read_text(encoding="utf-8")
        self.assertNotIn(FAKE_KEY, text)
        self.assertNotIn("three", text)
        self.assertEqual(len(self.rows()), 2)
        self.assertEqual(self.rows()[0]["tokens"]["input"], 100)
        self.assertEqual(self.rows("batch")[0]["rate_limits"]["requests-remaining"], 9)


class TestRequestFormat(unittest.TestCase):
    def test_thinking_is_off_and_json_is_required(self):
        request = build_jobs([group()], 1)[0][0].request
        self.assertEqual(request["thinking"], {"type": "disabled"})
        output = request["output_config"]["format"]
        self.assertEqual(output["type"], "json_schema")
        self.assertEqual(sorted(output["schema"]["required"]), ["ranking", "scores"])
        self.assertEqual(sorted(output["schema"]["properties"]["scores"]["required"]), list("ABCD"))


class TestBudgetResume(JudgeCase):
    def test_spend_from_the_log_counts_after_restart(self):
        self.judge(FakeClient())
        judge = ApiJudge(self.log, client=FakeClient(), budget_usd=1.0)
        try:
            self.assertGreater(judge.spent, 0)
        finally:
            judge.close()


class TestTopicRubrics(unittest.TestCase):
    def test_each_topic_gets_only_its_own_rubric(self):
        for topic, rubric in RUBRICS.items():
            prompt = system_prompt(topic)
            self.assertIn(rubric, prompt)
            for other, text in RUBRICS.items():
                if other != topic:
                    self.assertNotIn(text, prompt)

    def test_request_uses_the_group_topic(self):
        request = build_jobs([Group("g", "math_solving", "2+2?", ("4", "5"))], 1)[0][0].request
        self.assertEqual(request["system"][0]["text"], system_prompt("math_solving"))

    def test_unknown_topic_is_refused(self):
        with self.assertRaises(ValueError):
            system_prompt("cooking")


class TestKeyFile(unittest.TestCase):
    def test_key_is_read_from_the_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "key.txt"
            path.write_text(f"ANTHROPIC_API_KEY={FAKE_KEY}\n", encoding="utf-8")
            self.assertEqual(load_api_key(path), FAKE_KEY)


if __name__ == "__main__":
    unittest.main()
