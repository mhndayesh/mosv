"""Haiku judge as a helper process (the Gemma learner runs in a venv without anthropic).

Reads one JSON object on stdin: {"update": int, "run": str, "log": path, "groups": [[id, topic, prompt, [answers]]]}
Writes one JSON object on stdout: {"rewards": [[...] or null], "stats": {...}}. Uses rl/api_judge.py unchanged
(strict JSON verdicts, each topic's own rubric). Run with the ai-training-rocm-windows venv.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.api_judge import ApiJudge, Group  # noqa: E402


def main():
    job = json.loads(sys.stdin.read())
    judge = ApiJudge(job["log"], run=job["run"], budget_usd=job.get("budget", 20.0), concurrency=job.get("concurrency", 16))
    try:
        groups = [Group(g[0], g[1], g[2], tuple(g[3])) for g in job["groups"]]
        rewards, stats = judge.judge(groups, job["update"])
    finally:
        judge.close()
    print(json.dumps({"rewards": rewards, "stats": stats.summary()}))


if __name__ == "__main__":
    main()
