"""Dependency-light policy-gradient math, independently unit-tested."""

from __future__ import annotations

import math


def group_advantages(rewards: list[float]) -> list[float]:
    """Group-relative baseline, without std scaling (Dr.-GRPO-style)."""
    if len(rewards) < 2:
        raise ValueError("Need at least 2 rollouts per group")
    center = sum(rewards) / len(rewards)
    return [float(r - center) for r in rewards]


def reward_integer_answer(response: str, target: int) -> float:
    """Bounded, verifiable demo reward. Does NOT assess agent behavior."""
    import re

    answer = response.strip()
    if re.fullmatch(r"[+-]?\d+", answer) is None:
        return -0.25
    prediction = int(answer)
    if prediction == target:
        return 1.0
    error = abs(prediction - target)
    return 0.2 * math.exp(-min(error, 200) / 10.0)
