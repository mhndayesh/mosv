"""Tests for the v6.0.1 rules that have branches: the gate verdict, the keep rule, the scoring math, the data split.

Run (ai-training-rocm-windows venv, repo root):  python -m pytest test_v6_0_1.py -q
No GPU and no API call: these only test the decisions, not the stages that produce their inputs.
"""

from v6_0_1 import build_data, checks, gate
from v6_0_1.behaviors import BEHAVIORS, guide, questions
from v6_0_1.evaluate import compare
from v6_0_1.train import split


def test_every_topic_has_rules_four_checks_and_a_guard():
    for topic, entry in BEHAVIORS.items():
        assert entry["guide"].strip(), topic
        assert list(questions(topic)) == ["b1", "b2", "b3", "b4", "g1"], topic
        assert guide(topic).startswith("Follow these rules"), topic
    assert "math_solving" not in BEHAVIORS  # ability, not behavior: left untrained on purpose


def test_shares_counts_behavior_and_guard_separately():
    assert checks.shares({"b1": True, "b2": False, "b3": True, "b4": True, "g1": False}) == (0.75, 0.0)
    assert checks.shares({"b1": False, "b2": False, "b3": False, "b4": False, "g1": True}) == (0.0, 1.0)


def test_gate_trains_only_when_told_helps_guard_holds_and_checker_repeats():
    assert gate.verdict(0.40, 0.70, 0.90, 0.90, 0.95)[0] is True
    assert gate.verdict(0.40, 0.50, 0.90, 0.90, 0.95)[0] is False  # telling adds too little: nothing to teach
    assert gate.verdict(0.40, 0.70, 0.90, 0.70, 0.95)[0] is False  # telling breaks correctness
    assert gate.verdict(0.40, 0.70, 0.90, 0.90, 0.60)[0] is False  # the checker does not repeat itself
    assert gate.verdict(0.74, 0.89, 0.60, 0.55, 0.98)[0] is True   # exactly +15 points counts (0.89 - 0.74 is 0.1499.. in floats)
    assert gate.verdict(0.40, 0.70, 0.75, 0.65, 0.95)[0] is True   # a guard drop of exactly 10 points is still inside the limit


def test_repeat_rate_is_the_share_of_identical_answers():
    one = [{"id": "a", "variant": "plain", "copy": 0, "topic": "story", "pass": {"b1": True, "b2": True}}]
    two = [{"id": "a", "variant": "plain", "copy": 0, "topic": "story", "pass": {"b1": True, "b2": False}}]
    assert gate.repeat_rate(one, two) == {"story": 0.5}


def told(behavior, guard=1.0, finish="stop", tokens=100):
    return {"behavior": behavior, "guard": guard, "finish": finish, "tokens": tokens, "answer": "text"}


def test_choose_keeps_the_best_complete_safe_answer_that_beats_plain():
    best, reason = build_data.choose(0.25, [told(0.75, tokens=300), told(1.0, tokens=200)])
    assert reason == "kept" and best["behavior"] == 1.0
    best, _ = build_data.choose(0.25, [told(1.0, tokens=300), told(1.0, tokens=120)])
    assert best["tokens"] == 120  # same checks passed: the shorter answer wins


def test_choose_drops_cut_off_unsafe_weak_or_no_better_answers():
    assert build_data.choose(0.25, [told(1.0, finish="length")]) == (None, "cut off")
    assert build_data.choose(0.25, [told(1.0, guard=0.0)]) == (None, "guard failed")
    assert build_data.choose(0.25, [told(0.5)]) == (None, "too few checks passed")
    assert build_data.choose(1.0, [told(1.0)]) == (None, "not better than plain")


def test_split_is_stable_and_disjoint():
    rows = [{"id": f"{i:04x}"} for i in range(400)]
    train, held = split(rows)
    assert split(rows) == (train, held)                       # the same rows are held out on every run
    assert not {r["id"] for r in train} & {r["id"] for r in held}
    assert len(train) + len(held) == 400 and 20 <= len(held) <= 60  # about one row in ten


def test_claim_refuses_answers_written_under_other_rules(tmp_path):
    import pytest

    from v6_0_1.files import claim

    claim(tmp_path, "aaaa", fresh=False)                 # an empty folder is claimed for the current rules
    (tmp_path / "answers.jsonl").write_text("{}\n", encoding="utf-8")
    claim(tmp_path, "aaaa", fresh=False)                 # same rules: the cached answers are reused
    with pytest.raises(SystemExit):
        claim(tmp_path, "bbbb", fresh=False)             # the rules changed: stop instead of mixing
    claim(tmp_path, "bbbb", fresh=True)                  # --fresh: the old answers are deleted
    assert not (tmp_path / "answers.jsonl").exists()


def test_variety_caps_speech_openings_and_repeated_names():
    from v6_0_1 import variety

    assert variety.names('"Run," Elias said. The night was long.') == {"Elias"}
    rows = [{"answer": f'"Go," Elias said {i}.'} for i in range(5)] + [{"answer": f"Mara lifted the lid {i}."} for i in range(5)]
    kept, dropped = variety.limit(rows)
    assert len(dropped) == 3          # Elias may appear in at most 2 kept answers (10 rows: 5% rounds up to the minimum 2)
    assert sum(variety.opens_with_speech(r["answer"]) for r in kept) == 2


def test_compare_counts_a_tie_as_half_a_win():
    base = {"story": {"n": 2, "behavior": 0.5, "by_prompt": {"a": 0.5, "b": 0.5}}}
    new = {"story": {"n": 2, "behavior": 0.75, "by_prompt": {"a": 1.0, "b": 0.5}}}
    assert compare(base, new)["story"]["trained_wins_pct"] == 75.0
