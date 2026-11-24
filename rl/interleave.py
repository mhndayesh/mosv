"""Train all topics in one shuffled run: each small update belongs to one topic, and the next is usually another.

vLLM applies one vector at a time, so a single update cannot mix topics. Every topic keeps its own vector, optimizer,
best vector and checkpoint. One bank slot is loaded with the current topic's vector before each step.
"""

from __future__ import annotations

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rl import data as rl_data
from rl import pilot as core
from rl.data import read_split


@dataclass
class TopicState:
    expert: str
    train: list[dict]
    dev: list[dict]
    test: list[dict]
    updates: int
    vector: Any
    optimizer: Any
    caps: Any
    path: Path
    step: int = 0
    probe: Any = None
    base_answers: list = field(default_factory=list)
    base_dev: dict = field(default_factory=dict)
    best: tuple = ()
    history: list = field(default_factory=list)


def mixed_schedule(counts: dict[str, int], seed: int) -> list[str]:
    """A shuffled order with each topic's exact count, avoiding the same topic twice in a row while possible."""
    rng, left, order, last = random.Random(seed), {topic: counts[topic] for topic in sorted(counts)}, [], None
    while any(left.values()):
        choices = [topic for topic, count in left.items() if count and topic != last] or [last]
        last = rng.choices(choices, weights=[left[topic] for topic in choices])[0]
        left[last] -= 1
        order.append(last)
    return order


def remaining_schedule(order: list[str], done: dict[str, int]) -> list[str]:
    """Drop the updates each topic already finished, so a resumed run continues the same order."""
    skipped, out = dict.fromkeys(done, 0), []
    for topic in order:
        if skipped.get(topic, 0) < done.get(topic, 0):
            skipped[topic] += 1
        else:
            out.append(topic)
    return out


def rows_for(state: TopicState, step: int, prompts: int) -> list[dict]:
    return [state.train[(step * prompts + i) % len(state.train)] for i in range(prompts)]


def load_topic(run, bank, state: TopicState) -> None:
    with run.torch.no_grad():
        bank.vectors[0].copy_(state.vector)


def save_topic(run, bank, state: TopicState) -> None:
    best_vector, best_score, best_step = state.best
    core.save_atomic(run.torch, core.vector_state(run, bank, state.optimizer, state.step, state.expert, best_step=best_step,
                                                  best_vector=best_vector.detach().cpu(), best_score=best_score,
                                                  history=state.history), state.path)


def restore_topic(run, bank, state: TopicState) -> None:
    if not state.path.exists() or run.args.fresh:
        return
    saved = run.torch.load(state.path, map_location="cuda:0", weights_only=True)
    core.check_resume(saved, run.profile, state.path)
    bank.load_state_dict(saved["bank"], strict=False)
    state.optimizer.load_state_dict(saved["optimizer"])
    state.vector, state.step = bank.vectors[0].detach().clone(), saved["step"]
    state.history = saved.get("history", [])
    if "best_vector" in saved:
        state.best = (saved["best_vector"].to(state.vector.device), saved["best_score"], saved["best_step"])


def prepare_topic(run, bank, expert: str, folder: Path) -> TopicState:
    torch, args = run.torch, run.args
    loaded = [read_split(expert, name) for name in ("routed", "dev", "test")]
    train, dev, test = (core.fitting_rows(run, rows) for rows in loaded)
    random.Random(core.SEED).shuffle(train)
    dev, test = core.by_length(dev[:args.dev]), test[:args.test]
    updates = core.one_pass_updates(len(train), args.prompts) if args.one_pass else args.updates
    norms, typical = core.activation_norms(run, train)
    with torch.no_grad():
        bank.vectors[0].zero_()
    optimizer = core.make_optimizer(run, bank.vectors[0], norms)
    state = TopicState(expert, train, dev, test, updates, bank.vectors[0].detach().clone(), optimizer,
                       typical * args.cap_fraction, folder / f"{expert}.pt")
    state.best = (state.vector.clone(), core.JUDGE_NEUTRAL_SCORE, 0)
    restore_topic(run, bank, state)
    state.base_answers = core.eval_answers(run, bank, None, dev)
    base = core.score_answers(run, dev, state.base_answers, expert)
    state.base_dev = {**base, "checklist_score": base["score"], "score": core.JUDGE_NEUTRAL_SCORE}
    state.probe = core.encode(run, [row["prompt"] for row in train[:core.PROBE_PROMPTS]])
    core.say(run.card, expert=expert, prepared=True, start_step=state.step, updates=updates, train=len(train), dev=len(dev),
             test=len(test), degenerate=base["degenerate"], looping=base["looping"])
    return state


def mixed_step(run, bank, state: TopicState, job, start_next) -> tuple[dict, Any]:
    """Collect this batch, start the next one, then judge and learn on this topic's own vector."""
    swap_before = run.turns.swap_seconds
    run.torch.cuda.reset_peak_memory_stats()
    started = time.time()
    sampled, write_seconds = job.future.result()
    wait_seconds = time.time() - started
    next_job = start_next()
    load_topic(run, bank, state)
    run.judge_update = state.step
    rollouts = core.written_rollouts(run, job, sampled, state.expert)
    info = core.learn_step(run, bank, state.optimizer, state.caps, len(job.rows), rollouts, state.probe, write_seconds,
                           swap_before, job.vector)
    state.vector = bank.vectors[0].detach().clone()
    return {**info, "write_wait_s": round(wait_seconds, 2)}, next_job


def dev_check(run, bank, state: TopicState) -> None:
    load_topic(run, bank, state)
    run.judge_update = state.step
    result = core.judged_dev(run, bank, state.dev, state.base_answers, state.expert, f"dev_{state.step}")
    state.best = core.record_dev(run, bank, state.optimizer, state.history, result, bank.vectors[0].detach().clone(),
                                 state.step, state.base_dev, state.expert, state.best)
    state.vector = bank.vectors[0].detach().clone()
    save_topic(run, bank, state)


def finish_topic(run, bank, state: TopicState, report_dir: Path) -> None:
    best_vector, best_score, best_step = state.best
    core.save_best(run, bank, best_vector, state.optimizer, state.step, best_step, state.expert, state.path)
    state.vector = best_vector.clone()
    run.judge_update = state.step
    final = core.judged_final(run, bank, state.test, state.expert)
    report = {"expert": state.expert, "updates": state.step, "dev_base": state.base_dev["score"], "dev_history": state.history,
              "best_dev": best_score, "best_step": best_step, "test_prompts": len(state.test), "test": final,
              "settings": dict(vars(run.args))}
    (report_dir / f"{state.expert}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    core.say(run.card, expert=state.expert, finished=True, best_step=best_step, test_base=final["base"]["score"],
             test_steered=final["steered"]["score"], test_random=final["random"]["score"],
             beats_base_pct=final["steered"]["beats_base_pct"], steered_looping=final["steered"]["looping"])


def is_finished(run, expert: str, folder: Path, report_dir: Path) -> bool:
    return not run.args.fresh and (report_dir / f"{expert}.json").exists() and (folder / f"{expert}.pt").exists()


def train_mixed(run, experts: list[str]) -> None:
    from steering import SteeringBank

    args = run.args
    folder, report_dir = core.ROOT / "checkpoints" / "rl_pilot" / args.run, rl_data.OUT / args.run
    folder.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    bank = SteeringBank(["current_topic"], run.profile.steering_layers, run.model.config.hidden_size).to("cuda:0")
    bank.attach(run.model)
    todo = [expert for expert in experts if not is_finished(run, expert, folder, report_dir)]
    states = {expert: prepare_topic(run, bank, expert, folder) for expert in todo}
    order = remaining_schedule(mixed_schedule({e: s.updates for e, s in states.items()}, core.SEED),
                               {e: s.step for e, s in states.items()})
    core.say(run.card, mixed_updates=len(order), topics=len(states), first_20=order[:20])
    pool, pending, skipped_in_row, started = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vllm-writer"), None, 0, time.time()
    try:
        for index, topic in enumerate(order):
            state = states[topic]
            job = pending or core.start_write(run, pool, bank, rows_for(state, state.step, args.prompts), state.vector)
            check_due = (state.step + 1) % args.eval_every == 0 or state.step + 1 == state.updates

            def start_next():
                if check_due or index + 1 == len(order):
                    return None  # Scoring also uses vLLM, so no batch may be writing then.
                following = states[order[index + 1]]
                offset = 1 if following is state else 0
                return core.start_write(run, pool, bank, rows_for(following, following.step + offset, args.prompts), following.vector)

            info, pending = mixed_step(run, bank, state, job, start_next)
            skipped_in_row = skipped_in_row + 1 if info.get("skipped_update") else 0
            if skipped_in_row >= core.MAX_SKIPPED_UPDATES:
                raise RuntimeError(f"{skipped_in_row} updates in a row had too few judged prompts")
            state.step += 1
            save_topic(run, bank, state)
            core.say(run.card, expert=topic, step=state.step, of=state.updates, overall=index + 1, overall_of=len(order), **info,
                     seconds=round(time.time() - started, 1))
            if check_due:
                dev_check(run, bank, state)
            if state.step == state.updates:
                finish_topic(run, bank, state, report_dir)
    finally:
        pool.shutdown(wait=True)
    bank.detach()
