"""RL pilot: one model per card, each training its own behavior vectors.

Run in WSL: python -m rl.pilot --run pilot
Every result is compared with the unsteered model and a random vector of equal size.
Progress is saved after each update, so a stopped run resumes.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import multiprocessing as mp
import os
import queue
import random
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from behaviors.rubrics import is_looping, score_response
from model_profiles import PROFILES, ModelProfile, get_profile, model_width
from rl import data as rl_data
from rl.api_judge import ApiJudge, Group
from rl.data import BEHAVIORS, read_split
from rl.policy import (
    NormalizedMomentum, ShrinkableAdam, backward_policy_loss, build_policy_batch, kl_divergence, prepare_layers,
    token_chunks, trim_completion,
)
from rl_math import group_advantages
from router.pipeline import PIPELINES
from rl.vllm_sampler import (
    MAX_MODEL_LEN, SWAP_MODES, CardTurns, NoTurns, VllmSampler, WeightShuttle, open_weight_views, prepare_environment,
)

ROOT = Path(__file__).parent.parent
EVAL_BATCH = 32
CHAT_TOKENS = 16
PROMPT_LENGTH_STEP = 32  # Fewer distinct shapes means fewer slow first-time GPU kernels.
LOSS_SCALE = 4096.0
PROBE_PROMPTS = 16
PROBE_TOKENS = 8
MAX_BROKEN_RISE = 0.10  # Roll back if this many more dev answers break than in unsteered Qwen.
MAX_GRAD_NORM = 1.0
MIN_JUDGED_SHARE = 0.5  # With fewer judged prompts, skip the update instead of learning from a thin sample.
MAX_SKIPPED_UPDATES = 3  # This many skipped updates in a row means the judge is down, so the run stops.
JUDGE_NEUTRAL_SCORE = 0.5  # Base score on the judged dev set: a tie against the unsteered answer.
START_GAP_SECONDS = 3
SEED = 2026
ENGINE_STOP_SECONDS = 10
CARD_VARS = ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
# vLLM used about 2x PyTorch's CPU in the retrain, so it gets the P-core threads and PyTorch the card's E-cores.
ENGINE_CPUS = {"main": [0, 1, 2, 3, 4, 5], "second": [6, 7, 8, 9, 10, 11]}


@dataclass
class Run:
    torch: Any
    model: Any
    tokenizer: Any
    eos: set
    pad: int
    args: Any
    card: str
    profile: ModelProfile
    sampler: Any = None  # A VllmSampler when --sampler vllm, else None (Hugging Face generation).
    turns: Any = None  # Who holds the card. Every GPU step asks for its turn first.
    judge: Any = None  # An ApiJudge when --judge haiku, else the regex checklist scores answers.
    judge_update: int = 0
    last_judge: Any = None


def say(run_card: str, **row) -> None:
    print(json.dumps({"t": time.strftime("%H:%M:%S"), "card": run_card, **row}), flush=True)


def chat_texts(tokenizer, texts: list[str]) -> list[str]:
    return [tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False) for text in texts]


def encode(run: Run, texts: list[str]):
    chats = chat_texts(run.tokenizer, texts)
    batch = run.tokenizer(chats, padding=True, pad_to_multiple_of=PROMPT_LENGTH_STEP, return_tensors="pt",
                          add_special_tokens=False)
    return batch["input_ids"].to("cuda:0"), batch["attention_mask"].to("cuda:0")


def generate(run: Run, ids, mask, sample: bool, copies: int = 1) -> list[list[int]]:
    settings = {"do_sample": True, "temperature": 1.0, "top_k": 0, "top_p": 1.0, "num_return_sequences": copies} \
        if sample else {"do_sample": False, "temperature": None, "top_k": None, "top_p": None}
    with run.torch.inference_mode():
        output = run.model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=run.args.max_new,
                                    repetition_penalty=1.0, pad_token_id=run.pad, eos_token_id=sorted(run.eos),
                                    use_cache=True, cache_implementation="static", **settings)
    return [trim_completion(row, run.eos) for row in output[:, ids.shape[1]:].tolist()]


def decode(run: Run, tokens: list[int]) -> str:
    return run.tokenizer.decode(tokens, skip_special_tokens=True).strip()


def layer_sizes(tensor, mask) -> tuple[float, float]:
    """Average and typical (median) token size. Qwen's first token is ~8,200, which inflates only the average."""
    sizes = tensor[mask.bool()].float().norm(dim=-1)
    return sizes.mean().item(), sizes.median().item()


def activation_norms(run: Run, rows: list[dict]):
    """Returns (average, typical) per layer: the average sets the step size, the typical size sets the cap."""
    torch, totals, handles, layers = run.torch, {}, [], run.profile.steering_layers
    run.turns.for_torch()
    ids, mask = encode(run, [row["prompt"] for row in rows[:EVAL_BATCH]])

    def capture(slot):
        def hook(_module, _inputs, output):
            totals[slot] = layer_sizes(output[0] if isinstance(output, tuple) else output, mask)
        return hook

    for slot, layer in enumerate(layers):
        handles.append(run.model.model.layers[layer].register_forward_hook(capture(slot)))
    with torch.inference_mode():
        run.model.model(input_ids=ids, attention_mask=mask, use_cache=False)
    for handle in handles:
        handle.remove()
    average, typical = zip(*(totals[slot] for slot in range(len(layers))))
    return torch.tensor(average, device="cuda:0"), torch.tensor(typical, device="cuda:0")


def free_gpu(torch) -> None:
    """A finished generation's KV cache sits in reference cycles, so collect it before emptying the cache."""
    gc.collect()
    torch.cuda.empty_cache()


def eval_answers_hf(run: Run, bank, choice, rows: list[dict]) -> list[list[int]]:
    run.turns.for_torch()
    answers = []
    size = run.profile.eval_batch
    for start in range(0, len(rows), size):
        ids, mask = encode(run, [row["prompt"] for row in rows[start:start + size]])
        with bank.choose(choice):
            answers += generate(run, ids, mask, sample=False)
        free_gpu(run.torch)
    return answers


def eval_answers(run: Run, bank, choice, rows: list[dict]) -> list[list[int]]:
    """One repeatable answer per row, from the same engine that writes the training answers."""
    if run.sampler is None:
        return eval_answers_hf(run, bank, choice, rows)
    run.turns.for_vllm()
    prompts = run.tokenizer(chat_texts(run.tokenizer, [row["prompt"] for row in rows]),
                            add_special_tokens=False)["input_ids"]
    return run.sampler.best_answers(prompts, None if choice is None else bank.vectors[choice])


def by_length(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: len(row["prompt"]))


def evaluate(run: Run, bank, choice, rows: list[dict], expert: str) -> dict:
    ordered = by_length(rows)
    return score_answers(run, ordered, eval_answers(run, bank, choice, ordered), expert)


def score_answers(run: Run, ordered: list[dict], answers: list[list[int]], expert: str) -> dict:
    results, samples, looping = [], [], 0
    for row, tokens in zip(ordered, answers):
        text = decode(run, tokens)
        results.append(score_response(expert, text, row["prompt"]))
        looping += is_looping(text)
        if len(samples) < 3:
            samples.append({"prompt": row["prompt"][:160], "answer": text[:700]})
    names = sorted({name for result in results for name in result.checks})
    return {"score": round(sum(r.score for r in results) / len(results), 4),
            "degenerate": round(sum(r.degenerate for r in results) / len(results), 4),
            "looping": round(looping / len(results), 4),
            "on_topic": round(sum(r.topic for r in results) / len(results), 4),
            "checks": {name: round(sum(bool(r.checks.get(name)) for r in results) / len(results), 3) for name in names},
            "samples": samples}


def generation_chunks(run: Run, rows: list[dict]) -> list[list[int]]:
    """Long prompts use more cache memory, so they go in smaller batches."""
    encoded = run.tokenizer([row["prompt"] for row in rows], add_special_tokens=False)["input_ids"]
    lengths = [len(tokens) + CHAT_TOKENS + run.args.max_new for tokens in encoded]
    return token_chunks(list(range(len(rows))), lengths, run.profile.cells(run.card) // run.args.group)


def add_group(run: Run, expert: str, row: dict, prompt: list[int], answers: list[list[int]], out: tuple) -> None:
    """Score one prompt's answers and append one entry per answer to the four rollout lists."""
    scores = [score_response(expert, decode(run, tokens), row["prompt"]).score for tokens in answers]
    for target, values in zip(out, ([prompt] * len(answers), answers, scores, group_advantages(scores))):
        target += values


def sample_rollouts_hf(run: Run, bank, rows: list[dict], expert: str) -> tuple:
    run.turns.for_torch()
    group, out = run.args.group, ([], [], [], [])
    for part in generation_chunks(run, rows):
        chunk = [rows[i] for i in part]
        ids, mask = encode(run, [row["prompt"] for row in chunk])
        with bank.choose(0):
            made = generate(run, ids, mask, sample=True, copies=group)
        free_gpu(run.torch)
        for index, row in enumerate(chunk):
            add_group(run, expert, row, ids[index][mask[index].bool()].tolist(),
                      made[index * group:(index + 1) * group], out)
    return out


def sample_rollouts_vllm(run: Run, bank, rows: list[dict], expert: str) -> tuple:
    """Push the current vector into vLLM first, so the answers come from the policy being trained."""
    run.turns.for_vllm()
    run.sampler.push(bank.vectors[0])
    prompts = run.tokenizer(chat_texts(run.tokenizer, [row["prompt"] for row in rows]),
                            add_special_tokens=False)["input_ids"]  # Same tokens the gradient pass sees.
    sampled = run.sampler.sample(prompts)
    if getattr(run, "judge", None) is not None:
        return judged_rollouts(run, rows, prompts, sampled, expert)
    out = ([], [], [], [])
    for row, prompt, answers in zip(rows, prompts, sampled):
        add_group(run, expert, row, prompt, answers, out)
    return out


def judged_rollouts(run: Run, rows: list[dict], prompts: list, sampled: list, expert: str) -> tuple:
    """Haiku scores each prompt's answers. A prompt with no usable verdict is left out of this update."""
    groups = [Group(f"{expert}:{run.judge_update}:{index}", expert, row["prompt"], tuple(decode(run, tokens) for tokens in answers))
              for index, (row, answers) in enumerate(zip(rows, sampled))]
    rewards, stats = run.judge.judge(groups, run.judge_update)
    run.last_judge = stats.summary()
    out = ([], [], [], [])
    for prompt, answers, reward in zip(prompts, sampled, rewards):
        if reward is not None:
            for target, values in zip(out, ([prompt] * len(answers), answers, reward, group_advantages(reward))):
                target += values
    return out


def judge_fields(run: Run) -> dict:
    stats = getattr(run, "last_judge", None)
    if stats is None:
        return {}
    return {"judge_s": stats["seconds"], "judge_ok": stats["ok"], "judge_retried": stats["retried"],
            "judge_skipped": stats["skipped"], "judge_cost": stats["cost_usd"], "judge_scored": stats["groups_scored"]}


def compare_with_judge(run: Run, ordered: list[dict], answer_sets: dict[str, list], expert: str, tag: str) -> dict | None:
    """Mean judge reward for each answer set, and how often it beats the unsteered answer."""
    names = list(answer_sets)
    groups = [Group(f"{expert}:{tag}:{index}", expert, row["prompt"], tuple(decode(run, answer_sets[name][index]) for name in names))
              for index, row in enumerate(ordered)]
    rewards, stats = run.judge.judge(groups, run.judge_update)
    usable = [reward for reward in rewards if reward is not None]
    say(run.card, expert=expert, judge_eval=tag, scored=len(usable), of=len(groups), retried=stats.retried,
        skipped=stats.skipped, cost=round(stats.cost_usd, 5), seconds=round(stats.seconds, 1))
    if len(usable) < MIN_JUDGED_SHARE * len(groups):
        return None
    out = {}
    for index, name in enumerate(names):
        wins = sum((reward[index] > reward[0]) + 0.5 * (reward[index] == reward[0]) for reward in usable)
        out[name] = {"reward": round(sum(reward[index] for reward in usable) / len(usable), 4),
                     "beats_base_pct": round(100 * wins / len(usable), 1)}
    return out


def judged_dev(run: Run, bank, ordered: list[dict], base_answers: list, expert: str, tag: str) -> dict:
    answers = eval_answers(run, bank, 0, ordered)
    result = score_answers(run, ordered, answers, expert)
    means = compare_with_judge(run, ordered, {"base": base_answers, "steered": answers}, expert, tag)
    return {**result, "checklist_score": result["score"], "score": means["steered"]["reward"] if means else 0.0, "judge": means}


def judged_final(run: Run, bank, rows: list[dict], expert: str) -> dict:
    ordered = by_length(rows)
    sets = {"base": eval_answers(run, bank, None, ordered), "steered": eval_answers(run, bank, 0, ordered),
            "random": random_answers(run, bank, ordered)}
    means = compare_with_judge(run, ordered, sets, expert, "final_test")
    if means is None:
        raise RuntimeError(f"{expert}: the judge scored too few final test prompts")
    final = {}
    for name, answers in sets.items():
        result = score_answers(run, ordered, answers, expert)
        final[name] = {**result, "checklist_score": result["score"], "score": means[name]["reward"],
                       "beats_base_pct": means[name]["beats_base_pct"]}
    return final


def sample_rollouts(run: Run, bank, rows: list[dict], expert: str) -> tuple:
    """Returns prompts, completions, rewards, advantages: one entry per answer, group answers per prompt."""
    sample = sample_rollouts_hf if run.sampler is None else sample_rollouts_vllm
    return sample(run, bank, rows, expert)


def accumulate_gradient(run: Run, bank, prompts, completions, advantages, norm: float) -> tuple[float, int]:
    """Scale up during backprop so tiny float16 gradients do not round to zero."""
    torch, profile = run.torch, run.profile
    run.turns.for_torch()
    active = [i for i, value in enumerate(advantages) if abs(value) > 1e-12]
    lengths = [len(prompt) + len(completion) for prompt, completion in zip(prompts, completions)]
    loss = 0.0
    with bank.choose(0):
        for part in token_chunks(active, lengths, profile.chunk_tokens(run.card, run.args.sampler)):
            batch = build_policy_batch(torch, [prompts[i] for i in part], [completions[i] for i in part],
                                       run.pad, "cuda:0")
            loss += backward_policy_loss(torch, run.model.model, run.model.lm_head, batch,
                                         torch.tensor([advantages[i] for i in part]), norm / LOSS_SCALE,
                                         profile.head_chunk)
    for parameter in bank.parameters():
        if parameter.grad is not None:
            parameter.grad /= LOSS_SCALE
    return loss / LOSS_SCALE, len(active)


def probe_log_probs(run: Run, bank, probe):
    """Word probabilities at the last prompt tokens, used to measure how far a step moved Qwen."""
    run.turns.for_torch()
    ids, mask = probe
    with bank.choose(0), run.torch.inference_mode():
        hidden = run.model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        return run.torch.log_softmax(run.model.lm_head(hidden[:, -PROBE_TOKENS:, :]).float(), dim=-1)


def make_optimizer(run: Run, vector, norms):
    args = run.args
    if args.optimizer == "normalized":
        return NormalizedMomentum(run.torch, vector, norms * args.step_fraction, args.momentum)
    return ShrinkableAdam(run.torch, vector, args.lr_fraction * float(norms.mean()) / math.sqrt(vector.shape[-1]))


def check_vram(run: Run) -> None:
    """Stop before the driver does: PyTorch's peak plus what other programs hold on this card."""
    limit = run.profile.hard_stop_gb
    used = run.torch.cuda.max_memory_reserved() / 1e9 + run.profile.other_vram_gb[run.card]
    if limit is not None and used > limit:
        raise RuntimeError(f"{run.card}: {used:.2f} GB is over the {limit} GB limit")


@dataclass
class WriteJob:
    rows: list[dict]
    prompts: list
    vector: Any  # The exact vector that writes this batch.
    future: Future


def write_answers(sampler, prompts: list, vector) -> tuple[list, float]:
    started = time.time()
    sampler.push(vector)
    return sampler.sample(prompts), time.time() - started


def start_write(run: Run, pool, bank, rows: list[dict], topic_vector=None) -> WriteJob:
    """Start vLLM on the next batch. Tokens and the vector are copied here, so the writer thread shares nothing."""
    prompts = run.tokenizer(chat_texts(run.tokenizer, [row["prompt"] for row in rows]), add_special_tokens=False)["input_ids"]
    vector = (bank.vectors[0] if topic_vector is None else topic_vector).detach().cpu().clone()
    return WriteJob(rows, prompts, vector, pool.submit(write_answers, run.sampler, prompts, vector))


@contextmanager
def at_vector(run: Run, bank, vector):
    """Compute the gradient at the vector that wrote the answers, then return to the current vector."""
    if vector is None:
        yield
        return
    current = bank.vectors[0].detach().clone()
    with run.torch.no_grad():
        bank.vectors[0].copy_(vector.to(current.device))
    try:
        yield
    finally:
        with run.torch.no_grad():
            bank.vectors[0].copy_(current)


def written_rollouts(run: Run, job: WriteJob, sampled: list, expert: str) -> tuple:
    if getattr(run, "judge", None) is not None:
        return judged_rollouts(run, job.rows, job.prompts, sampled, expert)
    out = ([], [], [], [])
    for row, prompt, answers in zip(job.rows, job.prompts, sampled):
        add_group(run, expert, row, prompt, answers, out)
    return out


def overlap_step(run: Run, pool, bank, optimizer, caps, job: WriteJob, next_rows: list[dict] | None, expert: str,
                 probe) -> tuple[dict, WriteJob | None]:
    """Collect this batch, start writing the next one, then judge and learn while vLLM writes."""
    swap_before = run.turns.swap_seconds
    run.torch.cuda.reset_peak_memory_stats()
    started = time.time()
    sampled, write_seconds = job.future.result()
    wait_seconds = time.time() - started
    next_job = start_write(run, pool, bank, next_rows) if next_rows is not None else None
    rollouts = written_rollouts(run, job, sampled, expert)
    info = learn_step(run, bank, optimizer, caps, len(job.rows), rollouts, probe, write_seconds, swap_before, job.vector)
    return {**info, "write_wait_s": round(wait_seconds, 2)}, next_job


def policy_step(run: Run, bank, optimizer, caps, rows: list[dict], expert: str, probe) -> dict:
    swap_before = run.turns.swap_seconds
    run.torch.cuda.reset_peak_memory_stats()
    started = time.time()
    rollouts = sample_rollouts(run, bank, rows, expert)
    generate_seconds = time.time() - started - judge_fields(run).get("judge_s", 0.0)
    return learn_step(run, bank, optimizer, caps, len(rows), rollouts, probe, generate_seconds, swap_before)


def learn_step(run: Run, bank, optimizer, caps, row_count: int, rollouts: tuple, probe, generate_seconds: float,
               swap_before: float, sample_vector=None) -> dict:
    torch = run.torch
    prompts, completions, rewards, advantages = rollouts
    judged = judge_fields(run)
    if getattr(run, "judge", None) is not None and len(rewards) < MIN_JUDGED_SHARE * row_count * run.args.group:
        return {"skipped_update": True, "generate_s": round(generate_seconds, 2), **judged}
    generate_peak = torch.cuda.max_memory_allocated() / 1e9
    free_gpu(torch)
    torch.cuda.reset_peak_memory_stats()
    run.turns.for_torch()  # Before the timer, so gradient_s and swap_s do not overlap.
    optimizer.zero_grad(set_to_none=True)
    norm = row_count * run.args.group * run.args.max_new
    started = time.time()
    with at_vector(run, bank, sample_vector):
        loss, active = accumulate_gradient(run, bank, prompts, completions, advantages, norm)
    gradient_seconds = time.time() - started
    check_vram(run)
    grad_norm, step_kl, rejected = 0.0, 0.0, False
    if active:
        grad_norm = float(torch.nn.utils.clip_grad_norm_(bank.parameters(), MAX_GRAD_NORM))
        previous = bank.vectors[0].detach().clone()
        before = probe_log_probs(run, bank, probe)
        optimizer.step()
        with torch.no_grad():
            lengths = bank.vectors[0].norm(dim=-1).clamp_min(1e-9)
            bank.vectors[0].mul_((caps / lengths).clamp(max=1.0).unsqueeze(-1))
        step_kl = kl_divergence(torch, before, probe_log_probs(run, bank, probe))
        if step_kl > run.args.max_kl:
            with torch.no_grad():
                bank.vectors[0].copy_(previous)
            optimizer.shrink()
            rejected = True
    return {"reward": round(sum(rewards) / len(rewards), 4), "best": round(max(rewards), 4), "active": active,
            "step_kl": round(step_kl, 5), "rejected": rejected, "step_scale": round(optimizer.scale, 4),
            **judged, "generate_s": round(generate_seconds, 2), "gradient_s": round(gradient_seconds, 2),
            "swap_s": round(run.turns.swap_seconds - swap_before, 2),
            "loss": round(loss, 6), "grad_norm": round(grad_norm, 4),
            "tokens": sum(len(c) for c in completions),
            "generate_peak_gb": round(generate_peak, 2),
            "gradient_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
            "vector_norm": [round(float(v), 3) for v in bank.vectors[0].detach().norm(dim=-1)]}


def random_answers(run: Run, bank, ordered: list[dict]) -> list[list[int]]:
    """Answers from a random vector of the same size as the trained one."""
    torch = run.torch
    saved = bank.vectors[0].detach().clone()
    generator = torch.Generator(device="cuda:0").manual_seed(SEED)
    noise = torch.randn(saved.shape, device="cuda:0", generator=generator, dtype=saved.dtype)
    with torch.no_grad():
        bank.vectors[0].copy_(noise / noise.norm(dim=-1, keepdim=True) * saved.norm(dim=-1, keepdim=True))
    answers = eval_answers(run, bank, 0, ordered)
    with torch.no_grad():
        bank.vectors[0].copy_(saved)
    return answers


def random_control(run: Run, bank, rows: list[dict], expert: str) -> dict:
    ordered = by_length(rows)
    return score_answers(run, ordered, random_answers(run, bank, ordered), expert)


def save_atomic(torch, state: dict, path: Path) -> None:
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def vector_state(run: Run, bank, optimizer, step: int, expert: str, **extra) -> dict:
    return {"bank": {"vectors.0": bank.vectors[0].detach().cpu()}, "optimizer": optimizer.state_dict(),
            "step": step, "expert": expert, "layers": list(run.profile.steering_layers),
            "profile": run.profile.name, **extra}


def check_resume(state: dict, profile: ModelProfile, path: Path) -> None:
    """A vector only works on the model and layers it was trained on."""
    if tuple(state["layers"]) != profile.steering_layers or state.get("profile", "qwen06b") != profile.name:
        raise ValueError(f"{path} was trained for {state.get('profile', 'qwen06b')} layers {state['layers']}, "
                         f"not {profile.name} layers {list(profile.steering_layers)}. Use a new --run or --fresh.")


def one_pass_updates(train_rows: int, prompts_per_update: int) -> int:
    return math.ceil(train_rows / prompts_per_update)


def finished_best_step(state: dict, updates: int) -> int | None:
    """A file that finished training already holds its best vector. Keep that update's number on resume."""
    return state.get("best_step") if state["step"] >= updates else None


def save_best(run: Run, bank, best_vector, optimizer, step: int, best_step: int, expert: str, path: Path) -> None:
    """The finished file must hold the best dev vector, because inference loads this file."""
    with run.torch.no_grad():
        bank.vectors[0].copy_(best_vector)
    save_atomic(run.torch, vector_state(run, bank, optimizer, step, expert, best_step=best_step), path)


def record_dev(run: Run, bank, optimizer, history: list, result: dict, vector, step: int, base: dict,
               expert: str, best: tuple) -> tuple:
    """Log one dev score for `vector`. Returns the best (vector, score, step), rolling back if answers broke."""
    history.append({"step": step, "dev": result["score"], "degenerate": result["degenerate"],
                    "looping": result.get("looping"), "on_topic": result["on_topic"]})
    say(run.card, expert=expert, step=step, dev_steered=result["score"], dev_base=base["score"],
        degenerate=result["degenerate"], looping=result.get("looping"), on_topic=result["on_topic"])
    # With Haiku judging quality, the brake only catches output that is plainly broken.
    brake = "looping" if getattr(run, "judge", None) is not None else "degenerate"
    if result[brake] > base[brake] + MAX_BROKEN_RISE:
        with run.torch.no_grad():
            bank.vectors[0].copy_(best[0])
        optimizer.shrink()
        say(run.card, expert=expert, step=step, rolled_back_to=best[2], step_scale=optimizer.scale)
        return best
    return (vector, result["score"], step) if result["score"] > best[1] else best


def fitting_rows(run: Run, rows: list[dict]) -> list[dict]:
    """Keep prompts that leave room for a full answer inside the model length limit."""
    room = MAX_MODEL_LEN - run.args.max_new
    encoded = run.tokenizer(chat_texts(run.tokenizer, [row["prompt"] for row in rows]),
                            add_special_tokens=False)["input_ids"]
    return [row for row, tokens in zip(rows, encoded) if len(tokens) <= room]


def train_expert(run: Run, expert: str) -> dict:
    from steering import SteeringBank

    torch, args = run.torch, run.args
    folder = ROOT / "checkpoints" / "rl_pilot" / args.run
    report_dir = rl_data.OUT / args.run
    folder.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    loaded = [read_split(expert, name) for name in ("routed", "dev", "test")]
    train, dev, test = (fitting_rows(run, rows) for rows in loaded)
    say(run.card, expert=expert, too_long_dropped={name: len(before) - len(after) for name, before, after in
                                                   zip(("train", "dev", "test"), loaded, (train, dev, test))})
    random.Random(SEED).shuffle(train)
    dev, test = by_length(dev[:args.dev]), test[:args.test]
    updates = one_pass_updates(len(train), args.prompts) if args.one_pass else args.updates
    bank = SteeringBank([expert], run.profile.steering_layers, run.model.config.hidden_size).to("cuda:0")
    bank.attach(run.model)
    norms, typical = activation_norms(run, train)
    caps = typical * args.cap_fraction
    optimizer = make_optimizer(run, bank.vectors[0], norms)
    path, step, history, saved_best = folder / f"{expert}.pt", 0, [], None
    if path.exists() and not args.fresh:
        state = torch.load(path, map_location="cuda:0", weights_only=True)
        check_resume(state, run.profile, path)
        bank.load_state_dict(state["bank"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]
        saved_best = finished_best_step(state, updates)
    say(run.card, expert=expert, start_step=step, updates=updates, judge=args.judge, optimizer=args.optimizer,
        sampler=args.sampler, eval_engine=args.sampler,
        activation_norms=[round(float(v), 2) for v in norms], typical_norms=[round(float(v), 2) for v in typical],
        caps=[round(float(v), 2) for v in caps],
        prompts={"train": len(train), "dev": len(dev), "test": len(test)})
    base_answers = eval_answers(run, bank, None, dev)
    base_dev = score_answers(run, dev, base_answers, expert)
    if run.judge is not None:
        base_dev = {**base_dev, "checklist_score": base_dev["score"], "score": JUDGE_NEUTRAL_SCORE}
    say(run.card, expert=expert, dev_base=base_dev["score"], degenerate=base_dev["degenerate"])

    def dev_result(tag: str) -> dict:
        if run.judge is None:
            return evaluate(run, bank, 0, dev, expert)
        return judged_dev(run, bank, dev, base_answers, expert, tag)

    probe = encode(run, [row["prompt"] for row in train[:PROBE_PROMPTS]])
    best_vector, best_score, best_step = bank.vectors[0].detach().clone(), base_dev["score"], step
    if saved_best is not None:
        run.judge_update = step
        best_step, best_score = saved_best, dev_result(f"dev_resume_{step}")["score"]
    started = time.time()
    best, skipped_in_row, pending = (best_vector, best_score, best_step), 0, None
    rows_at = lambda index: [train[(index * args.prompts + i) % len(train)] for i in range(args.prompts)]
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vllm-writer") if args.overlap else None
    try:
        while step < updates:
            run.judge_update = step
            if pool is None:
                info = policy_step(run, bank, optimizer, caps, rows_at(step), expert, probe)
            else:
                job = pending or start_write(run, pool, bank, rows_at(step))
                # Scoring also uses vLLM, so no new batch is started right before a dev check.
                write_next = step + 1 < updates and (step + 1) % args.eval_every != 0
                info, pending = overlap_step(run, pool, bank, optimizer, caps, job, rows_at(step + 1) if write_next else None,
                                             expert, probe)
            skipped_in_row = skipped_in_row + 1 if info.get("skipped_update") else 0
            if skipped_in_row >= MAX_SKIPPED_UPDATES:
                raise RuntimeError(f"{expert}: {skipped_in_row} updates in a row had too few judged prompts")
            step += 1
            save_atomic(torch, vector_state(run, bank, optimizer, step, expert), path)
            say(run.card, expert=expert, step=step, of=updates, **info, seconds=round(time.time() - started, 1))
            if step % args.eval_every == 0 or step == updates:
                result = dev_result(f"dev_{step}")
                best = record_dev(run, bank, optimizer, history, result, bank.vectors[0].detach().clone(), step,
                                  base_dev, expert, best)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    best_vector, best_score, best_step = best
    save_best(run, bank, best_vector, optimizer, step, best_step, expert, path)
    run.judge_update = step
    if run.judge is not None:
        final = judged_final(run, bank, test, expert)
    else:
        final = {"base": evaluate(run, bank, None, test, expert), "steered": evaluate(run, bank, 0, test, expert),
                 "random": random_control(run, bank, test, expert)}
    report = {"expert": expert, "updates": step, "dev_base": base_dev["score"], "dev_history": history,
              "best_dev": best_score, "best_step": best_step,
              "test_prompts": len(test), "test": final, "settings": {k: v for k, v in vars(args).items()}}
    (report_dir / f"{expert}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    say(run.card, expert=expert, finished=True, test_base=final["base"]["score"],
        test_steered=final["steered"]["score"], test_random=final["random"]["score"],
        steered_degenerate=final["steered"]["degenerate"], steered_on_topic=final["steered"]["on_topic"])
    bank.detach()
    return report


def learner_cpus(name: str) -> list[int]:
    """The card's CPU threads that vLLM does not use."""
    spec = PIPELINES[name]
    return sorted(set(spec["build"] + spec["gpu"]) - set(ENGINE_CPUS[name]))


def pin_process(cpus: list[int]) -> None:
    """Move every thread of this process. sched_setaffinity(0) alone moves only the calling thread."""
    for thread in os.listdir("/proc/self/task"):
        try:
            os.sched_setaffinity(int(thread), cpus)
        except ProcessLookupError:
            pass  # The thread ended between listing and pinning.


def start_sampler(name: str, args, eos: set, torch, profile: ModelProfile):
    """vLLM inherits its card and CPUs when it starts; this process then moves to the learner CPUs."""
    separate = args.vllm_swap == "separate"
    engine_card = args.engine_card if separate else name
    cpus = sorted(PIPELINES[name]["build"] + PIPELINES[name]["gpu"]) if separate else learner_cpus(name)

    def on_engine_cpus(make):
        pin_process(ENGINE_CPUS[engine_card])
        saved = {key: os.environ.get(key) for key in CARD_VARS}
        os.environ.update(dict.fromkeys(CARD_VARS, PIPELINES[engine_card]["card"]))
        try:
            return make()
        finally:
            for key, value in saved.items():
                os.environ.pop(key, None) if value is None else os.environ.__setitem__(key, value)
            pin_process(cpus)
            torch.set_num_threads(len(cpus))

    total = torch.cuda.get_device_properties(0).total_memory if profile.vllm_memory_gb else None
    return VllmSampler.start(profile.path, args.group, args.max_new, eos, args.kv_cache_dtype, profile=profile,
                             card=engine_card, total_bytes=total, swap=args.vllm_swap, around_build=on_engine_cpus)


def limit_torch_vram(torch, profile: ModelProfile, card: str) -> None:
    """Keep PyTorch's cache under this card's budget, so it reuses freed blocks instead of growing."""
    limit = profile.torch_vram_gb.get(card)
    if limit:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, limit * 1e9 / total), 0)


def load_learner(torch, args, profile: ModelProfile, sampler):
    """Returns the frozen PyTorch model and who holds the card."""
    from transformers import AutoModelForCausalLM

    take_turns = profile.time_share and sampler is not None and args.vllm_swap != "separate"
    if take_turns and args.dtype != "float16":
        raise ValueError("Turn-taking needs float16; float32 weights would not leave room for gradients")
    model = AutoModelForCausalLM.from_pretrained(profile.path, dtype=getattr(torch, args.dtype)).eval()
    for weight in model.parameters():
        weight.requires_grad_(False)
    turns = NoTurns()
    if take_turns:
        shuttle = WeightShuttle(model, open_weight_views(profile.path), torch.float16, "cuda:0")
        shuttle.to_cpu()  # Drops the private RAM copy that from_pretrained made.
        turns = CardTurns(torch, sampler, shuttle)
    else:
        model = model.to("cuda:0")
    prepare_layers(torch, model.model.layers, profile.steering_layers[0], args.compile, profile.gradient_checkpointing)
    return model, turns


def close_worker() -> None:
    """vLLM's engine keeps this process and its VRAM alive after training, so end both explicitly."""
    for child in mp.active_children():
        child.terminate()
        child.join(ENGINE_STOP_SECONDS)
        if child.is_alive():
            child.kill()
    sys.stdout.flush()
    os._exit(0)


def worker(name: str, experts, args, ready) -> None:
    spec = PIPELINES[name]
    os.sched_setaffinity(0, spec["build"] + spec["gpu"])
    os.environ.update(HIP_VISIBLE_DEVICES=spec["card"], CUDA_VISIBLE_DEVICES=spec["card"], HF_HUB_OFFLINE="1",
                      OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="true")
    if args.sampler == "vllm":
        prepare_environment()
    import torch
    from transformers import AutoTokenizer

    profile = get_profile(args.profile)
    model_width(profile)  # Stops here if the folder is not this profile's model.
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Worker must see exactly one card")
    bus = getattr(torch.cuda.get_device_properties(0), "pci_bus_id", None)
    if bus is not None and bus != spec["bus"]:
        raise RuntimeError(f"Wrong card: bus {bus}, expected {spec['bus']}")
    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(profile.path, padding_side="left")
    eos = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|endoftext|>")}
    # vLLM goes first so it takes its share of the card before the PyTorch model and gradient batches.
    sampler = start_sampler(name, args, eos, torch, profile) if args.sampler == "vllm" else None
    limit_torch_vram(torch, profile, name)
    model, turns = load_learner(torch, args, profile, sampler)
    judge = None
    if args.judge == "haiku":
        judge = ApiJudge(rl_data.OUT / args.run / "judge_calls.jsonl", run=args.run, budget_usd=args.judge_budget, seed=SEED)
    run = Run(torch, model, tokenizer, eos, tokenizer.pad_token_id, args, name, profile, sampler, turns, judge)
    ready.set()
    if args.interleave:
        from rl.interleave import train_mixed

        topics = []
        while (expert := next_expert(experts)) is not None:
            topics.append(expert)
        say(name, mixed_topics=topics)
        train_mixed(run, topics)
    while (expert := next_expert(experts)) is not None:
        say(name, took_next_behavior=expert)
        train_expert(run, expert)
    say(name, peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 1e9, 2), all_done=True)
    if sampler is not None:
        close_worker()


def next_expert(experts) -> str | None:
    """A free card takes the next behavior. Each behavior goes to exactly one card."""
    try:
        return experts.get(timeout=2)
    except queue.Empty:
        return None


def parse_list(text: str, allowed) -> list[str]:
    items = [item.strip() for item in text.split(",") if item.strip()]
    unknown = [item for item in items if item not in allowed]
    if unknown or not items or len(items) != len(set(items)):
        raise ValueError(f"Bad list: {text}")
    return items


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="pilot")
    parser.add_argument("--profile", choices=sorted(PROFILES), default=None,
                        help="Base model. Default: $STEERING_PROFILE, else qwen06b.")
    parser.add_argument("--vllm-swap", choices=SWAP_MODES, default="sleep",
                        help="How vLLM leaves the card on turn-taking models. Use restart if sleep fails on this driver.")
    parser.add_argument("--cards", default="main,second")
    # Measured 2026-11-25 on 4B: the second card has no desktop, so vLLM gets a 25,088-token cache (main: 9,408)
    # and writes 32 answers in 11.4 s instead of 17.6 s. The learner fits on main at 13.4 GB with the desktop.
    parser.add_argument("--engine-card", choices=list(PIPELINES), default="second",
                        help="With --vllm-swap separate: the card vLLM writes on. --cards names the learner's card.")
    parser.add_argument("--experts", default=",".join(BEHAVIORS))
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--sampler", choices=["hf", "vllm"], default="hf",
                        help="Who writes the training and scoring answers.")
    parser.add_argument("--kv-cache-dtype", default="auto", help="vLLM cache type; fp8 drifted from PyTorch in tests.")
    parser.add_argument("--updates", type=int, default=40)
    parser.add_argument("--prompts", type=int, default=16)
    parser.add_argument("--group", type=int, default=4)
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--dev", type=int, default=32)
    parser.add_argument("--test", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--optimizer", choices=["normalized", "adam"], default="normalized")
    parser.add_argument("--step-fraction", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--max-kl", type=float, default=0.01)
    parser.add_argument("--lr-fraction", type=float, default=0.001)
    parser.add_argument("--cap-fraction", type=float, default=0.50, help="Largest vector, as a share of a typical token.")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--one-pass", action="store_true", help="Run enough updates to see every training prompt once.")
    parser.add_argument("--overlap", action="store_true", help="Write the next batch while judging and learning this one.")
    parser.add_argument("--interleave", action="store_true", help="Shuffle small updates across all topics in one run.")
    parser.add_argument("--judge", choices=["checklist", "haiku"], default="checklist", help="Who scores the answers.")
    parser.add_argument("--judge-budget", type=float, default=50.0, help="Hard stop for this run's Haiku spend, in USD.")
    return parser


def check_separate_cards(args) -> None:
    """In separate mode one learner drives the vLLM engine on the other card."""
    engine = getattr(args, "engine_card", "second")
    if args.vllm_swap == "separate" and (args.sampler != "vllm" or args.cards not in PIPELINES or args.cards == engine):
        raise ValueError("--vllm-swap separate needs --sampler vllm and one learner card (--cards) other than --engine-card")
    if getattr(args, "interleave", False) and (not args.overlap or args.judge != "haiku"):
        raise ValueError("--interleave needs --overlap and --judge haiku")
    if getattr(args, "overlap", False) and args.vllm_swap != "separate":
        raise ValueError("--overlap needs vLLM and the learner on separate cards (--vllm-swap separate)")
    if getattr(args, "judge", "checklist") == "haiku" and (args.sampler != "vllm" or args.group > 4):
        raise ValueError("--judge haiku needs --sampler vllm and at most 4 answers per prompt")


def main() -> int:
    args = make_parser().parse_args()
    check_separate_cards(args)
    context = mp.get_context("spawn")
    processes = []
    experts = context.Queue()
    for expert in parse_list(args.experts, BEHAVIORS):
        experts.put(expert)
    time.sleep(0.2)
    for name in parse_list(args.cards, PIPELINES):
        if processes and not processes[-1][1].wait(240):
            print(json.dumps({"FAILED": "previous card did not finish loading"}), flush=True)
            break
        if processes:
            time.sleep(START_GAP_SECONDS)
        ready = context.Event()
        process = context.Process(target=worker, args=(name, experts, args, ready), name=name)
        process.start()
        processes.append((process, ready))
    for process, _ in processes:
        process.join()
    codes = {process.name: process.exitcode for process, _ in processes}
    print(json.dumps({"exit_codes": codes}), flush=True)
    return max((code or 0) != 0 for code in codes.values())


if __name__ == "__main__":
    sys.exit(main())
