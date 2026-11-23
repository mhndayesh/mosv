"""Held-out evaluation: compare base vs routed steering on unseen arithmetic."""

from __future__ import annotations

import argparse
import random

import torch

from rl_math import reward_integer_answer
from steering import install_steering, router_features
from train_rl import format_prompt, load_base


@torch.no_grad()
def respond(model, tokenizer, prompt_ids, bank, expert, max_new_tokens):
    with bank.choose(expert):
        generated = model.generate(
            input_ids=prompt_ids,
            do_sample=False,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(generated[0, prompt_ids.size(1):], skip_special_tokens=True)


def evaluate(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tokenizer, model, device = load_base(checkpoint["model_id"])
    bank, router = install_steering(model, checkpoint["expert_names"], checkpoint["target_layers"])
    bank.load_state_dict(checkpoint["bank_state_dict"])
    router.load_state_dict(checkpoint["router_state_dict"])
    rng = random.Random(args.seed)
    counters = {"base_correct": 0, "steered_correct": 0, "router_correct": 0}

    for index in range(args.samples):
        task = index % 2
        a, b = rng.randint(3, 80), rng.randint(3, 80)
        answer = a + b if task == 0 else a - b
        ids = tokenizer(
            format_prompt(tokenizer, task, a, b),
            return_tensors="pt", add_special_tokens=False,
        ).input_ids.to(device)
        with torch.no_grad():
            chosen = int(router(router_features(model, ids)).argmax(dim=-1).item())
        base = respond(model, tokenizer, ids, bank, None, args.max_new_tokens)
        steered = respond(model, tokenizer, ids, bank, chosen, args.max_new_tokens)
        counters["base_correct"] += int(reward_integer_answer(base, answer) == 1.0)
        counters["steered_correct"] += int(reward_integer_answer(steered, answer) == 1.0)
        counters["router_correct"] += int(chosen == task)
        print(f"{index+1}/{args.samples}: target={answer}, route={chosen}, base={base!r}, steered={steered!r}")
    print("Held-out exact accuracy:", {
        key: round(value / args.samples, 4) for key, value in counters.items()
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/steering_bank.pt")
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--seed", type=int, default=98221)
    evaluate(ap.parse_args())
