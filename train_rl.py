"""Train ONLY in-model steering vectors with group-relative policy gradients.

This is a practical one-update-per-rollout group-relative REINFORCE prototype.
It is NOT full GRPO/PPO: no off-policy ratio clipping, no reference-model KL.
The router is trained using the known task label; vectors are trained by RL.
Run on a GPU with enough VRAM for backprop through the frozen LLM.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from rl_math import group_advantages, reward_integer_answer
from steering import count_parameters, install_steering, resolve_layer_indices, router_features


EXPERT_NAMES = ("addition", "subtraction")


def format_prompt(tokenizer, task: int, a: int, b: int) -> str:
    sign = "+" if task == 0 else "-"
    messages = [
        {"role": "system", "content": "You are a calculator. Reply with exactly one signed integer and nothing else."},
        {"role": "user", "content": f"Compute {a} {sign} {b}. Output the integer only."},
    ]
    if tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,  # Qwen3: prevent hidden reasoning text in reward
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
    return f"Calculator. Reply with only an integer.\nCompute {a} {sign} {b}.\nAnswer:"


@torch.no_grad()
def sample_completion(model, prompt_ids, eos_token_id: int | None, max_new_tokens: int):
    """Unfiltered temperature-1 on-policy sampling, with no stale KV cache."""
    context = prompt_ids
    generated = []
    for _ in range(max_new_tokens):
        logits = model(input_ids=context, use_cache=False).logits[:, -1, :].float()
        probabilities = F.softmax(logits, dim=-1)
        token = torch.multinomial(probabilities, num_samples=1)
        generated.append(token)
        context = torch.cat((context, token), dim=1)
        if eos_token_id is not None and int(token[0, 0]) == eos_token_id:
            break
    return torch.cat(generated, dim=1)


def completion_logprob(model, prompt_ids, completion_ids):
    """Exact differentiable log π(completion|prompt), including sampled EOS."""
    full = torch.cat((prompt_ids, completion_ids), dim=1)
    logits = model(input_ids=full[:, :-1], use_cache=False).logits.float()
    prefix_len = prompt_ids.size(1)
    # The logit predicting the first sampled token is at position prefix_len-1.
    scores = logits[:, prefix_len - 1 :, :]
    log_probs = F.log_softmax(scores, dim=-1)
    token_scores = log_probs.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)
    assert token_scores.size(1) == completion_ids.size(1)
    return token_scores.sum()


def load_base(model_id: str):
    # float32 on purpose: bf16 rounds away small steering updates.
    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model.to(device)
    model.eval()  # Disable dropout, but do NOT disable autograd for scoring.
    return tokenizer, model, device


def train(args):
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    tokenizer, model, device = load_base(args.model)
    target_layers = resolve_layer_indices(model, args.layers)
    bank, router = install_steering(model, EXPERT_NAMES, target_layers)
    stats = count_parameters(model, bank, router)
    print("Model parameter counts:", stats, flush=True)
    assert all(not p.requires_grad for name, p in model.named_parameters()
               if not name.startswith(("steering_bank.", "steering_router.")))

    optimizer = torch.optim.AdamW(
        [
            {"params": list(bank.parameters()), "lr": args.vector_lr},
            {"params": list(router.parameters()), "lr": args.router_lr},
        ], weight_decay=0.0,
    )

    for step in range(1, args.steps + 1):
        task = (step - 1) % len(EXPERT_NAMES)
        a = rng.randint(3, 80)
        b = rng.randint(3, 80)
        target = a + b if task == 0 else a - b
        prompt = format_prompt(tokenizer, task, a, b)
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        features = router_features(model, ids)

        # BOOTSTRAP: during training the skill label explicitly chooses the
        # expert. The router learns that label by supervised cross-entropy.
        # The vectors are NOT supervised: they learn only from rollout rewards.
        selected_expert = task
        completions = []
        replies = []
        rewards = []
        with bank.choose(selected_expert):
            for _ in range(args.group_size):
                completion = sample_completion(
                    model, ids, tokenizer.eos_token_id, args.max_new_tokens
                )
                answer = tokenizer.decode(completion[0].tolist(), skip_special_tokens=True)
                completions.append(completion)
                replies.append(answer)
                rewards.append(reward_integer_answer(answer, target))

        advantages = group_advantages(rewards)
        optimizer.zero_grad(set_to_none=True)
        policy_loss_number = 0.0

        # One backward per rollout keeps memory modest. Weights stay fixed;
        # gradients flow THROUGH frozen blocks to the selected steering vector.
        for completion, advantage in zip(completions, advantages):
            if abs(advantage) < 1e-12:
                continue  # All equal rewards => no RL signal for this group.
            with bank.choose(selected_expert):
                log_p = completion_logprob(model, ids, completion)
                loss = -(advantage * log_p) / (args.group_size * args.max_new_tokens)
                loss.backward()
                policy_loss_number += float(loss.detach())

        # Train the small prompt router, keeping embeddings and base frozen.
        route_logits = router(features)
        route_target = torch.tensor([task], device=device)
        router_loss = F.cross_entropy(route_logits, route_target)
        (args.router_ce_weight * router_loss).backward()

        # Regularize only the ACTIVE expert; other experts have grad=None.
        if args.vector_l2 > 0:
            (args.vector_l2 * bank.vectors[selected_expert].square().mean()).backward()

        torch.nn.utils.clip_grad_norm_(
            list(bank.parameters()) + list(router.parameters()), args.max_grad_norm
        )
        optimizer.step()

        # Sanity: unchosen experts are never used in the graph for this step.
        if step <= 2:
            untouched = [e for e in range(len(EXPERT_NAMES)) if e != selected_expert]
            assert all(bank.vectors[e].grad is None for e in untouched)

        if step == 1 or step % args.log_every == 0:
            print(json.dumps({
                "step": step, "task": EXPERT_NAMES[task], "target": target,
                "replies": replies[:2], "rewards": [round(r, 3) for r in rewards],
                "policy_loss": round(policy_loss_number, 5),
                "router_loss": round(float(router_loss.detach()), 5),
                "router_choice": int(route_logits.argmax(dim=-1).item()),
                "selected_vector_norm": round(float(bank.vectors[task].norm().detach()), 5),
            }), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_id": args.model,
        "expert_names": list(bank.expert_names),
        "target_layers": list(bank.target_layers),
        "bank_state_dict": {k: v.detach().cpu() for k, v in bank.state_dict().items()},
        "router_state_dict": {k: v.detach().cpu() for k, v in router.state_dict().items()},
        "steps": args.steps,
        "note": "Only steering and router were saved; pretrained base remains external and frozen",
    }, output)
    print(f"Saved steering experts + router to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--layers", default="auto", help="auto or indices such as 7,14,21")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--vector-lr", type=float, default=0.001)
    parser.add_argument("--router-lr", type=float, default=0.0001)
    parser.add_argument("--router-ce-weight", type=float, default=0.1)
    parser.add_argument("--vector-l2", type=float, default=0.0001)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output", default="checkpoints/steering_bank.pt")
    parameters = parser.parse_args()
    if parameters.group_size < 2:
        parser.error("--group-size must be at least 2")
    train(parameters)
