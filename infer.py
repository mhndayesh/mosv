"""Load the saved in-model steering experts/router and compare against base."""

from __future__ import annotations

import argparse

import torch

from steering import install_steering, router_features
from train_rl import load_base, format_prompt


@torch.no_grad()
def greedy_reply(model, tokenizer, ids, bank, choice, max_new_tokens):
    with bank.choose(choice):
        out = model.generate(
            input_ids=ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, ids.size(1):], skip_special_tokens=True).strip()


def main(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tokenizer, model, device = load_base(checkpoint["model_id"])
    bank, router = install_steering(
        model, checkpoint["expert_names"], checkpoint["target_layers"]
    )
    bank.load_state_dict(checkpoint["bank_state_dict"])
    router.load_state_dict(checkpoint["router_state_dict"])
    model.eval()

    prompt = format_prompt(tokenizer, args.task, args.a, args.b)
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    with torch.no_grad():
        routing_logits = router(router_features(model, ids))
        probabilities = routing_logits.softmax(dim=-1)[0]
    selected = int(torch.argmax(probabilities).item())
    if args.force_expert is not None:
        selected = args.force_expert

    print("Router probabilities:", {
        name: round(float(probabilities[i]), 4)
        for i, name in enumerate(bank.expert_names)
    })
    print("Selected:", bank.expert_names[selected])
    print("Without steering:", greedy_reply(model, tokenizer, ids, bank, None, args.max_new_tokens))
    print("With steering:", greedy_reply(model, tokenizer, ids, bank, selected, args.max_new_tokens))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/steering_bank.pt")
    parser.add_argument("--task", type=int, choices=[0, 1], default=0)
    parser.add_argument("--a", type=int, default=32)
    parser.add_argument("--b", type=int, default=17)
    parser.add_argument("--force-expert", type=int, choices=[0, 1])
    parser.add_argument("--max-new-tokens", type=int, default=16)
    main(parser.parse_args())
