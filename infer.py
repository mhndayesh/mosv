"""Answer one prompt: show the route, the steered answer and the unsteered answer.

Run: python infer.py "Write a short story about a fox" --run pilot_final
"""

from __future__ import annotations

import argparse

from system import MAX_NEW_TOKENS, NONE, load_system


def main(args) -> None:
    system = load_system(args.model, args.run, max_new_tokens=args.max_new_tokens)
    route = system.route([args.prompt])[0]
    chosen = args.force or route.name  # --force overrides the router's choice.
    unsteered = system.generate_fixed([args.prompt], None, args.max_new_tokens)[0]
    steered = unsteered if chosen == NONE else system.generate_fixed([args.prompt], chosen, args.max_new_tokens)[0]
    print(f"Route: {route.name} (confidence {route.confidence:.3f})" + (f", forced to {chosen}" if args.force else ""))
    print(f"\n--- Steered ({chosen}) ---\n{steered}")
    print(f"\n--- Unsteered ---\n{unsteered}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prompt")
    ap.add_argument("--run", default="pilot_final", help="folder under checkpoints/rl_pilot with the vectors")
    ap.add_argument("--model", default=None, help="Qwen3-0.6B folder (else $QWEN_MODEL_PATH, the WSL path, or the hub cache)")
    ap.add_argument("--force", choices=["engineering", "story", "teaching"], help="override the router's choice")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    main(ap.parse_args())
