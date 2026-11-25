"""Log what happens inside the model, token by token.

Run in WSL: python model_trace.py --run qwen4b_haiku_v3_test
For each prompt: the router's probabilities, then for every generated token what each steering layer
added, which trained vector that matches, and the word produced. Details go to data_review/traces/<run>.jsonl.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch

from model_profiles import PROFILES
from system import NONE, System, load_system

TRACE_DIR = Path(__file__).parent / "data_review" / "traces"
ZERO_NORM = 1e-3  # Below this, the layer added nothing.
MAX_NEW_TOKENS = 256
PROMPTS = (
    "Build a small command line todo app the right way.",
    "Set up a clean project for a weather dashboard.",
    "Write a story about a lighthouse keeper who finds a lost whale.",
    "Tell me a story about a young baker who loses her shop.",
    "Teach a second grader what fractions are.",
    "Give a lesson for young students on the water cycle.",
    "What is the capital of Japan?",
    "Translate 'good morning' into French.",
    "Write a Python function that reverses a string.",
    "Tell me a joke about cats.",
)


def tensor_of(output) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def match_vector(names, vectors: torch.Tensor, added: torch.Tensor) -> tuple[str, float]:
    """Which trained vector was added. vectors is experts x width for one layer."""
    if added.norm() < ZERO_NORM:
        return NONE, 1.0
    similarity = torch.nn.functional.cosine_similarity(vectors, added[None], dim=1)
    best = int(similarity.argmax())
    return names[best], round(float(similarity[best]), 4)


class ModelTrace:
    """Hooks around each steering layer, plus the output head, for one single-prompt generation."""

    def __init__(self, system: System):
        self.system, self.raw, self.layers, self.steps, self.handles = system, {}, {}, [], []

    def __enter__(self):
        model = self.system.model
        for slot, index in enumerate(self.system.bank.target_layers):
            layer = model.model.layers[index]
            self.handles.append(layer.register_forward_hook(partial(self._before, slot), prepend=True))
            self.handles.append(layer.register_forward_hook(partial(self._after, slot, index)))
        self.handles.append(model.lm_head.register_forward_hook(self._token))
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()

    def _before(self, slot, _module, _inputs, output) -> None:
        self.raw[slot] = tensor_of(output)[:, -1].detach().float()

    def _after(self, slot, index, _module, _inputs, output) -> None:
        raw = self.raw[slot][0]
        added = tensor_of(output)[0, -1].detach().float() - raw
        bank = self.system.bank
        vectors = torch.stack([v[slot].detach().float() for v in bank.vectors]).to(added.device)
        name, match = match_vector(bank.expert_names, vectors, added)
        self.layers[str(index)] = {"activation": round(float(raw.norm()), 2), "added": round(float(added.norm()), 3),
                                   "vector": name, "match": match}

    def _token(self, _module, _inputs, output) -> None:
        probabilities = output[0, -1].float().softmax(-1)
        token = int(probabilities.argmax())
        self.steps.append({"step": len(self.steps) + 1, "token": token,
                           "word": self.system.tokenizer.decode([token]),
                           "probability": round(float(probabilities[token]), 4), "layers": self.layers})
        self.layers = {}


def summarize(steps: list[dict]) -> dict:
    """Per layer: which vectors fired across the answer, and how big the push was."""
    out = {}
    for index in (steps[0]["layers"] if steps else {}):
        rows = [step["layers"][index] for step in steps]
        out[index] = {"vectors": sorted({row["vector"] for row in rows}),
                      "push_percent": round(100 * sum(r["added"] / max(r["activation"], 1e-6) for r in rows) / len(rows), 2),
                      "lowest_match": min(row["match"] for row in rows)}
    return out


def trace_prompt(system: System, prompt: str, max_new: int) -> dict:
    probabilities = system.router.probabilities(system._read([prompt]))[0]
    route = system.route([prompt])[0]
    with ModelTrace(system) as trace:
        answer = system._generate_batch([prompt], None if route.name == NONE else route.name, max_new)[0]
    tokens = [step["token"] for step in trace.steps]
    return {"prompt": prompt, "route": route.name, "confidence": round(route.confidence, 4),
            "router": {c: round(float(p), 4) for c, p in zip(system.router.classes, probabilities)},
            "tokens": len(tokens), "layers": summarize(trace.steps), "answer": answer,
            "trace_matches_answer": system.tokenizer.decode(tokens, skip_special_tokens=True).strip() == answer,
            "steps": trace.steps}


def print_record(record: dict) -> None:
    short = {key: value for key, value in record.items() if key not in ("steps", "answer")}
    print(json.dumps({**short, "answer_start": record["answer"][:150]}, ensure_ascii=False), flush=True)


def main(args) -> None:
    system = load_system(args.model, args.run, max_new_tokens=args.max_new_tokens, profile=args.profile)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    path = TRACE_DIR / f"{args.run}.jsonl"
    with path.open("w", encoding="utf-8") as file:
        for prompt in args.prompt or PROMPTS:
            record = trace_prompt(system, prompt, args.max_new_tokens)
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            print_record(record)
    print(json.dumps({"log": str(path)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="qwen4b_haiku_v3_test")
    parser.add_argument("--model", default=None)
    parser.add_argument("--profile", choices=sorted(PROFILES), default=None, help="Base model (else $STEERING_PROFILE)")
    parser.add_argument("--prompt", action="append", help="Repeat to trace your own prompts.")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    main(parser.parse_args())
