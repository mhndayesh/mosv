"""v5 step 1: train the per-token gate (none vs topic) at each steer layer, then freeze it for RL.

Data: topic prompts from data_review/clean_v5/<topic>/{train,dev}.jsonl and "none" prompts from the router's
labelled set (data_review/router_v2/prompts.jsonl, label none, splits train/val). Each prompt gets a short greedy
answer from the base GGUF (llama.cpp), so the gate also learns answer tokens. Hidden states x (the steer layer's
normed MLP input, what llama.cpp's gate sees) are captured in PyTorch; each layer's gate is a bias-free linear
classifier (classes x width), as the GGUF stores it. Held-out accuracy is reported per layer, on prompt and on
answer tokens. Locked test prompts are never read here.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from v5.steer import LAYERS

ROOT = Path(__file__).resolve().parents[1]
NONE_PROMPTS = ROOT / "data_review" / "router_v2" / "prompts.jsonl"
TOKENS_PER_ROW = 96
ANSWER_TOKENS = 160


def gate_rows(topic: str, per_class: int, seed: int) -> dict[str, list[dict]]:
    """{"fit": [...], "check": [...]}: rows of {"label", "prompt"}; check comes from dev/val splits only."""
    rng = random.Random(seed)
    clean = ROOT / "data_review" / "clean_v5" / topic
    read = lambda p: [json.loads(l) for l in p.open(encoding="utf-8")]  # noqa: E731
    none = [r for r in read(NONE_PROMPTS) if r["label"] == "none"]
    pick = lambda rows, n: rng.sample(rows, min(n, len(rows)))  # noqa: E731
    fit = ([{"label": topic, "prompt": r["prompt"]} for r in pick(read(clean / "train.jsonl"), per_class)]
           + [{"label": "none", "prompt": r["text"]} for r in pick([r for r in none if r["split"] == "train"], per_class)])
    check = ([{"label": topic, "prompt": r["prompt"]} for r in pick(read(clean / "dev.jsonl"), per_class // 5)]
             + [{"label": "none", "prompt": r["text"]} for r in pick([r for r in none if r["split"] == "val"], per_class // 5)])
    return {"fit": fit, "check": check}


@torch.no_grad()
def capture(model, ids_list: list[list[int]], prompt_lens: list[int], seed: int) -> tuple[dict, list, list]:
    """Per layer: (tokens, width) fp16 on CPU for up to TOKENS_PER_ROW positions per row; plus row index and is_answer."""
    rng = random.Random(seed)
    grabbed, hooks = {}, []
    for layer in LAYERS:
        hooks.append(model.model.layers[layer].mlp.register_forward_hook(
            lambda _m, inputs, _o, layer=layer: grabbed.__setitem__(layer, inputs[0][0])))
    xs = {l: [] for l in LAYERS}
    rows, answer = [], []
    try:
        for i, (ids, plen) in enumerate(zip(ids_list, prompt_lens)):
            model.model(input_ids=torch.tensor([ids], device="cuda:0"), use_cache=False)
            pos = sorted(rng.sample(range(len(ids)), min(TOKENS_PER_ROW, len(ids))))
            for layer in LAYERS:
                xs[layer].append(grabbed[layer][pos].half().cpu())
            rows += [i] * len(pos)
            answer += [p >= plen for p in pos]
    finally:
        for h in hooks:
            h.remove()
    return {l: torch.cat(v) for l, v in xs.items()}, rows, answer


def fit_layer(x: torch.Tensor, y: torch.Tensor, classes: int, steps: int = 400) -> torch.Tensor:
    x, y = x.cuda().float(), y.cuda()
    w = torch.zeros(classes, x.shape[1], device="cuda", requires_grad=True)
    weight = (len(y) / (classes * torch.bincount(y, minlength=classes).clamp_min(1))).float()
    opt = torch.optim.Adam([w], lr=1e-3)
    for _ in range(steps):
        loss = torch.nn.functional.cross_entropy(x @ w.T, y, weight=weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return w.detach()


def accuracy(w, x, y, answer) -> dict:
    pred = (x.cuda().float() @ w.T).argmax(-1).cpu()
    answer = torch.tensor(answer)
    share = lambda m: round(float((pred[m] == y[m]).float().mean()), 4) if bool(m.any()) else None  # noqa: E731
    return {"all": share(torch.ones_like(answer)), "prompt_tokens": share(~answer), "answer_tokens": share(answer)}
