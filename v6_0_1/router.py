"""The router: which expert each token goes to. One small linear layer per steer layer, fitted once before training.

In llama.cpp's gemma4 MoE layer every token picks the expert with the largest score, where
    scores = rms(h) / sqrt(width) @ router matrix          (h = the hidden state entering the feed-forward block)
Class 0 is "none": its expert is all zeros, so a token routed there passes through unchanged. That is how the
shipped model stays the plain model on everything that is not a trained topic.

Differences from v6's router (train_gate in v6/train.py):
  - the classes come from the gate: "none" plus only the topics being trained. math_solving and any skipped topic
    are labeled "none", so they keep the base behavior;
  - the texts are answers build_data.py already recorded, so nothing has to be generated here;
  - a trained topic is shown its plain answers AND its kept told answers, because after training the model's own
    words will look like the told ones.
"""

from __future__ import annotations

import random

POSITIONS = 48      # token positions sampled from each text
ANSWER_TOKENS = 96  # answer tokens read from each text (the prompt is always read in full)
STEPS, LR = 500, 1e-2  # logistic-regression fit, the same as v6


def collect(torch, model, tok, steer, items: list[tuple], rng: random.Random) -> tuple:
    """Run each (class index, prompt, answer) text through the model and return the router inputs at sampled tokens.

    Returns ({layer: features}, labels, is_answer_token). Expert strength is still zero here, so these are the base
    model's hidden states.
    """
    from v6.steer import rms
    from v6.train import prompt_ids, text_core

    grabbed = {}  # layer -> hidden states of the text being read (filled by the hooks below)
    hooks = [text_core(model).layers[l].pre_feedforward_layernorm.register_forward_pre_hook(
        lambda _module, args, l=l: grabbed.__setitem__(l, args[0][0])) for l in steer.layers]
    xs, ys, in_answer = {l: [] for l in steer.layers}, [], []
    try:
        with torch.inference_mode():
            for label, prompt, answer in items:
                head = prompt_ids(tok, [prompt])[0]  # the chat-formatted prompt, exactly as at test time
                full = head + tok.encode(answer, add_special_tokens=False)[:ANSWER_TOKENS]
                text_core(model)(input_ids=torch.tensor([full], device="cuda:0"), use_cache=False)
                pos = sorted(rng.sample(range(len(full)), min(POSITIONS, len(full))))
                for l in steer.layers:  # the same scaling llama.cpp applies before the router matrix
                    xs[l].append((rms(grabbed[l].float())[pos] * steer.width ** -0.5).half().cpu())
                ys += [label] * len(pos)
                in_answer += [p >= len(head) for p in pos]
    finally:
        for h in hooks:
            h.remove()
    return {l: torch.cat(v) for l, v in xs.items()}, torch.tensor(ys), torch.tensor(in_answer)


def fit(torch, model, tok, steer, rows: list[dict], per_class: int, seed: int) -> dict:
    """Fit steer.gate from labeled texts ({"label", "prompt", "answer"}); returns held-out accuracy per layer.

    One text in six of every class is held out and never used for fitting; accuracy is measured on those.
    """
    rng = random.Random(seed)
    parts = {"fit": [], "check": []}
    for index, name in enumerate(steer.classes):
        texts = [r for r in rows if r["label"] == name]
        if not texts:
            raise SystemExit(f"router: no texts for class '{name}' (run build_data first)")
        rng.shuffle(texts)
        texts = texts[:per_class]
        cut = max(1, len(texts) // 6)
        parts["check"] += [(index, r["prompt"], r["answer"]) for r in texts[:cut]]
        parts["fit"] += [(index, r["prompt"], r["answer"]) for r in texts[cut:]]
    data = {part: collect(torch, model, tok, steer, items, rng) for part, items in parts.items()}

    n, accuracy = len(steer.classes), {}
    for l in steer.layers:
        x, y = data["fit"][0][l].cuda().float(), data["fit"][1].cuda()
        w = torch.zeros(n, steer.width, device="cuda", requires_grad=True)
        weight = (len(y) / (n * torch.bincount(y, minlength=n).clamp_min(1))).float()  # every class counts the same
        opt = torch.optim.Adam([w], lr=LR)
        for _ in range(STEPS):
            loss = torch.nn.functional.cross_entropy(x @ w.T, y, weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
        steer.gate[l].copy_(w.detach())  # this matrix is written into the GGUF as the layer's router
        xc, yc, ac = data["check"][0][l].cuda().float(), data["check"][1], data["check"][2]
        pred = (xc @ w.detach().T).argmax(-1).cpu()
        accuracy[l] = {"all": round(float((pred == yc).float().mean()), 4),                      # every sampled token
                       "answer_tokens": round(float((pred[ac] == yc[ac]).float().mean()), 4),    # tokens of the answer only
                       "none_kept_none": round(float((pred[yc == 0] == 0).float().mean()), 4)}   # "leave alone" tokens left alone
    return {"texts": {part: len(items) for part, items in parts.items()}, "held_out_accuracy": accuracy}
