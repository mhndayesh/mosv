"""v4 step 2: teach MoE gates and experts to do what the v3 router + vectors do, per token.

Run in WSL on one card (README environment): HIP_VISIBLE_DEVICES=0 python moe/train.py
Target layout (llama.cpp arch hy_v3, see moe/export.py): at each steer layer l the MLP becomes
    out = original MLP(x)  [shared expert]  +  expert_k(x)  with k = argmax gate_l(x), one expert per token,
where x is that layer's normed MLP input. Expert 0 is all zeros ("none"); expert e is a small SwiGLU MLP.

1. collect: run the frozen model over data_review/moe_v4/answers.jsonl (prompt + v3 answer), with the v3 vector of
   the row's topic switched on, and keep x at every steer layer for a sample of tokens (label = the row's topic).
2. fit: per layer, gate = linear classifier x -> topic (cross-entropy); expert e = SwiGLU trained so expert_e(x)
   equals the v3 vector s[e, l] (mean squared error, on all tokens so a wrong pick still adds about s).
Writes checkpoints/moe_v4/moe.pt and prints per-layer gate accuracy and expert error on the val split.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_profiles import get_profile  # noqa: E402
from system import NONE, build_bank, load_router, load_vectors  # noqa: E402

DATA = ROOT / "data_review" / "moe_v4"
OUT = ROOT / "checkpoints" / "moe_v4"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", default="qwen4b_haiku_v3_test")
    p.add_argument("--tokens-per-row", type=int, default=96)
    p.add_argument("--ff", type=int, default=128, help="expert hidden size")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--recollect", action="store_true")
    return p.parse_args(argv)


def classes() -> list[str]:
    """Gate order: none first (expert 0 = zeros), then the router's topic order."""
    head, _ = load_router(get_profile("qwen4b").router_checkpoint, "cpu")
    return [NONE, *head.classes[:-1]]


def collect(args, names: list[str]) -> None:
    from transformers import AutoModelForCausalLM

    profile = get_profile("qwen4b")
    rows = [json.loads(line) for line in (DATA / "answers.jsonl").open(encoding="utf-8")]
    model = AutoModelForCausalLM.from_pretrained(profile.path, dtype=torch.float16).cuda().eval()
    topics = names[1:]
    vectors, layers = load_vectors(args.run, tuple(topics))
    bank = build_bank(tuple(topics), layers, vectors, "cuda")
    bank.attach(model)
    seen: dict[int, torch.Tensor] = {}
    hooks = [model.model.layers[l].mlp.register_forward_pre_hook(
        lambda _m, inputs, l=l: seen.__setitem__(l, inputs[0][0])) for l in layers]
    rng = np.random.default_rng(0)
    feats = {l: [] for l in layers}
    meta = []
    for i, row in enumerate(rows):
        ids = row["prompt_ids"] + row["answer_ids"]
        keep = np.sort(rng.choice(len(ids), min(args.tokens_per_row, len(ids)), replace=False))
        expert = None if row["label"] == NONE else topics.index(row["label"])
        with bank.choose(expert), torch.inference_mode():
            model.model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
        picked = torch.from_numpy(keep).cuda()
        for l in layers:
            feats[l].append(seen[l][picked].cpu().numpy().astype(np.float16))
        meta += [(names.index(row["label"]), row["split"] == "val", int(k >= len(row["prompt_ids"]))) for k in keep]
        if i % 200 == 0:
            print(json.dumps({"collected": i, "of": len(rows)}), flush=True)
    for h in hooks:
        h.remove()
    for l in layers:
        np.save(DATA / f"x_layer{l}.npy", np.concatenate(feats[l]))
    np.save(DATA / "meta.npy", np.array(meta, dtype=np.int64))  # label, is_val, is_answer_token
    (DATA / "layers.json").write_text(json.dumps(list(layers)))


class Experts(torch.nn.Module):
    """Expert 0 = zeros; experts 1..n are SwiGLU MLPs, the same maths llama.cpp runs for MoE experts."""

    def __init__(self, n: int, width: int, ff: int):
        super().__init__()
        self.gate = torch.nn.Parameter(torch.randn(n, ff, width) * width ** -0.5)
        self.up = torch.nn.Parameter(torch.randn(n, ff, width) * width ** -0.5)
        self.down = torch.nn.Parameter(torch.zeros(n, width, ff))

    def forward(self, x):  # x: (tokens, width) -> (n, tokens, width)
        hidden = torch.nn.functional.silu(torch.einsum("nfw,tw->ntf", self.gate, x)) * torch.einsum("nfw,tw->ntf", self.up, x)
        return torch.einsum("nwf,ntf->ntw", self.down, hidden)


def fit_layer(x, labels, train, target, args):
    """Returns gate weight (classes, width), expert weights, and val metrics for one layer."""
    width, n_cls = x.shape[1], target.shape[0] + 1
    gate = torch.nn.Linear(width, n_cls, bias=False).cuda()
    experts = Experts(target.shape[0], width, args.ff).cuda()
    opt = torch.optim.AdamW([*gate.parameters(), *experts.parameters()], lr=1e-3, weight_decay=0.0)
    counts = torch.bincount(labels[train], minlength=n_cls).float()
    weights = (counts.sum() / (counts.clamp_min(1) * n_cls)).cuda()  # Balance classes ("none" has 2x rows).
    idx = torch.nonzero(train).squeeze(1)
    for step in range(args.steps):
        batch = idx[torch.randint(len(idx), (4096,))]
        xb, yb = x[batch].cuda().float(), labels[batch].cuda()
        gate_loss = torch.nn.functional.cross_entropy(gate(xb), yb, weight=weights)
        made = experts(xb)  # (topics, tokens, width)
        expert_loss = ((made - target[:, None, :]) ** 2).mean() / (target ** 2).mean()
        opt.zero_grad()
        (gate_loss + expert_loss).backward()
        opt.step()
        if step % 500 == 0:
            print(json.dumps({"step": step, "gate_loss": round(gate_loss.item(), 4),
                              "expert_rel_mse": round(expert_loss.item(), 4)}), flush=True)
    return gate, experts


@torch.no_grad()
def evaluate(gate, experts, x, labels, val, answer, target) -> dict:
    xv, yv = x[val].cuda().float(), labels[val].cuda()
    pick = gate(xv).argmax(1)
    errors = [((experts(chunk) - target[:, None, :]).norm(dim=2) / target.norm(dim=1)[:, None])
              for chunk in xv.split(4096)]  # Chunks: all val tokens at once would need about 4 GB.
    rel = torch.cat(errors, dim=1).mean(1)
    return {"gate_acc": round(float((pick == yv).float().mean()), 4),
            "gate_acc_answer_tokens": round(float((pick == yv)[answer[val].cuda().bool()].float().mean()), 4),
            "expert_rel_error": [round(float(r), 3) for r in rel]}


def main():
    args = parse_args()
    names = classes()
    if args.recollect or not (DATA / "meta.npy").exists():
        collect(args, names)
    layers = json.loads((DATA / "layers.json").read_text())
    meta = torch.from_numpy(np.load(DATA / "meta.npy"))
    labels, val, answer = meta[:, 0], meta[:, 1].bool(), meta[:, 2]
    vectors, _ = load_vectors(args.run, tuple(names[1:]))
    saved = {"classes": names, "layers": layers, "ff": args.ff, "gate": {}, "experts": {}, "report": {}}
    for slot, l in enumerate(layers):
        x = torch.from_numpy(np.load(DATA / f"x_layer{l}.npy")).cuda()  # About 1.2 GB fp16 per layer.
        target = torch.stack([v[slot] for v in vectors]).cuda()  # (topics, width)
        gate, experts = fit_layer(x, labels, ~val, target, args)
        saved["gate"][l] = gate.weight.detach().cpu()
        saved["experts"][l] = {k: v.detach().cpu() for k, v in experts.state_dict().items()}
        saved["report"][l] = evaluate(gate, experts, x, labels, val, answer, target)
        print(json.dumps({"layer": l, **saved["report"][l]}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(saved, OUT / "moe.pt")


if __name__ == "__main__":
    main()
