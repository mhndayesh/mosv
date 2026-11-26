"""Check a v5 checkpoint really works, end to end. Run after training (needs both cards free):
    python -m v5.verify --run eng_pilot2
1. PyTorch (fp16 Bonsai + steer) vs llama.cpp (1-bit GGUF built from the same steer.pt): greedy tokens agree.
2. Trained GGUF vs plain Bonsai in llama.cpp: the expert changes the answers (how many, first split position).
3. Expert push: |expert(x)| / |mlp(x)| on real topic tokens, per layer (why KL may be ~0).
4. Routing share in PyTorch on topic prompts vs none prompts (the gate llama.cpp uses is the same tensor).
Prints one JSON object and saves it to checkpoints/v5/<run>/verify.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True)
    p.add_argument("--prompts", type=int, default=8)
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--torch-hip", default="0")
    args = p.parse_args(argv)
    os.environ["HIP_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"] = args.torch_hip
    from v5 import train as T

    sys.path.insert(0, str(T.GGUF_PY))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from moe import live_gguf
    from v5.gate import NONE_PROMPTS
    from v5.llama import Card, Pool
    from v5.steer import LAYERS, Steer

    folder = ROOT / "checkpoints" / "v5" / args.run
    saved = torch.load(folder / "steer.pt", weights_only=False)
    topic = saved["args"]["topic"]
    tok = AutoTokenizer.from_pretrained(T.BASE_HF)
    model = AutoModelForCausalLM.from_pretrained(T.BASE_HF, dtype=torch.float16).to("cuda:0").eval()
    steer = Steer(["none", topic], model.config.hidden_size, "cuda:0")
    steer.load(saved["steer"])
    steer.attach(model)
    rows = T.read_rows(topic, "dev")[: args.prompts]
    ids = T.prompt_ids(tok, [r["prompt"] for r in rows])
    out = {"run": args.run, "update": saved["update"]}

    # 3 + 4: expert push and routing, on topic prompts and on none prompts.
    pushes = {l: [] for l in LAYERS}

    def measure(layer):
        def hook(_mod, inputs, output):
            x = inputs[0]
            k = (x.float() @ steer.gate[layer].T).argmax(-1)
            e = steer.experts[layer]
            mask = k == 1
            if bool(mask.any()):
                h = x[mask]
                extra = (torch.nn.functional.silu(h @ e["gate"][0].T.half()) * (h @ e["up"][0].T.half())) @ e["down"][0].T.half()
                base = (output[mask] - extra).float().norm(dim=-1).mean()
                pushes[layer].append(float(extra.float().norm(dim=-1).mean() / base))
        return hook
    probes = [model.model.layers[l].mlp.register_forward_hook(measure(l)) for l in LAYERS]
    none_rows = [json.loads(l) for l in NONE_PROMPTS.open(encoding="utf-8")]
    none_ids = T.prompt_ids(tok, [r["text"] for r in none_rows if r["label"] == "none" and r["split"] == "val"][: args.prompts])
    with torch.no_grad():
        steer.routed_share()
        for i in ids:
            model(input_ids=torch.tensor([i], device="cuda:0"))
        topic_route = steer.routed_share()
        for i in none_ids:
            model(input_ids=torch.tensor([i], device="cuda:0"))
        none_route = steer.routed_share()
    for h in probes:
        h.remove()
    out["expert_push_ratio"] = {l: round(sum(v) / max(len(v), 1), 5) for l, v in pushes.items()}
    out["routed_to_expert"] = {"topic_prompts": {l: v[topic] for l, v in topic_route.items()},
                               "none_prompts": {l: v[topic] for l, v in none_route.items()}}

    # 1: PyTorch greedy tokens.
    with torch.no_grad():
        torch_tokens = []
        for i in ids:
            g = model.generate(torch.tensor([i], device="cuda:0"), max_new_tokens=args.tokens, do_sample=False)
            torch_tokens.append(g[0, len(i):].tolist())
    del model
    torch.cuda.empty_cache()

    # 1 + 2: llama.cpp greedy tokens, trained GGUF vs plain Bonsai.
    built = folder / "verify.gguf"
    live_gguf.build(T.BASE_GGUF, built, steer.to_moe())
    answers = {}
    for name, path in (("trained", built), ("base", T.BASE_GGUF)):
        pool = Pool(path, [Card("Vulkan0", 8150, 8, 2048)], folder)
        pool.start()
        try:
            answers[name] = [a[0]["tokens"][: args.tokens] for a in pool.generate(ids, 1, args.tokens, greedy=True)]
        finally:
            pool.stop()

    def first_split(a, b):
        return next((n for n, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    out["torch_vs_llama_trained"] = {"first_split": [first_split(a, b) for a, b in zip(torch_tokens, answers["trained"])],
                                     "identical": sum(a == b for a, b in zip(torch_tokens, answers["trained"]))}
    out["trained_vs_base_llama"] = {"first_split": [first_split(a, b) for a, b in zip(answers["trained"], answers["base"])],
                                    "identical": sum(a == b for a, b in zip(answers["trained"], answers["base"]))}
    out["of"] = len(ids)
    out["time"] = round(time.time())
    (folder / "verify.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
