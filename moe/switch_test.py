"""v4 check: does the per-token gate switch topic mid-conversation and mid-prompt?

llama.cpp does not report which expert it picks, so this rebuilds v4's exact maths in PyTorch: frozen Qwen3-4B
plus, at each steer layer, out = MLP(x) + expert_k(x), k = argmax(gate x) (sigmoid top-1 with normalised weight
= weight 1, as hy_v3 runs it), with the gates and experts from checkpoints/moe_v4/moe.pt. Answers are greedy and
written by this replica; then one pass over the whole chat records the expert picked at every token.

Run (Windows ROCm venv, one card): python moe/switch_test.py [--hf C:/llm/models/qwen3-4b-steer/hf]
Writes, in data_review/reports/v4_switch/:
- summary.json: per chat part, the share of tokens per expert at each layer, and the runs at layer 27
- <case>.tsv: one row per token: position, chat part, token text, expert at layer 9, 18, 27
- <case>.txt: the whole chat as text, cut into pieces wherever the layer-27 expert changes, each piece
  prefixed with its expert, e.g. "[story] Once upon a time ... [debugging] The KeyError means ..."
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_review" / "reports" / "v4_switch"

CASES = {
    "conversation: story -> debugging -> email": [
        "Write a short story (about 150 words) about a lighthouse keeper who finds a message in a bottle.",
        "Thanks. Now something different: my Python script crashes with KeyError: 'name' on the line "
        "print(user['name']). The data comes from json.load. How do I find and fix the bug?",
        "Great, that fixed it. Last thing: write a short professional email to my manager asking to move "
        "Friday's project review to Monday.",
    ],
    "one prompt: story + debugging": [
        "First, write a four-line poem about autumn leaves. Second, my JavaScript function returns undefined "
        "instead of the sum: function add(a, b) { a + b }. Explain the bug and fix it.",
    ],
    "one prompt: explanation + email": [
        "Explain in simple terms how photosynthesis works. Then write a short professional email to my "
        "manager asking for Friday off.",
    ],
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--hf", type=Path, default=Path("C:/llm/models/qwen3-4b-steer/hf"))
    p.add_argument("--moe", type=Path, default=ROOT / "checkpoints" / "moe_v4" / "moe.pt")
    p.add_argument("--max-new", type=int, default=400)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args(argv)


class V4Layers:
    """Adds v4's gate + experts to the steer layers' MLPs and records the picked expert per token."""

    def __init__(self, model, saved: dict, device: str):
        self.picks: dict[int, list[int]] = {}
        self.hooks = []
        for layer in saved["layers"]:
            gate = saved["gate"][layer].to(device, torch.float32)
            e = {k: v.to(device, torch.float16) for k, v in saved["experts"][layer].items()}
            mlp = model.model.layers[layer].mlp
            self.hooks.append(mlp.register_forward_hook(self._hook(layer, gate, e)))

    def _hook(self, layer, gate, e):
        def add_expert(_module, inputs, output):
            x = inputs[0]  # (batch, tokens, width): the layer's normed MLP input
            k = (x.float() @ gate.T).argmax(-1)  # sigmoid is monotonic, so top-1 = argmax of the logits
            self.picks.setdefault(layer, []).extend(k[0].tolist())
            extra = torch.zeros_like(output)
            for expert in k.unique().tolist():
                if expert == 0:
                    continue  # Expert 0 = "none" adds nothing.
                mask = k == expert
                h = x[mask]
                hidden = torch.nn.functional.silu(h @ e["gate"][expert - 1].T) * (h @ e["up"][expert - 1].T)
                extra[mask] = hidden @ e["down"][expert - 1].T
            return output + extra
        return add_expert


def chat_ids(tok, messages, add_generation_prompt: bool = False) -> list[int]:
    """Chat template -> token ids (text first: apply_chat_template's return type differs between versions)."""
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    return tok.encode(text, add_special_tokens=False)


def generate(model, tok, messages, max_new: int) -> str:
    ids = torch.tensor([chat_ids(tok, messages, True)], device=model.device)
    out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new, do_sample=False)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def spans(tok, messages) -> list[tuple[str, int, int]]:
    """Token ranges of each message in the chat template (role label included in the range)."""
    out, start = [], 0
    for i, m in enumerate(messages):
        end = len(chat_ids(tok, messages[:i + 1]))
        out.append((f"{m['role']} {i // 2 + 1}", start, end))
        start = end
    return out


def runs(picks: list[int], names: list[str]) -> list[str]:
    """Compress a pick sequence into 'expert x count' runs, dropping runs under 3 tokens."""
    out, last, count = [], None, 0
    for k in picks + [None]:
        if k == last:
            count += 1
            continue
        if last is not None and count >= 3:
            out.append(f"{names[last]} x{count}")
        last, count = k, 1
    return out


def write_logs(slug: str, tok, ids: list[int], parts, picks: dict[int, list[int]], names: list[str]) -> None:
    """Per-token TSV, and the chat as text split wherever the layer-27 expert changes."""
    part_of = [name for name, start, end in parts for _ in range(start, end)]
    layers = sorted(picks)
    rows = ["pos\tpart\ttoken\t" + "\t".join(f"layer{l}" for l in layers)]
    pieces, last = [], None
    for pos, token_id in enumerate(ids):
        text = tok.decode([token_id])
        rows.append(f"{pos}\t{part_of[pos]}\t{json.dumps(text, ensure_ascii=False)}\t"
                    + "\t".join(names[picks[l][pos]] for l in layers))
        label = names[picks[layers[-1]][pos]]
        if label != last:
            pieces.append(f"\n[{label}] ")
            last = label
        pieces.append(text)
    (OUT / f"{slug}.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (OUT / f"{slug}.txt").write_text("".join(pieces).lstrip() + "\n", encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    saved = torch.load(args.moe, map_location="cpu", weights_only=True)
    names = saved["classes"]
    tok = AutoTokenizer.from_pretrained(args.hf)
    model = AutoModelForCausalLM.from_pretrained(args.hf, dtype=torch.float16).to(args.device).eval()
    v4 = V4Layers(model, saved, args.device)
    report = {}
    OUT.mkdir(parents=True, exist_ok=True)
    for case, turns in CASES.items():
        messages = []
        with torch.inference_mode():
            for turn in turns:
                messages.append({"role": "user", "content": turn})
                messages.append({"role": "assistant", "content": generate(model, tok, messages, args.max_new)})
            v4.picks.clear()
            ids = chat_ids(tok, messages)
            model(torch.tensor([ids], device=args.device))
        slug = "".join(c if c.isalnum() else "_" for c in case.split(":")[1].strip()).strip("_")
        write_logs(slug, tok, ids, spans(tok, messages), v4.picks, names)
        parts = []
        for name, start, end in spans(tok, messages):
            per_layer = {}
            for layer, picks in v4.picks.items():
                share = Counter(names[k] for k in picks[start:end])
                per_layer[layer] = {n: round(c / (end - start), 2) for n, c in share.most_common(3)}
            parts.append({"part": name, "tokens": end - start, "experts": per_layer,
                          "runs_layer27": runs(v4.picks[27][start:end], names)})
            print(json.dumps({"case": case, "part": name, "tokens": end - start,
                              "layer18": per_layer[18], "layer27": per_layer[27]}), flush=True)
        report[case] = {"log": slug, "messages": messages, "parts": parts}
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
