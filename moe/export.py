"""v4 step 3: write Qwen3-4B + the trained gates/experts as ONE GGUF that stock llama.cpp runs (arch hy_v3).

Run with the gguf-tools venv (torch + gguf-py):
    python moe/export.py <qwen3-F16.gguf> <out.gguf> [--moe checkpoints/moe_v4/moe.pt]
Without --moe, all experts are zeros: the output then equals plain Qwen exactly, which checks the host
architecture (checked 2026-11-25: 5/5 greedy answers identical to the qwen3 GGUF, same speed).

Why hy_v3: it is the one llama.cpp architecture with Qwen3's attention (per-head q/k RMSNorm before RoPE, NEOX
RoPE, GQA) that lets each layer be dense OR MoE (a layer is MoE when it has ffn_gate_inp) and adds an always-on
shared expert. So every layer stays the original Qwen3 layer; at the steer layers the original MLP becomes the
shared expert and the topic experts sit next to it. Gating: sigmoid, top-1, weights normalised, so the picked
expert's output is added with weight exactly 1.
"""

import argparse
import sys
from pathlib import Path

import gguf
import numpy as np

STEER_LAYERS = (9, 18, 27)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("source", type=Path, help="Qwen3 GGUF (arch qwen3)")
    p.add_argument("out", type=Path)
    p.add_argument("--moe", type=Path, help="checkpoints/moe_v4/moe.pt")
    p.add_argument("--empty", type=int, default=9, help="without --moe: this many all-zero experts")
    return p.parse_args(argv)


def load_moe(path: Path) -> dict:
    """Per layer: gate (n, width) f32, gate/up exps (n, ff, width) and down exps (n, width, ff) f16, expert 0 = 0."""
    import torch

    saved = torch.load(path, map_location="cpu", weights_only=True)
    layers = {}
    for layer in saved["layers"]:
        e = saved["experts"][layer]
        zero = lambda t: torch.cat([torch.zeros_like(t[:1]), t])  # noqa: E731  Expert 0 = "none".
        layers[int(layer)] = {"gate_inp": saved["gate"][layer].float().numpy(),
                              "gate_exps": zero(e["gate"]).half().numpy(), "up_exps": zero(e["up"]).half().numpy(),
                              "down_exps": zero(e["down"]).half().numpy()}
    return {"classes": saved["classes"], "ff": int(saved["ff"]), "layers": layers}


def empty_moe(n: int, width: int, ff: int = 64) -> dict:
    blank = {"gate_inp": np.zeros((n, width), np.float32), "gate_exps": np.zeros((n, ff, width), np.float16),
             "up_exps": np.zeros((n, ff, width), np.float16), "down_exps": np.zeros((n, width, ff), np.float16)}
    return {"classes": [f"e{i}" for i in range(n)], "ff": ff, "layers": {l: blank for l in STEER_LAYERS}}


def copy_metadata(reader, writer, moe: dict) -> None:
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        kind = field.types[0]
        sub = field.types[-1] if kind == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name.replace("qwen3.", "hy_v3.", 1), field.contents(), kind, sub_type=sub)
    writer.add_expert_count(len(moe["classes"]))
    writer.add_expert_used_count(1)
    writer.add_expert_feed_forward_length(moe["ff"])
    writer.add_expert_shared_feed_forward_length(int(reader.fields["qwen3.feed_forward_length"].contents()))
    writer.add_expert_weights_norm(True)
    writer.add_expert_weights_scale(1.0)
    writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
    writer.add_array("steer.classes", moe["classes"])  # Extra key, ignored by llama.cpp; names expert k.


def tensor_list(reader, moe: dict) -> list:
    """Every Qwen3 tensor (steer-layer MLPs renamed to the shared expert) plus the new gate and expert tensors."""
    out = []
    for t in reader.tensors:
        name = t.name
        for layer in moe["layers"]:
            for part in ("gate", "up", "down"):
                if name == f"blk.{layer}.ffn_{part}.weight":
                    name = f"blk.{layer}.ffn_{part}_shexp.weight"
        out.append((name, t))
    for layer, parts in moe["layers"].items():
        out += [(f"blk.{layer}.ffn_{key}.weight", array) for key, array in parts.items()]
    return out


def main(argv=None):
    args = parse_args(argv)
    reader = gguf.GGUFReader(args.source)
    if reader.fields["general.architecture"].contents() != "qwen3":
        sys.exit("source must be a qwen3 GGUF")
    width = int(reader.fields["qwen3.embedding_length"].contents())
    moe = load_moe(args.moe) if args.moe else empty_moe(args.empty, width)
    writer = gguf.GGUFWriter(str(args.out), "hy_v3")
    copy_metadata(reader, writer, moe)
    tensors = tensor_list(reader, moe)
    for name, t in tensors:
        if isinstance(t, np.ndarray):
            writer.add_tensor_info(name, t.shape, t.dtype, t.nbytes)
        else:
            writer.add_tensor_info(name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for _, t in tensors:
        writer.write_tensor_data(t if isinstance(t, np.ndarray) else t.data)
    writer.close()
    print(f"wrote {args.out} with experts {moe['classes']} at layers {sorted(moe['layers'])}")


if __name__ == "__main__":
    main()
