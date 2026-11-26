"""Gemma 4 GGUF host for MoSV experts: add per-layer MoE (llama.cpp arch gemma4) at the steer layers.

llama.cpp's gemma4 graph treats a layer as MoE when it has ffn_gate_inp, and then computes
    out = post_ffw_norm( post_ffw_norm_1(original MLP) + post_ffw_norm_2(experts(pre_ffw_norm_2(h))) )
with the router on rms_norm(h) / sqrt(n_embd) * ffn_gate_inp.scale (softmax, top-1, weight normalised to 1).
post_ffw_norm_1 = ones keeps the original path exact (RMS norm applied twice = once); expert 0 = zeros gives
post_ffw_norm_2(0) = 0, so an all-zero build answers exactly like the source model.
Base tensors are copied unchanged (any quant type); new tensors are F32 (norms, router) and F16 (experts).
"""

from __future__ import annotations

from pathlib import Path

import gguf
import numpy as np

ARCH = "gemma4"


def empty_parts(n_expert: int, width: int, ff: int) -> dict[str, np.ndarray]:
    """GGUF tensor suffix -> array (numpy order = reversed ggml dims). Expert 0 and all experts start at zero."""
    ones = np.ones(width, np.float32)
    return {"ffn_gate_inp.weight": np.zeros((n_expert, width), np.float32),
            "ffn_gate_inp.scale": ones.copy(),
            "pre_ffw_norm_2.weight": ones.copy(),
            "post_ffw_norm_1.weight": ones.copy(),
            "post_ffw_norm_2.weight": ones.copy(),
            "ffn_gate_exps.weight": np.zeros((n_expert, ff, width), np.float16),
            "ffn_up_exps.weight": np.zeros((n_expert, ff, width), np.float16),
            "ffn_down_exps.weight": np.zeros((n_expert, width, ff), np.float16)}


def build(source: Path, out: Path, layers: dict[int, dict[str, np.ndarray]], classes: list[str], ff: int) -> None:
    """layers: steer layer -> {suffix: array} (see empty_parts). Writes `out`; base tensors copied as-is."""
    reader = gguf.GGUFReader(source)
    if reader.fields["general.architecture"].contents() != ARCH:
        raise SystemExit("source must be a gemma4 GGUF")
    writer = gguf.GGUFWriter(str(out), ARCH)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        kind = field.types[0]
        sub = field.types[-1] if kind == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), kind, sub_type=sub)
    writer.add_expert_count(len(classes))
    writer.add_expert_used_count(1)
    writer.add_expert_feed_forward_length(ff)
    writer.add_array("steer.classes", classes)
    tensors = [(t.name, t) for t in reader.tensors]
    tensors += [(f"blk.{l}.{suffix}", array) for l, parts in sorted(layers.items()) for suffix, array in parts.items()]
    for name, t in tensors:
        if isinstance(t, np.ndarray):
            writer.add_tensor_info(name, t.shape, t.dtype, t.nbytes)
        else:
            writer.add_tensor_info(name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for _, t in tensors:
        writer.write_tensor_data(np.ascontiguousarray(t) if isinstance(t, np.ndarray) else t.data)
    writer.close()
