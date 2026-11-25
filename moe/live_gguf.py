"""Keep one hy_v3 GGUF in step with the trainer: build it once, then overwrite only the gate/expert tensors in place.

The base model tensors never move. The gate and expert tensors are written after them (moe/export.py), so their
byte offsets are fixed for a given layout and a training update rewrites only those bytes (about 10-20 MB) before
the llama-server that samples from the file is restarted. Same shapes and dtypes only: a layout change needs build().
"""

from __future__ import annotations

import os
from pathlib import Path

import gguf
import numpy as np

from moe import export


def build(source: Path, out: Path, moe: dict) -> None:
    """Write the full GGUF once (base copied from `source`, a qwen3 GGUF, e.g. Bonsai-4B-Q1_0)."""
    export.write(export.source_reader(source), out, moe)


def moe_slots(path: Path) -> dict[str, tuple[int, int]]:
    """name -> (byte offset, byte count) for every gate and expert tensor in the file."""
    reader = gguf.GGUFReader(path)
    return {t.name: (int(t.data_offset), int(t.n_bytes)) for t in reader.tensors
            if t.name.endswith(("ffn_gate_inp.weight", "_exps.weight"))}


def moe_arrays(moe: dict) -> dict[str, np.ndarray]:
    """The tensors build() writes for `moe` (same names, dtypes and byte layout as export.tensor_list)."""
    return {f"blk.{layer}.ffn_{key}.weight": np.ascontiguousarray(array)
            for layer, parts in moe["layers"].items() for key, array in parts.items()}


def patch(path: Path, moe: dict, slots: dict[str, tuple[int, int]] | None = None) -> int:
    """Overwrite the gate/expert bytes in place. Returns bytes written. Refuses any size mismatch."""
    slots = slots or moe_slots(path)
    arrays = moe_arrays(moe)
    if set(arrays) != set(slots):
        raise ValueError(f"tensor names differ from the file: {sorted(set(arrays) ^ set(slots))}")
    written = 0
    with open(path, "r+b") as f:
        for name, array in arrays.items():
            offset, size = slots[name]
            if array.nbytes != size:
                raise ValueError(f"{name}: {array.nbytes} bytes, file slot holds {size}")
            f.seek(offset)
            f.write(array.tobytes())
            written += size
        f.flush()
        os.fsync(f.fileno())
    return written
