"""Write each topic's steering vector as a llama.cpp control-vector GGUF.

Run with any Python that has torch and gguf (llama.cpp's gguf-py):
    python v3/export_vectors.py [--run qwen4b_haiku_v3_test] [--out C:/llm/models/qwen3-4b-steer/vectors]

Training adds the vector to the OUTPUT of decoder layer l (0-based, a forward hook on model.layers[l]).
llama.cpp adds tensor "direction.N" to the output of layer N (0-based; layer 0 can never be steered), so the index
is used as it is. Use the file with: llama-server -m <base.gguf> --control-vector steer-<topic>.gguf
"""

import argparse
from pathlib import Path

import gguf
import torch

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "checkpoints" / "rl_pilot"


def directions(saved: dict) -> dict:
    """Layer index -> float32 vector, one per steered layer."""
    vectors = saved["bank"]["vectors.0"].float()
    layers = [int(layer) for layer in saved["layers"]]
    if vectors.shape[0] != len(layers) or min(layers) < 1:
        raise ValueError(f"Vector rows {tuple(vectors.shape)} do not fit layers {layers}")
    return dict(zip(layers, vectors))


def write(path: Path, topic: str, found: dict) -> None:
    writer = gguf.GGUFWriter(str(path), "controlvector")
    writer.add_string("controlvector.model_hint", "qwen3")
    writer.add_uint32("controlvector.layer_count", len(found))
    writer.add_string("general.name", f"steer-{topic}")
    for layer, vector in sorted(found.items()):
        writer.add_tensor(f"direction.{layer}", vector.numpy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", default="qwen4b_haiku_v3_test")
    p.add_argument("--out", type=Path, default=Path("C:/llm/models/qwen3-4b-steer/vectors"))
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for source in sorted((CHECKPOINTS / args.run).glob("*.pt")):
        saved = torch.load(source, map_location="cpu", weights_only=True)
        found = directions(saved)
        write(args.out / f"steer-{source.stem}.gguf", source.stem, found)
        print(source.stem, "step", saved.get("step"), "layers", sorted(found),
              "norms", [round(float(v.norm()), 2) for _, v in sorted(found.items())])


if __name__ == "__main__":
    main()
