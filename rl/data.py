"""Split the verified prompts into train, dev and test sets for each behavior.

Run: python -m rl.data                      (the original 3 topics, from the router prompts)
     python -m rl.data --topics v2          (8 topics, from data_review/clean_v2 into data_review/rl_v2)
Dev is watched during training. Test is only scored once, at the end of a run.

The topic list and output folder follow STEERING_TOPICS (v1, v2 or a comma list) so the trainer and
evaluator pick the same set. Unset means the original three topics and data_review/rl.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from router import TOPICS_V1, resolve_topics

ROOT = Path(__file__).parent.parent
PROMPTS = ROOT / "data_review" / "router" / "prompts.jsonl"
CLEAN_V2 = ROOT / "data_review" / "clean_v2"
SPLITS = {"train": "train", "val": "dev", "test": "test"}
SPLIT_NAMES = tuple(SPLITS.values())


def default_out(topics: tuple[str, ...]) -> Path:
    return ROOT / "data_review" / ("rl" if topics == TOPICS_V1 else "rl_v2")


ROUTE_BATCH = 128
BEHAVIORS = resolve_topics(os.environ.get("STEERING_TOPICS"))
OUT = Path(os.environ.get("STEERING_RL_OUT") or default_out(BEHAVIORS))


def split_rows(rows: list[dict], behavior: str) -> dict[str, list[dict]]:
    parts: dict[str, list[dict]] = {name: [] for name in SPLITS.values()}
    for row in rows:
        if row["label"] == behavior and row["split"] in SPLITS:
            parts[SPLITS[row["split"]]].append({"id": row["id"], "group": row["group"], "prompt": row["text"]})
    check_separate(parts, behavior)
    return parts


def clean_parts(behavior: str, clean_dir: Path = CLEAN_V2) -> dict[str, list[dict]]:
    """The cleaned train/dev/test files of one topic, kept as they are (already split by group)."""
    parts: dict[str, list[dict]] = {}
    for name in SPLIT_NAMES:
        with (clean_dir / behavior / f"{name}.jsonl").open(encoding="utf-8") as file:
            parts[name] = [{"id": r["id"], "group": r["group"], "prompt": r["prompt"]}
                           for r in map(json.loads, file) if r["prompt"].strip()]
    check_separate(parts, behavior)
    return parts


def check_separate(parts: dict[str, list[dict]], behavior: str) -> None:
    names = list(parts)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            if {r["group"] for r in parts[first]} & {r["group"] for r in parts[second]}:
                raise ValueError(f"{behavior}: {first} and {second} share a topic")


def read_split(behavior: str, split: str, folder: Path | None = None) -> list[dict]:
    with ((folder or OUT) / f"{behavior}_{split}.jsonl").open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def write_jsonl(path: Path, items: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for item in items:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_routed(topics: tuple[str, ...] = BEHAVIORS, out: Path = OUT, prompts: Path = PROMPTS,
                 checkpoint: Path = ROOT / "checkpoints" / "router.pt") -> None:
    """Run the router on the main GPU and keep training prompts it sends to the right behavior."""
    os.environ.update(HIP_VISIBLE_DEVICES="1", CUDA_VISIBLE_DEVICES="1")
    import torch

    from router import CLASSES
    from router.pipeline import load_audited, read_rows
    from router.train import decide

    features, _, ids = load_audited(read_rows(prompts))["train"]
    saved = torch.load(checkpoint, map_location="cuda:0", weights_only=True)
    router = torch.nn.Linear(features.shape[1], len(CLASSES)).to("cuda:0")
    router.load_state_dict(saved["state"])
    with torch.inference_mode():
        scaled = (torch.from_numpy(features).to("cuda:0") - saved["mean"]) / saved["std"]
        probabilities = (router(scaled) / saved["temperature"]).softmax(1).cpu().numpy()
    routed = dict(zip(ids, (CLASSES[i] for i in decide(probabilities, saved["threshold"]).tolist())))
    counts = {}
    for behavior in topics:
        rows = read_split(behavior, "train", out)
        kept = [row for row in rows if routed[row["id"]] == behavior]
        counts[behavior] = {"train": len(rows), "routed": len(kept)}
        write_jsonl(out / f"{behavior}_routed.jsonl", kept)
    print(json.dumps({"router_on_main_gpu": counts}), flush=True)


def length_batches(texts: list[str], size: int) -> list[list[int]]:
    """Indexes in batches of similar length, so little padding is computed."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    return [order[start:start + size] for start in range(0, len(order), size)]


def route_texts(torch, texts: list[str], saved: dict) -> list[str]:
    """Run the real router on any prompts, reading the model on the main GPU."""
    from transformers import AutoTokenizer

    from router import CLASSES
    from router.pipeline import MODEL, PIPELINES, chat_parts, encode, limit_vram, load_model, read_batch
    from router.train import decide

    limit_vram(torch)
    model = load_model(torch, PIPELINES["main"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    before, after, pad = chat_parts(MODEL)
    router = torch.nn.Linear(saved["mean"].numel(), len(CLASSES)).to("cuda:0")
    router.load_state_dict(saved["state"])
    names: list[str | None] = [None] * len(texts)
    with torch.inference_mode():
        for part in length_batches(texts, ROUTE_BATCH):
            batch = encode(tokenizer.backend_tokenizer, [texts[i] for i in part], before, after, pad)
            features = torch.from_numpy(read_batch(torch, model, batch)).to("cuda:0")
            logits = router((features - saved["mean"]) / saved["std"])
            choices = decide((logits / saved["temperature"]).softmax(1).cpu().numpy(), saved["threshold"],
                             len(CLASSES) - 1)
            for index, choice in zip(part, choices.tolist()):
                names[index] = CLASSES[choice]
    return names


def write_routed_all(topics: tuple[str, ...], out: Path, checkpoint: Path) -> None:
    """Keep every training prompt the router sends to its own topic, and show samples of what it dropped."""
    os.environ.update(HIP_VISIBLE_DEVICES="1", CUDA_VISIBLE_DEVICES="1")
    import torch

    rows = {behavior: read_split(behavior, "train", out) for behavior in topics}
    flat = [(behavior, row) for behavior in topics for row in rows[behavior]]
    saved = torch.load(checkpoint, map_location="cuda:0", weights_only=True)
    names = route_texts(torch, [row["prompt"] for _, row in flat], saved)
    report = {behavior: {"train": len(rows[behavior]), "routed": 0, "dropped_samples": []} for behavior in topics}
    kept: dict[str, list[dict]] = {behavior: [] for behavior in topics}
    for (behavior, row), name in zip(flat, names):
        if name == behavior:
            kept[behavior].append(row)
            report[behavior]["routed"] += 1
        elif len(report[behavior]["dropped_samples"]) < 3:
            report[behavior]["dropped_samples"].append({"sent_to": name, "prompt": row["prompt"][:110]})
    for behavior in topics:
        write_jsonl(out / f"{behavior}_routed.jsonl", kept[behavior])
    print(json.dumps({"routed": report}, ensure_ascii=False), flush=True)


def build(topics: tuple[str, ...], out: Path, source: str, prompts: Path, clean_dir: Path) -> dict:
    """Write `<topic>_<split>.jsonl` for every topic. `source` is `router` or `clean`."""
    out.mkdir(parents=True, exist_ok=True)
    rows = None
    if source == "router":
        with prompts.open(encoding="utf-8") as file:
            rows = [json.loads(line) for line in file]
    counts = {}
    for behavior in topics:
        parts = split_rows(rows, behavior) if rows is not None else clean_parts(behavior, clean_dir)
        counts[behavior] = {name: len(items) for name, items in parts.items()}
        for name, items in parts.items():
            write_jsonl(out / f"{behavior}_{name}.jsonl", items)
    return counts


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--topics", default=os.environ.get("STEERING_TOPICS"), help="v1, v2 or a comma list")
    parser.add_argument("--out", type=Path, help="default: data_review/rl for v1, data_review/rl_v2 otherwise")
    parser.add_argument("--source", choices=("router", "clean"), help="default: router for v1, clean otherwise")
    parser.add_argument("--prompts", type=Path, default=PROMPTS, help="router prompts file (source=router)")
    parser.add_argument("--clean-dir", type=Path, default=CLEAN_V2, help="cleaned topic folders (source=clean)")
    parser.add_argument("--route", action="store_true", help="keep only prompts the router sends to the right topic")
    parser.add_argument("--router-checkpoint", type=Path, default=ROOT / "checkpoints" / "router.pt")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    topics = resolve_topics(args.topics)
    out = args.out or default_out(topics)
    if args.route:
        if topics == TOPICS_V1:
            write_routed(topics, out, args.prompts, args.router_checkpoint)
        else:
            write_routed_all(topics, out, args.router_checkpoint)
        return
    source = args.source or ("router" if topics == TOPICS_V1 else "clean")
    print(json.dumps(build(topics, out, source, args.prompts, args.clean_dir), indent=1))


if __name__ == "__main__":
    main()
