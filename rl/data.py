"""Split the verified prompts into train, dev and test sets for each behavior.

Run: python -m rl.data
Dev is watched during training. Test is only scored once, at the end of a run.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
PROMPTS = ROOT / "data_review" / "router" / "prompts.jsonl"
OUT = ROOT / "data_review" / "rl"
BEHAVIORS = ("engineering", "story", "teaching")
SPLITS = {"train": "train", "val": "dev", "test": "test"}


def split_rows(rows: list[dict], behavior: str) -> dict[str, list[dict]]:
    parts: dict[str, list[dict]] = {name: [] for name in SPLITS.values()}
    for row in rows:
        if row["label"] == behavior and row["split"] in SPLITS:
            parts[SPLITS[row["split"]]].append({"id": row["id"], "group": row["group"], "prompt": row["text"]})
    check_separate(parts, behavior)
    return parts


def check_separate(parts: dict[str, list[dict]], behavior: str) -> None:
    names = list(parts)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            if {r["group"] for r in parts[first]} & {r["group"] for r in parts[second]}:
                raise ValueError(f"{behavior}: {first} and {second} share a topic")


def read_split(behavior: str, split: str) -> list[dict]:
    with (OUT / f"{behavior}_{split}.jsonl").open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def write_routed() -> None:
    """Run the router on the main GPU and keep training prompts it sends to the right behavior."""
    os.environ.update(HIP_VISIBLE_DEVICES="1", CUDA_VISIBLE_DEVICES="1")
    import torch

    from router import CLASSES
    from router.pipeline import load_audited, read_rows
    from router.train import decide

    features, _, ids = load_audited(read_rows())["train"]
    saved = torch.load(ROOT / "checkpoints" / "router.pt", map_location="cuda:0", weights_only=True)
    router = torch.nn.Linear(features.shape[1], len(CLASSES)).to("cuda:0")
    router.load_state_dict(saved["state"])
    with torch.inference_mode():
        scaled = (torch.from_numpy(features).to("cuda:0") - saved["mean"]) / saved["std"]
        probabilities = (router(scaled) / saved["temperature"]).softmax(1).cpu().numpy()
    routed = dict(zip(ids, (CLASSES[i] for i in decide(probabilities, saved["threshold"]).tolist())))
    counts = {}
    for behavior in BEHAVIORS:
        rows = read_split(behavior, "train")
        kept = [row for row in rows if routed[row["id"]] == behavior]
        counts[behavior] = {"train": len(rows), "routed": len(kept)}
        with (OUT / f"{behavior}_routed.jsonl").open("w", encoding="utf-8", newline="\n") as file:
            for row in kept:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"router_on_main_gpu": counts}), flush=True)


def main() -> None:
    if "--route" in sys.argv:
        write_routed()
        return
    with PROMPTS.open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file]
    OUT.mkdir(parents=True, exist_ok=True)
    counts = {}
    for behavior in BEHAVIORS:
        parts = split_rows(rows, behavior)
        counts[behavior] = {name: len(items) for name, items in parts.items()}
        for name, items in parts.items():
            with (OUT / f"{behavior}_{name}.jsonl").open("w", encoding="utf-8", newline="\n") as file:
                for item in items:
                    file.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
