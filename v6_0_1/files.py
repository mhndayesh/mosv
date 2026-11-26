"""v6.0.1 file locations, and the small helpers every stage uses to read prompts and JSON-lines files.

Everything a stage writes goes to one of three places:
  data_review/v6_0_1/   private working files: answers, check results, the training set. Not in git, because the
                        prompts come from research-only datasets.
  results/              small JSON reports that hold numbers only. In git.
  checkpoints/v6_0_1/   training runs: expert weights and logs. Not in git.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                           # the repo root
DATA = ROOT / "data_review" / "v6_0_1"                               # private working files of this version
PROMPTS = ROOT / "data_review" / "v6"                                # the v6 prompt splits, reused unchanged
RESULTS = ROOT / "results"                                           # numbers-only reports (tracked in git)
RUNS = ROOT / "checkpoints" / "v6_0_1"                               # one folder per training run
GGUF_DIR = Path("C:/llm/models/gemma4-12b/v6_0_1")                   # built GGUF files (7 GB each, outside the repo)
NONE_PROMPTS = ROOT / "data_review" / "router_v2" / "prompts.jsonl"  # everyday prompts that belong to no topic


def read_rows(path: Path) -> list[dict]:
    """All rows of a JSON-lines file. A missing file gives an empty list: that is simply a stage's first run."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_row(path: Path, row: dict) -> None:
    """Add one row to a JSON-lines file. Stages append after every answer, so a stop never loses finished work."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def claim(folder: Path, stamp: str, fresh: bool) -> None:
    """Make sure the cached answers in `folder` were written under the current rules (`stamp` = behaviors.fingerprint()).

    fresh=True deletes the folder's cached .jsonl files first. Otherwise, cached files written under different
    rules (or under unknown rules) stop the stage: mixing them with new answers would silently corrupt the numbers.
    """
    marker = folder / "behaviors.stamp"
    cached = list(folder.glob("*.jsonl")) if folder.exists() else []
    if fresh:
        for path in cached:
            path.unlink()
        cached = []
    if cached and (not marker.exists() or marker.read_text(encoding="utf-8").strip() != stamp):
        raise SystemExit(f"{folder} holds answers written under different rules (v6_0_1/behaviors.py changed). "
                         "Run this stage again with --fresh to rewrite them.")
    folder.mkdir(parents=True, exist_ok=True)
    marker.write_text(stamp, encoding="utf-8")


def prompts(topic: str, split: str, count: int) -> list[dict]:
    """The first `count` prompts of one topic and split ("train", "dev" or "test"), as {"id", "topic", "prompt"}.

    Taking the first rows (not a random sample) keeps every stage on the same prompts from run to run.
    """
    if topic == "none":                                  # everyday prompts: they teach the router "leave this alone"
        part = {"train": "train", "dev": "val"}[split]   # the router_v2 file calls its held-out part "val"
        rows = [r for r in read_rows(NONE_PROMPTS) if r["label"] == "none" and r["split"] == part]
        return [{"id": r["id"], "topic": "none", "prompt": r["text"]} for r in rows[:count]]
    if split == "test":                                  # the locked 300: locked_v6 refuses a file that differs from its lock
        from reports.answer_v5 import locked_v6
        rows = [r for r in locked_v6() if r["behavior"] == topic]
    else:
        rows = read_rows(PROMPTS / topic / f"{split}.jsonl")
    return [{"id": r["id"], "topic": topic, "prompt": r["prompt"]} for r in rows[:count]]
