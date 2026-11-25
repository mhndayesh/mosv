"""Pick each topic's test vector from a training run and write the run summary. CPU only.

Run from the repo root: python reports/pick_vectors.py [--run qwen4b_haiku_v3] [--out qwen4b_haiku_v3_test]

The owner's rule (2026-11-24): the judge is not fully trusted, so favour recent vectors. Of each topic's last 3
dev checks, take the one with the highest dev score (a tie goes to the later check). The vector comes from the
snapshot saved at that check. A topic with no dev check gets a zero vector, which is normal Qwen.

Dev checks come from the run's checkpoint history, or, when a finished topic no longer holds it, from the
training logs in data_review/rl_v4/<run>/train_logs. Final test results come from the same logs. The training
files are only read.
"""

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "checkpoints" / "rl_pilot"
RUNS = ROOT / "data_review" / "rl_v4"
REPORTS = ROOT / "data_review" / "reports"
TOPICS = ("engineering", "story", "teaching", "debugging", "math_solving", "explanation", "professional_email",
          "how_to_guide")
LATEST = 3


def log_rows(logs: list[Path]) -> list[dict]:
    rows = []
    for path in logs:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("{") and '"expert"' in line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # A line cut off when a run was stopped.
    return rows


def log_history(topic: str, rows: list[dict]) -> list[dict]:
    """Dev checks for one topic, one per step; a later log line for the same step wins (resumed runs)."""
    found = {r["step"]: {"step": r["step"], "dev": r["dev_steered"]}
             for r in rows if r.get("expert") == topic and "dev_steered" in r}
    return [found[s] for s in sorted(found)]


def final_result(topic: str, rows: list[dict]) -> dict | None:
    done = [r for r in rows if r.get("expert") == topic and r.get("finished") and "test_steered" in r]
    return done[-1] if done else None


def choose(history: list[dict]) -> tuple[dict | None, list[dict]]:
    """Best dev score among the last LATEST checks; on a tie the later check wins."""
    recent = [h for h in history if h.get("dev") is not None][-LATEST:]
    if not recent:
        return None, []
    return max(reversed(recent), key=lambda h: h["dev"]), recent


def pick(source: Path, target: Path, topic: str, rows: list[dict]) -> dict:
    live = torch.load(source / f"{topic}.pt", map_location="cpu", weights_only=True)
    picked, recent = choose(live.get("history") or log_history(topic, rows))
    if picked is None:
        vector, step = torch.zeros_like(live["bank"]["vectors.0"]), 0
    else:
        step = picked["step"]
        snapshot = torch.load(source / "snapshots" / f"{topic}_step{step:05d}.pt", map_location="cpu", weights_only=True)
        if snapshot["step"] != step:
            raise ValueError(f"{topic}: snapshot holds step {snapshot['step']}, expected {step}")
        vector = snapshot["bank"]["vectors.0"]
    torch.save({"bank": {"vectors.0": vector.clone()}, "step": step, "expert": topic, "layers": live["layers"],
                "profile": live["profile"], "picked_from": recent}, target / f"{topic}.pt")
    return {"topic": topic, "step": step, "dev": picked and picked["dev"], "trained_to": live["step"],
            "latest_3": [[h["step"], round(h["dev"], 3)] for h in recent], "final": final_result(topic, rows)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", default="qwen4b_haiku_v3", help="training run under checkpoints/rl_pilot")
    p.add_argument("--out", default="qwen4b_haiku_v3_test", help="folder for the picked vectors")
    p.add_argument("--summary", type=Path, default=REPORTS / "haiku_v3_summary.json")
    args = p.parse_args(argv)
    source, target = CHECKPOINTS / args.run, CHECKPOINTS / args.out
    target.mkdir(exist_ok=True)
    rows = log_rows(sorted((RUNS / args.run / "train_logs").glob("*.log")))
    summary = [pick(source, target, topic, rows) for topic in TOPICS]
    args.summary.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for item in summary:
        print(json.dumps({k: item[k] for k in ("topic", "trained_to", "step", "dev", "latest_3")}))


if __name__ == "__main__":
    main()
