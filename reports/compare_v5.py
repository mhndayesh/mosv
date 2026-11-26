"""Blind Haiku comparison of two recorded answer files on a locked v5 test split (e.g. base vs trained).

Run (any venv with anthropic; Windows: ai-training-rocm-windows):
    python reports/compare_v5.py --topic engineering --a bonsai_normal_2048 --b mosv_upd100
Each prompt's two answers go to Haiku as one group with the topic rubric (rl/judge_rubrics.py); the judge sees them
in both orders (rl/api_judge.py) and never learns which file is which. Writes data_review/v5_runs/<topic>/
compare_<a>_vs_<b>.json with per-prompt rewards, and prints win rate, mean reward and length, overall and per source.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(topic: str, tag: str) -> dict:
    path = ROOT / "data_review" / "v5_runs" / topic / f"{tag}.jsonl"
    rows = [json.loads(line) for line in path.open(encoding="utf-8")][1:]
    return {r["id"]: r for r in rows}


def summary(pairs: list[dict]) -> dict:
    n = max(len(pairs), 1)
    wins = sum((p["b_reward"] > p["a_reward"]) + 0.5 * (p["b_reward"] == p["a_reward"]) for p in pairs)
    return {"prompts": len(pairs), "b_beats_a_pct": round(100 * wins / n, 1),
            "a_reward": round(sum(p["a_reward"] for p in pairs) / n, 4),
            "b_reward": round(sum(p["b_reward"] for p in pairs) / n, 4),
            "a_tokens": round(sum(p["a_tokens"] for p in pairs) / n),
            "b_tokens": round(sum(p["b_tokens"] for p in pairs) / n),
            "a_hit_cap": sum(p["a_finish"] == "length" for p in pairs),
            "b_hit_cap": sum(p["b_finish"] == "length" for p in pairs),
            "a_loops": sum(p["a_finish"] == "loop" for p in pairs),
            "b_loops": sum(p["b_finish"] == "loop" for p in pairs)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--topic", default="engineering")
    p.add_argument("--a", required=True, help="first answer file tag (e.g. the baseline)")
    p.add_argument("--b", required=True, help="second answer file tag (e.g. the trained model)")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--budget", type=float, default=5.0)
    args = p.parse_args(argv)
    from rl.api_judge import ApiJudge, Group

    a, b = load(args.topic, args.a), load(args.topic, args.b)
    ids = sorted(set(a) & set(b))
    out_dir = ROOT / "data_review" / "v5_runs" / args.topic
    judge = ApiJudge(out_dir / f"judge_compare_{args.a}_vs_{args.b}.jsonl", run=f"compare_{args.a}_{args.b}",
                     budget_usd=args.budget, concurrency=args.concurrency)
    groups = [Group(f"{args.topic}:cmp:{i}", args.topic, a[i]["prompt"], (a[i]["answer"], b[i]["answer"])) for i in ids]
    try:
        rewards, stats = judge.judge(groups, 0)
    finally:
        judge.close()
    pairs = [{"id": i, "source": a[i]["source"], "a_reward": r[0], "b_reward": r[1],
              "a_tokens": a[i]["tokens"] or 0, "b_tokens": b[i]["tokens"] or 0,
              "a_finish": a[i]["finish"], "b_finish": b[i]["finish"]}
             for i, r in zip(ids, rewards) if r is not None]
    by_source = defaultdict(list)
    for pair in pairs:
        by_source[pair["source"]].append(pair)
    result = {"a": args.a, "b": args.b, "scored": len(pairs), "of": len(ids), "judge": stats.summary(),
              "overall": summary(pairs), "by_source": {s: summary(v) for s, v in sorted(by_source.items())}, "pairs": pairs}
    (out_dir / f"compare_{args.a}_vs_{args.b}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, indent=1))


if __name__ == "__main__":
    main()
