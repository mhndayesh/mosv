"""v4 step 1: answers written WITH the v3 vectors, as token ids, for teaching the MoE experts and gates.

Run in WSL on one card (README environment): HIP_VISIBLE_DEVICES=0 python moe/make_data.py
Prompts: the router's labelled prompts (data_review/router_v2/prompts.jsonl), train and val splits, a fixed sample
per class. Each topic's prompts are answered with that topic's v3 vector; "none" prompts with no vector. Greedy,
up to --max-new tokens, so the texts are what v3 itself writes.
Writes data_review/moe_v4/answers.jsonl: id, label, split, prompt_ids, answer_ids.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reports"))
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from model_profiles import get_profile  # noqa: E402
from system import NONE, load_vectors  # noqa: E402
from vllm_full_test import build_engine, push  # noqa: E402

OUT = ROOT / "data_review" / "moe_v4"
PROMPTS = ROOT / "data_review" / "router_v2" / "prompts.jsonl"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", default="qwen4b_haiku_v3_test")
    p.add_argument("--per-topic", type=int, default=200, help="train prompts per topic (val gets a fifth)")
    p.add_argument("--max-new", type=int, default=512)
    p.add_argument("--seed", type=int, default=4)
    # Engine settings for build_engine (one card, fp8 KV, as in the full test).
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--kv", default="fp8")
    p.add_argument("--util", type=float, default=0.9)
    p.add_argument("--max-seqs", type=int, default=256)
    p.add_argument("--batched", type=int, default=8192)
    p.add_argument("--kv-gib", type=float, default=4.0)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--eager", action="store_true")
    return p.parse_args(argv)


def sample(rows: list[dict], per_topic: int, seed: int) -> list[dict]:
    """Fixed sample: per class, per_topic train + per_topic // 5 val; "none" gets twice as many."""
    rng, chosen = random.Random(seed), []
    for label in sorted({r["label"] for r in rows}):
        scale = 2 if label == NONE else 1
        for split, count in (("train", per_topic * scale), ("val", per_topic * scale // 5)):
            pool = sorted((r for r in rows if r["label"] == label and r["split"] == split), key=lambda r: r["id"])
            chosen += rng.sample(pool, min(count, len(pool)))
    return chosen


def main():
    args = parse_args()
    from vllm import SamplingParams

    rows = sample([json.loads(line) for line in PROMPTS.open(encoding="utf-8")], args.per_topic, args.seed)
    llm = build_engine(args, get_profile("qwen4b"))
    tok = llm.get_tokenizer()
    for row in rows:
        text = tok.apply_chat_template([{"role": "user", "content": row["text"]}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        row["prompt_ids"] = tok.encode(text, add_special_tokens=False)[:args.max_model_len - args.max_new]
    eos = sorted({tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")})
    params = SamplingParams(temperature=0.0, max_tokens=args.max_new, stop_token_ids=eos, detokenize=False)
    topics = sorted({r["label"] for r in rows} - {NONE})
    vectors = dict(zip(topics, load_vectors(args.run, tuple(topics))[0]))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "answers.jsonl").open("w", encoding="utf-8") as out:
        for label in [NONE, *topics]:
            mine = [r for r in rows if r["label"] == label]
            push(llm, vectors.get(label))
            results = llm.generate([{"prompt_token_ids": r["prompt_ids"]} for r in mine], params, use_tqdm=False)
            for row, result in zip(mine, results):
                out.write(json.dumps({k: row[k] for k in ("id", "label", "split", "prompt_ids")}
                                     | {"answer_ids": list(result.outputs[0].token_ids)}) + "\n")
            print(json.dumps({"label": label, "prompts": len(mine),
                              "answer_tokens": sum(len(r.outputs[0].token_ids) for r in results)}), flush=True)
    push(llm, None)


if __name__ == "__main__":
    main()
