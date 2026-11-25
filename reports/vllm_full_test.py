"""Routed vs normal answers on vLLM: paged KV cache in fp8, long context, answers run until the model stops.

Run in WSL: python reports/vllm_full_test.py --tp 2 --max-model-len 262144
--probe only starts the engine and prints the KV cache capacity.

Reuses the router's picks from compare_4b.json: same 86 prompts, same trained router, so only the writing changes.
Steering is one vector per engine, so normal answers run all at once (the concurrency test) and steered answers
run one route at a time.
"""

import argparse
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # Engine in this process, so its stats are readable.

from model_profiles import get_profile  # noqa: E402
from rl.vllm_sampler import add_plugin_path, prepare_environment  # noqa: E402
from system import NONE, load_vectors  # noqa: E402

REPORTS = ROOT / "data_review" / "reports"
SAMPLE_EVERY = 25  # Engine steps between stat samples.


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", default="qwen4b_haiku_v3_test")
    p.add_argument("--source", type=Path, default=REPORTS / "compare_4b.json", help="prompts and router picks")
    p.add_argument("--out", type=Path, default=REPORTS / "vllm_full_test.json")
    p.add_argument("--tp", type=int, default=1, help="cards one engine spans")
    p.add_argument("--max-model-len", type=int, default=262144)
    p.add_argument("--kv", default="fp8")
    p.add_argument("--util", type=float, default=0.90)
    p.add_argument("--max-seqs", type=int, default=256)
    p.add_argument("--batched", type=int, default=8192, help="max tokens per engine step")
    p.add_argument("--stress", type=int, default=0, help="extra copies of the normal pass, to fill the cache")
    p.add_argument("--eager", action="store_true", help="skip CUDA graphs: less VRAM, slower")
    p.add_argument("--kv-gib", type=float, default=None, help="fixed KV cache per card; overrides --util's split")
    p.add_argument("--probe", action="store_true")
    return p.parse_args(argv)


def build_engine(args, profile):
    prepare_environment()
    from qwen_steering import ARCH, LAYERS_KEY
    from vllm import LLM

    fixed = {"kv_cache_memory_bytes": int(args.kv_gib * 2**30)} if args.kv_gib else {}
    return LLM(model=profile.path, dtype="float16", max_model_len=args.max_model_len, kv_cache_dtype=args.kv, **fixed,
               tensor_parallel_size=args.tp, gpu_memory_utilization=args.util, max_num_seqs=args.max_seqs,
               max_num_batched_tokens=args.batched, enable_prefix_caching=True, disable_log_stats=False,
               enforce_eager=args.eager,
               hf_overrides={"architectures": [ARCH], LAYERS_KEY: list(profile.steering_layers)})


def capacity(llm, args) -> dict:
    cache = llm.llm_engine.vllm_config.cache_config
    tokens = (cache.num_gpu_blocks or 0) * cache.block_size
    return {"kv_tokens": tokens, "block_size": cache.block_size, "kv_dtype": args.kv, "tp": args.tp,
            "max_model_len": args.max_model_len, "full_len_requests_at_once": round(tokens / args.max_model_len, 2)}


def stats(llm) -> dict:
    try:
        found = {m.name: getattr(m, "value", None) for m in llm.get_metrics()}
    except Exception as error:  # Stats are a report extra; the answers still count without them.
        return {"error": str(error)}
    keys = ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc",
            "vllm:num_preemptions")
    return {k.split(":")[1]: found.get(k) for k in keys}


def is_looping(tokens, gram: int = 20, floor: float = 0.5) -> bool:
    """True when the recent tokens are mostly one block repeated: few distinct 20-token pieces.
    A loop of period P leaves about P/window of the pieces distinct, so each window catches loops up to half its
    length; ordinary text stays near 1.0. Windows 2000 and 6000 catch short and long loops."""
    if len(tokens) % 128:
        return False
    for window in (2000, 6000):
        if len(tokens) >= window:
            tail = tuple(tokens[-window:])
            if len({tail[i:i + gram] for i in range(window - gram)}) / (window - gram) < floor:
                return True
    return False


ABORT_FILE = REPORTS / "abort.txt"  # Request ids written here are stopped at once (reason "manual").
PEEK_FILE = REPORTS / "peek.json"  # Latest text of the last few running answers, to check them by eye.
PEEK_WHEN = 5


def manual_stops(engine, latest: dict, done: dict) -> None:
    """Stop the running requests whose ids are listed in ABORT_FILE, keeping their text so far."""
    if not ABORT_FILE.exists():
        return
    wanted = set(ABORT_FILE.read_text(encoding="utf-8").split())
    for rid in [r for r in latest if r in wanted]:
        engine.abort_request([rid])
        done[int(rid)] = {"tokens": list(latest.pop(rid)), "reason": "manual"}


def run_batch(llm, prompts: list[list[int]], args, eos) -> tuple[list[dict], dict]:
    """All prompts at once, each allowed to the full context. Returns answers and the batch's peak stats."""
    from vllm import SamplingParams

    engine, done, peak = llm.llm_engine, {}, {"running": 0, "kv_usage": 0.0, "preemptions": 0}
    ABORT_FILE.unlink(missing_ok=True)  # Request ids restart at 0 in every batch, so old stops must not carry over.
    for i, ids in enumerate(prompts):
        params = SamplingParams(temperature=0.0, max_tokens=args.max_model_len - len(ids),
                                stop_token_ids=sorted(eos), detokenize=False)
        engine.add_request(str(i), {"prompt_token_ids": ids}, params)
    start, steps, latest = time.time(), 0, {}
    while engine.has_unfinished_requests():
        for out in engine.step():
            tokens = out.outputs[0].token_ids
            if out.finished:
                done[int(out.request_id)] = {"tokens": list(tokens), "reason": out.outputs[0].finish_reason}
                latest.pop(out.request_id, None)
            elif is_looping(tokens):  # Stuck answers would otherwise run to the full context.
                engine.abort_request([out.request_id])
                done[int(out.request_id)] = {"tokens": list(tokens), "reason": "repeat"}
                latest.pop(out.request_id, None)
            else:
                latest[out.request_id] = tokens
        steps += 1
        if steps % SAMPLE_EVERY == 0:
            manual_stops(engine, latest, done)
            if len(latest) <= PEEK_WHEN:
                tok = llm.get_tokenizer()
                PEEK_FILE.write_text(json.dumps({rid: {"tokens": len(t), "tail": tok.decode(list(t)[-1500:])}
                                                 for rid, t in latest.items()}, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
            s = stats(llm)
            peak["running"] = max(peak["running"], int(s.get("num_requests_running") or 0))
            peak["kv_usage"] = max(peak["kv_usage"], float(s.get("kv_cache_usage_perc") or 0))
            peak["preemptions"] = int(s.get("num_preemptions") or 0)
            print(json.dumps({"t": round(time.time() - start), "left": len(prompts) - len(done), **s}), flush=True)
    seconds = time.time() - start
    made = sum(len(d["tokens"]) for d in done.values())
    peak.update(seconds=round(seconds, 1), requests=len(prompts), answer_tokens=made,
                tok_per_s=round(made / seconds, 1), longest=max(len(d["tokens"]) for d in done.values()),
                hit_limit=sum(d["reason"] == "length" for d in done.values()),
                stopped_repeating=sum(d["reason"] == "repeat" for d in done.values()),
                stopped_by_hand=sum(d["reason"] == "manual" for d in done.values()))
    return [done[i] for i in range(len(prompts))], peak


def progress_path(args) -> Path:
    return args.out.with_suffix(f".partial_{args.max_model_len}_{args.run}.json")


def load_progress(args) -> dict:
    path = progress_path(args)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_progress(args, saved: dict) -> None:
    path = progress_path(args)
    path.with_suffix(".tmp").write_text(json.dumps(saved), encoding="utf-8")
    path.with_suffix(".tmp").replace(path)  # Never leave a half-written file behind.


def push(llm, vector) -> None:
    add_plugin_path()
    from qwen_steering import set_steering

    llm.apply_model(partial(set_steering, vectors=None if vector is None else vector.numpy()))
    llm.reset_prefix_cache()


def main():
    args = parse_args()
    profile = get_profile("qwen4b")
    llm = build_engine(args, profile)
    cap = capacity(llm, args)
    print(json.dumps({"capacity": cap}), flush=True)
    if args.probe:
        return
    tok = llm.get_tokenizer()
    rows = json.loads(args.source.read_text(encoding="utf-8"))
    chat = [tok.encode(tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False),
                       add_special_tokens=False) for r in rows]  # Same text as system._chat, as plain id lists.
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    text = partial(tok.decode, skip_special_tokens=True)

    saved = load_progress(args)  # Finished passes from an earlier run of the same settings are reused.
    if "normal" not in saved:
        push(llm, None)
        saved["normal"] = run_batch(llm, chat * (1 + args.stress), args, eos)
        save_progress(args, saved)
    normal, normal_stats = saved["normal"]
    routes = sorted({r["route"] for r in rows} - {NONE})
    vectors, _ = load_vectors(args.run, tuple(routes))
    steered, route_stats = {}, {}
    for route, vector in zip(routes, vectors):
        mine = [i for i, r in enumerate(rows) if r["route"] == route]
        if route not in saved.setdefault("steered", {}):
            push(llm, vector)
            saved["steered"][route] = run_batch(llm, [chat[i] for i in mine], args, eos)
            save_progress(args, saved)
        answers, route_stats[route] = saved["steered"][route]
        steered.update(zip(mine, answers))
    push(llm, None)

    out = [{**{k: r[k] for k in ("id", "type", "prompt", "route", "confidence")},
            "normal": text(normal[i]["tokens"]).strip(), "normal_tokens": len(normal[i]["tokens"]),
            "normal_end": normal[i]["reason"], "steered_end": (steered.get(i) or normal[i])["reason"],
            "steered": text(steered[i]["tokens"]).strip() if i in steered else text(normal[i]["tokens"]).strip(),
            "steered_tokens": len(steered[i]["tokens"]) if i in steered else len(normal[i]["tokens"])}
           for i, r in enumerate(rows)]
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    summary = {"capacity": cap, "normal_pass": normal_stats, "steered_by_route": route_stats}
    args.out.with_suffix(".stats.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({"done": len(out), "normal_pass": normal_stats}), flush=True)


if __name__ == "__main__":
    main()
