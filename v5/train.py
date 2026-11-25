"""v5 trainer (Bonsai-4B, llama.cpp sampling): RL on a gated expert that lives in the shipped GGUF layout.

Run (Windows, ai-training-rocm-windows venv, from the repo root):
    python -m v5.train --run eng_pilot --topic engineering --updates 20 --compile
Each update: 1) llama-server writes --group answers for --prompts train prompts from the live GGUF (the
sampling cards); 2) Haiku ranks each prompt's answers (strict JSON schema, rl/api_judge.py, the topic rubric);
3) PyTorch (frozen Bonsai fp16 = the same +-scale weights as the 1-bit GGUF) learns the expert from the
group-relative advantages, with a KL guard; 4) the new expert bytes are patched into the GGUF and the servers
restart. Step 0 trains the per-token gate (v5/gate.py) and freezes it.
Everything is logged as JSON lines in checkpoints/v5/<run>/: log.jsonl (one row per update and per event),
samples.jsonl (best/worst answer of one prompt per update), judge.jsonl (every Haiku call). Locked test prompts
are never read. Resume: rerun the same command; steer.pt and the update counter are reloaded.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".compiler-runtime"
CLANG = Path(sys.prefix) / "Lib" / "site-packages" / "_rocm_sdk_core" / "lib" / "llvm" / "bin"
BASE_GGUF = Path("C:/llm/models/bonsai-4b/Bonsai-4B-Q1_0.gguf")
BASE_HF = Path("C:/llm/models/bonsai-4b/unpacked")
GGUF_DIR = Path("C:/llm/models/bonsai-4b/v5")
GGUF_PY = Path("C:/llm/tools/llama.cpp/gguf-py")  # llama.cpp's gguf-py at the commit the README pins (abca692)
LOSS_SCALE, MAX_GRAD_NORM, PROBE_TOKENS, MIN_JUDGED_SHARE = 4096.0, 1.0, 8, 0.5
DEV_JUDGE_ID = 100_000  # judge "update" ids for dev evals, apart from training updates


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True)
    p.add_argument("--topic", default="engineering")
    p.add_argument("--updates", type=int, default=20)
    p.add_argument("--prompts", type=int, default=8, help="train prompts per update")
    p.add_argument("--group", type=int, default=4, help="answers per prompt")
    p.add_argument("--max-new", type=int, default=700, help="answer tokens (~512 words, the judge's window)")
    p.add_argument("--dev", type=int, default=32, help="dev prompts for the learning curve")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-kl", type=float, default=0.01, help="undo and halve lr above this step KL (nats/token)")
    p.add_argument("--judge-concurrency", type=int, default=16)
    p.add_argument("--judge-budget", type=float, default=10.0, help="USD hard stop")
    p.add_argument("--gate-per-class", type=int, default=400)
    p.add_argument("--torch-hip", default="0", help="HIP index for PyTorch; HIP 0 = llama.cpp Vulkan1 (not the display)")
    p.add_argument("--sample-cards", default="Vulkan0:32:1700",
                   help="llama.cpp cards as device:slots:slot_ctx, comma separated")
    p.add_argument("--chunk-tokens", type=int, default=8000, help="tokens per gradient chunk")
    p.add_argument("--compile", action="store_true", help="torch.compile the non-steer layers (project-local Triton)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def setup_env(args) -> None:
    """Pin PyTorch to one card before torch loads; make the project-local compiler visible."""
    os.environ["HIP_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"] = args.torch_hip
    sys.path.insert(0, str(GGUF_PY))  # read from source, not installed into the training venv
    if args.compile:
        sys.path.insert(0, str(RUNTIME))
        os.environ["PATH"] = str(CLANG) + os.pathsep + os.environ["PATH"]
        os.environ.setdefault("CC", str(CLANG / "clang-cl.exe"))
        os.environ.setdefault("CXX", str(CLANG / "clang-cl.exe"))
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(RUNTIME / "inductor-cache"))


class Log:
    def __init__(self, folder: Path):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, **row) -> None:
        row = {"time": round(time.time(), 1), **row}
        with (self.folder / name).open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if name == "log.jsonl":
            print(json.dumps(row, ensure_ascii=False), flush=True)


def read_rows(topic: str, split: str) -> list[dict]:
    path = ROOT / "data_review" / "clean_v5" / topic / f"{split}.jsonl"
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def prompt_ids(tok, texts: list[str]) -> list[list[int]]:
    chat = [tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True)
            for t in texts]
    return [tok.encode(c, add_special_tokens=False) for c in chat]


def prepare(torch, model, compile_them: bool) -> None:
    """Layers before the first steer layer: no graph. Later layers: checkpointed. Non-steer layers compiled."""
    from rl.policy import checkpointed, compiled_forward, without_gradients
    from v5.steer import LAYERS

    for i, layer in enumerate(model.model.layers):
        run = compiled_forward(torch, layer.forward) if compile_them and i not in LAYERS else layer.forward
        layer.forward = without_gradients(torch, run) if i < LAYERS[0] else checkpointed(torch, run)


def train_gate(torch, model, tok, steer, pool, args, log: Log) -> dict:
    from v5 import gate

    t = time.time()
    data = gate.gate_rows(args.topic, args.gate_per_class, args.seed)
    out = {}
    for part, rows in data.items():
        ids = prompt_ids(tok, [r["prompt"] for r in rows])
        answers = pool.generate(ids, 1, gate.ANSWER_TOKENS, greedy=True)
        full = [i + a[0]["tokens"] for i, a in zip(ids, answers)]
        x, row_of, is_answer = gate.capture(model, full, [len(i) for i in ids], args.seed)
        y = torch.tensor([steer.classes.index(rows[r]["label"]) for r in row_of])
        out[part] = (x, y, is_answer)
    acc = {}
    for layer in out["fit"][0]:
        w = gate.fit_layer(out["fit"][0][layer], out["fit"][1], len(steer.classes))
        steer.gate[layer].copy_(w)
        acc[layer] = gate.accuracy(w, out["check"][0][layer], out["check"][1], out["check"][2])
    result = {"event": "gate", "seconds": round(time.time() - t), "rows": {k: len(v) for k, v in data.items()},
              "held_out_accuracy": acc}
    log.write("log.jsonl", **result)
    return result


def probe_log_probs(torch, model, probe):
    with torch.inference_mode():
        hidden = model.model(input_ids=probe, use_cache=False).last_hidden_state
        return torch.log_softmax(model.lm_head(hidden[:, -PROBE_TOKENS:, :]).float(), dim=-1)


def learn(torch, model, steer, opt, prompts, completions, advantages, args, probe) -> dict:
    from rl.policy import backward_policy_loss, build_policy_batch, kl_divergence, token_chunks

    t = time.time()
    active = [i for i, a in enumerate(advantages) if abs(a) > 1e-12]
    lengths = [len(p) + len(c) for p, c in zip(prompts, completions)]
    norm = args.prompts * args.group * args.max_new
    opt.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    loss = 0.0
    for part in token_chunks(active, lengths, args.chunk_tokens):
        batch = build_policy_batch(torch, [prompts[i] for i in part], [completions[i] for i in part], 0, "cuda:0")
        loss += backward_policy_loss(torch, model.model, model.lm_head, batch,
                                     torch.tensor([advantages[i] for i in part]), norm / LOSS_SCALE)
    routed = steer.routed_share()  # counted on the sampled prompt+answer tokens of this update
    params = steer.parameters()
    for p in params:
        if p.grad is not None:
            p.grad /= LOSS_SCALE
    grad_norm = float(torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)) if active else 0.0
    kl, rejected = 0.0, False
    if active:
        before = probe_log_probs(torch, model, probe)
        saved = [p.detach().clone() for p in params]
        opt.step()
        kl = kl_divergence(torch, before, probe_log_probs(torch, model, probe))
        if kl > args.max_kl:
            with torch.no_grad():
                for p, s in zip(params, saved):
                    p.copy_(s)
            for g in opt.param_groups:
                g["lr"] *= 0.5
            rejected = True
    steer.routed_share()  # drop the probe's counts
    return {"learn_s": round(time.time() - t, 2), "loss": round(loss / LOSS_SCALE, 6), "active": len(active),
            "grad_norm": round(grad_norm, 4), "step_kl": round(kl, 5), "rejected": rejected,
            "lr": opt.param_groups[0]["lr"], "learn_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "routed": routed, "expert_norm": {l: round(float(e["down"].detach().norm()), 4) for l, e in steer.experts.items()}}


def judge_groups(judge, topic, update, prompts, answers_by_prompt) -> tuple:
    from rl.api_judge import Group

    groups = [Group(f"{topic}:{update}:{i}", topic, prompt, tuple(ans))
              for i, (prompt, ans) in enumerate(zip(prompts, answers_by_prompt))]
    rewards, stats = judge.judge(groups, update)
    return rewards, stats.summary()


def dev_eval(pool, tok, judge, args, dev_rows, dev_ids, base_answers, step: int, log: Log) -> dict:
    t = time.time()
    answers = [a[0] for a in pool.generate(dev_ids, 1, args.max_new, greedy=True)]
    texts = [tok.decode(a["tokens"], skip_special_tokens=True) for a in answers]
    rewards, stats = judge_groups(judge, args.topic, DEV_JUDGE_ID + step, [r["prompt"] for r in dev_rows],
                                  [(b, s) for b, s in zip(base_answers, texts)])
    usable = [r for r in rewards if r is not None]
    n = max(len(usable), 1)
    wins = sum((r[1] > r[0]) + 0.5 * (r[1] == r[0]) for r in usable)
    row = {"event": "dev", "update": step, "scored": len(usable), "of": len(dev_rows),
           "base_reward": round(sum(r[0] for r in usable) / n, 4), "steered_reward": round(sum(r[1] for r in usable) / n, 4),
           "steered_beats_base_pct": round(100 * wins / n, 1),
           "mean_tokens": round(sum(len(a["tokens"]) for a in answers) / len(answers)),
           "hit_cap": sum(a["stop"] == "limit" for a in answers), "seconds": round(time.time() - t), "judge": stats}
    log.write("log.jsonl", **row)
    return row


def sample_row(rows, rewards, texts, step) -> dict:
    """Best and worst answer of the first judged prompt, for reading by eye."""
    i = next((k for k, r in enumerate(rewards) if r is not None), None)
    if i is None:
        return {"update": step, "judged": False}
    r = rewards[i]
    best, worst = max(range(len(r)), key=r.__getitem__), min(range(len(r)), key=r.__getitem__)
    return {"update": step, "prompt": rows[i]["prompt"][:400], "rewards": r,
            "best": texts[i][best][:1500], "worst": texts[i][worst][:1500]}


def main(argv=None):
    args = parse_args(argv)
    setup_env(args)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from moe import live_gguf
    from reports.answer_v5 import is_loop
    from rl.api_judge import ApiJudge
    from rl_math import group_advantages
    from v5.llama import Card, Pool
    from v5.steer import Steer

    folder = ROOT / "checkpoints" / "v5" / args.run
    log = Log(folder)
    log.write("log.jsonl", event="start", args=vars(args))
    tok = AutoTokenizer.from_pretrained(BASE_HF)
    model = AutoModelForCausalLM.from_pretrained(BASE_HF, dtype=torch.float16).to("cuda:0").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    prepare(torch, model, args.compile)
    steer = Steer(["none", args.topic], model.config.hidden_size, "cuda:0", seed=args.seed)
    steer.attach(model)
    state_path, step = folder / "steer.pt", 0
    if state_path.exists():
        saved = torch.load(state_path, weights_only=False)
        steer.load(saved["steer"])
        step = saved["update"]
    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    gguf = GGUF_DIR / f"{args.run}.gguf"
    live_gguf.build(BASE_GGUF, gguf, steer.to_moe())
    cards = [Card(d, 8130 + i, int(s), int(c)) for i, (d, s, c) in
             enumerate(part.split(":") for part in args.sample_cards.split(","))]
    slots = live_gguf.moe_slots(gguf)  # byte offsets read once; re-reading the 600 MB file cost ~10 s per update
    pool = Pool(gguf, cards, folder)
    log.write("log.jsonl", event="servers", start_s=round(pool.start(), 2), cards=[vars(c) for c in cards])
    judge = ApiJudge(folder / "judge.jsonl", run=args.run, budget_usd=args.judge_budget,
                     concurrency=args.judge_concurrency, seed=args.seed)
    try:
        if not state_path.exists():
            train_gate(torch, model, tok, steer, pool, args, log)
            live_gguf.patch(gguf, steer.to_moe(), slots)
            pool.restart()
            torch.save({"steer": steer.state(), "update": 0, "args": vars(args)}, state_path)
        steer.routed_share()
        opt = torch.optim.AdamW(steer.parameters(), lr=args.lr, weight_decay=0.0)
        train = read_rows(args.topic, "train")
        random.Random(args.seed).shuffle(train)
        dev_rows = read_rows(args.topic, "dev")[: args.dev]
        dev_ids = prompt_ids(tok, [r["prompt"] for r in dev_rows])
        base_path = folder / "dev_base.json"
        if not base_path.exists():  # The expert's down weights start at zero, so these are the base model's answers.
            base = [a[0] for a in pool.generate(dev_ids, 1, args.max_new, greedy=True)]
            base_path.write_text(json.dumps([tok.decode(a["tokens"], skip_special_tokens=True) for a in base]),
                                 encoding="utf-8")
        base_answers = json.loads(base_path.read_text(encoding="utf-8"))
        probe = torch.tensor([i[-64:] for i in dev_ids[:8] if len(i) >= 64], device="cuda:0")
        while step < args.updates:
            step += 1
            t0 = time.time()
            rows = [train[((step - 1) * args.prompts + i) % len(train)] for i in range(args.prompts)]
            ids = prompt_ids(tok, [r["prompt"] for r in rows])
            sampled = pool.generate(ids, args.group, args.max_new, seed=args.seed * 1000 + step)
            sample_s = time.time() - t0
            texts = [[tok.decode(a["tokens"], skip_special_tokens=True) for a in group] for group in sampled]
            t1 = time.time()
            rewards, jstats = judge_groups(judge, args.topic, step, [r["prompt"] for r in rows], texts)
            judge_s = time.time() - t1
            prompts, completions, advantages, flat = [], [], [], []
            for i, (group, reward) in enumerate(zip(sampled, rewards)):
                if reward is None:
                    continue
                for answer, adv in zip(group, group_advantages(reward)):
                    prompts.append(ids[i])
                    completions.append(answer["tokens"])
                    advantages.append(adv)
                flat += reward
            answers = [a for g in sampled for a in g]
            gen_tokens = sum(len(a["tokens"]) for a in answers)
            row = {"event": "update", "update": step, "sample_s": round(sample_s, 2), "judge_s": round(judge_s, 2),
                   "gen_tokens": gen_tokens, "gen_tok_s": round(gen_tokens / max(sample_s, 1e-6)),
                   "mean_answer_tokens": round(gen_tokens / len(answers)),
                   "hit_cap": sum(a["stop"] == "limit" for a in answers),
                   "loops": sum(is_loop(t) for g in texts for t in g), "judge": jstats}
            if len(prompts) < MIN_JUDGED_SHARE * args.prompts * args.group:
                log.write("log.jsonl", **row, skipped="too few judged groups")
                continue
            row |= {"reward_mean": round(sum(flat) / len(flat), 4), "reward_max": max(flat), "reward_min": min(flat)}
            row |= learn(torch, model, steer, opt, prompts, completions, advantages, args, probe)
            t2 = time.time()
            live_gguf.patch(gguf, steer.to_moe(), slots)
            row["restart_s"] = round(pool.restart(), 2)
            row["sync_s"] = round(time.time() - t2, 2)
            row["update_s"] = round(time.time() - t0, 2)
            torch.save({"steer": steer.state(), "update": step, "args": vars(args)}, state_path)
            log.write("log.jsonl", **row)
            log.write("samples.jsonl", **sample_row(rows, rewards, texts, step))
            if step % args.eval_every == 0 or step == args.updates:
                dev_eval(pool, tok, judge, args, dev_rows, dev_ids, base_answers, step, log)
    finally:
        pool.stop()
        judge.close()
    log.write("log.jsonl", event="done", gguf=str(gguf), updates=step)


if __name__ == "__main__":
    main()
