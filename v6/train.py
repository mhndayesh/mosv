"""v6 trainer (Gemma 4 12B, Windows): RL on gated experts in the gemma4 GGUF layout, llama.cpp sampling.

Run (ai-training-rocm-windows venv: transformers 5.18 has gemma4_unified, bitsandbytes 4-bit works), from the repo root:
    C:/Users/mhnda/.venvs/ai-training-rocm-windows/Scripts/python.exe -m v6.train --run gemma_pilot --updates 20
Same settings as the locked baseline: the model's recommended sampling (from its generation_config, every sampler
explicit), thinking off, llama.cpp on the display card writing answers, PyTorch (4-bit NF4 base, frozen) learning on
the other card. 7 topics + none, one expert per topic at layers 12/24/36 (v6/steer.py = moe/gemma_host.py layout).
Length is not penalized (owner rule: only relative to the base answer, or off). Haiku runs in a helper process
(v6/judge_cli.py). Logs: checkpoints/v6/<run>/log.jsonl, samples.jsonl, judge.jsonl; a checkpoint per dev check.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF = Path("C:/llm/models/gemma4-12b/hf")
BASE_GGUF = Path("C:/llm/models/gemma4-12b/gemma-4-12b-it-Q4_K_M.gguf")
GGUF_DIR = Path("C:/llm/models/gemma4-12b/v6")
GGUF_PY = Path("C:/llm/tools/llama.cpp/gguf-py")
JUDGE_PY = Path("C:/Users/mhnda/.venvs/ai-training-rocm-windows/Scripts/python.exe")
TOPICS = ("story", "teaching", "debugging", "math_solving", "explanation", "professional_email", "engineering")
CLASSES = ["none", *TOPICS]
LAYERS = (12, 24, 36)
NONE_PROMPTS = ROOT / "data_review" / "router_v2" / "prompts.jsonl"
MAX_GRAD_NORM, PROBE_TOKENS, MIN_JUDGED_SHARE, DEV_ID = 1.0, 8, 0.5, 100_000


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True)
    p.add_argument("--updates", type=int, default=20)
    p.add_argument("--prompts", type=int, default=7, help="train prompts per update (one per topic by default)")
    p.add_argument("--group", type=int, default=4)
    p.add_argument("--max-new", type=int, default=1024)
    p.add_argument("--dev-per-topic", type=int, default=6)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-3, help="3e-3 tripped the KL guard twice in the smoke run")
    # 0.1, not v5's 0.01: the learner runs in bf16, where any nonzero expert output flips some hidden values by one
    # bf16 step (~0.008 near 1.0), a fixed KL floor of ~0.03 that does not shrink with lr (smoke run 2026-11-26;
    # two identical passes give KL 0). llama.cpp keeps these activations in fp32, so the shipped model sees far less.
    p.add_argument("--max-kl", type=float, default=0.1)
    p.add_argument("--gate-per-class", type=int, default=100)
    p.add_argument("--torch-hip", default="0", help="HIP 0 = Vulkan1, the card without the display")
    p.add_argument("--sample-cards", default="Vulkan0:12:2048")
    p.add_argument("--chunk-tokens", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def recommended_sampling() -> dict:
    """The model's own generation_config, with every other llama.cpp sampler written out as off (critical rule)."""
    g = json.loads((HF / "generation_config.json").read_text(encoding="utf-8"))
    return {"temperature": float(g.get("temperature", 1.0)), "top_k": int(g.get("top_k", 0)),
            "top_p": float(g.get("top_p", 1.0)), "min_p": 0.0, "typical_p": 1.0, "repeat_penalty": 1.0,
            "presence_penalty": 0.0, "frequency_penalty": 0.0, "top_n_sigma": -1, "xtc_probability": 0.0,
            "dry_multiplier": 0.0}


class Log:
    def __init__(self, folder: Path):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, **row) -> None:
        row = {"time": round(time.time(), 1), **row}
        with (self.folder / name).open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if name == "log.jsonl":
            print(json.dumps(row, ensure_ascii=False)[:600], flush=True)


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def balanced_train(seed: int) -> list[dict]:
    """Train prompts interleaved one topic at a time, so every update sees every topic."""
    rng = random.Random(seed)
    pools = {t: read(ROOT / "data_review" / "v6" / t / "train.jsonl") for t in TOPICS}
    for rows in pools.values():
        rng.shuffle(rows)
    longest = max(len(v) for v in pools.values())
    return [pools[t][i % len(pools[t])] for i in range(longest) for t in TOPICS]


def prompt_ids(tok, texts: list[str]) -> list[list[int]]:
    chats = [tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False) for t in texts]
    return [tok.encode(c, add_special_tokens=False) for c in chats]


def judge(groups: list, update: int, folder: Path, run: str) -> tuple[list, dict]:
    job = {"update": update, "run": run, "log": str(folder / "judge.jsonl"), "groups": groups}
    out = subprocess.run([str(JUDGE_PY), str(ROOT / "v6" / "judge_cli.py")], input=json.dumps(job), capture_output=True,
                         text=True, encoding="utf-8", cwd=str(ROOT), timeout=900)
    if out.returncode != 0:
        raise RuntimeError(f"judge failed: {out.stderr[-800:]}")
    result = json.loads(out.stdout.strip().splitlines()[-1])
    return result["rewards"], result["stats"]


def text_core(model):
    return model.model.language_model if hasattr(model.model, "language_model") else model.model


def logits(model, hidden, softcap: float):
    out = model.lm_head(hidden).float()
    return torch.tanh(out / softcap) * softcap if softcap else out


def policy_backward(model, softcap, batch, advantages, norm, head_chunk: int = 512) -> float:
    hidden = text_core(model)(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                              use_cache=False).last_hidden_state
    selected = hidden[batch["rows"], batch["cols"]]
    proxy = selected.detach().requires_grad_(True)
    weights = advantages.to(proxy.device, torch.float32)[batch["owners"]] / norm
    total = 0.0
    for start in range(0, proxy.shape[0], head_chunk):
        idx = torch.arange(start, min(start + head_chunk, proxy.shape[0]), device=proxy.device)
        lp = torch.log_softmax(logits(model, proxy[idx], softcap), dim=-1)
        loss = -(weights[idx] * lp.gather(1, batch["targets"][idx].unsqueeze(1)).squeeze(1)).sum()
        loss.backward()
        total += float(loss.detach())
    selected.backward(proxy.grad)
    return total


def probe_lp(model, softcap, probe):
    with torch.inference_mode():
        hidden = text_core(model)(input_ids=probe, use_cache=False).last_hidden_state
        return torch.log_softmax(logits(model, hidden[:, -PROBE_TOKENS:, :], softcap), dim=-1)


def train_gate(model, tok, steer, pool, args, log: Log) -> None:
    """Supervised per-token gate (7 topics + none) on prompts + short base answers; held-out accuracy logged."""
    from v6.steer import rms

    rng = random.Random(args.seed)
    none = [r for r in read(NONE_PROMPTS) if r["label"] == "none"]
    rows = {"fit": [], "check": []}
    for c in CLASSES:
        tr = [r["text"] for r in none if r["split"] == "train"] if c == "none" else [r["prompt"] for r in read(ROOT / "data_review" / "v6" / c / "train.jsonl")]
        dv = [r["text"] for r in none if r["split"] == "val"] if c == "none" else [r["prompt"] for r in read(ROOT / "data_review" / "v6" / c / "dev.jsonl")]
        rows["fit"] += [(c, t) for t in rng.sample(tr, min(args.gate_per_class, len(tr)))]
        rows["check"] += [(c, t) for t in rng.sample(dv, min(args.gate_per_class // 5, len(dv)))]
    grabbed = {}
    hooks = [text_core(model).layers[l].pre_feedforward_layernorm.register_forward_pre_hook(
        lambda _m, a, l=l: grabbed.__setitem__(l, a[0][0])) for l in LAYERS]
    data = {}
    t0 = time.time()
    for part, items in rows.items():
        ids = prompt_ids(tok, [t for _, t in items])
        answers = pool.generate(ids, 1, 96, greedy=True)
        xs, ys, ans = {l: [] for l in LAYERS}, [], []
        with torch.inference_mode():
            for (c, _), i, a in zip(items, ids, answers):
                full = i + a[0]["tokens"]
                text_core(model)(input_ids=torch.tensor([full], device="cuda:0"), use_cache=False)
                pos = sorted(rng.sample(range(len(full)), min(48, len(full))))
                for l in LAYERS:
                    xs[l].append((rms(grabbed[l].float())[pos] * steer.width ** -0.5).half().cpu())
                ys += [CLASSES.index(c)] * len(pos)
                ans += [p >= len(i) for p in pos]
        data[part] = ({l: torch.cat(v) for l, v in xs.items()}, torch.tensor(ys), torch.tensor(ans))
    for h in hooks:
        h.remove()
    acc = {}
    for l in LAYERS:
        x, y = data["fit"][0][l].cuda().float(), data["fit"][1].cuda()
        w = torch.zeros(len(CLASSES), steer.width, device="cuda", requires_grad=True)
        weight = (len(y) / (len(CLASSES) * torch.bincount(y, minlength=len(CLASSES)).clamp_min(1))).float()
        opt = torch.optim.Adam([w], lr=1e-2)
        for _ in range(500):
            loss = torch.nn.functional.cross_entropy(x @ w.T, y, weight=weight)
            opt.zero_grad(); loss.backward(); opt.step()
        steer.gate[l].copy_(w.detach())
        xc, yc, ac = data["check"][0][l].cuda().float(), data["check"][1], data["check"][2]
        pred = (xc @ w.detach().T).argmax(-1).cpu()
        acc[l] = {"all": round(float((pred == yc).float().mean()), 4), "answer_tokens": round(float((pred[ac] == yc[ac]).float().mean()), 4),
                  "none_kept_none": round(float((pred[yc == 0] == 0).float().mean()), 4)}
    log.write("log.jsonl", event="gate", seconds=round(time.time() - t0), rows={k: len(v) for k, v in rows.items()},
              held_out_accuracy=acc)


def learn(model, softcap, steer, opt, prompts, completions, advantages, args, probe) -> dict:
    from rl.policy import build_policy_batch, kl_divergence, token_chunks

    t = time.time()
    active = [i for i, a in enumerate(advantages) if abs(a) > 1e-12]
    lengths = [len(p) + len(c) for p, c in zip(prompts, completions)]
    norm = args.prompts * args.group * args.max_new
    opt.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    loss = 0.0
    for part in token_chunks(active, lengths, args.chunk_tokens):
        batch = build_policy_batch(torch, [prompts[i] for i in part], [completions[i] for i in part], 0, "cuda:0")
        loss += policy_backward(model, softcap, batch, torch.tensor([advantages[i] for i in part]), norm)
    routed = steer.routed_share()
    params = steer.parameters()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)) if active else 0.0
    kl, rejected = 0.0, False
    if active:
        before = probe_lp(model, softcap, probe)
        saved = [p.detach().clone() for p in params]
        opt.step()
        kl = kl_divergence(torch, before, probe_lp(model, softcap, probe))
        if kl > args.max_kl:
            with torch.no_grad():
                for p, s in zip(params, saved):
                    p.copy_(s)
            for g in opt.param_groups:
                g["lr"] *= 0.5
            rejected = True
    steer.routed_share()
    return {"learn_s": round(time.time() - t, 2), "loss": round(loss, 6), "active": len(active),
            "grad_norm": round(grad_norm, 4), "step_kl": round(kl, 5), "rejected": rejected,
            "lr": opt.param_groups[0]["lr"], "learn_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "routed": routed, "w2_norm": {l: round(float(e["w2"].detach().norm()), 4) for l, e in steer.experts.items()}}


def dev_eval(pool, tok, args, dev_rows, dev_ids, base_texts, step, folder, log) -> None:
    t = time.time()
    answers = [a[0] for a in pool.generate(dev_ids, 1, args.max_new, seed=777)]
    texts = [tok.decode(a["tokens"], skip_special_tokens=True) for a in answers]
    groups = [[f"dev:{step}:{i}", r["behavior"], r["prompt"], [b, s]] for i, (r, b, s) in enumerate(zip(dev_rows, base_texts, texts))]
    rewards, stats = judge(groups, DEV_ID + step, folder, args.run)
    usable = [(r, row["behavior"]) for r, row in zip(rewards, dev_rows) if r is not None]
    n = max(len(usable), 1)
    wins = sum((r[1] > r[0]) + 0.5 * (r[1] == r[0]) for r, _ in usable)
    per = {}
    for topic in TOPICS:
        rs = [r for r, tp in usable if tp == topic]
        per[topic] = round(100 * sum((r[1] > r[0]) + 0.5 * (r[1] == r[0]) for r in rs) / max(len(rs), 1), 1)
    log.write("log.jsonl", event="dev", update=step, scored=len(usable), of=len(dev_rows),
              steered_beats_base_pct=round(100 * wins / n, 1), per_topic_beats_pct=per,
              base_reward=round(sum(r[0] for r, _ in usable) / n, 4), steered_reward=round(sum(r[1] for r, _ in usable) / n, 4),
              mean_tokens=round(sum(len(a["tokens"]) for a in answers) / len(answers)), seconds=round(time.time() - t),
              judge=stats)


def main(argv=None):
    global torch
    args = parse_args(argv)
    os.environ["HIP_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"] = args.torch_hip
    sys.path.insert(0, str(GGUF_PY))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from moe import gemma_host
    from rl_math import group_advantages
    from v5.llama import Card, Pool
    from v6.steer import GemmaSteer

    folder = ROOT / "checkpoints" / "v6" / args.run
    log = Log(folder)
    sampling = recommended_sampling()
    log.write("log.jsonl", event="start", args=vars(args), sampling=sampling, thinking="off", layers=LAYERS)
    tok = AutoTokenizer.from_pretrained(HF)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(HF, quantization_config=quant, device_map={"": 0}, dtype=torch.bfloat16)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()  # HF only checkpoints layers in train mode; eval kept all 48 layers' activations (31-40 GB, spilled)
    cfg = getattr(model.config, "text_config", model.config)
    softcap = float(getattr(cfg, "final_logit_softcapping", 0.0) or 0.0)
    steer = GemmaSteer(CLASSES, cfg.hidden_size, LAYERS, "cuda:0", seed=args.seed)
    steer.attach(model)
    state_path, step = folder / "steer.pt", 0
    if state_path.exists():
        saved = torch.load(state_path, weights_only=False)
        steer.load(saved["steer"])
        step = saved["update"]
    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    gguf = GGUF_DIR / f"{args.run}.gguf"
    layers = steer.gguf_layers()
    gemma_host.build(BASE_GGUF, gguf, layers, CLASSES, steer.ff)
    where = gemma_host.slots(gguf, layers)
    cards = [Card(d, 8190 + i, int(s), int(c)) for i, (d, s, c) in enumerate(x.split(":") for x in args.sample_cards.split(","))]
    pool = Pool(gguf, cards, folder, top_k=sampling["top_k"], sampling=sampling, kv="q8_0")
    log.write("log.jsonl", event="servers", start_s=round(pool.start(), 2), model_s=round(time.time()), peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
    try:
        if not state_path.exists():
            train_gate(model, tok, steer, pool, args, log)
            gemma_host.patch(gguf, steer.gguf_layers(), where)
            pool.restart()
            torch.save({"steer": steer.state(), "update": 0, "args": vars(args)}, state_path)
        steer.routed_share()
        opt = torch.optim.AdamW(steer.parameters(), lr=args.lr, weight_decay=0.0)
        resumed = torch.load(state_path, weights_only=False)
        if "optimizer" in resumed:
            opt.load_state_dict(resumed["optimizer"])
            for g in opt.param_groups:  # a resume may raise or lower the lr on purpose (--lr wins over the saved one)
                g["lr"] = args.lr
        train = balanced_train(args.seed)
        dev_rows = [r for t in TOPICS for r in read(ROOT / "data_review" / "v6" / t / "dev.jsonl")[: args.dev_per_topic]]
        dev_ids = prompt_ids(tok, [r["prompt"] for r in dev_rows])
        base_path = folder / "dev_base.json"
        if not base_path.exists():  # step 0: w2 = 0, so the GGUF answers exactly like the base model
            base = [a[0] for a in pool.generate(dev_ids, 1, args.max_new, seed=777)]
            base_path.write_text(json.dumps([tok.decode(a["tokens"], skip_special_tokens=True) for a in base]), encoding="utf-8")
        base_texts = json.loads(base_path.read_text(encoding="utf-8"))
        probe = torch.tensor([i[-64:] for i in dev_ids[:8] if len(i) >= 64], device="cuda:0")
        while step < args.updates:
            step += 1
            t0 = time.time()
            rows = [train[((step - 1) * args.prompts + i) % len(train)] for i in range(args.prompts)]
            ids = prompt_ids(tok, [r["prompt"] for r in rows])
            sampled = pool.generate(ids, args.group, args.max_new, seed=args.seed * 1000 + step)
            sample_s = time.time() - t0
            texts = [[tok.decode(a["tokens"], skip_special_tokens=True) for a in g] for g in sampled]
            t1 = time.time()
            rewards, jstats = judge([[f"t:{step}:{i}", r["behavior"], r["prompt"], tx] for i, (r, tx) in enumerate(zip(rows, texts))],
                                    step, folder, args.run)
            judge_s = time.time() - t1
            prompts, completions, advantages, flat = [], [], [], []
            for i, (g, rw) in enumerate(zip(sampled, rewards)):
                if rw is None or any(a["stop"] == "error" for a in g):
                    continue
                for a, adv in zip(g, group_advantages(rw)):
                    prompts.append(ids[i]); completions.append(a["tokens"]); advantages.append(adv)
                flat += rw
            answers = [a for g in sampled for a in g]
            gen = sum(len(a["tokens"]) for a in answers)
            row = {"event": "update", "update": step, "sample_s": round(sample_s, 2), "judge_s": round(judge_s, 2),
                   "gen_tok_s": round(gen / max(sample_s, 1e-6)), "mean_answer_tokens": round(gen / len(answers)),
                   "hit_cap": sum(a["stop"] == "limit" for a in answers), "sample_errors": sum(a["stop"] == "error" for a in answers),
                   "topics": [r["behavior"] for r in rows], "judge": jstats}
            if len(prompts) < MIN_JUDGED_SHARE * args.prompts * args.group:
                log.write("log.jsonl", **row, skipped="too few judged groups")
                continue
            row |= {"reward_mean": round(sum(flat) / len(flat), 4), "reward_max": max(flat), "reward_min": min(flat)}
            row |= learn(model, softcap, steer, opt, prompts, completions, advantages, args, probe)
            t2 = time.time()
            gemma_host.patch(gguf, steer.gguf_layers(), where)
            row["restart_s"] = round(pool.restart(), 2)
            row["sync_s"] = round(time.time() - t2, 2)
            row["update_s"] = round(time.time() - t0, 2)
            torch.save({"steer": steer.state(), "update": step, "args": vars(args), "optimizer": opt.state_dict()}, state_path)
            log.write("log.jsonl", **row)
            i = next((k for k, r in enumerate(rewards) if r), None)
            if i is not None:
                r = rewards[i]
                log.write("samples.jsonl", update=step, topic=rows[i]["behavior"], prompt=rows[i]["prompt"][:400], rewards=r,
                          best=texts[i][max(range(len(r)), key=r.__getitem__)][:1500], worst=texts[i][min(range(len(r)), key=r.__getitem__)][:1500])
            if step % args.eval_every == 0 or step == args.updates:
                torch.save({"steer": steer.state(), "update": step, "args": vars(args)}, folder / f"steer_{step:04d}.pt")
                dev_eval(pool, tok, args, dev_rows, dev_ids, base_texts, step, folder, log)
    finally:
        pool.stop()
    log.write("log.jsonl", event="done", gguf=str(gguf), updates=step)


if __name__ == "__main__":
    main()
