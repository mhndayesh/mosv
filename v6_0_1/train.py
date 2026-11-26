"""v6.0.1 trainer: teach the experts to write the checked "told" answers from the PLAIN prompt. Supervised, no RL.

What is learned: the same gated experts as v6 (v6/steer.py: three layers, one expert per topic, stock llama.cpp
gemma4 layout). Only the experts change; the base model stays frozen.

How it is learned: the model reads the plain prompt followed by a kept told answer, and the experts are adjusted so
that each next token of that answer becomes more likely (cross-entropy). Every answer token gives a clear signal.
v6 instead rewarded whole sampled answers with a noisy score; its updates moved like a random walk.

Progress needs no judge. Three numbers are measured on held-out answers, as loss per token (lower = more likely):
  base_ce     the plain model reading the plain prompt: where training starts;
  teacher_ce  the plain model reading the prompt WITH the rules: the level to reach;
  val_ce      the model with experts reading the plain prompt: the student.
gap_closed = (base_ce - val_ce) / (base_ce - teacher_ce): 0 = nothing learned, 1 = behaves as if it had been told.

Run (ai-training-rocm-windows venv, repo root; PyTorch uses HIP 0, the card without the display):
    python -m v6_0_1.train --run gemma_v601
Reads data_review/v6_0_1/{train,router}.jsonl (from build_data.py).
Writes checkpoints/v6_0_1/<run>/{log.jsonl, steer_XXXX.pt, steer.pt}. Test a checkpoint with evaluate.py.
"""

from __future__ import annotations

import argparse
import os
import random
import time
import zlib

from v6.train import HF, LAYERS, Log  # the Gemma weights folder, the three steer layers, the JSON-lines logger
from v6_0_1.behaviors import guide
from v6_0_1.files import DATA, RUNS, read_rows

STOP = "<turn|>"     # Gemma 4's end-of-turn token: the last target of every answer, so the model learns where to stop
MIN_ROWS = 20        # a topic with fewer kept answers than this is not trained (its tokens go to "none")
MAX_GRAD_NORM = 1.0  # gradient clipping, the same value as v6
HEAD_CHUNK = 512     # positions sent through the 262k-word output layer at a time (keeps memory flat)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True, help="name of the run folder under checkpoints/v6_0_1/")
    p.add_argument("--epochs", type=int, default=3, help="passes over the kept answers")
    p.add_argument("--batch", type=int, default=16, help="answers per optimizer step")
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate of the expert weights")
    # w2 scales how strongly a layer's experts push. It starts at 0 (model unchanged). Supervised gradients point
    # the same way step after step, so a small rate is enough; in v6 w2 only drifted because its gradient was noise.
    p.add_argument("--w2-lr", type=float, default=3e-3)
    p.add_argument("--eval-every", type=int, default=25, help="steps between held-out checks (a checkpoint is saved at each)")
    p.add_argument("--chunk-tokens", type=int, default=4096, help="rows x longest row per forward pass (v6 value: 12.5 GB peak)")
    p.add_argument("--router-per-class", type=int, default=150, help="texts per class used to fit the router")
    p.add_argument("--torch-hip", default="0", help="HIP 0 = Vulkan1, the card without the display")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def split(rows: list[dict], val_every: int = 10) -> tuple[list[dict], list[dict]]:
    """(train, held-out): about one row in `val_every` is held out, chosen by a hash of its id, so it never changes."""
    held = [r for r in rows if zlib.crc32(r["id"].encode()) % val_every == 0]
    return [r for r in rows if zlib.crc32(r["id"].encode()) % val_every != 0], held


def load_learner(torch):
    """The frozen Gemma 4 12B in 4-bit, set up for training small add-on weights: (tokenizer, model, softcap, width)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(HF)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(HF, quantization_config=quant, device_map={"": 0}, dtype=torch.bfloat16)
    for p in model.parameters():  # the base model never changes
        p.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()  # Hugging Face only checkpoints layers in train mode; in eval mode the same pass needed 31-40 GB
    cfg = getattr(model.config, "text_config", model.config)
    return tok, model, float(getattr(cfg, "final_logit_softcapping", 0.0) or 0.0), cfg.hidden_size


def pairs(tok, rows: list[dict], told: bool = False) -> list[tuple[list[int], list[int]]]:
    """(prompt tokens, answer tokens) per row. told=True puts the topic's rules in a system message (the teacher's view).

    The prompt is formatted exactly as llama.cpp formats it at test time (same chat template, thinking off;
    checked 2026-11-26: identical text). The answer ends with the stop token.
    """
    stop = tok.convert_tokens_to_ids(STOP)
    out = []
    for row in rows:
        chat = [{"role": "user", "content": row["prompt"]}]
        if told:
            chat.insert(0, {"role": "system", "content": guide(row["topic"])})
        text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        out.append((tok.encode(text, add_special_tokens=False), tok.encode(row["answer"], add_special_tokens=False) + [stop]))
    return out


def chunks(torch, items: list[tuple], chunk_tokens: int):
    """Yield padded batches of (prompt, answer) pairs, each small enough for one forward pass."""
    from rl.policy import build_policy_batch, token_chunks

    lengths = [len(p) + len(a) for p, a in items]
    for part in token_chunks(list(range(len(items))), lengths, chunk_tokens):
        yield build_policy_batch(torch, [items[i][0] for i in part], [items[i][1] for i in part], 0, "cuda:0"), len(part)


def mean_ce(torch, model, softcap: float, items: list[tuple], chunk_tokens: int) -> float:
    """Average loss per answer token over `items` (lower = the model finds these answers more likely). No gradients."""
    from v6.train import logits, text_core

    total, count = 0.0, 0
    with torch.inference_mode():
        for batch, _ in chunks(torch, items, chunk_tokens):
            hidden = text_core(model)(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                      use_cache=False).last_hidden_state
            picked = hidden[batch["rows"], batch["cols"]]  # only the positions that predict an answer token
            for start in range(0, picked.shape[0], HEAD_CHUNK):
                lp = torch.log_softmax(logits(model, picked[start:start + HEAD_CHUNK], softcap), dim=-1)
                total -= float(lp.gather(1, batch["targets"][start:start + HEAD_CHUNK].unsqueeze(1)).sum())
            count += len(batch["targets"])
    return total / count


def teach(torch, model, softcap: float, opt, steer, items: list[tuple], chunk_tokens: int) -> float:
    """One optimizer step on a batch of pairs. Returns the batch's mean loss per answer token."""
    from v6.train import policy_backward

    targets = sum(len(a) for _, a in items)
    opt.zero_grad(set_to_none=True)
    loss = 0.0
    for batch, rows in chunks(torch, items, chunk_tokens):
        # v6's memory-saving backward pass, reused. With every "advantage" set to 1 and the norm set to the number
        # of answer tokens, its loss -(advantage x log-probability) / norm is exactly the mean cross-entropy.
        loss += policy_backward(model, softcap, batch, torch.ones(rows), targets)
    torch.nn.utils.clip_grad_norm_(steer.parameters(), MAX_GRAD_NORM)
    opt.step()
    return loss


def main(argv=None):
    args = parse_args(argv)
    os.environ["HIP_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"] = args.torch_hip  # before torch is imported
    import torch

    import v6.train
    v6.train.torch = torch  # the reused v6.train helpers expect the trainer's main() to have set this module global
    from v6.steer import GemmaSteer
    from v6_0_1 import router

    log = Log(RUNS / args.run)
    rows = read_rows(DATA / "train.jsonl")
    counts = {t: sum(r["topic"] == t for r in rows) for t in sorted({r["topic"] for r in rows})}
    topics = [t for t, n in counts.items() if n >= MIN_ROWS]
    if not topics:
        raise SystemExit(f"no topic has {MIN_ROWS} kept answers in {DATA / 'train.jsonl'} (run build_data first): {counts}")
    classes = ["none", *topics]  # class 0 = "none" = the all-zero expert
    train_rows, held_rows = split([r for r in rows if r["topic"] in topics])
    log.write("log.jsonl", event="start", args=vars(args), classes=classes, kept=counts,
              train=len(train_rows), held_out=len(held_rows), layers=LAYERS)

    tok, model, softcap, width = load_learner(torch)
    steer = GemmaSteer(classes, width, LAYERS, "cuda:0", seed=args.seed)
    steer.attach(model)  # hooks at the three layers; expert strength (w2) is 0, so the model is still unchanged

    t0 = time.time()
    labeled = [{**r, "label": r["label"] if r["label"] in classes else "none"} for r in read_rows(DATA / "router.jsonl")]
    log.write("log.jsonl", event="router", **router.fit(torch, model, tok, steer, labeled, args.router_per_class, args.seed),
              seconds=round(time.time() - t0))

    fits = lambda pair: len(pair[0]) + len(pair[1]) <= args.chunk_tokens  # noqa: E731 - a longer text cannot fit one pass
    train_pairs = [p for p in pairs(tok, train_rows) if fits(p)]
    plain_view, told_view = pairs(tok, held_rows), pairs(tok, held_rows, told=True)
    usable = [i for i in range(len(held_rows)) if fits(plain_view[i]) and fits(told_view[i])]  # the same answers in both views
    held_pairs, teacher_pairs = [plain_view[i] for i in usable], [told_view[i] for i in usable]
    if not train_pairs or not held_pairs:
        raise SystemExit(f"too few kept answers to train and hold some out: {len(train_pairs)} train, {len(held_pairs)} held out")
    base_ce = mean_ce(torch, model, softcap, held_pairs, args.chunk_tokens)        # the student before any training
    teacher_ce = mean_ce(torch, model, softcap, teacher_pairs, args.chunk_tokens)  # the plain model shown the rules
    steer.routed_share()  # reset the routing counters, so the first report covers training text only
    log.write("log.jsonl", event="levels", base_ce=round(base_ce, 4), teacher_ce=round(teacher_ce, 4),
              train_pairs=len(train_pairs), held_out_pairs=len(held_pairs))

    w2s = [e["w2"] for e in steer.experts.values()]
    rest = [t for t in steer.parameters() if all(t is not w for w in w2s)]
    opt = torch.optim.AdamW([{"params": rest, "lr": args.lr}, {"params": w2s, "lr": args.w2_lr}], weight_decay=0.0)
    best = {"step": 0, "val_ce": base_ce}

    def check_and_save(step: int) -> None:
        val = mean_ce(torch, model, softcap, held_pairs, args.chunk_tokens)
        closed = (base_ce - val) / max(base_ce - teacher_ce, 1e-9)
        torch.save({"steer": steer.state(), "update": step, "val_ce": val, "base_ce": base_ce, "teacher_ce": teacher_ce,
                    "args": vars(args)}, RUNS / args.run / f"steer_{step:04d}.pt")
        if val < best["val_ce"]:
            best.update(step=step, val_ce=val)
        log.write("log.jsonl", event="eval", step=step, val_ce=round(val, 4), gap_closed=round(closed, 3),
                  routed=steer.routed_share(),
                  w2_per_dim={l: round(float(e["w2"].detach().norm()) / width ** 0.5, 4) for l, e in steer.experts.items()})

    rng, step, order = random.Random(args.seed), 0, list(range(len(train_pairs)))
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(order)
        for start in range(0, len(order), args.batch):
            step += 1
            t = time.time()
            torch.cuda.reset_peak_memory_stats()
            loss = teach(torch, model, softcap, opt, steer, [train_pairs[i] for i in order[start:start + args.batch]], args.chunk_tokens)
            log.write("log.jsonl", event="step", step=step, epoch=epoch, loss=round(loss, 4), seconds=round(time.time() - t, 2),
                      peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
            if step % args.eval_every == 0:
                check_and_save(step)
    if step % args.eval_every != 0:  # the last steps since the previous check
        check_and_save(step)
    torch.save({"steer": steer.state(), "update": step, "args": vars(args)}, RUNS / args.run / "steer.pt")
    log.write("log.jsonl", event="done", steps=step, best_step=best["step"], best_val_ce=round(best["val_ce"], 4),
              base_ce=round(base_ce, 4), teacher_ce=round(teacher_ce, 4))


if __name__ == "__main__":
    main()
