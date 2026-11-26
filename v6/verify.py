"""Check that trained v6 experts act the same in PyTorch (the learner) and in llama.cpp (the sampler / shipped GGUF).

Run after training, cards free (ai-training-rocm-windows venv), from the repo root:
    python -m v6.verify --run gemma_w2
    python -m v6.verify --version v6_0_1 --run gemma_v601 --checkpoint steer_0100.pt
For a few dev prompts it takes the base GGUF's greedy answer (--tokens long) and scores that same token sequence four
ways: llama.cpp base, llama.cpp with experts, PyTorch base (w2 = 0), PyTorch with experts. Reported:
  floor      mean |logprob difference| PyTorch vs llama.cpp on the base model (4-bit NF4 vs Q4_K_M: expected mismatch)
  effect     mean |logprob change| the experts cause, in each engine
  agreement  correlation of that change between the engines (near 1 = same experts; near 0 = the learner trains a
             different function than the one llama.cpp samples from)
Also prints whether llama.cpp returns the stop token with an answer. Saves checkpoints/v6/<run>/verify.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT, TOP = 8195, 20


def server(model: Path, device: str, log: Path) -> subprocess.Popen:
    from v5.llama import SERVER
    cmd = [SERVER, "-m", str(model), "-dev", device, "-ngl", "99", "--port", str(PORT), "-fa", "on", "-c", "4096",
           "-np", "1", "--no-webui", "--no-jinja", "-ctk", "q8_0", "-ctv", "q8_0"]
    proc = subprocess.Popen(cmd, stdout=log.open("a", encoding="utf-8"), stderr=subprocess.STDOUT)
    for _ in range(600):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1)
            return proc
        except Exception:
            time.sleep(0.2)
    proc.terminate()
    raise SystemExit(f"llama-server did not start; see {log}")


def complete(prompt: list[int], n: int) -> dict:
    body = {"prompt": prompt, "n_predict": n, "return_tokens": True, "cache_prompt": True, "temperature": 0.0,
            "top_k": 1, "n_probs": TOP, "post_sampling_probs": False}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())


def tops(out: dict) -> list[dict[int, float]]:
    """Per generated token: {token id: logprob} for the top TOP tokens (before sampling)."""
    return [{int(t["id"]): float(t["logprob"]) for t in step["top_logprobs"]} for step in out["completion_probabilities"]]


def score(prompt: list[int], seq: list[int]) -> list[dict[int, float]]:
    """llama.cpp's top-TOP logprobs at every position of a fixed sequence (one 1-token request per position)."""
    return [tops(complete(prompt + seq[:t], 1))[0] for t in range(len(seq))]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True)
    p.add_argument("--version", default="v6", help="folder under checkpoints/: v6 or v6_0_1")
    p.add_argument("--checkpoint", default="steer.pt")
    p.add_argument("--per-topic", type=int, default=2)
    p.add_argument("--tokens", type=int, default=48)
    p.add_argument("--torch-hip", default="0")
    p.add_argument("--llama-card", default="Vulkan0")
    args = p.parse_args(argv)
    os.environ["HIP_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"] = args.torch_hip
    from v6.train import BASE_GGUF, GGUF_DIR, GGUF_PY, HF, LAYERS, TOPICS, logits, prompt_ids, read, text_core
    sys.path.insert(0, str(GGUF_PY))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    import v6.train
    v6.train.torch = torch  # v6.train.logits uses the module global that the trainer's main() sets
    from moe import gemma_host
    from v6.steer import GemmaSteer

    folder = ROOT / "checkpoints" / args.version / args.run
    saved = torch.load(folder / args.checkpoint, weights_only=False)
    classes = saved["steer"]["classes"]  # read from the checkpoint: v6 has 7 topics, v6.0.1 only the trained ones
    tok = AutoTokenizer.from_pretrained(HF)
    rows = [r for t in TOPICS for r in read(ROOT / "data_review" / "v6" / t / "dev.jsonl")[: args.per_topic]]
    prompts = prompt_ids(tok, [r["prompt"] for r in rows])

    cpu = GemmaSteer(classes, saved["steer"]["gate"][LAYERS[0]].shape[1], LAYERS, "cpu")
    cpu.load(saved["steer"])
    gguf = GGUF_DIR / f"{args.run}_verify.gguf"
    gemma_host.build(BASE_GGUF, gguf, cpu.gguf_layers(), classes, cpu.ff)
    log = folder / "verify_llama.log"

    proc = server(BASE_GGUF, args.llama_card, log)
    try:
        outs = [complete(pr, args.tokens) for pr in prompts]
        seqs = [o["tokens"] for o in outs]
        llama_base = [tops(o) for o in outs]
        short = complete(prompt_ids(tok, ["Reply with the single word: hello"])[0], 32)
    finally:
        proc.terminate(); proc.wait()
    proc = server(gguf, args.llama_card, log)
    try:
        llama_steer = [score(pr, s) for pr, s in zip(prompts, seqs)]
    finally:
        proc.terminate(); proc.wait()
    gguf.unlink()

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(HF, quantization_config=quant, device_map={"": 0}, dtype=torch.bfloat16)
    model.eval()
    cfg = getattr(model.config, "text_config", model.config)
    softcap = float(getattr(cfg, "final_logit_softcapping", 0.0) or 0.0)
    steer = GemmaSteer(classes, cfg.hidden_size, LAYERS, "cuda:0")
    steer.load(saved["steer"])
    steer.attach(model)
    trained = {l: e["w2"].detach().clone() for l, e in steer.experts.items()}

    def torch_lp(scale: float) -> list:
        with torch.inference_mode():
            for l, e in steer.experts.items():
                e["w2"].copy_(trained[l] * scale)
            res = []
            for pr, s in zip(prompts, seqs):
                hidden = text_core(model)(input_ids=torch.tensor([pr + s[:-1]], device="cuda:0"), use_cache=False).last_hidden_state
                res.append(torch.log_softmax(logits(model, hidden[0, len(pr) - 1:], softcap), dim=-1).float().cpu())
            return res

    pt_base, pt_steer = torch_lp(0.0), torch_lp(1.0)
    routed = steer.routed_share()

    floor, d_llama, d_pt, top1_base, top1_steer = [], [], [], [], []
    for i, s in enumerate(seqs):
        for t in range(len(s)):
            lb, ls = llama_base[i][t], llama_steer[i][t]
            floor += [abs(float(pt_base[i][t, k]) - v) for k, v in lb.items()]
            top1_base.append(int(pt_base[i][t].argmax()) == max(lb, key=lb.get))
            top1_steer.append(int(pt_steer[i][t].argmax()) == max(ls, key=ls.get))
            for k in lb.keys() & ls.keys():
                d_llama.append(ls[k] - lb[k])
                d_pt.append(float(pt_steer[i][t, k] - pt_base[i][t, k]))
    a, b = torch.tensor(d_llama), torch.tensor(d_pt)
    report = {"run": args.run, "checkpoint": args.checkpoint, "update": saved["update"], "prompts": len(prompts),
              "positions": sum(len(s) for s in seqs), "compared_values": len(d_llama),
              "floor_mean_abs_logprob_diff_base": round(sum(floor) / len(floor), 4),
              "top1_agree_base": round(sum(top1_base) / len(top1_base), 4),
              "top1_agree_steered": round(sum(top1_steer) / len(top1_steer), 4),
              "effect_mean_abs_llama": round(float(a.abs().mean()), 4), "effect_mean_abs_torch": round(float(b.abs().mean()), 4),
              "effect_agreement_corr": round(float(torch.corrcoef(torch.stack([a, b]))[0, 1]), 4),
              "w2_per_dim": {l: round(float(w.norm()) / w.numel() ** 0.5, 4) for l, w in trained.items()},
              "routed_torch": routed,
              "stop_token_returned": {"stop_type": short.get("stop_type"), "last_tokens": tok.convert_ids_to_tokens(short["tokens"][-3:]),
                                      "eos": tok.eos_token, "n": len(short["tokens"])}}
    (folder / "verify.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
