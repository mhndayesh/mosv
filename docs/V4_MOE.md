# v4: the steering as a real MoE layer that stock llama.cpp runs (branch `v4-moe`)

Goal (owner, 2026-11-25): one GGUF that llama.cpp runs with no custom code and no proxy, by reusing an MoE
architecture llama.cpp already has. v3 (tag `v3-control-vectors`) stays as it is.

## Host architecture: `hy_v3` (Hunyuan v3), checked in llama.cpp source and the local build b10269
- Same attention as Qwen3: separate q/k/v, per-head q/k RMSNorm before RoPE, NEOX RoPE, GQA, RMSNorm blocks.
- Each layer is dense or MoE: MoE when `blk.N.ffn_gate_inp` exists. MoE layers add an always-on shared expert.
- Gating: sigmoid + top-k with weights normalised; with top-1 the picked expert is added with weight exactly 1.
- **Feasibility check (2026-11-25):** Qwen3-4B rewritten as hy_v3 with MoE at layers 9/18/27 (shared expert = the
  original MLP, 9 all-zero experts) gives **5/5 identical greedy answers** to the qwen3 GGUF, at the same speed
  (18.6 vs 18.8 tok/s in that run). So the host is exact; only the experts and gates are new.

## Design
At layers 9, 18, 27 (the v3 gaps): `out = MLP(x) + expert_k(x)`, `k = argmax gate(x)`, per token.
- Expert 0 = zeros = "none". Experts 1-8 = small SwiGLU MLPs (hidden 128), one per topic.
- Gate = linear 2560 -> 9 on the layer's normed MLP input.

Differences from v3, and the risks they bring:
1. **v3 adds a constant vector; an MoE expert is an MLP.** The experts are trained to output the v3 vector
   for every input (the MLP input is RMS-normalised, so this is learnable). Measured as relative error per expert.
2. **v3 routes once per prompt; MoE routes every token at every MoE layer.** The gates are trained on every
   token of a prompt + v3 answer, labelled with the prompt's topic. Risk: the topic flips mid-answer. Measured as
   per-token gate accuracy on held-out rows, split into prompt and answer tokens.

## Steps (code in `moe/`)
1. `moe/make_data.py` (WSL, vLLM, one card): 200 train + 40 val prompts per topic, 400 + 80 "none", from the
   router's labelled set; each answered greedily (512 tokens) with its v3 vector.
2. `moe/train.py` (WSL, one card): collect the MLP inputs at 9/18/27 (96 tokens per row, v3 vector on), fit
   gates (cross-entropy) and experts (MSE to the v3 vector).
3. `moe/export.py` (gguf-tools venv): one `hy_v3` GGUF.
4. Test in stock llama-server: route agreement, openings against the v3 control-vector answers, then the
   86-prompt blind Haiku test against the normal answers (`reports/score_full.py`).
5. If the distilled version loses quality: RL fine-tune the experts and gates with the v3 recipe.

## Results (2026-11-25, all steps run)

**Data:** 2,400 rows (240 per topic, 480 none), 1.02M answer tokens, v3 vectors on (vLLM, one card).

**Distillation** (held-out rows, 96 sampled tokens per row):

| Layer | Gate accuracy per token | Answer tokens only | Expert error vs v3 vector |
|---|---|---|---|
| 9 | 95.0% | 95.4% | 0.6-1.8% |
| 18 | 99.3% | 99.4% | 0.5-1.0% |
| 27 | 99.4% | 99.5% | 0.7-2.3% |

Layers 18 and 27 are easier because the steering from layer 9 is already in the hidden state.

**One GGUF, stock llama.cpp:** `C:\llm\models\qwen3-4b-steer\Qwen3-4B-steer-v4-MoE-F16.gguf` (8.10 GB, 55 MB more
than Qwen). 86 full-length answers with `moe/full_test.py`, one full copy per card (never tensor parallel):
138,947 tokens in 379 s = **367 tok/s**, 1 answer hit the 8,192-token cap. Openings match the v3 steered answers.

**Blind Haiku scores** (same rubric and normal answers as v3; 172 calls, 0 failed, $0.10):

| Topic | Normal | v3 | v4 | v4 wins |
|---|---|---|---|---|
| Debugging | 1.70 | 4.12 | 4.09 | 10/10 |
| Professional email | 3.22 | 4.01 | 4.03 | 10/10 |
| Story | 2.55 | 3.82 | 3.66 | 10/10 |
| Teaching | 2.72 | 3.43 | 3.55 | 9/10 |
| Explanation | 2.90 | 3.45 | 3.49 | 8/10 |
| How-to | 3.00 | 2.86 | 3.21 | 7/10 |
| Engineering | 2.48 | 2.18 | 2.41 | 3/10 |
| Math | 4.53 | 4.47 | 4.43 | 2/10 |
| **All 80 topic prompts** | **2.89** | **3.54** | **3.61** | |

v4 keeps v3's quality; the differences are within Haiku's run-to-run noise.

**LM Studio:** its active engine (llama.cpp ROCm runtime 2.80.1) includes `hy_v3`. Running that engine's own
`llama-server.exe` on the v4 GGUF gives steered answers whose openings match stock llama.cpp v4 (92-221 chars).
Installed 2026-11-25 as a hardlink (no second 8 GB copy) at
`C:\Users\mhnda\.lmstudio\models\local\Qwen3-4B-steer-v4-MoE\Qwen3-4B-steer-v4-MoE-F16.gguf`; `lms ls` lists it as
`qwen3-4b-steer-v4-moe`, arch `hy_v3`, 4.0B, 8.10 GB. Needs an LM Studio runtime that includes `hy_v3`.

## How it maps onto MoE, and what is custom

| MoE part in llama.cpp | Tensor | What v4 puts there |
|---|---|---|
| Router | `blk.N.ffn_gate_inp` | Linear 2560 -> 9, the topic gate |
| Experts | `blk.N.ffn_{gate,up,down}_exps` | Expert 0 = zeros ("none"), experts 1-8 = topic SwiGLU MLPs (hidden 128) |
| Shared expert | `blk.N.ffn_{gate,up,down}_shexp` | Qwen's original MLP, unchanged |
| Top-k | `hy_v3` code | top-1, sigmoid, weights normalised (weight 1) |

Only layers 9/18/27 are MoE; the other 33 stay dense Qwen. Unlike a normal MoE (large experts replacing the MLP,
several active per token, trained from scratch), v4 keeps the full MLP and adds tiny experts beside it (+55 MB).
No custom code at run time. Custom code only at build time: `moe/train.py` and `moe/export.py`. The file shows as
arch `hy_v3`, not `qwen3`; the Qwen chat template is copied over. llama.cpp builds without `hy_v3` cannot load it.

## Routing is per token (kept on purpose)

v3 picks one vector per prompt. v4's gates pick an expert for **every token** (prompt and answer) at each of the
3 MoE layers, independently. The owner prefers this over v3's per-prompt pick (2026-11-25).
- Held-out agreement with the prompt's topic: 95.0% of tokens at layer 9, 99.3% at 18, 99.4% at 27. Layers 18
  and 27 see layer 9's steering, so they mostly lock onto the same topic.
- About 1 token in 20 at layer 9 gets a different expert (which one was not measured). Scores held anyway.
- A prompt that mixes two topics can get different experts on different tokens. Not tested yet.
- No per-request override (v3's `"steer"` field): to run unsteered, load the plain Qwen GGUF.
- If strict one-topic-per-answer is ever wanted: retrain the gates with a consistency loss; llama.cpp has no
  "lock the expert" setting.

Not done yet: RL fine-tuning of the experts (not needed so far); a Q8_0 version (`llama-quantize`).

## Reproduce

Exact commands, versions (HF revision `cdbee75f`, llama.cpp `abca692`, gguf-tools venv) and sha256 of every
artifact: README "Build v4", steps 9-14. Seeds: data sample 4, token sample 0, training 0 (`--seed`, added after
the published run, so a retrain matches closely, not bit for bit). Export is exact: re-exporting the published
`moe.pt` gave the same GGUF sha256 (`35d00cab...`, 2026-11-25). Tests: `test_moe.py`.

## LM Studio notes (2026-11-25)

- Flash Attention works. With Unified KV Cache on, every model in LM Studio's ROCm and Vulkan engines logs
  `Flash Attention over K/V rows not supported, set to disabled` (`llama-context.cpp` `resolve_fused_ops`). That
  disables only `fused_kv_rows`, an add-on that makes each slot read only its own part of a shared cache; Vulkan
  rejects it (`ggml-vulkan.cpp`, `src[5] != nullptr`). On gfx1201 others measured the fallback as faster.
- A Q4_1 KV cache has no Flash Attention kernel and is converted to F16 each step (slow): use Q8_0 or Q4_0.
- Unified off gives each slot its own full context: 4 slots x 108,544 at Q4_0 took about 28 GB.
- The warning `control-looking token: 128247 '</s>'` comes from Qwen's own vocabulary (same in the base GGUF).
