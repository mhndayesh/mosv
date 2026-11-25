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

LM Studio: it can load this only if its bundled llama.cpp includes `hy_v3` (added to llama.cpp in 2026).
