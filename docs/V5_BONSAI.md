# v5: MoSV on Bonsai-4B (1-bit), trained straight into the GGUF with llama.cpp sampling

Branch `bonsai` (forked from v4). Code: `v5/` (trainer), `behaviors/prepare_v5.py` (data), `reports/answer_v5.py`,
`reports/score_v5.py`, `reports/compare_v5.py`, `moe/live_gguf.py`. Status 2026-11-26: **engineering topic done.**

## What changed from v4

| | v4 (Qwen3-4B) | **v5 (Bonsai-4B)** |
|---|---|---|
| Base model | Qwen3-4B-Instruct-2507, F16 (8.1 GB GGUF) | **Bonsai-4B, 1-bit Qwen3-4B (PrismML), Q1_0 (0.54 GB GGUF)** |
| How the steering is learned | RL trains v3 vectors (vLLM sampling), then distilled into MoE experts | **RL trains the MoE expert directly** in the shipped GGUF layout (no vectors, no distillation) |
| Who writes training answers | vLLM (WSL) | **llama.cpp on the 1-bit GGUF itself** (Windows, Vulkan) |
| Learner | PyTorch (WSL) | PyTorch (Windows ROCm venv), compiled with project-local triton-windows |
| Topics | 8 | engineering (new data); others to follow the same flow |
| Shipped file | `MoSV-Qwen3-4B-F16.gguf` 8.10 GB | **`MoSV-Bonsai-4B-eng-upd100.gguf` 584 MB** |

## Layout (same as v4, hy_v3)

At layers 9, 18, 27: the original MLP is the shared expert; a per-token top-1 gate picks expert 0 ("none", all zeros)
or the engineering expert (SwiGLU, width 128, ~1M numbers per layer). Gate trained first (supervised, engineering vs
none prompts + short base answers; held-out accuracy 98.3-98.4% per layer) and frozen. PyTorch uses Bonsai's FP16
"unpacked" weights, which are exactly the 1-bit weights (+-1 x one scale per 128), so training = shipping.

## Data (`behaviors/prepare_v5.py`, behavior not knowledge)

| Source | Kept | What it teaches |
|---|---|---|
| bigcode self-oss-instruct (odc-by) | 1,000 train / 100 dev / 100 test | precise function specs |
| Magpie Qwen2.5-Coder, medium/hard "do" tasks | 1,000 / 100 / 100 | varied real-world coding tasks |
| StackExchange Code Review (NC) | 1,000 / 100 / 100 | improve real code |

Filters drop lookup questions ("what is", "explain"), web/game builds, prompts that refer to a missing snippet, and
anything over 2,500 characters. The test split is locked in `results/locked_v5/engineering.json` (300 ids + sha256;
the builder refuses to change it without `--force`); 0 overlap with train/dev by text and by near-duplicate group.

## Rubric

Written by a Fable subagent from the data and the base model's answers (`rl/judge_rubrics.py`, engineering): does
the asked task; gets to the work at once; careful and consistent; simple and proportionate; clear to a reader; minus
for restarts ("Wait", "Actually"), repeats, unsupported claims, boilerplate. Judge window 768 words for engineering.

## Training settings that worked (measured)

| Setting | Value | Why |
|---|---|---|
| Sampling | llama.cpp, display card only (Vulkan0), 32 slots x 2,048 tokens, `--no-jinja`, top_k 200 | sharing the learner's card spilled 1.1 GB to RAM; `--no-jinja` stops a chat-parser 500; top_k 200 = 610 vs 458 tok/s |
| Learner | PyTorch on HIP 0 (= Vulkan1), layers < 9 no graph, later checkpointed, non-steer layers compiled | ~5-9 s per update |
| Weight sync | expert bytes patched in place in the GGUF (offsets read once) + server restart | ~2 s per update |
| Batch | 8 prompts x 4 answers, up to 1,024 tokens | |
| Judge | Haiku, strict JSON scores + ranking, 16 calls at once | 0 errors in 2,500+ calls, p50 1.3 s |
| Optimizer | Adam, start lr 5e-3, KL guard 0.01 (undo + halve) | settled at 1.25e-3; lr 1e-4 learned nothing |
| Length | reward -= 0.1 x tokens/1,024 (cap = full 0.1) | helped to update ~100, then over-shortened (see below) |
| Update time | ~40-60 s | |

## Result

Dev check (96 dev prompts, base vs trained side by side, blind): 53.6% at update 25, 55.7% at 50, 64.6% at 75,
**72.9% at 100**, 70.3% at 125, 58.3% at 150. The length penalty kept pushing answers shorter after update 100 and
quality fell, so the run was stopped and **checkpoint 100** is the model.

**Locked test (300 prompts never used in training; each answer scored alone by Haiku, one call per answer):**

| | Base Bonsai-4B | **MoSV-Bonsai-4B, checkpoint 100** |
|---|---|---|
| Overall (1-5) | 1.95 | **2.42** |
| Trained scores higher on the same prompt | | **66.2%** |
| Mean answer length | 1,212 tokens | **268** |
| Loops / cut at 2,048 tokens | 30 / 107 | **0 / 0** |
| Time for all 300 (both cards) | 865 s | **167 s** |

Per rubric line (base -> trained): does the task 2.36 -> 2.61; gets to the work 2.35 -> **3.38**; careful 1.81 ->
2.37; simple 2.17 -> **3.08**; clear 2.22 -> 2.94. By source: self-oss 2.44 -> 2.90, Magpie 2.05 -> 2.65,
Code Review 1.37 -> 1.71 (smallest gain: reviews need explanation, which the length push trims). Base answers cut at
2,048 tokens were judged on what they show (the judge is told not to penalize unfinished answers).
Summary file: `results/v5_engineering_test_scores.json`.

Verifier (`v5/verify.py`, checkpoint at update 20 of the pilot): PyTorch vs llama.cpp greedy tokens agree 7/8 for
64 tokens; the gate sends 95-96% of engineering tokens and 0.5-2.8% of other tokens to the expert.

## Run it

```
llama-server -m MoSV-Bonsai-4B-eng-upd100.gguf -ngl 99 -fa on
```
LM Studio: listed as `mosv-bonsai-4b` (hardlink in `~/.lmstudio/models/local/MoSV-Bonsai-4B/`).
Engineering prompts get short, direct answers; other prompts are routed to "none" and behave like plain Bonsai.

## Reproduce (Windows; ai-training-rocm-windows venv unless noted)

1. Data + lock: `python -m behaviors.prepare_v5 --topic engineering`
2. Base answers on the locked test (gguf-tools venv): `python reports/answer_v5.py --topic engineering --model <Bonsai-4B-Q1_0.gguf> --tag bonsai_normal_2048`
3. Train: `python -m v5.train --run eng_full2 --topic engineering --updates 375 --compile --lr 5e-3 --dev 96 --eval-every 25`
   (checkpoints at every dev check; we stopped at 150 and used `steer_0100.pt`)
4. Build the GGUF from a checkpoint with `moe/live_gguf.build` (Steer.load + to_moe), then answer the test as in 2.
5. Score: `python reports/score_v5.py --topic engineering --tags bonsai_normal_2048 mosv_upd100`

## Lessons / next

- Turn the length penalty off (or only penalize answers longer than the base answer) once answers reach about half
  the base length; pick the checkpoint by dev score.
- Bigger models (Gemma 4 12B, Qwen3.5-9B): GGUF hosts checked in llama.cpp source (`gemma4` has a per-layer MoE
  switch like hy_v3; `qwen35moe` needs MoE on every layer + a doubled shared-expert scale). Memory needs a layer split
  across both cards. Unsloth shared-weight RL env prepared in WSL (`~/.venvs/unsloth-rl`), not yet tested on GPU.

## Known weakness and wrap-up (2026-11-26)

- **Code review / refactor requests get too little explanation.** Example (`results/v5_run_log.txt`): asked to
  refactor a small function, the model only renamed a variable and added spaces (47 tokens, no explanation, no
  simpler version). This matches the smallest test gain (Code Review 1.37 -> 1.71). Cause: the length penalty kept
  rewarding shorter answers after they were already short enough.
- **Verification of the shipped file** (`results/v5_verify_upd100.json`, checkpoint 100, 16 dev prompts): PyTorch vs
  llama.cpp greedy tokens identical 14/16 (2 early fp16-vs-Vulkan splits); trained vs plain Bonsai 0/16 identical
  (the expert is active); routing 95-97% of engineering tokens vs 1.5-3.6% of other tokens; expert output at 94% /
  90% / 59% of the original MLP output at layers 9/18/27; GGUF = hy_v3, 253 Q1_0 + 9 F16 + 148 F32 tensors.
- **Recommended use settings** (Bonsai's own `generation_config.json`): temperature 0.5, top_p 0.85, top_k 20,
  min_p 0, repetition penalty 1.0. Training sampled at temperature 1.0 / top_k 200 and the tests above were greedy.

## Rules for the next model (from this run)

1. **Length:** penalize length only when an answer is longer than the base model's answer to the same prompt, or
   switch the penalty off after the early updates. Never reward shorter-than-base answers.
2. **Sampling:** train and test with the model's own recommended settings from its `generation_config.json`
   (for Gemma 4 12B: temperature 1.0, top_k 64, top_p 0.95), and record them in every run log.
3. **Checkpoint choice:** pick by the dev check, not the last update; keep a checkpoint at every dev check.
