# Progress log

Goal: replace LoRA with tiny steering vectors inside a frozen model, trained by RL, with one router choosing
at most one expert per request. Research only. The checklists judge process and structure, not truth.

## Current stage: 8 topics on a 3-8B model (full auto, started 2026-11-23)

| Phase | Status |
|---|---|
| 1. Scout data and choose the model | Done: Qwen3-4B-Instruct-2507 |
| 2. Download and clean data, 8 topics | Running |
| 3. Checklists for the 5 new topics | Done: gold examples score 1.0, story "talks about a story" exploit closed |
| 4. Port the pipeline to the new model | Running |
| 5. Retrain the router (8 topics + none) | Waiting |
| 6. Small GPU test | Waiting |
| 7. Train all 8 vectors, one model per card | Waiting |
| 8. Final test, side-by-side PDF, report | Waiting |

Topics: engineering, story, teaching, debugging, math solving, explanation, professional email, how-to guide.

## Rules for this stage

- One model, VRAM, KV cache and queue per card. Nothing shared over the bus.
- float16, compiled, default vLLM KV cache. fp8 drifted from PyTorch in testing.
- Stop any run above 14 GB on a card or 2.5 GB shared.
- Small GPU test before every long run. Commit each verified phase locally. Do not push.

## Last verified result: Qwen3-0.6B, run `vllm_full`

| Behavior | Test prompts | Unsteered | Random | Trained | Broken |
|---|---|---|---|---|---|
| Story | 200 | 0.111 | 0.063 | 0.828 | 0% |
| Teaching | 81 | 0.200 | 0.134 | 0.698 | 0% |
| Engineering | 199 | 0.005 | 0.010 | 0.170 | 21.6% |

Router on 680 held-out prompts: 463 steered, 460 correct, 3 wrong, 19 missed.

Known limits:

- Engineering stalled after 8 rollbacks. Its `vllm_full` file holds update 86, not its best update 40.
- Some story answers describe a story instead of telling one, and still score well.
- Training step size still uses the average activation, which the huge first token inflates.

## Decisions

| Date | Decision | Why |
|---|---|---|
| 2026-11-23 | Default vLLM KV cache, not fp8 | fp8 matched PyTorch on 20-32% of steered words; default matched 89-95% |
| 2026-11-23 | Clear the vLLM prefix cache on every vector change | Cached prompt work otherwise uses the old vector |
| 2026-11-23 | Cap vectors at 50% of the median token size | Qwen's first token is about 8,200 and inflated the average |
| 2026-11-23 | vLLM on P-cores, PyTorch on E-cores, per card | vLLM used about 2x PyTorch's CPU |
| 2026-11-23 | Base model: Qwen3-4B-Instruct-2507 | Same Qwen3 class, Apache-2.0, text only. 8B is 16.4 GB in float16 and cannot fit |
| 2026-11-23 | vLLM and PyTorch take turns on each card for 4B | Two 8 GB copies cannot both stay on a 16 GB card |
| 2026-11-23 | Replace argument with explanation | No real argument-prompt dataset over 3,000 rows was found |
| 2026-11-23 | Real prompts only, capped at 12,000 per topic | No template padding; engineering and teaching may stay well below 10,000 |
