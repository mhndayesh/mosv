# Progress log

Goal: replace LoRA with tiny steering vectors inside a frozen model, trained by RL, with one router choosing
at most one expert per request. Research only. The checklists judge process and structure, not truth.

## Current stage: 8 topics on a 3-8B model (full auto, started 2026-11-23)

| Phase | Status |
|---|---|
| 1. Scout data and choose the model | Done: Qwen3-4B-Instruct-2507 |
| 2. Download and clean data, 8 topics | Done: see counts below |
| 3. Checklists for the 5 new topics | Done: gold examples score 1.0, story "talks about a story" exploit closed |
| 4. Port the pipeline to the new model | Done: PyTorch path runs on 4B |
| 5. Retrain the router (8 topics + none) | Done: 2,781 test prompts, 2,178 of 2,184 steers correct, 0 wrong expert, 6 false steers, 3 misses |
| 6. Small GPU test | Done: split-card run, 4 updates, 13.07 GB peak, 0 GPU faults, about 10-12 s per update |
| 7. Train all 8 vectors, one model per card | Waiting |
| 8. Final test, side-by-side PDF, report | Waiting |

Topics: engineering, story, teaching, debugging, math solving, explanation, professional email, how-to guide.

Real prompts per topic (train / dev / test):

| Topic | Train | Dev | Test |
|---|---|---|---|
| engineering | 2,739 | 344 | 321 |
| story | 9,598 | 1,203 | 1,199 |
| teaching | 728 | 88 | 81 |
| debugging | 9,397 | 1,163 | 1,176 |
| math_solving | 9,640 | 1,209 | 1,151 |
| explanation | 9,599 | 1,227 | 1,174 |
| professional_email | 4,000 | 488 | 512 |
| how_to_guide | 9,572 | 1,207 | 1,221 |

Teaching stays small because no suitable new source was found. About 920 of the 2,181 arena engineering prompts share one template.

## Rules for this stage

- One model, VRAM and KV cache per card. Only token IDs and steering vectors cross between cards.
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
| 2026-11-23 | 4B: vLLM on the main card writes and scores, PyTorch on the second card learns | Two 8 GB copies cannot share a card, vLLM sleep fails on this ROCm/WSL setup, and PyTorch-only generation took 54-112 s per update and hit a GPU memory fault. The split takes about 10-12 s per update with 0 faults |
| 2026-11-23 | Any GPU fault stops all GPU work until the owner is back | Retrying a real driver fault unattended could freeze the PC |
| 2026-11-23 | Replace argument with explanation | No real argument-prompt dataset over 3,000 rows was found |
| 2026-11-23 | Real prompts only, capped at 12,000 per topic | No template padding; engineering and teaching may stay well below 10,000 |
