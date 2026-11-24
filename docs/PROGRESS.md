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
| 7. Train all 8 vectors | Done on 2026-11-24: all 8 trained and tested |
| 8. Final test, side-by-side PDF, report | Done: `data_review/reports/qwen4b_final_review.pdf` (94 pages, 86 side-by-side answers) |
| 9. Large mixed story dataset | Scouted only: about 50,400 estimated prompts from 8 sources, nothing downloaded |

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

## Latest result: Qwen3-4B-Instruct, run `qwen4b_v2`

Test scores on held-out prompts. The checklist measures structure, not truth or quality.

| Topic | Test prompts | Unsteered | Random | Trained | Broken |
|---|---|---|---|---|---|
| Story | 300 | 0.070 | 0.075 | 0.778 | 3.3% |
| Teaching | 81 | 0.152 | 0.169 | 0.679 | 0% |
| Math solving | 300 | 0.171 | 0.169 | 0.488 | 0% |
| Debugging | 300 | 0.169 | 0.171 | 0.404 | 0.3% |
| Engineering | 300 | 0.011 | 0.009 | 0.391 | 0% |
| Explanation | 300 | 0.040 | 0.045 | 0.374 | 1.7% |
| Professional email | 300 | 0.404 | 0.392 | 1.000 (likely format only) | 0% |
| How-to guide | 300 | 0.183 | 0.172 | 0.718 | 0% |

Router on all held-out prompts, after the padding fix: 6,823 of 6,835 topic prompts got the right vector, 0 got a wrong vector, 12 got no steering. On 600 normal prompts, 6 were steered by mistake.

Open items:

- The run stopped at 18:01 with `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`, the second such fault today. Windows logged no GPU error and both cards report OK. The cause is unknown.
- The email dev score of 0.992 is not trustworthy. In the PDF (page 40) the steered email changed a policy update into a meeting request and still scored 1.00, so the checklist likely rewards format.
- Debugging excludes prompts too long to leave room for a full answer: 840 train, 105 dev, 114 test.
- The owner read the PDF, but gaming has not been measured across the full test set.
- Fixed: after the first generation, padding stayed on in the shared tokenizer and put pad tokens into the router's input. This caused the debugging-to-math errors in the first PDF. Older 0.6B router numbers from `evaluate.py` may be affected.
- Tested and removed: running dev scoring in the same vLLM queue as training answers saved no measurable time (about 24 s against 22-27 s).
- Story data samples: found responses have normal writing structure but score only 0.00-0.35 on the story checklist, so use the prompts only.
- Knowledge retention is untested: no factual-accuracy check has compared steered and normal answers.
- The unused vLLM turn-taking code (`CardTurns`, `WeightShuttle`) is still in `rl/vllm_sampler.py`.

Resume or retest a topic in WSL. A finished topic skips training and runs its final test:

```bash
cd /mnt/c/projects/llm_steering_rl
V=$HOME/vllm-dual-9070xt/.venv
LD_LIBRARY_PATH=$V/lib/python3.12/site-packages/_rocm_sdk_core/lib ROCPROFILER_REGISTER_ENABLED=0 \
VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_NO_USAGE_STATS=1 VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
HF_HUB_OFFLINE=1 STEERING_PROFILE=qwen4b STEERING_TOPICS=v2 \
$V/bin/python -m rl.pilot --profile qwen4b --run qwen4b_v2 --cards second \
  --experts how_to_guide --sampler vllm --vllm-swap separate --compile \
  --updates 86 --prompts 16 --group 4 --max-new 256 --dev 128 --test 300 --eval-every 5
```

## First side-by-side review: Qwen3-4B (2026-11-24)

This section describes the first 46-prompt PDF, made before the router fix. The current review is `data_review/reports/qwen4b_final_review.pdf`.

PDF: `data_review/reports/qwen4b_steered_vs_normal.pdf` (gitignored). It shows 46 random held-out prompts, normal answer left and steered answer right: 6 for each of the 7 trained topics, plus 4 normal prompts. How-to is excluded because it has no vector. Prompts over 512 tokens were skipped, answers stop at 256 tokens, and emojis were removed because the PDF font cannot draw them.

The run used the second card only: 179 s, 8.65 GB peak PyTorch memory, 0 GPU errors. Answers are saved in `data_review/reports/compare_4b.json` (gitignored). `reports/compare_4b.py` writes them on one GPU in WSL, and `reports/build_pdf_4b.py <out.pdf>` builds the PDF (needs reportlab).

| Topic (6 prompts) | Normal | Steered |
|---|---|---|
| Professional email | 0.35 | 1.00 |
| Teaching | 0.15 | 0.67 |
| Story | 0.08 | 0.64 |
| Math solving | 0.17 | 0.52 |
| Engineering | 0.00 | 0.39 |
| Debugging | 0.17 | 0.39 |
| Explanation | 0.00 | 0.36 |

These are checklist scores on a small sample. They measure structure, not truth or quality.

Router on these 46 prompts: 44 correct, 1 wrong expert (page 21, debugging sent to math), 1 miss (page 11, story left unsteered).

Pages to reread: 40 (email changed its purpose), 22 (debugging answer names an `updateFilter()` cause that is not in the bug report), 21 and 11 (router errors).

### Owner's observation (not a scientific claim)

After reading the PDF, the owner observed that on Qwen3-4B the steering vectors visibly change how the frozen model behaves, executes and structures its answers, regardless of output quality. It works like behavior fine-tuning and does not appear to add information. The owner's reasoning is that a large model already holds broad knowledge but is biased by its training toward certain tasks, so vectors could re-steer it per task or prompt while its weights stay fixed.

This is a logical observation from using the method. It has not been tested formally and is not a scientific fact. Retained knowledge, very large models and undoing a trained bias are all untested.

### Owner's hypothesis: minimize system prompts (untested)

A vector trained per workflow, or per task inside a workflow, can carry that workflow's behavior instructions, so its system prompt can shrink while keeping the same adherence. The goal is to **minimize, not replace**. Example: a novel-writing platform with separate creative workflows and technical workflows (ingestion, uploads, API calls, MCP servers), each with its own vector.

| Part of the system prompt | Plan |
|---|---|
| Tone, style, voice, format, step order | Candidate for a vector |
| Habits such as "ask before assuming" | Possible, needs testing |
| Facts, platform details, character sheets, story state | Stays in the prompt |
| API schemas, MCP tool definitions, exact JSON | Stays in the prompt |
| Hard rules, safety, permissions | Stays in the prompt and is enforced in code |

A workflow already knows which task it is running, so it can load its vector directly without the router.

Risks: stronger steering can break text, prompt caching already makes repeated prompts cheap, the method needs a self-hosted model, and each new model version needs new vectors.

Planned test, on the same prompts:

1. Full system prompt (baseline)
2. Short prompt only
3. Short prompt + trained vector
4. Short prompt + random vector

The hypothesis holds if #3 gets close to #1 and clearly beats #2 and #4, with no rise in broken or off-task answers. The main measure is how many prompt tokens can be removed before adherence falls below the full prompt's.

## Earlier result: Qwen3-0.6B, run `vllm_full`

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
| 2026-11-24 | 4B main-card vLLM budget 10.5 GB | 11.0 reached 14.03 GB and hit the safety stop; 10.0 slowed writing from about 8 s to 14 s; 10.5 ran at about 8.5 s with a 13.15 GB peak |
| 2026-11-23 | Default vLLM KV cache, not fp8 | fp8 matched PyTorch on 20-32% of steered words; default matched 89-95% |
| 2026-11-23 | Clear the vLLM prefix cache on every vector change | Cached prompt work otherwise uses the old vector |
| 2026-11-23 | Cap vectors at 50% of the median token size | Qwen's first token is about 8,200 and inflated the average |
| 2026-11-23 | vLLM on P-cores, PyTorch on E-cores, per card | vLLM used about 2x PyTorch's CPU |
| 2026-11-23 | Base model: Qwen3-4B-Instruct-2507 | Same Qwen3 class, Apache-2.0, text only. 8B is 16.4 GB in float16 and cannot fit |
| 2026-11-23 | 4B: vLLM on the main card writes and scores, PyTorch on the second card learns | Two 8 GB copies cannot share a card, vLLM sleep fails on this ROCm/WSL setup, and PyTorch-only generation took 54-112 s per update and hit a GPU memory fault. The split takes about 10-12 s per update with 0 faults |
| 2026-11-23 | Any GPU fault stops all GPU work until the owner is back | Retrying a real driver fault unattended could freeze the PC |
| 2026-11-23 | Drop prompts that leave no room for a 256-token answer | Up to 3,708-token debugging prompts exceed the 1,024-token limit; a run stalled when it reached them |
| 2026-11-23 | Replace argument with explanation | No real argument-prompt dataset over 3,000 rows was found |
| 2026-11-23 | Real prompts only, capped at 12,000 per topic | No template padding; engineering and teaching may stay well below 10,000 |
