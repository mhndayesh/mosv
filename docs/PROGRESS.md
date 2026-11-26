# Progress log

Goal: replace LoRA with tiny steering vectors inside a frozen model, trained by RL, with one router choosing at
most one vector per request. Research only. Failed ideas, old runs and their verdicts are in
[`../archive/FAILED_IDEAS.md`](../archive/FAILED_IDEAS.md).

Inspiration: Meral et al., "RL Post-Training in Diffusion Transformers Is Activation Steering"
([arXiv:2611.26627](https://arxiv.org/abs/2611.26627)). That paper uses one shared vector set on diffusion models.
This project adapts it to a text LLM, with one vector per topic and a router (see the README).

## Current state (2026-11-25)

| Part | Status |
|---|---|
| Base model | Qwen3-4B-Instruct-2507, frozen, float16 |
| Topics | engineering, story, teaching, debugging, math solving, explanation, professional email, how-to guide |
| Router | 8 topics + "none". 6,823 of 6,835 held-out topic prompts got the right vector, 0 a wrong one, 12 none; 6 of 600 normal prompts steered by mistake |
| Judge | Claude Haiku 5.5, one 5-criterion rubric per topic (`rl/judge_rubrics.py`) |
| Current vectors | Run `qwen4b_haiku_v3`, all 8 topics trained together (`--interleave`), stopped at about half the planned updates. Picked steps in `checkpoints/rl_pilot/qwen4b_haiku_v3_test` |
| Review | Full-length blind test on 86 prompts, PDF `data_review/reports/qwen4b_full_answers.pdf` |
| **Shipped model** | **v4**: one GGUF (`hy_v3` MoE), stock llama.cpp and LM Studio, blind average 3.61 (normal 2.89). See Result 4 |
| v3 | Control vectors + proxy, folder `v3/` (tag `v3-control-vectors`). Replaced by v4 as the main version; its vectors and router are v4's teacher |
| **v5 (2026-11-26, branch `bonsai`)** | Bonsai-4B 1-bit, expert trained by RL straight into the GGUF (llama.cpp sampling). Engineering only so far: locked test 66.2% of prompts better than base, 2.42 vs 1.95, answers 268 vs 1,212 tokens. See Result 5 and [V5_BONSAI.md](V5_BONSAI.md) |

## Result 1: training scores (Haiku, 256-token answers)

Dev = Haiku's preference for the steered answer against normal Qwen on 128 held-out dev prompts (0.50 = tie).
The picked step is the best of each topic's last 3 dev checks. Only 3 topics ran their final test before the stop.

| Topic | Trained to step | Picked step | Dev | Final test: normal / random / steered | Steered beats normal |
|---|---|---|---|---|---|
| Debugging | 1,069 | 1,069 | 0.871 | 0.257 / 0.248 / **0.889** | 100% |
| Professional email | 240 | 150 | 0.825 | not run | |
| Story | 1,004 | 900 | 0.821 | not run | |
| Math solving | 561 | 550 | 0.735 | not run | |
| Teaching | 91 | 91 | 0.732 | 0.466 / 0.426 / **0.730** | 78.4% |
| Explanation | 540 | 400 | 0.727 | not run | |
| Engineering | 342 | 300 | 0.714 | 0.347 / 0.335 / **0.747** | 89.3% |
| How-to guide | 558 | 450 | **0.483** | not run | below normal |

## Result 2: full-length blind test (the main result)

86 held-out prompts (10 per topic, 6 outside all topics), each answered in full twice: normal and with the
router's vector. vLLM on one card, greedy, no length cap; 4 of 172 answers were stopped for looping. Haiku saw
each pair as "Answer 1" and "Answer 2", judged it twice with the order swapped, and scored every rubric criterion
separately as strict JSON: 172 calls, 0 failed, $0.09.

| Topic | Normal | Steered | Steered wins / ties / losses |
|---|---|---|---|
| Debugging | 1.70 | **4.12** | 10 / 0 / 0 |
| Story | 2.55 | **3.82** | 10 / 0 / 0 |
| Professional email | 3.22 | **4.01** | 9 / 1 / 0 |
| Teaching | 2.72 | **3.43** | 9 / 0 / 1 |
| Explanation | 2.90 | **3.45** | 7 / 1 / 2 |
| Math solving | 4.53 | 4.47 | 2 / 7 / 1 |
| How-to guide | 3.00 | 2.86 | 3 / 1 / 6 |
| Engineering | 2.48 | 2.18 | 2 / 0 / 8 |
| **All 80** | **2.89** | **3.54** | |

The 6 prompts outside the topics got a general rubric. The router left them unsteered, so both answers are the
same text and scored the same.

What this shows, and what it does not:

- 5 topics clearly improve. Math is a tie, because normal Qwen is already strong there.
- **Engineering does not carry over to full length.** It beat normal at 256 tokens (test 0.747) but loses on
  full programs (2.18 against 2.48). Its steered answers were also the longest: 64,499 tokens for 10 answers,
  2 stopped for looping.
- **How-to is slightly worse with its vector**, as its dev score (0.483) predicted. The owner's view: how-to is
  closer to opinion than to a structured skill, unlike debugging.
- 10 prompts per topic is a small sample. Knowledge retention (factual accuracy) is still untested.
- The full test used an **fp8 KV cache**. Training and dev scoring used the default cache, because fp8 drifted from
  PyTorch in training tests. The two setups are not identical.

## Result 3: runs in stock llama.cpp (2026-11-25, tag `v3-control-vectors`; archived, replaced by v4)

GGUF base model + one llama.cpp control vector per topic + `gguf/serve.py` (router proxy over llama-server
router mode). Router picks 86/86 identical to the original; 23 ms per prompt on GPU (30 s on CPU, so GPU is the
default). End-to-end through the proxy: debugging, story, none routed correctly; steered openings match the
vLLM steered answers; streaming works; a topic switch reloads the model (8-10 s for a 120-token answer, F16).
Limits: LM Studio cannot load control vectors; the vectors cannot be baked into Qwen3 weights (no bias slot).

## Result 4: v4, one MoE GGUF for stock llama.cpp (2026-11-25, branch `v4-moe`)

The v3 router + vectors distilled into real MoE layers of the `hy_v3` architecture at layers 9/18/27 (original
MLP = shared expert, expert 0 = none, 8 small SwiGLU topic experts, per-token top-1 gate). One 8.10 GB GGUF, no
proxy, no custom code; also runs on LM Studio's engine. Blind Haiku average on the 80 topic prompts: normal 2.89,
v3 3.54, **v4 3.61**. 367 tok/s with one copy per card. Routing is per token (95-99% agree with the prompt's
topic); the owner prefers this to v3's per-prompt pick. Installed in LM Studio as `mosv-qwen3-4b`.
Reproducible from the repo: README steps 9-14, pinned HF revision and llama.cpp commit, sha256 of every artifact;
re-exporting from `moe.pt` gave the identical GGUF. Details: [V4_MOE.md](V4_MOE.md).

## Result 5: v5, Bonsai-4B 1-bit, engineering (2026-11-26, branch `bonsai`)

New engineering data (self-oss, Magpie coder medium/hard, Code Review; 3,000 train), a Fable-written behavior
rubric, and RL that trains the gated expert directly in the GGUF layout while llama.cpp writes the answers.
Dev win rate vs base (96 prompts, blind pairs): 53.6% (update 25), 55.7% (50), 64.6% (75), **72.9% (100)**,
70.3% (125), 58.3% (150); the length penalty over-shortened after 100, so the run stopped and checkpoint 100 ships.
Locked test, 300 prompts, each answer scored alone by Haiku: overall 1.95 -> **2.42**, trained higher on **66.2%**
of prompts, 1,212 -> **268** tokens, loops 30 -> 0, cut answers 107 -> 0, 865 s -> 167 s for all 300. Code Review
gains least (1.37 -> 1.71). Details, settings and reproduce steps: [V5_BONSAI.md](V5_BONSAI.md).

## Open items

- How-to: retrain or drop. Engineering: train on longer answers, or split the topic.
- Professional email, story, math and explanation never ran their final test.
- The unused vLLM turn-taking code (`CardTurns`, `WeightShuttle`) is still in `rl/vllm_sampler.py`.
- Debugging excludes prompts too long to leave room for an answer: 840 train, 105 dev, 114 test.
- Two `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` faults on 2026-11-24, cause unknown.
- About 1 in 4 long greedy answers changes between two identical vLLM runs, so dev scores from 0.45 to 0.55 can be chance.

## Data

Real prompts only, capped at 12,000 per topic. Training splits: `data_review/rl_v4` (68,944 train, 8,741 dev,
8,596 test), built by `python -m rl.prepare_v4`. Sexual-themed prompts are removed by keyword.

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

Story training uses the mixed set in `data_review/clean_v3/story` (29,364 real prompts: WritingPrompts 8,809,
WildChat 8,809, LMSYS 6,728, ShareGPT 2,215, no_robots 1,476, Arena 922, small human sets 405; built by
`python -m behaviors.prepare_story_mix`). Teaching stays small because no suitable new source was found. About
920 of the 2,181 arena engineering prompts share one template.

## Judge details (`rl/api_judge.py`)

- Training: one call per prompt in one order and its exact reverse, thinking off, strict JSON schema. A failed call
  is retried once, then skipped; a skipped call never counts as a low score. 16 parallel calls (32 hit rate limits).
- Review: `reports/score_full.py` scores each criterion separately for both answers, blind, in both orders.
- Rubric check: on 20 prompts, normal answers beat shuffled ones 97.5% of the time and wrong-prompt answers 100%.
- Logs: `data_review/rl_v4/<run>/judge_calls.jsonl`, with no API key or answer text. Budget: $50 hard stop per run.

## Hardware rules (measured)

| Rule | Why |
|---|---|
| Never tensor parallel across the two cards | It ran at about half the speed of one card (72 card-to-card syncs per token in WSL), and the display card spilled 3-8 GB into system RAM in 4 of 7 tries |
| Prefer the second card (no display) | Windows gives WSL only about 10-13 GB of the display card, and the amount changed between runs |
| Training: vLLM on one card writes, PyTorch on the other learns | Two 8 GB copies cannot share a card, and vLLM sleep fails on ROCm in WSL. About 10-12 s per update, 0 faults |
| Testing: one card, CUDA graphs on, fp8 KV | 80k context with a 5.7 GiB cache; 86 answers (99,186 tokens) in 153 s, up to 85 at once, 0 preemptions |
| Watch shared GPU memory during runs | Spills into system RAM are silent and slow; check `\GPU Adapter Memory(*)\Shared Usage` |

## Decisions

| Date | Decision | Why |
|---|---|---|
| 2026-11-25 | v4 MoE GGUF is the shipped model; v3 serving archived | One file, no proxy, runs in LM Studio, same quality |
| 2026-11-25 | Keep per-token routing in v4 | Owner prefers it to v3's per-prompt pick |
| 2026-11-25 | No tensor parallel; test on one card | See the hardware rules; owner rule |
| 2026-11-25 | Full-length answers with a loop check for the review | 256-token answers cut most answers mid-sentence; the loop check stops answers whose recent tokens are mostly one repeated block |
| 2026-11-25 | Blind per-criterion Haiku scoring for the review | Checklist scores rewarded format, not quality |
| 2026-11-24 | Haiku judge with per-topic rubrics for training | The story checklist agreed with human votes on 49.6% of 26,298 LitBench pairs |
| 2026-11-24 | 4B main-card vLLM budget 10.5 GB in training | 11.0 reached 14.03 GB; 10.0 slowed writing from about 8 s to 14 s |
| 2026-11-23 | Default vLLM KV cache in training, not fp8 | fp8 matched PyTorch on 20-32% of steered words; default matched 89-95% |
| 2026-11-23 | Clear the vLLM prefix cache on every vector change | Cached prompt work otherwise uses the old vector |
| 2026-11-23 | Cap vectors at 50% of the median token size | Qwen's first token is about 8,200 and inflated the average |
| 2026-11-23 | Base model Qwen3-4B-Instruct-2507 | Same Qwen3 class, Apache-2.0, text only; 8B is 16.4 GB in float16 and cannot fit one card |
| 2026-11-23 | Any GPU fault stops all GPU work until the owner is back | Retrying a real driver fault unattended could freeze the PC |
| 2026-11-23 | Drop prompts that leave no room for an answer | Debugging prompts up to 3,708 tokens stalled a run |
| 2026-11-23 | Explanation instead of argument | No real argument-prompt dataset over 3,000 rows was found |

## Owner's observations and hypotheses (untested)

**Behavior, not knowledge.** On Qwen3-4B the vectors visibly change how the frozen model structures and carries out
its answers. They seem to work like behavior fine-tuning and do not appear to add information. The owner's
reasoning: a large model already holds broad knowledge but is biased toward certain tasks, so vectors could
re-steer it per task while its weights stay fixed. Retained knowledge, very large models and undoing a trained bias
are all untested.

**Shorter system prompts.** A vector per workflow (or per task inside a workflow) could carry tone, style, format
and step order, so the system prompt can shrink. Facts, schemas, tool definitions and hard rules stay in the prompt.
A workflow knows its task, so it can load its vector without the router. Planned test on the same prompts: full
prompt, short prompt, short prompt + trained vector, short prompt + random vector. The hypothesis holds if the third
gets close to the first and clearly beats the second and fourth, with no rise in broken answers.
