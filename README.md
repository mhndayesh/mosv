# Steering vectors instead of LoRA: one frozen LLM, many behaviors

A frozen Qwen3-4B-Instruct-2507 gets a bank of tiny **steering vectors**, one per topic. A **router** reads each
prompt and picks at most one vector. The vectors are trained by **RL**, with Claude Haiku as the judge. The
model's weights never change: a vector is about 30 KB, against 8 GB for the model. **Research only.**

Status, results and decisions: **[docs/PROGRESS.md](docs/PROGRESS.md)**. Failed ideas and old runs:
**[archive/FAILED_IDEAS.md](archive/FAILED_IDEAS.md)** (the folder is gitignored).

## Where the idea comes from

This project adapts **"RL Post-Training in Diffusion Transformers Is Activation Steering"** by Tuna Han Salih Meral,
Kaan Oktay, Alec Helbling, Atakan Kara, Duen Horng Chau and Pinar Yanardag
([arXiv:2611.26627](https://arxiv.org/abs/2611.26627), project page <https://rl-is-steering.github.io>).

The paper shows that a fixed set of steering vectors, added to the block outputs of a frozen image/video diffusion
transformer, recovers most of what RL fine-tuning gains. They report 90.0-96.3% of the reward gains of six RL-tuned
image models, with 65x fewer parameters than the RL-tuned LoRA (0.29M against 18.8M on SD3.5-M). The vectors can
also be trained directly with online RL, starting from zero.

What this repo changes:
- **Text LLM instead of diffusion:** the vectors go on the residual stream of a frozen Qwen3-4B at 3 layers.
- **One vector per topic instead of one shared set**, with a **router** that picks a vector, or none, per prompt.
- **Training:** each vector is trained by group-relative RL, with Claude Haiku judging against a per-topic rubric.

## Current model: v4, one GGUF for stock llama.cpp and LM Studio (2026-11-25)

`Qwen3-4B-steer-v4-MoE-F16.gguf` (8.10 GB) is Qwen3-4B with the trained steering built in as real MoE layers of
`hy_v3`, an architecture llama.cpp already has. The router is inside the model as per-token gates. No proxy, no
custom build: `llama-server -m Qwen3-4B-steer-v4-MoE-F16.gguf -ngl 999 --jinja`, or load it in LM Studio. Design,
measurements and limits: **[docs/V4_MOE.md](docs/V4_MOE.md)**.

Full-length answers on 86 held-out prompts, scored blind by Claude Haiku on each topic's 5-criterion rubric
(1-5 scale; both answers shown as "Answer 1" and "Answer 2", judged twice with the order swapped). v3 is the
teacher v4 was distilled from (vectors + router, served with a proxy; archived).

| Topic | Normal | v3 | **v4** | v4 wins |
|---|---|---|---|---|
| Debugging | 1.70 | 4.12 | **4.09** | 10 / 10 |
| Professional email | 3.22 | 4.01 | **4.03** | 10 / 10 |
| Story | 2.55 | 3.82 | **3.66** | 10 / 10 |
| Teaching | 2.72 | 3.43 | **3.55** | 9 / 10 |
| Explanation | 2.90 | 3.45 | **3.49** | 8 / 10 |
| How-to guide | 3.00 | 2.86 | **3.21** | 7 / 10 |
| Engineering | 2.48 | 2.18 | **2.41** | 3 / 10 |
| Math solving | 4.53 | 4.47 | **4.43** | 2 / 10 |
| **All 80 topic prompts** | **2.89** | **3.54** | **3.61** | |

v3 vs v4 differences are within Haiku's run-to-run noise. PDF with every v3 answer side by side:
`data_review/reports/qwen4b_full_answers.pdf`; v4 answers and scores: `data_review/reports/v4_full_test.json`,
`v4_scores.json`.

## How it works

At chosen layers `l` (default about 25%, 50% and 75% of depth), every token's hidden state gets the vector of
the chosen topic `e` added: `h'[l,i] = h[l,i] + s[e,l]`. The router picks `e` once per prompt from a read-only
pass, or picks "none" and leaves the model untouched. Training samples a group of answers per prompt, Haiku
scores them against the topic rubric, and a group-relative policy gradient (REINFORCE with group-centered
rewards; no PPO clipping, no KL term) updates only that topic's vector.

v4 turns that into MoE layers. At layers 9, 18, 27 the original MLP becomes the always-on shared expert; next to
it sit expert 0 (all zeros = "none") and 8 small SwiGLU topic experts, picked top-1 per token by a linear gate.
The experts are distilled to output the v3 vectors and the gates to give the router's topic, from v3's own answers.

## Repository map

| Path | Job |
|---|---|
| `steering.py` | The vector bank and the layer hooks that add a vector |
| `system.py` | Loads model + router + vectors; routed and fixed generation |
| `model_profiles.py` | Per-model settings (0.6B and 4B): layers, paths, VRAM budgets |
| `rl_math.py` | Group-relative advantages |
| `behaviors/` | Data cleaning per topic, story mix, and the old regex checklists (`rubrics.py`, now only for tests and `evaluate.py`) |
| `router/` | Router dataset, training and pipeline |
| `rl/pilot.py`, `rl/interleave.py` | RL training loop; `--interleave` trains all 8 topics in one run |
| `rl/api_judge.py`, `rl/judge_rubrics.py` | Claude Haiku judge and the 8 topic rubrics |
| `rl/vllm_sampler.py`, `rl/vllm_plugin/` | vLLM answer writing with the live vector |
| `rl/data.py`, `rl/prepare_v4.py` | Training splits (`data_review/rl_v4`) |
| `reports/pick_vectors.py` | Picks each topic's test vector from a run (best of the last 3 dev checks) and writes the summary |
| `reports/compare_4b.py` | Picks the 86 review prompts and records the router's choice (`compare_4b.json`) |
| `reports/vllm_full_test.py` | Writes full-length normal and steered answers on vLLM |
| `reports/score_full.py` | Blind Haiku scoring, one score per criterion, strict JSON |
| `reports/build_pdf_full.py` | The side-by-side PDF with bookmarks (helpers in `build_pdf_4b.py`) |
| `moe/make_data.py`, `moe/train.py` | v4: v3's answers as training data; distil gates + experts (`checkpoints/moe_v4/moe.pt`) |
| `moe/export.py`, `moe/full_test.py` | v4: write the `hy_v3` GGUF; full-length test in llama-server |
| `evaluate.py`, `infer.py`, `model_trace.py` | Older checklist evaluation, one-off routed answers, per-token trace |
| `test_*.py` | Tests (`python -m pytest`); `test_reports.py` covers the review scripts with no GPU |
| `checkpoints/rl_pilot/qwen4b_haiku_v3*` | Current vectors (`_test` = the step picked per topic). Gitignored |
| `data_review/` | Datasets and reports. Gitignored, research-only licences |
| `archive/` | Failed and superseded work. Gitignored |

## Run the pipeline

Training, testing and scoring run in WSL (Ubuntu-24.04) with the venv `~/vllm-dual-9070xt/.venv`. Environment
for every command:

```bash
cd /mnt/c/projects/llm_steering_rl
V=$HOME/vllm-dual-9070xt/.venv
export LD_LIBRARY_PATH=$V/lib/python3.12/site-packages/_rocm_sdk_core/lib ROCPROFILER_REGISTER_ENABLED=0 \
  VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_NO_USAGE_STATS=1 VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
  HF_HUB_OFFLINE=1 STEERING_PROFILE=qwen4b
```

1. **Data:** `python -m behaviors.prepare_topics`, `python -m behaviors.prepare_story_mix`, `python -m rl.prepare_v4`.
2. **Router:** `python -m router.pipeline`.
3. **Train.** The exact settings of `qwen4b_haiku_v3` are saved in each finished topic's
   `data_review/rl_v4/qwen4b_haiku_v3/<topic>.json` (`settings`), and its training logs are in `train_logs/`:
   `$V/bin/python -m rl.pilot --profile qwen4b --run <name> --judge haiku --interleave --overlap --sampler vllm --vllm-swap separate --cards main --engine-card second --compile --one-pass --prompts 8 --group 4 --max-new 512 --dev 128 --test 300 --eval-every 50 --step-fraction 0.0025 --judge-budget 100`
4. **Pick the test vectors** (CPU, seconds): `python reports/pick_vectors.py` writes
   `checkpoints/rl_pilot/qwen4b_haiku_v3_test` and `data_review/reports/haiku_v3_summary.json`.
5. **Pick the 86 review prompts and record the router's choice:** `$V/bin/python reports/compare_4b.py`
   (seed 7, writes `data_review/reports/compare_4b.json`).
6. **Full-length test, one card** (`HIP_VISIBLE_DEVICES=0` is the second card, the one with no screen; the
   defaults are the settings used): `HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 $V/bin/python reports/vllm_full_test.py`
7. **Score:** `$V/bin/python reports/score_full.py` (Haiku key in `C:\projects\api\claude.txt`; only unscored pairs are sent).
8. **PDF** (Windows, any Python with reportlab): `python reports/build_pdf_full.py data_review/reports/qwen4b_full_answers.pdf`.

Package versions: `requirements.txt`. Light tests for steps 4-8, no GPU: `python -m pytest test_reports.py`.

## Build v4 (after steps 1-6 above; step 13 reuses step 6's normal answers)

Model folder `C:\llm\models\qwen3-4b-steer\`. Converter venv `C:\Users\mhnda\.venvs\gguf-tools` (torch 2.11.0+cpu,
transformers 4.57.6, numpy 2.2.6, gguf-py installed editable from `C:\llm\tools\llama.cpp` at commit `abca692`).

9. **Base GGUF** (Windows, gguf-tools venv): `hf\` holds hard links to the HF snapshot
   `Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554` (the WSL symlinks are unreadable from Windows):
   `python C:\llm\tools\llama.cpp\convert_hf_to_gguf.py C:\llm\models\qwen3-4b-steer\hf --outtype f16 --outfile C:\llm\models\qwen3-4b-steer\Qwen3-4B-Instruct-2507-F16.gguf`
10. **Training data** (WSL env above, one card): `HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 $V/bin/python moe/make_data.py`
    (seed 4; 2,400 prompts from `data_review/router_v2`, each answered greedily with its v3 vector).
11. **Distil gates + experts** (WSL, one card): `HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 $V/bin/python moe/train.py`
    (seed 0; writes `checkpoints/moe_v4/moe.pt` with its settings and per-layer accuracy).
12. **GGUF** (Windows, gguf-tools venv, about 2 min):
    `python moe/export.py C:\llm\models\qwen3-4b-steer\Qwen3-4B-Instruct-2507-F16.gguf C:\llm\models\qwen3-4b-steer\Qwen3-4B-steer-v4-MoE-F16.gguf --moe checkpoints/moe_v4/moe.pt`
    (without `--moe` it writes all-zero experts: output identical to plain Qwen, a check of the host architecture).
13. **Full-length test** (Windows, plain Python, both cards, one full copy each): `python moe/full_test.py C:\llm\models\qwen3-4b-steer\Qwen3-4B-steer-v4-MoE-F16.gguf`
    (`--llama` picks the llama-server.exe; default `C:\llm\tq-vulkan\build\bin\llama-server.exe`).
14. **Score:** `$V/bin/python reports/score_full.py --answers data_review/reports/v4_full_test.json --out data_review/reports/v4_scores.json`

Light tests, no GPU: `python -m pytest test_moe.py` (gguf-tools venv for the export test).

**Checksums of the files behind the published result** (sha256, to compare a rebuild against):

| File | sha256 |
|---|---|
| `data_review/moe_v4/answers.jsonl` | `e85014889a6462d443744de9484a5a97f5558119879dea3e84b7c0d824a8d9be` |
| `checkpoints/moe_v4/moe.pt` | `dd7b9aebf32882b8c67c31e60126f95296f0f0bceae6f022510006c2c6928c27` |
| `Qwen3-4B-Instruct-2507-F16.gguf` | `5dfe8dfb085a35074c6e6d787ac001f56706e67297df327a29a1f21b509d8772` |
| `Qwen3-4B-steer-v4-MoE-F16.gguf` | `35d00caba382fb0e1a593faeb07429289957fc628a47245c7b51e9847e139ba8` |

## Run v4

- **llama.cpp:** `llama-server -m Qwen3-4B-steer-v4-MoE-F16.gguf -ngl 999 -fa on --jinja` (any build that has
  arch `hy_v3`; tested on the tq-vulkan build b10269). Fits one 16 GB card; never tensor parallel.
- **LM Studio:** installed as a hard link in `C:\Users\mhnda\.lmstudio\models\local\Qwen3-4B-steer-v4-MoE\`, listed
  as `qwen3-4b-steer-v4-moe`. Works on LM Studio's ROCm 2.80.1+ and Vulkan 2.73+ engines. Use a Q8_0 or Q4_0
  KV cache (Q4_1 has no Flash Attention kernel). With Unified KV Cache on, the log warns `Flash Attention over K/V
  rows not supported`: that only disables an add-on for the shared cache, Flash Attention itself stays on, and
  it happens for every model in LM Studio's engines.
- Routing is per token; there is no per-request override. To answer unsteered, load the plain Qwen GGUF.

## Archived: v3 serving (control vectors + routing proxy)

Before v4, the vectors ran in llama.cpp as 31 KB control-vector GGUFs (`direction.9/18/27`) with a Python proxy
(`gguf/serve.py`) that ran the router and switched llama-server presets per request. It worked (router 86/86 same
picks, 23 ms on GPU) but needed the proxy, a model reload on each topic switch (8-10 s), and did not run in
LM Studio. v4 replaces it. The code is at tag **`v3-control-vectors`**
(`git checkout v3-control-vectors -- gguf test_gguf.py`) and in `archive/v3_control_vectors/`. The v3 vectors and
router themselves are still current: they are v4's teacher (steps 1-6).

## What reproduces exactly, and what does not

| Step | Exact on rerun? |
|---|---|
| Data splits, router labels, prompt pick (seeds fixed) | Yes |
| Picking the test vectors (step 4) | Yes: rebuilt bit for bit from the snapshots and logs (checked 2026-11-25) |
| Training (step 3) | No: answers are sampled and Haiku's verdicts vary, so a rerun gives different vectors |
| Full-length answers (step 6) | Mostly: greedy, but about 1 in 4 long answers changes between identical vLLM runs |
| Haiku scores (steps 7, 14) | Close, not exact: the order shuffle is seeded, but Haiku's replies can vary slightly |
| Base GGUF (step 9) | Yes, with the same HF revision and llama.cpp commit |
| v4 training data (step 10) | Mostly: greedy vLLM, same caveat as step 6 |
| v4 distillation (step 11) | Close, not exact: seeded since 2026-11-25 (the published `moe.pt` was trained before the seed was added), and GPU sums vary in the last bits |
| v4 GGUF from a given `moe.pt` (step 12) | Yes: re-export gives the same sha256 (checked 2026-11-25) |
| v4 full test (step 13) | Mostly: greedy, but parallel slots can change long answers |

## Hardware rules (measured on this box)

- **One model copy per card. Never tensor parallel.** Split layers (pipeline) only if a model does not fit on one card.
- Prefer the second card (no display). Windows gives WSL only about 10-13 GB of the display card, and anything
  over that spills into system RAM.
- Keep the test's step size at 4,096-8,192 tokens and CUDA graphs on (no `--eager`).
