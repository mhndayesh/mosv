# Steering vectors instead of LoRA: one frozen LLM, many behaviors

A frozen Qwen3-4B-Instruct-2507 gets a bank of tiny **steering vectors**, one per topic. A **router** reads each
prompt and picks at most one vector. The vectors are trained by **RL**, with Claude Haiku as the judge. The
model's weights never change: a vector is about 30 KB, against 8 GB for the model. **Research only.**

Status, results and decisions: **[docs/PROGRESS.md](docs/PROGRESS.md)**. Failed ideas and old runs:
**[archive/FAILED_IDEAS.md](archive/FAILED_IDEAS.md)** (the folder is gitignored).

## Current result (2026-11-25, run `qwen4b_haiku_v3`)

Full-length answers on 86 held-out prompts, scored blind by Claude Haiku on each topic's 5-criterion rubric
(1-5 scale; both answers shown as "Answer 1" and "Answer 2", judged twice with the order swapped).

| Topic | Normal | Steered | Steered wins |
|---|---|---|---|
| Debugging | 1.70 | **4.12** | 10 / 10 |
| Story | 2.55 | **3.82** | 10 / 10 |
| Professional email | 3.22 | **4.01** | 9 / 10 |
| Teaching | 2.72 | **3.43** | 9 / 10 |
| Explanation | 2.90 | **3.45** | 7 / 10 |
| Math solving | 4.53 | 4.47 | 2 / 10 (7 ties) |
| How-to guide | 3.00 | 2.86 | 3 / 10 |
| Engineering | 2.48 | 2.18 | 2 / 10 |
| **All 80 topic prompts** | **2.89** | **3.54** | |

PDF with every answer side by side and per-criterion scores: `data_review/reports/qwen4b_full_answers.pdf`.

## How it works

At chosen layers `l` (default about 25%, 50% and 75% of depth), every token's hidden state gets the vector of
the chosen topic `e` added: `h'[l,i] = h[l,i] + s[e,l]`. The router picks `e` once per prompt from a read-only
pass, or picks "none" and leaves the model untouched. Training samples a group of answers per prompt, Haiku
scores them against the topic rubric, and a group-relative policy gradient (REINFORCE with group-centered
rewards; no PPO clipping, no KL term) updates only that topic's vector.

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

## What reproduces exactly, and what does not

| Step | Exact on rerun? |
|---|---|
| Data splits, router labels, prompt pick (seeds fixed) | Yes |
| Picking the test vectors (step 4) | Yes: rebuilt bit for bit from the snapshots and logs (checked 2026-11-25) |
| Training (step 3) | No: answers are sampled and Haiku's verdicts vary, so a rerun gives different vectors |
| Full-length answers (step 6) | Mostly: greedy, but about 1 in 4 long answers changes between identical vLLM runs |
| Haiku scores (step 7) | Close, not exact: the order shuffle is seeded, but Haiku's replies can vary slightly |

## Hardware rules (measured on this box)

- **One model copy per card. Never tensor parallel.** Split layers (pipeline) only if a model does not fit on one card.
- Prefer the second card (no display). Windows gives WSL only about 10-13 GB of the display card, and anything
  over that spills into system RAM.
- Keep the test's step size at 4,096-8,192 tokens and CUDA graphs on (no `--eager`).
