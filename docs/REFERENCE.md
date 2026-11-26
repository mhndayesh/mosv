# MoSV reference: recipe, data, and training methods

The reusable part of this project in one place: how to take a new model from zero to a tested result, what every
data file looks like, the rules for making data, why the training data is built the way it is, and which training
methods worked. Numbers are measured on this box unless marked "estimate" or "not built".

The rules in short form: [TRAINING_BIBLE.md](TRAINING_BIBLE.md). Version write-ups: [V4_MOE.md](V4_MOE.md) (Qwen3-4B), [V5_BONSAI.md](V5_BONSAI.md) (Bonsai-4B),
[V6_GEMMA.md](V6_GEMMA.md) (Gemma 4 12B: what went wrong and v6.0.1). Status and decisions: [PROGRESS.md](PROGRESS.md).

## 1. The idea in six lines

- The base model is frozen. Small **experts** are added at three layers, one expert per topic.
- A **router** picks one expert per token. Expert 0 is "none": all zeros, so the model answers unchanged.
- Everything ships as **one GGUF** that stock llama.cpp and LM Studio run.
- Experts teach **behavior, not knowledge**: how an answer is structured and carried out, not new facts.
- An expert can only move the model toward something it **already does when asked**. It adds no ability.
- So the ceiling of any training is the **told level**: how the model answers when it is given the rules.

## 2. The recipe for a new model

Run from the repo root on Windows, `ai-training-rocm-windows` venv, both cards free. Every step has a pass rule;
do not go on when it fails.

| # | Step | Tool | Pass rule | Time here |
|---|---|---|---|---|
| 0 | Pick the model, lock the settings | see 2.1 | All five checks in 2.1 hold | about 1 hour |
| 1 | Build the prompt splits, lock the test | `python -m behaviors.prepare_v6` | Lock file written; 0 overlap between splits | minutes |
| 2 | Baseline on the locked test | `reports/answer_v5.py`, then `reports/score_v5.py` | 0 failed answers; score, tokens, loops and cut answers recorded | 10 min per 300 prompts |
| 3 | Write the behaviors | `v6_0_1/behaviors.py` | Each topic: rules, 4 yes/no checks, 1 correctness guard (section 4.3) | by hand |
| 4 | Headroom gate | `python -m v6_0_1.gate` | Per topic: told beats plain by 15+ points, guard loses at most 10, checker repeats 85%+ | 25 min |
| 5 | Build the training set | `python -m v6_0_1.build_data --per-topic 300` | Enough kept answers (20+ per topic); the rejected ones read and judged right | about 2 hours for 4 topics (estimate) |
| 6 | Train | `python -m v6_0_1.train --run NAME` | Held-out loss falls at every check; `gap_closed` rises | 4 s per step of 8 short answers |
| 7 | Check both engines | `python -m v6.verify --version v6_0_1 --run NAME --checkpoint steer_XXXX.pt` | Expert effect agrees between PyTorch and llama.cpp (0.81 on strong v6 experts) | 4 min |
| 8 | Evaluate on dev, pick the checkpoint | `python -m v6_0_1.evaluate --run NAME --checkpoint steer_XXXX.pt` | Behavior up, guard not down, untrained topics unchanged | 25 min (estimate) |
| 9 | Locked test, once | the same with `--split test` | The one final number; also the old overall score for comparison | 20 min (estimate) |
| 10 | Ship and document | built GGUF into LM Studio; results into `docs/` | File hash and settings recorded | minutes |

If step 4 fails for a topic: read its told answers. Too short, or guard down? Rewrite the rules and rerun with
`--fresh`. No gain even when told? Drop the topic: nothing can be taught there.

### 2.1 Step 0 in detail: the five checks before anything else

1. **The GGUF layout has a per-layer expert switch** in stock llama.cpp. Known: `hy_v3` (Qwen3-4B, Bonsai-4B) and
   `gemma4` (Gemma 4). An all-zero expert build must answer exactly like the plain model (`moe/gemma_host.py`).
2. **The learner fits one 16 GB card.** Gemma 4 12B needs 4-bit (NF4) and gradient checkpointing with the model in
   train mode: 12 GB peak. In eval mode the same pass needed 31-40 GB.
3. **Recommended sampling, every sampler written out.** Read the model's `generation_config.json`. llama-server adds
   `min_p 0.05` by itself, so set every sampler explicitly. Decide thinking on or off once. Use the same settings in
   baseline, data, training and test. Gemma 4 12B: temperature 1.0, top_k 64, top_p 0.95, min_p 0, thinking off.
4. **The prompt is the same in both engines.** Compare the Hugging Face chat template with llama-server's
   `/apply-template` (Gemma 4: identical, thinking off).
5. **The stop token comes back** with llama.cpp's answer tokens (Gemma 4: `<turn|>`), so training can teach where
   to stop.

### 2.2 Which training method?

Measure three things on the base model first:

| Question | How to measure | Bonsai-4B (RL worked) | Gemma 4 12B (RL failed) |
|---|---|---|---|
| Does the base make gross errors? | Baseline: loops and cut answers of 300 | 30 loops, 107 cut | 0 loops, 0 cut |
| Do answers to one prompt differ? | Judge the same group of 4 twice: same winner | 33.4% (chance 25%) | 25.5% |
| Is there room on the test? | Base score; told score; a strong model's score | Base 1.95 | 3.38 / 3.46 / 3.71 |

- Gross, varied errors and a low base score: **RL** (section 6.2) works and is the simplest path.
- A strong, consistent base: **tell, check, train** (section 6.4). RL has nothing to pick from.

## 3. Data structure

### 3.1 Folders

| Folder | Holds | In git? |
|---|---|---|
| `data_review/` | All prompts, answers, check results, training sets | No. Prompts come from research-only datasets |
| `data_review/v6/<topic>/{train,dev}.jsonl`, `data_review/v6/test.jsonl` | The prompt splits (7 topics; the test file is the locked 300) | No |
| `data_review/router_v2/prompts.jsonl` | Everyday prompts with no topic ("none") | No |
| `data_review/v6_0_1/gate/`, `build/`, `eval/` | v6.0.1 answers and verdicts, one folder per stage | No |
| `data_review/v6_0_1/{train,router,rejected}.jsonl` | The training set, the router texts, the dropped prompts | No |
| `data_review/v5_runs/<topic>/` | Baseline answers and scores in the older format | No |
| `results/` | Reports with numbers only, and the test locks | Yes |
| `checkpoints/<version>/<run>/` | Training runs: logs and expert weights | No |
| `C:/llm/models/<model>/` | Base and built GGUF files | No |

### 3.2 Files and fields

One line of a `.jsonl` file is one row.

| File | One row is | Fields |
|---|---|---|
| Prompt splits | one prompt | `id` (hash), `behavior` (the topic), `source` (dataset), `group` (near-duplicates share it), `prompt` |
| `router_v2/prompts.jsonl` | one everyday prompt | `id`, `split` (train / val / ...), `label` ("none"), `group`, `text` |
| `results/locked_v6.json` | the test lock | `seed`, `topics`, `size` (300), `sha256` of the prompts, `ids`: list of `[id, topic]` |
| v6.0.1 answers (`gate/answers.jsonl`, `build/answers.jsonl`, `eval/*.jsonl`) | one answer | `id`, `topic`, `variant` (`plain`, `told` or `reference`), `copy` (0, 1, ...), `prompt`, `answer`, `tokens`, `finish` (`stop`, `length` or `loop`) |
| v6.0.1 verdicts (`*/checks.jsonl`, `eval/*_checks.jsonl`) | one judged answer | `id`, `topic`, `variant`, `copy`, `round` (1, or 2 for the repeat test), `pass`: `{"b1": true, "b2": false, "b3": true, "b4": true, "g1": true}` |
| `v6_0_1/train.jsonl` | one kept pair | `id`, `topic`, `prompt`, `answer` (the told answer), `behavior` (share of checks it passed), `base_behavior` (the plain answer's share), `tokens`, `base_tokens` |
| `v6_0_1/router.jsonl` | one router text | `label` (a trained topic or "none"), `prompt`, `answer` |
| `v6_0_1/rejected.jsonl` | one dropped prompt | `id`, `topic`, `reason`, `plain_behavior`, `told`: list of `{behavior, guard, finish, tokens, answer}` |
| `<stage folder>/behaviors.stamp` | a marker | The fingerprint of `behaviors.py` the cached answers were written under |
| `results/v6_0_1_gate.json` | the gate report | `settings`; per topic: `plain`, `told`, `reference` (each `{n, behavior, guard, tokens, checks}`), `repeat`, `train` (true / false), `reason` |
| `results/v6_0_1_data.json` | the build report | per topic: `prompts`, `kept`, `outcomes` (why rows were dropped), `plain_behavior`, `kept_behavior`, `plain_tokens`, `kept_tokens` |
| `results/v6_0_1_eval_<split>_<run>_<checkpoint>.json` | an evaluation | per topic: `base` and `trained` (`{n, behavior, guard, tokens, checks}`), `prompts`, `trained_wins_pct` |
| `checkpoints/v6_0_1/<run>/log.jsonl` | one event | `event` = `start` (settings, classes, counts), `router` (held-out accuracy per layer), `levels` (`base_ce`, `teacher_ce`), `step` (`loss`, `seconds`, `peak_gb`), `eval` (`val_ce`, `gap_closed`, `routed`, `w2_per_dim`), `done` (`best_step`) |
| `checkpoints/.../steer_XXXX.pt` | a checkpoint | `steer` (`classes`, `layers`, `ff`, `gate` = router matrices, `experts`), `update` (step), `val_ce`, `base_ce`, `teacher_ce`, `args` |
| Older baseline answers (`v5_runs/<topic>/<tag>.jsonl`) | settings header, then one answer per row | `id`, `source`, `prompt`, `answer`, `tokens`, `chars`, `finish`, `seconds`, `thinking` |
| Older scores (`v5_runs/<topic>/score_<tag>.jsonl`) | one scored answer | `id`, `source`, `overall` (1-5), `criteria` (five 1-5 values), `tokens`, `finish` |
| RL run folders (`checkpoints/v5/`, `checkpoints/v6/`) | per update | `log.jsonl` (reward, KL, lr, undone steps, dev checks), `judge.jsonl` (every judge call, no answer text, no key), `samples.jsonl` (best and worst answer of one group) |

### 3.3 Identity rules

- One answer is identified by (`id`, `variant`, `copy`); a verdict adds `round`. A rerun skips what is already there.
- Every answer has a fixed seed made from its identity, so a rerun aims at the same text. Parallel slots can still
  change a few sampled words, so numbers move slightly between runs.
- `plain` is always one copy per prompt. `told` has as many copies as asked for. `reference` is the judge model
  answering the plain prompt.

## 4. Data-making rules

### 4.1 Prompts

1. **Real prompts only**, from datasets. Research-only sources are fine while the prompts stay private: only ids
   and hashes go into git.
2. **Tasks to do, not questions to look up.** "Write / implement / refactor / review this" teaches behavior; "what
   is / explain the difference" tests knowledge (`behaviors/prepare_v5.py` filters these by pattern).
3. **The prompt must contain what is needed to answer.** Drop prompts that point at a missing snippet. If a source
   lacks information by nature (bug reports without code, emails with empty key points), teach the behavior that
   fits: honesty about guesses, placeholders instead of invented facts.
4. **Leave room for the answer.** Cap prompt length (2,500 characters for engineering) and drop prompts so long
   that no answer fits.
5. **Splits by group.** Near-duplicates share a `group`, and the split comes from a hash of the group, so a test
   prompt can never appear in train or dev.
6. **Balance the sources** inside a topic and cap its size (3,000 train / 300 dev / 300 test for engineering).
7. **Filter by keyword carefully.** Sexual-themed prompts are removed; in code data, words like `std`, `sex`
   (a column name), `kill`, `strip` and `execute` are protected from that filter.
8. **Lock the test before any training.** `results/locked_v6.json` holds the ids and a sha256; a rebuild that would
   change it stops unless `--force` is given.
9. **Three splits, three jobs.** Train: only for teaching. Dev: every decision (gate, checkpoint choice). Test: one
   final number, run once.

### 4.2 Answers

1. **The same settings everywhere**: baseline, data, training and test. Recommended sampling with every sampler
   written out, the same thinking setting, KV cache type and token cap. Save the settings with the answers.
2. **Let answers run free** up to 2,048 tokens. Three stop rules, the same for every model: the stop token, a loop
   (the last 400 characters already appear 4 times earlier), or the cap.
3. **Save each answer the moment it ends**, and resume by identity. A stop never loses finished work.
4. **One model per card** in llama.cpp, with separate slots (13 on the display card, 16 on the other for Gemma at
   3,072 tokens per slot). Measured: about 0.6 answers per second in total.

### 4.3 Behaviors: rules and checks (`v6_0_1/behaviors.py`)

Rules told to the model:

- Say **what to do**, not only what to avoid. "Do not invent facts" gave nothing until the rule added "write a
  bracketed placeholder instead".
- Keep the answer whole. A rule like "answer in one or two sentences" shrank explanations from 807 to 102 tokens
  and correctness fell. Put "meet every requirement" before "keep it simple".
- Watch the token count of told answers in the gate. A large drop is the first sign of rules that are too tight.

Yes/no checks for the judge:

- **Four behavior checks and one correctness guard per topic.**
- Each question is answerable from the request and the answer alone, asks about **one** thing, and is phrased so
  that **yes is good**.
- Ask about something visible: "Does the answer speak to the child directly, instead of describing a lesson plan to
  a teacher?" (plain 0%, told 100%).
- A question the plain model already passes 90%+ of the time carries no gain and hides the others. Replace it.
- The **guard** asks whether the answer is still correct. It must not drop while behavior rises.

### 4.4 The keep rule: what goes into the training set (`build_data.choose`)

A told answer is kept only if all of these hold:

1. It is **complete** (ended on the stop token). A cut answer would teach stopping mid-way.
2. It passes **every guard**.
3. It passes **at least 3 of 4** behavior checks.
4. It passes **more checks than the plain answer** to the same prompt. Otherwise there is nothing to teach.
5. With several told copies, the best one is kept; on a tie, the shorter.

**Read the rejects.** Every dropped prompt is saved with its reason. Measured on 90 prompts: 51 kept; drops were
"too few checks passed" (19), "not better than plain" (17) and "guard failed" (3), and the samples read were right
to drop.

### 4.5 Sample sizes

| Decision | Prompts | Why |
|---|---|---|
| Gate | 60 per topic | With 20, the plain score of the same prompts moved up to 13 points between runs and one answer flipped a verdict |
| Checkpoint choice | 60 per topic on dev | In-training checks on 42 prompts swung from 0.49 to 0.60 on the base model alone |
| Final test | 300, locked | One run, one number |

### 4.6 Saved answers and changed rules

After editing `behaviors.py`, saved answers and verdicts are stale. Each stage stores a fingerprint of the rules
next to its files and **stops** if it differs. Rerun the stage with `--fresh`.

## 5. Training-data philosophy

1. **Teach how, not what.** Experts change how the model answers. Facts stay in the frozen weights. A topic whose
   score depends on getting the answer right (math) is left untrained.
2. **Use the model's own words.** The training answers are written by the same model, told the rules. No outside
   knowledge enters, and every answer is something this model can produce.
3. **Verified better, not hoped better.** Every kept answer passed the checks and beat the plain answer to the
   same prompt. v6 hoped one of four random samples would be better; it was not.
4. **The better answer must differ by more than the judge's noise.** Random samples of a strong model do not;
   a told answer does (teaching: 57% -> 99% of checks).
5. **Behavior must not cost correctness.** That is what the guard is for, in the gate and in the keep rule.
6. **Leave everything else alone.** Everyday prompts and untrained topics are routed to "none" and answer as before.
7. **Do not reward shortness for its own sake.** In v5 a flat length penalty kept shortening answers until quality
   fell (dev 72.9% -> 58.3%). Penalize length only against the base answer, or not at all.
8. **A small clean set beats a large noisy one.** One kept answer gives a clear signal at every token.
9. **Hold some answers out.** Progress is read on answers never trained on, and the locked test is touched once.

## 6. Training methods and what works

### 6.1 The methods used so far

| Method | Version, model | How | Result |
|---|---|---|---|
| RL on steering vectors | v3, Qwen3-4B | Group-relative RL, Haiku judge, vLLM writes the answers | Blind test, 80 prompts: 2.89 -> **3.54**. 5 topics improved; engineering and how-to did not |
| Distill vectors into experts | v4, Qwen3-4B | The v3 vectors and router copied into MoE layers of one GGUF | **3.61**, one file for stock llama.cpp and LM Studio |
| RL straight into the GGUF experts | v5, Bonsai-4B 1-bit | The same RL, llama.cpp writes the answers from the live GGUF | Locked 300: 1.95 -> **2.42**, better on 66.2% of prompts, 1,212 -> 268 tokens, loops 30 -> 0 |
| The same RL | v6, Gemma 4 12B | 7 topics | **No gain** (base 3.39, checkpoints 3.30 to 3.41) |
| Tell, check, train | v6.0.1, Gemma 4 12B | Supervised on checked told answers | Gate passes 4 topics; small test: held-out loss 1.214 -> 1.081 in 6 steps. **Full run not done** |

### 6.2 The RL method (v3, v5, v6) and the settings that worked on Bonsai

How one update works:

1. Sample a group of answers per prompt (8 prompts x 4 answers, up to 1,024 tokens each).
2. Haiku sees the group and returns a 1-5 score per answer plus a ranking, as strict JSON. Each group is judged in
   one label order and in its exact reverse, which cancels letter-order bias. A failed call is skipped and never
   counts as a low score.
3. Reward per answer = half from its rank in the group, half from its score, between 0 and 1.
4. Advantage = reward minus the group's average. No scaling.
5. Loss = minus (advantage x log-probability of the sampled tokens). Only the experts get a gradient.
6. Safety: after the step, measure how far the model's next-token choices moved (KL). Above the limit, undo the
   step and halve the learning rate.

| Setting | Value that worked (v5, Bonsai) | Note |
|---|---|---|
| Batch | 8 prompts x 4 answers | |
| Optimizer | Adam, start lr 5e-3, settled at 1.25e-3 | lr 1e-4 learned nothing |
| KL limit | 0.01, undo and halve | Save the optimizer state; warm up after a resume |
| Judge | Haiku, 16 calls at once | 0 errors in 2,500+ calls |
| Judge window | 768 words (the whole 1,024-token answer) | The older 190-word window hid rambling and endings |
| Length | reward minus 0.1 x tokens / 1,024 | Helped to update 100, then over-shortened. **Rule now: only against the base answer, or off** |
| Sampling | llama.cpp on the display card only, 32 slots | Sharing the learner's card spilled 1.1 GB into system RAM |
| Weight sync | Expert bytes patched in place in the GGUF, then server restart | About 2 s per update |
| Checkpoints | One at every dev check; pick by dev, not the last update | Dev peaked at update 100 (72.9%), fell to 58.3% by 150 |
| Update time | 40 to 60 s | |

Bonsai sampled at temperature 1.0 with top_k 200 for speed. The rule since then is the model's recommended
sampling everywhere (section 2.1).

### 6.3 When RL works, and when it does not

| Condition | Bonsai-4B: worked | Gemma 4 12B: failed |
|---|---|---|
| Base score on the locked test | 1.95 | 3.43 |
| Training answers scored 1 or 2 by the judge | 63% | debugging 54%, every other topic 15% or less |
| Judge picks the same winner twice | 33.4% | 25.5% (chance 25%) |
| Rank agreement between the two judgings | +0.21 | +0.04 |
| Groups where all four scores are equal | 14% | 40% |
| Gross failures in base answers (of 300) | 30 loops, 107 cut | 0 loops, 0 cut |

RL needs a better answer to show up among the samples, and a judge that can see it. A weak model varies a lot, so
it does. A strong model has fixed habits: every sample has the same flaw, and the judge picks at random.

**Before any RL run, check:** the judge agrees with itself well above chance, the base model has room on the test,
and the reward repeats. If not, use section 6.4.

### 6.4 Tell, check, train (v6.0.1)

1. **Gate:** plain vs told on dev prompts, judged with yes/no checks, twice.
2. **Build:** told answers on training prompts; keep the ones that pass the keep rule.
3. **Train:** supervised. The model reads the plain prompt and the kept answer; the experts are adjusted so each
   next token of that answer becomes more likely. The stop token is the last target.
4. **Progress without a judge:** loss on held-out answers. `base_ce` = plain model, `teacher_ce` = plain model shown
   the rules, `val_ce` = model with experts. `gap_closed` = (base - val) / (base - teacher).

| Setting | Value | Note |
|---|---|---|
| Batch | 16 answers per step | The small test used 8 |
| Optimizer | AdamW, experts lr 1e-3, expert strength lr 3e-3 | Strength starts at 0, so the model starts unchanged |
| Epochs | 3 | A check and a checkpoint every 25 steps |
| Router | Fitted once before training, on recorded plain and kept answers | Untrained topics and everyday prompts are labeled "none" |
| Memory | 11.9 GB peak at 4,096 tokens per pass | 4-bit learner with checkpointing |

Measured so far: the checker repeats 95 to 98%; the gate passes debugging, teaching, explanation and story; in a
6-step test the held-out loss fell at every check and the expert strength grew in a steady line.

### 6.5 What did not work

| Tried | Result |
|---|---|
| RL with a 4-answer ranking judge on a strong model | Reward at chance level, no gain (v6) |
| Raising the learning rate when nothing was learned | Bigger random pushes; dev fell to 38.1% (v6) |
| A flat length penalty | Answers kept shrinking past the useful point (v5) |
| One overall 1-5 score as the target | Behavior gains vanish in it (v6) |
| 256-token answers in training | The engineering gain did not carry over to full-length answers (v3) |
| Topics that are opinion or ability, not a skill | How-to got worse with its vector (v3); math is a tie everywhere |
| Tensor parallel across the two cards; both cards from WSL | Half the speed and RAM spills; a PC freeze |

The full list for v6 is in [V6_GEMMA.md](V6_GEMMA.md), section 2.

### 6.6 RL as a final polish (not built, not tested)

Once tell, check, train has put the behavior in, RL could refine it with a reward that repeats: the share of
yes/no checks one answer passes, with the guard as a gate, one answer per judge call, and several prompts per topic
in every update. Run the checks of section 6.3 first.

## 7. Checklists

Before training anything:

- [ ] Step 0 checks done (layout, memory, sampling, prompt format, stop token).
- [ ] Test locked; baseline recorded with the same settings the test will use.
- [ ] Gate run on 60 prompts per topic; only passing topics go on.
- [ ] The reward or checker repeats when the same answers are judged twice.
- [ ] Rejected training data read.

Before claiming a result:

- [ ] The checkpoint was picked on dev, with at least 60 prompts per topic.
- [ ] The locked test ran once, with plain prompts and the baseline's settings.
- [ ] The guard did not drop, and untrained topics are unchanged.
- [ ] The trained experts were checked in both engines.
- [ ] The command and its real output are in the write-up.

## 8. Where things are

| Need | Look in |
|---|---|
| Prompt building and locks | `behaviors/prepare_v5.py`, `behaviors/prepare_v6.py`, `results/locked_v6.json` |
| Baseline answers and the 1-5 scorer | `reports/answer_v5.py`, `reports/score_v5.py`, `rl/judge_rubrics.py` |
| Behaviors, checker, gate, data, training, evaluation | `v6_0_1/` (one file per stage) |
| Expert layout and GGUF writing | `v6/steer.py`, `moe/gemma_host.py` (Gemma 4); `moe/export.py`, `moe/live_gguf.py` (hy_v3) |
| RL trainers | `v5/train.py` (Bonsai), `v6/train.py` (Gemma), `rl/` (judge, batching, v3 pilot) |
| Engine agreement check | `v6/verify.py`, `v5/verify.py` |
| Hardware rules | [PROGRESS.md](PROGRESS.md), "Hardware rules"; `README.md` |
