# v6 on Gemma 4 12B: what went wrong, what is right, and v6.0.1

Branch `bonsai`, 2026-11-26. Code: `v6/` (the RL version that gave no gain) and `v6_0_1/` (the corrected version).
The reusable recipe, the data structure, the data rules and the training methods are in [REFERENCE.md](REFERENCE.md).

**In four lines**

- v6.0 moved the v5 method (RL with a Haiku judge) from Bonsai-4B to Gemma 4 12B. Two runs gave no gain.
- The code was not broken; the target was. On the same test Gemma scored 3.38, Gemma told the rubric 3.46 and
  Haiku itself 3.71. There was almost nothing to win, and the training reward was noise.
- v6.0.1 measures the room first, scores single behaviors with yes/no checks, and teaches by "tell, check, train"
  instead of trial and error.
- Status of v6.0.1: **done and shipped** (2026-11-27). Three topics trained (debugging, teaching, story). On the
  locked test the trained model passes more behavior checks on 98% / 81% / 62% of prompts. Section 5.6 has the
  final result; the write-up is [paper/paper_v6_0_1.md](../paper/paper_v6_0_1.md).

Words used below: **plain** = the user's prompt alone. **Told** = the same prompt with the topic's rules in a system
message. **Guard** = a yes/no question about correctness that must not get worse.

## 1. What v6.0 was

| Part | v6.0 |
|---|---|
| Model | Gemma 4 12B, frozen. Answers are written by `gemma-4-12b-it-Q4_K_M.gguf` in llama.cpp; learning runs on the same model in 4-bit (NF4) in PyTorch |
| What is trained | Gated experts at layers 12, 24 and 36: one expert per topic, a router picks one per token, expert 0 = "none". Stock llama.cpp gemma4 layout (`moe/gemma_host.py`, `v6/steer.py`) |
| Topics | story, teaching, debugging, math_solving, explanation, professional_email, engineering |
| Sampling | The model's recommended settings: temperature 1.0, top_k 64, top_p 0.95, min_p 0, thinking off |
| Method | RL. Per update: 7 prompts x 4 answers. Haiku sees the 4 answers together and returns 1-5 scores plus a ranking. Reward = half rank, half score, minus the group average |
| Baseline | Locked test, 300 prompts, each answer scored alone by Haiku: **3.43** |

| Run | Setting | Result |
|---|---|---|
| `gemma_full` | 125 updates, lr 1e-3, then 3e-3 | In-training dev checks (42 prompts): 59.5, 53.6, 58.3, 56.0, 52.4% wins. On 140 dev prompts scored one by one: base 3.39, checkpoints 3.30 to 3.41. **No gain** |
| `gemma_w2` | 96 updates, expert strength lr raised 10x (0.03) | Dev wins 53.6, 59.5, then **38.1%**. Stopped |

## 2. The issues: wrong techniques and what each one caused

| # | Wrong technique | What it caused | Evidence |
|---|---|---|---|
| 1 | Training before checking that the test has room | No method could have shown a gain: the most to win was about +0.3, by behavior alone about +0.1 | Same 140 dev prompts, same scorer: Gemma **3.38**, Gemma told the rubric **3.46**, Haiku **3.71**. Bonsai's base was 1.95, so v5 had room |
| 2 | Trial-and-error RL on a model whose answers do not vary | RL can only reward a better answer if one shows up among the samples. Gemma's flaws are habits, present in every sample | Scores of two separate answers to the same prompt correlate 0.86 (140 prompts, all topics together). The best of 6 answers is only 0.2 to 0.5 points above their average |
| 3 | A 4-answer comparison as the reward | The reward was noise | 1,539 groups judged twice: same winner **25.5%** (chance is 25%), rank agreement +0.04, all four scores equal in 40% of groups. Bonsai: 33.4%, +0.21, 14% |
| 4 | One overall 1-5 score | Knowledge and behavior are mixed, so real behavior changes disappear in it | Told the rubric, debugging "honest about guesses" went 1.4 -> 2.8 and explanation "plain words" 2.4 -> 4.0, yet the overall score moved +0.08 |
| 5 | Prompts that cannot be answered well | The score is capped by the prompt, not by behavior | Debugging prompts are bug reports with no code (Haiku itself scores 2.80). Email prompts list empty key points ("Compliance status"). Math measures ability: telling Gemma rules made it worse (4.10 -> 3.80) |
| 6 | Tiny batches | Each topic's expert learned from 4 answers per update | 7 prompts x 4 answers of about 650 tokens per update |
| 7 | Tuning the learning rate on noise | Two lr changes that could not help; the second made the model worse | Expert strength moved like a random walk: 0.304 per dimension after 94 updates; a random walk predicts 0.291, a real direction 2.82 |
| 8 | Trusting small dev checks | "59.5% wins" looked like learning | On 42 prompts the base model alone swung from 0.49 to 0.60 between checks |
| 9 | A safety guard that looked in the wrong place | Run 2 drifted to 38.1% with 0 rejected steps | The KL guard read only the last 8 token positions of at most 8 prompts, all from two topics |
| 10 | Old rubrics and a short judge window | The reward read knowledge, and only the start of each answer | 6 of 7 topics used the v3 rubrics; the training judge read the first 190 words of answers that averaged 600 to 750 tokens |
| 11 | No end-to-end check of trained experts before the runs | Two long runs without knowing the experts acted the same in both engines | The check was written afterwards. It passed (section 3), but it belongs before training |

Two wrong diagnoses made along the way, corrected later:

- "The experts are too weak." Run 2 raised their strength about 20x and the model got worse, not better.
- "The forced ranking adds letter-order noise." Each group is judged in two opposite label orders (669 of 669
  groups checked), which cancels that effect.

## 3. What was checked and found fine

| Check | Result |
|---|---|
| Trainer math | The sign of the loss, the token alignment and the stop token are right (`v6/train.py`, `rl/policy.py`) |
| Experts act the same in PyTorch and in the GGUF | `v6/verify.py`: the change the experts cause in token probabilities correlates **0.81** between the two engines, with the same size (1.53 vs 1.46) |
| Learner vs shipped model | The 4-bit learner and the Q4_K_M GGUF agree on the top token 90% of the time |
| Prompt format | The Hugging Face chat template and llama.cpp build the same prompt (thinking off) |
| Test scorer | One answer per call: the same answers scored twice got the same overall score **88%** of the time (correlation 0.93) |
| Sampling | Recommended settings, every sampler written out |

So the measuring tool and the plumbing were sound. The target and the reward were the problem.

## 4. The correct techniques, and why

| Correct technique | Why | Where in v6.0.1 |
|---|---|---|
| Measure the room before training | Experts can only move the model toward what it already does when asked. If telling it adds nothing, training adds nothing | `gate.py` |
| Score single behaviors with yes/no questions, one answer per call | A concrete question repeats; a 1-5 taste score does not. A correctness guard stops "better behavior, wrong answer" | `behaviors.py`, `checks.py` |
| Check that the reward repeats | A reward that does not repeat is noise, whatever the optimizer does with it | `gate.py` judges every answer twice |
| Make the better answer on purpose | Tell the model the rules, check the answer, keep it only if it passes and beats the plain answer to the same prompt | `build_data.py` |
| Teach with supervised training | Every token of a kept answer is a clear signal. No lucky sample is needed | `train.py` |
| Watch progress without a judge | Loss on held-out answers, placed between "plain model" and "plain model told the rules" | `train.py` (`gap_closed`) |
| Fit the behavior to what the prompt allows, and leave the rest alone | For a bug report with no code, the right behavior is honesty about guesses. Math and skipped topics are routed to "none" and stay unchanged | `behaviors.py`, `router.py` |
| Decide on enough prompts | One answer must not flip a verdict | Gate default: 60 prompts per topic. Pick the checkpoint on dev; run the locked test once |
| Keep saved answers honest | Answers written under old rules must never mix with new ones | `files.claim`, the `--fresh` option |
| RL only as a final polish | Once the behavior is in, RL with the yes/no reward can refine it | Not built yet |

## 5. What happens when it is done correctly

Everything in 5.1 to 5.5 is measured, on dev or training prompts. The locked test was not touched.

### 5.1 The reward repeats

| Reward | Same result when the same answers are judged twice |
|---|---|
| v6: Haiku ranks 4 answers | Same winner 25.5% of the time (chance 25%) |
| v6.0.1: Haiku answers yes/no questions about 1 answer | Same yes/no answer **95% to 98%** of the time, per topic (3,600 yes/no answers judged twice) |

### 5.2 The gate shows where the room is

`python -m v6_0_1.gate`, 60 dev prompts per topic, about 25 minutes. "Behavior" = share of the topic's four behavior
checks passed.

| Topic | Behavior, plain | Behavior, told | Gain | Haiku (plain) | Guard, plain -> told | Checker repeats | Tokens, plain -> told | Verdict |
|---|---|---|---|---|---|---|---|---|
| debugging | 0.17 | 0.61 | **+44** | 0.59 | 0.98 -> 1.00 | 95% | 844 -> 488 | **train** |
| teaching | 0.57 | 0.99 | **+42** | 0.81 | 0.90 -> 0.88 | 98% | 998 -> 280 | **train** |
| explanation | 0.34 | 0.69 | **+35** | 0.39 | 0.73 -> 0.68 | 96% | 772 -> 239 | **train** |
| story | 0.63 | 0.79 | **+16** | 0.76 | 0.98 -> 1.00 | 95% | 904 -> 569 | **train** |
| engineering | 0.75 | 0.89 | +14 | 0.80 | 0.57 -> 0.60 | 97% | 765 -> 271 | skip: gain under 15 |
| professional_email | 0.65 | 0.73 | +8 | 0.75 | 0.98 -> 0.82 | 95% | 289 -> 144 | skip: gain under 15 |

Read it like this: a topic is trained when telling the model adds at least 15 points, the correctness guard loses
at most 10 points, and the checker repeats itself at least 85% of the time. The Haiku column is Haiku answering the
plain prompt; it is shown for reference and is not part of the rule.

What the single questions show (plain -> told, 60 answers each; all of them are in `results/v6_0_1_gate.json`):

| Topic | Question | Plain | Told |
|---|---|---|---|
| teaching | Speaks to the child directly, not a lesson plan for a teacher | 0.00 | 1.00 |
| teaching | Language right for the stated grade | 0.42 | 1.00 |
| debugging | Says how to check or confirm a cause | 0.23 | 1.00 |
| debugging | Separates what the report states from what is guessed | 0.00 | 0.42 |
| explanation | Short sentences and everyday words | 0.02 | 0.60 |
| story | Opens with an action, a voice or a specific image | 0.13 | 0.37 |
| engineering | Keeps to what was asked, no extra features | 0.43 | 0.82 |
| professional_email | Every fact is given in the request or is a placeholder | 0.63 | 0.93 |

- Four topics have real room. That is a different picture from the old overall score, where all seven looked flat.
- Engineering misses by one point. Plain Gemma already passes 75% of its checks, and its weak spot is the
  correctness guard (0.57, Haiku 0.80), which is ability, not behavior.
- Email: "do not invent facts" works (0.63 -> 0.93), but the told emails are half as long and stop covering the key
  points (guard 0.98 -> 0.82, "no filler" 0.67 -> 0.42). Its rules need another pass before it can be trained.

### 5.3 The gate catches a bad instruction before any training

The first version of the rules made answers too short:

| Topic | First rules | What the gate showed (20 prompts) | After rewriting the rules (60 prompts) |
|---|---|---|---|
| explanation | "Answer in one or two sentences" | Told answers shrank from 807 to 102 tokens; correctness fell 0.70 -> 0.60 | 239 tokens; correctness 0.73 -> 0.68: passes |
| engineering | "Simplest design" | 746 -> 243 tokens; correctness fell 0.70 -> 0.50 | "Meet every requirement" first: correctness 0.57 -> 0.60, no drop |
| professional_email | No rule about coverage | 302 -> 135 tokens; "covers the key points" fell 1.00 -> 0.75 | "Cover every key point" added: still 0.98 -> 0.82, not fixed |
| story | "Open inside the action" | Gain +11: skipped | "First sentence = an action or a line of speech": gain +16, passes |

The first two gate runs are saved as `results/v6_0_1_gate_guides_v1.json` and
`results/v6_0_1_gate_guides_v2_20prompts.json`. Under v6 these instructions would have gone straight into the reward. The first two gate runs also exposed the
gate's own weakness: with 20 prompts per topic, the plain score of the same prompts moved by up to 13 points between
runs and one answer flipped a verdict. The default is now 60.

### 5.4 The training set is made of verified better answers

Small test, 30 training prompts per topic, one told answer each (`build_data.py`, report in
`results/v6_0_1_data.json`; run before explanation passed the gate, so it covers three topics):

| Topic | Kept | Behavior, plain answer | Behavior, kept answer | Tokens, plain -> kept | Why the others were dropped |
|---|---|---|---|---|---|
| debugging | 14 of 30 | 0.17 | 0.82 | 838 -> 504 | 16 passed fewer than 3 of 4 checks |
| story | 10 of 30 | 0.66 | 0.88 | 900 -> 565 | 17 were no better than the plain answer, 3 passed too few checks |
| teaching | 27 of 30 | 0.54 | 0.99 | 1,012 -> 291 | 3 failed the correctness guard |

The dropped answers were read: they were the right ones to drop (for example a phonics lesson with a wrong fact,
caught by the guard). Dropped prompts are saved with the reason in `data_review/v6_0_1/rejected.jsonl`.

### 5.5 Training moves in one direction

Small test, teaching only: 24 answers, 6 steps of 8 answers, about 4 seconds per step, 11.9 GB peak (`train.py`).
Loss per token on 3 held-out answers: plain model 1.214, plain model told the rules 0.589.

| Step | Held-out loss | Gap closed | Expert strength per dimension |
|---|---|---|---|
| 0 | 1.214 | 0% | 0 |
| 2 | 1.180 | 5% | 0.0051 |
| 4 | 1.132 | 13% | 0.0096 |
| 6 | 1.081 | 21% | 0.0143 |

The loss fell at every check and the strength grew in a straight line: 0.0143 after 6 steps, where a steady
direction gives at most 0.018 and a random walk 0.007. In v6 the same number followed the random walk. This is a
small test on 3 held-out answers: it shows the signal is real, not how good the final model will be.

### 5.6 The full run (2026-11-27)

Topics trained: debugging, teaching, story (clear passes in both 60-prompt gate runs; explanation and engineering
flipped between runs and were left out). Run `gemma_v601`:

- **Data:** 300 training prompts per topic, 2 told answers each; kept 221 / 272 / 181 (debugging / teaching / story).
- **Training:** 611 answers, 63 held out, 117 steps. Held-out loss 1.218 (plain) -> 0.642, below the told level
  0.684. Router held-out accuracy 83% / 97% / 97% at layers 12 / 24 / 36.
- **Checkpoint choice (dev, 60 prompts per topic):** step 100, best or tied best on all three topics
  (teaching 0.58 -> 0.99, story 0.68 -> 0.82, debugging 0.16 -> 0.40).

**Locked test, checkpoint 100** (300 prompts, plain prompts, answered by the built GGUF in llama.cpp, the baseline's
settings; `results/v6_0_1_eval_test_gemma_v601_steer_0100.json`):

| Topic | Behavior, base -> trained | Trained better on | Guard, base -> trained | Tokens, base -> trained |
|---|---|---|---|---|
| **Teaching** | 0.58 -> **0.97** | **98%** | 0.95 -> 0.91 | 986 -> 325 |
| **Debugging** | 0.17 -> **0.44** | **81%** | 0.98 -> 0.95 | 828 -> 544 |
| **Story** | 0.73 -> **0.81** | **62%** | 1.00 -> 1.00 | 860 -> 711 |
| Explanation (not trained) | 0.34 -> 0.34 | 50% | 0.70 -> 0.81 | 736 -> 739 |
| Email (not trained) | 0.69 -> 0.69 | 49% | 1.00 -> 0.98 | 292 -> 286 |
| Engineering (not trained) | 0.77 -> 0.71 | 39% | 0.69 -> 0.79 | 785 -> 765 |

- Teaching reaches its ceiling (told: 0.99); story is close (0.83); debugging more than doubles but stays below
  its ceiling (0.58).
- Correctness cost on teaching: 0.95 -> 0.91.
- **Engineering, not trained, lost ground** on the locked test (0.77 -> 0.71, 39% wins), though it gained on dev
  (0.75 -> 0.79). Likely cause: engineering tokens routed to the debugging expert. Fix to try: fit the router with
  engineering prompts as their own group of "none" texts.
- The old overall 1-5 score was not rerun on the trained model.
- **Checked inside the GGUF** (`v6/verify.py`): in llama.cpp the experts change next-token log-probabilities by
  1.6-2.0 on trained topics and 0.13-0.21 on untrained topics and everyday prompts (top token changed 22-33% vs
  0-2%); PyTorch and llama.cpp agree at 0.89. Chat demo: `results/v6_0_1_demo.md`.

## 6. v6.0.1: the code

Run every stage from the repo root in the `ai-training-rocm-windows` venv, with the cards free. Windows only.

| Step | Command | What it does | Time | Writes |
|---|---|---|---|---|
| 1 | `python -m v6_0_1.gate` | Plain, told and Haiku answers on dev prompts; yes/no checks, twice; a train / skip verdict per topic | about 25 min | `results/v6_0_1_gate.json` |
| 2 | `python -m v6_0_1.build_data --per-topic 300` | For the topics marked "train": plain and told answers on training prompts, checks, keeps the answers that pass | about 2 hours for the 4 passing topics (estimate from 0.6 answers per second) | `data_review/v6_0_1/{train,router,rejected}.jsonl`, `results/v6_0_1_data.json` |
| 3 | `python -m v6_0_1.train --run NAME` | Fits the router, then supervised training of the experts on the kept answers | measured 4 s per step of 8 short answers; about 30 min for 500 kept answers (estimate) | `checkpoints/v6_0_1/NAME/` |
| 4 | `python -m v6_0_1.evaluate --run NAME --checkpoint steer_XXXX.pt --split dev --per-topic 60` | Builds the GGUF, answers plain prompts with it and with the base model, checks both | about 25 min (estimate) | `results/v6_0_1_eval_dev_...json` |
| 5 | The same with `--split test`, once, for the picked checkpoint | The locked 300 | about 20 min (estimate) | `results/v6_0_1_eval_test_...json` |

After editing `v6_0_1/behaviors.py`, add `--fresh` to steps 1, 2 and 4: saved answers written under the old rules
are refused otherwise. For the old overall score on the locked test, use the existing tools:
`reports/answer_v5.py --topic v6 --model <the built GGUF> --tag v601 ...` then `reports/score_v5.py --topic v6`.

| File | Job |
|---|---|
| `v6_0_1/behaviors.py` | The one place that defines good behavior: per topic, the rules told to the model, four yes/no behavior checks and one correctness guard |
| `v6_0_1/checks.py` | The judge: one Haiku call per answer, yes/no only, strict JSON |
| `v6_0_1/answers.py` | Writes plain or told answers with llama.cpp on both cards, with the baseline's settings |
| `v6_0_1/gate.py` | The headroom gate (step 1) |
| `v6_0_1/build_data.py` | Tell, check, keep (step 2) |
| `v6_0_1/router.py` | Fits the per-token router; untrained topics and everyday prompts go to "none" |
| `v6_0_1/train.py` | Supervised training of the experts (step 3) |
| `v6_0_1/evaluate.py` | Builds the GGUF and compares it with the base model (steps 4 and 5) |
| `v6_0_1/files.py` | File locations and small read / write helpers |
| `test_v6_0_1.py` | Tests of the decision rules (gate verdict, keep rule, scoring, split, saved-answer check) |
| `v6/verify.py` | Check that trained experts act the same in PyTorch and in the GGUF (`--version v6_0_1` for v6.0.1 runs) |

Reused, not copied: the server start and streaming request (`reports/answer_v5.py`), the expert layout
(`v6/steer.py`, `moe/gemma_host.py`), the memory-saving backward pass (`v6/train.py`), the judge model and key
(`rl/api_judge.py`). Math is not trained: its prompts are routed to "none".

## 7. Rules for the next model

1. **Gate first.** Before any training, compare plain vs told (and a strong model for reference). Train only where
   telling the model clearly helps and correctness holds.
2. **The reward must repeat.** Judge the same answers twice. Do not touch the optimizer until this passes.
3. **Decide on enough prompts.** If one answer can flip the verdict, the sample is too small.
4. **Score behaviors, not overall quality.** Yes/no questions per behavior, plus a correctness guard.
5. **Give the model answers that differ.** If samples of one prompt score alike, make the better answer on purpose
   (tell, check) and teach it directly.
6. **Check trained experts in both engines** before a long run (`v6/verify.py`).
7. **Recommended sampling, every sampler written out**, in baseline, training and test (unchanged rule).

## 8. Limits and open items

- Engineering leaks into an expert on the locked test (section 5.6).
- The prompts were not replaced. The behaviors were chosen to fit them (honesty for bug reports without code,
  placeholders for emails without facts). Better prompt sources remain an option.
- Haiku is the checker, the reference writer and the old scorer. A second judge model would be a stronger check.
- The router is fitted before training and not refitted afterwards.
- There is no test yet that everyday prompts (the "none" class) answer exactly as before after training.
- The 4-bit learner and the shipped Q4_K_M model differ on 10% of top tokens, so a trained effect transfers at
  about 0.81, not 1.0.
- `v6/verify.py --version v6_0_1` runs on v6.0.1 checkpoints. On the 6-step test checkpoint the experts' effect is
  still tiny (0.12 in PyTorch, 0.15 in llama.cpp, against 1.5 for the strong v6 experts) and the two engines agree
  only 0.44 on it, because an effect that small is near the learner's rounding noise. Run it again on the full
  checkpoint; it should be back near 0.8.
- The yes/no checker is not perfect: its repeat rate is high, but single verdicts can be wrong (one story opening
  that starts with a line of speech was marked as not doing so).
