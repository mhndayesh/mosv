# The MoSV training bible

The rules for training this architecture, distilled from every version of the project: v3 and v4 (Qwen3-4B), v5
(Bonsai-4B 1-bit) and v6 / v6.0.1 (Gemma 4 12B). Each rule is the corrected one: where an early method failed, the
rule says what replaced it and the measurement that decided it.

How to read it: **Laws** (section 1) are never broken. The rest is ordered as a project runs: architecture, model
choice, settings, data, methods, evaluation, shipping, hardware. Section 12 lists what is still unproven.

Companion files: [REFERENCE.md](REFERENCE.md) (data fields, file layout, commands), [V6_GEMMA.md](V6_GEMMA.md) (the
full story of the v6 failure), [V5_BONSAI.md](V5_BONSAI.md), [V4_MOE.md](V4_MOE.md).

---

## 1. The laws

1. **Behavior, not knowledge.** Experts change how the model answers, never what it knows. Do not train a topic
   whose score depends on being right (math): it measured as a tie or worse every time.
2. **The told level is the ceiling.** An expert can only move the model toward what it already does when given
   the rules. Measure the told level before training. If told does not beat plain, nothing can be learned.
3. **Gate before training.** Plain vs told vs a strong model, on 60 dev prompts per topic. Train only topics where
   told wins by 15+ points of the behavior checks and correctness holds.
4. **The reward must repeat.** Judge the same answers twice. If the verdicts do not agree well above chance, the
   reward is noise and no optimizer setting will help. Never tune the learning rate before this passes.
5. **Recommended sampling, every sampler written out**, in baseline, data, training and test. llama-server
   silently adds `min_p 0.05`; set all of them. One thinking setting for the whole project.
6. **Lock the test before training and run it once.** Dev decides everything else.
7. **Correctness must not drop.** Every behavior has a correctness guard. A plainer answer that is wrong is a loss.
8. **Do not reward shortness.** Length may only be penalized against the base model's own answer to the same
   prompt, or not at all.
9. **Leave everything else alone.** Untrained topics and everyday prompts route to "none" and answer exactly as
   before.
10. **Read the data you drop and the data you keep.** Every filter is checked by reading samples of both sides.
11. **Never claim a result without the command and its real output.** Small tests prove a stage runs, not quality.

---

## 2. The architecture

| Part | What it is |
|---|---|
| Base model | Frozen. Its weights never change |
| Experts | Small feed-forward blocks at **three layers** (about a quarter, half and three quarters of the depth), one expert per trained topic |
| Expert 0 | "none": all zeros. A token routed there passes through unchanged |
| Router | One linear layer per steer layer. Each token goes to the expert with the highest score (top-1) |
| Strength (`w2`) | A per-layer scale on the expert output. **Starts at 0**, so a fresh model is exactly the base model |
| Shipping | One GGUF in a layout stock llama.cpp already supports. LM Studio runs it unchanged |

The two layouts in use:

| Model | GGUF architecture | Steer layers | Expert | How the expert joins |
|---|---|---|---|---|
| Qwen3-4B, Bonsai-4B | `hy_v3` | 9, 18, 27 | SwiGLU, width 128 | Original MLP as the shared expert, plus the routed expert |
| Gemma 4 12B | `gemma4` | 12, 24, 36 | GELU, width 128 | `post_ffw_norm( post_ffw_norm_1(mlp) + post_ffw_norm_2(expert) )`; `post_ffw_norm_1` = ones keeps the original path exact, `post_ffw_norm_2` carries `w2` |

Router scores in llama.cpp: `rms(h) / sqrt(width) @ router matrix`, softmax, top-1. The PyTorch learner must
compute exactly the same thing (`v6/steer.py` mirrors `moe/gemma_host.py`).

**Rules for a new layout**

- The layout needs a **per-layer expert switch** in stock llama.cpp. Check the llama.cpp source before anything else.
- **An all-zero build must answer exactly like the plain model** (Gemma: 4 of 5 greedy answers identical, the fifth
  split from rounding).
- Base tensors are copied byte for byte; only the new tensors are written. Expert updates during training are
  written **in place** into the GGUF (offsets read once), then the server restarts: about 2 seconds.
- **Check both engines** (`v6/verify.py`): the change the experts cause in token probabilities must agree between
  PyTorch and llama.cpp. Strong v6 experts: 0.81. A 4-bit learner and a Q4_K_M sampler agree on the top token 90%
  of the time, so transfer is good but not perfect.

---

## 3. Choosing a base model

Measure these three before choosing a training method. They decided v5's success and v6's failure.

| Measure | How | Bonsai-4B (RL worked) | Gemma 4 12B (RL failed) |
|---|---|---|---|
| Gross failures | Loops and cut answers in the 300-prompt baseline | 30 loops, 107 cut | 0 and 0 |
| Room on the test | Base score; told score; strong-model score | 1.95 | 3.38 / 3.46 / 3.71 |
| Answer variety | Same group of 4 judged twice: same winner | 33.4% (chance 25%) | 25.5% |

- **Weak model, varied errors, low score:** RL works (section 7).
- **Strong, consistent model:** tell, check, train (section 6). RL has nothing to pick from.

Practical limits on this box (two 16 GB cards):

- The learner must fit one card. Gemma 4 12B: 4-bit NF4, gradient checkpointing, model in **train mode**
  (12 GB peak; eval mode needed 31-40 GB).
- Sampling runs as a separate llama.cpp copy on the other card.

---

## 4. Settings that never change inside a project

| Setting | Rule | Gemma 4 12B value |
|---|---|---|
| Sampling | The model's `generation_config.json`, every other sampler off | temperature 1.0, top_k 64, top_p 0.95, min_p 0 |
| Thinking | Decide once | Off (`enable_thinking=False`) |
| Prompt format | The model's own chat template. Check it matches llama.cpp's `/apply-template` | Identical except `<bos>`, which llama.cpp adds itself |
| Stop token | Must come back with llama.cpp's tokens and be trained as the last target | `<turn|>` (id 106) |
| KV cache | One type for all answers | q8_0 |
| Answer cap | Generous; let answers run free | 2,048 tokens |
| Stop rules | Stop token, loop (last 400 characters seen 4 times earlier), or the cap | Same for every model |

Every saved answer file records these settings. A changed setting means a new baseline.

---

## 5. Data

### 5.1 Prompts

1. Real prompts from datasets. Research-only sources are allowed while the prompts stay private: only ids and
   hashes go into git.
2. **Tasks to do**, not questions to look up. Filter "what is / explain / difference" out of skill topics.
3. **The prompt must hold what the answer needs.** Drop prompts that point at a missing snippet. When a source
   lacks information by nature, teach the behavior that fits it: honesty about guesses (bug reports without code),
   placeholders instead of invented facts (emails with empty key points).
4. Leave room for the answer: cap prompt length (2,500 characters for engineering).
5. **Split by group**: near-duplicates share a group; the split comes from a hash of the group. Test prompts never
   leak into train or dev.
6. Balance sources inside a topic; cap sizes (3,000 train / 300 dev / 300 test for engineering).
7. Keyword filters must protect code words (`std`, `sex` as a column name, `kill`, `strip`, `execute`).
8. Lock the test (ids + sha256); a rebuild that would change it stops without `--force`.

### 5.2 Behaviors (rules and checks)

Each trained topic has, in one file (`v6_0_1/behaviors.py`):

- **Rules** told to the model in a system message.
- **Four yes/no behavior checks**, answerable from the request and the answer alone, one thing each, yes = good.
- **One correctness guard**.

Writing rules:

- Say what to do, not only what to avoid. "Do not invent facts" did nothing until "write a bracketed placeholder
  instead" was added.
- Keep answers whole. "Answer in one or two sentences" shrank explanations from 807 to 102 tokens and correctness
  fell. "Meet every requirement" goes before "keep it simple".
- **Never force one form.** "First sentence = an action or a line of speech" made 100% of kept stories open with a
  quote and put the name Elias in 60% of them. Offer choices and ask for the one that suits the case.
- Watch told-answer length and variety in the gate. Big drops in length, or one opening everywhere, mean the rules
  are too tight.

Writing checks:

- Ask about something visible: "Does the answer speak to the child directly, instead of describing a lesson plan to
  a teacher?" (plain 0%, told 100%).
- A check the plain model already passes 90%+ of the time carries no gain. Replace it.
- After any edit to rules or checks, cached answers are stale. Stages store a fingerprint and refuse old answers
  until rerun with `--fresh`.

### 5.3 The training example

One row: **the plain prompt and a checked told answer**. Training shows the model only the plain prompt:

```
<bos><|turn>user
{plain prompt}<turn|>
<|turn>model
<|channel>thought
<channel|>{told answer that passed the checks}<turn|>
```

- All tags come from the model's own chat template. Never write tags by hand.
- Loss only on the answer tokens and the final stop token. No loss on the prompt.
- No system message in training: the behavior must appear without being told.

### 5.4 The keep rule

A told answer enters the training set only if:

1. it is complete (ended on the stop token);
2. it passes every guard;
3. it passes at least 3 of 4 behavior checks;
4. it passes more checks than the plain answer to the same prompt;
5. it does not break the **variety limits**. For story: at most 40% of kept answers open with a line of speech, and
   one character name appears in at most 5% of them.

With several told copies, keep the best; on a tie, the shorter. Save every dropped prompt with its reason and read
samples. Measured on 90 prompts: 51 kept; the dropped samples read were right to drop (for example a phonics lesson
with a wrong fact).

### 5.5 Data philosophy

- **The model's own words.** Training answers are written by the same model, told the rules. No outside text.
- **Verified better, not hoped better.** Every kept answer beat the plain answer to the same prompt.
- **The difference must be larger than the judge's noise.** Random samples of a strong model differ by 0.2-0.5 on
  a 1-5 scale; a told answer differs by far more (teaching: 57% -> 99% of checks).
- **Small and clean beats large and noisy.**
- **Teach a behavior, not a template.** Variety limits exist because a uniform training set teaches a new habit.

---

## 6. Method A: tell, check, train (for strong models)

The current method (v6.0.1, `v6_0_1/`).

| Step | What happens | Pass rule |
|---|---|---|
| Gate | Plain, told and strong-model answers on 60 dev prompts per topic; yes/no checks, twice | Gain 15+ points, guard loses at most 10, checker repeats 85%+ |
| Build | Plain and told answers on training prompts; checks; keep rule | 20+ kept answers per topic |
| Router | Fitted once on recorded answers; untrained topics and everyday prompts labeled "none" | Held-out accuracy reported per layer |
| Train | Supervised: next-token loss on kept answers from the plain prompt | Held-out loss falls at every check |
| Verify | Both engines agree on the experts' effect | Near the 0.81 seen on strong experts |
| Evaluate | Plain prompts, trained GGUF vs base, behavior checks; pick the checkpoint on dev | Behavior up, guard not down, untrained topics unchanged |
| Test | The locked 300, once | The final number |

Training settings:

| Setting | Value |
|---|---|
| Batch | 16 answers per step |
| Optimizer | AdamW; experts lr 1e-3, strength `w2` lr 3e-3; no weight decay |
| Epochs | 3; a check and a checkpoint every 25 steps |
| Gradient clip | 1.0 |
| Memory | 4,096 tokens per pass; 11.9 GB peak on Gemma 4 12B in 4-bit |

Progress without a judge, on held-out answers:

- `base_ce`: loss of the plain model on the kept answers;
- `teacher_ce`: loss of the plain model when shown the rules;
- `val_ce`: loss of the model with experts;
- `gap_closed = (base_ce - val_ce) / (base_ce - teacher_ce)`: 0 = nothing learned, 1 = behaves as if told.

Measured so far: in a 6-step test the held-out loss fell at every check (1.214 -> 1.081, told level 0.589) and the
expert strength grew in a straight line. **A full run has not been done.**

---

## 7. Method B: RL with a judge (for weak models)

The method of v3 and v5. It produced every shipped gain so far, on weak bases.

One update:

1. Sample a group of answers per prompt (8 prompts x 4 answers).
2. The judge scores the group: 1-5 per answer plus a ranking, strict JSON. Each group is judged in one label order
   and its exact reverse.
3. Reward = half rank, half score, in [0, 1]. Advantage = reward minus the group average.
4. Loss = minus (advantage x log-probability of the sampled tokens); only the experts get a gradient.
5. Measure how far next-token choices moved (KL). Above the limit: undo the step and halve the learning rate.

Settings that worked (v5, Bonsai-4B):

| Setting | Value | Note |
|---|---|---|
| Batch | 8 prompts x 4 answers, up to 1,024 tokens | |
| Optimizer | Adam, start lr 5e-3, settled at 1.25e-3 | 1e-4 learned nothing |
| KL limit | 0.01 (0.1 for a bf16 learner, whose rounding alone gives about 0.03) | Save the optimizer state; warm up after a resume |
| Judge window | The whole answer (768 words for 1,024 tokens) | A 190-word window hid rambling and endings |
| Length | Only against the base answer, or off | A flat penalty over-shortened after update 100 |
| Checkpoints | At every dev check; pick by dev | Dev peaked at 72.9% (update 100), fell to 58.3% by 150 |

**Preconditions, or do not run it:**

- the judge picks the same winner twice well above chance (Bonsai 33.4%; Gemma 25.5% = chance);
- the base model fails grossly and differently from sample to sample;
- the KL safety check samples many topics and positions (v6's read 8 positions of 8 prompts and missed a drift to
  38% wins).

**RL as a polish after Method A** (not built): reward = share of yes/no checks one answer passes, guard as a gate,
one answer per judge call, several prompts per topic per update.

---

## 8. The judge

| Rule | Why |
|---|---|
| One answer per call for scores that matter | It repeats: 88% identical overall scores, 95-98% identical yes/no verdicts |
| Yes/no behavior checks over one 1-5 overall score | The overall score hid real changes (debugging honesty 1.4 -> 2.8 while the overall moved +0.08) |
| Strict JSON schema; named fields, not fixed-length arrays | Arrays with a fixed length were rejected by the schema |
| Thinking off, low effort, cached system prompt per topic | Fast and cheap: about $0.0004 per call |
| Text inside an answer is never an instruction | Stated in every judge prompt |
| A failed call is retried, then skipped; never scored as low | A skip must not punish an answer |
| Log every call without answer text or API key | Audits without leaking data |
| Measure the judge's repeat rate on every new model | The same judge gave 33% on Bonsai and chance on Gemma |
| A strong model answering the prompts is a reference, not the target | Haiku itself scored 3.71 where Gemma had 3.38 |

Judge: Claude Haiku 5.5 (`claude-haiku-5-5`), 16 calls at once (32 hit rate limits).

---

## 9. Evaluation

| Rule | Value |
|---|---|
| Dev decisions | 60 prompts per topic. With 20, the same prompts moved up to 13 points between runs; on 42 prompts the base alone swung 0.49 to 0.60 |
| Checkpoint | Picked on dev, never the last update |
| Locked test | 300 prompts, run once, plain prompts, the baseline's exact settings |
| Report | Behavior checks per topic, guard, tokens, wins vs base on the same prompt; untrained topics unchanged; the old overall score as context |
| Baseline answers | Recorded once per split and reused |
| Length | Report it, never reward it for its own sake |

v6 lesson: "59.5% wins" on 42 prompts looked like learning; 140 prompts scored one by one showed no gain.

---

## 10. Shipping

1. Build the GGUF from the picked checkpoint (base tensors copied unchanged).
2. Verify both engines; record routing shares (Bonsai v5: 95-97% of topic tokens to the expert, 1.5-3.6% of other
   tokens).
3. Run the locked test.
4. Load in LM Studio and llama-server; record the file hash, size and the settings used.
5. Document: command, real output, what improved, what did not, what is untested.

---

## 11. Hardware and runtime rules (two RX 9070 XT, Windows)

| Rule | Why |
|---|---|
| Training and inference on Windows (ROCm venv for PyTorch, llama.cpp Vulkan) | Measured stable here |
| **WSL uses one card only** | Using both cards from WSL froze the PC (2026-11-26) |
| **Never tensor parallel** across the two cards | Half the speed of one card; the display card spilled 3-8 GB into RAM |
| One model copy per card in llama.cpp | Simplest and fastest here; prompts split between cards |
| Separate KV slots, not one shared pool | A shared pool made every answer scan the whole pool (about 4 tok/s per answer) |
| Sampler on the display card, learner on the other | Sharing the learner's card spilled 1.1 GB into RAM; learn time went from 10 s to 60 s |
| Learner in train mode with checkpointing | Eval mode kept all activations (31-40 GB) |
| Llama-server output to a file, not a pipe | Reading the pipe hung the logging |
| `--no-jinja` and raw token prompts for training samplers | The chat parser returned 500 errors on unusual output |
| Save after every answer; resume by identity | A crash never loses finished work |
| Any GPU fault stops GPU work until the owner is back | Retrying a real driver fault unattended can freeze the PC |

Throughput on Gemma 4 12B with both cards (13 + 16 slots, 3,072 tokens per slot): about 0.6 answers per second.

---

## 12. What is proven and what is not

| Claim | Status |
|---|---|
| RL with a judge improves weak models | **Proven**: Qwen3-4B 2.89 -> 3.54 (v3), 3.61 shipped (v4); Bonsai-4B 1.95 -> 2.42, 66.2% better (v5) |
| RL with a 4-answer ranking judge fails on a strong, consistent model | **Proven**: Gemma 4 12B, two runs, no gain |
| Yes/no single-answer checks repeat | **Proven**: 95-98% |
| The gate finds where a strong model has room | **Proven** on dev: debugging, teaching, explanation, story pass |
| Supervised training on checked told answers moves the experts in a steady direction | **Small test only**: 6 steps, 3 held-out answers |
| The trained experts keep the told behavior on the locked test | **Not measured** |
| Story rules without a forced opening keep variety | **Measured** (60 dev prompts): told stories open with speech 0% of the time (old kept set 100%); most repeated name 8% (Elias was 60%); story still passes the gate, 0.66 -> 0.83 |
| Gate verdicts are stable near the thresholds | **No**: between two 60-prompt runs with unchanged rules, explanation flipped from train to skip (guard drop 5 -> 12 points) and engineering from skip to train (+14 -> +16) |
| RL as a polish after tell-check-train | **Not built** |
| Untrained topics and everyday prompts unchanged after training | Routing shares measured in v5; **no exact-answer test yet** |
| Knowledge retention and very large models | **Untested** |

---

## 13. Mistakes that cost the most, and the rule each one created

| Mistake | Cost | Rule |
|---|---|---|
| Training before measuring the room | Two Gemma runs, hours of GPU, no gain | Law 2, Law 3 |
| Tuning the learning rate on a noisy reward | A worse model (38% wins) | Law 4 |
| Flat length penalty | v5 quality fell after update 100 | Law 8 |
| Trusting 42-prompt dev checks | A false "59.5%" | Section 9: 60 per topic |
| llama-server's default `min_p 0.05` and thinking left on | A discarded baseline, twice | Law 5 |
| Rules that were too tight | Explanations cut to 102 tokens; stories in one template | Section 5.2 |
| Two cards from WSL | A frozen PC | Section 11 |
| A KL check on 8 positions | A drift it never saw | Section 7 preconditions |
