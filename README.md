# In-model RL steering experts for a normal LLM

**Purpose:** Replace trainable LoRA matrices with a tiny resident bank of
activation-steering vectors. Freeze the pretrained language model, train
the vectors through a **group-relative RL policy-gradient loss**, and train
an input router to pick a vector.

This is a **research prototype**, not a measured LLM performance claim.

## The exact internal math

At each selected decoder block `l` and expert `e`, for *every token* `i`:

`h'[l,i] = h[l,i] + s[e,l]`

`s[e,l]` is a learned vector of length `hidden_size`, permanently registered
inside the model. The selected expert is determined **once per prompt**;
inactive experts are never read or applied. The pretrained weights `theta`
are frozen. All trainable parameters are only the steering bank and router.

For `G` completions of one prompt, assign rewards `R_j`, and group-centered
advantages `A_j = R_j - mean(R)`. The prototype minimizes

`L_policy = -(1 / (G*T)) * sum_j [ A_j * sum_t log P_theta,s(a_jt | prompt, a_j<t) ]`

where `T` is a fixed maximum completion length, to avoid per-response-length
normalization bias. Rollouts are generated at the same policy as the gradient
update, so there are no stale-policy importance weights.

This is **group-relative REINFORCE (GRPO-inspired)**, not the complete GRPO
algorithm: no PPO ratio clipping and no reference-model KL penalty. For the
first experiment, the router is learned with supervised task labels; only
the **vectors** learn from rewards. A future version could optimize the
discrete routing action with its own policy-gradient term.

## What's included

| File | Purpose |
|---|---|
| `steering.py` | Two independent inside-model vector experts, layer hooks, router |
| `train_rl.py` | Freeze Qwen3; sample answer groups; reward, backprop, save checkpoint |
| `rl_math.py` | Group advantages and demo arithmetic reward |
| `infer.py` | Compare unsteered vs chosen expert on same prompt |
| `evaluate.py` | Held-out base/steered accuracy comparison |
| `test_steering.py` | Internal tests for active/inactive experts and gradients |
| `test_rl_math.py` | Reward and policy-gradient mathematical sanity tests |

## Requirements

- Python 3.10+, PyTorch, Transformers, Safetensors.
- Qwen3-0.6B weights (downloaded by Transformers or present in a local model path).
- **GPU strongly recommended.** Even though only the vectors train, gradient
  calculation must traverse the downstream frozen transformer layers. You do
  **not** save 90% of the backward compute merely by freezing their weights.
- The prototype uses a simple no-KV-cache sampler during training for
  correctness, which is slower than production inference.

## Install and train

```bash
pip install -r requirements.txt
python -m unittest test_steering.py test_rl_math.py

# Start with a small smoke run on a GPU:
python train_rl.py --model Qwen/Qwen3-0.6B --steps 5 --group-size 4 --max-new-tokens 12

# Larger experiment, still small enough for research:
python train_rl.py --model Qwen/Qwen3-0.6B --steps 150 --group-size 4 --layers auto

# Same model weights, steering disabled vs enabled:
python infer.py --checkpoint checkpoints/steering_bank.pt --task 0 --a 32 --b 17
python infer.py --checkpoint checkpoints/steering_bank.pt --task 1 --a 32 --b 17

# Evaluate on held-out arithmetic tasks:
python evaluate.py --checkpoint checkpoints/steering_bank.pt --samples 40
```

`--layers auto` attaches steering vectors after approximately 25%, 50%, and
75% of the transformer's depth. Specify `--layers 7,14,21` to target exact
indices. The default tasks are **addition** and **subtraction** because their
rewards are exact and easy to verify, not because arithmetic proves behavioral
skill learning.

## From arithmetic to behavioral skills

Replace `reward_integer_answer()` with an outcome-based checker that returns
a number for each whole assistant rollout. For example:

- `test_before_done`: 1 only when a real test command ran, exited successfully,
  and its recorded result matches the completion claim; 0 or penalty otherwise.
- `stage_before_live`: 1 only when independent deployment logs establish
  staging verification occurred before live publication.
- `research_before_implementation`: reward verifiable evidence collection
  before a later implementation action, with task-dependent allowances.

Do NOT reward mere strings like "I tested the code". To learn actual agent
behavior, a training rollout must include **tool execution, environmental
state, and verified outcomes**, and the environment must prevent bypasses.
This demo's text-only arithmetic evaluator does **not** train those skills.

## Safety checks and limitations

1. The model is frozen *before* the steering bank and router are attached.
2. An inactive expert has `grad=None` because experts are distinct Parameters;
   its AdamW optimizer state cannot update it when another expert trains.
3. Steering disabled is the original frozen LLM's computation.
4. Do not change selected experts halfway through a cached generation: prior
   KV cache entries were computed under the previous expert. Choose once per
   request or explicitly recompute the cache.
5. The router here is trained with known task labels; an RL-trained router
   needs separate exploration and selection-reward design.
6. Low parameter count does not imply low total RL GPU compute or guaranteed
   transfer to behavioral skills. Compare to same-budget LoRA on held-out tasks.
7. Save only steering parameters in the checkpoint; model weights are not
   duplicated. The `model_id` in the checkpoint must remain available.

## References

- [RL Post-Training in Diffusion Transformers Is Activation Steering](https://arxiv.org/abs/2611.26627)
- [RISER: Orchestrating Latent Reasoning Skills for Adaptive Activation Steering](https://arxiv.org/abs/2601.09269)
- [Hugging Face GRPO trainer](https://huggingface.co/docs/trl/grpo_trainer)
