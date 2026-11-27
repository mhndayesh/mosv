---
license: apache-2.0
base_model: Qwen/Qwen3-4B-Instruct-2507
language:
- en
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- lm-studio
- steering-vectors
- mixture-of-experts
- qwen3
---

# MoSV: Mixture of Steering Vectors (Qwen3-4B)

> **Newest: [MoSV-Gemma-4-12B v6.0.1](https://huggingface.co/mhndayesh/MoSV-Gemma-4-12B-v6.0.1)** - Gemma 4 12B with debugging, teaching and story experts. On 300 locked test prompts it passes more behavior checks than base on 98% of teaching, 81% of debugging and 62% of story prompts. Write-up: [paper_v6_0_1.md](https://github.com/mhndayesh/mosv/blob/main/paper/paper_v6_0_1.md).

**GitHub:** [github.com/mhndayesh/mosv](https://github.com/mhndayesh/mosv): code, training pipeline, tests and
write-ups (v3 to v6.0.1). A snapshot of that repo is also in this repo's [`code/`](./code) folder.

Qwen3-4B-Instruct-2507 with task steering built in. Small steering vectors, trained by RL with Claude Haiku as the
judge, are stored inside the model as mixture-of-experts layers, and a gate picks one per token. One GGUF, no proxy
and no custom code: it runs in stock llama.cpp and in LM Studio. **Research project.**

## Run it

```bash
llama-server -m MoSV-Qwen3-4B-F16.gguf -ngl 999 -fa on --jinja
```

Or download the file and load it in LM Studio. You need a llama.cpp build that includes the `hy_v3`
architecture (LM Studio's ROCm 2.80.1+ and Vulkan 2.73+ engines were tested). It is 8.1 GB at F16 and fits one
16 GB GPU. Use a Q8_0 or Q4_0 KV cache; Q4_1 has no Flash Attention kernel.

## How it works

At layers 9, 18 and 27 the original Qwen MLP stays as an always-on shared expert. Next to it sit expert 0 (all
zeros = "no steering") and 8 small SwiGLU topic experts: engineering, story, teaching, debugging, math solving,
explanation, professional email, how-to guide. A linear gate picks one expert per token (sigmoid, top-1, weight 1).
The rest of the model is unchanged Qwen. The experts and gates were distilled from v3, which adds one RL-trained
vector per topic after the same layers, with a router picking the vector once per prompt.

## Results

86 held-out prompts, full-length answers, scored blind by Claude Haiku on a 5-criterion rubric per topic (1-5),
each pair judged twice with the order swapped.

| Topic | Normal Qwen | v3 | **MoSV (v4)** |
|---|---|---|---|
| Debugging | 1.70 | 4.12 | **4.09** |
| Professional email | 3.22 | 4.01 | **4.03** |
| Story | 2.55 | 3.82 | **3.66** |
| Teaching | 2.72 | 3.43 | **3.55** |
| Explanation | 2.90 | 3.45 | **3.49** |
| How-to guide | 3.00 | 2.86 | **3.21** |
| Engineering | 2.48 | 2.18 | **2.41** |
| Math solving | 4.53 | 4.47 | **4.43** |
| **All 80 topic prompts** | **2.89** | **3.54** | **3.61** |

**Limits:**
- Engineering and math do not improve, and each topic has only 10 test prompts.
- Routing works well between turns of a conversation: 91-100% of each answer's tokens get the right expert.
- Inside one prompt that mixes two tasks, the second task's expert flickers token by token.
- There is no per-request override. To answer unsteered, use plain Qwen.
- Factual accuracy with the vectors on was not tested.

## Files

| File | What |
|---|---|
| `MoSV-Qwen3-4B-F16.gguf` | v4, the model (arch `hy_v3`). sha256 `35d00caba382fb0e1a593faeb07429289957fc628a47245c7b51e9847e139ba8` |
| `v3/vectors/steer-<topic>.gguf` | v3: the 8 vectors as llama.cpp control vectors (`--control-vector`, use with plain Qwen3-4B-Instruct-2507 F16) |
| `v3/router_qwen4b.pt` | v3: the router head (reads the hidden state after layer 9); used by `v3/serve.py` in the GitHub repo |
| `code/` | Snapshot of the [GitHub repo](https://github.com/mhndayesh/mosv) (v3 in `code/v3/`, v4 in `code/moe/`, papers in `code/paper/`). GitHub is the live copy |

## Credit and license

Inspired by Meral et al., "RL Post-Training in Diffusion Transformers Is Activation Steering"
([arXiv:2611.26627](https://arxiv.org/abs/2611.26627)), adapted from diffusion models to a text LLM with one vector
per topic and a router. Base model: [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
(Apache-2.0).

**Code, MoSV experts, routers and steering vectors: MIT** (free to use, change and sell, keep the notice). The GGUF also contains the base model's weights, which stay under Qwen's Apache-2.0; that license also allows commercial use.
