# Outline: Behavior vectors for a frozen LLM

Format: an honest technical write-up for GitHub / Hugging Face, by someone trying things out. Not a journal paper.
Plain language, real numbers, and failures shown as well as wins.

## Working title

**Teaching a small model how to work, not what to know: RL-trained steering vectors with a router**

## The one-line claim

A frozen Qwen3-4B keeps all its knowledge. Small RL-trained vectors, one per task and chosen by a router, change how
it does the work (structure, steps, style). A vector adds no knowledge. It changes how the model uses what it knows.

## Sections

1. **The idea** (about half a page)
   - Small models often know enough but don't use it well for a given task.
   - Inspiration: Meral et al. 2026 (arXiv 2611.26627) showed that RL post-training in image and video models acts
     like a steering vector. This write-up asks whether the same holds for a language model, with one vector per
     task.
   - How it differs from mixture of experts: MoE uses part of the model. This uses all of it and changes only the
     behavior.
   - Why not LoRA: a vector is about 30 KB, the model never changes, and vectors can be swapped per prompt.
     (Stacking vectors and turning their strength up or down are the owner's next steps, not yet tested.)

2. **How it works** (one diagram plus about a page)
   - The vector is added at layers 9, 18 and 27 of 36: `h' = h + s`.
   - Router: one linear layer reading layer 9. It picks 1 of 8 topics, or "none".
   - Training: 4 answers per prompt, scored by Claude Haiku against a 5-point rubric for each topic, with a
     group-relative policy gradient. Only that topic's vector learns. A KL check on a probe set rejects steps that
     move the model too far.
   - Size cap: each vector is kept to at most 50% of the typical activation size.

3. **Setup**
   - Model, hardware (two RX 9070 XT, vLLM writes answers on one card while PyTorch trains on the other).
   - The 8 topics and their data sizes (68,944 training prompts in total).
   - The judge: why Haiku, and the rubric check (97.5% / 100%).

4. **Results**
   - Router: 6,823 / 6,835 correct, 0 wrong vectors, 6 / 600 false steers.
   - Training-time test: steered against normal against a random vector (debugging 0.889 / 0.257 / 0.248, etc.).
   - **Main result:** the full-length blind test table (80 prompts, 2.89 to 3.54).
   - Example answers side by side: debugging and story (the clearest wins).

5. **What didn't work** (an honest section)
   - The regex checklist judge was a coin flip (49.6%) and got gamed ("The rain fell").
   - 0.6B was too small.
   - Engineering: weak data (2,739 prompts, about 920 of 2,181 arena prompts share one template). It won short
     answers but lost full projects.
   - How-to: never learned (dev 0.483).
   - Math: a tie, because the model is already strong there.

6. **Limits** (short list)
   - 10 prompts per topic, an LLM judge only, no human ratings.
   - Training stopped at about 50%.
   - The knowledge claim is not yet measured.
   - No LoRA comparison.

7. **Next steps**
   - Knowledge test with and without vectors.
   - Vector strength sweep, and stacking two vectors.
   - Better engineering data, and a MoE version with a router.

8. **How to reproduce** (link to the README pipeline)

## Open decisions for the owner

- [ ] Title
- [ ] Run the knowledge test before publishing, or list it as a limit?
- [ ] Where it lives: `paper/` in this repo, a Hugging Face model card, or both?
- [ ] How many example answers to show
