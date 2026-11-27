I taught a frozen Gemma 4 12B new behaviors with tiny "steering experts" (36 MB). RL failed, plain supervised training worked. Here's what I learned.

Hey all,

I've been working on a side project called MoSV (Mixture of Steering Vectors) and wanted to share where it's at, including the part that failed.

**Where the idea came from**

I read "RL Post-Training in Diffusion Transformers Is Activation Steering" (Meral et al., arXiv:2611.26627). They show that for image/video diffusion models, most of what RL fine-tuning gains can be recovered with a small set of steering vectors added inside the frozen model. That made me wonder if the same works for text LLMs: maybe "post-training" is mostly nudging the model in a direction, and you don't need to touch the weights at all.

**What I built**

The base model stays frozen. I add tiny experts at three layers, one per topic. A router picks an expert per token, or "none", in which case the model answers exactly like normal. The whole thing is baked into one GGUF in a layout stock llama.cpp already supports, so it runs in plain llama.cpp and LM Studio with no custom code.

The idea I'm testing: this teaches behavior, not knowledge. The model already knows things. What it often gets wrong is how it answers: writing a lesson plan for a teacher when you asked it to teach a kid, stating guesses as facts when debugging, opening every story with the weather.

**It worked on small models first**

On Qwen3-4B and a 1-bit Bonsai-4B I trained the experts with RL (sample several answers, a judge model scores them, reinforce the better ones). Both improved: Bonsai scored higher on 66% of 300 locked test prompts, with answers 4x shorter and no more looping.

[image: 3_results_table.png]

**Then it broke on a strong model**

On Gemma 4 12B, the same RL setup did nothing. Two full runs, no gain. So I stopped and measured why:

- There was almost no room. On my test, Gemma scored 3.38, Gemma told the rules directly scored 3.46, and the judge model itself scored 3.71.
- The reward was noise. I had the judge rank the same 4 answers twice, and it picked the same winner 25.5% of the time. That's chance.
- A strong model gives the same quality every time, so RL has nothing better to reinforce.

[image: 2_rl_reward_noise.png]

**The fix (way simpler, honestly)**

1. Gate first. Before training, compare the model answering plainly with the model told the rules. Only train topics where being told clearly helps. That's also the ceiling.
2. Yes/no checks instead of a 1-5 score. "Does the answer speak to the child directly?" "Does it separate what the bug report says from what's guessed?" Plus one correctness check that must not get worse. The judge gives the same answer twice 95-98% of the time.
3. Tell, check, train. Have the model answer with the rules, check each answer, keep only the ones that pass and beat the plain answer, then train the experts on those with the plain prompt only. That's just normal supervised training, like a LoRA. No RL.

Funny part: after all the RL work, what worked is basically "make good examples from the model itself, filter them, fine-tune the small parts."

**Results on Gemma 4 12B**

Locked test, 300 prompts never used in training, no instructions given to the model:

[image: 1_gemma_results.png]

- Teaching: 0.58 to 0.97 of the behavior checks, better on 98% of prompts. Answers went from ~990 to ~325 tokens because it stopped writing lesson plans and started actually teaching the kid.
- Debugging: 0.17 to 0.44, better on 81% of prompts.
- Story: 0.73 to 0.81, better on 62% of prompts.

The dashed line is the ceiling (the model told the rules). Teaching and story basically reach it without being told anything.

**Why I think this is a genuinely good idea**

- Tiny: 36 MB on top of a frozen 7 GB model.
- Behavior, not knowledge: you're not overwriting what the model knows.
- Per topic: each topic gets its own behavior and everything else is left alone. A normal LoRA changes the model for every prompt.
- One file that runs anywhere llama.cpp runs.
- It could replace a lot of long system prompts: tone, format and steps could live in an expert instead of 2,000 tokens of instructions.

**Caveats**

- One judge (Claude Haiku), and 43 test prompts per topic.
- Teaching got slightly less accurate (0.95 to 0.91 on the correctness check).
- Engineering, which I did NOT train, got a bit worse (0.77 to 0.71). Probably some code tokens get routed to the debugging expert. That's the next thing to fix.
- Done on a home setup with two RX 9070 XTs on Windows, which was its own adventure.

Code, paper and all the numbers: github.com/mhndayesh/mosv
Model: huggingface.co/mhndayesh/MoSV-Gemma-4-12B-v6.0.1

Happy to answer questions, and I'd love to hear if anyone has tried something similar on text models.
