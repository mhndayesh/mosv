"""Fast answer sampling with vLLM for the RL pilot (--sampler vllm).

Each card's worker owns one vLLM engine and one PyTorch model. vLLM only writes answers; PyTorch still
computes gradients. Before every sampling call the current steering vector is copied into the engine, so
answers come from the policy being trained. With this sampler, dev and test scoring also run on vLLM, so
base, steered and random scores always come from one engine.
vLLM is imported only when an engine is built, so the Hugging Face mode and the tests never need it.
"""

from __future__ import annotations

import os
import sys
from functools import partial
from pathlib import Path

from rl.policy import trim_completion

PLUGIN_DIR = Path(__file__).parent / "vllm_plugin"
GPU_MEMORY_UTILIZATION = 0.25  # 0.30 reached 14.17 GB next to PyTorch and the desktop on the main card.
MAX_NUM_SEQS = 256
MAX_NUM_BATCHED_TOKENS = 8192
MAX_MODEL_LEN = 1024  # Prompt plus answer must fit in this.
ENV_DEFAULTS = {
    "ROCPROFILER_REGISTER_ENABLED": "0",
    "VLLM_WSL2_ENABLE_PIN_MEMORY": "1",
    "VLLM_NO_USAGE_STATS": "1",
    "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",  # Lets apply_model send a function to the engine process.
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",  # PyTorch already owns the card in this process.
}


def add_plugin_path() -> None:
    """Make `qwen_steering` importable here and in the processes vLLM spawns."""
    plugin = str(PLUGIN_DIR)
    if plugin not in sys.path:
        sys.path.insert(0, plugin)
    if plugin not in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [plugin, os.environ.get("PYTHONPATH", "")]))


def prepare_environment() -> None:
    """Set the variables vLLM needs, then make the steering plugin importable."""
    for key, value in ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)
    libraries = Path(sys.prefix) / "lib" / "python3.12" / "site-packages" / "_rocm_sdk_core" / "lib"
    if libraries.is_dir() and str(libraries) not in os.environ.get("LD_LIBRARY_PATH", ""):
        os.environ["LD_LIBRARY_PATH"] = f"{libraries}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    add_plugin_path()


class VllmSampler:
    """Wraps a vLLM engine: push the vector, then sample answers for token-ID prompts."""

    def __init__(self, llm, params, group: int, max_new: int, eos: set, greedy=None):
        self.llm, self.params, self.group, self.max_new, self.eos = llm, params, group, max_new, eos
        self.greedy = greedy

    @classmethod
    def start(cls, model_path: str, group: int, max_new: int, eos: set, kv_cache_dtype: str = "auto"):
        prepare_environment()
        from qwen_steering import ARCH
        from vllm import LLM, SamplingParams

        llm = LLM(model=model_path, dtype="float16", hf_overrides={"architectures": [ARCH]},
                  gpu_memory_utilization=GPU_MEMORY_UTILIZATION, max_num_seqs=MAX_NUM_SEQS,
                  max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS, max_model_len=MAX_MODEL_LEN,
                  kv_cache_dtype=kv_cache_dtype, disable_log_stats=True)
        # Plain sampling from the model's own distribution, as the PyTorch gradient pass assumes.
        params = SamplingParams(n=group, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
                                max_tokens=max_new, stop_token_ids=sorted(eos), detokenize=False)
        greedy = SamplingParams(temperature=0.0, repetition_penalty=1.0, max_tokens=max_new,
                                stop_token_ids=sorted(eos), detokenize=False)
        return cls(llm, params, group, max_new, eos, greedy)

    def push(self, vectors) -> None:
        """Copy float32 vectors (layers x width) into the engine. None turns steering off."""
        add_plugin_path()
        from qwen_steering import set_steering

        array = None if vectors is None else vectors.detach().float().cpu().numpy()
        self.llm.apply_model(partial(set_steering, vectors=array))
        self.llm.reset_prefix_cache()  # Cached prompt work was computed with the old vector.

    def _generate(self, prompts: list[list[int]], params):
        too_long = [len(p) for p in prompts if len(p) + self.max_new > MAX_MODEL_LEN]
        if too_long:
            raise ValueError(f"Prompt of {max(too_long)} tokens + {self.max_new} answer tokens exceeds "
                             f"the vLLM limit of {MAX_MODEL_LEN}")
        return self.llm.generate([{"prompt_token_ids": p} for p in prompts], params, use_tqdm=False)

    def sample(self, prompts: list[list[int]]) -> list[list[list[int]]]:
        """Answers for each prompt: `group` token lists, each cut after its first end token."""
        return [[trim_completion(list(o.token_ids), self.eos) for o in sorted(r.outputs, key=lambda o: o.index)]
                for r in self._generate(prompts, self.params)]

    def best_answers(self, prompts: list[list[int]], vectors) -> list[list[int]]:
        """One repeatable answer per prompt for scoring. vectors=None scores the unsteered model."""
        self.push(vectors)
        return [trim_completion(list(r.outputs[0].token_ids), self.eos)
                for r in self._generate(prompts, self.greedy)]
