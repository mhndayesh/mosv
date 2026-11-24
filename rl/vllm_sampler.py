"""Fast answer sampling with vLLM for the RL pilot (--sampler vllm), and the turn-taking for large models.

Each card's worker owns one vLLM engine and one PyTorch model. vLLM only writes answers; PyTorch still
computes gradients. Before every sampling call the current steering vector is copied into the engine, so
answers come from the policy being trained. With this sampler, dev and test scoring also run on vLLM, so
base, steered and random scores always come from one engine.
vLLM is imported only when an engine is built, so the Hugging Face mode and the tests never need it.

Two 4B copies do not fit on one card, so CardTurns gives the card to one side at a time:
- vLLM sleeps (level 1: weights copied to RAM, KV cache dropped), or is restarted with --vllm-swap restart.
- PyTorch weights are frozen, so they never copy back. They read from the memory-mapped model files,
  which costs no private RAM, and leaving the card only repoints them at those files.
"""

from __future__ import annotations

import gc
import os
import sys
import time
from functools import partial
from pathlib import Path

from model_profiles import DEFAULT_PROFILE, PROFILES, ModelProfile
from rl.policy import trim_completion

PLUGIN_DIR = Path(__file__).parent / "vllm_plugin"
MAX_MODEL_LEN = 1024  # Prompt plus answer must fit in this.
SLEEP_LEVEL = 1  # Level 2 would reread 8 GB through the slow /mnt/c path on every wake.
SWAP_MODES = ("sleep", "restart", "separate")  # separate: vLLM on the main card, PyTorch on the second.
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


def engine_settings(profile: ModelProfile, card: str, total_bytes: int | None, swap: str) -> dict:
    """The LLM() arguments that depend on the model. Layers travel in the config to every child process."""
    from qwen_steering import ARCH, LAYERS_KEY

    settings = {"hf_overrides": {"architectures": [ARCH], LAYERS_KEY: list(profile.steering_layers)},
                "gpu_memory_utilization": profile.vllm_share(card, total_bytes),
                "max_num_seqs": profile.vllm_max_seqs,
                "max_num_batched_tokens": profile.vllm_max_batched_tokens}
    if card in profile.vllm_kv_bytes:  # vLLM still checks the share at startup, so both are set.
        settings["kv_cache_memory_bytes"] = profile.vllm_kv_bytes[card]
    if profile.time_share and swap == "sleep":
        settings["enable_sleep_mode"] = True
    return settings


class VllmSampler:
    """Wraps a vLLM engine: push the vector, then sample answers for token-ID prompts."""

    def __init__(self, llm, params, group: int, max_new: int, eos: set, greedy=None, build=None, swap: str = "sleep"):
        self.llm, self.params, self.group, self.max_new, self.eos = llm, params, group, max_new, eos
        self.greedy, self.build, self.swap = greedy, build, swap
        self.awake = True

    @classmethod
    def start(cls, model_path: str, group: int, max_new: int, eos: set, kv_cache_dtype: str = "auto",
              profile: ModelProfile = PROFILES[DEFAULT_PROFILE], card: str = "main", total_bytes: int | None = None,
              swap: str = "sleep", around_build=None):
        if swap not in SWAP_MODES:
            raise ValueError(f"Unknown vLLM swap mode: {swap}")
        prepare_environment()
        from vllm import LLM, SamplingParams

        settings = engine_settings(profile, card, total_bytes, swap)

        def build():
            make = partial(LLM, model=model_path, dtype="float16", max_model_len=MAX_MODEL_LEN,
                           kv_cache_dtype=kv_cache_dtype, disable_log_stats=True, **settings)
            return around_build(make) if around_build else make()

        # Plain sampling from the model's own distribution, as the PyTorch gradient pass assumes.
        params = SamplingParams(n=group, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
                                max_tokens=max_new, stop_token_ids=sorted(eos), detokenize=False)
        greedy = SamplingParams(temperature=0.0, repetition_penalty=1.0, max_tokens=max_new,
                                stop_token_ids=sorted(eos), detokenize=False)
        return cls(build(), params, group, max_new, eos, greedy, build, swap)

    def rest(self) -> None:
        """Free this engine's VRAM. Steering and the prefix cache are set again by the next push."""
        if self.swap == "restart":
            self.llm.llm_engine.engine_core.shutdown()
            self.llm = None
            gc.collect()
        else:
            self.llm.sleep(level=SLEEP_LEVEL)
        self.awake = False

    def wake(self) -> None:
        if self.swap == "restart":
            self.llm = self.build()
        else:
            self.llm.wake_up()
        self.awake = True

    def _require_awake(self) -> None:
        if not self.awake:
            raise RuntimeError("vLLM is asleep; call CardTurns.for_vllm() first")

    def push(self, vectors) -> None:
        """Copy float32 vectors (layers x width) into the engine. None turns steering off."""
        self._require_awake()
        add_plugin_path()
        from qwen_steering import set_steering

        if vectors is not None and not vectors.detach().any():
            vectors = None  # A kept zero vector is the unsteered model, so it takes the exact no-steering path.
        array = None if vectors is None else vectors.detach().float().cpu().numpy()
        self.llm.apply_model(partial(set_steering, vectors=array))
        self.llm.reset_prefix_cache()  # Cached prompt work was computed with the old vector.

    def _generate(self, prompts: list[list[int]], params):
        self._require_awake()
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


def open_weight_views(model_path: str) -> dict:
    """Memory-mapped tensors from the model files. Measured on 4B: 8.04 GB mapped, 0.02 GB private RAM."""
    from safetensors import safe_open

    views = {}
    for file in sorted(Path(model_path).glob("*.safetensors")):
        with safe_open(str(file), "pt") as handle:
            views.update({key: handle.get_tensor(key) for key in handle.keys()})
    return views


class WeightShuttle:
    """Moves frozen PyTorch weights onto the card from the mapped files. Leaving the card copies nothing."""

    def __init__(self, model, views: dict, dtype, device):
        self.dtype, self.device, self.pairs = dtype, device, []
        for name, weight in model.named_parameters():
            if name not in views or views[name].shape != weight.shape:
                raise ValueError(f"Model file has no matching tensor for {name}")
            self.pairs.append((weight, views[name]))
        self.bytes = sum(view.numel() * weight.element_size() for weight, view in self.pairs)
        for module in model.modules():  # Small buffers such as RoPE stay on the card the whole run.
            for name, buffer in module._buffers.items():
                if buffer is not None:
                    module._buffers[name] = buffer.to(device)
        self.on_gpu = False

    def to_gpu(self) -> None:
        for weight, view in self.pairs:
            weight.data = view.to(self.device).to(self.dtype)
        self.on_gpu = True

    def to_cpu(self) -> None:
        for weight, view in self.pairs:
            weight.data = view
        self.on_gpu = False


class NoTurns:
    """Both engines fit on the card, or only one exists: nothing to swap."""

    swap_seconds, swaps = 0.0, 0

    def for_vllm(self) -> None:
        pass

    def for_torch(self) -> None:
        pass


class CardTurns:
    """Gives the card to vLLM or PyTorch. Always frees one side before the other side loads."""

    def __init__(self, torch, sampler: VllmSampler, weights: WeightShuttle):
        if weights.on_gpu:
            raise RuntimeError("PyTorch weights must be off the card while vLLM holds it")
        self.torch, self.sampler, self.weights = torch, sampler, weights
        self.owner, self.swap_seconds, self.swaps = "vllm", 0.0, 0

    def _count(self, started: float) -> None:
        self.swap_seconds += time.time() - started
        self.swaps += 1

    def for_vllm(self) -> None:
        if self.owner == "vllm":
            return
        started = time.time()
        self.weights.to_cpu()
        gc.collect()
        self.torch.cuda.synchronize()
        self.torch.cuda.empty_cache()
        left = self.torch.cuda.memory_allocated()
        if left > self.weights.bytes / 2:
            raise RuntimeError(f"PyTorch still holds {left / 1e9:.2f} GB; refusing to wake vLLM")
        self.sampler.wake()
        self.owner = "vllm"
        self._count(started)

    def for_torch(self) -> None:
        if self.owner == "torch":
            return
        started = time.time()
        self.sampler.rest()
        self.owner = "moving"  # A failed upload must not look like vLLM still holds the card.
        self.weights.to_gpu()
        self.owner = "torch"
        self._count(started)
