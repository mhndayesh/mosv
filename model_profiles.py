"""One place for every fact about a base model: where it is, its layers and its memory budgets.

Pick one with --profile, or STEERING_PROFILE for the router jobs. The default stays Qwen3-0.6B, so
existing vectors, the router and the tests keep working. Width is read from the model's config.json.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).parent
HUB = "/mnt/c/Users/mhnda/.cache/huggingface/hub"
PROFILE_ENV = "STEERING_PROFILE"
DEFAULT_PROFILE = "qwen06b"


@dataclass(frozen=True)
class ModelProfile:
    name: str
    model_id: str
    path: str  # The WSL snapshot folder.
    num_layers: int
    steering_layers: tuple[int, ...]
    read_layer: int  # The router reads the prompt after this layer.
    router_dir: str  # Folder under data_review for prompts, features and the report.
    router_file: str  # File under checkpoints.
    time_share: bool  # True when vLLM and PyTorch must take turns on the card.
    vllm_fraction: float  # Share of the card for vLLM, used when vllm_memory_gb is empty.
    vllm_memory_gb: Mapping[str, float]  # vLLM's whole budget per card: weights, KV cache and workspace.
    vllm_max_seqs: int
    vllm_max_batched_tokens: int
    gradient_tokens_vllm: Mapping[str, int]  # Rows x longest row per gradient batch, with vLLM on the card.
    gradient_tokens_alone: Mapping[str, int]  # The same with Hugging Face sampling.
    generate_cells: int  # Answers x tokens per Hugging Face generation batch.
    head_chunk: int  # Answer tokens per pass through the 151,936-word output layer.
    gradient_checkpointing: bool
    other_vram_gb: Mapping[str, float]  # VRAM other programs hold on each card.
    hard_stop_gb: float | None  # Stop the run above this much VRAM on a card.
    torch_vram_gb: Mapping[str, float] = field(default_factory=dict)  # PyTorch's own limit per card.
    generate_cells_by_card: Mapping[str, int] = field(default_factory=dict)  # Overrides generate_cells.
    eval_batch: int = 32  # Prompts per scoring batch with Hugging Face generation.
    vllm_kv_bytes: Mapping[str, int] = field(default_factory=dict)  # Fixed KV cache per card, so vLLM cannot grow.

    def cells(self, card: str) -> int:
        return self.generate_cells_by_card.get(card, self.generate_cells)

    @property
    def router_folder(self) -> Path:
        return ROOT / "data_review" / self.router_dir

    @property
    def router_checkpoint(self) -> Path:
        return ROOT / "checkpoints" / self.router_file

    def chunk_tokens(self, card: str, sampler: str) -> int:
        budgets = self.gradient_tokens_vllm if sampler == "vllm" else self.gradient_tokens_alone
        return budgets[card]

    def vllm_share(self, card: str, total_bytes: int | None) -> float:
        """Share of the card for vLLM. A budget in GB is turned into a share of the real card size."""
        if not self.vllm_memory_gb:
            return self.vllm_fraction
        if not total_bytes:
            raise ValueError(f"{self.name}: the card size is needed to apply a VRAM budget")
        return self.vllm_memory_gb[card] * 1e9 / total_bytes


PROFILES = {
    "qwen06b": ModelProfile(
        name="qwen06b", model_id="Qwen/Qwen3-0.6B",
        path=f"{HUB}/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca",
        num_layers=28, steering_layers=(7, 14, 21), read_layer=7,
        router_dir="router", router_file="router.pt", time_share=False,
        vllm_fraction=0.25, vllm_memory_gb={},  # 0.30 reached 14.17 GB next to PyTorch and the desktop.
        vllm_max_seqs=256, vllm_max_batched_tokens=8192,
        gradient_tokens_vllm={"main": 2000, "second": 2000},
        gradient_tokens_alone={"main": 4000, "second": 4000},
        generate_cells=22000, head_chunk=1024, gradient_checkpointing=False,
        other_vram_gb={"main": 2.7, "second": 0.0}, hard_stop_gb=None,
    ),
    # Sizes for 4B: weights 8.05 GB. Measured gradient graph without checkpointing: 121 KB per token per
    # layer, so 3.1 MB per token over 26 layers. Checkpointing keeps about 0.27 MB per token.
    "qwen4b": ModelProfile(
        name="qwen4b", model_id="Qwen/Qwen3-4B-Instruct-2507",
        path=f"{HUB}/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554",
        num_layers=36, steering_layers=(9, 18, 27), read_layer=9,
        router_dir="router_v2", router_file="router_qwen4b.pt", time_share=True,
        vllm_fraction=0.0, vllm_memory_gb={"main": 10.5, "second": 13.0},  # Main also holds a 2.7-3.1 GB desktop; 11.0 reached 14.03 GB.
        vllm_max_seqs=128, vllm_max_batched_tokens=4096,
        gradient_tokens_vllm={"main": 4000, "second": 8000},
        gradient_tokens_alone={"main": 4000, "second": 8000},
        generate_cells=12000, head_chunk=256, gradient_checkpointing=True,
        other_vram_gb={"main": 2.7, "second": 0.0}, hard_stop_gb=14.0,
        # One 4B copy per card. With these PyTorch limits, 8,000 (main) and 12,000 (second) cells ran out of memory.
        torch_vram_gb={"main": 11.0, "second": 13.5}, generate_cells_by_card={"main": 5000, "second": 8000},
        eval_batch=16,
        # A budget lets vLLM reserve up to 11.23 GiB, or about 14.5 GiB with Windows apps. A fixed cache cannot grow.
        vllm_kv_bytes={"main": int(1.30 * 2**30)},
    ),
}


def get_profile(name: str | None = None) -> ModelProfile:
    """The named profile, else $STEERING_PROFILE, else the 0.6B default."""
    chosen = name or os.environ.get(PROFILE_ENV) or DEFAULT_PROFILE
    if chosen not in PROFILES:
        raise ValueError(f"Unknown model profile {chosen!r}; choose from {sorted(PROFILES)}")
    return PROFILES[chosen]


def profile_for_model(model_id: str) -> ModelProfile | None:
    return next((profile for profile in PROFILES.values() if profile.model_id == model_id), None)


def model_width(profile: ModelProfile, path: str | Path | None = None) -> int:
    """Width from config.json. Refuses a folder whose layer count is not the profile's."""
    with (Path(path or profile.path) / "config.json").open(encoding="utf-8") as file:
        config = json.load(file)
    if config["num_hidden_layers"] != profile.num_layers:
        raise ValueError(f"{profile.name} expects {profile.num_layers} layers, "
                         f"config has {config['num_hidden_layers']}")
    return int(config["hidden_size"])
