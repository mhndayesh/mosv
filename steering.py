"""Resident steering vectors + hard router for Hugging Face causal LMs.

The base transformer is frozen. Hooks add one expert vector AFTER each chosen
decoder block. Every vector is shared by all token positions in that block.
Only the chosen expert receives a gradient from an individual rollout.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

import torch
from torch import nn


def decoder_layers(model: nn.Module):
    """Resolve common Hugging Face decoder-layer layouts, failing clearly."""
    candidates = (
        ("model", "layers"),  # Qwen2/3, Llama, Mistral, Gemma
        ("transformer", "h"),  # GPT-2
        ("gpt_neox", "layers"),  # Pythia
    )
    for path in candidates:
        obj = model
        for part in path:
            obj = getattr(obj, part, None)
            if obj is None:
                break
        else:
            if isinstance(obj, (nn.ModuleList, list, tuple)):
                return obj
    raise TypeError("Unsupported decoder layout. Add its layer path in decoder_layers().")


class SteeringBank(nn.Module):
    def __init__(self, expert_names: Iterable[str], target_layers: Iterable[int], width: int):
        super().__init__()
        self.expert_names = tuple(expert_names)
        self.target_layers = tuple(target_layers)
        if len(self.expert_names) < 1 or len(self.target_layers) < 1:
            raise ValueError("At least one expert and one target layer are required")
        # Stored INSIDE the model, exactly like a resident MoE expert table.
        # Each expert is its OWN Parameter. With AdamW, an inactive expert has
        # grad=None and its old optimizer momentum cannot update it accidentally.
        self.vectors = nn.ParameterList([
            nn.Parameter(torch.zeros(len(self.target_layers), width, dtype=torch.float32))
            for _ in self.expert_names
        ])
        self.active_expert: int | None = None
        self._handles = []

    @contextmanager
    def choose(self, expert: int | None):
        if expert is not None and not (0 <= expert < len(self.expert_names)):
            raise ValueError(f"Unknown expert: {expert}")
        previous = self.active_expert
        self.active_expert = expert
        try:
            yield
        finally:
            self.active_expert = previous

    def attach(self, model: nn.Module):
        all_layers = decoder_layers(model)
        if self._handles:
            raise RuntimeError("Steering hooks were already attached")
        for slot, layer_idx in enumerate(self.target_layers):
            if not (0 <= layer_idx < len(all_layers)):
                raise ValueError(f"Layer {layer_idx} not present in model")

            def hook(_module, _inputs, output, vector_slot=slot):
                if self.active_expert is None:
                    return output  # Identity path: NO expert runs.
                tensor = output[0] if isinstance(output, tuple) else output
                if not torch.is_tensor(tensor):
                    raise TypeError("Expected tensor or tensor-first tuple from decoder layer")
                delta = self.vectors[self.active_expert][vector_slot].to(dtype=tensor.dtype)
                shifted = tensor + delta[None, None, :]
                if isinstance(output, tuple):
                    return (shifted, *output[1:])
                return shifted

            self._handles.append(all_layers[layer_idx].register_forward_hook(hook))

    def detach(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def install_steering(model: nn.Module, expert_names, layer_indices):
    """Freeze base FIRST; then register trainable vector bank and router."""
    for weight in model.parameters():
        weight.requires_grad_(False)
    config = model.config
    width = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None)
    if width is None:
        raise ValueError("Cannot determine model hidden size")
    device = next(model.parameters()).device
    bank = SteeringBank(expert_names, layer_indices, width).to(device)
    router = nn.Linear(width, len(bank.expert_names)).to(device=device, dtype=torch.float32)
    # Register as child modules: experts and router live INSIDE the model.
    model.add_module("steering_bank", bank)
    model.add_module("steering_router", router)
    bank.attach(model)
    return bank, router


@torch.no_grad()
def router_features(model, prompt_ids: torch.Tensor) -> torch.Tensor:
    # Frozen embedding summary provides a very cheap prompt-level router input.
    emb = model.get_input_embeddings()(prompt_ids)
    return emb.float().mean(dim=1).detach()


def resolve_layer_indices(model, csv: str):
    count = len(decoder_layers(model))
    if csv == "auto":
        return sorted(set([count // 4, count // 2, 3 * count // 4]))
    indices = sorted(set(int(s.strip()) for s in csv.split(",") if s.strip()))
    if not indices or any(i < 0 or i >= count for i in indices):
        raise ValueError(f"Selected layers {indices} must be in [0,{count-1}]")
    return indices


def count_parameters(model, bank, router):
    active = sum(p.numel() for p in (*bank.parameters(), *router.parameters()))
    original = sum(p.numel() for p in model.parameters()) - active
    return {"frozen_base": original, "trainable_steering_and_router": active}
