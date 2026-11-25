"""v5 steer layers in PyTorch, laid out exactly like the hy_v3 GGUF (moe/export.py) so training = shipping.

At each steer layer the original MLP stays (the GGUF's shared expert) and a gated expert is added:
    out = mlp(x) + expert_k(x),  k = argmax(gate @ x)   (llama.cpp: sigmoid, top-1, weight normalised to 1)
Expert 0 is "none" (adds nothing). x is the layer's normed MLP input, the same tensor llama.cpp feeds its gate.
The gate is trained first (v5/gate.py) and frozen; RL trains the experts.
"""

from __future__ import annotations

import numpy as np
import torch

LAYERS = (9, 18, 27)
FF = 128  # Expert width: sampling, patching and learning measured the same speed at 64-512 (2026-11-26).


class Steer:
    def __init__(self, classes: list[str], width: int, device: str, ff: int = FF, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        n = len(classes) - 1  # trainable experts; expert 0 = none
        self.classes, self.ff, self.width = list(classes), ff, width
        self.gate = {l: torch.zeros(len(classes), width, device=device) for l in LAYERS}
        small = lambda *s: (torch.randn(*s, generator=g) * 0.02).to(device).requires_grad_()  # noqa: E731
        # down starts at zero, so a fresh Steer leaves the model exactly unchanged.
        self.experts = {l: {"gate": small(n, ff, width), "up": small(n, ff, width),
                            "down": torch.zeros(n, width, ff, device=device, requires_grad=True)} for l in LAYERS}
        self.routed = {l: torch.zeros(len(classes), dtype=torch.long) for l in LAYERS}
        self.hooks = []

    def parameters(self) -> list[torch.Tensor]:
        return [t for parts in self.experts.values() for t in parts.values()]

    def attach(self, model) -> None:
        for layer in LAYERS:
            self.hooks.append(model.model.layers[layer].mlp.register_forward_hook(self._hook(layer)))

    def _hook(self, layer: int):
        def add_expert(_module, inputs, output):
            x = inputs[0]
            pick = (x.float() @ self.gate[layer].T).argmax(-1)
            self.routed[layer] += torch.bincount(pick.flatten(), minlength=len(self.classes)).cpu()
            extra = torch.zeros_like(output)
            e = self.experts[layer]
            for k in range(1, len(self.classes)):
                mask = pick == k
                if not bool(mask.any()):
                    continue
                h = x[mask]
                hidden = torch.nn.functional.silu(h @ e["gate"][k - 1].T.half()) * (h @ e["up"][k - 1].T.half())
                extra = extra.index_put((mask,), hidden @ e["down"][k - 1].T.half())
            return output + extra
        return add_expert

    def routed_share(self) -> dict:
        """Share of tokens each layer sent to each expert since the last call; then resets the counters."""
        out = {}
        for layer, counts in self.routed.items():
            total = max(int(counts.sum()), 1)
            out[layer] = {c: round(int(n) / total, 4) for c, n in zip(self.classes, counts)}
            counts.zero_()
        return out

    def to_moe(self) -> dict:
        """The dict moe/export.py and moe/live_gguf.py write (expert 0 = zeros)."""
        layers = {}
        for layer in LAYERS:
            e = {k: v.detach().float().cpu() for k, v in self.experts[layer].items()}
            zero = lambda t: torch.cat([torch.zeros_like(t[:1]), t]).half().numpy()  # noqa: E731
            layers[layer] = {"gate_inp": self.gate[layer].float().cpu().numpy().astype(np.float32),
                             "gate_exps": zero(e["gate"]), "up_exps": zero(e["up"]), "down_exps": zero(e["down"])}
        return {"classes": self.classes, "ff": self.ff, "layers": layers}

    def state(self) -> dict:
        return {"classes": self.classes, "ff": self.ff, "layers": list(LAYERS),
                "gate": {l: g.cpu() for l, g in self.gate.items()},
                "experts": {l: {k: v.detach().cpu() for k, v in e.items()} for l, e in self.experts.items()}}

    def load(self, state: dict) -> None:
        with torch.no_grad():
            for layer in LAYERS:
                self.gate[layer].copy_(state["gate"][layer])
                for key, value in state.get("experts", {}).get(layer, {}).items():
                    self.experts[layer][key].copy_(value)
