"""v6 steer layers for Gemma 4 in PyTorch, matching moe/gemma_host.py (llama.cpp gemma4 per-layer MoE) exactly.

At each steer layer, with h = the residual entering the feed-forward block and m = the original MLP output:
    ffn_out = post_ffw_norm( rms(m) + w2 * rms(expert_k(rms(h))) ),  k = argmax router(h)
router(h) = rms(h) / sqrt(width) @ gate.T (softmax top-1 -> weight 1), expert 0 = "none" (adds exactly 0).
In HF Gemma4 the post_feedforward_layernorm runs after the MLP, so the hook replaces the MLP output m with
rms(m) + w2 * rms(e); the layer's own post norm then gives llama.cpp's result. w2 (post_ffw_norm_2 weight) starts at
zero, so a fresh Steer leaves the model unchanged, then grows smoothly; expert directions are trained through it.
Experts: down(gelu_tanh(gate x) * up x), like llama.cpp's LLM_FFN_GELU expert path.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

FF = 128
EPS = 1e-6


def rms(x: torch.Tensor) -> torch.Tensor:
    return x * torch.pow(x.float().pow(2).mean(-1, keepdim=True) + EPS, -0.5).to(x.dtype)


class GemmaSteer:
    def __init__(self, classes: list[str], width: int, layers: tuple[int, ...], device: str, ff: int = FF, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        n = len(classes) - 1
        self.classes, self.width, self.layers, self.ff = list(classes), width, tuple(layers), ff
        rnd = lambda *s: (torch.randn(*s, generator=g) * 0.02).to(device).requires_grad_()  # noqa: E731
        self.gate = {l: torch.zeros(len(classes), width, device=device) for l in layers}
        self.experts = {l: {"gate": rnd(n, ff, width), "up": rnd(n, ff, width), "down": rnd(n, width, ff),
                            "w2": torch.zeros(width, device=device, requires_grad=True)} for l in layers}
        self.routed = {l: torch.zeros(len(classes), dtype=torch.long) for l in layers}
        self.h = {}
        self.hooks = []

    def parameters(self) -> list[torch.Tensor]:
        return [t for parts in self.experts.values() for t in parts.values()]

    def route(self, layer: int, h: torch.Tensor) -> torch.Tensor:
        x = rms(h.float()) * self.width ** -0.5
        return (x @ self.gate[layer].T).argmax(-1)

    def attach(self, model) -> None:
        layers = model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers
        for l in self.layers:
            block = layers[l]
            self.hooks.append(block.pre_feedforward_layernorm.register_forward_pre_hook(self._keep_h(l)))
            self.hooks.append(block.mlp.register_forward_hook(self._add(l)))

    def _keep_h(self, layer):
        def keep(_module, args):
            self.h[layer] = args[0]
        return keep

    def _add(self, layer):
        def add(_module, _inputs, m):
            h = self.h.pop(layer)
            pick = self.route(layer, h)
            self.routed[layer] += torch.bincount(pick.flatten(), minlength=len(self.classes)).cpu()
            e = self.experts[layer]
            hn = rms(h)
            extra = torch.zeros_like(m)
            for k in range(1, len(self.classes)):
                mask = pick == k
                if not bool(mask.any()):
                    continue
                x = hn[mask]
                y = F.gelu(x @ e["gate"][k - 1].T.to(x.dtype), approximate="tanh") * (x @ e["up"][k - 1].T.to(x.dtype))
                extra = extra.index_put((mask,), rms(y @ e["down"][k - 1].T.to(x.dtype)))
            return rms(m) + extra * e["w2"].to(m.dtype)
        return add

    def routed_share(self) -> dict:
        out = {}
        for layer, counts in self.routed.items():
            total = max(int(counts.sum()), 1)
            out[layer] = {c: round(int(v) / total, 4) for c, v in zip(self.classes, counts)}
            counts.zero_()
        return out

    def gguf_layers(self) -> dict[int, dict[str, np.ndarray]]:
        """Arrays for moe/gemma_host.build / patch (expert 0 = zeros). post_ffw_norm_2 carries w2."""
        out = {}
        for l in self.layers:
            e = {k: v.detach().float().cpu() for k, v in self.experts[l].items()}
            zero = lambda t: torch.cat([torch.zeros_like(t[:1]), t]).half().numpy()  # noqa: E731
            ones = np.ones(self.width, np.float32)
            out[l] = {"ffn_gate_inp.weight": self.gate[l].float().cpu().numpy(),
                      "ffn_gate_inp.scale": ones.copy(), "pre_ffw_norm_2.weight": ones.copy(),
                      "post_ffw_norm_1.weight": ones.copy(), "post_ffw_norm_2.weight": e["w2"].numpy().astype(np.float32),
                      "ffn_gate_exps.weight": zero(e["gate"]), "ffn_up_exps.weight": zero(e["up"]),
                      "ffn_down_exps.weight": zero(e["down"])}
        return out

    def state(self) -> dict:
        return {"classes": self.classes, "layers": list(self.layers), "ff": self.ff,
                "gate": {l: g.cpu() for l, g in self.gate.items()},
                "experts": {l: {k: v.detach().cpu() for k, v in e.items()} for l, e in self.experts.items()}}

    def load(self, state: dict) -> None:
        with torch.no_grad():
            for l in self.layers:
                self.gate[l].copy_(state["gate"][l])
                for k, v in state.get("experts", {}).get(l, {}).items():
                    self.experts[l][k].copy_(v)
