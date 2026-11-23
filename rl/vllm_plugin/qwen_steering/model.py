"""Qwen3 with a steering buffer added after selected layers."""

from functools import partial

import torch
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM

from . import steering_layers


def steered_forward(original, layer, positions, hidden_states, residual=None):
    output, residual = original(positions, hidden_states, residual)
    return output + layer.steering_delta, residual


class SteeredQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        width = self.config.hidden_size
        for index in steering_layers(self.config):
            layer = self.model.layers[index]
            weight = next(layer.parameters())
            layer.register_buffer("steering_delta", torch.zeros(width, dtype=weight.dtype, device=weight.device),
                                  persistent=False)
            layer.forward = partial(steered_forward, layer.forward, layer)
