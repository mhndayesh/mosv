"""vLLM plugin: add one steering vector after chosen Qwen3 layers."""

ARCH = "SteeredQwen3ForCausalLM"
LAYERS_KEY = "steering_layers"  # Set through hf_overrides, so every vLLM child process gets the same config.


def register() -> None:
    from vllm import ModelRegistry

    ModelRegistry.register_model(ARCH, "qwen_steering.model:SteeredQwen3ForCausalLM")


def steering_layers(config) -> tuple[int, ...]:
    """The layers to steer. Missing layers stop the engine, so a wrong default can never steer silently."""
    layers = getattr(config, LAYERS_KEY, None)
    if not layers:
        raise ValueError(f"Model config has no {LAYERS_KEY}; pass it in hf_overrides")
    return tuple(int(index) for index in layers)


def set_steering(model, vectors=None) -> list[float]:
    """Update the live vectors in place, or turn steering off with None."""
    import torch

    model = getattr(model, "_orig_mod", model)
    layers = steering_layers(model.config)
    if vectors is not None and len(vectors) != len(layers):
        raise ValueError(f"Got {len(vectors)} vectors for {len(layers)} steering layers")
    sizes = []
    for slot, index in enumerate(layers):
        delta = model.model.layers[index].steering_delta
        if vectors is None:
            delta.zero_()
        else:
            delta.copy_(torch.as_tensor(vectors[slot]).to(delta.device, dtype=delta.dtype))
        sizes.append(float(delta.float().norm()))
    return sizes