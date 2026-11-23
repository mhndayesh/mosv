"""vLLM plugin: add one steering vector after chosen Qwen3 layers."""

ARCH = "SteeredQwen3ForCausalLM"
LAYERS = (7, 14, 21)


def register() -> None:
    from vllm import ModelRegistry

    ModelRegistry.register_model(ARCH, "qwen_steering.model:SteeredQwen3ForCausalLM")


def set_steering(model, vectors=None) -> list[float]:
    """Update the live vectors in place, or turn steering off with None."""
    import torch

    model = getattr(model, "_orig_mod", model)
    sizes = []
    for slot, index in enumerate(LAYERS):
        delta = model.model.layers[index].steering_delta
        if vectors is None:
            delta.zero_()
        else:
            delta.copy_(torch.as_tensor(vectors[slot]).to(delta.device, dtype=delta.dtype))
        sizes.append(float(delta.float().norm()))
    return sizes