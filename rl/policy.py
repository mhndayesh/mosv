"""Policy-gradient math on batches of sampled answers. Torch is passed in, so card
selection in each worker happens before torch loads."""

from __future__ import annotations

HEAD_CHUNK = 1024  # Qwen has 151,936 output words, so full logits would use many GB.


class NormalizedMomentum:
    """Small steps sized to each layer. Noisy gradients disagree, cancel, and move less."""

    def __init__(self, torch, vector, step_sizes, momentum: float):
        self.torch, self.vector, self.step_sizes, self.momentum = torch, vector, step_sizes, momentum
        self.velocity = torch.zeros_like(vector)
        self.scale = 1.0

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.vector.grad = None

    def step(self) -> None:
        gradient = self.vector.grad.detach()
        unit = gradient / gradient.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        self.velocity = self.momentum * self.velocity + (1 - self.momentum) * unit
        with self.torch.no_grad():
            self.vector.sub_(self.velocity * self.step_sizes.unsqueeze(-1) * self.scale)

    def shrink(self) -> None:
        self.scale *= 0.5

    def state_dict(self) -> dict:
        return {"velocity": self.velocity.detach().cpu(), "scale": self.scale}

    def load_state_dict(self, state: dict) -> None:
        self.velocity, self.scale = state["velocity"].to(self.vector.device), state["scale"]


class ShrinkableAdam:
    """Adam baseline with the same undo-and-shrink safety as the normalized optimizer."""

    def __init__(self, torch, vector, learning_rate: float):
        self.inner = torch.optim.AdamW([vector], lr=learning_rate, weight_decay=0.0)
        self.scale = 1.0

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.inner.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.inner.step()

    def shrink(self) -> None:
        self.scale *= 0.5
        for group in self.inner.param_groups:
            group["lr"] *= 0.5

    def state_dict(self) -> dict:
        return {"inner": self.inner.state_dict(), "scale": self.scale}

    def load_state_dict(self, state: dict) -> None:
        self.inner.load_state_dict(state["inner"])
        self.scale = state["scale"]


def kl_divergence(torch, before, after) -> float:
    """How much the word probabilities changed, in nats per token."""
    return float((before.exp() * (before - after)).sum(dim=-1).mean())


def trim_completion(tokens: list[int], eos_ids: set[int]) -> list[int]:
    """Keep the answer up to and including its first end token."""
    for index, token in enumerate(tokens):
        if token in eos_ids:
            return tokens[:index + 1]
    return tokens


def token_chunks(indices: list[int], lengths: list[int], budget: int) -> list[list[int]]:
    """Group answers so rows x longest answer stays under the gradient memory budget."""
    chunks, current, longest = [], [], 0
    for index in sorted(indices, key=lambda i: lengths[i]):
        width = max(longest, lengths[index])
        if current and width * (len(current) + 1) > budget:
            chunks.append(current)
            current, width = [], lengths[index]
        current.append(index)
        longest = width
    return chunks + ([current] if current else [])


def build_policy_batch(torch, prompts: list[list[int]], completions: list[list[int]], pad_id: int, device) -> dict:
    """Each hidden position predicts the next token, so scoring starts at the prompt's last token."""
    inputs = [prompt + completion[:-1] for prompt, completion in zip(prompts, completions)]
    width = max(len(row) for row in inputs)
    ids = torch.full((len(inputs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(inputs), width), dtype=torch.long)
    rows, cols, targets, owners = [], [], [], []
    for index, (row, prompt, completion) in enumerate(zip(inputs, prompts, completions)):
        ids[index, :len(row)] = torch.tensor(row, dtype=torch.long)
        mask[index, :len(row)] = 1
        for offset, token in enumerate(completion):
            rows.append(index)
            cols.append(len(prompt) - 1 + offset)
            targets.append(token)
            owners.append(index)
    return {"input_ids": ids.to(device), "attention_mask": mask.to(device),
            "rows": torch.tensor(rows, device=device), "cols": torch.tensor(cols, device=device),
            "targets": torch.tensor(targets, device=device), "owners": torch.tensor(owners, device=device)}


def compiled_forward(torch, eager):
    """Compiled for gradient passes. Calls that use a KV cache stay eager, and steering hooks stay outside."""
    compiled = torch.compile(eager, dynamic=True)

    def forward(*args, **kwargs):
        uses_cache = kwargs.get("cache") is not None or kwargs.get("past_key_values") is not None
        return eager(*args, **kwargs) if uses_cache else compiled(*args, **kwargs)
    return forward


def without_gradients(torch, run):
    """Run the layer body with no graph. A steering hook runs after this, so its vector still gets a gradient."""
    def forward(*args, **kwargs):
        with torch.no_grad():
            return run(*args, **kwargs)
    return forward


def checkpointed(torch, run):
    """Keep only the layer's input during the forward pass and redo the layer during backward."""
    from torch.utils.checkpoint import checkpoint

    def forward(*args, **kwargs):
        hidden = args[0] if args else kwargs.get("hidden_states")
        uses_cache = kwargs.get("cache") is not None or kwargs.get("past_key_values") is not None
        if uses_cache or not torch.is_grad_enabled() or not hidden.requires_grad:
            return run(*args, **kwargs)
        return checkpoint(run, *args, use_reentrant=False, preserve_rng_state=False, **kwargs)
    return forward


def prepare_layers(torch, layers, first_steered: int, compile_them: bool, use_checkpoint: bool) -> None:
    """Layers up to the first steered one need no graph. Later ones may be checkpointed."""
    for index, layer in enumerate(layers):
        run = compiled_forward(torch, layer.forward) if compile_them else layer.forward
        if index <= first_steered:
            layer.forward = without_gradients(torch, run)
        elif use_checkpoint:
            layer.forward = checkpointed(torch, run)
        else:
            layer.forward = run


def backward_policy_loss(torch, core, head, batch: dict, advantages, norm: float,
                         head_chunk: int = HEAD_CHUNK) -> float:
    """Backprop -A * log p(answer), running the huge output layer in small chunks."""
    hidden = core(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                  use_cache=False).last_hidden_state
    selected = hidden[batch["rows"], batch["cols"]]
    proxy = selected.detach().requires_grad_(True)
    weights = advantages.to(proxy.device, dtype=torch.float32)[batch["owners"]] / norm
    total = 0.0
    for start in range(0, proxy.shape[0], head_chunk):
        part = range(start, min(start + head_chunk, proxy.shape[0]))
        index = torch.tensor(list(part), device=proxy.device)
        log_probs = torch.log_softmax(head(proxy[index]).float(), dim=-1)
        picked = log_probs.gather(1, batch["targets"][index].unsqueeze(1)).squeeze(1)
        loss = -(weights[index] * picked).sum()
        loss.backward()
        total += float(loss.detach())
    selected.backward(proxy.grad)
    return total
