"""One inference path: read the prompt (steering off), route it, then generate with that expert.

Qwen3-0.6B (float16) + the trained router + the trained vectors, all in one SteeringBank.
The expert is chosen once per request and stays fixed for every token of that answer.
`none` runs with no expert, so the original model computes the answer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from router.pipeline import StopAtLayer, chat_parts, encode, stop_hook
from router.train import decide
from steering import SteeringBank

ROOT = Path(__file__).parent
ROUTER_PATH = ROOT / "checkpoints" / "router.pt"
VECTORS_DIR = ROOT / "checkpoints" / "rl_pilot"
MODEL_ENV = "QWEN_MODEL_PATH"
NONE = "none"
MAX_NEW_TOKENS = 256
ROUTE_BATCH = 32
GENERATE_BATCH = 16


@dataclass(frozen=True)
class Route:
    name: str  # An expert name, or "none".
    confidence: float  # Probability of the top class.


@dataclass(frozen=True)
class Answer:
    text: str
    route: str
    confidence: float


@dataclass
class RouterHead:
    linear: torch.nn.Module
    mean: torch.Tensor
    std: torch.Tensor
    temperature: float
    threshold: float
    classes: tuple[str, ...]
    read_layer: int

    def probabilities(self, features: torch.Tensor) -> np.ndarray:
        scaled = (features.float() - self.mean) / self.std
        with torch.no_grad():
            return (self.linear(scaled) / self.temperature).softmax(1).cpu().numpy()


class System:
    def __init__(self, model, tokenizer, bank: SteeringBank, router: RouterHead, parts, max_new_tokens=MAX_NEW_TOKENS):
        if router.classes[-1] != NONE or tuple(router.classes[:-1]) != bank.expert_names:
            raise ValueError(f"Router classes {router.classes} do not match experts {bank.expert_names}")
        tokenizer.padding_side = "left"  # Batched generation needs the prompt at the right edge.
        self.model, self.tokenizer, self.bank, self.router = model, tokenizer, bank, router
        self.before, self.after, self.pad_id = parts
        self.max_new_tokens = max_new_tokens
        self.device = next(model.parameters()).device

    # ---- routing ----------------------------------------------------------------------------

    def _read(self, texts: list[str]) -> torch.Tensor:
        """Last-token hidden state after the read layer, unsteered, on the chat-formatted prompt."""
        batch = encode(self.tokenizer.backend_tokenizer, texts, self.before, self.after, self.pad_id)
        ids = torch.from_numpy(batch["input_ids"]).to(self.device)
        mask = torch.from_numpy(batch["attention_mask"]).to(self.device)
        positions = (mask.cumsum(1) - 1).clamp(min=0)
        handle = self.model.model.layers[self.router.read_layer].register_forward_hook(stop_hook)
        try:
            with self.bank.choose(None), torch.inference_mode():
                self.model.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False)
        except StopAtLayer as stopped:
            return stopped.hidden[:, -1, :].float()
        finally:
            handle.remove()
        raise RuntimeError("Read layer was never reached")

    def route(self, prompts: list[str]) -> list[Route]:
        routes = []
        for start in range(0, len(prompts), ROUTE_BATCH):
            probabilities = self.router.probabilities(self._read(prompts[start:start + ROUTE_BATCH]))
            choices = decide(probabilities, self.router.threshold, len(self.router.classes) - 1)
            routes += [Route(self.router.classes[c], float(p.max())) for c, p in zip(choices, probabilities)]
        return routes

    # ---- generation -------------------------------------------------------------------------

    def _chat(self, prompt: str) -> str:
        return self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)

    def _generate_batch(self, prompts: list[str], expert: str | None, max_new_tokens: int) -> list[str]:
        batch = self.tokenizer([self._chat(p) for p in prompts], padding=True, return_tensors="pt",
                               add_special_tokens=False).to(self.device)
        index = None if expert is None else self.bank.expert_names.index(expert)
        with self.bank.choose(index), torch.inference_mode():
            output = self.model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                                         pad_token_id=self.pad_id, use_cache=True)
        return [self.tokenizer.decode(row, skip_special_tokens=True).strip()
                for row in output[:, batch["input_ids"].shape[1]:]]

    def generate_fixed(self, prompts: list[str], expert: str | None, max_new_tokens: int | None = None,
                       batch_size: int = GENERATE_BATCH) -> list[str]:
        """Answer every prompt with one expert (or None = original model), in the input order."""
        steps = max_new_tokens or self.max_new_tokens
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))  # Similar lengths pad less.
        texts = [""] * len(prompts)
        for start in range(0, len(order), batch_size):
            chunk = order[start:start + batch_size]
            for i, text in zip(chunk, self._generate_batch([prompts[i] for i in chunk], expert, steps)):
                texts[i] = text
        return texts

    def generate(self, prompts: list[str], max_new_tokens: int | None = None,
                 batch_size: int = GENERATE_BATCH) -> list[Answer]:
        """Route first, then answer each group with its own expert fixed for the whole request."""
        routes = self.route(prompts)
        answers: list[Answer | None] = [None] * len(prompts)
        for name in dict.fromkeys(r.name for r in routes):
            members = [i for i, r in enumerate(routes) if r.name == name]
            texts = self.generate_fixed([prompts[i] for i in members], None if name == NONE else name,
                                        max_new_tokens, batch_size)
            for i, text in zip(members, texts):
                answers[i] = Answer(text, routes[i].name, routes[i].confidence)
        return answers


# ---- loading --------------------------------------------------------------------------------

def resolve_model_path(model_path: str | None, model_id: str) -> str:
    """Explicit path, then $QWEN_MODEL_PATH, then the WSL path used in training if it exists, then the hub id."""
    from router.pipeline import MODEL as WSL_PATH

    for candidate in (model_path, os.environ.get(MODEL_ENV), WSL_PATH):
        if candidate and Path(candidate).exists():
            return candidate
    if model_path or os.environ.get(MODEL_ENV):
        raise FileNotFoundError(f"Model path not found: {model_path or os.environ[MODEL_ENV]}")
    return model_id  # Resolved from the local Hugging Face cache.


def load_router(path: Path, device) -> tuple[RouterHead, dict]:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    classes = tuple(saved["classes"])
    linear = torch.nn.Linear(saved["mean"].numel(), len(classes))
    linear.load_state_dict(saved["state"])
    head = RouterHead(linear.to(device).eval(), saved["mean"].to(device), saved["std"].to(device),
                      float(saved["temperature"]), float(saved["threshold"]), classes, int(saved["read_layer"]))
    return head, saved


def load_vectors(run: str, experts: tuple[str, ...], vectors_dir: Path = VECTORS_DIR) -> tuple[list, tuple[int, ...]]:
    """One (layers, width) tensor per expert, in expert order. All must exist and share the same layers."""
    missing = [name for name in experts if not (vectors_dir / run / f"{name}.pt").exists()]
    if missing:
        raise FileNotFoundError(f"Run '{run}' has no vectors for {missing} in {vectors_dir / run}")
    saved = [torch.load(vectors_dir / run / f"{name}.pt", map_location="cpu", weights_only=True) for name in experts]
    layers = {tuple(s["layers"]) for s in saved}
    if len(layers) != 1:
        raise ValueError(f"Experts in run '{run}' were trained on different layers: {layers}")
    return [s["bank"]["vectors.0"].float() for s in saved], layers.pop()


def build_bank(experts: tuple[str, ...], layers: tuple[int, ...], vectors: list, device) -> SteeringBank:
    bank = SteeringBank(experts, layers, vectors[0].shape[-1])
    with torch.no_grad():
        for slot, vector in zip(bank.vectors, vectors):
            slot.copy_(vector)
    return bank.to(device)


def load_system(model_path: str | None = None, run: str = "pilot_final", router_path: Path = ROUTER_PATH,
                vectors_dir: Path = VECTORS_DIR, max_new_tokens: int = MAX_NEW_TOKENS) -> System:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    router, saved = load_router(router_path, device)
    path = resolve_model_path(model_path, saved["model_id"])
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16).to(device).eval()
    experts = router.classes[:-1]
    vectors, layers = load_vectors(run, experts, vectors_dir)
    bank = build_bank(experts, layers, vectors, device)
    bank.attach(model)
    return System(model, tokenizer, bank, router, chat_parts(path), max_new_tokens)
