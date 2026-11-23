import json
import random
import shutil
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from behaviors.rubrics import score_response
from evaluate import load_cases
from model_profiles import get_profile
from system import NONE, VECTORS_DIR, load_system

RUN = "qwen4b_v2"
SEED = 7
TRAINED = ("engineering", "story", "teaching", "debugging", "math_solving", "explanation", "professional_email")
UNTRAINED = "how_to_guide"
PER_TOPIC = 6
PER_NONE = 4
MAX_PROMPT_TOKENS = 512
BATCH = 8
VRAM_LIMIT_BYTES = 13.5e9
OUT = ROOT / "data_review" / "reports" / "compare_4b.json"
VECTORS = ROOT / "checkpoints" / "compare_4b_vectors"


def prepare_vectors():
    """Copy the 7 trained vectors and add an all-zero how-to vector, which leaves the model unchanged."""
    target = VECTORS / RUN
    target.mkdir(parents=True, exist_ok=True)
    for name in TRAINED:
        shutil.copy(VECTORS_DIR / RUN / f"{name}.pt", target / f"{name}.pt")
    saved = torch.load(target / f"{TRAINED[0]}.pt", map_location="cpu", weights_only=True)
    saved["bank"]["vectors.0"] = torch.zeros_like(saved["bank"]["vectors.0"])
    torch.save(saved, target / f"{UNTRAINED}.pt")


def fits(system, prompt):
    return len(system.tokenizer(system._chat(prompt), add_special_tokens=False)["input_ids"]) <= MAX_PROMPT_TOKENS


def pick(system, cases):
    rng, chosen = random.Random(SEED), []
    for label in (*TRAINED, NONE):
        rows = [c for c in cases if c["label"] == label and fits(system, c["prompt"])]
        chosen += rng.sample(rows, PER_NONE if label == NONE else PER_TOPIC)
    return chosen


def score(label, text, prompt):
    return None if label == NONE else round(score_response(label, text, prompt).score, 3)


def answer_group(system, group, first_id):
    prompts = [c["prompt"] for c in group]
    routed = system.generate(prompts, 256, BATCH)
    normal = system.generate_fixed(prompts, None, 256, BATCH)
    return [{"id": first_id + i, "type": c["label"], "prompt": c["prompt"], "route": a.route,
             "confidence": round(a.confidence, 3), "normal": n, "steered": a.text,
             "normal_score": score(c["label"], n, c["prompt"]),
             "steered_score": score(c["label"], a.text, c["prompt"])}
            for i, (c, a, n) in enumerate(zip(group, routed, normal))]


def main():
    prepare_vectors()
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, VRAM_LIMIT_BYTES / total), 0)
    profile = get_profile("qwen4b")
    system = load_system(None, RUN, vectors_dir=VECTORS, profile="qwen4b")
    chosen = pick(system, load_cases(None, TRAINED, profile.router_folder / "prompts.jsonl"))
    print(json.dumps({"picked": len(chosen)}), flush=True)
    rows = []
    for label in (*TRAINED, NONE):
        start = time.time()
        rows += answer_group(system, [c for c in chosen if c["label"] == label], len(rows) + 1)
        OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({"done": label, "rows": len(rows), "seconds": round(time.time() - start),
                          "peak_gb": round(torch.cuda.max_memory_allocated(0) / 1e9, 2)}), flush=True)
    print(json.dumps({"finished": len(rows), "routes": [r["route"] for r in rows]}), flush=True)


if __name__ == "__main__":
    main()
