"""Train the router from the shared stream, compare it with a plain offline fit, and test it."""

from __future__ import annotations

import json
import os
import queue
import time

import numpy as np

from model_profiles import get_profile
from router import CLASSES, NONE_INDEX

PROFILE = get_profile()
REPORT = PROFILE.router_folder / "report.json"
CHECKPOINT = PROFILE.router_checkpoint
WARMUP_PER_CLASS = 64
STREAM_PER_CLASS = 16
LEARNING_RATE = 0.01
MAX_EPOCHS = 400
PATIENCE = 40
WEIGHT_DECAYS = (1e-4, 1e-3, 1e-2)
MAX_FALSE_STEER = 0.01
MAX_WRONG_EXPERT = 0.01
MIN_PRECISION = 0.99
MIN_COVERAGE = 0.80  # Misses are safe, but a router that almost never steers is useless.
MAX_HARD_FALSE_STEER = 0.03


def decide(probabilities: np.ndarray, threshold: float, none_index: int = NONE_INDEX) -> np.ndarray:
    """Pick the top class, but fall back to none when not confident enough."""
    choices = probabilities.argmax(axis=1)
    choices[probabilities.max(axis=1) < threshold] = none_index
    return choices


def false_steer_rate(choices: np.ndarray, labels: np.ndarray, none_index: int = NONE_INDEX) -> float:
    negatives = labels == none_index
    return float((choices[negatives] != none_index).mean()) if negatives.any() else 0.0


def pick_threshold(probabilities: np.ndarray, labels: np.ndarray) -> float:
    for step in range(50, 200):  # Exact 0.005 steps: rounding must never lower the bar.
        value = step / 200
        if false_steer_rate(decide(probabilities, value), labels) <= MAX_FALSE_STEER:
            return value
    return 0.995


def recalls(choices: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {name: round(float((choices[labels == i] == i).mean()), 4)
            for i, name in enumerate(CLASSES) if (labels == i).any()}


def safety(choices: np.ndarray, labels: np.ndarray) -> dict:
    """A miss leaves the model unchanged. Only a steer to the wrong place does harm."""
    steered, task = choices != NONE_INDEX, labels != NONE_INDEX
    correct = int(((choices == labels) & steered).sum())
    wrong_expert = int(((choices != labels) & steered & task).sum())
    false_steer = int((steered & ~task).sum())
    return {"prompts": len(labels), "steered": int(steered.sum()), "correct_steers": correct,
            "wrong_expert": wrong_expert, "false_steer": false_steer,
            "missed_left_unchanged": int((~steered & task).sum()),
            "wrong_expert_rate": round(wrong_expert / max(int(task.sum()), 1), 4),
            "precision": round(correct / max(int(steered.sum()), 1), 4),
            "coverage": round(correct / max(int(task.sum()), 1), 4)}


def safety_pass(result: dict, false_steer_rate_value: float, hard_rate: float) -> dict[str, bool]:
    return {"wrong_expert": result["wrong_expert_rate"] <= MAX_WRONG_EXPERT,
            "false_steer": false_steer_rate_value <= MAX_FALSE_STEER,
            "hard_false_steer": hard_rate <= MAX_HARD_FALSE_STEER,
            "precision": result["precision"] >= MIN_PRECISION,
            "still_useful": result["coverage"] >= MIN_COVERAGE}


def calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    confidence, correct = probabilities.max(axis=1), probabilities.argmax(axis=1) == labels
    total = 0.0
    for low in np.arange(0, 1, 1 / bins):
        inside = (confidence > low) & (confidence <= low + 1 / bins)
        if inside.any():
            total += inside.mean() * abs(correct[inside].mean() - confidence[inside].mean())
    return round(float(total), 4)


def new_router(torch, device, width: int):
    return torch.nn.Linear(width, len(CLASSES)).to(device)


def stream_phase(torch, stream, abort, device) -> dict:
    """Learn from data as it arrives, always in class-balanced mini-batches."""
    pools = [[] for _ in CLASSES]
    seen, model, optimizer, mean, std, steps = set(), None, None, None, None, 0
    rng = np.random.default_rng(7)
    while True:
        try:
            item = stream.get(timeout=1)
        except queue.Empty:
            if abort.is_set():
                raise SystemExit(3)
            continue
        if item is None:
            break
        if seen & set(item["ids"]):
            raise ValueError("Duplicate id reached the training stream")
        seen.update(item["ids"])
        for label, vector in zip(item["labels"].tolist(), item["features"]):
            pools[label].append(vector)
        if model is None and all(len(pool) >= WARMUP_PER_CLASS for pool in pools):
            sample = torch.from_numpy(np.stack([v for pool in pools for v in pool])).to(device)
            mean, std = sample.mean(0), sample.std(0).clamp_min(1e-6)
            model = new_router(torch, device, sample.shape[1])
            optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
        while model is not None and all(len(pool) >= STREAM_PER_CLASS for pool in pools):
            vectors, labels = [], []
            for label, pool in enumerate(pools):
                for index in sorted(rng.choice(len(pool), STREAM_PER_CLASS, replace=False).tolist(), reverse=True):
                    vectors.append(pool.pop(index))
                    labels.append(label)
            x = (torch.from_numpy(np.stack(vectors)).to(device) - mean) / std
            loss = torch.nn.functional.cross_entropy(model(x), torch.tensor(labels, device=device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            steps += 1
    return {"model": model, "mean": mean, "std": std, "steps": steps, "prompts": len(seen)}


def fit(torch, train, val, mean, std, weight_decay, device, start_state=None) -> dict:
    x, y = ((torch.from_numpy(train[0]).to(device) - mean) / std, torch.from_numpy(train[1]).to(device))
    xv, yv = ((torch.from_numpy(val[0]).to(device) - mean) / std, torch.from_numpy(val[1]).to(device))
    counts = torch.bincount(y, minlength=len(CLASSES)).float()
    weights = counts.sum() / (len(CLASSES) * counts)
    model = new_router(torch, device, x.shape[1])
    if start_state is not None:
        model.load_state_dict(start_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)
    best, best_loss, bad = None, float("inf"), 0
    for _ in range(MAX_EPOCHS):
        loss = torch.nn.functional.cross_entropy(model(x), y, weight=weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            val_loss = float(torch.nn.functional.cross_entropy(model(xv), yv))
        if val_loss < best_loss - 1e-4:
            best, best_loss, bad = {k: v.detach().clone() for k, v in model.state_dict().items()}, val_loss, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best)
    with torch.no_grad():
        accuracy = float((model(xv).argmax(1) == yv).float().mean())
    return {"state": best, "val_loss": best_loss, "val_accuracy": accuracy, "mean": mean, "std": std,
            "weight_decay": weight_decay}


def fit_temperature(torch, logits, labels) -> float:
    log_t = torch.zeros(1, device=logits.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().detach())


def probabilities(torch, result, features, temperature, device) -> np.ndarray:
    model = new_router(torch, device, features.shape[1])
    model.load_state_dict(result["state"])
    x = (torch.from_numpy(features).to(device) - result["mean"]) / result["std"]
    with torch.no_grad():
        return (model(x) / temperature).softmax(1).cpu().numpy()


def choose_router(torch, online, data, device) -> tuple[dict, dict]:
    train, val = data["train"], data["val"]
    candidates = {}
    if online["model"] is not None:
        state = {k: v.detach().clone() for k, v in online["model"].state_dict().items()}
        candidates["stream_then_replay"] = fit(torch, train, val, online["mean"], online["std"], 1e-3, device, state)
    x = torch.from_numpy(train[0]).to(device)
    mean, std = x.mean(0), x.std(0).clamp_min(1e-6)
    for decay in WEIGHT_DECAYS:
        candidates[f"offline_wd_{decay:g}"] = fit(torch, train, val, mean, std, decay, device)
    summary = {name: {"val_loss": round(c["val_loss"], 4), "val_accuracy": round(c["val_accuracy"], 4)}
               for name, c in candidates.items()}
    best_name = min(candidates, key=lambda name: candidates[name]["val_loss"])
    return {**candidates[best_name], "name": best_name}, summary


def evaluate(torch, best, data, device) -> tuple[dict, float, float]:
    val_x, val_y, _ = data["val"]
    model = new_router(torch, device, val_x.shape[1])
    model.load_state_dict(best["state"])
    tune_x, tune_y, _ = data["tune"]
    held_x, held_y = np.concatenate([val_x, tune_x]), np.concatenate([val_y, tune_y])
    with torch.no_grad():  # Calibrate on data that contains real mistakes.
        logits = model((torch.from_numpy(held_x).to(device) - best["mean"]) / best["std"])
    temperature = fit_temperature(torch, logits, torch.from_numpy(held_y).to(device))
    threshold = pick_threshold(probabilities(torch, best, held_x, temperature, device), held_y)
    test_x, test_y, _ = data["test"]
    test_p = probabilities(torch, best, test_x, temperature, device)
    test_choice = decide(test_p, threshold)
    hard_choice = decide(probabilities(torch, best, data["hard"][0], temperature, device), threshold)
    per_class = recalls(test_choice, test_y)
    report = {"test_prompts": len(test_y), "test_recall": per_class,
              "test_accuracy": round(float((test_choice == test_y).mean()), 4),
              "false_steer_none": round(false_steer_rate(test_choice, test_y), 4),
              "false_steer_hard": round(false_steer_rate(hard_choice, data["hard"][1]), 4),
              "hard_prompts": len(hard_choice), "calibration_error": calibration_error(test_p, test_y)}
    if "natural" in data:
        natural_x, natural_y, _ = data["natural"]
        natural_choice = decide(probabilities(torch, best, natural_x, temperature, device), threshold)
        report["natural_prompts"] = len(natural_y)
        report["natural_recall"] = recalls(natural_choice, natural_y)
    if "final" in data:
        final_x, final_y, _ = data["final"]
        final_choice = decide(probabilities(torch, best, final_x, temperature, device), threshold)
        report["final_prompts"] = len(final_y)
        report["final_accuracy"] = round(float((final_choice == final_y).mean()), 4)
        report["final_recall"] = recalls(final_choice, final_y)
        report["final_false_steer"] = round(false_steer_rate(final_choice, final_y), 4)
        report["final_safety"] = safety(final_choice, final_y)
    report["test_safety"] = safety(test_choice, test_y)
    report["pass"] = safety_pass(report["test_safety"], report["false_steer_none"], report["false_steer_hard"])
    return report, temperature, threshold


def trainer(stream, audit_ok, abort, card) -> None:
    os.sched_setaffinity(0, [12, 13, 14, 15, 24, 25, 26, 27])
    os.environ.update(HIP_VISIBLE_DEVICES=card, CUDA_VISIBLE_DEVICES=card, OMP_NUM_THREADS="4")
    import torch

    from router.pipeline import load_audited, read_rows

    try:
        torch.manual_seed(13)
        device = "cuda:0"
        started = time.time()
        online = stream_phase(torch, stream, abort, device)
        while not audit_ok.is_set():
            if abort.is_set():
                raise SystemExit(3)
            time.sleep(0.1)
        data = load_audited(read_rows())
        best, candidates = choose_router(torch, online, data, device)
        report, temperature, threshold = evaluate(torch, best, data, device)
        report = {"chosen": best["name"], "candidates": candidates, "stream_steps": online["steps"],
                  "stream_prompts": online["prompts"], "temperature": round(temperature, 4),
                  "threshold": threshold, "train_seconds": round(time.time() - started, 1), **report}
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"classes": list(CLASSES), "read_layer": PROFILE.read_layer, "model_id": PROFILE.model_id,
                    "profile": PROFILE.name,
                    "state": {k: v.cpu() for k, v in best["state"].items()}, "mean": best["mean"].cpu(),
                    "std": best["std"].cpu(), "temperature": temperature, "threshold": threshold}, CHECKPOINT)
        REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"router": report}), flush=True)
    except SystemExit:
        raise
    except Exception:
        abort.set()
        raise
