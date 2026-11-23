"""Two separate GPU input pipelines feeding one shared training queue.

prompts A -> builders -> input queue A -> main card   --\\
                                                         -> record workers -> stream -> trainer
prompts B -> builders -> input queue B -> second card --/

Run in WSL: python -m router.pipeline
For another model or topic set: STEERING_PROFILE=qwen4b STEERING_TOPICS=v2 python -m router.pipeline
Each prompt belongs to exactly one card. Finished batches are saved at once, so
a stopped run resumes. Training only uses the full set after an exactly-once audit.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from model_profiles import get_profile, model_width
from router import CLASSES

PROFILE = get_profile()  # Child processes inherit $STEERING_PROFILE, so every stage reads the same model.
PROMPTS = PROFILE.router_folder / "prompts.jsonl"
CACHE = PROFILE.router_folder / "features"
MODEL = PROFILE.path
READ_LAYER = PROFILE.read_layer
BATCH_SIZE = 128
MAX_BODY_TOKENS = 160
INPUT_DEPTH = 8
RESULT_DEPTH = 16
STREAM_DEPTH = 32
START_GAP_SECONDS = 3  # Added after the first card has fully loaded.
STALL_SECONDS = 120
STATUS_SECONDS = 5

# 28 logical CPUs: each GPU pipeline owns 10, the shared training queue owns 8.
PIPELINES = {
    "main": {"card": "1", "bus": 3, "build": [0, 1, 2, 3, 4, 16, 17, 18], "gpu": [5, 19]},
    "second": {"card": "0", "bus": 6, "build": [6, 7, 8, 9, 10, 20, 21, 22], "gpu": [11, 23]},
}
RECORD_CPUS = [12, 13, 14, 15, 24, 25, 26, 27]


def read_rows(path: Path = PROMPTS) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def plan_batches(rows: list[dict], names=tuple(PIPELINES), size: int = BATCH_SIZE) -> dict[str, list[dict]]:
    """Fixed split by ID, then similar lengths together so padding stays small."""
    plans: dict[str, list[dict]] = {name: [] for name in names}
    ordered = sorted(rows, key=lambda row: row["id"])
    for index, name in enumerate(names):
        mine = sorted(ordered[index::len(names)], key=lambda row: (len(row["text"]), row["id"]))
        for start in range(0, len(mine), size):
            chunk = mine[start:start + size]
            plans[name].append({
                "batch_id": f"{name}_{start // size:05d}",
                "ids": [row["id"] for row in chunk],
                "texts": [row["text"] for row in chunk],
                "labels": [CLASSES.index(row["label"]) for row in chunk],
                "splits": [row["split"] for row in chunk],
            })
    return plans


def get_or_abort(source, abort):
    while not abort.is_set():
        try:
            return source.get(timeout=1)
        except queue.Empty:
            continue
    raise SystemExit(3)


def put_or_abort(target, item, abort) -> None:
    while not abort.is_set():
        try:
            target.put(item, timeout=1)
            return
        except queue.Full:
            continue
    raise SystemExit(3)


def chat_parts(model_path: str = MODEL) -> tuple[list[int], list[int], int]:
    """The fixed token IDs around every user message, computed once."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    marker = "QZXMARKERXZQ"
    text = tokenizer.apply_chat_template([{"role": "user", "content": marker}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    before, after = text.split(marker)
    return (tokenizer(before, add_special_tokens=False)["input_ids"],
            tokenizer(after, add_special_tokens=False)["input_ids"], tokenizer.pad_token_id)


def encode(tokenizer, texts: list[str], before: list[int], after: list[int], pad_id: int) -> dict:
    """Multicore tokenize, trim only the user text, and pad on the left."""
    rows = [before + item.ids[:MAX_BODY_TOKENS] + after
            for item in tokenizer.encode_batch(texts, add_special_tokens=False)]
    width = max(len(row) for row in rows)
    ids = np.full((len(rows), width), pad_id, dtype=np.int64)
    mask = np.zeros((len(rows), width), dtype=np.int64)
    for index, row in enumerate(rows):
        ids[index, width - len(row):] = row
        mask[index, width - len(row):] = 1
    return {"input_ids": ids, "attention_mask": mask}


def builder(tasks, inputs, cpus, parts, abort) -> None:
    os.sched_setaffinity(0, cpus)
    os.environ.update(TOKENIZERS_PARALLELISM="true", RAYON_NUM_THREADS=str(len(cpus)), OMP_NUM_THREADS="1")
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(f"{MODEL}/tokenizer.json")
    while (task := get_or_abort(tasks, abort)) is not None:
        put_or_abort(inputs, {**task, **encode(tokenizer, task["texts"], *parts), "texts": None}, abort)


class StopAtLayer(Exception):
    def __init__(self, hidden):
        self.hidden = hidden


def stop_hook(_module, _inputs, output):
    raise StopAtLayer(output[0] if isinstance(output, tuple) else output)


def read_batch(torch, model, batch):
    ids = torch.from_numpy(batch["input_ids"]).to("cuda:0")
    mask = torch.from_numpy(batch["attention_mask"]).to("cuda:0")
    positions = (mask.cumsum(1) - 1).clamp(min=0)
    try:
        model.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False)
    except StopAtLayer as stopped:
        return stopped.hidden[:, -1, :].float().cpu().numpy()
    raise RuntimeError("Read layer was never reached")


def load_model(torch, spec):
    from transformers import AutoModelForCausalLM

    if torch.cuda.device_count() != 1:
        raise RuntimeError("GPU worker must see exactly one card")
    bus = getattr(torch.cuda.get_device_properties(0), "pci_bus_id", None)
    if bus is not None and bus != spec["bus"]:
        raise RuntimeError(f"Wrong card: bus {bus}, expected {spec['bus']}")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to("cuda:0").eval()
    model.model.layers[READ_LAYER].register_forward_hook(stop_hook)
    return model


def gpu_worker(name, spec, inputs, results, beat, ready, abort) -> None:
    os.sched_setaffinity(0, spec["gpu"])
    os.environ.update(HIP_VISIBLE_DEVICES=spec["card"], CUDA_VISIBLE_DEVICES=spec["card"],
                      OMP_NUM_THREADS="2", HF_HUB_OFFLINE="1")
    import torch

    try:
        model = load_model(torch, PIPELINES[name])
        beat.value = time.time()
        ready.set()
        waited, computed, count = 0.0, 0.0, 0
        with torch.inference_mode():
            while True:
                started = time.time()
                batch = get_or_abort(inputs, abort)
                waited += time.time() - started
                if batch is None:
                    break
                started = time.time()
                features = read_batch(torch, model, batch)
                computed += time.time() - started
                count += len(batch["ids"])
                beat.value = time.time()
                put_or_abort(results, {k: batch[k] for k in ("batch_id", "ids", "labels", "splits")}
                             | {"features": features}, abort)
        print(json.dumps({"gpu": name, "prompts": count, "compute_s": round(computed, 2),
                          "waited_for_data_s": round(waited, 2),
                          "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2)}), flush=True)
    except SystemExit:
        raise
    except Exception:
        abort.set()
        raise


def check_result(result: dict, width: int) -> None:
    features, ids = result["features"], result["ids"]
    if features.shape != (len(ids), width):
        raise ValueError(f"{result['batch_id']}: bad shape {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError(f"{result['batch_id']}: non-finite features")
    if len(set(ids)) != len(ids) or len(result["labels"]) != len(ids) or len(result["splits"]) != len(ids):
        raise ValueError(f"{result['batch_id']}: ids, labels and splits do not line up")


def save_chunk(result: dict, cache: Path) -> None:
    temp = cache / f"{result['batch_id']}.tmp.npz"
    np.savez(temp, ids=np.array(result["ids"]), labels=np.array(result["labels"], dtype=np.int64),
             splits=np.array(result["splits"]), features=result["features"])
    os.replace(temp, cache / f"{result['batch_id']}.npz")


def record_worker(results, stream, cache, cpu, width, abort) -> None:
    os.sched_setaffinity(0, [cpu])
    try:
        while (result := get_or_abort(results, abort)) is not None:
            check_result(result, width)
            save_chunk(result, cache)
            keep = [i for i, split in enumerate(result["splits"]) if split == "train"]
            if keep:
                put_or_abort(stream, {"ids": [result["ids"][i] for i in keep],
                                      "labels": np.array(result["labels"], dtype=np.int64)[keep],
                                      "features": result["features"][keep]}, abort)
    except SystemExit:
        raise
    except Exception:
        abort.set()
        raise


def load_audited(rows: list[dict], cache: Path = CACHE) -> dict:
    """Every prompt exactly once, with its own label and split, or raise."""
    expected = {row["id"]: row for row in rows}
    seen: set[str] = set()
    parts: dict[str, list] = {}
    for path in sorted(cache.glob("*.npz")):
        if path.name.endswith(".tmp.npz"):
            continue
        with np.load(path) as chunk:
            ids, labels, splits, features = (chunk[k] for k in ("ids", "labels", "splits", "features"))
        for index, uid in enumerate(ids.tolist()):
            row = expected.get(uid)
            if row is None or uid in seen:
                raise ValueError(f"Unknown or duplicate id {uid} in {path.name}")
            if CLASSES[int(labels[index])] != row["label"] or str(splits[index]) != row["split"]:
                raise ValueError(f"Label or split mismatch for {uid} in {path.name}")
            seen.add(uid)
            parts.setdefault(row["split"], []).append((uid, int(labels[index]), features[index]))
    missing = len(expected) - len(seen)
    if missing:
        raise ValueError(f"{missing} prompts are missing from the saved features")
    output = {}
    for split, items in parts.items():
        items.sort(key=lambda item: item[0])
        output[split] = (np.stack([i[2] for i in items]), np.array([i[1] for i in items]), [i[0] for i in items])
    return output


class Supervisor:
    """Starts the workers, closes each stage in order, and stops everything on failure."""

    def __init__(self, rows, plans):
        self.ctx = mp.get_context("spawn")
        self.rows = rows
        self.abort = self.ctx.Event()
        self.audit_ok = self.ctx.Event()
        self.results = self.ctx.Queue(RESULT_DEPTH)
        self.stream = self.ctx.Queue(STREAM_DEPTH)
        self.inputs = {name: self.ctx.Queue(INPUT_DEPTH) for name in PIPELINES}
        self.tasks = {name: self.ctx.Queue() for name in PIPELINES}
        self.beats = {name: self.ctx.Value("d", time.time()) for name in PIPELINES}
        self.ready = {name: self.ctx.Event() for name in PIPELINES}
        self.parts = chat_parts()
        self.width = model_width(PROFILE)
        self.builders, self.gpus, self.records, self.trainer = {}, {}, [], None
        for name, plan in plans.items():
            for task in plan:
                self.tasks[name].put(task)
            self.tasks[name].put(None)

    def everyone(self):
        groups = [*self.gpus.values(), *self.records, *[p for ps in self.builders.values() for p in ps]]
        return groups + ([self.trainer] if self.trainer else [])

    def start(self) -> None:
        from router.train import trainer

        for cpu in RECORD_CPUS:
            self.records.append(self.ctx.Process(target=record_worker, args=(
                self.results, self.stream, CACHE, cpu, self.width, self.abort)))
            self.records[-1].start()
        self.trainer = self.ctx.Process(target=trainer, args=(self.stream, self.audit_ok, self.abort, PIPELINES["main"]["card"]))
        self.trainer.start()
        for name, spec in PIPELINES.items():
            self.builders[name] = [self.ctx.Process(target=builder, args=(
                self.tasks[name], self.inputs[name], spec["build"], self.parts, self.abort))]
            self.builders[name][0].start()
        previous = None
        for name, spec in PIPELINES.items():
            if previous and not self.ready[previous].wait(STALL_SECONDS):
                self.fail(f"{previous} GPU did not finish loading")
            if previous:
                time.sleep(START_GAP_SECONDS)
            self.beats[name].value = time.time()
            self.gpus[name] = self.ctx.Process(target=gpu_worker, args=(
                name, spec, self.inputs[name], self.results, self.beats[name], self.ready[name], self.abort))
            self.gpus[name].start()
            previous = name

    def fail(self, reason: str) -> None:
        print(json.dumps({"FAILED": reason}), flush=True)
        self.abort.set()
        time.sleep(2)
        for process in self.everyone():
            if process.is_alive():
                process.terminate()
        raise SystemExit(1)

    def check(self) -> None:
        if self.abort.is_set():
            self.fail("a worker reported an error")
        for process in self.everyone():
            if process.exitcode not in (None, 0):
                self.fail(f"{process.name} exited with {process.exitcode}")
        for name, process in self.gpus.items():
            if process.is_alive() and time.time() - self.beats[name].value > STALL_SECONDS:
                self.fail(f"{name} GPU stalled")

    def status(self) -> None:
        print(json.dumps({"t": time.strftime("%H:%M:%S"), "saved_batches": len(list(CACHE.glob("*.npz"))),
                          "input_main": self.inputs["main"].qsize(), "input_second": self.inputs["second"].qsize(),
                          "results": self.results.qsize(), "stream": self.stream.qsize()}), flush=True)

    def run(self) -> None:
        self.start()
        closed, results_closed, stream_closed, last = set(), False, False, 0.0
        while self.trainer.is_alive():
            self.check()
            for name, group in self.builders.items():
                if name not in closed and not any(p.is_alive() for p in group):
                    put_or_abort(self.inputs[name], None, self.abort)
                    closed.add(name)
            if not results_closed and len(closed) == len(PIPELINES) and not any(p.is_alive() for p in self.gpus.values()):
                for _ in self.records:
                    put_or_abort(self.results, None, self.abort)
                results_closed = True
            if results_closed and not stream_closed and not any(p.is_alive() for p in self.records):
                put_or_abort(self.stream, None, self.abort)
                stream_closed = True
                try:
                    load_audited(self.rows)
                except ValueError as error:
                    self.fail(f"audit: {error}")
                print(json.dumps({"audit": "passed", "prompts": len(self.rows)}), flush=True)
                self.audit_ok.set()
            if time.time() - last >= STATUS_SECONDS:
                self.status()
                last = time.time()
            time.sleep(0.2)
        self.check()
        if not self.audit_ok.is_set():
            self.fail("trainer stopped before the audit")


def main() -> None:
    if "--fresh" in sys.argv and CACHE.exists():
        shutil.rmtree(CACHE)
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = read_rows()
    plans = plan_batches(rows)
    done = {path.stem for path in CACHE.glob("*.npz")}
    plans = {name: [task for task in plan if task["batch_id"] not in done] for name, plan in plans.items()}
    print(json.dumps({"prompts": len(rows), "batches_to_do": {n: len(p) for n, p in plans.items()},
                      "already_saved": len(done)}), flush=True)
    started = time.time()
    Supervisor(rows, plans).run()
    print(json.dumps({"finished": True, "seconds": round(time.time() - started, 1)}), flush=True)


if __name__ == "__main__":
    main()
