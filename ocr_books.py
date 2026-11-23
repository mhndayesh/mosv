"""OCR scanned PDF books to text with PaddleOCR-VL, one book at a time.

Run: ocr-rocm python ocr_books.py [--pages 5]
Uses llama.cpp on Vulkan: measured 0.93 s/page vs about 25 s/page in ROCm torch.
One page at a time per card was fastest (65 pages/min vs 21-48 when concurrent).
Each finished page is saved at once, so a stopped run resumes where it left off.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import multiprocessing as mp
import os
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

SERVER = Path(r"C:\llm\tq-vulkan\build\bin\llama-server.exe")
MODEL_DIR = Path(r"C:\llm\models\paddleocr-vl-1.6")
MODEL = MODEL_DIR / "PaddleOCR-VL-1.6-GGUF.gguf"
VISION = MODEL_DIR / "PaddleOCR-VL-1.6-GGUF-mmproj.gguf"
ROOT = Path(__file__).parent / "data_review"
BASE_PORT = 8091
RENDER_AHEAD = 8
REQUEST_TIMEOUT = 600
START_GAP_SECONDS = 20  # Never start both cards at the same moment.


def log(gpu: str, message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} [gpu {gpu}] {message}", flush=True)


def page_path(book_dir: Path, number: int) -> Path:
    return book_dir / f"page_{number:05d}.txt"


def save_text(path: Path, text: str) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def looks_repeated(text: str) -> bool:
    """Catch the common OCR failure where the model loops on one phrase."""
    return re.search(r"(.{8,80}?)\1{5,}", text, flags=re.DOTALL) is not None


def render_pages(pdf: Path, numbers: list[int], dpi: int, out: queue.Queue) -> None:
    """Render and encode on a CPU thread so the GPU never waits for a page."""
    import pypdfium2

    document = pypdfium2.PdfDocument(pdf)
    try:
        for number in numbers:
            image = document[number - 1].render(scale=dpi / 72).to_pil().convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, "PNG", compress_level=1)
            out.put((number, base64.b64encode(buffer.getvalue()).decode()))
    except Exception as error:
        out.put((None, error))
    finally:
        document.close()
        out.put(None)


def read_page(port: int, image_b64: str, max_tokens: int) -> tuple[str, bool]:
    body = json.dumps({
        "model": "ocr", "temperature": 0, "max_tokens": max_tokens, "cache_prompt": False,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            {"type": "text", "text": "OCR:"},
        ]}],
    }).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", body,
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        choice = json.load(response)["choices"][0]
    return choice["message"]["content"].strip(), choice["finish_reason"] == "length"


def merge_book(book_dir: Path, target: Path, count: int) -> None:
    parts = []
    for number in range(1, count + 1):
        path = page_path(book_dir, number)
        body = path.read_text(encoding="utf-8") if path.exists() else "[PAGE FAILED]"
        parts.append(f"<!-- page {number} -->\n{body}")
    save_text(target, "\n\n".join(parts) + "\n")


def ocr_book(pdf: Path, out_root: Path, port: int, gpu: str, args) -> None:
    import pypdfium2

    book_dir = out_root / pdf.stem
    book_dir.mkdir(parents=True, exist_ok=True)
    document = pypdfium2.PdfDocument(pdf)
    total = len(document)
    document.close()
    count = min(total, args.pages) if args.pages else total
    todo = [n for n in range(1, count + 1) if not page_path(book_dir, n).exists()]
    log(gpu, f"{pdf.name}: {count} pages, {len(todo)} to do")

    pages: queue.Queue = queue.Queue(maxsize=RENDER_AHEAD)
    threading.Thread(target=render_pages, args=(pdf, todo, args.dpi, pages), daemon=True).start()
    started, done = time.time(), 0
    while (item := pages.get()) is not None:
        number, image = item
        if number is None:
            raise RuntimeError(f"Could not render {pdf.name}: {image}")
        text, cut_off = read_page(port, image, args.max_tokens)
        if cut_off or looks_repeated(text):
            reason = "hit token limit" if cut_off else "repeated text"
            with (book_dir / "issues.txt").open("a", encoding="utf-8") as issues:
                issues.write(f"page {number}: {reason}\n")
            log(gpu, f"{pdf.name} page {number}: {reason}")
        save_text(page_path(book_dir, number), text)
        done += 1
        if done == 1 or done % 10 == 0 or done == len(todo):
            rate = (time.time() - started) / done
            log(gpu, f"{pdf.name}: {done}/{len(todo)} pages, {rate:.2f} s/page")
    merge_book(book_dir, out_root / f"{pdf.stem}.md", count)
    log(gpu, f"{pdf.name}: finished")


def start_server(gpu: str, port: int, log_path: Path, context: int) -> subprocess.Popen:
    command = [
        str(SERVER), "-m", str(MODEL), "--mmproj", str(VISION),
        "--host", "127.0.0.1", "--port", str(port), "-dev", f"Vulkan{gpu}",
        "-ngl", "999", "-c", str(context), "-np", "1",
        "-b", "2048", "-ub", "2048", "-fa", "on", "--temp", "0", "--no-webui",
    ]
    with log_path.open("w", encoding="utf-8") as output:
        server = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
    for _ in range(240):
        if server.poll() is not None:
            raise RuntimeError(f"Server on Vulkan{gpu} exited; see {log_path}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as reply:
                if json.load(reply).get("status") == "ok":
                    return server
        except OSError:
            pass
        time.sleep(0.5)
    server.kill()
    raise RuntimeError(f"Server on Vulkan{gpu} never became ready; see {log_path}")


def assign_books(pdfs: list[Path], gpus: list[str]) -> dict[str, list[Path]]:
    """Fixed split: earlier cards get the extra book, so 5 books on 2 cards = 3 + 2."""
    share, extra = divmod(len(pdfs), len(gpus))
    assigned, start = {}, 0
    for index, gpu in enumerate(gpus):
        end = start + share + (index < extra)
        assigned[gpu] = pdfs[start:end]
        start = end
    return assigned


def worker(gpu: str, port: int, books: list[Path], delay: int, out_root: Path, args) -> None:
    """Runs one card alone: its own server, its own books, nothing shared."""
    time.sleep(delay)
    server = start_server(gpu, port, out_root / f"server_gpu{gpu}.log", args.max_tokens + 4096)
    log(gpu, f"server ready on Vulkan{gpu}, port {port}, books: {[b.name for b in books]}")
    failed = 0
    try:
        for pdf in books:
            try:
                ocr_book(pdf, out_root, port, gpu, args)
            except Exception as error:
                failed += 1
                log(gpu, f"{pdf.name} FAILED: {error!r}")
    finally:
        server.terminate()
        server.wait(timeout=30)
    if failed:
        sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--books", type=Path, default=ROOT / "books")
    parser.add_argument("--out", type=Path, default=ROOT / "books_text")
    # One card only: starting OCR on both at once froze this PC on 2026-11-23.
    parser.add_argument("--gpus", default="0", help="Vulkan index; do not use both cards")
    parser.add_argument("--pages", type=int, default=0, help="only the first N pages (0 = all)")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()

    pdfs = sorted(args.books.glob("*.pdf"))
    if not pdfs:
        print(f"No PDF files in {args.books}")
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    gpus = [g.strip() for g in args.gpus.split(",")][:len(pdfs)]
    assigned = assign_books(pdfs, gpus)
    workers = [
        context.Process(target=worker, args=(
            g, BASE_PORT + i, assigned[g], i * START_GAP_SECONDS, args.out, args,
        ))
        for i, g in enumerate(gpus)
    ]
    for process in workers:
        process.start()
    for process in workers:
        process.join()
    return max(process.exitcode or 0 for process in workers)


if __name__ == "__main__":
    raise SystemExit(main())
