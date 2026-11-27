"""Table images for the speed test (reads results/speed_v6_0_1/speed.json, written by speed_v6_0_1.py).

Run (any venv with matplotlib, repo root):  python reports/speed_chart_v6_0_1.py
Writes results/speed_v6_0_1/speed_summary.png and speed_per_prompt.png. Works on a partial run (only the card
setups already finished are drawn). Same look as docs/reddit/make_charts.py.
"""
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "results" / "speed_v6_0_1"
INK, INK2, GRID, HEAD, BLUE = "#191919", "#666666", "#e6e6e6", "#f4f4f4", "#257de0"
plt.rcParams.update({"font.family": "Segoe UI", "font.size": 11, "text.color": INK})
NAMES = {"base": "Base Gemma 4 12B", "mosv": "MoSV v6.0.1"}


def draw(rows, title, note, path, bold_rows=()):
    """One clean table image: each column as wide as its longest line, header shaded, `bold_rows` in blue bold."""
    chars = [max(len(line) for row in rows for line in row[c].split("\n")) + 2 for c in range(len(rows[0]))]
    fig, ax = plt.subplots(figsize=(max(0.105 * sum(chars), 9), 0.42 * len(rows) + 1.3), dpi=200)
    ax.axis("off")
    t = ax.table(cellText=rows, loc="center", cellLoc="center", colWidths=[c / sum(chars) for c in chars])
    t.auto_set_font_size(False)
    t.set_fontsize(10.5)
    t.scale(1, 1.6)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor(HEAD)
            cell.set_height(cell.get_height() * 1.6)  # two-line headers
        elif r in bold_rows:
            cell.set_text_props(weight="bold", color=BLUE)
    ax.set_title(title, loc="left", fontsize=13, weight="bold")
    fig.text(0.01, 0.01, note, color=INK2, fontsize=8.5, wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def pct(new, old):
    return f"{100 * (new / old - 1):+.0f}%"


def summary(data, setups):
    """Per setup: llama-bench speeds, server speed at fixed length, natural answer length and time."""
    req, bench = data["requests"], data["bench"]
    head = ["Cards", "Model", "Generation tok/s\n(llama-bench tg128)", "Prompt tok/s\n(llama-bench pp512)",
            "Generation tok/s\n(server, 512 tok)", "Answer length\n(avg tokens)", "Time per answer\n(avg s)"]
    rows, diff_rows = [head], []
    for setup in setups:
        vals = {}
        for m in NAMES:
            b = {x["test"]: x["tps"] for x in bench if x["setup"] == setup and x["model"] == m}
            fx = [x for x in req if x["setup"] == setup and x["model"] == m and x["mode"] == "fixed"]
            nt = [x for x in req if x["setup"] == setup and x["model"] == m and x["mode"] == "natural"]
            vals[m] = [b.get("tg128"), b.get("pp512"), statistics.mean(x["gen_tps"] for x in fx),
                       statistics.mean(x["gen_tokens"] for x in nt), statistics.mean(x["wall_s"] for x in nt)]
            rows.append([setup, NAMES[m]] + [f"{v:.1f}" if v is not None else "-" for v in vals[m]])
        diff_rows.append(len(rows))
        rows.append([setup, "MoSV vs base"] + [pct(n, o) if n and o else "-" for n, o in zip(vals["mosv"], vals["base"])])
    return rows, diff_rows


def per_prompt(data, setups):
    """Per prompt (natural answers): answer length and time for both models, per setup."""
    req, prompts = data["requests"], data["setup"]["prompts"]
    head = ["#", "Prompt kind"]
    for s in setups:
        head += [f"{s}\nbase tokens", f"{s}\nMoSV tokens", f"{s}\nbase seconds", f"{s}\nMoSV seconds"]
    rows, cols = [head], {s: [[], [], [], []] for s in setups}
    for i, (kind, _) in enumerate(prompts):
        row = [str(i + 1), kind]
        for s in setups:
            b, m = [next(x for x in req if x["setup"] == s and x["model"] == mo and x["mode"] == "natural"
                         and x["prompt_id"] == i) for mo in NAMES]
            vals = [b["gen_tokens"], m["gen_tokens"], b["wall_s"], m["wall_s"]]
            for col, v in zip(cols[s], vals):
                col.append(v)
            row += [str(vals[0]), str(vals[1]), f"{vals[2]:.1f}", f"{vals[3]:.1f}"]
        rows.append(row)
    avg = ["", "average"]
    for s in setups:
        bt, mt, bs, ms = (statistics.mean(c) for c in cols[s])
        avg += [f"{bt:.0f}", f"{mt:.0f} ({pct(mt, bt)})", f"{bs:.1f}", f"{ms:.1f} ({pct(ms, bs)})"]
    return rows + [avg]


def headline(data):
    """The short one-card table: speed the same, answers shorter and sooner."""
    req, bench = data["requests"], data["bench"]
    b = {(x["model"], x["test"]): x["tps"] for x in bench if x["setup"] == "single"}
    nat = {m: [x for x in req if x["setup"] == "single" and x["model"] == m and x["mode"] == "natural"] for m in NAMES}
    tok = {m: statistics.mean(x["gen_tokens"] for x in nat[m]) for m in NAMES}
    sec = {m: statistics.mean(x["wall_s"] for x in nat[m]) for m in NAMES}
    return [["One RX 9070 XT", "Base Gemma 4 12B", "MoSV v6.0.1", "Change"],
            ["Generation speed (llama-bench)", f"{b['base', 'tg128']:.1f} tok/s", f"{b['mosv', 'tg128']:.1f} tok/s",
             "same (within noise)"],
            ["Prompt reading (llama-bench)", f"{b['base', 'pp512']:,.0f} tok/s", f"{b['mosv', 'pp512']:,.0f} tok/s", "same"],
            ["Answer length (10 prompts)", f"{tok['base']:.0f} tokens", f"{tok['mosv']:.0f} tokens",
             f"{100 * (1 - tok['mosv'] / tok['base']):.0f}% shorter"],
            ["Time per answer", f"{sec['base']:.1f} s", f"{sec['mosv']:.1f} s",
             f"{100 * (1 - sec['mosv'] / sec['base']):.0f}% sooner"]]


def locked_lengths():
    """Answer length on the 300-prompt locked test (results/v6_0_1_eval_test_*), the larger sample."""
    test = json.loads((OUT.parent / "v6_0_1_eval_test_gemma_v601_steer_0100.json").read_text(encoding="utf-8"))
    trained = ("teaching", "debugging", "story")
    rows, total = [["Topic", "Prompts", "Base tokens", "MoSV tokens", "Change", "Behavior checks\nbase → MoSV"]], [0, 0, 0]
    for topic in sorted(test["topics"], key=lambda t: (t not in trained, t)):
        x = test["topics"][topic]
        b, m = x["base"], x["trained"]
        name = topic.replace("_", " ") + ("" if topic in trained else " (not trained)")
        rows.append([name, str(b["n"]), str(b["tokens"]), str(m["tokens"]), pct(m["tokens"], b["tokens"]),
                     f"{b['behavior']:.2f} → {m['behavior']:.2f}"])
        for i, v in enumerate((b["n"], b["n"] * b["tokens"], b["n"] * m["tokens"])):
            total[i] += v
    bt, mt = total[1] / total[0], total[2] / total[0]
    rows.append(["all topics", str(total[0]), f"{bt:.0f}", f"{mt:.0f}", pct(mt, bt), ""])
    return rows


def main():
    data = json.loads((OUT / "speed.json").read_text(encoding="utf-8"))
    setups = [s for s, _, _ in data["setup"]["setups"] if any(b["setup"] == s for b in data["bench"])]
    note = ("Same llama.cpp build, flags, prompts and seed; one model loaded at a time; Gemma's recommended sampling "
            "(temp 1.0, top_k 64, top_p 0.95, min_p 0); thinking off.\nDual = 2x RX 9070 XT, layer split; single = 1 card. "
            "Server speed: 10 prompts x 2 sessions, 512 tokens each. Answer length and time: natural answers, 10 prompts.")
    rows, diff_rows = summary(data, setups)
    draw(rows, "Speed: base Gemma 4 12B vs MoSV v6.0.1", note, OUT / "speed_summary.png", diff_rows)
    pp = per_prompt(data, setups)
    draw(pp, "Answer length and time per prompt (natural answers)", note, OUT / "speed_per_prompt.png", [len(pp) - 1])
    if "single" in setups:
        draw(headline(data), "MoSV v6.0.1: same speed, shorter answers", note, OUT / "speed_one_card.png", [3, 4])
    lt = locked_lengths()
    draw(lt, "Answer length on the locked test (prompts never used in training)",
         "Plain prompts, no instructions, answered by the shipped GGUF in llama.cpp with Gemma's recommended sampling; "
         "average tokens per answer.\nBehavior checks judged by Claude Haiku with yes/no questions "
         "(results/v6_0_1_eval_test_gemma_v601_steer_0100.json).", OUT / "answer_length_locked_test.png", [len(lt) - 1])
    print("saved", sorted(p.name for p in OUT.glob("*.png")), "setups:", setups)


if __name__ == "__main__":
    main()
