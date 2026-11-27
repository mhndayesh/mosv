"""Charts for the Reddit post (static PNGs, light surface). Numbers are copied from results/ and docs/."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"
BASE, TRAINED = "#b3b3b3", "#257de0"   # neutral gray for the reference, series-1 blue for the result
plt.rcParams.update({"font.family": "Segoe UI", "font.size": 12, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID})


def frame(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


# 1. Gemma: base vs trained vs told ceiling
topics = ["Teaching", "Story", "Debugging"]
base, trained, told = [0.58, 0.73, 0.17], [0.97, 0.81, 0.44], [0.99, 0.83, 0.58]  # locked test; told = dev gate
fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
x, w = range(len(topics)), 0.34
for i in x:
    ax.bar(i - w / 2 - 0.01, base[i], w, color=BASE)
    ax.bar(i + w / 2 + 0.01, trained[i], w, color=TRAINED)
    ax.text(i - w / 2 - 0.01, base[i] + 0.02, f"{base[i]:.2f}", ha="center", color=INK2, fontsize=11)
    ax.text(i + w / 2 + 0.01, trained[i] - 0.06, f"{trained[i]:.2f}", ha="center", color="white", fontsize=11, weight="bold")  # inside the bar: the ceiling line sits just above
    ax.plot([i - w - 0.06, i + w + 0.06], [told[i]] * 2, color=INK, linestyle=(0, (4, 3)), linewidth=1.5)
ax.text(2 + w + 0.08, told[2], "ceiling: model\ntold the rules", va="center", color=INK2, fontsize=10)
ax.set_xticks(list(x), topics)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Share of behavior checks passed")
frame(ax)
ax.bar(0, 0, color=BASE, label="Base Gemma 4 12B (plain prompt)")
ax.bar(0, 0, color=TRAINED, label="With trained experts (plain prompt)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, frameon=False, fontsize=11)
ax.set_title("Gemma 4 12B: behavior learned without being told", loc="left", pad=34, fontsize=14, weight="bold")
fig.text(0.01, 0.01, "Locked test: 43 prompts per topic never used in training, judged by Claude Haiku with yes/no checks. Ceiling measured on dev prompts.",
         color=INK2, fontsize=9)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUT / "1_gemma_results.png")

# 2. Why RL failed on Gemma: judge agreement with itself
fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
names, vals = ["Bonsai-4B\n(RL worked)", "Gemma 4 12B\n(RL failed)"], [33.4, 25.5]
ax.bar([0, 1], vals, 0.5, color=[TRAINED, BASE])
for i, v in enumerate(vals):
    ax.text(i, v + 0.8, f"{v}%", ha="center", weight="bold")
ax.axhline(25, color=INK, linestyle=(0, (4, 3)), linewidth=1.5)
ax.text(1.74, 25.6, "pure chance (1 in 4)", ha="right", va="bottom", color=INK2, fontsize=10)
ax.set_xticks([0, 1], names)
ax.set_xlim(-0.5, 1.75)
ax.set_ylim(0, 40)
ax.set_ylabel("Same winner picked twice")
frame(ax)
ax.set_title("Same 4 answers judged twice: does the judge agree with itself?", loc="left", fontsize=13, weight="bold")
fig.tight_layout()
fig.savefig(OUT / "2_rl_reward_noise.png")

# 3. Results table across models
rows = [["Model", "Method", "Before", "After", "Note"],
        ["Qwen3-4B", "RL, vectors then MoE GGUF", "2.89", "3.61", "blind judge score, 1-5"],
        ["Bonsai-4B (1-bit)", "RL into the GGUF", "1.95", "2.42", "better on 66% of 300 locked prompts"],
        ["Gemma 4 12B", "same RL", "3.39", "3.30-3.41", "no gain"],
        ["Gemma 4 12B", "tell, check, train", "see chart", "see chart", "behavior wins: teaching 98%, debugging 81%, story 62%"]]
fig, ax = plt.subplots(figsize=(10, 2.6), dpi=150)
ax.axis("off")
t = ax.table(cellText=rows, loc="center", cellLoc="left", colWidths=[0.18, 0.26, 0.1, 0.12, 0.34])
t.auto_set_font_size(False)
t.set_fontsize(11)
t.scale(1, 1.7)
for (r, c), cell in t.get_celld().items():
    cell.set_edgecolor(GRID)
    if r == 0:
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#f4f4f4")
fig.tight_layout()
fig.savefig(OUT / "3_results_table.png", bbox_inches="tight")
print("saved", sorted(p.name for p in OUT.glob("*.png")))
