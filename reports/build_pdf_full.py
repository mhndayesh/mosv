"""PDF of the full-length test: every answer in full, normal left and steered right, with blind Haiku scores per criterion.

Run: python reports/build_pdf_full.py data_review/reports/qwen4b_full_answers.pdf
Reads vllm_full_test.json (answers) and full_scores.json (scores). Long answers continue over as many pages as needed.
"""

import json
import sys
from itertools import zip_longest
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import Flowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_pdf_4b import (FRAME_W, HERE, LABELS, MARGIN, MUTED, PAD, THEMES, answer_flowables,  # noqa: E402
                          dropped, footer, grid, header_row, header_table, height, inline, register_fonts, style)

class Mark(Flowable):
    """Invisible marker: a side-tab bookmark and a link target at this point of the document."""

    def __init__(self, title: str, key: str, level: int = 0):
        super().__init__()
        self.title, self.key, self.level = title, key, level

    def wrap(self, *_):
        return 0, 0

    def draw(self):
        pass


class NavDoc(SimpleDocTemplate):
    """Turns every Mark into a bookmark (side tab) and a named destination for internal links."""

    def afterFlowable(self, flowable):
        if isinstance(flowable, Mark):
            self.canv.bookmarkPage(flowable.key)
            self.canv.addOutlineEntry(flowable.title, flowable.key, flowable.level, closed=flowable.level == 0)
            self.canv.showOutline()  # Open the side panel with the bookmarks when the PDF opens.


def link(text: str, key: str) -> str:
    return f'<a href="#{key}"><u>{text}</u></a>'


SIZE = 8.5  # One fixed text size: answers flow over pages instead of shrinking.
CHUNK_PT = 150  # Height of one table row; tables can only break between rows.
GREY = colors.HexColor("#475569")


def chunks(flowables, width):
    """Group an answer's lines into blocks of about CHUNK_PT so the side-by-side table can break across pages."""
    out, block = [], []
    for item in flowables:
        block.append(item)
        if height(block, width) >= CHUNK_PT:
            out.append(block)
            block = []
    return out + ([block] if block else [])


def end_note(reason, tokens):
    return {"stop": f"{tokens:,} tokens, finished", "repeat": f"{tokens:,} tokens, STOPPED: repeating",
            "length": f"{tokens:,} tokens, hit the context limit",
            "manual": f"{tokens:,} tokens, STOPPED by hand: looping"}.get(reason, f"{tokens:,} tokens")


def score_table(score):
    """Criteria down the side, normal and steered scores across, averaged over Haiku's two blind orders."""
    cell, center = style("s", 8, after=0), style("s", 8.5, after=0, alignment=TA_CENTER)
    data = [header_row(["Criterion (Haiku, blind)", "Normal", "Steered", "Change"])]
    for name, n, s in zip(score["criteria"], score["normal"], score["steered"]):
        color = "#047857" if s > n else "#B91C1C" if s < n else "#64748B"
        data.append([Paragraph(name, cell), Paragraph(f"{n:.1f}", center), Paragraph(f"{s:.1f}", center),
                     Paragraph(f'<font color="{color}"><b>{s - n:+.1f}</b></font>', center)])
    data.append([Paragraph("<b>Average of the criteria</b>", cell), Paragraph(f"<b>{score['normal_total']:.2f}</b>", center),
                 Paragraph(f"<b>{score['steered_total']:.2f}</b>", center), Paragraph("", center)])
    return grid(data, [FRAME_W - 3 * 60, 60, 60, 60])


def blind_note(score):
    seen = [c["answer_1_is"] for c in score["calls"]]
    failed = sum("error" in c for c in score["calls"])
    text = f"Haiku judged twice; it saw Answer 1 = {seen[0]} the first time and {seen[1]} the second, never the names."
    return Paragraph(text + (f" {failed} call failed and was left out." if failed else ""),
                     style("b", 8, color=MUTED, after=4))


def answers_table(row):
    route = row["route"]
    main, tint = (colors.HexColor(c) for c in THEMES[route])
    col = FRAME_W / 2
    inner = col - 2 * PAD
    left = chunks(answer_flowables(row["normal"], SIZE, GREY), inner)
    right = chunks(answer_flowables(row["steered"], SIZE, main), inner)
    head = style("colhead", 9.5, "BodyB", colors.white, 0)
    name = "NOT STEERED (same as normal)" if route == "none" else f"STEERED: {LABELS[route]}"
    data = [[Paragraph(f'NORMAL &nbsp; <font name="Body">{end_note(row["normal_end"], row["normal_tokens"])}</font>', head),
             Paragraph(f'{name} &nbsp; <font name="Body">{end_note(row["steered_end"], row["steered_tokens"])}</font>', head)]]
    data += [[a or "", b or ""] for a, b in zip_longest(left, right)]
    table = Table(data, colWidths=[col, col], repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), GREY), ("BACKGROUND", (1, 0), (1, 0), main),
                               ("BACKGROUND", (1, 1), (1, -1), tint), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("BOX", (0, 0), (0, -1), 0.7, GREY), ("BOX", (1, 0), (1, -1), 0.9, main),
                               ("LEFTPADDING", (0, 0), (-1, -1), PAD), ("RIGHTPADDING", (0, 0), (-1, -1), PAD),
                               ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return table


def color_change(change: float) -> str:
    color = "#047857" if change > 0.05 else "#B91C1C" if change < -0.05 else "#64748B"
    return f'<font color="{color}"><b>{change:+.2f}</b></font>'


def topic_key(topic: str) -> str:
    return f"topic_{topic}"


def detail(row, score, first_of_topic: bool):
    marks = [Mark(f"{LABELS[row['type']]} ({row['count']})", topic_key(row["type"]), 0)] if first_of_topic else []
    short = row["prompt"][:60] + ("..." if len(row["prompt"]) > 60 else "")
    marks.append(Mark(f"#{row['id']} {short}", f"p{row['id']}", 1))
    parts = marks + [header_table(row), Spacer(1, 5)]
    if score and score["normal"]:
        parts += [score_table(score), blind_note(score)]
    return parts + [answers_table(row), PageBreak()]


def per_topic(scores) -> list[dict]:
    out = []
    for topic in dict.fromkeys(s["topic"] for s in scores):
        mine = [s for s in scores if s["topic"] == topic]
        n = sum(s["normal_total"] for s in mine) / len(mine)
        t = sum(s["steered_total"] for s in mine) / len(mine)
        wins = sum(s["steered_total"] > s["normal_total"] for s in mine)
        ties = sum(s["steered_total"] == s["normal_total"] for s in mine)
        out.append({"topic": topic, "pairs": len(mine), "normal": n, "steered": t, "wins": wins, "ties": ties,
                    "losses": len(mine) - wins - ties})
    return sorted(out, key=lambda r: r["normal"] - r["steered"])


def topic_table(rows_by_topic) -> Table:
    """One line per topic, best gain first; the topic name jumps to its pages."""
    cell, center = style("a", 10, after=0), style("a", 10, after=0, alignment=TA_CENTER)
    data = [header_row(["Topic", "Pairs", "Normal", "Steered", "Change", "Steered wins / ties / losses"])]
    for r in rows_by_topic:
        name = link(f'<font color="{THEMES[r["topic"]][0]}"><b>{LABELS[r["topic"]]}</b></font>', topic_key(r["topic"]))
        data.append([Paragraph(name, cell), Paragraph(str(r["pairs"]), center), Paragraph(f"{r['normal']:.2f}", center),
                     Paragraph(f"{r['steered']:.2f}", center), Paragraph(color_change(r["steered"] - r["normal"]), center),
                     Paragraph(f"{r['wins']} / {r['ties']} / {r['losses']}", center)])
    return grid(data, [150, 55, 70, 70, 70, FRAME_W - 150 - 55 - 3 * 70])


def criteria_table(scores) -> Table:
    """Per topic: average of each criterion, normal vs steered."""
    cell = style("t", 8.5, after=0)
    data = [header_row(["Topic", "Criterion", "Normal", "Steered", "Change"])]
    for topic in dict.fromkeys(s["topic"] for s in scores):
        mine = [s for s in scores if s["topic"] == topic]
        for i, name in enumerate(mine[0]["criteria"]):
            n = sum(s["normal"][i] for s in mine) / len(mine)
            t = sum(s["steered"][i] for s in mine) / len(mine)
            label = link(f'<font color="{THEMES[topic][0]}"><b>{LABELS[topic]}</b></font>', topic_key(topic)) if i == 0 else ""
            data.append([Paragraph(label, cell), Paragraph(name, cell), Paragraph(f"{n:.2f}", cell),
                         Paragraph(f"{t:.2f}", cell), Paragraph(color_change(t - n), cell)])
    return grid(data, [110, FRAME_W - 110 - 3 * 55, 55, 55, 55], tinted=False)


def abstract_page(scores, stats) -> list:
    topics = [s for s in scores if s["topic"] != "none"]
    n = sum(s["normal_total"] for s in topics) / len(topics)
    t = sum(s["steered_total"] for s in topics) / len(topics)
    wins = sum(s["steered_total"] > s["normal_total"] for s in topics)
    ties = sum(s["steered_total"] == s["normal_total"] for s in topics)
    table = per_topic(topics)
    better = [LABELS[r["topic"]] for r in table if r["steered"] - r["normal"] > 0.3]
    worse = [LABELS[r["topic"]] for r in table if r["steered"] - r["normal"] < -0.05]
    cap, normal_pass = stats["capacity"], stats["normal_pass"]
    body = style("n", 10.5, color=colors.HexColor("#334155"), after=6)
    text = [f"Frozen Qwen3-4B with a router and 8 trained steering vectors. {len(scores)} held-out prompts were answered "
            f"twice in full, once normally and once with the router's vector, and Claude Haiku judged every pair blind.",
            f"<b>Result on the {len(topics)} topic pairs:</b> steered <b>{t:.2f}</b> vs normal <b>{n:.2f}</b> on a 1-5 scale "
            f"(average of 5 criteria). Steered wins {wins}, ties {ties}, loses {len(topics) - wins - ties}.",
            f"<b>Clearly better with the vector:</b> {', '.join(better)}. <b>Worse:</b> {', '.join(worse) or 'none'}.",
            "<b>How it was judged:</b> the two answers were shown as Answer 1 and Answer 2, never named; each pair was "
            "judged twice with the order swapped; Haiku scored every criterion of the topic's rubric separately and "
            "returned strict JSON. The 6 prompts outside the 8 topics got a general rubric; the router left them "
            "unsteered, so both answers are the same text.",
            f"<b>How the answers were written:</b> vLLM on one card, paged fp8 KV cache, {cap['max_model_len']:,}-token "
            f"context, greedy, no length cap; answers that started repeating were stopped and are marked. Normal pass: "
            f"{normal_pass['requests']} answers, {normal_pass['answer_tokens']:,} tokens in {normal_pass['seconds']:.0f} s.",
            "<b>Navigate:</b> use the side bookmarks, or click a topic name below to jump to its answers."]
    return [Mark("Abstract", "abstract", 0), Paragraph("Steered vs normal Qwen3-4B: full answers, blind Haiku scores",
            style("title", 19, "BodyB", after=6)), *[Paragraph(x, body) for x in text], Spacer(1, 6),
            topic_table(table), PageBreak(),
            Mark("Scores by criterion", "criteria", 0),
            Paragraph("Scores by criterion", style("title", 16, "BodyB", after=6)), criteria_table(scores), PageBreak()]


def fmt(value):
    return "n/a" if value is None else f"{value:.2f}"


def index_page(rows, by_id):
    """Every prompt on one list; the prompt text jumps to its page."""
    center, cell = style("i", 8.5, after=0, alignment=TA_CENTER), style("i", 8.5, after=0)
    data = [header_row(["#", "Type", "Prompt (click to open)", "Router", "Normal", "Steered", "Tokens N / S"])]
    for r in rows:
        s = by_id.get(r["id"]) or {}
        prompt = r["prompt"][:90] + ("..." if len(r["prompt"]) > 90 else "")
        data.append([Paragraph(str(r["id"]), center),
                     Paragraph(link(f'<font color="{THEMES[r["type"]][0]}"><b>{LABELS[r["type"]]}</b></font>',
                                    topic_key(r["type"])), cell),
                     Paragraph(link(inline(prompt, "Body"), f"p{r['id']}"), cell),
                     Paragraph("off" if r["route"] == "none" else LABELS[r["route"]], center),
                     Paragraph(fmt(s.get("normal_total")), center), Paragraph(fmt(s.get("steered_total")), center),
                     Paragraph(f'{r["normal_tokens"]:,} / {r["steered_tokens"]:,}', center)])
    return [Mark("All prompts", "index", 0), grid(data, [24, 80, FRAME_W - 24 - 80 - 80 - 48 - 48 - 80, 80, 48, 48, 80]),
            PageBreak()]


def main():
    rows = json.loads((HERE / "vllm_full_test.json").read_text(encoding="utf-8"))
    scores = json.loads((HERE / "full_scores.json").read_text(encoding="utf-8"))
    stats = json.loads((HERE / "vllm_full_test.stats.json").read_text(encoding="utf-8"))
    by_id = {s["id"]: s for s in scores}
    counts = {t: sum(r["type"] == t for r in rows) for t in {r["type"] for r in rows}}
    rows = sorted(rows, key=lambda r: (list(dict.fromkeys(x["type"] for x in rows)).index(r["type"]), r["id"]))
    out = Path(sys.argv[1])
    register_fonts()
    story = abstract_page(scores, stats) + index_page(rows, by_id)
    seen = set()
    for row in rows:
        story += detail({**row, "count": counts[row["type"]]}, by_id.get(row["id"]), row["type"] not in seen)
        seen.add(row["type"])
    story.pop()
    doc = NavDoc(str(out), pagesize=landscape(A4), leftMargin=MARGIN, rightMargin=MARGIN,
                 topMargin=MARGIN, bottomMargin=MARGIN + 14, title="Qwen3-4B full answers")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(json.dumps({"pdf": str(out), "pages": doc.page, "symbols_not_drawable": len(dropped)}))


if __name__ == "__main__":
    main()
