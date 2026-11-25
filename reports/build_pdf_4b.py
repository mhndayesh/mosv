import json
import re
import sys
from html import escape
from io import BytesIO
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import HRFlowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = Path(__file__).resolve().parents[1] / "data_review" / "reports"
FONTS = Path("C:/Windows/Fonts")
PAGE_W, PAGE_H = landscape(A4)
MARGIN = 24
FRAME_W = PAGE_W - 2 * MARGIN
FRAME_H = PAGE_H - 2 * MARGIN - 14 - 12
HEADER_H = 24
PAD = 9
SIZES = (10.5, 10, 9.5, 9, 8.5, 8, 7.5, 7, 6.5, 6)
INK, MUTED, LINE = colors.HexColor("#0F172A"), colors.HexColor("#64748B"), colors.HexColor("#CBD5E1")
THEMES = {"engineering": ("#1D4ED8", "#EFF6FF"), "story": ("#7C3AED", "#F5F3FF"),
          "teaching": ("#047857", "#ECFDF5"), "debugging": ("#B91C1C", "#FEF2F2"), "math_solving": ("#B45309", "#FFFBEB"), "explanation": ("#0E7490", "#ECFEFF"), "professional_email": ("#BE185D", "#FDF2F8"), "how_to_guide": ("#4D7C0F", "#F7FEE7"), "none": ("#475569", "#F1F5F9")}
LABELS = {"engineering": "Engineering", "story": "Story", "teaching": "Teaching", "debugging": "Debugging", "math_solving": "Math solving", "explanation": "Explanation", "professional_email": "Professional email", "how_to_guide": "How-to guide", "none": "Normal prompt"}
TREE = re.compile("[├└│┌┐┘┴┬┼─]")
dropped = set()
page_offset = 0


def register_fonts():
    for name, file in (("Body", "segoeui.ttf"), ("BodyB", "segoeuib.ttf"), ("Mono", "consola.ttf"), ("MonoB", "consolab.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(FONTS / file)))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="BodyB", italic="Body", boldItalic="BodyB")
    pdfmetrics.registerFontFamily("Mono", normal="Mono", bold="MonoB", italic="Mono", boldItalic="MonoB")


def keep_drawable(text, font):
    glyphs = pdfmetrics.getFont(font).face.charToGlyph
    dropped.update(c for c in text if ord(c) not in glyphs and c not in "\n\t")
    return "".join(c for c in text if ord(c) in glyphs or c in "\n\t")


def inline(line, font):
    text = escape(keep_drawable(line, font), quote=False)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text).replace("**", "")
    return re.sub(r"`([^`]+)`", r'<font name="Mono">\1</font>', text)


def style(name, size, font="Body", color=INK, after=1.5, **extra):
    return ParagraphStyle(name, fontName=font, fontSize=size, leading=size * 1.32, textColor=color,
                          spaceAfter=after, splitLongWords=1, **extra)


def mono_line(line, size):
    body = line.rstrip()
    lead = len(body) - len(body.lstrip(" "))
    text = "&nbsp;" * lead + escape(keep_drawable(body.lstrip(" "), "Mono"), quote=False)
    return Paragraph(text or "&nbsp;", style("code", size * 0.94, "Mono", colors.HexColor("#1E293B"), 0))


def line_flowable(line, size, accent):
    stripped = line.strip()
    if not stripped:
        return Spacer(1, size * 0.45)
    if re.fullmatch(r"[-*_]{3,}", stripped):
        return HRFlowable(width="100%", thickness=0.5, color=LINE, spaceBefore=2, spaceAfter=3)
    heading = re.match(r"#{1,6}\s+(.*)", stripped)
    if heading:
        return Paragraph(inline(heading.group(1), "BodyB").replace("<b>", "").replace("</b>", ""),
                         style("head", size + 0.7, "BodyB", accent, 2))
    return Paragraph(inline(line.strip(), "Body"), style("body", size))


def answer_flowables(text, size, accent):
    out, code = [], False
    for line in (text or "(empty answer)").split("\n"):
        if line.strip().startswith("```"):
            code = not code
            out.append(Spacer(1, size * 0.25))
        elif code or TREE.search(line):
            out.append(mono_line(line, size))
        else:
            out.append(line_flowable(line, size, accent))
    return out


def height(flowables, width):
    total = 0
    for item in flowables:
        total += item.wrap(width, 10000)[1] + item.getSpaceAfter() + item.getSpaceBefore()
    return total


def trim(flowables, width, room, size):
    """Last resort at the smallest size: drop lines from the end and say so, so one page still holds the answer."""
    note = Paragraph("<i>[... rest cut to fit the page]</i>", style("cut", size, color=MUTED, after=0))
    while flowables and height(flowables + [note], width) > room:
        flowables = flowables[:-1]
    return flowables + [note]


def fit(left, right, width, room, accents):
    for size in SIZES:
        a, b = answer_flowables(left, size, accents[0]), answer_flowables(right, size, accents[1])
        if max(height(a, width), height(b, width)) <= room:
            return a, b, size
    a = a if height(a, width) <= room else trim(a, width, room, size)
    b = b if height(b, width) <= room else trim(b, width, room, size)
    return a, b, size


def score_text(value):
    return "n/a" if value is None else f"{value:.2f}"


def header_table(row):
    color = colors.HexColor(THEMES[row["type"]][0])
    route = row["route"]
    route_text = "No steering (router stayed off)" if route == "none" else f"{LABELS[route]} vector"
    top = Paragraph(f'<b>#{row["id"]} &nbsp; {LABELS[row["type"]].upper()}</b> &nbsp;&nbsp;|&nbsp;&nbsp; '
                    f'Router: <b>{route_text}</b> &nbsp;({row["confidence"]:.0%} sure)',
                    style("top", 10, "Body", colors.white, 0))
    prompt = Paragraph(f'<b>Prompt:</b> {inline(row["prompt"], "Body")}', style("prompt", 10.5, after=0))
    table = Table([[top], [prompt]], colWidths=[FRAME_W])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), color), ("BACKGROUND", (0, 1), (0, 1), colors.white),
                               ("BOX", (0, 0), (-1, -1), 0.8, color), ("LEFTPADDING", (0, 0), (-1, -1), PAD),
                               ("RIGHTPADDING", (0, 0), (-1, -1), PAD), ("TOPPADDING", (0, 0), (-1, -1), 4),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return table


def compare_table(row, room):
    main, tint = (colors.HexColor(c) for c in THEMES[row["route"]])
    grey = colors.HexColor("#475569")
    col = FRAME_W / 2
    inner = col - 2 * PAD
    left, right, _ = fit(row["normal"], row["steered"], inner, room - HEADER_H - 12, (grey, main))
    steered_name = "NOT STEERED (same as normal)" if row["route"] == "none" else f'STEERED: {LABELS[row["route"]]}'
    head = style("colhead", 10, "BodyB", colors.white, 0)
    table = Table([[Paragraph(f'NORMAL &nbsp; <font name="Body">checklist {score_text(row["normal_score"])}</font>', head),
                    Paragraph(f'{steered_name} &nbsp; <font name="Body">checklist {score_text(row["steered_score"])}</font>', head)],
                   [left, right]], colWidths=[col, col], rowHeights=[HEADER_H, room - HEADER_H])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), grey), ("BACKGROUND", (1, 0), (1, 0), main),
                               ("BACKGROUND", (1, 1), (1, 1), tint), ("VALIGN", (0, 0), (-1, 0), "MIDDLE"),
                               ("VALIGN", (0, 1), (-1, 1), "TOP"), ("BOX", (0, 0), (0, -1), 0.7, grey),
                               ("BOX", (1, 0), (1, -1), 0.9, main), ("LEFTPADDING", (0, 0), (-1, -1), PAD),
                               ("RIGHTPADDING", (0, 0), (-1, -1), PAD), ("TOPPADDING", (0, 1), (-1, 1), 6),
                               ("BOTTOMPADDING", (0, 1), (-1, 1), 4)]))
    return table


def detail_page(row):
    head = header_table(row)
    used = head.wrap(FRAME_W, FRAME_H)[1]
    return [head, Spacer(1, 6), compare_table(row, FRAME_H - used - 6), PageBreak()]


def change_cell(row):
    if row["normal_score"] is None:
        return Paragraph("unchanged", style("c", 8.5, color=MUTED, after=0, alignment=TA_CENTER))
    change = row["steered_score"] - row["normal_score"]
    color = "#047857" if change > 0.005 else "#B91C1C" if change < -0.005 else "#64748B"
    return Paragraph(f'<font color="{color}"><b>{change:+.2f}</b></font>', style("c", 9, after=0, alignment=TA_CENTER))


def index_row(row):
    center = style("i", 8.5, after=0, alignment=TA_CENTER)
    color = THEMES[row["type"]][0]
    prompt = row["prompt"] if len(row["prompt"]) <= 84 else row["prompt"][:81] + "..."
    route = "off" if row["route"] == "none" else LABELS[row["route"]]
    return [Paragraph(str(row["id"]), center), Paragraph(f'<font color="{color}"><b>{LABELS[row["type"]]}</b></font>', style("i", 8.5, after=0)),
            Paragraph(inline(prompt, "Body"), style("i", 8.5, after=0)), Paragraph(route, center),
            Paragraph(score_text(row["normal_score"]), center), Paragraph(score_text(row["steered_score"]), center),
            change_cell(row), Paragraph(str(row["id"] + page_offset), center)]


def index_page(rows):
    scored = [r for r in rows if r["normal_score"] is not None]
    normal = sum(r["normal_score"] for r in scored) / len(scored)
    steered = sum(r["steered_score"] for r in scored) / len(scored)
    title = Paragraph("Steered vs normal answers: Qwen3-4B", style("title", 19, "BodyB", after=3))
    note = Paragraph(f"{len(rows)} held-out prompts picked at random, not hand-chosen. Each has its own page: <b>normal on the left, steered on the right</b>. "
                     f"Answers stop at 256 tokens, so many end mid-sentence. The checklist scores structure, not truth or quality. "
                     f"Average on the {len(scored)} behavior prompts: <b>normal {normal:.2f}, steered {steered:.2f}</b>.",
                     style("note", 9.5, color=colors.HexColor("#334155"), after=7))
    head = style("h", 8.5, "BodyB", colors.white, 0, alignment=TA_CENTER)
    labels = ["#", "Type", "Prompt", "Router", "Normal", "Steered", "Change", "Page"]
    data = [[Paragraph(x, head) for x in labels]] + [index_row(r) for r in rows]
    widths = [24, 78, FRAME_W - 24 - 78 - 74 - 48 - 48 - 56 - 34, 74, 48, 48, 56, 34]
    commands = [("BACKGROUND", (0, 0), (-1, 0), INK), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LINEBELOW", (0, 0), (-1, -1), 0.35, LINE), ("TOPPADDING", (0, 0), (-1, -1), 2.2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.8)]
    commands += [("BACKGROUND", (0, i + 1), (-1, i + 1), colors.HexColor(THEMES[r["type"]][1]))
                 for i, r in enumerate(rows) if i % 2 == 0]
    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle(commands))
    return [title, note, table, PageBreak()]


def grid(data, widths, tinted=True):
    commands = [("BACKGROUND", (0, 0), (-1, 0), INK), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEBELOW", (0, 0), (-1, -1), 0.35, LINE), ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5)]
    if tinted:
        commands += [("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F8FAFC")) for i in range(2, len(data), 2)]
    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle(commands))
    return table


def header_row(labels):
    head = style("h", 9, "BodyB", colors.white, 0)
    return [Paragraph(label, head) for label in labels]


def router_summary(rows):
    topic = [r for r in rows if r["type"] != "none"]
    right = sum(r["route"] == r["type"] for r in topic)
    missed = sum(r["route"] == "none" for r in topic)
    plain = [r for r in rows if r["type"] == "none"]
    wrongly = sum(r["route"] != "none" for r in plain)
    return (f"<b>Router in this test:</b> {right} of {len(topic)} topic prompts got their own vector, "
            f"{len(topic) - right - missed} got another topic's vector, {missed} got no steering. "
            f"{wrongly} of {len(plain)} normal prompts were steered by mistake.")


def summary_cell(item, cell):
    final = item["final"]
    if not final:
        return [Paragraph("not run (stopped at ceiling)", style("m", 9.5, color=MUTED, after=0))] + [Paragraph("", cell)] * 3
    return [Paragraph(f'{final["test_base"]:.2f}', cell), Paragraph(f'{final["test_random"]:.2f}', cell),
            Paragraph(f'<b>{final["test_steered"]:.2f}</b>', cell), Paragraph(f'{final["beats_base_pct"]:.0f}%', cell)]


def results_page(rows):
    summary = json.loads((HERE / "haiku_v3_summary.json").read_text(encoding="utf-8"))
    summary.sort(key=lambda s: -s["dev"])
    cell = style("r", 10.5, after=0)
    data = [header_row(["Topic", "Vector step", "Dev score (Haiku)", "Final test: normal", "Random vector",
                        "Trained vector", "Beats normal"])]
    for item in summary:
        data.append([Paragraph(f'<font color="{THEMES[item["topic"]][0]}"><b>{LABELS[item["topic"]]}</b></font>', cell),
                     Paragraph(str(item["step"]), cell), Paragraph(f'<b>{item["dev"]:.2f}</b>', cell),
                     *summary_cell(item, cell)])
    notes = ["<b>Dev score:</b> Claude Haiku compares the steered answer with normal Qwen's answer on 128 held-out prompts, "
             "twice with the order swapped. 0.50 is a tie; above 0.50 means the vector helps.",
             "<b>Vector step:</b> the best of each topic's last 3 dev checks, not the best score overall, because the judge is not fully trusted.",
             "<b>Final test:</b> Haiku scores on held-out test prompts, 0 to 1. Only the 3 topics that trained to the end ran it.",
             "<b>How-to guide:</b> its vector is worse than normal Qwen (dev 0.48). It is kept here so you can see it.",
             router_summary(rows),
             "<b>Scores on the answer pages</b> come from the topic's local checklist (structure and process), not from Haiku."]
    body = style("n", 10, color=colors.HexColor("#334155"), after=5)
    return [Paragraph("Qwen3-4B with router and 8 steering vectors", style("title", 20, "BodyB", after=4)),
            Paragraph("Frozen model. Vectors trained with RL and a Claude Haiku judge. All prompts held out from training.",
                      style("sub", 11, color=MUTED, after=9)),
            grid(data, [FRAME_W * share for share in (0.2, 0.1, 0.13, 0.15, 0.13, 0.15, 0.14)]), Spacer(1, 11),
            *[Paragraph(note, body) for note in notes], PageBreak()]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Body", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, 14, "Steering vectors on frozen Qwen3-4B | run qwen4b_haiku_v3 | research only")
    canvas.drawRightString(PAGE_W - MARGIN, 14, f"Page {doc.page}")
    canvas.restoreState()


def build(target, rows) -> int:
    story = results_page(rows) + index_page(rows)
    for row in rows:
        story += detail_page(row)
    story.pop()
    doc = SimpleDocTemplate(target, pagesize=landscape(A4), leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=MARGIN, bottomMargin=MARGIN + 14, title="Qwen3-4B steering review",
                            author="llm_steering_rl")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return doc.page


def main():
    global page_offset
    rows = json.loads((HERE / "compare_4b.json").read_text(encoding="utf-8"))
    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    register_fonts()
    page_offset = build(BytesIO(), rows) - len(rows)  # Each answer has one page, after the opening pages.
    pages = build(str(out), rows)
    print(json.dumps({"pdf": str(out), "pages": pages, "first_answer_page": page_offset + 1,
                      "symbols_not_drawable": len(dropped)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
