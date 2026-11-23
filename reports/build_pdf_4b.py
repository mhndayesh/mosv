import json
import re
import sys
from html import escape
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
LABELS = {"engineering": "Engineering", "story": "Story", "teaching": "Teaching", "debugging": "Debugging", "math_solving": "Math solving", "explanation": "Explanation", "professional_email": "Professional email", "how_to_guide": "How-to (untrained, no effect)", "none": "Normal prompt"}
TREE = re.compile("[├└│┌┐┘┴┬┼─]")
dropped = set()


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


def fit(left, right, width, room, accents):
    for size in SIZES:
        a, b = answer_flowables(left, size, accents[0]), answer_flowables(right, size, accents[1])
        if max(height(a, width), height(b, width)) <= room:
            return a, b, size
    raise ValueError(f"Answer does not fit even at {SIZES[-1]} pt")


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
            change_cell(row), Paragraph(str(row["id"] + 2), center)]


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


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Body", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, 14, "Steering vectors on frozen Qwen3-4B | run qwen4b_v2 | research only")
    canvas.drawRightString(PAGE_W - MARGIN, 14, f"Page {doc.page}")
    canvas.restoreState()


def main():
    rows = json.loads((HERE / "compare_4b.json").read_text(encoding="utf-8"))
    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    register_fonts()
    story = index_page(rows)
    for row in rows:
        story += detail_page(row)
    story.pop()
    doc = SimpleDocTemplate(str(out), pagesize=landscape(A4), leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=MARGIN, bottomMargin=MARGIN + 14, title="Steered vs normal answers",
                            author="llm_steering_rl")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(json.dumps({"pdf": str(out), "pages": doc.page, "symbols_not_drawable": sorted(dropped)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
