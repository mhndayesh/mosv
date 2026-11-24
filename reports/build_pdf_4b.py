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
RESULTS = HERE.parent / "rl_v2" / "qwen4b_v2"
TOPICS = ("engineering", "story", "teaching", "debugging", "math_solving", "explanation", "professional_email", "how_to_guide")
SHOWN_SAMPLES = {"WildChat-1M": (1, 2), "LMSYS Chat 1M (clean)": (1, 2), "WritingPrompts": (0, 2),
                 "Arena preference 140k": (1,), "ShareGPT V3": (1, 2), "no_robots": (0, 2), "Storium-5K": (0, 2)}


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


def router_summary():
    audit = json.loads((HERE / "route_audit.json").read_text(encoding="utf-8"))
    topics = {name: item for name, item in audit.items() if name != "none"}
    total = sum(item["prompts"] for item in topics.values())
    right = sum(item["routes"].get(name, 0) for name, item in topics.items())
    missed = sum(item["routes"].get("none", 0) for item in topics.values())
    plain = audit["none"]
    return (f"<b>Router on all held-out prompts:</b> {right:,} of {total:,} topic prompts got the right vector, "
            f"{total - right - missed} got a wrong vector, {missed} got no steering. On {plain['prompts']} normal prompts, "
            f"{plain['prompts'] - plain['routes'].get('none', 0)} were steered by mistake.")


def results_page():
    reports = [json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8")) for name in TOPICS]
    reports.sort(key=lambda r: -r["test"]["steered"]["score"])
    cell = style("r", 10.5, after=0)
    data = [header_row(["Topic", "Test prompts", "Unsteered", "Random vector", "Trained vector", "Broken answers"])]
    for r in reports:
        test = r["test"]
        data.append([Paragraph(f'<font color="{THEMES[r["expert"]][0]}"><b>{LABELS[r["expert"]]}</b></font>', cell),
                     Paragraph(str(r["test_prompts"]), cell), Paragraph(f'{test["base"]["score"]:.3f}', cell),
                     Paragraph(f'{test["random"]["score"]:.3f}', cell), Paragraph(f'<b>{test["steered"]["score"]:.3f}</b>', cell),
                     Paragraph(f'{test["steered"]["degenerate"]:.1%}', cell)])
    notes = ["<b>What the score means:</b> the checklist measures structure and process, not truth or writing quality.",
             "<b>Professional email:</b> treat its score as format only. Its checklist has passed an email that changed the request's purpose.",
             "<b>Debugging:</b> prompts too long to leave room for a full answer were excluded.",
             router_summary(),
             "<b>Router bug fixed in this version:</b> after the first generation, pad tokens leaked into the router's input and sent many debugging prompts to the math vector. The earlier 46-prompt PDF was affected on page 21."]
    body = style("n", 10, color=colors.HexColor("#334155"), after=5)
    return [Paragraph("Qwen3-4B steering vectors: final results", style("title", 20, "BodyB", after=4)),
            Paragraph("All 8 topics, trained on a frozen model and tested on held-out prompts.", style("sub", 11, color=MUTED, after=9)),
            grid(data, [FRAME_W * share for share in (0.29, 0.13, 0.14, 0.15, 0.15, 0.14)]), Spacer(1, 11),
            *[Paragraph(note, body) for note in notes], PageBreak()]


def clean(text, limit):
    flat = " ".join(keep_drawable(text, "Body").split())
    return escape(flat if len(flat) <= limit else flat[:limit - 3] + "...", quote=False)


def data_pages():
    sources = json.loads((HERE / "story_samples.json").read_text(encoding="utf-8"))
    cell = style("d", 9.2, after=0)
    data = [header_row(["Source", "Who wrote it", "Rows measured", "Median words", "3+ paragraphs", "Dialogue", "Markdown", "Story checklist"])]
    for s in sources:
        data.append([Paragraph(f'<b>{s["source"]}</b>', cell), Paragraph(s["origin"], cell), Paragraph(str(s["measured"]), cell),
                     Paragraph(str(s["median_words"]), cell), Paragraph(f'{s["pct_3plus_paragraphs"]}%', cell),
                     Paragraph(f'{s["pct_dialogue"]}%', cell), Paragraph(f'{s["pct_markdown"]}%', cell),
                     Paragraph(f'{s["story_checklist"]:.2f}', cell)])
    verdict = ("<b>Do the responses have structure?</b> They have normal writing structure, such as paragraphs, dialogue, verses and script format, "
               "but <b>not the process structure our checklist rewards</b>. Their story checklist averages are 0.00 to 0.35."
               "So they are mostly natural text, close to what a model already saw in pretraining. "
               "<b>We would use the prompts only.</b> The RL loop writes its own answers, so these responses are not training targets. "
               "Storium's replies are mostly short character descriptions, so its replies are not usable as answers. "
               "These are small samples of 8 to 60 rows per source, after a safety filter, and a few non-story rows passed the keyword match.")
    samples = [header_row(["Source", "Real prompt", "Its response (start)"])]
    small = style("s", 8.4, after=0)
    for s in sources:
        for index in SHOWN_SAMPLES[s["source"]]:
            item = s["samples"][index]
            samples.append([Paragraph(f'<b>{s["source"]}</b>', small), Paragraph(clean(item["prompt"], 330), small),
                            Paragraph(clean(item["response"], 520), small)])
    return [Paragraph("New story data: what we found", style("title", 20, "BodyB", after=4)),
            Paragraph("Not downloaded yet. Rows below were read from the public dataset viewer.", style("sub", 11, color=MUTED, after=8)),
            grid(data, [FRAME_W * share for share in (0.2, 0.24, 0.09, 0.09, 0.1, 0.09, 0.09, 0.1)]), Spacer(1, 10),
            Paragraph(verdict, style("v", 10.2, color=colors.HexColor("#334155"), after=0)), PageBreak(),
            Paragraph("Sample prompts from each source", style("title", 17, "BodyB", after=7)),
            grid(samples, [FRAME_W * 0.14, FRAME_W * 0.36, FRAME_W * 0.50]), PageBreak()]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Body", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, 14, "Steering vectors on frozen Qwen3-4B | run qwen4b_v2 | research only")
    canvas.drawRightString(PAGE_W - MARGIN, 14, f"Page {doc.page}")
    canvas.restoreState()


def build(target, rows) -> int:
    story = results_page() + data_pages() + index_page(rows)
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
