"""Render docs/MANUAL.md to a PDF.

    .venv/bin/pip install reportlab
    .venv/bin/python scripts/make_manual.py

Kept in the repository so the PDF can be regenerated rather than hand-edited
and drifting from the Markdown it came from.
"""
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, HRFlowable, KeepTogether,
                                PageBreak, PageTemplate, Paragraph, Preformatted,
                                Spacer, Table, TableStyle)

SRC = Path("docs/MANUAL.md")
OUT = Path("Cadence-Auto-Manual.pdf")

INK = colors.HexColor("#2a2724")
SOFT = colors.HexColor("#55504a")
FAINT = colors.HexColor("#8a8179")
ACCENT = colors.HexColor("#b4552d")
RULE = colors.HexColor("#e5ded3")
PAPER = colors.HexColor("#faf8f4")

# reportlab's built-in fonts use WinAnsi, which has no glyph for these; they
# would render as solid black boxes.
SUBS = {"→": "->", "≥": ">=", "≤": "<="}


def clean(text: str) -> str:
    for bad, good in SUBS.items():
        text = text.replace(bad, good)
    return text


def inline(text: str) -> str:
    """Markdown inline markup -> reportlab markup, escaping everything else."""
    text = clean(text)
    out, i = [], 0
    # Protect code spans from escaping twice.
    for part in re.split(r"(`[^`]+`)", text):
        if part.startswith("`") and part.endswith("`") and len(part) > 1:
            out.append(
                f'<font face="Courier" size="9" color="#b4552d">'
                f"{html.escape(part[1:-1])}</font>")
        else:
            esc = html.escape(part)
            esc = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", esc)
            esc = re.sub(r"(?<![\w*])\*([^*]+?)\*(?![\w*])", r"<i>\1</i>", esc)
            esc = re.sub(r"\[(.+?)\]\((.+?)\)",
                         r'<link href="\2" color="#b4552d">\1</link>', esc)
            out.append(esc)
    return "".join(out)


ss = getSampleStyleSheet()
S = {
    "h1": ParagraphStyle("h1", parent=ss["Title"], fontName="Helvetica-Bold",
                         fontSize=26, leading=30, textColor=INK,
                         alignment=TA_LEFT, spaceAfter=4),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=15,
                         leading=19, textColor=ACCENT, spaceBefore=18,
                         spaceAfter=7),
    "h3": ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=11.5,
                         leading=15, textColor=INK, spaceBefore=13,
                         spaceAfter=5),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.6,
                           leading=14.2, textColor=SOFT, spaceAfter=7),
    "bullet": ParagraphStyle("bullet", fontName="Helvetica", fontSize=9.6,
                             leading=14.2, textColor=SOFT, leftIndent=13,
                             bulletIndent=3, spaceAfter=3.5),
    "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=8.6,
                           leading=12, textColor=SOFT),
    "cellh": ParagraphStyle("cellh", fontName="Helvetica-Bold", fontSize=8.6,
                            leading=12, textColor=INK),
    "code": ParagraphStyle("code", fontName="Courier", fontSize=8.3,
                           leading=11.4, textColor=INK, leftIndent=8),
}


def build_table(rows):
    header, body = rows[0], rows[1:]
    data = [[Paragraph(inline(c), S["cellh"]) for c in header]]
    data += [[Paragraph(inline(c), S["cell"]) for c in r] for r in body]
    t = Table(data, hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
        ("LINEBELOW", (0, 1), (-1, -2), 0.3, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def parse(md: str):
    story, lines, i = [], md.splitlines(), 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("```"):
            block, i = [], i + 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(clean(lines[i]))
                i += 1
            i += 1
            story.append(Spacer(1, 3))
            story.append(Preformatted("\n".join(block), S["code"]))
            story.append(Spacer(1, 8))
            continue

        if line.startswith("|") and i + 1 < len(lines) and set(
                lines[i + 1].replace("|", "").strip()) <= set("-: "):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                if not set("".join(cells)) <= set("-: "):
                    rows.append(cells)
                i += 1
            story.append(Spacer(1, 3))
            story.append(build_table(rows))
            story.append(Spacer(1, 10))
            continue

        if line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), S["h3"])); i += 1; continue
        if line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), S["h2"])); i += 1; continue
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), S["h1"])); i += 1; continue

        if line.strip() == "---":
            story.append(Spacer(1, 7))
            story.append(HRFlowable(width="100%", thickness=0.6, color=RULE))
            story.append(Spacer(1, 5)); i += 1; continue

        m = re.match(r"^(\d+)\.\s+(.*)", line)
        if m:
            text = m.group(2)
            while i + 1 < len(lines) and lines[i + 1].startswith("   "):
                i += 1
                text += " " + lines[i].strip()
            story.append(Paragraph(inline(text), S["bullet"],
                                   bulletText=f"{m.group(1)}."))
            i += 1; continue

        if line.startswith("- "):
            text = line[2:]
            while i + 1 < len(lines) and lines[i + 1].startswith("  ") \
                    and not lines[i + 1].lstrip().startswith("-"):
                i += 1
                text += " " + lines[i].strip()
            story.append(Paragraph(inline(text), S["bullet"], bulletText="•"))
            i += 1; continue

        if not line.strip():
            i += 1; continue

        para = [line]
        while i + 1 < len(lines) and lines[i + 1].strip() and not re.match(
                r"^(#|```|\||-\s|\d+\.\s|---$)", lines[i + 1]):
            i += 1
            para.append(lines[i])
        story.append(Paragraph(inline(" ".join(para)), S["body"]))
        i += 1
    return story


def decorate(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(PAPER)
    canvas.rect(0, 0, A4[0], A4[1], stroke=0, fill=1)
    canvas.setFont("Helvetica", 7.6)
    canvas.setFillColor(FAINT)
    if doc.page > 1:
        canvas.drawString(20 * mm, 12 * mm, "Cadence Auto - Manual")
    canvas.drawRightString(A4[0] - 20 * mm, 12 * mm, str(doc.page))
    canvas.restoreState()


md = SRC.read_text()
doc = BaseDocTemplate(str(OUT), pagesize=A4,
                      leftMargin=20 * mm, rightMargin=20 * mm,
                      topMargin=18 * mm, bottomMargin=20 * mm,
                      title="Cadence Auto - Manual", author="Amritha S")
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=decorate)])
doc.build(parse(md))
print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
