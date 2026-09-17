"""Build a readable PDF from MATS_EXECUTIVE_SUMMARY_DRAFT.md."""

from html import escape
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Spacer,
)


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "MATS_EXECUTIVE_SUMMARY_DRAFT.md"
OUTPUT = ROOT / "MATS_EXECUTIVE_SUMMARY_DRAFT.pdf"
NAVY = colors.HexColor("#17365D")
BLUE = colors.HexColor("#176B87")
GREY = colors.HexColor("#5A6573")


def inline_markup(text: str) -> str:
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", text)
    text = text.replace(r"\(p=0.0039\)", "<i>p</i> = 0.0039")
    text = text.replace(r"\(p=0.25\)", "<i>p</i> = 0.25")
    return text


def scaled_image(path: Path, max_width: float, max_height: float) -> Image:
    width, height = ImageReader(str(path)).getSize()
    scale = min(max_width / width, max_height / height)
    image = Image(str(path), width=width * scale, height=height * scale)
    image.hAlign = "CENTER"
    return image


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D8DEE6"))
    canvas.line(18 * mm, 13 * mm, A4[0] - 18 * mm, 13 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(GREY)
    canvas.drawString(18 * mm, 8.5 * mm, "Nikhil Makkar — MATS application project")
    canvas.drawRightString(A4[0] - 18 * mm, 8.5 * mm, f"Executive summary  |  {doc.page}")
    canvas.restoreState()


def main() -> None:
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCustom", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=20, leading=23, textColor=NAVY, alignment=TA_LEFT,
        spaceAfter=5 * mm,
    )
    h2 = ParagraphStyle(
        "H2Custom", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=11.5, leading=14, textColor=BLUE, spaceBefore=2.5 * mm,
        spaceAfter=1.4 * mm,
    )
    body = ParagraphStyle(
        "BodyCustom", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=9.2, leading=12.1, textColor=colors.HexColor("#202832"),
        alignment=TA_LEFT, spaceAfter=2.3 * mm,
    )
    caption = ParagraphStyle(
        "CaptionCustom", parent=body, fontSize=7.7, leading=9.4,
        textColor=GREY, alignment=TA_CENTER, spaceBefore=0.8 * mm,
        spaceAfter=2.3 * mm,
    )

    doc = BaseDocTemplate(
        str(OUTPUT), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=15 * mm, bottomMargin=17 * mm,
        title="Does Qwen know when technical language is useful?",
        author="Nikhil Makkar",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates(PageTemplate(id="main", frames=[frame], onPage=footer))

    story = []
    paragraphs = []

    def flush() -> None:
        if paragraphs:
            story.append(Paragraph(inline_markup(" ".join(paragraphs)), body))
            paragraphs.clear()

    figure_number = 0
    for raw in SOURCE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        if line.startswith("!["):
            flush()
            match = re.match(r"!\[(.+?)\]\((.+?)\)", line)
            if match:
                figure_number += 1
                image_path = ROOT / match.group(2)
                figure = scaled_image(image_path, doc.width, 67 * mm)
                figure_caption = Paragraph(
                    f"<b>Figure {figure_number}.</b> {escape(match.group(1))}.", caption
                )
                story.append(KeepTogether([figure, figure_caption]))
            continue
        if line.startswith("# "):
            flush()
            story.append(Paragraph(inline_markup(line[2:]), title))
            continue
        if line == "## Executive summary":
            continue
        if line.startswith("### "):
            flush()
            story.append(Paragraph(inline_markup(line[4:]), h2))
            continue
        paragraphs.append(line)
    flush()

    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    main()
