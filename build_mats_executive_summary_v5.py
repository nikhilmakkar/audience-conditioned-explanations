"""Build the application-calibrated MATS executive summary and its two custom figures."""

from html import escape
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate, Paragraph, Spacer


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "report_assets"
SOURCE = ROOT / "MATS_EXECUTIVE_SUMMARY_V5.md"
OUTPUT = ROOT / "MATS_EXECUTIVE_SUMMARY_V5.pdf"
NAVY = "#17365D"
TEAL = "#176B87"
GREEN = "#23856D"
AMBER = "#D97706"
RED = "#B54747"
GREY = "#5A6573"


def task_figure() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 4.15))
    ax.set_xlim(0, 11.2); ax.set_ylim(0, 4.15); ax.axis("off")
    fig.patch.set_facecolor("white")
    passage = ("DriftGate predicts a hidden state,\nchecks each observation for outliers,\n"
               "and updates from accepted measurements.")
    tech = ("Robust Kalman-style filter; gates observations\nusing squared Mahalanobis innovation, then\n"
            "applies the covariance-derived gain.")
    accessible = ("Predicts the next value and its uncertainty,\nignores implausible measurements, then corrects\n"
                  "the prediction using accepted ones.")
    boxes = [
        (0.15, 2.55, 3.0, 1.30, "TECHNICAL PASSAGE", passage, "#EEF4FA", NAVY),
        (3.55, 2.55, 3.35, 1.30, "RELEVANT EXPERT", "Kalman filtering and\nfeedback control", "#E8F6F1", GREEN),
        (7.3, 2.55, 3.55, 1.30, "OTHER-FIELD EXPERT", "Population genetics and\nphylogenetics", "#FFF4E8", AMBER),
        (0.35, 0.45, 5.0, 1.45, "TECHNICAL SUMMARY", tech, "#EAF2FB", TEAL),
        (5.85, 0.45, 5.0, 1.45, "ACCESSIBLE SUMMARY", accessible, "#FFF8E1", AMBER),
    ]
    for x, y, w, h, label, body, face, edge in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.05,rounding_size=.06",
                                    facecolor=face, edgecolor=edge, linewidth=1.25))
        ax.text(x+.13, y+h-.16, label, fontsize=8.0, color=edge, weight="bold", va="top")
        ax.text(x+.13, y+h-.50, body, fontsize=7.8, linespacing=1.22, color="#263442", va="top")
    ax.annotate("", xy=(3.47, 3.20), xytext=(3.15, 3.20), arrowprops=dict(arrowstyle="->", color=GREY))
    ax.annotate("", xy=(7.22, 3.20), xytext=(6.9, 3.20), arrowprops=dict(arrowstyle="->", color=GREY))
    fig.savefig(ASSETS / "application_task.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def mechanism_figure() -> None:
    labels = ["Original\nmean-difference", "SVD\ncomponent 1", "Optimized 1D\n(exploratory)", "Rank-3\nsubspace"]
    vals = np.array([0.047, 0.039, 0.156, 0.180])
    fig, ax = plt.subplots(figsize=(10.9, 4.3))
    fig.patch.set_facecolor("white")
    bars = ax.bar(np.arange(4), vals, color=[AMBER, "#C9A227", "#4F9DA6", NAVY], width=.64, zorder=3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, val+.007, f"{val:.3f}", ha="center", weight="bold", color=NAVY)
    ax.axhline(.594, color="#7B8794", linestyle="--", linewidth=1.2)
    ax.text(3.48, .598, "Natural audience gap = 0.594", ha="right", va="bottom", fontsize=9, color=GREY)
    ax.set_xticks(np.arange(4), labels)
    ax.set_ylim(0, .66)
    ax.set_ylabel("Audience gap removed by ablation")
    ax.set_title("The easiest direction to extract was not the one Qwen relied on most", loc="left",
                 fontsize=14, weight="bold", color=NAVY, pad=10)
    ax.text(.02, .87, "8% removed", transform=ax.transAxes, color=AMBER, weight="bold")
    ax.text(.60, .38, "one direction nearly\nmatches all three", transform=ax.transAxes,
            ha="center", color=TEAL, weight="bold")
    ax.annotate("", xy=(2.85, .18), xytext=(2.15, .156), arrowprops=dict(arrowstyle="<->", color=TEAL, lw=1.5))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E7EB", zorder=0)
    fig.tight_layout()
    fig.savefig(ASSETS / "application_mechanism_result.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def falsification_figure() -> None:
    rows = [
        ("Topic or label shortcut?", "8 invented methods; A/B → X/Y → 1/2", "PASS", GREEN),
        ("Exact profile wording?", "replace every reader-profile sentence", "PASS", GREEN),
        ("Arbitrary perturbation?", "100 random directions", "PASS", GREEN),
        ("Extraction pipeline artifact?", "100 shuffled-label directions", "NEAR MISS", AMBER),
        ("Normal causal mechanism?", "remove direction instead of adding it", "WEAK", AMBER),
        ("Three dimensions necessary?", "validation-selected 1D inside rank-3", "NOT SHOWN", AMBER),
        ("Original free-generation effect?", "technical-term coverage under steering", "FAIL", RED),
        ("Same result in another family?", "Phi-3.5 Mini replication", "FAIL", RED),
    ]
    fig, ax = plt.subplots(figsize=(11.1, 5.65))
    ax.set_xlim(0, 11.1); ax.set_ylim(0, 9.1); ax.axis("off"); fig.patch.set_facecolor("white")
    ax.text(.15, 8.75, "I treated every positive result as a new thing to falsify", fontsize=15,
            weight="bold", color=NAVY, va="top")
    ax.text(.15, 8.18, "Alternative explanation", fontsize=8.5, weight="bold", color=GREY)
    ax.text(4.15, 8.18, "Test", fontsize=8.5, weight="bold", color=GREY)
    y = 7.72
    for i, (question, test, status, color) in enumerate(rows):
        if i % 2 == 0:
            ax.add_patch(FancyBboxPatch((.08, y-.29), 10.85, .64, boxstyle="round,pad=.02",
                                        facecolor="#F6F8FA", edgecolor="none"))
        ax.text(.2, y, question, fontsize=9.2, color="#263442", va="center")
        ax.text(4.15, y, test, fontsize=9.2, color="#263442", va="center")
        ax.add_patch(FancyBboxPatch((9.15, y-.20), 1.55, .40, boxstyle="round,pad=.03,rounding_size=.08",
                                    facecolor=color, edgecolor="none"))
        ax.text(9.925, y, status, ha="center", va="center", fontsize=8.2, color="white", weight="bold")
        y -= .82
    ax.text(.2, .48,
            "Scope: 2 model families • layer/token sweep • 16 source domains • 3 answer encodings • "
            "dose sweeps • addition + ablation • 200 null directions • free generation",
            fontsize=8.8, color=NAVY, weight="bold")
    fig.savefig(ASSETS / "application_falsification_map.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def markup(text: str) -> str:
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", text)
    return text


def scaled_image(path: Path, max_width: float, max_height: float) -> Image:
    width, height = ImageReader(str(path)).getSize()
    scale = min(max_width / width, max_height / height)
    result = Image(str(path), width=width*scale, height=height*scale)
    result.hAlign = "CENTER"
    return result


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D8DEE6")); canvas.line(17*mm, 13*mm, A4[0]-17*mm, 13*mm)
    canvas.setFont("Helvetica", 7.3); canvas.setFillColor(colors.HexColor(GREY))
    canvas.drawString(17*mm, 8.5*mm, "Nikhil Makkar — MATS application project")
    canvas.drawRightString(A4[0]-17*mm, 8.5*mm, f"Executive summary  |  {doc.page}")
    canvas.restoreState()


def build_pdf() -> None:
    styles = getSampleStyleSheet()
    title = ParagraphStyle("TitleX", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=21,
                           leading=23.5, textColor=colors.HexColor(NAVY), alignment=TA_LEFT, spaceAfter=2*mm)
    subtitle = ParagraphStyle("SubtitleX", parent=styles["BodyText"], fontName="Helvetica", fontSize=11,
                              leading=13, textColor=colors.HexColor(TEAL), spaceAfter=4*mm)
    h2 = ParagraphStyle("H2X", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12,
                        leading=14, textColor=colors.HexColor(TEAL), spaceBefore=2.3*mm, spaceAfter=1.1*mm)
    body = ParagraphStyle("BodyX", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.3,
                          leading=12.0, textColor=colors.HexColor("#202832"), spaceAfter=2.2*mm)
    bullet = ParagraphStyle("BulletX", parent=body, leftIndent=4*mm, firstLineIndent=-3*mm,
                            bulletIndent=0, spaceAfter=1.5*mm)
    caption = ParagraphStyle("CapX", parent=body, fontSize=7.5, leading=9, textColor=colors.HexColor(GREY),
                             spaceBefore=.6*mm, spaceAfter=1.8*mm)
    doc = BaseDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=17*mm, rightMargin=17*mm,
                          topMargin=13*mm, bottomMargin=17*mm, title=SOURCE.stem, author="Nikhil Makkar")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates(PageTemplate(id="main", frames=[frame], onPage=footer))
    story, paragraph = [], []
    def flush():
        if paragraph:
            story.append(Paragraph(markup(" ".join(paragraph)), body)); paragraph.clear()
    for raw in SOURCE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line: flush(); continue
        if line.startswith("# "):
            flush(); story.append(Paragraph(markup(line[2:]), title)); continue
        if line.startswith("### "):
            flush(); story.append(Paragraph(markup(line[4:]), subtitle)); continue
        if line.startswith("## "):
            flush(); story.append(Paragraph(markup(line[3:]), h2)); continue
        if line.startswith("- "):
            flush(); story.append(Paragraph(markup(line[2:]), bullet, bulletText="•")); continue
        match = re.match(r"!\[(.+?)\]\((.+?)\)", line)
        if match:
            flush(); img = scaled_image(ROOT/match.group(2), doc.width, 88*mm)
            story.append(KeepTogether([img, Paragraph(markup(match.group(1)), caption)])); continue
        if line == "[PAGEBREAK]":
            flush(); story.append(PageBreak()); continue
        paragraph.append(line)
    flush(); doc.build(story)


def main() -> None:
    task_figure(); mechanism_figure(); falsification_figure(); build_pdf(); print(OUTPUT)


if __name__ == "__main__":
    main()
