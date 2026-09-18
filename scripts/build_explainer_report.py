"""Build a readable, figure-heavy PDF explaining the full project.

All quantitative figures are regenerated from CSV artifacts in results/.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "report" / "AUDIENCE_STEERING_EXPLAINER.pdf"
ASSETS = ROOT / "report/report_assets"

NAVY = "#183153"
BLUE = "#2878B5"
TEAL = "#2A9D8F"
ORANGE = "#E76F51"
GOLD = "#E9C46A"
RED = "#C44536"
GREEN = "#3A7D44"
GREY = "#667085"
LIGHT = "#F2F5F8"


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def f(row, key):
    return float(row[key])


def avg_se(values):
    return mean(values), stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def finish(fig, name):
    path = ASSETS / name
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def base_axes(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=NAVY, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)


def plot_question():
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 4.5); ax.axis("off")
    boxes = [
        (0.2, 1.25, 2.2, 2.0, "Same technical\npassage", LIGHT, NAVY),
        (3.05, 2.45, 2.4, 1.2, "Reader: domain\nexpert", "#E7F5F2", GREEN),
        (3.05, 0.65, 2.4, 1.2, "Reader: expert in\nanother field", "#FFF2E8", ORANGE),
        (6.25, 2.45, 2.0, 1.2, "Technical\nsummary", "#EAF2FB", BLUE),
        (6.25, 0.65, 2.0, 1.2, "Accessible\nsummary", "#FFF7DB", "#9A6B00"),
        (8.85, 1.25, 1.4, 2.0, "Does one\nactivation\ndirection\ncontrol this?", "#F4EAFE", "#6C4AB6"),
    ]
    for x, y, w, h, text, face, edge in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                                    facecolor=face, edgecolor=edge, linewidth=1.5))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=10,
                color=NAVY, fontweight="bold" if x > 8 else "normal")
    for start, end in [((2.4, 2.25), (3.05, 3.05)), ((2.4, 2.25), (3.05, 1.25)),
                       ((5.45, 3.05), (6.25, 3.05)), ((5.45, 1.25), (6.25, 1.25)),
                       ((8.25, 3.05), (8.85, 2.75)), ((8.25, 1.25), (8.85, 1.75))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                    color=GREY, linewidth=1.3))
    ax.text(5.25, 4.2, "Research question", ha="center", fontsize=16,
            fontweight="bold", color=NAVY)
    return finish(fig, "01_question.png")


def plot_pipeline():
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.set_xlim(0, 11); ax.set_ylim(0, 4.8); ax.axis("off")
    labels = [
        ("1", "Build contrasts", "Same passage;\ndifferent reader fields"),
        ("2", "Capture activations", "Residual stream at\nlayer and token"),
        ("3", "Extract direction", "mean(domain expert)\n- mean(other expert)"),
        ("4", "Intervene", "Add, subtract,\nor ablate vector"),
        ("5", "Falsify", "Held-out domains, labels,\nrandoms, generation"),
    ]
    xs = np.linspace(0.2, 8.95, 5)
    for x, (num, title, desc) in zip(xs, labels):
        ax.add_patch(FancyBboxPatch((x, 1.25), 1.85, 2.1, boxstyle="round,pad=0.05",
                                    facecolor="white", edgecolor=BLUE, linewidth=1.5))
        ax.add_patch(plt.Circle((x+0.28, 3.08), 0.2, facecolor=BLUE, edgecolor="none"))
        ax.text(x+0.28, 3.08, num, color="white", ha="center", va="center", fontweight="bold")
        ax.text(x+0.925, 2.68, title, ha="center", va="center", color=NAVY,
                fontweight="bold", fontsize=10)
        ax.text(x+0.925, 1.85, desc, ha="center", va="center", color=GREY, fontsize=8.7)
    for x in xs[:-1]:
        ax.add_patch(FancyArrowPatch((x+1.87, 2.3), (x+2.14, 2.3), arrowstyle="-|>",
                                    mutation_scale=12, color=TEAL, linewidth=1.5))
    ax.text(5.5, 4.25, "The experimental loop", ha="center", fontsize=16,
            fontweight="bold", color=NAVY)
    ax.text(5.5, 0.55, "A positive steering result is not the endpoint: each later box tests a different alternative explanation.",
            ha="center", fontsize=10, color=GREY)
    return finish(fig, "02_pipeline.png")


def plot_behavior():
    model_paths = {
        "Qwen2.5-3B": RESULTS/"qwen25_3b_matched_behavior"/"behavior_summary.csv",
        "Qwen3.5-4B": RESULTS/"qwen35_4b_matched_behavior"/"behavior_summary.csv",
    }
    sources = ["kriging", "resnet", "transformer", "unet"]
    computed = {}
    for model, path in model_paths.items():
        data = rows(path)
        vals = []
        for source in sources:
            subset = [r for r in data if r["source_id"] == source and r["split"] == "test"]
            rel = mean(f(r, "mean_technical_margin") for r in subset if r["condition"].startswith("relevant"))
            irr = mean(f(r, "mean_technical_margin") for r in subset if r["condition"].startswith("irrelevant"))
            vals.append(rel-irr)
        computed[model] = vals
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(sources)); width = 0.34
    for offset, (model, vals), color in zip((-width/2, width/2), computed.items(), (ORANGE, BLUE)):
        ax.bar(x+offset, vals, width, label=model, color=color)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, [s.title() for s in sources])
    base_axes(ax, "Behavioral baseline: exactly relevant expertise changes summary preference",
              ylabel="relevant - unrelated expert margin")
    ax.legend(frameon=False)
    return finish(fig, "03_behavior.png")


def plot_lodo():
    data = rows(RESULTS/"exact_relevance_folds.csv")
    sources = [r["source"] for r in data]
    x = np.arange(len(sources)); width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    series = [
        ("Addition", "addition_effect", BLUE),
        ("Ablate selected layer", "single_layer_ablation_reduction", TEAL),
        ("Ablate all layers", "all_layer_ablation_reduction", ORANGE),
    ]
    for i, (label, key, color) in enumerate(series):
        ax.bar(x+(i-1)*width, [f(r, key) for r in data], width, label=label, color=color)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, [s.upper() if s in {"bert", "gcn", "ddpm"} else s.title() for s in sources], rotation=25)
    base_axes(ax, "Leave-one-domain-out: each direction is tested on a domain excluded from extraction",
              ylabel="signed margin change / gap reduction")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    return finish(fig, "04_lodo.png")


def plot_fixed_domains():
    data = rows(RESULTS/"qwen35_4b_fixed_direction_prospective"/"addition_by_domain.csv")
    sources = sorted({r["source"] for r in data}); enc = ["A/B", "X/Y", "1/2"]
    x = np.arange(len(sources)); width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (encoding, color) in enumerate(zip(enc, (BLUE, TEAL, GOLD))):
        vals = [f(next(r for r in data if r["source"] == source and r["encoding"] == encoding), "addition_effect")
                for source in sources]
        ax.bar(x+(i-1)*width, vals, width, label=encoding, color=color)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, [s.title() for s in sources], rotation=27)
    base_axes(ax, "One fixed Qwen direction transfers to eight invented methods",
              ylabel="signed addition effect")
    ax.legend(title="Answer labels", frameon=False, ncol=3)
    return finish(fig, "05_fixed_domains.png")


def aggregate_controls(path):
    data = rows(path)
    output = defaultdict(dict)
    for r in data:
        output[r["kind"]][int(r["seed"])] = f(r, "aggregate_mean")
    return {kind: list(by_seed.values()) for kind, by_seed in output.items()}


def plot_controls():
    controls = aggregate_controls(RESULTS/"qwen35_4b_fixed_direction_prospective"/"controls.csv")
    additions = rows(RESULTS/"qwen35_4b_fixed_direction_prospective"/"addition_by_domain.csv")
    learned = mean(f(r, "addition_effect") for r in additions if r["encoding"] == "A/B")
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    bins = np.linspace(min(controls["random"]+controls["permuted_labels"]),
                       max(controls["random"]+controls["permuted_labels"]+[learned]), 25)
    ax.hist(controls["random"], bins=bins, alpha=.65, label="Isotropic random", color=BLUE)
    ax.hist(controls["permuted_labels"], bins=bins, alpha=.58, label="Shuffled discovery labels", color=ORANGE)
    ax.axvline(learned, color=NAVY, linewidth=2.5, label=f"Learned = {learned:.3f}")
    ax.text(learned, ax.get_ylim()[1]*.82, "6 shuffled controls\nwere >= learned", ha="right",
            va="top", fontsize=9, color=RED)
    base_axes(ax, "Specificity control: the strict prospective near-miss",
              xlabel="aggregate signed addition effect", ylabel="control count")
    ax.legend(frameon=False, fontsize=8)
    return finish(fig, "06_controls.png")


def plot_dose():
    files = [
        ("Original synthetic profiles", RESULTS/"qwen35_4b_forced_choice_dose"/"dose_rows.csv"),
        ("Rewritten reader wording", RESULTS/"qwen35_4b_profile_paraphrase_dose"/"dose_rows.csv"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for ax, (title, path) in zip(axes, files):
        data = rows(path)
        for condition, label, color in (("domain_expert", "Domain expert", BLUE),
                                         ("matched_other_expert", "Other-field expert", ORANGE)):
            xs, ys, es = [], [], []
            for alpha in (-2, -1, 0, 1, 2):
                vals = [f(r, "technical_margin") for r in data if r["encoding"] == "A/B"
                        and r["condition"] == condition and f(r, "alpha") == alpha]
                y, e = avg_se(vals); xs.append(alpha); ys.append(y); es.append(e)
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=label, color=color)
        base_axes(ax, title, xlabel="steering coefficient",
                  ylabel="technical - accessible margin" if ax is axes[0] else "")
        ax.axvline(0, color="0.75", linewidth=1)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Monotonic forced-choice control survives profile paraphrasing",
                 x=.06, ha="left", fontsize=13, fontweight="bold", color=NAVY)
    fig.tight_layout(rect=(0,0,1,.94))
    return finish(fig, "07_dose.png")


def plot_projection():
    data = rows(RESULTS/"qwen35_4b_projection_transfer"/"projection_by_domain.csv")
    sources = sorted({r["source"] for r in data}); sets = sorted({r["test_set"] for r in data})
    labels = {
        "stimuli_prospective_synthetic": "Original reader wording",
        "stimuli_prospective_profile_paraphrase": "Rewritten reader wording",
    }
    x = np.arange(len(sources)); width=.35
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for off, test, color in zip((-width/2, width/2), sets, (BLUE, TEAL)):
        vals = [f(next(r for r in data if r["source"] == source and r["test_set"] == test), "projection_gap")
                for source in sources]
        ax.bar(x+off, vals, width, label=labels[test], color=color)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_xticks(x, [s.title() for s in sources], rotation=27)
    base_axes(ax, "Natural activations separate reader relevance along the frozen direction",
              ylabel="domain-expert - other-expert projection")
    ax.legend(frameon=False)
    return finish(fig, "08_projection.png")


def plot_ablation_damage():
    abl = rows(RESULTS/"qwen35_4b_fixed_direction_prospective"/"ablation_by_domain.csv")
    coh = rows(RESULTS/"qwen35_4b_fixed_direction_prospective"/"selected_coherence_controls.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    x=np.arange(len(abl)); width=.35
    axes[0].bar(x-width/2, [f(r,"selected_layer_ablation_reduction") for r in abl], width,
                label="Layer 17 only", color=TEAL)
    axes[0].bar(x+width/2, [f(r,"all_layer_ablation_reduction") for r in abl], width,
                label="All layers", color=ORANGE)
    axes[0].axhline(0,color="black",linewidth=.8)
    axes[0].set_xticks(x,[r["source"].title() for r in abl],rotation=35,fontsize=8)
    base_axes(axes[0], "Ablation reduces part of the natural gap", ylabel="gap reduction")
    axes[0].legend(frameon=False,fontsize=8)
    random_kl=[f(r,"mean_next_token_kl") for r in coh if r["kind"]=="random"]
    learned=f(next(r for r in coh if r["kind"]=="learned"),"mean_next_token_kl")
    axes[1].scatter(np.ones(len(random_kl)),random_kl,color=BLUE,s=35,alpha=.75,label="random directions")
    axes[1].scatter([1], [learned], color=RED, marker="*", s=170, label="learned direction", zorder=5)
    axes[1].set_xlim(.7,1.3); axes[1].set_xticks([1],["Layer 17 ablation"])
    base_axes(axes[1], "Ordinary-language disruption is tiny", ylabel="mean next-token KL")
    axes[1].legend(frameon=False,fontsize=8)
    fig.tight_layout()
    return finish(fig, "09_ablation_damage.png")


def plot_generation():
    data = rows(RESULTS/"qwen35_4b_fixed_direction_generation"/"generation_by_domain.csv")
    data = [r for r in data if r["metric"]=="term_coverage"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5,4.5))
    for condition,label,color in (("pooled","Pooled (both readers)",NAVY),("domain_expert","Domain expert",BLUE),
                                   ("matched_other_expert","Other-field expert",ORANGE)):
        subset=[r for r in data if r["condition"]==condition]
        xs=[];ys=[];es=[]
        for alpha in (-2,-1,0,1,2):
            vals=[f(r,f"alpha_{alpha:g}") for r in subset]
            y,e=avg_se(vals);xs.append(alpha);ys.append(y);es.append(e)
        axes[0].errorbar(xs,ys,yerr=es,marker="o",capsize=3,label=label,color=color)
    base_axes(axes[0],"Automated generation metric",xlabel="steering coefficient",ylabel="technical-term coverage")
    axes[0].legend(frameon=False,fontsize=8)
    pooled=[r for r in data if r["condition"]=="pooled"]
    effects=[f(r,"plus2_minus_minus2") for r in pooled]
    tolerance = 1e-12
    axes[1].bar(np.arange(len(pooled)), effects,
                color=[GREEN if v > tolerance else RED if v < -tolerance else "#B8C2CC" for v in effects])
    axes[1].axhline(0,color="black",linewidth=.8)
    axes[1].set_xticks(np.arange(len(pooled)),[r["source_id"].title() for r in pooled],rotation=35,fontsize=8)
    base_axes(axes[1],"Effect by source: only 2/8 meaningfully positive",ylabel="coverage at +2 minus -2")
    fig.tight_layout()
    return finish(fig,"10_generation.png")


def plot_learning_curve():
    data=rows(RESULTS/"qwen35_4b_direction_learning_curve"/"subset_results.csv")
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.5))
    sizes=[1,2,4,6,8]
    for ax,key,title,ylabel in ((axes[0],"mean_effect","Causal effect stays stable","mean held-out addition effect"),
                                (axes[1],"cosine_to_full","Direction geometry stabilizes","cosine to 8-domain vector")):
        for size in sizes:
            vals=[f(r,key) for r in data if int(r["train_domains"])==size]
            jitter=np.linspace(-.12,.12,len(vals)) if len(vals)>1 else [0]
            ax.scatter(np.array([size]*len(vals))+jitter,vals,color=BLUE,alpha=.35,s=16)
            ax.plot([size],[mean(vals)],marker="D",color=RED,markersize=6)
        base_axes(ax,title,xlabel="number of discovery domains",ylabel=ylabel)
        ax.set_xticks(sizes)
    axes[0].axhline(0,color="black",linewidth=.8)
    axes[1].set_ylim(0,1.05)
    fig.tight_layout()
    return finish(fig,"11_learning_curve.png")


def plot_cross_family():
    q=rows(RESULTS/"qwen35_4b_fixed_direction_prospective"/"encoding_summary.csv")
    p=rows(RESULTS/"phi35_38b_fixed_direction_prospective"/"encoding_summary.csv")
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.5))
    enc=["A/B","X/Y","1/2"];x=np.arange(3);width=.35
    axes[0].bar(x-width/2,[f(r,"mean_addition_effect") for r in q],width,
                yerr=[f(r,"standard_error") for r in q],capsize=3,label="Qwen3.5-4B",color=BLUE)
    axes[0].bar(x+width/2,[f(r,"mean_effect") for r in p],width,
                yerr=[f(r,"standard_error") for r in p],capsize=3,label="Phi-3.5 Mini",color=ORANGE)
    axes[0].set_xticks(x,enc);base_axes(axes[0],"Both models move, but consistency differs",ylabel="mean signed effect")
    axes[0].legend(frameon=False,fontsize=8)
    counts=[]
    for path, addpath, field in [
        (RESULTS/"qwen35_4b_fixed_direction_prospective"/"controls.csv",
         RESULTS/"qwen35_4b_fixed_direction_prospective"/"addition_by_domain.csv","addition_effect"),
        (RESULTS/"phi35_38b_fixed_direction_prospective"/"controls.csv",
         RESULTS/"phi35_38b_fixed_direction_prospective"/"addition_by_domain.csv","addition_effect")]:
        ctrl=aggregate_controls(path); add=rows(addpath)
        learned=mean(f(r,field) for r in add if r["encoding"]=="A/B")
        counts.extend([sum(v>=learned for v in ctrl["random"]),sum(v>=learned for v in ctrl["permuted_labels"])])
    positions=[0,1,3,4]
    control_bars = axes[1].bar(positions,counts,color=[BLUE,TEAL,ORANGE,GOLD])
    for bar, count in zip(control_bars, counts):
        axes[1].text(bar.get_x()+bar.get_width()/2, max(count, 0)+.45,
                     f"{count}/100", ha="center", va="bottom",
                     fontsize=9, fontweight="bold", color=NAVY)
    axes[1].axhline(5,color="black",linestyle="--",linewidth=1,label="frozen maximum")
    axes[1].set_xticks(positions,["Qwen\nrandom","Qwen\nshuffled","Phi\nrandom","Phi\nshuffled"])
    axes[1].set_ylim(0,21);base_axes(axes[1],"Controls matching or exceeding learned",ylabel="number of controls (out of 100)")
    axes[1].legend(frameon=False,fontsize=8)
    fig.tight_layout()
    return finish(fig,"12_cross_family.png")


def plot_evidence_ladder():
    fig,ax=plt.subplots(figsize=(10.5,5.2));ax.set_xlim(0,10.5);ax.set_ylim(0,5.2);ax.axis("off")
    items=[
        ("1. Behavioral contrast","PASS","Model changes summary preference with reader relevance",GREEN),
        ("2. Linear representation","PASS","Frozen direction separates unseen reader profiles",GREEN),
        ("3. Intervention sufficiency","PASS","Adding the vector monotonically moves forced-choice logits",GREEN),
        ("4. Normal use / mediation","PARTIAL","Ablation reduces only part of the natural decision gap","#B7791F"),
        ("5. Specificity","NEAR-MISS","6/100 shuffled controls beat learned; cutoff allowed 5",ORANGE),
        ("6. Generated explanations","FAIL / PENDING","Term metric fails; blinded human scoring remains",RED),
        ("7. Cross-family mechanism","FAIL","Phi effect does not beat specificity and ablation standards",RED),
    ]
    y=4.65
    for title,status,desc,color in items:
        ax.add_patch(FancyBboxPatch((.3,y-.48),9.9,.58,boxstyle="round,pad=.03",
                                    facecolor="#FFFFFF",edgecolor="#D0D5DD"))
        ax.text(.55,y-.18,title,va="center",fontsize=10,fontweight="bold",color=NAVY)
        ax.text(4.15,y-.18,status,va="center",ha="center",fontsize=9,fontweight="bold",color=color)
        ax.text(5.05,y-.18,desc,va="center",fontsize=8.7,color=GREY)
        y-=.65
    ax.text(5.25,5.0,"Evidence ladder: what each experiment actually licenses",ha="center",
            fontsize=15,fontweight="bold",color=NAVY)
    return finish(fig,"13_evidence_ladder.png")


def make_assets():
    ASSETS.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.labelcolor":NAVY,
                         "xtick.color":GREY,"ytick.color":GREY})
    return {
        "question":plot_question(), "pipeline":plot_pipeline(), "behavior":plot_behavior(),
        "lodo":plot_lodo(), "fixed":plot_fixed_domains(), "controls":plot_controls(),
        "dose":plot_dose(), "projection":plot_projection(), "ablation":plot_ablation_damage(),
        "generation":plot_generation(), "learning":plot_learning_curve(),
        "cross_family":plot_cross_family(), "ladder":plot_evidence_ladder(),
    }


class NumberedCanvasMixin:
    pass


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D0D5DD"));canvas.line(1.7*cm,1.35*cm,19.3*cm,1.35*cm)
    canvas.setFont("Helvetica",7.5);canvas.setFillColor(colors.HexColor(GREY))
    canvas.drawString(1.7*cm,.92*cm,"Audience-conditioned explanation depth — working research report")
    canvas.drawRightString(19.3*cm,.92*cm,f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(images):
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CoverTitle",fontName="Helvetica-Bold",fontSize=27,leading=32,
                              textColor=colors.HexColor(NAVY),alignment=TA_LEFT,spaceAfter=14))
    styles.add(ParagraphStyle(name="CoverSub",fontName="Helvetica",fontSize=13,leading=19,
                              textColor=colors.HexColor(GREY),spaceAfter=12))
    styles.add(ParagraphStyle(name="H1x",fontName="Helvetica-Bold",fontSize=19,leading=23,
                              textColor=colors.HexColor(NAVY),spaceBefore=5,spaceAfter=10))
    styles.add(ParagraphStyle(name="H2x",fontName="Helvetica-Bold",fontSize=13,leading=16,
                              textColor=colors.HexColor(BLUE),spaceBefore=8,spaceAfter=6))
    styles.add(ParagraphStyle(name="Bodyx",fontName="Helvetica",fontSize=9.4,leading=14,
                              textColor=colors.HexColor("#27364A"),spaceAfter=7))
    styles.add(ParagraphStyle(name="Smallx",fontName="Helvetica",fontSize=8,leading=11,
                              textColor=colors.HexColor(GREY),spaceAfter=5))
    styles.add(ParagraphStyle(name="Captionx",fontName="Helvetica-Oblique",fontSize=8,leading=11,
                              textColor=colors.HexColor(GREY),spaceBefore=4,spaceAfter=9))
    styles.add(ParagraphStyle(name="Calloutx",fontName="Helvetica-Bold",fontSize=10,leading=15,
                              textColor=colors.HexColor(NAVY),backColor=colors.HexColor("#EAF2FB"),
                              borderPadding=9,spaceBefore=6,spaceAfter=10))
    styles.add(ParagraphStyle(name="Failx",fontName="Helvetica-Bold",fontSize=10,leading=15,
                              textColor=colors.HexColor("#7A271A"),backColor=colors.HexColor("#FEECE8"),
                              borderPadding=9,spaceBefore=6,spaceAfter=10))
    styles.add(ParagraphStyle(name="Bulletx",parent=styles["Bodyx"],leftIndent=13,firstLineIndent=-7,
                              bulletIndent=3,spaceAfter=4))

    doc=BaseDocTemplate(str(OUT),pagesize=A4,rightMargin=1.7*cm,leftMargin=1.7*cm,
                        topMargin=1.55*cm,bottomMargin=1.65*cm,
                        title="Audience-conditioned explanation depth",
                        author="Nikhil Makkar",
                        subject="Step-by-step mechanistic-interpretability experiment report")
    frame=Frame(doc.leftMargin,doc.bottomMargin,doc.width,doc.height,id="normal")
    doc.addPageTemplates(PageTemplate(id="main",frames=frame,onPage=footer))
    story=[]
    P=lambda text,style="Bodyx":story.append(Paragraph(text,styles[style]))
    def heading(text): P(text,"H1x")
    def sub(text): P(text,"H2x")
    def bullet(text): story.append(Paragraph(text,styles["Bulletx"],bulletText="•"))
    def figure(key,caption,width=17.2*cm):
        im=Image(str(images[key]));im._restrictSize(width,10.8*cm);story.append(im);P(caption,"Captionx")
    def table(data,widths=None,font=8):
        cooked=[]
        for row in data:
            cooked.append([Paragraph(str(cell),styles["Smallx"]) for cell in row])
        t=Table(cooked,colWidths=widths,repeatRows=1,hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor(NAVY)),("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#D0D5DD")),
            ("VALIGN",(0,0),(-1,-1),"TOP"),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor(LIGHT)]),
            ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
            ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ]));story.append(t);story.append(Spacer(1,8))

    story.append(Spacer(1,1.4*cm));P("Audience-conditioned<br/>explanation depth","CoverTitle")
    P("A step-by-step mechanistic investigation of whether a language model represents what a reader knows — and whether one activation direction controls the technical level of its explanation.","CoverSub")
    story.append(Spacer(1,.6*cm));figure("question","The project isolates reader-domain relevance while holding the source passage and candidate summaries fixed.",17.2*cm)
    story.append(Spacer(1,.35*cm));P("Working research report · Nikhil Makkar · 8 September 2026","Smallx")
    P("Purpose: understand the completed experiments and provide reusable figures for a later MATS application report. This document separates confirmed results, near-misses, failures, and pending human validation.","Smallx")
    story.append(PageBreak())

    heading("Executive summary")
    P("The motivating observation was simple: when asked for the <i>gist</i> of a technical paper, a model may over-explain for an imagined novice instead of giving a concise, technically precise account. The research question became narrower and testable: <b>does an LLM internally represent whether the reader's expertise is relevant to the source, and can a linear activation direction control the summary it prefers?</b>")
    P("The answer is mixed. For Qwen3.5-4B, one residual-stream direction is strongly readable and causally controls a forced-choice decision between technical and accessible summaries. It transfers across unfamiliar domains, answer-label encodings, steering strengths, and new reader-profile wording. Ablating it reduces part of the model's natural audience-conditioned preference while barely disturbing ordinary language modelling.","Calloutx")
    P("But the strongest broader claims fail. The fixed prospective intervention narrowly misses its shuffled-label specificity threshold; the predeclared free-generation term metric does not move reliably; and Phi-3.5 Mini does not show direction-specific cross-family replication. Therefore this is <b>not</b> evidence for a universal 'explanation-depth mechanism.'","Failx")
    table([
        ["Question","Result","Meaning"],
        ["Can the vector separate unseen reader profiles?","Yes: 8/8 domains; AUC 0.81–0.84","Strong linear representation evidence in Qwen3.5"],
        ["Does adding it change forced-choice preference?","Yes: 8/8 across three label encodings","Strong intervention sufficiency"],
        ["Does removing it reduce natural behavior?","Partly: 7/8; selected-layer p=0.0078","Partial mediation, not a complete mechanism"],
        ["Is it unique to correct discovery labels?","Near-miss: 6/100 shuffled >= learned; cutoff 5","Specificity remains unresolved"],
        ["Does it control generated gists?","Automated test failed; human review pending","Cannot claim generation control"],
        ["Does it replicate in another family?","No clean Phi replication","No cross-family claim"],
    ],[5.0*cm,4.1*cm,7.2*cm])
    story.append(PageBreak())

    heading("1. From an observation to a falsifiable question")
    P("The original idea was not 'models use jargon badly.' That is too broad and too subjective. We turned it into a controlled contrast in which both summaries are factually acceptable, but one preserves domain terminology and the other explains the same mechanism more accessibly.")
    figure("question","Figure 1. The only intended causal difference is whether the named reader field is relevant to the source passage.")
    sub("Why this is related to steering-vector research")
    P("The project borrows a pattern from work on refusal and reasoning behaviors: construct contrastive examples, extract a mean-difference direction from internal activations, then test addition and ablation. The important distinction is that methodological similarity does not imply the same strength of claim. Refusal is an overt behavior; 'appropriate explanation depth' is more subjective and harder to measure.")
    bullet("<b>Representation:</b> can reader relevance be decoded from activations?")
    bullet("<b>Sufficiency:</b> does injecting the direction move the model's decision?")
    bullet("<b>Necessity or mediation:</b> does removing the direction reduce natural behavior?")
    bullet("<b>Ecological validity:</b> does the result survive free-form generation?")
    story.append(PageBreak())

    heading("2. What one experimental example looks like")
    P("Each prompt contains a reader profile, one self-contained technical passage, and two candidate summaries. Candidate order is balanced: the technical summary is A half the time and B half the time. This prevents a fixed preference for A or B from creating the result.")
    table([
        ["Prompt component","Illustrative content","What is controlled"],
        ["Reader profile","Expert in Kalman filtering vs expert in population genetics","Both are experts; only domain relevance changes"],
        ["Source","A robust state estimator rejects high-Mahalanobis innovations","Identical in the pair"],
        ["Technical candidate","Uses state covariance, innovation gating, and a Kalman gain","Factually valid; domain terminology retained"],
        ["Accessible candidate","Predicts a hidden quantity, rejects implausible measurements, then corrects","Factually valid; less assumed background"],
        ["Output","One label token: A/B, X/Y, or 1/2","Measured as a log-probability margin"],
    ],[3.4*cm,7.3*cm,6.0*cm])
    P("Primary behavioral metric:","H2x")
    P("<font name='Courier'><b>technical margin = log P(technical label) - log P(accessible label)</b></font>")
    P("A positive audience gap means the model prefers technical summaries more for domain experts than for equally expert readers in unrelated fields. The independent unit for final inference is the source domain, not each duplicated prompt.")
    figure("pipeline","Figure 2. The project repeatedly moved from an apparent positive result to a harder test designed to disprove it.")
    story.append(PageBreak())

    heading("3. Step 1 — establish behavior before inspecting internals")
    P("The first question was whether the models changed summary preference at all. Early expert-versus-novice contrasts were strong but confounded: 'novice' also implies a need for definitions and simpler prose. We therefore matched overall sophistication and compared relevant experts with practitioners or researchers from unrelated fields.")
    figure("behavior","Figure 3. Test-split differences from the matched-profile behavioral experiment. Positive bars mean exactly relevant expertise increases preference for the technical summary.")
    P("This motivated an <b>exact-relevance</b> direction rather than a generic expert/novice direction. It also exposed model and domain variation; the behavior was not identical everywhere. That is why later inference treats domains as independent units and includes Qwen2.5 as a falsifying replication rather than pooling all prompts.")
    sub("What this step does not show")
    bullet("It does not identify an internal representation.")
    bullet("It does not show causality; prompts alone can change behavior through many routes.")
    bullet("It does not tell us whether free-form generations change meaningfully.")
    story.append(PageBreak())

    heading("4. Step 2 — extract and intervene on a direction")
    P("At a chosen transformer layer and token position, we record the residual-stream activation for every contrastive prompt. The candidate direction is the difference between the mean domain-expert activation and the mean other-domain-expert activation. Addition moves activations along the direction; ablation removes their projection onto it.")
    figure("pipeline","Figure 4. Extraction and causal testing use held-out data: representation discovery, intervention selection, and final evaluation are separated.")
    table([
        ["Operation","Question answered","Safe interpretation"],
        ["Linear projection / decoding","Is relevance information readable along this vector?","The information is represented linearly enough for this probe"],
        ["Add +v or -v","Can this activation feature control the output?","The intervention is sufficient to affect this readout"],
        ["Ablate v","Does natural computation depend on this feature?","A reduction supports partial mediation, subject to damage controls"],
        ["Random direction","Would almost any perturbation work?","Tests generic sensitivity at the same norm"],
        ["Shuffled-label direction","Does the correct contrast matter?","Tests extraction-pipeline specificity"],
    ],[3.3*cm,5.6*cm,7.8*cm])
    P("The report deliberately avoids saying 'causal mechanism' from addition alone. Causal control at an intervention site is weaker than demonstrating that the model normally computes and uses one unique mechanism.","Calloutx")
    story.append(PageBreak())

    heading("5. Step 3 — leave one real domain out")
    P("A direction that works only on the same paper used to construct it may encode source-specific words. The eight-domain leave-one-domain-out (LODO) experiment learns from seven real ML domains, selects layer/token coordinates using only their validation profiles, and evaluates once on the excluded eighth domain.")
    figure("lodo","Figure 5. Qwen3.5 LODO results. Addition is positive for every excluded domain; ablation is weaker and sometimes negative.")
    table([
        ["LODO result","Value"],
        ["Mean addition effect","+0.186 ± 0.022 SE"],
        ["Positive held-out domains","8/8"],
        ["Exact one-sided sign-flip p","0.0039"],
        ["Answer re-encodings","A/B +0.186; X/Y +0.203; 1/2 +0.230; each 8/8"],
        ["Selected-layer ablation","+0.078; 6/8; p=0.0195"],
        ["All-layer ablation","+0.129; 7/8; p=0.0078"],
    ],[8.1*cm,8.4*cm])
    P("This is strong evidence that the <i>extraction procedure</i> generalizes across real domains. It is not yet evidence for one fixed vector because each fold may select a different layer and learn a slightly different vector.")
    story.append(PageBreak())

    heading("6. Step 4 — freeze one vector and test unfamiliar methods")
    P("To remove the fold-specific-vector ambiguity, a stricter protocol was written before evaluation. One Qwen3.5 vector was fitted once from all eight real discovery domains at frozen layer 17. It was then applied unchanged to eight invented, self-contained methods spanning unrelated technical areas. Invented names and mechanisms reduce the possibility that the model is recalling a known paper-summary pairing.")
    figure("fixed","Figure 6. Signed addition effects for one frozen direction. Every domain is positive under A/B, X/Y, and 1/2 labels.")
    table([
        ["Encoding","Mean effect","Positive domains","Exact p"],
        ["A/B","+0.156","8/8","0.0039"],
        ["X/Y","+0.178","8/8","0.0039"],
        ["1/2","+0.184","8/8","0.0039"],
    ],[4.1*cm]*4)
    P("Re-encoding matters because a vector might accidentally favor the token A or interact with its embedding. Similar effects under three encodings make that explanation implausible.")
    story.append(PageBreak())

    heading("7. The most important near-miss: shuffled-label specificity")
    P("Two null distributions were predeclared. Isotropic controls are arbitrary directions with the learned vector's norm. Shuffled-label controls run the same mean-difference pipeline after randomly assigning discovery examples to two balanced groups. The frozen criterion allowed at most five of 100 controls to match the learned aggregate.")
    figure("controls","Figure 7. No isotropic direction matched the learned Qwen effect, but 6/100 shuffled-label directions did. The predeclared cutoff was therefore missed by one.")
    P("This result must remain a <b>failure of the complete prospective protocol</b>, even though it is close. Increasing permutations afterward or changing the cutoff would turn uncertainty into researcher degrees of freedom. The correct interpretation is: the fixed direction is highly effective and non-random, but the extraction labels are not uniquely specific at the predeclared 5% standard.","Failx")
    P("Why can shuffled directions work? Prompt activations contain many correlated features: field names, lexical templates, candidate-summary content, and reader descriptions. A random balanced split can sometimes yield a direction aligned with a high-gain downstream feature. This is exactly why isotropic random controls alone were not enough.")
    story.append(PageBreak())

    heading("8. Step 5 — test monotonicity and prompt wording")
    P("The initial intervention pooled +v on other-domain experts with -v on domain experts. A stronger interpretation requires a single coefficient axis: increasing alpha should increase the technical-summary margin in either context. We tested alpha = -2, -1, 0, +1, +2 and repeated the full test after replacing every profile carrier sentence.")
    figure("dose","Figure 8. A/B curves shown; X/Y and 1/2 produce the same qualitative result. Every encoding-by-condition cell has 8/8 positive source slopes and exact p=0.0039 in both profile sets.")
    P("This rules out two important alternatives: the effect is not merely created by the original counterfactual sign convention, and it is not tied to phrases such as 'deeply familiar with.' The direction behaves like a context-independent control knob for the <b>forced-choice technical-summary logit margin</b>.","Calloutx")
    story.append(PageBreak())

    heading("9. Step 6 — verify natural representation, not only intervention")
    P("An intervention could change a decision even if the model's natural activations do not use that axis to distinguish readers. We therefore projected unseen prompt activations onto the frozen direction without steering.")
    figure("projection","Figure 9. Within every synthetic source, domain-expert activations project higher than other-domain-expert activations. The result survives wholly new profile-carrier wording.")
    table([
        ["Profile set","Mean within-domain gap","Positive domains","AUC","Fixed-midpoint accuracy","Controls >= learned"],
        ["Original synthetic","+0.285","8/8","0.809","62.5%","0/100 random; 0/100 shuffled"],
        ["New carrier wording","+0.329","8/8","0.836","71.9%","0/100 random; 0/100 shuffled"],
    ],[3.2*cm,3.1*cm,2.4*cm,1.7*cm,2.7*cm,3.5*cm])
    P("AUC and within-domain gaps transfer better than the absolute midpoint threshold. This means relative separation is strong, but the whole activation distribution shifts between the real discovery prompts and synthetic prompts. Saying 'perfectly decodable' would therefore overstate the result.")
    story.append(PageBreak())

    heading("10. Step 7 — ablation and language damage")
    P("Addition demonstrates sufficiency. To test whether the model normally relies on this direction, we remove the projection of its activations along the unit vector. A valid ablation must reduce the audience-conditioned gap without broadly breaking next-token prediction.")
    figure("ablation","Figure 10. Left: selected-layer and all-layer ablation reductions by source. Right: WikiText next-token KL for the learned layer-17 direction and ten matched random directions.")
    table([
        ["Diagnostic","Result","Interpretation"],
        ["Selected-layer ablation","Mean +0.047; 7/8; p=0.0078","Supports partial natural use"],
        ["All-layer ablation","Mean +0.063; 7/8; p=0.043","Distributed but not uniform effect"],
        ["WikiText NLL change","-0.0011 over 4,080 tokens","No measured loss degradation"],
        ["Mean next-token KL","0.00073","Very small absolute distribution change"],
        ["Top-1 agreement","98.4%","Most token predictions unchanged"],
    ],[4.3*cm,5.0*cm,7.3*cm])
    P("Ablation does not erase the whole behavioral gap, so the evidence supports <b>partial mediation</b>, not a single complete mechanism. Component-level attention/MLP localization also failed earlier; we do not have a head- or circuit-level account.")
    story.append(PageBreak())

    heading("11. Step 8 — free generation is the ecological test")
    P("The original observation concerned generated explanations, not selecting A or B. We therefore generated 160 two-to-three-sentence gists: eight unfamiliar methods × two reader conditions × two profile phrasings × five steering coefficients, using greedy decoding. The primary frozen metric was coverage of source-specific technical terms.")
    figure("generation","Figure 11. The automatic term metric is nearly flat. The pooled +2 minus -2 contrast is +0.011, meaningfully positive for only 2/8 domains (p=0.25); the five-point slope is positive for 3/8 and also non-significant (p=0.1875).")
    P("This experiment fails its predeclared criteria. It is possible that generated paraphrases change technicality without using the exact term bank; the PhaseNest examples qualitatively suggest some wording shifts. But inspecting examples after the metric fails cannot replace a frozen outcome. The correct next step is blinded human scoring for technicality, factual correctness, and coherence.","Failx")
    P("The human-review sheet contains all outputs in randomized order together with their source passages. Coefficients and reader conditions are kept in a separate key that should remain unopened until scoring is complete.")
    story.append(PageBreak())

    heading("12. Step 9 — how much discovery diversity is necessary?")
    P("The full direction averages eight discovery domains. To test whether one favorable domain drives it, we exhaustively fitted every subset of 1, 2, 4, 6, and 8 domains: 135 directions in total. Every vector was evaluated unchanged on all eight synthetic methods.")
    figure("learning","Figure 12. Dots are every subset; diamonds are size means. Mean causal effect is already near +0.15, while geometric agreement with the full vector rises steadily with domain diversity.")
    table([
        ["Discovery domains","Subsets","Meet 7/8 rule","All 8/8 positive","Mean cosine to full"],
        ["1","8","6/8","6/8","0.579"],
        ["2","28","26/28","25/28","0.735"],
        ["4","70","70/70","70/70","0.884"],
        ["6","28","28/28","28/28","0.957"],
        ["8","1","1/1","1/1","1.000"],
    ],[3.4*cm,2.2*cm,3.3*cm,3.3*cm,3.7*cm])
    P("Four diverse discovery domains are sufficient for stability in this stimulus set. More domains mainly align the direction geometrically and reduce dependence on the specific training subset; they do not create the average causal effect from nothing.")
    story.append(PageBreak())

    heading("13. Step 10 — non-Qwen cross-family replication")
    P("Two Qwen checkpoints are not independent architectural evidence. A prospective replication used Microsoft Phi-3.5 Mini (3.8B). Five layer candidates were evaluated only with real-domain LODO discovery; layer 12 was selected before scoring synthetic methods.")
    figure("cross_family","Figure 13. Phi has positive average additions, but more random and shuffled controls match the learned effect. X/Y is positive in only 6/8 domains, below the frozen 7/8 requirement.")
    table([
        ["Phi diagnostic","Result","Frozen outcome"],
        ["A/B","+0.170; 7/8; p=0.0078","Passes domain sign test"],
        ["X/Y","+0.094; 6/8; p=0.0234","Fails 7/8 consistency"],
        ["1/2","+0.170; 8/8; p=0.0039","Passes domain sign test"],
        ["Aggregate controls","6 random; 18 shuffled >= learned","Fails specificity"],
        ["Individual random controls","Only 2/8 domains clear 95th percentile","Fails specificity"],
        ["Selected-layer ablation","5/8; p=0.0703","Inconclusive"],
    ],[4.2*cm,6.1*cm,6.4*cm])
    P("Phi is perturbable in the same broad direction, but the learned direction is not special enough relative to controls and is not clearly necessary. This is a failed cross-family mechanistic replication, not evidence that the phenomenon is absent from Phi under every possible design.")
    story.append(PageBreak())

    heading("14. How the evidence fits together")
    figure("ladder","Figure 14. Each rung requires a stronger conclusion. The project reaches reliable representation and forced-choice control, but not general generated-explanation control.")
    sub("The strongest defensible claim")
    P("<b>In Qwen3.5-4B, reader-domain relevance is linearly represented along a stable residual-stream direction, and adding or removing that direction causally affects the model's forced-choice preference between technical and accessible summaries across unfamiliar domains.</b>","Calloutx")
    sub("Claims the evidence does not support")
    bullet("There is one universal 'technicality neuron' or one complete mechanism.")
    bullet("The vector controls the quality or depth of arbitrary generated explanations.")
    bullet("The mechanism is caused by constitutional training or alignment training.")
    bullet("The same direction or extraction procedure works cleanly across model families.")
    bullet("The attention head or MLP circuit implementing the behavior has been localized.")
    story.append(PageBreak())

    heading("15. Validity threats and how we handled them")
    table([
        ["Threat","Control performed","Remaining limitation"],
        ["A/B token bias","Balanced order; repeated X/Y and 1/2","Sequence scoring differs slightly for Phi digits"],
        ["Generic sophistication","Relevant vs equally expert unrelated readers","Profiles remain synthetic descriptions"],
        ["Paper-word memorization","LODO real domains; invented held-out mechanisms","Invented stimuli need independent human validation"],
        ["Layer/token cherry-picking","Validation-only selection; prospective frozen layer","Early pilot informed later protocols"],
        ["Any perturbation works","Norm-matched isotropic random controls","Finite Monte Carlo samples"],
        ["Arbitrary mean split works","Balanced shuffled-label controls","Qwen prospective result narrowly misses cutoff"],
        ["General model damage","WikiText NLL, KL, and top-1 agreement","Only one ordinary-text dataset"],
        ["One favorable domain","Eight-domain inference; exhaustive subset curve","Eight sources are still a small universe"],
        ["Forced choice equals generation","Five-point greedy generation experiment","Human scoring still pending"],
        ["One model family","Phi prospective replication","Only one non-Qwen model tested"],
    ],[4.0*cm,6.0*cm,6.7*cm],font=7.5)
    sub("Statistical note")
    P("With eight independent domains, an all-positive result has exact one-sided sign-flip probability 1/256 = 0.00390625. Prompt variants and answer-order duplicates are averaged within domains; treating all prompt rows as independent would produce misleadingly small p-values.")
    story.append(PageBreak())

    heading("16. What remains before using this in an application")
    P("The compute-heavy falsification suite is complete. The remaining work is primarily scientific validation and communication, not another undirected model sweep.")
    table([
        ["Priority","Action","Why it matters"],
        ["1","Review all eight invented passages and both summaries in results/prospective_stimulus_review.csv","A flawed candidate pair can create an artificial preference"],
        ["2","Score all 160 rows in results/qwen35_4b_fixed_direction_generation/blind_review_with_sources.csv before opening blind_key.csv","Determines whether the term-bank failure is a metric failure or true generation failure"],
        ["3","Manually verify several raw logit-margin rows from the CSV and read the core scripts","You should be able to explain every headline number"],
        ["4","Choose the application narrative only after human scoring","The final story may be a positive forced-choice result or a readout-generation dissociation"],
    ],[1.3*cm,10.0*cm,5.4*cm])
    sub("A compelling application narrative")
    P("The strongest narrative is not 'I found a universal alignment vector.' It is: <i>I began from a vague observation about over-simplified paper summaries, operationalized reader relevance, progressively falsified confounds, found a robust Qwen forced-choice feature, and discovered sharp limits in specificity, generated text, and cross-family replication.</i> That demonstrates research taste through the sequence of tests, including the failures.")
    story.append(PageBreak())

    heading("Appendix A — experiment chronology")
    table([
        ["Stage","Main question","Outcome"],
        ["Expert vs novice pilot","Is technical-summary preference audience-conditioned?","Large effect, but coarse-cue confound"],
        ["Matched expertise","Does exact domain relevance matter?","Yes behaviorally, with model/domain variation"],
        ["Layer/token sweep","Where is a controllable residual feature?","Late-middle layers; final prompt token"],
        ["Random and decoder controls","Is the result arbitrary or label-token-specific?","Mixed; motivated stronger protocols"],
        ["Component localization","Can attention/MLP outputs localize it?","No clean component"],
        ["Eight-domain LODO","Does extraction generalize?","Strong Qwen3.5; Qwen2.5 fails re-encoding"],
        ["Fixed synthetic transfer","Does one vector generalize?","Strong additions; shuffled-label near-miss"],
        ["Dose and carriers","Is it monotonic and wording-robust?","Yes, all six cells 8/8"],
        ["Projection transfer","Is it naturally represented?","Yes within domains; absolute threshold shifts"],
        ["Ablation and coherence","Is it used without general damage?","Partial Qwen mediation; tiny damage"],
        ["Free generation","Does it change actual gists?","Frozen automatic metric fails"],
        ["Learning curve","Is the result data-subset dependent?","Stable from four discovery domains"],
        ["Phi replication","Is it cross-family?","No direction-specific replication"],
    ],[3.3*cm,7.2*cm,6.2*cm],font=7.3)
    story.append(PageBreak())

    heading("Appendix B — key files and reproducibility")
    P("All quantitative plots in this PDF are regenerated from CSV outputs by <font name='Courier'>build_explainer_report.py</font>. The report does not recompute model activations.")
    table([
        ["Purpose","File or directory"],
        ["Submitted full report","report/FULL_REPORT.md"],
        ["Fixed Qwen prospective outputs","results/qwen35_4b_fixed_direction_prospective/"],
        ["Forced-choice dose responses","results/qwen35_4b_forced_choice_dose/ and qwen35_4b_profile_paraphrase_dose/"],
        ["Free-generation outputs","results/qwen35_4b_fixed_direction_generation/"],
        ["Representation transfer","results/qwen35_4b_projection_transfer/"],
        ["Discovery-size curve","results/qwen35_4b_direction_learning_curve/"],
        ["Cross-family outputs","results/phi35_38b_fixed_direction_prospective/"],
        ["Frozen protocols","PROSPECTIVE_*.md, *_PROTOCOL.md, CROSS_FAMILY_PROTOCOL.md"],
    ],[5.0*cm,11.7*cm])
    sub("Related work")
    P("Arditi et al. (2024), <i>Refusal in Language Models Is Mediated by a Single Direction</i>. https://arxiv.org/abs/2406.11717")
    P("Venhoff et al. (2025), <i>Understanding Reasoning in Thinking Language Models via Steering Vectors</i>. https://arxiv.org/abs/2506.18167")
    P("Cloud et al. (2025), <i>Subliminal Learning: Language models transmit behavioral traits via hidden signals in data</i>. https://arxiv.org/abs/2507.14805")
    P("These papers motivated the contrastive-direction / causal-intervention template. The present project asks a different and narrower question about reader-relevance-conditioned summary selection.")
    sub("Model scope")
    P("Main models: Qwen2.5-3B-Instruct and Qwen3.5-4B. Cross-family replication: Microsoft Phi-3.5 Mini Instruct (3.8B dense decoder-only Transformer). Results apply only to the tested checkpoints, prompts, layers, and interventions.")

    doc.build(story)


def main():
    images=make_assets()
    build_pdf(images)
    print(OUT)


if __name__ == "__main__":
    main()
