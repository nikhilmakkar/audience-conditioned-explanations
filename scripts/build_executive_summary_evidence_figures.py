"""Evidence-rich figures for the MATS executive summary.

Unlike the earlier headline figures, these expose the task and domain-level data.
"""

import csv
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "report/report_assets"
NAVY = "#17365D"
BLUE = "#176B87"
TEAL = "#4F9DA6"
ORANGE = "#D97706"
RED = "#B45353"
GREY = "#667085"
LIGHT = "#E5E7EB"
PALE = "#A8DADC"

FIELD = {
    "anchorshift": "Causal inference", "driftgate": "Control",
    "fluxpatch": "Fluid dynamics", "motifbridge": "Comp. biology",
    "phasenest": "Audio", "phasesar": "SAR",
    "quorumweave": "Cryptography", "shardbloom": "Databases",
}


def read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def clean(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=LIGHT, linewidth=.8, zorder=0)


def save(fig, stem):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, xywh, heading, body, edge, face, fontsize=9.0):
    x, y, w, h = xywh
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=.035,rounding_size=.05",
        facecolor=face, edgecolor=edge, linewidth=1.5,
    ))
    ax.text(x + .13, y + h - .14, heading, va="top", color=edge,
            fontsize=8.4, fontweight="bold")
    ax.text(x + .13, y + h - .49, body, va="top", color="#273444",
            fontsize=fontsize, linespacing=1.24)


def figure_design():
    fig, ax = plt.subplots(figsize=(11.2, 5.65))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 11.2); ax.set_ylim(0, 5.65); ax.axis("off")
    ax.text(.12, 5.48, "What the forced-choice experiment actually measures",
            fontsize=16, fontweight="bold", color=NAVY, va="top")

    ax.text(.2, 5.02, "HELD CONSTANT", fontsize=8.7, color=NAVY, fontweight="bold")
    passage = ("A residual block learns F(x) and adds the input x, producing F(x)+x.\n"
               "Identity shortcuts make identity mappings easy; projections handle dimension changes.")
    box(ax, (.15, 3.88, 6.55, .98), "SAME PASSAGE: RESNET", passage, NAVY, "#EEF4FA", 8.55)

    ax.text(7.25, 5.02, "ONLY VARIABLE: READER PROFILE", fontsize=8.7,
            color=GREY, fontweight="bold")
    box(ax, (7.22, 3.88, 3.78, .98), "RELEVANT EXPERT",
        "Researches deep convolutional networks\nfor computer vision", TEAL, "#EAF6F4", 8.45)

    technical = ("ResNet parameterizes blocks as\nF(x)+x. Identity shortcuts ease\n"
                 "optimization; projections handle\ndimension changes.")
    accessible = ("Each block learns a change to its\ninput, then adds the original input\n"
                  "back. Adjusted shortcuts handle\ndifferent representation sizes.")
    box(ax, (.15, 2.43, 3.15, 1.16), "CANDIDATE A: TECHNICAL", technical, BLUE, "#EAF2FB", 7.4)
    box(ax, (3.55, 2.43, 3.15, 1.16), "CANDIDATE B: ACCESSIBLE", accessible, ORANGE, "#FFF8E1", 7.4)
    box(ax, (7.22, 2.53, 3.78, .98), "UNRELATED EXPERT",
        "Researches computational fluid dynamics", ORANGE, "#FFF4E8", 8.45)

    ax.text(3.43, 2.18, "Every reader sees the same passage and both summaries",
            ha="center", fontsize=8.6, color=GREY, fontweight="bold")
    for start, end in [((3.43, 2.08), (3.43, 1.55)), ((9.1, 2.40), (8.4, 1.55))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11,
                                    color="#98A2B3", linewidth=1.0))

    ax.add_patch(FancyBboxPatch((1.05, .68), 9.1, .76, boxstyle="round,pad=.04",
                                facecolor="#F4F6F8", edgecolor="#98A2B3", linewidth=1.2))
    ax.text(5.6, 1.20, "MEASURED OUTPUT", ha="center", va="center", fontsize=8.4,
            color=GREY, fontweight="bold")
    ax.text(5.6, .91,
            "margin = log P(technical label) − log P(accessible label)   •   positive means technical is preferred",
            ha="center", va="center", fontsize=9.6, color=NAVY, fontweight="bold")
    ax.text(5.6, .27,
            "Direction extraction: 8 real ML topics  →  freeze layer/token/vector  →  evaluate unchanged on 8 invented methods",
            ha="center", va="center", fontsize=9.3, color="#344054")
    save(fig, "exec_evidence_1_design")


def control_values(rows, kind):
    by_seed = {int(r["seed"]): float(r["aggregate_mean"]) for r in rows if r["kind"] == kind}
    return np.array([by_seed[k] for k in sorted(by_seed)])


def figure_robustness():
    additions = read(R / "qwen35_4b_fixed_direction_prospective/addition_by_domain.csv")
    projections = read(R / "qwen35_4b_projection_transfer/projection_by_domain.csv")
    controls = read(R / "qwen35_4b_fixed_direction_prospective/controls.csv")
    encodings = ["A/B", "X/Y", "1/2"]
    domains = list(dict.fromkeys(r["source"] for r in additions))

    matrix = np.array([[float(next(r["addition_effect"] for r in additions
                                   if r["source"] == d and r["encoding"] == e))
                        for e in encodings] for d in domains])
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 5.1), gridspec_kw={"width_ratios": [1.0, 1.05, 1.25]})
    fig.patch.set_facecolor("white")
    fig.suptitle("Domain-level transfer, profile rewording, and null distributions",
                 x=.055, y=1.01, ha="left", fontsize=16, fontweight="bold", color=NAVY)

    ax = axes[0]
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=.33, aspect="auto")
    for i in range(8):
        for j in range(3):
            ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center",
                    color="white" if matrix[i,j] > .19 else NAVY, fontsize=8.4, fontweight="bold")
    ax.set_xticks(range(3), encodings)
    ax.set_yticks(range(8), [FIELD[d] for d in domains], fontsize=8.5)
    ax.set_title("A. Steering effect for every test", loc="left")
    ax.set_xlabel("Answer-label encoding")
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=.046, pad=.03)

    ax = axes[1]
    original = {r["source"]: float(r["projection_gap"]) for r in projections
                if r["test_set"] == "stimuli_prospective_synthetic"}
    rewritten = {r["source"]: float(r["projection_gap"]) for r in projections
                 if r["test_set"] == "stimuli_prospective_profile_paraphrase"}
    ypos = np.arange(len(domains))
    for i, d in enumerate(domains):
        ax.plot([original[d], rewritten[d]], [i, i], color="#B8C2CC", linewidth=1.2, zorder=1)
        ax.scatter([original[d]], [i], color=TEAL, s=34, zorder=2,
                   label="Original profiles" if i == 0 else None)
        ax.scatter([rewritten[d]], [i], color=NAVY, s=34, zorder=2,
                   label="Rewritten profiles" if i == 0 else None)
    ax.axvline(0, color=GREY, linewidth=.9)
    ax.set_xlim(-.02, .58); ax.set_ylim(-.6, 7.6)
    ax.set_yticks(ypos, [FIELD[d] for d in domains], fontsize=7.7)
    ax.set_xlabel("Relevant − unrelated projection")
    ax.set_title("B. Projection gap by domain", loc="left")
    ax.legend(frameon=False, fontsize=7.6, loc="lower right")
    clean(ax, grid="x")

    ax = axes[2]
    rand = control_values(controls, "random")
    shuf = control_values(controls, "permuted_labels")
    learned = matrix[:, 0].mean()
    bins = np.linspace(min(rand.min(), shuf.min()) - .01, max(shuf.max(), learned) + .02, 23)
    ax.hist(rand, bins=bins, color=PALE, alpha=.88, label="Random directions", zorder=2)
    ax.hist(shuf, bins=bins, histtype="step", color=RED, linewidth=2,
            label="Shuffled audience labels", zorder=3)
    ax.axvline(learned, color=NAVY, linewidth=2.3, label="Learned direction")
    ax.set_xlabel("Mean steering effect")
    ax.set_ylabel("Control directions")
    ax.set_title("C. Full control distributions", loc="left")
    ax.text(.97, .16, f"random ≥ learned: {(rand >= learned).sum()}/100\n"
                      f"shuffled ≥ learned: {(shuf >= learned).sum()}/100",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.8, fontweight="bold")
    ax.legend(frameon=False, fontsize=7.8, loc="upper left")
    clean(ax)

    fig.subplots_adjust(left=.09, right=.985, top=.84, bottom=.19, wspace=.50)
    save(fig, "exec_evidence_2_robustness")


def figure_mechanism():
    ablation = read(R / "qwen35_4b_fixed_direction_prospective/ablation_by_domain.csv")
    intrinsic = read(R / "qwen35_4b_intrinsic_rank_minimal/evaluation_by_domain.csv")
    domains = [r["source"] for r in ablation]
    interventions = ["Original direction", "Selected 1D", "Rank-3"]
    colors = [ORANGE, TEAL, NAVY]
    orig = {r["source"]: float(r["selected_layer_ablation_reduction"]) for r in ablation}
    selected = {r["source"]: float(r["gap_reduction"]) for r in intrinsic
                if r["test_set"] == "synthetic" and r["intervention"] == "selected_1d_mixture"}
    rank3 = {r["source"]: float(r["gap_reduction"]) for r in intrinsic
             if r["test_set"] == "synthetic" and r["intervention"] == "rank3_subspace"}

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.0), gridspec_kw={"width_ratios": [1.0, 1.25]})
    fig.patch.set_facecolor("white")
    fig.suptitle("What changed when the candidate directions were removed",
                 x=.06, y=1.01, ha="left", fontsize=16, fontweight="bold", color=NAVY)

    ax = axes[0]
    values = np.array([[orig[d], selected[d], rank3[d]] for d in domains])
    for row in values:
        ax.plot(range(3), row, color="#C8D0D8", linewidth=1, alpha=.9, zorder=1)
        ax.scatter(range(3), row, color=colors, s=30, alpha=.85, zorder=2)
    means = values.mean(axis=0)
    ax.plot(range(3), means, color="#111827", linewidth=2.4, marker="D", markersize=7,
            label="Mean across domains", zorder=4)
    for i, m in enumerate(means):
        ax.text(i, m + .018, f"mean {m:.3f}", ha="center", fontsize=8.7, fontweight="bold")
    ax.axhline(0, color=GREY, linewidth=.9)
    ax.set_xticks(range(3), interventions)
    ax.set_ylabel("Audience-gap reduction")
    ax.set_title("A. Each line is one invented domain", loc="left")
    ax.legend(frameon=False, fontsize=8.5)
    clean(ax)

    ax = axes[1]
    sets = ["real_test", "synthetic", "new_carriers"]
    set_labels = ["Held-out real profiles", "Invented methods", "Rewritten profiles"]
    methods = ["svd_component_1", "selected_1d_mixture", "rank3_subspace"]
    method_labels = ["Variance-leading 1D", "Selected 1D", "Rank-3"]
    offsets = [-.23, 0, .23]
    rng_offsets = np.linspace(-.055, .055, 8)
    for j, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
        for i, test_set in enumerate(sets):
            vals = np.array([float(r["gap_reduction"]) for r in intrinsic
                             if r["test_set"] == test_set and r["intervention"] == method])
            xpos = i + offsets[j]
            ax.scatter(xpos + rng_offsets, vals, color=color, alpha=.38, s=17, zorder=2)
            ax.errorbar([xpos], [vals.mean()], yerr=[vals.std(ddof=1)/np.sqrt(len(vals))],
                        fmt="D", color=color, markersize=6, capsize=3, linewidth=1.5,
                        label=label if i == 0 else None, zorder=4)
    ax.axhline(0, color=GREY, linewidth=.9)
    ax.set_xticks(range(3), set_labels)
    ax.set_ylabel("Audience-gap reduction")
    ax.set_title("B. Dots expose the variation behind each mean", loc="left")
    ax.legend(frameon=False, fontsize=8.2, loc="upper left")
    clean(ax)

    fig.subplots_adjust(left=.08, right=.985, top=.82, bottom=.17, wspace=.34)
    save(fig, "exec_evidence_3_mechanism")


plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5,
    "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.labelsize": 9.5,
})
figure_design()
figure_robustness()
figure_mechanism()
