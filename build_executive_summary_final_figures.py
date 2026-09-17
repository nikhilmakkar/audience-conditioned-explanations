"""Three simple, standalone figures for the MATS executive summary."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = ROOT / "report_assets"

NAVY = "#17365D"
BLUE = "#176B87"
TEAL = "#4F9DA6"
ORANGE = "#D97706"
RED = "#B45353"
GREY = "#667085"
LIGHT = "#E5E7EB"

FIELDS = {
    "anchorshift": "AnchorShift · causal inference",
    "driftgate": "DriftGate · control",
    "fluxpatch": "FluxPatch · fluid dynamics",
    "motifbridge": "MotifBridge · computational biology",
    "phasenest": "PhaseNest · audio",
    "phasesar": "PhaseSAR · radar",
    "quorumweave": "QuorumWeave · cryptography",
    "shardbloom": "ShardBloom · databases",
}


def read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def style(ax, grid="x"):
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=LIGHT, linewidth=.8, zorder=0)


def save(fig, stem):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def transfer_figure():
    rows = read(RESULTS / "qwen35_4b_fixed_direction_prospective/addition_by_domain.csv")
    domains = list(dict.fromkeys(r["source"] for r in rows))
    encodings = ["A/B", "X/Y", "1/2"]
    colors = [NAVY, TEAL, ORANGE]
    offsets = [-.24, 0, .24]

    fig, ax = plt.subplots(figsize=(8.4, 5.25))
    y = np.arange(len(domains))
    for encoding, color, offset in zip(encodings, colors, offsets):
        values = [float(next(r["addition_effect"] for r in rows
                             if r["source"] == d and r["encoding"] == encoding))
                  for d in domains]
        ax.barh(y + offset, values, height=.20, color=color, label=encoding, zorder=3)

    ax.axvline(0, color="#111827", linewidth=1.1)
    ax.set_yticks(y, [FIELDS[d] for d in domains])
    ax.invert_yaxis()
    ax.set_xlim(-.015, .34)
    ax.set_xlabel("Steering effect on technical-summary preference  →")
    ax.set_title("Transfer across invented domains", loc="left", color=NAVY,
                 fontsize=12.5, pad=10, fontweight="bold")
    ax.legend(title="Answer-label encoding", frameon=False, ncols=3, loc="lower right",
              fontsize=8.7, title_fontsize=8.7)
    style(ax)
    fig.subplots_adjust(left=.35, right=.97, top=.90, bottom=.15)
    save(fig, "exec_final_1_transfer")


def intervention_figure():
    additions = read(RESULTS / "qwen35_4b_fixed_direction_prospective/addition_by_domain.csv")
    ablations = read(RESULTS / "qwen35_4b_fixed_direction_prospective/ablation_by_domain.csv")
    intrinsic = read(RESULTS / "qwen35_4b_intrinsic_rank_minimal/evaluation_by_domain.csv")

    added = np.mean([float(r["addition_effect"]) for r in additions if r["encoding"] == "A/B"])
    natural_gap = np.mean([float(r["baseline_gap"]) for r in ablations])
    original_removed = np.mean([float(r["selected_layer_ablation_reduction"]) for r in ablations])
    selected_removed = float(next(r["aggregate_mean"] for r in intrinsic
                                  if r["test_set"] == "synthetic"
                                  and r["intervention"] == "selected_1d_mixture"))
    rank3_removed = float(next(r["aggregate_mean"] for r in intrinsic
                              if r["test_set"] == "synthetic"
                              and r["intervention"] == "rank3_subspace"))

    labels = ["Add original\ndirection", "Remove original\ndirection",
              "Remove selected\n1D direction*", "Remove rank-3\nsubspace*"]
    values = [added, original_removed, selected_removed, rank3_removed]
    colors = [BLUE, ORANGE, TEAL, NAVY]

    fig, ax = plt.subplots(figsize=(8.6, 5.15))
    x = np.arange(4)
    bars = ax.bar(x, values, color=colors, width=.64, zorder=3)
    ax.axhline(natural_gap, color=GREY, linewidth=1.5, linestyle="--", zorder=2)
    ax.text(3.48, natural_gap + .014, f"Natural audience gap = {natural_gap:.3f}",
            ha="right", color=GREY, fontsize=9.3, fontweight="bold")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value + .014, f"{value:.3f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold", color="#1F2937")
    ax.text(1, original_removed/2, "8%", ha="center", va="center",
            color="white", fontsize=10, fontweight="bold")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, .68)
    ax.set_ylabel("Change in technical-summary margin")
    ax.set_title("Steering and ablation effects", loc="left", color=NAVY,
                 fontsize=12.5, pad=10, fontweight="bold")
    ax.text(.99, -.22,
            "*Selected-1D and rank-3 results are preliminary: selection-matched shuffled controls were not run.",
            transform=ax.transAxes, ha="right", fontsize=8.6, color=GREY)
    style(ax, grid="y")
    fig.subplots_adjust(left=.12, right=.97, top=.88, bottom=.25)
    save(fig, "exec_final_2_interventions")


def controls_figure():
    rows = read(RESULTS / "qwen35_4b_fixed_direction_prospective/controls.csv")
    additions = read(RESULTS / "qwen35_4b_fixed_direction_prospective/addition_by_domain.csv")
    learned = np.mean([float(r["addition_effect"]) for r in additions if r["encoding"] == "A/B"])

    def values(kind):
        by_seed = {int(r["seed"]): float(r["aggregate_mean"])
                   for r in rows if r["kind"] == kind}
        return np.array([by_seed[k] for k in sorted(by_seed)])

    random = values("random")
    shuffled = values("permuted_labels")
    rng = np.random.default_rng(7)

    fig, ax = plt.subplots(figsize=(8.1, 5.2))
    for xpos, vals, color, label in [(0, random, BLUE, "Random directions"),
                                     (1, shuffled, RED, "Shuffled audience labels")]:
        jitter = rng.uniform(-.18, .18, len(vals))
        ax.scatter(xpos + jitter, vals, s=25, alpha=.58, color=color,
                   edgecolor="none", zorder=3)
        ax.plot([xpos-.24, xpos+.24], [np.median(vals), np.median(vals)],
                color="#111827", linewidth=2.2, zorder=4)

    ax.axhline(learned, color=NAVY, linewidth=2, linestyle="--", zorder=2,
               label=f"Learned direction ({learned:.3f})")
    ax.text(0, learned + .014, f"{(random >= learned).sum()}/100 reached learned effect",
            ha="center", fontsize=9.5, fontweight="bold", color=BLUE)
    ax.text(1, learned + .014, f"{(shuffled >= learned).sum()}/100 reached learned effect",
            ha="center", fontsize=9.5, fontweight="bold", color=RED)
    ax.set_xticks([0, 1], ["Random directions", "Directions learned after\nshuffling audience labels"])
    ax.set_xlim(-.55, 1.55)
    ax.set_ylabel("Mean steering effect across eight domains")
    ax.set_title("Control-direction steering effects", loc="left", color=NAVY,
                 fontsize=12.5, pad=10, fontweight="bold")
    ax.legend(frameon=False, loc="lower left")
    style(ax, grid="y")
    fig.subplots_adjust(left=.14, right=.97, top=.88, bottom=.20)
    save(fig, "exec_final_3_controls")


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
})

transfer_figure()
intervention_figure()
controls_figure()
