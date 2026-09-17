"""One purpose-built figure for the MATS executive-summary narrative."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results/qwen35_4b_fixed_direction_prospective"
OUT = ROOT / "report_assets/exec_summary_core_result.png"


def main() -> None:
    with (RESULTS / "addition_by_domain.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    encodings = ["A/B", "X/Y", "1/2"]
    values = {
        encoding: [
            float(row["addition_effect"])
            for row in rows
            if row["encoding"] == encoding
        ]
        for encoding in encodings
    }

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.patch.set_facecolor("white")

    # Panel A: fixed direction, all held-out domains, all answer encodings.
    ax = axes[0]
    colors = ["#176B87", "#4F9DA6", "#8CC0C7"]
    x = np.arange(len(encodings))
    means = [np.mean(values[encoding]) for encoding in encodings]
    ax.bar(x, means, color=colors, width=0.58, zorder=2)
    offsets = np.linspace(-0.16, 0.16, 8)
    for i, encoding in enumerate(encodings):
        ax.scatter(
            np.full(8, i) + offsets,
            values[encoding],
            s=24,
            color="#17365D",
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        ax.text(i, means[i] + 0.015, "8/8 positive", ha="center", va="bottom", fontsize=9, weight="bold")
    ax.axhline(0, color="#6B7280", linewidth=0.9)
    ax.set_xticks(x, encodings)
    ax.set_ylim(-0.015, 0.35)
    ax.set_ylabel("Change in technical-summary preference")
    ax.set_title("A. One frozen direction steers every held-out method", loc="left", weight="bold")
    ax.text(
        0.0, -0.22,
        "Learned on 8 real ML topics → tested here on 8 invented methods\n"
        "0/100 random directions matched it; 6/100 shuffled-label directions did",
        transform=ax.transAxes,
        fontsize=9,
        color="#374151",
        va="top",
    )

    # Panel B: how much of the natural gap different ablations remove.
    ax = axes[1]
    labels = ["Mean-difference\ndirection", "Optimized 1D\n(exploratory)", "Rank-3\nsubspace"]
    reductions = np.array([0.047, 0.156, 0.180])
    baseline_gap = 0.594
    fractions = 100 * reductions / baseline_gap
    bars = ax.bar(np.arange(3), fractions, color=["#D97706", "#4F9DA6", "#17365D"], width=0.62)
    for bar, fraction in zip(bars, fractions):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            fraction + 1.5,
            f"{fraction:.0f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )
    ax.set_xticks(np.arange(3), labels)
    ax.set_ylim(0, 38)
    ax.set_ylabel("Natural audience gap removed")
    ax.set_title("B. Successful steering was a weak causal explanation", loc="left", weight="bold")
    ax.text(
        0.0, -0.22,
        "A better 1D mixture nearly matches the full rank-3 subspace:\n"
        "variance explained and causal leverage are not the same objective",
        transform=ax.transAxes,
        fontsize=9,
        color="#374151",
        va="top",
    )

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.8, zorder=0)

    fig.suptitle(
        "The direction looked convincing—until I tried to remove the behaviour",
        x=0.06,
        y=1.02,
        ha="left",
        fontsize=16,
        weight="bold",
        color="#111827",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.29, wspace=0.30)
    fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
    print(OUT)


if __name__ == "__main__":
    main()
