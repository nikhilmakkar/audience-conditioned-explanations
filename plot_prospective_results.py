"""Create a compact figure for the prospective and robustness experiments."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def avg_se(values):
    return mean(values), stdev(values)/math.sqrt(len(values)) if len(values) > 1 else 0


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))

    dose = read(RESULTS/"qwen35_4b_forced_choice_dose"/"dose_rows.csv")
    ax = axes[0, 0]
    for condition, label, color in (("domain_expert", "domain expert", "#176B87"),
                                     ("matched_other_expert", "other-domain expert", "#D05A3A")):
        xs, ys, es = [], [], []
        for alpha in (-2, -1, 0, 1, 2):
            vals = [float(r["technical_margin"]) for r in dose
                    if r["encoding"] == "A/B" and r["condition"] == condition
                    and float(r["alpha"]) == alpha]
            y, e = avg_se(vals); xs.append(alpha); ys.append(y); es.append(e)
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=label, color=color)
    ax.set(title="A. Qwen forced-choice dose response", xlabel="steering coefficient",
           ylabel="technical - accessible log-probability")
    ax.legend(frameon=False, fontsize=8)
    ax.axvline(0, color="0.8", linewidth=1)

    generation = read(RESULTS/"qwen35_4b_fixed_direction_generation"/"generation_by_domain.csv")
    ax = axes[0, 1]
    rows = [r for r in generation if r["metric"] == "term_coverage" and r["condition"] == "pooled"]
    xs, ys, es = [], [], []
    for alpha in (-2, -1, 0, 1, 2):
        vals = [float(r[f"alpha_{alpha:g}"]) for r in rows]
        y, e = avg_se(vals); xs.append(alpha); ys.append(y); es.append(e)
    ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, color="#6C4AB6")
    ax.set(title="B. Free generation: frozen term metric", xlabel="steering coefficient",
           ylabel="source-term coverage")
    ax.axvline(0, color="0.8", linewidth=1)

    ax = axes[1, 0]
    qcontrols = read(RESULTS/"qwen35_4b_fixed_direction_prospective"/"controls.csv")
    pcontrols = read(RESULTS/"phi35_38b_fixed_direction_prospective"/"controls.csv")
    qlearn = mean(float(r["addition_effect"]) for r in read(
        RESULTS/"qwen35_4b_fixed_direction_prospective"/"addition_by_domain.csv") if r["encoding"] == "A/B")
    plearn = mean(float(r["addition_effect"]) for r in read(
        RESULTS/"phi35_38b_fixed_direction_prospective"/"addition_by_domain.csv") if r["encoding"] == "A/B")
    counts = []
    for controls, learned in ((qcontrols, qlearn), (pcontrols, plearn)):
        for kind in ("random", "permuted_labels"):
            by_seed = {int(r["seed"]): float(r["aggregate_mean"])
                       for r in controls if r["kind"] == kind}
            counts.append(sum(value >= learned for value in by_seed.values()))
    positions = [0, 1, 3, 4]
    ax.bar(positions, counts, color=["#176B87", "#76A5AF", "#D05A3A", "#E6A57E"])
    ax.axhline(5, color="black", linestyle="--", linewidth=1, label="frozen maximum")
    ax.set_xticks(positions, ["Qwen\nrandom", "Qwen\nshuffled", "Phi\nrandom", "Phi\nshuffled"])
    ax.set(title="C. Controls matching learned aggregate", ylabel="count out of 100", ylim=(0, 21))
    ax.legend(frameon=False, fontsize=8)

    curve = read(RESULTS/"qwen35_4b_direction_learning_curve"/"learning_curve_summary.csv")
    ax = axes[1, 1]
    sizes = [1, 2, 4, 6, 8]
    effects = [float(next(r["mean"] for r in curve if int(r["train_domains"]) == size
                          and r["metric"] == "mean_effect")) for size in sizes]
    cosines = [float(next(r["mean"] for r in curve if int(r["train_domains"]) == size
                          and r["metric"] == "cosine_to_full")) for size in sizes]
    ax.plot(sizes, effects, marker="o", color="#176B87", label="held-out effect")
    ax.set(title="D. Discovery-domain learning curve", xlabel="discovery domains",
           ylabel="mean addition effect", ylim=(0, 0.2))
    twin = ax.twinx()
    twin.plot(sizes, cosines, marker="s", color="#D05A3A", label="cosine to full")
    twin.set_ylabel("cosine to full direction", color="#D05A3A")
    twin.set_ylim(0, 1.05)
    lines = ax.lines + twin.lines
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=8, loc="lower right")

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    output = RESULTS/"prospective_results.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
