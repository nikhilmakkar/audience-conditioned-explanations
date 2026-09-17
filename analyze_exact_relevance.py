"""Summarize the frozen eight-domain exact-relevance experiment."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def read(path: Path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def exact_sign_flip_p(values):
    observed = mean(values)
    null = [
        mean(sign * value for sign, value in zip(signs, values))
        for signs in itertools.product((-1, 1), repeat=len(values))
    ]
    return sum(value >= observed for value in null) / len(null)


def encoding_domain_effects(rows, encoding):
    groups = defaultdict(list)
    for row in rows:
        if row["encoding"] == encoding:
            groups[row["heldout_source"]].append(float(row["addition_signed_effect"]))
    return {source: mean(values) for source, values in sorted(groups.items())}


def write(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lodo-dir", type=Path, required=True)
    parser.add_argument("--followup-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--model-label", default="Qwen3.5-4B")
    args = parser.parse_args()

    folds = read(args.lodo_dir / "lodo_test_summary.csv")
    reencoded = read(args.followup_dir / "reencoded_rows.csv")
    random = read(args.followup_dir / "random_addition_controls.csv")
    random_by_domain = defaultdict(list)
    for row in random:
        random_by_domain[row["heldout_source"]].append(float(row["mean_addition_effect"]))

    rows = []
    for fold in folds:
        source = fold["heldout_source"]
        learned = float(fold["addition_effect"])
        controls = random_by_domain[source]
        rows.append({
            "source": source,
            "selected_layer": fold["selected_layer"],
            "selected_position": fold["selected_tail_position"],
            "baseline_gap": fold["baseline_gap"],
            "addition_effect": learned,
            "single_layer_ablation_reduction": fold["selected_layer_ablation_reduction"],
            "all_layer_ablation_reduction": fold["all_layer_ablation_reduction"],
            "random_additions_ge_learned": sum(value >= learned for value in controls),
            "random_additions_n": len(controls),
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write(args.output_dir / "exact_relevance_folds.csv", rows)

    summaries = []
    for encoding in ("A/B", "X/Y", "1/2"):
        values = list(encoding_domain_effects(reencoded, encoding).values())
        leave_one_out = [mean(values[:i] + values[i + 1:]) for i in range(len(values))]
        summaries.append({
            "encoding": encoding,
            "n_domains": len(values),
            "mean_addition_effect": mean(values),
            "standard_error_across_domains": stdev(values) / math.sqrt(len(values)),
            "positive_domains": sum(value > 0 for value in values),
            "exact_domain_sign_flip_p": exact_sign_flip_p(values),
            "minimum_leave_one_domain_out_mean": min(leave_one_out),
            "maximum_leave_one_domain_out_mean": max(leave_one_out),
        })
    write(args.output_dir / "exact_relevance_encoding_summary.csv", summaries)

    import matplotlib.pyplot as plt

    sources = [row["source"] for row in rows]
    additions = [float(row["addition_effect"]) for row in rows]
    single = [float(row["single_layer_ablation_reduction"]) for row in rows]
    all_layer = [float(row["all_layer_ablation_reduction"]) for row in rows]
    x = list(range(len(sources)))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar([value - width for value in x], additions, width, label="Addition")
    ax.bar(x, single, width, label="Selected-layer ablation")
    ax.bar([value + width for value in x], all_layer, width, label="All-layer ablation")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, sources, rotation=30, ha="right")
    ax.set_ylabel("Signed technical-margin effect")
    ax.set_title(f"Exact-relevance interventions on held-out domains ({args.model_label})")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(args.output_dir / "exact_relevance_folds.png", dpi=180)


if __name__ == "__main__":
    main()
