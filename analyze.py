"""Create paired summaries and compact figures from experiment CSV files."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def read_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    rows = list(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sem(values):
    return stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def paired_behavior(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["source_id"], row["condition"], row["split"], row["profile_index"]
        )
        groups[key].append(float(row["technical_margin"]))
    result = []
    for key, values in sorted(groups.items()):
        if len(values) != 2:
            raise ValueError(f"Expected two A/B orders for {key}, got {len(values)}")
        result.append(
            dict(
                zip(
                    ("source_id", "condition", "split", "profile_index"), key
                )
            )
            | {"technical_margin": mean(values)}
        )
    return result


def paired_steering(rows):
    groups = defaultdict(list)
    for row in rows:
        profile = row["example_id"].rsplit(":", 1)[0]
        key = (
            row["direction"], row["source_id"], row["condition"],
            float(row["alpha"]), profile,
        )
        groups[key].append(float(row["technical_margin"]))
    result = []
    for key, values in sorted(groups.items()):
        if len(values) != 2:
            raise ValueError(f"Expected two A/B orders for {key}, got {len(values)}")
        result.append(
            dict(zip(("direction", "source_id", "condition", "alpha", "profile"), key))
            | {"technical_margin": mean(values)}
        )
    return result


def summarize_steering(pairs):
    groups = defaultdict(list)
    for row in pairs:
        key = row["direction"], row["source_id"], row["condition"], row["alpha"]
        groups[key].append(row["technical_margin"])
    return [
        dict(zip(("direction", "source_id", "condition", "alpha"), key))
        | {"n_profiles": len(values), "mean_technical_margin": mean(values), "standard_error": sem(values)}
        for key, values in sorted(groups.items())
    ]


def plot_behavior(pairs, output):
    import matplotlib.pyplot as plt

    conditions = ["expert", "academic_nonml", "neutral", "novice"]
    labels = ["ML expert", "Non-ML academic", "Neutral", "ML novice"]
    colors = {"train": "#4c78a8", "validation": "#f58518", "test": "#e45756"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, source in zip(axes, ("resnet", "transformer")):
        for x, condition in enumerate(conditions):
            values = [r for r in pairs if r["source_id"] == source and r["condition"] == condition]
            for offset, row in enumerate(values):
                axis.scatter(x + (offset - 3.5) * 0.035, row["technical_margin"],
                             color=colors[row["split"]], s=24, alpha=0.8)
            axis.scatter(x, mean([r["technical_margin"] for r in values]),
                         color="black", marker="_", s=180, linewidth=2)
        axis.axhline(0, color="gray", linestyle="--", linewidth=1)
        axis.set_title(source.capitalize())
        axis.set_xticks(range(4), labels, rotation=20, ha="right")
        axis.set_ylabel("Technical-summary log-probability margin")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_steering(summary, output):
    import matplotlib.pyplot as plt

    labels = {
        "ml_expert_minus_novice": "ML expert − novice",
        "nonml_academic_minus_novice": "Non-ML academic − novice",
    }
    colors = {"ml_expert_minus_novice": "#4c78a8", "nonml_academic_minus_novice": "#f58518"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for axis, source in zip(axes, ("resnet", "transformer")):
        for direction in labels:
            rows = sorted(
                [r for r in summary if r["source_id"] == source and r["condition"] == "novice" and r["direction"] == direction],
                key=lambda row: row["alpha"],
            )
            axis.errorbar(
                [r["alpha"] for r in rows], [r["mean_technical_margin"] for r in rows],
                yerr=[r["standard_error"] for r in rows], marker="o",
                label=labels[direction], color=colors[direction], capsize=3,
            )
        axis.axhline(0, color="gray", linestyle="--", linewidth=1)
        axis.set_title(source.capitalize())
        axis.set_xlabel("Steering coefficient alpha")
        axis.set_ylabel("Held-out novice technical margin")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior-dir", type=Path, required=True)
    parser.add_argument("--direction-dir", type=Path, required=True)
    args = parser.parse_args()

    behavior = paired_behavior(read_rows(args.behavior_dir / "behavior_rows.csv"))
    write_rows(args.behavior_dir / "behavior_profile_pairs.csv", behavior)
    plot_behavior(behavior, args.behavior_dir / "behavior_paired.png")

    steering = paired_steering(read_rows(args.direction_dir / "steering_rows.csv"))
    summary = summarize_steering(steering)
    write_rows(args.direction_dir / "steering_paired_summary.csv", summary)
    plot_steering(summary, args.direction_dir / "steering_novice.png")


if __name__ == "__main__":
    main()
