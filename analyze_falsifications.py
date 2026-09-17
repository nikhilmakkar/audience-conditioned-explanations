"""Summarize and plot the three main falsification experiments."""

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SOURCES = ["resnet", "transformer", "unet", "kriging"]


def read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def sem(values):
    return stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def paired_behavior(path):
    groups = defaultdict(list)
    for row in read(path):
        key = row["source_id"], row["condition"], int(row["profile_index"])
        groups[key].append(float(row["technical_margin"]))
    assert all(len(values) == 2 for values in groups.values())
    return {key: mean(values) for key, values in groups.items()}


def main():
    broad = paired_behavior(
        RESULTS / "qwen25_3b_matched_behavior" / "behavior_rows.csv"
    )
    exact = paired_behavior(
        RESULTS / "qwen25_3b_domain_matched_behavior" / "behavior_rows.csv"
    )
    generation = read(
        RESULTS / "qwen25_3b_domain_matched_generation_180" / "generation_rows.csv"
    )
    random_rows = read(
        RESULTS / "qwen25_3b_random_baseline" / "random_steering_summary.csv"
    )

    broad_effects = {}
    exact_effects = {}
    generation_effects = {}
    for source in SOURCES:
        ml_cue = [
            (
                broad[(source, "relevant_researcher", index)]
                + broad[(source, "relevant_practitioner", index)]
                - broad[(source, "irrelevant_researcher", index)]
                - broad[(source, "irrelevant_practitioner", index)]
            ) / 2
            for index in range(9)
        ]
        broad_effects[source] = (mean(ml_cue), sem(ml_cue))
        domain = [
            exact[(source, "domain_expert", index)]
            - exact[(source, "matched_other_expert", index)]
            for index in range(8)
        ]
        exact_effects[source] = (mean(domain), sem(domain))

        rows = {
            (row["condition"], int(row["profile_index"])): row
            for row in generation if row["source_id"] == source
        }
        term_effect = [
            float(rows[("domain_expert", index)]["technical_term_count"])
            - float(rows[("matched_other_expert", index)]["technical_term_count"])
            for index in range(8)
        ]
        generation_effects[source] = (mean(term_effect), sem(term_effect))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    x = range(len(SOURCES))
    labels = ["ResNet", "Transformer", "U-Net", "Kriging"]
    for axis, effects, title, ylabel in [
        (axes[0], broad_effects, "Coarse ML cue", "Technical-margin difference"),
        (axes[1], exact_effects, "Exact domain relevance", "Technical-margin difference"),
        (axes[2], generation_effects, "Free-form term retention", "Technical-term count difference"),
    ]:
        axis.errorbar(
            list(x), [effects[s][0] for s in SOURCES],
            yerr=[effects[s][1] for s in SOURCES], fmt="o", capsize=4,
            color="#4c78a8",
        )
        axis.axhline(0, color="gray", linestyle="--", linewidth=1)
        axis.set_xticks(list(x), labels, rotation=25, ha="right")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(RESULTS / "falsification_summary.png", dpi=200)
    plt.close(fig)

    with (RESULTS / "falsification_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_id", "coarse_ml_cue", "coarse_se", "exact_domain", "exact_se", "generation_terms", "generation_se"],
        )
        writer.writeheader()
        for source in SOURCES:
            writer.writerow(
                {
                    "source_id": source,
                    "coarse_ml_cue": broad_effects[source][0],
                    "coarse_se": broad_effects[source][1],
                    "exact_domain": exact_effects[source][0],
                    "exact_se": exact_effects[source][1],
                    "generation_terms": generation_effects[source][0],
                    "generation_se": generation_effects[source][1],
                }
            )

    for source in ("resnet", "transformer"):
        learned = float(next(
            row["mean_slope"] for row in random_rows
            if row["source_id"] == source and row["kind"] == "learned"
        ))
        random = [
            float(row["mean_slope"]) for row in random_rows
            if row["source_id"] == source and row["kind"] == "random"
        ]
        print(
            source, "learned slope", round(learned, 3),
            "random mean", round(mean(random), 3),
            "random range", (round(min(random), 3), round(max(random), 3)),
        )


if __name__ == "__main__":
    main()
