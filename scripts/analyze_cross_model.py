"""Plot coarse versus exact audience effects for both evaluated models."""

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["resnet", "transformer", "unet", "kriging"]
MODELS = {"qwen25_3b": "Qwen2.5-3B", "qwen35_4b": "Qwen3.5-4B"}


def paired_behavior(path):
    groups = defaultdict(list)
    with path.open() as handle:
        for row in csv.DictReader(handle):
            key = row["source_id"], row["condition"], int(row["profile_index"])
            groups[key].append(float(row["technical_margin"]))
    assert all(len(values) == 2 for values in groups.values())
    return {key: mean(values) for key, values in groups.items()}


def sem(values):
    return stdev(values) / math.sqrt(len(values))


def effects(model):
    broad = paired_behavior(
        ROOT / f"results/{model}_matched_behavior/behavior_rows.csv"
    )
    exact = paired_behavior(
        ROOT / f"results/{model}_domain_matched_behavior/behavior_rows.csv"
    )
    result = {}
    for source in SOURCES:
        coarse_values = [
            (
                broad[(source, "relevant_researcher", index)]
                + broad[(source, "relevant_practitioner", index)]
                - broad[(source, "irrelevant_researcher", index)]
                - broad[(source, "irrelevant_practitioner", index)]
            ) / 2
            for index in range(9)
        ]
        exact_values = [
            exact[(source, "domain_expert", index)]
            - exact[(source, "matched_other_expert", index)]
            for index in range(8)
        ]
        result[source] = {
            "coarse": (mean(coarse_values), sem(coarse_values)),
            "exact": (mean(exact_values), sem(exact_values)),
        }
    return result


def main():
    values = {model: effects(model) for model in MODELS}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    labels = ["ResNet", "Transformer", "U-Net", "Kriging"]
    colors = {"coarse": "#4c78a8", "exact": "#e45756"}
    for axis, (model, model_label) in zip(axes, MODELS.items()):
        for offset, kind in ((-0.09, "coarse"), (0.09, "exact")):
            axis.errorbar(
                [index + offset for index in range(4)],
                [values[model][source][kind][0] for source in SOURCES],
                yerr=[values[model][source][kind][1] for source in SOURCES],
                fmt="o", capsize=3, color=colors[kind], label=kind.capitalize(),
            )
        axis.axhline(0, color="gray", linestyle="--", linewidth=1)
        axis.set_title(model_label)
        axis.set_xticks(range(4), labels, rotation=25, ha="right")
        axis.set_ylabel("Technical-margin difference")
        axis.legend(frameon=False)
    fig.suptitle("Coarse audience cues dominate exact domain relevance")
    fig.tight_layout()
    fig.savefig(ROOT / "results/cross_model_behavior.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
