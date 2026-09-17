"""Summarize and plot the layer/token, permutation, and ecological controls."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODELS = {
    "Qwen2.5-3B": "qwen25_3b",
    "Qwen3.5-4B": "qwen35_4b",
}


def read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    summaries = []
    heatmaps = {}
    generation = {}
    for display, slug in MODELS.items():
        directory = RESULTS / f"{slug}_layer_token_sweep"
        candidates = read(directory / "candidate_sweep.csv")
        layers = max(int(row["layer"]) for row in candidates) + 1
        positions = sorted({int(row["tail_position"]) for row in candidates})
        matrix = np.full((layers, len(positions)), np.nan)
        for row in candidates:
            matrix[int(row["layer"]), positions.index(int(row["tail_position"]))] = float(
                row["selection_score"]
            )
        heatmaps[display] = (matrix, positions)

        tests = read(directory / "test_summary.csv")
        permutations = read(directory / "decoder_permutations.csv")
        true = next(row for row in permutations if row["is_true_labels"] == "True")
        coherence = read(directory / "wikitext_coherence.csv")
        learned_all = next(
            row for row in coherence if row["kind"] == "learned" and row["scope"] == "all"
        )
        selected = max(candidates, key=lambda row: float(row["selection_score"]))
        for test in tests:
            source = test["source_id"]
            accuracy = float(true[f"test_accuracy_{source}"])
            p_value = sum(
                float(row[f"test_accuracy_{source}"]) >= accuracy for row in permutations
            ) / len(permutations)
            summaries.append({
                "model": display,
                "source": source,
                "selected_layer": selected["layer"],
                "selected_tail_position": selected["tail_position"],
                "causal_selected_decoding_accuracy": test["decoding_accuracy"],
                "permutation_selected_layer": true["selected_layer"],
                "permutation_selected_tail_position": true["selected_tail_position"],
                "permutation_pipeline_test_accuracy": accuracy,
                "exact_permutation_p": p_value,
                "addition_effect": test["addition_effect"],
                "single_layer_ablation_fraction": (
                    float(test["selected_layer_ablation_reduction"])
                    / float(test["baseline_gap"])
                ),
                "all_layer_ablation_fraction": (
                    float(test["all_layer_ablation_reduction"])
                    / float(test["baseline_gap"])
                ),
                "wikitext_nll_increase_all_layer": learned_all["nll_increase"],
                "wikitext_kl_all_layer": learned_all["mean_next_token_kl"],
                "wikitext_top1_agreement_all_layer": learned_all["top1_agreement"],
            })

        rows = read(RESULTS / f"{slug}_steered_generation" / "generation_rows.csv")
        slopes = {}
        for source in ("resnet", "transformer"):
            for condition in ("expert", "novice"):
                relevant = [
                    row for row in rows
                    if row["source_id"] == source and row["condition"] == condition
                ]
                per_example = []
                for example_id in sorted({row["example_id"] for row in relevant}):
                    values = {
                        float(row["alpha"]): float(row["technical_term_count"])
                        for row in relevant if row["example_id"] == example_id
                    }
                    per_example.append((values[1.0] - values[-1.0]) / 2)
                slopes[(source, condition)] = sum(per_example) / len(per_example)
        generation[display] = slopes

    with (RESULTS / "deep_controls_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, (display, (matrix, positions)) in zip(axes[0], heatmaps.items()):
        image = axis.imshow(matrix.T, aspect="auto", origin="lower", cmap="coolwarm")
        axis.set_title(f"{display}: validation causal score")
        axis.set_xlabel("Layer")
        axis.set_ylabel("Tail position")
        axis.set_yticks(range(len(positions)), positions)
        fig.colorbar(image, ax=axis, shrink=0.8)

    axis = axes[1, 0]
    x = np.arange(4)
    labels, fractions = [], []
    for row in summaries:
        labels.append(f"{row['model'].split('-')[0]}\n{row['source']}")
        fractions.append(100 * float(row["all_layer_ablation_fraction"]))
    axis.bar(x, fractions, color=["#4c78a8", "#4c78a8", "#f58518", "#f58518"])
    axis.set_xticks(x, labels)
    axis.set_ylabel("Natural audience gap removed (%)")
    axis.set_title("All-layer, all-position ablation")
    axis.axhline(0, color="black", linewidth=0.7)

    axis = axes[1, 1]
    width = 0.18
    keys = [
        ("resnet", "expert"), ("resnet", "novice"),
        ("transformer", "expert"), ("transformer", "novice"),
    ]
    for offset, (display, slopes) in zip((-width / 2, width / 2), generation.items()):
        axis.bar(
            np.arange(len(keys)) + offset,
            [slopes[key] for key in keys],
            width=width,
            label=display,
        )
    axis.set_xticks(
        np.arange(len(keys)), ["ResNet\nexpert", "ResNet\nnovice", "Transformer\nexpert", "Transformer\nnovice"]
    )
    axis.axhline(0, color="black", linewidth=0.7)
    axis.set_ylabel("Technical-term slope per alpha")
    axis.set_title("Generated gists: inconsistent transfer")
    axis.legend(frameon=False)

    fig.savefig(RESULTS / "deep_controls.png", dpi=180)


if __name__ == "__main__":
    main()
