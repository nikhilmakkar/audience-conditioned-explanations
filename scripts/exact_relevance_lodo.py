"""Leave-one-domain-out causal test of an exact-relevance direction."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from experiment import build_examples, get_layers, label_token_ids, load_model, load_stimuli
from layer_token_sweep import (
    ablate_all_positions,
    add_all_positions,
    capture_tail,
    prepare_batch,
    run_batch,
)


POSITIVE = "domain_expert"
NEGATIVE = "matched_other_expert"


def direction(train, activations, layer, tail_index):
    positive = torch.stack([
        activations[e["example_id"]][layer, tail_index]
        for e in train if e["condition"] == POSITIVE
    ]).mean(0)
    negative = torch.stack([
        activations[e["example_id"]][layer, tail_index]
        for e in train if e["condition"] == NEGATIVE
    ]).mean(0)
    return positive - negative


def mean_source_gap(examples, margins):
    sources = sorted({e["source_id"] for e in examples})
    gaps = []
    for source in sources:
        values = {POSITIVE: [], NEGATIVE: []}
        for example, margin in zip(examples, margins):
            if example["source_id"] == source:
                values[example["condition"]].append(margin)
        gaps.append(
            sum(values[POSITIVE]) / len(values[POSITIVE])
            - sum(values[NEGATIVE]) / len(values[NEGATIVE])
        )
    return sum(gaps) / len(gaps)


def signed_effect(examples, baseline, changed):
    effects = []
    for example, before, after in zip(examples, baseline, changed):
        sign = 1.0 if example["condition"] == NEGATIVE else -1.0
        effects.append(sign * (after - before))
    return sum(effects) / len(effects)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--stimuli", type=Path, default=Path("data/stimuli_domain_matched.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tail-width", type=int, default=5)
    parser.add_argument("--random-controls", type=int, default=20)
    args = parser.parse_args()

    examples = build_examples(load_stimuli(args.stimuli))
    sources = sorted({e["source_id"] for e in examples})
    model, tokenizer = load_model(args.model)
    layers = list(get_layers(model))
    labels = label_token_ids(tokenizer)
    activations = {}
    for index, example in enumerate(examples, 1):
        activations[example["example_id"]] = capture_tail(
            model, tokenizer, example["prompt"], args.tail_width
        )
        print(f"[activations {index}/{len(examples)}] {example['example_id']}")

    candidate_rows = []
    test_rows = []
    random_rows = []
    for heldout in sources:
        train = [
            e for e in examples
            if e["source_id"] != heldout and e["split"] == "train"
        ]
        validation = [
            e for e in examples
            if e["source_id"] != heldout and e["split"] == "validation"
        ]
        test = [
            e for e in examples
            if e["source_id"] == heldout and e["split"] == "test"
        ]
        validation_inputs = prepare_batch(model, tokenizer, validation)
        baseline_validation = run_batch(
            model, validation_inputs, validation, labels
        )
        baseline_validation_gap = mean_source_gap(validation, baseline_validation)
        coefficients = torch.tensor([
            1.0 if e["condition"] == NEGATIVE else -1.0 for e in validation
        ])
        vectors = {}
        fold_candidates = []
        for layer_index, layer in enumerate(layers):
            for tail_index in range(args.tail_width):
                vector = direction(train, activations, layer_index, tail_index)
                vectors[(layer_index, tail_index)] = vector
                added = run_batch(
                    model, validation_inputs, validation, labels,
                    add_all_positions(layer, vector, coefficients),
                )
                unit = vector / vector.norm().clamp_min(1e-12)
                ablated = run_batch(
                    model, validation_inputs, validation, labels,
                    ablate_all_positions([layer], unit),
                )
                row = {
                    "heldout_source": heldout,
                    "layer": layer_index,
                    "tail_position": tail_index - args.tail_width,
                    "direction_norm": float(vector.norm()),
                    "validation_addition_effect": signed_effect(
                        validation, baseline_validation, added
                    ),
                    "validation_ablation_reduction": (
                        baseline_validation_gap - mean_source_gap(validation, ablated)
                    ),
                }
                row["selection_score"] = min(
                    row["validation_addition_effect"],
                    row["validation_ablation_reduction"],
                )
                fold_candidates.append(row)
                candidate_rows.append(row)
        selected = max(fold_candidates, key=lambda row: row["selection_score"])
        layer_index = int(selected["layer"])
        tail_index = int(selected["tail_position"]) + args.tail_width
        vector = vectors[(layer_index, tail_index)]
        unit = vector / vector.norm().clamp_min(1e-12)
        test_inputs = prepare_batch(model, tokenizer, test)
        baseline = run_batch(model, test_inputs, test, labels)
        base_gap = mean_source_gap(test, baseline)
        coefficients = torch.tensor([
            1.0 if e["condition"] == NEGATIVE else -1.0 for e in test
        ])
        added = run_batch(
            model, test_inputs, test, labels,
            add_all_positions(layers[layer_index], vector, coefficients),
        )
        selected_ablated = run_batch(
            model, test_inputs, test, labels,
            ablate_all_positions([layers[layer_index]], unit),
        )
        global_ablated = run_batch(
            model, test_inputs, test, labels,
            ablate_all_positions(layers, unit),
        )
        test_rows.append({
            "heldout_source": heldout,
            "selected_layer": layer_index,
            "selected_tail_position": selected["tail_position"],
            "validation_selection_score": selected["selection_score"],
            "baseline_gap": base_gap,
            "addition_effect": signed_effect(test, baseline, added),
            "selected_layer_ablation_reduction": (
                base_gap - mean_source_gap(test, selected_ablated)
            ),
            "all_layer_ablation_reduction": (
                base_gap - mean_source_gap(test, global_ablated)
            ),
        })

        learned_addition = test_rows[-1]["addition_effect"]
        learned_ablation = test_rows[-1]["all_layer_ablation_reduction"]
        random_rows.append({
            "heldout_source": heldout, "kind": "learned", "seed": -1,
            "addition_effect": learned_addition,
            "all_layer_ablation_reduction": learned_ablation,
        })
        for seed in range(args.random_controls):
            generator = torch.Generator().manual_seed(seed)
            random = torch.randn(vector.shape, generator=generator)
            random *= vector.norm() / random.norm()
            random_unit = random / random.norm()
            random_added = run_batch(
                model, test_inputs, test, labels,
                add_all_positions(layers[layer_index], random, coefficients),
            )
            random_ablated = run_batch(
                model, test_inputs, test, labels,
                ablate_all_positions(layers, random_unit),
            )
            random_rows.append({
                "heldout_source": heldout, "kind": "random", "seed": seed,
                "addition_effect": signed_effect(test, baseline, random_added),
                "all_layer_ablation_reduction": (
                    base_gap - mean_source_gap(test, random_ablated)
                ),
            })
        print("fold", test_rows[-1])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "candidate_sweep.csv", candidate_rows)
    write_rows(args.output_dir / "lodo_test_summary.csv", test_rows)
    write_rows(args.output_dir / "random_controls.csv", random_rows)


if __name__ == "__main__":
    main()
