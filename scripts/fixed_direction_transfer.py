"""Prospective fixed-direction transfer to unfamiliar synthetic methods."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import torch

from exact_relevance_lodo import NEGATIVE, POSITIVE, mean_source_gap, signed_effect
from experiment import build_examples, get_layers, load_model, load_stimuli
from layer_token_sweep import (
    ablate_all_positions,
    add_all_positions,
    capture_tail,
    prepare_batch,
    run_batch,
)
from reencoding_control import ids_for, relabel


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def group_activations(examples, activations, layer):
    grouped = defaultdict(list)
    for example in examples:
        key = example["source_id"], example["condition"], example["profile_index"]
        grouped[key].append(activations[example["example_id"]][layer, -1])
    return {key: torch.stack(values).mean(0) for key, values in grouped.items()}


def true_direction(grouped):
    positive = torch.stack([
        value for key, value in grouped.items() if key[1] == POSITIVE
    ]).mean(0)
    negative = torch.stack([
        value for key, value in grouped.items() if key[1] == NEGATIVE
    ]).mean(0)
    return positive - negative


def permuted_direction(grouped, seed):
    values = [grouped[key] for key in sorted(grouped)]
    generator = torch.Generator().manual_seed(seed)
    assignment = torch.randperm(len(values), generator=generator)
    half = len(values) // 2
    positive = torch.stack([values[index] for index in assignment[:half]]).mean(0)
    negative = torch.stack([values[index] for index in assignment[half:]]).mean(0)
    return positive - negative


def domain_effects(examples, baseline, changed):
    grouped = defaultdict(list)
    for example, before, after in zip(examples, baseline, changed):
        sign = 1.0 if example["condition"] == NEGATIVE else -1.0
        grouped[example["source_id"]].append(sign * (after - before))
    return {source: mean(values) for source, values in sorted(grouped.items())}


def exact_sign_flip_p(values):
    observed = mean(values)
    null = [
        mean(sign * value for sign, value in zip(signs, values))
        for signs in itertools.product((-1, 1), repeat=len(values))
    ]
    return sum(value >= observed for value in null) / len(null)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--random-controls", type=int, default=100)
    parser.add_argument("--permutation-controls", type=int, default=100)
    args = parser.parse_args()

    discovery = [
        example for example in build_examples(load_stimuli(args.discovery))
        if example["split"] == "train"
    ]
    original_test = build_examples(load_stimuli(args.test))
    model, tokenizer = load_model(args.model)
    layers = list(get_layers(model))
    if not 0 <= args.layer < len(layers):
        raise ValueError(f"layer {args.layer} outside 0..{len(layers) - 1}")

    activations = {}
    for index, example in enumerate(discovery, 1):
        activations[example["example_id"]] = capture_tail(
            model, tokenizer, example["prompt"], 1
        )
        print(f"[discovery activations {index}/{len(discovery)}] {example['example_id']}")
    grouped = group_activations(discovery, activations, args.layer)
    vector = true_direction(grouped)
    unit = vector / vector.norm().clamp_min(1e-12)

    rows = []
    encoding_summaries = []
    ablation_rows = []
    original_inputs = None
    original_baseline = None
    original_coefficients = None
    for left, right in (("A", "B"), ("X", "Y"), ("1", "2")):
        examples = [relabel(example, left, right) for example in original_test]
        label_ids = ids_for(tokenizer, (left, right))
        inputs = prepare_batch(model, tokenizer, examples)
        baseline = run_batch(model, inputs, examples, label_ids)
        coefficients = torch.tensor([
            1.0 if example["condition"] == NEGATIVE else -1.0
            for example in examples
        ])
        changed = run_batch(
            model,
            inputs,
            examples,
            label_ids,
            add_all_positions(layers[args.layer], vector, coefficients),
        )
        effects = domain_effects(examples, baseline, changed)
        values = list(effects.values())
        encoding = f"{left}/{right}"
        encoding_summaries.append({
            "encoding": encoding,
            "n_domains": len(values),
            "mean_addition_effect": mean(values),
            "standard_error": stdev(values) / math.sqrt(len(values)),
            "positive_domains": sum(value > 0 for value in values),
            "exact_sign_flip_p": exact_sign_flip_p(values),
        })
        for source, effect in effects.items():
            rows.append({
                "encoding": encoding,
                "source": source,
                "addition_effect": effect,
            })
        if encoding == "A/B":
            original_inputs = inputs
            original_baseline = baseline
            original_coefficients = coefficients

    selected_ablated = run_batch(
        model,
        original_inputs,
        original_test,
        ids_for(tokenizer, ("A", "B")),
        ablate_all_positions([layers[args.layer]], unit),
    )
    all_ablated = run_batch(
        model,
        original_inputs,
        original_test,
        ids_for(tokenizer, ("A", "B")),
        ablate_all_positions(layers, unit),
    )
    for source in sorted({example["source_id"] for example in original_test}):
        indices = [
            index for index, example in enumerate(original_test)
            if example["source_id"] == source
        ]
        examples = [original_test[index] for index in indices]
        baseline = [original_baseline[index] for index in indices]
        selected = [selected_ablated[index] for index in indices]
        all_layers = [all_ablated[index] for index in indices]
        baseline_gap = mean_source_gap(examples, baseline)
        ablation_rows.append({
            "source": source,
            "baseline_gap": baseline_gap,
            "selected_layer_ablation_reduction": (
                baseline_gap - mean_source_gap(examples, selected)
            ),
            "all_layer_ablation_reduction": (
                baseline_gap - mean_source_gap(examples, all_layers)
            ),
        })

    control_rows = []
    for kind, count in (
        ("random", args.random_controls),
        ("permuted_labels", args.permutation_controls),
    ):
        for seed in range(count):
            if kind == "random":
                generator = torch.Generator().manual_seed(seed)
                control = torch.randn(vector.shape, generator=generator)
                control *= vector.norm() / control.norm()
            else:
                control = permuted_direction(grouped, seed)
                control *= vector.norm() / control.norm().clamp_min(1e-12)
            changed = run_batch(
                model,
                original_inputs,
                original_test,
                ids_for(tokenizer, ("A", "B")),
                add_all_positions(
                    layers[args.layer], control, original_coefficients
                ),
            )
            effects = domain_effects(original_test, original_baseline, changed)
            for source, effect in effects.items():
                control_rows.append({
                    "kind": kind,
                    "seed": seed,
                    "source": source,
                    "addition_effect": effect,
                    "aggregate_mean": mean(effects.values()),
                })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "addition_by_domain.csv", rows)
    write_rows(args.output_dir / "encoding_summary.csv", encoding_summaries)
    write_rows(args.output_dir / "ablation_by_domain.csv", ablation_rows)
    write_rows(args.output_dir / "controls.csv", control_rows)
    torch.save({"layer": args.layer, "vector": vector}, args.output_dir / "fixed_direction.pt")
    (args.output_dir / "config.json").write_text(json.dumps({
        "model": args.model,
        "discovery": str(args.discovery),
        "test": str(args.test),
        "layer": args.layer,
        "random_controls": args.random_controls,
        "permutation_controls": args.permutation_controls,
    }, indent=2) + "\n")
    for summary in encoding_summaries:
        print(summary)


if __name__ == "__main__":
    main()
