"""Frozen-selection follow-up for leave-one-domain-out relevance steering."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import torch

from exact_relevance_lodo import NEGATIVE, POSITIVE, direction
from experiment import build_examples, get_layers, load_model, load_stimuli
from layer_token_sweep import (
    ablate_all_positions,
    add_all_positions,
    capture_tail,
    prepare_batch,
    run_batch,
)
from reencoding_control import ids_for, relabel


def read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_effects(rows, field):
    by_order = defaultdict(list)
    for row in rows:
        key = row["heldout_source"], row["condition"], int(row["profile_index"])
        by_order[key].append(float(row[field]))
    condition_values = {key: mean(values) for key, values in by_order.items()}
    effects = []
    for source in sorted({key[0] for key in condition_values}):
        indices = sorted({key[2] for key in condition_values if key[0] == source})
        for index in indices:
            effects.append((
                condition_values[(source, POSITIVE, index)]
                + condition_values[(source, NEGATIVE, index)]
            ) / 2)
    return effects


def domain_effects(rows, field):
    """Aggregate repeated carrier profiles before cross-domain inference."""
    carrier_keys = sorted({
        (row["heldout_source"], int(row["profile_index"])) for row in rows
    })
    by_domain = defaultdict(list)
    for value, (source, _index) in zip(paired_effects(rows, field), carrier_keys):
        by_domain[source].append(value)
    return [mean(by_domain[source]) for source in sorted(by_domain)]


def sign_flip_p(values):
    observed = mean(values)
    null = [
        mean(sign * value for sign, value in zip(signs, values))
        for signs in itertools.product((-1, 1), repeat=len(values))
    ]
    return sum(value >= observed for value in null) / len(null)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--stimuli", type=Path, default=Path("data/stimuli_domain_matched.json"))
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tail-width", type=int, default=5)
    parser.add_argument("--random-controls", type=int, default=100)
    args = parser.parse_args()

    selections = {row["heldout_source"]: row for row in read(args.selections)}
    examples = build_examples(load_stimuli(args.stimuli))
    model, tokenizer = load_model(args.model)
    layers = list(get_layers(model))
    activations = {}
    for index, example in enumerate(examples, 1):
        activations[example["example_id"]] = capture_tail(
            model, tokenizer, example["prompt"], args.tail_width
        )
        print(f"[activations {index}/{len(examples)}] {example['example_id']}")

    directions = {}
    rows = []
    random_rows = []
    for heldout, selection in selections.items():
        layer_index = int(selection["selected_layer"])
        tail_index = int(selection["selected_tail_position"]) + args.tail_width
        train = [
            e for e in examples
            if e["source_id"] != heldout and e["split"] == "train"
        ]
        vector = direction(train, activations, layer_index, tail_index)
        unit = vector / vector.norm().clamp_min(1e-12)
        directions[heldout] = {
            "layer": layer_index,
            "tail_position": int(selection["selected_tail_position"]),
            "vector": vector,
        }
        original_test = [
            e for e in examples
            if e["source_id"] == heldout and e["split"] == "test"
        ]
        for left, right in (("A", "B"), ("X", "Y"), ("1", "2")):
            test = [relabel(e, left, right) for e in original_test]
            label_ids = ids_for(tokenizer, (left, right))
            inputs = prepare_batch(model, tokenizer, test)
            baseline = run_batch(model, inputs, test, label_ids)
            coefficients = torch.tensor([
                1.0 if e["condition"] == NEGATIVE else -1.0 for e in test
            ])
            added = run_batch(
                model, inputs, test, label_ids,
                add_all_positions(layers[layer_index], vector, coefficients),
            )
            ablated = run_batch(
                model, inputs, test, label_ids,
                ablate_all_positions(layers, unit),
            )
            for example, before, after, after_ablation in zip(
                test, baseline, added, ablated
            ):
                sign = 1.0 if example["condition"] == NEGATIVE else -1.0
                rows.append({
                    "encoding": f"{left}/{right}",
                    "heldout_source": heldout,
                    "example_id": example["example_id"],
                    "condition": example["condition"],
                    "profile_index": example["profile_index"],
                    "technical_label": example["technical_label"],
                    "baseline_margin": before,
                    "addition_signed_effect": sign * (after - before),
                    "ablated_margin": after_ablation,
                })

        # Random-direction specificity is evaluated only under the original A/B encoding.
        test = original_test
        label_ids = ids_for(tokenizer, ("A", "B"))
        inputs = prepare_batch(model, tokenizer, test)
        baseline = run_batch(model, inputs, test, label_ids)
        coefficients = torch.tensor([
            1.0 if e["condition"] == NEGATIVE else -1.0 for e in test
        ])
        for seed in range(args.random_controls):
            generator = torch.Generator().manual_seed(seed)
            random = torch.randn(vector.shape, generator=generator)
            random *= vector.norm() / random.norm()
            changed = run_batch(
                model, inputs, test, label_ids,
                add_all_positions(layers[layer_index], random, coefficients),
            )
            effects = [
                (1.0 if e["condition"] == NEGATIVE else -1.0) * (after - before)
                for e, before, after in zip(test, baseline, changed)
            ]
            random_rows.append({
                "heldout_source": heldout,
                "seed": seed,
                "mean_addition_effect": mean(effects),
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "reencoded_rows.csv", rows)
    write_rows(args.output_dir / "random_addition_controls.csv", random_rows)
    torch.save(directions, args.output_dir / "frozen_fold_directions.pt")

    summaries = []
    for encoding in ("A/B", "X/Y", "1/2"):
        relevant = [row for row in rows if row["encoding"] == encoding]
        effects = domain_effects(relevant, "addition_signed_effect")
        summaries.append({
            "encoding": encoding,
            "n_domains": len(effects),
            "mean_addition_effect": mean(effects),
            "standard_error": stdev(effects) / math.sqrt(len(effects)),
            "positive_domains": sum(value > 0 for value in effects),
            "exact_sign_flip_p": sign_flip_p(effects),
        })
    write_rows(args.output_dir / "reencoded_summary.csv", summaries)
    for row in summaries:
        print(row)


if __name__ == "__main__":
    main()
