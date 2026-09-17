"""Test whether a steering effect survives changing forced-choice label tokens."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from experiment import build_examples, get_layers, load_model, load_stimuli
from layer_token_sweep import (
    ablate_all_positions,
    add_all_positions,
    audience_gap,
    prepare_batch,
    run_batch,
    signed_addition_effect,
)


def relabel(example, left, right):
    prompt = example["prompt"]
    prompt = prompt.replace("exactly A or B", f"exactly {left} or {right}")
    prompt = prompt.replace("Summary A:", f"Summary {left}:")
    prompt = prompt.replace("Summary B:", f"Summary {right}:")
    changed = dict(example)
    changed["prompt"] = prompt
    changed["technical_label"] = left if example["technical_label"] == "A" else right
    return changed


def ids_for(tokenizer, labels):
    result = {}
    for label in labels:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Label {label!r} is not one token: {ids}")
        result[label] = ids[0]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--stimuli", type=Path, default=Path("stimuli.json"))
    parser.add_argument("--direction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    saved = torch.load(args.direction, map_location="cpu", weights_only=True)
    layer_index = int(saved["selected_layer"])
    vector = saved["selected_direction"].float()
    unit = vector / vector.norm().clamp_min(1e-12)
    model, tokenizer = load_model(args.model)
    layers = list(get_layers(model))
    original = [
        e for e in build_examples(load_stimuli(args.stimuli))
        if e["condition"] in {"expert", "novice"} and e["split"] == "test"
    ]

    rows = []
    for left, right in (("A", "B"), ("X", "Y"), ("1", "2")):
        label_ids = ids_for(tokenizer, (left, right))
        for source in ("resnet", "transformer"):
            examples = [
                relabel(e, left, right) for e in original if e["source_id"] == source
            ]
            inputs = prepare_batch(model, tokenizer, examples)
            baseline = run_batch(model, inputs, examples, label_ids)
            coefficients = torch.tensor([
                1.0 if e["condition"] == "novice" else -1.0 for e in examples
            ])
            added = run_batch(
                model, inputs, examples, label_ids,
                add_all_positions(layers[layer_index], vector, coefficients),
            )
            ablated = run_batch(
                model, inputs, examples, label_ids,
                ablate_all_positions(layers, unit),
            )
            baseline_gap = audience_gap(examples, baseline)
            ablated_gap = audience_gap(examples, ablated)
            rows.append({
                "encoding": f"{left}/{right}",
                "source_id": source,
                "baseline_gap": baseline_gap,
                "addition_effect": signed_addition_effect(examples, baseline, added),
                "all_layer_ablation_gap": ablated_gap,
                "all_layer_ablation_reduction": baseline_gap - ablated_gap,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
