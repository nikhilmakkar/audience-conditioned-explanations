"""Ablate a saved fixed direction on balanced A/B held-out examples."""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path
from statistics import mean

import torch

from exact_relevance_lodo import mean_source_gap
from experiment import build_examples, get_layers, load_model, load_stimuli
from layer_token_sweep import ablate_all_positions, prepare_batch, run_batch
from reencoding_control import ids_for


def signflip(values):
    observed = mean(values)
    null = [mean(s*v for s, v in zip(signs, values))
            for signs in itertools.product((-1, 1), repeat=len(values))]
    return sum(value >= observed - 1e-12 for value in null) / len(null)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--stimuli", type=Path, required=True)
    parser.add_argument("--direction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    saved = torch.load(args.direction, map_location="cpu", weights_only=True)
    layer_index = int(saved.get("selected_layer", saved.get("layer")))
    vector = saved.get("selected_direction", saved.get("vector")).float()
    unit = vector/vector.norm().clamp_min(1e-12)
    model, tokenizer = load_model(args.model)
    layers = list(get_layers(model))
    examples = build_examples(load_stimuli(args.stimuli))
    inputs = prepare_batch(model, tokenizer, examples)
    label_ids = ids_for(tokenizer, ("A", "B"))
    baseline = run_batch(model, inputs, examples, label_ids)
    selected = run_batch(model, inputs, examples, label_ids,
                         ablate_all_positions([layers[layer_index]], unit))
    all_layers = run_batch(model, inputs, examples, label_ids,
                           ablate_all_positions(layers, unit))
    rows = []
    for source in sorted({e["source_id"] for e in examples}):
        indices = [i for i, e in enumerate(examples) if e["source_id"] == source]
        subset = [examples[i] for i in indices]
        gap = mean_source_gap(subset, [baseline[i] for i in indices])
        selected_gap = mean_source_gap(subset, [selected[i] for i in indices])
        all_gap = mean_source_gap(subset, [all_layers[i] for i in indices])
        rows.append({"source": source, "baseline_gap": gap,
                     "selected_layer_ablation_reduction": gap-selected_gap,
                     "all_layer_ablation_reduction": gap-all_gap})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    for key in ("selected_layer_ablation_reduction", "all_layer_ablation_reduction"):
        values = [float(row[key]) for row in rows]
        print(key, mean(values), sum(v > 0 for v in values), signflip(values))


if __name__ == "__main__":
    main()
