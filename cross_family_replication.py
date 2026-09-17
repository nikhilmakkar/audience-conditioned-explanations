"""Discovery-selected, prospectively held-out cross-family replication."""

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

from experiment import build_examples, get_layers, load_model, load_stimuli
from fixed_direction_transfer import group_activations, permuted_direction, true_direction
from layer_token_sweep import add_all_positions, capture_tail, prepare_batch, run_batch
from reencoding_control import ids_for, relabel


POSITIVE = "domain_expert"
NEGATIVE = "matched_other_expert"


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def signflip(values):
    observed = mean(values)
    null = [mean(s*v for s, v in zip(signs, values))
            for signs in itertools.product((-1, 1), repeat=len(values))]
    return sum(value >= observed - 1e-12 for value in null) / len(null)


def source_effects(examples, baseline, changed):
    grouped = defaultdict(list)
    for example, before, after in zip(examples, baseline, changed):
        sign = 1.0 if example["condition"] == NEGATIVE else -1.0
        grouped[example["source_id"]].append(sign * (after - before))
    return {source: mean(values) for source, values in sorted(grouped.items())}


def subset_inputs(model, tokenizer, examples):
    return prepare_batch(model, tokenizer, examples)


def prepare_label_pair(tokenizer, inputs, left, right):
    """Append a shared label-token prefix and return the final differing IDs."""
    left_ids = tokenizer.encode(" " + left, add_special_tokens=False)
    right_ids = tokenizer.encode(" " + right, add_special_tokens=False)
    prefix_length = 0
    while (prefix_length < min(len(left_ids), len(right_ids))
           and left_ids[prefix_length] == right_ids[prefix_length]):
        prefix_length += 1
    if len(left_ids) != prefix_length + 1 or len(right_ids) != prefix_length + 1:
        raise ValueError(f"Labels do not differ in exactly one final token: {left_ids}, {right_ids}")
    if prefix_length:
        prefix = torch.tensor(left_ids[:prefix_length], device=inputs["input_ids"].device)
        prefix = prefix[None, :].expand(inputs["input_ids"].shape[0], -1)
        inputs = dict(inputs)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], prefix], dim=1)
        inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(prefix)], dim=1)
    return inputs, {left: left_ids[-1], right: right_ids[-1]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--discovery", type=Path, default=Path("stimuli_domain_matched_8.json"))
    parser.add_argument("--test", type=Path, default=Path("stimuli_prospective_synthetic.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-controls", type=int, default=100)
    parser.add_argument("--permutation-controls", type=int, default=100)
    args = parser.parse_args()

    discovery = [e for e in build_examples(load_stimuli(args.discovery)) if e["split"] == "train"]
    test_original = build_examples(load_stimuli(args.test))
    model, tokenizer = load_model(args.model)
    layers = list(get_layers(model))
    candidate_layers = sorted({round(fraction * (len(layers)-1))
                               for fraction in (0.25, 0.375, 0.5, 0.625, 0.75)})
    activations = {}
    for index, example in enumerate(discovery, 1):
        activations[example["example_id"]] = capture_tail(model, tokenizer, example["prompt"], 1)
        print(f"[capture {index}/{len(discovery)}]", flush=True)

    discovery_ids = ids_for(tokenizer, ("A", "B"))
    discovery_domains = sorted({e["source_id"] for e in discovery})
    selection_rows = []
    for layer_index in candidate_layers:
        fold_effects = []
        for heldout in discovery_domains:
            train = [e for e in discovery if e["source_id"] != heldout]
            validation = [e for e in discovery if e["source_id"] == heldout]
            grouped = group_activations(train, activations, layer_index)
            vector = true_direction(grouped)
            inputs = subset_inputs(model, tokenizer, validation)
            baseline = run_batch(model, inputs, validation, discovery_ids)
            coefficients = torch.tensor([1.0 if e["condition"] == NEGATIVE else -1.0
                                         for e in validation])
            changed = run_batch(model, inputs, validation, discovery_ids,
                                add_all_positions(layers[layer_index], vector, coefficients))
            effect = mean(source_effects(validation, baseline, changed).values())
            fold_effects.append(effect)
            selection_rows.append({"layer": layer_index, "heldout_domain": heldout,
                                   "addition_effect": effect})
        print(f"layer {layer_index}: {mean(fold_effects):.6f}", flush=True)

    means = {layer: mean([r["addition_effect"] for r in selection_rows if r["layer"] == layer])
             for layer in candidate_layers}
    selected_layer = max(candidate_layers, key=lambda layer: (means[layer], -layer))
    grouped = group_activations(discovery, activations, selected_layer)
    vector = true_direction(grouped)
    print(f"selected layer {selected_layer}", flush=True)

    addition_rows, summaries = [], []
    ab_inputs = ab_examples = ab_ids = ab_baseline = ab_coefficients = None
    for left, right in (("A", "B"), ("X", "Y"), ("1", "2")):
        examples = [relabel(e, left, right) for e in test_original]
        inputs = subset_inputs(model, tokenizer, examples)
        inputs, label_ids = prepare_label_pair(tokenizer, inputs, left, right)
        baseline = run_batch(model, inputs, examples, label_ids)
        coefficients = torch.tensor([1.0 if e["condition"] == NEGATIVE else -1.0
                                     for e in examples])
        changed = run_batch(model, inputs, examples, label_ids,
                            add_all_positions(layers[selected_layer], vector, coefficients))
        effects = source_effects(examples, baseline, changed)
        values = list(effects.values())
        encoding = f"{left}/{right}"
        summaries.append({"encoding": encoding, "mean_effect": mean(values),
                          "standard_error": stdev(values)/math.sqrt(len(values)),
                          "positive_domains": sum(v > 0 for v in values),
                          "exact_one_sided_p": signflip(values)})
        addition_rows.extend({"encoding": encoding, "source": source, "addition_effect": effect}
                             for source, effect in effects.items())
        if encoding == "A/B":
            ab_inputs, ab_examples, ab_ids = inputs, examples, label_ids
            ab_baseline, ab_coefficients = baseline, coefficients

    controls = []
    for kind, count in (("random", args.random_controls),
                        ("permuted_labels", args.permutation_controls)):
        for seed in range(count):
            if kind == "random":
                generator = torch.Generator().manual_seed(seed)
                control = torch.randn(vector.shape, generator=generator)
                control *= vector.norm()/control.norm()
            else:
                control = permuted_direction(grouped, seed)
                control *= vector.norm()/control.norm().clamp_min(1e-12)
            changed = run_batch(model, ab_inputs, ab_examples, ab_ids,
                                add_all_positions(layers[selected_layer], control, ab_coefficients))
            effects = source_effects(ab_examples, ab_baseline, changed)
            aggregate = mean(effects.values())
            controls.extend({"kind": kind, "seed": seed, "source": source,
                             "addition_effect": effect, "aggregate_mean": aggregate}
                            for source, effect in effects.items())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write(args.output_dir/"selection_lodo.csv", selection_rows)
    write(args.output_dir/"addition_by_domain.csv", addition_rows)
    write(args.output_dir/"encoding_summary.csv", summaries)
    write(args.output_dir/"controls.csv", controls)
    torch.save({"layer": selected_layer, "vector": vector}, args.output_dir/"fixed_direction.pt")
    (args.output_dir/"config.json").write_text(json.dumps({
        "model": args.model, "discovery": str(args.discovery), "test": str(args.test),
        "candidate_layers": candidate_layers, "selected_layer": selected_layer,
        "random_controls": args.random_controls,
        "permutation_controls": args.permutation_controls,
    }, indent=2)+"\n")
    for row in summaries:
        print(row)


if __name__ == "__main__":
    main()
