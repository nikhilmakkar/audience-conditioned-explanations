"""Exhaustive discovery-domain subset learning curve for a fixed layer."""

from __future__ import annotations

import argparse
import csv
import itertools
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import torch

from experiment import build_examples, get_layers, load_model, load_stimuli
from fixed_direction_transfer import group_activations, true_direction
from layer_token_sweep import add_all_positions, capture_tail, prepare_batch, run_batch
from reencoding_control import ids_for


def effects(examples, baseline, changed):
    grouped = defaultdict(list)
    for example, before, after in zip(examples, baseline, changed):
        sign = 1.0 if example["condition"] == "matched_other_expert" else -1.0
        grouped[example["source_id"]].append(sign*(after-before))
    return {source: mean(values) for source, values in sorted(grouped.items())}


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--discovery", type=Path, default=Path("stimuli_domain_matched_8.json"))
    parser.add_argument("--test", type=Path, default=Path("stimuli_prospective_synthetic.json"))
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    discovery = [e for e in build_examples(load_stimuli(args.discovery)) if e["split"] == "train"]
    test = build_examples(load_stimuli(args.test))
    domains = sorted({e["source_id"] for e in discovery})
    model, tokenizer = load_model(args.model)
    layers = list(get_layers(model))
    activations = {}
    for index, example in enumerate(discovery, 1):
        activations[example["example_id"]] = capture_tail(model, tokenizer, example["prompt"], 1)
        print(f"[capture {index}/{len(discovery)}]", flush=True)
    grouped_all = group_activations(discovery, activations, args.layer)
    full = true_direction(grouped_all)
    full_unit = full/full.norm().clamp_min(1e-12)

    inputs = prepare_batch(model, tokenizer, test)
    label_ids = ids_for(tokenizer, ("A", "B"))
    baseline = run_batch(model, inputs, test, label_ids)
    coefficients = torch.tensor([1.0 if e["condition"] == "matched_other_expert" else -1.0
                                 for e in test])
    subset_rows, domain_rows = [], []
    for size in (1, 2, 4, 6, 8):
        for subset in itertools.combinations(domains, size):
            allowed = set(subset)
            grouped = {key: value for key, value in grouped_all.items() if key[0] in allowed}
            vector = true_direction(grouped)
            changed = run_batch(model, inputs, test, label_ids,
                                add_all_positions(layers[args.layer], vector, coefficients))
            per_domain = effects(test, baseline, changed)
            cosine = float(torch.nn.functional.cosine_similarity(vector, full, dim=0))
            subset_id = "+".join(subset)
            subset_rows.append({"train_domains": size, "subset": subset_id,
                                "mean_effect": mean(per_domain.values()),
                                "positive_test_domains": sum(v > 0 for v in per_domain.values()),
                                "cosine_to_full": cosine, "direction_norm": float(vector.norm())})
            domain_rows.extend({"train_domains": size, "subset": subset_id,
                                "test_source": source, "addition_effect": effect}
                               for source, effect in per_domain.items())
        print(f"size {size} complete", flush=True)

    summaries = []
    for size in (1, 2, 4, 6, 8):
        rows = [row for row in subset_rows if row["train_domains"] == size]
        for metric in ("mean_effect", "positive_test_domains", "cosine_to_full", "direction_norm"):
            values = [float(row[metric]) for row in rows]
            summaries.append({"train_domains": size, "n_subsets": len(rows), "metric": metric,
                              "mean": mean(values), "median": median(values),
                              "minimum": min(values), "maximum": max(values)})
    write(args.output_dir/"subset_results.csv", subset_rows)
    write(args.output_dir/"domain_results.csv", domain_rows)
    write(args.output_dir/"learning_curve_summary.csv", summaries)
    for row in summaries:
        print(row)


if __name__ == "__main__":
    main()
