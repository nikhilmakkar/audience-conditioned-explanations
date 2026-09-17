"""Projection transfer of one frozen direction to synthetic reader profiles."""

from __future__ import annotations

import argparse
import csv
import itertools
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch

from experiment import build_examples, get_layers, load_model, load_stimuli
from fixed_direction_transfer import group_activations, permuted_direction
from layer_token_sweep import capture_tail


def signflip(values):
    observed = mean(values)
    null = [mean(s*v for s, v in zip(signs, values))
            for signs in itertools.product((-1, 1), repeat=len(values))]
    return sum(v >= observed-1e-12 for v in null)/len(null)


def grouped(examples, activations, layer):
    values = defaultdict(list)
    for example in examples:
        key = example["source_id"], example["condition"], example["profile_index"]
        values[key].append(activations[example["example_id"]][layer, -1])
    return {key: torch.stack(items).mean(0) for key, items in values.items()}


def auc(labels, scores):
    positive = [s for y, s in zip(labels, scores) if y]
    negative = [s for y, s in zip(labels, scores) if not y]
    return mean(1.0 if p > n else 0.5 if p == n else 0.0 for p in positive for n in negative)


def evaluate(values, direction, midpoint):
    unit = direction/direction.norm().clamp_min(1e-12)
    scores = {key: float(value@unit) for key, value in values.items()}
    domains = sorted({key[0] for key in scores})
    gaps = {domain: mean([score for key, score in scores.items()
                          if key[0] == domain and key[1] == "domain_expert"])
                    - mean([score for key, score in scores.items()
                            if key[0] == domain and key[1] == "matched_other_expert"])
            for domain in domains}
    labels = [key[1] == "domain_expert" for key in scores]
    score_list = list(scores.values())
    accuracy = mean((score > midpoint) == label for score, label in zip(score_list, labels))
    return gaps, auc(labels, score_list), accuracy


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--discovery", type=Path, default=Path("data/stimuli_domain_matched_8.json"))
    parser.add_argument("--tests", type=Path, nargs="+", required=True)
    parser.add_argument("--direction", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    saved = torch.load(args.direction, map_location="cpu", weights_only=True)
    layer = int(saved.get("layer", saved.get("selected_layer")))
    learned = saved.get("vector", saved.get("selected_direction")).float()
    model, tokenizer = load_model(args.model)
    discovery = [e for e in build_examples(load_stimuli(args.discovery)) if e["split"] == "train"]
    all_sets = [("discovery", discovery)] + [(path.stem, build_examples(load_stimuli(path)))
                                               for path in args.tests]
    activation_sets = {}
    total = sum(len(examples) for _, examples in all_sets)
    done = 0
    for name, examples in all_sets:
        activations = {}
        for example in examples:
            activations[example["example_id"]] = capture_tail(model, tokenizer, example["prompt"], 1)
            done += 1
            print(f"[capture {done}/{total}]", flush=True)
        activation_sets[name] = grouped(examples, activations, layer)

    discovery_values = activation_sets["discovery"]
    unit = learned/learned.norm().clamp_min(1e-12)
    pos = [float(v@unit) for key, v in discovery_values.items() if key[1] == "domain_expert"]
    neg = [float(v@unit) for key, v in discovery_values.items() if key[1] == "matched_other_expert"]
    midpoint = (mean(pos)+mean(neg))/2
    randoms = []
    for seed in range(100):
        generator = torch.Generator().manual_seed(seed)
        vector = torch.randn(learned.shape, generator=generator)
        randoms.append(("random", seed, vector))
        randoms.append(("permuted_labels", seed, permuted_direction(discovery_values, seed)))

    summary, domains, controls = [], [], []
    for name, _ in all_sets[1:]:
        values = activation_sets[name]
        gaps, test_auc, accuracy = evaluate(values, learned, midpoint)
        vals = list(gaps.values())
        summary.append({"test_set": name, "mean_gap": mean(vals),
                        "positive_domains": sum(v > 0 for v in vals),
                        "exact_one_sided_p": signflip(vals), "auc": test_auc,
                        "discovery_midpoint_accuracy": accuracy})
        domains.extend({"test_set": name, "source": source, "projection_gap": gap}
                       for source, gap in gaps.items())
        for kind, seed, vector in randoms:
            control_gaps, _, _ = evaluate(values, vector, midpoint)
            controls.append({"test_set": name, "kind": kind, "seed": seed,
                             "aggregate_gap": mean(control_gaps.values())})
    write(args.output_dir/"projection_summary.csv", summary)
    write(args.output_dir/"projection_by_domain.csv", domains)
    write(args.output_dir/"projection_controls.csv", controls)
    for row in summary:
        matching = {kind: sum(float(r["aggregate_gap"]) >= row["mean_gap"] for r in controls
                              if r["test_set"] == row["test_set"] and r["kind"] == kind)
                    for kind in ("random", "permuted_labels")}
        print(row, matching)


if __name__ == "__main__":
    main()
