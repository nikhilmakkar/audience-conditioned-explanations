"""Uniform-sign forced-choice dose response for the frozen direction."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import torch

from experiment import build_examples, get_layers, load_model, load_stimuli
from layer_token_sweep import add_all_positions, prepare_batch, run_batch
from reencoding_control import ids_for, relabel


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def slope(xs, ys):
    xbar, ybar = mean(xs), mean(ys)
    return sum((x-xbar)*(y-ybar) for x, y in zip(xs, ys)) / sum((x-xbar)**2 for x in xs)


def signflip(values):
    observed = mean(values)
    null = [mean(s*v for s, v in zip(signs, values))
            for signs in itertools.product((-1, 1), repeat=len(values))]
    return sum(value >= observed - 1e-12 for value in null) / len(null)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--stimuli", type=Path, default=Path("data/stimuli_prospective_synthetic.json"))
    parser.add_argument("--direction", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    saved = torch.load(args.direction, map_location="cpu", weights_only=True)
    layer_index = int(saved.get("selected_layer", saved.get("layer")))
    vector = saved.get("selected_direction", saved.get("vector")).float()
    model, tokenizer = load_model(args.model)
    layers = list(get_layers(model))
    original = build_examples(load_stimuli(args.stimuli))
    alphas = (-2.0, -1.0, 0.0, 1.0, 2.0)
    detail = []

    for left, right in (("A", "B"), ("X", "Y"), ("1", "2")):
        encoding = f"{left}/{right}"
        examples = [relabel(example, left, right) for example in original]
        ids = ids_for(tokenizer, (left, right))
        inputs = prepare_batch(model, tokenizer, examples)
        for alpha in alphas:
            coefficients = torch.full((len(examples),), alpha)
            margins = run_batch(
                model, inputs, examples, ids,
                None if alpha == 0 else add_all_positions(layers[layer_index], vector, coefficients),
            )
            grouped = defaultdict(list)
            for example, margin in zip(examples, margins):
                grouped[(example["source_id"], example["condition"])].append(margin)
            for (source, condition), values in sorted(grouped.items()):
                detail.append({
                    "encoding": encoding, "source": source, "condition": condition,
                    "alpha": alpha, "technical_margin": mean(values),
                })
        print(encoding, flush=True)

    summaries = []
    for encoding in ("A/B", "X/Y", "1/2"):
        for condition in ("domain_expert", "matched_other_expert"):
            slopes = []
            for source in sorted({row["source"] for row in detail}):
                rows = [row for row in detail if row["encoding"] == encoding
                        and row["condition"] == condition and row["source"] == source]
                rows.sort(key=lambda row: row["alpha"])
                slopes.append(slope([row["alpha"] for row in rows],
                                    [row["technical_margin"] for row in rows]))
            summaries.append({
                "encoding": encoding, "condition": condition,
                "mean_slope": mean(slopes),
                "standard_error": stdev(slopes) / math.sqrt(len(slopes)),
                "positive_domains": sum(value > 0 for value in slopes),
                "exact_one_sided_p": signflip(slopes),
            })

    write(args.output_dir / "dose_rows.csv", detail)
    write(args.output_dir / "dose_summary.csv", summaries)
    for row in summaries:
        print(row)


if __name__ == "__main__":
    main()
