"""Nested SVD relevance-subspace ablations with rank-matched nulls."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from statistics import mean, stdev

import torch

from exact_relevance_lodo import mean_source_gap
from experiment import build_examples, get_layers, load_model, load_stimuli
from fixed_direction_transfer import group_activations, true_direction
from layer_token_sweep import capture_tail, prepare_batch, run_batch
from reencoding_control import ids_for


POSITIVE = "domain_expert"
NEGATIVE = "matched_other_expert"


def write(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader(); writer.writerows(values)


def exact_signflip(values):
    observed = mean(values)
    null = [mean(sign*value for sign, value in zip(signs, values))
            for signs in itertools.product((-1, 1), repeat=len(values))]
    return sum(value >= observed-1e-12 for value in null)/len(null)


def domain_contrasts(grouped):
    output = []
    for source in sorted({key[0] for key in grouped}):
        positive = torch.stack([value for key, value in grouped.items()
                                if key[0] == source and key[1] == POSITIVE]).mean(0)
        negative = torch.stack([value for key, value in grouped.items()
                                if key[0] == source and key[1] == NEGATIVE]).mean(0)
        output.append(positive-negative)
    return torch.stack(output)


def svd_basis(contrasts):
    return torch.linalg.svd(contrasts.float(), full_matrices=False).Vh


def shuffled_contrasts(grouped, seed):
    generator = torch.Generator().manual_seed(seed)
    output = []
    for source in sorted({key[0] for key in grouped}):
        values = [grouped[key] for key in sorted(grouped) if key[0] == source]
        order = torch.randperm(len(values), generator=generator)
        half = len(values)//2
        positive = torch.stack([values[i] for i in order[:half]]).mean(0)
        negative = torch.stack([values[i] for i in order[half:]]).mean(0)
        output.append(positive-negative)
    return torch.stack(output)


def random_basis(width, seed, rank=8):
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(width, rank, generator=generator)
    return torch.linalg.qr(matrix, mode="reduced").Q.T.contiguous()


@contextmanager
def ablate_basis(layer, basis):
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        changed = hidden.clone()
        local = basis.to(device=hidden.device, dtype=torch.float32)
        coefficients = changed.float() @ local.T
        changed -= (coefficients @ local).to(changed.dtype)
        return (changed,) + output[1:] if isinstance(output, tuple) else changed

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def reductions(examples, baseline, changed):
    output = {}
    for source in sorted({example["source_id"] for example in examples}):
        indices = [i for i, example in enumerate(examples) if example["source_id"] == source]
        subset = [examples[i] for i in indices]
        before = mean_source_gap(subset, [baseline[i] for i in indices])
        after = mean_source_gap(subset, [changed[i] for i in indices])
        output[source] = before-after
    return output


def evaluate_basis(model, layer, inputs, examples, label_ids, baseline, basis):
    changed = run_batch(model, inputs, examples, label_ids, ablate_basis(layer, basis))
    return reductions(examples, baseline, changed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--discovery", type=Path, default=Path("stimuli_domain_matched_8.json"))
    parser.add_argument("--test", type=Path, default=Path("stimuli_prospective_synthetic.json"))
    parser.add_argument("--paraphrase-test", type=Path,
                        default=Path("stimuli_prospective_profile_paraphrase.json"))
    parser.add_argument("--fixed-direction", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--controls", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    discovery = [example for example in build_examples(load_stimuli(args.discovery))
                 if example["split"] == "train"]
    tests = {
        "original": build_examples(load_stimuli(args.test)),
        "new_carrier": build_examples(load_stimuli(args.paraphrase_test)),
    }
    model, tokenizer = load_model(args.model)
    layers = list(get_layers(model))
    activations = {}
    for index, example in enumerate(discovery, 1):
        activations[example["example_id"]] = capture_tail(
            model, tokenizer, example["prompt"], 1
        )
        print(f"[capture {index}/{len(discovery)}]", flush=True)
    grouped = group_activations(discovery, activations, args.layer)
    contrasts = domain_contrasts(grouped)
    basis = svd_basis(contrasts)
    singular = torch.linalg.svdvals(contrasts.float())
    energy = singular.square()/singular.square().sum()
    saved = torch.load(args.fixed_direction, map_location="cpu", weights_only=True)
    fixed = saved.get("vector", saved.get("selected_direction")).float()
    cosine = abs(float(torch.nn.functional.cosine_similarity(basis[0], fixed, dim=0)))

    prepared = {}
    label_ids = ids_for(tokenizer, ("A", "B"))
    for name, examples in tests.items():
        inputs = prepare_batch(model, tokenizer, examples)
        baseline = run_batch(model, inputs, examples, label_ids)
        prepared[name] = inputs, examples, baseline

    learned_rows = []
    for name, (inputs, examples, baseline) in prepared.items():
        for rank in range(1, 9):
            effects = evaluate_basis(model, layers[args.layer], inputs, examples,
                                     label_ids, baseline, basis[:rank])
            values = list(effects.values())
            for source, reduction in effects.items():
                learned_rows.append({"test_set": name, "rank": rank, "source": source,
                                     "ablation_reduction": reduction,
                                     "aggregate_mean": mean(values),
                                     "positive_domains": sum(value > 0 for value in values),
                                     "exact_one_sided_p": exact_signflip(values)})
        print(f"learned ranks complete: {name}", flush=True)

    inputs, examples, baseline = prepared["original"]
    control_rows = []
    for kind in ("random", "shuffled_labels"):
        for seed in range(args.controls):
            control_basis = (random_basis(fixed.numel(), seed) if kind == "random"
                             else svd_basis(shuffled_contrasts(grouped, seed)))
            for rank in range(1, 9):
                effects = evaluate_basis(model, layers[args.layer], inputs, examples,
                                         label_ids, baseline, control_basis[:rank])
                aggregate = mean(effects.values())
                for source, reduction in effects.items():
                    control_rows.append({"kind": kind, "seed": seed, "rank": rank,
                                         "source": source, "ablation_reduction": reduction,
                                         "aggregate_mean": aggregate})
            if (seed+1) % 10 == 0:
                print(f"{kind} controls {seed+1}/{args.controls}", flush=True)

    spectrum_rows = [{"component": i+1, "singular_value": float(value),
                      "energy_fraction": float(energy[i]),
                      "cumulative_energy": float(energy[:i+1].sum())}
                     for i, value in enumerate(singular)]
    summary_rows = []
    for name in tests:
        for rank in range(1, 9):
            subset = [row for row in learned_rows if row["test_set"] == name and row["rank"] == rank]
            first = subset[0]
            result = {"test_set": name, "rank": rank,
                      "mean_reduction": first["aggregate_mean"],
                      "positive_domains": first["positive_domains"],
                      "exact_one_sided_p": first["exact_one_sided_p"],
                      "random_ge_learned": "", "shuffled_ge_learned": ""}
            if name == "original":
                for kind, key in (("random", "random_ge_learned"),
                                  ("shuffled_labels", "shuffled_ge_learned")):
                    controls = {int(row["seed"]): float(row["aggregate_mean"])
                                for row in control_rows if row["kind"] == kind and row["rank"] == rank}
                    result[key] = sum(value >= first["aggregate_mean"] for value in controls.values())
            summary_rows.append(result)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write(args.output_dir/"spectrum.csv", spectrum_rows)
    write(args.output_dir/"learned_ablation_by_domain.csv", learned_rows)
    write(args.output_dir/"control_ablation_by_domain.csv", control_rows)
    write(args.output_dir/"rank_summary.csv", summary_rows)
    torch.save({"layer": args.layer, "basis": basis, "singular_values": singular},
               args.output_dir/"relevance_subspace.pt")
    (args.output_dir/"config.json").write_text(json.dumps({
        "model": args.model, "layer": args.layer, "controls": args.controls,
        "discovery": str(args.discovery), "test": str(args.test),
        "paraphrase_test": str(args.paraphrase_test),
        "rank1_abs_cosine_to_fixed_mean_direction": cosine,
    }, indent=2)+"\n")
    print(f"rank-1 |cosine| to fixed mean direction: {cosine:.6f}")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
