"""Track learned audience directions/subspaces through residual-stream layers."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from experiment import build_examples, get_layers, load_model, load_stimuli
from fixed_direction_transfer import group_activations
from layer_token_sweep import capture_tail
from multidimensional_subspace import domain_contrasts, svd_basis


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--discovery", type=Path, default=Path("stimuli_domain_matched_8.json"))
    parser.add_argument("--selected-layer", type=int, default=17)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    examples = [e for e in build_examples(load_stimuli(args.discovery)) if e["split"] == "train"]
    model, tokenizer = load_model(args.model)
    n_layers = len(list(get_layers(model)))
    activations = {}
    for index, example in enumerate(examples, 1):
        activations[example["example_id"]] = capture_tail(model, tokenizer, example["prompt"], 1)
        if index % 16 == 0:
            print(f"capture {index}/{len(examples)}", flush=True)

    bases, energies = [], []
    for layer in range(n_layers):
        contrasts = domain_contrasts(group_activations(examples, activations, layer))
        basis = svd_basis(contrasts)
        singular = torch.linalg.svdvals(contrasts.float())
        bases.append(basis)
        energies.append(singular.square() / singular.square().sum())

    selected = bases[args.selected_layer]
    summary = []
    for layer, basis in enumerate(bases):
        component_cosine = abs(float(torch.dot(basis[0], selected[0])))
        principal = torch.linalg.svdvals(basis[:args.rank] @ selected[:args.rank].T)
        adjacent = float("nan") if layer == 0 else abs(float(torch.dot(basis[0], bases[layer-1][0])))
        summary.append({
            "layer": layer,
            "rank1_abs_cosine_to_selected": component_cosine,
            "rank1_abs_cosine_to_previous": adjacent,
            f"rank{args.rank}_mean_principal_cosine_to_selected": float(principal.mean()),
            f"rank{args.rank}_min_principal_cosine_to_selected": float(principal.min()),
            "component1_energy": float(energies[layer][0]),
            f"cumulative_rank{args.rank}_energy": float(energies[layer][:args.rank].sum()),
        })

    pairwise = []
    for left, left_basis in enumerate(bases):
        row = {"layer": left}
        for right, right_basis in enumerate(bases):
            row[f"layer_{right}"] = abs(float(torch.dot(left_basis[0], right_basis[0])))
        pairwise.append(row)

    write(args.output_dir / "layer_geometry_summary.csv", summary)
    write(args.output_dir / "rank1_pairwise_abs_cosine.csv", pairwise)
    torch.save({"bases": torch.stack(bases), "energies": torch.stack(energies)},
               args.output_dir / "layer_subspaces.pt")


if __name__ == "__main__":
    main()
