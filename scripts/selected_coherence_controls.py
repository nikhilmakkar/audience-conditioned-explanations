"""Selected-layer norm-matched coherence controls for a saved direction."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from ablation_coherence import evaluate_direction, load_wikitext
from experiment import get_layers, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--direction", type=Path, required=True)
    parser.add_argument("--wikitext", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--max-blocks", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--random-controls", type=int, default=10)
    args = parser.parse_args()

    saved = torch.load(args.direction, map_location="cpu", weights_only=True)
    layer_index = int(saved.get("selected_layer", saved.get("layer")))
    learned = saved.get("selected_direction", saved.get("vector")).float()
    learned /= learned.norm().clamp_min(1e-12)
    model, tokenizer = load_model(args.model)
    layers = list(get_layers(model))
    ids = tokenizer.encode(load_wikitext(args.wikitext), add_special_tokens=False)
    usable = min(len(ids)//args.block_size, args.max_blocks)*args.block_size
    blocks = torch.tensor(ids[:usable], dtype=torch.long).reshape(-1, args.block_size)
    scope = f"selected:{layer_index}"

    rows = []
    learned_row = evaluate_direction(model, layers, blocks, learned, scope, args.batch_size)
    learned_row.update({"kind": "learned", "seed": -1})
    rows.append(learned_row); print(learned_row)
    for seed in range(args.random_controls):
        generator = torch.Generator().manual_seed(seed)
        control = torch.randn(learned.shape, generator=generator)
        control /= control.norm()
        row = evaluate_direction(model, layers, blocks, control, scope, args.batch_size)
        row.update({"kind": "random", "seed": seed})
        rows.append(row); print(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
