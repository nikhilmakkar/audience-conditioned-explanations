"""Ordinary-text damage curve for a learned residual-stream subspace."""

from __future__ import annotations

import argparse
import csv
import json
import math
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F

from ablation_coherence import load_wikitext, metrics
from experiment import get_layers, load_model


@contextmanager
def ablate_subspace(layer, basis):
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        vectors = basis.to(device=hidden.device, dtype=torch.float32)
        changed = hidden.clone()
        projection = (changed.float() @ vectors.T) @ vectors
        changed -= projection.to(changed.dtype)
        if isinstance(output, tuple):
            return (changed,) + output[1:]
        return changed

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def model_forward(model, input_ids, context=None):
    mask = torch.ones_like(input_ids)
    if context is None:
        with torch.inference_mode():
            return model(input_ids=input_ids, attention_mask=mask).logits
    with context:
        with torch.inference_mode():
            return model(input_ids=input_ids, attention_mask=mask).logits


def evaluate(model, layer, blocks, basis, batch_size):
    totals = {"base_ce": 0.0, "changed_ce": 0.0, "kl": 0.0, "agree": 0, "tokens": 0}
    device = next(model.parameters()).device
    for start in range(0, len(blocks), batch_size):
        ids = blocks[start:start + batch_size].to(device)
        base = model_forward(model, ids)
        labels = ids[:, 1:]
        totals["base_ce"] += float(F.cross_entropy(
            base[:, :-1].float().reshape(-1, base.shape[-1]), labels.reshape(-1), reduction="sum"
        ).cpu())
        changed = model_forward(model, ids, ablate_subspace(layer, basis))
        ce, kl, agree, tokens = metrics(base, changed, ids)
        totals["changed_ce"] += ce
        totals["kl"] += kl
        totals["agree"] += agree
        totals["tokens"] += tokens
        del base, changed
    n = totals["tokens"]
    base_nll = totals["base_ce"] / n
    changed_nll = totals["changed_ce"] / n
    return {
        "tokens": n,
        "baseline_nll": base_nll,
        "changed_nll": changed_nll,
        "nll_increase": changed_nll - base_nll,
        "mean_next_token_kl": totals["kl"] / n,
        "top1_agreement": totals["agree"] / n,
    }


def random_basis(width, rank, seed):
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(width, rank, generator=generator)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q.T.contiguous()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--subspace", type=Path, required=True)
    parser.add_argument("--wikitext", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-blocks", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--random-controls", type=int, default=10)
    args = parser.parse_args()

    saved = torch.load(args.subspace, map_location="cpu", weights_only=True)
    basis = saved["basis"].float()
    layer_number = int(saved["layer"])
    model, tokenizer = load_model(args.model)
    layer = list(get_layers(model))[layer_number]
    ids = tokenizer.encode(load_wikitext(args.wikitext), add_special_tokens=False)
    usable = min(len(ids) // args.block_size, args.max_blocks) * args.block_size
    blocks = torch.tensor(ids[:usable], dtype=torch.long).reshape(-1, args.block_size)

    rows = []
    for rank in range(1, basis.shape[0] + 1):
        row = evaluate(model, layer, blocks, basis[:rank], args.batch_size)
        row.update({"kind": "learned", "seed": -1, "rank": rank, "layer": layer_number})
        rows.append(row)
        print(row, flush=True)
    for seed in range(args.random_controls):
        full = random_basis(basis.shape[1], basis.shape[0], seed)
        for rank in range(1, basis.shape[0] + 1):
            row = evaluate(model, layer, blocks, full[:rank], args.batch_size)
            row.update({"kind": "random", "seed": seed, "rank": rank, "layer": layer_number})
            rows.append(row)
        print(f"random damage {seed + 1}/{args.random_controls}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
