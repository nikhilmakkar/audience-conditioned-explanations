"""Measure language-model damage caused by residual-direction ablation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F

from experiment import get_layers, load_model


@contextmanager
def ablate_all_positions(layers, unit):
    handles = []

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        modified = hidden.clone()
        direction = unit.to(device=hidden.device, dtype=torch.float32)
        projections = modified.float() @ direction
        modified -= projections.to(modified.dtype)[..., None] * direction.to(modified.dtype)
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified

    for layer in layers:
        handles.append(layer.register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def load_wikitext(path: Path) -> str:
    payload = json.loads(path.read_text())
    texts = [row["row"]["text"] for row in payload["rows"]]
    return "\n".join(text for text in texts if text.strip())


def metrics(reference_logits, changed_logits, input_ids):
    reference = reference_logits[:, :-1].float()
    changed = changed_logits[:, :-1].float()
    labels = input_ids[:, 1:]
    changed_ce = F.cross_entropy(
        changed.reshape(-1, changed.shape[-1]), labels.reshape(-1), reduction="sum"
    )
    reference_logp = torch.log_softmax(reference, dim=-1)
    changed_logp = torch.log_softmax(changed, dim=-1)
    reference_p = reference_logp.exp()
    kl = (reference_p * (reference_logp - changed_logp)).sum(dim=-1).sum()
    agreement = (reference.argmax(-1) == changed.argmax(-1)).sum()
    return float(changed_ce.cpu()), float(kl.cpu()), int(agreement.cpu()), labels.numel()


def forward(model, input_ids, context=None):
    attention_mask = torch.ones_like(input_ids)
    if context is None:
        with torch.inference_mode():
            return model(input_ids=input_ids, attention_mask=attention_mask).logits
    with context:
        with torch.inference_mode():
            return model(input_ids=input_ids, attention_mask=attention_mask).logits


def evaluate_direction(model, layers, blocks, unit, scope, batch_size):
    selected_layer = None
    if scope.startswith("selected:"):
        selected_layer = int(scope.split(":", 1)[1])
        target_layers = [layers[selected_layer]]
    elif scope == "all":
        target_layers = layers
    else:
        raise ValueError(scope)

    totals = {"baseline_ce": 0.0, "changed_ce": 0.0, "kl": 0.0, "agree": 0, "tokens": 0}
    device = next(model.parameters()).device
    for start in range(0, len(blocks), batch_size):
        input_ids = blocks[start:start + batch_size].to(device)
        baseline = forward(model, input_ids)
        labels = input_ids[:, 1:]
        totals["baseline_ce"] += float(F.cross_entropy(
            baseline[:, :-1].float().reshape(-1, baseline.shape[-1]),
            labels.reshape(-1), reduction="sum"
        ).cpu())
        changed = forward(
            model, input_ids, ablate_all_positions(target_layers, unit)
        )
        changed_ce, kl, agree, tokens = metrics(baseline, changed, input_ids)
        totals["changed_ce"] += changed_ce
        totals["kl"] += kl
        totals["agree"] += agree
        totals["tokens"] += tokens
        del baseline, changed
    n = totals["tokens"]
    baseline_nll = totals["baseline_ce"] / n
    changed_nll = totals["changed_ce"] / n
    return {
        "scope": scope,
        "tokens": n,
        "baseline_nll": baseline_nll,
        "changed_nll": changed_nll,
        "nll_increase": changed_nll - baseline_nll,
        "baseline_perplexity": math.exp(baseline_nll),
        "changed_perplexity": math.exp(changed_nll),
        "mean_next_token_kl": totals["kl"] / n,
        "top1_agreement": totals["agree"] / n,
    }


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
    selected_layer = int(saved["selected_layer"])
    learned = saved["selected_direction"].float()
    learned /= learned.norm().clamp_min(1e-12)
    model, tokenizer = load_model(args.model)
    layers = list(get_layers(model))
    ids = tokenizer.encode(load_wikitext(args.wikitext), add_special_tokens=False)
    usable = min(len(ids) // args.block_size, args.max_blocks) * args.block_size
    blocks = torch.tensor(ids[:usable], dtype=torch.long).reshape(-1, args.block_size)

    rows = []
    for scope in (f"selected:{selected_layer}", "all"):
        row = evaluate_direction(model, layers, blocks, learned, scope, args.batch_size)
        row.update({"kind": "learned", "seed": -1})
        rows.append(row)
        print(rows[-1])

    for seed in range(args.random_controls):
        generator = torch.Generator().manual_seed(seed)
        random = torch.randn(learned.shape, generator=generator)
        random /= random.norm()
        row = evaluate_direction(model, layers, blocks, random, "all", args.batch_size)
        row.update({"kind": "random", "seed": seed})
        rows.append(row)
        print(rows[-1])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
