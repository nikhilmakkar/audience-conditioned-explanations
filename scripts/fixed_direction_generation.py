"""Prospectively frozen dose-response generation with one fixed direction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch

from experiment import build_generation_examples, format_prompt, get_layers, load_model, load_stimuli


@contextmanager
def add_direction(layer, vector, alpha):
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        changed = hidden + alpha * vector.to(device=hidden.device, dtype=hidden.dtype)[None, None, :]
        return (changed,) + output[1:] if isinstance(output, tuple) else changed

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def generate(model, tokenizer, prompt, layer, vector, alpha, max_new_tokens):
    rendered = format_prompt(tokenizer, prompt)
    inputs = tokenizer(rendered, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    context = add_direction(layer, vector, alpha) if alpha else nullcontext()
    with context, torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    new = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new, skip_special_tokens=True).strip()


def term_hits(text, terms):
    lower = text.lower()
    return sum(
        bool(re.search(r"(?<!\w)" + re.escape(term.lower()) + r"(?!\w)", lower))
        for term in terms
    )


def code(example_id, alpha):
    payload = f"prospective-generation-v1:{example_id}:{alpha}"
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--stimuli", type=Path, default=Path("data/stimuli_prospective_synthetic.json"))
    parser.add_argument("--direction", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    data = load_stimuli(args.stimuli)
    banks = {source["id"]: source["technical_terms"] for source in data["sources"]}
    saved = torch.load(args.direction, map_location="cpu", weights_only=True)
    layer_index = int(saved.get("selected_layer", saved.get("layer")))
    vector = saved.get("selected_direction", saved.get("vector")).float()
    model, tokenizer = load_model(args.model)
    layer = get_layers(model)[layer_index]
    examples = build_generation_examples(data)
    alphas = (-2.0, -1.0, 0.0, 1.0, 2.0)

    rows = []
    for index, example in enumerate(examples, 1):
        terms = banks[example["source_id"]]
        for alpha in alphas:
            text = generate(
                model, tokenizer, example["prompt"], layer, vector, alpha,
                args.max_new_tokens,
            )
            hits = term_hits(text, terms)
            rows.append({
                "blind_code": code(example["example_id"], alpha),
                "example_id": example["example_id"],
                "source_id": example["source_id"],
                "condition": example["condition"],
                "profile_index": example["profile_index"],
                "alpha": alpha,
                "term_hits": hits,
                "term_bank_size": len(terms),
                "term_coverage": hits / len(terms),
                "word_count": len(text.split()),
                "text": text,
            })
        print(f"[{index}/{len(examples)}] {example['example_id']}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (args.output_dir / "generation_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    review = [{
        "blind_code": row["blind_code"],
        "source_id": row["source_id"],
        "text": row["text"],
        "human_technicality_1_to_5": "",
        "human_correct_0_or_1": "",
        "human_coherent_0_or_1": "",
    } for row in rows]
    random.Random(20260908).shuffle(review)
    with (args.output_dir / "blind_review.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review[0]))
        writer.writeheader()
        writer.writerows(review)

    with (args.output_dir / "blind_key.csv").open("w", newline="") as handle:
        keys = ["blind_code", "example_id", "condition", "alpha"]
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row[key] for key in keys} for row in rows)

    config = {
        "model": args.model,
        "stimuli": str(args.stimuli),
        "direction": str(args.direction),
        "layer": layer_index,
        "alphas": alphas,
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
        "rows": len(rows),
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
