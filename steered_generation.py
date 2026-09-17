"""Generate held-out gists under bidirectional audience-direction steering."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch

from experiment import build_generation_examples, format_prompt, get_layers, load_model, load_stimuli


TERM_BANK = {
    "resnet": [
        "residual", "f(x)", "identity mapping", "identity shortcut",
        "projection shortcut", "parameterization", "degradation", "optimization", "cnn",
    ],
    "transformer": [
        "multi-head", "self-attention", "query", "queries", "key", "keys",
        "value", "values", "scaled dot-product", "positional encoding",
        "causal mask", "recurrence", "convolution",
    ],
}


@contextmanager
def add_generation_direction(layer, vector, alpha):
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        modified = hidden.clone()
        direction = vector.to(device=hidden.device, dtype=hidden.dtype)
        modified += alpha * direction[None, None, :]
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified

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
    context = (
        add_generation_direction(layer, vector, alpha)
        if alpha != 0 else nullcontext()
    )
    with context:
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def term_count(text, source):
    lower = text.lower()
    return sum(bool(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", lower))
               for term in TERM_BANK[source])


def blind_code(example_id, alpha):
    digest = hashlib.sha256(f"audience-steering:{example_id}:{alpha}".encode()).hexdigest()
    return digest[:10]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--stimuli", type=Path, default=Path("stimuli.json"))
    parser.add_argument("--direction", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()

    saved = torch.load(args.direction, map_location="cpu", weights_only=True)
    layer_index = int(saved["selected_layer"])
    vector = saved["selected_direction"].float()
    model, tokenizer = load_model(args.model)
    layer = get_layers(model)[layer_index]
    examples = [
        e for e in build_generation_examples(load_stimuli(args.stimuli))
        if e["condition"] in {"expert", "novice"} and e["split"] == "test"
    ]

    rows = []
    for index, example in enumerate(examples, 1):
        for alpha in (-1.0, 0.0, 1.0):
            text = generate(
                model, tokenizer, example["prompt"], layer, vector,
                alpha, args.max_new_tokens,
            )
            rows.append({
                "example_id": example["example_id"],
                "source_id": example["source_id"],
                "condition": example["condition"],
                "alpha": alpha,
                "word_count": len(text.split()),
                "technical_term_count": term_count(text, example["source_id"]),
                "text": text,
                "blind_code": blind_code(example["example_id"], alpha),
            })
        print(f"[{index}/{len(examples)}] {example['example_id']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "generation_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    blinded = [
        {
            "blind_code": row["blind_code"],
            "source_id": row["source_id"],
            "text": row["text"],
            "human_technicality_1_to_5": "",
            "human_correct_0_or_1": "",
            "human_coherent_0_or_1": "",
        }
        for row in sorted(rows, key=lambda row: row["blind_code"])
    ]
    with (args.output_dir / "blind_review.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(blinded[0]))
        writer.writeheader()
        writer.writerows(blinded)
    with (args.output_dir / "blind_key.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["blind_code", "example_id", "condition", "alpha"]
        )
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows)


if __name__ == "__main__":
    main()
