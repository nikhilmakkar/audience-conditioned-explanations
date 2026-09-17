"""Layer/token causal sweep and exact decoder permutation test.

The sweep uses ResNet train prompts to extract expert-minus-novice directions
from the last K input positions at every residual-stream layer.  Candidate
selection uses ResNet validation prompts only.  The frozen candidate is then
evaluated on held-out ResNet and Transformer prompts.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from contextlib import contextmanager
from pathlib import Path

import torch

from experiment import (
    build_examples,
    format_prompt,
    get_layers,
    label_token_ids,
    load_model,
    load_stimuli,
)


def write_rows(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def capture_tail(model, tokenizer, prompt: str, width: int) -> torch.Tensor:
    layers = get_layers(model)
    cache = [None] * len(layers)
    handles = []

    def make_hook(index):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.shape[1] < width:
                raise ValueError(f"Prompt has only {hidden.shape[1]} tokens")
            cache[index] = hidden[0, -width:].detach().float().cpu()
        return hook

    for index, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(index)))
    try:
        rendered = format_prompt(tokenizer, prompt)
        inputs = tokenizer(rendered, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            model(**inputs)
    finally:
        for handle in handles:
            handle.remove()
    return torch.stack(cache)  # [layer, tail_position, d_model]


def prepare_batch(model, tokenizer, examples):
    tokenizer.padding_side = "left"
    rendered = [format_prompt(tokenizer, example["prompt"]) for example in examples]
    inputs = tokenizer(rendered, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    return {key: value.to(device) for key, value in inputs.items()}


def margins_from_logits(logits, examples, label_ids):
    log_probs = torch.log_softmax(logits[:, -1].float(), dim=-1)
    margins = []
    for row, example in enumerate(examples):
        technical = example["technical_label"]
        accessible = next(label for label in label_ids if label != technical)
        margins.append(float(
            (log_probs[row, label_ids[technical]] - log_probs[row, label_ids[accessible]])
            .detach().cpu()
        ))
    return margins


@contextmanager
def add_all_positions(layer, vector, coefficients):
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        modified = hidden.clone()
        direction = vector.to(device=hidden.device, dtype=hidden.dtype)
        scale = coefficients.to(device=hidden.device, dtype=hidden.dtype)
        modified += scale[:, None, None] * direction[None, None, :]
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


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


def run_batch(model, inputs, examples, label_ids, context=None):
    if context is None:
        with torch.inference_mode():
            logits = model(**inputs).logits
    else:
        with context:
            with torch.inference_mode():
                logits = model(**inputs).logits
    return margins_from_logits(logits, examples, label_ids)


def signed_addition_effect(examples, baseline, changed):
    values = []
    for example, before, after in zip(examples, baseline, changed):
        sign = 1.0 if example["condition"] == "novice" else -1.0
        values.append(sign * (after - before))
    return sum(values) / len(values)


def audience_gap(examples, margins):
    by_condition = {"expert": [], "novice": []}
    for example, margin in zip(examples, margins):
        by_condition[example["condition"]].append(margin)
    return (
        sum(by_condition["expert"]) / len(by_condition["expert"])
        - sum(by_condition["novice"]) / len(by_condition["novice"])
    )


def projection_accuracy(examples, activations, vector, layer, tail_index, midpoint):
    unit = vector / vector.norm().clamp_min(1e-12)
    correct = 0
    for example in examples:
        score = float(activations[example["example_id"]][layer, tail_index] @ unit)
        prediction = "expert" if score > midpoint else "novice"
        correct += prediction == example["condition"]
    return correct / len(examples)


def direction_and_midpoint(train, activations, layer, tail_index, positive_ids=None):
    if positive_ids is None:
        positive_ids = {
            (e["condition"], e["profile_index"])
            for e in train if e["condition"] == "expert"
        }
    positive, negative = [], []
    for example in train:
        key = (example["condition"], example["profile_index"])
        destination = positive if key in positive_ids else negative
        destination.append(activations[example["example_id"]][layer, tail_index])
    pos_mean = torch.stack(positive).mean(0)
    neg_mean = torch.stack(negative).mean(0)
    vector = pos_mean - neg_mean
    unit = vector / vector.norm().clamp_min(1e-12)
    midpoint = float(((pos_mean @ unit) + (neg_mean @ unit)) / 2)
    return vector, midpoint


def exact_permutation_test(train, validation, tests, activations, n_layers, width):
    profile_ids = sorted({(e["condition"], e["profile_index"]) for e in train})
    true_positive = frozenset(key for key in profile_ids if key[0] == "expert")
    rows = []
    for positive_tuple in itertools.combinations(profile_ids, len(profile_ids) // 2):
        positive_ids = frozenset(positive_tuple)
        candidates = []
        for layer in range(n_layers):
            for tail_index in range(width):
                vector, midpoint = direction_and_midpoint(
                    train, activations, layer, tail_index, positive_ids
                )
                accuracy = projection_accuracy(
                    validation, activations, vector, layer, tail_index, midpoint
                )
                # Break ties using validation distance from the decision boundary.
                unit = vector / vector.norm().clamp_min(1e-12)
                signed_distances = []
                for example in validation:
                    score = float(
                        activations[example["example_id"]][layer, tail_index] @ unit
                    )
                    sign = 1 if example["condition"] == "expert" else -1
                    signed_distances.append(sign * (score - midpoint))
                candidates.append((accuracy, sum(signed_distances), layer, tail_index, vector, midpoint))
        selected = max(candidates, key=lambda row: (row[0], row[1]))
        accuracy, distance, layer, tail_index, vector, midpoint = selected
        row = {
            "is_true_labels": positive_ids == true_positive,
            "validation_accuracy": accuracy,
            "validation_signed_distance": distance,
            "selected_layer": layer,
            "selected_tail_position": tail_index - width,
        }
        for source, examples in tests.items():
            row[f"test_accuracy_{source}"] = projection_accuracy(
                examples, activations, vector, layer, tail_index, midpoint
            )
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--stimuli", type=Path, default=Path("stimuli.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tail-width", type=int, default=5)
    args = parser.parse_args()

    examples = [
        e for e in build_examples(load_stimuli(args.stimuli))
        if e["condition"] in {"expert", "novice"}
    ]
    train = [e for e in examples if e["source_id"] == "resnet" and e["split"] == "train"]
    validation = [e for e in examples if e["source_id"] == "resnet" and e["split"] == "validation"]
    tests = {
        source: [e for e in examples if e["source_id"] == source and e["split"] == "test"]
        for source in ("resnet", "transformer")
    }

    model, tokenizer = load_model(args.model)
    layers = list(get_layers(model))
    label_ids = label_token_ids(tokenizer)
    relevant = train + validation + tests["resnet"] + tests["transformer"]
    activations = {}
    for index, example in enumerate(relevant, 1):
        activations[example["example_id"]] = capture_tail(
            model, tokenizer, example["prompt"], args.tail_width
        )
        print(f"[activations {index}/{len(relevant)}] {example['example_id']}")

    validation_inputs = prepare_batch(model, tokenizer, validation)
    baseline_validation = run_batch(
        model, validation_inputs, validation, label_ids
    )
    baseline_validation_gap = audience_gap(validation, baseline_validation)
    coefficients = torch.tensor([
        1.0 if e["condition"] == "novice" else -1.0 for e in validation
    ])

    candidate_rows = []
    candidate_vectors = {}
    for layer_index, layer in enumerate(layers):
        for tail_index in range(args.tail_width):
            vector, midpoint = direction_and_midpoint(
                train, activations, layer_index, tail_index
            )
            candidate_vectors[(layer_index, tail_index)] = vector
            changed = run_batch(
                model,
                validation_inputs,
                validation,
                label_ids,
                add_all_positions(layer, vector, coefficients),
            )
            unit = vector / vector.norm().clamp_min(1e-12)
            ablated = run_batch(
                model,
                validation_inputs,
                validation,
                label_ids,
                ablate_all_positions([layer], unit),
            )
            row = {
                "layer": layer_index,
                "tail_position": tail_index - args.tail_width,
                "direction_norm": float(vector.norm()),
                "validation_decoding_accuracy": projection_accuracy(
                    validation, activations, vector, layer_index, tail_index, midpoint
                ),
                "validation_addition_effect": signed_addition_effect(
                    validation, baseline_validation, changed
                ),
                "validation_ablation_reduction": (
                    baseline_validation_gap - audience_gap(validation, ablated)
                ),
            }
            row["selection_score"] = min(
                row["validation_addition_effect"],
                row["validation_ablation_reduction"],
            )
            candidate_rows.append(row)
        print(f"[sweep {layer_index + 1}/{len(layers)}]")

    selected = max(candidate_rows, key=lambda row: row["selection_score"])
    selected_layer = int(selected["layer"])
    selected_tail_index = int(selected["tail_position"]) + args.tail_width
    vector = candidate_vectors[(selected_layer, selected_tail_index)]
    unit = vector / vector.norm().clamp_min(1e-12)

    test_rows = []
    for source, source_examples in tests.items():
        inputs = prepare_batch(model, tokenizer, source_examples)
        baseline = run_batch(model, inputs, source_examples, label_ids)
        coefficients = torch.tensor([
            1.0 if e["condition"] == "novice" else -1.0 for e in source_examples
        ])
        added = run_batch(
            model, inputs, source_examples, label_ids,
            add_all_positions(layers[selected_layer], vector, coefficients),
        )
        selected_ablation = run_batch(
            model, inputs, source_examples, label_ids,
            ablate_all_positions([layers[selected_layer]], unit),
        )
        global_ablation = run_batch(
            model, inputs, source_examples, label_ids,
            ablate_all_positions(layers, unit),
        )
        _, midpoint = direction_and_midpoint(
            train, activations, selected_layer, selected_tail_index
        )
        test_rows.append({
            "source_id": source,
            "baseline_gap": audience_gap(source_examples, baseline),
            "addition_effect": signed_addition_effect(source_examples, baseline, added),
            "selected_layer_ablation_gap": audience_gap(source_examples, selected_ablation),
            "selected_layer_ablation_reduction": (
                audience_gap(source_examples, baseline)
                - audience_gap(source_examples, selected_ablation)
            ),
            "all_layer_ablation_gap": audience_gap(source_examples, global_ablation),
            "all_layer_ablation_reduction": (
                audience_gap(source_examples, baseline)
                - audience_gap(source_examples, global_ablation)
            ),
            "decoding_accuracy": projection_accuracy(
                source_examples, activations, vector,
                selected_layer, selected_tail_index, midpoint
            ),
        })

    permutation_rows = exact_permutation_test(
        train, validation, tests, activations, len(layers), args.tail_width
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "candidate_sweep.csv", candidate_rows)
    write_rows(args.output_dir / "test_summary.csv", test_rows)
    write_rows(args.output_dir / "decoder_permutations.csv", permutation_rows)
    torch.save({
        "selected_layer": selected_layer,
        "selected_tail_position": int(selected["tail_position"]),
        "selected_direction": vector,
        "selection_rule": "maximize min(validation addition effect, validation ablation reduction)",
    }, args.output_dir / "selected_direction.pt")
    (args.output_dir / "config.json").write_text(json.dumps({
        "model": args.model,
        "stimuli": str(args.stimuli),
        "tail_width": args.tail_width,
        "selection_rule": "maximize min(validation addition effect, validation ablation reduction)",
    }, indent=2))
    print("selected", selected)
    for row in test_rows:
        print(row)


if __name__ == "__main__":
    main()
