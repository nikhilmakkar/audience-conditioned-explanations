"""Localize direction-aligned causal effects to attention or MLP outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from experiment import build_examples, get_layers, label_token_ids, load_model, load_stimuli
from layer_token_sweep import (
    ablate_all_positions,
    audience_gap,
    margins_from_logits,
    prepare_batch,
    run_batch,
)


def component_modules(layer):
    attention = getattr(layer, "self_attn", None)
    if attention is None:
        attention = getattr(layer, "linear_attn", None)
    mlp = getattr(layer, "mlp", None)
    if attention is None or mlp is None:
        raise ValueError("Expected an attention/linear-attention module and layer.mlp")
    return {"attention": attention, "mlp": mlp}


def capture_component_projections(model, inputs, layers, unit, tail_index):
    cache = {}
    handles = []

    def make_hook(layer_index, component):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            position = hidden.shape[1] + tail_index
            direction = unit.to(device=hidden.device, dtype=torch.float32)
            cache[(layer_index, component)] = (
                hidden[:, position].float() @ direction
            ).detach().cpu()
        return hook

    for layer_index, layer in enumerate(layers):
        for component, module in component_modules(layer).items():
            handles.append(module.register_forward_hook(make_hook(layer_index, component)))
    try:
        with torch.inference_mode():
            model(**inputs)
    finally:
        for handle in handles:
            handle.remove()
    return cache


def projection_gap(examples, values):
    grouped = {"expert": [], "novice": []}
    for example, value in zip(examples, values):
        grouped[example["condition"]].append(float(value))
    return (
        sum(grouped["expert"]) / len(grouped["expert"])
        - sum(grouped["novice"]) / len(grouped["novice"])
    )


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--stimuli", type=Path, default=Path("stimuli.json"))
    parser.add_argument("--direction", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    saved = torch.load(args.direction, map_location="cpu", weights_only=True)
    selected_direction = saved["selected_direction"].float()
    unit = selected_direction / selected_direction.norm().clamp_min(1e-12)
    tail_position = int(saved["selected_tail_position"])
    model, tokenizer = load_model(args.model)
    layers = list(get_layers(model))
    labels = label_token_ids(tokenizer)
    examples = [
        e for e in build_examples(load_stimuli(args.stimuli))
        if e["condition"] in {"expert", "novice"}
    ]
    validation = [
        e for e in examples if e["source_id"] == "resnet" and e["split"] == "validation"
    ]
    tests = {
        source: [e for e in examples if e["source_id"] == source and e["split"] == "test"]
        for source in ("resnet", "transformer")
    }

    validation_inputs = prepare_batch(model, tokenizer, validation)
    baseline_validation = run_batch(model, validation_inputs, validation, labels)
    baseline_gap = audience_gap(validation, baseline_validation)
    projections = capture_component_projections(
        model, validation_inputs, layers, unit, tail_position
    )

    sweep_rows = []
    for layer_index, layer in enumerate(layers):
        for component, module in component_modules(layer).items():
            changed = run_batch(
                model, validation_inputs, validation, labels,
                ablate_all_positions([module], unit),
            )
            sweep_rows.append({
                "layer": layer_index,
                "component": component,
                "validation_projection_gap": projection_gap(
                    validation, projections[(layer_index, component)]
                ),
                "validation_baseline_gap": baseline_gap,
                "validation_ablated_gap": audience_gap(validation, changed),
                "validation_gap_reduction": baseline_gap - audience_gap(validation, changed),
            })

    selected_by_component = {}
    for component in ("attention", "mlp"):
        candidates = [row for row in sweep_rows if row["component"] == component]
        selected_by_component[component] = max(
            candidates, key=lambda row: row["validation_gap_reduction"]
        )

    test_rows = []
    for source, source_examples in tests.items():
        inputs = prepare_batch(model, tokenizer, source_examples)
        baseline = run_batch(model, inputs, source_examples, labels)
        base_gap = audience_gap(source_examples, baseline)
        chosen_modules = []
        for component, selected in selected_by_component.items():
            layer_index = int(selected["layer"])
            module = component_modules(layers[layer_index])[component]
            chosen_modules.append(module)
            changed = run_batch(
                model, inputs, source_examples, labels,
                ablate_all_positions([module], unit),
            )
            changed_gap = audience_gap(source_examples, changed)
            test_rows.append({
                "source_id": source,
                "intervention": component,
                "selected_layer": layer_index,
                "baseline_gap": base_gap,
                "ablated_gap": changed_gap,
                "gap_reduction": base_gap - changed_gap,
            })
        changed = run_batch(
            model, inputs, source_examples, labels,
            ablate_all_positions(chosen_modules, unit),
        )
        changed_gap = audience_gap(source_examples, changed)
        test_rows.append({
            "source_id": source,
            "intervention": "selected_attention_and_mlp",
            "selected_layer": "+".join(
                str(selected_by_component[name]["layer"])
                for name in ("attention", "mlp")
            ),
            "baseline_gap": base_gap,
            "ablated_gap": changed_gap,
            "gap_reduction": base_gap - changed_gap,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "component_validation_sweep.csv", sweep_rows)
    write_rows(args.output_dir / "component_test_summary.csv", test_rows)
    print("selected", selected_by_component)
    for row in test_rows:
        print(row)


if __name__ == "__main__":
    main()
