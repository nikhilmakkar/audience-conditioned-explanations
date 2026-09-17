"""Test whether the learned audience direction is necessary at one layer/token."""

import argparse
import csv
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from statistics import mean

import torch

from experiment import (
    build_examples,
    capture_final_activations,
    get_layers,
    label_token_ids,
    load_model,
    load_stimuli,
    next_token_scores,
    prepare_inputs,
    technical_margin,
)


@contextmanager
def set_final_projection(layer, unit_direction, target_projection):
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        modified = hidden.clone()
        unit = unit_direction.to(device=hidden.device, dtype=torch.float32)
        current = modified[:, -1, :].float() @ unit
        delta = (target_projection - current).to(modified.dtype)
        modified[:, -1, :] += delta[:, None] * unit.to(modified.dtype)[None, :]
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def ablated_scores(model, tokenizer, prompt, label_ids, layer, unit, target):
    inputs = prepare_inputs(model, tokenizer, prompt)
    with set_final_projection(layer, unit, target):
        with torch.inference_mode():
            logits = model(**inputs).logits[0, -1].float()
            log_probs = torch.log_softmax(logits, dim=-1)
    return {
        label: float(log_probs[token_id].cpu())
        for label, token_id in label_ids.items()
    }


def paired_condition_means(rows, field):
    pairs = defaultdict(list)
    for row in rows:
        profile = row["example_id"].rsplit(":", 1)[0]
        pairs[(row["source_id"], row["condition"], profile)].append(row[field])
    condition = defaultdict(list)
    for (source, label, _profile), values in pairs.items():
        assert len(values) == 2
        condition[(source, label)].append(mean(values))
    return {key: mean(values) for key, values in condition.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--stimuli", type=Path, default=Path("data/stimuli.json"))
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()

    saved = torch.load(args.directions, map_location="cpu", weights_only=True)
    selected_layer = int(saved["selected_layer"])
    learned = saved["expert_minus_novice"][selected_layer].float()
    learned /= learned.norm()

    examples = build_examples(load_stimuli(args.stimuli))
    train = [
        example for example in examples
        if example["source_id"] == "resnet"
        and example["split"] == "train"
        and example["condition"] in {"expert", "novice"}
    ]
    test = [
        example for example in examples
        if example["split"] == "test"
        and example["condition"] in {"expert", "novice"}
    ]

    model, tokenizer = load_model(args.model)
    layer = get_layers(model)[selected_layer]
    label_ids = label_token_ids(tokenizer)
    train_hidden = torch.stack([
        capture_final_activations(model, tokenizer, example["prompt"])[selected_layer]
        for example in train
    ])

    baseline_rows = []
    for example in test:
        scores = next_token_scores(model, tokenizer, example["prompt"], label_ids)
        baseline_rows.append(
            {
                "example_id": example["example_id"],
                "source_id": example["source_id"],
                "condition": example["condition"],
                "technical_margin": technical_margin(scores, example["technical_label"]),
            }
        )
    baseline_means = paired_condition_means(baseline_rows, "technical_margin")

    vectors = [("learned", -1, learned)]
    for seed in range(args.seeds):
        generator = torch.Generator().manual_seed(seed)
        vector = torch.randn(learned.shape, generator=generator)
        vectors.append(("random", seed, vector / vector.norm()))

    rows = []
    summaries = []
    for kind, seed, unit in vectors:
        target = float((train_hidden @ unit).mean())
        current_rows = []
        for example in test:
            scores = ablated_scores(
                model, tokenizer, example["prompt"], label_ids,
                layer, unit, target,
            )
            row = {
                "kind": kind,
                "seed": seed,
                "example_id": example["example_id"],
                "source_id": example["source_id"],
                "condition": example["condition"],
                "technical_label": example["technical_label"],
                "technical_margin": technical_margin(scores, example["technical_label"]),
            }
            rows.append(row)
            current_rows.append(row)
        ablated_means = paired_condition_means(current_rows, "technical_margin")
        for source in sorted({example["source_id"] for example in test}):
            baseline_gap = (
                baseline_means[(source, "expert")]
                - baseline_means[(source, "novice")]
            )
            ablated_gap = (
                ablated_means[(source, "expert")]
                - ablated_means[(source, "novice")]
            )
            summaries.append(
                {
                    "kind": kind,
                    "seed": seed,
                    "source_id": source,
                    "baseline_gap": baseline_gap,
                    "ablated_gap": ablated_gap,
                    "gap_reduction": baseline_gap - ablated_gap,
                }
            )
        print(f"completed {kind} ablation {seed}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in (("ablation_rows.csv", rows), ("ablation_summary.csv", summaries)):
        with (args.output_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)


if __name__ == "__main__":
    main()
