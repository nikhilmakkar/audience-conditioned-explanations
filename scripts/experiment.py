"""Behavioral and activation-steering pilot for audience-conditioned summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ROOT = Path(__file__).resolve().parents[1]


def load_stimuli(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def build_examples(data: dict[str, Any]) -> list[dict[str, Any]]:
    examples = []
    for source in data["sources"]:
        profile_bank = source.get("profiles", data.get("profiles"))
        if profile_bank is None:
            raise ValueError(f"No profiles available for source {source['id']}")
        for condition, profiles in profile_bank.items():
            for profile_index, profile in enumerate(profiles):
                for technical_label in ("A", "B"):
                    summaries = {
                        technical_label: source["technical_summary"],
                        "B" if technical_label == "A" else "A": source["accessible_summary"],
                    }
                    prompt = (
                        f"{data['instruction']}\n\n"
                        f"Reader profile:\n{profile['text']}\n\n"
                        f"Source passage:\n{source['passage']}\n\n"
                        f"Summary A:\n{summaries['A']}\n\n"
                        f"Summary B:\n{summaries['B']}\n\n"
                        "Answer:"
                    )
                    examples.append(
                        {
                            "example_id": (
                                f"{source['id']}:{condition}:{profile['split']}:"
                                f"{profile_index}:{technical_label}"
                            ),
                            "source_id": source["id"],
                            "source_role": source["role"],
                            "condition": condition,
                            "split": profile["split"],
                            "profile_index": profile_index,
                            "technical_label": technical_label,
                            "prompt": prompt,
                        }
                    )
    return examples


def build_generation_examples(data: dict[str, Any]) -> list[dict[str, Any]]:
    examples = []
    for source in data["sources"]:
        profile_bank = source.get("profiles", data.get("profiles"))
        if profile_bank is None:
            raise ValueError(f"No profiles available for source {source['id']}")
        for condition, profiles in profile_bank.items():
            for profile_index, profile in enumerate(profiles):
                prompt = (
                    "Write a concise 2-3 sentence gist of the source passage for the "
                    "described reader. Preserve the central mechanism and use the level "
                    "of technical detail useful to that reader. Do not mention the reader "
                    "profile.\n\n"
                    f"Reader profile:\n{profile['text']}\n\n"
                    f"Source passage:\n{source['passage']}\n\n"
                    "Gist:"
                )
                examples.append(
                    {
                        "example_id": f"{source['id']}:{condition}:{profile['split']}:{profile_index}",
                        "source_id": source["id"],
                        "condition": condition,
                        "split": profile["split"],
                        "profile_index": profile_index,
                        "technical_terms": source.get("technical_terms", []),
                        "prompt": prompt,
                    }
                )
    return examples


def get_layers(model):
    candidates = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]
    for parent_name, child_name in candidates:
        parent = getattr(model, parent_name, None)
        if parent is not None and hasattr(parent, child_name):
            return getattr(parent, child_name)
    raise ValueError("Unsupported architecture: could not locate transformer layers")


def load_model(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def format_prompt(tokenizer, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return prompt


def label_token_ids(tokenizer) -> dict[str, int]:
    result = {}
    for label in ("A", "B"):
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"Label {label!r} tokenizes to {ids}; choose single-token labels for this model"
            )
        result[label] = ids[0]
    return result


def prepare_inputs(model, tokenizer, prompt: str):
    rendered = format_prompt(tokenizer, prompt)
    inputs = tokenizer(rendered, return_tensors="pt")
    device = next(model.parameters()).device
    return {key: value.to(device) for key, value in inputs.items()}


def next_token_scores(model, tokenizer, prompt: str, label_ids: dict[str, int]):
    import torch

    inputs = prepare_inputs(model, tokenizer, prompt)
    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1].float()
        log_probs = torch.log_softmax(logits, dim=-1)
    return {label: float(log_probs[token_id].cpu()) for label, token_id in label_ids.items()}


def technical_margin(scores: dict[str, float], technical_label: str) -> float:
    accessible_label = "B" if technical_label == "A" else "A"
    return scores[technical_label] - scores[accessible_label]


def write_rows(path: Path, rows: Iterable[dict[str, Any]]):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_behavior(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # A/B swaps are repeated measurements of one profile, not independent rows.
    profile_groups = defaultdict(list)
    for row in rows:
        key = (
            row["source_id"],
            row["condition"],
            row["split"],
            row["profile_index"],
        )
        profile_groups[key].append(float(row["technical_margin"]))
    groups = defaultdict(list)
    for key, values in profile_groups.items():
        if len(values) != 2:
            raise ValueError(
                f"Expected two A/B order measurements for {key}, got {len(values)}"
            )
        groups[key[:3]].append(sum(values) / len(values))
    summary = []
    for (source_id, condition, split), values in sorted(groups.items()):
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
        summary.append(
            {
                "source_id": source_id,
                "condition": condition,
                "split": split,
                "n_profiles": len(values),
                "mean_technical_margin": mean,
                "standard_error": math.sqrt(variance / len(values)),
            }
        )
    return summary


def run_behavior(args):
    data = load_stimuli(args.stimuli)
    examples = build_examples(data)
    model, tokenizer = load_model(args.model)
    label_ids = label_token_ids(tokenizer)
    rows = []
    for index, example in enumerate(examples, start=1):
        scores = next_token_scores(model, tokenizer, example["prompt"], label_ids)
        rows.append(
            {
                key: value for key, value in example.items() if key != "prompt"
            }
            | {
                "logp_A": scores["A"],
                "logp_B": scores["B"],
                "technical_margin": technical_margin(scores, example["technical_label"]),
            }
        )
        print(f"[{index}/{len(examples)}] {example['example_id']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "behavior_rows.csv", rows)
    summary = summarize_behavior(rows)
    write_rows(args.output_dir / "behavior_summary.csv", summary)
    (args.output_dir / "config.json").write_text(
        json.dumps({"model": args.model, "stimuli": str(args.stimuli)}, indent=2)
    )
    for row in summary:
        if row["split"] == "test":
            print(row)


def capture_final_activations(model, tokenizer, prompt: str):
    import torch

    layers = get_layers(model)
    cache = [None] * len(layers)
    handles = []

    def make_hook(layer_index):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            cache[layer_index] = hidden[0, -1].detach().float().cpu()
        return hook

    for layer_index, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(layer_index)))
    try:
        inputs = prepare_inputs(model, tokenizer, prompt)
        with torch.inference_mode():
            model(**inputs)
    finally:
        for handle in handles:
            handle.remove()
    return torch.stack(cache)


def mean_tensor(tensors):
    import torch

    return torch.stack(tensors).mean(dim=0)


def projection(vector, direction):
    import torch

    unit = direction / direction.norm().clamp_min(1e-12)
    return float((vector @ unit).item())


def fit_directions(examples, activations, positive="expert", negative="novice"):
    pos = [
        activations[e["example_id"]]
        for e in examples
        if e["source_id"] == "resnet" and e["split"] == "train" and e["condition"] == positive
    ]
    neg = [
        activations[e["example_id"]]
        for e in examples
        if e["source_id"] == "resnet" and e["split"] == "train" and e["condition"] == negative
    ]
    if not pos or not neg:
        raise ValueError(f"Missing training examples for {positive} versus {negative}")
    return mean_tensor(pos) - mean_tensor(neg), mean_tensor(pos), mean_tensor(neg)


def layer_accuracy(examples, activations, directions, pos_mean, neg_mean, split):
    accuracies = []
    for layer_index in range(directions.shape[0]):
        direction = directions[layer_index]
        midpoint = (
            projection(pos_mean[layer_index], direction)
            + projection(neg_mean[layer_index], direction)
        ) / 2
        correct = total = 0
        for example in examples:
            if not (
                example["source_id"] == "resnet"
                and example["split"] == split
                and example["condition"] in {"expert", "novice"}
            ):
                continue
            score = projection(activations[example["example_id"]][layer_index], direction)
            predicted = "expert" if score > midpoint else "novice"
            correct += predicted == example["condition"]
            total += 1
        accuracies.append(correct / total)
    return accuracies


@contextmanager
def add_at_final_token(layer, vector, alpha: float):
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        modified = hidden.clone()
        modified[:, -1, :] += alpha * vector.to(device=hidden.device, dtype=hidden.dtype)
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def steered_scores(model, tokenizer, prompt, label_ids, layer, vector, alpha):
    import torch

    inputs = prepare_inputs(model, tokenizer, prompt)
    with add_at_final_token(layer, vector, alpha):
        with torch.inference_mode():
            logits = model(**inputs).logits[0, -1].float()
            log_probs = torch.log_softmax(logits, dim=-1)
    return {label: float(log_probs[token_id].cpu()) for label, token_id in label_ids.items()}


def validation_causal_effect(
    model, tokenizer, examples, label_ids, layer, vector, baseline_margins
):
    """Expected signed effect of the direction on held-out ResNet validation prompts."""
    effects = []
    for example in examples:
        if not (
            example["source_id"] == "resnet"
            and example["split"] == "validation"
            and example["condition"] in {"expert", "novice"}
        ):
            continue
        alpha = 1.0 if example["condition"] == "novice" else -1.0
        scores = steered_scores(
            model, tokenizer, example["prompt"], label_ids, layer, vector, alpha
        )
        steered_margin = technical_margin(scores, example["technical_label"])
        baseline = baseline_margins[example["example_id"]]
        effects.append(
            steered_margin - baseline
            if example["condition"] == "novice"
            else baseline - steered_margin
        )
    return sum(effects) / len(effects)


def run_direction(args):
    import torch

    data = load_stimuli(args.stimuli)
    examples = build_examples(data)
    model, tokenizer = load_model(args.model)
    label_ids = label_token_ids(tokenizer)

    relevant = [
        example
        for example in examples
        if example["condition"] in {"expert", "novice", "academic_nonml"}
    ]
    activations = {}
    for index, example in enumerate(relevant, start=1):
        activations[example["example_id"]] = capture_final_activations(
            model, tokenizer, example["prompt"]
        )
        print(f"[activations {index}/{len(relevant)}] {example['example_id']}")

    directions, pos_mean, neg_mean = fit_directions(relevant, activations)
    validation_accuracy = layer_accuracy(
        relevant, activations, directions, pos_mean, neg_mean, "validation"
    )
    layers = get_layers(model)
    # Break validation-accuracy ties with a causal criterion on validation prompts.
    # Test prompts remain untouched until after the layer has been selected.
    best_validation = max(validation_accuracy)
    candidates = [
        index
        for index, accuracy in enumerate(validation_accuracy)
        if accuracy == best_validation
    ]
    validation_examples = [
        example
        for example in relevant
        if example["source_id"] == "resnet"
        and example["split"] == "validation"
        and example["condition"] in {"expert", "novice"}
    ]
    baseline_margins = {}
    for example in validation_examples:
        scores = steered_scores(
            model, tokenizer, example["prompt"], label_ids, layers[0], directions[0], 0.0
        )
        baseline_margins[example["example_id"]] = technical_margin(
            scores, example["technical_label"]
        )
    validation_effect = [float("nan")] * len(validation_accuracy)
    for layer_index in candidates:
        validation_effect[layer_index] = validation_causal_effect(
            model,
            tokenizer,
            relevant,
            label_ids,
            layers[layer_index],
            directions[layer_index],
            baseline_margins,
        )
    selected_layer = max(candidates, key=lambda index: validation_effect[index])
    test_accuracy = layer_accuracy(
        relevant, activations, directions, pos_mean, neg_mean, "test"
    )

    academic_direction, _, _ = fit_directions(
        relevant, activations, positive="academic_nonml", negative="novice"
    )
    steering_rows = []
    test_examples = [example for example in relevant if example["split"] == "test"]
    for direction_name, direction_bank in (
        ("ml_expert_minus_novice", directions),
        ("nonml_academic_minus_novice", academic_direction),
    ):
        vector = direction_bank[selected_layer]
        for example in test_examples:
            for alpha in args.alphas:
                scores = steered_scores(
                    model,
                    tokenizer,
                    example["prompt"],
                    label_ids,
                    layers[selected_layer],
                    vector,
                    alpha,
                )
                steering_rows.append(
                    {
                        "direction": direction_name,
                        "selected_layer": selected_layer,
                        "example_id": example["example_id"],
                        "source_id": example["source_id"],
                        "condition": example["condition"],
                        "technical_label": example["technical_label"],
                        "alpha": alpha,
                        "logp_A": scores["A"],
                        "logp_B": scores["B"],
                        "technical_margin": technical_margin(scores, example["technical_label"]),
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "expert_minus_novice": directions,
            "academic_minus_novice": academic_direction,
            "selected_layer": selected_layer,
        },
        args.output_dir / "directions.pt",
    )
    layer_rows = [
        {
            "layer": layer_index,
            "validation_accuracy": validation_accuracy[layer_index],
            "test_accuracy": test_accuracy[layer_index],
            "validation_causal_effect": validation_effect[layer_index],
            "direction_norm": float(directions[layer_index].norm().item()),
        }
        for layer_index in range(len(validation_accuracy))
    ]
    write_rows(args.output_dir / "layer_results.csv", layer_rows)
    write_rows(args.output_dir / "steering_rows.csv", steering_rows)
    print(
        f"Selected layer {selected_layer}: validation accuracy "
        f"{validation_accuracy[selected_layer]:.3f}, validation causal effect "
        f"{validation_effect[selected_layer]:.3f}, test accuracy "
        f"{test_accuracy[selected_layer]:.3f}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("behavior", "direction"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--stimuli", type=Path, default=ROOT / "data" / "stimuli.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=[-2, -1, 0, 1, 2])
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "behavior":
        run_behavior(args)
    else:
        run_direction(args)


if __name__ == "__main__":
    main()
