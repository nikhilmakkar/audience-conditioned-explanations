"""Compare the learned audience direction with norm-matched random directions."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch

from experiment import (
    build_examples,
    get_layers,
    label_token_ids,
    load_model,
    load_stimuli,
    steered_scores,
    technical_margin,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--stimuli", type=Path, default=Path("data/stimuli.json"))
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=2.0)
    args = parser.parse_args()

    saved = torch.load(args.directions, map_location="cpu", weights_only=True)
    selected_layer = int(saved["selected_layer"])
    learned = saved["expert_minus_novice"][selected_layer].float()
    model, tokenizer = load_model(args.model)
    layer = get_layers(model)[selected_layer]
    label_ids = label_token_ids(tokenizer)
    examples = [
        example
        for example in build_examples(load_stimuli(args.stimuli))
        if example["condition"] == "novice" and example["split"] == "test"
    ]

    vectors = [("learned", -1, learned)]
    for seed in range(args.seeds):
        generator = torch.Generator().manual_seed(seed)
        vector = torch.randn(learned.shape, generator=generator)
        vector *= learned.norm() / vector.norm()
        vectors.append(("random", seed, vector))

    rows = []
    for kind, seed, vector in vectors:
        for example in examples:
            for alpha in (-args.alpha, args.alpha):
                scores = steered_scores(
                    model, tokenizer, example["prompt"], label_ids,
                    layer, vector, alpha,
                )
                rows.append(
                    {
                        "kind": kind,
                        "seed": seed,
                        "example_id": example["example_id"],
                        "source_id": example["source_id"],
                        "profile_index": example["profile_index"],
                        "technical_label": example["technical_label"],
                        "alpha": alpha,
                        "technical_margin": technical_margin(
                            scores, example["technical_label"]
                        ),
                    }
                )
        print(f"completed {kind} direction {seed}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "random_steering_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        profile = row["example_id"].rsplit(":", 1)[0]
        grouped[(row["kind"], row["seed"], row["source_id"], profile, row["alpha"])].append(
            row["technical_margin"]
        )
    paired = {key: mean(values) for key, values in grouped.items()}
    summaries = []
    for kind, seed, _ in vectors:
        for source in sorted({example["source_id"] for example in examples}):
            profiles = sorted(
                {key[3] for key in paired if key[:3] == (kind, seed, source)}
            )
            slopes = [
                (
                    paired[(kind, seed, source, profile, args.alpha)]
                    - paired[(kind, seed, source, profile, -args.alpha)]
                )
                / (2 * args.alpha)
                for profile in profiles
            ]
            summaries.append(
                {"kind": kind, "seed": seed, "source_id": source, "mean_slope": mean(slopes)}
            )
    with (args.output_dir / "random_steering_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (args.output_dir / "config.json").write_text(
        json.dumps(vars(args), default=str, indent=2)
    )


if __name__ == "__main__":
    main()
