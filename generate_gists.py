"""Generate matched-reader gists and save objective surface metrics."""

import argparse
import csv
import json
from pathlib import Path

import torch

from experiment import build_generation_examples, load_model, load_stimuli, prepare_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--stimuli", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    args = parser.parse_args()

    examples = build_generation_examples(load_stimuli(args.stimuli))
    model, tokenizer = load_model(args.model)
    rows = []
    for index, example in enumerate(examples, start=1):
        inputs = prepare_inputs(model, tokenizer, example["prompt"])
        input_length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            output[0, input_length:], skip_special_tokens=True
        ).strip()
        lowered = generated.lower()
        matched_terms = [
            term for term in example["technical_terms"] if term.lower() in lowered
        ]
        rows.append(
            {
                key: value
                for key, value in example.items()
                if key not in {"prompt", "technical_terms"}
            }
            | {
                "word_count": len(generated.split()),
                "technical_term_count": len(matched_terms),
                "matched_terms": "|".join(matched_terms),
                "generation": generated,
            }
        )
        print(f"[{index}/{len(examples)}] {example['example_id']}: {generated}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "generation_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "config.json").write_text(
        json.dumps({"model": args.model, "stimuli": str(args.stimuli)}, indent=2)
    )


if __name__ == "__main__":
    main()
