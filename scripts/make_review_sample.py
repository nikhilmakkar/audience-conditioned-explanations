"""Create a deterministic, stratified raw-output sheet for human review."""

import csv
import json
import random
from pathlib import Path

from experiment import build_generation_examples


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260907


def main():
    stimuli = json.loads((ROOT / "data" / "stimuli_domain_matched.json").read_text())
    prompts = {
        row["example_id"]: row["prompt"]
        for row in build_generation_examples(stimuli)
    }
    rng = random.Random(SEED)
    sample = []
    for model in ("qwen25_3b", "qwen35_4b"):
        path = ROOT / f"results/{model}_domain_matched_generation_180/generation_rows.csv"
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        for source in sorted({row["source_id"] for row in rows}):
            for condition in ("domain_expert", "matched_other_expert"):
                candidates = [
                    row for row in rows
                    if row["source_id"] == source and row["condition"] == condition
                ]
                chosen = dict(rng.choice(candidates))
                chosen["model"] = model
                sample.append(chosen)
    rng.shuffle(sample)

    output = []
    for row in sample:
        output.append(
            {
                "model": row["model"],
                "example_id": row["example_id"],
                "prompt": prompts[row["example_id"]],
                "generation": row["generation"],
                "human_factually_correct_yes_no": "",
                "human_audience_fit_1_to_5": "",
                "human_notes": "",
            }
        )
    path = ROOT / "results/manual_review_sample.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"Wrote {len(output)} examples using seed {SEED} to {path}")


if __name__ == "__main__":
    main()
