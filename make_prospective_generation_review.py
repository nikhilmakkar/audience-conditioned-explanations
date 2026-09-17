"""Join blinded generation outputs to source passages for human review."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stimuli", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = {s["id"]: s for s in json.loads(args.stimuli.read_text())["sources"]}
    with args.review.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    output = []
    for row in rows:
        source = sources[row["source_id"]]
        output.append({
            "blind_code": row["blind_code"],
            "source_id": row["source_id"],
            "source_passage": source["passage"],
            "generation": row["text"],
            "human_technicality_1_to_5": "",
            "human_correct_0_or_1": "",
            "human_coherent_0_or_1": "",
            "notes": "",
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader(); writer.writerows(output)
    print(f"wrote {len(output)} blinded rows to {args.output}")


if __name__ == "__main__":
    main()
