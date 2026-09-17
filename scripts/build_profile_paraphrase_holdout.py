"""Replace profile carrier language without changing fields or source passages."""

from __future__ import annotations

import json
from pathlib import Path


SOURCE = Path("data/stimuli_prospective_synthetic.json")
OUTPUT = Path("data/stimuli_prospective_profile_paraphrase.json")


def field(text):
    prefix = "The reader is deeply familiar with "
    if not text.startswith(prefix):
        raise ValueError(text)
    return text[len(prefix):].split(". They want", 1)[0]


def main():
    data = json.loads(SOURCE.read_text())
    for source in data["sources"]:
        for profiles in source["profiles"].values():
            subject = field(profiles[0]["text"])
            profiles[:] = [
                {"split": "test", "text": (
                    f"The intended audience specializes professionally in {subject}. "
                    "Give them a compact account of how the method works."
                )},
                {"split": "test", "text": (
                    f"The explanation is for researchers whose main area is {subject}. "
                    "Present the central operation at a level suited to their background."
                )},
            ]
    OUTPUT.write_text(json.dumps(data, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
