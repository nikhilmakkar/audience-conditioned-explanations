import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    data = json.loads((ROOT / "stimuli_domain_matched.json").read_text())
    assert "profiles" not in data
    for source in data["sources"]:
        assert set(source["profiles"]) == {"domain_expert", "matched_other_expert"}
        left = source["profiles"]["domain_expert"]
        right = source["profiles"]["matched_other_expert"]
        assert len(left) == len(right) == 8
        assert [p["split"] for p in left] == [p["split"] for p in right]
        for a, b in zip(left, right):
            assert a["text"].split(". ", 1)[1] == b["text"].split(". ", 1)[1]
    total = sum(
        2 * sum(len(v) for v in source["profiles"].values())
        for source in data["sources"]
    )
    assert total == 128
    print(f"Domain-matched checks passed: 4 sources, {total} balanced prompts.")


if __name__ == "__main__":
    main()
