import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    data = json.loads((ROOT / "data" / "stimuli_matched.json").read_text())
    assert len(data["sources"]) == 4
    assert set(data["profiles"]) == {
        "relevant_researcher", "irrelevant_researcher",
        "relevant_practitioner", "irrelevant_practitioner",
    }
    for condition, profiles in data["profiles"].items():
        assert len(profiles) == 9, condition
        assert [p["split"] for p in profiles].count("train") == 4
        assert [p["split"] for p in profiles].count("validation") == 2
        assert [p["split"] for p in profiles].count("test") == 3
        assert len({p["text"] for p in profiles}) == 9
    for source in data["sources"]:
        technical_words = len(source["technical_summary"].split())
        accessible_words = len(source["accessible_summary"].split())
        assert 0.75 <= technical_words / accessible_words <= 1.25, (
            source["id"], technical_words, accessible_words
        )
    total = len(data["sources"]) * sum(len(v) for v in data["profiles"].values()) * 2
    assert total == 288
    print(f"Matched design checks passed: 4 sources, {total} balanced prompts.")


if __name__ == "__main__":
    main()
