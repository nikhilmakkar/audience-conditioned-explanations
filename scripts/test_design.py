import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_data():
    return json.loads((ROOT / "data" / "stimuli.json").read_text())


def main():
    data = load_data()
    source_ids = [source["id"] for source in data["sources"]]
    assert source_ids == ["resnet", "transformer"]

    required_splits = {"train", "validation", "test"}
    for condition, profiles in data["profiles"].items():
        assert len(profiles) >= 6, condition
        assert required_splits <= {profile["split"] for profile in profiles}, condition
        assert len({profile["text"] for profile in profiles}) == len(profiles), condition

    for source in data["sources"]:
        assert source["technical_summary"] != source["accessible_summary"]
        technical_words = len(source["technical_summary"].split())
        accessible_words = len(source["accessible_summary"].split())
        ratio = technical_words / accessible_words
        assert 0.70 <= ratio <= 1.30, (source["id"], technical_words, accessible_words)

    # Every constructed cell will be exactly balanced for label order.
    total = 0
    for source in data["sources"]:
        for profiles in data["profiles"].values():
            total += 2 * len(profiles)
    print(f"Design checks passed: {len(data['sources'])} sources, {total} prompts.")


if __name__ == "__main__":
    main()
