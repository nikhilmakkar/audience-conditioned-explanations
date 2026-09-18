from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

from exact_relevance_lodo import NEGATIVE, POSITIVE, mean_source_gap, signed_effect
from experiment import build_examples, load_stimuli, summarize_behavior, technical_margin
from reencoding_control import ids_for, relabel


ROOT = Path(__file__).resolve().parents[1]


def test_build_examples_balances_answer_order() -> None:
    data = load_stimuli(ROOT / "data" / "stimuli_prospective_synthetic.json")
    examples = build_examples(data)
    grouped: dict[tuple[object, ...], list[dict]] = defaultdict(list)

    for example in examples:
        key = (
            example["source_id"],
            example["condition"],
            example["split"],
            example["profile_index"],
        )
        grouped[key].append(example)

    assert grouped
    assert all({row["technical_label"] for row in rows} == {"A", "B"}
               for rows in grouped.values())
    assert all(len(rows) == 2 for rows in grouped.values())
    assert len(examples) == 64


def test_relabel_changes_labels_without_changing_content() -> None:
    data = load_stimuli(ROOT / "data" / "stimuli_prospective_synthetic.json")
    original = build_examples(data)[0]
    changed = relabel(original, "X", "Y")

    assert "exactly X or Y" in changed["prompt"]
    assert "Summary X:" in changed["prompt"]
    assert "Summary Y:" in changed["prompt"]
    assert changed["technical_label"] == (
        "X" if original["technical_label"] == "A" else "Y"
    )
    assert original["prompt"] != changed["prompt"]


def test_ids_for_rejects_multitoken_labels() -> None:
    class Tokenizer:
        def encode(self, label, add_special_tokens=False):
            assert not add_special_tokens
            return [1] if label == "A" else [2, 3]

    with pytest.raises(ValueError, match="not one token"):
        ids_for(Tokenizer(), ("A", "long"))


def test_summary_averages_answer_order_within_profile() -> None:
    rows = [
        {"source_id": "s", "condition": "c", "split": "test",
         "profile_index": 0, "technical_margin": 1.0},
        {"source_id": "s", "condition": "c", "split": "test",
         "profile_index": 0, "technical_margin": 3.0},
        {"source_id": "s", "condition": "c", "split": "test",
         "profile_index": 1, "technical_margin": 5.0},
        {"source_id": "s", "condition": "c", "split": "test",
         "profile_index": 1, "technical_margin": 7.0},
    ]

    summary = summarize_behavior(rows)
    assert len(summary) == 1
    assert summary[0]["n_profiles"] == 2
    assert summary[0]["mean_technical_margin"] == pytest.approx(4.0)


def test_margin_and_audience_metrics_have_explicit_orientation() -> None:
    assert technical_margin({"A": -1.0, "B": -3.0}, "A") == 2.0
    examples = [
        {"source_id": "s1", "condition": POSITIVE},
        {"source_id": "s1", "condition": NEGATIVE},
        {"source_id": "s2", "condition": POSITIVE},
        {"source_id": "s2", "condition": NEGATIVE},
    ]
    baseline = [2.0, 0.0, 4.0, 1.0]
    changed = [1.5, 0.5, 3.5, 1.5]

    assert mean_source_gap(examples, baseline) == pytest.approx(2.5)
    assert signed_effect(examples, baseline, changed) == pytest.approx(0.5)
