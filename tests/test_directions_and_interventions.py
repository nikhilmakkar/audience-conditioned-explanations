from __future__ import annotations

import torch
from torch import nn

from fixed_direction_transfer import (
    domain_effects,
    exact_sign_flip_p,
    permuted_direction,
    true_direction,
)
from layer_token_sweep import ablate_all_positions, add_all_positions
from multidimensional_subspace import random_basis, svd_basis


def test_mean_direction_uses_the_declared_audience_orientation() -> None:
    grouped = {
        ("s1", "domain_expert", 0): torch.tensor([3.0, 1.0]),
        ("s1", "matched_other_expert", 0): torch.tensor([1.0, 1.0]),
        ("s2", "domain_expert", 0): torch.tensor([5.0, 2.0]),
        ("s2", "matched_other_expert", 0): torch.tensor([1.0, 2.0]),
    }

    assert torch.equal(true_direction(grouped), torch.tensor([3.0, 0.0]))
    assert torch.equal(permuted_direction(grouped, 7),
                       permuted_direction(grouped, 7))


def test_domain_effects_orient_both_conditions_toward_the_prediction() -> None:
    examples = [
        {"source_id": "s", "condition": "domain_expert"},
        {"source_id": "s", "condition": "matched_other_expert"},
    ]
    effects = domain_effects(examples, [0.0, 0.0], [-0.5, 0.5])
    assert effects == {"s": 0.5}


def test_exact_sign_flip_probability_for_eight_positive_domains() -> None:
    assert exact_sign_flip_p([1.0] * 8) == 1 / 256


def test_addition_hook_changes_every_position_and_is_removed_afterward() -> None:
    layer = nn.Identity()
    hidden = torch.zeros(2, 3, 2)
    vector = torch.tensor([1.0, -2.0])
    coefficients = torch.tensor([1.0, -0.5])

    with add_all_positions(layer, vector, coefficients):
        changed = layer(hidden)

    expected = hidden + coefficients[:, None, None] * vector[None, None, :]
    assert torch.equal(changed, expected)
    assert torch.equal(layer(hidden), hidden)


def test_ablation_removes_projection_and_is_removed_afterward() -> None:
    layer = nn.Identity()
    hidden = torch.tensor([[[3.0, 4.0], [-2.0, 5.0]]])
    unit = torch.tensor([1.0, 0.0])

    with ablate_all_positions([layer], unit):
        changed = layer(hidden)

    assert torch.equal(changed, torch.tensor([[[0.0, 4.0], [0.0, 5.0]]]))
    assert torch.equal(layer(hidden), hidden)


def test_svd_and_random_bases_are_orthonormal() -> None:
    contrasts = torch.tensor([
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0],
    ])
    learned = svd_basis(contrasts)
    random = random_basis(width=4, seed=11, rank=3)
    identity = torch.eye(3)

    assert torch.allclose(learned @ learned.T, identity, atol=1e-6)
    assert torch.allclose(random @ random.T, identity, atol=1e-6)
