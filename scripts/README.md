# Script index

The repository keeps the complete experimental record, so this directory contains more than one polished entry point. The main scripts are:

## Core experiment

- `experiment.py` — shared model loading, prompt construction, activation capture, and baseline experiment code
- `fixed_direction_transfer.py` — frozen-direction transfer to the invented prospective domains
- `fixed_direction_ablation.py` — projection-removal test
- `fixed_direction_projection.py` — representation/projection analysis
- `forced_choice_dose.py` — steering-strength sweep

## Falsification and controls

- `random_steering_baseline.py` — norm-matched random directions
- `reencoding_control.py` — alternate answer-label encodings
- `cross_family_replication.py` — Phi-3.5 Mini replication
- `fixed_direction_generation.py` — free-generation experiment
- `direction_learning_curve.py` — discovery-data scaling

## Multidimensional follow-up

- `multidimensional_subspace.py` — SVD subspace construction and intervention
- `intrinsic_rank_falsification.py` — rank and selected-one-dimensional comparison
- `evaluate_selected_1d_minimal_v2.py` — held-out evaluation of the selected direction
- `component_interpretation.py` — component-level effects

Files beginning with `analyze_`, `build_`, `make_`, or `plot_` process saved outputs, construct stimuli, or regenerate report assets. The remaining scripts preserve intermediate experiments and their saved outputs.

Run commands from the repository root so relative `data/` and `results/` paths resolve consistently.
