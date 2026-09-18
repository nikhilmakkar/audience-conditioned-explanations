# Audience-Conditioned Explanation Steering

[![Tests](https://github.com/nikhilmakkar/audience-conditioned-explanations/actions/workflows/tests.yml/badge.svg)](https://github.com/nikhilmakkar/audience-conditioned-explanations/actions/workflows/tests.yml)

This repository implements experiments for learning linear activation directions from reader-profile contrasts and testing whether those directions change a language model's preference for technical or accessible explanations.

The code covers the complete intervention loop: build matched prompts, capture residual-stream activations, learn a direction or subspace, add or remove it during inference, and compare the result with random and shuffled-label controls. Saved outputs and the research reports are included, so the experiments can be inspected without downloading model weights.

## What is implemented

- Matched expert-versus-expert prompts that vary whether the reader's knowledge is relevant to the passage.
- Residual-stream activation capture across layers and token positions.
- Difference-of-means directions, linear decoding, activation addition and projection removal.
- Alternate answer-label encodings, dose-response sweeps, random directions and shuffled-label controls.
- SVD subspaces and rank-matched null interventions.
- Prospective transfer to invented technical domains, free-generation evaluation and cross-family replication.
- CSV and tensor outputs for every reported experiment.

## Code tour

| File | Role |
|---|---|
| [`scripts/experiment.py`](scripts/experiment.py) | Prompt construction, model loading, next-token scoring and baseline experiments |
| [`scripts/layer_token_sweep.py`](scripts/layer_token_sweep.py) | Activation capture, forward hooks, direction selection, steering and projection removal |
| [`scripts/fixed_direction_transfer.py`](scripts/fixed_direction_transfer.py) | Main prospective experiment with alternate labels, ablation, random controls and shuffled-label controls |
| [`scripts/multidimensional_subspace.py`](scripts/multidimensional_subspace.py) | SVD subspace construction, rank ablations and rank-matched controls |
| [`scripts/fixed_direction_generation.py`](scripts/fixed_direction_generation.py) | Free-generation intervention and frozen automatic metric |
| [`scripts/cross_family_replication.py`](scripts/cross_family_replication.py) | Discovery-only layer selection and replication on Phi-3.5 Mini |
| [`tests/`](tests) | CPU tests for stimulus construction, metrics, direction geometry and intervention hooks |

The [script index](scripts/README.md) maps the remaining experiment, analysis and figure-building files.

## Experiment flow

1. Construct pairs that differ in the stated reader but keep the passage and candidate summaries fixed.
2. Capture final-prompt-token residual activations on the discovery domains.
3. Compute the relevant-expert minus unrelated-expert mean activation difference.
4. Freeze the direction, layer and evaluation protocol before testing the prospective domains.
5. Add the direction to test sufficiency and remove its projection to test necessity.
6. Repeat with random directions, shuffled audience labels, rewritten profiles, new label tokens, generated text and another model family.

## Installation

The reported experiments used Python 3.10, CUDA 12.4, PyTorch 2.6.0 and Transformers 5.16.1.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Model-running scripts require a CUDA-capable GPU and download weights from Hugging Face. The unit tests run on CPU:

```bash
pip install -r requirements-dev.txt
pytest
```

## Reproducing the main experiment

Run commands from the repository root. The main fixed-direction prospective test is:

```bash
python scripts/fixed_direction_transfer.py \
  --model Qwen/Qwen3.5-4B \
  --discovery data/stimuli_domain_matched_8.json \
  --test data/stimuli_prospective_synthetic.json \
  --output-dir results/reproduction_fixed_direction
```

This writes per-domain steering effects, label-encoding summaries, ablation results, control distributions, the learned direction and the complete run configuration.

## Repository layout

```text
data/       passages, reader profiles and candidate summaries
protocols/  prospective protocols written before follow-up experiments
report/     submitted executive summary, full report and figures
results/    committed CSV outputs, configurations, generations and directions
scripts/    experiment, analysis and report-generation code
tests/      CPU unit tests for the reusable experimental logic
```

## Result in brief

The main result was mixed. In Qwen3.5-4B, a frozen direction learned from eight real ML topics steered forced-choice summary preference across eight invented domains and survived answer-label and profile-wording changes. However, removing the direction reduced only about 8% of the unmodified audience gap. Six of 100 shuffled-label directions matched its steering effect, and the result did not reliably transfer to free generation or Phi-3.5 Mini.

![The learned direction could steer the forced-choice answer, but removing it explained little of the unmodified audience effect.](report/report_assets/exec_final_2_interventions.png)

The useful methodological result is narrower than the original hypothesis: a direction can encode a behavioural contrast and steer an answer without explaining much of the computation used by the unmodified model.

For the research narrative and limitations, see the [executive summary](report/EXECUTIVE_SUMMARY.md) or [full report](report/FULL_REPORT.md).

## Scope and limitations

- The main endpoint is forced-choice summary preference, not generated explanation quality.
- The prospective methods are synthetic and received only light human review.
- The strongest effect is specific to Qwen3.5-4B; the Phi replication failed its controls.
- The committed outputs are sufficient to audit the reported calculations, but rerunning model interventions requires the model weights and a suitable GPU.

## AI assistance

Codex was used heavily for implementation, experiment orchestration, plotting and first-pass prose. I chose the research question, decided which claims required further tests, manually checked the main stimuli and figures, and rejected or revised interpretations that were not supported by the results.

## Author

Nikhil Makkar
