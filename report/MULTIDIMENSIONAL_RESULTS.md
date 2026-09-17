# Multidimensional audience-relevance results

## Question

The single direction was strongly decodable and sufficient for steering, but its
selected-layer ablation removed only part of the natural audience-conditioned
relevance gap. One possible explanation was that the relevant representation is
low-rank but not rank-1.

The protocol was frozen in `MULTIDIMENSIONAL_SUBSPACE_PROTOCOL.md` before the
outcomes below were inspected.

## Method

- Model: Qwen3.5-4B.
- Location: residual stream, layer 17, final prompt token.
- Discovery data: eight real technical domains.
- For each domain, calculate a domain-expert minus matched-other-expert contrast.
- Stack the eight contrast vectors and use uncentered SVD. Centering was avoided
  because it would remove their shared mean contrast.
- Ablate nested subspaces of ranks 1 through 8 at all token positions in layer 17.
- Primary evaluation: eight unfamiliar invented methods, balanced answer order.
- Transfer evaluation: the same methods with new reader-profile wording.
- Controls: 100 same-rank random orthonormal subspaces and 100 subspaces learned
  from within-domain shuffled labels.
- Damage check: WikiText next-token NLL, KL divergence, and top-1 agreement.

## Main result

| Rank | Original gap reduction | New carriers | Positive domains | Random >= learned | Shuffled >= learned |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.039 | 0.086 | 5/8 | 0/100 | 20/100 |
| 2 | 0.086 | 0.141 | 8/8 | 0/100 | 21/100 |
| **3** | **0.180** | **0.238** | **8/8** | **0/100** | **2/100** |
| 4 | 0.176 | 0.270 | 8/8 | 0/100 | 9/100 |
| 5 | 0.156 | 0.246 | 8/8 | 0/100 | 28/100 |
| 6 | 0.184 | 0.273 | 8/8 | 0/100 | 23/100 |
| 7 | 0.266 | 0.359 | 8/8 | 0/100 | 5/100 |
| 8 | 0.262 | 0.387 | 8/8 | 0/100 | 7/100 |

For rank 3, the exact one-sided domain sign-flip p-value is 0.00390625. Its
effect on the original test is 4.6 times the rank-1 effect. Rank 3 passes both
predeclared control standards and transfers to new carrier wording. Rank 7 has
a larger behavioral effect but only just meets the shuffled-label threshold;
rank 8 fails it. Therefore the result is not simply “more dimensions are always
better.”

The first SVD component is close to the previous fixed mean direction
(absolute cosine 0.965), confirming that this experiment extends rather than
replaces the original direction analysis. The first three components explain
68.8% of the energy across the eight domain contrasts.

## Ordinary-text damage

At rank 3, over 1,016 WikiText next-token predictions:

- baseline NLL: 3.0583;
- NLL increase: 0.00656, about 0.21% of baseline;
- mean next-token KL: 0.00248;
- top-1 agreement: 96.75%.

The mean random rank-3 subspace has KL 0.00110 and top-1 agreement 97.90%.
Thus the learned subspace perturbs ordinary text more than typical random
rank-matched spaces, although the absolute change is modest. The defensible
description is “large targeted behavioral effect with modest, detectable
general-language impact,” not “damage-free mediation.”

## Cross-layer geometry

Directions learned in layers 0–14 have little similarity to the selected
layer-17 direction. Similarity begins to rise at layers 15–16, changes sharply
at layer 17, is 0.70 at layer 18, and then evolves smoothly downstream. Adjacent
rank-1 cosine is mostly 0.84–0.95 across layers 20–30. Rank-3 principal-angle
similarity shows the same broad pattern.

This is consistent with audience-relevance geometry consolidating near the
middle of the network and subsequently being carried forward. It is descriptive
evidence, not localization of an attention head, MLP, or complete circuit.

## Strongest defensible claim

In Qwen3.5-4B, the forced-choice effect of reader-domain relevance is mediated
more completely by a small learned residual-stream subspace than by its leading
direction alone. A rank-3 subspace generalizes across eight unfamiliar methods
and new profile wording, and outperforms rank-matched random and shuffled-label
controls while causing modest ordinary-text change.

Do **not** claim that expertise itself is three-dimensional. The experiment
measures audience-conditioned relevance for technical-summary selection, and
the eight discovery contrasts limit the learned subspace to at most rank eight.

## Next tests

1. Run rank-3 ablation at neighboring layers to connect geometric emergence to
   causal leverage.
2. Compare learned and control interventions at matched KL, not only matched
   dimensionality.
3. Bootstrap discovery profiles to estimate subspace stability.
4. Complete blinded scoring of free-form generations.
5. Attempt cross-model geometry only with a frozen, held-out representation
   alignment; raw cross-model cosine is not interpretable.

## Artifacts

- `MULTIDIMENSIONAL_SUBSPACE_PROTOCOL.md`
- `multidimensional_subspace.py`
- `multidimensional_coherence.py`
- `layer_subspace_geometry.py`
- `results/qwen35_4b_multidimensional_subspace/`
- `MULTIDIMENSIONAL_SUBSPACE_ADDENDUM.pdf`
- `AUDIENCE_STEERING_FULL_REPORT.pdf`
