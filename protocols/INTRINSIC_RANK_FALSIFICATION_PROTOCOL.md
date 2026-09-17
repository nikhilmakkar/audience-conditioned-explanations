# Intrinsic-rank falsification protocol

Written before optimizing or evaluating any one-dimensional component mixture.

## Motivation

The leading SVD direction maximizes discovery-contrast variance, not causal
behavioral mediation. A rank-3 subspace outperforming that direction therefore
does not establish that three dimensions are necessary. A different single
direction inside the learned subspace might be more causally effective.

## Frozen data split

- Learn the SVD basis from the existing real-domain training profiles.
- Search for one-dimensional mixtures using only the existing real-domain
  validation profiles.
- Evaluate selected directions once on:
  1. untouched real-domain test profiles;
  2. the eight invented synthetic methods;
  3. the rewritten synthetic profile carriers.
- Do not use any of these three evaluation sets for direction selection.

## Candidate one-dimensional directions

- Previous leading SVD direction.
- Each of the eight individual SVD components.
- 256 fixed-seed random unit mixtures inside the top-three SVD span.
- 256 fixed-seed random unit mixtures inside the full rank-eight SVD span.
- Direction sign is irrelevant for ablation.
- Select the candidate with greatest mean validation gap reduction separately
  for the top-three and full-eight searches.

## Comparators

- Leading SVD direction.
- Full rank-3 SVD subspace.
- Full rank-8 SVD subspace.

## Interpretation

- If a validation-selected one-dimensional mixture approaches or exceeds the
  rank-3 effect on both untouched real profiles and synthetic domains, evidence
  that three dimensions are necessary is falsified.
- If rank 3 remains materially stronger, this supports—but cannot prove—a
  multidimensional causal account. Finite random search cannot exclude every
  possible one-dimensional direction.
- This analysis concerns dimensionality of the intervention effect. It does not
  identify three distinct semantic mechanisms.
