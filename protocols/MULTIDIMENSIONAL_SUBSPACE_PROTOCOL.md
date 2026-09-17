# Multidimensional relevance-subspace protocol

Frozen before evaluating any multidimensional subspace intervention.

## Question

Is the Qwen3.5-4B audience-relevance result unusually one-dimensional, or do
additional contrastive components explain natural behavior beyond the single vector?

This is exploratory follow-up analysis. The synthetic domains have already been used
by earlier experiments, so it cannot become a new confirmatory result.

## Subspace construction

- Model: `Qwen/Qwen3.5-4B`.
- Frozen representation site: residual stream, layer 17, final prompt token.
- Discovery data: train profiles from the eight real domains in
  `data/stimuli_domain_matched_8.json`.
- For each discovery domain, compute one domain-expert minus matched-other-expert
  activation contrast, averaging profile phrasings and balanced A/B order.
- Stack the eight oriented contrast vectors and take an uncentered SVD.
- The learned rank-k subspace is the first k right-singular vectors, for every nested
  rank k = 1,...,8. Uncentered SVD deliberately retains the shared mean component.
- Also report the cosine between the rank-1 SVD direction and the previous fixed
  mean-difference direction, plus the singular-value energy curve.

## Causal evaluation

- Primary test: the eight unfamiliar sources in
  `data/stimuli_prospective_synthetic.json` with balanced A/B order.
- Intervention: at layer 17 and every token position, orthogonally remove the
  activation projection onto the learned rank-k basis.
- Outcome: reduction in the natural domain-expert minus other-expert technical-margin
  gap. Source/domain (`n=8`) is the independent unit.
- Report mean reduction, number of positive domains, and exact one-sided sign-flip p
  for each rank.
- Repeat learned-rank ablations on the new profile-carrier set as a robustness result.

## Controls

At every rank, compare with:

1. 100 isotropic random orthonormal subspaces, using one seeded eight-vector QR basis
   per seed and nested ranks from that basis.
2. 100 shuffled-label subspaces. Within each discovery domain, randomly divide its
   eight condition/profile activation means into two balanced groups, form a pseudo-
   contrast, and take the same uncentered SVD. Shuffling occurs within domain so source
   identity cannot itself define the pseudo-label.

For each learned rank, report how many of 100 controls have aggregate gap reduction at
least as large. A learned rank is control-specific only if at most 5/100 controls of
each kind match it.

## Interpretation

- Evidence for additional dimensions requires more than a numerically larger
  ablation. Rank k>1 must increase the learned mean reduction beyond rank 1, retain a
  positive domain-level exact p <= 0.05, and beat both same-rank control distributions.
- General language-model damage must be measured for learned ranks before calling an
  increase mediation. Larger subspaces mechanically remove more activation, so a
  behavior gain accompanied by comparable generic damage is not relevance-specific.
- No rank will replace the one-vector prospective result, and ranks or thresholds will
  not be selected after inspecting the test curve.
