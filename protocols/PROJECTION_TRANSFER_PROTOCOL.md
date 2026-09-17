# Fixed-direction projection-transfer protocol

Frozen before this representation analysis. The synthetic sets were used by earlier
intervention experiments, so this is mechanistic triangulation rather than a fresh
confirmatory holdout.

- Use the already fitted Qwen3.5 layer-17 direction without refitting.
- Capture final-prompt-token residual activations from the original synthetic profiles
  and the profile-carrier paraphrase set, averaging balanced A/B candidate orders.
- Primary statistic: within-source domain-expert minus matched-other-expert projection
  gap, with source (`n=8`) as the independent unit and an exact one-sided sign-flip test.
- Report threshold-free AUC and accuracy using the midpoint of discovery condition
  means; neither threshold is tuned on synthetic examples.
- Compare the aggregate projection gap with 100 isotropic directions and 100 balanced
  shuffled-discovery-label directions.

Robust transfer requires at least 7/8 positive source gaps, exact p <= 0.05, and no
more than 5/100 controls of either kind matching the learned aggregate in both profile
sets.

