# Frozen discovery-domain learning curve

Frozen before running this analysis. This is descriptive robustness analysis on the
already used Qwen synthetic holdout, not a new confirmatory test.

- Model/vector location: Qwen3.5-4B, layer 17, final prompt-token extraction.
- Discovery pool: the eight real-domain train partitions.
- Training sizes: every domain subset of size 1, 2, 4, 6, and 8 (135 subsets total).
- Evaluation: balanced A/B counterfactual-sign addition on all eight synthetic sources,
  at all token positions, coefficient magnitude 1.
- Outcomes per subset: mean addition effect, number of positive synthetic sources, and
  cosine similarity to the direction fitted on all eight discovery domains.

Results are summarized across all subsets at each size. Because subsets overlap and
the synthetic set has already been evaluated, no subset-level p-values are interpreted
as fresh confirmation and no subset is selected as a replacement direction.

