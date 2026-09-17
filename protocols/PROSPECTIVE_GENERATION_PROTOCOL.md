# Prospective fixed-direction generation protocol

Frozen before inspecting any generated outputs. This is an exploratory follow-up to
the fixed-direction forced-choice experiment, not a repair or redefinition of its
predeclared shuffled-label near-miss.

## Question

Does the one fixed Qwen3.5-4B direction learned from the original eight real domains
produce a graded change in free-form explanations of eight unseen, invented methods?

## Frozen setup

- Model: `Qwen/Qwen3.5-4B`.
- Direction: the already saved fixed discovery direction in
  `results/qwen35_4b_fixed_direction_prospective/fixed_direction.pt`.
- Intervention: add the direction at transformer layer 17 at every token position.
- Coefficients: `-2, -1, 0, +1, +2`.
- Decoding: greedy, maximum 128 new tokens.
- Stimuli: all eight sources in `data/stimuli_prospective_synthetic.json`, both profile
  phrasings, and both reader conditions (domain expert and matched other-domain
  expert). No stimulus may be replaced after outputs are viewed.
- Independent analysis unit: source/domain (`n=8`), averaging profile phrasings and
  reader conditions unless a condition-specific analysis is named.

## Outcomes

Primary automated outcome: coverage of the source-specific technical-term bank,
reported as a proportion so banks of different sizes are comparable. Secondary
descriptive outcomes are word count and the two reader-condition strata.

The main contrast is the within-source `alpha=+2` minus `alpha=-2` change. A graded
effect is supported only if:

1. the technical-term coverage contrast is positive for at least 7 of 8 sources;
2. its exact one-sided sign-flip p-value is at most 0.05; and
3. the mean within-source linear slope across all five coefficients is positive with
   an exact one-sided sign-flip p-value at most 0.05.

The direction is interpreted as source-domain specificity rather than generic
verbosity only if the term-coverage result is not explained by a comparable increase
in word count. Reader-condition results are reported separately; no favorable stratum
will replace the pooled primary result.

## Human validation

All outputs are written with opaque codes and randomized into a review sheet. Before
unblinding coefficients, a reviewer scores technicality (1--5), factual correctness
(0/1), and coherence (0/1). The activation result alone cannot establish that more
technical outputs are better explanations. Human scores are confirmatory evidence and
remain pending until actually completed.

## Interpretation boundary

Passing this protocol would show that injecting this fixed direction is sufficient to
shift an observable feature of generated explanations on these stimuli. It would not
show that the direction is necessary, uniquely represents audience modelling, or is a
general mechanism across model families. Failure is retained as failure; coefficients,
term banks, and thresholds will not be changed after generation.
