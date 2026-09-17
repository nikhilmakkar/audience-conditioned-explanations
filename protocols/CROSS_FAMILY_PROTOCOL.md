# Prospective cross-family replication protocol

Frozen before downloading/loading Phi-3.5 Mini or evaluating it on any project
stimulus.

## Model and purpose

Replicate the extraction-and-intervention procedure in
`microsoft/Phi-3.5-mini-instruct`, a 3.8B dense decoder-only model from a non-Qwen
family. This tests a procedure, not transfer of Qwen's numerical activation vector,
since residual spaces are not shared across architectures.

## Discovery-only selection

- Discovery data: train profiles from `stimuli_domain_matched_8.json` only.
- Representation: final prompt token of the residual stream.
- Candidate layers: the unique rounded indices at 25%, 37.5%, 50%, 62.5%, and 75%
  of model depth.
- For each candidate, run leave-one-discovery-domain-out extraction and measure the
  balanced A/B counterfactual-sign addition effect on the held-out discovery domain.
- Select the layer with the largest mean effect across the eight held-out domains;
  ties choose the earlier layer.
- Refit one fixed positive-minus-negative direction at that layer using all discovery
  examples.

The synthetic test set is not used for layer selection, sign selection, scaling, or
any other decision.

## Frozen test

- Test data: all eight sources in `stimuli_prospective_synthetic.json`.
- Add the fixed vector at all token positions, using `+v` for matched-other-expert
  prompts and `-v` for domain-expert prompts.
- Encodings: A/B, X/Y, and 1/2.
- Controls on A/B: 100 isotropic norm-matched random directions and 100 balanced
  shuffled-discovery-label directions, normalized to the learned norm.
- Unit of inference: held-out synthetic source/domain (`n=8`).

## Success criteria

1. Each encoding has at least 7/8 positive domains and exact one-sided sign-flip
   p <= 0.05.
2. Learned A/B aggregate exceeds at least 95/100 isotropic and at least 95/100
   shuffled-label controls.
3. At least 6/8 individual domains exceed the 95th percentile of their isotropic
   random controls.

No model, candidate layers, threshold, stimulus, or label encoding is changed after
test results are observed. Ablation and language-model damage are follow-ups if the
addition test is non-null; absence of those follow-ups limits mediation claims.

## Tokenizer implementation deviation

The first run stopped at the tokenizer check before saving any result: Phi encodes the leading space shared by `1` and `2` as a separate token. For a pair with an identical token prefix, that common prefix is appended to both prompts and the final differing token is compared. This equals the two sequence log-probability difference because the common-prefix probability cancels. Labels are not replaced.
