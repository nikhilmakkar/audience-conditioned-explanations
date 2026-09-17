# Prospective fixed-direction transfer protocol

## Motivation

The eight-domain leave-one-domain-out result fits a separate direction and
selects a layer for each held-out domain. It establishes that the extraction
procedure transfers, but not that one fixed representation transfers unchanged.
The existing sources are also famous methods that may be recognized from model
pre-training.

This follow-up is frozen before any model evaluation on the new sources.

## Claim under test

A single residual-stream direction extracted from exact-domain-versus-unrelated-
domain expert contrasts transfers unchanged to unfamiliar technical methods and
causally changes which summary the model considers useful for each reader.

## Frozen discovery choices

- Discovery model: `Qwen/Qwen3.5-4B` with thinking disabled.
- Discovery data: all training profiles in `data/stimuli_domain_matched_8.json`.
- Positive class: domain expert.
- Negative class: equally experienced expert in an unrelated field.
- Direction: raw difference of class-mean residual activations.
- Layer: transformer residual block 17.
- Extraction token: final prompt token.
- No layer, token, direction scale, or source is selected using the prospective
  test results.

Layer 17 and the final token are frozen because all eight prior Qwen3.5 folds
selected the final token and five selected layer 17. This is an explicitly
follow-up choice, not an independently motivated architectural prior.

## Prospective test set

Eight invented methods will be written across technical fields not used to fit
the direction. Each passage is self-contained, and the method name is novel, so
recognition of a memorized paper cannot determine the answer.

Fields are frozen as:

1. state-space estimation and control;
2. finite-volume numerical fluid dynamics;
3. causal inference with instrumental variables;
4. database indexing and approximate membership;
5. computational biology and profile sequence models;
6. synthetic-aperture radar remote sensing;
7. threshold cryptography;
8. time-frequency audio signal processing.

Each source has two factually intended summaries: a compressed technical
summary and an accessible mechanism-preserving summary. Reader profiles use two
held-out carrier sentences and differ only in whether the expert field matches
the passage. Both A/B summary orders are evaluated.

Human verification remains required before submission; any invalid pair
invalidates that source and must be reported, not silently replaced after seeing
results.

## Frozen interventions and outcomes

- Baseline outcome: technical-summary log-probability margin.
- Addition: add the same raw discovery direction at every sequence position of
  layer 17. Use opposite signs for matched and unrelated experts so the signed
  effect asks whether steering moves both conditions in the intended direction.
- Selected-layer ablation: remove the unit-direction projection at layer 17.
- All-layer ablation: remove it at every residual block.
- Repeat the frozen addition under A/B, X/Y, and 1/2 answer encodings.
- Test 100 isotropic random directions matched to the learned direction's norm.
- Test shuffled discovery labels through 100 balanced permutations, fitting a
  direction at the same frozen layer and token for each permutation.
- Use one domain mean as the independent statistical unit.
- Primary inference: exact one-sided sign-flip test across eight domain effects.

## Success and failure criteria

The fixed-direction claim survives only if all are true:

1. Addition is positive in at least seven of eight prospective domains under
   each of A/B, X/Y, and 1/2.
2. The one-sided exact domain sign-flip probability is at most 0.05 under every
   encoding.
3. The learned mean exceeds at least 95 of 100 aggregate norm-matched random
   controls and 95 of 100 aggregate shuffled-label controls.
4. At least six of eight individual domains exceed the 95th percentile of their
   norm-matched random controls.

Ablation is evaluated separately. Evidence for mediation additionally requires:

5. Positive mean gap reduction with exact p at most 0.05.
6. Ordinary-language damage small enough that the behavioral reduction cannot
   plausibly be explained by broad model degradation. Damage metrics are
   next-token KL, NLL/perplexity change, and top-1 agreement on the same fixed
   WikiText sample used previously.

Failure of addition criteria rejects a fixed transferable control direction.
Successful addition with failed ablation supports sufficiency but not normal
mediation. Successful forced choice without blinded generated-summary transfer
does not explain the original free-form summarization phenomenon.

## Analysis boundary

No source deletion, layer change, token change, rescaling, or altered success
threshold is permitted after test logits are inspected. Exploratory follow-ups
must be labeled exploratory and cannot rescue this test.
