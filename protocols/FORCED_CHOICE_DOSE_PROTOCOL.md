# Fixed-direction forced-choice dose-response protocol

Frozen after the prospective generation primary analysis and before running this
follow-up. This diagnoses the meaning of the earlier counterfactual-sign intervention;
it does not alter that experiment's outcome.

## Setup

- Qwen3.5-4B, the already frozen fixed direction, layer 17, all token positions.
- The eight unseen invented sources, both reader conditions and both profile phrasings.
- Coefficients `-2, -1, 0, +1, +2`, applied with the same sign regardless of reader
  condition.
- Balanced candidate order and all three pre-existing label encodings: A/B, X/Y, 1/2.
- Technical-minus-accessible log-probability margin is the outcome.
- Source/domain (`n=8`) is the independent unit.

## Tests

For each encoding and reader condition, average label orders and profile phrasings,
fit the margin's linear slope over the five coefficients within each source, and use
an exact one-sided sign-flip test across sources.

A context-independent technicality-axis interpretation requires, for every encoding:

1. positive mean slope and at least 7/8 positive source slopes in domain-expert prompts;
2. the same in matched-other-expert prompts; and
3. exact one-sided p <= 0.05 in both conditions.

Failure means the earlier condition-specific counterfactual intervention cannot be
summarized as one context-independent knob that simply increases technical content.

