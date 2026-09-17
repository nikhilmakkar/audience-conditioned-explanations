# Pre-analysis protocol

## Research question

When choosing between a technically compressed and an accessible summary, does
an instruction-tuned LLM internally represent the reader's relevant expertise,
and does that representation causally affect its choice?

## Why use forced choice?

Free-form summaries confound audience adaptation with length, writing quality,
factuality, and evaluator taste. Forced choice gives an exact next-token metric
while preserving the core decision: which information and presentation is more
useful for this reader?

This loses ecological validity. Therefore, a successful forced-choice result is
a feasibility result, not a complete explanation of the original Claude
interaction.

## Pre-declared comparisons

### Behavioral

Primary:

- Expert versus novice technical-margin difference on ResNet test profiles.

Secondary:

- Expert versus neutral on ResNet.
- Non-ML academic versus novice on ResNet. This distinguishes domain expertise
  from general research literacy.
- Expert versus novice on Transformer. This is a content-transfer check.
- A/B order effect. A large effect is a warning that the task is poorly
  calibrated.

### Representation

For every transformer layer, calculate the difference between mean final-token
activations for ResNet expert-train prompts and novice-train prompts. Balance A/B
order within both classes.

For each layer, choose a classification threshold using only training profiles.
Keep layers tied for maximal validation-profile accuracy, then select among them
using the expected signed intervention effect on validation prompts: add the
direction to novice prompts and subtract it from expert prompts. Report
test-profile accuracy separately and never use it for layer selection.

Linear separability is not evidence of causal use.

### Intervention

At the selected layer and final prompt position, add multiples of the raw
expert-minus-novice mean-difference vector. Evaluate the change in technical
margin for held-out profiles.

Use alphas `[-2, -1, 0, 1, 2]`. Report:

- technical-margin change;
- monotonicity across alpha;
- A/B-balanced results;
- ResNet and Transformer results separately.

A useful negative/control comparison is the non-ML-academic-minus-novice
direction. If it performs as well as the ML-expert direction, the relevant
feature may be general academic sophistication rather than ML expertise.

## Success criteria

Do not use a p-value threshold for this tiny pilot. Look for all of:

1. A/B-balanced expert prompts have a higher mean technical margin than novice
   prompts on held-out ResNet profiles.
2. The direction classifies held-out profiles above chance.
3. Positive steering increases technical margin and negative steering decreases
   it without causing invalid behavior.
4. The direction outperforms, or behaves meaningfully differently from, the
   non-ML academic control.

Evidence is stronger if it transfers to Transformer, but transfer is not required
for a worthwhile ResNet case study.

## Main validity threats

1. **Famous-paper memorization:** the model may use learned canned summaries
   rather than the supplied passages. A later control can rename components or
   use an unfamiliar method while preserving the information structure.
2. **Lexical profile artifacts:** a direction may encode words such as
   "machine learning" rather than an inferred user model. Held-out paraphrases
   and implicit profiles partially address this.
3. **Summary-pair quality:** one candidate may simply be better. A/B balancing
   does not fix this. Neutral-profile preference and manual verification matter.
4. **Forced-choice limitation:** selecting A/B may not share the mechanism used
   in free generation.
5. **Direction interpretation:** difference-of-means directions contain every
   systematic difference between the two prompt sets.
6. **Layer selection overfitting:** validation selects the layer; test profiles
   must remain untouched until the final evaluation.

## Manual verification checklist

Before looking at model results, record yes/no and any edits for each source:

- [ ] The source passage is factually accurate.
- [ ] The technical summary is factually accurate.
- [ ] The accessible summary is factually accurate.
- [ ] Neither summary omits a fact that makes it misleading.
- [ ] The summaries have comparable length.
- [ ] The primary difference is presentation/technical compression, not quality.

## Suggested 20-hour stopping schedule

- Hours 0-1: verify/edit stimuli and run design tests.
- Hours 1-3: install/load model and obtain behavioral results.
- At hour 3: stop or pivot if expert/novice context has essentially no effect.
- Hours 3-8: debug controls and establish a reliable behavioral result.
- Hours 8-13: activation extraction and held-out layer selection.
- Hours 13-16: steering sweep and academic-control direction.
- Hours 16-20: plots, interpretation, limitations, and main write-up.

Do not add more papers, models, or complex interpretability techniques until the
primary causal experiment works.
