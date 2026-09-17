# Pilot findings

> **Superseded by [FALSIFICATION_REPORT.md](FALSIFICATION_REPORT.md).** These were the initial pilot
> findings. Later matched-profile and free-generation controls rejected the
> interpretation that the direction tracks exact domain expertise.

## Bottom line

Qwen2.5-3B-Instruct changes its preferred summary when the stated reader changes.
A ResNet-trained expert-minus-novice residual-stream direction also causally
changes that preference on held-out ResNet and Transformer prompts. However, a
non-ML-academic-minus-novice direction produces the same qualitative effect and
has cosine similarity 0.703 with the ML direction. The safest conclusion is
therefore that the model represents and uses a broader technical/academic
sophistication feature; this pilot does not isolate ML expertise.

## Design

- Model: `Qwen/Qwen2.5-3B-Instruct`.
- Discovery source: ResNet; transfer source: Transformer.
- Outcome: `log P(technical label) - log P(accessible label)`.
- Every profile was evaluated with both A/B orders and those two measurements
  were averaged before analysis.
- Directions used only ResNet training profiles. Candidate layers were those
  with maximal ResNet validation decoding accuracy; ties were resolved by the
  expected signed steering effect on validation prompts. Test prompts were not
  used for selection.

## Behavioral result

On held-out profiles, expert-minus-novice technical-margin differences were
6.625 for ResNet and 7.188 for Transformer. The corresponding
non-ML-academic-minus-novice differences were 6.688 and 10.688. Thus the model
clearly conditions on reader descriptions, but domain specificity is already
doubtful before inspecting activations.

There was substantial raw label-position bias: mean `log P(A)-log P(B)` was
4.594 on ResNet test prompts and -0.984 on Transformer test prompts. Perfect A/B
pairing is therefore essential; unpaired results would be misleading.

## Representation and intervention

Validation-only selection chose layer 26. The expert-minus-novice direction had
1.00 validation decoding accuracy and 0.75 held-out test accuracy.

For held-out novice profiles, mean technical margins under the ML direction
were:

| Source | alpha -2 | alpha -1 | alpha 0 | alpha +1 | alpha +2 |
|---|---:|---:|---:|---:|---:|
| ResNet | -8.187 | -7.688 | -5.750 | -2.625 | -0.188 |
| Transformer | -5.688 | -4.812 | -2.813 | -0.375 | 1.563 |

The non-ML academic control also shifted held-out novice prompts:

| Source | alpha -2 | alpha -1 | alpha 0 | alpha +1 | alpha +2 |
|---|---:|---:|---:|---:|---:|
| ResNet | -8.687 | -7.250 | -5.750 | -4.188 | -2.813 |
| Transformer | -5.437 | -4.187 | -2.813 | -1.563 | -0.500 |

The ML direction is stronger at positive alphas, but the control prevents a
claim that the causal feature is specifically ML expertise.

## What this supports

The pilot supports a narrow claim: in this forced-choice task, the model has a
linearly accessible audience-related feature that participates causally in
summary selection and transfers across two technical topics.

It does not show that the same mechanism controls free-form summarization, that
the feature is a single semantic axis, or that constitutional/preference
training created it. With only two held-out profiles per condition, the numbers
are descriptive rather than inferential.

## Best next experiment

Hold general academic sophistication constant and vary only domain relevance:
compare an ML researcher, a researcher in an unrelated technical field, and an
educated non-researcher using tightly matched profile templates. Use an
unfamiliar or synthetic source passage to reduce famous-paper memorization. If
the ML direction then outperforms the matched controls and transfers to free
generation, the domain-expertise interpretation becomes substantially stronger.
