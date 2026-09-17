# Falsification report

## Question

Does an instruction-tuned model adapt technical summaries to what a reader
actually knows, or does it respond to cruder cues such as being called an ML
expert, asking for technical detail, or appearing generally sophisticated?

## Result in one sentence

Two Qwen models respond strongly to coarse expertise cues, while Qwen3.5 also
supports a smaller exact-relevance steering feature under leave-one-domain-out
extraction that survives intervention, ablation, answer re-encoding, and random
controls. The result is specific to forced choice and does not establish one
fixed cross-domain vector, a complete user-expertise mechanism, or transfer to
generated gists.

## Falsification sequence

### 1. Initial positive result

On two summary pairs, explicit ML-expert profiles had larger held-out technical
margins than novice profiles: +6.625 on ResNet and +7.188 on Transformer. A
non-ML academic condition produced similarly large differences, immediately
challenging an ML-specific interpretation.

### 2. More content and a 2x2 audience design

Four summary pairs (ResNet, Transformer, U-Net, and kriging) crossed ML versus
non-ML experience with researcher versus practitioner status. Across 36 paired
source/profile observations, the ML cue increased technical margin by 2.962
(SE 0.480) in Qwen2.5-3B and 1.348 (SE 0.183) in Qwen3.5-4B. In Qwen2.5,
researcher status increased it by only 1.115 (SE 0.265). This rules out the
narrow explanation that the model merely reacts to academic status, but it does
not establish sensitivity to relevant knowledge. Raw logit scales should not
be compared across models.

### 3. Exact-domain matched control

For each passage, two profiles used the same carrier sentence and described
equally experienced researchers. Only the field of expertise differed. Across
32 pairs, exact-topic expertise changed technical margin by -0.148 (SE 0.176)
in Qwen2.5 and +0.195 (SE 0.052) in Qwen3.5. The latter is about one-seventh of
its coarse-cue effect and was source-dependent: near zero for ResNet and U-Net,
positive for Transformer and kriging. Thus a newer model has some
relevance-sensitive behavior, but the original strong domain-knowledge
interpretation failed its strongest test.

### 4. Free-form ecological-validity check

The same 32 pairs generated 2-3 sentence gists with a nonbinding 180-token cap.
Exact-topic experts retained only 0.125 additional predeclared technical terms
(SE 0.200) in Qwen2.5 and 0.031 (SE 0.145) in Qwen3.5. Mean length differences
were +3.31 and +1.63 words respectively, with standard errors 1.85 and 1.48.
The result is not generally rescued by moving from forced choice to generation.

Raw inspection also found factual problems: several ResNet generations invoked
vanishing gradients even though the supplied passage did not. Summary
adaptation and summary correctness must therefore be evaluated separately.

An initial Qwen3.5 run produced visible reasoning and hit the token cap before
answering. Inspection of the official template showed that thinking needed to
be explicitly disabled. Those scores were discarded and every Qwen3.5 result
above comes from corrected direct-answer prompts.

### 5. Random steering baseline

At each model's validation-selected layer, the learned broad
expert-minus-novice direction produced technical-margin slopes of 2.000 on
ResNet and 1.813 on Transformer in Qwen2.5, and 0.422 and 0.344 in Qwen3.5.
Twenty random directions with exactly the same norm were tested independently
in each model. Their maxima were 0.750 and 0.609 in Qwen2.5, and 0.094 and 0.102
in Qwen3.5. None matched its model's learned direction. Thus the intervention
effect replicates qualitatively and is not explained by arbitrary perturbations
of the same size, though its semantics are broad sophistication rather than
exact domain knowledge.

### 6. Direction ablation and random ablation controls

At the selected layer and final prompt token, the learned-direction projection
was set to the mean of the balanced ResNet expert/novice training projections.
In Qwen2.5 layer 26, the held-out expert-minus-novice technical-margin gap fell
from 6.625 to 5.188 on ResNet and from 7.188 to 5.938 on Transformer: reductions
of 1.438 and 1.250. In Qwen3.5 layer 18, it fell from 3.313 to 2.937 and from
3.031 to 2.625: reductions of 0.375 and 0.406. In both models and sources, none
of 20 random-direction ablations removed as much of the gap.

This is evidence of partial necessity at one layer and token position. About
78% and 83% of the Qwen2.5 gaps, and 89% and 87% of the Qwen3.5 gaps, remained
after ablation. It is therefore not evidence that a single direction fully
mediates audience conditioning.

### 7. Pre-specified layer-by-token causal sweep

Before examining new test results, the candidate set and selection rule were
fixed: extract directions from the final five input positions at every residual
layer, then maximize the weaker of validation addition and validation ablation
effects. The test sets were not used for selection. The sweep selected Qwen2.5
layer 26, position -1, and Qwen3.5 layer 18, position -5. Qwen2.5's strongest
region was concentrated at position -1 over layers 24-27; Qwen3.5 showed a
broader region over layers 12-20 at positions -5 and -1. Thus neither result is
an isolated single-layer spike.

With addition over every position at the selected layer, held-out signed
technical-margin effects were 2.688 and 1.938 for Qwen2.5 and 1.469 and 1.453
for Qwen3.5 on ResNet and Transformer respectively.

An exact decoder permutation test enumerated all 70 balanced assignments of the
eight training profiles and repeated validation-based layer/token selection.
The true-label pipeline's held-out accuracies were not unusual under this null:
the one-sided exact probabilities were 0.300 and 0.171 for Qwen2.5 and 0.243 and
0.100 for Qwen3.5. The small dataset therefore does not establish robust linear
decodability, despite high raw validation accuracy.

### 8. Strong ablation, answer re-encoding, and model-damage controls

Ablating the selected direction at its selected layer over all positions
removed 37% and 29% of the Qwen2.5 audience gaps and 22% and 20% of the Qwen3.5
gaps. Following the refusal-direction protocol more closely, ablating it at
every layer and every position removed 92% and 73% of the Qwen2.5 gaps and 64%
and 65% of the Qwen3.5 gaps.

The all-layer intervention also changed ordinary language modeling. On 4,080
WikiText-2 tokens, Qwen2.5 perplexity rose from 14.61 to 16.56, mean next-token
KL was 0.113, and top-1 agreement was 83.8%. Its large behavioral ablation is
therefore partly confounded by general disruption. Qwen3.5 perplexity rose only
from 16.54 to 16.75, with KL 0.0149 and 93.8% agreement, making its 64-65%
reduction substantially more surgical.

The causal result was not tied to A/B answer tokens. In both models and both
passages, addition and all-layer ablation retained the expected sign after
re-encoding the identical choices as X/Y and 1/2.

### 9. Component localization and generated-gist transfer

Validation-selected attention and MLP output ablations produced small and
inconsistent held-out reductions. No single component replicated strongly
across both passages and models. The residual effect therefore has not been
localized to an attention block, MLP, head, or circuit.

Greedy generation under negative, zero, and positive steering remained coherent
but did not show a consistent technicality dose response. Both models increased
technical-term use for novice ResNet summaries, but Qwen2.5 reversed on novice
Transformer summaries and Qwen3.5 was null there; expert-summary effects were
also mixed. These samples are small and require blinded human scoring, but they
currently falsify general transfer from the forced-choice intervention to
explanation depth in generated gists.

### 10. Eight-domain exact-relevance falsification

The four-domain pilot left only four independent domain units, so its apparent
Qwen3.5 addition consistency could not attain a one-sided domain-level exact
probability below 0.0625. After observing that limitation, the stimulus set was
frozen at eight domains by adding BERT, GCN, DDPM, and XGBoost. Their passages
and intended summary pairs were written from the canonical primary papers, and
they retained the same carrier sentences and predeclared 4/2/2
train/validation/test split. This is a prospectively frozen extension of the
pilot, not a preregistered study.

For each fold, the exact-relevance direction was estimated from the seven other
domains' training profiles. Every residual layer and the final five prompt
positions were evaluated on only those domains' validation profiles, selecting
the candidate that maximized the weaker of addition and single-layer ablation.
The chosen direction was then evaluated once on the unseen domain. All eight
folds selected the final prompt token and layers 17, 18, or 21.

Addition increased the relevance-conditioned technical margin in all eight
held-out domains. The mean was +0.186 (SE 0.022 across domains), with a one-sided
exact sign-flip probability of 0.0039. The smallest and largest leave-one-domain-
out means were +0.172 and +0.196, so no single domain carried the result. None
of 100 independently sampled, norm-matched random directions per fold matched
its learned addition effect.

The effect was not tied to the original answer labels. Repeating the frozen
interventions produced positive effects in all eight domains under A/B, X/Y,
and 1/2 encodings, with means +0.186, +0.203, and +0.230 respectively; each
domain-level exact sign-flip probability was 0.0039.

Ablation was weaker but systematic. Selected-layer ablation reduced the natural
exact-relevance gap by +0.078 on average (6/8 positive; exact p=0.0195), while
all-layer ablation reduced it by +0.129 (7/8 positive; exact p=0.0078). BERT's
all-layer effect was null, and kriging and Transformer had negative effects.
The data are consistent with partial, distributed mediation rather than full
mediation by one direction at one layer. However, these fold-specific
exact-relevance ablations do not yet have their own ordinary-language
model-damage control, so even partial mediation remains provisional.

The identical eight-domain Qwen2.5 run did not replicate this robust result.
Its A/B additions were positive in all eight domains (mean +0.195, exact
p=0.0039) and the aggregate exceeded all 100 aggregate random controls, but
individual-domain specificity was poor. More decisively, the frozen effect
collapsed under answer re-encoding: X/Y produced +0.070 (5/8 positive,
p=0.2266) and 1/2 produced +0.016 (3/8 positive, p=0.4453). Selected-layer
ablation was inconclusive (4/8 positive, p=0.0625), while all-layer ablation
reversed on average (-0.242, p=0.75). Qwen2.5's A/B result is therefore an
unstable forced-choice control, not a replicated exact-relevance direction.

This experiment changes the narrow conclusion for Qwen3.5: exact relevance is
not merely a source-dependent behavioral artifact. In each fold, a direction
learned without the test domain is sufficient to control the forced-choice
relevance decision, and removing that direction reduces part of the natural gap
on average. It does not change the failures of free-form generation transfer,
Qwen2.5 replication, permutation-tested broad-expertise decoding, or component
localization.

### 11. Fixed-direction prospective transfer

A stricter protocol was frozen before evaluation. One Qwen3.5 direction was fitted once at layer 17 from all real-domain training profiles and then applied unchanged to eight unfamiliar, invented technical methods. The methods span state estimation, numerical PDEs, causal inference, databases, computational biology, InSAR, threshold cryptography, and audio processing, limiting direct paper memorization as an explanation.

Addition passed the domain-level and re-encoding tests: A/B, X/Y, and 1/2 means were +0.156, +0.178, and +0.184, each positive in 8/8 domains with exact p=0.0039. No isotropic control matched the learned aggregate or any individual-domain effect. Selected-layer ablation reduced the natural relevance gap by +0.047 (7/8, p=0.0078); all-layer ablation reduced it by +0.063 (7/8, p=0.0430).

The prospectively defined overall test nevertheless failed its shuffled-label specificity criterion by one control: 6/100 shuffled-discovery-label directions matched or exceeded the learned aggregate, whereas the frozen cutoff allowed at most 5/100. This is a near-miss, not a positive confirmatory result. It supplies strong suggestive evidence for a fixed transferable control direction, but does not establish that the correct discovery labels uniquely identify it.

The actual layer-17 intervention was surgically small on 4,080 WikiText tokens: NLL changed by -0.0011, mean next-token KL was 0.00073, and top-1 agreement was 98.4%. Its KL was slightly higher than all ten norm-matched selected-layer random directions, but the absolute disturbance remained small. Applying the direction at every layer caused more damage and is not the primary intervention.

### 12. Dose response, generation, and profile-carrier robustness

A uniform-sign five-coefficient test showed that the fixed direction is a monotonic forced-choice control rather than an artifact of pooling opposite condition-specific interventions. Across A/B, X/Y, and 1/2, the technical-minus-accessible margin slope was positive in 8/8 domains for both domain-expert and matched-other-expert prompts; all six exact probabilities were 0.0039.

The same result survived replacing every reader-profile carrier with two new, balanced sentence frames that never appeared in discovery. Again all six encoding-by-condition cells were positive in 8/8 domains with p=0.0039. Thus the Qwen forced-choice effect is not tied to the original profile sentence template.

Free generation did not pass its frozen test. Across 160 greedily decoded gists at coefficients -2, -1, 0, +1, and +2, source-specific technical-term coverage changed by only +0.011 from -2 to +2 (3/8 positive domains, p=0.25), and the linear slope was also non-significant (3/8, p=0.1875). Word count decreased by 3.1 words on average, so verbosity does not explain a hidden positive result. The automated term bank is an incomplete measure of paraphrased technicality; the frozen failure stands, while blinded human technicality and correctness scores remain necessary to evaluate construct validity.

### 13. Non-Qwen cross-family replication

A prospective replication used Microsoft's 3.8B Phi-3.5 Mini. Five candidate layers were compared only by leave-one-real-domain-out discovery performance; layer 12 was selected before the synthetic set was scored. A tokenizer-specific shared-prefix sequence scorer was required for 1/2 after the first run stopped at a single-token assertion and before any result was saved.

Phi showed a real but non-specific addition signal. A/B was +0.170 (7/8, p=0.0078), X/Y was +0.094 (6/8, p=0.0234), and 1/2 was +0.170 (8/8, p=0.0039). It failed the frozen criteria because X/Y missed 7/8, six isotropic controls matched the A/B aggregate, 18 shuffled-label controls matched it, and only 2/8 individual domains exceeded their isotropic 95th percentiles. Selected-layer ablation was inconclusive (+0.297, 5/8, p=0.0703), as was all-layer ablation (+0.352, 5/8, p=0.2031). Its selected-layer language damage was not exceptional relative to norm-matched random directions. This does not replicate a direction-specific, normally used feature across model families.

### 14. Discovery-data sensitivity

An exhaustive Qwen learning curve fitted directions from every subset of 1, 2, 4, 6, and 8 real discovery domains (135 directions). Six of eight one-domain directions and 26/28 two-domain directions met the 7/8 synthetic-domain consistency rule. Every one of the 70 four-domain subsets, all 28 six-domain subsets, and the full set produced positive effects in all 8/8 test domains. Mean addition stayed near +0.15 across sizes, while mean cosine similarity to the full direction rose from 0.579 with one domain to 0.735 with two, 0.884 with four, and 0.957 with six. Domain diversity stabilizes the vector; the full-data effect is not carried by one favorable training domain.

### 15. Fixed-direction projection transfer

The frozen layer-17 direction also separates natural activations for unseen reader fields, independently of intervention. The within-source domain-expert minus other-expert projection gap was positive in 8/8 original synthetic domains (mean +0.285, p=0.0039, AUC 0.809) and 8/8 new-carrier domains (mean +0.329, p=0.0039, AUC 0.836). In both sets, 0/100 isotropic and 0/100 shuffled-label directions matched the aggregate gap.

A threshold fixed at the discovery-condition midpoint transferred less well: accuracy was 62.5% on the original synthetic profiles and 71.9% under new carriers. Thus relative ranking and within-domain separation transfer strongly, while the absolute projection location shifts across stimulus distributions.

## Defensible conclusion

In these models and prompts, audience-conditioned technicality is strongly driven by coarse cues correlated with sophistication and need for introductory teaching. Qwen3.5 additionally has a stable exact-relevance control feature in forced choice.

For Qwen3.5, both leave-one-domain-out and one-time fixed extraction generalize across eight unfamiliar held-out domains. The fixed vector gives strong within-domain linear separation under two profile carriers, survives answer-label changes, shows a monotonic dose response, tolerates new profile wording, and partially reduces the natural gap when ablated. The fixed prospective test narrowly missed its shuffled-label intervention cutoff, however, and the result did not transfer to the frozen automated free-generation outcome or replicate specifically in Phi. The defensible positive result is therefore a stable, linearly extractable and causally effective control feature for Qwen3.5's forced-choice relevance decision. It is not yet a general mechanism for adaptive explanation, a localized circuit, or a cross-family feature.

## Required human checks before submission

- Independently review all eight passages and summary pairs; the added BERT,
  GCN, DDPM, and XGBoost items have not yet received human verification.
- Annotate all 16 examples in `results/manual_review_sample.csv`, which covers
  both models, both reader conditions, and every source.
- Review the invented prospective passages in `results/prospective_stimulus_review.csv`, then score `results/qwen35_4b_fixed_direction_generation/blind_review_with_sources.csv` before opening its `blind_key.csv`.
- Score both `results/qwen25_3b_steered_generation/blind_review.csv` and
  `results/qwen35_4b_steered_generation/blind_review.csv` before opening their
  corresponding `blind_key.csv` files.
- Verify at least four A/B logit margins directly from `behavior_rows.csv`.
- Read `experiment.py`, `generate_gists.py`, and
  `random_steering_baseline.py` and `ablation_baseline.py` until every headline
  number is understandable.
- Write the application summary in your own voice and state which design
  decisions were yours.
