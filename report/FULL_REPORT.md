# Is Explanation Depth Audience-Conditioned?

## A small mechanistic study of what happens when the reader changes

## The question

This project started from a small but recurring frustration. When I ask my regular AI apps for the *gist* of a paper, they often remove exactly the technical terms that would make the explanation shorter and more useful. Instead, I get definitions, generic analogies and sometimes a long story explaining something I already know.

At first my question was very broad: does the model know what the reader knows, and can this be controlled internally? That phrasing sounds nice but is almost impossible to test. “Expertise” can mean knowledge of a field, general sophistication, familiarity with one method, confidence, preferred writing style, or simply that the prompt contains the word *expert*.

So I narrowed the project to one measurable question:

> When a prompt states the reader’s expertise, does a language model represent whether that expertise is relevant to the passage? Can a direction in its activations control whether it prefers a technical or accessible summary?

To be very clear, I am not testing whether the model has a complete representation of a person, and I am not testing broad behaviours such as sandbagging. I am studying one controlled instance in which changing the stated reader changes the model’s preferred answer.

The work is methodologically inspired by activation-steering studies such as [*Refusal in Language Models Is Mediated by a Single Direction*](https://arxiv.org/abs/2406.11717) and [*Understanding Reasoning in Thinking Language Models via Steering Vectors*](https://arxiv.org/abs/2506.18167). Those papers study much cleaner behavioural targets. Audience-appropriate explanation is subjective, which made it especially important to separate the different claims I could make.

I use four levels of evidence throughout the report:

1. **Behaviour:** does the unmodified model change its answer when the reader changes?
2. **Representation:** can the reader contrast be read from normal activations?
3. **Causal influence:** does adding or removing the activation feature change the answer?
4. **Generalisation:** does the result survive new domains, prompt wording, answer labels, free generation and another model family?

The main result is not simply that a steering vector “worked.” The more interesting result is that a direction could encode the audience contrast and steer the answer while explaining little of the computation the model normally used.

## Making the question measurable

### Why I did not compare an expert with a novice

My first instinct was to compare an expert reader with a novice. I quickly realised this would be a poor experiment. An expert profile and a novice profile differ in too many ways: vocabulary, education, confidence and whether introductory teaching is expected. If the output changed, I would not know whether the model understood relevant knowledge or merely reacted to a generic sophistication cue.

I therefore compared two equally technical readers:

- one expert in the field of the passage;
- one expert in an unrelated field.

Both readers receive the same technical passage, the same instruction and the same two candidate summaries. One candidate is concise and retains domain terminology. The other explains the same mechanism more explicitly in accessible language. Both are intended to be accurate. Only the relevance of the reader’s expertise changes.

![Controlled forced-choice task. The passage and candidate answers stay fixed; only the reader’s field changes.](report_assets/application_task.png)

*Figure 1. Example of the controlled comparison. This is a design figure, not a result.*

Here is a held-out example using an invented method:

> **Passage:** DriftGate estimates a hidden dynamical state from noisy controls and observations. It alternates a linear prediction with a measurement update, rejecting observations whose covariance-normalised innovation is above a fixed gate.
>
> **Relevant reader:** The reader is deeply familiar with state-space estimation, Kalman filtering and feedback control. They want the core idea explained concisely.
>
> **Unrelated expert:** The reader is deeply familiar with population genetics and phylogenetic reconstruction. They want the core idea explained concisely.
>
> **Technical summary:** DriftGate is a robust Kalman-style filter that propagates state covariance, gates observations by squared Mahalanobis innovation and applies the covariance-derived gain only to accepted measurement updates.
>
> **Accessible summary:** DriftGate tracks a changing hidden quantity by first predicting its next value and uncertainty, then correcting that prediction with new measurements. It ignores measurements that are implausibly far from the prediction and trusts accepted measurements according to their noise.

In the main experiments the model does not generate an explanation. It chooses between labelled candidates. I measure

\[
m = \log P(\text{technical label})-\log P(\text{accessible label}).
\]

A positive margin means the model prefers the technical summary. The **audience gap** is the margin for the relevant expert minus the margin for the unrelated expert. Candidate order and prompt variants are balanced, averaged within a domain, and the domain—not each repeated prompt—is treated as the independent unit.

This forced-choice setup is less realistic than free generation. I used it because it gives a clean outcome and keeps summary content and correctness fixed. I later return to free generation as a separate external-validity test.

## Was there actually a behaviour to explain?

Before looking at activations I checked whether changing reader relevance affected the unmodified model at all. This matters because a direction cannot explain an audience-conditioned behaviour that does not exist.

The early experiments also exposed the original confound. Both Qwen models reacted strongly to broad “expert” cues, including expertise unrelated to the passage. After switching to matched experts, the remaining exact-relevance effect was smaller. Qwen3.5-4B showed the cleaner and more consistent effect, so it became the main mechanistic model. Qwen2.5-3B remained useful as an initial behavioural comparison rather than evidence for the complete mechanism.

![Relevant-minus-unrelated expert technical-summary margins in the unmodified Qwen models.](report_assets/03_behavior.png)

*Figure 2. The matched-expert comparison asks about relevant knowledge rather than generic sophistication. The Qwen3.5 result was consistent enough to justify investigating its activations.*

This first correction changed the project. The initial phenomenon was not simply “the model knows the user is an expert.” Much of the large effect came from coarse cues. The narrower relevance-conditioned behaviour existed, but it was smaller and model-dependent.

## Extracting a direction without letting the test domain leak in

For each prompt I recorded the residual-stream activation at the final prompt token. A simple audience-contrast direction is the mean activation for relevant experts minus the mean for unrelated experts:

\[
v = \mathbb{E}[h\mid\text{relevant expert}]
  - \mathbb{E}[h\mid\text{unrelated expert}].
\]

I swept layers and token positions on validation data rather than choosing a convenient-looking location after seeing the final test. The main fixed direction came from layer 17 and the final prompt token in Qwen3.5-4B.

There were still two easy ways to fool myself:

1. a vector could encode the particular ML topics used to construct it;
2. a vector could be refitted separately for every test domain and therefore never demonstrate one common feature.

I addressed these in two stages. First, leave-one-domain-out experiments learned a direction on seven real domains and tested it on the excluded eighth. Addition was positive in all eight held-out domains. Then I used the stricter test: learn one direction once from all eight real ML domains, freeze its layer, token position, norm and coefficient, and apply it unchanged to eight invented technical methods from unrelated fields.

The invented methods included control, causal inference, numerical fluid dynamics, databases, computational biology, radar, cryptography and audio processing. They were self-contained descriptions rather than known papers, reducing the possibility that a memorised title-summary pairing determined the answer.

## The fixed direction transferred

The frozen Qwen3.5 direction increased preference for the technical summary in all eight invented domains. The mean effect was 0.156 using A/B answer labels. Changing the answer labels to X/Y and 1/2 gave mean effects of 0.178 and 0.184, again positive in every domain.

![Frozen-direction steering effects for every invented domain under three answer-label encodings.](report_assets/exec_final_1_transfer.png)

*Figure 3. Every bar is to the right of zero. Re-encoding the answer rules out a result tied only to the original A or B token.*

With eight independent domains all having the predicted sign, the exact one-sided sign-flip probability is \(1/2^8=0.0039\) for each encoding. I use this test because the number of domains is small and I do not want repeated prompt variants to create artificial precision.

I also tested whether the effect behaved like a control axis rather than a single lucky comparison. Across coefficients \(-2,-1,0,1,2\), technical-summary preference increased monotonically for both reader conditions and all three answer encodings. The same pattern survived rewriting every reader description with sentence templates not used during direction extraction.

![Five-point intervention sweep for the original and rewritten reader profiles.](report_assets/07_dose.png)

*Figure 4. Increasing the coefficient moves the forced-choice margin in the predicted direction. Rewritten profiles test dependence on one sentence template.*

At this point the direction looked surprisingly robust: new topics, invented mechanisms, new answer tokens and new profile wording all gave the same qualitative result. But this established only that the direction was sufficient to influence the forced-choice answer. It did not show that the unmodified model naturally represented or relied on it.

## Could the audience contrast be read from normal activations?

For the representation test I did not alter the model. I projected the unmodified activations onto the frozen direction and asked whether relevant-expert prompts landed above unrelated-expert prompts.

The within-domain projection gap was positive in all eight invented domains. Pooling the ranking information gave an area under the ROC curve (AUC) of 0.809. After rewriting every reader profile, the gap remained positive in all eight domains and AUC increased to 0.836.

![Natural separation between relevant and unrelated reader profiles along the frozen direction.](report_assets/08_projection.png)

*Figure 5. This tests whether the audience contrast exists in the model before any steering. Each bar is the relevant-reader projection minus the unrelated-reader projection for one domain; a positive bar means the learned direction ranks the relevant reader higher. All eight domains remain positive after the reader descriptions are rewritten, so the result is not tied only to one profile sentence template.*

AUC here has a simple interpretation: if I randomly take one relevant-reader activation and one unrelated-reader activation, the projection ranks the relevant one higher roughly 81–84% of the time. A fixed threshold transferred less well—62.5% accuracy on the original invented profiles and 71.9% after rewriting—so the relative separation is stronger than absolute calibration.

This gives evidence that the contrast is linearly readable in normal activations. “Readable” is deliberately weaker than “the model uses this direction to make its decision.”

## Was the steering effect special?

Neural networks are high-dimensional and sensitive to intervention. A direction changing the answer is not interesting if any vector with the same norm does the same thing.

I used two controls:

- **Random directions:** isotropic vectors matched to the learned direction’s norm. These test whether an arbitrary perturbation works.
- **Shuffled-label directions:** the full extraction pipeline is repeated after destroying the relevant-versus-unrelated labels. These are a harder control because they preserve structure in the prompt activations while removing the intended semantic grouping.

A small example makes the shuffled control clearer:

| Activation came from | Correct group | One possible shuffled group |
|---|---|---|
| ResNet + relevant expert | relevant | positive |
| ResNet + unrelated expert | unrelated | negative |
| Transformer + relevant expert | relevant | negative |
| Transformer + unrelated expert | unrelated | positive |

The real direction subtracts the mean of all unrelated-expert activations from the mean of all relevant-expert activations. For each shuffled control, I instead pooled the discovery activations and randomly divided them into two equally sized groups, ignoring the real audience condition. I then subtracted one random-group mean from the other and normalised the resulting vector to the same norm as the learned direction. The table shows only four examples; the actual control used all discovery activations and repeated this process with 100 random assignments.

This preserves the geometry and structure of real prompt activations but removes the proposed meaning. If these shuffled directions steer equally well, the extraction procedure may be finding generally influential axes rather than something specific to audience relevance.

None of 100 random directions matched the learned aggregate effect. However, six of 100 shuffled-label directions matched or exceeded it.

![Steering effects of all random and shuffled-label control directions compared with the learned direction.](report_assets/exec_final_3_controls.png)

*Figure 6. Random perturbation is not an adequate explanation, but the shuffled-label result prevents a clean specificity claim.*

I had frozen a cutoff of at most five matches out of 100. The result missed it by one. More importantly, the cutoff itself should not turn 5/100 into truth and 6/100 into falsehood. The useful interpretation is that the direction clearly beats arbitrary perturbations, while the extraction procedure can occasionally find equally strong directions even after the correct audience labels are destroyed.

Therefore I do not treat the fixed intervention direction as uniquely specific to audience relevance.

## Steering the model is not the same as explaining its normal behaviour

Adding the direction asks whether it can influence the answer. Ablation asks the harder question: how much does the unmodified model normally rely on it?

The natural audience gap across the invented domains was 0.594. Removing the original direction at layer 17 reduced this gap by only 0.047—approximately 8%. Seven of the eight domain-level reductions had the predicted sign, but most of the original behaviour remained.

The intervention was not simply destroying the language model. On 4,080 WikiText next-token predictions, selected-layer removal changed negative log-likelihood by -0.0011, produced mean KL divergence of 0.00073, and retained the original top prediction 98.4% of the time. The operation was small in ordinary-language terms.

![Comparison of steering and ablation effects against the natural audience gap.](report_assets/exec_final_2_interventions.png)

*Figure 7. The original direction has visible leverage when inserted, but its removal accounts for only a small part of the unmodified audience gap.*

This was the first result that made the clean single-direction story difficult to maintain. The direction was readable and steerable, but weak as a mediator of the normal decision. In other words, the model can be pushed using a feature without depending heavily on that exact feature by default.

## Maybe the behaviour needed more than one direction?

Expertise relevance is not an obviously one-dimensional phenomenon. Once the single-direction ablation removed only 8%, the natural next hypothesis was that the remaining behaviour occupied a small subspace rather than one direction.

I formed one relevant-minus-unrelated contrast vector for each of the eight real discovery domains, stacked them, and applied uncentred singular value decomposition (SVD). Uncentred SVD was used because centring would remove the shared mean contrast I wanted to study. The resulting components are ordered by how much variation in the eight contrast vectors they capture.

I ablated nested subspaces with ranks from one to eight. The maximum possible learned rank was eight because there were only eight domain-contrast vectors—not because the 2,560-dimensional activation space contained only eight possible directions.

The leading SVD component removed 0.039 of the natural gap. The rank-3 subspace removed 0.180, or about 30%, and the result was positive in all eight invented domains. No rank-matched random subspace out of 100 matched it, while two of 100 shuffled-label rank-3 subspaces did. Rank 7 removed more of the raw gap, but its shuffled-label specificity was weaker. The result was therefore not simply that removing more dimensions always improved the evidence.

My first interpretation was that the behaviour was low-rank but not rank-1. That was premature.

SVD orders components by activation variance. It does not know which direction has the greatest causal effect on the output. To test whether three dimensions were actually necessary, I searched for a single direction inside the rank-3 span using only real-domain validation profiles. I froze the selected mixture and evaluated it on untouched data.

![Held-out ablation performance of the variance-leading direction, a validation-selected one-dimensional mixture and the complete rank-3 subspace.](report_assets/intrinsic_rank_correction.png)

*Figure 8. One selected direction nearly matches rank 3 on invented methods and rewritten profiles, and exceeds it on held-out real profiles.*

The selected one-dimensional direction removed:

- 0.168 on held-out real profiles, compared with 0.117 for rank 3;
- 0.156 on invented methods, compared with 0.180 for rank 3;
- 0.234 after rewriting profiles, compared with 0.238 for rank 3.

All three selected-direction effects were positive in all eight domains. Therefore my experiments do not establish that three dimensions are necessary. What they show is more specific: the variance-leading direction is not the most causally effective direction, and a small learned subspace contains a much better one-dimensional causal axis.

This correction is itself preliminary. I selected the direction from many candidate mixtures but did not repeat the complete selection procedure inside many shuffled-label subspaces. Without that selection-matched null, the selected direction’s performance may be optimistic.

## Where did the result stop working?

### Free generation

The motivating observation concerned freely written explanations, not selection between prepared answers. So this had to be tested even though it was harder to measure.

I generated 160 short gists across eight invented methods, two reader conditions, two profile phrasings and five steering coefficients using greedy decoding. The frozen automatic outcome was coverage of source-specific technical terms.

The result was essentially flat. Moving from coefficient -2 to +2 changed term coverage by 0.011, with only two of eight domains meaningfully moving in the predicted direction (one-sided sign-flip \(p=0.25\)). A third value was \(1.1\times10^{-16}\), which is floating-point noise rather than a visible positive effect. The fitted five-point slope was positive in three domains and was also not significant (\(p=0.1875\)). Word count decreased slightly, so a hidden positive result was not being masked by longer outputs.

![Technical-term coverage in freely generated gists under five steering coefficients.](report_assets/10_generation.png)

*Figure 9. The automatic free-generation metric found no reliable steering effect.*

The metric is limited: a good explanation can paraphrase a technical term, and simply counting terminology does not measure correctness or usefulness. But that does not make the registered result positive. The honest conclusion is that forced-choice control did not transfer to the automatic free-generation outcome.

### A different model family

Qwen2.5 and Qwen3.5 are not independent architectural evidence. I therefore repeated the fixed-direction protocol prospectively on Phi-3.5 Mini, selecting layer 12 using only real-domain discovery performance before scoring the invented set.

Phi produced positive average steering effects: 0.170 for A/B, 0.094 for X/Y and 0.170 for 1/2. Looking only at averages would make this seem like a replication. It was not.

X/Y was positive in only six of eight domains, six random controls matched the A/B aggregate, 18 shuffled-label controls matched it, and only two individual domains exceeded their isotropic 95th percentiles. Selected-layer and all-layer ablation were also inconclusive.

![Qwen and Phi steering effects together with their random and shuffled-label controls.](report_assets/12_cross_family.png)

*Figure 10. Positive average steering is not sufficient for replication. Phi failed the robustness, specificity and ablation criteria.*

I therefore found no cross-family replication of a direction-specific, normally used audience-relevance feature.

## What I think the results mean

The positive result is narrow but real. In Qwen3.5-4B, one frozen residual-stream direction:

- separates relevant from unrelated reader profiles in unmodified activations;
- transfers from real ML topics to eight invented methods;
- changes the forced-choice technical-summary margin in every test domain;
- survives answer-label changes, reader-profile rewriting and a dose-response test;
- causes little change to ordinary next-token prediction.

But each stronger interpretation encountered a boundary:

- shuffled-label extraction occasionally produced equally strong steering;
- removing the original direction explained only 8% of the natural audience gap;
- rank 3 removed more, but a selected single direction nearly matched it;
- the automatic free-generation result failed;
- Phi did not replicate the complete result.

So I do not think the evidence supports a “single explanation-depth direction” in language models. It supports a transferable Qwen3.5 feature for one forced-choice audience-relevance decision.

The distinction I find most useful is between **representation**, **steerability** and **mediation**. These are easy to collapse into one sentence:

> “The model represents audience relevance in a direction which controls its behaviour.”

My experiments show why that sentence is too compressed. A direction can provide a good readout and a useful intervention handle while accounting for little of the unmodified computation. Likewise, a higher-rank subspace can look more causal merely because the one-dimensional comparison was chosen for variance instead of causal leverage.

## Limitations

### The task measures summary preference, not explanation depth in general

The main outcome is a forced choice between two prepared summaries. It does not measure every way a model can adapt an explanation: choice of examples, assumed prerequisites, sentence structure, length, confidence or pedagogical strategy. The failed generation test makes this distinction especially important.

### Eight domains are eight independent units

There are many prompt variants, answer orders and intervention coefficients, but these do not create hundreds of independent domains. The principal statistical unit is the source domain, giving eight discovery and eight prospective evaluation domains. Exact sign tests avoid exaggerated precision, but the estimates remain coarse.

### The prospective methods are synthetic

Invented methods help rule out direct paper memorisation, but they are shorter and cleaner than real papers. A model may behave differently when a passage contains ambiguous terminology, missing prerequisites or a long chain of argument.

### Candidate summaries were only lightly human-validated

The experiment assumes that the technical and accessible summaries are both accurate and comparably useful for their intended readers. I manually reviewed the small prospective set for plausibility, but there was no large expert annotation study. Errors or uneven information content could affect the measured margin.

### The intervention-specificity result is not clean

Six of 100 shuffled-label directions matched the learned steering effect. This is small relative to 100 but large enough that I cannot say the correct audience labels uniquely produced the useful direction. The result beats isotropic randomness more clearly than it beats a structured extraction null.

### The selected one-dimensional result is exploratory

The selected mixture was chosen after searching many directions within the rank-3 span. I used separate validation and test data, but I did not run the same search inside many shuffled-label subspaces. This missing selection-matched control is the most important unfinished mechanistic experiment.

### The free-generation metric is incomplete

Technical-term coverage can miss paraphrases and does not directly measure whether an explanation is correct, appropriately concise or useful. Blinded human evaluation would be better. Nevertheless, the predefined automatic result failed and is reported as such.

### The result did not generalise across model families

The complete positive result is specific to Qwen3.5-4B. Qwen2.5 supported the initial behavioural observation but not all robustness tests, and Phi failed the prospective replication criteria. Model-specific activation geometry should not be described as a universal LLM mechanism.

### This is not a circuit-level explanation

The analysis concerns residual-stream directions and subspaces at selected layers. I did not identify the attention heads or MLP features that construct, transmit or read the audience signal. Cross-layer cosine similarity is descriptive and cannot by itself localise a causal circuit.

## What I would do next

The most important next experiment is not another steering coefficient or another random seed. It is to repeat the complete selected-one-dimensional search inside many shuffled-label rank-3 subspaces and evaluate every selected direction on a fresh domain set. That would test whether the improved single direction reflects audience structure or the flexibility of searching a three-dimensional space.

The second priority is a blinded evaluation of free generations. Reviewers should see the source passage and score technical usefulness, omitted mechanisms and factual accuracy without seeing the intervention coefficient. This would tell us whether the flat terminology metric missed a real qualitative change or whether forced-choice steering simply does not transfer to generation.

After those, I would test larger and more diverse model families and then localise which components write and read the selected causal direction. Raw cosine similarity between different models would not be sufficient because their activation spaces are not aligned.

## Experimental details

| Item | Main setting |
|---|---|
| Main model | Qwen3.5-4B |
| Behavioural comparison | Qwen2.5-3B and Qwen3.5-4B |
| Cross-family test | Phi-3.5 Mini |
| Discovery domains | Eight real ML methods/topics |
| Prospective domains | Eight invented methods from unrelated fields |
| Activation location | Residual stream, layer 17, final prompt token |
| Direction | Mean relevant-expert activation minus matched unrelated-expert activation |
| Primary outcome | Technical-label log probability minus accessible-label log probability |
| Main intervention | Frozen direction added at the selected residual-stream location |
| Necessity test | Projection removal at all token positions in the selected layer |
| Controls | 100 norm-matched random directions and 100 shuffled-label directions |
| Generation | Greedy decoding; five coefficients from -2 to +2 |
| Inference | Domain aggregation and exact one-sided sign-flip tests |

### Additional checks

An exhaustive discovery-data sensitivity analysis fitted directions from all subsets of one, two, four, six and eight real domains—135 directions in total. Six of eight one-domain directions and 26 of 28 two-domain directions passed the seven-of-eight consistency rule. Every four-, six- and eight-domain direction produced positive effects in all eight prospective domains. Mean steering remained near 0.15, while cosine similarity to the full direction increased from 0.579 with one discovery domain to 0.957 with six. More domains mainly stabilised the geometry rather than creating the behavioural effect.

The rank-3 ablation caused modest but detectable ordinary-text disturbance. Over 1,016 WikiText next-token predictions, negative log-likelihood increased by 0.00656 (about 0.21%), mean KL divergence was 0.00248 and top-1 agreement was 96.75%. This is more disruption than the mean random rank-3 subspace but still small in absolute terms. I therefore describe rank-3 intervention as targeted with modest general-language impact, not damage-free.

Across layers, directions learned before layer 15 had little similarity to the selected layer-17 direction. Similarity rose around layers 15–17 and then evolved more smoothly downstream. This is consistent with the geometry consolidating near the middle of the network, but neighbouring-layer ablations would be needed before making a causal localisation claim.

## References

1. Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W. and Nanda, N. (2024). [*Refusal in Language Models Is Mediated by a Single Direction*](https://arxiv.org/abs/2406.11717).
2. Venhoff, C., Arcuschin, I., Torr, P., Conmy, A. and Nanda, N. (2025). [*Understanding Reasoning in Thinking Language Models via Steering Vectors*](https://arxiv.org/abs/2506.18167).
3. Cloud, A., Le, M., Chua, J., Betley, J., Sztyber-Betley, A., Hilton, J., Marks, S. and Evans, O. (2025). [*Subliminal Learning: Language Models Transmit Behavioral Traits via Hidden Signals in Data*](https://arxiv.org/abs/2507.14805).

## Reproducibility notes

The protocols, scripts, stimulus tables and result CSVs are retained in the project directory. The main prospective protocol was frozen before scoring the invented methods, and the dimensionality and cross-family extensions have their own protocol files. Any public version should link the code repository and state clearly which parts of implementation, analysis and writing used AI assistance.
