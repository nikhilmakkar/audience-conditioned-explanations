# Component interpretation and qualitative damage protocol

This protocol was written before inspecting component-wise interventions or
rank-3/rank-7 generations.

## Questions

1. Do the first three SVD components independently reduce the natural audience
   gap, or does the rank-3 result require their combination?
2. Are components associated with particular discovery domains?
3. Does rank-7 ablation visibly degrade ordinary generated answers more than
   rank-3 ablation?

## Frozen model and intervention

- Qwen3.5-4B.
- Previously learned layer-17 relevance basis; no refitting.
- Ablation at all token positions of residual layer 17.
- Greedy decoding for qualitative outputs.

## Component experiment

- Evaluate every non-empty subset of components 1–3 on the original eight
  synthetic methods.
- Evaluate components 4–8 individually for context.
- Report mean relevance-gap reduction and every source-level reduction.
- These are exploratory functional signatures, not independent confirmatory
  tests.
- Non-additivity is expected: the effect of ablating components 1 and 2 together
  need not equal the sum of their separate effects.

## Discovery loadings

- Recompute the eight frozen domain contrast vectors.
- Project them onto each of the eight saved SVD components.
- Plot signed loadings by domain and component.
- Because SVD signs are arbitrary, interpret relative patterns rather than the
  positive or negative name of a component.
- Components with close singular values may rotate under resampling; semantic
  names are not justified without a stability analysis.

## Qualitative damage comparison

- Use a frozen set of twelve ordinary prompts unrelated to the audience task.
- Generate baseline, rank-3-ablated, and rank-7-ablated answers with greedy
  decoding and the same token limit.
- Randomize the three answers within prompt in a blinded review sheet.
- Score coherence, correctness, instruction following, and obvious corruption
  before opening the condition key.
- Qualitative inspection is supplementary. The frozen WikiText NLL/KL analysis
  remains the primary general-damage measurement.
