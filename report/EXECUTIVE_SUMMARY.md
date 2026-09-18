# Is Explanation Depth Audience-Conditioned?

## Executive Summary

**Question -** Does a language model represent what the reader knows? And can one activation direction control how technical its explanation is?

**Motivation -** When I asked regular AI apps for the gist of papers, they often collapse to generic definitions and analogies, removing technical terms that would make the explanation shorter and more useful. This was frustrating but also made me wonder whether it was only a prompting failure, or whether the model represented what kind of explanation suited the reader.

I think this question matters beyond my frustration. If models adjust their behavior based on who they think the user is, understanding it would help us audit them.

**Setup -** I quickly realized “expert versus novice” confounds relevant knowledge with general sophistication. I therefore used a narrower comparison that was easier to measure: one expert in the passage’s field and one in an unrelated field. Both received the same passage and two accurate summaries - one concise and terminology-heavy and the other more accessible and explicit.

In the main experiments, the model sees the passage and both summaries rather than generating free text. I measure the log-probability difference between their labels. A positive margin means the model favors the technical summary.

I extracted an audience-contrast direction from residual-stream activations using eight real ML topics, froze it, and tested it on eight invented methods to remove reliance on familiar papers.

Qwen3.5-4B was the main experimental model. Qwen2.5-3B was used for the initial behavioral comparison, and Phi-3.5 Mini for cross-family replication.


## Key Takeaways

### 1. Was there actually anything to explain or was it just me and one model?
Both Qwen models preferred the technical summary more when the stated reader had relevant expertise. In Qwen3.5-4B, one frozen direction increased this preference in all eight invented domains, including after changing A/B labels to X/Y and 1/2 and rewriting the reader profiles.

![Frozen-direction transfer across the eight invented domains and three answer-label encodings.](report_assets/exec_final_1_transfer.png)

### 2. The direction worked - for now. But was it what the model normally used?

Unmodified activations separated the readers along this direction (AUC = 0.81; 0.84 with rewritten profiles), and none of the 100 random directions matched its aggregate steering effect. But removing it reduced only 0.047 of the original 0.594 audience gap - approximately 8%. Therefore, it influenced the answer when added but accounted for little of the model’s unmodified audience effect.

![Comparison of the steering and ablation effects with the original audience gap.](report_assets/exec_final_2_interventions.png)

### 3. Maybe behavior needed more than one direction?

A rank-3 SVD subspace removed 0.180 of the gap (30%) suggesting a low-rank but not rank-1 representation. But SVD orders components by activation variance, not by causal effect. Still sceptical, I selected and froze one direction inside the subspace using validation data. It nearly matched rank 3 subspace on invented methods (0.156 vs 0.180) and rewritten profile (0.234 vs 0.238), and exceeded it on held-out real profiles (0.168 vs 0.117). So the variance-leading direction was not the most causally effective. This result remains preliminary without selection-matched shuffled-label controls.

### 4. Where did the results stop holding up?

Six of 100 shuffled-label directions matched or exceeded the learned effect. Although 6/100 is small, this means the learned direction was not cleanly specific to the correct audience labels.
Also, an automatic technical-term metric found no reliable effect in free generation.

And Phi-3.5 Mini produced positive average steering, but failed the answer-label, control and ablation tests. I therefore found no cross family replication.

![Random and shuffled-label controls compared with the learned direction.](report_assets/exec_final_3_controls.png)

What I find most interesting (at least in Qwen3.5-4B) is that a direction can encode a behavioral contrast, steer the output, and still explain little of the computation the model normally uses. I did not find the convincing evidence that this generalises across models or to free generation.
