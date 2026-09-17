# Is Explanation Depth Audience-Conditioned?

## Executive summary

**Question —** Does a language model represent what it thinks the reader knows? More specifically, when told the reader’s expertise, does one activation direction represent its relevance and control preference between technical and accessible explanations?

**Motivation —** When I ask my regular AI apps for the *gist* of papers, they often collapse to generic definitions and analogies, removing technical terms that would make the answer shorter and more useful. This was frustrating and made me wonder whether it was only a prompting failure or whether the model represented what explanation suited the reader.

This matters beyond my frustration: if models adjust their behaviour based on who they think the user is, understanding it would help us audit them. I am not testing broad claims such as sandbagging—only one measurable instance of behaviour changing with the stated user.

**Setup —** I quickly realised that “expert versus novice” confounds relevant knowledge with general sophistication. I instead compared two equally technical readers: one expert in the passage’s field and one in an unrelated field. Both received the same passage and two accurate summaries—one terminology-heavy, the other more accessible.

In the main experiments, the model sees the passage and both summaries rather than generating free text. I measure the log-probability difference between their labels; a positive margin favours the technical summary.

I extracted an audience-contrast direction from residual-stream activations using eight real ML topics, froze it, and tested it on eight invented methods to remove reliance on familiar papers. Qwen3.5-4B was the main model; Qwen2.5-3B and Phi-3.5 Mini provided behavioural and cross-family checks.

![The frozen direction increased technical-summary preference across all eight invented domains and three answer-label encodings.](report_assets/exec_final_1_transfer.png)

## Key takeaways

**1. Was there actually anything to explain—or was it just me and one model?**

Both Qwen models preferred the technical summary more for readers with relevant expertise. In Qwen3.5-4B, one frozen direction increased this preference in all eight invented domains, including after changing A/B labels to X/Y and 1/2 and rewriting the reader profiles.

**2. The direction worked. But was it what the model normally used?**

Unmodified activations separated the readers along this direction (AUC 0.81; 0.84 with rewritten profiles), and none of 100 random directions matched its steering effect. But removing it erased only 0.047 of the original 0.594 audience gap—about 8%. It could steer the answer while explaining little of the unmodified decision.

**3. Maybe the behaviour needed more than one direction?**

A rank-3 SVD subspace removed 0.180 of the gap (30%), suggesting a low-rank but not rank-1 representation. I did not stop there. Using validation data, I selected and froze one direction inside that subspace. It nearly matched rank 3 on invented methods (0.156 versus 0.180) and rewritten profiles, and exceeded it on held-out real profiles. Three necessary dimensions were not established: the variance-leading direction was not the most causally effective. This remains preliminary without selection-matched shuffled-label controls.

![The direction could steer the answer, but removing it explained only a small part of the natural audience effect.](report_assets/exec_final_2_interventions.png)

**4. Where did the result stop working?**

Six of 100 shuffled-label directions matched or exceeded the learned effect. Although 6/100 is small, this means the learned direction was not cleanly specific to the correct audience labels. The automatic technical-term metric found no reliable effect in free generation. Phi showed positive average steering, but failed the answer-label, control and ablation checks. I therefore found no convincing cross-family replication.

![None of 100 random directions matched the learned effect, while six of 100 shuffled-label directions did.](report_assets/exec_final_3_controls.png)

What I find most interesting is that, at least in Qwen3.5-4B, a direction can encode a behavioural contrast, steer the output, and still explain little of the computation the model normally uses. This is a forced-choice result, not a general explanation-depth mechanism.
