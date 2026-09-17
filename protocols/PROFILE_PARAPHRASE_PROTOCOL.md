# Frozen profile-carrier paraphrase robustness test

Frozen before model evaluation. This is a robustness follow-up on the existing eight
synthetic sources, not a new independent-domain confirmation.

The already fitted Qwen3.5-4B vector and frozen layer 17 are reused unchanged. Only
the reader-profile carrier wording is replaced. Domain-expert and matched-other-expert
profiles use exactly the same new syntactic frames and differ only in the named field.
Neither new frame occurs in discovery or the first synthetic test.

Using the uniform-sign five-point forced-choice dose test (`-2,-1,0,+1,+2`), a robust
context-independent direction requires, for each of A/B, X/Y, and 1/2 and for each
reader condition, at least 7/8 positive source slopes and exact one-sided p <= 0.05.
No vector, layer, wording, or threshold changes after results are observed.

