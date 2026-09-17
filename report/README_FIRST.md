# MATS submission bundle

This folder contains only the writing and figures needed to assemble the Google Doc. It deliberately excludes code, raw CSVs, protocols, old figures and PDFs.

## Recommended Google Docs tabs

### Tab 1 — Executive Summary

Copy the contents of `EXECUTIVE_SUMMARY.md`. Do not create subtabs for Question, Motivation, Setup or individual takeaways. Use bold inline labels instead.

Recommended figures, in order:

1. `report_assets/exec_final_1_transfer.png`
2. `report_assets/exec_final_2_interventions.png`
3. `report_assets/exec_final_3_controls.png`

### Tab 2 — Full Report

Use `FULL_REPORT.md` as the text source. Suggested subtabs:

1. Question and experimental setup
2. Finding and testing the direction
3. What did the model normally use?
4. Was one direction enough?
5. Where the results broke
6. Interpretation and limitations

### Tab 3 — Experimental Details

Move the Experimental details and References sections here if the Full Report tab becomes too long. Otherwise leave them at the end of the Full Report.

## Important corrected labels

- “Rewritten reader wording” replaces the internal term “new carrier”.
- The free-generation endpoint has 2/8 meaningfully positive domains. A third stored value was floating-point zero. The five-point slope is positive in 3/8.
- Phi has 18/100 shuffled-label controls matching or exceeding the learned effect.
- Qwen has 0/100 random and 6/100 shuffled-label controls matching or exceeding the learned effect.

## Figure list

- `application_task.png` — controlled task example
- `03_behavior.png` — initial behavioural comparison
- `exec_final_1_transfer.png` — frozen-direction transfer
- `07_dose.png` — dose response and rewritten reader wording
- `08_projection.png` — natural activation separation
- `exec_final_3_controls.png` — Qwen random and shuffled controls
- `exec_final_2_interventions.png` — steering versus ablation
- `intrinsic_rank_correction.png` — rank 3 versus selected 1D
- `10_generation.png` — failed free-generation metric
- `12_cross_family.png` — Qwen versus Phi
