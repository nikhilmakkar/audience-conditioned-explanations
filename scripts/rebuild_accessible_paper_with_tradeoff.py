"""Rebuild the accessible paper with an explicit rank/evidence trade-off page."""
import build_accessible_paper as paper

tradeoff = (
    'Figure 14b — Raw effect and specificity are different',
    'multidim_rank_tradeoff.png',
    'A larger raw gap reduction is not automatically stronger evidence. Higher-rank ablation removes more activation axes and can capture both the intended audience feature and unrelated structure.',
    'Express each reduction as a fraction of the original behavior, then ask how often the same-rank pipeline reproduces it after the semantic labels are destroyed.',
    'The left panel divides reduction by the original mean gap of 0.59375. The right counts shuffled-label controls whose reduction is at least as large. Lower control counts mean stronger evidence that the labels mattered.',
    'Rank 3 removes 30.3% of the measured gap and has 2/100 exceedances: corrected Monte Carlo p = 3/101 = 0.030. Rank 7 removes 44.7% but has 5/100 exceedances: corrected p = 6/101 = 0.059. Rank 8 removes 44.1% and has 7/100: corrected p = 8/101 = 0.079.',
    'Rank 3 has the clearest shuffled-label specificity; ranks 7 and 8 have larger raw effects but weaker evidence.',
    'IMPORTANT TRADE-OFF'
)

index = next(i for i, row in enumerate(paper.FIGURES)
             if row[0].startswith('Figure 15'))
paper.FIGURES.insert(index, tradeoff)
paper.build()
