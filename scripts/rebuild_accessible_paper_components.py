"""Rebuild the accessible paper with rank trade-off and component pages."""
import build_accessible_paper as paper

tradeoff = (
    'Figure 14b — Raw effect and specificity are different',
    'multidim_rank_tradeoff.png',
    'A larger raw gap reduction is not automatically stronger evidence. Higher-rank ablation removes more activation axes and can capture both the intended audience feature and unrelated structure.',
    'Express each reduction as a fraction of the original behavior, then ask how often the same-rank pipeline reproduces it after the semantic labels are destroyed.',
    'The left panel divides reduction by the original mean gap of 0.59375. The right counts shuffled-label controls whose reduction is at least as large. Lower control counts mean stronger evidence that the labels mattered.',
    'Rank 3 removes 30.3% and has corrected Monte Carlo p = 3/101 = 0.030. Rank 7 removes 44.7% but has p = 6/101 = 0.059. Rank 8 removes 44.1% and has p = 8/101 = 0.079.',
    'Rank 3 has the clearest shuffled-label specificity; ranks 7 and 8 have larger raw effects but weaker evidence.',
    'IMPORTANT TRADE-OFF'
)

components = (
    'Figure 17 — Can the individual directions be interpreted?',
    'component_interpretation.png',
    'A subspace can have a reliable causal effect even when its individual coordinate axes do not correspond to simple human concepts. We therefore ablated components separately and in combinations, and inspected how real discovery domains load onto them.',
    'Determine whether rank 3 is one useful direction plus two passengers, or whether multiple components contribute; then look for domain-specific component structure.',
    'Left: each component alone, with the number of positive test domains. Middle: every combination of the top three. Right: signed discovery-domain loadings; color sign is arbitrary, while relative patterns are meaningful.',
    'Components 1, 2, and 3 remove 0.039, 0.047, and 0.090 alone; all three remove 0.180. Component 7 alone removes 0.117. Loadings differ strongly by discovery domain. Components have functional signatures, but stable semantic names require resampling and targeted factor datasets.',
    'MULTIPLE COMPONENTS MATTER; SEMANTICS UNRESOLVED'
)

rank15 = next(i for i,row in enumerate(paper.FIGURES) if row[0].startswith('Figure 15'))
paper.FIGURES.insert(rank15, tradeoff)
paper.FIGURES.append(components)
paper.build()
