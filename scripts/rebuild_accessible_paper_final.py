"""Final accessible-paper rebuild with rank and component interpretation."""
import build_accessible_paper as paper

paper.FIGURES.insert(
    next(i for i,row in enumerate(paper.FIGURES) if row[0].startswith('Figure 15')),
    (
        'Figure 14b — Raw effect and specificity are different',
        'multidim_rank_tradeoff.png',
        'A larger raw gap reduction is not automatically stronger evidence. Higher-rank ablation removes more activation axes and can capture both the intended audience feature and unrelated structure.',
        'Express each reduction as a fraction of the original behavior, then ask how often the same-rank pipeline reproduces it after the semantic labels are destroyed.',
        'A useful learned rank should remove a substantial fraction of the gap and be rarely matched by shuffled-label subspaces of the same rank.',
        'The left panel divides reduction by the original mean gap of 0.59375. The right counts shuffled-label controls whose reduction is at least as large. Lower counts mean stronger evidence that the labels mattered.',
        'Rank 3 removes 30.3% and has corrected Monte Carlo p = 0.030. Rank 7 removes 44.7% but has p = 0.059. Rank 8 removes 44.1% and has p = 0.079. Rank 3 has the clearest specificity.',
        'IMPORTANT TRADE-OFF'
    )
)

paper.FIGURES.append((
    'Figure 17 — Can the individual directions be interpreted?',
    'component_interpretation.png',
    'A subspace can have a reliable causal effect even when its coordinate axes do not correspond to simple human concepts. We ablated components separately and in combinations, then inspected discovery-domain loadings.',
    'Determine whether rank 3 is one useful direction plus two passengers, or whether multiple components contribute; then look for domain-specific structure.',
    'If several components matter, their individual and combined removals should reduce the held-out gap. Loading patterns can suggest hypotheses but cannot establish semantic meanings by themselves.',
    'Left: each component alone and positive-domain count. Middle: top-three combinations. Right: signed domain loadings; sign is arbitrary, while relative patterns are meaningful.',
    'Components 1, 2, and 3 remove 0.039, 0.047, and 0.090 alone; together they remove 0.180. Component 7 alone removes 0.117. Multiple components matter, but stable semantic names require resampling and targeted factor datasets.',
    'FUNCTIONS DIFFER; SEMANTICS UNRESOLVED'
))

paper.build()
