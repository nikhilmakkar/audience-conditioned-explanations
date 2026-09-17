"""Add the rank-versus-specificity explainer to the accessible paper."""
import build_accessible_paper as paper

tradeoff = (
    'Figure 14b — Why rank 3 is cleaner than rank 7',
    'multidim_rank_tradeoff.png',
    'A larger raw reduction is not automatically stronger evidence. Higher-rank interventions remove more activation space, giving both correctly learned and nonsense-label subspaces more opportunities to affect behavior.',
    'Show the effect in understandable units and compare it directly with the difficulty of reproducing that effect after destroying the discovery labels.',
    'The left panel expresses reduction as a percentage of the original 0.594 audience gap. The right counts how many of 100 shuffled-label subspaces do at least as well; lower is more specific, and at most five was allowed.',
    'Rank 3 removes 30.3% of the natural gap and only 2/100 shuffled controls match it. Rank 7 removes more, 44.7%, but 5/100 match it. Rank 8 removes 44.1%, while 7/100 match it and therefore fails the frozen specificity rule.',
    'Rank 3 is the cleanest evidence; rank 7 has the largest passing raw effect.',
    'IMPORTANT TRADE-OFF',
)

index = next(i for i, figure in enumerate(paper.FIGURES)
             if figure[0].startswith('Figure 15'))
paper.FIGURES.insert(index, tradeoff)
paper.OUT = paper.ROOT / 'report' / 'AUDIENCE_STEERING_ACCESSIBLE_PAPER_V2.pdf'

if __name__ == '__main__':
    paper.build()
