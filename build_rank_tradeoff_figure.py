"""Plot behavioral fraction removed against shuffled-label specificity."""
from pathlib import Path
import csv
from statistics import mean
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
ASSETS = ROOT / 'report_assets'

def read(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))

baseline_rows = read(RESULTS / 'qwen35_4b_fixed_direction_prospective' / 'ablation_by_domain.csv')
baseline = mean(float(row['baseline_gap']) for row in baseline_rows)
summary = [row for row in read(
    RESULTS / 'qwen35_4b_multidimensional_subspace' / 'rank_summary.csv'
) if row['test_set'] == 'original']
summary.sort(key=lambda row: int(row['rank']))

ranks = [int(row['rank']) for row in summary]
reductions = [float(row['mean_reduction']) for row in summary]
percent = [100 * value / baseline for value in reductions]
shuffled = [int(row['shuffled_ge_learned']) for row in summary]
colors = ['#2878B5' if rank != 3 else '#2A9D8F' for rank in ranks]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
axes[0].bar(ranks, percent, color=colors)
axes[0].set_title('How much natural audience behavior is removed?', loc='left',
                  fontsize=12, fontweight='bold', color='#183153')
axes[0].set_xlabel('subspace rank')
axes[0].set_ylabel('original audience gap removed (%)')
axes[0].set_xticks(ranks)
axes[0].set_ylim(0, 52)
for rank, pct, raw in zip(ranks, percent, reductions):
    axes[0].text(rank, pct + 1.2, f'{pct:.1f}%\n({raw:.3f})',
                 ha='center', va='bottom', fontsize=7.5)

axes[1].bar(ranks, shuffled, color=colors)
axes[1].axhline(5, color='#C44536', linestyle='--', linewidth=1.5,
                label='predeclared maximum: 5/100')
axes[1].set_title('How often does a nonsense-label subspace do as well?', loc='left',
                  fontsize=12, fontweight='bold', color='#183153')
axes[1].set_xlabel('subspace rank')
axes[1].set_ylabel('shuffled controls ≥ learned effect (out of 100)')
axes[1].set_xticks(ranks)
axes[1].set_ylim(0, 32)
for rank, count in zip(ranks, shuffled):
    axes[1].text(rank, count + .7, str(count), ha='center', fontsize=8)
axes[1].legend(frameon=False, fontsize=8)

for ax in axes:
    ax.grid(axis='y', alpha=.18)
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle(
    f'Rank 3 is smaller in raw effect than rank 7, but more specific  '
    f'(baseline gap = {baseline:.3f})',
    fontsize=14, fontweight='bold', color='#183153', y=1.02)
fig.tight_layout()
ASSETS.mkdir(exist_ok=True)
out = ASSETS / 'multidim_rank_tradeoff.png'
fig.savefig(out, dpi=190, bbox_inches='tight', facecolor='white')
print(out)
