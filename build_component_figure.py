"""Visualize component-wise causal effects and discovery-domain loadings."""
from pathlib import Path
import csv
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
R = ROOT / 'results/qwen35_4b_component_interpretation'
A = ROOT / 'report_assets'

def rows(name):
    with (R/name).open(newline='') as handle:
        return list(csv.DictReader(handle))

ablation = rows('component_subset_ablation.csv')
loadings = rows('discovery_component_loadings.csv')
individual = []
positive = []
for component in range(1, 9):
    values = [float(row['gap_reduction']) for row in ablation
              if row['components'] == str(component)]
    individual.append(mean(values))
    positive.append(sum(value > 0 for value in values))

subsets = ['1', '2', '3', '1+2', '1+3', '2+3', '1+2+3']
subset_means = [float(next(row['aggregate_mean'] for row in ablation
                           if row['components'] == subset)) for subset in subsets]
sources = sorted(set(row['source'] for row in loadings))
matrix = np.array([[float(next(row['loading_over_contrast_norm'] for row in loadings
                               if row['source'] == source and int(row['component']) == component))
                    for component in range(1,9)] for source in sources])

fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), gridspec_kw={'width_ratios':[1,1.1,1.65]})
colors = ['#2A9D8F' if i < 3 else '#9DB7CE' for i in range(8)]
axes[0].bar(range(1,9), individual, color=colors)
axes[0].axhline(0,color='black',linewidth=.8)
axes[0].set_xticks(range(1,9)); axes[0].set_xlabel('individual SVD component')
axes[0].set_ylabel('gap reduction')
axes[0].set_title('Each component alone',loc='left',fontweight='bold',color='#183153')
for x,(value,count) in enumerate(zip(individual,positive),1):
    axes[0].text(x,value+(.006 if value>=0 else -.012),f'{count}/8',
                 ha='center',va='bottom' if value>=0 else 'top',fontsize=7)

axes[1].bar(range(len(subsets)), subset_means, color='#2878B5')
axes[1].set_xticks(range(len(subsets)),subsets,rotation=35,ha='right')
axes[1].set_xlabel('components removed together');axes[1].set_ylabel('gap reduction')
axes[1].set_title('Top-three combinations',loc='left',fontweight='bold',color='#183153')
for x,value in enumerate(subset_means):
    axes[1].text(x,value+.005,f'{value:.3f}',ha='center',fontsize=7)

im=axes[2].imshow(matrix,cmap='RdBu_r',vmin=-.9,vmax=.9,aspect='auto')
axes[2].set_xticks(range(8),[f'C{i}' for i in range(1,9)])
axes[2].set_yticks(range(len(sources)),sources)
axes[2].set_title('Domain loading patterns',loc='left',fontweight='bold',color='#183153')
axes[2].set_xlabel('SVD component')
fig.colorbar(im,ax=axes[2],fraction=.046,pad=.04,label='signed loading / contrast norm')
for ax in axes[:2]:
    ax.grid(axis='y',alpha=.18);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Components have causal signatures, but not yet semantic names',
             fontsize=14,fontweight='bold',color='#183153',y=1.03)
fig.tight_layout()
A.mkdir(exist_ok=True)
out=A/'component_interpretation.png'
fig.savefig(out,dpi=190,bbox_inches='tight',facecolor='white')
print(out)
