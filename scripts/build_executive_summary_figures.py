import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; R=ROOT/"results"; OUT=ROOT / "report/report_assets"
NAVY="#17365D"; BLUE="#176B87"; TEAL="#4F9DA6"; ORANGE="#D97706"
RED="#B45353"; GREY="#6B7280"; LIGHT="#E5E7EB"; PALE="#A8DADC"

def read(p):
    with p.open(newline="") as f: return list(csv.DictReader(f))
def clean(ax):
    ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",color=LIGHT,lw=.8,zorder=0)
def save(fig,name):
    fig.savefig(OUT/f"{name}.png",dpi=300,bbox_inches="tight",facecolor="white")
    fig.savefig(OUT/f"{name}.pdf",bbox_inches="tight",facecolor="white"); plt.close(fig)

def fig1():
    rows=read(R/"qwen35_4b_fixed_direction_prospective/addition_by_domain.csv")
    proj=read(R/"qwen35_4b_projection_transfer/projection_summary.csv")
    enc=["A/B","X/Y","1/2"]; dom=list(dict.fromkeys(r["source"] for r in rows))
    field_name={
        "anchorshift":"Causal inference", "driftgate":"Control", "fluxpatch":"Fluid dynamics",
        "motifbridge":"Comp. biology", "phasenest":"Audio", "phasesar":"SAR",
        "quorumweave":"Cryptography", "shardbloom":"Databases",
    }
    names=[field_name[d] for d in dom]
    v={(r["encoding"],r["source"]):float(r["addition_effect"]) for r in rows}
    fig,a=plt.subplots(1,2,figsize=(10.8,4.4),gridspec_kw={"width_ratios":[1.55,.85]})
    fig.suptitle("A frozen direction transferred beyond its training topics",x=.075,y=1.01,ha="left",fontsize=16,weight="bold")
    x=np.arange(8)
    for e,off,c,m in zip(enc,[-.22,0,.22],[BLUE,TEAL,NAVY],["o","s","^"]):
        y=np.array([v[e,d] for d in dom]); a[0].scatter(x+off,y,s=43,color=c,marker=m,edgecolor="white",lw=.6,label=e,zorder=3)
        a[0].hlines(y.mean(),-.45,7.45,color=c,lw=1.2,alpha=.32)
    a[0].axhline(0,color=GREY,lw=1); a[0].set(xticks=x,xticklabels=names,ylim=(-.02,.35),ylabel="Increase in technical-summary preference")
    a[0].tick_params(axis="x",rotation=28,labelsize=8.8)
    for t in a[0].get_xticklabels(): t.set_ha("right")
    a[0].set_title("A. Positive in every invented domain and encoding",loc="left")
    a[0].legend(title="Answer labels",frameon=False,ncol=3,loc="upper right")
    a[0].text(.01,.98,"24/24 domain × encoding tests positive",transform=a[0].transAxes,va="top",fontsize=9.5,weight="bold",c=NAVY); clean(a[0])
    auc=[float(r["auc"]) for r in proj]; bars=a[1].bar(range(2),auc,color=[TEAL,NAVY],width=.58,zorder=2)
    a[1].axhline(.5,color=GREY,ls="--",lw=1.1)
    for b,val in zip(bars,auc): a[1].text(b.get_x()+b.get_width()/2,val+.018,f"AUC {val:.2f}",ha="center",weight="bold")
    a[1].set(xticks=range(2),xticklabels=["Invented\nmethods","Rewritten\nprofiles"],ylim=(0,1),ylabel="Relevant-vs-unrelated expert AUC")
    a[1].set_title("B. Readable in unmodified activations",loc="left"); a[1].text(1.06,.5,"chance",c=GREY,va="center",fontsize=9); clean(a[1])
    fig.subplots_adjust(left=.08,right=.98,top=.80,bottom=.24,wspace=.33); save(fig,"exec_figure_1_generalization")

def fig2():
    rows=read(R/"qwen35_4b_intrinsic_rank_minimal/evaluation_by_domain.csv")
    ivs=["svd_component_1","selected_1d_mixture","rank3_subspace"]; labels=["Variance-leading 1D","Selected 1D (exploratory)","Rank-3 subspace"]; colors=[ORANGE,TEAL,NAVY]
    fig,a=plt.subplots(1,2,figsize=(10.8,4.45)); fig.suptitle("Steering and mediation came apart",x=.075,y=1.01,ha="left",fontsize=16,weight="bold")
    reductions=np.array([.047,.15625,.1796875]); pct=100*reductions/.594; bars=a[0].bar(range(3),pct,color=colors,width=.62,zorder=2)
    for b,d,p in zip(bars,reductions,pct):
        a[0].text(b.get_x()+b.get_width()/2,p+1.2,f"{p:.0f}%",ha="center",weight="bold")
        a[0].text(b.get_x()+b.get_width()/2,max(1,p*.48),f"{d:.3f}",ha="center",c="white" if p>13 else NAVY,fontsize=9.3,weight="bold")
    a[0].set(xticks=range(3),xticklabels=["Original mean-\ndifference direction","Selected 1D\n(exploratory)","Rank-3\nsubspace"],ylim=(0,35),ylabel="Unmodified audience gap removed")
    a[0].set_title("A. The original direction accounted for only 8%",loc="left"); a[0].text(.02,.98,"Unmodified gap = 0.594",transform=a[0].transAxes,va="top",fontsize=9,c=GREY); clean(a[0])
    sets=["real_test","synthetic","new_carriers"]; x=np.arange(3); w=.23
    for j,(iv,label,c) in enumerate(zip(ivs,labels,colors)):
        means=[]; ses=[]
        for s in sets:
            z=np.array([float(r["gap_reduction"]) for r in rows if r["test_set"]==s and r["intervention"]==iv]); means.append(z.mean()); ses.append(z.std(ddof=1)/np.sqrt(len(z)))
        a[1].bar(x+(j-1)*w,means,w,yerr=ses,capsize=2.5,color=c,label=label,zorder=2)
    a[1].axhline(0,color=GREY,lw=.9); a[1].set(xticks=x,xticklabels=["Held-out real\nprofiles","Invented\nmethods","Rewritten\nprofiles"],ylim=(-.01,.29),ylabel="Audience-gap reduction")
    a[1].set_title("B. A selected 1D mixture nearly matched rank 3",loc="left"); a[1].legend(frameon=False,fontsize=8.2,loc="upper left"); clean(a[1])
    fig.subplots_adjust(left=.08,right=.98,top=.80,bottom=.22,wspace=.31); save(fig,"exec_figure_2_mediation")

def controls(rows,kind):
    x={int(r["seed"]):float(r["aggregate_mean"]) for r in rows if r["kind"]==kind}; return np.array([x[k] for k in sorted(x)])
def fig3():
    cr=read(R/"qwen35_4b_fixed_direction_prospective/controls.csv"); gen=read(R/"qwen35_4b_fixed_direction_generation/generation_summary.csv")
    rand=controls(cr,"random"); shuf=controls(cr,"permuted_labels"); learned=.1562500019
    term=next(r for r in gen if r["metric"]=="term_coverage" and r["condition"]=="pooled"); mean=float(term["contrast_mean"]); se=float(term["contrast_se"])
    fig,a=plt.subplots(1,2,figsize=(10.8,4.45),gridspec_kw={"width_ratios":[1.35,.85]}); fig.suptitle("Where the result stopped holding up",x=.075,y=1.01,ha="left",fontsize=16,weight="bold")
    bins=np.linspace(min(rand.min(),shuf.min())-.01,max(shuf.max(),learned)+.02,22)
    a[0].hist(rand,bins=bins,color=PALE,alpha=.88,label="Random directions",zorder=2); a[0].hist(shuf,bins=bins,histtype="step",color=RED,lw=2,label="Shuffled audience labels",zorder=3); a[0].axvline(learned,color=NAVY,lw=2.3,label="Learned direction")
    a[0].set(xlabel="Mean steering effect across 8 invented domains",ylabel="Control directions"); a[0].set_title("A. Six shuffled-label directions matched or exceeded learned",loc="left"); a[0].legend(frameon=False,fontsize=8.8,loc="upper left")
    a[0].text(.98,.95,f"Random: {(rand>=learned).sum()}/100 ≥ learned\nShuffled: {(shuf>=learned).sum()}/100 ≥ learned",transform=a[0].transAxes,ha="right",va="top",fontsize=9.5,weight="bold"); clean(a[0])
    a[1].errorbar([mean],[0],xerr=[1.96*se],fmt="o",color=ORANGE,ecolor=ORANGE,capsize=5,ms=8,lw=2); a[1].axvline(0,c=GREY,ls="--",lw=1.1)
    a[1].set(xlim=(-.01,.032),ylim=(-.8,.8),yticks=[],xlabel="Technical-term coverage change (mean ± 95% CI)"); a[1].set_title("B. No reliable free-generation effect",loc="left")
    a[1].text(.5,.25,f"change = {mean:.3f}\np = {float(term['contrast_exact_one_sided_p']):.2f}",transform=a[1].transAxes,ha="center",fontsize=12,weight="bold",c=ORANGE)
    fig.subplots_adjust(left=.08,right=.98,top=.80,bottom=.22,wspace=.35); save(fig,"exec_figure_3_limits")

OUT.mkdir(exist_ok=True); plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.titlesize":12,"axes.labelsize":10,"axes.titleweight":"bold"})
fig1(); fig2(); fig3()
