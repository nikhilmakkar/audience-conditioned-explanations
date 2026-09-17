"""Build a compact PDF addendum for the relevance-subspace experiments."""
from pathlib import Path
import csv
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, PageBreak, Table, TableStyle, Spacer

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/"results/qwen35_4b_multidimensional_subspace"
A=ROOT / "report/report_assets"; A.mkdir(exist_ok=True)
OUT=ROOT/"report"/"MULTIDIMENSIONAL_SUBSPACE_ADDENDUM.pdf"
NAVY="#183153";BLUE="#2878B5";TEAL="#2A9D8F";ORANGE="#E76F51";GREY="#667085"

def rows(name):
    with (R/name).open(newline="") as h:return list(csv.DictReader(h))
def val(r,k):return float(r[k])
def save(fig,name):
    p=A/name;fig.savefig(p,dpi=190,bbox_inches="tight",facecolor="white");plt.close(fig);return p
def style(ax,title,x,y):
    ax.set_title(title,loc="left",fontweight="bold",color=NAVY);ax.set_xlabel(x);ax.set_ylabel(y)
    ax.grid(axis="y",alpha=.18);ax.spines[["top","right"]].set_visible(False)

def rank_plot():
    summary=rows("rank_summary.csv");controls=rows("control_ablation_by_domain.csv")
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for test,color,label in [("original",BLUE,"Original profiles"),("new_carrier",TEAL,"New profile wording")]:
        z=sorted((r for r in summary if r["test_set"]==test),key=lambda r:int(r["rank"]))
        axes[0].plot([int(r["rank"]) for r in z],[val(r,"mean_reduction") for r in z],marker="o",lw=2,color=color,label=label)
    for kind,color,label in [("random",GREY,"Random 95th percentile"),("shuffled_labels",ORANGE,"Shuffled 95th percentile")]:
        curve=[]
        for rank in range(1,9):
            byseed={int(r["seed"]):val(r,"aggregate_mean") for r in controls if r["kind"]==kind and int(r["rank"])==rank}
            curve.append(np.percentile(list(byseed.values()),95))
        axes[0].plot(range(1,9),curve,ls="--",color=color,label=label)
    axes[0].axvline(3,color=NAVY,alpha=.25);axes[0].set_xticks(range(1,9));style(axes[0],"Behavioral mediation by rank","components removed","gap reduction");axes[0].legend(frameon=False,fontsize=8)
    spectrum=rows("spectrum.csv")
    axes[1].bar(range(1,9),[100*val(r,"energy_fraction") for r in spectrum],color=BLUE,alpha=.8)
    axes[1].plot(range(1,9),[100*val(r,"cumulative_energy") for r in spectrum],color=ORANGE,marker="o",label="cumulative")
    axes[1].set_xticks(range(1,9));axes[1].set_ylim(0,105);style(axes[1],"Domain contrasts are not rank-1","SVD component","contrast energy (%)");axes[1].legend(frameon=False)
    fig.tight_layout();return save(fig,"multidim_rank.png")

def damage_plot():
    d=rows("coherence_by_rank.csv");fig,ax=plt.subplots(figsize=(9.5,4.5));ranks=list(range(1,9))
    learned=[];means=[];los=[];his=[]
    for rank in ranks:
        learned.append(val(next(r for r in d if r["kind"]=="learned" and int(r["rank"])==rank),"mean_next_token_kl"))
        z=[val(r,"mean_next_token_kl") for r in d if r["kind"]=="random" and int(r["rank"])==rank]
        means.append(mean(z));los.append(min(z));his.append(max(z))
    ax.fill_between(ranks,los,his,color=GREY,alpha=.18,label="10 random subspaces: range")
    ax.plot(ranks,means,color=GREY,ls="--",marker="o",label="random mean")
    ax.plot(ranks,learned,color=ORANGE,lw=2.2,marker="o",label="learned")
    ax.axvline(3,color=NAVY,alpha=.25);ax.set_xticks(ranks);style(ax,"Ordinary-text change remains small but is not zero","components removed","mean next-token KL");ax.legend(frameon=False);fig.tight_layout()
    return save(fig,"multidim_damage.png")

def layer_plot():
    d=rows("layer_geometry_summary.csv");layers=[int(r["layer"]) for r in d]
    fig,ax=plt.subplots(figsize=(10,4.5))
    ax.plot(layers,[val(r,"rank1_abs_cosine_to_selected") for r in d],color=BLUE,lw=2,marker="o",ms=3,label="rank-1 to layer 17")
    ax.plot(layers,[val(r,"rank3_mean_principal_cosine_to_selected") for r in d],color=TEAL,lw=2,label="rank-3 to layer 17")
    ax.plot(layers,[float(r["rank1_abs_cosine_to_previous"]) for r in d],color=ORANGE,ls="--",label="rank-1 to previous layer")
    ax.axvline(17,color=NAVY,ls=":",label="selected layer");ax.set_ylim(0,1.05);ax.set_xticks(range(0,32,2));style(ax,"Audience geometry consolidates near layer 17","residual layer","absolute cosine similarity");ax.legend(frameon=False,ncol=2,fontsize=8);fig.tight_layout()
    return save(fig,"multidim_layers.png")

def footer(canvas,doc):
    canvas.saveState();canvas.setFont("Helvetica",8);canvas.setFillColor(colors.HexColor(GREY));canvas.drawRightString(19.2*cm,.8*cm,f"Addendum · {doc.page}");canvas.restoreState()

def build():
    pics=[rank_plot(),damage_plot(),layer_plot()]
    ss=getSampleStyleSheet();ss.add(ParagraphStyle(name="TitleX",fontName="Helvetica-Bold",fontSize=25,leading=30,textColor=colors.HexColor(NAVY),spaceAfter=12));ss.add(ParagraphStyle(name="H1X",fontName="Helvetica-Bold",fontSize=18,leading=22,textColor=colors.HexColor(NAVY),spaceAfter=10));ss.add(ParagraphStyle(name="BX",fontSize=10,leading=15,textColor=colors.HexColor("#27364A"),spaceAfter=8));ss.add(ParagraphStyle(name="CallX",fontName="Helvetica-Bold",fontSize=10.5,leading=16,textColor=colors.HexColor(NAVY),backColor=colors.HexColor("#EAF2FB"),borderPadding=9,spaceAfter=10))
    doc=SimpleDocTemplate(str(OUT),pagesize=A4,leftMargin=1.7*cm,rightMargin=1.7*cm,topMargin=1.6*cm,bottomMargin=1.4*cm,title="Multidimensional audience-relevance subspace",author="Nikhil Makkar")
    S=[];P=lambda t,s="BX":S.append(Paragraph(t,ss[s]))
    S.append(Spacer(1,1.2*cm));P("Beyond one direction","TitleX");P("A falsification-driven extension of the audience-conditioned explanation-depth study")
    P("Question: is the partial single-vector ablation weak because the representation is not rank-1? We extracted eight domain-specific expert-minus-other-expert contrast vectors at Qwen3.5-4B residual layer 17, applied uncentered SVD, and ablated nested subspaces of ranks 1–8.","CallX")
    P("The analysis was frozen before examining outcomes. Each rank was compared with 100 same-rank random orthonormal subspaces and 100 subspaces learned after within-domain label shuffling. The intervention was also transferred to entirely new reader-profile wording. Ordinary-text damage was measured separately on WikiText.")
    P("Why uncentered SVD? Centering the eight contrasts would delete their common expert-minus-other direction—the exact shared feature under investigation. Uncentered SVD retains that mean-like component and asks whether structured residual components add causal leverage.")
    S.append(PageBreak());P("1. Rank 3 is the clean specificity result","H1X");S.append(Image(str(pics[0]),width=17.2*cm,height=7.0*cm))
    data=[["Rank","Original reduction","New carriers","Domains +","Random ≥","Shuffled ≥"],["1","0.039","0.086","5/8","0/100","20/100"],["2","0.086","0.141","8/8","0/100","21/100"],["3","0.180","0.238","8/8","0/100","2/100"],["7","0.266","0.359","8/8","0/100","5/100"],["8","0.262","0.387","8/8","0/100","7/100"]]
    t=Table(data,colWidths=[1.5*cm,2.7*cm,2.5*cm,2.0*cm,2.1*cm,2.3*cm]);t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor(NAVY)),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.3,colors.lightgrey),("ALIGN",(1,1),(-1,-1),"CENTER"),("FONTSIZE",(0,0),(-1,-1),8)]));S.append(t)
    P("Rank 3 is the strongest predeclared specificity result: its effect is 4.6 times the rank-1 effect, all eight source domains move in the predicted direction (exact one-sided p = 0.0039), no random space matches it, and only 2/100 shuffled-label spaces match it. It also transfers to new carrier wording. Rank 7 is larger but only just meets the shuffled threshold; rank 8 fails it.","CallX")
    S.append(PageBreak());P("2. The effect is targeted, not damage-free","H1X");S.append(Image(str(pics[1]),width=17.2*cm,height=8.1*cm));P("At rank 3, WikiText next-token KL is 0.00248, top-1 predictions agree 96.75% of the time, and NLL rises by 0.00656 from a 3.058 baseline (about 0.21%). A typical random rank-3 space changes the distribution less (mean KL 0.00110). Thus the learned subspace has a large targeted behavioral effect with modest but detectable general-language impact. It should not be described as damage-free.")
    P("The behavior result is nevertheless not explained merely by rank: every one of the 100 random subspaces has a smaller relevance-gap reduction at rank 3. A useful follow-up would compare learned and control interventions at matched KL, not only matched dimensionality.")
    S.append(PageBreak());P("3. Where the geometry appears","H1X");S.append(Image(str(pics[2]),width=17.2*cm,height=7.7*cm));P("Early-layer directions are almost orthogonal to the selected layer-17 direction. Similarity starts rising at layers 15–16, changes sharply at layer 17, and then remains substantial at layer 18 before evolving smoothly downstream; adjacent-layer cosine is mostly 0.84–0.95 in layers 20–30. The rank-3 principal-angle curve tells the same broad story.")
    P("This is consistent with the audience-relevance representation consolidating in the middle of the network and then being carried forward. It is descriptive evidence, not localization of a head, MLP, or complete circuit. The sharp change at layer 17 is especially worth checking with neighboring-layer rank-3 ablations.","CallX")
    S.append(PageBreak());P("4. What can be claimed—and what cannot","H1X")
    P("Strongest current claim","CallX");P("In Qwen3.5-4B, the forced-choice effect of reader-domain relevance is mediated more completely by a small learned residual-stream subspace than by its leading direction alone. A rank-3 subspace generalizes across eight unfamiliar mechanisms and new profile wording, and outperforms rank-matched random and shuffled-label controls while causing modest ordinary-text change.")
    P("Do not claim that ‘expertise is three-dimensional.’ The stimuli isolate audience-conditioned relevance to technical-summary choice, not expertise as a general cognitive variable. Do not call rank 3 the true dimensionality: rank 7 produces a larger intervention but is less specific, and only eight domain contrasts bound the learned rank at eight.")
    P("Best next experiments: (1) neighboring-layer rank-3 causal sweep; (2) damage-matched controls; (3) repeat subspace extraction under bootstrap resampling of profiles; (4) blinded human scoring of free generations. Cross-model cosine should come later and requires a held-out representation alignment such as orthogonal Procrustes or CCA; raw cosine across separately trained models is not meaningful.")
    P("Artifacts: MULTIDIMENSIONAL_SUBSPACE_PROTOCOL.md; multidimensional_subspace.py; multidimensional_coherence.py; layer_subspace_geometry.py; and results/qwen35_4b_multidimensional_subspace/.")
    doc.build(S,onFirstPage=footer,onLaterPages=footer)
    print(OUT)

if __name__=="__main__":build()
