"""Build a concise correction to the multidimensional interpretation."""
from pathlib import Path
import csv
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, Table, TableStyle

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/"results/qwen35_4b_intrinsic_rank_minimal/evaluation_by_domain.csv"
rows=list(csv.DictReader(R.open()))
sets=["real_test","synthetic","new_carriers"]
set_labels=["Held-out real profiles","Invented methods","Rewritten reader wording"]
methods=["svd_component_1","selected_1d_mixture","rank3_subspace"]
labels=["Leading SVD\ncomponent","Optimized 1D\nmixture","Rank-3\nsubspace"]
values=[[float(next(r["aggregate_mean"] for r in rows if r["test_set"]==s and r["intervention"]==m))
         for m in methods] for s in sets]
fig,ax=plt.subplots(figsize=(9,4.6));x=range(3);w=.24
for j,(s,label,color) in enumerate(zip(sets,set_labels,["#2878B5","#2A9D8F","#E76F51"])):
    ax.bar([i+(j-1)*w for i in x],values[j],width=w,label=label,color=color)
ax.set_xticks(list(x),labels);ax.set_ylabel("natural audience-gap reduction")
ax.set_title("An optimized single direction nearly matches rank 3",loc="left",fontweight="bold",color="#183153")
ax.grid(axis="y",alpha=.18);ax.spines[["top","right"]].set_visible(False);ax.legend(frameon=False)
fig.tight_layout();img=ROOT / "report/report_assets/intrinsic_rank_correction.png"
fig.savefig(img,dpi=190,bbox_inches="tight",facecolor="white")

out=ROOT/"report"/"INTRINSIC_RANK_CORRECTION.pdf";ss=getSampleStyleSheet()
ss.add(ParagraphStyle(name="T",fontName="Helvetica-Bold",fontSize=22,leading=27,textColor=colors.HexColor("#183153"),spaceAfter=10))
ss.add(ParagraphStyle(name="B",fontSize=10,leading=15,textColor=colors.HexColor("#27364A"),spaceAfter=8))
ss.add(ParagraphStyle(name="C",fontName="Helvetica-Bold",fontSize=10.5,leading=16,textColor=colors.HexColor("#183153"),backColor=colors.HexColor("#EAF2FB"),borderPadding=9,spaceAfter=10))
doc=SimpleDocTemplate(str(out),pagesize=A4,leftMargin=1.7*cm,rightMargin=1.7*cm,topMargin=1.6*cm,bottomMargin=1.6*cm,title="Intrinsic-rank correction",author="Nikhil Makkar")
story=[Paragraph("Correction: rank 3 is not shown to be necessary",ss["T"]),
Paragraph("The earlier comparison showed that ablating the first three SVD components was stronger than ablating the leading component. But the leading component maximizes activation variance, not causal effect. We therefore selected a one-dimensional mixture using only real-domain validation profiles, froze it, and evaluated it on three untouched sets.",ss["B"]),
Image(str(img),width=16.8*cm,height=8.1*cm),Spacer(1,6)]
data=[["Evaluation set","Leading SVD","Optimized 1D","Rank 3"],
      ["Real test","0.109 (7/8)","0.168 (8/8)","0.117 (7/8)"],
      ["Synthetic","0.039 (5/8)","0.156 (8/8)","0.180 (8/8)"],
      ["Rewritten reader wording","0.086 (7/8)","0.234 (8/8)","0.238 (8/8)"]]
t=Table(data,colWidths=[4.3*cm]*4);t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#183153")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.4,colors.lightgrey),("ALIGN",(1,1),(-1,-1),"CENTER"),("FONTSIZE",(0,0),(-1,-1),8.5),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]));story.extend([t,Spacer(1,10)])
story.extend([Paragraph("Corrected conclusion",ss["C"]),
Paragraph("The rank-3 SVD subspace contains a substantially better causal direction than its variance-leading component. Current evidence does not show that three dimensions are intrinsically required: one validation-selected mixture nearly reproduces the rank-3 effect and transfers in 8/8 domains on all three evaluation sets.",ss["B"]),
Paragraph("The selected mixture is -0.844 C1 - 0.177 C2 + 0.507 C3 (overall sign is irrelevant for ablation). Because it was selected from 259 candidates, it now requires a selection-matched shuffled-label control before claiming that this optimized extraction procedure is uniquely specific. The previous rank-3 result remains evidence about a useful subspace, but not evidence for three distinct semantic mechanisms.",ss["B"])])
doc.build(story);print(out)
