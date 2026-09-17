"""Build a blinded side-by-side PDF for baseline/rank-3/rank-7 generations."""
from pathlib import Path
import csv, random
from collections import defaultdict
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Table, TableStyle, Spacer

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'results/qwen35_4b_component_interpretation/qualitative_generation_rows.csv'
OUT=ROOT/'report'/'BLIND_RANK3_RANK7_QUALITATIVE_REVIEW.pdf'
with SRC.open(newline='') as handle:
    rows=list(csv.DictReader(handle))
grouped=defaultdict(list)
for row in rows: grouped[int(row['prompt_index'])].append(row)

styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleX',fontName='Helvetica-Bold',fontSize=18,leading=22,textColor=colors.HexColor('#183153'),spaceAfter=8))
styles.add(ParagraphStyle(name='BodyX',fontSize=8.5,leading=12,textColor=colors.HexColor('#27364A')))
styles.add(ParagraphStyle(name='SmallX',fontSize=7.5,leading=10,textColor=colors.HexColor('#667085')))
doc=SimpleDocTemplate(str(OUT),pagesize=A4,leftMargin=1.5*cm,rightMargin=1.5*cm,topMargin=1.4*cm,bottomMargin=1.4*cm,title='Blinded rank-3/rank-7 qualitative review',author='Nikhil Makkar')
story=[]
for index in sorted(grouped):
    prompt=grouped[index][0]['prompt']
    story.append(Paragraph(f'Prompt {index+1}: {prompt}',styles['TitleX']))
    story.append(Paragraph('Three answers are baseline, rank-3 ablation, and rank-7 ablation in a hidden order. Score before opening blind_qualitative_key.csv.',styles['SmallX']))
    story.append(Spacer(1,6))
    candidates=list(grouped[index])
    random.Random(20260908+index).shuffle(candidates)
    for letter,row in zip('ABC',candidates):
        data=[[Paragraph(f'Answer {letter} · code {row["blind_code"]}',styles['BodyX'])],
              [Paragraph(row['text'].replace('\n','<br/>'),styles['BodyX'])],
              [Paragraph('Coherence __/5   Correctness __/5   Instruction following __/5   Obvious corruption 0/1   Notes: ____________________',styles['SmallX'])]]
        table=Table(data,colWidths=[17.8*cm])
        table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EAF2FB')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#B8C4D1')),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        story.append(table);story.append(Spacer(1,7))
    if index < max(grouped): story.append(PageBreak())
doc.build(story)
print(OUT)
