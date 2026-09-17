"""Build a paper-style PDF with a plain-language guide beside each figure."""
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, PageBreak,
    PageTemplate, Paragraph, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'report_assets'
OUT = ROOT / 'AUDIENCE_STEERING_ACCESSIBLE_PAPER.pdf'
NAVY = '#183153'
BLUE = '#2878B5'
GREY = '#667085'
LIGHT = '#F2F5F8'

# title, image, context, why, test, reading instructions, verdict, status
FIGURES = [
('Figure 1 — From observation to controlled question','01_question.png',
 'The original observation that models over-simplify papers was too subjective. We narrowed it to a choice between two accurate summaries of the same passage.',
 'Establish exactly what changes. Both readers are experts; only whether their field is relevant changes.',
 'Relevant expertise should increase preference for the technical summary compared with equally sophisticated but unrelated expertise.',
 'This is a design diagram, not a result. Follow the identical passage through the two reader descriptions.',
 'We obtained a measurable question about audience-conditioned summary preference, not expertise or simplification in general.','DESIGN'),
('Figure 2 — Experimental logic','02_pipeline.png',
 'Behavior, a readable activation, and a causal mechanism are different claims. The pipeline tests them separately.',
 'Prevent an activation correlation from being mistaken for a feature that the model actually uses.',
 'Behavior must exist; a feature must generalize; intervention must change behavior; removal should reduce natural behavior; controls should fail.',
 'Read left to right. Each step earns a stronger claim and creates a new way for the idea to fail.',
 'All stages were completed, with strong representation and steering, partial mediation, and failed generation and cross-family tests.','ROADMAP'),
('Figure 3 — Does the behavioral effect exist?','03_behavior.png',
 'Before inspecting activations, we ask whether the model naturally changes its summary choice when the reader field becomes relevant.',
 'There is no internal phenomenon to explain if the output behavior is absent. Matched experts remove the easy expert-versus-novice confound.',
 'Positive bars support the hypothesis: relevant expertise increases the technical-summary margin.',
 'Each bar is relevant-expert preference minus unrelated-expert preference. Zero means no measured audience effect.',
 'The effect is sufficiently consistent in Qwen3.5 to investigate, but varies across models and domains.','PASS, MODEL-SPECIFIC'),
('Figure 4 — Leave-one-domain-out causal transfer','04_lodo.png',
 'A direction learned on one topic might encode topic words rather than the relationship between reader and source.',
 'Test whether the extraction procedure transfers to a domain completely excluded while learning the direction.',
 'Train on seven domains and intervene on the eighth; repeat eight times. Held-out effects should remain positive.',
 'Addition asks whether inserting the feature moves the answer. Ablation asks whether removing it reduces the natural gap.',
 'Addition passes: 8/8 held-out domains, mean +0.186, p=0.0039. Ablation is weaker. Each fold still uses a different vector.','PASS FOR ADDITION'),
('Figure 5 — One frozen direction on unfamiliar methods','05_fixed_domains.png',
 'The previous study refitted a vector for every fold. Here one vector is learned once and applied unchanged to eight invented methods.',
 'Rule out transfer that depends on refitting for each target domain or recalling familiar papers.',
 'The frozen vector should work in all domains and after answer labels change from A/B to X/Y and 1/2.',
 'Bars above zero mean the vector changes technical-summary preference in the predicted direction.',
 'All three encodings produce 8/8 positive domains, with mean effects around 0.16 to 0.18.','PASS'),
('Figure 6 — Could meaningless directions work too?','06_controls.png',
 'Neural networks are sensitive. A learned direction is interesting only if suitable null directions do not work equally well.',
 'Test arbitrary perturbation and whether the pipeline manufactures strong vectors after the semantic labels are destroyed.',
 'At most 5 of 100 random or shuffled-label controls were allowed to equal the learned effect.',
 'The learned effect is compared with two null distributions. Greater separation is better.',
 'Random controls pass at 0/100. Shuffled labels reach 6/100, missing the frozen threshold by one.','MIXED, NEAR-MISS'),
('Figure 7 — Is it a continuous control knob?','07_dose.png',
 'A single plus-versus-minus comparison might exploit an intervention convention rather than reveal a genuine axis.',
 'Test monotonicity and independence from reader condition, answer tokens, and exact profile wording.',
 'Technical preference should rise as alpha moves from -2 to +2 in every condition.',
 'Moving right means stronger steering along the same direction. A rising curve is the predicted dose response.',
 'Every encoding-by-reader cell has positive source-level slopes in 8/8 domains, including rewritten profiles.','PASS'),
('Figure 8 — Is the feature naturally represented?','08_projection.png',
 'Artificial steering can work even if unmodified activations do not use that axis to distinguish readers.',
 'Separate intervention sufficiency from natural linear representation.',
 'Relevant-expert prompts should project higher onto the frozen direction within each source domain.',
 'AUC measures relative ranking; midpoint accuracy asks whether one absolute threshold transfers.',
 'All domain gaps are positive, AUC is 0.81 to 0.84, and 0/100 random or shuffled directions match it. The absolute threshold shifts.','PASS WITH CALIBRATION SHIFT'),
('Figure 9 — Is the one direction naturally necessary?','09_ablation_damage.png',
 'Addition shows that a vector can influence the model. Removal tests whether the model normally relies on it.',
 'Reduce the audience gap without broadly breaking ordinary next-token prediction.',
 'Behavioral bars should be positive; WikiText KL and prediction changes should remain small.',
 'The left panel is gap reduction by source. The right panel measures ordinary-language disturbance.',
 'Layer-17 removal reduces the gap by 0.047 in 7/8 domains, p=0.0078, with tiny damage. It mediates only part of the behavior.','PARTIAL PASS'),
('Figure 10 — Does it change freely written explanations?','10_generation.png',
 'Prepared choices are easy to measure, but the motivating observation concerned explanations generated by the model itself.',
 'Test whether forced-choice control transfers to realistic generation.',
 'Predeclared technical-term coverage should rise consistently with steering strength across domains.',
 'A rising curve supports generation control. A flat curve means this automatic outcome detects no reliable change.',
 'The +2 versus -2 change is only +0.011, 3/8 domains are positive, and p=0.25. Human scoring remains pending.','FAIL ON FROZEN METRIC'),
('Figure 11 — How many discovery domains are needed?','11_learning_curve.png',
 'The final vector could depend on one favorable domain or require all eight carefully chosen domains.',
 'Measure stability across every subset of 1, 2, 4, 6, and 8 domains: 135 learned directions.',
 'Effects should survive many subsets, while geometric similarity should increase with discovery diversity.',
 'Dots are individual subsets. Diamonds summarize each subset size. Effect and cosine answer different questions.',
 'All 70 four-domain subsets yield 8/8 positive test effects. More domains mainly stabilize direction geometry.','PASS'),
('Figure 12 — Does it replicate outside Qwen?','12_cross_family.png',
 'Two Qwen checkpoints are not independent architectural evidence, so Phi-3.5 provides a different-family test.',
 'Test whether the same extraction procedure finds a specific and naturally used feature in another family.',
 'Phi must pass domain consistency, answer re-encoding, random and shuffled specificity, and ablation.',
 'A positive mean alone is insufficient. The control counts show whether the learned vector is special.',
 'X/Y reaches 6/8; 6 random and 18 shuffled controls match the aggregate; ablation is inconclusive.','FAIL'),
('Figure 13 — Evidence ladder','13_evidence_ladder.png',
 'Behavior, representation, causal influence, necessity, generation, and cross-model replication support different claims.',
 'Prevent positive lower-level findings from being summarized as a universal mechanism.',
 'Every rung must be supported by the experiment designed for it; success below cannot replace failure above.',
 'Green is supported, amber is partial or unresolved, and red is failed or pending.',
 'The study establishes Qwen representation and forced-choice control, partial mediation, and clear external-validity limits.','SYNTHESIS'),
('Figure 14 — Is the representation larger than one direction?','multidim_rank.png',
 'Weak one-direction ablation suggested that the natural representation might occupy several activation dimensions.',
 'Test whether a learned rank-k subspace removes more of the gap than rank 1 and beats controls of the same rank.',
 'Extra components must improve over rank 1, transfer to new profile wording, and beat random and shuffled subspaces.',
 'Left: causal reduction versus rank and control thresholds. Right: contrast variation captured by each component.',
 'Rank 3 reduces 0.180 versus 0.039 at rank 1; it has 8/8 domains, 0/100 random and 2/100 shuffled controls matching it.','PASS, MAIN NEW RESULT'),
('Figure 15 — Is rank 3 simply more damaging?','multidim_damage.png',
 'Removing more dimensions changes more of the network. A larger behavioral effect might reflect broad disruption.',
 'Compare ordinary-text changes at every learned rank with same-rank random subspaces.',
 'Targeted behavior should exceed controls while absolute language degradation remains modest.',
 'KL measures movement of the full next-token distribution. Zero means no change; lower is less disturbance.',
 'At rank 3, KL is 0.00248, top-1 agreement is 96.75%, and NLL rises about 0.21%. Damage is modest but not zero.','PARTIAL PASS'),
('Figure 16 — Where does the geometry appear?','multidim_layers.png',
 'The audience feature could exist unchanged throughout the network or consolidate during a processing stage.',
 'Compare independently learned rank-1 directions and rank-3 subspaces across all residual layers.',
 'High similarity everywhere suggests propagation; a localized rise suggests formation or consolidation near that stage.',
 'Blue and green compare layers with layer 17. Orange compares each layer with its predecessor. These are geometric, not causal, results.',
 'Similarity is low through layer 14, rises at 15 to 16, changes sharply at 17, then evolves smoothly. Neighbor-layer ablations are still needed.','INTERESTING, DESCRIPTIVE'),
]

def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#D0D5DD'))
    canvas.line(1.6*cm, 1.25*cm, 19.4*cm, 1.25*cm)
    canvas.setFont('Helvetica', 7.5)
    canvas.setFillColor(colors.HexColor(GREY))
    canvas.drawString(1.6*cm, .85*cm, 'Audience-conditioned explanation depth — accessible paper')
    canvas.drawRightString(19.4*cm, .85*cm, f'Page {doc.page}')
    canvas.restoreState()

def build():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleX', fontName='Helvetica-Bold',
        fontSize=25, leading=30, textColor=colors.HexColor(NAVY), spaceAfter=12))
    styles.add(ParagraphStyle(name='SubX', fontSize=12, leading=18,
        textColor=colors.HexColor(GREY), spaceAfter=12))
    styles.add(ParagraphStyle(name='H1X', fontName='Helvetica-Bold',
        fontSize=18, leading=22, textColor=colors.HexColor(NAVY), spaceAfter=8))
    styles.add(ParagraphStyle(name='H2X', fontName='Helvetica-Bold',
        fontSize=12, leading=15, textColor=colors.HexColor(BLUE), spaceBefore=7, spaceAfter=4))
    styles.add(ParagraphStyle(name='BodyX', fontSize=9.5, leading=14,
        textColor=colors.HexColor('#27364A'), spaceAfter=7))
    styles.add(ParagraphStyle(name='SmallX', fontSize=8, leading=11,
        textColor=colors.HexColor(GREY)))
    styles.add(ParagraphStyle(name='GuideHead', fontName='Helvetica-Bold',
        fontSize=8.2, leading=11, textColor=colors.HexColor(NAVY)))
    styles.add(ParagraphStyle(name='GuideBody', fontSize=8.2, leading=11,
        textColor=colors.HexColor('#27364A')))
    styles.add(ParagraphStyle(name='CallX', fontName='Helvetica-Bold',
        fontSize=10, leading=15, textColor=colors.HexColor(NAVY),
        backColor=colors.HexColor('#EAF2FB'), borderPadding=9, spaceAfter=9))
    doc = BaseDocTemplate(str(OUT), pagesize=A4, leftMargin=1.6*cm,
        rightMargin=1.6*cm, topMargin=1.45*cm, bottomMargin=1.55*cm,
        title='When should an LLM speak technically?', author='Nikhil Makkar')
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='frame')
    doc.addPageTemplates(PageTemplate(id='main', frames=frame, onPage=footer))
    story = []
    def P(text, style='BodyX'):
        story.append(Paragraph(text, styles[style]))

    story.append(Spacer(1, 1.1*cm))
    P('When should an LLM<br/>speak technically?', 'TitleX')
    P('Evidence for a low-rank audience-relevance representation in Qwen3.5-4B', 'SubX')
    P('Nikhil Makkar · Working research report · 8 September 2026', 'SmallX')
    story.append(Spacer(1, .5*cm))
    P('Abstract', 'H1X')
    P("Language models adapt explanations to an imagined reader, but it is unclear whether reader relevance is represented in a simple, causally useful form. We compare a reader who is expert in the source domain with an equally sophisticated expert from another domain while holding the passage and candidate summaries fixed. In Qwen3.5-4B, a residual-stream direction separates the conditions and causally changes forced-choice preference across unfamiliar domains, answer encodings, and profile wording. Single-direction removal eliminates only part of the natural effect. A frozen extension finds that a rank-3 subspace produces substantially stronger mediation than rank 1 and beats 100 rank-matched random and 98 of 100 shuffled-label subspaces, with modest but detectable ordinary-text impact. The automatic free-generation metric and cross-family Phi replication fail. We therefore find a model-specific low-rank representation for audience-conditioned summary selection, not a universal explanation-depth mechanism.")
    P('Main conclusion in one sentence', 'H2X')
    P("Qwen3.5 appears to encode whether a reader's expertise is relevant in a small activation subspace that influences summary choice, but we have not shown reliable control of freely generated explanations or replication across model families.", 'CallX')
    P('How to read this report', 'H2X')
    P('Every result page keeps its figure beside four explanations: WHY gives the scientific reason; TEST states what would support or weaken the idea; HOW TO READ explains the visual; and VERDICT gives the narrow conclusion.')

    story.append(PageBreak())
    P('1. Question and claim hierarchy', 'H1X')
    P("The phrase ‘give me the gist’ does not specify a measurable level of technicality. We operationalize one component: whether the model expects the reader to understand terminology from the source domain. We compare relevant experts with equally sophisticated experts from unrelated fields, avoiding the broad expert-versus-novice contrast.")
    P('A positive technical margin means the model prefers the prepared technical summary over the accessible summary. An audience gap is the relevant expert margin minus the unrelated expert margin. Prompt duplicates are averaged within each of eight source domains before inference.')
    data = [
        ['Claim', 'Question'],
        ['Behavior', 'Does the reader description change summary choice?'],
        ['Representation', 'Can natural activations distinguish reader relevance?'],
        ['Sufficiency', 'Does inserting the feature change the choice?'],
        ['Mediation', 'Does removing the feature reduce natural behavior?'],
        ['Specificity', 'Do random and mislabeled features fail?'],
        ['External validity', 'Does it affect generated text and other model families?'],
    ]
    cooked = [[Paragraph(c, styles['GuideHead' if i == 0 else 'GuideBody'])
               for c in row] for i, row in enumerate(data)]
    table = Table(cooked, colWidths=[4*cm, 12.5*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor(NAVY)),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('GRID',(0,0),(-1,-1),.35,colors.lightgrey),
        ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor(LIGHT)]),
        ('LEFTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),
        ('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    story.append(table)
    P('With eight independent domains, an 8/8 directional result has exact one-sided sign-flip probability 1/256 = 0.00390625.', 'SmallX')

    for title, image, context, why, test, reading, verdict, status in FIGURES:
        story.append(PageBreak())
        P(title, 'H1X')
        P(context)
        im = Image(str(ASSETS / image))
        im._restrictSize(17.4*cm, 8.7*cm)
        story.append(im)
        story.append(Spacer(1, 5))
        guide = [
            [Paragraph('WHY',styles['GuideHead']),Paragraph(why,styles['GuideBody'])],
            [Paragraph('TEST',styles['GuideHead']),Paragraph(test,styles['GuideBody'])],
            [Paragraph('HOW TO READ',styles['GuideHead']),Paragraph(reading,styles['GuideBody'])],
            [Paragraph('VERDICT',styles['GuideHead']),Paragraph(verdict,styles['GuideBody'])],
        ]
        table = Table(guide, colWidths=[2.5*cm,14.2*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(0,-1),colors.HexColor('#EAF2FB')),
            ('GRID',(0,0),(-1,-1),.35,colors.HexColor('#D0D5DD')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),
            ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
            ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        story.append(table)
        story.append(Spacer(1, 7))
        P('Outcome: ' + status, 'CallX')

    story.append(PageBreak())
    P('Discussion', 'H1X')
    P("The evidence supports a compact mechanistic description within Qwen3.5-4B. Reader-domain relevance is naturally linearly represented, and its geometry is causally connected to preference between prepared technical and accessible summaries. One direction captures a stable shared component, but removal shows it is incomplete. The rank-3 experiment is therefore central: additional structured dimensions close substantially more of the gap, while rank-matched controls usually do not.")
    P("The word mechanism still requires care. We intervene on the residual stream, a shared communication channel between transformer blocks. We have not localized a head, MLP, or sparse circuit. Successful forced-choice control also does not imply control of every aspect of generated explanation style.")
    P('Strongest defensible claim', 'H2X')
    P('In Qwen3.5-4B, the forced-choice effect of reader-domain relevance is mediated more completely by a small learned residual-stream subspace than by its leading direction alone. A rank-3 subspace transfers across unfamiliar methods and new profile wording, beats rank-matched random and shuffled-label controls, and causes modest ordinary-text change.', 'CallX')
    P('Not established', 'H2X')
    for item in [
        'Expertise itself is three-dimensional.',
        'Rank 3 is the true or unique dimensionality.',
        'The subspace controls arbitrary generated explanations.',
        'The mechanism is universal across model families.',
        'Alignment or constitutional training caused the representation.',
        'Layer 17 contains a complete localized circuit.',
    ]:
        P('• ' + item)
    P('Best next experiments', 'H2X')
    P('Run neighboring-layer rank-3 ablations; compare controls at matched measured KL rather than only matched rank; bootstrap the learned subspace over discovery profiles; and complete blinded human scoring of free generations. Cross-model cosine requires a frozen held-out representation alignment—raw cosine between separately trained models is not meaningful.')

    story.append(PageBreak())
    P('Methods and reproducibility', 'H1X')
    P('Model and representation. Main results use Qwen3.5-4B residual-stream activations. The fixed analysis selects layer 17 and the final prompt token. The multidimensional analysis stacks eight domain-specific relevant-minus-unrelated expert contrasts and applies uncentered SVD. Centering would remove their shared mean contrast.')
    P('Intervention. Addition moves activations by alpha times a unit vector. Ablation removes the projection onto one direction or an orthonormal rank-k basis at every token position in the selected layer.')
    P('Controls. Isotropic vectors test arbitrary directions. Shuffled labels preserve the fitting procedure while destroying the intended meaning. Re-encoded answers test label-token artifacts. Rewritten profiles test carrier wording. WikiText NLL, KL, and top-1 agreement measure ordinary-language disturbance.')
    P('Primary local artifacts', 'H2X')
    for item in [
        'FALSIFICATION_REPORT.md — chronological experimental record',
        'MULTIDIMENSIONAL_SUBSPACE_PROTOCOL.md — frozen rank protocol',
        'MULTIDIMENSIONAL_RESULTS.md — detailed rank results',
        'results/qwen35_4b_multidimensional_subspace/ — numerical outputs',
        'build_accessible_paper.py — generator for this report',
    ]:
        P('• ' + item)
    P('Related methodological template', 'H2X')
    P("The contrastive-direction and causal-intervention design is inspired by Arditi et al., Refusal in Language Models Is Mediated by a Single Direction, and Venhoff et al., Understanding Reasoning in Thinking Language Models via Steering Vectors. This project asks a narrower question about audience relevance and summary selection; using a similar method does not license equally broad claims.")
    doc.build(story)
    print(OUT)

if __name__ == '__main__':
    build()
