"""Build the frozen unfamiliar-method test set from the prospective protocol."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SOURCES = [
    {
        "id": "driftgate",
        "role": "control",
        "citation": "Invented method; self-contained state-space mechanism",
        "passage": "DriftGate estimates a hidden dynamical state from noisy controls and observations. It alternates a linear prediction, which propagates both the state estimate and its covariance, with a measurement update based on the innovation. Before updating, it computes the innovation's covariance-normalized squared magnitude. Measurements above a fixed gate are treated as outliers and skipped. Accepted observations receive a Kalman-style gain that balances predicted uncertainty against measurement noise.",
        "technical_summary": "DriftGate is a robust Kalman-style filter that propagates state covariance, gates observations by squared Mahalanobis innovation, and applies the covariance-derived gain only to accepted measurement updates.",
        "accessible_summary": "DriftGate tracks a changing hidden quantity by first predicting its next value and uncertainty, then correcting that prediction with new measurements. It ignores measurements that are implausibly far from the prediction and trusts accepted measurements according to their noise.",
        "technical_terms": ["Kalman", "covariance", "innovation", "Mahalanobis", "gain"],
        "expert_field": "state-space estimation, Kalman filtering, and feedback control",
        "other_field": "population genetics and phylogenetic reconstruction",
    },
    {
        "id": "fluxpatch",
        "role": "numerical_pde",
        "citation": "Invented method; self-contained finite-volume mechanism",
        "passage": "FluxPatch solves conservation laws on a mesh whose cells can be refined locally. Each cell stores an average conserved state, and neighboring states determine numerical fluxes through shared faces. Fine cells take smaller time steps near shocks. At every coarse-fine boundary, the method replaces the coarse face flux with the sum of the corresponding fine-grid fluxes. This refluxing correction ensures that refinement does not create or destroy the conserved quantity.",
        "technical_summary": "FluxPatch is an adaptive finite-volume scheme with face-based numerical fluxes, local subcycling, and coarse-fine refluxing that restores discrete conservation across refinement interfaces.",
        "accessible_summary": "FluxPatch simulates flowing quantities by tracking how much crosses each cell boundary and using smaller cells and time steps near sharp changes. Where coarse and fine regions meet, it reconciles their boundary flows so the total quantity remains conserved.",
        "technical_terms": ["finite-volume", "numerical flux", "subcycling", "reflux", "discrete conservation"],
        "expert_field": "finite-volume methods, conservation laws, and computational fluid dynamics",
        "other_field": "information retrieval and web search ranking",
    },
    {
        "id": "anchorshift",
        "role": "causal_inference",
        "citation": "Invented method; self-contained instrumental-variable mechanism",
        "passage": "AnchorShift estimates a causal effect when treatment is correlated with unobserved causes of the outcome. It uses an auxiliary variable that predicts treatment but is assumed to affect the outcome only through treatment. Separate sample folds learn the treatment and outcome nuisance regressions. On held-out folds, residualized instrument, treatment, and outcome values form an orthogonal moment equation whose solution estimates the effect. Cross-fitting limits bias from reusing observations to fit nuisance models and estimate the target.",
        "technical_summary": "AnchorShift is a cross-fitted instrumental-variable estimator using residualized variables and an orthogonal moment condition, relying on instrument relevance and exclusion to address unobserved treatment-outcome confounding.",
        "accessible_summary": "AnchorShift estimates a treatment's effect using an outside factor that changes treatment but has no separate route to the outcome. It learns background relationships on one part of the data and estimates the effect on another to reduce overfitting bias.",
        "technical_terms": ["instrumental", "cross-fit", "residuali", "orthogonal moment", "exclusion"],
        "expert_field": "instrumental variables, semiparametric causal inference, and econometrics",
        "other_field": "computer graphics and physically based rendering",
    },
    {
        "id": "shardbloom",
        "role": "databases",
        "citation": "Invented method; self-contained approximate-membership mechanism",
        "passage": "ShardBloom accelerates key lookup across immutable database segments. Each segment stores a Bloom filter whose bits are set by several hashes of every resident key. A negative filter result safely skips that segment, whereas a positive result triggers an exact index lookup because collisions can produce false positives. Filters are arranged in the same hierarchy as the segments, and their bit budgets are allocated using observed query rates so frequently accessed levels receive lower false-positive probabilities.",
        "technical_summary": "ShardBloom attaches multi-hash Bloom filters to immutable segments and query-weighted hierarchy levels, enabling no-false-negative pruning while retaining exact lookups after possible false positives.",
        "accessible_summary": "ShardBloom gives each database segment a compact checklist that can prove a key is absent but may occasionally say it is present by mistake. The database skips definitely irrelevant segments and verifies every possible match, spending more checklist space on busy levels.",
        "technical_terms": ["Bloom filter", "hash", "false positive", "no-false-negative", "bit budget"],
        "expert_field": "database storage engines, indexing, and approximate membership structures",
        "other_field": "continuum mechanics and material fracture",
    },
    {
        "id": "motifbridge",
        "role": "computational_biology",
        "citation": "Invented method; self-contained profile-HMM mechanism",
        "passage": "MotifBridge aligns a protein sequence to a family model with match, insertion, and deletion states at each consensus position. Match states emit amino acids using position-specific probabilities, while transitions model gaps of different lengths. Pseudocounts keep unseen residues and transitions from receiving zero probability. The forward algorithm sums probability over all possible alignments for family scoring, while Viterbi decoding returns the single highest-scoring alignment for interpretation.",
        "technical_summary": "MotifBridge uses a pseudocount-smoothed profile HMM with position-specific emissions and match/insert/delete transitions; forward inference scores a sequence marginally, while Viterbi decoding yields its maximum-probability alignment.",
        "accessible_summary": "MotifBridge compares a protein with a family pattern while allowing residues to match, be inserted, or be skipped at each position. It can either combine evidence from every possible alignment to score membership or return the one best alignment to inspect.",
        "technical_terms": ["profile HMM", "emission", "match", "insert", "delete", "Viterbi", "forward algorithm"],
        "expert_field": "profile hidden Markov models and computational protein sequence analysis",
        "other_field": "distributed systems and consensus protocols",
    },
    {
        "id": "phasesar",
        "role": "remote_sensing",
        "citation": "Invented method; self-contained interferometric SAR mechanism",
        "passage": "PhaseSAR estimates surface displacement from repeated synthetic-aperture radar acquisitions. It forms interferometric phase differences, removes the phase predicted by imaging geometry and a reference elevation model, and weights each observation by coherence. Because phase is observed modulo a full cycle, neighboring pixels are unwrapped along paths that favor high-coherence edges. A weighted least-squares inversion across acquisition pairs then separates a displacement time series from residual atmospheric and measurement errors.",
        "technical_summary": "PhaseSAR performs coherence-weighted interferometric phase correction and path-guided unwrapping, then uses a multi-pair weighted least-squares inversion to recover displacement through time.",
        "accessible_summary": "PhaseSAR compares repeated radar images to measure small ground movements. It corrects predictable geometric effects, resolves the radar signal's repeating phase using reliable neighboring pixels, and combines many image pairs while giving less influence to noisy measurements.",
        "technical_terms": ["interferometric", "coherence", "unwrap", "phase", "least-squares", "displacement"],
        "expert_field": "interferometric synthetic-aperture radar and geodetic remote sensing",
        "other_field": "psycholinguistics and first-language acquisition",
    },
    {
        "id": "quorumweave",
        "role": "cryptography",
        "citation": "Invented method; self-contained threshold-signature mechanism",
        "passage": "QuorumWeave distributes a signing key among several participants using polynomial secret sharing, so no machine stores the complete key. A distributed setup protocol creates consistent shares without reconstructing the secret. To authorize a message, any threshold number of participants produce verifiable partial signatures. Lagrange interpolation in the exponent combines valid partials into one standard signature, while fewer than the threshold reveal no signing key and cannot create a valid aggregate.",
        "technical_summary": "QuorumWeave uses distributed key generation and polynomial secret shares for threshold signing; verifiable partial signatures are combined by interpolation in the exponent without reconstructing the private key.",
        "accessible_summary": "QuorumWeave splits signing authority across several machines so that no single machine possesses the whole secret. Enough participants can combine their individually checkable approvals into one ordinary signature, while a smaller group cannot sign or recover the key.",
        "technical_terms": ["distributed key", "secret share", "threshold", "partial signature", "Lagrange", "interpolation"],
        "expert_field": "threshold cryptography, secret sharing, and digital signature protocols",
        "other_field": "atmospheric chemistry and aerosol microphysics",
    },
    {
        "id": "phasenest",
        "role": "audio_signal_processing",
        "citation": "Invented method; self-contained time-frequency mechanism",
        "passage": "PhaseNest denoises audio in the short-time Fourier domain at two window sizes. A long-window branch estimates stable harmonic structure, while a short-window branch follows transients. The network predicts bounded complex masks that modify both magnitude and phase in each branch. Their inverse transforms are fused with learned time-varying weights, and a consistency projection maps the fused waveform back onto signals representable by the chosen overlapping analysis windows before final overlap-add reconstruction.",
        "technical_summary": "PhaseNest combines long- and short-window STFT branches, bounded complex masking, learned time-varying fusion, and spectrogram-consistency projection before overlap-add waveform reconstruction.",
        "accessible_summary": "PhaseNest cleans audio using one time-frequency view suited to steady tones and another suited to sudden sounds. It adjusts both strength and timing information, blends the two reconstructions over time, and enforces that the result corresponds to a valid overlapping-window analysis.",
        "technical_terms": ["STFT", "complex mask", "magnitude", "phase", "consistency", "overlap-add"],
        "expert_field": "time-frequency audio processing, complex spectral masking, and source enhancement",
        "other_field": "labor economics and social policy evaluation",
    },
]

CARRIERS = [
    "The reader is deeply familiar with {field}. They want the core idea explained concisely.",
    "The reader is an experienced researcher in {field}. They requested a brief summary of the key mechanism.",
]

INSTRUCTION = "The source is followed by two factually intended summaries. Choose the summary that would be more useful to the described reader. Judge usefulness for that reader, not which prose style you personally prefer. Reply with exactly A or B and no explanation."


def main():
    sources = []
    for specification in SOURCES:
        source = dict(specification)
        expert = source.pop("expert_field")
        other = source.pop("other_field")
        source["profiles"] = {
            "domain_expert": [
                {"split": "test", "text": template.format(field=expert)}
                for template in CARRIERS
            ],
            "matched_other_expert": [
                {"split": "test", "text": template.format(field=other)}
                for template in CARRIERS
            ],
        }
        sources.append(source)
    output = ROOT / "data" / "stimuli_prospective_synthetic.json"
    output.write_text(json.dumps({"sources": sources, "instruction": INSTRUCTION}, indent=2) + "\n")

    review = ROOT / "results" / "prospective_stimulus_review.csv"
    review.parent.mkdir(parents=True, exist_ok=True)
    fields = ["source", "passage", "technical_summary", "accessible_summary", "passage_accurate", "technical_summary_accurate", "accessible_summary_accurate", "pair_comparably_useful", "notes"]
    with review.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for source in sources:
            writer.writerow({
                "source": source["id"],
                "passage": source["passage"],
                "technical_summary": source["technical_summary"],
                "accessible_summary": source["accessible_summary"],
                "passage_accurate": "",
                "technical_summary_accurate": "",
                "accessible_summary_accurate": "",
                "pair_comparably_useful": "",
                "notes": "",
            })
    print(f"wrote {output} and {review} with {len(sources)} frozen sources")


if __name__ == "__main__":
    main()
