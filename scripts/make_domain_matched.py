"""Build source-specific, tightly matched domain-expertise stimuli."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DOMAINS = {
    "resnet": ("deep convolutional networks for computer vision", "computational fluid dynamics"),
    "transformer": ("attention-based language and sequence models", "computational fluid dynamics"),
    "unet": ("neural networks for biomedical image segmentation", "molecular genetics"),
    "kriging": ("geostatistics, variograms, and spatial interpolation", "deep neural-network optimization"),
}

TECHNICAL_TERMS = {
    "resnet": ["residual", "F(x)", "identity shortcut", "projection", "degradation"],
    "transformer": ["self-attention", "multi-head", "query", "key", "value", "positional", "causal mask", "recurrence"],
    "unet": ["encoder", "decoder", "skip", "concaten", "dense segmentation", "contracting", "expanding"],
    "kriging": ["variogram", "covariance", "unbiased", "unit-sum", "kriging variance", "redundan"],
}

TEMPLATES = [
    ("train", "The reader researches {domain}. They regularly read papers and want a concise summary of this passage."),
    ("train", "The reader has extensive professional knowledge of {domain}. They want to understand the main mechanism briefly."),
    ("train", "The reader publishes technical work on {domain}. Give them whichever short summary would be more useful."),
    ("train", "The reader has graduate-level expertise in {domain}. They requested a compact account of the central contribution."),
    ("validation", "The reader routinely evaluates research about {domain}. They want a short, informative summary."),
    ("validation", "The reader works professionally on {domain}. They asked for the gist of the passage."),
    ("test", "The reader is deeply familiar with {domain}. They want the core idea explained concisely."),
    ("test", "The reader is an experienced researcher in {domain}. They requested a brief summary of the key mechanism."),
]


def profiles(domain):
    return [
        {"split": split, "text": template.format(domain=domain)}
        for split, template in TEMPLATES
    ]


def main():
    base = json.loads((ROOT / "data" / "stimuli_matched.json").read_text())
    sources = []
    for source in base["sources"]:
        relevant, other = DOMAINS[source["id"]]
        source = dict(source)
        source["technical_terms"] = TECHNICAL_TERMS[source["id"]]
        source["profiles"] = {
            "domain_expert": profiles(relevant),
            "matched_other_expert": profiles(other),
        }
        sources.append(source)
    output = {"sources": sources, "instruction": base["instruction"]}
    (ROOT / "data" / "stimuli_domain_matched.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
