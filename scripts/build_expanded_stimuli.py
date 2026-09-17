"""Build the frozen eight-domain exact-relevance stimulus set.

The four original sources are preserved byte-for-byte at the object level.  New
profiles use the same eight carrier sentences and split assignment, changing
only the named field of expertise between conditions.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


NEW_SOURCES = [
    {
        "id": "bert",
        "role": "language_representation",
        "citation": "Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, NAACL 2019",
        "passage": (
            "BERT pre-trains a Transformer encoder on unlabeled text so that every layer can use both left and right context. "
            "Its masked-language-model objective replaces or obscures selected input tokens and trains the network to recover them from the surrounding sequence. "
            "The original model also uses a next-sentence-prediction objective over pairs of text segments. "
            "After pre-training, the same encoder can be fine-tuned for tasks such as question answering and natural-language inference with only a small task-specific output layer."
        ),
        "technical_summary": (
            "BERT pre-trains a deeply bidirectional Transformer encoder with masked language modeling and, in the original formulation, next-sentence prediction. Fine-tuning adds a small task head while updating the shared contextual representations end to end."
        ),
        "accessible_summary": (
            "BERT first learns from unlabeled text by hiding some words and predicting them using context on both sides. The resulting model can then be adapted to tasks such as question answering by adding a small output component and training on task examples."
        ),
        "technical_terms": ["bidirectional", "masked language", "next-sentence", "fine-tun", "Transformer encoder"],
        "expert_field": "bidirectional language-model pre-training and Transformer encoders",
        "other_field": "database query optimization and transaction processing",
    },
    {
        "id": "gcn",
        "role": "graph_ml",
        "citation": "Kipf and Welling, Semi-Supervised Classification with Graph Convolutional Networks, ICLR 2017",
        "passage": (
            "A graph convolutional network updates node representations by mixing each node's features with features from its neighbors. "
            "Kipf and Welling derive a simple propagation rule from a localized first-order approximation to spectral graph convolution. "
            "Adding self-connections and normalizing by node degrees stabilizes this aggregation. "
            "Stacking layers lets supervision from a small labeled subset shape representations that incorporate both node attributes and local graph structure, with computation scaling linearly in the number of edges."
        ),
        "technical_summary": (
            "The GCN uses a renormalized first-order spectral propagation rule to aggregate self and neighbor features. Stacked message-passing layers support semi-supervised node classification by combining attributes with local topology at cost linear in the edge count."
        ),
        "accessible_summary": (
            "A GCN represents each graph node by repeatedly combining its own information with information from connected neighbors. This allows labels available for only some nodes to guide predictions elsewhere while keeping computation efficient on sparse graphs."
        ),
        "technical_terms": ["spectral", "renorm", "message-passing", "node classification", "edge count"],
        "expert_field": "graph neural networks and semi-supervised node classification",
        "other_field": "clinical trial design and causal epidemiology",
    },
    {
        "id": "ddpm",
        "role": "generative_models",
        "citation": "Ho et al., Denoising Diffusion Probabilistic Models, NeurIPS 2020",
        "passage": (
            "A denoising diffusion probabilistic model defines a forward Markov process that gradually adds Gaussian noise to data according to a variance schedule. "
            "A neural network learns the reverse transitions, commonly by predicting the noise present at a randomly selected timestep. "
            "To generate a sample, the model starts from Gaussian noise and repeatedly applies the learned denoising transitions. "
            "The training objective is connected to a variational bound and to denoising score matching."
        ),
        "technical_summary": (
            "A DDPM fixes a Gaussian forward noising chain and learns its parameterized reverse process, typically through timestep-conditioned noise prediction. Sampling iteratively denoises from the Gaussian prior; the objective relates a variational bound to denoising score matching."
        ),
        "accessible_summary": (
            "A diffusion model learns to reverse a process that slowly corrupts training examples with random noise. It creates new examples by starting with noise and removing it over many steps, using a network trained to estimate the corruption at each stage."
        ),
        "technical_terms": ["Markov", "Gaussian", "reverse process", "noise prediction", "variational", "score matching"],
        "expert_field": "diffusion probabilistic models and score-based image generation",
        "other_field": "compiler construction and programming-language semantics",
    },
    {
        "id": "xgboost",
        "role": "tabular_ml",
        "citation": "Chen and Guestrin, XGBoost: A Scalable Tree Boosting System, KDD 2016",
        "passage": (
            "Gradient tree boosting builds an additive predictor by fitting each new decision tree to improve the current ensemble under a differentiable loss. "
            "XGBoost uses a regularized objective and a second-order approximation involving gradients and Hessians to score candidate tree structures and leaf weights. "
            "Its sparsity-aware split algorithm assigns a learned default direction to missing values. "
            "A weighted quantile sketch supports approximate split finding, while cache-aware and out-of-core system designs make training scalable."
        ),
        "technical_summary": (
            "XGBoost performs regularized additive tree boosting with second-order gradient statistics for split and leaf scoring. Sparsity-aware default directions, weighted quantile sketches, and cache-aware out-of-core execution provide scalable approximate tree construction."
        ),
        "accessible_summary": (
            "XGBoost builds a strong predictor by adding decision trees one at a time, with each tree correcting the current ensemble. It controls tree complexity and uses specialized methods for missing values, approximate splits, and efficient processing of large datasets."
        ),
        "technical_terms": ["regulariz", "second-order", "Hessian", "sparsity-aware", "quantile sketch", "out-of-core"],
        "expert_field": "gradient-boosted decision trees and scalable tabular machine learning",
        "other_field": "radio astronomy and interferometric imaging",
    },
]


def profiles(field: str, carriers: list[dict]) -> list[dict]:
    result = []
    for item in carriers:
        text = item["text"]
        start = text.index("researches") if "researches" in text else -1
        replacements = {
            "deep convolutional networks for computer vision": field,
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        if text == item["text"]:
            # All original carrier forms place the old field before the first period.
            prefixes = (
                "The reader has extensive professional knowledge of ",
                "The reader publishes technical work on ",
                "The reader has graduate-level expertise in ",
                "The reader routinely evaluates research about ",
                "The reader works professionally on ",
                "The reader is deeply familiar with ",
                "The reader is an experienced researcher in ",
            )
            for prefix in prefixes:
                if text.startswith(prefix):
                    rest = text[text.index("."):]
                    text = prefix + field + rest
                    break
            if start >= 0 and text == item["text"]:
                rest = text[text.index(".", start):]
                text = "The reader researches " + field + rest
        result.append({"split": item["split"], "text": text})
    return result


def main() -> None:
    base = json.loads((ROOT / "data" / "stimuli_domain_matched.json").read_text())
    carrier = base["sources"][0]["profiles"]["domain_expert"]
    for specification in NEW_SOURCES:
        source = copy.deepcopy(specification)
        expert_field = source.pop("expert_field")
        other_field = source.pop("other_field")
        source["profiles"] = {
            "domain_expert": profiles(expert_field, carrier),
            "matched_other_expert": profiles(other_field, carrier),
        }
        base["sources"].append(source)
    output = ROOT / "data" / "stimuli_domain_matched_8.json"
    output.write_text(json.dumps(base, indent=2) + "\n")
    print(f"wrote {output} with {len(base['sources'])} frozen sources")


if __name__ == "__main__":
    main()
