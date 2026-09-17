"""Component-wise ablation signatures and blinded rank-3/rank-7 generations."""
from __future__ import annotations
import argparse, csv, hashlib, itertools, json, random
from contextlib import nullcontext
from pathlib import Path
from statistics import mean
import torch

from experiment import build_examples, format_prompt, get_layers, load_model, load_stimuli
from fixed_direction_transfer import group_activations
from layer_token_sweep import capture_tail, prepare_batch, run_batch
from multidimensional_subspace import ablate_basis, domain_contrasts, reductions
from reencoding_control import ids_for

PROMPTS = [
    "Explain why Earth has seasons in three concise paragraphs.",
    "Give an intuitive explanation of gradient descent, including one limitation.",
    "Summarize the main causes of inflation without assuming an economics degree.",
    "Write Python pseudocode for binary search and explain its time complexity.",
    "Explain the difference between correlation and causation with a concrete example.",
    "Describe how a vaccine trains the immune system without claiming it guarantees immunity.",
    "Compare TCP and UDP for someone choosing a protocol for a live video application.",
    "Explain why regularization can improve a machine-learning model's test performance.",
    "Describe photosynthesis accurately in under 120 words.",
    "Explain what a confidence interval does and does not mean.",
    "Give a balanced explanation of nuclear fission versus nuclear fusion.",
    "Explain how public-key encryption permits secure communication over an insecure channel.",
]

def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def generate(model, tokenizer, prompt, layer, basis, max_new_tokens):
    rendered = format_prompt(tokenizer, prompt)
    inputs = tokenizer(rendered, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    context = nullcontext() if basis is None else ablate_basis(layer, basis)
    with context, torch.inference_mode():
        output = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                                pad_token_id=tokenizer.eos_token_id)
    new = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new, skip_special_tokens=True).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--subspace", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, default=Path("data/stimuli_domain_matched_8.json"))
    parser.add_argument("--test", type=Path, default=Path("data/stimuli_prospective_synthetic.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    saved = torch.load(args.subspace, map_location="cpu", weights_only=True)
    basis = saved["basis"].float()
    layer_index = int(saved["layer"])
    model, tokenizer = load_model(args.model)
    layer = list(get_layers(model))[layer_index]

    test = build_examples(load_stimuli(args.test))
    inputs = prepare_batch(model, tokenizer, test)
    label_ids = ids_for(tokenizer, ("A", "B"))
    baseline = run_batch(model, inputs, test, label_ids)
    subset_rows = []
    subsets = []
    for size in range(1, 4):
        subsets.extend(itertools.combinations(range(3), size))
    subsets.extend((index,) for index in range(3, 8))
    for subset in subsets:
        chosen = basis[list(subset)]
        changed = run_batch(model, inputs, test, label_ids, ablate_basis(layer, chosen))
        effects = reductions(test, baseline, changed)
        for source, effect in effects.items():
            subset_rows.append({
                "components": "+".join(str(i+1) for i in subset),
                "rank": len(subset), "source": source,
                "gap_reduction": effect, "aggregate_mean": mean(effects.values()),
            })
        print("components", subset, "mean", mean(effects.values()), flush=True)

    discovery = [e for e in build_examples(load_stimuli(args.discovery)) if e["split"] == "train"]
    activations = {}
    for index, example in enumerate(discovery, 1):
        activations[example["example_id"]] = capture_tail(model, tokenizer, example["prompt"], 1)
        if index % 16 == 0:
            print(f"capture {index}/{len(discovery)}", flush=True)
    contrasts = domain_contrasts(group_activations(discovery, activations, layer_index))
    sources = sorted({e["source_id"] for e in discovery})
    coefficients = contrasts.float() @ basis.T
    loading_rows = []
    for source_index, source in enumerate(sources):
        norm = contrasts[source_index].norm().clamp_min(1e-12)
        for component in range(8):
            loading_rows.append({
                "source": source, "component": component+1,
                "signed_loading": float(coefficients[source_index, component]),
                "loading_over_contrast_norm": float(coefficients[source_index, component]/norm),
            })

    generation_rows = []
    conditions = [("baseline", None), ("rank3", basis[:3]), ("rank7", basis[:7])]
    for prompt_index, prompt in enumerate(PROMPTS):
        for condition, chosen in conditions:
            text = generate(model, tokenizer, prompt, layer, chosen, args.max_new_tokens)
            code = hashlib.sha256(f"component-review:{prompt_index}:{condition}".encode()).hexdigest()[:12]
            generation_rows.append({
                "blind_code": code, "prompt_index": prompt_index,
                "condition": condition, "prompt": prompt, "text": text,
                "word_count": len(text.split()),
            })
        print(f"generation {prompt_index+1}/{len(PROMPTS)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write(args.output_dir/"component_subset_ablation.csv", subset_rows)
    write(args.output_dir/"discovery_component_loadings.csv", loading_rows)
    write(args.output_dir/"qualitative_generation_rows.csv", generation_rows)
    review = [{
        "blind_code": row["blind_code"], "prompt": row["prompt"], "text": row["text"],
        "coherence_1_to_5": "", "correctness_1_to_5": "",
        "instruction_following_1_to_5": "", "obvious_corruption_0_or_1": "", "notes": "",
    } for row in generation_rows]
    random.Random(20260908).shuffle(review)
    write(args.output_dir/"blind_qualitative_review.csv", review)
    write(args.output_dir/"blind_qualitative_key.csv", [{
        "blind_code": row["blind_code"], "prompt_index": row["prompt_index"],
        "condition": row["condition"], "word_count": row["word_count"],
    } for row in generation_rows])
    (args.output_dir/"config.json").write_text(json.dumps({
        "model": args.model, "layer": layer_index, "subspace": str(args.subspace),
        "test": str(args.test), "discovery": str(args.discovery),
        "generation_prompts": len(PROMPTS), "decoding": "greedy",
        "max_new_tokens": args.max_new_tokens,
    }, indent=2)+"\n")

if __name__ == "__main__":
    main()
