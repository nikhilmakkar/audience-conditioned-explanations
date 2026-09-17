"""Minimal held-out test of the validation-selected one-dimensional mixture."""
import csv
from pathlib import Path
from statistics import mean
import torch
from experiment import build_examples, get_layers, load_model, load_stimuli
from layer_token_sweep import prepare_batch, run_batch
from multidimensional_subspace import ablate_basis, reductions
from reencoding_control import ids_for

ROOT=Path(__file__).resolve().parents[1]
saved=torch.load(ROOT/"results/qwen35_4b_multidimensional_subspace/relevance_subspace.pt",
                 map_location="cpu",weights_only=True)
basis=saved["basis"].float(); layer_index=int(saved["layer"])
coef=torch.tensor([-0.8438711762428284,-0.1767401099205017,0.5066007971763611])
direction=coef@basis[:3]; direction/=direction.norm()
model,tokenizer=load_model("Qwen/Qwen3.5-4B"); layer=list(get_layers(model))[layer_index]
real=build_examples(load_stimuli(ROOT / "data" / "stimuli_domain_matched_8.json"))
sets={"real_test":[e for e in real if e["split"]=="test"],
      "synthetic":build_examples(load_stimuli(ROOT / "data" / "stimuli_prospective_synthetic.json")),
      "new_carriers":build_examples(load_stimuli(ROOT / "data" / "stimuli_prospective_profile_paraphrase.json"))}
comparators={"svd_component_1":basis[:1],"selected_1d_mixture":direction[None,:],
             "rank3_subspace":basis[:3]}
rows=[]
for set_name,examples in sets.items():
    labels=ids_for(tokenizer,("A","B")); inputs=prepare_batch(model,tokenizer,examples)
    baseline=run_batch(model,inputs,examples,labels)
    for name,candidate in comparators.items():
        changed=run_batch(model,inputs,examples,labels,ablate_basis(layer,candidate))
        effects=reductions(examples,baseline,changed)
        aggregate=mean(effects.values()); positive=sum(v>0 for v in effects.values())
        print(set_name,name,aggregate,positive,flush=True)
        for source,value in effects.items():
            rows.append({"test_set":set_name,"intervention":name,"source":source,
                         "gap_reduction":value,"aggregate_mean":aggregate,
                         "positive_domains":positive})
out=ROOT/"results/qwen35_4b_intrinsic_rank_minimal"; out.mkdir(parents=True,exist_ok=True)
with (out/"evaluation_by_domain.csv").open("w",newline="") as handle:
    writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
torch.save({"layer":layer_index,"coefficients":coef,"direction":direction},
           out/"selected_1d_mixture.pt")
