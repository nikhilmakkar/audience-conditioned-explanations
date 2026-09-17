"""Search validation-only one-dimensional mixtures before rank-3 claims."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from statistics import mean
import torch

from experiment import build_examples, get_layers, load_model, load_stimuli
from exact_relevance_lodo import mean_source_gap
from layer_token_sweep import prepare_batch, run_batch
from multidimensional_subspace import ablate_basis, reductions
from reencoding_control import ids_for

def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)

def candidates(basis, span, count, seed):
    output=[]
    for component in range(span):
        coefficient=torch.zeros(span);coefficient[component]=1
        output.append((f"component_{component+1}",coefficient,basis[component:component+1]))
    generator=torch.Generator().manual_seed(seed)
    for index in range(count):
        coefficient=torch.randn(span,generator=generator)
        coefficient/=coefficient.norm().clamp_min(1e-12)
        direction=coefficient@basis[:span]
        direction/=direction.norm().clamp_min(1e-12)
        output.append((f"random_mix_{index}",coefficient,direction[None,:]))
    return output

def prepare(model,tokenizer,examples):
    label_ids=ids_for(tokenizer,("A","B"))
    inputs=prepare_batch(model,tokenizer,examples)
    baseline=run_batch(model,inputs,examples,label_ids)
    return inputs,label_ids,baseline

def eval_basis(model,layer,examples,prepared,basis):
    inputs,label_ids,baseline=prepared
    changed=run_batch(model,inputs,examples,label_ids,ablate_basis(layer,basis))
    effects=reductions(examples,baseline,changed)
    return effects

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",default="Qwen/Qwen3.5-4B")
    parser.add_argument("--subspace",type=Path,required=True)
    parser.add_argument("--discovery",type=Path,default=Path("stimuli_domain_matched_8.json"))
    parser.add_argument("--synthetic",type=Path,default=Path("stimuli_prospective_synthetic.json"))
    parser.add_argument("--new-carriers",type=Path,default=Path("stimuli_prospective_profile_paraphrase.json"))
    parser.add_argument("--mixtures",type=int,default=256)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()

    saved=torch.load(args.subspace,map_location="cpu",weights_only=True)
    basis=saved["basis"].float();layer_index=int(saved["layer"])
    model,tokenizer=load_model(args.model);layer=list(get_layers(model))[layer_index]
    real=build_examples(load_stimuli(args.discovery))
    sets={
        "real_validation":[e for e in real if e["split"]=="validation"],
        "real_test":[e for e in real if e["split"]=="test"],
        "synthetic":build_examples(load_stimuli(args.synthetic)),
        "new_carriers":build_examples(load_stimuli(args.new_carriers)),
    }
    prepared={name:prepare(model,tokenizer,examples) for name,examples in sets.items()}
    validation_rows=[];selected={}
    for span,seed in ((3,20260908),(8,20260909)):
        best=None
        for index,(name,coefficient,direction) in enumerate(candidates(basis,span,args.mixtures,seed),1):
            effects=eval_basis(model,layer,sets["real_validation"],prepared["real_validation"],direction)
            score=mean(effects.values())
            validation_rows.append({"span":span,"candidate":name,"mean_reduction":score,
                                    "positive_domains":sum(v>0 for v in effects.values())})
            if best is None or score>best[0]:
                best=(score,name,coefficient.clone(),direction.clone())
            if index%64==0:print(f"span {span}: {index}/{args.mixtures+span}",flush=True)
        selected[f"best_1d_in_rank{span}"]=best
        print("selected",span,best[0],best[1],best[2].tolist(),flush=True)

    comparators={
        "svd_component_1":basis[:1],
        "rank3_subspace":basis[:3],
        "rank8_subspace":basis[:8],
        "best_1d_in_rank3":selected["best_1d_in_rank3"][3],
        "best_1d_in_rank8":selected["best_1d_in_rank8"][3],
    }
    result_rows=[]
    for set_name,examples in sets.items():
        for name,direction in comparators.items():
            effects=eval_basis(model,layer,examples,prepared[set_name],direction)
            for source,value in effects.items():
                result_rows.append({"test_set":set_name,"intervention":name,"source":source,
                                    "gap_reduction":value,"aggregate_mean":mean(effects.values()),
                                    "positive_domains":sum(v>0 for v in effects.values())})
        print("evaluation complete",set_name,flush=True)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    write(args.output_dir/"validation_search.csv",validation_rows)
    write(args.output_dir/"evaluation_by_domain.csv",result_rows)
    torch.save({"layer":layer_index,
                "best_1d_in_rank3":selected["best_1d_in_rank3"][3],
                "best_1d_in_rank8":selected["best_1d_in_rank8"][3],
                "rank3_coefficients":selected["best_1d_in_rank3"][2],
                "rank8_coefficients":selected["best_1d_in_rank8"][2]},
               args.output_dir/"selected_1d_mixtures.pt")
    (args.output_dir/"config.json").write_text(json.dumps({
        "model":args.model,"layer":layer_index,"mixtures_per_span":args.mixtures,
        "selection_set":"real-domain validation profiles",
        "evaluation_sets":["real-domain test profiles","synthetic methods","new carriers"],
        "rank3_selected_validation_reduction":selected["best_1d_in_rank3"][0],
        "rank8_selected_validation_reduction":selected["best_1d_in_rank8"][0],
    },indent=2)+"\n")

if __name__=="__main__":
    main()
