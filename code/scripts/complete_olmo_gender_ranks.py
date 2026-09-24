"""Restore the sole missing full-row LRE rank vector; do not refit parameters."""
import json
from functools import partial
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens import model_sweep as sweep

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"results/all_models_gender_original_continents/olmo3_7b"
OUT = ROOT/"results/paper_full_evaluation/olmo_gender_lre_ranks.json"

def main():
    torch.set_num_threads(2)
    manifest = json.loads((BASE/"manifest.json").read_text())
    old = json.loads((BASE/"measurements/name_gender.json").read_text())
    data = json.loads((ROOT/"data/relations/name_gender.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(manifest["model"],revision=manifest["revision"],local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(manifest["model"],revision=manifest["revision"],local_files_only=True,dtype=torch.float32,attn_implementation="eager",device_map={"":"cuda:0"}).eval().requires_grad_(False)
    model.forward = partial(model.forward,use_cache=False,logits_to_keep=1)
    model.config.use_cache = False
    layer = int(old["faithfulness"]["best_key"].split("/")[0])
    verification = sweep.verify(model,tokenizer,layer,8)
    sweep.enable_prefix_cache(model,tokenizer,layer,8)
    split = tuple(old["lre_split"][k] for k in ["demonstrations","references","evaluation"])
    trailing = sweep.trailing_space_needed(tokenizer,"name_gender",data["samples"])
    result = sweep.measure_faith(model,tokenizer,data["samples"],data["prompt_templates"][0],"main",[layer],trailing,8,None,split_override=split)
    assert f"{result['best_layer']}/{float(result['best_beta'])}" == old["faithfulness"]["best_key"]
    assert result["best_faith"] == old["faithfulness"]["best_faith"]
    for a,b in zip(result["rows"],old["faithfulness"]["rows"],strict=True):
        for key in ["subject","object","model_top1_id","hit"]:assert a[key] == b[key]
    import hashlib
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    sweep.write(OUT,dict(model=manifest["model"],revision=manifest["revision"],verification=verification,rows=result["rows"],best_key=old["faithfulness"]["best_key"],source_sha256=digest(BASE/"measurements/name_gender.json"),runner_sha256=digest(Path(__file__))))
    print("COMPLETE",len(result["rows"]),flush=True)

if __name__ == "__main__":main()
