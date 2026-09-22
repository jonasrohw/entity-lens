"""Refresh the original OLMo/GPT-J free-generation diagnostic, with newline prefixes."""
import argparse
import json
import socket
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens import model_sweep as sweep, splits
from jlens.matching import distinctive
from jlens.measure_prominence import RELATIONS
from jlens.prominence_statistics import leaks
from run_all_model_newline import ROOT, OUTPUT, sha, specs

PROMPTS = ["\n{} is best known for", "\nHere are some facts about {}. First,"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", choices=["olmo3_7b", "gptj"], required=True)
    p.add_argument("--worker", required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    spec = specs()[a.name]
    tok = AutoTokenizer.from_pretrained(spec["model_id"],revision=spec["revision"],local_files_only=True,padding_side="left")
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(spec["model_id"],revision=spec["revision"],local_files_only=True,dtype=torch.float32,attn_implementation="eager",device_map={"":"cuda:0"}).eval().requires_grad_(False)
    dest = ROOT / "results/shared_baselines" / a.name / "prominence"
    meta = dict(**spec,prompts=PROMPTS,max_new_tokens=90,do_sample=False,dtype="float32",batch_size=8,host=socket.gethostname(),worker=a.worker,script_sha256=sha(Path(__file__)))
    for relation in RELATIONS:
        source = OUTPUT / a.name / "shared_split" / f"{relation}.json"
        if not source.exists():
            continue
        target = dest / source.name
        with sweep.measurement_claim(target) as acquired:
            if not acquired or target.exists():
                continue
            old = json.loads(source.read_text())
            data = json.loads((sweep.DATA / source.name).read_text())
            samples = splits.filter_identifying_objects(data["samples"],tok,old["trailing_space"])
            demos,refs,tests = sweep.partitions(samples,"shared_split",lens=True)
            assert splits.split_fingerprint(demos,refs,tests)==old["split_fingerprint"]
            rows = []
            for x, lens in zip(tests[:40],old["lens"]["rows"][:40],strict=True):
                words = sorted(distinctive(x["object"],x["subject"]))
                if not words:
                    continue
                tops = [t.strip().lower() for t in lens["top_diff"]]
                row = dict(**x,words=words,gold_in_lens=any(len(t)>=3 and any(w.startswith(t) or t.startswith(w) for w in words) for t in tops))
                row["leak"] = leaks(row)
                rows.append(row)
            if len(rows)<5:
                sweep.write(target,dict(metadata=meta,relation=relation,skipped="fewer_than_five_scorable_rows",n=len(rows)))
                continue
            print("START",relation,len(rows),flush=True)
            for start in range(0,len(rows),8):
                batch = rows[start:start+8]
                continuations = [[] for _ in batch]
                for template in PROMPTS:
                    ids = tok([template.format(r["subject"]) for r in batch],return_tensors="pt",padding=True).to("cuda:0")
                    with torch.no_grad():
                        output = model.generate(**ids,max_new_tokens=90,do_sample=False,pad_token_id=tok.eos_token_id)
                    for texts, cont in zip(continuations,tok.batch_decode(output[:,ids.input_ids.shape[1]:]),strict=True):
                        texts.append(cont)
                for row,texts in zip(batch,continuations,strict=True):
                    cont = " ".join(texts).lower()
                    row.update(continuations=texts,hit=any(w in cont for w in row["words"]),strict=all(w in cont for w in row["words"]))
            sweep.write(target,dict(metadata=meta,relation=relation,source_sha256=sha(source),split_fingerprint=old["split_fingerprint"],n=len(rows),rows=rows))
            print("DONE",relation,flush=True)
    print("COMPLETE",a.name,flush=True)


if __name__ == "__main__":
    main()
