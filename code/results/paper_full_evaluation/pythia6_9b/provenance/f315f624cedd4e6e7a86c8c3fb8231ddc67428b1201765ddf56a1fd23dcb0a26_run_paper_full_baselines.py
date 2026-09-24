"""Full original-28 baseline readouts using the approved all-row protocol.

Reuse LRE grids and target predictions. Preserve every row and duplicate. Gender
alternatives apply to every readout; original continent references follow the
user's explicit choice. Score year exclusions only at report time. No row caps.
"""
import argparse
import hashlib
import json
import os
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens import model_sweep as sweep
from jlens.all_row_protocol import row_multiset
from jlens.measure_answer_prior import NEUTRAL_CONTEXTS
from run_targeted_corrections import TERMS

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/all_models_gender_original_continents"
NEWLINE = ROOT / "results/gptj_pythia_full_row_diagnosis"
OUT = ROOT / "results/paper_full_evaluation"
KINDS = ["diff", "lre", "raw_jacobian", "logit_lens", "differential_logit_lens"]


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--worker", required=True)
    p.add_argument("--batch-size", type=int, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    manifest = read(BASE/a.name/"manifest.json")
    prompt = "\n{} is known for" if a.name in ["gptj", "pythia6_9b"] else manifest["prompt"]
    assert len(manifest["dataset"]) == 28
    assert sum(d["n_original"] for d in manifest["dataset"].values()) == 9747
    dest = OUT/a.name
    tok = AutoTokenizer.from_pretrained(manifest["model"], revision=manifest["revision"], local_files_only=True)
    kwargs = dict(dtype=torch.float32, attn_implementation="eager", local_files_only=True, device_map={"": "cuda:0"})
    if torch.cuda.device_count() > 1:
        kwargs.update(device_map="balanced", max_memory={i: f"{int(torch.cuda.get_device_properties(i).total_memory/2**30)-11}GiB" for i in range(torch.cuda.device_count())})
    print("LOAD", a.name, flush=True)
    model = AutoModelForCausalLM.from_pretrained(manifest["model"], revision=manifest["revision"], **kwargs).eval().requires_grad_(False)
    assert all(p.device.type == "cuda" for p in model.parameters())
    model.config.use_cache = model.config.get_text_config().use_cache = False
    model.forward = partial(model.forward, use_cache=False, logits_to_keep=1)
    conv = sweep.enable_fast_qwen_convolution(model, tok, manifest["lens_layer"])
    verification = sweep.verify(model, tok, manifest["lens_layer"], a.batch_size)
    prefix = sweep.enable_prefix_cache(model, tok, manifest["lens_layer"], a.batch_size)
    hashes = {}
    for path in [Path(__file__), Path(sweep.__file__), ROOT/"jlens/model.py", ROOT/"jlens/prefix_intervention.py", ROOT/"scripts/run_targeted_corrections.py", ROOT/"jlens/measure_answer_prior.py"]:
        digest = sha(path)
        archive = dest/"provenance"/(digest+"_"+path.name)
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(path.read_bytes())
        hashes[str(path.relative_to(ROOT))] = digest
    sweep.write(dest/"workers"/(a.worker+".json"), dict(worker=a.worker, pid=os.getpid(), model=manifest["model"], revision=manifest["revision"], prompt=prompt, batch_size=a.batch_size, manifest_sha256=sha(BASE/a.name/"manifest.json"), verification=verification, convolution_check=conv, prefix_cache_check=prefix, source_sha256=hashes))
    with torch.no_grad():
        priors = []
        for context in NEUTRAL_CONTEXTS:
            text = context if context else tok.bos_token or tok.eos_token or "<|endoftext|>"
            ids = tok(text, return_tensors="pt").input_ids.to(model.get_input_embeddings().weight.device)
            priors.append(model(input_ids=ids).logits[0, -1].float().log_softmax(-1).cpu())
        prior = torch.stack(priors).mean(0).numpy()
    prior_path = dest/("answer_prior_"+a.worker+".npz")
    np.savez_compressed(prior_path, mean_log_probability=prior)
    print("VERIFIED", verification, flush=True)
    for rel in sorted(manifest["dataset"], key=lambda r: -manifest["dataset"][r]["n_original"]):
        target = dest/"measurements"/(rel+".json")
        with sweep.measurement_claim(target) as claimed:
            if not claimed or target.exists():
                continue
            start = time.time()
            old_path = BASE/a.name/"measurements"/(rel+".json")
            old = read(old_path)
            lens_path = (NEWLINE if a.name in ["gptj", "pythia6_9b"] else BASE)/a.name/"measurements"/(rel+".json")
            previous = read(lens_path)["lens"]
            plan = manifest["dataset"][rel]
            data_path = sweep.DATA/(rel+".json")
            data = read(data_path)
            assert sha(data_path) == old["data_sha256"] == plan["data_sha256"]
            split = tuple(plan["lens_split"][k] for k in ["demonstrations", "references", "evaluation"])
            assert row_multiset(split[2]) == row_multiset(data["samples"])
            assert previous["neutral_prompt"] == prompt
            for r, s in zip(previous["rows"], split[2], strict=True):
                assert (r["subject"], r["object"]) == (s["subject"], s["object"])
            print("START", rel, len(split[2]), flush=True)
            term_ids = {}
            if rel == "name_gender":
                for word in sum(TERMS.values(), []):
                    ids = tok.encode(" "+word, add_special_tokens=False)
                    assert len(ids) == 1
                    term_ids[word] = ids[0]
            call_index, term_ranks, top50 = 0, {}, []
            original_decode = sweep.decode

            def capture(m, norm, hidden):
                nonlocal call_index
                logits = original_decode(m, norm, hidden)
                if call_index < len(KINDS):
                    kind = KINDS[call_index]
                    if term_ids:
                        term_ranks[kind] = {w: (logits > logits[:, t:t+1]).sum(-1).cpu().tolist() for w, t in term_ids.items()}
                    if kind == "logit_lens":
                        top50.extend(logits.topk(50).indices.cpu().tolist())
                call_index += 1
                return logits

            sweep.decode = capture
            try:
                lens = sweep.measure_lens(model, tok, data["samples"], data["prompt_templates"][0], "main", manifest["lens_layer"], old["trailing_space"], a.batch_size, None, neutral_prompt=prompt, include_baselines=True, split_override=split, cached_targets=[r["model_top1_id"] for r in previous["rows"]])
            finally:
                sweep.decode = original_decode
            for i, (r, s, b) in enumerate(zip(lens["rows"], split[2], previous["rows"], strict=True)):
                r["object"] = s["object"]
                for key in ["subject", "model_top1", "model_top1_id", "n_references"]:
                    assert r[key] == b[key]
                r["logit_lens_top50_ids"] = top50[i]
                r["answer_prior"] = float(prior[r["model_top1_id"]])
                if term_ids:
                    group = next((words for words in TERMS.values() if r["model_top1"] in words), [])
                    r["gender_target_terms"] = group or [r["model_top1"]]
                    for kind in KINDS:
                        r["rank_"+kind+"_literal"] = r["rank_"+kind]
                        expanded = {w: term_ranks[kind][w][i] for w in group}
                        r["gender_ranks_"+kind] = expanded
                        r["rank_"+kind] = min([r["rank_"+kind]]+list(expanded.values()))
            for kind in KINDS:
                lens["agg"][kind] = sweep.aggregate([r["rank_"+kind] for r in lens["rows"]])
            deltas = [abs(r["rank_diff"]-b["rank_diff"]) for r, b in zip(lens["rows"], previous["rows"], strict=True)]
            aggregate_delta = abs(lens["agg"]["diff"]["median_lograng"]-previous["agg"]["diff"]["median_lograng"])
            assert aggregate_delta < .001, (rel, aggregate_delta)
            sweep.write(target, dict(relation=rel, worker=a.worker, data_sha256=sha(data_path), manifest_sha256=sha(BASE/a.name/"manifest.json"), lre_source=str(old_path.relative_to(ROOT)), lre_sha256=sha(old_path), approved_lens_source=str(lens_path.relative_to(ROOT)), approved_lens_sha256=sha(lens_path), n_original=len(data["samples"]), lens=lens, diff_replay_max_rank_delta=max(deltas), diff_replay_aggregate_delta=aggregate_delta, gender_alternatives_all_readouts=True, diverse_continent_references=False, seconds=time.time()-start))
            print("DONE", rel, round(time.time()-start, 1), flush=True)
    print("WORKER_COMPLETE", a.worker, flush=True)


if __name__ == "__main__":
    main()
