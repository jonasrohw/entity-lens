"""Full-row newline diagnostic; reuse unchanged relation-context LRE and targets.

Original 28 relations, all 9,747 rows, original seeded reference sets, accepted
gender wordings. The user's explicit original-continent and original-28 choices
override the broader standing defaults. No row filtering or source mutation.
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
from run_targeted_corrections import TERMS, expanded_ranks

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/all_models_gender_original_continents"
OUT = ROOT / "results/gptj_pythia_full_row_diagnosis"
PROMPT = "\n{} is known for"


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", choices=["gptj", "pythia6_9b"], required=True)
    p.add_argument("--worker", required=True)
    p.add_argument("--batch-size", type=int, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    source = BASE / a.name
    dest = OUT / a.name
    manifest = read(source / "manifest.json")
    assert len(manifest["dataset"]) == 28
    assert sum(x["n_original"] for x in manifest["dataset"].values()) == 9747
    tok = AutoTokenizer.from_pretrained(manifest["model"], revision=manifest["revision"], local_files_only=True)
    print("LOAD", a.name, flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        manifest["model"], revision=manifest["revision"], local_files_only=True,
        dtype=torch.float32, attn_implementation="eager", device_map={"": "cuda:0"},
    ).eval().requires_grad_(False)
    model.config.use_cache = model.config.get_text_config().use_cache = False
    model.forward = partial(model.forward, use_cache=False, logits_to_keep=1)
    verification = sweep.verify(model, tok, manifest["lens_layer"], a.batch_size)
    prefix = sweep.enable_prefix_cache(model, tok, manifest["lens_layer"], a.batch_size)
    sources = [Path(__file__), Path(sweep.__file__), ROOT / "scripts/run_targeted_corrections.py", ROOT / "jlens/model.py", ROOT / "jlens/prefix_intervention.py"]
    hashes = {}
    for path in sources:
        digest = sha(path)
        archive = dest / "provenance" / f"{digest}_{path.name}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(path.read_bytes())
        hashes[str(path.relative_to(ROOT))] = digest
    sweep.write(dest / "workers" / f"{a.worker}.json", dict(
        model=manifest["model"], revision=manifest["revision"], worker=a.worker,
        batch_size=a.batch_size, pid=os.getpid(), verification=verification,
        prefix_cache_check=prefix, source_sha256=hashes,
        manifest_sha256=sha(source / "manifest.json"), dtype="float32", attention="eager",
    ))
    print("VERIFIED", verification, flush=True)
    # Cooperating processes claim whole relations; the completed experiment
    # must contain all 28 and every original row before reporting correlations.
    names = sorted(manifest["dataset"], key=lambda r: -manifest["dataset"][r]["n_original"])
    for rel in names:
        target = dest / "measurements" / f"{rel}.json"
        with sweep.measurement_claim(target) as claimed:
            if not claimed or target.exists():
                continue
            start = time.time()
            plan = manifest["dataset"][rel]
            old_path = source / "measurements" / f"{rel}.json"
            old = read(old_path)
            data_path = sweep.DATA / f"{rel}.json"
            data = read(data_path)
            assert sha(data_path) == old["data_sha256"] == plan["data_sha256"]
            assert old["lens_split"] == plan["lens_split"]
            split = tuple(plan["lens_split"][k] for k in ["demonstrations", "references", "evaluation"])
            assert row_multiset(split[2]) == row_multiset(data["samples"])
            assert not old["date_audit"]["excluded"]
            for x, row in zip(split[2], old["lens"]["rows"], strict=True):
                assert (x["subject"], x["object"]) == (row["subject"], row["object"])
            targets = [r["model_top1_id"] for r in old["lens"]["rows"]]
            print("START", rel, len(targets), flush=True)
            norms = {}
            original_cache = sweep.cache_prompt

            def capture_cache(m, tokenizer, prompt, subject, layers):
                before = original_cache(m, tokenizer, manifest["prompt"].format(subject), subject, layers)
                after = original_cache(m, tokenizer, prompt, subject, layers)
                enc = tokenizer(prompt, return_offsets_mapping=True)
                begin = prompt.rindex(subject)
                end = begin + len(subject)
                expected = max(i for i, (b, e) in enumerate(enc.offset_mapping) if b < end and e > begin)
                assert expected == after["position"] and expected > 0
                layer = manifest["lens_layer"]
                norms[subject] = dict(
                    original_position=before["position"], newline_position=after["position"],
                    original_subject_norm=float(before["states"][layer].norm()),
                    newline_subject_norm=float(after["states"][layer].norm()),
                    original_final_norm=float(before["states"]["final"].norm()),
                    newline_final_norm=float(after["states"]["final"].norm()),
                    original_ids=before["ids"].cpu().tolist()[0],
                    newline_ids=after["ids"].cpu().tolist()[0],
                )
                return after

            term_ids = {w: tok.encode(" " + w, add_special_tokens=False) for w in sum(TERMS.values(), [])} if rel == "name_gender" else {}
            assert all(len(v) == 1 for v in term_ids.values())
            term_ids = {k: v[0] for k, v in term_ids.items()}
            term_ranks, hidden_diagnostics = {}, {}
            original_decode = sweep.decode

            def capture_decode(m, norm, hidden):
                logits = original_decode(m, norm, hidden)
                if not hidden_diagnostics:
                    hidden_diagnostics.update(diff_norms=hidden.norm(dim=-1).cpu().tolist())
                    if term_ids:
                        term_ranks.update({w: (logits > logits[:, tid:tid+1]).sum(-1).cpu().tolist() for w, tid in term_ids.items()})
                return logits

            sweep.cache_prompt = capture_cache
            sweep.decode = capture_decode
            try:
                lens = sweep.measure_lens(
                    model, tok, data["samples"], data["prompt_templates"][0], "main",
                    manifest["lens_layer"], old["trailing_space"], a.batch_size, None,
                    neutral_prompt=PROMPT, split_override=split, cached_targets=targets,
                )
            finally:
                sweep.cache_prompt = original_cache
                sweep.decode = original_decode
            if term_ids:
                expanded_ranks(lens, term_ids, term_ranks)
            for row, x, previous in zip(lens["rows"], split[2], old["lens"]["rows"], strict=True):
                row["object"] = x["object"]
                for key in ["subject", "model_top1_id", "model_top1", "n_references"]:
                    assert row[key] == previous[key]
            assert lens["n"] == plan["n_original"]
            assert np.median(np.log10(np.array([r["rank_diff"] for r in lens["rows"]]) + 1)) == lens["agg"]["diff"]["median_lograng"]
            sweep.write(target, dict(
                relation=rel, worker=a.worker, n_original=lens["n"],
                data_sha256=sha(data_path), baseline_path=str(old_path.relative_to(ROOT)),
                baseline_sha256=sha(old_path), lens=lens, neutral_prompt_diagnostics=norms,
                hidden_diagnostics=hidden_diagnostics, elapsed_seconds=time.time()-start,
                lre_reused=True, targets_reused=True, gender_alternatives=True,
                diverse_continent_references=False,
            ))
            print("DONE", rel, round(time.time()-start, 1), flush=True)
    print("WORKER_COMPLETE", a.worker, flush=True)


if __name__ == "__main__":
    main()
