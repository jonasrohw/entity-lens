"""Reproduce model targets and measure prior-only LRE agreement, without JVPs."""

import argparse
import json
import math
import socket
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens import model_sweep as sweep
from jlens import splits

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / "generated/cross_model_audit"


def original(name, relation, protocol):
    if name in {"olmo3_7b", "gptj"}:
        model = "olmo" if name == "olmo3_7b" else "gptj"
        folder = BASE / "results" / model / "results"
        suffix = (
            "_ts" if protocol == "main" and relation.startswith("presidents_") else ""
        )
        path = (
            folder
            / ("main" if protocol == "main" else "shared_split")
            / f"faithfulness_{relation}{suffix}.json"
        )
        return json.loads(path.read_text()) if path.exists() else None
    path = BASE / "results/five_model_sweep" / name / protocol / f"{relation}.json"
    return json.loads(path.read_text()).get("faithfulness")


def token_audit(tokenizer):
    count = 0
    differences = []
    years = {}
    for path in sorted(sweep.DATA.glob("*.json")):
        data = json.loads(path.read_text())
        for sample in data["samples"]:
            subject = sample["subject"]
            for template in [data["prompt_templates"][0], "{} is known for"]:
                prompt = template.format(subject)
                start = prompt.rindex(subject)
                end = start + len(subject)
                tokens = tokenizer(prompt, return_offsets_mapping=True)
                actual = min(
                    len(tokenizer(prompt[:end]).input_ids) - 1,
                    len(tokens.input_ids) - 1,
                )
                expected = max(
                    i
                    for i, (a, b) in enumerate(tokens.offset_mapping)
                    if a < end and b > start
                )
                count += 1
                if actual != expected:
                    differences.append([path.stem, subject, template, actual, expected])
        if path.stem.startswith("presidents_"):
            years[path.stem] = {
                obj: {
                    "leading_space": tokenizer.tokenize(" " + obj),
                    "bare": tokenizer.tokenize(obj),
                }
                for obj in sorted({s["object"] for s in data["samples"]})
            }
    return dict(
        positions_checked=count, differences=differences, year_tokenization=years
    )


def measure(model, tokenizer, name, relation, data, protocol):
    samples = data["samples"]
    trailing = sweep.trailing_space_needed(tokenizer, relation, samples)
    if name == "gptj" and protocol == "main" and relation.startswith("presidents_"):
        trailing = True  # Match the saved GPT-J prompt variant, explicitly.
    if protocol == "shared_split":
        samples = splits.filter_identifying_objects(samples, tokenizer, trailing)
    previous = original(name, relation, protocol)
    if previous is None:
        return dict(skipped="excluded_from_original_faith_run")
    demos, refs, tests = sweep.partitions(samples, protocol)
    fingerprint = splits.split_fingerprint(demos, refs, tests)
    assert fingerprint == previous["split_fingerprint"], (name, relation, protocol)
    template = data["prompt_templates"][0]
    context = "".join(
        template.format(s["subject"]) + " " + s["object"] + "\n" for s in demos
    )
    final_states = []
    targets = []
    for i, sample in enumerate(refs + tests):
        prompt = (
            context + template.format(sample["subject"]) + (" " if trailing else "")
        )
        c = sweep.cache_prompt(model, tokenizer, prompt, sample["subject"], [])
        if i < len(refs):
            final_states.append(c["states"]["final"])
        else:
            targets.append(c["target"])
    _, norm = sweep.layout(model)
    prior_id = int(
        sweep.decode(model, norm, torch.stack(final_states).mean(0)).argmax()
    )
    gold_ids = [
        splits.first_object_token(s["object"], tokenizer, trailing) for s in tests
    ]
    correct = [int(a == b) for a, b in zip(targets, gold_ids)]
    legacy_gold_ids = [
        tokenizer.encode(
            ("" if trailing else " ") + s["object"], add_special_tokens=False
        )[0]
        for s in tests
    ]
    legacy_correct = [int(a == b) for a, b in zip(targets, legacy_gold_ids)]
    assert sum(legacy_correct) == previous["lm_correct"], (
        name,
        relation,
        protocol,
        sum(legacy_correct),
        previous["lm_correct"],
    )
    hits = [previous["per_subject"][s["subject"]] for s in tests]
    assert np.mean(hits) == previous["best_faith"]
    counts = Counter(targets)
    n = len(tests)
    return dict(
        relation=relation,
        protocol=protocol,
        n=n,
        split_fingerprint=fingerprint,
        trailing_space=trailing,
        subjects=[s["subject"] for s in tests],
        targets=targets,
        target_tokens=[tokenizer.decode([t]) for t in targets],
        gold_ids=gold_ids,
        correct=correct,
        legacy_gold_ids=legacy_gold_ids,
        legacy_correct=legacy_correct,
        selected_lre_hits=hits,
        prior_id=prior_id,
        prior_token=tokenizer.decode([prior_id]),
        prior_faith=float(np.mean(np.asarray(targets) == prior_id)),
        lm_accuracy=float(np.mean(correct)),
        target_unique=len(counts),
        target_modal_fraction=max(counts.values()) / n,
        target_entropy_bits=-sum(c / n * math.log2(c / n) for c in counts.values()),
        selected_faith=previous["best_faith"],
        gold_correct_subject_faith=float(
            np.mean(np.asarray(hits)[np.asarray(correct, dtype=bool)])
        )
        if any(correct)
        else None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    audit = token_audit(tokenizer)
    assert not audit["differences"]
    sweep.write(ROOT / "targets" / args.name / "tokenizer.json", audit)
    kwargs = dict(
        dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
        device_map={"": "cuda:0"},
    )
    if torch.cuda.device_count() > 1:
        kwargs.update(
            device_map="balanced",
            max_memory={
                i: f"{int(torch.cuda.get_device_properties(i).total_memory / 2**30) - 10}GiB"
                for i in range(torch.cuda.device_count())
            },
        )
    model = (
        AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
        .eval()
        .requires_grad_(False)
    )
    assert all(p.device.type == "cuda" for p in model.parameters())
    model.config.use_cache = False
    model.config.get_text_config().use_cache = False
    model.forward = partial(model.forward, use_cache=False, logits_to_keep=1)
    sweep.write(
        ROOT / "targets" / args.name / "metadata.json",
        dict(
            model=args.model,
            host=socket.gethostname(),
            dtype="float32",
            revision=Path(
                hf_hub_download(args.model, "config.json", local_files_only=True)
            ).parent.name,
            note="Ordinary HF forwards; no convolution substitution, prefix cache, or Jacobian approximation.",
        ),
    )
    with torch.no_grad():
        for protocol in ["main", "shared_split"]:
            for path in sorted(sweep.DATA.glob("*.json")):
                target = ROOT / "targets" / args.name / protocol / path.name
                with sweep.measurement_claim(target) as acquired:
                    if not acquired or target.exists():
                        continue
                    print("TARGET_START", protocol, path.stem, flush=True)
                    result = measure(
                        model,
                        tokenizer,
                        args.name,
                        path.stem,
                        json.loads(path.read_text()),
                        protocol,
                    )
                    sweep.write(target, result)
                    print(
                        "TARGET_DONE",
                        protocol,
                        path.stem,
                        result.get("prior_faith"),
                        flush=True,
                    )
    print("TARGET_COMPLETE", args.name, flush=True)


if __name__ == "__main__":
    main()
