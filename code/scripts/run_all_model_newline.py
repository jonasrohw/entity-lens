"""Rerun each retained model's neutral lens with exactly one leading newline."""

import argparse
import hashlib
import json
import socket
import time
from functools import partial
from pathlib import Path

import torch
import transformers
from report_cross_model_audit import LABELS, records
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens import model_sweep as sweep
from jlens import splits

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/all_model_newline"
PROMPT = "\n{} is known for"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def specs():
    models = {
        m["name"]: m
        for m in json.loads((ROOT / "results/five_model_sweep/models.json").read_text())
    }
    revisions = json.loads(
        (ROOT / "results/five_model_sweep/revisions.json").read_text()
    )
    for name in ("gptj", "olmo3_7b"):
        meta = json.loads(
            (
                ROOT / f"results/cross_model_audit/raw/targets/{name}/metadata.json"
            ).read_text()
        )
        models[name] = dict(
            name=name, model_id=meta["model"], lens_layer=15 if name == "gptj" else 17
        )
        revisions[meta["model"]] = meta["revision"]
    for m in models.values():
        m["revision"] = revisions[m["model_id"]]
    return models


def baseline_files(name, protocol, relation):
    if name not in ("gptj", "olmo3_7b"):
        return [ROOT / f"results/five_model_sweep/{name}/{protocol}/{relation}.json"]
    folder = ROOT / "results" / ("olmo" if name == "olmo3_7b" else name) / "results"
    lens = folder / (
        "jacobian_lens.json"
        if protocol == "main"
        else "shared_split/jacobian_lens.json"
    )
    suffix = "_ts" if protocol == "main" and relation.startswith("presidents_") else ""
    if suffix:
        lens = folder / "jacobian_lens_years.json"
    return [lens, folder / protocol / f"faithfulness_{relation}{suffix}.json"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", choices=LABELS, required=True)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--root", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(2)
    spec = specs()[args.name]
    old_records = records()[args.name]
    tokenizer = AutoTokenizer.from_pretrained(
        spec["model_id"], revision=spec["revision"], local_files_only=True
    )
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
                i: f"{int(torch.cuda.get_device_properties(i).total_memory / 2**30) - 11}GiB"
                for i in range(torch.cuda.device_count())
            },
        )
    print("LOAD", spec, flush=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            spec["model_id"], revision=spec["revision"], **kwargs
        )
        .eval()
        .requires_grad_(False)
    )
    assert all(p.device.type == "cuda" for p in model.parameters())
    model.config.use_cache = False
    model.config.get_text_config().use_cache = False
    model.forward = partial(model.forward, use_cache=False, logits_to_keep=1)
    convolution = sweep.enable_fast_qwen_convolution(
        model, tokenizer, spec["lens_layer"]
    )
    verification = sweep.verify(model, tokenizer, spec["lens_layer"], args.batch_size)
    prefix = (
        sweep.enable_prefix_cache(model, tokenizer, spec["lens_layer"], args.batch_size)
        if model.config.get_text_config().model_type == "qwen3_5_text"
        else None
    )
    metadata = dict(
        **spec,
        prompt=PROMPT,
        host=socket.gethostname(),
        worker=args.worker,
        batch_size=args.batch_size,
        dtype="float32",
        attention="eager",
        torch=torch.__version__,
        transformers=transformers.__version__,
        verification=verification,
        convolution_check=convolution,
        prefix_cache_check=prefix,
        source_sha256={
            str(p.relative_to(ROOT)): sha(p)
            for p in (
                Path(__file__),
                ROOT / "scripts/report_cross_model_audit.py",
                Path(sweep.__file__),
                ROOT / "jlens/model.py",
                ROOT / "jlens/splits.py",
                ROOT / "jlens/prefix_intervention.py",
            )
        },
    )
    sweep.write(args.root / args.name / f"worker_{args.worker}.json", metadata)
    print("VERIFIED", verification, flush=True)
    for protocol in ("main", "shared_split"):
        # Large tasks first to avoid a long tail across cooperating processes.
        names = sorted(
            old_records[protocol], key=lambda n: -old_records[protocol][n]["lens"]["n"]
        )
        for relation in names:
            target = args.root / args.name / protocol / f"{relation}.json"
            with sweep.measurement_claim(target) as acquired:
                if not acquired or target.exists():
                    continue
                start = time.time()
                print("START", protocol, relation, flush=True)
                old = old_records[protocol][relation]
                assert old["lens"]["layer"] == spec["lens_layer"]
                data_path = sweep.DATA / f"{relation}.json"
                data = json.loads(data_path.read_text())
                trailing = sweep.trailing_space_needed(
                    tokenizer, relation, data["samples"]
                )
                if (
                    args.name == "gptj"
                    and protocol == "main"
                    and relation.startswith("presidents_")
                ):
                    trailing = True
                samples = (
                    data["samples"]
                    if protocol == "main"
                    else splits.filter_identifying_objects(
                        data["samples"], tokenizer, trailing
                    )
                )
                demos, refs, tests = sweep.partitions(samples, protocol, lens=True)
                positions, token_changes = {}, []
                for subject in dict.fromkeys(x["subject"] for x in refs + tests):
                    text = PROMPT.format(subject)
                    begin = text.rindex(subject)
                    end = begin + len(subject)
                    tokens = tokenizer(text, return_offsets_mapping=True)
                    expected = max(
                        i
                        for i, (a, b) in enumerate(tokens.offset_mapping)
                        if a < end and b > begin
                    )
                    actual = min(
                        len(tokenizer(text[:end]).input_ids) - 1,
                        len(tokens.input_ids) - 1,
                    )
                    assert actual == expected and actual > 0, (
                        subject,
                        actual,
                        expected,
                    )
                    positions[subject] = actual
                    # Audit content tokens separately from automatically added BOS/EOS.
                    old_ids = tokenizer(text[1:], add_special_tokens=False).input_ids
                    new_ids = tokenizer(text, add_special_tokens=False).input_ids
                    newline_ids = tokenizer("\n", add_special_tokens=False).input_ids
                    if new_ids != newline_ids + old_ids:
                        token_changes.append(
                            dict(
                                subject=subject,
                                original_ids=old_ids,
                                new_ids=new_ids,
                                newline_ids=newline_ids,
                            )
                        )
                lens = sweep.measure_lens(
                    model,
                    tokenizer,
                    samples,
                    data["prompt_templates"][0],
                    protocol,
                    spec["lens_layer"],
                    trailing,
                    args.batch_size,
                    100,
                    neutral_prompt=PROMPT,
                )
                keys = ["subject", "model_top1"]
                if "model_top1_id" in old["lens"]["rows"][0]:
                    keys += ["model_top1_id", "n_references"]
                assert [[r[k] for k in keys] for r in lens["rows"]] == [
                    [r[k] for k in keys] for r in old["lens"]["rows"]
                ], (args.name, protocol, relation, "target or subject mismatch")
                assert lens["split_fingerprint"] == old["lens"]["split_fingerprint"]
                sources = baseline_files(args.name, protocol, relation)
                sweep.write(
                    target,
                    dict(
                        relation=relation,
                        protocol=protocol,
                        model=args.name,
                        worker=args.worker,
                        lens=lens,
                        faithfulness=old["faithfulness"],
                        n_lre=old["n_test"],
                        baseline_lens=old["lens"],
                        trailing_space=trailing,
                        positions=positions,
                        content_tokenization_changes=token_changes,
                        split_fingerprint=splits.split_fingerprint(demos, refs, tests),
                        baseline_sha256={
                            str(p.relative_to(ROOT)): sha(p) for p in sources
                        },
                        data_sha256=sha(data_path),
                        elapsed_seconds=time.time() - start,
                    ),
                )
                print(
                    "DONE",
                    protocol,
                    relation,
                    round(time.time() - start, 1),
                    "token_changes",
                    len(token_changes),
                    flush=True,
                )
    print("COMPLETE", args.name, flush=True)


if __name__ == "__main__":
    main()
