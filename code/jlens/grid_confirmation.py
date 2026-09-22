"""Re-fit selected configurations on a fresh split without reselecting them."""

import json
import time

import numpy as np

from jlens import grid_search as grid
from jlens import model_sweep as sweep
from jlens import splits


def unique_unambiguous_subjects(samples):
    """Collapse identical rows and exclude subjects with multiple gold objects."""
    objects = {}
    first = {}
    for sample in samples:
        subject = sample["subject"]
        objects.setdefault(subject, set()).add(sample["object"])
        first.setdefault(subject, sample)
    return [sample for subject, sample in first.items() if len(objects[subject]) == 1]


def measure(model, tokenizer, data, relation, config, batch_size):
    samples = data["samples"]
    input_count = len(samples)
    subject_policy = config.get("subject_policy", "original_rows")
    if subject_policy == "unique_unambiguous_subjects":
        samples = unique_unambiguous_subjects(samples)
    elif subject_policy != "original_rows":
        raise ValueError(subject_policy)
    cleaned_count = len(samples)
    template = data["prompt_templates"][0]
    trailing = sweep.trailing_space_needed(tokenizer, relation, samples)
    protocol = config.get("protocol", "main")
    if protocol == "shared_split":
        samples = splits.filter_identifying_objects(samples, tokenizer, trailing)
    faith_configs = config["faith_configs"][relation]
    faith_layers = sorted({cell["layer"] for cell in faith_configs.values()})
    faith_betas = sorted({cell["beta"] for cell in faith_configs.values()} | {0.0})
    seed = config["seed"]
    faith = grid.faith_grid(
        model,
        tokenizer,
        samples,
        template,
        protocol,
        faith_layers,
        trailing,
        batch_size,
        faith_betas,
        seed=seed,
    )
    lens_configs = config["lens_configs"]
    lens_layers = sorted({cell["layer"] for cell in lens_configs.values()})
    lens_betas = sorted(
        {cell["beta"] for cell in lens_configs.values() if cell["beta"] is not None}
        | {1.0}
    )
    prompts = {
        name: grid.PROMPTS[name]
        for name in {cell["prompt"] for cell in lens_configs.values()}
    }
    lens = grid.lens_grid(
        model,
        tokenizer,
        samples,
        template,
        protocol,
        lens_layers,
        trailing,
        batch_size,
        lens_betas,
        seed=seed,
        prompts=prompts,
    )
    faith_results = {}
    for name, cell in faith_configs.items():
        result = faith["grid"][str(cell["layer"])]["8"][str(cell["beta"])]
        faith_results[name] = dict(
            layer=cell["layer"],
            beta=cell["beta"],
            score=float(np.mean(result["hits"])),
            hits=result["hits"],
        )
    lens_results = {}
    for name, cell in lens_configs.items():
        values = lens["prompts"][cell["prompt"]][str(cell["layer"])][
            str(cell["references"])
        ]
        result = (
            values["diff"]
            if cell["readout"] == "diff"
            else values["affine"][str(cell["beta"])]
        )
        lens_results[name] = dict(config=cell, result=result)
    gold_ids = [
        splits.first_object_token(gold, tokenizer, trailing) for gold in faith["gold"]
    ]
    return dict(
        relation=relation,
        protocol=protocol,
        subject_policy=subject_policy,
        input_samples=input_count,
        cleaned_samples=cleaned_count,
        seed=seed,
        faith=faith_results,
        lens=lens_results,
        lm_accuracy=float(
            np.mean(np.asarray(gold_ids) == np.asarray(faith["targets"]))
        ),
        prior_only_faith=float(
            np.mean(faith["grid"][str(faith_layers[0])]["8"]["0.0"]["hits"])
        ),
        n_faith=faith["n"],
        n_lens=lens["n"],
        faith_subjects=faith["subjects"],
        lens_subjects=lens["subjects"],
        faith_split_fingerprint=faith["split_fingerprint"],
        lens_split_fingerprint=lens["split_fingerprint"],
    )


def run_worker(model, tokenizer, args):
    path = args.root / "confirmation_config.json"
    while not path.exists():
        time.sleep(2)
    config = json.loads(path.read_text())
    for source in sorted(sweep.DATA.glob("*.json")):
        target = args.root / "confirmation" / source.name
        with sweep.measurement_claim(target) as acquired:
            if not acquired or target.exists():
                continue
            if source.stem not in config["faith_configs"]:
                sweep.write(
                    target,
                    dict(
                        relation=source.stem,
                        skipped="not_in_frozen_configuration_cohort",
                    ),
                )
                continue
            print("CONFIRM_START", source.stem, flush=True)
            start = time.time()
            result = measure(
                model,
                tokenizer,
                json.loads(source.read_text()),
                source.stem,
                config,
                args.batch_size,
            )
            result.update(worker=args.worker, elapsed_seconds=time.time() - start)
            sweep.write(target, result)
            print("CONFIRM_DONE", source.stem, time.time() - start, flush=True)
    print("CONFIRM_COMPLETE", args.worker, flush=True)
