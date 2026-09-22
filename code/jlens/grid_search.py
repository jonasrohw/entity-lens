"""Expanded Mistral grids with retained products and explicit selection splits."""

import json
import random
import socket
import time

import numpy as np
import torch

from jlens import model_sweep as sweep
from jlens import splits
from jlens.prefix_intervention import PrefixIntervention

BETAS = (
    [0.0, 0.0625, 0.125]
    + [i / 4 for i in range(1, 33)]
    + [10.0, 15.0, 20.0, 30.0, 50.0, 100.0]
)
REF_COUNTS = (4, 8)
PROMPTS = {"known_for": "{} is known for", "bare_subject": "{}"}


def state_modules(model):
    blocks, _ = sweep.layout(model)
    return [model.get_input_embeddings(), *blocks]


def cache(model, tokenizer, prompt, subject, layers):
    modules = state_modules(model)
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
        model.get_input_embeddings().weight.device
    )
    end = prompt.rindex(subject) + len(subject)
    position = min(len(tokenizer(prompt[:end]).input_ids) - 1, ids.shape[1] - 1)
    states = {}
    handles = []
    for layer in set(layers) | {len(modules) - 1}:

        def capture(module, inputs, output, index=layer):
            hidden = output[0] if isinstance(output, tuple) else output
            states[index] = hidden[0, position].detach().clone()
            if index == len(modules) - 1:
                states["final"] = hidden[0, -1].detach().clone()

        handles.append(modules[layer].register_forward_hook(capture))
    try:
        with torch.no_grad():
            result = model(input_ids=ids, use_cache=False, logits_to_keep=1)
            target = int(result.logits[0, -1].argmax())
    finally:
        for handle in handles:
            handle.remove()
    return dict(ids=ids, position=position, states=states, target=target)


def partition(samples, protocol, lens=False, seed=0):
    if seed == 0:
        return sweep.partitions(samples, protocol, lens=lens, n_test=100)
    pool = list(samples)
    random.Random((17 if lens and protocol == "main" else 0) + seed).shuffle(pool)
    if protocol == "main" and not lens:
        random.Random(1000 + seed).shuffle(pool)
    if protocol == "main" and lens:
        return pool[:8], pool[8:16], pool[8:28]
    return pool[:8], pool[8:16], pool[16:116]


def rank_summary(ranks):
    ranks = np.asarray(ranks, dtype=float)
    return dict(
        median_logrank=float(np.median(np.log10(ranks + 1))),
        mean_logrank=float(np.mean(np.log10(ranks + 1))),
        top1=float(np.mean(ranks == 0)),
        top10=float(np.mean(ranks < 10)),
        mrr=float(np.mean(1 / (ranks + 1))),
    )


def decode_grid(model, base, products, targets, betas, chunk_size=4):
    _, norm = sweep.layout(model)
    targets = torch.tensor(targets, device=products.device)
    output = {}
    with torch.no_grad():
        for start in range(0, len(betas), chunk_size):
            chunk = betas[start : start + chunk_size]
            scales = torch.tensor(chunk, device=products.device)[:, None, None]
            logits = sweep.decode(model, norm, base[None] + scales * products[None])
            target_logits = logits.gather(
                -1, targets[None, :, None].expand(len(chunk), -1, -1)
            )
            ranks = (logits > target_logits).sum(-1).cpu().tolist()
            hits = (logits.argmax(-1) == targets[None]).cpu().int().tolist()
            constant = (logits.amax(-1) == logits.amin(-1)).sum(-1).cpu().tolist()
            for beta, rank, hit, n_constant in zip(chunk, ranks, hits, constant):
                summary = rank_summary(rank)
                summary["optimistic_top1"] = summary["top1"]
                summary["top1"] = float(np.mean(hit))
                output[str(beta)] = dict(
                    ranks=rank, hits=hit, n_constant_outputs=n_constant, **summary
                )
    return output


def faith_grid(
    model,
    tokenizer,
    samples,
    template,
    protocol,
    layers,
    trailing,
    batch_size,
    betas=BETAS,
    seed=0,
    tensors=None,
):
    demos, refs, tests = partition(samples, protocol, seed=seed)
    context = "".join(
        template.format(x["subject"]) + " " + x["object"] + "\n" for x in demos
    )
    cached = {
        x["subject"]: cache(
            model,
            tokenizer,
            context + template.format(x["subject"]) + (" " if trailing else ""),
            x["subject"],
            layers,
        )
        for x in refs + tests
    }
    targets = [cached[x["subject"]]["target"] for x in tests]
    modules = state_modules(model)
    intervention = PrefixIntervention(model, modules[-1])
    grid = {}
    for layer in layers:
        products = torch.zeros(
            (len(tests), model.config.hidden_size),
            device=model.get_output_embeddings().weight.device,
        )
        base = torch.zeros(model.config.hidden_size, device=products.device)
        grid[str(layer)] = {}
        handle = modules[layer].register_forward_hook(intervention.hook)
        try:
            for index, sample in enumerate(refs, 1):
                c = cached[sample["subject"]]
                reference = c["states"][layer]
                directions = torch.stack(
                    [cached[x["subject"]]["states"][layer] - reference for x in tests]
                )
                products += sweep.jvps(
                    intervention,
                    reference,
                    directions,
                    c["ids"],
                    c["position"],
                    batch_size,
                )
                base += c["states"]["final"]
                if index in REF_COUNTS:
                    mean_products, mean_base = products / index, base / index
                    grid[str(layer)][str(index)] = decode_grid(
                        model, mean_base[None], mean_products, targets, betas
                    )
                    if tensors is not None:
                        tensors[f"faith/{layer}/{index}/products"] = mean_products.cpu()
                        tensors[f"faith/{layer}/{index}/base"] = mean_base.cpu()
        finally:
            handle.remove()
        print("FAITH_LAYER", protocol, layer, flush=True)
    return dict(
        grid=grid,
        subjects=[x["subject"] for x in tests],
        targets=targets,
        gold=[x["object"] for x in tests],
        n=len(tests),
        split_fingerprint=splits.split_fingerprint(demos, refs, tests),
    )


def lens_grid(
    model,
    tokenizer,
    samples,
    template,
    protocol,
    layers,
    trailing,
    batch_size,
    betas=BETAS,
    seed=0,
    tensors=None,
    prompts=PROMPTS,
):
    demos, refs, tests = partition(samples, protocol, lens=True, seed=seed)
    if len(tests) < 2:
        return dict(skipped="fewer_than_two_lens_subjects", n=len(tests))
    context = "".join(
        template.format(x["subject"]) + " " + x["object"] + "\n" for x in demos
    )
    targets = [
        cache(
            model,
            tokenizer,
            context + template.format(x["subject"]) + (" " if trailing else ""),
            x["subject"],
            [],
        )["target"]
        for x in tests
    ]
    modules = state_modules(model)
    all_prompts = {}
    for prompt_name, prompt in prompts.items():
        cached = {
            x["subject"]: cache(
                model, tokenizer, prompt.format(x["subject"]), x["subject"], layers
            )
            for x in refs + tests
        }
        grid = {}
        intervention = PrefixIntervention(model, modules[-1])
        for layer in layers:
            products = torch.zeros(
                (len(tests), model.config.hidden_size),
                device=model.get_output_embeddings().weight.device,
            )
            base = torch.zeros_like(products)
            counts = torch.zeros(len(tests), device=products.device)
            grid[str(layer)] = {}
            handle = modules[layer].register_forward_hook(intervention.hook)
            try:
                for index, sample in enumerate(refs, 1):
                    c = cached[sample["subject"]]
                    reference = c["states"][layer]
                    directions = torch.stack(
                        [
                            cached[x["subject"]]["states"][layer] - reference
                            for x in tests
                        ]
                    )
                    values = sweep.jvps(
                        intervention,
                        reference,
                        directions,
                        c["ids"],
                        c["position"],
                        batch_size,
                    )
                    valid = torch.tensor(
                        [x["subject"] != sample["subject"] for x in tests],
                        device=products.device,
                    )
                    products += values * valid[:, None]
                    base += c["states"]["final"][None] * valid[:, None]
                    counts += valid
                    if index in REF_COUNTS:
                        assert counts.min() > 0
                        mean_products, mean_base = (
                            products / counts[:, None],
                            base / counts[:, None],
                        )
                        affine = decode_grid(
                            model, mean_base, mean_products, targets, betas
                        )
                        differential = decode_grid(
                            model,
                            torch.zeros_like(mean_base),
                            mean_products,
                            targets,
                            [1.0],
                        )["1.0"]
                        grid[str(layer)][str(index)] = dict(
                            affine=affine, diff=differential
                        )
                        if tensors is not None:
                            key = f"lens/{prompt_name}/{layer}/{index}"
                            tensors[key + "/products"] = mean_products.cpu()
                            tensors[key + "/base"] = mean_base.cpu()
            finally:
                handle.remove()
            print("LENS_LAYER", protocol, prompt_name, layer, flush=True)
        all_prompts[prompt_name] = grid
    return dict(
        prompts=all_prompts,
        subjects=[x["subject"] for x in tests],
        targets=targets,
        n=len(tests),
        split_fingerprint=splits.split_fingerprint(demos, refs, tests),
    )


def run_worker(model, tokenizer, args):
    layers = list(range(len(state_modules(model))))
    relations = sorted(
        sweep.DATA.glob("*.json"),
        key=lambda p: -len(json.loads(p.read_text())["samples"]),
    )
    for protocol in ("main", "shared_split"):
        for source in relations:
            target = args.root / protocol / source.name
            with sweep.measurement_claim(target) as acquired:
                if not acquired or target.exists():
                    continue
                data = json.loads(source.read_text())
                samples = data["samples"]
                trailing = sweep.trailing_space_needed(tokenizer, source.stem, samples)
                if protocol == "shared_split":
                    samples = splits.filter_identifying_objects(
                        samples, tokenizer, trailing
                    )
                if len(samples) < 17:
                    sweep.write(
                        target,
                        dict(
                            relation=source.stem,
                            protocol=protocol,
                            skipped="too_few_samples",
                            n_samples=len(samples),
                        ),
                    )
                    continue
                print("START", protocol, source.stem, flush=True)
                start = time.time()
                tensors = {}
                faith = faith_grid(
                    model,
                    tokenizer,
                    samples,
                    data["prompt_templates"][0],
                    protocol,
                    layers,
                    trailing,
                    args.batch_size,
                    tensors=tensors,
                )
                lens = lens_grid(
                    model,
                    tokenizer,
                    samples,
                    data["prompt_templates"][0],
                    protocol,
                    layers,
                    trailing,
                    args.batch_size,
                    tensors=tensors,
                )
                tensor_path = target.with_suffix(".pt")
                temporary = tensor_path.with_suffix(f".{args.worker}.tmp")
                torch.save(tensors, temporary)
                temporary.replace(tensor_path)
                sweep.write(
                    target,
                    dict(
                        relation=source.stem,
                        protocol=protocol,
                        model_id="mistralai/Mistral-7B-v0.3",
                        revision=model.config._commit_hash,
                        dtype="float32",
                        layers=layers,
                        betas=BETAS,
                        ref_counts=REF_COUNTS,
                        seed=0,
                        host=socket.gethostname(),
                        worker=args.worker,
                        trailing_space=trailing,
                        faith=faith,
                        lens=lens,
                        elapsed_seconds=time.time() - start,
                    ),
                )
                print("DONE", protocol, source.stem, time.time() - start, flush=True)
    print("COMPLETE", args.worker, flush=True)
