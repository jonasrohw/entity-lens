"""Resumable, relation-sharded model comparisons with reusable Jacobian graphs.

Each worker keeps one model loaded. Each reference/layer needs one forward and
one reverse graph; batches of subject directions share it. Protocol sizes and
the layer/beta selection rule match the original measurement commands.
"""

import argparse
import fcntl
import json
import os
import random
import socket
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens import affine_beta, splits
from jlens.model import HiddenStateIntervention, resolve_blocks, resolve_final_norm
from jlens.prefix_intervention import PrefixIntervention

BETAS = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0)
DATA = Path(__file__).resolve().parents[1] / "data/relations"


@contextmanager
def measurement_claim(path, enabled=True):
    """An NFS advisory lock, automatically released if a worker exits."""
    if not enabled:
        yield True
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def causal_conv1d_shifts(hidden_states, weight, bias=None, activation=None, **kwargs):
    """HF's short depthwise causal convolution without slow grouped-conv AD.

    For a width-four kernel this is four shifted elementwise products. It
    supports ordinary PyTorch double backward and avoids the pathological
    grouped-convolution batching fallback on the hybrid Qwen models.
    """
    from transformers.activations import ACT2FN

    length = hidden_states.shape[-1]
    padded = F.pad(hidden_states.to(weight.dtype), (weight.shape[-1] - 1, 0))
    out = padded[:, :, :length] * weight[None, :, 0, None]
    for offset in range(1, weight.shape[-1]):
        out = (
            out + padded[:, :, offset : offset + length] * weight[None, :, offset, None]
        )
    if bias is not None:
        out = out + bias[None, :, None]
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


def enable_fast_qwen_convolution(model, tokenizer, layer):
    if model.config.get_text_config().model_type != "qwen3_5_text":
        return None
    from transformers.models.qwen3_5 import modeling_qwen3_5 as implementation

    original = implementation.causal_conv1d_fn
    blocks, _ = layout(model)
    prompt = "France is known for"
    before = cache_prompt(model, tokenizer, prompt, "France", [layer])
    reference = before["states"][layer]
    torch.manual_seed(42)
    direction = torch.randn_like(reference)
    intervention = HiddenStateIntervention(model, blocks[-1])
    handle = blocks[layer - 1].register_forward_hook(intervention.hook)
    try:
        old_product = intervention.jvp(
            reference, direction, before["ids"], before["position"]
        )
        implementation.causal_conv1d_fn = causal_conv1d_shifts
        new_product = intervention.jvp(
            reference, direction, before["ids"], before["position"]
        )
        torch.testing.assert_close(new_product, old_product, rtol=1e-3, atol=2e-4)
    except Exception:
        implementation.causal_conv1d_fn = original
        raise
    finally:
        handle.remove()
    after = cache_prompt(model, tokenizer, prompt, "France", [layer])
    torch.testing.assert_close(after["logits"], before["logits"], rtol=1e-4, atol=1e-3)
    return dict(
        logits_max_abs_error=float((after["logits"] - before["logits"]).abs().max()),
        jvp_relative_error=float(
            (new_product - old_product).norm() / old_product.norm()
        ),
    )


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def layout(model):
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox.layers, model.gpt_neox.final_layer_norm
    return resolve_blocks(model), resolve_final_norm(model)


def decode(model, norm, hidden):
    # Use the actual HF module: Gemma2 and Qwen3.8 use (1 + weight) RMSNorm.
    normalized = norm(hidden.to(norm.weight.device))
    head = model.get_output_embeddings()
    logits = head(normalized.to(head.weight.device)).float()
    cap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
    return cap * torch.tanh(logits / cap) if cap is not None else logits


def cache_prompt(model, tokenizer, prompt, subject, layers):
    blocks, _ = layout(model)
    tokens = tokenizer(prompt, return_tensors="pt")
    ids = tokens.input_ids.to(model.get_input_embeddings().weight.device)
    end = prompt.rindex(subject) + len(subject)
    position = min(len(tokenizer(prompt[:end]).input_ids) - 1, ids.shape[1] - 1)
    captured = {}
    handles = []

    def capture(layer):
        def hook(module, inputs, output):
            states = output[0] if isinstance(output, tuple) else output
            captured[layer] = states[0, position].detach().clone()
            if layer == len(blocks):
                captured["final"] = states[0, -1].detach().clone()

        return hook

    for layer in sorted(set(layers) | {len(blocks)}):
        handles.append(blocks[layer - 1].register_forward_hook(capture(layer)))
    try:
        with torch.no_grad():
            output = model(input_ids=ids, use_cache=False, logits_to_keep=1)
            logits = (
                output.logits[0, -1]
                .detach()
                .to(model.get_output_embeddings().weight.device)
            )
    finally:
        for handle in handles:
            handle.remove()
    return dict(
        ids=ids,
        position=position,
        states=captured,
        target=int(logits.argmax()),
        logits=logits,
    )


def jvps(intervention, reference, directions, ids, position, batch_size=2):
    """Exact double-backward products, reusing a graph across directions."""
    hidden = reference.detach().clone().requires_grad_(True)
    output = intervention.forward(hidden, ids, position)
    cotangent = torch.zeros_like(output, requires_grad=True)
    gradient = torch.autograd.grad(
        output, hidden, grad_outputs=cotangent, create_graph=True
    )[0]
    chunks = []
    for start in range(0, len(directions), batch_size):
        batch = directions[start : start + batch_size]
        keep = start + batch_size < len(directions)
        product = torch.autograd.grad(
            gradient,
            cotangent,
            grad_outputs=batch,
            is_grads_batched=True,
            retain_graph=keep,
        )[0]
        chunks.append(product.detach())
    return torch.cat(chunks)


def make_intervention(model, final_block):
    cls = (
        PrefixIntervention
        if getattr(model, "_sweep_prefix_cache", False)
        else HiddenStateIntervention
    )
    return cls(model, final_block)


def enable_prefix_cache(model, tokenizer, layer, batch_size=2):
    """Gate suffix caching against full-prompt HF derivatives before enabling."""
    blocks, _ = layout(model)
    checks = []
    for prompt, subject, state_layer in [
        ("France is known for", "France", layer),
        (
            "The capital of Germany is Berlin.\n" * 8 + "The capital of France is",
            "France",
            10,
        ),
    ]:
        c = cache_prompt(model, tokenizer, prompt, subject, [state_layer])
        reference = c["states"][state_layer]
        torch.manual_seed(43)
        directions = torch.randn(
            max(3, batch_size), reference.numel(), device=reference.device
        )
        outputs = []
        timings = []
        for cls in (HiddenStateIntervention, PrefixIntervention):
            intervention = cls(model, blocks[-1])
            handle = blocks[state_layer - 1].register_forward_hook(intervention.hook)
            start = time.time()
            try:
                with torch.no_grad():
                    state = intervention.forward(reference, c["ids"], c["position"])
                products = jvps(
                    intervention,
                    reference,
                    directions,
                    c["ids"],
                    c["position"],
                    batch_size,
                )
                torch.cuda.synchronize()
                outputs.append((state, products))
                timings.append(time.time() - start)
            finally:
                handle.remove()
        for original, cached in zip(*outputs):
            torch.testing.assert_close(cached, original, rtol=1e-3, atol=1e-3)
        checks.append(
            dict(
                length=c["ids"].shape[1],
                prefix_length=c["position"],
                layer=state_layer,
                state_relative_error=float(
                    (outputs[1][0] - outputs[0][0]).norm() / outputs[0][0].norm()
                ),
                jvp_relative_error=float(
                    (outputs[1][1] - outputs[0][1]).norm() / outputs[0][1].norm()
                ),
                full_seconds=timings[0],
                cached_seconds=timings[1],
            )
        )
    model._sweep_prefix_cache = True
    return checks


def verify(model, tokenizer, layer, batch_size):
    blocks, norm = layout(model)
    cache = cache_prompt(model, tokenizer, "France is known for", "France", [layer])
    raw = cache["states"]["final"]
    with torch.no_grad():
        logits = decode(model, norm, raw)
    torch.testing.assert_close(logits, cache["logits"], rtol=1e-4, atol=2e-4)
    reference = cache["states"][layer]
    torch.manual_seed(41)
    directions = torch.randn(
        max(3, batch_size), reference.numel(), device=reference.device
    )
    directions *= reference.norm() / directions.norm(dim=1, keepdim=True)
    intervention = make_intervention(model, blocks[-1])
    handle = blocks[layer - 1].register_forward_hook(intervention.hook)
    try:
        with torch.no_grad():
            noop = intervention.forward(reference, cache["ids"], cache["position"])
        torch.testing.assert_close(noop, raw, rtol=0, atol=0)
        batched = jvps(
            intervention,
            reference,
            directions,
            cache["ids"],
            cache["position"],
            batch_size,
        )
        scalar = intervention.jvp(
            reference, directions[0], cache["ids"], cache["position"]
        )
        torch.testing.assert_close(batched[0], scalar, rtol=3e-4, atol=1e-4)
        eps = 0.003
        with torch.no_grad():
            plus = intervention.forward(
                reference + eps * directions[0], cache["ids"], cache["position"]
            )
            minus = intervention.forward(
                reference - eps * directions[0], cache["ids"], cache["position"]
            )
        error = float(((plus - minus) / (2 * eps) - scalar).norm() / scalar.norm())
        assert error < 0.02, error
    finally:
        handle.remove()
    return dict(
        logits_max_abs_error=float((logits - cache["logits"]).abs().max()),
        batched_scalar_max_abs_error=float((batched[0] - scalar).abs().max()),
        finite_difference_relative_error=error,
    )


def trailing_space_needed(tokenizer, relation, samples):
    if not relation.startswith("presidents_"):
        return False
    first = [
        tokenizer.decode([splits.first_object_token(x["object"], tokenizer)])
        for x in samples
    ]
    return all(not token.strip() for token in first)


def partitions(samples, protocol, *, lens=False, n_test=100):
    if protocol == "main":
        pool = list(samples)
        random.Random(17 if lens else 0).shuffle(pool)
        if not lens:
            random.Random(1000).shuffle(pool)
        demos = pool[:8]
        if lens:
            tests = pool[8:28]
            return demos, tests[:8], tests
        return demos, pool[8:16], pool[16 : 16 + n_test]
    return splits.split_samples(samples, n_test=n_test)


def measure_faith(
    model, tokenizer, samples, template, protocol, layers, trailing, batch_size, n_test
):
    blocks, norm = layout(model)
    demos, refs, tests = partitions(samples, protocol, n_test=n_test)
    context = "".join(
        template.format(x["subject"]) + " " + x["object"] + "\n" for x in demos
    )
    cache = {
        x["subject"]: cache_prompt(
            model,
            tokenizer,
            context + template.format(x["subject"]) + (" " if trailing else ""),
            x["subject"],
            layers,
        )
        for x in refs + tests
    }
    targets = torch.tensor(
        [cache[x["subject"]]["target"] for x in tests],
        device=model.get_output_embeddings().weight.device,
    )
    base = torch.stack([cache[x["subject"]]["states"]["final"] for x in refs]).mean(0)
    grid, cells = {}, {}
    intervention = make_intervention(model, blocks[-1])
    for layer in layers:
        handle = blocks[layer - 1].register_forward_hook(intervention.hook)
        try:
            products = torch.zeros((len(tests), base.numel()), device=base.device)
            for x in refs:
                c = cache[x["subject"]]
                reference = c["states"][layer]
                directions = torch.stack(
                    [cache[t["subject"]]["states"][layer] - reference for t in tests]
                )
                products += jvps(
                    intervention,
                    reference,
                    directions,
                    c["ids"],
                    c["position"],
                    batch_size,
                )
            products /= len(refs)
        finally:
            handle.remove()
        grid[str(layer)] = {}
        for beta in BETAS:
            with torch.no_grad():
                pred = decode(model, norm, base[None] + beta * products).argmax(-1)
            hits = (pred == targets).int().tolist()
            score = float(np.mean(hits))
            grid[str(layer)][str(beta)] = [score, 0.0]
            cells[layer, beta] = hits
        print(
            "LAYER",
            protocol,
            layer,
            max(v[0] for v in grid[str(layer)].values()),
            flush=True,
        )
    best_layer, best_beta = max(cells, key=lambda cell: np.mean(cells[cell]))
    correct = sum(
        cache[x["subject"]]["target"]
        == splits.first_object_token(x["object"], tokenizer, trailing)
        for x in tests
    )
    return dict(
        grid=grid,
        best_layer=best_layer,
        best_beta=best_beta,
        best_faith=float(np.mean(cells[best_layer, best_beta])),
        best_sd=0.0,
        trials=1,
        n_test=len(tests),
        requested_n_test=n_test,
        lm_correct=correct,
        per_subject={
            x["subject"]: h for x, h in zip(tests, cells[best_layer, best_beta])
        },
        split_fingerprint=splits.split_fingerprint(demos, refs, tests),
        legacy_split=protocol == "main",
        distinct_first_token=protocol != "main",
    )


def aggregate(ranks):
    r = np.asarray(ranks, dtype=float)
    return dict(
        median_rank=float(np.median(r)),
        median_lograng=float(np.median(np.log10(r + 1))),
        agree_at1=float((r == 0).mean()),
        agree_at10=float((r < 10).mean()),
    )


def measure_lens(
    model,
    tokenizer,
    samples,
    template,
    protocol,
    layer,
    trailing,
    batch_size,
    n_test,
    *,
    neutral_prompt="{} is known for",
):
    blocks, norm = layout(model)
    demos, refs, tests = partitions(samples, protocol, lens=True, n_test=n_test)
    if len(tests) < 2:
        return dict(skipped="fewer_than_two_lens_subjects", n=len(tests))
    context = "".join(
        template.format(x["subject"]) + " " + x["object"] + "\n" for x in demos
    )
    targets = []
    with torch.no_grad():
        for x in tests:
            ids = tokenizer(
                context + template.format(x["subject"]) + (" " if trailing else ""),
                return_tensors="pt",
            ).input_ids.to(model.get_input_embeddings().weight.device)
            targets.append(
                int(
                    model(input_ids=ids, use_cache=False, logits_to_keep=1)
                    .logits[0, -1]
                    .argmax()
                )
            )
    names = list(dict.fromkeys(x["subject"] for x in refs + tests))
    cache = {
        s: cache_prompt(model, tokenizer, neutral_prompt.format(s), s, [layer])
        for s in names
    }
    output_device = model.get_output_embeddings().weight.device
    width = cache[names[0]]["states"]["final"].numel()
    diff = torch.zeros(len(tests), width, device=output_device)
    affine = torch.zeros_like(diff)
    counts = torch.zeros(len(tests), device=output_device)
    intervention = make_intervention(model, blocks[-1])
    handle = blocks[layer - 1].register_forward_hook(intervention.hook)
    try:
        for x in refs:
            s = x["subject"]
            c = cache[s]
            ref = c["states"][layer]
            directions = torch.stack(
                [cache[t["subject"]]["states"][layer] - ref for t in tests]
            )
            v = jvps(
                intervention, ref, directions, c["ids"], c["position"], batch_size
            ).to(output_device)
            valid = torch.tensor(
                [t["subject"] != s for t in tests], device=output_device
            )
            diff += v * valid[:, None]
            affine += (v + c["states"]["final"].to(output_device)) * valid[:, None]
            counts += valid
    finally:
        handle.remove()
    diff /= counts[:, None]
    affine /= counts[:, None]
    target_tensor = torch.tensor(targets, device=output_device)
    rows = [
        dict(
            subject=t["subject"],
            model_top1_id=tid,
            model_top1=tokenizer.decode([tid]).strip(),
            n_references=int(count),
        )
        for t, tid, count in zip(tests, targets, counts)
    ]
    agg = {}
    with torch.no_grad():
        for kind, hidden in [("diff", diff), ("lre", affine)]:
            logits = decode(model, norm, hidden)
            ranks = (logits > logits.gather(1, target_tensor[:, None])).sum(-1).tolist()
            tops = logits.topk(5).indices.tolist()
            agg[kind] = aggregate(ranks)
            for row, rank, top in zip(rows, ranks, tops):
                row["rank_" + kind] = rank
                row["top_" + kind] = [tokenizer.decode([i]).strip() for i in top]
        for row in rows:
            row["rank_lre_beta"] = {"1.0": row["rank_lre"]}
        for beta in BETAS:
            if beta == 1:
                continue
            logits = decode(model, norm, affine + (beta - 1) * diff)
            ranks = (logits > logits.gather(1, target_tensor[:, None])).sum(-1).tolist()
            for row, rank in zip(rows, ranks):
                row["rank_lre_beta"][str(beta)] = rank
    return dict(
        n=len(tests),
        layer=layer,
        template=template,
        agg=agg,
        rows=rows,
        beta_sweep=affine_beta.summarize(rows, BETAS, "top1"),
        split_fingerprint=None
        if protocol == "main"
        else splits.split_fingerprint(demos, refs, tests),
        legacy_split=protocol == "main",
        neutral_prompt=neutral_prompt,
    )


def summarize(root, verbose=True):
    report = {}
    expected_relations = {p.stem for p in DATA.glob("*.json")}
    for model_dir in sorted(Path(root).glob("*")):
        if not model_dir.is_dir():
            continue
        variants = {}
        for protocol in ("main", "shared_split"):
            paths = sorted((model_dir / protocol).glob("*.json"))
            rows = []
            for p in paths:
                d = json.loads(p.read_text())
                if "faithfulness" in d and "agg" in d.get("lens", {}):
                    rows.append(d)
            if not rows:
                continue
            y = [d["faithfulness"]["best_faith"] for d in rows]
            correlations = {}
            for kind in ("diff", "lre"):
                r, p = spearmanr(
                    [-d["lens"]["agg"][kind]["median_lograng"] for d in rows], y
                )
                correlations[kind] = dict(rho=float(r), p=float(p))
            for metric in affine_beta.SELECTION_METRICS:
                r, p = spearmanr(
                    [
                        affine_beta.tuned_score(d["lens"]["beta_sweep"], metric)[1]
                        for d in rows
                    ],
                    y,
                )
                correlations["affine_tuned_" + metric] = dict(
                    rho=float(r), p=float(p), in_sample=True
                )
            fixed_beta = {}
            for beta in BETAS:
                r, p = spearmanr(
                    [
                        -d["lens"]["beta_sweep"]["agg"][str(beta)]["median_lograng"]
                        for d in rows
                    ],
                    y,
                )
                fixed_beta[str(beta)] = dict(rho=float(r), p=float(p))
            variants[protocol] = dict(
                n_relations=len(rows),
                complete={p.stem for p in paths} == expected_relations,
                n_finished=len(paths),
                correlations=correlations,
                fixed_beta_correlations=fixed_beta,
                completed_relations=[d["relation"] for d in rows],
                excluded_relations=sorted(
                    {p.stem for p in paths} - {d["relation"] for d in rows}
                ),
            )
        if variants:
            report[model_dir.name] = variants
    write(Path(root) / "summary.json", report)
    if verbose:
        print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--name")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--layers", default="5,11,14,17,20,23")
    parser.add_argument("--lens-layer", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--prefix-cache",
        action="store_true",
        help="verify and cache the causally unchanged prefix for this architecture",
    )
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument(
        "--dynamic", action="store_true", help="claim unfinished relations across hosts"
    )
    parser.add_argument("--relations", default="")
    parser.add_argument("--protocols", default="main,shared_split")
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        return summarize(args.root)
    torch.set_num_threads(2)
    start = time.time()
    print("LOAD", socket.gethostname(), args.model, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
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
    model = (
        AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
        .eval()
        .requires_grad_(False)
    )
    placement = getattr(model, "hf_device_map", None)
    if placement is None:
        # Some HF single-device loaders omit Accelerate's metadata attribute.
        placement = {
            f"observed_parameter_device_{i}": device
            for i, device in enumerate(
                sorted({str(p.device) for p in model.parameters()})
            )
        }
    device_map = {key: str(value) for key, value in placement.items()}
    if any(device in {"cpu", "disk"} for device in device_map.values()):
        raise RuntimeError(f"This sweep requires GPU-only placement: {device_map}")
    model.config.use_cache = False
    model.config.get_text_config().use_cache = False
    model.forward = partial(model.forward, use_cache=False, logits_to_keep=1)
    convolution_check = enable_fast_qwen_convolution(model, tokenizer, args.lens_layer)
    verification = verify(model, tokenizer, args.lens_layer, args.batch_size)
    prefix_check = (
        enable_prefix_cache(model, tokenizer, args.lens_layer, args.batch_size)
        if args.prefix_cache
        or model.config.get_text_config().model_type == "qwen3_5_text"
        else None
    )
    metadata = dict(
        model_id=args.model,
        revision=Path(
            hf_hub_download(args.model, "config.json", local_files_only=True)
        ).parent.name,
        dtype="float32",
        host=socket.gethostname(),
        layers=args.layers,
        lens_layer=args.lens_layer,
        verification=verification,
        torch=torch.__version__,
        batch_size=args.batch_size,
        convolution_check=convolution_check,
        prefix_cache_check=prefix_check,
        device_map=device_map,
    )
    write(
        args.root
        / args.name
        / f"verification_{socket.gethostname()}_{args.shard}.json",
        metadata,
    )
    print("VERIFIED", verification, flush=True)
    if args.verify_only:
        return
    relations = (
        args.relations.split(",")
        if args.relations
        else sorted(p.stem for p in DATA.glob("*.json"))
    )
    # Longest relations first, balanced across persistent workers.
    relations = sorted(
        relations,
        key=lambda r: -len(json.loads((DATA / f"{r}.json").read_text())["samples"]),
    )
    for protocol in args.protocols.split(","):
        for relation in (
            relations if args.dynamic else relations[args.shard :: args.shards]
        ):
            data = json.loads((DATA / f"{relation}.json").read_text())
            samples = data["samples"]
            template = data["prompt_templates"][0]
            trailing = trailing_space_needed(tokenizer, relation, samples)
            path = args.root / args.name / protocol / f"{relation}.json"
            with measurement_claim(path, args.dynamic) as acquired:
                if not acquired or path.exists():
                    continue
                pool = (
                    samples
                    if protocol == "main"
                    else splits.filter_identifying_objects(samples, tokenizer, trailing)
                )
                if len(pool) < 17:
                    write(
                        path,
                        dict(
                            relation=relation,
                            skipped="too_few_samples",
                            n_samples=len(pool),
                            **metadata,
                        ),
                    )
                    continue
                print("START", relation, protocol, len(pool), flush=True)
                t = time.time()
                faith = measure_faith(
                    model,
                    tokenizer,
                    pool,
                    template,
                    protocol,
                    [int(x) for x in args.layers.split(",")],
                    trailing,
                    args.batch_size,
                    args.n_test,
                )
                lens = measure_lens(
                    model,
                    tokenizer,
                    pool,
                    template,
                    protocol,
                    args.lens_layer,
                    trailing,
                    args.batch_size,
                    args.n_test,
                )
                write(
                    path,
                    dict(
                        relation=relation,
                        protocol=protocol,
                        faithfulness=faith,
                        lens=lens,
                        trailing_space=trailing,
                        elapsed_seconds=time.time() - t,
                        **metadata,
                    ),
                )
                print(
                    "DONE",
                    relation,
                    protocol,
                    "faith",
                    faith["best_faith"],
                    "seconds",
                    time.time() - t,
                    flush=True,
                )
    print("COMPLETE", args.name, args.shard, time.time() - start, flush=True)


if __name__ == "__main__":
    main()
