"""Compare subject-end and prompt-end differential lenses against frozen LRE scores.

Run one worker per model, or shard the 26 relations across separate GPUs.
Inputs freeze the original main-protocol subjects, references, targets and y.
The prompt, layer, differential formula and native readout are unchanged.
"""

import argparse
import hashlib
import json
import platform
import sys
import time
from functools import partial
from pathlib import Path

BASE = Path(__file__).resolve().parents[2] / "code"
sys.path.insert(0, str(BASE))
import numpy as np
import torch
import transformers
from jlens.model import HiddenStateIntervention, resolve_blocks, resolve_final_norm
from transformers import AutoModelForCausalLM, AutoTokenizer


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def aggregate(ranks):
    r = np.asarray(ranks, dtype=float)
    return {
        "median_rank": float(np.median(r)),
        "median_lograng": float(np.median(np.log10(r + 1))),
        "agree_at1": float((r == 0).mean()),
        "agree_at10": float((r < 10).mean()),
    }


def cache_prompt(model, tokenizer, subject, layer):
    prompt = subject + " is known for"
    tokens = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True)
    ids = tokens.input_ids.to("cuda:0")
    offsets = tokens.offset_mapping[0].tolist()
    pos = max(i for i, (a, b) in enumerate(offsets) if a < len(subject) and b > 0)
    end = ids.shape[1] - 1
    assert tokenizer.decode([int(ids[0, end])]).strip() == "for"
    blocks = resolve_blocks(model)
    states = {}

    def capture(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        states["subject"] = h[0, pos].detach().clone()
        states["last"] = h[0, end].detach().clone()

    def capture_final(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        states["final"] = h[0, -1].detach().clone()

    handles = [
        blocks[layer - 1].register_forward_hook(capture),
        blocks[-1].register_forward_hook(capture_final),
    ]
    try:
        with torch.no_grad():
            output = model(input_ids=ids)
            readout = model.get_output_embeddings()(
                resolve_final_norm(model)(states["final"])
            )
            torch.testing.assert_close(
                readout, output.logits[0, -1], rtol=1e-4, atol=2e-4
            )
    finally:
        for h in handles:
            h.remove()
    return {"ids": ids, "positions": {"subject": pos, "last": end}, "states": states}


def jvps(intervention, reference, directions, ids, position, batch_size):
    hidden = reference.detach().clone().requires_grad_(True)
    output = intervention.forward(hidden, ids, position)
    cotangent = torch.zeros_like(output, requires_grad=True)
    gradient = torch.autograd.grad(
        output, hidden, grad_outputs=cotangent, create_graph=True
    )[0]
    chunks = []
    for start in range(0, len(directions), batch_size):
        batch = directions[start : start + batch_size]
        product = torch.autograd.grad(
            gradient,
            cotangent,
            grad_outputs=batch,
            is_grads_batched=True,
            retain_graph=start + batch_size < len(directions),
        )[0]
        chunks.append(product.detach())
    return torch.cat(chunks)


def check_derivative(intervention, c, direction, product):
    h = c["states"]["last"]
    pos = c["positions"]["last"]
    checks = []
    for eps in (0.01, 0.003, 0.001):
        with torch.no_grad():
            plus = intervention.forward(h + eps * direction, c["ids"], pos)
            minus = intervention.forward(h - eps * direction, c["ids"], pos)
        fd = (plus - minus) / (2 * eps)
        checks.append(
            {
                "epsilon": eps,
                "relative_error": float((fd - product).norm() / product.norm()),
            }
        )
    assert min(r["relative_error"] for r in checks) < 0.02, checks
    scalar = intervention.jvp(h, direction, c["ids"], pos)
    torch.testing.assert_close(product, scalar, rtol=1e-3, atol=1e-3)
    return {
        "finite_differences": checks,
        "batched_scalar_relative_error": float(
            (scalar - product).norm() / product.norm()
        ),
    }


def measure(model, tokenizer, spec, layer, batch_size, verify):
    targets = spec["targets"]
    names = [r["subject"] for r in targets]
    refs = spec["references"]
    context = "".join(
        spec["template"].format(d["subject"]) + " " + d["object"] + "\n"
        for d in spec["demonstrations"]
    )
    target_ids = []
    with torch.no_grad():
        for row in targets:
            ids = tokenizer(
                context + spec["template"].format(row["subject"]), return_tensors="pt"
            ).input_ids.to("cuda:0")
            tid = int(model(input_ids=ids).logits[0, -1].argmax())
            assert tokenizer.decode([tid]).strip() == row["model_top1"], (
                spec["relation"],
                row,
                tid,
            )
            assert row["model_top1_id"] is None or tid == row["model_top1_id"]
            target_ids.append(tid)
    cache = {name: cache_prompt(model, tokenizer, name, layer) for name in names}
    blocks, norm = resolve_blocks(model), resolve_final_norm(model)
    intervention = HiddenStateIntervention(model, blocks[-1])
    handle = blocks[layer - 1].register_forward_hook(intervention.hook)
    result = {}
    checks = []
    try:
        for mode in ("subject", "last"):
            total = torch.zeros(
                (len(names), cache[names[0]]["states"]["final"].numel()),
                device="cuda:0",
            )
            counts = torch.zeros(len(names), device="cuda:0")
            contributions = []
            for anchor in refs:
                c = cache[anchor]
                h = c["states"][mode]
                pos = c["positions"][mode]
                with torch.no_grad():
                    noop = intervention.forward(h, c["ids"], pos)
                torch.testing.assert_close(noop, c["states"]["final"], rtol=0, atol=0)
                directions = torch.stack([cache[s]["states"][mode] - h for s in names])
                v = jvps(intervention, h, directions, c["ids"], pos, batch_size)
                assert bool(torch.isfinite(v).all())
                if mode == "last" and verify and anchor == refs[0]:
                    j = next(i for i, s in enumerate(names) if s != anchor)
                    checks.append(
                        dict(
                            reference=anchor,
                            subject=names[j],
                            **check_derivative(intervention, c, directions[j], v[j]),
                        )
                    )
                valid = torch.tensor([s != anchor for s in names], device="cuda:0")
                total += v * valid[:, None]
                counts += valid
                contributions.append(
                    {
                        "reference": anchor,
                        "norms": v.norm(dim=1).tolist(),
                        "direction_to_reference_norm": (
                            directions.norm(dim=1) / h.norm()
                        ).tolist(),
                    }
                )
            assert bool((counts > 0).all())
            hidden = total / counts[:, None]
            with torch.no_grad():
                logits = model.get_output_embeddings()(norm(hidden))
                tids = torch.tensor(target_ids, device="cuda:0")
                ranks = (logits > logits.gather(1, tids[:, None])).sum(-1).tolist()
                tops = logits.topk(5).indices.tolist()
            result[mode] = {
                "agg": aggregate(ranks),
                "contributions": contributions,
                "rows": [
                    {
                        "subject": s,
                        "position": cache[s]["positions"][mode],
                        "input_ids": cache[s]["ids"][0].tolist(),
                        "state_norm": float(cache[s]["states"][mode].norm()),
                        "n_references": int(counts[i]),
                        "model_top1_id": target_ids[i],
                        "model_top1": targets[i]["model_top1"],
                        "rank": ranks[i],
                        "top_ids": tops[i],
                        "top_tokens": [tokenizer.decode([t]).strip() for t in tops[i]],
                    }
                    for i, s in enumerate(names)
                ],
            }
    finally:
        handle.remove()
    return {
        "relation": spec["relation"],
        "faithfulness": spec["faithfulness"],
        "original_x": spec["original_x"],
        "original_ranks": [r["original_rank"] for r in targets],
        "modes": result,
        "derivative_checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("qwen", "olmo"), required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "generated/last_position_correlation"
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    inputs = BASE / "docs/last-position-lre-correlation-inputs.json"
    spec = json.loads(inputs.read_text())["models"][args.model]
    print("LOAD", args.model, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        spec["model"], revision=spec["revision"], local_files_only=True
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            spec["model"],
            revision=spec["revision"],
            local_files_only=True,
            dtype=torch.float32,
            attn_implementation="eager",
            device_map={"": "cuda:0"},
        )
        .eval()
        .requires_grad_(False)
    )
    model.config.use_cache = False
    model.forward = partial(model.forward, use_cache=False, logits_to_keep=1)
    provenance = {
        "model": spec["model"],
        "revision": spec["revision"],
        "layer": spec["layer"],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(),
        "dtype": "float32",
        "attention": "eager",
        "neutral_prompt": "{} is known for",
        "input_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "intervention_sha256": hashlib.sha256(
            (BASE / "jlens/model.py").read_bytes()
        ).hexdigest(),
    }
    relations = spec["relations"][args.shard :: args.shards]
    for index, relation in enumerate(relations):
        out = args.output_dir / args.model / (relation["relation"] + ".json")
        if out.exists():
            old = json.loads(out.read_text())
            assert old["provenance"] == provenance
            continue
        dataset = BASE / "data/relations" / (relation["relation"] + ".json")
        assert (
            hashlib.sha256(dataset.read_bytes()).hexdigest()
            == relation["dataset_sha256"]
        )
        start = time.monotonic()
        result = measure(
            model,
            tokenizer,
            relation,
            spec["layer"],
            args.batch_size,
            index == 0 or relation["relation"] == "country_language",
        )
        result.update(provenance=provenance, seconds=time.monotonic() - start)
        write(out, result)
        print(
            "DONE",
            args.model,
            relation["relation"],
            {k: v["agg"] for k, v in result["modes"].items()},
            result["seconds"],
            flush=True,
        )


if __name__ == "__main__":
    main()
