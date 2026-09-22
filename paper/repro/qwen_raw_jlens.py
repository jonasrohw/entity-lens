"""Qwen raw Jacobian lens: mean_i J_i h_e, at subject and prompt ends.

Uses the frozen 26-relation inputs from the position experiment. No subtraction,
reference output, or beta is added to the raw readout. The differential control
is also evaluated directly with the same graphs, not reconstructed from ranks.
"""

import argparse
import hashlib
import json
import platform
import time
from functools import partial
from pathlib import Path

import torch
import transformers
from jlens.model import HiddenStateIntervention, resolve_blocks, resolve_final_norm
from last_position_correlation import BASE, aggregate, cache_prompt, jvps, write
from transformers import AutoModelForCausalLM, AutoTokenizer


def measure(model, tokenizer, spec, layer, verify):
    names = [r["subject"] for r in spec["targets"]]
    target_ids = [r["model_top1_id"] for r in spec["targets"]]
    assert all(isinstance(t, int) for t in target_ids)
    context = "".join(
        spec["template"].format(d["subject"]) + " " + d["object"] + "\n"
        for d in spec["demonstrations"]
    )
    with torch.no_grad():
        for name, tid in zip(names, target_ids, strict=True):
            ids = tokenizer(
                context + spec["template"].format(name), return_tensors="pt"
            ).input_ids.to("cuda:0")
            assert int(model(input_ids=ids).logits[0, -1].argmax()) == tid
    cache = {s: cache_prompt(model, tokenizer, s, layer) for s in names}
    refs = spec["references"]
    blocks, norm = resolve_blocks(model), resolve_final_norm(model)
    intervention = HiddenStateIntervention(model, blocks[-1])
    handle = blocks[layer - 1].register_forward_hook(intervention.hook)
    results, checks = {}, []
    try:
        for position in ("subject", "last"):
            states = torch.stack([cache[s]["states"][position] for s in names])
            totals = {kind: torch.zeros_like(states) for kind in ("raw", "diff")}
            counts = torch.zeros(len(names), device="cuda:0")
            contributions = []
            for anchor in refs:
                c = cache[anchor]
                h = c["states"][position]
                pos = c["positions"][position]
                with torch.no_grad():
                    noop = intervention.forward(h, c["ids"], pos)
                torch.testing.assert_close(noop, c["states"]["final"], rtol=0, atol=0)
                # Direct raw and difference products plus J_i h_i for an identity check.
                directions = torch.cat([states, states - h, h[None]], dim=0)
                products = jvps(intervention, h, directions, c["ids"], pos, 2)
                raw, diff, anchor_product = (
                    products[: len(names)],
                    products[len(names) : -1],
                    products[-1],
                )
                assert bool(torch.isfinite(products).all())
                identity_errors = (raw - anchor_product - diff).norm(dim=1) / (
                    raw.norm(dim=1) + anchor_product.norm() + diff.norm(dim=1)
                ).clamp_min(1e-20)
                assert float(identity_errors.max()) < 1e-4
                if verify and anchor == refs[0]:
                    j = next(i for i, s in enumerate(names) if s != anchor)
                    direction, product = states[j], raw[j]
                    fd_checks = []
                    # Small perturbations, even when h_e dwarfs the reference h_i.
                    scale = min(1.0, float(h.norm() / direction.norm()))
                    for epsilon in (0.01, 0.003, 0.001):
                        eps = epsilon * scale
                        with torch.no_grad():
                            plus = intervention.forward(
                                h + eps * direction, c["ids"], pos
                            )
                            minus = intervention.forward(
                                h - eps * direction, c["ids"], pos
                            )
                        fd = (plus - minus) / (2 * eps)
                        fd_checks.append(
                            {
                                "epsilon": eps,
                                "relative_error": float(
                                    (fd - product).norm() / product.norm()
                                ),
                            }
                        )
                    assert min(r["relative_error"] for r in fd_checks) < 0.03, fd_checks
                    scalar = intervention.jvp(h, direction, c["ids"], pos)
                    torch.testing.assert_close(scalar, product, rtol=1e-3, atol=1e-3)
                    checks.append(
                        {
                            "position": position,
                            "reference": anchor,
                            "subject": names[j],
                            "finite_differences": fd_checks,
                            "batched_scalar_relative_error": float(
                                (scalar - product).norm() / product.norm()
                            ),
                        }
                    )
                valid = torch.tensor([s != anchor for s in names], device="cuda:0")
                totals["raw"] += raw * valid[:, None]
                totals["diff"] += diff * valid[:, None]
                counts += valid
                contributions.append(
                    {
                        "reference": anchor,
                        "raw_norms": raw.norm(dim=1).tolist(),
                        "diff_norms": diff.norm(dim=1).tolist(),
                        "anchor_product_norm": float(anchor_product.norm()),
                        "linearity_relative_errors": identity_errors.tolist(),
                    }
                )
            assert bool((counts > 0).all())
            modes = {}
            for kind, total in totals.items():
                with torch.no_grad():
                    logits = model.get_output_embeddings()(
                        norm(total / counts[:, None])
                    )
                    tids = torch.tensor(target_ids, device="cuda:0")
                    ranks = (logits > logits.gather(1, tids[:, None])).sum(-1).tolist()
                    tops = logits.topk(5).indices.tolist()
                modes[kind] = {
                    "agg": aggregate(ranks),
                    "rows": [
                        {
                            "subject": s,
                            "position": cache[s]["positions"][position],
                            "state_norm": float(states[i].norm()),
                            "n_references": int(counts[i]),
                            "target_id": target_ids[i],
                            "target": spec["targets"][i]["model_top1"],
                            "rank": ranks[i],
                            "top_ids": tops[i],
                            "top_tokens": [
                                tokenizer.decode([t]).strip() for t in tops[i]
                            ],
                        }
                        for i, s in enumerate(names)
                    ],
                }
            results[position] = {"modes": modes, "contributions": contributions}
    finally:
        handle.remove()
    return {
        "relation": spec["relation"],
        "faithfulness": spec["faithfulness"],
        "positions": results,
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "generated/qwen_raw_jlens"
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    inputs = BASE / "docs/last-position-lre-correlation-inputs.json"
    spec = json.loads(inputs.read_text())["models"]["qwen"]
    print("LOAD", args.shard, flush=True)
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
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
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
        "input_sha256": sha(inputs),
        "script_sha256": sha(Path(__file__)),
        "helper_sha256": sha(Path(__file__).with_name("last_position_correlation.py")),
        "intervention_sha256": sha(BASE / "jlens/model.py"),
    }
    for i, relation in enumerate(spec["relations"][args.shard :: args.shards]):
        out = args.output_dir / (relation["relation"] + ".json")
        if out.exists():
            assert json.loads(out.read_text())["provenance"] == provenance
            continue
        assert (
            sha(BASE / "data/relations" / (relation["relation"] + ".json"))
            == relation["dataset_sha256"]
        )
        start = time.monotonic()
        result = measure(
            model,
            tokenizer,
            relation,
            spec["layer"],
            i == 0 or relation["relation"] == "country_language",
        )
        result.update(provenance=provenance, seconds=time.monotonic() - start)
        write(out, result)
        print(
            "DONE",
            relation["relation"],
            {
                p: {k: v["agg"] for k, v in d["modes"].items()}
                for p, d in result["positions"].items()
            },
            result["seconds"],
            flush=True,
        )


if __name__ == "__main__":
    main()
