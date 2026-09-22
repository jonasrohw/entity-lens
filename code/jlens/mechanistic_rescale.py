"""Supplementary gain-range control; same frozen operators and held-out splits.

Keeps float32, eager attention and all model parameters on two GPUs. Recreates
training intercepts and four-Jacobian centering offsets, then rescores cached
operator products. The original beta grid is verified against saved outputs
before the wider validation-only gain selection is reported.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from jlens.mechanistic import BETAS, MODEL, Runner, save_json

WIDE_BETAS = [
    0.0,
    0.01,
    0.03,
    0.1,
    0.3,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    5.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1000.0,
    10000.0,
]


def run_one(runner, root, item):
    t0 = time.time()
    name, plan, nv = item["relation"], item["plan"], item["nval"]
    if item["model_revision"] != getattr(runner.model.config, "_commit_hash", None):
        raise RuntimeError("Model checkpoint revision changed between runs")
    arrays = torch.load(
        root / f"{name}.pt", map_location=runner.device, weights_only=True
    )
    template = plan["template"]
    context = "".join(
        template.format(x["subject"]) + " " + x["object"] + "\n"
        for x in plan["parts"]["icl"]
    )
    trailing = " " if plan["trailing_space"] else ""
    refs = [
        runner.cache(context + template.format(x["subject"]) + trailing, x["subject"])
        for x in plan["parts"]["references"]
    ]
    with torch.no_grad():
        train_finals = []
        for x in plan["parts"]["train"]:
            ids, pos = runner.encode(
                context + template.format(x["subject"]) + trailing, x["subject"]
            )
            final, _, _ = runner.forward(ids, pos)
            train_finals.append(final.detach())
        base = torch.stack(train_finals).mean(0)
    result = {
        "relation": name,
        "nval": nv,
        "betas": WIDE_BETAS,
        "layers": {},
        "supplementary": True,
    }
    for layer, a in arrays.items():
        mean = a["train_relation"].mean(0)
        offset = runner.products(refs, layer, mean[None], subtract_reference=True)[0]
        products = {
            "relation": a["relation_products"] - offset,
            "shared": a["shared_products"],
            "matched_neutral": a["neutral_products_same_input"],
        }
        original = item["layers"][str(layer)]
        record = {}
        for kind, directions in products.items():
            grid = {
                str(b): runner.score(base + b * directions, item["targets"])
                for b in WIDE_BETAS
            }
            best = max(
                WIDE_BETAS,
                key=lambda b: (
                    np.mean(grid[str(b)]["hit"][:nv]),
                    np.mean(grid[str(b)]["logp"][:nv]),
                    -b,
                ),
            )
            old_grid = original[
                {
                    "relation": "calibrated_relation_grids",
                    "shared": "shared_grids",
                    "matched_neutral": "matched_neutral_grids",
                }[kind]
            ]
            max_logp_error = max(
                float(
                    np.max(
                        np.abs(
                            np.array(grid[str(b)]["logp"]) - old_grid[str(b)]["logp"]
                        )
                    )
                )
                for b in BETAS
            )
            mismatches = sum(
                sum(
                    a != b
                    for a, b in zip(grid[str(beta)]["top"], old_grid[str(beta)]["top"])
                )
                for beta in BETAS
            )
            if max_logp_error > 1e-3 or mismatches:
                raise RuntimeError(
                    f"{name} layer {layer} {kind}: rescore differs: logp={max_logp_error}, top mismatches={mismatches}"
                )
            record[kind] = {
                "beta": best,
                "scores": grid[str(best)],
                "grid": grid,
                "verification_max_logp_error": max_logp_error,
                "verification_top_mismatches": mismatches,
                "mean_product_norm": float(directions.norm(dim=-1).mean()),
            }
        result["layers"][str(layer)] = record
    result["complete"] = True
    result["seconds"] = time.time() - t0
    save_json(root / "calibration" / f"{name}.json", result)
    print(
        f"{name}: wider-gain control complete in {result['seconds']:.1f}s", flush=True
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument(
        "--differential-only",
        action="store_true",
        help="Score the paper's unscaled differential neutral lens from cached vectors",
    )
    args = parser.parse_args()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = AutoConfig.from_pretrained(MODEL, local_files_only=True)
    mapping = {
        "model.embed_tokens": 0,
        "model.rotary_emb": 0,
        "model.norm": 1,
        "lm_head": 1,
    }
    mapping.update(
        {
            f"model.layers.{i}": int(i >= config.num_hidden_layers // 2)
            for i in range(config.num_hidden_layers)
        }
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
        device_map=mapping,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        assert parameter.dtype == torch.float32 and parameter.device.type == "cuda"
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    manifest = json.loads((args.directory / "manifest.json").read_text())
    names = [name for name, plan in manifest["relations"].items() if "parts" in plan]
    runner = Runner(model, tokenizer, manifest["layers"])
    if args.differential_only:
        for name in names:
            item = json.loads((args.directory / f"{name}.json").read_text())
            assert item["complete"]
            tensors = torch.load(
                args.directory / f"{name}.pt",
                map_location=runner.device,
                weights_only=True,
            )
            result = {"relation": name, "layers": {}, "complete": True}
            for layer, arrays in tensors.items():
                neutral = arrays["neutral_affine"] - arrays["neutral_base"]
                result["layers"][str(layer)] = runner.score(neutral, item["targets"])
            save_json(args.directory / "differential" / f"{name}.json", result)
            print(f"{name}: differential readout scored", flush=True)
        return
    while True:
        remaining = []
        for name in names:
            if (args.directory / "calibration" / f"{name}.json").exists():
                continue
            remaining.append(name)
            path = args.directory / f"{name}.json"
            if path.exists():
                item = json.loads(path.read_text())
                if item.get("complete"):
                    run_one(runner, args.directory, item)
        if not remaining or not args.wait:
            break
        time.sleep(20)


if __name__ == "__main__":
    main()
