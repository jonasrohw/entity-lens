"""Refine positive LRE beta continuously using saved Jacobian products."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.mistral.modeling_mistral import MistralRMSNorm

from jlens import model_sweep as sweep
from jlens.beta_intervals import rank_candidates, top1_intervals
from jlens.grid_search import BETAS

ROOT = Path(__file__).resolve().parents[1]
MODEL = "mistralai/Mistral-7B-v0.3"


class Readout(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.config = AutoConfig.from_pretrained(MODEL, local_files_only=True)
        directory = Path(
            hf_hub_download(MODEL, "config.json", local_files_only=True)
        ).parent
        index = json.loads((directory / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        values = {}
        for name in ["model.norm.weight", "lm_head.weight"]:
            with safe_open(
                directory / index[name], framework="pt", device="cpu"
            ) as shard:
                values[name] = shard.get_tensor(name).float().to(device)
        self.norm = MistralRMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps
        ).to(device)
        self.norm.weight = torch.nn.Parameter(
            values["model.norm.weight"], requires_grad=False
        )
        self.head = torch.nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False, device=device
        )
        self.head.weight = torch.nn.Parameter(
            values["lm_head.weight"], requires_grad=False
        )
        self.double_weight = self.head.weight.double()
        self.eval().requires_grad_(False)

    def get_output_embeddings(self):
        return self.head

    def affine_logits(self, base, products):
        weight = self.norm.weight.double()
        return F.linear(base.double() * weight, self.double_weight), F.linear(
            products.double() * weight, self.double_weight
        )


def actual_hits(model, base, products, targets, beta):
    logits = sweep.decode(model, model.norm, base[None] + beta * products)
    return (
        (logits.argmax(-1) == torch.tensor(targets, device=products.device))
        .int()
        .cpu()
        .tolist()
    )


def select_beta(model, base, products, targets, bounds, selection, dense):
    values, scores = rank_candidates(bounds, selection, [b for b in BETAS if b > 0])
    expected = int(scores[0])
    best_beta = max(
        (b for b in BETAS if b > 0),
        key=lambda b: sum(dense[str(b)]["hits"][i] for i in selection),
    )
    best_hits = dense[str(best_beta)]["hits"]
    best_count = sum(best_hits[i] for i in selection)
    tried = 0
    for beta, theoretical in zip(values, scores):
        if theoretical < expected:
            break
        hits = actual_hits(model, base, products, targets, float(beta))
        count = sum(hits[i] for i in selection)
        tried += 1
        if count > best_count or (count == best_count and beta < best_beta):
            best_beta, best_hits, best_count = float(beta), hits, count
        if count >= expected:
            break
    return dict(
        beta=best_beta,
        hits=best_hits,
        selection_faith=best_count / len(selection),
        analytic_max_faith=expected / len(selection),
        verified_optimum=best_count >= expected,
        evaluated_candidates=tried,
        selection_indices=selection,
    )


def refine(model, source, tensor_path):
    data = json.loads(source.read_text())
    if "faith" not in data:
        return dict(
            relation=data["relation"],
            protocol=data["protocol"],
            skipped=data.get("skipped"),
        )
    tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)
    targets = data["faith"]["targets"]
    count = len(targets)
    calibration = list(range(count // 2))
    evaluation = list(range(count // 2, count))
    grid = {}
    max_error = 0.0
    for layer in data["layers"]:
        grid[str(layer)] = {}
        for refs in data["ref_counts"]:
            key = f"faith/{layer}/{refs}"
            base = tensors[key + "/base"].to(model.norm.weight.device)
            products = tensors[key + "/products"].to(model.norm.weight.device)
            a, b = model.affine_logits(base[None], products)
            hidden = base[None] + products
            linear = (a + b) / torch.sqrt(
                hidden.double().square().mean(-1, keepdim=True)
                + model.config.rms_norm_eps
            )
            hf = sweep.decode(model, model.norm, hidden)
            torch.testing.assert_close(linear.float(), hf, rtol=5e-4, atol=5e-4)
            max_error = max(max_error, float((linear.float() - hf).abs().max()))
            bounds = top1_intervals(a, b, targets)
            dense = data["faith"]["grid"][str(layer)][str(refs)]
            all_fit = select_beta(
                model, base, products, targets, bounds, list(range(count)), dense
            )
            calibration_fit = None
            if count >= 4:
                calibration_fit = select_beta(
                    model, base, products, targets, bounds, calibration, dense
                )
                calibration_fit["evaluation_indices"] = evaluation
                calibration_fit["evaluation_faith"] = float(
                    np.mean([calibration_fit["hits"][i] for i in evaluation])
                )
            grid[str(layer)][str(refs)] = dict(
                all_subjects=all_fit,
                calibration=calibration_fit,
                input_dependent=bool(torch.count_nonzero(products)),
                lower=[float(v) if np.isfinite(v) else None for v in bounds["lower"]],
                upper=[float(v) if np.isfinite(v) else None for v in bounds["upper"]],
                valid=bounds["valid"].tolist(),
            )
    return dict(
        relation=data["relation"],
        protocol=data["protocol"],
        n_test=count,
        grid=grid,
        readout_max_abs_error=max_error,
        method="positive-beta target top-1 interval overlap; optima verified with HF float32 readout",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=ROOT / "generated/mistral_full_grid"
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = Readout("cuda")
    print("READOUT_LOADED", flush=True)
    expected = 2 * len(list(sweep.DATA.glob("*.json")))
    while True:
        for protocol in ["main", "shared_split"]:
            for source in sorted((args.root / protocol).glob("*.json")):
                target = args.root / "continuous" / protocol / source.name
                if target.exists():
                    continue
                start = time.time()
                with torch.no_grad():
                    result = refine(model, source, source.with_suffix(".pt"))
                result["elapsed_seconds"] = time.time() - start
                sweep.write(target, result)
                print(
                    "REFINED",
                    protocol,
                    source.stem,
                    result["elapsed_seconds"],
                    flush=True,
                )
        completed = len(list((args.root / "continuous").glob("*/*.json")))
        if args.once or completed == expected:
            break
        time.sleep(5)
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
