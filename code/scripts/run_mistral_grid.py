"""Keep Mistral resident while running the expanded, resumable grid."""

import argparse
import json
import os
import socket
import time
from functools import partial
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens.model_sweep import enable_prefix_cache, verify

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--root", type=Path, default=ROOT / "generated/mistral_full_grid"
    )
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / f"{args.worker}.pid").write_text(str(os.getpid()) + "\n")
    torch.set_num_threads(2)
    model_id = args.model
    print("LOAD", socket.gethostname(), model_id, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_id,
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
    checks = dict(
        standard=verify(model, tokenizer, 17, args.batch_size),
        prefix=enable_prefix_cache(model, tokenizer, 17, args.batch_size),
    )
    (args.root / f"{args.worker}.verification.json").write_text(
        json.dumps(checks, indent=2) + "\n"
    )
    print("VERIFIED", checks, flush=True)
    while not (args.root / "READY").exists():
        time.sleep(2)
    if args.confirm:
        from jlens.grid_confirmation import run_worker
    else:
        from jlens.grid_search import run_worker

    run_worker(model, tokenizer, args)


if __name__ == "__main__":
    main()
