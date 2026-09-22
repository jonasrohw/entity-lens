"""Matched newline/shared-split baselines; preserves authoritative LRE targets."""

import argparse
import json
import socket
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import transformers
from run_all_model_newline import OUTPUT, PROMPT, ROOT, sha, specs
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens import model_sweep as sweep
from jlens import splits
from jlens.geva_baseline import subject_key
from jlens.measure_answer_prior import NEUTRAL_CONTEXTS

DEST = ROOT / "results/shared_baselines"
CACHE = ROOT / "generated/shared_baselines"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--worker", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(2)
    spec = specs()[a.name]
    layer = spec["lens_layer"]
    tok = AutoTokenizer.from_pretrained(
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
    model.config.use_cache = model.config.get_text_config().use_cache = False
    model.forward = partial(model.forward, use_cache=False, logits_to_keep=1)
    convolution = sweep.enable_fast_qwen_convolution(model, tok, layer)
    verification = sweep.verify(model, tok, layer, a.batch_size)
    prefix = (
        sweep.enable_prefix_cache(model, tok, layer, a.batch_size)
        if model.config.get_text_config().model_type == "qwen3_5_text"
        else None
    )
    blocks, norm = sweep.layout(model)
    device = model.get_output_embeddings().weight.device
    input_device = model.get_input_embeddings().weight.device
    meta = dict(
        **spec,
        prompt=PROMPT,
        protocol="shared_split",
        dtype="float32",
        attention="eager",
        host=socket.gethostname(),
        worker=a.worker,
        batch_size=a.batch_size,
        verification=verification,
        convolution_check=convolution,
        prefix_cache_check=prefix,
        torch=torch.__version__,
        transformers=transformers.__version__,
        source_sha256={
            str(f.relative_to(ROOT)): sha(f)
            for f in (
                Path(__file__),
                Path(sweep.__file__),
                ROOT / "jlens/model.py",
                ROOT / "jlens/splits.py",
                ROOT / "jlens/prefix_intervention.py",
                ROOT / "jlens/measure_answer_prior.py",
            )
        },
    )
    sweep.write(DEST / a.name / f"worker_{a.worker}.json", meta)
    print("VERIFIED", verification, flush=True)
    with torch.no_grad():
        prior = (
            torch.stack(
                [
                    torch.log_softmax(
                        model(
                            input_ids=tok(
                                c or tok.bos_token or "<|endoftext|>",
                                return_tensors="pt",
                            ).input_ids.to(input_device)
                        )
                        .logits[0, -1]
                        .float(),
                        -1,
                    )
                    for c in NEUTRAL_CONTEXTS
                ]
            )
            .mean(0)
            .cpu()
            .numpy()
        )
    cache_dir = CACHE / a.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / f"prior_{a.worker}.npy", prior)
    cache = {}

    def get(subject):
        if subject not in cache:
            c = sweep.cache_prompt(model, tok, PROMPT.format(subject), subject, [layer])
            with torch.no_grad():
                values = sweep.decode(model, norm, c["states"][layer]).cpu().numpy()
            path = cache_dir / f"{subject_key(subject)}.npz"
            # Worker-specific temporary files keep concurrent caches atomic.
            temporary = path.with_suffix(f".{a.worker}.tmp")
            with temporary.open("wb") as f:
                np.savez_compressed(
                    f, subject=subject, logits=values, top_ids=np.argsort(-values)[:50]
                )
            temporary.replace(path)
            c.pop("logits")
            cache[subject] = c
        return cache[subject]

    sources = sorted(
        (OUTPUT / a.name / "shared_split").glob("*.json"),
        key=lambda f: -json.loads(f.read_text())["lens"]["n"],
    )
    for source in sources:
        target = DEST / a.name / "shared_split" / source.name
        with sweep.measurement_claim(target) as acquired:
            if not acquired or target.exists():
                continue
            start = time.time()
            old = json.loads(source.read_text())
            relation = old["relation"]
            print("START", relation, flush=True)
            data_path = sweep.DATA / source.name
            data = json.loads(data_path.read_text())
            assert sha(data_path) == old["data_sha256"]
            samples = splits.filter_identifying_objects(
                data["samples"], tok, old["trailing_space"]
            )
            demos, refs, tests = sweep.partitions(samples, "shared_split", lens=True)
            assert (
                splits.split_fingerprint(demos, refs, tests) == old["split_fingerprint"]
            )
            assert [x["subject"] for x in tests] == [
                r["subject"] for r in old["lens"]["rows"]
            ]
            assert (
                old["lens"]["neutral_prompt"] == PROMPT
                and old["lens"]["layer"] == layer
            )
            for x in refs + tests:
                get(x["subject"])
            test_states = torch.stack(
                [get(t["subject"])["states"][layer] for t in tests]
            )
            width = get(tests[0]["subject"])["states"]["final"].numel()
            raw = torch.zeros(len(tests), width, device=device)
            centers = torch.zeros_like(test_states)
            counts = torch.zeros(len(tests), device=device)
            intervention = sweep.make_intervention(model, blocks[-1])
            handle = blocks[layer - 1].register_forward_hook(intervention.hook)
            try:
                for x in refs:
                    c = get(x["subject"])
                    v = sweep.jvps(
                        intervention,
                        c["states"][layer],
                        test_states,
                        c["ids"],
                        c["position"],
                        a.batch_size,
                    ).to(device)
                    valid = torch.tensor(
                        [t["subject"] != x["subject"] for t in tests], device=device
                    )
                    raw += v * valid[:, None]
                    centers += c["states"][layer] * valid.to(centers.device)[:, None]
                    counts += valid
            finally:
                handle.remove()
            assert (counts > 0).all()
            raw /= counts[:, None]
            centers /= counts.to(centers.device)[:, None]
            rows = []
            for t, oldrow, count in zip(
                tests, old["lens"]["rows"], counts.tolist(), strict=True
            ):
                assert int(count) == oldrow["n_references"]
                gold_id = splits.first_object_token(
                    t["object"], tok, old["trailing_space"]
                )
                rows.append(
                    dict(
                        subject=t["subject"],
                        object=t["object"],
                        model_top1_id=oldrow["model_top1_id"],
                        gold_id=gold_id,
                        n_references=int(count),
                        target_prior=float(prior[oldrow["model_top1_id"]]),
                        gold_prior=float(prior[gold_id]),
                    )
                )
            targets = torch.tensor([r["model_top1_id"] for r in rows], device=device)
            agg = {}
            with torch.no_grad():
                for name, hidden in (
                    ("raw_jacobian", raw),
                    ("logit_lens", test_states),
                    ("differential_logit_lens", test_states - centers),
                ):
                    values = sweep.decode(model, norm, hidden)
                    ranks = (
                        (values > values.gather(1, targets[:, None])).sum(-1).tolist()
                    )
                    agg[name] = sweep.aggregate(ranks)
                    for row, rank in zip(rows, ranks):
                        row["rank_" + name] = rank
            sweep.write(
                target,
                dict(
                    relation=relation,
                    model=a.name,
                    worker=a.worker,
                    protocol="shared_split",
                    prompt=PROMPT,
                    layer=layer,
                    split_fingerprint=old["split_fingerprint"],
                    source_sha256=sha(source),
                    data_sha256=sha(data_path),
                    n=len(rows),
                    rows=rows,
                    agg=agg,
                    answer_prior_contexts=NEUTRAL_CONTEXTS,
                    cardinality_full=len(data["samples"])
                    / len({x["object"] for x in data["samples"]}),
                    cardinality_filtered=len(samples)
                    / len({x["object"] for x in samples}),
                    cardinality_test=len(tests) / len({x["object"] for x in tests}),
                    elapsed_seconds=time.time() - start,
                ),
            )
            print("DONE", relation, round(time.time() - start, 1), flush=True)
    print("COMPLETE", a.name, flush=True)


if __name__ == "__main__":
    main()
