"""Launch persistent relation workers across the three authorized GPU hosts."""

import argparse
import json
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "generated/five_model_sweep"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small-only", action="store_true")
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    models = {
        m["name"]: m
        for m in json.loads((ROOT / "results/five_model_sweep/models.json").read_text())
    }
    assignments = []
    for host, name, first_gpu in [
        ("rho1", "qwen3_8b_base", 0),
        ("rho1", "mistral7b", 4),
        ("rho2", "pythia6_9b", 0),
        ("rho2", "gemma2_9b", 4),
    ]:
        for shard in range(4):
            assignments.append((host, name, str(first_gpu + shard), shard, 4))
    if not args.small_only:
        assignments += [
            ("beta", "qwen3_8_27b", "0,1", 0, 2),
            ("beta", "qwen3_8_27b", "2,3", 1, 2),
        ]
    manifest = []
    for host, name, devices, shard, shards in assignments:
        model = models[name]
        log = RUN / f"{name}_{shard}.log"
        command = [
            str(ROOT / ".venv/bin/python"),
            "-u",
            "-m",
            "jlens.model_sweep",
            "--model",
            model["model_id"],
            "--name",
            name,
            "--lens-layer",
            str(model["lens_layer"]),
            "--layers",
            ",".join(map(str, model["layers"])),
            "--root",
            str(RUN / "results"),
            "--shard",
            str(shard),
            "--shards",
            str(shards),
            "--batch-size",
            "2",
        ]
        if name == "qwen3_8_27b":
            command.append("--dynamic")
        pidfile = RUN / f"{name}_{shard}.pid"
        script = (
            f"cd {shlex.quote(str(ROOT))} || exit 1\n"
            f"if test -f {shlex.quote(str(pidfile))} && kill -0 "
            f"$(cat {shlex.quote(str(pidfile))}) 2>/dev/null; then exit 0; fi; "
            f"CUDA_VISIBLE_DEVICES={shlex.quote(devices)} OMP_NUM_THREADS=2 "
            f"MKL_NUM_THREADS=2 HF_HUB_DISABLE_PROGRESS_BARS=1 "
            f"nohup {shlex.join(command)} > {shlex.quote(str(log))} 2>&1 < /dev/null & "
            f"echo $! > {shlex.quote(str(pidfile))}"
        )
        subprocess.run(["ssh", "-o", "BatchMode=yes", host, script], check=True)
        manifest.append(
            dict(
                host=host,
                name=name,
                devices=devices,
                shard=shard,
                shards=shards,
                command=command,
                log=str(log),
            )
        )
        print(host, name, shard, devices, flush=True)
    (RUN / "workers.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
