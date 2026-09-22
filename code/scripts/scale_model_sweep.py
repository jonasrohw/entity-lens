"""Move completed small-model GPU allocations to the large Qwen comparison.

Run once on beta. Existing static Qwen workers migrate after their next atomic
relation result; advisory locks reserve their remaining assignments until then.
A singleton lock prevents two coordinators from managing the same workers.
"""

import fcntl
import json
import os
import shlex
import signal
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path

from launch_model_sweep import ROOT, RUN


def remote(host, command):
    return subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", host, command], text=True
    ).strip()


def launch(host, devices, index, model):
    command = [
        str(ROOT / ".venv/bin/python"),
        "-u",
        "-m",
        "jlens.model_sweep",
        "--model",
        model["model_id"],
        "--name",
        model["name"],
        "--lens-layer",
        str(model["lens_layer"]),
        "--layers",
        ",".join(map(str, model["layers"])),
        "--root",
        str(RUN / "results"),
        "--shard",
        str(index),
        "--batch-size",
        "8",
        "--dynamic",
    ]
    log = RUN / f"{model['name']}_{index}.log"
    if log.exists():
        log.rename(log.with_suffix(f".{int(time.time())}.previous"))
    pidfile = RUN / f"{model['name']}_{index}.pid"
    script = (
        f"cd {shlex.quote(str(ROOT))} || exit 1\n"
        f"CUDA_VISIBLE_DEVICES={shlex.quote(devices)} OMP_NUM_THREADS=2 "
        f"MKL_NUM_THREADS=2 HF_HUB_DISABLE_PROGRESS_BARS=1 "
        f"nohup {shlex.join(command)} > {shlex.quote(str(log))} 2>&1 < /dev/null & "
        f"echo $! > {shlex.quote(str(pidfile))}"
    )
    remote(host, script)
    entry = dict(host=host, devices=devices, shard=index, command=command, log=str(log))
    (RUN / f"dynamic_worker_{index}.json").write_text(
        json.dumps(entry, indent=2) + "\n"
    )
    print("LAUNCHED", host, devices, index, flush=True)


def idle(host, devices):
    query = (
        f"nvidia-smi -i {devices} --query-compute-apps=pid "
        "--format=csv,noheader,nounits"
    )
    return not remote(host, query)


def stop_worker(index, model):
    pid = int((RUN / f"{model['name']}_{index}.pid").read_text())
    process = Path(f"/proc/{pid}/cmdline")
    if process.exists():
        command = process.read_bytes().replace(b"\0", b" ")
        if (
            b"jlens.model_sweep" not in command
            or model["model_id"].encode() not in command
        ):
            raise RuntimeError(f"PID {pid} is not our Qwen worker")
        os.kill(pid, signal.SIGTERM)
        for _ in range(100):
            if not process.exists() or not process.read_bytes():
                break
            time.sleep(0.1)
        else:
            raise RuntimeError(f"Worker {pid} did not exit")


def main():
    singleton = (RUN / "scaler.lock").open("a")
    fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
    model = next(
        m
        for m in json.loads((ROOT / "results/five_model_sweep/models.json").read_text())
        if m["name"] == "qwen3_8_27b"
    )
    relations = sorted(
        (ROOT / "data/relations").glob("*.json"),
        key=lambda p: -len(json.loads(p.read_text())["samples"]),
    )
    reserved = {}
    for index in range(2):
        if (RUN / f"dynamic_worker_{index}.json").exists():
            continue
        stack = ExitStack()
        for protocol in ("main", "shared_split"):
            for relation in relations[index::2]:
                path = RUN / "results" / model["name"] / protocol / relation.name
                path.parent.mkdir(parents=True, exist_ok=True)
                handle = stack.enter_context(path.with_suffix(".lock").open("a"))
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reserved[index] = stack
    slots = [
        ("rho1", "0,1,2,3", 2),
        ("rho1", "4,5,6,7", 3),
        ("rho2", "0,1,2,3", 4),
        ("rho2", "4,5,6,7", 5),
    ]
    slots = [s for s in slots if not (RUN / f"dynamic_worker_{s[2]}.json").exists()]
    upgrades = [
        index
        for index in range(2)
        if (RUN / f"dynamic_worker_{index}.json").exists()
        and not (RUN / f"batch_upgrade_{index}.json").exists()
    ]
    while reserved or slots or upgrades:
        for index in list(upgrades):
            log = (RUN / f"{model['name']}_{index}.log").read_text()
            if any(line.startswith("DONE ") for line in log.splitlines()):
                stop_worker(index, model)
                launch("beta", "0,1" if index == 0 else "2,3", index, model)
                (RUN / f"batch_upgrade_{index}.json").write_text('{"batch_size": 8}\n')
                upgrades.remove(index)
        for index, stack in list(reserved.items()):
            log = (RUN / f"{model['name']}_{index}.log").read_text()
            if not any(line.startswith("DONE ") for line in log.splitlines()):
                continue
            stop_worker(index, model)
            stack.close()
            del reserved[index]
            launch("beta", "0,1" if index == 0 else "2,3", index, model)
        for slot in list(slots):
            host, _, index = slot
            pairs = (
                [("qwen3_8b_base", 0), ("mistral7b", 4)]
                if host == "rho1"
                else [("pythia6_9b", 0), ("gemma2_9b", 4)]
            )
            used = set()
            for path in RUN.glob("dynamic_worker_*.json"):
                worker = json.loads(path.read_text())
                if worker["host"] == host:
                    used.update(map(int, worker["devices"].split(",")))
            eligible = []
            for small_model, first_gpu in pairs:
                for shard in range(4):
                    log = (RUN / f"{small_model}_{shard}.log").read_text()
                    if first_gpu + shard not in used and any(
                        line.startswith("COMPLETE ") for line in log.splitlines()
                    ):
                        eligible.append(first_gpu + shard)
            available = []
            for device in sorted(eligible):
                if idle(host, str(device)):
                    available.append(device)
                if len(available) == 3:
                    break
            if len(available) == 3:
                devices = ",".join(map(str, available))
                launch(host, devices, index, model)
                slots.remove(slot)
        time.sleep(2)
    print("ALL ALLOCATIONS MIGRATED", flush=True)


if __name__ == "__main__":
    main()
