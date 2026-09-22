"""Confirm frozen settings as grid workers finish, then resume saved Qwen jobs."""

import fcntl
import json
import shlex
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "generated/mistral_full_grid"
OLD = ROOT / "generated/five_model_sweep"


def remote(host, command):
    return subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", host, command], text=True
    ).strip()


def enough_memory(host, devices, minimum_gib):
    free = remote(
        host,
        f"nvidia-smi -i {devices} --query-gpu=memory.free --format=csv,noheader,nounits",
    )
    return all(int(value) >= minimum_gib * 1024 for value in free.splitlines())


def launch(host, devices, command, log):
    script = (
        f"cd {shlex.quote(str(ROOT))} || exit 1\n"
        f"CUDA_VISIBLE_DEVICES={shlex.quote(devices)} OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_DISABLE_PROGRESS_BARS=1 "
        f"nohup {shlex.join(command)} > {shlex.quote(str(log))} 2>&1 < /dev/null &"
        f"\necho $! > {shlex.quote(str(log.with_suffix('.pid')))}"
    )
    remote(host, script)


def completed(path, prefix):
    return path.exists() and any(
        line.startswith(prefix) for line in path.read_text().splitlines()
    )


def main():
    singleton = (RUN / "orchestration.lock").open("a")
    fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
    workers = [("beta", i) for i in range(4)] + [("rho2", i) for i in range(4, 8)]
    while True:
        for host, device in workers:
            worker = f"{host}_{device}"
            marker = RUN / f"confirmation_worker_{worker}.json"
            if marker.exists() or not (RUN / "confirmation_config.json").exists():
                continue
            if completed(RUN / f"{worker}.log", "COMPLETE ") and enough_memory(
                host, str(device), 35
            ):
                command = [
                    str(ROOT / ".venv/bin/python"),
                    "-u",
                    "scripts/run_mistral_grid.py",
                    "--worker",
                    f"confirm_{worker}",
                    "--confirm",
                ]
                launch(host, str(device), command, RUN / f"confirm_{worker}.log")
                marker.write_text(
                    json.dumps(
                        dict(host=host, device=device, command=command), indent=2
                    )
                    + "\n"
                )
                print("CONFIRM_LAUNCHED", worker, flush=True)
        resumed = 0
        for index, host, devices, paused in [
            (0, "beta", "0,1", "paused_qwen_worker.json"),
            (1, "beta", "2,3", "paused_qwen_worker_1.json"),
            (4, "rho2", "5,6,7", "paused_qwen_worker_4.json"),
        ]:
            marker = RUN / f"qwen_resumed_{index}.json"
            if marker.exists():
                resumed += 1
                continue
            if not all(
                completed(RUN / f"confirm_{host}_{device}.log", "CONFIRM_COMPLETE ")
                for device in devices.split(",")
            ):
                continue
            if not enough_memory(host, devices, 70 if host == "beta" else 45):
                continue
            state = json.loads((RUN / paused).read_text())
            command = state["command"]
            log = OLD / f"qwen3_8_27b_{index}.log"
            if log.exists():
                log.rename(log.with_suffix(f".{int(time.time())}.before_grid_resume"))
            launch(host, devices, command, log)
            marker.write_text(
                json.dumps(dict(host=host, devices=devices, command=command), indent=2)
                + "\n"
            )
            print("QWEN_RESUMED", index, flush=True)
            resumed += 1
        if resumed == 3:
            break
        time.sleep(3)
    print("TRANSFERS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
