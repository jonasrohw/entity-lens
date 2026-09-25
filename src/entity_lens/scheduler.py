from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class Job(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    command: tuple[str, ...]
    gpus: int = Field(ge=1)


class Running(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    job: Job
    devices: tuple[int, ...]
    process: subprocess.Popen[bytes]
    log: Path


def available_devices(idle_memory_mib: int) -> list[int]:
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    usage = {int(index): int(used) for index, used in (line.split(",") for line in output.strip().splitlines())}
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    candidates = [int(device) for device in visible.split(",") if device.strip()] if visible else sorted(usage)
    return [device for device in candidates if usage[device] <= idle_memory_mib]


def run_jobs(jobs: Sequence[Job], log_dir: Path, devices: Sequence[int]) -> None:
    free = list(devices)
    if any(job.gpus > len(free) for job in jobs):
        message = f"A job needs more GPUs than the {len(free)} available"
        raise RuntimeError(message)
    log_dir.mkdir(parents=True, exist_ok=True)
    queue = sorted(jobs, key=lambda job: -job.gpus)
    running: list[Running] = []
    failures: list[str] = []
    while queue or running:
        for job in list(queue):
            if job.gpus <= len(free):
                assigned, free = tuple(free[: job.gpus]), free[job.gpus :]
                log = log_dir / f"{job.name}.log"
                environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, assigned))}
                handle = log.open("wb")
                process = subprocess.Popen(job.command, env=environment, stdout=handle, stderr=subprocess.STDOUT)
                running.append(Running(job=job, devices=assigned, process=process, log=log))
                queue.remove(job)
                print(f"started {job.name} on GPUs {assigned}", flush=True)
        for entry in list(running):
            code = entry.process.poll()
            if code is not None:
                running.remove(entry)
                free.extend(entry.devices)
                print(f"finished {entry.job.name} with exit code {code}", flush=True)
                if code:
                    failures.append(f"{entry.job.name} (see {entry.log})")
        time.sleep(5)
    if failures:
        message = "Failed jobs: " + ", ".join(failures)
        raise RuntimeError(message)


def cli_command(*arguments: str) -> tuple[str, ...]:
    return (sys.executable, "-m", "entity_lens", *arguments)
