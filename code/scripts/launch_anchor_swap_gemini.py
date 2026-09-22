"""Wait for the source audits, freeze inputs, then run four local GPU workers."""

import concurrent.futures
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/anchor_swap_gemini"
RUN = ROOT / "generated/anchor_swap_gemini"
PYTHON = str(ROOT / ".venv/bin/python")
DOMAINS = ["athletes", "authors", "cities", "countries", "films", "landmarks"]


def worker(gpu, names):
    for name in names:
        with (RUN / f"{name}.log").open("w") as f:
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=str(gpu),
                OMP_NUM_THREADS="2",
                MKL_NUM_THREADS="2",
                PYTHONPATH=str(ROOT),
            )
            p = subprocess.run(
                [
                    PYTHON,
                    "-u",
                    str(ROOT / "scripts/run_anchor_swap_gemini.py"),
                    "--domain",
                    name,
                ],
                cwd=ROOT,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
            )
        if p.returncode:
            raise RuntimeError(f"{name} failed; see {RUN / name}.log")
        print("FINISHED", name, flush=True)


if __name__ == "__main__":
    while not (
        all((DATA / f"{name}_sourced.json").exists() for name in DOMAINS)
        and (DATA / "authors_wikidata_complete.flag").exists()
    ):
        time.sleep(10)
    subprocess.run(
        [PYTHON, str(ROOT / "scripts/run_anchor_swap_gemini.py"), "--freeze"],
        cwd=ROOT,
        check=True,
    )
    free = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    assert len(free.splitlines()) >= 4 and all(
        int(x) > 60000 for x in free.splitlines()[:4]
    ), "Four idle 80GB GPUs required"
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(worker, i, names)
            for i, names in enumerate(
                [
                    ["athletes", "authors"],
                    ["cities", "countries"],
                    ["films"],
                    ["landmarks"],
                ]
            )
        ]
        for f in concurrent.futures.as_completed(futures):
            f.result()
    subprocess.run(
        [PYTHON, str(ROOT / "scripts/report_anchor_swap_gemini.py")],
        cwd=ROOT,
        check=True,
    )
