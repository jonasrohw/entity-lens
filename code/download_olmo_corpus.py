#!/usr/bin/env python3
"""Download every published file for Olmo-3-1025-7B's three training stages.

Requires huggingface_hub. Run with --plan to pin revisions and check disk space.
Re-run the same command to resume; keep manifest.json and verified.jsonl.
The published 7B data contains upstream redactions (see its dataset card).
"""

import argparse
import concurrent.futures as futures
import fcntl
import hashlib
import json
import logging
import os
import shutil
import socket
import time
from pathlib import Path, PurePosixPath

# Bound Xet's per-file concurrency and keep temporary caches on the data mount.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_XET_NUM_CONCURRENT_RANGE_GETS", "4")
os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")

REPOS = (
    "allenai/dolma3_mix-6T-1025-7B",
    "allenai/dolma3_dolmino_mix-100B-1025",
    "allenai/dolma3_longmino_mix-50B-1025",
)
DEFAULT_ROOT = Path("/mounts/work/jonas/data/olmo-3-1025-7b-training")


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def retry(call, attempts, label):
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:
            if attempt == attempts:
                raise
            delay = min(300, 5 * 2 ** min(attempt - 1, 6))
            logging.warning(
                "%s: %s; retry %s/%s in %ss",
                label,
                type(exc).__name__,
                attempt,
                attempts,
                delay,
            )
            time.sleep(delay)


def manifest_for(root, attempts):
    from huggingface_hub import HfApi

    path = root / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if [r["repo_id"] for r in manifest["repos"]] != list(REPOS):
            raise ValueError(
                "Existing manifest does not match the required repositories"
            )
        return manifest
    api = HfApi()
    repos = []
    for repo_id in REPOS:
        logging.info("Enumerating ALL files: %s", repo_id)
        info = retry(
            lambda: api.dataset_info(repo_id, files_metadata=True), attempts, repo_id
        )
        files = []
        for item in info.siblings:
            p = PurePosixPath(item.rfilename)
            if p.is_absolute() or ".." in p.parts or item.size is None:
                raise ValueError(f"Invalid file metadata: {item.rfilename}")
            digest = item.lfs.sha256 if item.lfs else item.blob_id
            if not digest:
                raise ValueError(f"Missing checksum: {item.rfilename}")
            files.append(
                {
                    "path": item.rfilename,
                    "size": item.size,
                    "algorithm": "sha256" if item.lfs else "git-sha1",
                    "digest": digest,
                }
            )
        if not files:
            raise ValueError(f"Empty file listing: {repo_id}")
        repos.append({"repo_id": repo_id, "revision": info.sha, "files": files})
        logging.info(
            "%s: %s files, %.3f TB, revision %s",
            repo_id,
            len(files),
            sum(f["size"] for f in files) / 1e12,
            info.sha,
        )
    manifest = {
        "model": "allenai/Olmo-3-1025-7B",
        "created_at": time.time(),
        "scope": "All files in all three published dataset repositories",
        "limitation": "Upstream redacted PDF texts cannot be restored.",
        "repos": repos,
    }
    atomic_json(path, manifest)
    return manifest


def destination(root, repo, item):
    return root / repo["repo_id"].split("/")[1] / item["path"]


def checksum_ok(path, item):
    if not path.is_file() or path.stat().st_size != item["size"]:
        return False
    h = hashlib.sha256() if item["algorithm"] == "sha256" else hashlib.sha1()
    if item["algorithm"] == "git-sha1":
        h.update(f"blob {item['size']}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest() == item["digest"]


def fetch(root, repo, item, attempts):
    from huggingface_hub import hf_hub_download

    path = destination(root, repo, item)
    if checksum_ok(path, item):
        return path.stat().st_mtime_ns

    def transfer():
        # Force replacement only for an already completed but corrupt file.
        hf_hub_download(
            repo_id=repo["repo_id"],
            repo_type="dataset",
            revision=repo["revision"],
            filename=item["path"],
            local_dir=root / repo["repo_id"].split("/")[1],
            force_download=path.exists(),
        )
        if not checksum_ok(path, item):
            raise ValueError(f"Checksum mismatch: {path}")
        return path.stat().st_mtime_ns

    return retry(transfer, attempts, f"{repo['repo_id']}/{item['path']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=20)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.attempts < 1:
        parser.error("workers and attempts must be positive")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_XET_CACHE", str(root / ".xet"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(root / "download.log")],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    lock = (root / "download.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another downloader is already running for this directory")

    manifest = manifest_for(root, args.attempts)
    jobs = [(repo, item) for repo in manifest["repos"] for item in repo["files"]]
    total_bytes = sum(item["size"] for _, item in jobs)
    existing_bytes = 0
    for repo, item in jobs:
        p = destination(root, repo, item)
        if p.is_file() and p.stat().st_size == item["size"]:
            existing_bytes += item["size"]
    remaining_bytes = total_bytes - existing_bytes
    free = shutil.disk_usage(root).free
    logging.info(
        "Plan: %s files, %.3f TB total, %.3f TB remaining, %.3f TB free",
        len(jobs),
        total_bytes / 1e12,
        remaining_bytes / 1e12,
        free / 1e12,
    )
    if free < remaining_bytes + 50 * 1024**3:
        raise SystemExit("Insufficient disk space (including 50 GiB reserve)")
    if args.plan:
        return

    ledger_path = root / "verified.jsonl"
    verified = {}
    if ledger_path.exists():
        for line in ledger_path.read_text().splitlines():
            try:
                record = json.loads(line)
                verified[(record["repo_id"], record["path"])] = record
            except (ValueError, KeyError):
                pass  # A terminated process may leave a partial final record.
        # Remove any truncated final record before appending new ones.
        ledger_path.write_text("".join(json.dumps(r) + "\n" for r in verified.values()))
    pending = []
    done = done_bytes = 0
    for repo, item in jobs:
        record = verified.get((repo["repo_id"], item["path"]), {})
        p = destination(root, repo, item)
        if (
            record.get("digest") == item["digest"]
            and p.is_file()
            and p.stat().st_size == item["size"]
            and p.stat().st_mtime_ns == record.get("mtime_ns")
        ):
            done += 1
            done_bytes += item["size"]
        else:
            pending.append((repo, item))

    started = time.time()
    initial_bytes = done_bytes
    failures = []

    def status(state, active=0):
        value = {
            "state": state,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_at": started,
            "updated_at": time.time(),
            "files_verified": done,
            "files_total": len(jobs),
            "bytes_verified": done_bytes,
            "bytes_total": total_bytes,
            "active_files": active,
            "failed_files": failures,
            "verified_MB_per_second": (done_bytes - initial_bytes)
            / 1e6
            / max(1, time.time() - started),
        }
        atomic_json(root / "status.json", value)
        logging.info(
            "%s: %s/%s files verified, %.2f/%.2f GB, %s failures",
            state,
            done,
            len(jobs),
            done_bytes / 1e9,
            total_bytes / 1e9,
            len(failures),
        )

    (root / "download.pid").write_text(str(os.getpid()) + "\n")
    status("downloading")
    last_report = time.monotonic()
    try:
        with (
            ledger_path.open("a", buffering=1) as ledger,
            futures.ThreadPoolExecutor(max_workers=args.workers) as pool,
        ):
            queue = iter(pending)
            active = {}

            def fill():
                while len(active) < args.workers:
                    job = next(queue, None)
                    if job is None:
                        break
                    repo, item = job
                    active[pool.submit(fetch, root, repo, item, args.attempts)] = job

            fill()
            while active:
                finished, _ = futures.wait(
                    active, timeout=30, return_when=futures.FIRST_COMPLETED
                )
                for future in finished:
                    repo, item = active.pop(future)
                    try:
                        mtime = future.result()
                    except Exception as exc:
                        failures.append(
                            {
                                "repo_id": repo["repo_id"],
                                "path": item["path"],
                                "error_type": type(exc).__name__,
                            }
                        )
                        logging.error(
                            "Failed: %s/%s (%s)",
                            repo["repo_id"],
                            item["path"],
                            type(exc).__name__,
                        )
                    else:
                        ledger.write(
                            json.dumps(
                                {"repo_id": repo["repo_id"], **item, "mtime_ns": mtime}
                            )
                            + "\n"
                        )
                        done += 1
                        done_bytes += item["size"]
                fill()
                if time.monotonic() - last_report >= 30:
                    status("downloading", len(active))
                    last_report = time.monotonic()
        status("failed" if failures else "complete")
    except BaseException:
        status("interrupted")
        raise
    if failures:
        raise SystemExit(
            "Some files failed; re-run to resume only missing/unverified files"
        )


if __name__ == "__main__":
    main()
