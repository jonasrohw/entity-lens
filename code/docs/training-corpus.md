# OLMo 3 7B training corpus download

`download_olmo_corpus.py` downloads every file from the three published datasets
for `allenai/Olmo-3-1025-7B`:

- `allenai/dolma3_mix-6T-1025-7B`
- `allenai/dolma3_dolmino_mix-100B-1025`
- `allenai/dolma3_longmino_mix-50B-1025`

It applies no topic, partition, or shard filters. Files stay in their published
compressed format. The initial inventory on 2026-09-17 contains 228,692 files,
totalling approximately 3.617 TB (decimal).

The dataset card reports that some science-PDF texts have been replaced with
`[REMOVED]`. This retrieves the complete currently published release, not the
unredacted historical training corpus. See
<https://huggingface.co/datasets/allenai/dolma3_mix-6T-1025-7B>.

## Run or resume

From the repository root, using a Python environment with `huggingface_hub`
(the existing `code/.venv` has version 1.31.0):

```bash
code/.venv/bin/python code/download_olmo_corpus.py --plan
code/.venv/bin/python -u code/download_olmo_corpus.py --workers 8
```

Default destination: `/mounts/work/jonas/data/olmo-3-1025-7b-training`.
Override with `--root /path/to/storage`. No GPU is needed.

The manifest pins repository commits on first run and is reused on subsequent
runs. Keep it to resume the same release. A lock prevents concurrent downloaders
from writing to the same destination. Interrupted transfers resume through the
Hugging Face local download cache. Each completed file is checked against its
published SHA-256 (LFS files) or Git blob SHA-1 (ordinary files). The verification
ledger skips previously verified files whose size and modification time match.

The worker retries failures up to 20 times with backoff. Exhausted failures are
recorded in the final status, and the process exits unsuccessfully. Rerun to retry
them. There is no automatic restart after a server reboot.

## Monitor

```bash
cat /mounts/work/jonas/data/olmo-3-1025-7b-training/status.json
tail -f /mounts/work/jonas/data/olmo-3-1025-7b-training/download.log
```

`manifest.json` lists every expected file, byte size, and checksum.
`verified.jsonl` records successful checksum checks. `status.json` updates about
every 30 seconds during downloading; `state: complete` means every manifest file
has passed verification. Check the PID/host and update time as well, since a
forcibly killed process cannot update its status. `download.pid` contains the PID
of the latest download process.

The initial job runs in the `olmo-corpus-download` tmux session on host `beta`.
To stop it, run `kill "$(cat /mounts/work/jonas/data/olmo-3-1025-7b-training/download.pid)"`
on that host. Downloaded data remains available for a later resume.
