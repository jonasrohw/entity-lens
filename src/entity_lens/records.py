from __future__ import annotations

import gzip
import hashlib
import os
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from entity_lens.splits import Split


class Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DerivativeCheck(Record):
    decode_max_abs_error: float
    batched_scalar_relative_error: float
    finite_difference_relative_error: float


class PrefixCheck(Record):
    n_tokens: int
    prefix_length: int
    layer: int
    state_relative_error: float
    jvp_relative_error: float


class ConvolutionCheck(Record):
    logits_max_abs_error: float
    jvp_relative_error: float


class RuntimeChecks(Record):
    convolution: ConvolutionCheck | None
    derivative: DerivativeCheck
    prefix: tuple[PrefixCheck, ...]


class Readout(StrEnum):
    CEL = "cel"
    NEL = "nel"
    LOGIT_LENS = "logit_lens"
    DIFFERENTIAL_LOGIT_LENS = "differential_logit_lens"


class GridCell(Record):
    layer: int
    beta: float
    faithfulness: float


class LreRow(Record):
    subject: str
    object: str
    target_id: int
    target: str
    gold_id: int
    prediction_id: int
    hit: bool
    rank: int
    grid_predictions: tuple[int, ...]


class LreMeasurement(Record):
    split: Split
    trailing_space: bool
    grid: tuple[GridCell, ...]
    best_layer: int
    best_beta: float
    faithfulness: float
    rows: tuple[LreRow, ...]


class LensRow(Record):
    subject: str
    object: str
    target_id: int
    target: str
    n_references: int
    ranks: dict[Readout, int]
    literal_ranks: dict[Readout, int]
    anchored_ranks: dict[float, int]
    top_tokens: dict[Readout, tuple[str, ...]]
    logit_lens_top_ids: tuple[int, ...]
    answer_prior: float


class LensMeasurement(Record):
    split: Split
    prompt: str
    layer: int
    rows: tuple[LensRow, ...]

    def score(self, readout: Readout) -> float:
        import numpy as np

        ranks = np.array([row.ranks[readout] for row in self.rows], dtype=float)
        return -float(np.median(np.log10(ranks + 1)))


class YearRow(Record):
    subject: str
    object: str
    first_token: str
    joint_token: str
    ignored: bool


class YearAudit(Record):
    rows: tuple[YearRow, ...]

    @property
    def excluded(self) -> bool:
        return any(row.ignored for row in self.rows)


class RelationMeasurement(Record):
    model: str
    relation: str
    data_sha256: str
    lre: LreMeasurement
    lens: LensMeasurement
    year_audit: YearAudit | None
    seconds: float


class WorkerRecord(Record):
    model: str
    hf_id: str
    revision: str
    torch: str
    transformers: str
    device: str
    checks: RuntimeChecks
    source_sha256: dict[str, str]


def write(path: Path, record: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (record.model_dump_json(indent=1) + "\n").encode()
    if path.suffix == ".gz":
        payload = gzip.compress(payload, compresslevel=9, mtime=0)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def read[RecordT: BaseModel](path: Path, kind: type[RecordT]) -> RecordT:
    payload = path.read_bytes()
    if path.suffix == ".gz":
        payload = gzip.decompress(payload)
    return kind.model_validate_json(payload)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
