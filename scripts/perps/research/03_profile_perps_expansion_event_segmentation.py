#!/usr/bin/env python
"""
Profile deterministic event segmentation for 15-minute / 50-bps expansion labels.

Purpose
-------
The upstream label builder correctly produces one positive label for every
context-ready decision second whose exact future path from ``t + 1`` through
``t + 901`` contains at least 50 bps of high-low expansion. Those rolling
labels are deliberately overlapping and must not be treated as independent
market events.

This script does not train a model and does not alter the upstream labels. It
compares several deterministic ways of grouping positive rolling anchors into
finer event instances:

* ``bridge_000s``: strictly contiguous positive runs;
* ``bridge_030s``: bridge at most 30 intervening non-positive seconds;
* ``bridge_060s``: bridge at most 60 intervening non-positive seconds;
* ``bridge_300s``: bridge at most 300 intervening non-positive seconds;
* ``bridge_900s``: bridge at most 900 intervening non-positive seconds;
* ``overlap_reference``: reproduce the existing connected-overlapping-future-
  interval regime rule exactly.

Bridge-gap semantics
--------------------
For consecutive positive decision seconds ``p`` and ``q``:

    inactive_gap_seconds = q - p - 1

They remain in the same candidate event when:

    inactive_gap_seconds <= bridge_gap_seconds

Therefore a zero-second bridge means every decision second in an event is
positive. The existing future-interval overlap rule is equivalent to a maximum
inactive gap of 899 seconds under the fixed ``[t + 1, t + 901]`` path contract.
The script reconstructs both forms independently and validates that equivalence.
The requested 900-second candidate is retained as a sensitivity case and is one
second more permissive than the exact overlap reference.

Outputs
-------
The script writes separate research outputs and never adds future-derived event
IDs to causal feature files or upstream row-label partitions.

Default dataset root:
    data/derived/perps/research/expansion_event_segmentation_sensitivity_v1

Default report root:
    reports/perps_expansion_event_segmentation_sensitivity

Dependencies
------------
    pip install numpy pandas pyarrow

Typical Git Bash usage
----------------------
    python scripts/perps/research/03_profile_perps_expansion_event_segmentation.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sqlite3
import sys
import traceback
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


PROFILER_VERSION = "1.0.0"
SEGMENTATION_SCHEMA_VERSION = 1
EXPECTED_LABEL_SCHEMA_VERSION = 1
EXPECTED_LABEL_BUILDER_VERSION = "1.0.0"
HORIZON_SECONDS = 15 * 60
PATH_OBSERVATIONS = HORIZON_SECONDS + 1
THRESHOLD_BPS = 50.0
OVERLAP_EQUIVALENT_INACTIVE_GAP_SECONDS = HORIZON_SECONDS - 1
DEFAULT_BRIDGE_GAPS = (0, 30, 60, 300, 900)
DEFAULT_TREND_DOMINANCE_RATIO = 0.75
SYMBOL_PREFIX = "symbol="
DATE_PREFIX = "date="
SEGMENTATION_PREFIX = "segmentation="

REQUIRED_LABEL_COLUMNS = (
    "symbol",
    "decision_second",
    "entry_second",
    "path_end_second",
    "future_swing_15m_bps",
    "expansion_50_15m",
    "future_upside_15m_bps",
    "future_downside_15m_bps",
    "future_terminal_return_15m_bps",
    "future_high_first",
    "future_high_second",
    "future_low_second",
    "first_50bps_realization_second",
)

POSITIVE_NUMERIC_COLUMNS = (
    "decision_second",
    "entry_second",
    "path_end_second",
    "future_swing_15m_bps",
    "future_upside_15m_bps",
    "future_downside_15m_bps",
    "future_terminal_return_15m_bps",
    "future_high_second",
    "future_low_second",
    "first_50bps_realization_second",
)

REPORT_FILENAMES = (
    "perps_expansion_event_segmentation_comparison.csv",
    "perps_expansion_event_segmentation_transition.csv",
    "perps_expansion_event_segmentation_metric_profile.csv",
    "perps_expansion_event_segmentation_duration_distribution.csv",
    "perps_expansion_event_segmentation_daily.csv",
    "perps_expansion_event_segmentation_reference_fragmentation.csv",
    "perps_expansion_event_segmentation_top_longest_events.csv",
    "perps_expansion_event_segmentation_invariants.csv",
    "perps_expansion_event_segmentation_summary.json",
    "perps_expansion_event_segmentation_config.json",
    "perps_expansion_event_segmentation_findings.md",
)


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat().replace("+00:00", "Z")


def iso_from_second(value: int | float | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return datetime.fromtimestamp(int(value), tz=timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def date_from_second(value: int) -> str:
    return datetime.fromtimestamp(int(value), tz=timezone.utc).date().isoformat()


def parse_date(value: str | None, label: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise SystemExit(f"{label} must use YYYY-MM-DD format: {value}") from exc


def compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def percent(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return float("nan")
    return 100.0 * float(numerator) / float(denominator)


def finite_values(values: Iterable[Any]) -> pd.Series:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce")
    return series[np.isfinite(series.to_numpy(dtype=float, na_value=np.nan))]


def safe_stat(values: Iterable[Any], statistic: str) -> float:
    series = finite_values(values)
    if not len(series):
        return float("nan")
    if statistic == "mean":
        return float(series.mean())
    if statistic == "median":
        return float(series.median())
    if statistic == "minimum":
        return float(series.min())
    if statistic == "maximum":
        return float(series.max())
    if statistic.startswith("p"):
        return float(series.quantile(float(statistic[1:]) / 100.0))
    raise ValueError(f"Unknown statistic: {statistic}")


def parse_bridge_gaps(value: str) -> tuple[int, ...]:
    try:
        gaps = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    except ValueError as exc:
        raise SystemExit("--bridge-gaps must be comma-separated non-negative integers") from exc
    if not gaps:
        raise SystemExit("--bridge-gaps must contain at least one value")
    if gaps[0] < 0:
        raise SystemExit("--bridge-gaps values cannot be negative")
    if gaps[-1] > 86_400:
        raise SystemExit("--bridge-gaps values above 86,400 seconds are not supported")
    return tuple(gaps)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - runtime dependency check
        raise SystemExit(
            "Missing dependency 'pyarrow'. Activate the project virtual environment "
            "and run: python -m pip install numpy pandas pyarrow"
        ) from exc
    return pa, pq


def metadata_bytes(metadata: Mapping[str, Any]) -> dict[bytes, bytes]:
    return {
        str(key).encode("utf-8"): str(value).encode("utf-8")
        for key, value in metadata.items()
        if value is not None
    }


def decode_metadata(schema: Any) -> dict[str, str]:
    raw = schema.metadata or {}
    return {key.decode("utf-8"): value.decode("utf-8") for key, value in raw.items()}


def dataframe_to_table(frame: pd.DataFrame, metadata: Mapping[str, Any]) -> Any:
    pa, _ = require_pyarrow()
    table = pa.Table.from_pandas(frame, preserve_index=False)
    return table.replace_schema_metadata(metadata_bytes(metadata))


def atomic_write_parquet(
    path: Path,
    frame: pd.DataFrame,
    metadata: Mapping[str, Any],
    compression_level: int,
) -> None:
    _, pq = require_pyarrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    table = dataframe_to_table(frame, metadata)
    try:
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            compression_level=compression_level,
            use_dictionary=True,
            write_statistics=True,
        )
        written = pq.ParquetFile(temporary)
        try:
            if written.metadata.num_rows != len(frame):
                raise RuntimeError(
                    f"Atomic Parquet validation failed for {path}: "
                    f"expected {len(frame):,} rows, found {written.metadata.num_rows:,}"
                )
            written_metadata = decode_metadata(written.schema_arrow)
            if written_metadata.get("input_signature") != str(metadata.get("input_signature")):
                raise RuntimeError(f"Input-signature metadata mismatch for {path}")
        finally:
            close = getattr(written, "close", None)
            if callable(close):
                close()
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def cleanup_extra_parquet_files(directory: Path, keep: Path) -> None:
    if not directory.exists():
        return
    for path in directory.glob("*.parquet"):
        if path.resolve() != keep.resolve():
            path.unlink()


# ---------------------------------------------------------------------------
# Input discovery and state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelPartition:
    decision_date: date
    files: tuple[Path, ...]
    rows: int
    minimum_second: int | None
    maximum_second: int | None
    dependency_signatures: tuple[str, ...]
    signature: str


def extract_partition_date(path: Path) -> date:
    for parent in path.parents:
        if parent.name.startswith(DATE_PREFIX):
            return date.fromisoformat(parent.name[len(DATE_PREFIX) :])
    raise RuntimeError(f"Could not infer decision date from path: {path}")


def discover_label_partitions(
    input_root: Path,
    symbol: str,
    expected_builder_version: str,
    hash_input_files: bool,
    start_date: date | None,
    end_date: date | None,
    max_dates: int | None,
) -> dict[date, LabelPartition]:
    _, pq = require_pyarrow()
    symbol_root = input_root / "row_labels" / f"{SYMBOL_PREFIX}{symbol}"
    files = sorted(symbol_root.glob(f"{DATE_PREFIX}*/*.parquet"))
    if not files:
        raise RuntimeError(f"No row-label Parquet files found under: {symbol_root}")

    grouped: dict[date, list[Path]] = {}
    for path in files:
        day = extract_partition_date(path)
        if start_date and day < start_date:
            continue
        if end_date and day > end_date:
            continue
        grouped.setdefault(day, []).append(path)
    days = sorted(grouped)
    if max_dates is not None:
        if max_dates < 1:
            raise SystemExit("--max-dates must be at least 1")
        days = days[:max_dates]
    if not days:
        raise RuntimeError("No label partitions remain after date filtering")

    output: dict[date, LabelPartition] = {}
    for day in days:
        rows = 0
        minimum_second: int | None = None
        maximum_second: int | None = None
        dependencies: list[str] = []
        signatures: list[dict[str, Any]] = []
        for path in sorted(grouped[day]):
            parquet_file = pq.ParquetFile(path)
            try:
                schema_names = set(parquet_file.schema_arrow.names)
                missing = sorted(set(REQUIRED_LABEL_COLUMNS) - schema_names)
                if missing:
                    raise RuntimeError(f"{path} is missing required columns: {missing}")
                metadata = decode_metadata(parquet_file.schema_arrow)
                builder = metadata.get("perps_expansion_label_builder_version")
                schema_version = metadata.get("perps_expansion_label_schema_version")
                file_symbol = metadata.get("symbol")
                threshold = metadata.get("threshold_bps")
                metadata_date = metadata.get("decision_date")
                if expected_builder_version and builder != expected_builder_version:
                    raise RuntimeError(
                        f"{path} label builder version is {builder!r}; expected "
                        f"{expected_builder_version!r}"
                    )
                if schema_version != str(EXPECTED_LABEL_SCHEMA_VERSION):
                    raise RuntimeError(
                        f"{path} label schema version is {schema_version!r}; expected "
                        f"{EXPECTED_LABEL_SCHEMA_VERSION}"
                    )
                if file_symbol and file_symbol.upper() != symbol:
                    raise RuntimeError(f"{path} symbol metadata is {file_symbol!r}")
                if metadata_date and metadata_date != day.isoformat():
                    raise RuntimeError(f"{path} decision-date metadata is {metadata_date!r}")
                if threshold is not None and not math.isclose(
                    float(threshold), THRESHOLD_BPS, rel_tol=0, abs_tol=1e-12
                ):
                    raise RuntimeError(f"{path} threshold is {threshold!r}; expected 50 bps")
                file_rows = int(parquet_file.metadata.num_rows)
                rows += file_rows
                dependency = metadata.get("dependency_signature", "")
                dependencies.append(dependency)
                stat = path.stat()
                signature_row: dict[str, Any] = {
                    "path": str(path.resolve()),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "rows": file_rows,
                    "dependency_signature": dependency,
                }
                if hash_input_files:
                    signature_row["sha256"] = sha256_file(path)
                signatures.append(signature_row)

                if file_rows:
                    table = parquet_file.read(
                        columns=["decision_second"],
                        use_threads=True,
                    )
                    values = table.column(0).to_numpy(zero_copy_only=False)
                    file_min = int(np.min(values))
                    file_max = int(np.max(values))
                    minimum_second = (
                        file_min if minimum_second is None else min(minimum_second, file_min)
                    )
                    maximum_second = (
                        file_max if maximum_second is None else max(maximum_second, file_max)
                    )
            finally:
                close = getattr(parquet_file, "close", None)
                if callable(close):
                    close()
        signature = sha256_text(compact_json(signatures))
        output[day] = LabelPartition(
            decision_date=day,
            files=tuple(sorted(grouped[day])),
            rows=rows,
            minimum_second=minimum_second,
            maximum_second=maximum_second,
            dependency_signatures=tuple(dependencies),
            signature=signature,
        )
    return output


def connect_state(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            started_at_utc TEXT NOT NULL,
            completed_at_utc TEXT,
            status TEXT NOT NULL,
            symbol TEXT NOT NULL,
            input_signature TEXT,
            config_signature TEXT,
            positive_rows INTEGER,
            segmentation_count INTEGER,
            error_text TEXT,
            configuration_json TEXT NOT NULL,
            summary_json TEXT
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS input_partitions (
            symbol TEXT NOT NULL,
            decision_date TEXT NOT NULL,
            partition_signature TEXT NOT NULL,
            rows INTEGER NOT NULL,
            minimum_second INTEGER,
            maximum_second INTEGER,
            files_json TEXT NOT NULL,
            last_seen_run_id TEXT NOT NULL,
            PRIMARY KEY (symbol, decision_date)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS latest_build (
            symbol TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            input_signature TEXT NOT NULL,
            config_signature TEXT NOT NULL,
            completed_at_utc TEXT,
            positive_rows INTEGER,
            segmentation_count INTEGER,
            output_manifest_json TEXT,
            error_text TEXT
        )
        """
    )
    connection.commit()
    return connection


def record_partitions(
    connection: sqlite3.Connection,
    symbol: str,
    run_id: str,
    partitions: Mapping[date, LabelPartition],
) -> None:
    for day, partition in partitions.items():
        connection.execute(
            """
            INSERT INTO input_partitions (
                symbol, decision_date, partition_signature, rows,
                minimum_second, maximum_second, files_json, last_seen_run_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, decision_date) DO UPDATE SET
                partition_signature=excluded.partition_signature,
                rows=excluded.rows,
                minimum_second=excluded.minimum_second,
                maximum_second=excluded.maximum_second,
                files_json=excluded.files_json,
                last_seen_run_id=excluded.last_seen_run_id
            """,
            (
                symbol,
                day.isoformat(),
                partition.signature,
                partition.rows,
                partition.minimum_second,
                partition.maximum_second,
                json.dumps([str(path) for path in partition.files]),
                run_id,
            ),
        )
    connection.commit()


def combined_input_signature(partitions: Mapping[date, LabelPartition]) -> str:
    return sha256_text(
        compact_json(
            [
                {
                    "decision_date": day.isoformat(),
                    "signature": partitions[day].signature,
                    "rows": partitions[day].rows,
                }
                for day in sorted(partitions)
            ]
        )
    )


# ---------------------------------------------------------------------------
# Positive-anchor loading and validation
# ---------------------------------------------------------------------------


def read_positive_rows(path: Path) -> pd.DataFrame:
    _, pq = require_pyarrow()
    try:
        table = pq.read_table(
            path,
            columns=list(REQUIRED_LABEL_COLUMNS),
            filters=[("expansion_50_15m", "=", True)],
            use_threads=True,
        )
    except TypeError:
        # Older PyArrow versions may not support filters through read_table for
        # every file source. The fallback still reads only required columns.
        table = pq.read_table(path, columns=list(REQUIRED_LABEL_COLUMNS), use_threads=True)
    frame = table.to_pandas(split_blocks=True, self_destruct=True)
    if len(frame):
        frame = frame[frame["expansion_50_15m"].fillna(False).astype(bool)].copy()
    return frame


def load_positive_anchors(
    partitions: Mapping[date, LabelPartition],
    symbol: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    daily_rows: list[dict[str, Any]] = []
    invariants: list[dict[str, Any]] = []

    for day in sorted(partitions):
        day_frames = [read_positive_rows(path) for path in partitions[day].files]
        day_frames = [frame for frame in day_frames if len(frame)]
        day_frame = (
            pd.concat(day_frames, ignore_index=True)
            if day_frames
            else pd.DataFrame(columns=list(REQUIRED_LABEL_COLUMNS))
        )
        if len(day_frame):
            if not day_frame["symbol"].astype(str).str.upper().eq(symbol).all():
                raise RuntimeError(f"Unexpected symbol found in positive rows for {day}")
            frames.append(day_frame)
        daily_rows.append(
            {
                "decision_date": day.isoformat(),
                "valid_label_rows": int(partitions[day].rows),
                "positive_rows": int(len(day_frame)),
                "positive_share_pct": percent(len(day_frame), partitions[day].rows),
            }
        )

    if frames:
        positives = pd.concat(frames, ignore_index=True)
    else:
        positives = pd.DataFrame(columns=list(REQUIRED_LABEL_COLUMNS))

    if len(positives):
        positives = positives.sort_values("decision_second", kind="mergesort").reset_index(
            drop=True
        )
        for column in POSITIVE_NUMERIC_COLUMNS:
            positives[column] = pd.to_numeric(positives[column], errors="raise")
        integer_columns = (
            "decision_second",
            "entry_second",
            "path_end_second",
            "future_high_second",
            "future_low_second",
            "first_50bps_realization_second",
        )
        for column in integer_columns:
            positives[column] = positives[column].astype("int64")
        positives["future_high_first"] = positives["future_high_first"].astype("boolean")

    def add(name: str, passed: bool, detail: str) -> None:
        invariants.append(
            {
                "scope": "INPUT",
                "segmentation_id": None,
                "invariant": name,
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
            }
        )

    add(
        "POSITIVE_DECISION_SECONDS_UNIQUE",
        not positives["decision_second"].duplicated().any() if len(positives) else True,
        f"positive_rows={len(positives):,}",
    )
    add(
        "POSITIVE_DECISION_SECONDS_STRICTLY_INCREASING",
        bool(
            len(positives) < 2
            or np.all(np.diff(positives["decision_second"].to_numpy(dtype=np.int64)) > 0)
        ),
        f"positive_rows={len(positives):,}",
    )
    add(
        "ENTRY_IS_EXACTLY_DECISION_PLUS_ONE",
        bool(
            not len(positives)
            or np.all(
                positives["entry_second"].to_numpy(dtype=np.int64)
                == positives["decision_second"].to_numpy(dtype=np.int64) + 1
            )
        ),
        "entry_second=decision_second+1",
    )
    add(
        "PATH_END_IS_EXACTLY_ENTRY_PLUS_900",
        bool(
            not len(positives)
            or np.all(
                positives["path_end_second"].to_numpy(dtype=np.int64)
                == positives["entry_second"].to_numpy(dtype=np.int64) + HORIZON_SECONDS
            )
        ),
        f"horizon_seconds={HORIZON_SECONDS}",
    )
    add(
        "ALL_ROWS_AT_OR_ABOVE_50_BPS",
        bool(
            not len(positives)
            or np.all(
                positives["future_swing_15m_bps"].to_numpy(dtype=float) + 1e-10
                >= THRESHOLD_BPS
            )
        ),
        f"threshold_bps={THRESHOLD_BPS}",
    )
    add(
        "REALIZATION_INSIDE_EXACT_FUTURE_PATH",
        bool(
            not len(positives)
            or np.all(
                (
                    positives["first_50bps_realization_second"].to_numpy(dtype=np.int64)
                    >= positives["entry_second"].to_numpy(dtype=np.int64)
                )
                & (
                    positives["first_50bps_realization_second"].to_numpy(dtype=np.int64)
                    <= positives["path_end_second"].to_numpy(dtype=np.int64)
                )
            )
        ),
        "entry_second<=first_realization<=path_end_second",
    )
    failures = [row for row in invariants if row["status"] == "FAIL"]
    if failures:
        raise RuntimeError(
            "Input-label validation failed: "
            + ", ".join(row["invariant"] for row in failures)
        )
    return positives, pd.DataFrame(daily_rows), invariants


# ---------------------------------------------------------------------------
# Segmentation logic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SegmentationSpec:
    segmentation_id: str
    rule_type: str
    bridge_gap_seconds: int | None
    description: str


def bridge_spec(gap: int) -> SegmentationSpec:
    width = max(3, len(str(gap)))
    code = f"bridge_{gap:0{width}d}s"
    return SegmentationSpec(
        segmentation_id=code,
        rule_type="MAX_INACTIVE_GAP",
        bridge_gap_seconds=gap,
        description=f"Continue while intervening non-positive seconds <= {gap}",
    )


def reference_spec() -> SegmentationSpec:
    return SegmentationSpec(
        segmentation_id="overlap_reference",
        rule_type="OVERLAPPING_FUTURE_INTERVALS",
        bridge_gap_seconds=None,
        description="Connected components of closed future intervals [t+1, t+901]",
    )


def bridge_group_ordinals(decision_seconds: np.ndarray, bridge_gap: int) -> np.ndarray:
    if len(decision_seconds) == 0:
        return np.empty(0, dtype=np.int64)
    inactive_gaps = np.diff(decision_seconds) - 1
    starts = np.concatenate((np.array([True]), inactive_gaps > bridge_gap))
    return np.cumsum(starts, dtype=np.int64)


def overlap_reference_ordinals(
    entry_seconds: np.ndarray,
    path_end_seconds: np.ndarray,
) -> np.ndarray:
    if len(entry_seconds) == 0:
        return np.empty(0, dtype=np.int64)
    groups = np.empty(len(entry_seconds), dtype=np.int64)
    ordinal = 1
    groups[0] = ordinal
    union_end = int(path_end_seconds[0])
    for index in range(1, len(entry_seconds)):
        start = int(entry_seconds[index])
        end = int(path_end_seconds[index])
        if start <= union_end:
            union_end = max(union_end, end)
        else:
            ordinal += 1
            union_end = end
        groups[index] = ordinal
    return groups


def group_bounds(ordinals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(ordinals) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    starts = np.flatnonzero(np.concatenate((np.array([True]), ordinals[1:] != ordinals[:-1])))
    ends = np.concatenate((starts[1:], np.array([len(ordinals)])))
    return starts.astype(np.int64), ends.astype(np.int64)


def classify_path(
    upside_bps: float,
    downside_bps: float,
    swing_bps: float,
    dominance_ratio: float,
) -> tuple[bool, str, float]:
    if not math.isfinite(swing_bps) or swing_bps <= 0:
        return False, "TWO_SIDED", float("nan")
    share = max(upside_bps, downside_bps) / swing_bps
    if share + 1e-12 < dominance_ratio:
        return False, "TWO_SIDED", share
    if upside_bps > downside_bps:
        return True, "TREND_UP", share
    if downside_bps > upside_bps:
        return True, "TREND_DOWN", share
    return False, "TWO_SIDED", share


def stable_event_id(symbol: str, segmentation_id: str, first_second: int) -> str:
    stamp = datetime.fromtimestamp(first_second, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    compact_segmentation = segmentation_id.upper().replace("_", "-")
    return f"EXP50-15M-EVT-{symbol}-{compact_segmentation}-{stamp}"


def build_events_and_membership(
    positives: pd.DataFrame,
    spec: SegmentationSpec,
    ordinals: np.ndarray,
    reference_ordinals: np.ndarray,
    reference_event_ids: Sequence[str] | None,
    trend_dominance_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    decision = positives["decision_second"].to_numpy(dtype=np.int64)
    entry = positives["entry_second"].to_numpy(dtype=np.int64)
    path_end = positives["path_end_second"].to_numpy(dtype=np.int64)
    swing = positives["future_swing_15m_bps"].to_numpy(dtype=float)
    upside = positives["future_upside_15m_bps"].to_numpy(dtype=float)
    downside = positives["future_downside_15m_bps"].to_numpy(dtype=float)
    terminal = positives["future_terminal_return_15m_bps"].to_numpy(dtype=float)
    future_high = positives["future_high_second"].to_numpy(dtype=np.int64)
    future_low = positives["future_low_second"].to_numpy(dtype=np.int64)
    realization = positives["first_50bps_realization_second"].to_numpy(dtype=np.int64)
    high_first_series = positives["future_high_first"]

    starts, ends = group_bounds(ordinals)
    event_rows: list[dict[str, Any]] = []
    event_ids: list[str] = []
    representative_indices = np.zeros(len(starts), dtype=np.int64)
    first_realization_indices = np.zeros(len(starts), dtype=np.int64)
    previous_end: int | None = None
    previous_union_end: int | None = None

    for event_index, (start_raw, end_raw) in enumerate(zip(starts, ends), start=1):
        start = int(start_raw)
        end = int(end_raw)
        group_slice = slice(start, end)
        first_decision = int(decision[start])
        last_decision = int(decision[end - 1])
        event_span = last_decision - first_decision + 1
        group_count = end - start
        internal_gaps = np.diff(decision[group_slice]) - 1
        positive_runs = 1 + int(np.sum(internal_gaps > 0)) if group_count else 0
        maximum_inactive_gap = int(np.max(internal_gaps)) if len(internal_gaps) else 0
        total_internal_inactive = int(np.sum(internal_gaps)) if len(internal_gaps) else 0

        local_representative = int(np.argmax(swing[group_slice]))
        representative_index = start + local_representative
        representative_indices[event_index - 1] = representative_index
        local_realization = int(np.argmin(realization[group_slice]))
        realization_index = start + local_realization
        first_realization_indices[event_index - 1] = realization_index

        trend_like, path_type, dominant_share = classify_path(
            float(upside[representative_index]),
            float(downside[representative_index]),
            float(swing[representative_index]),
            trend_dominance_ratio,
        )
        high_first_raw = high_first_series.iloc[representative_index]
        high_first = None if pd.isna(high_first_raw) else bool(high_first_raw)
        first_realization = int(realization[realization_index])
        union_start = int(entry[start])
        union_end = int(np.max(path_end[group_slice]))
        reference_count = int(len(np.unique(reference_ordinals[group_slice])))
        identifier = stable_event_id(
            str(positives["symbol"].iloc[0]) if len(positives) else "UNKNOWN",
            spec.segmentation_id,
            first_decision,
        )
        event_ids.append(identifier)

        event_rows.append(
            {
                "segmentation_schema_version": SEGMENTATION_SCHEMA_VERSION,
                "segmentation_profiler_version": PROFILER_VERSION,
                "segmentation_id": spec.segmentation_id,
                "rule_type": spec.rule_type,
                "bridge_gap_seconds": spec.bridge_gap_seconds,
                "event_ordinal": event_index,
                "event_id": identifier,
                "symbol": str(positives["symbol"].iloc[0]) if len(positives) else None,
                "event_start_second": first_decision,
                "event_start_timestamp_utc": pd.to_datetime(first_decision, unit="s", utc=True),
                "event_end_second": last_decision,
                "event_end_timestamp_utc": pd.to_datetime(last_decision, unit="s", utc=True),
                "event_start_date": date_from_second(first_decision),
                "event_end_date": date_from_second(last_decision),
                "crosses_utc_date": date_from_second(first_decision)
                != date_from_second(last_decision),
                "event_duration_seconds": event_span,
                "number_of_positive_anchors": group_count,
                "positive_anchor_coverage_pct": percent(group_count, event_span),
                "number_of_contiguous_positive_runs": positive_runs,
                "total_internal_inactive_seconds": total_internal_inactive,
                "maximum_internal_inactive_gap_seconds": maximum_inactive_gap,
                "mean_internal_inactive_gap_seconds": float(np.mean(internal_gaps))
                if len(internal_gaps)
                else 0.0,
                "median_internal_inactive_gap_seconds": float(np.median(internal_gaps))
                if len(internal_gaps)
                else 0.0,
                "future_interval_union_start_second": union_start,
                "future_interval_union_end_second": union_end,
                "future_interval_union_duration_seconds": union_end - union_start + 1,
                "first_50bps_realization_second": first_realization,
                "first_50bps_realization_timestamp_utc": pd.to_datetime(
                    first_realization, unit="s", utc=True
                ),
                "oracle_label_onset_to_realization_seconds": first_realization
                - first_decision,
                "first_executable_entry_to_realization_seconds": first_realization
                - int(entry[start]),
                "first_realization_anchor_decision_second": int(decision[realization_index]),
                "representative_decision_second": int(decision[representative_index]),
                "maximum_future_swing_15m_bps": float(swing[representative_index]),
                "maximum_future_upside_15m_bps": float(np.max(upside[group_slice])),
                "maximum_future_downside_15m_bps": float(np.max(downside[group_slice])),
                "representative_terminal_return_15m_bps": float(terminal[representative_index]),
                "representative_future_high_second": int(future_high[representative_index]),
                "representative_future_high_timestamp_utc": pd.to_datetime(
                    int(future_high[representative_index]), unit="s", utc=True
                ),
                "representative_future_low_second": int(future_low[representative_index]),
                "representative_future_low_timestamp_utc": pd.to_datetime(
                    int(future_low[representative_index]), unit="s", utc=True
                ),
                "representative_high_first": high_first,
                "trend_like": trend_like,
                "path_type": path_type,
                "dominant_excursion_share": dominant_share,
                "seconds_since_previous_event": None
                if previous_end is None
                else first_decision - previous_end - 1,
                "seconds_since_previous_event_future_union": None
                if previous_union_end is None
                else union_start - previous_union_end - 1,
                "reference_regime_count": reference_count,
            }
        )
        previous_end = last_decision
        previous_union_end = union_end

    events = pd.DataFrame(event_rows)
    if len(events):
        events["bridge_gap_seconds"] = events["bridge_gap_seconds"].astype("Int64")
        events["representative_high_first"] = events["representative_high_first"].astype(
            "boolean"
        )
        events["seconds_since_previous_event"] = events[
            "seconds_since_previous_event"
        ].astype("Int64")
        events["seconds_since_previous_event_future_union"] = events[
            "seconds_since_previous_event_future_union"
        ].astype("Int64")

    event_id_by_ordinal = np.asarray(event_ids, dtype=object)
    event_ids_per_anchor = event_id_by_ordinal[ordinals - 1] if len(ordinals) else np.array([], dtype=object)
    start_rank = np.repeat(starts, ends - starts)
    anchor_rank = np.arange(len(ordinals), dtype=np.int64) - start_rank + 1
    event_sizes = (ends - starts).astype(np.int64)
    event_size_per_anchor = event_sizes[ordinals - 1] if len(ordinals) else np.array([], dtype=np.int64)
    representative_mask = np.zeros(len(ordinals), dtype=bool)
    representative_mask[representative_indices] = True
    first_realization_mask = np.zeros(len(ordinals), dtype=bool)
    first_realization_mask[first_realization_indices] = True

    if reference_event_ids is None:
        reference_ids_per_anchor = event_ids_per_anchor
    else:
        reference_ids_per_anchor = np.asarray(reference_event_ids, dtype=object)[
            reference_ordinals - 1
        ]

    membership = pd.DataFrame(
        {
            "segmentation_schema_version": np.full(
                len(ordinals), SEGMENTATION_SCHEMA_VERSION, dtype=np.int16
            ),
            "segmentation_profiler_version": pd.Categorical(
                np.repeat(PROFILER_VERSION, len(ordinals))
            ),
            "segmentation_id": pd.Categorical(
                np.repeat(spec.segmentation_id, len(ordinals))
            ),
            "event_id": event_ids_per_anchor,
            "reference_regime_id": reference_ids_per_anchor,
            "symbol": pd.Categorical(positives["symbol"].astype(str)),
            "decision_second": decision,
            "event_ordinal": ordinals,
            "anchor_rank_within_event": anchor_rank,
            "is_first_positive_anchor": anchor_rank == 1,
            "is_last_positive_anchor": anchor_rank == event_size_per_anchor,
            "is_representative_max_swing_anchor": representative_mask,
            "is_first_realization_anchor": first_realization_mask,
        }
    )

    pair_frame = pd.DataFrame(
        {
            "event_ordinal": ordinals,
            "reference_ordinal": reference_ordinals,
        }
    )
    mapping = (
        pair_frame.groupby(["event_ordinal", "reference_ordinal"], sort=True)
        .size()
        .rename("shared_positive_anchors")
        .reset_index()
    )
    if len(mapping):
        candidate_sizes = pd.Series(event_sizes, index=np.arange(1, len(event_sizes) + 1))
        reference_sizes = pd.Series(
            np.bincount(reference_ordinals, minlength=int(reference_ordinals.max()) + 1)[1:],
            index=np.arange(1, int(reference_ordinals.max()) + 1),
        )
        mapping["segmentation_id"] = spec.segmentation_id
        mapping["event_id"] = mapping["event_ordinal"].map(
            pd.Series(event_ids, index=np.arange(1, len(event_ids) + 1))
        )
        reference_id_series = pd.Series(
            reference_event_ids if reference_event_ids is not None else event_ids,
            index=np.arange(1, int(reference_ordinals.max()) + 1),
        )
        mapping["reference_regime_id"] = mapping["reference_ordinal"].map(
            reference_id_series
        )
        mapping["candidate_event_positive_anchors"] = mapping["event_ordinal"].map(
            candidate_sizes
        )
        mapping["reference_regime_positive_anchors"] = mapping["reference_ordinal"].map(
            reference_sizes
        )
        mapping["candidate_anchor_share_pct"] = (
            mapping["shared_positive_anchors"]
            / mapping["candidate_event_positive_anchors"]
            * 100.0
        )
        mapping["reference_anchor_share_pct"] = (
            mapping["shared_positive_anchors"]
            / mapping["reference_regime_positive_anchors"]
            * 100.0
        )
        mapping = mapping[
            [
                "segmentation_id",
                "event_id",
                "event_ordinal",
                "reference_regime_id",
                "reference_ordinal",
                "shared_positive_anchors",
                "candidate_event_positive_anchors",
                "reference_regime_positive_anchors",
                "candidate_anchor_share_pct",
                "reference_anchor_share_pct",
            ]
        ]

    invariant_rows: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        invariant_rows.append(
            {
                "scope": "SEGMENTATION",
                "segmentation_id": spec.segmentation_id,
                "invariant": name,
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
            }
        )

    add(
        "ALL_POSITIVE_ANCHORS_ASSIGNED_ONCE",
        len(membership) == len(positives)
        and not membership["decision_second"].duplicated().any(),
        f"membership={len(membership):,}, positives={len(positives):,}",
    )
    add(
        "EVENT_POSITIVE_COUNTS_RECONCILE",
        int(events["number_of_positive_anchors"].sum()) == len(positives)
        if len(events)
        else len(positives) == 0,
        f"events_sum={int(events['number_of_positive_anchors'].sum()) if len(events) else 0:,}",
    )
    add(
        "EVENT_IDS_UNIQUE",
        not events["event_id"].duplicated().any() if len(events) else True,
        f"events={len(events):,}",
    )
    add(
        "EVENTS_STRICTLY_ORDERED",
        bool(
            len(events) < 2
            or np.all(
                events["event_start_second"].to_numpy(dtype=np.int64)[1:]
                > events["event_end_second"].to_numpy(dtype=np.int64)[:-1]
            )
        ),
        "successive decision-state event spans do not overlap",
    )
    add(
        "EVENT_MAX_SWING_AT_OR_ABOVE_THRESHOLD",
        bool(
            not len(events)
            or np.all(
                events["maximum_future_swing_15m_bps"].to_numpy(dtype=float) + 1e-10
                >= THRESHOLD_BPS
            )
        ),
        f"threshold_bps={THRESHOLD_BPS}",
    )
    add(
        "REFERENCE_MAPPING_RECONCILES",
        int(mapping["shared_positive_anchors"].sum()) == len(positives)
        if len(mapping)
        else len(positives) == 0,
        f"mapping_sum={int(mapping['shared_positive_anchors'].sum()) if len(mapping) else 0:,}",
    )

    if spec.rule_type == "MAX_INACTIVE_GAP" and len(decision) > 1:
        gaps = np.diff(decision) - 1
        inside = ordinals[1:] == ordinals[:-1]
        boundary = ~inside
        add(
            "INSIDE_EVENT_GAPS_RESPECT_BRIDGE_LIMIT",
            bool(not np.any(inside) or np.all(gaps[inside] <= int(spec.bridge_gap_seconds))),
            f"bridge_gap_seconds={spec.bridge_gap_seconds}",
        )
        add(
            "EVENT_BOUNDARIES_EXCEED_BRIDGE_LIMIT",
            bool(not np.any(boundary) or np.all(gaps[boundary] > int(spec.bridge_gap_seconds))),
            f"bridge_gap_seconds={spec.bridge_gap_seconds}",
        )
    elif spec.rule_type == "OVERLAPPING_FUTURE_INTERVALS" and len(entry) > 1:
        # Validate against its exact fixed-horizon equivalent independently.
        equivalent = bridge_group_ordinals(
            decision, OVERLAP_EQUIVALENT_INACTIVE_GAP_SECONDS
        )
        add(
            "OVERLAP_REFERENCE_EQUALS_BRIDGE_899S",
            bool(np.array_equal(ordinals, equivalent)),
            f"equivalent_inactive_gap_seconds={OVERLAP_EQUIVALENT_INACTIVE_GAP_SECONDS}",
        )

    failures = [row for row in invariant_rows if row["status"] == "FAIL"]
    if failures:
        raise RuntimeError(
            f"Segmentation validation failed for {spec.segmentation_id}: "
            + ", ".join(row["invariant"] for row in failures)
        )
    return events, membership, mapping, invariant_rows


# ---------------------------------------------------------------------------
# Profiling and reports
# ---------------------------------------------------------------------------


def event_metric_profile(events_by_segmentation: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    columns = {
        "event_duration_seconds": "event_duration_seconds",
        "positive_anchors_per_event": "number_of_positive_anchors",
        "positive_anchor_coverage_pct": "positive_anchor_coverage_pct",
        "maximum_internal_inactive_gap_seconds": "maximum_internal_inactive_gap_seconds",
        "maximum_future_swing_15m_bps": "maximum_future_swing_15m_bps",
        "oracle_label_onset_to_realization_seconds": "oracle_label_onset_to_realization_seconds",
        "seconds_between_events": "seconds_since_previous_event",
    }
    rows: list[dict[str, Any]] = []
    for segmentation_id, events in events_by_segmentation.items():
        for metric, column in columns.items():
            values = pd.to_numeric(events.get(column, pd.Series(dtype=float)), errors="coerce")
            values = values[np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))]
            for statistic in ("count", "mean", "median", "p75", "p90", "p95", "maximum"):
                if statistic == "count":
                    value = float(len(values))
                else:
                    value = safe_stat(values, statistic)
                rows.append(
                    {
                        "segmentation_id": segmentation_id,
                        "metric": metric,
                        "statistic": statistic,
                        "value": value,
                    }
                )
    return pd.DataFrame(rows)


def duration_distribution(events_by_segmentation: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    buckets = (
        ("under_5m", 0, 300),
        ("5m_to_under_15m", 300, 900),
        ("15m_to_under_30m", 900, 1800),
        ("30m_to_under_60m", 1800, 3600),
        ("1h_to_under_2h", 3600, 7200),
        ("2h_or_more", 7200, None),
    )
    rows: list[dict[str, Any]] = []
    for segmentation_id, events in events_by_segmentation.items():
        values = pd.to_numeric(
            events.get("event_duration_seconds", pd.Series(dtype=float)), errors="coerce"
        ).to_numpy(dtype=float)
        total = len(values)
        for order, (label, lower, upper) in enumerate(buckets, start=1):
            mask = values >= lower
            if upper is not None:
                mask &= values < upper
            count = int(np.sum(mask))
            rows.append(
                {
                    "segmentation_id": segmentation_id,
                    "bucket_order": order,
                    "duration_bucket": label,
                    "lower_seconds": lower,
                    "upper_seconds_exclusive": upper,
                    "event_count": count,
                    "event_pct": percent(count, total),
                }
            )
    return pd.DataFrame(rows)


def build_comparison(
    specs: Sequence[SegmentationSpec],
    events_by_segmentation: Mapping[str, pd.DataFrame],
    positive_rows: int,
    observed_days: int,
    positive_days: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        events = events_by_segmentation[spec.segmentation_id]
        count = len(events)
        high_first = int(events["representative_high_first"].fillna(False).astype(bool).sum())
        low_first = int(events["representative_high_first"].eq(False).sum())
        tie_count = count - high_first - low_first
        path_counts = events["path_type"].value_counts().to_dict()
        reference_counts = events["reference_regime_count"].astype(int)
        rows.append(
            {
                "segmentation_id": spec.segmentation_id,
                "rule_type": spec.rule_type,
                "bridge_gap_seconds": spec.bridge_gap_seconds,
                "description": spec.description,
                "positive_rows": positive_rows,
                "event_count": count,
                "events_per_observed_day": count / observed_days if observed_days else float("nan"),
                "events_per_positive_day": count / positive_days if positive_days else float("nan"),
                "positive_anchors_per_event_mean": positive_rows / count if count else float("nan"),
                "duration_mean_seconds": safe_stat(events["event_duration_seconds"], "mean"),
                "duration_median_seconds": safe_stat(events["event_duration_seconds"], "median"),
                "duration_p75_seconds": safe_stat(events["event_duration_seconds"], "p75"),
                "duration_p90_seconds": safe_stat(events["event_duration_seconds"], "p90"),
                "duration_p95_seconds": safe_stat(events["event_duration_seconds"], "p95"),
                "duration_maximum_seconds": safe_stat(events["event_duration_seconds"], "maximum"),
                "events_at_least_15m_pct": percent(
                    int((events["event_duration_seconds"] >= 900).sum()), count
                ),
                "events_at_least_30m_pct": percent(
                    int((events["event_duration_seconds"] >= 1800).sum()), count
                ),
                "events_at_least_60m_pct": percent(
                    int((events["event_duration_seconds"] >= 3600).sum()), count
                ),
                "events_at_least_2h_pct": percent(
                    int((events["event_duration_seconds"] >= 7200).sum()), count
                ),
                "coverage_median_pct": safe_stat(events["positive_anchor_coverage_pct"], "median"),
                "maximum_internal_gap_p95_seconds": safe_stat(
                    events["maximum_internal_inactive_gap_seconds"], "p95"
                ),
                "maximum_swing_median_bps": safe_stat(
                    events["maximum_future_swing_15m_bps"], "median"
                ),
                "maximum_swing_p90_bps": safe_stat(
                    events["maximum_future_swing_15m_bps"], "p90"
                ),
                "maximum_swing_p95_bps": safe_stat(
                    events["maximum_future_swing_15m_bps"], "p95"
                ),
                "high_first_count": high_first,
                "high_first_pct": percent(high_first, count),
                "low_first_count": low_first,
                "low_first_pct": percent(low_first, count),
                "same_second_extremes_count": tie_count,
                "trend_up_count": int(path_counts.get("TREND_UP", 0)),
                "trend_up_pct": percent(int(path_counts.get("TREND_UP", 0)), count),
                "trend_down_count": int(path_counts.get("TREND_DOWN", 0)),
                "trend_down_pct": percent(int(path_counts.get("TREND_DOWN", 0)), count),
                "two_sided_count": int(path_counts.get("TWO_SIDED", 0)),
                "two_sided_pct": percent(int(path_counts.get("TWO_SIDED", 0)), count),
                "inter_event_gap_median_seconds": safe_stat(
                    events["seconds_since_previous_event"], "median"
                ),
                "inter_event_gap_p90_seconds": safe_stat(
                    events["seconds_since_previous_event"], "p90"
                ),
                "cross_midnight_events": int(events["crosses_utc_date"].sum()),
                "candidate_events_spanning_multiple_reference_regimes": int(
                    (reference_counts > 1).sum()
                ),
                "candidate_events_spanning_multiple_reference_regimes_pct": percent(
                    int((reference_counts > 1).sum()), count
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame["bridge_gap_seconds"] = frame["bridge_gap_seconds"].astype("Int64")
    return frame


def build_transition_profile(
    bridge_specs: Sequence[SegmentationSpec],
    events_by_segmentation: Mapping[str, pd.DataFrame],
    ordinals_by_segmentation: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ordered = sorted(bridge_specs, key=lambda item: int(item.bridge_gap_seconds or 0))
    for lower, higher in zip(ordered, ordered[1:]):
        lower_events = events_by_segmentation[lower.segmentation_id]
        higher_events = events_by_segmentation[higher.segmentation_id]
        lower_ordinals = ordinals_by_segmentation[lower.segmentation_id]
        higher_ordinals = ordinals_by_segmentation[higher.segmentation_id]
        lower_boundaries = set(np.flatnonzero(lower_ordinals[1:] != lower_ordinals[:-1]) + 1)
        higher_boundaries = set(np.flatnonzero(higher_ordinals[1:] != higher_ordinals[:-1]) + 1)
        retained = len(lower_boundaries & higher_boundaries)
        rows.append(
            {
                "lower_segmentation_id": lower.segmentation_id,
                "higher_segmentation_id": higher.segmentation_id,
                "lower_bridge_gap_seconds": lower.bridge_gap_seconds,
                "higher_bridge_gap_seconds": higher.bridge_gap_seconds,
                "lower_event_count": len(lower_events),
                "higher_event_count": len(higher_events),
                "events_merged": len(lower_events) - len(higher_events),
                "event_count_drop_pct": percent(
                    len(lower_events) - len(higher_events), len(lower_events)
                ),
                "lower_internal_boundaries": len(lower_boundaries),
                "boundaries_retained_at_higher_threshold": retained,
                "boundary_retention_pct": percent(retained, len(lower_boundaries)),
                "lower_median_duration_seconds": safe_stat(
                    lower_events["event_duration_seconds"], "median"
                ),
                "higher_median_duration_seconds": safe_stat(
                    higher_events["event_duration_seconds"], "median"
                ),
                "lower_p95_duration_seconds": safe_stat(
                    lower_events["event_duration_seconds"], "p95"
                ),
                "higher_p95_duration_seconds": safe_stat(
                    higher_events["event_duration_seconds"], "p95"
                ),
            }
        )
    return pd.DataFrame(rows)


def reference_fragmentation_profile(
    specs: Sequence[SegmentationSpec],
    mappings: Mapping[str, pd.DataFrame],
    reference_event_count: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        mapping = mappings[spec.segmentation_id]
        per_reference = mapping.groupby("reference_regime_id")["event_id"].nunique()
        per_candidate = mapping.groupby("event_id")["reference_regime_id"].nunique()
        rows.append(
            {
                "segmentation_id": spec.segmentation_id,
                "reference_regime_count": reference_event_count,
                "candidate_event_count": int(mapping["event_id"].nunique()),
                "candidate_events_per_reference_mean": float(per_reference.mean()),
                "candidate_events_per_reference_median": float(per_reference.median()),
                "candidate_events_per_reference_p90": float(per_reference.quantile(0.90)),
                "candidate_events_per_reference_p95": float(per_reference.quantile(0.95)),
                "candidate_events_per_reference_maximum": int(per_reference.max()),
                "reference_regimes_split_into_2_or_more": int((per_reference >= 2).sum()),
                "reference_regimes_split_into_2_or_more_pct": percent(
                    int((per_reference >= 2).sum()), reference_event_count
                ),
                "reference_regimes_split_into_3_or_more": int((per_reference >= 3).sum()),
                "reference_regimes_split_into_5_or_more": int((per_reference >= 5).sum()),
                "candidate_events_spanning_2_or_more_reference_regimes": int(
                    (per_candidate >= 2).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def daily_profile(
    specs: Sequence[SegmentationSpec],
    events_by_segmentation: Mapping[str, pd.DataFrame],
    input_daily: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    input_lookup = input_daily.set_index("decision_date").to_dict("index")
    dates = list(input_daily["decision_date"])
    for spec in specs:
        events = events_by_segmentation[spec.segmentation_id]
        groups = {
            key: group
            for key, group in events.groupby("event_start_date", sort=True)
        }
        for day in dates:
            day_events = groups.get(day, pd.DataFrame())
            input_row = input_lookup[day]
            rows.append(
                {
                    "segmentation_id": spec.segmentation_id,
                    "decision_date": day,
                    "valid_label_rows": int(input_row["valid_label_rows"]),
                    "positive_rows": int(input_row["positive_rows"]),
                    "positive_share_pct": float(input_row["positive_share_pct"]),
                    "events_started": int(len(day_events)),
                    "median_event_duration_seconds": safe_stat(
                        day_events.get("event_duration_seconds", []), "median"
                    ),
                    "p90_event_duration_seconds": safe_stat(
                        day_events.get("event_duration_seconds", []), "p90"
                    ),
                    "median_maximum_swing_bps": safe_stat(
                        day_events.get("maximum_future_swing_15m_bps", []), "median"
                    ),
                    "trend_up_events": int(
                        (day_events.get("path_type", pd.Series(dtype=str)) == "TREND_UP").sum()
                    ),
                    "trend_down_events": int(
                        (day_events.get("path_type", pd.Series(dtype=str)) == "TREND_DOWN").sum()
                    ),
                    "two_sided_events": int(
                        (day_events.get("path_type", pd.Series(dtype=str)) == "TWO_SIDED").sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def top_longest_events(events_by_segmentation: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    columns = [
        "segmentation_id",
        "event_id",
        "event_start_timestamp_utc",
        "event_end_timestamp_utc",
        "event_duration_seconds",
        "number_of_positive_anchors",
        "positive_anchor_coverage_pct",
        "number_of_contiguous_positive_runs",
        "maximum_internal_inactive_gap_seconds",
        "maximum_future_swing_15m_bps",
        "path_type",
        "reference_regime_count",
    ]
    for events in events_by_segmentation.values():
        frame = events.nlargest(25, "event_duration_seconds")[columns].copy()
        frame.insert(1, "duration_rank_within_segmentation", np.arange(1, len(frame) + 1))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def global_nested_invariants(
    bridge_specs: Sequence[SegmentationSpec],
    ordinals_by_segmentation: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordered = sorted(bridge_specs, key=lambda item: int(item.bridge_gap_seconds or 0))
    for lower, higher in zip(ordered, ordered[1:]):
        lower_ordinals = ordinals_by_segmentation[lower.segmentation_id]
        higher_ordinals = ordinals_by_segmentation[higher.segmentation_id]
        # Every higher-threshold event must be a union of complete lower-threshold events.
        lower_boundary = lower_ordinals[1:] != lower_ordinals[:-1]
        higher_boundary = higher_ordinals[1:] != higher_ordinals[:-1]
        passed = bool(np.all(~higher_boundary | lower_boundary))
        rows.append(
            {
                "scope": "GLOBAL",
                "segmentation_id": f"{lower.segmentation_id}->{higher.segmentation_id}",
                "invariant": "BRIDGE_SEGMENTATIONS_ARE_NESTED",
                "status": "PASS" if passed else "FAIL",
                "detail": "higher-threshold boundaries must be a subset of lower-threshold boundaries",
            }
        )
    return rows


def markdown_findings(
    comparison: pd.DataFrame,
    transition: pd.DataFrame,
    fragmentation: pd.DataFrame,
    summary: Mapping[str, Any],
) -> str:
    lines = [
        "# Perps Expansion Event Segmentation Sensitivity",
        "",
        "## Purpose",
        "",
        "This profile compares deterministic ways of turning overlapping 50-bps / "
        "15-minute positive rolling labels into event instances. It does not choose a "
        "final event definition automatically and it does not train a model.",
        "",
        "## Input",
        "",
        f"- Positive rolling anchors: **{int(summary['positive_rows']):,}**",
        f"- Valid labelled rows: **{int(summary['valid_label_rows']):,}**",
        f"- Observed decision dates: **{int(summary['observed_dates']):,}**",
        f"- Failed invariants: **{int(summary['failed_invariants']):,}**",
        "",
        "## Candidate comparison",
        "",
        "| Segmentation | Events | Median duration | P95 duration | >=1h | Median max swing | Ref. regimes split |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    frag_lookup = fragmentation.set_index("segmentation_id")
    for _, row in comparison.iterrows():
        frag = frag_lookup.loc[row["segmentation_id"]]
        lines.append(
            "| "
            f"{row['segmentation_id']} | "
            f"{int(row['event_count']):,} | "
            f"{float(row['duration_median_seconds']):,.1f}s | "
            f"{float(row['duration_p95_seconds']):,.1f}s | "
            f"{float(row['events_at_least_60m_pct']):.2f}% | "
            f"{float(row['maximum_swing_median_bps']):.2f} bps | "
            f"{float(frag['reference_regimes_split_into_2_or_more_pct']):.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Adjacent-threshold sensitivity",
            "",
            "| Lower | Higher | Events merged | Count drop | Boundary retention |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in transition.iterrows():
        lines.append(
            "| "
            f"{row['lower_segmentation_id']} | {row['higher_segmentation_id']} | "
            f"{int(row['events_merged']):,} | "
            f"{float(row['event_count_drop_pct']):.2f}% | "
            f"{float(row['boundary_retention_pct']):.2f}% |"
        )

    reference = comparison[comparison["segmentation_id"] == "overlap_reference"].iloc[0]
    bridge900 = comparison[comparison["segmentation_id"] == "bridge_900s"]
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            f"- The exact overlap reference contains **{int(reference['event_count']):,}** broad regimes.",
            "- `oracle_label_onset_to_realization_seconds` is a future-label diagnostic, not model warning time.",
            "- Lower bridge thresholds split broad regimes into finer label-state instances; this does not prove that every split is economically independent.",
            "- Random train/test splitting remains invalid. Any later event evaluation must use chronological walk-forward splits and a horizon embargo.",
            "- No threshold is selected automatically. Review event-count stability, duration tails, reference fragmentation, and representative paths together.",
        ]
    )
    if len(bridge900):
        difference = int(bridge900.iloc[0]["event_count"] - reference["event_count"])
        lines.append(
            f"- `bridge_900s` differs from the exact overlap rule by **{difference:+,} events** because the exact fixed-horizon equivalence is 899 intervening seconds."
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output paths and manifest validation
# ---------------------------------------------------------------------------


def events_path(output_root: Path, symbol: str, segmentation_id: str) -> Path:
    return (
        output_root
        / "events"
        / f"{SEGMENTATION_PREFIX}{segmentation_id}"
        / f"{SYMBOL_PREFIX}{symbol}"
        / "part-00000.parquet"
    )


def membership_path(output_root: Path, symbol: str, segmentation_id: str) -> Path:
    return (
        output_root
        / "membership"
        / f"{SEGMENTATION_PREFIX}{segmentation_id}"
        / f"{SYMBOL_PREFIX}{symbol}"
        / "part-00000.parquet"
    )


def mapping_path(output_root: Path, symbol: str) -> Path:
    return output_root / "reference_mapping" / f"{SYMBOL_PREFIX}{symbol}" / "part-00000.parquet"


def validate_manifest(manifest: Sequence[Mapping[str, Any]], input_signature: str) -> bool:
    _, pq = require_pyarrow()
    try:
        for item in manifest:
            path = Path(str(item["path"]))
            if not path.exists():
                return False
            parquet_file = pq.ParquetFile(path)
            try:
                if parquet_file.metadata.num_rows != int(item["rows"]):
                    return False
                metadata = decode_metadata(parquet_file.schema_arrow)
                if metadata.get("input_signature") != input_signature:
                    return False
            finally:
                close = getattr(parquet_file, "close", None)
                if callable(close):
                    close()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def run_self_test() -> None:
    decision = np.array([10, 11, 12, 43, 44, 106, 1007, 1008], dtype=np.int64)
    contiguous = bridge_group_ordinals(decision, 0)
    expected_contiguous = np.array([1, 1, 1, 2, 2, 3, 4, 4], dtype=np.int64)
    if not np.array_equal(contiguous, expected_contiguous):
        raise AssertionError((contiguous, expected_contiguous))

    bridge30 = bridge_group_ordinals(decision, 30)
    expected_bridge30 = np.array([1, 1, 1, 1, 1, 2, 3, 3], dtype=np.int64)
    if not np.array_equal(bridge30, expected_bridge30):
        raise AssertionError((bridge30, expected_bridge30))

    # Exact overlap rule: p and q connect when q-p-1 <= 899.
    overlap_decision = np.array([0, 900, 1801, 2701], dtype=np.int64)
    entry = overlap_decision + 1
    path_end = entry + HORIZON_SECONDS
    overlap = overlap_reference_ordinals(entry, path_end)
    equivalent = bridge_group_ordinals(
        overlap_decision, OVERLAP_EQUIVALENT_INACTIVE_GAP_SECONDS
    )
    expected_overlap = np.array([1, 1, 2, 2], dtype=np.int64)
    if not np.array_equal(overlap, expected_overlap):
        raise AssertionError((overlap, expected_overlap))
    if not np.array_equal(overlap, equivalent):
        raise AssertionError("Overlap reference must equal bridge_899s")

    bridge900 = bridge_group_ordinals(overlap_decision, 900)
    expected_bridge900 = np.array([1, 1, 1, 1], dtype=np.int64)
    if not np.array_equal(bridge900, expected_bridge900):
        raise AssertionError((bridge900, expected_bridge900))

    starts, ends = group_bounds(expected_contiguous)
    if not np.array_equal(starts, np.array([0, 3, 5, 6])):
        raise AssertionError(starts)
    if not np.array_equal(ends, np.array([3, 5, 6, 8])):
        raise AssertionError(ends)

    trend = classify_path(60.0, 10.0, 70.0, 0.75)
    if trend[1] != "TREND_UP":
        raise AssertionError(trend)
    two_sided = classify_path(30.0, 25.0, 55.0, 0.75)
    if two_sided[1] != "TWO_SIDED":
        raise AssertionError(two_sided)

    print("SELF-TEST PASS")
    print("Validated strict-contiguous, bridge-gap, overlap-reference, off-by-one, bounds and path classification logic.")


# ---------------------------------------------------------------------------
# CLI and orchestration
# ---------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Profile deterministic segmentation sensitivity for positive 15-minute / "
            "50-bps perps expansion labels."
        )
    )
    value.add_argument(
        "--input-root",
        default="./data/derived/perps/research/expansion_episodes_15m_50bps_v1",
        help="Root containing upstream row_labels/ partitions.",
    )
    value.add_argument(
        "--output-root",
        default="./data/derived/perps/research/expansion_event_segmentation_sensitivity_v1",
        help="Separate event-sensitivity dataset root.",
    )
    value.add_argument(
        "--report-root",
        default="./reports/perps_expansion_event_segmentation_sensitivity",
        help="CSV, JSON and Markdown report root.",
    )
    value.add_argument("--symbol", default="BTCUSDT")
    value.add_argument(
        "--bridge-gaps",
        default=",".join(str(value) for value in DEFAULT_BRIDGE_GAPS),
        help="Comma-separated maximum intervening non-positive seconds.",
    )
    value.add_argument(
        "--trend-dominance-ratio",
        type=float,
        default=DEFAULT_TREND_DOMINANCE_RATIO,
        help="Minimum dominant-excursion share for TREND_UP or TREND_DOWN.",
    )
    value.add_argument(
        "--expected-label-builder-version",
        default=EXPECTED_LABEL_BUILDER_VERSION,
        help="Expected upstream label builder version; empty disables this check.",
    )
    value.add_argument("--start-date", help="Optional inclusive YYYY-MM-DD filter.")
    value.add_argument("--end-date", help="Optional inclusive YYYY-MM-DD filter.")
    value.add_argument(
        "--max-dates",
        type=int,
        help="Optional date limit after filtering; use only with isolated smoke outputs.",
    )
    value.add_argument(
        "--hash-input-files",
        action="store_true",
        help="Use full SHA-256 file hashes in change detection.",
    )
    value.add_argument(
        "--force",
        action="store_true",
        help="Rebuild outputs even when input and configuration signatures are unchanged.",
    )
    value.add_argument(
        "--compression-level",
        type=int,
        default=6,
        help="Zstandard Parquet compression level from 1 through 22.",
    )
    value.add_argument(
        "--self-test",
        action="store_true",
        help="Run deterministic in-memory tests without reading Parquet files.",
    )
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    if not 0.5 <= args.trend_dominance_ratio <= 1.0:
        raise SystemExit("--trend-dominance-ratio must be between 0.5 and 1.0")
    if not 1 <= args.compression_level <= 22:
        raise SystemExit("--compression-level must be between 1 and 22")

    bridge_gaps = parse_bridge_gaps(args.bridge_gaps)
    # Keep the requested 900-second sensitivity case available in the standard
    # report. Custom runs may intentionally omit it.
    bridge_specs = [bridge_spec(gap) for gap in bridge_gaps]
    specs = [*bridge_specs, reference_spec()]

    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    report_root = Path(args.report_root).expanduser().resolve()
    symbol = str(args.symbol).strip().upper()
    start_date = parse_date(args.start_date, "--start-date")
    end_date = parse_date(args.end_date, "--end-date")
    if start_date and end_date and start_date > end_date:
        raise SystemExit("--start-date cannot be after --end-date")

    run_id = utc_now().strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "_build_state" / "build_state.sqlite"

    config = {
        "profiler_version": PROFILER_VERSION,
        "segmentation_schema_version": SEGMENTATION_SCHEMA_VERSION,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "report_root": str(report_root),
        "symbol": symbol,
        "horizon_seconds": HORIZON_SECONDS,
        "path_observations": PATH_OBSERVATIONS,
        "threshold_bps": THRESHOLD_BPS,
        "bridge_gaps_seconds": list(bridge_gaps),
        "overlap_equivalent_inactive_gap_seconds": OVERLAP_EQUIVALENT_INACTIVE_GAP_SECONDS,
        "trend_dominance_ratio": float(args.trend_dominance_ratio),
        "expected_label_builder_version": args.expected_label_builder_version,
        "expected_label_schema_version": EXPECTED_LABEL_SCHEMA_VERSION,
        "hash_input_files": bool(args.hash_input_files),
        "start_date": start_date.isoformat() if start_date else None,
        "end_date": end_date.isoformat() if end_date else None,
        "max_dates": args.max_dates,
    }
    config_signature = sha256_text(
        compact_json(
            {
                key: value
                for key, value in config.items()
                if key not in {"report_root"}
            }
        )
    )

    connection = connect_state(state_path)
    connection.execute(
        """
        INSERT INTO runs (
            run_id, started_at_utc, status, symbol, config_signature,
            configuration_json
        ) VALUES (?, ?, 'RUNNING', ?, ?, ?)
        """,
        (run_id, utc_now_iso(), symbol, config_signature, json.dumps(config, sort_keys=True)),
    )
    connection.commit()

    try:
        print("Discovering expansion-label partitions...")
        partitions = discover_label_partitions(
            input_root=input_root,
            symbol=symbol,
            expected_builder_version=args.expected_label_builder_version,
            hash_input_files=bool(args.hash_input_files),
            start_date=start_date,
            end_date=end_date,
            max_dates=args.max_dates,
        )
        input_signature = combined_input_signature(partitions)
        record_partitions(connection, symbol, run_id, partitions)
        connection.execute(
            "UPDATE runs SET input_signature=? WHERE run_id=?",
            (input_signature, run_id),
        )
        connection.commit()

        print(f"Input dates       : {len(partitions):,}")
        print(f"Date range        : {min(partitions)} through {max(partitions)}")
        print(f"Valid label rows  : {sum(item.rows for item in partitions.values()):,}")
        print(f"Segmentations     : {', '.join(spec.segmentation_id for spec in specs)}")
        print(f"Output root       : {output_root}")
        print(f"State database    : {state_path}")

        latest = connection.execute(
            "SELECT * FROM latest_build WHERE symbol=?", (symbol,)
        ).fetchone()
        if (
            not args.force
            and latest is not None
            and latest["status"] == "COMPLETED"
            and latest["input_signature"] == input_signature
            and latest["config_signature"] == config_signature
            and latest["output_manifest_json"]
        ):
            manifest = json.loads(latest["output_manifest_json"])
            reports_exist = all((report_root / name).exists() for name in REPORT_FILENAMES)
            if reports_exist and validate_manifest(manifest, input_signature):
                connection.execute(
                    """
                    UPDATE runs SET completed_at_utc=?, status='SKIPPED_UNCHANGED',
                        positive_rows=?, segmentation_count=?, summary_json=?
                    WHERE run_id=?
                    """,
                    (
                        utc_now_iso(),
                        int(latest["positive_rows"] or 0),
                        int(latest["segmentation_count"] or 0),
                        json.dumps(
                            {
                                "status": "SKIPPED_UNCHANGED",
                                "input_signature": input_signature,
                                "config_signature": config_signature,
                            },
                            sort_keys=True,
                        ),
                        run_id,
                    ),
                )
                connection.commit()
                print("SKIP: inputs, configuration, Parquet outputs and reports are unchanged.")
                return 0

        connection.execute(
            """
            INSERT INTO latest_build (
                symbol, status, input_signature, config_signature, error_text
            ) VALUES (?, 'RUNNING', ?, ?, NULL)
            ON CONFLICT(symbol) DO UPDATE SET
                status='RUNNING',
                input_signature=excluded.input_signature,
                config_signature=excluded.config_signature,
                error_text=NULL
            """,
            (symbol, input_signature, config_signature),
        )
        connection.commit()

        print("Loading positive rolling anchors...")
        positives, input_daily, input_invariants = load_positive_anchors(partitions, symbol)
        if not len(positives):
            raise RuntimeError("No positive 50-bps expansion anchors were found")
        print(f"Positive anchors  : {len(positives):,}")

        decision = positives["decision_second"].to_numpy(dtype=np.int64)
        entry = positives["entry_second"].to_numpy(dtype=np.int64)
        path_end = positives["path_end_second"].to_numpy(dtype=np.int64)
        reference_ordinals = overlap_reference_ordinals(entry, path_end)
        reference = reference_spec()

        # Build the exact overlap reference first so all candidate memberships can
        # carry stable reference-regime IDs and fragmentation can be measured.
        reference_events, reference_membership, reference_mapping, reference_invariants = (
            build_events_and_membership(
                positives=positives,
                spec=reference,
                ordinals=reference_ordinals,
                reference_ordinals=reference_ordinals,
                reference_event_ids=None,
                trend_dominance_ratio=float(args.trend_dominance_ratio),
            )
        )
        reference_event_ids = list(reference_events["event_id"].astype(str))

        events_by_segmentation: dict[str, pd.DataFrame] = {
            reference.segmentation_id: reference_events
        }
        mappings: dict[str, pd.DataFrame] = {
            reference.segmentation_id: reference_mapping
        }
        ordinals_by_segmentation: dict[str, np.ndarray] = {
            reference.segmentation_id: reference_ordinals
        }
        all_invariants = [*input_invariants, *reference_invariants]
        output_manifest: list[dict[str, Any]] = []
        mapping_frames: list[pd.DataFrame] = [reference_mapping]

        common_metadata = {
            "segmentation_profiler_version": PROFILER_VERSION,
            "segmentation_schema_version": SEGMENTATION_SCHEMA_VERSION,
            "source_label_builder_version": args.expected_label_builder_version,
            "source_label_schema_version": EXPECTED_LABEL_SCHEMA_VERSION,
            "symbol": symbol,
            "horizon_seconds": HORIZON_SECONDS,
            "threshold_bps": THRESHOLD_BPS,
            "trend_dominance_ratio": args.trend_dominance_ratio,
            "input_signature": input_signature,
            "config_signature": config_signature,
            "created_at_utc": utc_now_iso(),
        }

        def write_segmentation_outputs(
            spec: SegmentationSpec,
            events: pd.DataFrame,
            membership: pd.DataFrame,
        ) -> None:
            event_file = events_path(output_root, symbol, spec.segmentation_id)
            membership_file = membership_path(output_root, symbol, spec.segmentation_id)
            metadata = {
                **common_metadata,
                "segmentation_id": spec.segmentation_id,
                "rule_type": spec.rule_type,
                "bridge_gap_seconds": spec.bridge_gap_seconds,
            }
            atomic_write_parquet(
                event_file,
                events,
                {**metadata, "dataset_role": "candidate_events"},
                args.compression_level,
            )
            atomic_write_parquet(
                membership_file,
                membership,
                {**metadata, "dataset_role": "positive_anchor_event_membership"},
                args.compression_level,
            )
            cleanup_extra_parquet_files(event_file.parent, event_file)
            cleanup_extra_parquet_files(membership_file.parent, membership_file)
            output_manifest.extend(
                [
                    {"path": str(event_file), "rows": len(events)},
                    {"path": str(membership_file), "rows": len(membership)},
                ]
            )

        print(
            f"PROFILE   {reference.segmentation_id}: "
            f"events={len(reference_events):,}"
        )
        write_segmentation_outputs(reference, reference_events, reference_membership)

        for spec in bridge_specs:
            ordinals = bridge_group_ordinals(decision, int(spec.bridge_gap_seconds))
            events, membership, mapping, invariants = build_events_and_membership(
                positives=positives,
                spec=spec,
                ordinals=ordinals,
                reference_ordinals=reference_ordinals,
                reference_event_ids=reference_event_ids,
                trend_dominance_ratio=float(args.trend_dominance_ratio),
            )
            events_by_segmentation[spec.segmentation_id] = events
            mappings[spec.segmentation_id] = mapping
            ordinals_by_segmentation[spec.segmentation_id] = ordinals
            mapping_frames.append(mapping)
            all_invariants.extend(invariants)
            print(
                f"PROFILE   {spec.segmentation_id}: "
                f"events={len(events):,}, "
                f"median_duration={safe_stat(events['event_duration_seconds'], 'median'):,.1f}s, "
                f"p95={safe_stat(events['event_duration_seconds'], 'p95'):,.1f}s"
            )
            write_segmentation_outputs(spec, events, membership)
            del membership

        all_invariants.extend(global_nested_invariants(bridge_specs, ordinals_by_segmentation))
        invariant_frame = pd.DataFrame(all_invariants)
        failed_invariants = int((invariant_frame["status"] == "FAIL").sum())
        if failed_invariants:
            failures = invariant_frame[invariant_frame["status"] == "FAIL"]
            raise RuntimeError(
                "Global segmentation invariants failed: "
                + ", ".join(failures["invariant"].astype(str).unique())
            )

        all_mapping = pd.concat(mapping_frames, ignore_index=True)
        map_file = mapping_path(output_root, symbol)
        atomic_write_parquet(
            map_file,
            all_mapping,
            {**common_metadata, "dataset_role": "candidate_to_overlap_reference_mapping"},
            args.compression_level,
        )
        cleanup_extra_parquet_files(map_file.parent, map_file)
        output_manifest.append({"path": str(map_file), "rows": len(all_mapping)})

        ordered_specs = [*bridge_specs, reference]
        observed_days = len(input_daily)
        positive_days = int((input_daily["positive_rows"] > 0).sum())
        comparison = build_comparison(
            ordered_specs,
            events_by_segmentation,
            len(positives),
            observed_days,
            positive_days,
        )
        transition = build_transition_profile(
            bridge_specs, events_by_segmentation, ordinals_by_segmentation
        )
        metric = event_metric_profile(events_by_segmentation)
        durations = duration_distribution(events_by_segmentation)
        daily = daily_profile(ordered_specs, events_by_segmentation, input_daily)
        fragmentation = reference_fragmentation_profile(
            ordered_specs, mappings, len(reference_events)
        )
        longest = top_longest_events(events_by_segmentation)

        summary = {
            "status": "COMPLETED",
            "run_id": run_id,
            "profiler_version": PROFILER_VERSION,
            "created_at_utc": utc_now_iso(),
            "symbol": symbol,
            "input_root": str(input_root),
            "output_root": str(output_root),
            "report_root": str(report_root),
            "input_signature": input_signature,
            "config_signature": config_signature,
            "observed_dates": observed_days,
            "first_date": min(partitions).isoformat(),
            "last_date": max(partitions).isoformat(),
            "valid_label_rows": int(input_daily["valid_label_rows"].sum()),
            "positive_rows": len(positives),
            "positive_share_pct": percent(
                len(positives), int(input_daily["valid_label_rows"].sum())
            ),
            "segmentation_count": len(ordered_specs),
            "overlap_reference_events": len(reference_events),
            "candidate_event_counts": {
                segmentation_id: len(events)
                for segmentation_id, events in events_by_segmentation.items()
            },
            "failed_invariants": failed_invariants,
            "output_manifest": output_manifest,
        }
        findings = markdown_findings(comparison, transition, fragmentation, summary)

        run_report_root = report_root / "runs" / run_id
        report_frames = {
            "perps_expansion_event_segmentation_comparison.csv": comparison,
            "perps_expansion_event_segmentation_transition.csv": transition,
            "perps_expansion_event_segmentation_metric_profile.csv": metric,
            "perps_expansion_event_segmentation_duration_distribution.csv": durations,
            "perps_expansion_event_segmentation_daily.csv": daily,
            "perps_expansion_event_segmentation_reference_fragmentation.csv": fragmentation,
            "perps_expansion_event_segmentation_top_longest_events.csv": longest,
            "perps_expansion_event_segmentation_invariants.csv": invariant_frame,
        }
        for filename, frame in report_frames.items():
            atomic_write_csv(report_root / filename, frame)
            atomic_write_csv(run_report_root / filename, frame)
        for root in (report_root, run_report_root):
            atomic_write_json(
                root / "perps_expansion_event_segmentation_summary.json", summary
            )
            atomic_write_json(
                root / "perps_expansion_event_segmentation_config.json", config
            )
            atomic_write_text(
                root / "perps_expansion_event_segmentation_findings.md", findings
            )

        atomic_write_json(output_root / "_metadata" / "dataset_config.json", config)
        atomic_write_json(output_root / "_metadata" / "latest_run_summary.json", summary)

        connection.execute(
            """
            UPDATE latest_build SET
                status='COMPLETED', completed_at_utc=?, positive_rows=?,
                segmentation_count=?, output_manifest_json=?, error_text=NULL
            WHERE symbol=?
            """,
            (
                utc_now_iso(),
                len(positives),
                len(ordered_specs),
                json.dumps(output_manifest, sort_keys=True),
                symbol,
            ),
        )
        connection.execute(
            """
            UPDATE runs SET completed_at_utc=?, status='COMPLETED',
                positive_rows=?, segmentation_count=?, summary_json=?
            WHERE run_id=?
            """,
            (
                utc_now_iso(),
                len(positives),
                len(ordered_specs),
                json.dumps(summary, sort_keys=True),
                run_id,
            ),
        )
        connection.commit()

        print("COMPLETED")
        print(f"Positive anchors       : {len(positives):,}")
        print(f"Overlap regimes        : {len(reference_events):,}")
        for spec in bridge_specs:
            print(
                f"{spec.segmentation_id:<23}: "
                f"{len(events_by_segmentation[spec.segmentation_id]):,} events"
            )
        print(f"Failed invariants      : {failed_invariants:,}")
        print(f"Reports                : {report_root}")
        return 0

    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        connection.execute(
            """
            UPDATE runs SET completed_at_utc=?, status='FAILED', error_text=?
            WHERE run_id=?
            """,
            (utc_now_iso(), error_text, run_id),
        )
        connection.execute(
            """
            INSERT INTO latest_build (
                symbol, status, input_signature, config_signature, error_text
            ) VALUES (?, 'FAILED', '', ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                status='FAILED', error_text=excluded.error_text
            """,
            (symbol, config_signature, error_text),
        )
        connection.commit()
        return 1
    finally:
        connection.close()


if __name__ == "__main__":
    sys.exit(main())
