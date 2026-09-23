#!/usr/bin/env python
"""
Profile expansion-target and event-segmentation sensitivity.

Research purpose
----------------
The current baseline target is a 50-bps high-low expansion within an exact
15-minute future path, with practical events grouped using a 60-second bridge
between positive rolling labels. That baseline is useful, but neither the
50-bps threshold, the 15-minute horizon, nor the 60-second grouping gap should
be assumed to be universally optimal.

This script jointly profiles three dimensions before any model is trained:

1. future expansion threshold in basis points;
2. exact forward horizon;
3. deterministic event grouping gap.

Causal label contract
---------------------
For every context-ready decision row at exact second ``t``:

* entry price is the exact price at ``t + 1``;
* an H-second future path contains every exact second from ``t + 1`` through
  ``t + 1 + H`` inclusive;
* a path therefore contains ``H + 1`` observations spanning exactly H elapsed
  seconds;
* future swing is ``(future_high - future_low) / entry_price * 10,000``;
* rows are retained only when the complete exact path exists with finite,
  positive prices;
* missing seconds are never filled, interpolated or carried;
* all outputs are research labels and are never written into causal features.

Event grouping contract
-----------------------
For consecutive positive decision seconds ``p`` and ``q``:

    inactive_gap_seconds = q - p - 1

A bridge segmentation keeps them in the same event when:

    inactive_gap_seconds <= bridge_gap_seconds

The script also constructs an independent ``overlap_reference`` segmentation.
Every positive anchor owns the closed future interval ``[t + 1, t + 1 + H]``.
Anchors belong to the same broad regime while those intervals remain connected.
For a fixed H, that reference is equivalent to bridging at most ``H - 1`` fully
inactive seconds, but the implementation is independent and validated.

Default target grid
-------------------
    5 minutes:  35, 50, 75 bps
    10 minutes: 35, 50, 75 bps
    15 minutes: 35, 50, 75, 100 bps
    20 minutes: 50, 75, 100 bps
    30 minutes: 50, 75, 100 bps

Default bridge gaps
-------------------
    0, 15, 30, 60, 120 and 300 seconds, plus overlap_reference

Resumability
------------
Exact horizon-label partitions are stored separately by horizon and UTC decision
calendar date. A date depends on its own audited feature partition and the next
UTC partition, when present. Therefore a changed input date invalidates its own
labels and the prior day's cross-midnight labels through dependency signatures.
Partition writes are atomic and tracked in SQLite. Global target/event profiles
are deterministically rematerialized from the completed horizon labels.

Outputs
-------
Default dataset root:
    data/derived/perps/research/expansion_target_segmentation_sensitivity_v1

Default report root:
    reports/perps_expansion_target_segmentation_sensitivity

Dependencies
------------
    pip install numpy pandas pyarrow numba

Typical Git Bash usage
----------------------
    python scripts/perps/research/04_profile_expansion_target_and_segmentation_sensitivity.py
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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:  # pragma: no cover - pure-Python fallback remains correct
    def njit(*args: Any, **kwargs: Any):  # type: ignore[misc]
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(function: Any) -> Any:
            return function

        return decorator


PROFILER_VERSION = "1.0.0"
LABEL_SCHEMA_VERSION = 1
EVENT_SCHEMA_VERSION = 1
EXPECTED_FEATURE_SCHEMA_VERSION = 1
EXPECTED_FEATURE_BUILDER_VERSION = "1.0.2"
SECONDS_PER_DAY = 86_400
DATE_PREFIX = "date="
SYMBOL_PREFIX = "symbol="
HORIZON_PREFIX = "horizon="
TARGET_PREFIX = "target="
SEGMENTATION_PREFIX = "segmentation="
DEFAULT_TARGET_GRID = (
    "5=35|50|75;"
    "10=35|50|75;"
    "15=35|50|75|100;"
    "20=50|75|100;"
    "30=50|75|100"
)
DEFAULT_BRIDGE_GAPS = (0, 15, 30, 60, 120, 300)
DEFAULT_TREND_DOMINANCE_RATIO = 0.75
DEFAULT_MIN_VIABLE_EVENTS = 300
DEFAULT_MIN_VIABLE_ISOLATED_EVENTS = 150
DEFAULT_MAX_POSITIVE_RATE_PCT = 40.0
DEFAULT_MIN_POSITIVE_RATE_PCT = 1.0

REQUIRED_FEATURE_COLUMNS = (
    "feature_schema_version",
    "symbol",
    "receive_second",
    "price_for_features",
    "flow_feature_context_ready",
)

LABEL_COLUMNS = (
    "symbol",
    "decision_date",
    "horizon_seconds",
    "decision_second",
    "entry_second",
    "path_end_second",
    "entry_price",
    "future_high_price",
    "future_low_price",
    "future_terminal_price",
    "future_high_second",
    "future_low_second",
    "future_swing_bps",
    "future_upside_bps",
    "future_downside_bps",
    "future_terminal_return_bps",
    "future_high_first",
)

EVENT_COLUMNS = (
    "event_schema_version",
    "profiler_version",
    "event_id",
    "target_id",
    "symbol",
    "horizon_seconds",
    "horizon_minutes",
    "threshold_bps",
    "segmentation_id",
    "segmentation_rule_type",
    "bridge_gap_seconds",
    "event_ordinal",
    "event_start_second",
    "event_end_second",
    "event_start_timestamp_utc",
    "event_end_timestamp_utc",
    "event_start_date",
    "event_end_date",
    "crosses_utc_date",
    "event_duration_seconds",
    "number_of_positive_anchors",
    "positive_anchor_coverage_pct",
    "number_of_contiguous_positive_runs",
    "total_internal_inactive_seconds",
    "maximum_internal_inactive_gap_seconds",
    "future_interval_union_start_second",
    "future_interval_union_end_second",
    "future_interval_union_duration_seconds",
    "maximum_future_swing_bps",
    "maximum_future_upside_bps",
    "maximum_future_downside_bps",
    "representative_terminal_return_bps",
    "representative_future_high_second",
    "representative_future_low_second",
    "representative_high_first",
    "dominant_excursion_share",
    "trend_like",
    "path_type",
    "seconds_since_previous_event",
    "isolated_onset",
)

REPORT_FILENAMES = (
    "perps_expansion_target_prevalence.csv",
    "perps_expansion_target_segmentation_comparison.csv",
    "perps_expansion_target_segmentation_transition.csv",
    "perps_expansion_target_stable_ranges.csv",
    "perps_expansion_target_viability.csv",
    "perps_expansion_target_duration_distribution.csv",
    "perps_expansion_target_daily.csv",
    "perps_expansion_target_overlap.csv",
    "perps_expansion_target_top_longest_events.csv",
    "perps_expansion_target_invariants.csv",
    "perps_expansion_target_summary.json",
    "perps_expansion_target_config.json",
    "perps_expansion_target_findings.md",
)

_PA: Any | None = None
_PQ: Any | None = None


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------


def require_arrow() -> tuple[Any, Any]:
    global _PA, _PQ
    if _PA is None or _PQ is None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - environment check
            raise SystemExit(
                "Missing dependency 'pyarrow'. Activate the project virtual "
                "environment and run: pip install numpy pandas pyarrow numba"
            ) from exc
        _PA, _PQ = pa, pq
    return _PA, _PQ


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat().replace("+00:00", "Z")


def epoch_second_for_date(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())


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


def percentage(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return float("nan")
    return 100.0 * float(numerator) / float(denominator)


def finite_series(values: Iterable[Any]) -> pd.Series:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce")
    array = series.to_numpy(dtype=float, na_value=np.nan)
    return series[np.isfinite(array)]


def safe_stat(values: Iterable[Any], name: str) -> float:
    series = finite_series(values)
    if series.empty:
        return float("nan")
    if name == "mean":
        return float(series.mean())
    if name == "median":
        return float(series.median())
    if name == "minimum":
        return float(series.min())
    if name == "maximum":
        return float(series.max())
    if name.startswith("p"):
        return float(series.quantile(float(name[1:]) / 100.0))
    raise ValueError(f"Unsupported statistic: {name}")


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
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def metadata_bytes(values: Mapping[str, Any]) -> dict[bytes, bytes]:
    return {
        str(key).encode("utf-8"): str(value).encode("utf-8")
        for key, value in values.items()
    }


def decode_metadata(schema: Any) -> dict[str, str]:
    return {
        key.decode("utf-8", errors="replace"): value.decode(
            "utf-8", errors="replace"
        )
        for key, value in (schema.metadata or {}).items()
    }


def atomic_write_parquet_frame(
    path: Path,
    frame: pd.DataFrame,
    metadata: Mapping[str, Any],
    compression_level: int,
) -> None:
    pa, pq = require_arrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        merged = dict(table.schema.metadata or {})
        merged.update(metadata_bytes(metadata))
        table = table.replace_schema_metadata(merged)
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            compression_level=compression_level,
            use_dictionary=True,
            write_statistics=True,
        )
        check = pq.ParquetFile(temporary)
        try:
            if int(check.metadata.num_rows) != len(frame):
                raise RuntimeError(
                    f"Parquet validation failed for {path}: expected {len(frame):,} "
                    f"rows, wrote {check.metadata.num_rows:,}"
                )
        finally:
            close = getattr(check, "close", None)
            if callable(close):
                close()
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def cleanup_extra_parquet_files(directory: Path, keep: Path) -> None:
    for candidate in directory.glob("*.parquet"):
        if candidate.resolve() != keep.resolve():
            candidate.unlink()


def numeric_token(value: float) -> str:
    rounded = round(float(value), 8)
    if math.isclose(rounded, round(rounded), abs_tol=1e-8):
        return f"{int(round(rounded)):03d}"
    text = f"{rounded:.8f}".rstrip("0").rstrip(".")
    integer, decimal = text.split(".")
    return f"{int(integer):03d}P{decimal}"


def target_id(horizon_minutes: int, threshold_bps: float) -> str:
    return f"H{horizon_minutes:02d}M_T{numeric_token(threshold_bps)}BPS"


def horizon_id(horizon_seconds: int) -> str:
    minutes = horizon_seconds // 60
    return f"{minutes:02d}m"


def segmentation_id_for_gap(gap: int) -> str:
    return f"bridge_{gap:03d}s"


def parse_target_grid(value: str) -> dict[int, tuple[float, ...]]:
    output: dict[int, tuple[float, ...]] = {}
    for raw_group in value.split(";"):
        group = raw_group.strip()
        if not group:
            continue
        if "=" not in group:
            raise SystemExit(
                "--target-grid groups must use MINUTES=THRESHOLD|THRESHOLD syntax"
            )
        minute_text, threshold_text = group.split("=", 1)
        try:
            minutes = int(minute_text.strip())
        except ValueError as exc:
            raise SystemExit(f"Invalid horizon minutes: {minute_text}") from exc
        if minutes <= 0 or minutes > 24 * 60:
            raise SystemExit("Target horizons must be between 1 and 1,440 minutes")
        thresholds: set[float] = set()
        for part in threshold_text.split("|"):
            if not part.strip():
                continue
            try:
                threshold = float(part.strip())
            except ValueError as exc:
                raise SystemExit(f"Invalid threshold: {part}") from exc
            if not math.isfinite(threshold) or threshold <= 0 or threshold > 10_000:
                raise SystemExit("Target thresholds must be finite and between 0 and 10,000 bps")
            thresholds.add(threshold)
        if not thresholds:
            raise SystemExit(f"No thresholds supplied for {minutes} minutes")
        if minutes in output:
            thresholds.update(output[minutes])
        output[minutes] = tuple(sorted(thresholds))
    if not output:
        raise SystemExit("--target-grid produced no targets")
    return dict(sorted(output.items()))


def parse_bridge_gaps(value: str) -> tuple[int, ...]:
    try:
        gaps = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise SystemExit("--bridge-gaps must be comma-separated integers") from exc
    if not gaps:
        raise SystemExit("--bridge-gaps must include at least one gap")
    if gaps[0] < 0 or gaps[-1] > SECONDS_PER_DAY:
        raise SystemExit("Bridge gaps must be between 0 and 86,400 seconds")
    return tuple(gaps)


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InputPartition:
    receive_date: date
    path: Path
    rows: int
    minimum_second: int
    maximum_second: int
    feature_builder_version: str | None
    feature_schema_version: int | None
    size_bytes: int
    modified_time_ns: int
    content_sha256: str | None
    signature: str


def parquet_min_max_int(parquet_file: Any, column_name: str) -> tuple[int, int]:
    index = parquet_file.schema_arrow.get_field_index(column_name)
    if index < 0:
        raise RuntimeError(f"Required column is missing: {column_name}")
    minima: list[int] = []
    maxima: list[int] = []
    for row_group_index in range(parquet_file.num_row_groups):
        column = parquet_file.metadata.row_group(row_group_index).column(index)
        statistics = column.statistics
        if statistics is None or not statistics.has_min_max:
            minima = []
            maxima = []
            break
        minima.append(int(statistics.min))
        maxima.append(int(statistics.max))
    if minima:
        return min(minima), max(maxima)
    values = parquet_file.read(columns=[column_name]).column(column_name).to_numpy(
        zero_copy_only=False
    )
    if not len(values):
        raise RuntimeError(f"No values found in {column_name}")
    return int(np.min(values)), int(np.max(values))


def discover_feature_partitions(
    feature_root: Path,
    symbol: str,
    expected_builder_version: str,
    hash_input_files: bool,
) -> dict[date, InputPartition]:
    _, pq = require_arrow()
    symbol_root = feature_root / f"{SYMBOL_PREFIX}{symbol}"
    if not symbol_root.exists():
        raise SystemExit(f"Feature symbol directory does not exist: {symbol_root}")

    output: dict[date, InputPartition] = {}
    for date_directory in sorted(symbol_root.glob(f"{DATE_PREFIX}*")):
        if not date_directory.is_dir():
            continue
        try:
            receive_date = date.fromisoformat(date_directory.name[len(DATE_PREFIX) :])
        except ValueError:
            continue
        files = sorted(path for path in date_directory.glob("*.parquet") if path.is_file())
        if len(files) != 1:
            raise SystemExit(
                f"Expected exactly one feature Parquet file for {receive_date}, "
                f"found {len(files)} in {date_directory}"
            )
        path = files[0]
        parquet_file = pq.ParquetFile(path)
        try:
            missing = [
                column
                for column in REQUIRED_FEATURE_COLUMNS
                if column not in parquet_file.schema_arrow.names
            ]
            if missing:
                raise SystemExit(f"{path} is missing required columns: {missing}")
            metadata = decode_metadata(parquet_file.schema_arrow)
            builder_version = metadata.get("perps_flow_feature_builder_version")
            raw_schema_version = metadata.get("perps_flow_feature_schema_version")
            schema_version = int(raw_schema_version) if raw_schema_version else None
            if expected_builder_version and builder_version != expected_builder_version:
                raise SystemExit(
                    f"{path} feature builder version is {builder_version!r}; "
                    f"expected {expected_builder_version!r}"
                )
            if schema_version != EXPECTED_FEATURE_SCHEMA_VERSION:
                raise SystemExit(
                    f"{path} feature schema version is {schema_version!r}; "
                    f"expected {EXPECTED_FEATURE_SCHEMA_VERSION}"
                )
            minimum_second, maximum_second = parquet_min_max_int(
                parquet_file, "receive_second"
            )
            rows = int(parquet_file.metadata.num_rows)
        finally:
            close = getattr(parquet_file, "close", None)
            if callable(close):
                close()

        stat = path.stat()
        content_hash = sha256_file(path) if hash_input_files else None
        payload = {
            "path": str(path.resolve()),
            "size_bytes": int(stat.st_size),
            "modified_time_ns": int(stat.st_mtime_ns),
            "rows": rows,
            "minimum_second": minimum_second,
            "maximum_second": maximum_second,
            "feature_builder_version": builder_version,
            "feature_schema_version": schema_version,
            "content_sha256": content_hash,
        }
        output[receive_date] = InputPartition(
            receive_date=receive_date,
            path=path,
            rows=rows,
            minimum_second=minimum_second,
            maximum_second=maximum_second,
            feature_builder_version=builder_version,
            feature_schema_version=schema_version,
            size_bytes=int(stat.st_size),
            modified_time_ns=int(stat.st_mtime_ns),
            content_sha256=content_hash,
            signature=sha256_text(compact_json(payload)),
        )
    if not output:
        raise SystemExit(f"No feature partitions found under {symbol_root}")
    return output


def selected_dates(
    partitions: Mapping[date, InputPartition],
    start_date: date | None,
    end_date: date | None,
    max_dates: int | None,
) -> list[date]:
    values = sorted(
        day
        for day in partitions
        if (start_date is None or day >= start_date)
        and (end_date is None or day <= end_date)
    )
    if max_dates is not None:
        if max_dates <= 0:
            raise SystemExit("--max-dates must be positive")
        values = values[:max_dates]
    if not values:
        raise SystemExit("No feature dates matched the requested filters")
    return values


def dependency_signature(
    day: date,
    horizon_seconds: int,
    partitions: Mapping[date, InputPartition],
    label_logic_signature: str,
) -> str:
    dependencies = [partitions[day]]
    next_partition = partitions.get(day + timedelta(days=1))
    if next_partition is not None:
        dependencies.append(next_partition)
    payload = {
        "decision_date": day.isoformat(),
        "horizon_seconds": horizon_seconds,
        "label_logic_signature": label_logic_signature,
        "dependencies": [
            {
                "receive_date": item.receive_date.isoformat(),
                "signature": item.signature,
            }
            for item in dependencies
        ],
    }
    return sha256_text(compact_json(payload))


# ---------------------------------------------------------------------------
# SQLite build state
# ---------------------------------------------------------------------------


def connect_state(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            started_at_utc TEXT NOT NULL,
            completed_at_utc TEXT,
            status TEXT NOT NULL,
            configuration_json TEXT NOT NULL,
            rebuilt_partitions INTEGER NOT NULL DEFAULT 0,
            skipped_partitions INTEGER NOT NULL DEFAULT 0,
            failed_partitions INTEGER NOT NULL DEFAULT 0,
            error_text TEXT
        );

        CREATE TABLE IF NOT EXISTS input_partitions (
            symbol TEXT NOT NULL,
            receive_date TEXT NOT NULL,
            path TEXT NOT NULL,
            signature TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            modified_time_ns INTEGER NOT NULL,
            rows INTEGER NOT NULL,
            minimum_second INTEGER NOT NULL,
            maximum_second INTEGER NOT NULL,
            feature_builder_version TEXT,
            feature_schema_version INTEGER,
            content_sha256 TEXT,
            last_seen_run_id TEXT NOT NULL,
            PRIMARY KEY(symbol, receive_date)
        );

        CREATE TABLE IF NOT EXISTS label_partitions (
            symbol TEXT NOT NULL,
            decision_date TEXT NOT NULL,
            horizon_seconds INTEGER NOT NULL,
            dependency_signature TEXT NOT NULL,
            output_path TEXT NOT NULL,
            output_rows INTEGER NOT NULL DEFAULT 0,
            minimum_decision_second INTEGER,
            maximum_decision_second INTEGER,
            status TEXT NOT NULL,
            started_at_utc TEXT,
            completed_at_utc TEXT,
            error_text TEXT,
            PRIMARY KEY(symbol, decision_date, horizon_seconds)
        );
        """
    )
    connection.commit()
    return connection


def record_inputs(
    connection: sqlite3.Connection,
    symbol: str,
    run_id: str,
    partitions: Mapping[date, InputPartition],
) -> None:
    for item in partitions.values():
        connection.execute(
            """
            INSERT INTO input_partitions (
                symbol, receive_date, path, signature, size_bytes,
                modified_time_ns, rows, minimum_second, maximum_second,
                feature_builder_version, feature_schema_version,
                content_sha256, last_seen_run_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, receive_date) DO UPDATE SET
                path=excluded.path,
                signature=excluded.signature,
                size_bytes=excluded.size_bytes,
                modified_time_ns=excluded.modified_time_ns,
                rows=excluded.rows,
                minimum_second=excluded.minimum_second,
                maximum_second=excluded.maximum_second,
                feature_builder_version=excluded.feature_builder_version,
                feature_schema_version=excluded.feature_schema_version,
                content_sha256=excluded.content_sha256,
                last_seen_run_id=excluded.last_seen_run_id
            """,
            (
                symbol,
                item.receive_date.isoformat(),
                str(item.path),
                item.signature,
                item.size_bytes,
                item.modified_time_ns,
                item.rows,
                item.minimum_second,
                item.maximum_second,
                item.feature_builder_version,
                item.feature_schema_version,
                item.content_sha256,
                run_id,
            ),
        )
    connection.commit()


def get_label_state(
    connection: sqlite3.Connection,
    symbol: str,
    day: date,
    horizon_seconds: int,
) -> sqlite3.Row | None:
    return connection.execute(
        """
        SELECT * FROM label_partitions
        WHERE symbol=? AND decision_date=? AND horizon_seconds=?
        """,
        (symbol, day.isoformat(), horizon_seconds),
    ).fetchone()


def mark_label_running(
    connection: sqlite3.Connection,
    symbol: str,
    day: date,
    horizon_seconds: int,
    dependency: str,
    output_path: Path,
) -> None:
    connection.execute(
        """
        INSERT INTO label_partitions (
            symbol, decision_date, horizon_seconds, dependency_signature,
            output_path, status, started_at_utc
        ) VALUES (?, ?, ?, ?, ?, 'RUNNING', ?)
        ON CONFLICT(symbol, decision_date, horizon_seconds) DO UPDATE SET
            dependency_signature=excluded.dependency_signature,
            output_path=excluded.output_path,
            output_rows=0,
            minimum_decision_second=NULL,
            maximum_decision_second=NULL,
            status='RUNNING',
            started_at_utc=excluded.started_at_utc,
            completed_at_utc=NULL,
            error_text=NULL
        """,
        (
            symbol,
            day.isoformat(),
            horizon_seconds,
            dependency,
            str(output_path),
            utc_now_iso(),
        ),
    )
    connection.commit()


def mark_label_completed(
    connection: sqlite3.Connection,
    symbol: str,
    day: date,
    horizon_seconds: int,
    dependency: str,
    output_path: Path,
    rows: int,
    minimum_second: int | None,
    maximum_second: int | None,
) -> None:
    connection.execute(
        """
        UPDATE label_partitions SET
            dependency_signature=?, output_path=?, output_rows=?,
            minimum_decision_second=?, maximum_decision_second=?,
            status='COMPLETED', completed_at_utc=?, error_text=NULL
        WHERE symbol=? AND decision_date=? AND horizon_seconds=?
        """,
        (
            dependency,
            str(output_path),
            rows,
            minimum_second,
            maximum_second,
            utc_now_iso(),
            symbol,
            day.isoformat(),
            horizon_seconds,
        ),
    )
    connection.commit()


def mark_label_failed(
    connection: sqlite3.Connection,
    symbol: str,
    day: date,
    horizon_seconds: int,
    error_text: str,
) -> None:
    connection.execute(
        """
        UPDATE label_partitions SET
            status='FAILED', completed_at_utc=?, error_text=?
        WHERE symbol=? AND decision_date=? AND horizon_seconds=?
        """,
        (utc_now_iso(), error_text, symbol, day.isoformat(), horizon_seconds),
    )
    connection.commit()


# ---------------------------------------------------------------------------
# Exact forward-label construction
# ---------------------------------------------------------------------------


@njit(cache=True)
def forward_extrema_first_indices(
    values: np.ndarray,
    valid: np.ndarray,
    window_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact O(n) forward high/low with earliest tied-extreme positions."""
    length = len(values)
    highs = np.full(length, np.nan, dtype=np.float64)
    lows = np.full(length, np.nan, dtype=np.float64)
    high_indices = np.full(length, -1, dtype=np.int64)
    low_indices = np.full(length, -1, dtype=np.int64)
    complete = np.zeros(length, dtype=np.uint8)

    maximum_queue = np.empty(length, dtype=np.int64)
    minimum_queue = np.empty(length, dtype=np.int64)
    maximum_head = 0
    maximum_tail = 0
    minimum_head = 0
    minimum_tail = 0
    invalid_count = 0

    for index in range(length):
        expired = index - window_length
        while maximum_head < maximum_tail and maximum_queue[maximum_head] <= expired:
            maximum_head += 1
        while minimum_head < minimum_tail and minimum_queue[minimum_head] <= expired:
            minimum_head += 1

        if not valid[index]:
            invalid_count += 1
        else:
            while (
                maximum_head < maximum_tail
                and values[maximum_queue[maximum_tail - 1]] < values[index]
            ):
                maximum_tail -= 1
            maximum_queue[maximum_tail] = index
            maximum_tail += 1

            while (
                minimum_head < minimum_tail
                and values[minimum_queue[minimum_tail - 1]] > values[index]
            ):
                minimum_tail -= 1
            minimum_queue[minimum_tail] = index
            minimum_tail += 1

        if index >= window_length and not valid[index - window_length]:
            invalid_count -= 1

        if index >= window_length - 1:
            start = index - window_length + 1
            if (
                invalid_count == 0
                and maximum_head < maximum_tail
                and minimum_head < minimum_tail
            ):
                maximum_index = maximum_queue[maximum_head]
                minimum_index = minimum_queue[minimum_head]
                highs[start] = values[maximum_index]
                lows[start] = values[minimum_index]
                high_indices[start] = maximum_index
                low_indices[start] = minimum_index
                complete[start] = 1

    return highs, lows, high_indices, low_indices, complete


@dataclass
class DenseLocalData:
    base_second: int
    price: np.ndarray
    observed: np.ndarray
    ready: np.ndarray


def read_feature_partition(item: InputPartition, symbol: str) -> pd.DataFrame:
    _, pq = require_arrow()
    parquet_file = pq.ParquetFile(item.path)
    try:
        frame = parquet_file.read(
            columns=[
                "symbol",
                "receive_second",
                "price_for_features",
                "flow_feature_context_ready",
            ]
        ).to_pandas()
    finally:
        close = getattr(parquet_file, "close", None)
        if callable(close):
            close()
    frame = frame[frame["symbol"].astype(str).str.upper() == symbol].copy()
    frame["receive_second"] = pd.to_numeric(
        frame["receive_second"], errors="raise"
    ).astype(np.int64)
    frame["price_for_features"] = pd.to_numeric(
        frame["price_for_features"], errors="coerce"
    )
    frame["flow_feature_context_ready"] = frame[
        "flow_feature_context_ready"
    ].fillna(False).astype(bool)
    if frame["receive_second"].duplicated().any():
        raise RuntimeError(f"Duplicate receive_second values in {item.path}")
    return frame.sort_values("receive_second", kind="mergesort").reset_index(drop=True)


def load_local_data(
    day: date,
    partitions: Mapping[date, InputPartition],
    symbol: str,
) -> DenseLocalData:
    frames = [read_feature_partition(partitions[day], symbol)]
    next_partition = partitions.get(day + timedelta(days=1))
    if next_partition is not None:
        frames.append(read_feature_partition(next_partition, symbol))
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.sort_values("receive_second", kind="mergesort").reset_index(drop=True)
    if frame["receive_second"].duplicated().any():
        duplicates = frame.loc[
            frame["receive_second"].duplicated(keep=False), "receive_second"
        ].head().tolist()
        raise RuntimeError(f"Cross-partition duplicate seconds near {day}: {duplicates}")

    base_second = epoch_second_for_date(day)
    seconds = frame["receive_second"].to_numpy(dtype=np.int64)
    positions = seconds - base_second
    keep = positions >= 0
    frame = frame.loc[keep].reset_index(drop=True)
    positions = positions[keep]
    if not len(positions):
        raise RuntimeError(f"No feature seconds exist on or after {day}")
    length = int(np.max(positions)) + 1
    dense_price = np.full(length, np.nan, dtype=np.float64)
    dense_observed = np.zeros(length, dtype=np.bool_)
    dense_ready = np.zeros(length, dtype=np.bool_)
    dense_price[positions] = frame["price_for_features"].to_numpy(dtype=float)
    dense_observed[positions] = True
    dense_ready[positions] = frame["flow_feature_context_ready"].to_numpy(dtype=bool)
    return DenseLocalData(
        base_second=base_second,
        price=dense_price,
        observed=dense_observed,
        ready=dense_ready,
    )


def build_horizon_label_frame(
    day: date,
    local: DenseLocalData,
    symbol: str,
    horizon_seconds: int,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    path_observations = horizon_seconds + 1
    finite_positive = local.observed & np.isfinite(local.price) & (local.price > 0.0)
    highs, lows, high_positions, low_positions, complete = forward_extrema_first_indices(
        local.price, finite_positive, path_observations
    )

    day_end_position = min(SECONDS_PER_DAY, len(local.price))
    anchor_positions = np.arange(day_end_position, dtype=np.int64)
    eligible_ready = local.observed[anchor_positions] & local.ready[anchor_positions]
    ready_count = int(np.sum(eligible_ready))
    anchor_positions = anchor_positions[eligible_ready]
    entry_positions = anchor_positions + 1
    in_bounds = entry_positions + horizon_seconds < len(local.price)
    anchor_positions = anchor_positions[in_bounds]
    entry_positions = entry_positions[in_bounds]
    complete_mask = complete[entry_positions].astype(bool)
    anchor_positions = anchor_positions[complete_mask]
    entry_positions = entry_positions[complete_mask]

    if not len(anchor_positions):
        empty = pd.DataFrame(columns=LABEL_COLUMNS)
        summary = {
            "decision_date": day.isoformat(),
            "horizon_seconds": horizon_seconds,
            "eligible_ready_anchors": ready_count,
            "valid_label_rows": 0,
            "invalid_or_incomplete_ready_anchors": ready_count,
        }
        return empty, summary, []

    terminal_positions = entry_positions + horizon_seconds
    entry_price = local.price[entry_positions]
    future_high = highs[entry_positions]
    future_low = lows[entry_positions]
    terminal_price = local.price[terminal_positions]
    future_high_positions = high_positions[entry_positions]
    future_low_positions = low_positions[entry_positions]

    decision_seconds = anchor_positions + local.base_second
    entry_seconds = entry_positions + local.base_second
    path_end_seconds = terminal_positions + local.base_second
    high_seconds = future_high_positions + local.base_second
    low_seconds = future_low_positions + local.base_second

    swing = (future_high - future_low) / entry_price * 10_000.0
    upside = (future_high / entry_price - 1.0) * 10_000.0
    downside = (1.0 - future_low / entry_price) * 10_000.0
    terminal_return = (terminal_price / entry_price - 1.0) * 10_000.0

    tolerance = 1e-8
    for values in (swing, upside, downside):
        if np.any(values < -tolerance):
            raise RuntimeError(f"Materially negative excursion value for {day}")
        values[(values < 0.0) & (values >= -tolerance)] = 0.0

    high_first = pd.array(
        [
            True if high < low else False if low < high else pd.NA
            for high, low in zip(high_seconds, low_seconds)
        ],
        dtype="boolean",
    )

    frame = pd.DataFrame(
        {
            "symbol": symbol,
            "decision_date": day.isoformat(),
            "horizon_seconds": np.full(len(decision_seconds), horizon_seconds, dtype=np.int32),
            "decision_second": decision_seconds.astype(np.int64),
            "entry_second": entry_seconds.astype(np.int64),
            "path_end_second": path_end_seconds.astype(np.int64),
            "entry_price": entry_price.astype(np.float64),
            "future_high_price": future_high.astype(np.float64),
            "future_low_price": future_low.astype(np.float64),
            "future_terminal_price": terminal_price.astype(np.float64),
            "future_high_second": high_seconds.astype(np.int64),
            "future_low_second": low_seconds.astype(np.int64),
            "future_swing_bps": swing.astype(np.float64),
            "future_upside_bps": upside.astype(np.float64),
            "future_downside_bps": downside.astype(np.float64),
            "future_terminal_return_bps": terminal_return.astype(np.float64),
            "future_high_first": high_first,
        }
    )

    invariants = validate_label_frame(frame, day, horizon_seconds)
    summary = {
        "decision_date": day.isoformat(),
        "horizon_seconds": horizon_seconds,
        "eligible_ready_anchors": ready_count,
        "valid_label_rows": len(frame),
        "invalid_or_incomplete_ready_anchors": ready_count - len(frame),
        "minimum_decision_second": int(frame["decision_second"].min()),
        "maximum_decision_second": int(frame["decision_second"].max()),
    }
    return frame, summary, invariants


def invariant_record(
    scope: str,
    check: str,
    passed: bool,
    details: str,
    target_id_value: str | None = None,
    segmentation_id: str | None = None,
) -> dict[str, Any]:
    return {
        "scope": scope,
        "target_id": target_id_value,
        "segmentation_id": segmentation_id,
        "check": check,
        "passed": bool(passed),
        "details": details,
    }


def validate_label_frame(
    frame: pd.DataFrame,
    day: date,
    horizon_seconds: int,
) -> list[dict[str, Any]]:
    scope = f"LABEL:{day}:{horizon_seconds}s"
    records: list[dict[str, Any]] = []
    if frame.empty:
        records.append(invariant_record(scope, "EMPTY_SCHEMA", list(frame.columns) == list(LABEL_COLUMNS), "empty partition"))
        return records
    seconds = frame["decision_second"].to_numpy(dtype=np.int64)
    records.append(invariant_record(scope, "SORTED_UNIQUE_DECISION_SECONDS", bool(np.all(np.diff(seconds) > 0)), f"rows={len(frame)}"))
    records.append(invariant_record(scope, "ENTRY_T_PLUS_1", bool(np.all(frame["entry_second"].to_numpy(dtype=np.int64) == seconds + 1)), "entry=decision+1"))
    records.append(invariant_record(scope, "EXACT_PATH_END", bool(np.all(frame["path_end_second"].to_numpy(dtype=np.int64) - frame["entry_second"].to_numpy(dtype=np.int64) == horizon_seconds)), f"horizon={horizon_seconds}"))
    reconstructed = (
        (frame["future_high_price"].to_numpy(dtype=float) - frame["future_low_price"].to_numpy(dtype=float))
        / frame["entry_price"].to_numpy(dtype=float)
        * 10_000.0
    )
    records.append(invariant_record(scope, "SWING_RECONCILIATION", bool(np.allclose(reconstructed, frame["future_swing_bps"].to_numpy(dtype=float), rtol=1e-10, atol=1e-8)), "(high-low)/entry"))
    records.append(invariant_record(scope, "SWING_EQUALS_UPSIDE_PLUS_DOWNSIDE", bool(np.allclose(frame["future_swing_bps"].to_numpy(dtype=float), frame["future_upside_bps"].to_numpy(dtype=float) + frame["future_downside_bps"].to_numpy(dtype=float), rtol=1e-10, atol=1e-8)), "swing=upside+downside"))
    extreme_bounds = (
        (frame["future_high_second"].to_numpy(dtype=np.int64) >= frame["entry_second"].to_numpy(dtype=np.int64))
        & (frame["future_high_second"].to_numpy(dtype=np.int64) <= frame["path_end_second"].to_numpy(dtype=np.int64))
        & (frame["future_low_second"].to_numpy(dtype=np.int64) >= frame["entry_second"].to_numpy(dtype=np.int64))
        & (frame["future_low_second"].to_numpy(dtype=np.int64) <= frame["path_end_second"].to_numpy(dtype=np.int64))
    )
    records.append(invariant_record(scope, "EXTREMES_INSIDE_PATH", bool(np.all(extreme_bounds)), "high/low timestamps inside path"))
    records.append(invariant_record(scope, "FINITE_POSITIVE_PRICES", bool(np.all(np.isfinite(frame[["entry_price", "future_high_price", "future_low_price", "future_terminal_price"]].to_numpy(dtype=float))) and np.all(frame[["entry_price", "future_high_price", "future_low_price", "future_terminal_price"]].to_numpy(dtype=float) > 0)), "finite positive prices"))
    return records


def label_output_path(
    output_root: Path,
    symbol: str,
    day: date,
    horizon_seconds: int,
) -> Path:
    return (
        output_root
        / "row_labels"
        / f"{HORIZON_PREFIX}{horizon_id(horizon_seconds)}"
        / f"{SYMBOL_PREFIX}{symbol}"
        / f"{DATE_PREFIX}{day.isoformat()}"
        / "part-00000.parquet"
    )


def validate_written_label_file(
    path: Path,
    expected_rows: int,
    expected_dependency: str,
    day: date,
    horizon_seconds: int,
) -> None:
    _, pq = require_arrow()
    parquet_file = pq.ParquetFile(path)
    try:
        if int(parquet_file.metadata.num_rows) != expected_rows:
            raise RuntimeError(f"Row-count mismatch in {path}")
        metadata = decode_metadata(parquet_file.schema_arrow)
        if metadata.get("dependency_signature") != expected_dependency:
            raise RuntimeError(f"Dependency signature mismatch in {path}")
        if metadata.get("decision_date") != day.isoformat():
            raise RuntimeError(f"Decision date metadata mismatch in {path}")
        if int(metadata.get("horizon_seconds", "-1")) != horizon_seconds:
            raise RuntimeError(f"Horizon metadata mismatch in {path}")
        missing = [column for column in LABEL_COLUMNS if column not in parquet_file.schema_arrow.names]
        if missing:
            raise RuntimeError(f"Missing label columns in {path}: {missing}")
    finally:
        close = getattr(parquet_file, "close", None)
        if callable(close):
            close()


# ---------------------------------------------------------------------------
# Target and segmentation construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SegmentationSpec:
    segmentation_id: str
    rule_type: str
    bridge_gap_seconds: int | None


def segmentation_specs(bridge_gaps: Sequence[int]) -> list[SegmentationSpec]:
    return [
        SegmentationSpec(segmentation_id_for_gap(gap), "BRIDGE_GAP", gap)
        for gap in bridge_gaps
    ] + [SegmentationSpec("overlap_reference", "OVERLAPPING_FUTURE_INTERVALS", None)]


def bridge_group_ordinals(decision_seconds: np.ndarray, bridge_gap: int) -> np.ndarray:
    if not len(decision_seconds):
        return np.array([], dtype=np.int64)
    inactive = np.diff(decision_seconds) - 1
    starts = np.concatenate((np.array([True]), inactive > bridge_gap))
    return np.cumsum(starts, dtype=np.int64) - 1


def overlap_reference_ordinals(
    entry_seconds: np.ndarray,
    path_end_seconds: np.ndarray,
) -> np.ndarray:
    if not len(entry_seconds):
        return np.array([], dtype=np.int64)
    output = np.empty(len(entry_seconds), dtype=np.int64)
    group = 0
    output[0] = 0
    union_end = int(path_end_seconds[0])
    for index in range(1, len(entry_seconds)):
        if int(entry_seconds[index]) <= union_end:
            union_end = max(union_end, int(path_end_seconds[index]))
        else:
            group += 1
            union_end = int(path_end_seconds[index])
        output[index] = group
    return output


def group_bounds(ordinals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not len(ordinals):
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    starts = np.flatnonzero(np.concatenate((np.array([True]), np.diff(ordinals) != 0)))
    ends = np.concatenate((starts[1:], np.array([len(ordinals)])))
    return starts.astype(np.int64), ends.astype(np.int64)


def classify_path(upside: float, downside: float, dominance_ratio: float) -> tuple[float, bool, str]:
    total = upside + downside
    if not math.isfinite(total) or total <= 0:
        return float("nan"), False, "UNCLASSIFIED"
    share = max(upside, downside) / total
    if share + 1e-12 < dominance_ratio:
        return share, False, "TWO_SIDED"
    return share, True, "TREND_UP" if upside >= downside else "TREND_DOWN"


def stable_event_id(target: str, segmentation: str, first_second: int) -> str:
    digest = hashlib.sha1(
        f"{target}|{segmentation}|{first_second}".encode("utf-8")
    ).hexdigest()[:16]
    return f"EVT-{digest}"


def empty_event_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=EVENT_COLUMNS)


def build_events(
    positive: pd.DataFrame,
    target: str,
    symbol: str,
    horizon_seconds: int,
    threshold_bps: float,
    spec: SegmentationSpec,
    trend_dominance_ratio: float,
) -> pd.DataFrame:
    if positive.empty:
        return empty_event_frame()
    frame = positive.sort_values("decision_second", kind="mergesort").reset_index(drop=True)
    decisions = frame["decision_second"].to_numpy(dtype=np.int64)
    entries = frame["entry_second"].to_numpy(dtype=np.int64)
    path_ends = frame["path_end_second"].to_numpy(dtype=np.int64)
    if spec.rule_type == "BRIDGE_GAP":
        assert spec.bridge_gap_seconds is not None
        ordinals = bridge_group_ordinals(decisions, spec.bridge_gap_seconds)
    else:
        ordinals = overlap_reference_ordinals(entries, path_ends)
    starts, ends = group_bounds(ordinals)

    rows: list[dict[str, Any]] = []
    previous_end: int | None = None
    for ordinal, (start_index, end_index) in enumerate(zip(starts, ends), start=1):
        block = frame.iloc[int(start_index) : int(end_index)]
        block_decisions = block["decision_second"].to_numpy(dtype=np.int64)
        differences = np.diff(block_decisions)
        inactive_gaps = np.maximum(differences - 1, 0)
        event_start = int(block_decisions[0])
        event_end = int(block_decisions[-1])
        duration = event_end - event_start + 1
        maximum_swing = float(block["future_swing_bps"].max())
        representative_candidates = block.index[
            np.isclose(
                block["future_swing_bps"].to_numpy(dtype=float),
                maximum_swing,
                rtol=0.0,
                atol=1e-10,
            )
        ]
        representative = block.loc[int(representative_candidates[0])]
        dominant_share, trend_like, path_type = classify_path(
            float(representative["future_upside_bps"]),
            float(representative["future_downside_bps"]),
            trend_dominance_ratio,
        )
        if pd.isna(representative["future_high_first"]):
            representative_high_first: bool | None = None
        else:
            representative_high_first = bool(representative["future_high_first"])
        gap_from_previous = (
            None if previous_end is None else event_start - previous_end - 1
        )
        isolated = previous_end is None or bool(
            gap_from_previous is not None and gap_from_previous > horizon_seconds
        )
        row = {
            "event_schema_version": EVENT_SCHEMA_VERSION,
            "profiler_version": PROFILER_VERSION,
            "event_id": stable_event_id(target, spec.segmentation_id, event_start),
            "target_id": target,
            "symbol": symbol,
            "horizon_seconds": horizon_seconds,
            "horizon_minutes": horizon_seconds // 60,
            "threshold_bps": threshold_bps,
            "segmentation_id": spec.segmentation_id,
            "segmentation_rule_type": spec.rule_type,
            "bridge_gap_seconds": spec.bridge_gap_seconds,
            "event_ordinal": ordinal,
            "event_start_second": event_start,
            "event_end_second": event_end,
            "event_start_timestamp_utc": iso_from_second(event_start),
            "event_end_timestamp_utc": iso_from_second(event_end),
            "event_start_date": date_from_second(event_start),
            "event_end_date": date_from_second(event_end),
            "crosses_utc_date": date_from_second(event_start) != date_from_second(event_end),
            "event_duration_seconds": duration,
            "number_of_positive_anchors": len(block),
            "positive_anchor_coverage_pct": percentage(len(block), duration),
            "number_of_contiguous_positive_runs": 1 + int(np.sum(differences > 1)),
            "total_internal_inactive_seconds": int(np.sum(inactive_gaps)) if len(inactive_gaps) else 0,
            "maximum_internal_inactive_gap_seconds": int(np.max(inactive_gaps)) if len(inactive_gaps) else 0,
            "future_interval_union_start_second": int(block["entry_second"].min()),
            "future_interval_union_end_second": int(block["path_end_second"].max()),
            "future_interval_union_duration_seconds": int(block["path_end_second"].max() - block["entry_second"].min() + 1),
            "maximum_future_swing_bps": maximum_swing,
            "maximum_future_upside_bps": float(block["future_upside_bps"].max()),
            "maximum_future_downside_bps": float(block["future_downside_bps"].max()),
            "representative_terminal_return_bps": float(representative["future_terminal_return_bps"]),
            "representative_future_high_second": int(representative["future_high_second"]),
            "representative_future_low_second": int(representative["future_low_second"]),
            "representative_high_first": representative_high_first,
            "dominant_excursion_share": dominant_share,
            "trend_like": trend_like,
            "path_type": path_type,
            "seconds_since_previous_event": gap_from_previous,
            "isolated_onset": isolated,
        }
        rows.append(row)
        previous_end = event_end
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def event_output_path(
    output_root: Path,
    symbol: str,
    target: str,
    segmentation: str,
) -> Path:
    return (
        output_root
        / "events"
        / f"{TARGET_PREFIX}{target}"
        / f"{SEGMENTATION_PREFIX}{segmentation}"
        / f"{SYMBOL_PREFIX}{symbol}"
        / "part-00000.parquet"
    )


def validate_events(
    events: pd.DataFrame,
    positive: pd.DataFrame,
    target: str,
    spec: SegmentationSpec,
    horizon_seconds: int,
) -> list[dict[str, Any]]:
    scope = "EVENTS"
    records: list[dict[str, Any]] = []
    expected_positive = len(positive)
    observed_positive = int(events["number_of_positive_anchors"].sum()) if len(events) else 0
    records.append(invariant_record(scope, "POSITIVE_ANCHOR_RECONCILIATION", observed_positive == expected_positive, f"expected={expected_positive}, observed={observed_positive}", target, spec.segmentation_id))
    if events.empty:
        records.append(invariant_record(scope, "EMPTY_WHEN_NO_POSITIVES", expected_positive == 0, f"positive_rows={expected_positive}", target, spec.segmentation_id))
        return records
    starts = events["event_start_second"].to_numpy(dtype=np.int64)
    ends = events["event_end_second"].to_numpy(dtype=np.int64)
    records.append(invariant_record(scope, "EVENTS_SORTED_NONOVERLAPPING", bool(np.all(np.diff(starts) > 0) and np.all(starts[1:] > ends[:-1])), f"events={len(events)}", target, spec.segmentation_id))
    records.append(invariant_record(scope, "POSITIVE_COVERAGE_BOUNDS", bool(events["positive_anchor_coverage_pct"].between(0.0, 100.0, inclusive="both").all()), "coverage within [0,100]", target, spec.segmentation_id))
    if spec.rule_type == "BRIDGE_GAP":
        assert spec.bridge_gap_seconds is not None
        records.append(invariant_record(scope, "MAX_INTERNAL_GAP_WITHIN_BRIDGE", bool((events["maximum_internal_inactive_gap_seconds"] <= spec.bridge_gap_seconds).all()), f"bridge={spec.bridge_gap_seconds}", target, spec.segmentation_id))
    else:
        records.append(invariant_record(scope, "OVERLAP_REFERENCE_HORIZON", bool((events["maximum_internal_inactive_gap_seconds"] <= horizon_seconds - 1).all()), f"equivalent_max_gap={horizon_seconds - 1}", target, spec.segmentation_id))
    return records


# ---------------------------------------------------------------------------
# Report construction
# ---------------------------------------------------------------------------


def target_prevalence_row(
    target: str,
    horizon_seconds: int,
    threshold_bps: float,
    daily: pd.DataFrame,
) -> dict[str, Any]:
    valid_rows = int(daily["valid_label_rows"].sum())
    positive_rows = int(daily["positive_rows"].sum())
    return {
        "target_id": target,
        "horizon_minutes": horizon_seconds // 60,
        "horizon_seconds": horizon_seconds,
        "threshold_bps": threshold_bps,
        "dates": int(len(daily)),
        "valid_label_rows": valid_rows,
        "positive_rows": positive_rows,
        "positive_rate_pct": percentage(positive_rows, valid_rows),
        "mean_daily_positive_rate_pct": safe_stat(daily["positive_rate_pct"], "mean"),
        "median_daily_positive_rate_pct": safe_stat(daily["positive_rate_pct"], "median"),
        "p90_daily_positive_rate_pct": safe_stat(daily["positive_rate_pct"], "p90"),
        "maximum_daily_positive_rate_pct": safe_stat(daily["positive_rate_pct"], "maximum"),
        "zero_positive_dates": int((daily["positive_rows"] == 0).sum()),
    }


def segmentation_comparison_row(
    target: str,
    horizon_seconds: int,
    threshold_bps: float,
    spec: SegmentationSpec,
    events: pd.DataFrame,
    positive_rows: int,
    observed_dates: int,
) -> dict[str, Any]:
    durations = events["event_duration_seconds"] if len(events) else pd.Series(dtype=float)
    return {
        "target_id": target,
        "horizon_minutes": horizon_seconds // 60,
        "threshold_bps": threshold_bps,
        "segmentation_id": spec.segmentation_id,
        "segmentation_rule_type": spec.rule_type,
        "bridge_gap_seconds": spec.bridge_gap_seconds,
        "positive_rows": positive_rows,
        "event_count": len(events),
        "isolated_onset_count": int(events["isolated_onset"].sum()) if len(events) else 0,
        "events_per_observed_date": float(len(events) / observed_dates) if observed_dates else float("nan"),
        "median_duration_seconds": safe_stat(durations, "median"),
        "p75_duration_seconds": safe_stat(durations, "p75"),
        "p90_duration_seconds": safe_stat(durations, "p90"),
        "p95_duration_seconds": safe_stat(durations, "p95"),
        "maximum_duration_seconds": safe_stat(durations, "maximum"),
        "events_at_least_15m": int((durations >= 900).sum()) if len(events) else 0,
        "events_at_least_30m": int((durations >= 1_800).sum()) if len(events) else 0,
        "events_at_least_60m": int((durations >= 3_600).sum()) if len(events) else 0,
        "events_at_least_120m": int((durations >= 7_200).sum()) if len(events) else 0,
        "share_events_at_least_60m_pct": percentage(int((durations >= 3_600).sum()), len(events)),
        "median_positive_anchors_per_event": safe_stat(events["number_of_positive_anchors"] if len(events) else [], "median"),
        "median_positive_anchor_coverage_pct": safe_stat(events["positive_anchor_coverage_pct"] if len(events) else [], "median"),
        "median_maximum_internal_gap_seconds": safe_stat(events["maximum_internal_inactive_gap_seconds"] if len(events) else [], "median"),
        "maximum_internal_gap_seconds": safe_stat(events["maximum_internal_inactive_gap_seconds"] if len(events) else [], "maximum"),
        "median_maximum_swing_bps": safe_stat(events["maximum_future_swing_bps"] if len(events) else [], "median"),
        "p90_maximum_swing_bps": safe_stat(events["maximum_future_swing_bps"] if len(events) else [], "p90"),
        "high_first_pct": percentage(int((events["representative_high_first"] == True).sum()), int(events["representative_high_first"].notna().sum())) if len(events) else float("nan"),
        "trend_up_pct": percentage(int((events["path_type"] == "TREND_UP").sum()), len(events)),
        "trend_down_pct": percentage(int((events["path_type"] == "TREND_DOWN").sum()), len(events)),
        "two_sided_pct": percentage(int((events["path_type"] == "TWO_SIDED").sum()), len(events)),
        "cross_midnight_events": int(events["crosses_utc_date"].sum()) if len(events) else 0,
    }


def build_transition_profile(comparison: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    bridge = comparison[comparison["segmentation_rule_type"] == "BRIDGE_GAP"].copy()
    for target, group in bridge.groupby("target_id", sort=False):
        group = group.sort_values("bridge_gap_seconds")
        records = group.to_dict("records")
        for previous, current in zip(records, records[1:]):
            previous_count = int(previous["event_count"])
            current_count = int(current["event_count"])
            merged = previous_count - current_count
            rows.append(
                {
                    "target_id": target,
                    "horizon_minutes": int(current["horizon_minutes"]),
                    "threshold_bps": float(current["threshold_bps"]),
                    "previous_bridge_gap_seconds": int(previous["bridge_gap_seconds"]),
                    "current_bridge_gap_seconds": int(current["bridge_gap_seconds"]),
                    "previous_event_count": previous_count,
                    "current_event_count": current_count,
                    "events_merged": merged,
                    "event_count_drop_pct": percentage(merged, previous_count),
                }
            )
    return pd.DataFrame(rows)


def build_stable_ranges(
    transition: pd.DataFrame,
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target, group in transition.groupby("target_id", sort=False):
        finite = group[np.isfinite(pd.to_numeric(group["event_count_drop_pct"], errors="coerce"))].copy()
        if finite.empty:
            base = comparison[comparison["target_id"] == target].iloc[0]
            rows.append(
                {
                    "target_id": target,
                    "horizon_minutes": int(base["horizon_minutes"]),
                    "threshold_bps": float(base["threshold_bps"]),
                    "candidate_strict_bridge_seconds": None,
                    "candidate_primary_bridge_seconds": None,
                    "stable_transition_drop_pct": float("nan"),
                    "next_transition_drop_pct": float("nan"),
                    "stability_class": "INSUFFICIENT",
                }
            )
            continue
        best_index = finite["event_count_drop_pct"].astype(float).idxmin()
        best = finite.loc[best_index]
        next_rows = group[
            group["previous_bridge_gap_seconds"]
            == int(best["current_bridge_gap_seconds"])
        ]
        next_drop = float(next_rows.iloc[0]["event_count_drop_pct"]) if len(next_rows) else float("nan")
        drop = float(best["event_count_drop_pct"])
        if drop <= 5.0:
            stability_class = "STABLE"
        elif drop <= 10.0:
            stability_class = "MODERATE"
        else:
            stability_class = "UNSTABLE"
        rows.append(
            {
                "target_id": target,
                "horizon_minutes": int(best["horizon_minutes"]),
                "threshold_bps": float(best["threshold_bps"]),
                "candidate_strict_bridge_seconds": int(best["previous_bridge_gap_seconds"]),
                "candidate_primary_bridge_seconds": int(best["current_bridge_gap_seconds"]),
                "stable_transition_drop_pct": drop,
                "next_transition_drop_pct": next_drop,
                "stability_class": stability_class,
            }
        )
    return pd.DataFrame(rows)


def build_viability_profile(
    prevalence: pd.DataFrame,
    comparison: pd.DataFrame,
    stable_ranges: pd.DataFrame,
    min_events: int,
    min_isolated_events: int,
    min_positive_rate_pct: float,
    max_positive_rate_pct: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    prevalence_by_target = prevalence.set_index("target_id")
    comparison_indexed = comparison.set_index(["target_id", "segmentation_id"])
    for stable in stable_ranges.to_dict("records"):
        target = str(stable["target_id"])
        prevalence_row = prevalence_by_target.loc[target]
        primary_gap = stable["candidate_primary_bridge_seconds"]
        if primary_gap is None or pd.isna(primary_gap):
            event_count = 0
            isolated_count = 0
            segmentation = None
        else:
            segmentation = segmentation_id_for_gap(int(primary_gap))
            comparison_row = comparison_indexed.loc[(target, segmentation)]
            event_count = int(comparison_row["event_count"])
            isolated_count = int(comparison_row["isolated_onset_count"])
        positive_rate = float(prevalence_row["positive_rate_pct"])
        reasons: list[str] = []
        if event_count < min_events:
            reasons.append(f"events<{min_events}")
        if isolated_count < min_isolated_events:
            reasons.append(f"isolated<{min_isolated_events}")
        if not (min_positive_rate_pct <= positive_rate <= max_positive_rate_pct):
            reasons.append(
                f"positive_rate_outside_{min_positive_rate_pct:g}-{max_positive_rate_pct:g}%"
            )
        if stable["stability_class"] == "UNSTABLE":
            reasons.append("no_clear_gap_plateau")
        if not reasons:
            sample_viability = "STRONG"
        elif event_count >= max(100, min_events // 2) and isolated_count >= max(50, min_isolated_events // 2):
            sample_viability = "CAUTION"
        else:
            sample_viability = "LIMITED"
        rows.append(
            {
                "target_id": target,
                "horizon_minutes": int(prevalence_row["horizon_minutes"]),
                "threshold_bps": float(prevalence_row["threshold_bps"]),
                "positive_rate_pct": positive_rate,
                "candidate_primary_segmentation": segmentation,
                "candidate_primary_bridge_seconds": primary_gap,
                "candidate_primary_event_count": event_count,
                "candidate_primary_isolated_onset_count": isolated_count,
                "stability_class": stable["stability_class"],
                "sample_viability": sample_viability,
                "caution_reasons": ";".join(reasons),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["sample_viability", "candidate_primary_isolated_onset_count", "candidate_primary_event_count"],
        ascending=[True, False, False],
        kind="mergesort",
    )


def duration_distribution_rows(
    target: str,
    spec: SegmentationSpec,
    events: pd.DataFrame,
) -> list[dict[str, Any]]:
    buckets = (
        ("LT_5M", 0, 300),
        ("5_TO_15M", 300, 900),
        ("15_TO_30M", 900, 1_800),
        ("30_TO_60M", 1_800, 3_600),
        ("1_TO_2H", 3_600, 7_200),
        ("GE_2H", 7_200, None),
    )
    rows: list[dict[str, Any]] = []
    durations = events["event_duration_seconds"] if len(events) else pd.Series(dtype=float)
    for bucket, lower, upper in buckets:
        mask = durations >= lower
        if upper is not None:
            mask &= durations < upper
        count = int(mask.sum())
        rows.append(
            {
                "target_id": target,
                "segmentation_id": spec.segmentation_id,
                "duration_bucket": bucket,
                "event_count": count,
                "event_share_pct": percentage(count, len(events)),
            }
        )
    return rows


def daily_event_rows(target: str, spec: SegmentationSpec, events: pd.DataFrame) -> list[dict[str, Any]]:
    if events.empty:
        return []
    grouped = events.groupby("event_start_date", sort=True)
    return [
        {
            "target_id": target,
            "segmentation_id": spec.segmentation_id,
            "event_start_date": event_date,
            "event_count": len(group),
            "isolated_onset_count": int(group["isolated_onset"].sum()),
            "median_duration_seconds": safe_stat(group["event_duration_seconds"], "median"),
            "maximum_swing_bps": safe_stat(group["maximum_future_swing_bps"], "maximum"),
        }
        for event_date, group in grouped
    ]


def build_target_overlap(
    target_catalog: Mapping[str, dict[str, Any]],
    valid_seconds_by_horizon: Mapping[int, np.ndarray],
    positive_seconds_by_target: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    targets = sorted(target_catalog)
    valid_intersections: dict[tuple[int, int], int] = {}
    positive_on_horizon: dict[tuple[str, int], int] = {}
    horizons = sorted(valid_seconds_by_horizon)
    for left_index, left_horizon in enumerate(horizons):
        for right_horizon in horizons[left_index:]:
            key = (left_horizon, right_horizon)
            count = int(
                np.intersect1d(
                    valid_seconds_by_horizon[left_horizon],
                    valid_seconds_by_horizon[right_horizon],
                    assume_unique=True,
                ).size
            )
            valid_intersections[key] = count
            valid_intersections[(right_horizon, left_horizon)] = count
    for target in targets:
        positives = positive_seconds_by_target[target]
        for horizon in horizons:
            positive_on_horizon[(target, horizon)] = int(
                np.intersect1d(
                    positives,
                    valid_seconds_by_horizon[horizon],
                    assume_unique=True,
                ).size
            )

    rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(targets):
        left_horizon = int(target_catalog[left]["horizon_seconds"])
        for right in targets[left_index:]:
            right_horizon = int(target_catalog[right]["horizon_seconds"])
            left_positive = positive_seconds_by_target[left]
            right_positive = positive_seconds_by_target[right]
            intersection = int(
                np.intersect1d(left_positive, right_positive, assume_unique=True).size
            )
            left_common = positive_on_horizon[(left, right_horizon)]
            right_common = positive_on_horizon[(right, left_horizon)]
            union = left_common + right_common - intersection
            rows.append(
                {
                    "target_a": left,
                    "target_b": right,
                    "common_valid_decision_rows": valid_intersections[(left_horizon, right_horizon)],
                    "target_a_positive_on_common_rows": left_common,
                    "target_b_positive_on_common_rows": right_common,
                    "both_positive_rows": intersection,
                    "positive_union_rows": union,
                    "jaccard_pct": percentage(intersection, union),
                    "share_a_also_b_pct": percentage(intersection, left_common),
                    "share_b_also_a_pct": percentage(intersection, right_common),
                }
            )
    return pd.DataFrame(rows)


def markdown_findings(
    summary: Mapping[str, Any],
    prevalence: pd.DataFrame,
    stable_ranges: pd.DataFrame,
    viability: pd.DataFrame,
    baseline_target: str,
) -> str:
    lines = [
        "# Perps Expansion Target and Segmentation Sensitivity",
        "",
        "## Run summary",
        "",
        f"- Status: **{summary['status']}**",
        f"- Feature dates: **{summary['selected_dates']:,}**",
        f"- Horizons: **{summary['horizon_count']:,}**",
        f"- Targets: **{summary['target_count']:,}**",
        f"- Segmentation definitions per target: **{summary['segmentation_count']:,}**",
        f"- Failed invariants: **{summary['failed_invariants']:,}**",
        "",
        "## Interpretation rule",
        "",
        "This report measures label prevalence and event-definition stability. It does not measure predictive model performance, profitability, or executable warning time.",
        "",
        "## Target prevalence",
        "",
        "| Target | Positive rate | Positive rows |",
        "|---|---:|---:|",
    ]
    for row in prevalence.sort_values(["horizon_minutes", "threshold_bps"]).to_dict("records"):
        lines.append(
            f"| {row['target_id']} | {row['positive_rate_pct']:.2f}% | {int(row['positive_rows']):,} |"
        )
    lines.extend(
        [
            "",
            "## Data-supported bridge candidates",
            "",
            "The candidate range is the adjacent bridge-gap transition that changes the event count least. It is a profiling aid, not an automatically frozen research decision.",
            "",
            "| Target | Strict candidate | Primary candidate | Event-count drop | Stability |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in stable_ranges.sort_values(["horizon_minutes", "threshold_bps"]).to_dict("records"):
        strict = "n/a" if pd.isna(row["candidate_strict_bridge_seconds"]) else f"{int(row['candidate_strict_bridge_seconds'])}s"
        primary = "n/a" if pd.isna(row["candidate_primary_bridge_seconds"]) else f"{int(row['candidate_primary_bridge_seconds'])}s"
        drop = "n/a" if not math.isfinite(float(row["stable_transition_drop_pct"])) else f"{float(row['stable_transition_drop_pct']):.2f}%"
        lines.append(f"| {row['target_id']} | {strict} | {primary} | {drop} | {row['stability_class']} |")
    lines.extend(
        [
            "",
            "## Sample viability",
            "",
            "| Target | Candidate events | Isolated onsets | Positive rate | Viability | Cautions |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    for row in viability.sort_values(["horizon_minutes", "threshold_bps"]).to_dict("records"):
        lines.append(
            f"| {row['target_id']} | {int(row['candidate_primary_event_count']):,} | "
            f"{int(row['candidate_primary_isolated_onset_count']):,} | "
            f"{float(row['positive_rate_pct']):.2f}% | {row['sample_viability']} | "
            f"{row['caution_reasons'] or 'None'} |"
        )
    baseline = viability[viability["target_id"] == baseline_target]
    if len(baseline):
        row = baseline.iloc[0]
        lines.extend(
            [
                "",
                "## Frozen baseline reference",
                "",
                f"The existing baseline `{baseline_target}` remains present for direct comparison. Its profiling candidate is `{row['candidate_primary_segmentation']}` with {int(row['candidate_primary_event_count']):,} events and {int(row['candidate_primary_isolated_onset_count']):,} isolated onsets in this run.",
            ]
        )
    lines.extend(
        [
            "",
            "## Next decision",
            "",
            "Use the prevalence, stability and isolated-event sample sizes to shortlist target/grouping definitions. Freeze those definitions before comparing feature predictability or training models, so target selection is not optimized on model scores.",
            "",
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def run_self_test() -> None:
    parsed = parse_target_grid("5=35|50;15=50|75")
    assert parsed == {5: (35.0, 50.0), 15: (50.0, 75.0)}
    assert parse_bridge_gaps("0,15,30,60") == (0, 15, 30, 60)

    values = np.array([100.0, 100.0, 101.0, 99.0, 102.0, 102.0], dtype=np.float64)
    valid = np.ones(len(values), dtype=np.bool_)
    highs, lows, high_indices, low_indices, complete = forward_extrema_first_indices(values, valid, 4)
    assert complete[0] == 1
    assert highs[0] == 101.0
    assert lows[0] == 99.0
    assert high_indices[0] == 2
    assert low_indices[0] == 3

    decisions = np.array([100, 101, 132, 193, 194], dtype=np.int64)
    assert bridge_group_ordinals(decisions, 0).tolist() == [0, 0, 1, 2, 2]
    assert bridge_group_ordinals(decisions, 30).tolist() == [0, 0, 0, 1, 1]
    assert bridge_group_ordinals(decisions, 60).tolist() == [0, 0, 0, 0, 0]

    entries = decisions + 1
    path_ends = entries + 60
    reference = overlap_reference_ordinals(entries, path_ends)
    equivalent = bridge_group_ordinals(decisions, 59)
    assert np.array_equal(reference, equivalent)

    positive = pd.DataFrame(
        {
            "symbol": "BTCUSDT",
            "decision_date": "1970-01-01",
            "horizon_seconds": 60,
            "decision_second": decisions,
            "entry_second": entries,
            "path_end_second": path_ends,
            "entry_price": 100.0,
            "future_high_price": 101.0,
            "future_low_price": 99.0,
            "future_terminal_price": 100.5,
            "future_high_second": entries + 20,
            "future_low_second": entries + 40,
            "future_swing_bps": 200.0,
            "future_upside_bps": 100.0,
            "future_downside_bps": 100.0,
            "future_terminal_return_bps": 50.0,
            "future_high_first": True,
        }
    )
    events = build_events(
        positive,
        "H01M_T050BPS",
        "BTCUSDT",
        60,
        50.0,
        SegmentationSpec("bridge_030s", "BRIDGE_GAP", 30),
        0.75,
    )
    assert len(events) == 2
    assert int(events["number_of_positive_anchors"].sum()) == len(positive)
    assert int(events.iloc[0]["maximum_internal_inactive_gap_seconds"]) == 30

    comparison = pd.DataFrame(
        {
            "target_id": ["X"] * 4,
            "horizon_minutes": [15] * 4,
            "threshold_bps": [50.0] * 4,
            "segmentation_rule_type": ["BRIDGE_GAP"] * 4,
            "bridge_gap_seconds": [0, 30, 60, 300],
            "event_count": [1000, 900, 875, 600],
        }
    )
    transitions = build_transition_profile(comparison)
    stable = build_stable_ranges(transitions, comparison)
    assert int(stable.iloc[0]["candidate_strict_bridge_seconds"]) == 30
    assert int(stable.iloc[0]["candidate_primary_bridge_seconds"]) == 60

    print("SELF-TEST PASS")
    print(
        "Validated target-grid parsing, exact forward extrema, bridge grouping, "
        "overlap equivalence, event aggregation and stable-gap profiling."
    )


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Profile expansion thresholds, horizons and event grouping gaps."
    )
    value.add_argument(
        "--feature-root",
        default="./data/derived/perps/flow_features_1s_v1",
        help="Root of the audited perps flow-feature dataset.",
    )
    value.add_argument(
        "--output-root",
        default="./data/derived/perps/research/expansion_target_segmentation_sensitivity_v1",
        help="Research-label and event-sensitivity dataset root.",
    )
    value.add_argument(
        "--report-root",
        default="./reports/perps_expansion_target_segmentation_sensitivity",
        help="CSV, JSON and Markdown report root.",
    )
    value.add_argument("--symbol", default="BTCUSDT")
    value.add_argument(
        "--target-grid",
        default=DEFAULT_TARGET_GRID,
        help=(
            "Semicolon-separated MINUTES=THRESHOLD|THRESHOLD groups. "
            f"Default: {DEFAULT_TARGET_GRID}"
        ),
    )
    value.add_argument(
        "--bridge-gaps",
        default=",".join(str(value) for value in DEFAULT_BRIDGE_GAPS),
        help="Comma-separated inactive-gap candidates in seconds.",
    )
    value.add_argument("--start-date", help="Optional inclusive YYYY-MM-DD filter.")
    value.add_argument("--end-date", help="Optional inclusive YYYY-MM-DD filter.")
    value.add_argument(
        "--max-dates",
        type=int,
        help="Optional date limit after filtering; useful for isolated smoke tests.",
    )
    value.add_argument(
        "--expected-feature-builder-version",
        default=EXPECTED_FEATURE_BUILDER_VERSION,
        help="Expected upstream feature builder version; empty disables the check.",
    )
    value.add_argument(
        "--trend-dominance-ratio",
        type=float,
        default=DEFAULT_TREND_DOMINANCE_RATIO,
    )
    value.add_argument("--min-viable-events", type=int, default=DEFAULT_MIN_VIABLE_EVENTS)
    value.add_argument(
        "--min-viable-isolated-events",
        type=int,
        default=DEFAULT_MIN_VIABLE_ISOLATED_EVENTS,
    )
    value.add_argument(
        "--min-positive-rate-pct",
        type=float,
        default=DEFAULT_MIN_POSITIVE_RATE_PCT,
    )
    value.add_argument(
        "--max-positive-rate-pct",
        type=float,
        default=DEFAULT_MAX_POSITIVE_RATE_PCT,
    )
    value.add_argument(
        "--hash-input-files",
        action="store_true",
        help="Include full SHA-256 hashes in input change detection.",
    )
    value.add_argument(
        "--force",
        action="store_true",
        help="Rebuild selected horizon-label partitions even if unchanged.",
    )
    value.add_argument("--compression-level", type=int, default=6)
    value.add_argument(
        "--self-test",
        action="store_true",
        help="Run pure deterministic unit tests without reading Parquet inputs.",
    )
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    if not 0.5 <= args.trend_dominance_ratio <= 1.0:
        raise SystemExit("--trend-dominance-ratio must be between 0.5 and 1.0")
    if args.min_viable_events <= 0 or args.min_viable_isolated_events <= 0:
        raise SystemExit("Viability event thresholds must be positive")
    if not 0.0 <= args.min_positive_rate_pct < args.max_positive_rate_pct <= 100.0:
        raise SystemExit("Positive-rate bounds must satisfy 0 <= min < max <= 100")
    if not 1 <= args.compression_level <= 22:
        raise SystemExit("--compression-level must be between 1 and 22")

    target_grid = parse_target_grid(args.target_grid)
    bridge_gaps = parse_bridge_gaps(args.bridge_gaps)
    specs = segmentation_specs(bridge_gaps)
    horizons_seconds = tuple(minutes * 60 for minutes in target_grid)
    target_catalog: dict[str, dict[str, Any]] = {}
    for minutes, thresholds in target_grid.items():
        for threshold in thresholds:
            identifier = target_id(minutes, threshold)
            target_catalog[identifier] = {
                "target_id": identifier,
                "horizon_minutes": minutes,
                "horizon_seconds": minutes * 60,
                "threshold_bps": threshold,
            }

    feature_root = Path(args.feature_root).expanduser().resolve()
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

    configuration = {
        "profiler_version": PROFILER_VERSION,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "feature_root": str(feature_root),
        "output_root": str(output_root),
        "report_root": str(report_root),
        "symbol": symbol,
        "target_grid": target_grid,
        "bridge_gaps": bridge_gaps,
        "include_overlap_reference": True,
        "trend_dominance_ratio": args.trend_dominance_ratio,
        "expected_feature_builder_version": args.expected_feature_builder_version,
        "expected_feature_schema_version": EXPECTED_FEATURE_SCHEMA_VERSION,
        "hash_input_files": bool(args.hash_input_files),
        "start_date": start_date.isoformat() if start_date else None,
        "end_date": end_date.isoformat() if end_date else None,
        "max_dates": args.max_dates,
        "viability": {
            "min_events": args.min_viable_events,
            "min_isolated_events": args.min_viable_isolated_events,
            "min_positive_rate_pct": args.min_positive_rate_pct,
            "max_positive_rate_pct": args.max_positive_rate_pct,
        },
    }
    label_logic_signature = sha256_text(
        compact_json(
            {
                "profiler_version": PROFILER_VERSION,
                "label_schema_version": LABEL_SCHEMA_VERSION,
                "causal_contract": "decision_t_entry_t_plus_1_exact_path_h_plus_1",
                "expected_feature_schema_version": EXPECTED_FEATURE_SCHEMA_VERSION,
            }
        )
    )

    connection = connect_state(state_path)
    connection.execute(
        "INSERT INTO runs (run_id, started_at_utc, status, configuration_json) "
        "VALUES (?, ?, 'RUNNING', ?)",
        (run_id, utc_now_iso(), json.dumps(configuration, sort_keys=True)),
    )
    connection.commit()

    rebuilt_partitions = 0
    skipped_partitions = 0
    failed_partitions = 0
    label_summaries: list[dict[str, Any]] = []
    invariant_rows: list[dict[str, Any]] = []

    try:
        print("Discovering audited flow-feature partitions...")
        partitions = discover_feature_partitions(
            feature_root,
            symbol,
            args.expected_feature_builder_version,
            args.hash_input_files,
        )
        dates = selected_dates(partitions, start_date, end_date, args.max_dates)
        record_inputs(connection, symbol, run_id, partitions)
        print(f"Input dates       : {len(partitions):,}")
        print(f"Selected dates    : {len(dates):,}")
        print(f"Target horizons   : {len(horizons_seconds):,}")
        print(f"Target definitions: {len(target_catalog):,}")
        print(f"Segmentations     : {len(specs):,} per target")
        print(f"Output root       : {output_root}")

        for date_index, day in enumerate(dates, start=1):
            plans: list[tuple[int, str, Path, bool, sqlite3.Row | None]] = []
            for horizon_seconds in horizons_seconds:
                dependency = dependency_signature(
                    day, horizon_seconds, partitions, label_logic_signature
                )
                path = label_output_path(output_root, symbol, day, horizon_seconds)
                state = get_label_state(connection, symbol, day, horizon_seconds)
                unchanged = (
                    not args.force
                    and state is not None
                    and state["status"] == "COMPLETED"
                    and state["dependency_signature"] == dependency
                    and Path(state["output_path"]).exists()
                )
                if unchanged:
                    try:
                        validate_written_label_file(
                            Path(state["output_path"]),
                            int(state["output_rows"]),
                            dependency,
                            day,
                            horizon_seconds,
                        )
                    except Exception:
                        unchanged = False
                plans.append((horizon_seconds, dependency, path, unchanged, state))

            if all(plan[3] for plan in plans):
                skipped_partitions += len(plans)
                print(
                    f"SKIP DATE [{date_index:>3}/{len(dates)}] {day}: "
                    f"all {len(plans)} horizons unchanged"
                )
                for horizon_seconds, _, _, _, state in plans:
                    assert state is not None
                    label_summaries.append(
                        {
                            "decision_date": day.isoformat(),
                            "horizon_seconds": horizon_seconds,
                            "status": "SKIPPED_UNCHANGED",
                            "valid_label_rows": int(state["output_rows"]),
                        }
                    )
                continue

            local = load_local_data(day, partitions, symbol)
            for horizon_seconds, dependency, path, unchanged, state in plans:
                if unchanged:
                    skipped_partitions += 1
                    assert state is not None
                    label_summaries.append(
                        {
                            "decision_date": day.isoformat(),
                            "horizon_seconds": horizon_seconds,
                            "status": "SKIPPED_UNCHANGED",
                            "valid_label_rows": int(state["output_rows"]),
                        }
                    )
                    continue
                mark_label_running(
                    connection,
                    symbol,
                    day,
                    horizon_seconds,
                    dependency,
                    path,
                )
                try:
                    frame, summary, checks = build_horizon_label_frame(
                        day, local, symbol, horizon_seconds
                    )
                    metadata = {
                        "perps_expansion_target_profiler_version": PROFILER_VERSION,
                        "perps_expansion_horizon_label_schema_version": LABEL_SCHEMA_VERSION,
                        "source_feature_builder_version": partitions[day].feature_builder_version,
                        "source_feature_schema_version": EXPECTED_FEATURE_SCHEMA_VERSION,
                        "symbol": symbol,
                        "decision_date": day.isoformat(),
                        "horizon_seconds": horizon_seconds,
                        "path_observations": horizon_seconds + 1,
                        "causal_entry_contract": "decision_t_entry_t_plus_1_exact_path_h_plus_1",
                        "dependency_signature": dependency,
                        "created_at_utc": utc_now_iso(),
                    }
                    atomic_write_parquet_frame(
                        path, frame, metadata, args.compression_level
                    )
                    cleanup_extra_parquet_files(path.parent, path)
                    validate_written_label_file(
                        path, len(frame), dependency, day, horizon_seconds
                    )
                    minimum_second = (
                        int(frame["decision_second"].min()) if len(frame) else None
                    )
                    maximum_second = (
                        int(frame["decision_second"].max()) if len(frame) else None
                    )
                    mark_label_completed(
                        connection,
                        symbol,
                        day,
                        horizon_seconds,
                        dependency,
                        path,
                        len(frame),
                        minimum_second,
                        maximum_second,
                    )
                    rebuilt_partitions += 1
                    summary["status"] = "REBUILT"
                    label_summaries.append(summary)
                    invariant_rows.extend(checks)
                    print(
                        f"BUILD     [{date_index:>3}/{len(dates)}] {day} "
                        f"H={horizon_seconds // 60:>2}m rows={len(frame):,}"
                    )
                except Exception as exc:
                    failed_partitions += 1
                    mark_label_failed(
                        connection,
                        symbol,
                        day,
                        horizon_seconds,
                        traceback.format_exc(),
                    )
                    print(
                        f"FAILED    {day} H={horizon_seconds // 60}m: {exc}",
                        file=sys.stderr,
                    )
                    raise

        _, pq = require_arrow()
        prevalence_rows: list[dict[str, Any]] = []
        comparison_rows: list[dict[str, Any]] = []
        duration_rows: list[dict[str, Any]] = []
        daily_event_profile_rows: list[dict[str, Any]] = []
        longest_rows: list[pd.DataFrame] = []
        positive_seconds_by_target: dict[str, np.ndarray] = {}
        valid_seconds_by_horizon: dict[int, np.ndarray] = {}

        targets_by_horizon: dict[int, list[dict[str, Any]]] = {}
        for item in target_catalog.values():
            targets_by_horizon.setdefault(int(item["horizon_seconds"]), []).append(item)

        print("Profiling targets and segmentations...")
        for horizon_seconds in horizons_seconds:
            horizon_targets = sorted(
                targets_by_horizon[horizon_seconds], key=lambda row: row["threshold_bps"]
            )
            valid_arrays: list[np.ndarray] = []
            positive_frames: dict[str, list[pd.DataFrame]] = {
                str(item["target_id"]): [] for item in horizon_targets
            }
            daily_by_target: dict[str, list[dict[str, Any]]] = {
                str(item["target_id"]): [] for item in horizon_targets
            }

            for day in dates:
                path = label_output_path(output_root, symbol, day, horizon_seconds)
                parquet_file = pq.ParquetFile(path)
                try:
                    frame = parquet_file.read(columns=list(LABEL_COLUMNS)).to_pandas()
                finally:
                    close = getattr(parquet_file, "close", None)
                    if callable(close):
                        close()
                if len(frame):
                    frame["decision_second"] = pd.to_numeric(
                        frame["decision_second"], errors="raise"
                    ).astype(np.int64)
                    frame["future_swing_bps"] = pd.to_numeric(
                        frame["future_swing_bps"], errors="raise"
                    ).astype(float)
                    frame = frame.sort_values("decision_second", kind="mergesort")
                    valid_arrays.append(frame["decision_second"].to_numpy(dtype=np.int64))
                else:
                    valid_arrays.append(np.array([], dtype=np.int64))
                for target_spec in horizon_targets:
                    target = str(target_spec["target_id"])
                    threshold = float(target_spec["threshold_bps"])
                    positive = frame[
                        frame["future_swing_bps"].to_numpy(dtype=float) + 1e-10
                        >= threshold
                    ].copy()
                    if len(positive):
                        positive_frames[target].append(positive)
                    daily_by_target[target].append(
                        {
                            "decision_date": day.isoformat(),
                            "valid_label_rows": len(frame),
                            "positive_rows": len(positive),
                            "positive_rate_pct": percentage(len(positive), len(frame)),
                        }
                    )

            valid_seconds = (
                np.concatenate(valid_arrays).astype(np.int64, copy=False)
                if valid_arrays
                else np.array([], dtype=np.int64)
            )
            if len(valid_seconds) and not np.all(np.diff(valid_seconds) > 0):
                raise RuntimeError(
                    f"Horizon {horizon_seconds}s valid seconds are not globally sorted and unique"
                )
            valid_seconds_by_horizon[horizon_seconds] = valid_seconds

            for target_spec in horizon_targets:
                target = str(target_spec["target_id"])
                threshold = float(target_spec["threshold_bps"])
                positive = (
                    pd.concat(positive_frames[target], ignore_index=True)
                    if positive_frames[target]
                    else pd.DataFrame(columns=LABEL_COLUMNS)
                )
                positive = positive.sort_values(
                    "decision_second", kind="mergesort"
                ).reset_index(drop=True)
                positive_seconds = (
                    positive["decision_second"].to_numpy(dtype=np.int64)
                    if len(positive)
                    else np.array([], dtype=np.int64)
                )
                if len(positive_seconds) and not np.all(np.diff(positive_seconds) > 0):
                    raise RuntimeError(f"Positive seconds are not unique for {target}")
                positive_seconds_by_target[target] = positive_seconds
                daily_target = pd.DataFrame(daily_by_target[target])
                prevalence_rows.append(
                    target_prevalence_row(
                        target,
                        horizon_seconds,
                        threshold,
                        daily_target,
                    )
                )

                reference_ordinals = (
                    overlap_reference_ordinals(
                        positive["entry_second"].to_numpy(dtype=np.int64),
                        positive["path_end_second"].to_numpy(dtype=np.int64),
                    )
                    if len(positive)
                    else np.array([], dtype=np.int64)
                )
                equivalent_ordinals = (
                    bridge_group_ordinals(positive_seconds, horizon_seconds - 1)
                    if len(positive)
                    else np.array([], dtype=np.int64)
                )
                invariant_rows.append(
                    invariant_record(
                        "TARGET",
                        "OVERLAP_REFERENCE_EQUIVALENCE",
                        bool(np.array_equal(reference_ordinals, equivalent_ordinals)),
                        f"equivalent_bridge_gap={horizon_seconds - 1}",
                        target,
                        "overlap_reference",
                    )
                )

                for spec in specs:
                    events = build_events(
                        positive,
                        target,
                        symbol,
                        horizon_seconds,
                        threshold,
                        spec,
                        args.trend_dominance_ratio,
                    )
                    invariant_rows.extend(
                        validate_events(
                            events, positive, target, spec, horizon_seconds
                        )
                    )
                    event_path = event_output_path(
                        output_root, symbol, target, spec.segmentation_id
                    )
                    event_metadata = {
                        "perps_expansion_target_profiler_version": PROFILER_VERSION,
                        "perps_expansion_event_schema_version": EVENT_SCHEMA_VERSION,
                        "target_id": target,
                        "horizon_seconds": horizon_seconds,
                        "threshold_bps": threshold,
                        "segmentation_id": spec.segmentation_id,
                        "segmentation_rule_type": spec.rule_type,
                        "bridge_gap_seconds": spec.bridge_gap_seconds,
                        "source_label_logic_signature": label_logic_signature,
                        "created_at_utc": utc_now_iso(),
                    }
                    atomic_write_parquet_frame(
                        event_path,
                        events,
                        event_metadata,
                        args.compression_level,
                    )
                    cleanup_extra_parquet_files(event_path.parent, event_path)
                    comparison_rows.append(
                        segmentation_comparison_row(
                            target,
                            horizon_seconds,
                            threshold,
                            spec,
                            events,
                            len(positive),
                            len(dates),
                        )
                    )
                    duration_rows.extend(
                        duration_distribution_rows(target, spec, events)
                    )
                    daily_event_profile_rows.extend(
                        daily_event_rows(target, spec, events)
                    )
                    if len(events):
                        longest = events.nlargest(
                            min(10, len(events)), "event_duration_seconds"
                        ).copy()
                        longest_rows.append(
                            longest[
                                [
                                    "target_id",
                                    "segmentation_id",
                                    "event_id",
                                    "event_start_timestamp_utc",
                                    "event_end_timestamp_utc",
                                    "event_duration_seconds",
                                    "number_of_positive_anchors",
                                    "positive_anchor_coverage_pct",
                                    "maximum_future_swing_bps",
                                    "isolated_onset",
                                ]
                            ]
                        )

        prevalence = pd.DataFrame(prevalence_rows).sort_values(
            ["horizon_minutes", "threshold_bps"], kind="mergesort"
        )
        comparison = pd.DataFrame(comparison_rows).sort_values(
            ["horizon_minutes", "threshold_bps", "segmentation_rule_type", "bridge_gap_seconds"],
            kind="mergesort",
            na_position="last",
        )
        transition = build_transition_profile(comparison)
        stable_ranges = build_stable_ranges(transition, comparison)
        viability = build_viability_profile(
            prevalence,
            comparison,
            stable_ranges,
            args.min_viable_events,
            args.min_viable_isolated_events,
            args.min_positive_rate_pct,
            args.max_positive_rate_pct,
        )
        duration_distribution = pd.DataFrame(duration_rows)
        daily_event_profile = pd.DataFrame(daily_event_profile_rows)
        target_overlap = build_target_overlap(
            target_catalog,
            valid_seconds_by_horizon,
            positive_seconds_by_target,
        )
        top_longest = (
            pd.concat(longest_rows, ignore_index=True)
            if longest_rows
            else pd.DataFrame(
                columns=[
                    "target_id",
                    "segmentation_id",
                    "event_id",
                    "event_start_timestamp_utc",
                    "event_end_timestamp_utc",
                    "event_duration_seconds",
                    "number_of_positive_anchors",
                    "positive_anchor_coverage_pct",
                    "maximum_future_swing_bps",
                    "isolated_onset",
                ]
            )
        )

        bridge_counts = comparison[
            comparison["segmentation_rule_type"] == "BRIDGE_GAP"
        ].copy()
        for target, group in bridge_counts.groupby("target_id", sort=False):
            ordered = group.sort_values("bridge_gap_seconds")
            counts = ordered["event_count"].to_numpy(dtype=np.int64)
            invariant_rows.append(
                invariant_record(
                    "TARGET",
                    "EVENT_COUNT_NONINCREASING_WITH_BRIDGE",
                    bool(np.all(np.diff(counts) <= 0)),
                    f"counts={counts.tolist()}",
                    target,
                    None,
                )
            )

        invariants = pd.DataFrame(invariant_rows)
        if invariants.empty:
            invariants = pd.DataFrame(
                columns=[
                    "scope",
                    "target_id",
                    "segmentation_id",
                    "check",
                    "passed",
                    "details",
                ]
            )
        failed_invariants = int((~invariants["passed"].astype(bool)).sum())
        status = "COMPLETED" if failed_invariants == 0 else "FAILED_VALIDATION"
        summary = {
            "run_id": run_id,
            "status": status,
            "profiler_version": PROFILER_VERSION,
            "started_at_utc": connection.execute(
                "SELECT started_at_utc FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()[0],
            "completed_at_utc": utc_now_iso(),
            "symbol": symbol,
            "selected_dates": len(dates),
            "first_date": dates[0].isoformat(),
            "last_date": dates[-1].isoformat(),
            "horizon_count": len(horizons_seconds),
            "target_count": len(target_catalog),
            "segmentation_count": len(specs),
            "rebuilt_label_partitions": rebuilt_partitions,
            "skipped_label_partitions": skipped_partitions,
            "failed_label_partitions": failed_partitions,
            "invariant_checks": len(invariants),
            "failed_invariants": failed_invariants,
            "baseline_target_id": "H15M_T050BPS",
        }

        report_root.mkdir(parents=True, exist_ok=True)
        atomic_write_csv(report_root / REPORT_FILENAMES[0], prevalence)
        atomic_write_csv(report_root / REPORT_FILENAMES[1], comparison)
        atomic_write_csv(report_root / REPORT_FILENAMES[2], transition)
        atomic_write_csv(report_root / REPORT_FILENAMES[3], stable_ranges)
        atomic_write_csv(report_root / REPORT_FILENAMES[4], viability)
        atomic_write_csv(report_root / REPORT_FILENAMES[5], duration_distribution)
        atomic_write_csv(report_root / REPORT_FILENAMES[6], daily_event_profile)
        atomic_write_csv(report_root / REPORT_FILENAMES[7], target_overlap)
        atomic_write_csv(report_root / REPORT_FILENAMES[8], top_longest)
        atomic_write_csv(report_root / REPORT_FILENAMES[9], invariants)
        atomic_write_json(report_root / REPORT_FILENAMES[10], summary)
        atomic_write_json(report_root / REPORT_FILENAMES[11], configuration)
        findings = markdown_findings(
            summary,
            prevalence,
            stable_ranges,
            viability,
            "H15M_T050BPS",
        )
        atomic_write_text(report_root / REPORT_FILENAMES[12], findings)

        run_report_root = report_root / "runs" / run_id
        run_report_root.mkdir(parents=True, exist_ok=True)
        for filename in REPORT_FILENAMES:
            source = report_root / filename
            if source.exists():
                shutil.copy2(source, run_report_root / filename)

        metadata_root = output_root / "_metadata"
        atomic_write_json(metadata_root / "dataset_config.json", configuration)
        atomic_write_json(metadata_root / "latest_run_summary.json", summary)

        connection.execute(
            """
            UPDATE runs SET
                completed_at_utc=?, status=?, rebuilt_partitions=?,
                skipped_partitions=?, failed_partitions=?, error_text=NULL
            WHERE run_id=?
            """,
            (
                summary["completed_at_utc"],
                status,
                rebuilt_partitions,
                skipped_partitions,
                failed_partitions,
                run_id,
            ),
        )
        connection.commit()

        print("\nBuild summary")
        print(f"Status             : {status}")
        print(f"Targets            : {len(target_catalog):,}")
        print(f"Segmentations      : {len(specs):,} per target")
        print(f"Label partitions   : rebuilt={rebuilt_partitions:,}, skipped={skipped_partitions:,}")
        print(f"Failed invariants  : {failed_invariants:,}")
        print(f"Reports            : {report_root}")
        if failed_invariants:
            return 2
        return 0
    except Exception:
        connection.execute(
            """
            UPDATE runs SET completed_at_utc=?, status='FAILED',
                rebuilt_partitions=?, skipped_partitions=?,
                failed_partitions=?, error_text=?
            WHERE run_id=?
            """,
            (
                utc_now_iso(),
                rebuilt_partitions,
                skipped_partitions,
                max(1, failed_partitions),
                traceback.format_exc(),
                run_id,
            ),
        )
        connection.commit()
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
