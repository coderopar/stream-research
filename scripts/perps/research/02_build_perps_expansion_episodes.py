#!/usr/bin/env python
"""
Build exact 15-minute / 50-bps perps expansion labels and distinct episodes.

Research contract
-----------------
For a decision row at second ``t``:

* entry price is the exact feature price at ``t + 1``;
* the future path is every exact second from ``t + 1`` through ``t + 901``
  inclusive (901 observations spanning exactly 900 elapsed seconds);
* ``future_swing_15m_bps`` is ``(future_high - future_low) / entry * 10,000``;
* ``expansion_50_15m`` is true when that exact path swing is at least 50 bps;
* rows are emitted only when the decision row is context-ready and the entire
  future path exists with finite, positive prices. Missing seconds are never
  filled or carried.

Episode rule
------------
Every positive decision row owns a future interval ``[t + 1, t + 901]``. Positive
rows are sorted by decision second and clustered as connected components of
those closed intervals. A row joins the current episode when its interval start
is less than or equal to the current component's maximum interval end; otherwise
it starts a new episode. The rule is transitive, deterministic, symbol-local and
does not reset at UTC midnight.

The row-level labels remain in a separate research dataset. Global episode IDs
are written to a separate positive-anchor membership file so partition-level
label rebuilding does not rewrite causal feature files or unchanged label dates.

Resumability and cross-day dependency
-------------------------------------
A decision-date label partition depends on the feature partition for that UTC
calendar date and, when present, the immediately following UTC date. Therefore,
a changed date automatically invalidates its own labels and the previous day's
cross-midnight labels. Row partitions are rebuilt independently and atomically.
Episode, membership and report outputs are global cross-day products and are
rebuilt whenever any row-label partition changes or disappears.

Expected dependencies
---------------------
    pip install numpy pandas pyarrow numba

Typical Git Bash usage
----------------------
    python scripts/perps/research/02_build_perps_expansion_episodes.py
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
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - runtime environment check
    raise SystemExit(
        "Missing dependency 'pyarrow'. Activate the project virtual environment "
        "and run: pip install numpy pandas pyarrow numba"
    ) from exc

try:
    from numba import njit, prange
except ImportError as exc:  # pragma: no cover - runtime environment check
    raise SystemExit(
        "Missing dependency 'numba'. Activate the project virtual environment "
        "and run: pip install numpy pandas pyarrow numba"
    ) from exc


BUILDER_VERSION = "1.0.0"
LABEL_SCHEMA_VERSION = 1
EPISODE_SCHEMA_VERSION = 1
EXPECTED_FEATURE_SCHEMA_VERSION = 1
EXPECTED_FEATURE_BUILDER_VERSION = "1.0.2"
HORIZON_SECONDS = 15 * 60
PATH_OBSERVATIONS = HORIZON_SECONDS + 1
THRESHOLD_BPS = 50.0
DEFAULT_TREND_DOMINANCE_RATIO = 0.75
SECONDS_PER_DAY = 86_400
DATE_PREFIX = "date="
SYMBOL_PREFIX = "symbol="
EPISODE_RULE_VERSION = "overlapping_future_intervals_v1"

REQUIRED_FEATURE_COLUMNS = (
    "feature_schema_version",
    "symbol",
    "receive_second",
    "price_for_features",
    "flow_feature_context_ready",
)

LABEL_COLUMNS = (
    "label_schema_version",
    "label_builder_version",
    "symbol",
    "decision_date",
    "decision_second",
    "decision_timestamp_utc",
    "entry_second",
    "entry_timestamp_utc",
    "path_end_second",
    "path_end_timestamp_utc",
    "entry_price",
    "future_high_price",
    "future_low_price",
    "future_terminal_price",
    "future_high_second",
    "future_low_second",
    "future_high_timestamp_utc",
    "future_low_timestamp_utc",
    "future_swing_15m_bps",
    "expansion_50_15m",
    "future_upside_15m_bps",
    "future_downside_15m_bps",
    "future_terminal_return_15m_bps",
    "future_high_first",
    "seconds_to_future_high",
    "seconds_to_future_low",
    "seconds_to_50bps_realization",
    "first_50bps_realization_second",
    "first_50bps_realization_timestamp_utc",
)

POSITIVE_COLUMNS = (
    "symbol",
    "decision_date",
    "decision_second",
    "entry_second",
    "path_end_second",
    "future_swing_15m_bps",
    "future_upside_15m_bps",
    "future_downside_15m_bps",
    "future_terminal_return_15m_bps",
    "future_high_first",
    "future_high_second",
    "future_low_second",
    "seconds_to_50bps_realization",
    "first_50bps_realization_second",
)


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat().replace("+00:00", "Z")


def epoch_second_for_date(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())


def iso_from_second(value: int | float | None) -> str | None:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return datetime.fromtimestamp(int(value), tz=timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def date_from_second(value: int) -> date:
    return datetime.fromtimestamp(int(value), tz=timezone.utc).date()


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


def safe_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if len(array) else float("nan")


def safe_median(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if len(array) else float("nan")


def safe_quantile(values: Iterable[float], q: float) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, q, method="linear")) if len(array) else float("nan")


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_write_parquet(
    path: Path,
    table: pa.Table,
    compression: str = "zstd",
    compression_level: int = 6,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        pq.write_table(
            table,
            temporary,
            compression=compression,
            compression_level=compression_level,
            use_dictionary=True,
            write_statistics=True,
        )
        check = pq.ParquetFile(temporary)
        try:
            if check.metadata.num_rows != table.num_rows:
                raise RuntimeError(
                    f"Atomic Parquet validation failed for {path}: "
                    f"expected {table.num_rows:,}, wrote {check.metadata.num_rows:,}"
                )
        finally:
            close = getattr(check, "close", None)
            if callable(close):
                close()
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def decode_metadata(schema: pa.Schema) -> dict[str, str]:
    return {
        key.decode("utf-8", errors="replace"): value.decode(
            "utf-8", errors="replace"
        )
        for key, value in (schema.metadata or {}).items()
    }


def metadata_bytes(values: Mapping[str, Any]) -> dict[bytes, bytes]:
    return {
        str(key).encode("utf-8"): str(value).encode("utf-8")
        for key, value in values.items()
    }


def clean_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def timestamp_array_from_seconds(values: Sequence[int | None] | np.ndarray) -> pa.Array:
    """Build a nullable UTC second-resolution timestamp array from epoch seconds."""
    integers = pa.array(values, type=pa.int64())
    return integers.cast(pa.timestamp("s", tz="UTC"))


def cleanup_extra_parquet_files(directory: Path, keep: Path) -> None:
    for candidate in directory.glob("*.parquet"):
        if candidate.resolve() != keep.resolve():
            candidate.unlink()


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


def parquet_min_max_int(parquet_file: pq.ParquetFile, column_name: str) -> tuple[int, int]:
    index = parquet_file.schema_arrow.get_field_index(column_name)
    if index < 0:
        raise RuntimeError(f"Required column is missing: {column_name}")
    minima: list[int] = []
    maxima: list[int] = []
    for row_group_index in range(parquet_file.num_row_groups):
        column = parquet_file.metadata.row_group(row_group_index).column(index)
        stats = column.statistics
        if stats is None or not stats.has_min_max:
            minima = []
            maxima = []
            break
        minima.append(int(stats.min))
        maxima.append(int(stats.max))
    if minima:
        return min(minima), max(maxima)
    values = parquet_file.read(columns=[column_name]).column(column_name).to_numpy(
        zero_copy_only=False
    )
    if not len(values):
        raise RuntimeError(f"No values found in {column_name}")
    return int(np.min(values)), int(np.max(values))


def discover_partitions(
    feature_root: Path,
    symbol: str,
    expected_builder_version: str,
    hash_input_files: bool,
) -> dict[date, InputPartition]:
    symbol_root = feature_root / f"{SYMBOL_PREFIX}{symbol}"
    if not symbol_root.exists():
        raise SystemExit(f"Feature symbol directory does not exist: {symbol_root}")

    output: dict[date, InputPartition] = {}
    for date_dir in sorted(symbol_root.glob(f"{DATE_PREFIX}*")):
        if not date_dir.is_dir():
            continue
        try:
            receive_date = date.fromisoformat(date_dir.name[len(DATE_PREFIX) :])
        except ValueError:
            continue
        files = sorted(path for path in date_dir.glob("*.parquet") if path.is_file())
        if len(files) != 1:
            raise SystemExit(
                f"Expected exactly one feature Parquet file for {receive_date}, "
                f"found {len(files)} in {date_dir}"
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
            schema_version = (
                int(raw_schema_version) if raw_schema_version is not None else None
            )
            if expected_builder_version and builder_version != expected_builder_version:
                raise SystemExit(
                    f"{path} builder version is {builder_version!r}; expected "
                    f"{expected_builder_version!r}"
                )
            if schema_version != EXPECTED_FEATURE_SCHEMA_VERSION:
                raise SystemExit(
                    f"{path} feature schema version is {schema_version!r}; expected "
                    f"{EXPECTED_FEATURE_SCHEMA_VERSION}"
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
        signature_payload = {
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
            signature=sha256_text(compact_json(signature_payload)),
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
        raise SystemExit("No input dates matched the requested date range")
    return values


def dependency_signature(
    day: date,
    partitions: Mapping[date, InputPartition],
    label_config_signature: str,
) -> str:
    dependencies = [partitions[day]]
    next_partition = partitions.get(day + timedelta(days=1))
    if next_partition is not None:
        dependencies.append(next_partition)
    return sha256_text(
        compact_json(
            {
                "decision_date": day.isoformat(),
                "label_config_signature": label_config_signature,
                "dependencies": [
                    {
                        "receive_date": item.receive_date.isoformat(),
                        "signature": item.signature,
                    }
                    for item in dependencies
                ],
            }
        )
    )


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
            rebuilt_dates INTEGER NOT NULL DEFAULT 0,
            skipped_dates INTEGER NOT NULL DEFAULT 0,
            failed_dates INTEGER NOT NULL DEFAULT 0,
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
            PRIMARY KEY (symbol, receive_date)
        );

        CREATE TABLE IF NOT EXISTS label_partitions (
            symbol TEXT NOT NULL,
            decision_date TEXT NOT NULL,
            dependency_signature TEXT NOT NULL,
            output_path TEXT NOT NULL,
            output_rows INTEGER NOT NULL DEFAULT 0,
            positive_rows INTEGER NOT NULL DEFAULT 0,
            minimum_decision_second INTEGER,
            maximum_decision_second INTEGER,
            status TEXT NOT NULL,
            started_at_utc TEXT,
            completed_at_utc TEXT,
            error_text TEXT,
            PRIMARY KEY (symbol, decision_date)
        );

        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
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
                symbol, receive_date, path, signature, size_bytes, modified_time_ns,
                rows, minimum_second, maximum_second, feature_builder_version,
                feature_schema_version, content_sha256, last_seen_run_id
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


def label_state(
    connection: sqlite3.Connection, symbol: str, day: date
) -> sqlite3.Row | None:
    return connection.execute(
        "SELECT * FROM label_partitions WHERE symbol=? AND decision_date=?",
        (symbol, day.isoformat()),
    ).fetchone()


def mark_label_running(
    connection: sqlite3.Connection,
    symbol: str,
    day: date,
    dependency: str,
    output_path: Path,
) -> None:
    connection.execute(
        """
        INSERT INTO label_partitions (
            symbol, decision_date, dependency_signature, output_path, status,
            started_at_utc, completed_at_utc, error_text
        ) VALUES (?, ?, ?, ?, 'RUNNING', ?, NULL, NULL)
        ON CONFLICT(symbol, decision_date) DO UPDATE SET
            dependency_signature=excluded.dependency_signature,
            output_path=excluded.output_path,
            output_rows=0,
            positive_rows=0,
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
    dependency: str,
    output_path: Path,
    rows: int,
    positive_rows: int,
    minimum_second: int | None,
    maximum_second: int | None,
) -> None:
    connection.execute(
        """
        UPDATE label_partitions SET
            dependency_signature=?, output_path=?, output_rows=?, positive_rows=?,
            minimum_decision_second=?, maximum_decision_second=?, status='COMPLETED',
            completed_at_utc=?, error_text=NULL
        WHERE symbol=? AND decision_date=?
        """,
        (
            dependency,
            str(output_path),
            rows,
            positive_rows,
            minimum_second,
            maximum_second,
            utc_now_iso(),
            symbol,
            day.isoformat(),
        ),
    )
    connection.commit()


def mark_label_failed(
    connection: sqlite3.Connection,
    symbol: str,
    day: date,
    error_text: str,
) -> None:
    connection.execute(
        """
        UPDATE label_partitions SET status='FAILED', completed_at_utc=?, error_text=?
        WHERE symbol=? AND decision_date=?
        """,
        (utc_now_iso(), error_text, symbol, day.isoformat()),
    )
    connection.commit()


def completed_label_states(
    connection: sqlite3.Connection, symbol: str
) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT * FROM label_partitions
        WHERE symbol=? AND status='COMPLETED'
        ORDER BY decision_date
        """,
        (symbol,),
    ).fetchall()


# ---------------------------------------------------------------------------
# Exact forward calculations
# ---------------------------------------------------------------------------


@njit(cache=True)
def forward_extrema_first_indices(
    values: np.ndarray, valid: np.ndarray, window_length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """O(n) exact forward high/low and earliest occurrence index per window."""
    n = len(values)
    highs = np.full(n, np.nan, dtype=np.float64)
    lows = np.full(n, np.nan, dtype=np.float64)
    high_indices = np.full(n, -1, dtype=np.int64)
    low_indices = np.full(n, -1, dtype=np.int64)
    complete = np.zeros(n, dtype=np.uint8)

    max_queue = np.empty(n, dtype=np.int64)
    min_queue = np.empty(n, dtype=np.int64)
    max_head = 0
    max_tail = 0
    min_head = 0
    min_tail = 0
    invalid_count = 0

    for index in range(n):
        expired = index - window_length
        while max_head < max_tail and max_queue[max_head] <= expired:
            max_head += 1
        while min_head < min_tail and min_queue[min_head] <= expired:
            min_head += 1

        if not valid[index]:
            invalid_count += 1
        else:
            # Strict comparisons preserve the earliest index when values tie.
            while max_head < max_tail and values[max_queue[max_tail - 1]] < values[index]:
                max_tail -= 1
            max_queue[max_tail] = index
            max_tail += 1

            while min_head < min_tail and values[min_queue[min_tail - 1]] > values[index]:
                min_tail -= 1
            min_queue[min_tail] = index
            min_tail += 1

        if index >= window_length and not valid[index - window_length]:
            invalid_count -= 1

        if index >= window_length - 1:
            start = index - window_length + 1
            if invalid_count == 0 and max_head < max_tail and min_head < min_tail:
                max_index = max_queue[max_head]
                min_index = min_queue[min_head]
                highs[start] = values[max_index]
                lows[start] = values[min_index]
                high_indices[start] = max_index
                low_indices[start] = min_index
                complete[start] = 1

    return highs, lows, high_indices, low_indices, complete


@njit(cache=True, parallel=True)
def first_range_realization_offsets(
    values: np.ndarray,
    entry_positions: np.ndarray,
    horizon_seconds: int,
    threshold_bps: float,
) -> np.ndarray:
    """Earliest offset where running high-low range reaches the threshold."""
    output = np.full(len(entry_positions), -1, dtype=np.int32)
    for row_index in prange(len(entry_positions)):
        entry_position = entry_positions[row_index]
        entry_price = values[entry_position]
        running_high = entry_price
        running_low = entry_price
        for offset in range(horizon_seconds + 1):
            price = values[entry_position + offset]
            if price > running_high:
                running_high = price
            if price < running_low:
                running_low = price
            swing_bps = (running_high - running_low) / entry_price * 10_000.0
            if swing_bps + 1e-10 >= threshold_bps:
                output[row_index] = offset
                break
    return output


@dataclass
class DenseLocalData:
    base_second: int
    price: np.ndarray
    observed: np.ndarray
    ready: np.ndarray


def read_feature_partition(item: InputPartition, symbol: str) -> pd.DataFrame:
    parquet_file = pq.ParquetFile(item.path)
    try:
        table = parquet_file.read(columns=list(REQUIRED_FEATURE_COLUMNS))
    finally:
        close = getattr(parquet_file, "close", None)
        if callable(close):
            close()
    frame = table.to_pandas()
    if len(frame) != item.rows:
        raise RuntimeError(
            f"Feature row-count mismatch for {item.path}: metadata={item.rows:,}, "
            f"read={len(frame):,}"
        )
    versions = pd.to_numeric(frame["feature_schema_version"], errors="coerce")
    if versions.isna().any() or not bool(
        (versions == EXPECTED_FEATURE_SCHEMA_VERSION).all()
    ):
        raise RuntimeError(f"Unexpected feature_schema_version in {item.path}")
    symbols = frame["symbol"].astype(str)
    if not bool((symbols == symbol).all()):
        bad = sorted(symbols[symbols != symbol].unique().tolist())[:5]
        raise RuntimeError(f"Unexpected symbols in {item.path}: {bad}")
    frame = frame.copy()
    frame["receive_second"] = pd.to_numeric(
        frame["receive_second"], errors="raise"
    ).astype("int64")
    frame["price_for_features"] = pd.to_numeric(
        frame["price_for_features"], errors="coerce"
    ).astype(float)
    frame["flow_feature_context_ready"] = frame[
        "flow_feature_context_ready"
    ].fillna(False).astype(bool)
    seconds = frame["receive_second"].to_numpy(dtype=np.int64)
    if len(seconds) > 1 and np.any(np.diff(seconds) <= 0):
        raise RuntimeError(f"receive_second is not strictly increasing in {item.path}")
    expected_dates = pd.to_datetime(seconds, unit="s", utc=True).date
    if any(value != item.receive_date for value in expected_dates):
        raise RuntimeError(
            f"UTC partition-date mismatch in {item.path}; expected {item.receive_date}"
        )
    return frame


def load_local_data(
    day: date,
    partitions: Mapping[date, InputPartition],
    symbol: str,
) -> DenseLocalData:
    dependencies = [partitions[day]]
    next_item = partitions.get(day + timedelta(days=1))
    if next_item is not None:
        dependencies.append(next_item)

    frames = [read_feature_partition(item, symbol) for item in dependencies]
    frame = pd.concat(frames, ignore_index=True)
    seconds = frame["receive_second"].to_numpy(dtype=np.int64)
    if len(seconds) > 1 and np.any(np.diff(seconds) <= 0):
        raise RuntimeError(
            f"Dependency seconds are not globally strictly increasing for {day}"
        )
    base_second = epoch_second_for_date(day)
    maximum_second = int(seconds[-1])
    length = maximum_second - base_second + 1
    if length <= 0 or length > 2 * SECONDS_PER_DAY + 1:
        raise RuntimeError(
            f"Unexpected local dense span for {day}: length={length:,}, "
            f"base={base_second}, max={maximum_second}"
        )
    positions = seconds - base_second
    if np.any(positions < 0) or np.any(positions >= length):
        raise RuntimeError(f"Dependency second outside local grid for {day}")

    dense_price = np.full(length, np.nan, dtype=np.float64)
    dense_observed = np.zeros(length, dtype=np.bool_)
    dense_ready = np.zeros(length, dtype=np.bool_)
    prices = frame["price_for_features"].to_numpy(dtype=float)
    ready = frame["flow_feature_context_ready"].to_numpy(dtype=bool)
    dense_price[positions] = prices
    dense_observed[positions] = True
    dense_ready[positions] = ready
    return DenseLocalData(
        base_second=base_second,
        price=dense_price,
        observed=dense_observed,
        ready=dense_ready,
    )


def label_schema() -> pa.Schema:
    utc_seconds = pa.timestamp("s", tz="UTC")
    return pa.schema(
        [
            pa.field("label_schema_version", pa.int16(), nullable=False),
            pa.field("label_builder_version", pa.string(), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("decision_date", pa.date32(), nullable=False),
            pa.field("decision_second", pa.int64(), nullable=False),
            pa.field("decision_timestamp_utc", utc_seconds, nullable=False),
            pa.field("entry_second", pa.int64(), nullable=False),
            pa.field("entry_timestamp_utc", utc_seconds, nullable=False),
            pa.field("path_end_second", pa.int64(), nullable=False),
            pa.field("path_end_timestamp_utc", utc_seconds, nullable=False),
            pa.field("entry_price", pa.float64(), nullable=False),
            pa.field("future_high_price", pa.float64(), nullable=False),
            pa.field("future_low_price", pa.float64(), nullable=False),
            pa.field("future_terminal_price", pa.float64(), nullable=False),
            pa.field("future_high_second", pa.int64(), nullable=False),
            pa.field("future_low_second", pa.int64(), nullable=False),
            pa.field("future_high_timestamp_utc", utc_seconds, nullable=False),
            pa.field("future_low_timestamp_utc", utc_seconds, nullable=False),
            pa.field("future_swing_15m_bps", pa.float64(), nullable=False),
            pa.field("expansion_50_15m", pa.bool_(), nullable=False),
            pa.field("future_upside_15m_bps", pa.float64(), nullable=False),
            pa.field("future_downside_15m_bps", pa.float64(), nullable=False),
            pa.field("future_terminal_return_15m_bps", pa.float64(), nullable=False),
            pa.field("future_high_first", pa.bool_(), nullable=True),
            pa.field("seconds_to_future_high", pa.int32(), nullable=False),
            pa.field("seconds_to_future_low", pa.int32(), nullable=False),
            pa.field("seconds_to_50bps_realization", pa.int32(), nullable=True),
            pa.field("first_50bps_realization_second", pa.int64(), nullable=True),
            pa.field("first_50bps_realization_timestamp_utc", utc_seconds, nullable=True),
        ]
    )


def empty_label_table(metadata: Mapping[str, Any]) -> pa.Table:
    schema = label_schema().with_metadata(metadata_bytes(metadata))
    return pa.Table.from_arrays(
        [pa.array([], type=field.type) for field in schema], schema=schema
    )


def build_label_table(
    day: date,
    partitions: Mapping[date, InputPartition],
    symbol: str,
    threshold_bps: float,
    dependency: str,
) -> tuple[pa.Table, dict[str, Any], list[dict[str, Any]]]:
    local = load_local_data(day, partitions, symbol)
    finite_positive = (
        local.observed & np.isfinite(local.price) & (local.price > 0.0)
    )
    highs, lows, high_indices, low_indices, complete = forward_extrema_first_indices(
        local.price, finite_positive, PATH_OBSERVATIONS
    )

    day_start_position = 0
    day_end_position = min(SECONDS_PER_DAY, len(local.price))
    anchor_positions = np.arange(day_start_position, day_end_position, dtype=np.int64)
    eligible_anchor = local.observed[anchor_positions] & local.ready[anchor_positions]
    anchor_positions = anchor_positions[eligible_anchor]
    entry_positions = anchor_positions + 1
    in_bounds = entry_positions + HORIZON_SECONDS < len(local.price)
    anchor_positions = anchor_positions[in_bounds]
    entry_positions = entry_positions[in_bounds]
    complete_mask = complete[entry_positions].astype(bool)
    anchor_positions = anchor_positions[complete_mask]
    entry_positions = entry_positions[complete_mask]

    metadata = {
        "perps_expansion_label_builder_version": BUILDER_VERSION,
        "perps_expansion_label_schema_version": LABEL_SCHEMA_VERSION,
        "source_feature_builder_version": partitions[day].feature_builder_version,
        "source_feature_schema_version": EXPECTED_FEATURE_SCHEMA_VERSION,
        "symbol": symbol,
        "decision_date": day.isoformat(),
        "horizon_seconds": HORIZON_SECONDS,
        "path_observations": PATH_OBSERVATIONS,
        "threshold_bps": threshold_bps,
        "causal_entry_contract": "decision_t_entry_t_plus_1_path_through_t_plus_901",
        "dependency_signature": dependency,
        "created_at_utc": utc_now_iso(),
    }

    if not len(anchor_positions):
        summary = {
            "decision_date": day.isoformat(),
            "eligible_ready_anchors": int(
                np.sum(local.observed[:day_end_position] & local.ready[:day_end_position])
            ),
            "valid_label_rows": 0,
            "positive_rows": 0,
            "invalid_or_incomplete_ready_anchors": int(
                np.sum(local.observed[:day_end_position] & local.ready[:day_end_position])
            ),
        }
        return empty_label_table(metadata), summary, []

    entry_price = local.price[entry_positions]
    terminal_positions = entry_positions + HORIZON_SECONDS
    terminal_price = local.price[terminal_positions]
    future_high = highs[entry_positions]
    future_low = lows[entry_positions]
    future_high_positions = high_indices[entry_positions]
    future_low_positions = low_indices[entry_positions]

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
    for name, values in (
        ("future_swing_15m_bps", swing),
        ("future_upside_15m_bps", upside),
        ("future_downside_15m_bps", downside),
    ):
        if np.any(values < -tolerance):
            raise RuntimeError(f"{name} contains a material negative value")
        values[(values < 0) & (values >= -tolerance)] = 0.0

    positive = swing + 1e-10 >= threshold_bps
    realization_offsets = np.full(len(swing), -1, dtype=np.int32)
    positive_indices = np.flatnonzero(positive)
    if len(positive_indices):
        realized = first_range_realization_offsets(
            local.price,
            entry_positions[positive_indices].astype(np.int64),
            HORIZON_SECONDS,
            threshold_bps,
        )
        if np.any(realized < 0):
            raise RuntimeError(
                f"Positive labels without an exact threshold realization on {day}"
            )
        realization_offsets[positive_indices] = realized

    first_realization_seconds = np.full(len(swing), -1, dtype=np.int64)
    first_realization_seconds[positive] = (
        entry_seconds[positive] + realization_offsets[positive].astype(np.int64)
    )

    high_first_values: list[bool | None] = []
    for high_second, low_second in zip(high_seconds, low_seconds):
        if high_second < low_second:
            high_first_values.append(True)
        elif low_second < high_second:
            high_first_values.append(False)
        else:
            high_first_values.append(None)

    realization_offset_values = [
        int(value) if value >= 0 else None for value in realization_offsets
    ]
    realization_second_values = [
        int(value) if value >= 0 else None for value in first_realization_seconds
    ]

    schema = label_schema()
    arrays = [
        pa.array(np.full(len(swing), LABEL_SCHEMA_VERSION, dtype=np.int16), type=pa.int16()),
        pa.array([BUILDER_VERSION] * len(swing), type=pa.string()),
        pa.array([symbol] * len(swing), type=pa.string()),
        pa.array([day] * len(swing), type=pa.date32()),
        pa.array(decision_seconds, type=pa.int64()),
        timestamp_array_from_seconds(decision_seconds),
        pa.array(entry_seconds, type=pa.int64()),
        timestamp_array_from_seconds(entry_seconds),
        pa.array(path_end_seconds, type=pa.int64()),
        timestamp_array_from_seconds(path_end_seconds),
        pa.array(entry_price, type=pa.float64()),
        pa.array(future_high, type=pa.float64()),
        pa.array(future_low, type=pa.float64()),
        pa.array(terminal_price, type=pa.float64()),
        pa.array(high_seconds, type=pa.int64()),
        pa.array(low_seconds, type=pa.int64()),
        timestamp_array_from_seconds(high_seconds),
        timestamp_array_from_seconds(low_seconds),
        pa.array(swing, type=pa.float64()),
        pa.array(positive, type=pa.bool_()),
        pa.array(upside, type=pa.float64()),
        pa.array(downside, type=pa.float64()),
        pa.array(terminal_return, type=pa.float64()),
        pa.array(high_first_values, type=pa.bool_()),
        pa.array((future_high_positions - entry_positions).astype(np.int32), type=pa.int32()),
        pa.array((future_low_positions - entry_positions).astype(np.int32), type=pa.int32()),
        pa.array(realization_offset_values, type=pa.int32()),
        pa.array(realization_second_values, type=pa.int64()),
        timestamp_array_from_seconds(realization_second_values),
    ]
    table = pa.Table.from_arrays(
        arrays,
        schema=schema.with_metadata(metadata_bytes(metadata)),
    )

    eligible_ready = int(
        np.sum(local.observed[:day_end_position] & local.ready[:day_end_position])
    )
    summary = {
        "decision_date": day.isoformat(),
        "eligible_ready_anchors": eligible_ready,
        "valid_label_rows": int(len(swing)),
        "positive_rows": int(np.sum(positive)),
        "invalid_or_incomplete_ready_anchors": eligible_ready - int(len(swing)),
        "minimum_decision_second": int(decision_seconds[0]),
        "maximum_decision_second": int(decision_seconds[-1]),
    }
    invariants = validate_label_arrays(
        day=day,
        decision_seconds=decision_seconds,
        entry_seconds=entry_seconds,
        path_end_seconds=path_end_seconds,
        swing=swing,
        upside=upside,
        downside=downside,
        terminal_return=terminal_return,
        positive=positive,
        high_seconds=high_seconds,
        low_seconds=low_seconds,
        realization_offsets=realization_offsets,
        threshold_bps=threshold_bps,
    )
    return table, summary, invariants


def validate_label_arrays(
    day: date,
    decision_seconds: np.ndarray,
    entry_seconds: np.ndarray,
    path_end_seconds: np.ndarray,
    swing: np.ndarray,
    upside: np.ndarray,
    downside: np.ndarray,
    terminal_return: np.ndarray,
    positive: np.ndarray,
    high_seconds: np.ndarray,
    low_seconds: np.ndarray,
    realization_offsets: np.ndarray,
    threshold_bps: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        rows.append(
            {
                "scope": day.isoformat(),
                "invariant": name,
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
            }
        )

    add(
        "DECISION_SECONDS_STRICTLY_INCREASING",
        bool(len(decision_seconds) < 2 or np.all(np.diff(decision_seconds) > 0)),
        f"rows={len(decision_seconds):,}",
    )
    add(
        "ENTRY_IS_DECISION_PLUS_ONE",
        bool(np.all(entry_seconds == decision_seconds + 1)),
        "entry_second = decision_second + 1",
    )
    add(
        "PATH_END_IS_ENTRY_PLUS_900",
        bool(np.all(path_end_seconds == entry_seconds + HORIZON_SECONDS)),
        f"horizon_seconds={HORIZON_SECONDS}",
    )
    add(
        "SWING_DECOMPOSES_INTO_UPSIDE_PLUS_DOWNSIDE",
        bool(np.allclose(swing, upside + downside, rtol=0, atol=1e-7)),
        "future_swing = future_upside + future_downside",
    )
    add(
        "EXPANSION_LABEL_MATCHES_THRESHOLD",
        bool(np.array_equal(positive, swing + 1e-10 >= threshold_bps)),
        f"threshold_bps={threshold_bps}",
    )
    add(
        "EXTREMA_OFFSETS_WITHIN_PATH",
        bool(
            np.all((high_seconds - entry_seconds >= 0) & (high_seconds <= path_end_seconds))
            and np.all((low_seconds - entry_seconds >= 0) & (low_seconds <= path_end_seconds))
        ),
        "high and low timestamps lie inside the exact future path",
    )
    add(
        "REALIZATION_PRESENT_ONLY_FOR_POSITIVES",
        bool(
            np.all(realization_offsets[positive] >= 0)
            and np.all(realization_offsets[positive] <= HORIZON_SECONDS)
            and np.all(realization_offsets[~positive] == -1)
        ),
        "positive offsets in [0, 900], negatives are null sentinel",
    )
    add(
        "TERMINAL_RETURN_FINITE",
        bool(np.isfinite(terminal_return).all()),
        f"finite_rows={int(np.isfinite(terminal_return).sum()):,}",
    )
    failures = [row for row in rows if row["status"] == "FAIL"]
    if failures:
        detail = "; ".join(row["invariant"] for row in failures)
        raise RuntimeError(f"Label validation failed for {day}: {detail}")
    return rows


def validate_written_label_file(
    path: Path,
    expected_rows: int,
    expected_dependency: str,
    expected_day: date,
) -> None:
    parquet_file = pq.ParquetFile(path)
    try:
        if int(parquet_file.metadata.num_rows) != expected_rows:
            raise RuntimeError(
                f"Written label row count mismatch for {path}: "
                f"expected={expected_rows:,}, actual={parquet_file.metadata.num_rows:,}"
            )
        missing = [name for name in LABEL_COLUMNS if name not in parquet_file.schema_arrow.names]
        if missing:
            raise RuntimeError(f"Written label file is missing columns: {missing}")
        metadata = decode_metadata(parquet_file.schema_arrow)
        if metadata.get("dependency_signature") != expected_dependency:
            raise RuntimeError(f"Dependency signature mismatch in {path}")
        if metadata.get("decision_date") != expected_day.isoformat():
            raise RuntimeError(f"Decision-date metadata mismatch in {path}")
    finally:
        close = getattr(parquet_file, "close", None)
        if callable(close):
            close()


# ---------------------------------------------------------------------------
# Episode clustering
# ---------------------------------------------------------------------------


def load_all_positive_anchors(states: Sequence[sqlite3.Row]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for state in states:
        path = Path(state["output_path"])
        if not path.exists():
            raise RuntimeError(f"Completed label partition is missing: {path}")
        parquet_file = pq.ParquetFile(path)
        try:
            table = parquet_file.read(columns=list(POSITIVE_COLUMNS))
        finally:
            close = getattr(parquet_file, "close", None)
            if callable(close):
                close()
        frame = table.to_pandas()
        if len(frame):
            frame = frame[frame["future_swing_15m_bps"] + 1e-10 >= THRESHOLD_BPS]
            if len(frame):
                frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=list(POSITIVE_COLUMNS))
    output = pd.concat(frames, ignore_index=True)
    output = output.sort_values("decision_second", kind="mergesort").reset_index(drop=True)
    if output["decision_second"].duplicated().any():
        duplicates = int(output["decision_second"].duplicated().sum())
        raise RuntimeError(f"Duplicate positive decision seconds found: {duplicates:,}")
    return output


def episode_id(symbol: str, first_decision_second: int) -> str:
    stamp = datetime.fromtimestamp(first_decision_second, tz=timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    return f"EXP50_15M_{symbol}_{stamp}"


def classify_path(
    upside_bps: float, downside_bps: float, swing_bps: float, dominance_ratio: float
) -> tuple[bool, str, float]:
    if swing_bps <= 0:
        return False, "TWO_SIDED", float("nan")
    ratio = max(upside_bps, downside_bps) / swing_bps
    if ratio + 1e-12 < dominance_ratio:
        return False, "TWO_SIDED", ratio
    if upside_bps > downside_bps:
        return True, "TREND_UP", ratio
    if downside_bps > upside_bps:
        return True, "TREND_DOWN", ratio
    return False, "TWO_SIDED", ratio


def cluster_episodes(
    positives: pd.DataFrame,
    symbol: str,
    trend_dominance_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    episode_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    invariant_rows: list[dict[str, Any]] = []

    if positives.empty:
        return pd.DataFrame(), pd.DataFrame(), [
            {
                "scope": "GLOBAL",
                "invariant": "NO_POSITIVE_ROWS_NO_EPISODES",
                "status": "PASS",
                "detail": "positive_rows=0, episodes=0",
            }
        ]

    starts: list[int] = []
    ends: list[int] = []
    current_start = 0
    current_union_end = int(positives.loc[0, "path_end_second"])
    for index in range(1, len(positives)):
        interval_start = int(positives.loc[index, "entry_second"])
        interval_end = int(positives.loc[index, "path_end_second"])
        if interval_start <= current_union_end:
            if interval_end > current_union_end:
                current_union_end = interval_end
        else:
            starts.append(current_start)
            ends.append(index)
            current_start = index
            current_union_end = interval_end
    starts.append(current_start)
    ends.append(len(positives))

    previous_last_decision: int | None = None
    previous_union_end: int | None = None
    assigned_rows = 0

    for ordinal, (start, end) in enumerate(zip(starts, ends), start=1):
        group = positives.iloc[start:end].copy()
        assigned_rows += len(group)
        first_decision = int(group["decision_second"].iloc[0])
        last_decision = int(group["decision_second"].iloc[-1])
        first_entry = int(group["entry_second"].iloc[0])
        union_end = int(group["path_end_second"].max())
        identifier = episode_id(symbol, first_decision)

        max_swing = float(group["future_swing_15m_bps"].max())
        representative_candidates = group[
            np.isclose(
                group["future_swing_15m_bps"].to_numpy(dtype=float),
                max_swing,
                rtol=0,
                atol=1e-10,
            )
        ]
        representative_index = int(representative_candidates.index[0])
        representative = positives.loc[representative_index]

        first_realization = int(group["first_50bps_realization_second"].min())
        realization_anchor_index = int(
            group["first_50bps_realization_second"].astype("int64").idxmin()
        )
        realization_anchor = positives.loc[realization_anchor_index]

        trend_like, path_type, dominant_ratio = classify_path(
            float(representative["future_upside_15m_bps"]),
            float(representative["future_downside_15m_bps"]),
            float(representative["future_swing_15m_bps"]),
            trend_dominance_ratio,
        )
        high_first_raw = representative["future_high_first"]
        high_first = None if pd.isna(high_first_raw) else bool(high_first_raw)

        positive_span_seconds = last_decision - first_decision + 1
        row = {
            "episode_schema_version": EPISODE_SCHEMA_VERSION,
            "episode_builder_version": BUILDER_VERSION,
            "episode_rule_version": EPISODE_RULE_VERSION,
            "episode_ordinal": ordinal,
            "episode_id": identifier,
            "symbol": symbol,
            "episode_start": first_decision,
            "episode_start_timestamp_utc": iso_from_second(first_decision),
            "episode_end": last_decision,
            "episode_end_timestamp_utc": iso_from_second(last_decision),
            "first_positive_decision_second": first_decision,
            "last_positive_decision_second": last_decision,
            "episode_start_date": date_from_second(first_decision).isoformat(),
            "episode_end_date": date_from_second(last_decision).isoformat(),
            "crosses_utc_date": date_from_second(first_decision)
            != date_from_second(last_decision),
            "episode_duration_seconds": positive_span_seconds,
            "number_of_positive_rolling_anchors": int(len(group)),
            "positive_anchor_coverage_pct": percent(len(group), positive_span_seconds),
            "future_interval_union_start_second": first_entry,
            "future_interval_union_end_second": union_end,
            "future_interval_union_duration_seconds": union_end - first_entry + 1,
            "first_50bps_realization_second": first_realization,
            "first_50bps_realization_timestamp_utc": iso_from_second(first_realization),
            "seconds_from_episode_start_to_first_50bps_realization": first_realization
            - first_decision,
            "seconds_from_first_executable_entry_to_first_50bps_realization": first_realization
            - first_entry,
            "first_realization_anchor_decision_second": int(
                realization_anchor["decision_second"]
            ),
            "representative_decision_second": int(representative["decision_second"]),
            "maximum_future_swing_15m_bps": max_swing,
            "maximum_future_upside_15m_bps": float(
                group["future_upside_15m_bps"].max()
            ),
            "maximum_future_downside_15m_bps": float(
                group["future_downside_15m_bps"].max()
            ),
            "representative_terminal_return_15m_bps": float(
                representative["future_terminal_return_15m_bps"]
            ),
            "representative_future_high_second": int(
                representative["future_high_second"]
            ),
            "representative_future_high_timestamp_utc": iso_from_second(
                int(representative["future_high_second"])
            ),
            "representative_future_low_second": int(
                representative["future_low_second"]
            ),
            "representative_future_low_timestamp_utc": iso_from_second(
                int(representative["future_low_second"])
            ),
            "representative_high_first": high_first,
            "trend_like": trend_like,
            "path_type": path_type,
            "dominant_excursion_share": dominant_ratio,
            "seconds_since_previous_episode": None
            if previous_last_decision is None
            else first_decision - previous_last_decision - 1,
            "seconds_since_previous_episode_future_union": None
            if previous_union_end is None
            else first_entry - previous_union_end - 1,
        }
        episode_rows.append(row)

        group_indices = list(group.index)
        for rank, source_index in enumerate(group_indices, start=1):
            source = positives.loc[source_index]
            membership_rows.append(
                {
                    "episode_id": identifier,
                    "symbol": symbol,
                    "decision_date": str(source["decision_date"]),
                    "decision_second": int(source["decision_second"]),
                    "anchor_rank_within_episode": rank,
                    "is_first_positive_anchor": rank == 1,
                    "is_last_positive_anchor": rank == len(group_indices),
                    "is_representative_max_swing_anchor": source_index
                    == representative_index,
                }
            )

        previous_last_decision = last_decision
        previous_union_end = union_end

    episodes = pd.DataFrame(episode_rows)
    membership = pd.DataFrame(membership_rows)

    def add(name: str, passed: bool, detail: str) -> None:
        invariant_rows.append(
            {
                "scope": "GLOBAL",
                "invariant": name,
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
            }
        )

    add(
        "ALL_POSITIVE_ANCHORS_ASSIGNED_ONCE",
        assigned_rows == len(positives)
        and len(membership) == len(positives)
        and not membership["decision_second"].duplicated().any(),
        f"positive_rows={len(positives):,}, membership_rows={len(membership):,}",
    )
    add(
        "EPISODE_IDS_UNIQUE",
        not episodes["episode_id"].duplicated().any(),
        f"episodes={len(episodes):,}",
    )
    add(
        "EPISODES_STRICTLY_ORDERED_AND_NONOVERLAPPING",
        bool(
            len(episodes) < 2
            or np.all(
                episodes["future_interval_union_start_second"].to_numpy(dtype=np.int64)[1:]
                > episodes["future_interval_union_end_second"].to_numpy(dtype=np.int64)[:-1]
            )
        ),
        "successive connected components have disjoint future-interval unions",
    )
    add(
        "EPISODE_POSITIVE_COUNTS_RECONCILE",
        int(episodes["number_of_positive_rolling_anchors"].sum()) == len(positives),
        f"episode_sum={int(episodes['number_of_positive_rolling_anchors'].sum()):,}",
    )
    add(
        "EPISODE_MAX_SWING_AT_OR_ABOVE_THRESHOLD",
        bool((episodes["maximum_future_swing_15m_bps"] + 1e-10 >= THRESHOLD_BPS).all()),
        f"threshold_bps={THRESHOLD_BPS}",
    )
    failures = [row for row in invariant_rows if row["status"] == "FAIL"]
    if failures:
        raise RuntimeError(
            "Episode validation failed: "
            + ", ".join(row["invariant"] for row in failures)
        )
    return episodes, membership, invariant_rows


def dataframe_to_parquet_table(
    frame: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> pa.Table:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    return table.replace_schema_metadata(metadata_bytes(metadata))


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def metric_profile(episodes: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "episode_duration_seconds": "episode_duration_seconds",
        "warning_lead_seconds": "seconds_from_episode_start_to_first_50bps_realization",
        "maximum_future_swing_15m_bps": "maximum_future_swing_15m_bps",
        "seconds_between_episodes": "seconds_since_previous_episode",
    }
    rows: list[dict[str, Any]] = []
    for label, column in metrics.items():
        values = pd.to_numeric(episodes.get(column, pd.Series(dtype=float)), errors="coerce")
        values = values[np.isfinite(values)]
        for statistic, value in (
            ("count", float(len(values))),
            ("mean", float(values.mean()) if len(values) else float("nan")),
            ("median", float(values.median()) if len(values) else float("nan")),
            ("p75", float(values.quantile(0.75)) if len(values) else float("nan")),
            ("p90", float(values.quantile(0.90)) if len(values) else float("nan")),
            ("p95", float(values.quantile(0.95)) if len(values) else float("nan")),
            ("maximum", float(values.max()) if len(values) else float("nan")),
        ):
            rows.append({"metric": label, "statistic": statistic, "value": value})
    return pd.DataFrame(rows)


def duration_distribution(episodes: pd.DataFrame) -> pd.DataFrame:
    edges = np.array([0, 30, 60, 120, 300, 600, 900, 1800, 3600, np.inf], dtype=float)
    labels = [
        "0-29s",
        "30-59s",
        "60-119s",
        "120-299s",
        "300-599s",
        "600-899s",
        "900-1799s",
        "1800-3599s",
        "3600s+",
    ]
    values = pd.to_numeric(
        episodes.get("episode_duration_seconds", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    counts, _ = np.histogram(values.to_numpy(dtype=float), bins=edges)
    total = int(np.sum(counts))
    return pd.DataFrame(
        [
            {
                "bucket_order": index + 1,
                "duration_bucket": labels[index],
                "lower_seconds": int(edges[index]),
                "upper_seconds_exclusive": None
                if math.isinf(edges[index + 1])
                else int(edges[index + 1]),
                "episode_count": int(count),
                "episode_pct": percent(int(count), total),
            }
            for index, count in enumerate(counts)
        ]
    )


def build_daily_profile(
    states: Sequence[sqlite3.Row], episodes: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if len(episodes):
        episodes_by_start = {
            key: group.copy()
            for key, group in episodes.groupby("episode_start_date", sort=True)
        }
    else:
        episodes_by_start = {}

    for state in states:
        day = str(state["decision_date"])
        day_episodes = episodes_by_start.get(day, pd.DataFrame())
        valid_rows = int(state["output_rows"])
        positive_rows = int(state["positive_rows"])
        if len(day_episodes):
            high_first_count = int(
                day_episodes["representative_high_first"].fillna(False).astype(bool).sum()
            )
            low_first_count = int(
                (~day_episodes["representative_high_first"].fillna(True).astype(bool)).sum()
            )
            trend_count = int(day_episodes["trend_like"].astype(bool).sum())
            two_sided_count = len(day_episodes) - trend_count
            durations = day_episodes["episode_duration_seconds"].astype(float)
            leads = day_episodes[
                "seconds_from_episode_start_to_first_50bps_realization"
            ].astype(float)
            swings = day_episodes["maximum_future_swing_15m_bps"].astype(float)
            gaps = pd.to_numeric(
                day_episodes["seconds_since_previous_episode"], errors="coerce"
            ).dropna()
        else:
            high_first_count = low_first_count = trend_count = two_sided_count = 0
            durations = pd.Series(dtype=float)
            leads = pd.Series(dtype=float)
            swings = pd.Series(dtype=float)
            gaps = pd.Series(dtype=float)
        rows.append(
            {
                "decision_date": day,
                "valid_label_rows": valid_rows,
                "expansion_positive_rows": positive_rows,
                "share_time_expansion_positive_pct": percent(positive_rows, valid_rows),
                "episodes_started": int(len(day_episodes)),
                "mean_episode_duration_seconds": float(durations.mean())
                if len(durations)
                else float("nan"),
                "median_episode_duration_seconds": float(durations.median())
                if len(durations)
                else float("nan"),
                "mean_warning_lead_seconds": float(leads.mean())
                if len(leads)
                else float("nan"),
                "median_warning_lead_seconds": float(leads.median())
                if len(leads)
                else float("nan"),
                "mean_maximum_swing_bps": float(swings.mean())
                if len(swings)
                else float("nan"),
                "maximum_swing_bps": float(swings.max())
                if len(swings)
                else float("nan"),
                "representative_high_first_episodes": high_first_count,
                "representative_low_first_episodes": low_first_count,
                "trend_like_episodes": trend_count,
                "two_sided_episodes": two_sided_count,
                "mean_seconds_since_previous_episode": float(gaps.mean())
                if len(gaps)
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_overall_profile(
    states: Sequence[sqlite3.Row], episodes: pd.DataFrame
) -> pd.DataFrame:
    valid_rows = sum(int(row["output_rows"]) for row in states)
    positive_rows = sum(int(row["positive_rows"]) for row in states)
    observed_days = len(states)
    episode_count = len(episodes)
    high_first = (
        int(episodes["representative_high_first"].fillna(False).astype(bool).sum())
        if episode_count
        else 0
    )
    known_order = (
        int(episodes["representative_high_first"].notna().sum()) if episode_count else 0
    )
    trend_count = int(episodes["trend_like"].astype(bool).sum()) if episode_count else 0
    two_sided_count = episode_count - trend_count
    crosses_date = int(episodes["crosses_utc_date"].astype(bool).sum()) if episode_count else 0
    rows = [
        ("completed_label_dates", observed_days),
        ("valid_row_level_labels", valid_rows),
        ("expansion_positive_rows", positive_rows),
        ("share_time_expansion_positive_pct", percent(positive_rows, valid_rows)),
        ("distinct_expansion_episodes", episode_count),
        ("episodes_per_completed_label_day", episode_count / observed_days if observed_days else float("nan")),
        (
            "mean_episode_duration_seconds",
            float(episodes["episode_duration_seconds"].mean()) if episode_count else float("nan"),
        ),
        (
            "median_episode_duration_seconds",
            float(episodes["episode_duration_seconds"].median()) if episode_count else float("nan"),
        ),
        (
            "p90_episode_duration_seconds",
            float(episodes["episode_duration_seconds"].quantile(0.90)) if episode_count else float("nan"),
        ),
        (
            "mean_warning_lead_seconds",
            float(
                episodes[
                    "seconds_from_episode_start_to_first_50bps_realization"
                ].mean()
            )
            if episode_count
            else float("nan"),
        ),
        (
            "median_warning_lead_seconds",
            float(
                episodes[
                    "seconds_from_episode_start_to_first_50bps_realization"
                ].median()
            )
            if episode_count
            else float("nan"),
        ),
        (
            "mean_maximum_episode_swing_bps",
            float(episodes["maximum_future_swing_15m_bps"].mean())
            if episode_count
            else float("nan"),
        ),
        (
            "maximum_episode_swing_bps",
            float(episodes["maximum_future_swing_15m_bps"].max())
            if episode_count
            else float("nan"),
        ),
        ("representative_high_first_episodes", high_first),
        ("representative_low_first_episodes", known_order - high_first),
        ("trend_like_episodes", trend_count),
        ("two_sided_episodes", two_sided_count),
        ("cross_utc_date_episodes", crosses_date),
        (
            "mean_seconds_between_episodes",
            float(
                pd.to_numeric(
                    episodes["seconds_since_previous_episode"], errors="coerce"
                ).mean()
            )
            if episode_count > 1
            else float("nan"),
        ),
        (
            "median_seconds_between_episodes",
            float(
                pd.to_numeric(
                    episodes["seconds_since_previous_episode"], errors="coerce"
                ).median()
            )
            if episode_count > 1
            else float("nan"),
        ),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def report_invariants(
    states: Sequence[sqlite3.Row],
    positives: pd.DataFrame,
    episodes: pd.DataFrame,
    membership: pd.DataFrame,
    episode_invariants: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    rows = list(episode_invariants)

    def add(name: str, passed: bool, detail: str) -> None:
        rows.append(
            {
                "scope": "REPORT",
                "invariant": name,
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
            }
        )

    expected_positive = sum(int(row["positive_rows"]) for row in states)
    add(
        "STATE_POSITIVE_ROWS_MATCH_LOADED_POSITIVES",
        expected_positive == len(positives),
        f"state={expected_positive:,}, loaded={len(positives):,}",
    )
    add(
        "MEMBERSHIP_ROWS_MATCH_POSITIVES",
        len(membership) == len(positives),
        f"membership={len(membership):,}, positives={len(positives):,}",
    )
    add(
        "EPISODE_COUNTS_MATCH_MEMBERSHIP",
        bool(
            not len(episodes)
            or int(episodes["number_of_positive_rolling_anchors"].sum())
            == len(membership)
        ),
        f"episodes={len(episodes):,}",
    )
    add(
        "ALL_COMPLETED_LABEL_OUTPUTS_EXIST",
        all(Path(row["output_path"]).exists() for row in states),
        f"completed_states={len(states):,}",
    )
    return pd.DataFrame(rows)


def findings_markdown(
    config: Mapping[str, Any],
    overall: pd.DataFrame,
    episodes: pd.DataFrame,
    invariants: pd.DataFrame,
) -> str:
    lookup = dict(zip(overall["metric"], overall["value"]))
    episode_count = int(lookup.get("distinct_expansion_episodes", 0))
    lines = [
        "# Perps 15-Minute 50-bps Expansion Episodes",
        "",
        f"Generated at: `{utc_now_iso()}`",
        "",
        "## Exact label contract",
        "",
        "- Decision row: second `t`.",
        "- Executable entry: exact price at `t + 1`.",
        "- Future path: all exact seconds from `t + 1` through `t + 901` inclusive.",
        "- Positive target: future high-low swing is at least 50 bps relative to entry.",
        "- Eligibility: `flow_feature_context_ready = true` and a complete exact path.",
        "- Missing seconds are invalidating; they are never filled.",
        "",
        "## Episode clustering",
        "",
        "Each positive row contributes the closed future interval `[t + 1, t + 901]`. "
        "Chronologically adjacent positive rows belong to the same episode when their "
        "future intervals overlap the current component's interval union. Clustering is "
        "transitive and continues across UTC midnight.",
        "",
        "`episode_start` and `episode_end` describe the span of positive decision states. "
        "The separate future-interval union fields describe the future market path covered "
        "by those labels.",
        "",
        "## Path classification",
        "",
        f"The representative row is the earliest row with the episode's maximum 15-minute "
        f"swing. A path is trend-like when its larger entry-relative excursion contributes "
        f"at least {float(config['trend_dominance_ratio']):.0%} of total high-low swing; "
        "otherwise it is two-sided. This is an excursion-shape label, not proof of monotonic "
        "price movement.",
        "",
        "## Overall profile",
        "",
        f"- Valid row-level labels: **{int(lookup.get('valid_row_level_labels', 0)):,}**",
        f"- Positive rolling labels: **{int(lookup.get('expansion_positive_rows', 0)):,}**",
        f"- Share of valid labelled seconds positive: **{float(lookup.get('share_time_expansion_positive_pct', float('nan'))):.3f}%**",
        f"- Distinct episodes: **{episode_count:,}**",
        f"- Episodes per completed label day: **{float(lookup.get('episodes_per_completed_label_day', float('nan'))):.3f}**",
        f"- Median episode duration: **{float(lookup.get('median_episode_duration_seconds', float('nan'))):.1f} seconds**",
        f"- Median warning lead: **{float(lookup.get('median_warning_lead_seconds', float('nan'))):.1f} seconds**",
        "",
        "## Interpretation",
        "",
        "The row-level dataset is suitable for causal decision-cadence sampling. The episode "
        "dataset is the primary unit for prevalence, held-out episode recall, warning-time "
        "analysis and later walk-forward evaluation. It should not be randomly split.",
        "",
        "## Validation",
        "",
        f"- Failed invariants: **{int((invariants['status'] == 'FAIL').sum()) if len(invariants) else 0}**",
        f"- Total invariants: **{len(invariants):,}**",
        "",
    ]
    if episode_count:
        largest = episodes.nlargest(5, "maximum_future_swing_15m_bps")
        lines.extend(
            [
                "## Five largest episodes by representative swing",
                "",
                "| Episode | Start | Positive anchors | Duration (s) | Max swing (bps) | Path type |",
                "|---|---|---:|---:|---:|---|",
            ]
        )
        for _, row in largest.iterrows():
            lines.append(
                f"| `{row['episode_id']}` | {row['episode_start_timestamp_utc']} | "
                f"{int(row['number_of_positive_rolling_anchors']):,} | "
                f"{int(row['episode_duration_seconds']):,} | "
                f"{float(row['maximum_future_swing_15m_bps']):.3f} | "
                f"{row['path_type']} |"
            )
        lines.append("")
    return "\n".join(lines)


def write_reports(
    report_root: Path,
    run_id: str,
    config: Mapping[str, Any],
    states: Sequence[sqlite3.Row],
    positives: pd.DataFrame,
    episodes: pd.DataFrame,
    membership: pd.DataFrame,
    episode_invariants: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], pd.DataFrame]:
    daily = build_daily_profile(states, episodes)
    overall = build_overall_profile(states, episodes)
    metric_stats = metric_profile(episodes)
    duration = duration_distribution(episodes)
    invariants = report_invariants(
        states, positives, episodes, membership, episode_invariants
    )
    failed = int((invariants["status"] == "FAIL").sum()) if len(invariants) else 0

    summary = {
        "builder_version": BUILDER_VERSION,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "episode_schema_version": EPISODE_SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at_utc": utc_now_iso(),
        "status": "PASS" if failed == 0 else "FAIL",
        "configuration": dict(config),
        "completed_label_dates": len(states),
        "valid_row_level_labels": sum(int(row["output_rows"]) for row in states),
        "positive_row_level_labels": sum(int(row["positive_rows"]) for row in states),
        "distinct_episodes": len(episodes),
        "failed_invariants": failed,
    }
    findings = findings_markdown(config, overall, episodes, invariants)

    run_dir = report_root / "runs" / run_id
    output_map = {
        "perps_expansion_episode_overall.csv": overall,
        "perps_expansion_episode_daily.csv": daily,
        "perps_expansion_episode_metric_profile.csv": metric_stats,
        "perps_expansion_episode_duration_distribution.csv": duration,
        "perps_expansion_episode_invariants.csv": invariants,
    }
    for filename, frame in output_map.items():
        atomic_write_csv(run_dir / filename, frame)
        atomic_write_csv(report_root / filename, frame)
    atomic_write_json(run_dir / "perps_expansion_episode_summary.json", summary)
    atomic_write_json(report_root / "perps_expansion_episode_summary.json", summary)
    atomic_write_json(run_dir / "perps_expansion_episode_config.json", dict(config))
    atomic_write_json(report_root / "perps_expansion_episode_config.json", dict(config))
    atomic_write_text(run_dir / "perps_expansion_episode_findings.md", findings)
    atomic_write_text(report_root / "perps_expansion_episode_findings.md", findings)
    return summary, invariants


# ---------------------------------------------------------------------------
# Main build orchestration
# ---------------------------------------------------------------------------


def label_output_path(output_root: Path, symbol: str, day: date) -> Path:
    return (
        output_root
        / "row_labels"
        / f"{SYMBOL_PREFIX}{symbol}"
        / f"{DATE_PREFIX}{day.isoformat()}"
        / "part-00000.parquet"
    )


def episode_output_path(output_root: Path, symbol: str) -> Path:
    return output_root / "episodes" / f"{SYMBOL_PREFIX}{symbol}" / "part-00000.parquet"


def membership_output_path(output_root: Path, symbol: str) -> Path:
    return (
        output_root
        / "episode_membership"
        / f"{SYMBOL_PREFIX}{symbol}"
        / "part-00000.parquet"
    )


def remove_stale_label_dates(
    connection: sqlite3.Connection,
    output_root: Path,
    symbol: str,
    current_dates: set[date],
) -> list[str]:
    removed: list[str] = []
    rows = connection.execute(
        "SELECT decision_date, output_path FROM label_partitions WHERE symbol=?",
        (symbol,),
    ).fetchall()
    for row in rows:
        state_day = date.fromisoformat(row["decision_date"])
        if state_day in current_dates:
            continue
        path = Path(row["output_path"])
        if path.exists():
            directory = path.parent
            shutil.rmtree(directory)
            parent = directory.parent
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
        connection.execute(
            "DELETE FROM label_partitions WHERE symbol=? AND decision_date=?",
            (symbol, state_day.isoformat()),
        )
        removed.append(state_day.isoformat())
    connection.commit()
    return removed


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Build causal 15-minute 50-bps perps expansion labels and episodes."
    )
    value.add_argument(
        "--feature-root",
        default="./data/derived/perps/flow_features_1s_v1",
        help="Root of the audited perps flow-feature dataset.",
    )
    value.add_argument(
        "--output-root",
        default="./data/derived/perps/research/expansion_episodes_15m_50bps_v1",
        help="Research-label and episode dataset root.",
    )
    value.add_argument(
        "--report-root",
        default="./reports/perps_expansion_episodes_15m_50bps",
        help="CSV, JSON and Markdown report root.",
    )
    value.add_argument("--symbol", default="BTCUSDT")
    value.add_argument("--start-date", help="Optional inclusive YYYY-MM-DD filter.")
    value.add_argument("--end-date", help="Optional inclusive YYYY-MM-DD filter.")
    value.add_argument(
        "--max-dates",
        type=int,
        help="Optional limit after date filtering; useful for isolated smoke tests.",
    )
    value.add_argument(
        "--expected-feature-builder-version",
        default=EXPECTED_FEATURE_BUILDER_VERSION,
        help="Expected upstream flow-feature builder version; empty disables the check.",
    )
    value.add_argument(
        "--threshold-bps",
        type=float,
        default=THRESHOLD_BPS,
        help="Expansion threshold. The v1 dataset contract expects 50 bps.",
    )
    value.add_argument(
        "--trend-dominance-ratio",
        type=float,
        default=DEFAULT_TREND_DOMINANCE_RATIO,
        help="Minimum dominant-excursion share for TREND_UP/TREND_DOWN.",
    )
    value.add_argument(
        "--hash-input-files",
        action="store_true",
        help="Use full SHA-256 content hashes in input change detection.",
    )
    value.add_argument(
        "--force",
        action="store_true",
        help="Rebuild selected label partitions even when state says unchanged.",
    )
    value.add_argument(
        "--compression-level",
        type=int,
        default=6,
        help="Zstandard Parquet compression level.",
    )
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not math.isclose(args.threshold_bps, THRESHOLD_BPS, rel_tol=0, abs_tol=1e-12):
        raise SystemExit(
            "This v1 dataset is fixed to 50 bps. Use --threshold-bps 50 or create a "
            "separately versioned output dataset for another threshold."
        )
    if not 0.5 <= args.trend_dominance_ratio <= 1.0:
        raise SystemExit("--trend-dominance-ratio must be between 0.5 and 1.0")
    if not 1 <= args.compression_level <= 22:
        raise SystemExit("--compression-level must be between 1 and 22")

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

    label_config = {
        "builder_version": BUILDER_VERSION,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "episode_schema_version": EPISODE_SCHEMA_VERSION,
        "feature_root": str(feature_root),
        "output_root": str(output_root),
        "report_root": str(report_root),
        "symbol": symbol,
        "horizon_seconds": HORIZON_SECONDS,
        "path_observations": PATH_OBSERVATIONS,
        "threshold_bps": float(args.threshold_bps),
        "trend_dominance_ratio": float(args.trend_dominance_ratio),
        "episode_rule_version": EPISODE_RULE_VERSION,
        "expected_feature_builder_version": args.expected_feature_builder_version,
        "expected_feature_schema_version": EXPECTED_FEATURE_SCHEMA_VERSION,
        "hash_input_files": bool(args.hash_input_files),
        "start_date": start_date.isoformat() if start_date else None,
        "end_date": end_date.isoformat() if end_date else None,
        "max_dates": args.max_dates,
    }
    label_config_signature = sha256_text(
        compact_json(
            {
                key: value
                for key, value in label_config.items()
                if key not in {"report_root", "start_date", "end_date", "max_dates"}
            }
        )
    )

    connection = connect_state(state_path)
    connection.execute(
        "INSERT INTO runs (run_id, started_at_utc, status, configuration_json) "
        "VALUES (?, ?, 'RUNNING', ?)",
        (run_id, utc_now_iso(), json.dumps(label_config, sort_keys=True)),
    )
    connection.commit()

    rebuilt_dates = 0
    skipped_dates = 0
    failed_dates = 0
    removed_dates: list[str] = []
    partition_summaries: list[dict[str, Any]] = []
    label_invariants: list[dict[str, Any]] = []

    try:
        print("Discovering audited flow-feature partitions...")
        partitions = discover_partitions(
            feature_root=feature_root,
            symbol=symbol,
            expected_builder_version=args.expected_feature_builder_version,
            hash_input_files=args.hash_input_files,
        )
        dates = selected_dates(partitions, start_date, end_date, args.max_dates)
        record_inputs(connection, symbol, run_id, partitions)

        full_unfiltered_run = (
            start_date is None and end_date is None and args.max_dates is None
        )
        if full_unfiltered_run:
            removed_dates = remove_stale_label_dates(
                connection, output_root, symbol, set(partitions)
            )
            if removed_dates:
                print(f"Removed stale label dates: {', '.join(removed_dates)}")

        print(f"Input dates      : {len(partitions):,}")
        print(f"Selected dates   : {len(dates):,}")
        print(f"Date range       : {dates[0]} through {dates[-1]}")
        print(f"Output root      : {output_root}")
        print(f"State database   : {state_path}")

        for index, day in enumerate(dates, start=1):
            dependency = dependency_signature(day, partitions, label_config_signature)
            path = label_output_path(output_root, symbol, day)
            state = label_state(connection, symbol, day)
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
                    )
                except Exception:
                    unchanged = False
            if unchanged:
                skipped_dates += 1
                print(
                    f"SKIP      [{index:>3}/{len(dates)}] {day}: "
                    f"rows={int(state['output_rows']):,}, positives={int(state['positive_rows']):,}"
                )
                partition_summaries.append(
                    {
                        "decision_date": day.isoformat(),
                        "status": "SKIPPED_UNCHANGED",
                        "valid_label_rows": int(state["output_rows"]),
                        "positive_rows": int(state["positive_rows"]),
                    }
                )
                continue

            mark_label_running(connection, symbol, day, dependency, path)
            try:
                table, summary, invariants = build_label_table(
                    day=day,
                    partitions=partitions,
                    symbol=symbol,
                    threshold_bps=float(args.threshold_bps),
                    dependency=dependency,
                )
                atomic_write_parquet(
                    path,
                    table,
                    compression="zstd",
                    compression_level=args.compression_level,
                )
                cleanup_extra_parquet_files(path.parent, path)
                validate_written_label_file(path, table.num_rows, dependency, day)
                positive_rows = int(summary["positive_rows"])
                minimum_second = summary.get("minimum_decision_second")
                maximum_second = summary.get("maximum_decision_second")
                mark_label_completed(
                    connection,
                    symbol,
                    day,
                    dependency,
                    path,
                    table.num_rows,
                    positive_rows,
                    minimum_second,
                    maximum_second,
                )
                rebuilt_dates += 1
                partition_summaries.append({**summary, "status": "COMPLETED"})
                label_invariants.extend(invariants)
                print(
                    f"COMPLETED [{index:>3}/{len(dates)}] {day}: "
                    f"rows={table.num_rows:,}, positives={positive_rows:,}, "
                    f"share={percent(positive_rows, table.num_rows):.3f}%"
                )
            except Exception as exc:
                failed_dates += 1
                error_text = f"{type(exc).__name__}: {exc}"
                mark_label_failed(connection, symbol, day, error_text)
                print(f"FAILED    [{index:>3}/{len(dates)}] {day}: {error_text}")
                traceback.print_exc()

        if failed_dates:
            raise RuntimeError(
                f"{failed_dates} label partition(s) failed; global episodes were not rebuilt"
            )

        states = completed_label_states(connection, symbol)
        if not states:
            raise RuntimeError("No completed row-label partitions are available")

        # Reports are cheap enough to refresh even when row labels are unchanged.
        print("Loading positive anchors for global cross-day clustering...")
        positives = load_all_positive_anchors(states)
        episodes, membership, episode_invariants = cluster_episodes(
            positives,
            symbol,
            float(args.trend_dominance_ratio),
        )

        episode_metadata = {
            "perps_expansion_episode_builder_version": BUILDER_VERSION,
            "perps_expansion_episode_schema_version": EPISODE_SCHEMA_VERSION,
            "episode_rule_version": EPISODE_RULE_VERSION,
            "symbol": symbol,
            "horizon_seconds": HORIZON_SECONDS,
            "threshold_bps": THRESHOLD_BPS,
            "trend_dominance_ratio": args.trend_dominance_ratio,
            "created_at_utc": utc_now_iso(),
        }
        membership_metadata = {
            **episode_metadata,
            "dataset_role": "positive_anchor_episode_membership",
        }
        # Global outputs are re-materialized atomically after every successful run.
        # This is inexpensive relative to label construction and repairs any damaged
        # derived file even when all date partitions were correctly skipped.
        episode_table = dataframe_to_parquet_table(episodes, episode_metadata)
        membership_table = dataframe_to_parquet_table(
            membership, membership_metadata
        )
        episode_path = episode_output_path(output_root, symbol)
        membership_path = membership_output_path(output_root, symbol)
        atomic_write_parquet(
            episode_path,
            episode_table,
            compression="zstd",
            compression_level=args.compression_level,
        )
        atomic_write_parquet(
            membership_path,
            membership_table,
            compression="zstd",
            compression_level=args.compression_level,
        )
        cleanup_extra_parquet_files(episode_path.parent, episode_path)
        cleanup_extra_parquet_files(membership_path.parent, membership_path)

        all_invariants = episode_invariants + label_invariants
        summary, report_checks = write_reports(
            report_root=report_root,
            run_id=run_id,
            config=label_config,
            states=states,
            positives=positives,
            episodes=episodes,
            membership=membership,
            episode_invariants=all_invariants,
        )
        failed_invariants = int((report_checks["status"] == "FAIL").sum())
        status = "COMPLETED" if failed_invariants == 0 else "FAILED_VALIDATION"

        latest_run = {
            "run_id": run_id,
            "builder_version": BUILDER_VERSION,
            "status": status,
            "started_or_completed_at_utc": utc_now_iso(),
            "configuration": label_config,
            "input_dates": len(partitions),
            "selected_dates": len(dates),
            "rebuilt_dates": rebuilt_dates,
            "skipped_dates": skipped_dates,
            "failed_dates": failed_dates,
            "removed_dates": removed_dates,
            "completed_label_dates": len(states),
            "row_label_rows": sum(int(row["output_rows"]) for row in states),
            "positive_rows": len(positives),
            "episodes": len(episodes),
            "failed_invariants": failed_invariants,
            "partition_summaries": partition_summaries,
        }
        metadata_dir = output_root / "_metadata"
        atomic_write_json(metadata_dir / "latest_run_summary.json", latest_run)
        atomic_write_json(metadata_dir / "dataset_config.json", label_config)

        connection.execute(
            """
            UPDATE runs SET completed_at_utc=?, status=?, rebuilt_dates=?,
                skipped_dates=?, failed_dates=?, error_text=NULL
            WHERE run_id=?
            """,
            (
                utc_now_iso(),
                status,
                rebuilt_dates,
                skipped_dates,
                failed_dates,
                run_id,
            ),
        )
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES('latest_run_id', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (run_id,),
        )
        connection.commit()

        print("\nExpansion episode build finished.")
        print(f"Status            : {status}")
        print(f"Dates rebuilt     : {rebuilt_dates:,}")
        print(f"Dates unchanged   : {skipped_dates:,}")
        print(f"Row-level labels  : {latest_run['row_label_rows']:,}")
        print(f"Positive labels   : {len(positives):,}")
        print(f"Distinct episodes : {len(episodes):,}")
        print(f"Failed invariants : {failed_invariants:,}")
        print(f"Dataset root      : {output_root}")
        print(f"Reports           : {report_root}")
        return 0 if status == "COMPLETED" else 1

    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        connection.execute(
            """
            UPDATE runs SET completed_at_utc=?, status='FAILED', rebuilt_dates=?,
                skipped_dates=?, failed_dates=?, error_text=? WHERE run_id=?
            """,
            (
                utc_now_iso(),
                rebuilt_dates,
                skipped_dates,
                failed_dates,
                error_text,
                run_id,
            ),
        )
        connection.commit()
        print(f"\nExpansion episode build failed: {error_text}", file=sys.stderr)
        return 1
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
