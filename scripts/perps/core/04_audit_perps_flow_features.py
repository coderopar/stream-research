#!/usr/bin/env python
"""
Independently audit the causal perps flow-feature one-second layer.

This auditor validates the output of::

    scripts/perps/core/03_build_perps_flow_features.py

against the audited perps core one-second layer. It does not trust the feature
builder's run summary as proof of correctness. For every date it:

* reconciles core, feature, and SQLite state ownership;
* validates the complete Arrow schema and Parquet metadata contract;
* verifies that the current core context manifest matches the build manifest;
* reconstructs every feature from core rows using only receive seconds <= t;
* checks exact row alignment, timestamps, price selection, returns, rolling
  windows, OI changes, liquidation measurements, z-scores, readiness flags,
  and cumulative CVD;
* validates cross-date carry signatures and state continuity;
* emits machine-readable CSV/JSON reports and a concise Markdown finding file.

The audit is deliberately strict. Any formula, causal-alignment, schema,
state, or manifest mismatch is a failure. Inherited source-grid gaps are
reported as informational observations when the feature layer handles them
correctly by nulling incomplete rolling windows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - environment specific
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]


AUDITOR_VERSION = "1.0.0"
EXPECTED_FEATURE_BUILDER_VERSION = "1.0.2"
EXPECTED_FEATURE_SCHEMA_VERSION = 1
EXPECTED_CORE_SCHEMA_VERSION = 1
SECONDS_PER_DAY = 86_400
DATE_PARTITION_RE = re.compile(r"^date=(\d{4}-\d{2}-\d{2})$")
SYMBOL_PARTITION_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")

DEFAULT_WINDOWS = (5, 15, 30, 60, 300, 900, 3600)
DEFAULT_ZSCORE_WINDOWS = (300, 900, 3600)

REQUIRED_CORE_COLUMNS = (
    "core_schema_version",
    "symbol",
    "receive_second",
    "timestamp_utc",
    "available_after_second",
    "available_after_utc",
    "seconds_from_day_start",
    "core_context_ready",
    "trade_count",
    "quote_volume",
    "base_volume",
    "aggressive_buy_quote_volume",
    "aggressive_sell_quote_volume",
    "signed_quote_flow",
    "signed_base_flow",
    "max_trade_notional",
    "last_trade_price",
    "trade_price_fresh",
    "mark_price",
    "mark_fresh",
    "open_interest_updated",
    "open_interest",
    "open_interest_change",
    "liquidation_event_count",
    "long_liquidation_notional",
    "short_liquidation_notional",
    "liquidation_notional",
    "net_liquidation_notional",
    "max_liquidation_notional",
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureConfig:
    windows: tuple[int, ...]
    zscore_windows: tuple[int, ...]
    zscore_min_fraction: float
    min_impact_flow_notional: float

    @property
    def max_window(self) -> int:
        return max((*self.windows, *self.zscore_windows, 1))

    def signature(self) -> str:
        return sha256_text(compact_json(asdict(self)))


@dataclass(frozen=True)
class CarryState:
    cvd_quote: float = 0.0
    cvd_base: float = 0.0

    def signature(self) -> str:
        return sha256_text(compact_json(asdict(self)))


@dataclass(frozen=True)
class DatePartition:
    day: date
    files: tuple[Path, ...]


@dataclass
class WindowBounds:
    left: np.ndarray
    row_count: np.ndarray
    complete: np.ndarray


@dataclass
class Issue:
    severity: str
    code: str
    receive_date: str
    column_name: str
    message: str
    mismatch_count: int = 0
    first_receive_second: int | None = None
    expected_value: str = ""
    actual_value: str = ""
    max_abs_error: float | None = None
    max_rel_error: float | None = None

    def row(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "receive_date": self.receive_date,
            "column_name": self.column_name,
            "message": self.message,
            "mismatch_count": self.mismatch_count,
            "first_receive_second": self.first_receive_second,
            "expected_value": self.expected_value,
            "actual_value": self.actual_value,
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
        }


@dataclass
class Invariant:
    invariant: str
    status: str
    failures: int
    warnings: int
    details: str

    def row(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------


def require_arrow() -> None:
    if pa is None or pq is None:
        raise SystemExit(
            "Missing dependency. Activate the project virtual environment and run: "
            "pip install pandas pyarrow"
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def create_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{uuid.uuid4().hex[:8]}"


def compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_date(value: str | None, argument_name: str) -> date | None:
    if value is None:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SystemExit(f"{argument_name} must use YYYY-MM-DD; received {value!r}") from exc


def safe_symbol_partition(symbol: str) -> str:
    safe = SYMBOL_PARTITION_SAFE_RE.sub("_", symbol.strip().upper())
    if not safe:
        raise SystemExit("--symbol resolved to an empty partition name")
    return safe


def day_start_second(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())


def receive_day_from_second(second: int) -> date:
    return datetime.fromtimestamp(int(second), tz=timezone.utc).date()


def load_json(value: str | bytes | None, default: Any) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)):
            return "null"
        return f"{float(value):.15g}"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.bool_, bool)):
        return "true" if bool(value) else "false"
    return str(value)


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copy_latest(run_root: Path, latest_root: Path, filenames: Sequence[str]) -> None:
    latest_root.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        source = run_root / filename
        if source.exists():
            shutil.copy2(source, latest_root / filename)


def add_issue(
    issues: list[Issue],
    severity: str,
    code: str,
    day: date | str | None,
    message: str,
    *,
    column_name: str = "",
    mismatch_count: int = 0,
    first_receive_second: int | None = None,
    expected_value: Any = "",
    actual_value: Any = "",
    max_abs_error: float | None = None,
    max_rel_error: float | None = None,
) -> None:
    issues.append(
        Issue(
            severity=severity,
            code=code,
            receive_date=(day.isoformat() if isinstance(day, date) else str(day or "")),
            column_name=column_name,
            message=message,
            mismatch_count=int(mismatch_count),
            first_receive_second=first_receive_second,
            expected_value=format_scalar(expected_value),
            actual_value=format_scalar(actual_value),
            max_abs_error=max_abs_error,
            max_rel_error=max_rel_error,
        )
    )


# ---------------------------------------------------------------------------
# Arrow schema and metadata
# ---------------------------------------------------------------------------


def expected_feature_schema(config: FeatureConfig) -> "pa.Schema":
    require_arrow()
    fields = [
        pa.field("feature_schema_version", pa.int16(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("receive_second", pa.int64(), nullable=False),
        pa.field("timestamp_utc", pa.timestamp("s", tz="UTC"), nullable=False),
        pa.field("available_after_second", pa.int64(), nullable=False),
        pa.field("available_after_utc", pa.timestamp("s", tz="UTC"), nullable=False),
        pa.field("seconds_from_day_start", pa.int32(), nullable=False),
        pa.field("core_schema_version", pa.int16(), nullable=False),
        pa.field("core_context_ready", pa.bool_(), nullable=False),
        pa.field("price_for_features", pa.float64()),
        pa.field("price_source_code", pa.int8(), nullable=False),
        pa.field("price_fresh", pa.bool_(), nullable=False),
        pa.field("log_return_1s", pa.float64()),
        pa.field("return_1s_bps", pa.float64()),
        pa.field("abs_return_1s_bps", pa.float64()),
        pa.field("trade_count_1s", pa.int64(), nullable=False),
        pa.field("quote_volume_1s", pa.float64(), nullable=False),
        pa.field("base_volume_1s", pa.float64(), nullable=False),
        pa.field("signed_quote_flow_1s", pa.float64(), nullable=False),
        pa.field("signed_base_flow_1s", pa.float64(), nullable=False),
        pa.field("aggressive_flow_imbalance_1s", pa.float64()),
        pa.field("max_trade_notional_1s", pa.float64(), nullable=False),
        pa.field("max_trade_share_1s", pa.float64()),
        pa.field("open_interest_updated_1s", pa.bool_(), nullable=False),
        pa.field("open_interest", pa.float64()),
        pa.field("observed_open_interest_change_1s", pa.float64()),
        pa.field("liquidation_event_count_1s", pa.int64(), nullable=False),
        pa.field("liquidation_notional_1s", pa.float64(), nullable=False),
        pa.field("net_liquidation_notional_1s", pa.float64(), nullable=False),
        pa.field("liquidation_imbalance_1s", pa.float64()),
        pa.field("max_liquidation_notional_1s", pa.float64(), nullable=False),
        pa.field("cvd_quote", pa.float64(), nullable=False),
        pa.field("cvd_base", pa.float64(), nullable=False),
    ]
    for window in config.windows:
        suffix = f"{window}s"
        fields.extend(
            [
                pa.field(f"window_observation_count_{suffix}", pa.int32(), nullable=False),
                pa.field(f"window_complete_{suffix}", pa.bool_(), nullable=False),
                pa.field(f"return_lag_available_{suffix}", pa.bool_(), nullable=False),
                pa.field(f"log_return_{suffix}", pa.float64()),
                pa.field(f"return_{suffix}_bps", pa.float64()),
                pa.field(f"realized_vol_{suffix}_bps", pa.float64()),
                pa.field(f"trade_count_{suffix}", pa.int64()),
                pa.field(f"quote_volume_{suffix}", pa.float64()),
                pa.field(f"signed_quote_flow_{suffix}", pa.float64()),
                pa.field(f"aggressive_flow_imbalance_{suffix}", pa.float64()),
                pa.field(f"open_interest_change_{suffix}", pa.float64()),
                pa.field(f"open_interest_change_{suffix}_pct", pa.float64()),
                pa.field(f"liquidation_notional_{suffix}", pa.float64()),
                pa.field(f"net_liquidation_notional_{suffix}", pa.float64()),
                pa.field(f"liquidation_imbalance_{suffix}", pa.float64()),
                pa.field(f"liquidation_to_quote_ratio_{suffix}", pa.float64()),
                pa.field(f"price_flow_alignment_{suffix}", pa.int8()),
                pa.field(f"price_response_bps_per_1m_signed_flow_{suffix}", pa.float64()),
            ]
        )
    for window in config.zscore_windows:
        for feature_name in (
            "signed_quote_flow_1s",
            "quote_volume_1s",
            "liquidation_notional_1s",
            "abs_return_1s_bps",
        ):
            fields.append(pa.field(f"{feature_name}_z_{window}s", pa.float64()))
    fields.append(pa.field("flow_feature_context_ready", pa.bool_(), nullable=False))
    return pa.schema(fields)


def semantic_type_equal(actual: "pa.DataType", expected: "pa.DataType") -> bool:
    if actual == expected:
        return True
    if pa.types.is_timestamp(actual) and pa.types.is_timestamp(expected):
        # Parquet may widen timestamp[s] to timestamp[ms]. Values are audited.
        return actual.tz == expected.tz and actual.unit in {"s", "ms", "us", "ns"}
    return False


def semantic_schema_equal(actual: "pa.Schema", expected: "pa.Schema") -> bool:
    if actual.names != expected.names or len(actual) != len(expected):
        return False
    for actual_field, expected_field in zip(actual, expected):
        if actual_field.nullable != expected_field.nullable:
            return False
        if not semantic_type_equal(actual_field.type, expected_field.type):
            return False
    return True


def metadata_text(schema: "pa.Schema") -> dict[str, str]:
    return {
        key.decode("utf-8", errors="replace"): value.decode("utf-8", errors="replace")
        for key, value in (schema.metadata or {}).items()
    }


def infer_config(feature_files: Sequence[Path], issues: list[Issue]) -> FeatureConfig:
    require_arrow()
    if not feature_files:
        raise SystemExit("No feature Parquet files were found")
    parquet_file = pq.ParquetFile(feature_files[0])
    try:
        metadata = metadata_text(parquet_file.schema_arrow)
    finally:
        parquet_file.close()

    windows_payload = load_json(metadata.get("windows_seconds"), list(DEFAULT_WINDOWS))
    zscore_payload = load_json(metadata.get("zscore_windows_seconds"), list(DEFAULT_ZSCORE_WINDOWS))
    try:
        windows = tuple(int(item) for item in windows_payload)
        zscore_windows = tuple(int(item) for item in zscore_payload)
        fraction = float(metadata.get("zscore_min_fraction", "0.8"))
        impact = float(metadata.get("min_impact_flow_notional", "100000"))
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Could not infer feature configuration from Parquet metadata: {exc}") from exc

    if not windows or not zscore_windows or any(item <= 0 for item in (*windows, *zscore_windows)):
        raise SystemExit("Invalid feature windows in Parquet metadata")
    if not 0 < fraction <= 1:
        raise SystemExit("Invalid zscore_min_fraction in Parquet metadata")
    if impact < 0:
        raise SystemExit("Invalid min_impact_flow_notional in Parquet metadata")

    required_metadata = (
        "perps_flow_feature_schema_version",
        "perps_flow_feature_builder_version",
        "source_core_schema_version",
        "symbol",
        "receive_date",
        "causal_clock",
        "availability_rule",
        "windows_seconds",
        "zscore_windows_seconds",
        "zscore_min_fraction",
        "min_impact_flow_notional",
        "input_signature",
        "carry_in_signature",
        "carry_out_signature",
        "input_manifest_json",
    )
    missing = [key for key in required_metadata if key not in metadata]
    if missing:
        add_issue(
            issues,
            "FAIL",
            "METADATA_REQUIRED",
            None,
            f"First feature file is missing required metadata keys: {missing}",
        )
    return FeatureConfig(
        windows=tuple(sorted(set(windows))),
        zscore_windows=tuple(sorted(set(zscore_windows))),
        zscore_min_fraction=fraction,
        min_impact_flow_notional=impact,
    )


# ---------------------------------------------------------------------------
# Partition discovery and manifests
# ---------------------------------------------------------------------------


def discover_partitions(root: Path, symbol: str) -> dict[date, DatePartition]:
    symbol_root = root / f"symbol={safe_symbol_partition(symbol)}"
    if not symbol_root.exists():
        return {}
    result: dict[date, DatePartition] = {}
    for child in sorted(symbol_root.iterdir()):
        if not child.is_dir():
            continue
        match = DATE_PARTITION_RE.match(child.name)
        if not match:
            continue
        day = datetime.strptime(match.group(1), "%Y-%m-%d").date()
        result[day] = DatePartition(day=day, files=tuple(sorted(child.glob("*.parquet"))))
    return result


def context_days_for(day: date, all_days: Sequence[date], max_window_seconds: int) -> list[date]:
    earliest_second = day_start_second(day) - max_window_seconds - 1
    return [
        candidate
        for candidate in all_days
        if day_start_second(candidate) + SECONDS_PER_DAY - 1 >= earliest_second
        and candidate <= day
    ]


def file_manifest(path: Path, root: Path) -> dict[str, Any]:
    require_arrow()
    stat = path.stat()
    parquet_file = pq.ParquetFile(path)
    try:
        schema_metadata = parquet_file.schema_arrow.metadata or {}
        return {
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "rows": int(parquet_file.metadata.num_rows),
            "row_groups": int(parquet_file.metadata.num_row_groups),
            "schema_names": list(parquet_file.schema_arrow.names),
            "core_schema_version": (schema_metadata.get(b"perps_core_schema_version") or b"").decode(),
            "core_builder_version": (schema_metadata.get(b"builder_version") or b"").decode(),
        }
    finally:
        parquet_file.close()


def manifest_signature(manifests: Sequence[Mapping[str, Any]]) -> str:
    return sha256_text(compact_json(list(manifests)))


def current_context_manifest(
    core_root: Path,
    core_partitions: Mapping[date, DatePartition],
    all_core_days: Sequence[date],
    day: date,
    max_window: int,
) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for context_day in context_days_for(day, all_core_days, max_window):
        for path in core_partitions[context_day].files:
            item = file_manifest(path, core_root)
            item["context_receive_date"] = context_day.isoformat()
            manifests.append(item)
    return manifests


# ---------------------------------------------------------------------------
# Input readers
# ---------------------------------------------------------------------------


def read_core_partition(partition: DatePartition) -> pd.DataFrame:
    require_arrow()
    frames: list[pd.DataFrame] = []
    for path in partition.files:
        parquet_file = pq.ParquetFile(path)
        try:
            missing = [name for name in REQUIRED_CORE_COLUMNS if name not in parquet_file.schema_arrow.names]
            if missing:
                raise RuntimeError(f"Core file {path} is missing required columns: {missing}")
            frames.append(parquet_file.read(columns=list(REQUIRED_CORE_COLUMNS)).to_pandas())
        finally:
            parquet_file.close()
    if not frames:
        return pd.DataFrame(columns=REQUIRED_CORE_COLUMNS)
    frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    frame = frame.sort_values("receive_second", kind="mergesort").reset_index(drop=True)
    return frame


def read_core_context(
    partitions: Mapping[date, DatePartition],
    days: Sequence[date],
) -> pd.DataFrame:
    frames = [read_core_partition(partitions[item]) for item in days]
    if not frames:
        return pd.DataFrame(columns=REQUIRED_CORE_COLUMNS)
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.sort_values("receive_second", kind="mergesort").reset_index(drop=True)
    return frame


def read_feature_partition(partition: DatePartition) -> tuple[pd.DataFrame, "pa.Schema", int, int]:
    require_arrow()
    frames: list[pd.DataFrame] = []
    schemas: list[pa.Schema] = []
    row_groups = 0
    physical_rows = 0
    for path in partition.files:
        parquet_file = pq.ParquetFile(path)
        try:
            schemas.append(parquet_file.schema_arrow)
            row_groups += int(parquet_file.metadata.num_row_groups)
            physical_rows += int(parquet_file.metadata.num_rows)
            frames.append(parquet_file.read().to_pandas())
        finally:
            parquet_file.close()
    if not frames:
        return pd.DataFrame(), pa.schema([]), 0, 0
    frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    frame = frame.sort_values("receive_second", kind="mergesort").reset_index(drop=True)
    return frame, schemas[0], physical_rows, row_groups


# ---------------------------------------------------------------------------
# Independent feature engine
# ---------------------------------------------------------------------------


def choose_feature_price(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trade_price = pd.to_numeric(frame["last_trade_price"], errors="coerce").to_numpy(float)
    mark_price = pd.to_numeric(frame["mark_price"], errors="coerce").to_numpy(float)
    trade_fresh = frame["trade_price_fresh"].fillna(False).to_numpy(bool)
    mark_fresh = frame["mark_fresh"].fillna(False).to_numpy(bool)
    valid_trade = np.isfinite(trade_price) & (trade_price > 0)
    valid_mark = np.isfinite(mark_price) & (mark_price > 0)
    price = np.full(len(frame), np.nan, dtype=float)
    source = np.zeros(len(frame), dtype=np.int8)
    mask = trade_fresh & valid_trade
    price[mask] = trade_price[mask]
    source[mask] = 1
    mask = (source == 0) & mark_fresh & valid_mark
    price[mask] = mark_price[mask]
    source[mask] = 2
    mask = (source == 0) & valid_trade
    price[mask] = trade_price[mask]
    source[mask] = 3
    mask = (source == 0) & valid_mark
    price[mask] = mark_price[mask]
    source[mask] = 4
    return price, source, np.isin(source, [1, 2])


def exact_lag(values: np.ndarray, seconds: np.ndarray, lag_seconds: int) -> tuple[np.ndarray, np.ndarray]:
    target = seconds - int(lag_seconds)
    positions = np.searchsorted(seconds, target, side="left")
    available = positions < len(seconds)
    if len(seconds):
        safe = np.minimum(positions, len(seconds) - 1)
        available &= seconds[safe] == target
    result = np.full(len(seconds), np.nan, dtype=float)
    if len(seconds):
        result[available] = values[positions[available]]
    return result, available


def calculate_window_bounds(seconds: np.ndarray, window: int) -> WindowBounds:
    left_targets = seconds - int(window) + 1
    left = np.searchsorted(seconds, left_targets, side="left")
    row_count = np.arange(len(seconds), dtype=np.int64) - left + 1
    complete = (row_count == int(window)) & (left < len(seconds)) & (seconds[left] == left_targets)
    return WindowBounds(left=left, row_count=row_count, complete=complete)


def rolling_sum(values: np.ndarray, bounds: WindowBounds) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    safe = np.where(finite, values, 0.0)
    cumulative = np.concatenate(([0.0], np.cumsum(safe, dtype=float)))
    count_cumulative = np.concatenate(([0], np.cumsum(finite.astype(np.int64))))
    right = np.arange(len(values), dtype=np.int64) + 1
    return (
        cumulative[right] - cumulative[bounds.left],
        count_cumulative[right] - count_cumulative[bounds.left],
    )


def rolling_zscore(values: np.ndarray, bounds: WindowBounds, minimum_count: int) -> np.ndarray:
    finite = np.isfinite(values)
    safe = np.where(finite, values, 0.0)
    square = np.where(finite, values * values, 0.0)
    cumulative = np.concatenate(([0.0], np.cumsum(safe, dtype=float)))
    cumulative_square = np.concatenate(([0.0], np.cumsum(square, dtype=float)))
    cumulative_count = np.concatenate(([0], np.cumsum(finite.astype(np.int64))))
    right = np.arange(len(values), dtype=np.int64) + 1
    sums = cumulative[right] - cumulative[bounds.left]
    sums_square = cumulative_square[right] - cumulative_square[bounds.left]
    counts = cumulative_count[right] - cumulative_count[bounds.left]
    means = np.divide(sums, counts, out=np.full(len(values), np.nan), where=counts > 0)
    variances = np.divide(
        sums_square,
        counts,
        out=np.full(len(values), np.nan),
        where=counts > 0,
    ) - means * means
    variances = np.where(variances < 0, np.maximum(variances, 0.0), variances)
    std = np.sqrt(variances)
    result = np.full(len(values), np.nan)
    valid = bounds.complete & finite & (counts >= minimum_count) & np.isfinite(std) & (std > 1e-12)
    result[valid] = (values[valid] - means[valid]) / std[valid]
    return result


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full(len(numerator), np.nan, dtype=float)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
    result[valid] = numerator[valid] / denominator[valid]
    return result


def bounded_ratio(numerator: np.ndarray, denominator: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
    result = safe_ratio(numerator, denominator)
    finite = np.isfinite(result)
    small = finite & (np.abs(result) > 1.0) & (np.abs(result) <= 1.0 + tolerance)
    result[small] = np.clip(result[small], -1.0, 1.0)
    return result


def component_imbalance(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)
    return bounded_ratio(positive - negative, positive + negative)


def signed_alignment(left: np.ndarray, right: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.full(len(left), np.nan, dtype=float)
    finite = valid & np.isfinite(left) & np.isfinite(right)
    result[finite] = np.sign(left[finite]) * np.sign(right[finite])
    return result


def expected_feature_columns(
    context: pd.DataFrame,
    output_day: date,
    incoming_carry: CarryState,
    config: FeatureConfig,
) -> tuple[dict[str, Any], CarryState, dict[str, Any]]:
    if context.empty:
        raise RuntimeError("Core feature context is empty")
    seconds = pd.to_numeric(context["receive_second"], errors="raise").to_numpy(np.int64)
    if np.any(np.diff(seconds) <= 0):
        raise RuntimeError("Core context receive_second is not strictly increasing")
    start = day_start_second(output_day)
    end = start + SECONDS_PER_DAY - 1
    current_mask = (seconds >= start) & (seconds <= end)
    if not current_mask.any():
        raise RuntimeError(f"No core rows exist for {output_day}")
    current_indices = np.flatnonzero(current_mask)
    output_slice = slice(int(current_indices[0]), int(current_indices[-1]) + 1)
    if not current_mask[output_slice].all():
        raise RuntimeError(f"Core rows for {output_day} are not contiguous inside context")
    current = context.iloc[output_slice].reset_index(drop=True)

    price, price_source, price_fresh = choose_feature_price(context)
    lag_1s_price, lag_1s_available = exact_lag(price, seconds, 1)
    log_return_1s = np.full(len(context), np.nan)
    valid_1s = (
        lag_1s_available
        & np.isfinite(price)
        & np.isfinite(lag_1s_price)
        & (price > 0)
        & (lag_1s_price > 0)
    )
    log_return_1s[valid_1s] = np.log(price[valid_1s] / lag_1s_price[valid_1s])
    return_1s_bps = log_return_1s * 10_000.0
    abs_return_1s_bps = np.abs(return_1s_bps)

    trade_count = pd.to_numeric(context["trade_count"], errors="coerce").fillna(0).to_numpy(float)
    quote_volume = pd.to_numeric(context["quote_volume"], errors="coerce").fillna(0).to_numpy(float)
    base_volume = pd.to_numeric(context["base_volume"], errors="coerce").fillna(0).to_numpy(float)
    buy_quote = pd.to_numeric(context["aggressive_buy_quote_volume"], errors="coerce").fillna(0).to_numpy(float)
    sell_quote = pd.to_numeric(context["aggressive_sell_quote_volume"], errors="coerce").fillna(0).to_numpy(float)
    signed_quote = pd.to_numeric(context["signed_quote_flow"], errors="coerce").fillna(0).to_numpy(float)
    signed_base = pd.to_numeric(context["signed_base_flow"], errors="coerce").fillna(0).to_numpy(float)
    max_trade = pd.to_numeric(context["max_trade_notional"], errors="coerce").fillna(0).to_numpy(float)
    open_interest = pd.to_numeric(context["open_interest"], errors="coerce").to_numpy(float)
    observed_oi_change = pd.to_numeric(context["open_interest_change"], errors="coerce").to_numpy(float)
    oi_updated = context["open_interest_updated"].fillna(False).to_numpy(bool)
    long_liq = pd.to_numeric(context["long_liquidation_notional"], errors="coerce").fillna(0).to_numpy(float)
    short_liq = pd.to_numeric(context["short_liquidation_notional"], errors="coerce").fillna(0).to_numpy(float)
    total_liq = pd.to_numeric(context["liquidation_notional"], errors="coerce").fillna(0).to_numpy(float)
    net_liq = pd.to_numeric(context["net_liquidation_notional"], errors="coerce").fillna(0).to_numpy(float)
    max_liq = pd.to_numeric(context["max_liquidation_notional"], errors="coerce").fillna(0).to_numpy(float)

    current_signed_quote = signed_quote[output_slice]
    current_signed_base = signed_base[output_slice]
    cvd_quote = incoming_carry.cvd_quote + np.cumsum(current_signed_quote, dtype=float)
    cvd_base = incoming_carry.cvd_base + np.cumsum(current_signed_base, dtype=float)
    outgoing = CarryState(float(cvd_quote[-1]), float(cvd_base[-1]))

    columns: dict[str, Any] = {
        "feature_schema_version": np.full(len(current), EXPECTED_FEATURE_SCHEMA_VERSION, dtype=np.int16),
        "symbol": current["symbol"].astype(str).to_numpy(),
        "receive_second": seconds[output_slice],
        "timestamp_utc": seconds[output_slice],
        "available_after_second": pd.to_numeric(current["available_after_second"], errors="raise").to_numpy(np.int64),
        "available_after_utc": pd.to_numeric(current["available_after_second"], errors="raise").to_numpy(np.int64),
        "seconds_from_day_start": pd.to_numeric(current["seconds_from_day_start"], errors="raise").to_numpy(np.int32),
        "core_schema_version": pd.to_numeric(current["core_schema_version"], errors="raise").to_numpy(np.int16),
        "core_context_ready": current["core_context_ready"].fillna(False).to_numpy(bool),
        "price_for_features": price[output_slice],
        "price_source_code": price_source[output_slice],
        "price_fresh": price_fresh[output_slice],
        "log_return_1s": log_return_1s[output_slice],
        "return_1s_bps": return_1s_bps[output_slice],
        "abs_return_1s_bps": abs_return_1s_bps[output_slice],
        "trade_count_1s": pd.to_numeric(current["trade_count"], errors="raise").to_numpy(np.int64),
        "quote_volume_1s": quote_volume[output_slice],
        "base_volume_1s": base_volume[output_slice],
        "signed_quote_flow_1s": signed_quote[output_slice],
        "signed_base_flow_1s": signed_base[output_slice],
        "aggressive_flow_imbalance_1s": component_imbalance(buy_quote, sell_quote)[output_slice],
        "max_trade_notional_1s": max_trade[output_slice],
        "max_trade_share_1s": safe_ratio(max_trade, quote_volume)[output_slice],
        "open_interest_updated_1s": oi_updated[output_slice],
        "open_interest": open_interest[output_slice],
        "observed_open_interest_change_1s": observed_oi_change[output_slice],
        "liquidation_event_count_1s": pd.to_numeric(current["liquidation_event_count"], errors="raise").to_numpy(np.int64),
        "liquidation_notional_1s": total_liq[output_slice],
        "net_liquidation_notional_1s": net_liq[output_slice],
        "liquidation_imbalance_1s": bounded_ratio(short_liq - long_liq, total_liq)[output_slice],
        "max_liquidation_notional_1s": max_liq[output_slice],
        "cvd_quote": cvd_quote,
        "cvd_base": cvd_base,
    }

    window_summary: dict[str, Any] = {}
    squared_returns = log_return_1s * log_return_1s
    for window in config.windows:
        suffix = f"{window}s"
        bounds = calculate_window_bounds(seconds, window)
        lag_price, lag_available = exact_lag(price, seconds, window)
        valid_return = (
            lag_available
            & np.isfinite(price)
            & np.isfinite(lag_price)
            & (price > 0)
            & (lag_price > 0)
        )
        horizon_log_return = np.full(len(context), np.nan)
        horizon_log_return[valid_return] = np.log(price[valid_return] / lag_price[valid_return])
        horizon_return_bps = horizon_log_return * 10_000.0

        trade_sum, _ = rolling_sum(trade_count, bounds)
        quote_sum, _ = rolling_sum(quote_volume, bounds)
        buy_sum, _ = rolling_sum(buy_quote, bounds)
        sell_sum, _ = rolling_sum(sell_quote, bounds)
        signed_sum = buy_sum - sell_sum
        long_sum, _ = rolling_sum(long_liq, bounds)
        short_sum, _ = rolling_sum(short_liq, bounds)
        total_liq_sum, _ = rolling_sum(total_liq, bounds)
        net_liq_sum = short_sum - long_sum
        rv_sum, rv_count = rolling_sum(squared_returns, bounds)
        realized_vol = np.sqrt(np.maximum(rv_sum, 0.0)) * 10_000.0

        lag_oi, lag_oi_available = exact_lag(open_interest, seconds, window)
        oi_change = np.full(len(context), np.nan)
        oi_pct = np.full(len(context), np.nan)
        valid_oi = lag_oi_available & np.isfinite(open_interest) & np.isfinite(lag_oi)
        oi_change[valid_oi] = open_interest[valid_oi] - lag_oi[valid_oi]
        valid_oi_pct = valid_oi & (lag_oi != 0)
        oi_pct[valid_oi_pct] = oi_change[valid_oi_pct] / lag_oi[valid_oi_pct]

        rolling_ready = bounds.complete
        return_ready = bounds.complete & lag_available
        rv_ready = return_ready & (rv_count == window)
        flow_imbalance = component_imbalance(buy_sum, sell_sum)
        liq_imbalance = bounded_ratio(net_liq_sum, total_liq_sum)
        liq_to_quote = safe_ratio(total_liq_sum, quote_sum)
        alignment = signed_alignment(horizon_return_bps, signed_sum, return_ready)
        response = np.full(len(context), np.nan)
        impact_ready = (
            return_ready
            & np.isfinite(horizon_return_bps)
            & np.isfinite(signed_sum)
            & (np.abs(signed_sum) >= config.min_impact_flow_notional)
        )
        response[impact_ready] = horizon_return_bps[impact_ready] / (signed_sum[impact_ready] / 1_000_000.0)

        for values in (trade_sum, quote_sum, signed_sum, flow_imbalance, total_liq_sum, net_liq_sum, liq_imbalance, liq_to_quote):
            values[~rolling_ready] = np.nan
        for values in (horizon_log_return, horizon_return_bps, alignment, response):
            values[~return_ready] = np.nan
        oi_ready = bounds.complete & lag_oi_available
        oi_change[~oi_ready] = np.nan
        oi_pct[~oi_ready] = np.nan
        realized_vol[~rv_ready] = np.nan

        columns[f"window_observation_count_{suffix}"] = bounds.row_count[output_slice].astype(np.int32)
        columns[f"window_complete_{suffix}"] = bounds.complete[output_slice]
        columns[f"return_lag_available_{suffix}"] = lag_available[output_slice]
        columns[f"log_return_{suffix}"] = horizon_log_return[output_slice]
        columns[f"return_{suffix}_bps"] = horizon_return_bps[output_slice]
        columns[f"realized_vol_{suffix}_bps"] = realized_vol[output_slice]
        trade_current = trade_sum[output_slice]
        columns[f"trade_count_{suffix}"] = np.where(np.isfinite(trade_current), np.rint(trade_current), np.nan)
        columns[f"quote_volume_{suffix}"] = quote_sum[output_slice]
        columns[f"signed_quote_flow_{suffix}"] = signed_sum[output_slice]
        columns[f"aggressive_flow_imbalance_{suffix}"] = flow_imbalance[output_slice]
        columns[f"open_interest_change_{suffix}"] = oi_change[output_slice]
        columns[f"open_interest_change_{suffix}_pct"] = oi_pct[output_slice]
        columns[f"liquidation_notional_{suffix}"] = total_liq_sum[output_slice]
        columns[f"net_liquidation_notional_{suffix}"] = net_liq_sum[output_slice]
        columns[f"liquidation_imbalance_{suffix}"] = liq_imbalance[output_slice]
        columns[f"liquidation_to_quote_ratio_{suffix}"] = liq_to_quote[output_slice]
        columns[f"price_flow_alignment_{suffix}"] = alignment[output_slice]
        columns[f"price_response_bps_per_1m_signed_flow_{suffix}"] = response[output_slice]
        window_summary[suffix] = {
            "complete_rows": int(bounds.complete[output_slice].sum()),
            "return_ready_rows": int(return_ready[output_slice].sum()),
            "realized_vol_ready_rows": int(rv_ready[output_slice].sum()),
            "impact_ready_rows": int(impact_ready[output_slice].sum()),
        }

    z_sources = {
        "signed_quote_flow_1s": signed_quote,
        "quote_volume_1s": quote_volume,
        "liquidation_notional_1s": total_liq,
        "abs_return_1s_bps": abs_return_1s_bps,
    }
    for window in config.zscore_windows:
        bounds = calculate_window_bounds(seconds, window)
        minimum_count = max(2, int(math.ceil(window * config.zscore_min_fraction)))
        for feature_name, values in z_sources.items():
            columns[f"{feature_name}_z_{window}s"] = rolling_zscore(values, bounds, minimum_count)[output_slice]

    max_complete = calculate_window_bounds(seconds, config.max_window).complete[output_slice]
    columns["flow_feature_context_ready"] = (
        current["core_context_ready"].fillna(False).to_numpy(bool)
        & np.isfinite(price[output_slice])
        & max_complete
    )
    summary = {
        "rows": len(current),
        "first_receive_second": int(seconds[output_slice][0]),
        "last_receive_second": int(seconds[output_slice][-1]),
        "ready_rows": int(np.asarray(columns["flow_feature_context_ready"], dtype=bool).sum()),
        "window_summary": window_summary,
    }
    return columns, outgoing, summary


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def datetime_to_epoch_seconds(series: pd.Series) -> np.ndarray:
    converted = pd.to_datetime(series, utc=True, errors="coerce")
    values_ns = converted.to_numpy(dtype="datetime64[ns]").astype("int64")
    result = values_ns.astype(float) / 1_000_000_000.0
    result[values_ns == np.iinfo(np.int64).min] = np.nan
    return result


def series_to_actual_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column in {"timestamp_utc", "available_after_utc"}:
        return datetime_to_epoch_seconds(frame[column])
    series = frame[column]
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False).to_numpy(bool)
    if pd.api.types.is_string_dtype(series.dtype) or series.dtype == object:
        return series.astype("string").fillna(pd.NA).to_numpy(object)
    return pd.to_numeric(series, errors="coerce").to_numpy(float)


def compare_column(
    day: date,
    seconds: np.ndarray,
    column: str,
    expected: Any,
    actual_frame: pd.DataFrame,
    issues: list[Issue],
    reconciliations: list[dict[str, Any]],
    *,
    atol: float = 1e-9,
    rtol: float = 1e-10,
) -> None:
    if column not in actual_frame.columns:
        add_issue(issues, "FAIL", "COLUMN_MISSING", day, f"Feature output is missing {column}", column_name=column)
        reconciliations.append({
            "receive_date": day.isoformat(), "column_name": column, "status": "FAIL",
            "rows": len(seconds), "mismatch_count": len(seconds), "max_abs_error": "", "max_rel_error": "",
        })
        return
    expected_array = np.asarray(expected)
    actual = series_to_actual_array(actual_frame, column)
    if len(actual) != len(expected_array):
        add_issue(
            issues, "FAIL", "COLUMN_LENGTH", day,
            f"{column} length differs ({len(actual)} != {len(expected_array)})",
            column_name=column, mismatch_count=abs(len(actual) - len(expected_array)),
        )
        reconciliations.append({
            "receive_date": day.isoformat(), "column_name": column, "status": "FAIL",
            "rows": len(seconds), "mismatch_count": abs(len(actual) - len(expected_array)),
            "max_abs_error": "", "max_rel_error": "",
        })
        return

    if expected_array.dtype.kind in {"U", "S", "O"}:
        exp = pd.Series(expected_array, dtype="string")
        act = pd.Series(actual, dtype="string")
        equal = (exp.isna() & act.isna()) | exp.eq(act)
        mismatch = ~equal.to_numpy(bool)
        max_abs = max_rel = None
    elif expected_array.dtype.kind == "b":
        exp = expected_array.astype(bool)
        act = np.asarray(actual, dtype=bool)
        mismatch = exp != act
        max_abs = max_rel = None
    else:
        exp = expected_array.astype(float)
        act = np.asarray(actual, dtype=float)
        equal = np.isclose(exp, act, atol=atol, rtol=rtol, equal_nan=True)
        mismatch = ~equal
        both = mismatch & np.isfinite(exp) & np.isfinite(act)
        if both.any():
            abs_error = np.abs(exp[both] - act[both])
            denominator = np.maximum(np.abs(exp[both]), 1e-300)
            rel_error = abs_error / denominator
            max_abs = float(abs_error.max())
            max_rel = float(rel_error.max())
        else:
            max_abs = max_rel = None

    mismatch_count = int(mismatch.sum())
    status = "PASS" if mismatch_count == 0 else "FAIL"
    reconciliations.append({
        "receive_date": day.isoformat(),
        "column_name": column,
        "status": status,
        "rows": len(seconds),
        "mismatch_count": mismatch_count,
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
    })
    if mismatch_count:
        first = int(np.flatnonzero(mismatch)[0])
        add_issue(
            issues,
            "FAIL",
            "FORMULA_MISMATCH",
            day,
            f"{column} differs from the independently reconstructed value",
            column_name=column,
            mismatch_count=mismatch_count,
            first_receive_second=int(seconds[first]) if first < len(seconds) else None,
            expected_value=expected_array[first],
            actual_value=actual[first],
            max_abs_error=max_abs,
            max_rel_error=max_rel,
        )


# ---------------------------------------------------------------------------
# State and per-date audit
# ---------------------------------------------------------------------------


def open_state_readonly(path: Path) -> sqlite3.Connection | None:
    if not path.exists():
        return None
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def read_state_rows(connection: sqlite3.Connection | None, symbol: str) -> dict[date, sqlite3.Row]:
    if connection is None:
        return {}
    try:
        rows = connection.execute(
            "SELECT * FROM date_builds WHERE symbol=? ORDER BY receive_date",
            (symbol,),
        ).fetchall()
    except sqlite3.DatabaseError:
        return {}
    return {datetime.strptime(row["receive_date"], "%Y-%m-%d").date(): row for row in rows}


def audit_date(
    day: date,
    core_root: Path,
    feature_root: Path,
    core_partitions: Mapping[date, DatePartition],
    feature_partitions: Mapping[date, DatePartition],
    all_core_days: Sequence[date],
    state_row: sqlite3.Row | None,
    symbol: str,
    config: FeatureConfig,
    incoming_carry: CarryState,
    issues: list[Issue],
    reconciliations: list[dict[str, Any]],
    state_rows_out: list[dict[str, Any]],
    window_rows: list[dict[str, Any]],
) -> tuple[CarryState, dict[str, Any]]:
    core_partition = core_partitions.get(day)
    feature_partition = feature_partitions.get(day)
    summary: dict[str, Any] = {
        "receive_date": day.isoformat(),
        "status": "PASS",
        "core_files": len(core_partition.files) if core_partition else 0,
        "feature_files": len(feature_partition.files) if feature_partition else 0,
        "core_rows": 0,
        "feature_rows": 0,
        "ready_rows": 0,
        "not_ready_rows": 0,
        "formula_mismatch_columns": 0,
        "failure_issues": 0,
        "warning_issues": 0,
        "first_receive_second": None,
        "last_receive_second": None,
    }
    issue_start = len(issues)

    if core_partition is None:
        add_issue(issues, "FAIL", "CORE_DATE_MISSING", day, "Feature/state date has no core partition")
        summary["status"] = "FAIL"
        return incoming_carry, summary
    if feature_partition is None:
        add_issue(issues, "FAIL", "FEATURE_DATE_MISSING", day, "Core/state date has no feature partition")
        summary["status"] = "FAIL"
        return incoming_carry, summary
    if not feature_partition.files:
        add_issue(issues, "FAIL", "FEATURE_FILE_MISSING", day, "Feature date directory contains no Parquet file")
        summary["status"] = "FAIL"
        return incoming_carry, summary
    if len(feature_partition.files) != 1:
        add_issue(
            issues, "FAIL", "FEATURE_FILE_COUNT", day,
            f"Expected exactly one feature Parquet file, found {len(feature_partition.files)}",
        )

    context_days = context_days_for(day, all_core_days, config.max_window)
    context = read_core_context(core_partitions, context_days)
    core_current = context.loc[
        (pd.to_numeric(context["receive_second"], errors="raise") >= day_start_second(day))
        & (pd.to_numeric(context["receive_second"], errors="raise") < day_start_second(day) + SECONDS_PER_DAY)
    ].reset_index(drop=True)
    summary["core_rows"] = len(core_current)

    if core_current["receive_second"].duplicated().any():
        add_issue(issues, "FAIL", "CORE_DUPLICATE_SECOND", day, "Core current date contains duplicate seconds")
    if not core_current["receive_second"].is_monotonic_increasing:
        add_issue(issues, "FAIL", "CORE_ORDER", day, "Core current date is not strictly ordered")

    actual, schema, physical_rows, row_groups = read_feature_partition(feature_partition)
    summary["feature_rows"] = len(actual)
    if not actual.empty:
        summary["first_receive_second"] = int(pd.to_numeric(actual["receive_second"], errors="coerce").iloc[0])
        summary["last_receive_second"] = int(pd.to_numeric(actual["receive_second"], errors="coerce").iloc[-1])
    if physical_rows != len(actual):
        add_issue(
            issues, "FAIL", "PHYSICAL_ROW_COUNT", day,
            f"Parquet metadata rows differ from loaded rows ({physical_rows} != {len(actual)})",
        )
    if len(actual) != len(core_current):
        add_issue(
            issues, "FAIL", "ROW_COUNT", day,
            f"Feature rows differ from core rows ({len(actual)} != {len(core_current)})",
        )

    expected_schema = expected_feature_schema(config)
    if not semantic_schema_equal(schema.remove_metadata(), expected_schema):
        add_issue(
            issues, "FAIL", "OUTPUT_SCHEMA", day,
            "Feature Arrow schema differs from the expected v1 contract",
            expected_value=str(expected_schema), actual_value=str(schema.remove_metadata()),
        )

    metadata = metadata_text(schema)
    expected_metadata = {
        "perps_flow_feature_schema_version": str(EXPECTED_FEATURE_SCHEMA_VERSION),
        "perps_flow_feature_builder_version": EXPECTED_FEATURE_BUILDER_VERSION,
        "source_core_schema_version": str(EXPECTED_CORE_SCHEMA_VERSION),
        "symbol": symbol,
        "receive_date": day.isoformat(),
        "causal_clock": "receive_second",
        "availability_rule": "available_after_second=receive_second+1",
        "windows_seconds": compact_json(list(config.windows)),
        "zscore_windows_seconds": compact_json(list(config.zscore_windows)),
        "zscore_min_fraction": str(config.zscore_min_fraction),
        "min_impact_flow_notional": str(config.min_impact_flow_notional),
    }
    for key, expected_value in expected_metadata.items():
        if metadata.get(key) != expected_value:
            add_issue(
                issues, "FAIL", "METADATA_VALUE", day,
                f"Metadata {key} differs from the expected contract",
                column_name=key, expected_value=expected_value, actual_value=metadata.get(key),
            )

    manifests = current_context_manifest(core_root, core_partitions, all_core_days, day, config.max_window)
    input_signature = manifest_signature(manifests)
    metadata_manifest = load_json(metadata.get("input_manifest_json"), [])
    if metadata_manifest != manifests:
        add_issue(issues, "FAIL", "INPUT_MANIFEST", day, "Parquet input manifest is stale or different from current core context")
    if metadata.get("input_signature") != input_signature:
        add_issue(
            issues, "FAIL", "INPUT_SIGNATURE", day,
            "Parquet input signature differs from current core context",
            expected_value=input_signature, actual_value=metadata.get("input_signature"),
        )
    if metadata.get("carry_in_signature") != incoming_carry.signature():
        add_issue(
            issues, "FAIL", "CARRY_IN_SIGNATURE", day,
            "Parquet carry-in signature differs from independently reconstructed prior carry",
            expected_value=incoming_carry.signature(), actual_value=metadata.get("carry_in_signature"),
        )

    if state_row is None:
        add_issue(issues, "FAIL", "STATE_DATE_MISSING", day, "Feature output has no date_builds state row")
        state_rows_out.append({
            "receive_date": day.isoformat(), "status": "MISSING", "builder_version": "",
            "output_rows": "", "input_signature_match": False, "carry_in_signature_match": False,
            "carry_out_signature_match": False, "output_path_match": False,
        })
    else:
        expected_relative = feature_partition.files[0].relative_to(feature_root).as_posix()
        state_final = load_json(state_row["final_carry_json"], {})
        state_rows_out.append({
            "receive_date": day.isoformat(),
            "status": state_row["status"],
            "builder_version": state_row["builder_version"],
            "output_rows": state_row["output_rows"],
            "input_signature_match": state_row["input_signature"] == input_signature,
            "carry_in_signature_match": state_row["carry_in_signature"] == incoming_carry.signature(),
            "carry_out_signature_match": "pending",
            "output_path_match": state_row["output_relative_path"] == expected_relative,
        })
        if state_row["status"] != "COMPLETE":
            add_issue(issues, "FAIL", "STATE_STATUS", day, f"State status is {state_row['status']!r}, expected COMPLETE")
        if state_row["builder_version"] != EXPECTED_FEATURE_BUILDER_VERSION:
            add_issue(
                issues, "FAIL", "STATE_BUILDER_VERSION", day,
                "State builder version differs from expected",
                expected_value=EXPECTED_FEATURE_BUILDER_VERSION, actual_value=state_row["builder_version"],
            )
        if int(state_row["output_schema_version"]) != EXPECTED_FEATURE_SCHEMA_VERSION:
            add_issue(issues, "FAIL", "STATE_SCHEMA_VERSION", day, "State output schema version differs")
        if state_row["config_signature"] != config.signature():
            add_issue(issues, "FAIL", "STATE_CONFIG_SIGNATURE", day, "State configuration signature differs")
        if state_row["input_signature"] != input_signature:
            add_issue(issues, "FAIL", "STATE_INPUT_SIGNATURE", day, "State input signature is stale")
        if load_json(state_row["input_manifest_json"], []) != manifests:
            add_issue(issues, "FAIL", "STATE_INPUT_MANIFEST", day, "State input manifest is stale")
        if state_row["carry_in_signature"] != incoming_carry.signature():
            add_issue(issues, "FAIL", "STATE_CARRY_IN", day, "State carry-in signature differs")
        if int(state_row["output_rows"]) != len(actual):
            add_issue(issues, "FAIL", "STATE_OUTPUT_ROWS", day, "State output row count differs from physical output")
        if state_row["output_relative_path"] != expected_relative:
            add_issue(issues, "FAIL", "STATE_OUTPUT_PATH", day, "State output path differs from physical output")
        if state_final and not {"cvd_quote", "cvd_base"}.issubset(state_final):
            add_issue(issues, "FAIL", "STATE_FINAL_CARRY", day, "State final carry is incomplete")

    expected_columns, outgoing_carry, expected_summary = expected_feature_columns(
        context=context,
        output_day=day,
        incoming_carry=incoming_carry,
        config=config,
    )
    seconds = np.asarray(expected_columns["receive_second"], dtype=np.int64)
    if len(actual) == len(seconds):
        for column in expected_feature_schema(config).names:
            expected = expected_columns[column]
            if column in {
                "feature_schema_version", "receive_second", "available_after_second",
                "seconds_from_day_start", "core_schema_version", "price_source_code",
                "trade_count_1s", "liquidation_event_count_1s",
            } or column.startswith("window_observation_count_"):
                compare_column(day, seconds, column, expected, actual, issues, reconciliations, atol=0.0, rtol=0.0)
            elif column == "symbol" or column.endswith("_complete_5s"):
                compare_column(day, seconds, column, expected, actual, issues, reconciliations, atol=0.0, rtol=0.0)
            elif column.startswith("window_complete_") or column.startswith("return_lag_available_") or column in {
                "core_context_ready", "price_fresh", "open_interest_updated_1s", "flow_feature_context_ready",
            }:
                compare_column(day, seconds, column, expected, actual, issues, reconciliations, atol=0.0, rtol=0.0)
            elif column.startswith("price_flow_alignment_") or column.startswith("trade_count_"):
                compare_column(day, seconds, column, expected, actual, issues, reconciliations, atol=0.0, rtol=0.0)
            else:
                compare_column(day, seconds, column, expected, actual, issues, reconciliations, atol=1e-8, rtol=1e-10)

    if metadata.get("carry_out_signature") != outgoing_carry.signature():
        add_issue(
            issues, "FAIL", "CARRY_OUT_SIGNATURE", day,
            "Parquet carry-out signature differs from independently reconstructed CVD carry",
            expected_value=outgoing_carry.signature(), actual_value=metadata.get("carry_out_signature"),
        )
    if state_row is not None:
        state_final = load_json(state_row["final_carry_json"], {})
        state_carry = CarryState(float(state_final.get("cvd_quote", 0.0)), float(state_final.get("cvd_base", 0.0)))
        state_rows_out[-1]["carry_out_signature_match"] = state_carry.signature() == outgoing_carry.signature()
        if state_carry.signature() != outgoing_carry.signature():
            add_issue(issues, "FAIL", "STATE_CARRY_OUT", day, "State final carry differs from independently reconstructed CVD carry")
        state_summary = load_json(state_row["summary_json"], {})
        if state_summary:
            if int(state_summary.get("output_rows", -1)) != len(actual):
                add_issue(issues, "FAIL", "STATE_SUMMARY_ROWS", day, "State summary output_rows differs")
            if int(state_summary.get("flow_feature_context_ready_rows", -1)) != expected_summary["ready_rows"]:
                add_issue(issues, "FAIL", "STATE_SUMMARY_READY", day, "State summary ready-row count differs")

    if "flow_feature_context_ready" in actual.columns:
        ready = actual["flow_feature_context_ready"].fillna(False).to_numpy(bool)
        summary["ready_rows"] = int(ready.sum())
        summary["not_ready_rows"] = int((~ready).sum())
    for window in config.windows:
        suffix = f"{window}s"
        complete_column = f"window_complete_{suffix}"
        if complete_column in actual.columns:
            complete = actual[complete_column].fillna(False).to_numpy(bool)
            window_rows.append({
                "receive_date": day.isoformat(),
                "window_seconds": window,
                "rows": len(actual),
                "complete_rows": int(complete.sum()),
                "incomplete_rows": int((~complete).sum()),
                "expected_complete_rows": expected_summary["window_summary"][suffix]["complete_rows"],
                "return_ready_rows": expected_summary["window_summary"][suffix]["return_ready_rows"],
                "realized_vol_ready_rows": expected_summary["window_summary"][suffix]["realized_vol_ready_rows"],
                "impact_ready_rows": expected_summary["window_summary"][suffix]["impact_ready_rows"],
            })

    date_issues = issues[issue_start:]
    failures = sum(item.severity == "FAIL" for item in date_issues)
    warnings = sum(item.severity == "WARN" for item in date_issues)
    summary["failure_issues"] = failures
    summary["warning_issues"] = warnings
    summary["formula_mismatch_columns"] = sum(item.code == "FORMULA_MISMATCH" for item in date_issues)
    summary["status"] = "FAIL" if failures else ("WARN" if warnings else "PASS")

    # Correctly handled inherited grid discontinuities are informational only.
    if summary["not_ready_rows"] and day != all_core_days[0]:
        max_complete_expected = np.asarray(expected_columns["flow_feature_context_ready"], dtype=bool)
        if (~max_complete_expected).any():
            add_issue(
                issues,
                "INFO",
                "CONTEXT_NOT_READY",
                day,
                f"{int((~max_complete_expected).sum()):,} rows are correctly marked not ready because core/price/3600s context is incomplete",
                mismatch_count=int((~max_complete_expected).sum()),
            )
    return outgoing_carry, summary


# ---------------------------------------------------------------------------
# Findings and invariants
# ---------------------------------------------------------------------------


def build_invariants(issues: Sequence[Issue], date_rows: Sequence[Mapping[str, Any]]) -> list[Invariant]:
    groups = [
        ("Date ownership and physical outputs", {"CORE_DATE_MISSING", "FEATURE_DATE_MISSING", "FEATURE_FILE_MISSING", "FEATURE_FILE_COUNT", "PHYSICAL_ROW_COUNT", "ROW_COUNT"}),
        ("Arrow schema and metadata", {"OUTPUT_SCHEMA", "METADATA_REQUIRED", "METADATA_VALUE"}),
        ("Build state and input freshness", {"STATE_DATE_MISSING", "STATE_STATUS", "STATE_BUILDER_VERSION", "STATE_SCHEMA_VERSION", "STATE_CONFIG_SIGNATURE", "STATE_INPUT_SIGNATURE", "STATE_INPUT_MANIFEST", "STATE_OUTPUT_ROWS", "STATE_OUTPUT_PATH", "STATE_SUMMARY_ROWS", "STATE_SUMMARY_READY", "INPUT_MANIFEST", "INPUT_SIGNATURE"}),
        ("Cumulative carry continuity", {"CARRY_IN_SIGNATURE", "CARRY_OUT_SIGNATURE", "STATE_CARRY_IN", "STATE_CARRY_OUT", "STATE_FINAL_CARRY"}),
        ("Exact row alignment and causal timestamps", {"CORE_DUPLICATE_SECOND", "CORE_ORDER", "COLUMN_MISSING", "COLUMN_LENGTH"}),
        ("Independent feature formulas", {"FORMULA_MISMATCH"}),
    ]
    invariants: list[Invariant] = []
    for name, codes in groups:
        relevant = [item for item in issues if item.code in codes]
        failures = sum(item.severity == "FAIL" for item in relevant)
        warnings = sum(item.severity == "WARN" for item in relevant)
        status = "FAIL" if failures else ("WARN" if warnings else "PASS")
        invariants.append(Invariant(name, status, failures, warnings, f"{len(relevant)} issue records"))
    failed_dates = sum(row.get("status") == "FAIL" for row in date_rows)
    invariants.append(
        Invariant(
            "All audited dates pass",
            "FAIL" if failed_dates else "PASS",
            failed_dates,
            0,
            f"{len(date_rows) - failed_dates}/{len(date_rows)} dates passed",
        )
    )
    return invariants


def write_findings(
    path: Path,
    summary: Mapping[str, Any],
    invariants: Sequence[Invariant],
    issues: Sequence[Issue],
) -> None:
    lines = [
        "# Perps Flow-Feature Audit Findings",
        "",
        f"- **Status:** {summary['status']}",
        f"- **Run ID:** `{summary['run_id']}`",
        f"- **Auditor version:** `{summary['auditor_version']}`",
        f"- **Expected builder version:** `{summary['expected_builder_version']}`",
        f"- **Symbol:** `{summary['symbol']}`",
        f"- **Date range:** `{summary['date_start']}` through `{summary['date_end']}`",
        f"- **Dates audited:** {summary['dates_audited']:,}",
        f"- **Feature rows audited:** {summary['feature_rows']:,}",
        f"- **Context-ready rows:** {summary['ready_rows']:,}",
        f"- **Failure issues:** {summary['failure_issues']:,}",
        f"- **Warning issues:** {summary['warning_issues']:,}",
        "",
        "## Invariants",
        "",
        "| Invariant | Status | Failures | Warnings | Details |",
        "|---|---:|---:|---:|---|",
    ]
    for item in invariants:
        lines.append(f"| {item.invariant} | {item.status} | {item.failures} | {item.warnings} | {item.details} |")
    actionable = [item for item in issues if item.severity in {"FAIL", "WARN"}]
    lines.extend(["", "## Actionable issues", ""])
    if not actionable:
        lines.append("No failures or warnings were found.")
    else:
        for item in actionable[:100]:
            location = f" on {item.receive_date}" if item.receive_date else ""
            column = f" [{item.column_name}]" if item.column_name else ""
            lines.append(f"- **{item.severity} {item.code}**{location}{column}: {item.message}")
        if len(actionable) > 100:
            lines.append(f"- {len(actionable) - 100} additional issues are available in `perps_flow_feature_audit_issues.csv`.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A PASS means the stored feature layer matches an independent reconstruction from the core layer, including cross-day context and cumulative CVD. It also means the current core input manifests still match those used by the builder. Informational context-not-ready observations do not indicate corruption; they document rows that are deliberately excluded because required historical context is incomplete.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Independently audit the perps flow-feature one-second layer.")
    parser.add_argument("--core-root", default="./data/derived/perps/core_1s_v1")
    parser.add_argument("--feature-root", default="./data/derived/perps/flow_features_1s_v1")
    parser.add_argument("--reports-root", default="./reports/perps_flow_features_audit")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--fail-on-warnings", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    require_arrow()
    args = build_parser().parse_args(argv)
    start_date = parse_date(args.start_date, "--start-date")
    end_date = parse_date(args.end_date, "--end-date")
    if start_date and end_date and start_date > end_date:
        raise SystemExit("--start-date cannot be after --end-date")
    if args.progress_every <= 0:
        raise SystemExit("--progress-every must be positive")

    core_root = Path(args.core_root).expanduser().resolve()
    feature_root = Path(args.feature_root).expanduser().resolve()
    reports_root = Path(args.reports_root).expanduser().resolve()
    symbol = args.symbol.strip().upper()
    if not core_root.exists():
        raise SystemExit(f"Core root does not exist: {core_root}")
    if not feature_root.exists():
        raise SystemExit(f"Feature root does not exist: {feature_root}")

    issues: list[Issue] = []
    core_partitions = discover_partitions(core_root, symbol)
    feature_partitions = discover_partitions(feature_root, symbol)
    all_feature_files = [path for partition in feature_partitions.values() for path in partition.files]
    config = infer_config(all_feature_files, issues)

    state_path = feature_root / "_build_state" / "build_state.sqlite"
    state_connection = open_state_readonly(state_path)
    state_rows = read_state_rows(state_connection, symbol)

    union_days = sorted(set(core_partitions) | set(feature_partitions) | set(state_rows))
    selected_days = [
        day for day in union_days
        if (start_date is None or day >= start_date) and (end_date is None or day <= end_date)
    ]
    if not selected_days:
        raise SystemExit("No dates remain after applying the requested date range")
    all_core_days = sorted(core_partitions)
    if not all_core_days:
        raise SystemExit("No core date partitions were found")

    run_identifier = create_run_id()
    run_root = reports_root / "runs" / run_identifier
    run_root.mkdir(parents=True, exist_ok=True)

    print("Auditing perps flow-feature one-second layer...")
    print(f"Run ID          : {run_identifier}")
    print(f"Core root       : {core_root}")
    print(f"Feature root    : {feature_root}")
    print(f"State database  : {state_path}")
    print(f"Reports root    : {run_root}")
    print(f"Symbol          : {symbol}")
    print(f"Date range      : {selected_days[0]} to {selected_days[-1]} ({len(selected_days)} dates)")
    print(f"Windows         : {','.join(map(str, config.windows))}")
    print(f"Z-score windows : {','.join(map(str, config.zscore_windows))}")

    # Independent CVD carry before a filtered start date.
    incoming_carry = CarryState()
    prior_days = [day for day in all_core_days if day < selected_days[0]]
    for prior_day in prior_days:
        prior = read_core_partition(core_partitions[prior_day])
        prior_quote = pd.to_numeric(
            prior["signed_quote_flow"], errors="coerce"
        ).fillna(0).to_numpy(float)
        prior_base = pd.to_numeric(
            prior["signed_base_flow"], errors="coerce"
        ).fillna(0).to_numpy(float)
        # Match the builder's sequential cumulative-sum order exactly. A plain
        # reduction can produce a different final floating-point bit pattern.
        quote_end = (
            incoming_carry.cvd_quote
            if len(prior_quote) == 0
            else incoming_carry.cvd_quote + float(np.cumsum(prior_quote, dtype=float)[-1])
        )
        base_end = (
            incoming_carry.cvd_base
            if len(prior_base) == 0
            else incoming_carry.cvd_base + float(np.cumsum(prior_base, dtype=float)[-1])
        )
        incoming_carry = CarryState(quote_end, base_end)

    date_rows: list[dict[str, Any]] = []
    reconciliations: list[dict[str, Any]] = []
    state_reconciliation: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    fatal_error: str | None = None

    try:
        for position, day in enumerate(selected_days, start=1):
            incoming_carry, day_summary = audit_date(
                day=day,
                core_root=core_root,
                feature_root=feature_root,
                core_partitions=core_partitions,
                feature_partitions=feature_partitions,
                all_core_days=all_core_days,
                state_row=state_rows.get(day),
                symbol=symbol,
                config=config,
                incoming_carry=incoming_carry,
                issues=issues,
                reconciliations=reconciliations,
                state_rows_out=state_reconciliation,
                window_rows=window_rows,
            )
            date_rows.append(day_summary)
            if position % args.progress_every == 0 or position == len(selected_days):
                print(
                    f"AUDITED   [{position:3d}/{len(selected_days)}] {day}: "
                    f"rows={day_summary['feature_rows']:,}, "
                    f"fails={day_summary['failure_issues']}, "
                    f"warns={day_summary['warning_issues']}"
                )
    except Exception as exc:
        fatal_error = f"{type(exc).__name__}: {exc}"
        add_issue(issues, "FAIL", "AUDITOR_EXCEPTION", None, fatal_error)
        traceback.print_exc()
    finally:
        if state_connection is not None:
            state_connection.close()

    invariants = build_invariants(issues, date_rows)
    failure_issues = sum(item.severity == "FAIL" for item in issues)
    warning_issues = sum(item.severity == "WARN" for item in issues)
    status = "FAIL" if failure_issues else ("WARN" if warning_issues else "PASS")
    feature_rows = sum(int(row.get("feature_rows", 0)) for row in date_rows)
    ready_rows = sum(int(row.get("ready_rows", 0)) for row in date_rows)
    summary = {
        "status": status,
        "run_id": run_identifier,
        "started_or_completed_at_utc": utc_now_iso(),
        "auditor_version": AUDITOR_VERSION,
        "expected_builder_version": EXPECTED_FEATURE_BUILDER_VERSION,
        "expected_feature_schema_version": EXPECTED_FEATURE_SCHEMA_VERSION,
        "symbol": symbol,
        "core_root": str(core_root),
        "feature_root": str(feature_root),
        "state_database": str(state_path),
        "date_start": selected_days[0].isoformat(),
        "date_end": selected_days[-1].isoformat(),
        "dates_audited": len(date_rows),
        "feature_rows": feature_rows,
        "ready_rows": ready_rows,
        "not_ready_rows": feature_rows - ready_rows,
        "failure_issues": failure_issues,
        "warning_issues": warning_issues,
        "informational_issues": sum(item.severity == "INFO" for item in issues),
        "configuration": asdict(config),
        "fatal_error": fatal_error,
    }

    filenames = [
        "perps_flow_feature_audit_findings.md",
        "perps_flow_feature_audit_summary.json",
        "perps_flow_feature_audit_invariants.csv",
        "perps_flow_feature_audit_issues.csv",
        "perps_flow_feature_date_summary.csv",
        "perps_flow_feature_column_reconciliation.csv",
        "perps_flow_feature_state_reconciliation.csv",
        "perps_flow_feature_window_summary.csv",
    ]
    write_findings(run_root / filenames[0], summary, invariants, issues)
    write_json(run_root / filenames[1], summary)
    write_csv(run_root / filenames[2], [item.row() for item in invariants], ["invariant", "status", "failures", "warnings", "details"])
    write_csv(
        run_root / filenames[3],
        [item.row() for item in issues],
        [
            "severity", "code", "receive_date", "column_name", "message", "mismatch_count",
            "first_receive_second", "expected_value", "actual_value", "max_abs_error", "max_rel_error",
        ],
    )
    write_csv(run_root / filenames[4], date_rows)
    write_csv(run_root / filenames[5], reconciliations)
    write_csv(run_root / filenames[6], state_reconciliation)
    write_csv(run_root / filenames[7], window_rows)
    copy_latest(run_root, reports_root, filenames)

    print("\nPerps flow-feature audit finished.")
    print(f"Status           : {status}")
    print(f"Dates audited    : {len(date_rows):,}")
    print(f"Feature rows     : {feature_rows:,}")
    print(f"Context-ready    : {ready_rows:,}")
    print(f"Failure issues   : {failure_issues:,}")
    print(f"Warning issues   : {warning_issues:,}")
    print(f"Findings         : {run_root / filenames[0]}")
    print(f"Latest reports   : {reports_root}")

    if failure_issues:
        return 1
    if warning_issues and args.fail_on_warnings:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
