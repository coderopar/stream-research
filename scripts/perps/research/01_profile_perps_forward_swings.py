#!/usr/bin/env python
"""
Profile causal forward BTC perpetual price swings across multiple horizons.

This script consumes the audited perps flow-feature layer and answers:

* What is the average high-low swing over 5, 10, 15, 20, 25 and 30 minutes?
* How are those swings distributed in 5-basis-point buckets through 50 bps?
* How often does each horizon reach at least 5, 10, ..., 50 bps?
* What are the corresponding upside, downside and maximum-absolute excursions?

Causal window contract
----------------------
A feature row at decision second ``t`` becomes usable after second ``t``. The
executable entry is therefore the exact feature price at ``t + 1``. For a
horizon of H seconds, the future path includes every exact second from
``t + 1`` through ``t + 1 + H`` inclusive. Consequently, a 15-minute horizon
contains 901 price observations spanning exactly 900 elapsed seconds.

The primary swing metric is:

    high_low_swing_bps = (future_high - future_low) / entry_price * 10,000

Only windows with every required second present and every path price finite are
included. By default, anchors must also have ``flow_feature_context_ready`` set,
so the resulting opportunity profile matches rows usable by downstream flow
research.

Sampling modes
--------------
Two profiles are emitted from the same calculations:

* ``rolling_1s``: every eligible decision second. This is the dense opportunity
  distribution, but neighboring samples overlap heavily.
* ``utc_non_overlapping``: entry seconds aligned to UTC multiples of the
  horizon. This provides a lower-correlation robustness view.

The script writes aggregate reports only; it does not materialize tens of
millions of window-level rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - handled at runtime
    pa = None
    pq = None


PROFILER_VERSION = "1.0.0"
EXPECTED_FEATURE_SCHEMA_VERSION = 1
EXPECTED_BUILDER_VERSION = "1.0.2"
DEFAULT_HORIZONS_MINUTES = (5, 10, 15, 20, 25, 30)
DEFAULT_BUCKET_STEP_BPS = 5.0
DEFAULT_BUCKET_MAX_BPS = 50.0
DATE_PARTITION_PREFIX = "date="
SYMBOL_PARTITION_PREFIX = "symbol="
SECONDS_PER_DAY = 86_400

REQUIRED_COLUMNS = (
    "feature_schema_version",
    "symbol",
    "receive_second",
    "price_for_features",
    "flow_feature_context_ready",
)

PRIMARY_METRIC = "high_low_swing_bps"
METRICS = (
    "high_low_swing_bps",
    "up_excursion_bps",
    "down_excursion_bps",
    "max_abs_excursion_bps",
    "terminal_abs_move_bps",
)


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


def parse_int_list(value: str, label: str) -> tuple[int, ...]:
    parsed: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            number = int(token)
        except ValueError as exc:
            raise SystemExit(f"Invalid integer in {label}: {token}") from exc
        if number <= 0:
            raise SystemExit(f"All {label} values must be positive: {number}")
        parsed.append(number)
    output = tuple(sorted(set(parsed)))
    if not output:
        raise SystemExit(f"{label} cannot be empty")
    return output


def parse_date(value: str | None, label: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise SystemExit(f"{label} must use YYYY-MM-DD format: {value}") from exc


def compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    frame = pd.DataFrame(list(rows), columns=list(columns))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def percentile(values: np.ndarray, q: float) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.quantile(values, q, method="linear"))


def safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def safe_std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=0)) if len(values) else float("nan")


def percent(numerator: int | float, denominator: int | float) -> float:
    return 100.0 * float(numerator) / float(denominator) if denominator else float("nan")


# ---------------------------------------------------------------------------
# Input discovery and validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InputPartition:
    receive_date: date
    path: Path
    rows: int
    minimum_second: int
    maximum_second: int
    builder_version: str | None
    schema_version: int | None


def decode_metadata(schema: "pa.Schema") -> dict[str, str]:
    metadata = schema.metadata or {}
    return {
        key.decode("utf-8", errors="replace"): value.decode("utf-8", errors="replace")
        for key, value in metadata.items()
    }


def parquet_min_max_int(parquet_file: "pq.ParquetFile", column_name: str) -> tuple[int, int]:
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

    table = parquet_file.read(columns=[column_name])
    values = table.column(column_name).to_numpy(zero_copy_only=False)
    if len(values) == 0:
        raise RuntimeError(f"Partition has no values in {column_name}")
    return int(np.min(values)), int(np.max(values))


def discover_partitions(
    feature_root: Path,
    symbol: str,
    start_date: date | None,
    end_date: date | None,
    expected_builder_version: str,
) -> list[InputPartition]:
    symbol_root = feature_root / f"{SYMBOL_PARTITION_PREFIX}{symbol}"
    if not symbol_root.exists():
        raise SystemExit(f"Feature symbol directory does not exist: {symbol_root}")

    partitions: list[InputPartition] = []
    for date_dir in sorted(symbol_root.glob(f"{DATE_PARTITION_PREFIX}*")):
        if not date_dir.is_dir():
            continue
        raw_date = date_dir.name[len(DATE_PARTITION_PREFIX) :]
        try:
            receive_date = date.fromisoformat(raw_date)
        except ValueError:
            continue
        if start_date is not None and receive_date < start_date:
            continue
        if end_date is not None and receive_date > end_date:
            continue

        files = sorted(path for path in date_dir.glob("*.parquet") if path.is_file())
        if len(files) != 1:
            raise SystemExit(
                f"Expected exactly one Parquet output for {receive_date}, found {len(files)}: "
                f"{date_dir}"
            )
        path = files[0]
        parquet_file = pq.ParquetFile(path)
        try:
            names = parquet_file.schema_arrow.names
            missing = [column for column in REQUIRED_COLUMNS if column not in names]
            if missing:
                raise SystemExit(f"{path} is missing required columns: {missing}")

            metadata = decode_metadata(parquet_file.schema_arrow)
            builder_version = metadata.get("perps_flow_feature_builder_version")
            schema_raw = metadata.get("perps_flow_feature_schema_version")
            schema_version = int(schema_raw) if schema_raw is not None else None

            if expected_builder_version and builder_version != expected_builder_version:
                raise SystemExit(
                    f"{path} builder version is {builder_version!r}; expected "
                    f"{expected_builder_version!r}"
                )
            if schema_version != EXPECTED_FEATURE_SCHEMA_VERSION:
                raise SystemExit(
                    f"{path} schema version is {schema_version!r}; expected "
                    f"{EXPECTED_FEATURE_SCHEMA_VERSION}"
                )

            minimum_second, maximum_second = parquet_min_max_int(
                parquet_file, "receive_second"
            )
            partitions.append(
                InputPartition(
                    receive_date=receive_date,
                    path=path,
                    rows=int(parquet_file.metadata.num_rows),
                    minimum_second=minimum_second,
                    maximum_second=maximum_second,
                    builder_version=builder_version,
                    schema_version=schema_version,
                )
            )
        finally:
            close = getattr(parquet_file, "close", None)
            if callable(close):
                close()

    if not partitions:
        raise SystemExit("No feature partitions matched the requested symbol/date range")
    return partitions


def load_feature_arrays(
    partitions: Sequence[InputPartition], symbol: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seconds_parts: list[np.ndarray] = []
    price_parts: list[np.ndarray] = []
    ready_parts: list[np.ndarray] = []
    date_parts: list[np.ndarray] = []

    for index, partition in enumerate(partitions, start=1):
        parquet_file = pq.ParquetFile(partition.path)
        try:
            table = parquet_file.read(columns=list(REQUIRED_COLUMNS))
        finally:
            close = getattr(parquet_file, "close", None)
            if callable(close):
                close()

        frame = table.to_pandas()
        if len(frame) != partition.rows:
            raise RuntimeError(
                f"Row-count mismatch while reading {partition.path}: "
                f"metadata={partition.rows}, read={len(frame)}"
            )

        versions = pd.to_numeric(frame["feature_schema_version"], errors="coerce")
        if versions.isna().any() or not bool(
            (versions == EXPECTED_FEATURE_SCHEMA_VERSION).all()
        ):
            raise RuntimeError(f"Unexpected feature_schema_version in {partition.path}")

        symbols = frame["symbol"].astype(str)
        if not bool((symbols == symbol).all()):
            bad = sorted(symbols[symbols != symbol].unique().tolist())[:5]
            raise RuntimeError(f"Unexpected symbols in {partition.path}: {bad}")

        seconds = pd.to_numeric(frame["receive_second"], errors="raise").to_numpy(
            dtype=np.int64
        )
        prices = pd.to_numeric(frame["price_for_features"], errors="coerce").to_numpy(
            dtype=float
        )
        ready = frame["flow_feature_context_ready"].fillna(False).to_numpy(dtype=bool)

        if len(seconds) and np.any(np.diff(seconds) <= 0):
            raise RuntimeError(f"receive_second is not strictly increasing: {partition.path}")

        utc_dates = pd.to_datetime(seconds, unit="s", utc=True).date
        expected_date = partition.receive_date
        if any(value != expected_date for value in utc_dates):
            raise RuntimeError(
                f"Partition-date mismatch in {partition.path}; expected {expected_date}"
            )

        seconds_parts.append(seconds)
        price_parts.append(prices)
        ready_parts.append(ready)
        date_parts.append(
            np.full(len(seconds), expected_date.toordinal(), dtype=np.int32)
        )
        print(
            f"LOADED    [{index:>3}/{len(partitions)}] {partition.receive_date}: "
            f"rows={len(frame):,}"
        )

    seconds_all = np.concatenate(seconds_parts)
    price_all = np.concatenate(price_parts)
    ready_all = np.concatenate(ready_parts)
    date_all = np.concatenate(date_parts)

    if np.any(np.diff(seconds_all) <= 0):
        duplicate_count = int(np.sum(np.diff(seconds_all) == 0))
        raise RuntimeError(
            "Feature seconds are not globally strictly increasing; "
            f"duplicate transitions={duplicate_count:,}"
        )

    return seconds_all, price_all, ready_all, date_all


# ---------------------------------------------------------------------------
# Dense grid and forward-window calculations
# ---------------------------------------------------------------------------


@dataclass
class DenseFeatureGrid:
    minimum_second: int
    maximum_second: int
    price: np.ndarray
    observed: np.ndarray
    ready: np.ndarray

    @property
    def seconds(self) -> np.ndarray:
        return np.arange(
            self.minimum_second, self.maximum_second + 1, dtype=np.int64
        )


def make_dense_grid(
    seconds: np.ndarray, prices: np.ndarray, ready: np.ndarray
) -> DenseFeatureGrid:
    minimum_second = int(seconds[0])
    maximum_second = int(seconds[-1])
    length = maximum_second - minimum_second + 1
    positions = seconds - minimum_second

    if length > len(seconds) + 10_000_000:
        raise RuntimeError(
            "Dense grid would contain unexpectedly many absent seconds: "
            f"span={length:,}, observed={len(seconds):,}"
        )

    dense_price = np.full(length, np.nan, dtype=float)
    dense_observed = np.zeros(length, dtype=bool)
    dense_ready = np.zeros(length, dtype=bool)

    dense_price[positions] = prices
    dense_observed[positions] = True
    dense_ready[positions] = ready

    return DenseFeatureGrid(
        minimum_second=minimum_second,
        maximum_second=maximum_second,
        price=dense_price,
        observed=dense_observed,
        ready=dense_ready,
    )


def forward_extrema(values: np.ndarray, window_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Return exact forward max/min over [i, i + window_length - 1]."""
    series = pd.Series(values)
    high = (
        series.rolling(window_length, min_periods=window_length)
        .max()
        .shift(-(window_length - 1))
        .to_numpy(dtype=float)
    )
    low = (
        series.rolling(window_length, min_periods=window_length)
        .min()
        .shift(-(window_length - 1))
        .to_numpy(dtype=float)
    )
    return high, low


def normalize_nonnegative(name: str, values: np.ndarray, tolerance: float = 1e-8) -> np.ndarray:
    finite = np.isfinite(values)
    if np.any(values[finite] < -tolerance):
        minimum = float(np.min(values[finite]))
        raise RuntimeError(f"{name} contains negative values; minimum={minimum}")
    output = values.copy()
    tiny_negative = finite & (output < 0)
    output[tiny_negative] = 0.0
    return output


@dataclass
class WindowMetrics:
    anchor_seconds: np.ndarray
    entry_seconds: np.ndarray
    entry_price: np.ndarray
    terminal_price: np.ndarray
    high_low_swing_bps: np.ndarray
    up_excursion_bps: np.ndarray
    down_excursion_bps: np.ndarray
    max_abs_excursion_bps: np.ndarray
    terminal_abs_move_bps: np.ndarray

    def subset(self, mask: np.ndarray) -> "WindowMetrics":
        return WindowMetrics(
            anchor_seconds=self.anchor_seconds[mask],
            entry_seconds=self.entry_seconds[mask],
            entry_price=self.entry_price[mask],
            terminal_price=self.terminal_price[mask],
            high_low_swing_bps=self.high_low_swing_bps[mask],
            up_excursion_bps=self.up_excursion_bps[mask],
            down_excursion_bps=self.down_excursion_bps[mask],
            max_abs_excursion_bps=self.max_abs_excursion_bps[mask],
            terminal_abs_move_bps=self.terminal_abs_move_bps[mask],
        )

    def metric(self, name: str) -> np.ndarray:
        value = getattr(self, name)
        return np.asarray(value)

    def __len__(self) -> int:
        return len(self.anchor_seconds)


def calculate_window_metrics(
    grid: DenseFeatureGrid,
    horizon_seconds: int,
    require_ready: bool,
) -> tuple[WindowMetrics, dict[str, int]]:
    # A decision at t enters at t+1. The path spans entry through entry+H,
    # which is H+1 exact-second observations.
    window_length = horizon_seconds + 1
    future_high, future_low = forward_extrema(grid.price, window_length)

    anchor_positions = np.arange(len(grid.price), dtype=np.int64)
    entry_positions = anchor_positions + 1
    terminal_positions = entry_positions + horizon_seconds

    anchor_observed = grid.observed
    anchor_ready = grid.ready if require_ready else grid.observed
    within_bounds = terminal_positions < len(grid.price)

    candidate = anchor_observed & anchor_ready & within_bounds
    candidate_positions = anchor_positions[candidate]
    candidate_entry = entry_positions[candidate]
    candidate_terminal = terminal_positions[candidate]

    entry_price = grid.price[candidate_entry]
    terminal_price = grid.price[candidate_terminal]
    high = future_high[candidate_entry]
    low = future_low[candidate_entry]

    valid = (
        np.isfinite(entry_price)
        & (entry_price > 0)
        & np.isfinite(terminal_price)
        & (terminal_price > 0)
        & np.isfinite(high)
        & np.isfinite(low)
        & (high > 0)
        & (low > 0)
        & (high >= low)
    )

    anchor_positions = candidate_positions[valid]
    entry_positions = candidate_entry[valid]
    terminal_positions = candidate_terminal[valid]
    entry_price = entry_price[valid]
    terminal_price = terminal_price[valid]
    high = high[valid]
    low = low[valid]

    swing = (high - low) / entry_price * 10_000.0
    up = (high / entry_price - 1.0) * 10_000.0
    down = (1.0 - low / entry_price) * 10_000.0
    max_abs = np.maximum(up, down)
    terminal_abs = np.abs(terminal_price / entry_price - 1.0) * 10_000.0

    swing = normalize_nonnegative("high_low_swing_bps", swing)
    up = normalize_nonnegative("up_excursion_bps", up)
    down = normalize_nonnegative("down_excursion_bps", down)
    max_abs = normalize_nonnegative("max_abs_excursion_bps", max_abs)
    terminal_abs = normalize_nonnegative("terminal_abs_move_bps", terminal_abs)

    metrics = WindowMetrics(
        anchor_seconds=anchor_positions + grid.minimum_second,
        entry_seconds=entry_positions + grid.minimum_second,
        entry_price=entry_price,
        terminal_price=terminal_price,
        high_low_swing_bps=swing,
        up_excursion_bps=up,
        down_excursion_bps=down,
        max_abs_excursion_bps=max_abs,
        terminal_abs_move_bps=terminal_abs,
    )

    counts = {
        "observed_anchor_rows": int(np.sum(anchor_observed)),
        "eligible_anchor_rows": int(np.sum(anchor_observed & anchor_ready)),
        "candidate_anchor_rows": int(np.sum(candidate)),
        "valid_windows": len(metrics),
        "invalid_or_incomplete_windows": int(np.sum(candidate) - len(metrics)),
    }
    return metrics, counts


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def summary_row(
    mode: str,
    horizon_minutes: int,
    metrics: WindowMetrics,
    counts: Mapping[str, int],
) -> dict[str, Any]:
    swing = metrics.high_low_swing_bps
    up = metrics.up_excursion_bps
    down = metrics.down_excursion_bps
    max_abs = metrics.max_abs_excursion_bps
    terminal_abs = metrics.terminal_abs_move_bps

    up_dominant = int(np.sum(up > down + 1e-12))
    down_dominant = int(np.sum(down > up + 1e-12))
    balanced = len(metrics) - up_dominant - down_dominant

    row: dict[str, Any] = {
        "sampling_mode": mode,
        "horizon_minutes": horizon_minutes,
        "horizon_seconds": horizon_minutes * 60,
        "valid_windows": len(metrics),
        "invalid_or_incomplete_windows": counts.get(
            "invalid_or_incomplete_windows", 0
        ),
        "mean_swing_bps": safe_mean(swing),
        "median_swing_bps": percentile(swing, 0.50),
        "std_swing_bps": safe_std(swing),
        "p75_swing_bps": percentile(swing, 0.75),
        "p90_swing_bps": percentile(swing, 0.90),
        "p95_swing_bps": percentile(swing, 0.95),
        "p99_swing_bps": percentile(swing, 0.99),
        "maximum_swing_bps": float(np.max(swing)) if len(swing) else float("nan"),
        "mean_up_excursion_bps": safe_mean(up),
        "mean_down_excursion_bps": safe_mean(down),
        "mean_max_abs_excursion_bps": safe_mean(max_abs),
        "mean_terminal_abs_move_bps": safe_mean(terminal_abs),
        "up_dominant_windows": up_dominant,
        "down_dominant_windows": down_dominant,
        "balanced_windows": balanced,
        "up_dominant_pct": percent(up_dominant, len(metrics)),
        "down_dominant_pct": percent(down_dominant, len(metrics)),
        "balanced_pct": percent(balanced, len(metrics)),
    }
    return row


def bucket_edges(step_bps: float, maximum_bps: float) -> np.ndarray:
    if step_bps <= 0:
        raise SystemExit("--bucket-step-bps must be positive")
    if maximum_bps <= 0:
        raise SystemExit("--bucket-max-bps must be positive")
    quotient = maximum_bps / step_bps
    if not math.isclose(quotient, round(quotient), rel_tol=0, abs_tol=1e-9):
        raise SystemExit("--bucket-max-bps must be an exact multiple of --bucket-step-bps")
    finite = np.arange(0.0, maximum_bps + step_bps, step_bps, dtype=float)
    return np.concatenate((finite, [np.inf]))


def bucket_label(lower: float, upper: float) -> str:
    if math.isinf(upper):
        return f"{lower:g}+ bps"
    return f"{lower:g}-{upper:g} bps"


def distribution_rows(
    mode: str,
    horizon_minutes: int,
    metrics: WindowMetrics,
    edges: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric_name in METRICS:
        values = metrics.metric(metric_name)
        counts, _ = np.histogram(values, bins=edges)
        total = int(np.sum(counts))
        cumulative = 0
        for index, count in enumerate(counts):
            lower = float(edges[index])
            upper = float(edges[index + 1])
            cumulative += int(count)
            rows.append(
                {
                    "sampling_mode": mode,
                    "horizon_minutes": horizon_minutes,
                    "horizon_seconds": horizon_minutes * 60,
                    "metric": metric_name,
                    "bucket_order": index + 1,
                    "bucket_lower_bps": lower,
                    "bucket_upper_bps": None if math.isinf(upper) else upper,
                    "bucket_label": bucket_label(lower, upper),
                    "count": int(count),
                    "pct": percent(int(count), total),
                    "cumulative_count": cumulative,
                    "cumulative_pct": percent(cumulative, total),
                }
            )
    return rows


def threshold_rows(
    mode: str,
    horizon_minutes: int,
    metrics: WindowMetrics,
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric_name in METRICS:
        values = metrics.metric(metric_name)
        for threshold in thresholds:
            at_least = int(np.sum(values >= threshold))
            below = len(values) - at_least
            rows.append(
                {
                    "sampling_mode": mode,
                    "horizon_minutes": horizon_minutes,
                    "horizon_seconds": horizon_minutes * 60,
                    "metric": metric_name,
                    "threshold_bps": threshold,
                    "at_least_count": at_least,
                    "at_least_pct": percent(at_least, len(values)),
                    "below_count": below,
                    "below_pct": percent(below, len(values)),
                }
            )
    return rows


def daily_rows(
    mode: str,
    horizon_minutes: int,
    metrics: WindowMetrics,
) -> list[dict[str, Any]]:
    if len(metrics) == 0:
        return []
    date_ordinals = np.floor_divide(metrics.entry_seconds, SECONDS_PER_DAY).astype(
        np.int64
    )
    boundaries = np.flatnonzero(np.diff(date_ordinals)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(metrics)]))

    rows: list[dict[str, Any]] = []
    for start, end in zip(starts, ends):
        ordinal_since_epoch = int(date_ordinals[start])
        day = date.fromordinal(date(1970, 1, 1).toordinal() + ordinal_since_epoch)
        swing = metrics.high_low_swing_bps[start:end]
        max_abs = metrics.max_abs_excursion_bps[start:end]
        rows.append(
            {
                "sampling_mode": mode,
                "horizon_minutes": horizon_minutes,
                "receive_date": day.isoformat(),
                "valid_windows": len(swing),
                "mean_swing_bps": safe_mean(swing),
                "median_swing_bps": percentile(swing, 0.50),
                "p90_swing_bps": percentile(swing, 0.90),
                "p95_swing_bps": percentile(swing, 0.95),
                "maximum_swing_bps": float(np.max(swing)),
                "mean_max_abs_excursion_bps": safe_mean(max_abs),
                "share_swing_ge_25bps_pct": percent(int(np.sum(swing >= 25.0)), len(swing)),
                "share_swing_ge_35bps_pct": percent(int(np.sum(swing >= 35.0)), len(swing)),
                "share_swing_ge_50bps_pct": percent(int(np.sum(swing >= 50.0)), len(swing)),
            }
        )
    return rows


def apply_sampling_mode(
    metrics: WindowMetrics,
    mode: str,
    horizon_seconds: int,
) -> WindowMetrics:
    if mode == "rolling_1s":
        return metrics
    if mode == "utc_non_overlapping":
        mask = (metrics.entry_seconds % horizon_seconds) == 0
        return metrics.subset(mask)
    raise ValueError(f"Unknown sampling mode: {mode}")


# ---------------------------------------------------------------------------
# Findings and invariants
# ---------------------------------------------------------------------------


def build_findings_markdown(
    config: Mapping[str, Any],
    summary_rows: Sequence[Mapping[str, Any]],
    threshold_rows_all: Sequence[Mapping[str, Any]],
    input_partitions: Sequence[InputPartition],
    total_rows: int,
    dense_span_rows: int,
    missing_seconds: int,
) -> str:
    rolling = [row for row in summary_rows if row["sampling_mode"] == "rolling_1s"]
    rolling = sorted(rolling, key=lambda row: int(row["horizon_minutes"]))

    threshold_lookup: dict[tuple[int, float], float] = {}
    for row in threshold_rows_all:
        if (
            row["sampling_mode"] == "rolling_1s"
            and row["metric"] == PRIMARY_METRIC
            and float(row["threshold_bps"]) in (25.0, 35.0, 50.0)
        ):
            threshold_lookup[
                (int(row["horizon_minutes"]), float(row["threshold_bps"]))
            ] = float(row["at_least_pct"])

    lines = [
        "# Perps Forward Swing Profile",
        "",
        f"Generated at: `{utc_now_iso()}`",
        "",
        "## Contract",
        "",
        "- Decision anchor: feature row at second `t`.",
        "- Entry: exact price at `t + 1`.",
        "- Path: every exact second from entry through entry plus the horizon, inclusive.",
        "- Primary swing: `(future high - future low) / entry price × 10,000`.",
        f"- Anchor readiness: `{config['anchor_readiness']}`.",
        "- `rolling_1s` windows overlap; `utc_non_overlapping` is supplied as a robustness view.",
        "",
        "## Input coverage",
        "",
        f"- Feature partitions: **{len(input_partitions):,}**",
        f"- Feature rows: **{total_rows:,}**",
        f"- Dense elapsed-second span: **{dense_span_rows:,}**",
        f"- Missing seconds inside the span: **{missing_seconds:,}**",
        f"- Date range: **{input_partitions[0].receive_date} through {input_partitions[-1].receive_date}**",
        "",
        "## Rolling one-second results",
        "",
        "| Horizon | Valid windows | Mean swing | Median | P90 | P95 | ≥25 bps | ≥35 bps | ≥50 bps |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rolling:
        horizon = int(row["horizon_minutes"])
        lines.append(
            "| "
            f"{horizon} min | {int(row['valid_windows']):,} | "
            f"{float(row['mean_swing_bps']):.3f} | "
            f"{float(row['median_swing_bps']):.3f} | "
            f"{float(row['p90_swing_bps']):.3f} | "
            f"{float(row['p95_swing_bps']):.3f} | "
            f"{threshold_lookup.get((horizon, 25.0), float('nan')):.3f}% | "
            f"{threshold_lookup.get((horizon, 35.0), float('nan')):.3f}% | "
            f"{threshold_lookup.get((horizon, 50.0), float('nan')):.3f}% |"
        )

    lines.extend(
        [
            "",
            "## Interpretation notes",
            "",
            "The mean high-low swing measures total intrawindow travel relative to the entry price. "
            "It is not the same as a realizable one-direction trade return because the high and low "
            "may occur in either order.",
            "",
            "Use `max_abs_excursion_bps` when asking how far price moved in either direction from "
            "entry, and use the separate upside/downside excursion distributions for directional "
            "target and stop calibration.",
            "",
            "The dense rolling profile is best for opportunity prevalence. The UTC non-overlapping "
            "profile is better for checking whether conclusions survive heavy overlap reduction.",
            "",
        ]
    )
    return "\n".join(lines)


def invariant_rows(
    seconds: np.ndarray,
    grid: DenseFeatureGrid,
    summary_rows: Sequence[Mapping[str, Any]],
    distribution_rows_all: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        rows.append(
            {
                "invariant": name,
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
            }
        )

    add(
        "GLOBAL_SECONDS_STRICTLY_INCREASING",
        bool(len(seconds) < 2 or np.all(np.diff(seconds) > 0)),
        f"rows={len(seconds):,}",
    )
    add(
        "DENSE_GRID_OBSERVED_COUNT",
        int(np.sum(grid.observed)) == len(seconds),
        f"observed={int(np.sum(grid.observed)):,}, rows={len(seconds):,}",
    )
    add(
        "SUMMARY_HAS_ALL_HORIZONS_AND_MODES",
        len(summary_rows) > 0,
        f"summary_rows={len(summary_rows):,}",
    )

    distribution_frame = pd.DataFrame(distribution_rows_all)
    distribution_ok = True
    details: list[str] = []
    if len(distribution_frame):
        grouped = distribution_frame.groupby(
            ["sampling_mode", "horizon_minutes", "metric"], dropna=False
        )
        for key, group in grouped:
            total_pct = float(group["pct"].sum())
            if not math.isclose(total_pct, 100.0, rel_tol=0, abs_tol=1e-7):
                distribution_ok = False
                details.append(f"{key}: pct={total_pct}")
    add(
        "DISTRIBUTION_PERCENTAGES_SUM_TO_100",
        distribution_ok,
        "; ".join(details[:10]) if details else "all groups sum to 100%",
    )

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile causal forward perps price swings across minute horizons."
    )
    parser.add_argument(
        "--feature-root",
        default="./data/derived/perps/flow_features_1s_v1",
        help="Root of the audited perps flow-feature dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="./reports/perps_forward_swing_profile",
        help="Directory for CSV, JSON and Markdown reports.",
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument(
        "--horizons-minutes",
        default=",".join(str(value) for value in DEFAULT_HORIZONS_MINUTES),
        help="Comma-separated forward horizons in minutes.",
    )
    parser.add_argument("--start-date", help="Optional inclusive YYYY-MM-DD filter.")
    parser.add_argument("--end-date", help="Optional inclusive YYYY-MM-DD filter.")
    parser.add_argument(
        "--bucket-step-bps",
        type=float,
        default=DEFAULT_BUCKET_STEP_BPS,
        help="Distribution bucket width in basis points.",
    )
    parser.add_argument(
        "--bucket-max-bps",
        type=float,
        default=DEFAULT_BUCKET_MAX_BPS,
        help="Last finite bucket edge; a final tail bucket is added above it.",
    )
    parser.add_argument(
        "--anchor-readiness",
        choices=("flow_ready", "all_observed"),
        default="flow_ready",
        help="Require flow_feature_context_ready at the decision anchor or use all rows.",
    )
    parser.add_argument(
        "--expected-builder-version",
        default=EXPECTED_BUILDER_VERSION,
        help="Expected perps flow-feature builder version. Use an empty string to disable.",
    )
    parser.add_argument(
        "--skip-non-overlapping",
        action="store_true",
        help="Emit only dense rolling_1s results.",
    )
    parser.add_argument(
        "--hash-input-files",
        action="store_true",
        help="Compute SHA-256 for every feature partition in the JSON manifest.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    require_arrow()
    args = build_parser().parse_args(argv)

    feature_root = Path(args.feature_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    symbol = str(args.symbol).upper().strip()
    horizons_minutes = parse_int_list(args.horizons_minutes, "horizons-minutes")
    start_date = parse_date(args.start_date, "start-date")
    end_date = parse_date(args.end_date, "end-date")
    if start_date and end_date and start_date > end_date:
        raise SystemExit("--start-date cannot be after --end-date")

    edges = bucket_edges(args.bucket_step_bps, args.bucket_max_bps)
    thresholds = tuple(
        float(value)
        for value in np.arange(
            args.bucket_step_bps,
            args.bucket_max_bps + args.bucket_step_bps,
            args.bucket_step_bps,
        )
    )
    modes = ["rolling_1s"]
    if not args.skip_non_overlapping:
        modes.append("utc_non_overlapping")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = output_dir / "runs" / run_id

    print("Profiling forward perps swings...")
    print(f"Run ID          : {run_id}")
    print(f"Feature root    : {feature_root}")
    print(f"Output directory: {run_dir}")
    print(f"Symbol          : {symbol}")
    print(f"Horizons        : {','.join(str(value) for value in horizons_minutes)} minutes")
    print(f"Buckets         : {args.bucket_step_bps:g} bps through {args.bucket_max_bps:g} bps + tail")
    print(f"Anchor readiness: {args.anchor_readiness}")
    print(f"Sampling modes  : {','.join(modes)}")

    partitions = discover_partitions(
        feature_root=feature_root,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        expected_builder_version=args.expected_builder_version,
    )
    seconds, prices, ready, _ = load_feature_arrays(partitions, symbol)
    grid = make_dense_grid(seconds, prices, ready)

    dense_span_rows = len(grid.price)
    missing_seconds = dense_span_rows - int(np.sum(grid.observed))
    finite_prices = int(np.sum(np.isfinite(grid.price) & (grid.price > 0)))

    print(f"Observed rows    : {len(seconds):,}")
    print(f"Dense span       : {dense_span_rows:,} seconds")
    print(f"Missing seconds  : {missing_seconds:,}")
    print(f"Finite prices    : {finite_prices:,}")

    summary_rows_all: list[dict[str, Any]] = []
    distribution_rows_all: list[dict[str, Any]] = []
    threshold_rows_all: list[dict[str, Any]] = []
    daily_rows_all: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    require_ready = args.anchor_readiness == "flow_ready"

    for horizon_index, horizon_minutes in enumerate(horizons_minutes, start=1):
        horizon_seconds = horizon_minutes * 60
        print(
            f"CALCULATE [{horizon_index:>2}/{len(horizons_minutes)}] "
            f"{horizon_minutes} minutes..."
        )
        metrics, counts = calculate_window_metrics(
            grid=grid,
            horizon_seconds=horizon_seconds,
            require_ready=require_ready,
        )

        coverage_rows.append(
            {
                "horizon_minutes": horizon_minutes,
                "horizon_seconds": horizon_seconds,
                **counts,
                "valid_window_pct_of_eligible": percent(
                    counts["valid_windows"], counts["eligible_anchor_rows"]
                ),
            }
        )

        for mode in modes:
            sampled = apply_sampling_mode(metrics, mode, horizon_seconds)
            mode_counts = dict(counts)
            if mode != "rolling_1s":
                mode_counts["invalid_or_incomplete_windows"] = 0
            summary_rows_all.append(
                summary_row(mode, horizon_minutes, sampled, mode_counts)
            )
            distribution_rows_all.extend(
                distribution_rows(mode, horizon_minutes, sampled, edges)
            )
            threshold_rows_all.extend(
                threshold_rows(mode, horizon_minutes, sampled, thresholds)
            )
            daily_rows_all.extend(daily_rows(mode, horizon_minutes, sampled))

            primary = sampled.high_low_swing_bps
            print(
                f"PROFILE   {mode:<20} windows={len(sampled):,}, "
                f"mean={safe_mean(primary):.3f} bps, "
                f"median={percentile(primary, 0.50):.3f} bps, "
                f">=50bps={percent(int(np.sum(primary >= 50.0)), len(primary)):.3f}%"
            )

        del metrics

    invariants = invariant_rows(
        seconds=seconds,
        grid=grid,
        summary_rows=summary_rows_all,
        distribution_rows_all=distribution_rows_all,
    )
    failed_invariants = [row for row in invariants if row["status"] == "FAIL"]

    config = {
        "profiler_version": PROFILER_VERSION,
        "expected_feature_schema_version": EXPECTED_FEATURE_SCHEMA_VERSION,
        "expected_builder_version": args.expected_builder_version,
        "feature_root": str(feature_root),
        "output_dir": str(output_dir),
        "symbol": symbol,
        "horizons_minutes": list(horizons_minutes),
        "bucket_step_bps": args.bucket_step_bps,
        "bucket_max_bps": args.bucket_max_bps,
        "thresholds_bps": list(thresholds),
        "anchor_readiness": args.anchor_readiness,
        "sampling_modes": modes,
        "hash_input_files": bool(args.hash_input_files),
        "causal_window_contract": {
            "decision_second": "t",
            "entry_second": "t+1",
            "future_path": "t+1 through t+1+horizon_seconds inclusive",
            "path_observations": "horizon_seconds+1",
        },
        "primary_metric": PRIMARY_METRIC,
        "metrics": list(METRICS),
    }

    summary_payload = {
        "status": "FAIL" if failed_invariants else "PASS",
        "run_id": run_id,
        "profiler_version": PROFILER_VERSION,
        "created_at_utc": utc_now_iso(),
        "symbol": symbol,
        "date_start": partitions[0].receive_date.isoformat(),
        "date_end": partitions[-1].receive_date.isoformat(),
        "partitions": len(partitions),
        "feature_rows": len(seconds),
        "dense_span_seconds": dense_span_rows,
        "missing_seconds": missing_seconds,
        "finite_price_seconds": finite_prices,
        "failed_invariants": len(failed_invariants),
        "configuration": config,
        "window_summaries": summary_rows_all,
        "input_manifest": [
            {
                "receive_date": partition.receive_date.isoformat(),
                "path": str(partition.path),
                "rows": partition.rows,
                "minimum_second": partition.minimum_second,
                "maximum_second": partition.maximum_second,
                "builder_version": partition.builder_version,
                "schema_version": partition.schema_version,
                "size_bytes": partition.path.stat().st_size,
                "modified_time_ns": partition.path.stat().st_mtime_ns,
                "sha256": sha256_file(partition.path) if args.hash_input_files else None,
            }
            for partition in partitions
        ],
    }

    findings = build_findings_markdown(
        config=config,
        summary_rows=summary_rows_all,
        threshold_rows_all=threshold_rows_all,
        input_partitions=partitions,
        total_rows=len(seconds),
        dense_span_rows=dense_span_rows,
        missing_seconds=missing_seconds,
    )

    summary_columns = [
        "sampling_mode",
        "horizon_minutes",
        "horizon_seconds",
        "valid_windows",
        "invalid_or_incomplete_windows",
        "mean_swing_bps",
        "median_swing_bps",
        "std_swing_bps",
        "p75_swing_bps",
        "p90_swing_bps",
        "p95_swing_bps",
        "p99_swing_bps",
        "maximum_swing_bps",
        "mean_up_excursion_bps",
        "mean_down_excursion_bps",
        "mean_max_abs_excursion_bps",
        "mean_terminal_abs_move_bps",
        "up_dominant_windows",
        "down_dominant_windows",
        "balanced_windows",
        "up_dominant_pct",
        "down_dominant_pct",
        "balanced_pct",
    ]
    distribution_columns = [
        "sampling_mode",
        "horizon_minutes",
        "horizon_seconds",
        "metric",
        "bucket_order",
        "bucket_lower_bps",
        "bucket_upper_bps",
        "bucket_label",
        "count",
        "pct",
        "cumulative_count",
        "cumulative_pct",
    ]
    threshold_columns = [
        "sampling_mode",
        "horizon_minutes",
        "horizon_seconds",
        "metric",
        "threshold_bps",
        "at_least_count",
        "at_least_pct",
        "below_count",
        "below_pct",
    ]
    daily_columns = [
        "sampling_mode",
        "horizon_minutes",
        "receive_date",
        "valid_windows",
        "mean_swing_bps",
        "median_swing_bps",
        "p90_swing_bps",
        "p95_swing_bps",
        "maximum_swing_bps",
        "mean_max_abs_excursion_bps",
        "share_swing_ge_25bps_pct",
        "share_swing_ge_35bps_pct",
        "share_swing_ge_50bps_pct",
    ]
    coverage_columns = [
        "horizon_minutes",
        "horizon_seconds",
        "observed_anchor_rows",
        "eligible_anchor_rows",
        "candidate_anchor_rows",
        "valid_windows",
        "invalid_or_incomplete_windows",
        "valid_window_pct_of_eligible",
    ]
    invariant_columns = ["invariant", "status", "detail"]

    output_files: dict[str, tuple[str, Any]] = {
        "perps_forward_swing_summary.csv": ("csv", (summary_rows_all, summary_columns)),
        "perps_forward_swing_distribution.csv": (
            "csv",
            (distribution_rows_all, distribution_columns),
        ),
        "perps_forward_swing_thresholds.csv": (
            "csv",
            (threshold_rows_all, threshold_columns),
        ),
        "perps_forward_swing_daily_summary.csv": (
            "csv",
            (daily_rows_all, daily_columns),
        ),
        "perps_forward_swing_coverage.csv": ("csv", (coverage_rows, coverage_columns)),
        "perps_forward_swing_invariants.csv": ("csv", (invariants, invariant_columns)),
        "perps_forward_swing_summary.json": ("json", summary_payload),
        "perps_forward_swing_config.json": ("json", config),
        "perps_forward_swing_findings.md": ("text", findings),
    }

    for filename, (kind, payload) in output_files.items():
        run_path = run_dir / filename
        latest_path = output_dir / filename
        if kind == "csv":
            rows, columns = payload
            atomic_write_csv(run_path, rows, columns)
            atomic_write_csv(latest_path, rows, columns)
        elif kind == "json":
            atomic_write_json(run_path, payload)
            atomic_write_json(latest_path, payload)
        else:
            atomic_write_text(run_path, str(payload))
            atomic_write_text(latest_path, str(payload))

    status = "FAIL" if failed_invariants else "PASS"
    print("\nForward swing profile finished.")
    print(f"Status           : {status}")
    print(f"Feature rows     : {len(seconds):,}")
    print(f"Horizons         : {len(horizons_minutes)}")
    print(f"Summary rows     : {len(summary_rows_all):,}")
    print(f"Distribution rows: {len(distribution_rows_all):,}")
    print(f"Threshold rows   : {len(threshold_rows_all):,}")
    print(f"Failed invariants: {len(failed_invariants):,}")
    print(f"Findings         : {run_dir / 'perps_forward_swing_findings.md'}")
    print(f"Latest reports   : {output_dir}")

    return 1 if failed_invariants else 0


if __name__ == "__main__":
    raise SystemExit(main())
