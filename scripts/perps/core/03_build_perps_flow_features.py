#!/usr/bin/env python
"""
Build causal rolling flow features from the audited perps one-second core layer.

Input
-----
The audited output of ``scripts/perps/core/01_build_perps_core_1s.py``::

    data/derived/perps/core_1s_v1/
      symbol=BTCUSDT/
        date=YYYY-MM-DD/
          part-YYYY-MM-DD.parquet

Output
------
A compact feature-only layer keyed by ``symbol`` and ``receive_second``::

    data/derived/perps/flow_features_1s_v1/
      symbol=BTCUSDT/
        date=YYYY-MM-DD/
          part-YYYY-MM-DD.parquet

The output intentionally does not repeat the complete 90-column core table. It
contains stable identifiers, current one-second measurements, cumulative CVD,
and strictly causal rolling features. Join it back to the core table using
``(symbol, receive_second)`` when additional raw measurements are needed.

Causal contract
---------------
* Receive time remains the availability clock.
* A feature row for receive second ``t`` is available only after that second
  closes, at ``available_after_second = t + 1``.
* Every rolling window uses only seconds less than or equal to ``t``.
* Exact returns compare ``t`` with the exact second ``t - horizon``. They are
  null when that lag second is absent.
* A rolling window is marked complete only when every expected core second is
  present. Features that would otherwise use partial windows are kept null.
* Missing seconds are never silently synthesized or treated as observed zeros.
* No future outcomes, targets, labels, or trade-entry assumptions are created.

Feature groups
--------------
* Exact log returns and basis-point returns.
* Realized volatility from one-second log returns.
* Trade activity and aggressive signed quote flow.
* Aggressive-flow imbalance.
* Dataset-to-date cumulative volume delta (CVD), with resumable carry state.
* Exact open-interest changes over each horizon.
* Liquidation intensity, net liquidation pressure and liquidation imbalance.
* Price/flow alignment and price response per one million units of signed quote
  flow, guarded by a configurable minimum flow notional.
* Rolling z-scores for one-second signed flow, quote volume, liquidation
  notional and absolute return.

Operational properties
----------------------
* Date-level resumability through SQLite.
* Current core input manifests, rolling-lookback manifests, configuration and
  incoming cumulative carry state are all part of the rebuild signature.
* Changed prior-date context automatically rebuilds dependent dates.
* Output replacement is atomic.
* Dates are processed sequentially because cumulative CVD is stateful.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import gc
import os
import re
import shutil
import time
import sqlite3
import sys
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

try:  # Kept optional at import time so the pure feature engine can be tested.
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - environment-specific
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]


BUILDER_VERSION = "1.0.2"
OUTPUT_SCHEMA_VERSION = 1
EXPECTED_CORE_SCHEMA_VERSION = 1
SECONDS_PER_DAY = 86_400
NANOSECONDS_PER_SECOND = 1_000_000_000
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
    "trade_observed",
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


def compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_iso_date(value: str | None, argument_name: str) -> date | None:
    if value is None:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SystemExit(f"{argument_name} must use YYYY-MM-DD; received {value!r}") from exc


def parse_positive_int_list(value: str, argument_name: str) -> tuple[int, ...]:
    try:
        values = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    except ValueError as exc:
        raise SystemExit(f"{argument_name} must be a comma-separated list of integers") from exc
    if not values or any(item <= 0 for item in values):
        raise SystemExit(f"{argument_name} must contain positive integers")
    return values


def safe_symbol_partition(symbol: str) -> str:
    safe = SYMBOL_PARTITION_SAFE_RE.sub("_", symbol.strip().upper())
    if not safe:
        raise SystemExit("--symbol resolved to an empty partition name")
    return safe


def day_start_second(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())


def receive_day_from_second(second: int) -> date:
    return datetime.fromtimestamp(int(second), tz=timezone.utc).date()


def replace_with_retry(
    source: Path,
    destination: Path,
    attempts: int = 20,
    initial_delay_seconds: float = 0.05,
) -> None:
    """Atomically replace a file, tolerating short-lived Windows file locks."""
    delay = initial_delay_seconds
    for attempt in range(1, attempts + 1):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt >= attempts:
                raise
            gc.collect()
            time.sleep(delay)
            delay = min(delay * 1.5, 0.5)


def unlink_with_retry(
    path: Path,
    attempts: int = 10,
    initial_delay_seconds: float = 0.05,
) -> None:
    """Best-effort cleanup for temporary files after handles have been closed."""
    delay = initial_delay_seconds
    for attempt in range(1, attempts + 1):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt >= attempts:
                return
            gc.collect()
            time.sleep(delay)
            delay = min(delay * 1.5, 0.5)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        replace_with_retry(temporary, path)
    finally:
        unlink_with_retry(temporary)


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        replace_with_retry(temporary, path)
    finally:
        unlink_with_retry(temporary)


def ensure_free_space(path: Path, minimum_gib: float) -> float:
    path.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(path).free / (1024 ** 3)
    if free_gib < minimum_gib:
        raise RuntimeError(
            f"Free-space guard triggered for {path}: {free_gib:.2f} GiB available, "
            f"{minimum_gib:.2f} GiB required"
        )
    return free_gib


def file_manifest(path: Path, root: Path) -> dict[str, Any]:
    require_arrow()
    stat = path.stat()
    parquet_file = pq.ParquetFile(path)
    try:
        metadata = parquet_file.metadata
        schema_metadata = parquet_file.schema_arrow.metadata or {}
        return {
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "rows": int(metadata.num_rows),
            "row_groups": int(metadata.num_row_groups),
            "schema_names": list(parquet_file.schema_arrow.names),
            "core_schema_version": (schema_metadata.get(b"perps_core_schema_version") or b"").decode(),
            "core_builder_version": (schema_metadata.get(b"builder_version") or b"").decode(),
        }
    finally:
        parquet_file.close()


def manifest_signature(manifests: Sequence[Mapping[str, Any]]) -> str:
    return sha256_text(compact_json(list(manifests)))



# ---------------------------------------------------------------------------
# Partition discovery and input reading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatePartition:
    day: date
    files: tuple[Path, ...]


def discover_partitions(
    core_root: Path,
    symbol: str,
    start_date: date | None,
    end_date: date | None,
) -> list[DatePartition]:
    symbol_root = core_root / f"symbol={safe_symbol_partition(symbol)}"
    if not symbol_root.exists():
        raise SystemExit(f"Core symbol partition does not exist: {symbol_root}")

    partitions: list[DatePartition] = []
    for child in sorted(symbol_root.iterdir()):
        if not child.is_dir():
            continue
        match = DATE_PARTITION_RE.match(child.name)
        if not match:
            continue
        day = datetime.strptime(match.group(1), "%Y-%m-%d").date()
        if start_date is not None and day < start_date:
            continue
        if end_date is not None and day > end_date:
            continue
        files = tuple(sorted(child.glob("*.parquet")))
        if not files:
            continue
        partitions.append(DatePartition(day=day, files=files))

    if not partitions:
        raise SystemExit("No matching perps core date partitions were found")
    return partitions


def discover_all_partitions(core_root: Path, symbol: str) -> dict[date, DatePartition]:
    partitions = discover_partitions(core_root, symbol, None, None)
    return {partition.day: partition for partition in partitions}


def context_days_for(
    day: date,
    all_days: Sequence[date],
    max_window_seconds: int,
) -> list[date]:
    earliest_second = day_start_second(day) - max_window_seconds - 1
    selected = [
        candidate
        for candidate in all_days
        if day_start_second(candidate) + SECONDS_PER_DAY - 1 >= earliest_second
        and candidate <= day
    ]
    return selected


def read_partition(partition: DatePartition) -> pd.DataFrame:
    require_arrow()
    frames: list[pd.DataFrame] = []
    for path in partition.files:
        parquet_file = pq.ParquetFile(path)
        try:
            missing = [
                name for name in REQUIRED_CORE_COLUMNS
                if name not in parquet_file.schema_arrow.names
            ]
            if missing:
                raise RuntimeError(f"Core file {path} is missing required columns: {missing}")
            frames.append(
                parquet_file.read(columns=list(REQUIRED_CORE_COLUMNS)).to_pandas()
            )
        finally:
            parquet_file.close()
    frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    frame = frame.sort_values("receive_second", kind="mergesort").reset_index(drop=True)
    if frame["receive_second"].duplicated().any():
        duplicates = frame.loc[frame["receive_second"].duplicated(), "receive_second"].head().tolist()
        raise RuntimeError(f"Core partition {partition.day} has duplicate seconds: {duplicates}")
    expected_symbol = str(frame["symbol"].iloc[0]) if not frame.empty else ""
    if not frame["symbol"].eq(expected_symbol).all():
        raise RuntimeError(f"Core partition {partition.day} contains multiple symbols")
    return frame


def read_context(
    partitions_by_day: Mapping[date, DatePartition],
    context_days: Sequence[date],
) -> pd.DataFrame:
    frames = [read_partition(partitions_by_day[item]) for item in context_days]
    context = pd.concat(frames, ignore_index=True)
    context = context.sort_values("receive_second", kind="mergesort").reset_index(drop=True)
    if context["receive_second"].duplicated().any():
        duplicates = context.loc[context["receive_second"].duplicated(), "receive_second"].head().tolist()
        raise RuntimeError(f"Context contains duplicate receive seconds: {duplicates}")
    return context


# ---------------------------------------------------------------------------
# Pure feature engine
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


@dataclass
class WindowBounds:
    left: np.ndarray
    row_count: np.ndarray
    complete: np.ndarray


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
    source[mask] = 1  # fresh carried trade price

    mask = (source == 0) & mark_fresh & valid_mark
    price[mask] = mark_price[mask]
    source[mask] = 2  # fresh mark price fallback

    mask = (source == 0) & valid_trade
    price[mask] = trade_price[mask]
    source[mask] = 3  # stale but causal trade state

    mask = (source == 0) & valid_mark
    price[mask] = mark_price[mask]
    source[mask] = 4  # stale but causal mark state

    fresh = np.isin(source, [1, 2])
    return price, source, fresh


def exact_lag(values: np.ndarray, seconds: np.ndarray, lag_seconds: int) -> tuple[np.ndarray, np.ndarray]:
    target = seconds - int(lag_seconds)
    positions = np.searchsorted(seconds, target, side="left")
    available = (positions < len(seconds))
    safe_positions = np.minimum(positions, max(len(seconds) - 1, 0))
    if len(seconds):
        available &= seconds[safe_positions] == target
    result = np.full(len(seconds), np.nan, dtype=float)
    if len(seconds):
        result[available] = values[positions[available]]
    return result, available


def calculate_window_bounds(seconds: np.ndarray, window: int) -> WindowBounds:
    left_targets = seconds - int(window) + 1
    left = np.searchsorted(seconds, left_targets, side="left")
    row_count = np.arange(len(seconds), dtype=np.int64) - left + 1
    complete = (
        (row_count == int(window))
        & (left < len(seconds))
        & (seconds[left] == left_targets)
    )
    return WindowBounds(left=left, row_count=row_count, complete=complete)


def rolling_sum(values: np.ndarray, bounds: WindowBounds) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    safe = np.where(finite, values, 0.0)
    cumulative = np.concatenate(([0.0], np.cumsum(safe, dtype=float)))
    count_cumulative = np.concatenate(([0], np.cumsum(finite.astype(np.int64))))
    right = np.arange(len(values), dtype=np.int64) + 1
    sums = cumulative[right] - cumulative[bounds.left]
    counts = count_cumulative[right] - count_cumulative[bounds.left]
    return sums, counts


def rolling_zscore(
    values: np.ndarray,
    bounds: WindowBounds,
    minimum_count: int,
) -> np.ndarray:
    finite = np.isfinite(values)
    safe = np.where(finite, values, 0.0)
    safe_square = np.where(finite, values * values, 0.0)
    cumulative = np.concatenate(([0.0], np.cumsum(safe, dtype=float)))
    cumulative_square = np.concatenate(([0.0], np.cumsum(safe_square, dtype=float)))
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
    valid = (
        bounds.complete
        & finite
        & (counts >= minimum_count)
        & np.isfinite(std)
        & (std > 1e-12)
    )
    result[valid] = (values[valid] - means[valid]) / std[valid]
    return result


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full(len(numerator), np.nan, dtype=float)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
    result[valid] = numerator[valid] / denominator[valid]
    return result


def bounded_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """Return a ratio expected to lie in [-1, 1], correcting only FP drift.

    Values that exceed the mathematical bound by more than ``tolerance`` are
    deliberately left unchanged so output validation still exposes a genuine
    accounting inconsistency. Tiny floating-point overshoots are clipped.
    """

    result = safe_ratio(numerator, denominator)
    finite = np.isfinite(result)
    small_overshoot = finite & (np.abs(result) > 1.0) & (np.abs(result) <= 1.0 + tolerance)
    result[small_overshoot] = np.clip(result[small_overshoot], -1.0, 1.0)
    return result


def component_imbalance(
    positive_component: np.ndarray,
    negative_component: np.ndarray,
) -> np.ndarray:
    """Stable (positive - negative) / (positive + negative) imbalance.

    Both the numerator and denominator are derived from the same component
    sums. This avoids catastrophic cancellation from independently accumulated
    signed-flow and gross-flow series in short, low-volume windows.
    """

    positive = np.asarray(positive_component, dtype=float)
    negative = np.asarray(negative_component, dtype=float)
    numerator = positive - negative
    denominator = positive + negative
    return bounded_ratio(numerator, denominator)


def signed_alignment(left: np.ndarray, right: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.full(len(left), np.nan, dtype=float)
    finite = valid & np.isfinite(left) & np.isfinite(right)
    result[finite] = np.sign(left[finite]) * np.sign(right[finite])
    return result


def build_feature_frame(
    context: pd.DataFrame,
    output_day: date,
    incoming_carry: CarryState,
    config: FeatureConfig,
) -> tuple[pd.DataFrame, CarryState, dict[str, Any]]:
    if context.empty:
        raise RuntimeError("Feature context is empty")

    seconds = pd.to_numeric(context["receive_second"], errors="raise").to_numpy(np.int64)
    if np.any(np.diff(seconds) <= 0):
        raise RuntimeError("Context receive_second is not strictly increasing")

    current_start = day_start_second(output_day)
    current_end = current_start + SECONDS_PER_DAY - 1
    current_mask = (seconds >= current_start) & (seconds <= current_end)
    if not current_mask.any():
        raise RuntimeError(f"No core rows exist for output date {output_day}")
    current_indices = np.flatnonzero(current_mask)
    output_slice = slice(int(current_indices[0]), int(current_indices[-1]) + 1)
    if not current_mask[output_slice].all():
        raise RuntimeError(f"Core rows for {output_day} are not contiguous within context ordering")

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
    max_trade_notional = pd.to_numeric(context["max_trade_notional"], errors="coerce").fillna(0).to_numpy(float)

    open_interest = pd.to_numeric(context["open_interest"], errors="coerce").to_numpy(float)
    open_interest_change = pd.to_numeric(context["open_interest_change"], errors="coerce").to_numpy(float)
    open_interest_updated = context["open_interest_updated"].fillna(False).to_numpy(bool)

    liquidation_count = pd.to_numeric(context["liquidation_event_count"], errors="coerce").fillna(0).to_numpy(float)
    long_liq = pd.to_numeric(context["long_liquidation_notional"], errors="coerce").fillna(0).to_numpy(float)
    short_liq = pd.to_numeric(context["short_liquidation_notional"], errors="coerce").fillna(0).to_numpy(float)
    total_liq = pd.to_numeric(context["liquidation_notional"], errors="coerce").fillna(0).to_numpy(float)
    net_liq = pd.to_numeric(context["net_liquidation_notional"], errors="coerce").fillna(0).to_numpy(float)
    max_liq = pd.to_numeric(context["max_liquidation_notional"], errors="coerce").fillna(0).to_numpy(float)

    # Derive imbalance from the same buy/sell components used by its gross
    # denominator. Independently accumulated signed and gross flows can differ
    # by a few floating-point ULPs in low-volume windows.
    flow_imbalance_1s = component_imbalance(buy_quote, sell_quote)
    max_trade_share_1s = safe_ratio(max_trade_notional, quote_volume)
    liquidation_imbalance_1s = bounded_ratio(short_liq - long_liq, total_liq)

    current_signed_quote = signed_quote[output_slice]
    current_signed_base = signed_base[output_slice]
    cvd_quote = incoming_carry.cvd_quote + np.cumsum(current_signed_quote, dtype=float)
    cvd_base = incoming_carry.cvd_base + np.cumsum(current_signed_base, dtype=float)
    outgoing_carry = CarryState(
        cvd_quote=float(cvd_quote[-1]),
        cvd_base=float(cvd_base[-1]),
    )

    current = context.iloc[output_slice].reset_index(drop=True)
    frame = pd.DataFrame({
        "feature_schema_version": np.full(len(current), OUTPUT_SCHEMA_VERSION, dtype=np.int16),
        "symbol": current["symbol"].astype(str).to_numpy(),
        "receive_second": seconds[output_slice],
        "timestamp_utc": pd.to_datetime(seconds[output_slice], unit="s", utc=True),
        "available_after_second": pd.to_numeric(current["available_after_second"], errors="raise").to_numpy(np.int64),
        "available_after_utc": pd.to_datetime(
            pd.to_numeric(current["available_after_second"], errors="raise"), unit="s", utc=True
        ),
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
        "aggressive_flow_imbalance_1s": flow_imbalance_1s[output_slice],
        "max_trade_notional_1s": max_trade_notional[output_slice],
        "max_trade_share_1s": max_trade_share_1s[output_slice],
        "open_interest_updated_1s": open_interest_updated[output_slice],
        "open_interest": open_interest[output_slice],
        "observed_open_interest_change_1s": open_interest_change[output_slice],
        "liquidation_event_count_1s": pd.to_numeric(
            current["liquidation_event_count"], errors="raise"
        ).to_numpy(np.int64),
        "liquidation_notional_1s": total_liq[output_slice],
        "net_liquidation_notional_1s": net_liq[output_slice],
        "liquidation_imbalance_1s": liquidation_imbalance_1s[output_slice],
        "max_liquidation_notional_1s": max_liq[output_slice],
        "cvd_quote": cvd_quote,
        "cvd_base": cvd_base,
    })

    derived_columns: dict[str, Any] = {}
    window_summaries: dict[str, Any] = {}
    for window in config.windows:
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

        trade_count_sum, _ = rolling_sum(trade_count, bounds)
        quote_volume_sum, _ = rolling_sum(quote_volume, bounds)
        buy_quote_sum, _ = rolling_sum(buy_quote, bounds)
        sell_quote_sum, _ = rolling_sum(sell_quote, bounds)
        # Use the same component sums for signed and gross flow. This preserves
        # |signed flow| <= gross flow by construction and avoids short-window
        # cancellation errors from a separate signed-flow cumulative sum.
        signed_quote_sum = buy_quote_sum - sell_quote_sum
        long_liq_sum, _ = rolling_sum(long_liq, bounds)
        short_liq_sum, _ = rolling_sum(short_liq, bounds)
        total_liq_sum, _ = rolling_sum(total_liq, bounds)
        net_liq_sum = short_liq_sum - long_liq_sum

        squared_returns = log_return_1s * log_return_1s
        rv_sum, rv_count = rolling_sum(squared_returns, bounds)
        realized_vol_bps = np.sqrt(np.maximum(rv_sum, 0.0)) * 10_000.0

        lag_oi, lag_oi_available = exact_lag(open_interest, seconds, window)
        oi_change = np.full(len(context), np.nan)
        oi_change_pct = np.full(len(context), np.nan)
        valid_oi = lag_oi_available & np.isfinite(open_interest) & np.isfinite(lag_oi)
        oi_change[valid_oi] = open_interest[valid_oi] - lag_oi[valid_oi]
        valid_oi_pct = valid_oi & (lag_oi != 0)
        oi_change_pct[valid_oi_pct] = oi_change[valid_oi_pct] / lag_oi[valid_oi_pct]

        rolling_ready = bounds.complete
        return_ready = bounds.complete & lag_available
        rv_ready = return_ready & (rv_count == window)
        flow_imbalance = component_imbalance(buy_quote_sum, sell_quote_sum)
        liquidation_imbalance = bounded_ratio(net_liq_sum, total_liq_sum)
        liquidation_to_quote = safe_ratio(total_liq_sum, quote_volume_sum)
        alignment = signed_alignment(horizon_return_bps, signed_quote_sum, return_ready)

        price_response = np.full(len(context), np.nan)
        valid_impact = (
            return_ready
            & np.isfinite(horizon_return_bps)
            & np.isfinite(signed_quote_sum)
            & (np.abs(signed_quote_sum) >= config.min_impact_flow_notional)
        )
        price_response[valid_impact] = (
            horizon_return_bps[valid_impact] / (signed_quote_sum[valid_impact] / 1_000_000.0)
        )

        # Partial windows are explicitly null rather than silently biased.
        for values in (
            trade_count_sum,
            quote_volume_sum,
            signed_quote_sum,
            flow_imbalance,
            total_liq_sum,
            net_liq_sum,
            liquidation_imbalance,
            liquidation_to_quote,
        ):
            values[~rolling_ready] = np.nan
        for values in (
            horizon_log_return,
            horizon_return_bps,
            alignment,
            price_response,
        ):
            values[~return_ready] = np.nan
        oi_ready = bounds.complete & lag_oi_available
        oi_change[~oi_ready] = np.nan
        oi_change_pct[~oi_ready] = np.nan
        realized_vol_bps[~rv_ready] = np.nan

        suffix = f"{window}s"
        derived_columns[f"window_observation_count_{suffix}"] = bounds.row_count[output_slice].astype(np.int32)
        derived_columns[f"window_complete_{suffix}"] = bounds.complete[output_slice]
        derived_columns[f"return_lag_available_{suffix}"] = lag_available[output_slice]
        derived_columns[f"log_return_{suffix}"] = horizon_log_return[output_slice]
        derived_columns[f"return_{suffix}_bps"] = horizon_return_bps[output_slice]
        derived_columns[f"realized_vol_{suffix}_bps"] = realized_vol_bps[output_slice]
        derived_columns[f"trade_count_{suffix}"] = pd.Series(trade_count_sum[output_slice]).round().astype("Int64")
        derived_columns[f"quote_volume_{suffix}"] = quote_volume_sum[output_slice]
        derived_columns[f"signed_quote_flow_{suffix}"] = signed_quote_sum[output_slice]
        derived_columns[f"aggressive_flow_imbalance_{suffix}"] = flow_imbalance[output_slice]
        derived_columns[f"open_interest_change_{suffix}"] = oi_change[output_slice]
        derived_columns[f"open_interest_change_{suffix}_pct"] = oi_change_pct[output_slice]
        derived_columns[f"liquidation_notional_{suffix}"] = total_liq_sum[output_slice]
        derived_columns[f"net_liquidation_notional_{suffix}"] = net_liq_sum[output_slice]
        derived_columns[f"liquidation_imbalance_{suffix}"] = liquidation_imbalance[output_slice]
        derived_columns[f"liquidation_to_quote_ratio_{suffix}"] = liquidation_to_quote[output_slice]
        derived_columns[f"price_flow_alignment_{suffix}"] = pd.Series(alignment[output_slice]).astype("Int8")
        derived_columns[f"price_response_bps_per_1m_signed_flow_{suffix}"] = price_response[output_slice]

        window_summaries[suffix] = {
            "complete_rows": int(bounds.complete[output_slice].sum()),
            "return_ready_rows": int(return_ready[output_slice].sum()),
            "realized_vol_ready_rows": int(rv_ready[output_slice].sum()),
            "impact_ready_rows": int(valid_impact[output_slice].sum()),
        }

    zscore_sources = {
        "signed_quote_flow_1s": signed_quote,
        "quote_volume_1s": quote_volume,
        "liquidation_notional_1s": total_liq,
        "abs_return_1s_bps": abs_return_1s_bps,
    }
    for window in config.zscore_windows:
        bounds = calculate_window_bounds(seconds, window)
        minimum_count = max(2, int(math.ceil(window * config.zscore_min_fraction)))
        for feature_name, values in zscore_sources.items():
            zscore = rolling_zscore(values, bounds, minimum_count)
            derived_columns[f"{feature_name}_z_{window}s"] = zscore[output_slice]

    max_context_complete = calculate_window_bounds(seconds, config.max_window).complete[output_slice]
    derived_columns["flow_feature_context_ready"] = (
        frame["core_context_ready"].astype(bool).to_numpy()
        & frame["price_for_features"].notna().to_numpy()
        & max_context_complete
    )
    frame = pd.concat(
        [frame, pd.DataFrame(derived_columns, index=frame.index)],
        axis=1,
    )

    summary = {
        "output_rows": int(len(frame)),
        "first_receive_second": int(frame["receive_second"].iloc[0]),
        "last_receive_second": int(frame["receive_second"].iloc[-1]),
        "price_available_rows": int(frame["price_for_features"].notna().sum()),
        "price_fresh_rows": int(frame["price_fresh"].sum()),
        "core_context_ready_rows": int(frame["core_context_ready"].sum()),
        "flow_feature_context_ready_rows": int(frame["flow_feature_context_ready"].sum()),
        "trade_seconds": int((frame["trade_count_1s"] > 0).sum()),
        "liquidation_seconds": int((frame["liquidation_event_count_1s"] > 0).sum()),
        "cvd_quote_start": float(cvd_quote[0]),
        "cvd_quote_end": float(cvd_quote[-1]),
        "cvd_base_start": float(cvd_base[0]),
        "cvd_base_end": float(cvd_base[-1]),
        "window_summaries": window_summaries,
    }
    return frame, outgoing_carry, summary


# ---------------------------------------------------------------------------
# Arrow schema, metadata and validation
# ---------------------------------------------------------------------------


def output_schema(config: FeatureConfig):
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
        fields.extend([
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
        ])

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


def frame_to_table(frame: pd.DataFrame, config: FeatureConfig, metadata: Mapping[str, str]):
    schema = output_schema(config).with_metadata(
        {str(key).encode(): str(value).encode() for key, value in metadata.items()}
    )
    missing = [field.name for field in schema if field.name not in frame.columns]
    extra = [name for name in frame.columns if name not in schema.names]
    if missing or extra:
        raise RuntimeError(f"Output schema mismatch: missing={missing}, extra={extra}")
    arrays = []
    for field in schema:
        try:
            arrays.append(pa.array(frame[field.name], type=field.type, from_pandas=True))
        except Exception as exc:
            raise RuntimeError(
                f"Failed converting feature column {field.name} to {field.type}: {exc}"
            ) from exc
    return pa.Table.from_arrays(arrays, schema=schema)


def validate_output(frame: pd.DataFrame, core_current: pd.DataFrame, day: date, config: FeatureConfig) -> None:
    failures: list[str] = []
    if frame.empty:
        failures.append("output is empty")
    if len(frame) != len(core_current):
        failures.append(f"row count differs from core ({len(frame)} != {len(core_current)})")
    if frame["receive_second"].duplicated().any():
        failures.append("receive_second contains duplicates")
    if not frame["receive_second"].is_monotonic_increasing:
        failures.append("receive_second is not strictly increasing")
    if not np.array_equal(
        frame["receive_second"].to_numpy(np.int64),
        core_current["receive_second"].to_numpy(np.int64),
    ):
        failures.append("output seconds do not exactly match current core rows")
    if not (frame["available_after_second"] == frame["receive_second"] + 1).all():
        failures.append("available_after_second is not receive_second + 1")
    if not frame["receive_second"].map(receive_day_from_second).eq(day).all():
        failures.append("output contains a row outside its date partition")
    if not frame["core_schema_version"].eq(EXPECTED_CORE_SCHEMA_VERSION).all():
        failures.append("unexpected core schema version")

    for column in (
        "quote_volume_1s",
        "base_volume_1s",
        "max_trade_notional_1s",
        "liquidation_notional_1s",
        "max_liquidation_notional_1s",
    ):
        if (frame[column] < 0).any():
            failures.append(f"{column} contains negative values")

    bounded_columns = [
        "aggressive_flow_imbalance_1s",
        "liquidation_imbalance_1s",
    ]
    bounded_columns.extend(f"aggressive_flow_imbalance_{window}s" for window in config.windows)
    bounded_columns.extend(f"liquidation_imbalance_{window}s" for window in config.windows)
    for column in bounded_columns:
        valid = frame[column].dropna()
        outside = (valid < -1.000000001) | (valid > 1.000000001)
        if outside.any():
            offending = valid[outside]
            first_index = offending.index[0]
            failures.append(
                f"{column} falls outside [-1, 1] "
                f"(min={valid.min():.12g}, max={valid.max():.12g}, "
                f"first_receive_second={int(frame.loc[first_index, 'receive_second'])}, "
                f"first_value={float(offending.iloc[0]):.12g})"
            )

    for window in config.windows:
        suffix = f"{window}s"
        complete = frame[f"window_complete_{suffix}"]
        observation_count = frame[f"window_observation_count_{suffix}"]
        if not observation_count[complete].eq(window).all():
            failures.append(f"complete {suffix} windows do not contain exactly {window} rows")
        partial = ~complete
        columns_that_must_be_null = (
            f"log_return_{suffix}",
            f"return_{suffix}_bps",
            f"realized_vol_{suffix}_bps",
            f"trade_count_{suffix}",
            f"quote_volume_{suffix}",
            f"signed_quote_flow_{suffix}",
            f"aggressive_flow_imbalance_{suffix}",
            f"open_interest_change_{suffix}",
            f"open_interest_change_{suffix}_pct",
            f"liquidation_notional_{suffix}",
            f"net_liquidation_notional_{suffix}",
            f"liquidation_imbalance_{suffix}",
            f"liquidation_to_quote_ratio_{suffix}",
            f"price_flow_alignment_{suffix}",
            f"price_response_bps_per_1m_signed_flow_{suffix}",
        )
        for column in columns_that_must_be_null:
            if frame.loc[partial, column].notna().any():
                failures.append(f"{column} is populated for an incomplete window")

    numeric = frame.select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
        failures.append("numeric output contains infinity")

    if failures:
        raise RuntimeError("Feature output validation failed: " + "; ".join(failures))


# ---------------------------------------------------------------------------
# Build state
# ---------------------------------------------------------------------------


def open_state_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS build_runs (
            run_id TEXT PRIMARY KEY,
            started_at_utc TEXT NOT NULL,
            completed_at_utc TEXT,
            status TEXT NOT NULL,
            core_root TEXT NOT NULL,
            output_root TEXT NOT NULL,
            arguments_json TEXT NOT NULL,
            discovered_dates INTEGER NOT NULL DEFAULT 0,
            selected_dates INTEGER NOT NULL DEFAULT 0,
            skipped_dates INTEGER NOT NULL DEFAULT 0,
            completed_dates INTEGER NOT NULL DEFAULT 0,
            failed_dates INTEGER NOT NULL DEFAULT 0,
            output_rows INTEGER NOT NULL DEFAULT 0,
            error_message TEXT
        );

        CREATE TABLE IF NOT EXISTS date_builds (
            symbol TEXT NOT NULL,
            receive_date TEXT NOT NULL,
            status TEXT NOT NULL,
            builder_version TEXT NOT NULL,
            output_schema_version INTEGER NOT NULL,
            config_signature TEXT NOT NULL,
            input_signature TEXT NOT NULL,
            carry_in_signature TEXT NOT NULL,
            input_manifest_json TEXT NOT NULL,
            output_relative_path TEXT,
            output_rows INTEGER NOT NULL DEFAULT 0,
            first_receive_second INTEGER,
            last_receive_second INTEGER,
            final_carry_json TEXT,
            summary_json TEXT,
            started_at_utc TEXT,
            completed_at_utc TEXT,
            last_run_id TEXT,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            error_message TEXT,
            updated_at_utc TEXT NOT NULL,
            PRIMARY KEY(symbol, receive_date)
        );

        CREATE INDEX IF NOT EXISTS idx_flow_date_build_status
            ON date_builds(status);
        """
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        ("builder_version", BUILDER_VERSION),
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        ("output_schema_version", str(OUTPUT_SCHEMA_VERSION)),
    )
    connection.execute(
        "UPDATE date_builds SET status='INTERRUPTED', updated_at_utc=? WHERE status='RUNNING'",
        (utc_now_iso(),),
    )
    connection.commit()
    return connection


def create_run(
    connection: sqlite3.Connection,
    run_id: str,
    core_root: Path,
    output_root: Path,
    args: argparse.Namespace,
    discovered_dates: int,
) -> None:
    connection.execute(
        """
        INSERT INTO build_runs(
            run_id, started_at_utc, status, core_root, output_root,
            arguments_json, discovered_dates
        ) VALUES (?, ?, 'RUNNING', ?, ?, ?, ?)
        """,
        (
            run_id,
            utc_now_iso(),
            str(core_root),
            str(output_root),
            compact_json(vars(args)),
            discovered_dates,
        ),
    )
    connection.commit()


def get_date_state(connection: sqlite3.Connection, symbol: str, day: date):
    return connection.execute(
        "SELECT * FROM date_builds WHERE symbol=? AND receive_date=?",
        (symbol, day.isoformat()),
    ).fetchone()


def state_carry(row: sqlite3.Row | None) -> CarryState | None:
    if row is None or row["status"] != "COMPLETE" or not row["final_carry_json"]:
        return None
    payload = json.loads(row["final_carry_json"])
    return CarryState(
        cvd_quote=float(payload.get("cvd_quote", 0.0)),
        cvd_base=float(payload.get("cvd_base", 0.0)),
    )


def mark_date_running(
    connection: sqlite3.Connection,
    symbol: str,
    day: date,
    run_id: str,
    config_signature: str,
    input_signature: str,
    carry_signature: str,
    manifests: Sequence[Mapping[str, Any]],
) -> None:
    now = utc_now_iso()
    connection.execute(
        """
        INSERT INTO date_builds(
            symbol, receive_date, status, builder_version, output_schema_version,
            config_signature, input_signature, carry_in_signature,
            input_manifest_json, started_at_utc, last_run_id, attempt_count,
            updated_at_utc
        ) VALUES (?, ?, 'RUNNING', ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
        ON CONFLICT(symbol, receive_date) DO UPDATE SET
            status='RUNNING',
            builder_version=excluded.builder_version,
            output_schema_version=excluded.output_schema_version,
            config_signature=excluded.config_signature,
            input_signature=excluded.input_signature,
            carry_in_signature=excluded.carry_in_signature,
            input_manifest_json=excluded.input_manifest_json,
            started_at_utc=excluded.started_at_utc,
            completed_at_utc=NULL,
            last_run_id=excluded.last_run_id,
            attempt_count=date_builds.attempt_count + 1,
            error_message=NULL,
            updated_at_utc=excluded.updated_at_utc
        """,
        (
            symbol,
            day.isoformat(),
            BUILDER_VERSION,
            OUTPUT_SCHEMA_VERSION,
            config_signature,
            input_signature,
            carry_signature,
            compact_json(list(manifests)),
            now,
            run_id,
            now,
        ),
    )
    connection.commit()


def mark_date_complete(
    connection: sqlite3.Connection,
    symbol: str,
    day: date,
    output_relative_path: str,
    summary: Mapping[str, Any],
    carry: CarryState,
) -> None:
    now = utc_now_iso()
    connection.execute(
        """
        UPDATE date_builds SET
            status='COMPLETE',
            output_relative_path=?,
            output_rows=?,
            first_receive_second=?,
            last_receive_second=?,
            final_carry_json=?,
            summary_json=?,
            completed_at_utc=?,
            error_message=NULL,
            updated_at_utc=?
        WHERE symbol=? AND receive_date=?
        """,
        (
            output_relative_path,
            int(summary["output_rows"]),
            int(summary["first_receive_second"]),
            int(summary["last_receive_second"]),
            compact_json(asdict(carry)),
            compact_json(summary),
            now,
            now,
            symbol,
            day.isoformat(),
        ),
    )
    connection.commit()


def mark_date_failed(
    connection: sqlite3.Connection,
    symbol: str,
    day: date,
    error_message: str,
) -> None:
    connection.execute(
        """
        UPDATE date_builds SET
            status='FAILED', error_message=?, completed_at_utc=?, updated_at_utc=?
        WHERE symbol=? AND receive_date=?
        """,
        (error_message, utc_now_iso(), utc_now_iso(), symbol, day.isoformat()),
    )
    connection.commit()


def finish_run(
    connection: sqlite3.Connection,
    run_id: str,
    status: str,
    selected: int,
    skipped: int,
    completed: int,
    failed: int,
    rows: int,
    error_message: str | None = None,
) -> None:
    connection.execute(
        """
        UPDATE build_runs SET
            completed_at_utc=?, status=?, selected_dates=?, skipped_dates=?,
            completed_dates=?, failed_dates=?, output_rows=?, error_message=?
        WHERE run_id=?
        """,
        (
            utc_now_iso(),
            status,
            selected,
            skipped,
            completed,
            failed,
            rows,
            error_message,
            run_id,
        ),
    )
    connection.commit()


# ---------------------------------------------------------------------------
# Build orchestration
# ---------------------------------------------------------------------------


def output_path_for(output_root: Path, symbol: str, day: date) -> Path:
    return (
        output_root
        / f"symbol={safe_symbol_partition(symbol)}"
        / f"date={day.isoformat()}"
        / f"part-{day.isoformat()}.parquet"
    )


def output_is_current(
    state: sqlite3.Row | None,
    output_path: Path,
    config_signature: str,
    input_signature: str,
    carry_signature: str,
) -> bool:
    if state is None or state["status"] != "COMPLETE" or not output_path.exists():
        return False
    if state["builder_version"] != BUILDER_VERSION:
        return False
    if int(state["output_schema_version"]) != OUTPUT_SCHEMA_VERSION:
        return False
    if state["config_signature"] != config_signature:
        return False
    if state["input_signature"] != input_signature:
        return False
    if state["carry_in_signature"] != carry_signature:
        return False
    try:
        require_arrow()
        parquet_file = pq.ParquetFile(output_path)
        try:
            return int(parquet_file.metadata.num_rows) == int(state["output_rows"])
        finally:
            parquet_file.close()
    except Exception:
        return False


def build_date(
    core_root: Path,
    output_root: Path,
    partitions_by_day: Mapping[date, DatePartition],
    all_days: Sequence[date],
    day: date,
    symbol: str,
    incoming_carry: CarryState,
    config: FeatureConfig,
    compression: str,
    compression_level: int | None,
    row_group_size: int,
    manifests: Sequence[Mapping[str, Any]],
    input_signature: str,
) -> tuple[dict[str, Any], CarryState, Path]:
    context_days = context_days_for(day, all_days, config.max_window)
    context = read_context(partitions_by_day, context_days)
    core_current = context.loc[
        (context["receive_second"] >= day_start_second(day))
        & (context["receive_second"] < day_start_second(day) + SECONDS_PER_DAY)
    ].reset_index(drop=True)

    frame, outgoing_carry, summary = build_feature_frame(
        context=context,
        output_day=day,
        incoming_carry=incoming_carry,
        config=config,
    )
    validate_output(frame, core_current, day, config)

    output_path = output_path_for(output_root, symbol, day)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")

    metadata = {
        "perps_flow_feature_schema_version": str(OUTPUT_SCHEMA_VERSION),
        "perps_flow_feature_builder_version": BUILDER_VERSION,
        "source_core_schema_version": str(EXPECTED_CORE_SCHEMA_VERSION),
        "symbol": symbol,
        "receive_date": day.isoformat(),
        "causal_clock": "receive_second",
        "availability_rule": "available_after_second=receive_second+1",
        "windows_seconds": compact_json(list(config.windows)),
        "zscore_windows_seconds": compact_json(list(config.zscore_windows)),
        "zscore_min_fraction": str(config.zscore_min_fraction),
        "min_impact_flow_notional": str(config.min_impact_flow_notional),
        "price_source_codes": compact_json({
            "0": "unavailable",
            "1": "fresh_trade_state",
            "2": "fresh_mark_state",
            "3": "stale_trade_state",
            "4": "stale_mark_state",
        }),
        "price_flow_alignment": "sign(return_bps)*sign(signed_quote_flow)",
        "net_liquidation_sign": "short_liquidation_notional-long_liquidation_notional",
        "input_signature": input_signature,
        "carry_in_signature": incoming_carry.signature(),
        "carry_out_signature": outgoing_carry.signature(),
        "input_manifest_json": compact_json(list(manifests)),
        "created_at_utc": utc_now_iso(),
    }

    table = frame_to_table(frame, config, metadata)
    write_kwargs: dict[str, Any] = {
        "compression": compression,
        "use_dictionary": True,
        "write_statistics": True,
        "row_group_size": row_group_size,
    }
    if compression_level is not None and compression in {"zstd", "gzip", "brotli"}:
        write_kwargs["compression_level"] = compression_level

    try:
        pq.write_table(table, temporary, **write_kwargs)

        # ParquetFile keeps an OS-level handle open on Windows. It must be
        # closed before the atomic rename or os.replace can raise WinError 32.
        written = pq.ParquetFile(temporary)
        try:
            written_rows = int(written.metadata.num_rows)
        finally:
            written.close()

        if written_rows != len(frame):
            raise RuntimeError(
                f"Temporary output row mismatch: {written_rows} != {len(frame)}"
            )

        del table
        gc.collect()
        replace_with_retry(temporary, output_path)
    finally:
        unlink_with_retry(temporary)

    return summary, outgoing_carry, output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build causal perps flow features from the audited core 1-second layer."
    )
    parser.add_argument(
        "--core-root",
        default="./data/derived/perps/core_1s_v1",
        help="Input perps core root.",
    )
    parser.add_argument(
        "--output-root",
        default="./data/derived/perps/flow_features_1s_v1",
        help="Output feature-layer root.",
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument(
        "--windows",
        default=",".join(str(item) for item in DEFAULT_WINDOWS),
        help="Exact return and rolling aggregation windows in seconds.",
    )
    parser.add_argument(
        "--zscore-windows",
        default=",".join(str(item) for item in DEFAULT_ZSCORE_WINDOWS),
        help="Rolling z-score windows in seconds.",
    )
    parser.add_argument(
        "--zscore-min-fraction",
        type=float,
        default=0.80,
        help="Minimum finite-value fraction required inside a complete z-score window.",
    )
    parser.add_argument(
        "--min-impact-flow-notional",
        type=float,
        default=100_000.0,
        help="Minimum absolute signed quote flow for price-response features.",
    )
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--compression-level", type=int, default=3)
    parser.add_argument("--row-group-size", type=int, default=86_400)
    parser.add_argument("--min-free-space-gb", type=float, default=10.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    require_arrow()
    parser = build_parser()
    args = parser.parse_args(argv)

    if not (0 < args.zscore_min_fraction <= 1):
        raise SystemExit("--zscore-min-fraction must be in (0, 1]")
    if args.min_impact_flow_notional < 0:
        raise SystemExit("--min-impact-flow-notional cannot be negative")
    if args.row_group_size <= 0:
        raise SystemExit("--row-group-size must be positive")

    start_date = parse_iso_date(args.start_date, "--start-date")
    end_date = parse_iso_date(args.end_date, "--end-date")
    if start_date and end_date and start_date > end_date:
        raise SystemExit("--start-date cannot be after --end-date")

    windows = parse_positive_int_list(args.windows, "--windows")
    zscore_windows = parse_positive_int_list(args.zscore_windows, "--zscore-windows")
    config = FeatureConfig(
        windows=windows,
        zscore_windows=zscore_windows,
        zscore_min_fraction=float(args.zscore_min_fraction),
        min_impact_flow_notional=float(args.min_impact_flow_notional),
    )

    core_root = Path(args.core_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    symbol = args.symbol.strip().upper()
    if not core_root.exists():
        raise SystemExit(f"Core root does not exist: {core_root}")

    all_partitions = discover_all_partitions(core_root, symbol)
    all_days = sorted(all_partitions)
    selected_days = [
        item
        for item in all_days
        if (start_date is None or item >= start_date)
        and (end_date is None or item <= end_date)
    ]
    if not selected_days:
        raise SystemExit("No dates remain after applying the requested date range")

    output_free = ensure_free_space(output_root, args.min_free_space_gb)
    state_path = output_root / "_build_state" / "build_state.sqlite"
    connection = open_state_database(state_path)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{uuid.uuid4().hex[:8]}"
    create_run(connection, run_id, core_root, output_root, args, len(selected_days))

    print("Planning perps flow-feature build...")
    print(f"Run ID          : {run_id}")
    print(f"Core root       : {core_root}")
    print(f"Output root     : {output_root}")
    print(f"State database  : {state_path}")
    print(f"Symbol          : {symbol}")
    print(f"Date range      : {selected_days[0]} to {selected_days[-1]} ({len(selected_days)} dates)")
    print(f"Windows         : {','.join(map(str, windows))}")
    print(f"Z-score windows : {','.join(map(str, zscore_windows))}")
    print(f"Maximum lookback: {config.max_window:,} seconds")
    print(f"Output free     : {output_free:.2f} GiB")
    print(f"Free-space guard: {args.min_free_space_gb:.2f} GiB")

    selected = skipped = completed = failed = rows_written = 0
    date_rows: list[dict[str, Any]] = []
    incoming_carry = CarryState()

    first_all_day = all_days[0]
    if selected_days[0] != first_all_day:
        previous_days = [item for item in all_days if item < selected_days[0]]
        if previous_days:
            previous_state = get_date_state(connection, symbol, previous_days[-1])
            restored = state_carry(previous_state)
            if restored is None:
                raise SystemExit(
                    "A later --start-date requires a COMPLETE prior date state so cumulative CVD "
                    "can be restored. Build from the dataset start first."
                )
            incoming_carry = restored

    try:
        for position, day in enumerate(selected_days, start=1):
            context_days = context_days_for(day, all_days, config.max_window)
            manifests: list[dict[str, Any]] = []
            for context_day in context_days:
                partition = all_partitions[context_day]
                for path in partition.files:
                    item = file_manifest(path, core_root)
                    item["context_receive_date"] = context_day.isoformat()
                    manifests.append(item)
            input_signature = manifest_signature(manifests)
            config_signature = config.signature()
            carry_signature = incoming_carry.signature()
            output_path = output_path_for(output_root, symbol, day)
            state = get_date_state(connection, symbol, day)

            if not args.force and output_is_current(
                state,
                output_path,
                config_signature,
                input_signature,
                carry_signature,
            ):
                restored = state_carry(state)
                if restored is None:
                    raise RuntimeError(f"COMPLETE state for {day} has no final carry")
                incoming_carry = restored
                skipped += 1
                summary = json.loads(state["summary_json"] or "{}")
                output_rows = int(state["output_rows"])
                date_rows.append({
                    "receive_date": day.isoformat(),
                    "status": "SKIPPED",
                    "output_rows": output_rows,
                    "input_signature": input_signature,
                    "carry_out_signature": incoming_carry.signature(),
                })
                print(f"SKIPPED   [{position:3d}/{len(selected_days)}] {day} rows={output_rows:,}")
                continue

            selected += 1
            if args.dry_run:
                print(f"SELECTED  [{position:3d}/{len(selected_days)}] {day} (dry run)")
                date_rows.append({
                    "receive_date": day.isoformat(),
                    "status": "SELECTED_DRY_RUN",
                    "output_rows": 0,
                    "input_signature": input_signature,
                    "carry_in_signature": carry_signature,
                })
                # A dry run cannot know changed carry. Stop after the first selected date,
                # otherwise later freshness decisions could be misleading.
                break

            ensure_free_space(output_root, args.min_free_space_gb)
            mark_date_running(
                connection,
                symbol,
                day,
                run_id,
                config_signature,
                input_signature,
                carry_signature,
                manifests,
            )
            try:
                summary, outgoing_carry, written_path = build_date(
                    core_root=core_root,
                    output_root=output_root,
                    partitions_by_day=all_partitions,
                    all_days=all_days,
                    day=day,
                    symbol=symbol,
                    incoming_carry=incoming_carry,
                    config=config,
                    compression=args.compression,
                    compression_level=args.compression_level,
                    row_group_size=args.row_group_size,
                    manifests=manifests,
                    input_signature=input_signature,
                )
                mark_date_complete(
                    connection,
                    symbol,
                    day,
                    written_path.relative_to(output_root).as_posix(),
                    summary,
                    outgoing_carry,
                )
                incoming_carry = outgoing_carry
                completed += 1
                rows_written += int(summary["output_rows"])
                date_rows.append({
                    "receive_date": day.isoformat(),
                    "status": "COMPLETED",
                    "output_rows": int(summary["output_rows"]),
                    "first_receive_second": int(summary["first_receive_second"]),
                    "last_receive_second": int(summary["last_receive_second"]),
                    "flow_feature_context_ready_rows": int(summary["flow_feature_context_ready_rows"]),
                    "input_signature": input_signature,
                    "carry_out_signature": outgoing_carry.signature(),
                })
                print(
                    f"COMPLETED [{position:3d}/{len(selected_days)}] {day} "
                    f"rows={int(summary['output_rows']):,}, "
                    f"ready={int(summary['flow_feature_context_ready_rows']):,}"
                )
            except Exception as exc:
                failed += 1
                message = f"{type(exc).__name__}: {exc}"
                mark_date_failed(connection, symbol, day, message)
                date_rows.append({
                    "receive_date": day.isoformat(),
                    "status": "FAILED",
                    "output_rows": 0,
                    "error_message": message,
                })
                print(f"FAILED    [{position:3d}/{len(selected_days)}] {day}: {message}", file=sys.stderr)
                traceback.print_exc()
                break

        status = "FAILED" if failed else ("DRY_RUN" if args.dry_run else "COMPLETED")
        finish_run(
            connection,
            run_id,
            status,
            selected,
            skipped,
            completed,
            failed,
            rows_written,
        )
    except Exception as exc:
        finish_run(
            connection,
            run_id,
            "FAILED",
            selected,
            skipped,
            completed,
            failed + 1,
            rows_written,
            f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        connection.close()

    summary_payload = {
        "run_id": run_id,
        "status": "FAILED" if failed else ("DRY_RUN" if args.dry_run else "COMPLETED"),
        "builder_version": BUILDER_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "core_root": str(core_root),
        "output_root": str(output_root),
        "symbol": symbol,
        "date_start": selected_days[0].isoformat(),
        "date_end": selected_days[-1].isoformat(),
        "discovered_dates": len(selected_days),
        "selected_dates": selected,
        "skipped_dates": skipped,
        "completed_dates": completed,
        "failed_dates": failed,
        "rows_written": rows_written,
        "configuration": asdict(config),
        "completed_at_utc": utc_now_iso(),
        "dates": date_rows,
    }
    state_root = output_root / "_build_state"
    atomic_write_json(state_root / "latest_run_summary.json", summary_payload)
    atomic_write_csv(state_root / "latest_date_summary.csv", date_rows)
    run_root = state_root / "runs" / run_id
    atomic_write_json(run_root / "run_summary.json", summary_payload)
    atomic_write_csv(run_root / "date_summary.csv", date_rows)

    print("\nPerps flow-feature build finished.")
    print(f"Status           : {summary_payload['status']}")
    print(f"Selected dates   : {selected}")
    print(f"Skipped dates    : {skipped}")
    print(f"Completed dates  : {completed}")
    print(f"Failed dates     : {failed}")
    print(f"Rows written     : {rows_written:,}")
    print(f"Summary          : {state_root / 'latest_run_summary.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
