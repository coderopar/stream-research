#!/usr/bin/env python
"""
Build the canonical Binance perpetual-futures one-second core measurement layer.

Input
-----
Canonical event families produced by ``scripts/01_extract_all_market_events.py``::

    perps/trades
    perps/mark_price
    perps/open_interest
    perps/liquidations

Output
------
One Parquet file per symbol and UTC receive date::

    data/derived/perps/core_1s_v1/
      symbol=BTCUSDT/
        date=YYYY-MM-DD/
          part-YYYY-MM-DD.parquet

The output is a measurement layer, not a trading-feature or labelled dataset.
Each row represents one receive-time UTC second and is available only after that
second closes. Aggregates use records whose ``ts_recv_ns`` falls inside that
second. Mark price, index price, funding and open interest are carried forward
causally with explicit update, age and freshness fields.

Important properties
--------------------
* Receive time is the availability clock. Exchange timestamps are retained only
  as audit/data-quality fields.
* Exact replay duplicates are removed for derived measurements, while raw and
  unique record counts are both retained for reconciliation.
* ``buyer_is_maker=True`` means the aggressive side was SELL; false means BUY.
* Binance force-order side SELL is classified as a long liquidation; BUY is a
  short liquidation.
* Builds are resumable at symbol/date level through SQLite state.
* Input signatures include current canonical shards, builder configuration and
  the incoming causal carry state. Changed inputs automatically rebuild the
  affected date and propagate changed carry state into later dates.
* Output replacement is atomic. An interrupted date is retried on the next run.

This script intentionally does not calculate rolling returns, z-scores, Volume
Profiles, order-book features, participant states, labels or future outcomes.
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
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

try:
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - environment-specific
    raise SystemExit(
        "Missing dependency. Activate the project virtual environment and run: "
        "pip install pandas pyarrow"
    ) from exc


BUILDER_VERSION = "1.0.1"
OUTPUT_SCHEMA_VERSION = 1
SECONDS_PER_DAY = 86_400
NANOSECONDS_PER_SECOND = 1_000_000_000
DATE_PARTITION_RE = re.compile(r"^date=(\d{4}-\d{2}-\d{2})$")
SYMBOL_PARTITION_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")

FAMILIES = ("trades", "mark_price", "open_interest", "liquidations")
FAMILY_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "trades": (
        "symbol",
        "ts_recv_ns",
        "receive_second",
        "event_id",
        "source_file_id",
        "source_record_number",
        "exchange_event_ms",
        "trade_time_ms",
        "aggregate_trade_id",
        "price",
        "quantity",
        "normalized_quantity",
        "buyer_is_maker",
    ),
    "mark_price": (
        "symbol",
        "ts_recv_ns",
        "receive_second",
        "event_id",
        "source_file_id",
        "source_record_number",
        "exchange_event_ms",
        "mark_price",
        "alternate_mark_price",
        "estimated_settle_price",
        "index_price",
        "funding_rate",
        "next_funding_time_ms",
    ),
    "open_interest": (
        "symbol",
        "ts_recv_ns",
        "receive_second",
        "event_id",
        "source_file_id",
        "source_record_number",
        "exchange_time_ms",
        "open_interest",
    ),
    "liquidations": (
        "symbol",
        "ts_recv_ns",
        "receive_second",
        "event_id",
        "source_file_id",
        "source_record_number",
        "exchange_event_ms",
        "side",
        "original_quantity",
        "price",
        "average_price",
        "last_filled_quantity",
        "cumulative_filled_quantity",
        "trade_time_ms",
    ),
}


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------


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


def parse_csv_values(value: str | None) -> set[str] | None:
    if not value:
        return None
    result = {item.strip() for item in value.split(",") if item.strip()}
    return result or None


def safe_symbol_partition(symbol: str) -> str:
    cleaned = SYMBOL_PARTITION_SAFE_RE.sub("_", symbol.strip()).strip("_")
    if not cleaned:
        raise SystemExit(f"Invalid --symbol value: {symbol!r}")
    return cleaned


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def available_bytes(path: Path) -> int:
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return int(shutil.disk_usage(probe).free)


def format_gib(value: int) -> str:
    return f"{value / (1024 ** 3):.2f} GiB"


def day_start_second(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())


def receive_day_from_second(second: int) -> date:
    return datetime.fromtimestamp(second, tz=timezone.utc).date()


def nullable_int_series(values: np.ndarray, missing_sentinel: int = -1) -> pd.Series:
    series = pd.Series(values.copy())
    series = series.mask(series.eq(missing_sentinel))
    return series.astype("Int64")


def ensure_finite_numeric(
    frame: pd.DataFrame,
    column: str,
    *,
    allow_null: bool,
    context: str,
) -> pd.Series:
    original = frame[column]
    values = pd.to_numeric(original, errors="coerce")
    supplied = original.notna() & original.astype(str).str.strip().ne("")
    invalid = supplied & (~np.isfinite(values))
    if invalid.any():
        samples = original.loc[invalid].astype(str).head(5).tolist()
        raise RuntimeError(
            f"{context}: {int(invalid.sum()):,} non-numeric {column} values; "
            f"examples={samples}"
        )
    if not allow_null and values.isna().any():
        raise RuntimeError(f"{context}: {int(values.isna().sum()):,} missing {column} values")
    return values.astype("float64")


def read_parquet_metadata_rows(path: Path) -> int:
    try:
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception as exc:
        raise RuntimeError(f"Cannot read Parquet footer {path}: {exc}") from exc


def iter_row_groups(path: Path, columns: Sequence[str]) -> Iterator[pd.DataFrame]:
    try:
        parquet_file = pq.ParquetFile(path)
    except Exception as exc:
        raise RuntimeError(f"Cannot open Parquet file {path}: {exc}") from exc

    available = set(parquet_file.schema_arrow.names)
    missing = sorted(set(columns) - available)
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {missing}")

    for row_group_index in range(parquet_file.num_row_groups):
        try:
            table = parquet_file.read_row_group(row_group_index, columns=list(columns))
            yield table.to_pandas(split_blocks=True, self_destruct=True)
        except Exception as exc:
            raise RuntimeError(
                f"Failed reading row group {row_group_index} from {path}: {exc}"
            ) from exc


def discover_partition_dates(canonical_root: Path) -> list[date]:
    values: set[date] = set()
    for family in FAMILIES:
        family_dir = canonical_root / "perps" / family
        if not family_dir.exists():
            continue
        for child in family_dir.iterdir():
            if not child.is_dir():
                continue
            match = DATE_PARTITION_RE.match(child.name)
            if not match:
                continue
            try:
                values.add(datetime.strptime(match.group(1), "%Y-%m-%d").date())
            except ValueError:
                continue
    return sorted(values)


def family_files(canonical_root: Path, family: str, day: date) -> list[Path]:
    directory = canonical_root / "perps" / family / f"date={day.isoformat()}"
    return sorted(directory.glob("*.parquet")) if directory.exists() else []


def input_manifest(canonical_root: Path, day: date) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[Path]]]:
    manifest: dict[str, list[dict[str, Any]]] = {}
    paths_by_family: dict[str, list[Path]] = {}
    for family in FAMILIES:
        paths = family_files(canonical_root, family, day)
        paths_by_family[family] = paths
        entries: list[dict[str, Any]] = []
        for path in paths:
            stat = path.stat()
            entries.append(
                {
                    "path": path.relative_to(canonical_root).as_posix(),
                    "size_bytes": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "rows": read_parquet_metadata_rows(path),
                }
            )
        manifest[family] = entries
    return manifest, paths_by_family


# ---------------------------------------------------------------------------
# Carry state
# ---------------------------------------------------------------------------


@dataclass
class CarryState:
    trade_price: float | None = None
    trade_ts_recv_ns: int | None = None
    trade_exchange_event_ms: int | None = None
    trade_time_ms: int | None = None

    mark_price: float | None = None
    alternate_mark_price: float | None = None
    estimated_settle_price: float | None = None
    index_price: float | None = None
    funding_rate: float | None = None
    next_funding_time_ms: int | None = None
    mark_ts_recv_ns: int | None = None
    mark_exchange_event_ms: int | None = None

    open_interest: float | None = None
    open_interest_ts_recv_ns: int | None = None
    open_interest_exchange_time_ms: int | None = None

    def signature(self) -> str:
        return sha256_text(compact_json(asdict(self)))

    @classmethod
    def from_json(cls, value: str | None) -> "CarryState":
        if not value:
            return cls()
        parsed = json.loads(value)
        allowed = {field_name for field_name in cls.__dataclass_fields__}
        return cls(**{key: parsed.get(key) for key in allowed})


@dataclass
class LatestCandidate:
    ts_recv_ns: int = -1
    source_file_id: str = ""
    source_record_number: int = -1
    values: dict[str, Any] = field(default_factory=dict)

    def consider(self, row: Mapping[str, Any], values: dict[str, Any]) -> None:
        ts = int(row["ts_recv_ns"])
        source_file_id = str(row.get("source_file_id") or "")
        source_record_number = int(row.get("source_record_number") or -1)
        key = (ts, source_file_id, source_record_number)
        current = (self.ts_recv_ns, self.source_file_id, self.source_record_number)
        if key > current:
            self.ts_recv_ns = ts
            self.source_file_id = source_file_id
            self.source_record_number = source_record_number
            self.values = values


def latest_state_from_date(
    canonical_root: Path,
    symbol: str,
    day: date,
    incoming: CarryState,
) -> CarryState:
    """Update carry state from the latest canonical observations on one date.

    This lightweight scan is used for dry-run propagation and initial carry
    discovery. Actual builds produce the same final carry during aggregation.
    """
    carry = CarryState(**asdict(incoming))

    trade_candidate = LatestCandidate()
    for path in family_files(canonical_root, "trades", day):
        columns = (
            "symbol", "ts_recv_ns", "source_file_id", "source_record_number",
            "exchange_event_ms", "trade_time_ms", "price",
        )
        for frame in iter_row_groups(path, columns):
            frame = frame[frame["symbol"].astype(str).eq(symbol)]
            if frame.empty:
                continue
            frame["price_num"] = pd.to_numeric(frame["price"], errors="coerce")
            frame = frame[np.isfinite(frame["price_num"]) & frame["price_num"].gt(0)]
            if frame.empty:
                continue
            frame = frame.sort_values(
                ["ts_recv_ns", "source_file_id", "source_record_number"]
            )
            row = frame.iloc[-1]
            trade_candidate.consider(
                row,
                {
                    "trade_price": float(row["price_num"]),
                    "trade_ts_recv_ns": int(row["ts_recv_ns"]),
                    "trade_exchange_event_ms": int(row["exchange_event_ms"])
                    if pd.notna(row["exchange_event_ms"]) else None,
                    "trade_time_ms": int(row["trade_time_ms"])
                    if pd.notna(row["trade_time_ms"]) else None,
                },
            )
    if trade_candidate.values:
        for key, value in trade_candidate.values.items():
            setattr(carry, key, value)

    mark_candidate = LatestCandidate()
    mark_value_columns = (
        "mark_price", "alternate_mark_price", "estimated_settle_price",
        "index_price", "funding_rate",
    )
    for path in family_files(canonical_root, "mark_price", day):
        columns = (
            "symbol", "ts_recv_ns", "source_file_id", "source_record_number",
            "exchange_event_ms", *mark_value_columns, "next_funding_time_ms",
        )
        for frame in iter_row_groups(path, columns):
            frame = frame[frame["symbol"].astype(str).eq(symbol)]
            if frame.empty:
                continue
            for column in mark_value_columns:
                frame[f"{column}_num"] = pd.to_numeric(frame[column], errors="coerce")
            frame = frame.sort_values(
                ["ts_recv_ns", "source_file_id", "source_record_number"]
            )
            row = frame.iloc[-1]
            mark_candidate.consider(
                row,
                {
                    "mark_price": float(row["mark_price_num"])
                    if pd.notna(row["mark_price_num"]) else None,
                    "alternate_mark_price": float(row["alternate_mark_price_num"])
                    if pd.notna(row["alternate_mark_price_num"]) else None,
                    "estimated_settle_price": float(row["estimated_settle_price_num"])
                    if pd.notna(row["estimated_settle_price_num"]) else None,
                    "index_price": float(row["index_price_num"])
                    if pd.notna(row["index_price_num"]) else None,
                    "funding_rate": float(row["funding_rate_num"])
                    if pd.notna(row["funding_rate_num"]) else None,
                    "next_funding_time_ms": int(row["next_funding_time_ms"])
                    if pd.notna(row["next_funding_time_ms"]) else None,
                    "mark_ts_recv_ns": int(row["ts_recv_ns"]),
                    "mark_exchange_event_ms": int(row["exchange_event_ms"])
                    if pd.notna(row["exchange_event_ms"]) else None,
                },
            )
    if mark_candidate.values:
        for key, value in mark_candidate.values.items():
            setattr(carry, key, value)

    oi_candidate = LatestCandidate()
    for path in family_files(canonical_root, "open_interest", day):
        columns = (
            "symbol", "ts_recv_ns", "source_file_id", "source_record_number",
            "exchange_time_ms", "open_interest",
        )
        for frame in iter_row_groups(path, columns):
            frame = frame[frame["symbol"].astype(str).eq(symbol)]
            if frame.empty:
                continue
            frame["open_interest_num"] = pd.to_numeric(frame["open_interest"], errors="coerce")
            frame = frame[np.isfinite(frame["open_interest_num"])]
            if frame.empty:
                continue
            frame = frame.sort_values(
                ["ts_recv_ns", "source_file_id", "source_record_number"]
            )
            row = frame.iloc[-1]
            oi_candidate.consider(
                row,
                {
                    "open_interest": float(row["open_interest_num"]),
                    "open_interest_ts_recv_ns": int(row["ts_recv_ns"]),
                    "open_interest_exchange_time_ms": int(row["exchange_time_ms"])
                    if pd.notna(row["exchange_time_ms"]) else None,
                },
            )
    if oi_candidate.values:
        for key, value in oi_candidate.values.items():
            setattr(carry, key, value)

    return carry


def find_initial_carry(
    canonical_root: Path,
    symbol: str,
    first_day: date,
    lookback_days: int,
) -> CarryState:
    carry = CarryState()
    start = first_day - timedelta(days=lookback_days)
    current = start
    while current < first_day:
        carry = latest_state_from_date(canonical_root, symbol, current, carry)
        current += timedelta(days=1)
    return carry


# ---------------------------------------------------------------------------
# Date buffers and family aggregation
# ---------------------------------------------------------------------------


@dataclass
class DayStats:
    canonical_rows: dict[str, int] = field(default_factory=lambda: {family: 0 for family in FAMILIES})
    symbol_rows: dict[str, int] = field(default_factory=lambda: {family: 0 for family in FAMILIES})
    unique_rows: dict[str, int] = field(default_factory=lambda: {family: 0 for family in FAMILIES})
    duplicates_removed: dict[str, int] = field(default_factory=lambda: {family: 0 for family in FAMILIES})
    numeric_quantity_fallback_rows: int = 0
    unknown_liquidation_side_rows: int = 0


class DayBuffers:
    def __init__(self, day: date) -> None:
        self.day = day
        self.day_start = day_start_second(day)
        self.observed_min_index = SECONDS_PER_DAY
        self.observed_max_index = -1
        n = SECONDS_PER_DAY

        # Raw and unique observations by family.
        self.trade_raw_count = np.zeros(n, dtype=np.int64)
        self.trade_count = np.zeros(n, dtype=np.int64)
        self.mark_raw_count = np.zeros(n, dtype=np.int64)
        self.mark_update_count = np.zeros(n, dtype=np.int64)
        self.oi_raw_count = np.zeros(n, dtype=np.int64)
        self.oi_update_count = np.zeros(n, dtype=np.int64)
        self.liq_raw_count = np.zeros(n, dtype=np.int64)
        self.liq_count = np.zeros(n, dtype=np.int64)

        # Trade aggregates.
        self.trade_open = np.full(n, np.nan)
        self.trade_high = np.full(n, np.nan)
        self.trade_low = np.full(n, np.nan)
        self.trade_close = np.full(n, np.nan)
        self.trade_first_ts = np.full(n, -1, dtype=np.int64)
        self.trade_last_ts = np.full(n, -1, dtype=np.int64)
        self.trade_first_exchange_ms = np.full(n, -1, dtype=np.int64)
        self.trade_last_exchange_ms = np.full(n, -1, dtype=np.int64)
        self.trade_first_trade_ms = np.full(n, -1, dtype=np.int64)
        self.trade_last_trade_ms = np.full(n, -1, dtype=np.int64)
        self.base_volume = np.zeros(n)
        self.quote_volume = np.zeros(n)
        self.buy_base_volume = np.zeros(n)
        self.sell_base_volume = np.zeros(n)
        self.unknown_base_volume = np.zeros(n)
        self.buy_quote_volume = np.zeros(n)
        self.sell_quote_volume = np.zeros(n)
        self.unknown_quote_volume = np.zeros(n)
        self.buy_trade_count = np.zeros(n, dtype=np.int64)
        self.sell_trade_count = np.zeros(n, dtype=np.int64)
        self.unknown_trade_count = np.zeros(n, dtype=np.int64)
        self.max_trade_base = np.zeros(n)
        self.max_trade_notional = np.zeros(n)

        # Latest mark observation inside each second.
        self.mark_event_ts = np.full(n, -1, dtype=np.int64)
        self.mark_exchange_ms = np.full(n, -1, dtype=np.int64)
        self.mark_event_value = np.full(n, np.nan)
        self.alt_mark_event_value = np.full(n, np.nan)
        self.settle_event_value = np.full(n, np.nan)
        self.index_event_value = np.full(n, np.nan)
        self.funding_event_value = np.full(n, np.nan)
        self.next_funding_event_ms = np.full(n, -1, dtype=np.int64)

        # Latest OI observation inside each second.
        self.oi_event_ts = np.full(n, -1, dtype=np.int64)
        self.oi_exchange_ms = np.full(n, -1, dtype=np.int64)
        self.oi_event_value = np.full(n, np.nan)

        # Liquidations.
        self.long_liq_count = np.zeros(n, dtype=np.int64)
        self.short_liq_count = np.zeros(n, dtype=np.int64)
        self.unknown_liq_count = np.zeros(n, dtype=np.int64)
        self.long_liq_base = np.zeros(n)
        self.short_liq_base = np.zeros(n)
        self.unknown_liq_base = np.zeros(n)
        self.long_liq_notional = np.zeros(n)
        self.short_liq_notional = np.zeros(n)
        self.unknown_liq_notional = np.zeros(n)
        self.max_liq_notional = np.zeros(n)

    def note_indices(self, indices: np.ndarray) -> None:
        if indices.size == 0:
            return
        self.observed_min_index = min(self.observed_min_index, int(indices.min()))
        self.observed_max_index = max(self.observed_max_index, int(indices.max()))

    def indices_for_seconds(self, seconds: pd.Series, context: str) -> np.ndarray:
        values = pd.to_numeric(seconds, errors="coerce")
        if values.isna().any():
            raise RuntimeError(f"{context}: missing or invalid receive_second")
        indices = values.astype("int64").to_numpy() - self.day_start
        invalid = (indices < 0) | (indices >= SECONDS_PER_DAY)
        if invalid.any():
            samples = values.iloc[np.flatnonzero(invalid)[:5]].tolist()
            raise RuntimeError(
                f"{context}: rows fall outside receive partition {self.day}: {samples}"
            )
        self.note_indices(indices)
        return indices.astype(np.int64)


@dataclass
class DedupTracker:
    primary_seen: set[Any] = field(default_factory=set)
    event_seen: set[str] = field(default_factory=set)
    source_seen: set[tuple[str, int]] = field(default_factory=set)


def deduplicate_frame(
    frame: pd.DataFrame,
    *,
    primary_key: str,
    tracker: DedupTracker,
) -> tuple[pd.DataFrame, int]:
    """Remove exact replay duplicates with vectorised chunk-local checks.

    The preferred stable identifier is ``aggregate_trade_id`` for aggTrade and
    ``event_id`` for the other canonical families. Missing identifiers fall
    back to immutable source provenance. Seen sets are date-local, which keeps
    memory bounded while removing duplicate replay rows across shards and row
    groups inside the same receive date.
    """
    if frame.empty:
        return frame, 0

    keep = np.ones(len(frame), dtype=bool)
    unresolved = np.ones(len(frame), dtype=bool)

    if primary_key != "event_id":
        primary = pd.to_numeric(frame[primary_key], errors="coerce")
        valid_positions = np.flatnonzero(primary.notna().to_numpy())
        if valid_positions.size:
            values = primary.iloc[valid_positions].astype("int64")
            duplicate_mask = values.duplicated(keep="first").to_numpy(copy=True)
            if tracker.primary_seen:
                # Pandas may expose Arrow-backed boolean arrays as read-only NumPy
                # views. Avoid in-place mutation of such views by materialising a
                # writable copy and combining into a new array.
                seen_mask = values.isin(tracker.primary_seen).to_numpy(copy=True)
                duplicate_mask = np.logical_or(duplicate_mask, seen_mask)
            keep[valid_positions] = ~duplicate_mask
            accepted = values.iloc[np.flatnonzero(~duplicate_mask)].tolist()
            tracker.primary_seen.update(int(value) for value in accepted)
            unresolved[valid_positions] = False

    # For event-id-primary families, or rows missing the numeric primary key.
    event_positions = np.flatnonzero(unresolved)
    if event_positions.size:
        event_values = frame["event_id"].iloc[event_positions].astype("string")
        valid_event = event_values.notna() & event_values.str.strip().ne("")
        valid_local_positions = np.flatnonzero(valid_event.to_numpy())
        if valid_local_positions.size:
            positions = event_positions[valid_local_positions]
            values = event_values.iloc[valid_local_positions].astype(str)
            duplicate_mask = values.duplicated(keep="first").to_numpy(copy=True)
            if tracker.event_seen:
                # See numeric-primary branch above: keep this writable and avoid
                # in-place operators on potentially read-only Arrow-backed views.
                seen_mask = values.isin(tracker.event_seen).to_numpy(copy=True)
                duplicate_mask = np.logical_or(duplicate_mask, seen_mask)
            keep[positions] = ~duplicate_mask
            accepted = values.iloc[np.flatnonzero(~duplicate_mask)].tolist()
            tracker.event_seen.update(accepted)
            unresolved[positions] = False

    # Source provenance is unique by construction and is the final fallback.
    source_positions = np.flatnonzero(unresolved)
    if source_positions.size:
        source_keys = [
            (str(file_id), int(record_number))
            for file_id, record_number in zip(
                frame["source_file_id"].iloc[source_positions],
                frame["source_record_number"].iloc[source_positions],
                strict=True,
            )
        ]
        for local_index, key in enumerate(source_keys):
            position = int(source_positions[local_index])
            if key in tracker.source_seen:
                keep[position] = False
            else:
                tracker.source_seen.add(key)

    duplicate_count = int((~keep).sum())
    return frame.loc[keep].copy(), duplicate_count


def aggregate_trades(
    paths: Sequence[Path],
    symbol: str,
    buffers: DayBuffers,
    stats: DayStats,
) -> CarryState:
    tracker = DedupTracker()
    latest = LatestCandidate()

    for path in paths:
        stats.canonical_rows["trades"] += read_parquet_metadata_rows(path)
        for frame in iter_row_groups(path, FAMILY_REQUIRED_COLUMNS["trades"]):
            frame = frame[frame["symbol"].astype(str).eq(symbol)].copy()
            stats.symbol_rows["trades"] += len(frame)
            if frame.empty:
                continue
            indices_all = buffers.indices_for_seconds(frame["receive_second"], f"trades:{path}")
            np.add.at(buffers.trade_raw_count, indices_all, 1)

            frame, duplicate_count = deduplicate_frame(
                frame, primary_key="aggregate_trade_id", tracker=tracker
            )
            stats.duplicates_removed["trades"] += duplicate_count
            stats.unique_rows["trades"] += len(frame)
            if frame.empty:
                continue

            frame["second_index"] = buffers.indices_for_seconds(
                frame["receive_second"], f"trades:{path}"
            )
            price = ensure_finite_numeric(
                frame, "price", allow_null=False, context=f"trades:{path}"
            )
            quantity = pd.to_numeric(frame["quantity"], errors="coerce")
            normalized = pd.to_numeric(frame["normalized_quantity"], errors="coerce")
            use_fallback = (~np.isfinite(quantity)) & np.isfinite(normalized)
            stats.numeric_quantity_fallback_rows += int(use_fallback.sum())
            quantity = quantity.where(~use_fallback, normalized)
            if (~np.isfinite(quantity)).any():
                raise RuntimeError(
                    f"trades:{path}: {int((~np.isfinite(quantity)).sum()):,} rows have no usable quantity"
                )
            if (price <= 0).any() or (quantity <= 0).any():
                raise RuntimeError(f"trades:{path}: non-positive price or quantity encountered")

            frame["price_num"] = price
            frame["quantity_num"] = quantity.astype("float64")
            frame["quote_num"] = frame["price_num"] * frame["quantity_num"]
            frame["buyer_is_maker_bool"] = frame["buyer_is_maker"].astype("boolean")

            grouped = frame.groupby("second_index", sort=False)
            counts = grouped.size()
            idx = counts.index.to_numpy(dtype=np.int64)
            buffers.trade_count[idx] += counts.to_numpy(dtype=np.int64)
            buffers.base_volume[idx] += grouped["quantity_num"].sum().to_numpy()
            buffers.quote_volume[idx] += grouped["quote_num"].sum().to_numpy()

            high = grouped["price_num"].max()
            low = grouped["price_num"].min()
            for second_index, value in high.items():
                current = buffers.trade_high[int(second_index)]
                buffers.trade_high[int(second_index)] = value if np.isnan(current) else max(current, value)
            for second_index, value in low.items():
                current = buffers.trade_low[int(second_index)]
                buffers.trade_low[int(second_index)] = value if np.isnan(current) else min(current, value)

            base_max = grouped["quantity_num"].max()
            quote_max = grouped["quote_num"].max()
            for second_index, value in base_max.items():
                buffers.max_trade_base[int(second_index)] = max(
                    buffers.max_trade_base[int(second_index)], float(value)
                )
            for second_index, value in quote_max.items():
                buffers.max_trade_notional[int(second_index)] = max(
                    buffers.max_trade_notional[int(second_index)], float(value)
                )

            buy = frame[frame["buyer_is_maker_bool"].eq(False)]
            sell = frame[frame["buyer_is_maker_bool"].eq(True)]
            unknown = frame[frame["buyer_is_maker_bool"].isna()]
            for subset, count_buffer, base_buffer, quote_buffer in (
                (buy, buffers.buy_trade_count, buffers.buy_base_volume, buffers.buy_quote_volume),
                (sell, buffers.sell_trade_count, buffers.sell_base_volume, buffers.sell_quote_volume),
                (unknown, buffers.unknown_trade_count, buffers.unknown_base_volume, buffers.unknown_quote_volume),
            ):
                if subset.empty:
                    continue
                side_grouped = subset.groupby("second_index", sort=False)
                side_counts = side_grouped.size()
                side_idx = side_counts.index.to_numpy(dtype=np.int64)
                count_buffer[side_idx] += side_counts.to_numpy(dtype=np.int64)
                base_buffer[side_idx] += side_grouped["quantity_num"].sum().to_numpy()
                quote_buffer[side_idx] += side_grouped["quote_num"].sum().to_numpy()

            ordered = frame.sort_values(
                ["second_index", "ts_recv_ns", "source_file_id", "source_record_number"]
            )
            first_rows = ordered.drop_duplicates("second_index", keep="first")
            last_rows = ordered.drop_duplicates("second_index", keep="last")

            for row in first_rows.itertuples(index=False):
                second_index = int(row.second_index)
                ts = int(row.ts_recv_ns)
                if buffers.trade_first_ts[second_index] < 0 or ts < buffers.trade_first_ts[second_index]:
                    buffers.trade_first_ts[second_index] = ts
                    buffers.trade_open[second_index] = float(row.price_num)
                    buffers.trade_first_exchange_ms[second_index] = (
                        int(row.exchange_event_ms) if pd.notna(row.exchange_event_ms) else -1
                    )
                    buffers.trade_first_trade_ms[second_index] = (
                        int(row.trade_time_ms) if pd.notna(row.trade_time_ms) else -1
                    )

            for row in last_rows.itertuples(index=False):
                second_index = int(row.second_index)
                ts = int(row.ts_recv_ns)
                if ts >= buffers.trade_last_ts[second_index]:
                    buffers.trade_last_ts[second_index] = ts
                    buffers.trade_close[second_index] = float(row.price_num)
                    buffers.trade_last_exchange_ms[second_index] = (
                        int(row.exchange_event_ms) if pd.notna(row.exchange_event_ms) else -1
                    )
                    buffers.trade_last_trade_ms[second_index] = (
                        int(row.trade_time_ms) if pd.notna(row.trade_time_ms) else -1
                    )

            last_row = ordered.iloc[-1]
            latest.consider(
                last_row,
                {
                    "trade_price": float(last_row["price_num"]),
                    "trade_ts_recv_ns": int(last_row["ts_recv_ns"]),
                    "trade_exchange_event_ms": int(last_row["exchange_event_ms"])
                    if pd.notna(last_row["exchange_event_ms"]) else None,
                    "trade_time_ms": int(last_row["trade_time_ms"])
                    if pd.notna(last_row["trade_time_ms"]) else None,
                },
            )

    return CarryState(**latest.values) if latest.values else CarryState()


def aggregate_mark_price(
    paths: Sequence[Path],
    symbol: str,
    buffers: DayBuffers,
    stats: DayStats,
) -> CarryState:
    tracker = DedupTracker()
    latest = LatestCandidate()
    numeric_columns = (
        "mark_price", "alternate_mark_price", "estimated_settle_price",
        "index_price", "funding_rate",
    )

    for path in paths:
        stats.canonical_rows["mark_price"] += read_parquet_metadata_rows(path)
        for frame in iter_row_groups(path, FAMILY_REQUIRED_COLUMNS["mark_price"]):
            frame = frame[frame["symbol"].astype(str).eq(symbol)].copy()
            stats.symbol_rows["mark_price"] += len(frame)
            if frame.empty:
                continue
            indices_all = buffers.indices_for_seconds(frame["receive_second"], f"mark_price:{path}")
            np.add.at(buffers.mark_raw_count, indices_all, 1)

            frame, duplicate_count = deduplicate_frame(frame, primary_key="event_id", tracker=tracker)
            stats.duplicates_removed["mark_price"] += duplicate_count
            stats.unique_rows["mark_price"] += len(frame)
            if frame.empty:
                continue

            frame["second_index"] = buffers.indices_for_seconds(
                frame["receive_second"], f"mark_price:{path}"
            )
            for column in numeric_columns:
                frame[f"{column}_num"] = ensure_finite_numeric(
                    frame,
                    column,
                    allow_null=column not in {"mark_price", "index_price"},
                    context=f"mark_price:{path}",
                )

            counts = frame.groupby("second_index", sort=False).size()
            idx = counts.index.to_numpy(dtype=np.int64)
            buffers.mark_update_count[idx] += counts.to_numpy(dtype=np.int64)

            ordered = frame.sort_values(
                ["second_index", "ts_recv_ns", "source_file_id", "source_record_number"]
            )
            last_rows = ordered.drop_duplicates("second_index", keep="last")
            for row in last_rows.itertuples(index=False):
                second_index = int(row.second_index)
                ts = int(row.ts_recv_ns)
                if ts < buffers.mark_event_ts[second_index]:
                    continue
                buffers.mark_event_ts[second_index] = ts
                buffers.mark_exchange_ms[second_index] = (
                    int(row.exchange_event_ms) if pd.notna(row.exchange_event_ms) else -1
                )
                buffers.mark_event_value[second_index] = float(row.mark_price_num)
                buffers.alt_mark_event_value[second_index] = float(row.alternate_mark_price_num)
                buffers.settle_event_value[second_index] = float(row.estimated_settle_price_num)
                buffers.index_event_value[second_index] = float(row.index_price_num)
                buffers.funding_event_value[second_index] = float(row.funding_rate_num)
                buffers.next_funding_event_ms[second_index] = (
                    int(row.next_funding_time_ms) if pd.notna(row.next_funding_time_ms) else -1
                )

            last_row = ordered.iloc[-1]
            latest.consider(
                last_row,
                {
                    "mark_price": float(last_row["mark_price_num"])
                    if pd.notna(last_row["mark_price_num"]) else None,
                    "alternate_mark_price": float(last_row["alternate_mark_price_num"])
                    if pd.notna(last_row["alternate_mark_price_num"]) else None,
                    "estimated_settle_price": float(last_row["estimated_settle_price_num"])
                    if pd.notna(last_row["estimated_settle_price_num"]) else None,
                    "index_price": float(last_row["index_price_num"])
                    if pd.notna(last_row["index_price_num"]) else None,
                    "funding_rate": float(last_row["funding_rate_num"])
                    if pd.notna(last_row["funding_rate_num"]) else None,
                    "next_funding_time_ms": int(last_row["next_funding_time_ms"])
                    if pd.notna(last_row["next_funding_time_ms"]) else None,
                    "mark_ts_recv_ns": int(last_row["ts_recv_ns"]),
                    "mark_exchange_event_ms": int(last_row["exchange_event_ms"])
                    if pd.notna(last_row["exchange_event_ms"]) else None,
                },
            )

    return CarryState(**latest.values) if latest.values else CarryState()


def aggregate_open_interest(
    paths: Sequence[Path],
    symbol: str,
    buffers: DayBuffers,
    stats: DayStats,
) -> CarryState:
    tracker = DedupTracker()
    latest = LatestCandidate()

    for path in paths:
        stats.canonical_rows["open_interest"] += read_parquet_metadata_rows(path)
        for frame in iter_row_groups(path, FAMILY_REQUIRED_COLUMNS["open_interest"]):
            frame = frame[frame["symbol"].astype(str).eq(symbol)].copy()
            stats.symbol_rows["open_interest"] += len(frame)
            if frame.empty:
                continue
            indices_all = buffers.indices_for_seconds(frame["receive_second"], f"open_interest:{path}")
            np.add.at(buffers.oi_raw_count, indices_all, 1)

            frame, duplicate_count = deduplicate_frame(frame, primary_key="event_id", tracker=tracker)
            stats.duplicates_removed["open_interest"] += duplicate_count
            stats.unique_rows["open_interest"] += len(frame)
            if frame.empty:
                continue

            frame["second_index"] = buffers.indices_for_seconds(
                frame["receive_second"], f"open_interest:{path}"
            )
            frame["open_interest_num"] = ensure_finite_numeric(
                frame, "open_interest", allow_null=False, context=f"open_interest:{path}"
            )
            if (frame["open_interest_num"] < 0).any():
                raise RuntimeError(f"open_interest:{path}: negative open interest encountered")

            counts = frame.groupby("second_index", sort=False).size()
            idx = counts.index.to_numpy(dtype=np.int64)
            buffers.oi_update_count[idx] += counts.to_numpy(dtype=np.int64)

            ordered = frame.sort_values(
                ["second_index", "ts_recv_ns", "source_file_id", "source_record_number"]
            )
            last_rows = ordered.drop_duplicates("second_index", keep="last")
            for row in last_rows.itertuples(index=False):
                second_index = int(row.second_index)
                ts = int(row.ts_recv_ns)
                if ts < buffers.oi_event_ts[second_index]:
                    continue
                buffers.oi_event_ts[second_index] = ts
                buffers.oi_exchange_ms[second_index] = (
                    int(row.exchange_time_ms) if pd.notna(row.exchange_time_ms) else -1
                )
                buffers.oi_event_value[second_index] = float(row.open_interest_num)

            last_row = ordered.iloc[-1]
            latest.consider(
                last_row,
                {
                    "open_interest": float(last_row["open_interest_num"]),
                    "open_interest_ts_recv_ns": int(last_row["ts_recv_ns"]),
                    "open_interest_exchange_time_ms": int(last_row["exchange_time_ms"])
                    if pd.notna(last_row["exchange_time_ms"]) else None,
                },
            )

    return CarryState(**latest.values) if latest.values else CarryState()


def aggregate_liquidations(
    paths: Sequence[Path],
    symbol: str,
    buffers: DayBuffers,
    stats: DayStats,
) -> None:
    tracker = DedupTracker()

    for path in paths:
        stats.canonical_rows["liquidations"] += read_parquet_metadata_rows(path)
        for frame in iter_row_groups(path, FAMILY_REQUIRED_COLUMNS["liquidations"]):
            frame = frame[frame["symbol"].astype(str).eq(symbol)].copy()
            stats.symbol_rows["liquidations"] += len(frame)
            if frame.empty:
                continue
            indices_all = buffers.indices_for_seconds(frame["receive_second"], f"liquidations:{path}")
            np.add.at(buffers.liq_raw_count, indices_all, 1)

            frame, duplicate_count = deduplicate_frame(frame, primary_key="event_id", tracker=tracker)
            stats.duplicates_removed["liquidations"] += duplicate_count
            stats.unique_rows["liquidations"] += len(frame)
            if frame.empty:
                continue

            frame["second_index"] = buffers.indices_for_seconds(
                frame["receive_second"], f"liquidations:{path}"
            )
            original = ensure_finite_numeric(
                frame, "original_quantity", allow_null=True, context=f"liquidations:{path}"
            )
            cumulative = ensure_finite_numeric(
                frame, "cumulative_filled_quantity", allow_null=True, context=f"liquidations:{path}"
            )
            last_filled = ensure_finite_numeric(
                frame, "last_filled_quantity", allow_null=True, context=f"liquidations:{path}"
            )
            order_price = ensure_finite_numeric(
                frame, "price", allow_null=True, context=f"liquidations:{path}"
            )
            average_price = ensure_finite_numeric(
                frame, "average_price", allow_null=True, context=f"liquidations:{path}"
            )

            quantity = cumulative.where(cumulative > 0, original)
            quantity = quantity.where(quantity > 0, last_filled)
            price = average_price.where(average_price > 0, order_price)
            invalid = (~np.isfinite(quantity)) | (~np.isfinite(price)) | (quantity <= 0) | (price <= 0)
            if invalid.any():
                raise RuntimeError(
                    f"liquidations:{path}: {int(invalid.sum()):,} rows lack usable filled quantity/price"
                )

            frame["quantity_num"] = quantity.astype("float64")
            frame["price_num"] = price.astype("float64")
            frame["notional_num"] = frame["quantity_num"] * frame["price_num"]
            frame["side_norm"] = frame["side"].astype(str).str.upper()

            grouped = frame.groupby("second_index", sort=False)
            counts = grouped.size()
            idx = counts.index.to_numpy(dtype=np.int64)
            buffers.liq_count[idx] += counts.to_numpy(dtype=np.int64)

            for row in frame.itertuples(index=False):
                second_index = int(row.second_index)
                side = str(row.side_norm)
                quantity_value = float(row.quantity_num)
                notional_value = float(row.notional_num)
                buffers.max_liq_notional[second_index] = max(
                    buffers.max_liq_notional[second_index], notional_value
                )
                if side == "SELL":
                    buffers.long_liq_count[second_index] += 1
                    buffers.long_liq_base[second_index] += quantity_value
                    buffers.long_liq_notional[second_index] += notional_value
                elif side == "BUY":
                    buffers.short_liq_count[second_index] += 1
                    buffers.short_liq_base[second_index] += quantity_value
                    buffers.short_liq_notional[second_index] += notional_value
                else:
                    buffers.unknown_liq_count[second_index] += 1
                    buffers.unknown_liq_base[second_index] += quantity_value
                    buffers.unknown_liq_notional[second_index] += notional_value
                    stats.unknown_liquidation_side_rows += 1


# ---------------------------------------------------------------------------
# Causal state expansion and output construction
# ---------------------------------------------------------------------------


def expand_state(
    event_values: np.ndarray,
    event_ts_ns: np.ndarray,
    carry_value: float | int | None,
    carry_ts_ns: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(event_values)
    update_mask = event_ts_ns >= 0
    latest_indices = np.where(update_mask, np.arange(n, dtype=np.int64), -1)
    np.maximum.accumulate(latest_indices, out=latest_indices)

    values = np.full(n, np.nan, dtype=np.float64)
    timestamps = np.full(n, -1, dtype=np.int64)
    has_event = latest_indices >= 0
    if has_event.any():
        values[has_event] = event_values[latest_indices[has_event]]
        timestamps[has_event] = event_ts_ns[latest_indices[has_event]]
    before_first = ~has_event
    if carry_value is not None:
        values[before_first] = float(carry_value)
    if carry_ts_ns is not None:
        timestamps[before_first] = int(carry_ts_ns)
    return values, timestamps


def expand_int_state(
    event_values: np.ndarray,
    event_ts_ns: np.ndarray,
    carry_value: int | None,
) -> np.ndarray:
    n = len(event_values)
    update_mask = event_ts_ns >= 0
    latest_indices = np.where(update_mask, np.arange(n, dtype=np.int64), -1)
    np.maximum.accumulate(latest_indices, out=latest_indices)
    values = np.full(n, -1, dtype=np.int64)
    has_event = latest_indices >= 0
    if has_event.any():
        values[has_event] = event_values[latest_indices[has_event]]
    if carry_value is not None:
        values[~has_event] = int(carry_value)
    return values


def state_age_seconds(receive_seconds: np.ndarray, state_ts_ns: np.ndarray) -> np.ndarray:
    ages = np.full(len(receive_seconds), np.nan)
    valid = state_ts_ns >= 0
    ages[valid] = (
        (receive_seconds[valid].astype(np.float64) * NANOSECONDS_PER_SECOND)
        - state_ts_ns[valid].astype(np.float64)
    ) / NANOSECONDS_PER_SECOND
    # A state received within the current second can be younger than the second
    # start, yielding a small negative value. At end-of-second availability, age
    # is measured against the next second boundary instead.
    ages[valid] += 1.0
    return np.maximum(ages, 0.0, where=np.isfinite(ages), out=ages)


def merge_carry(incoming: CarryState, *updates: CarryState) -> CarryState:
    result = CarryState(**asdict(incoming))
    for update in updates:
        for key, value in asdict(update).items():
            if value is not None:
                setattr(result, key, value)
    return result


def build_output_frame(
    symbol: str,
    buffers: DayBuffers,
    incoming: CarryState,
    mark_max_age_seconds: int,
    oi_max_age_seconds: int,
    trade_max_age_seconds: int,
) -> tuple[pd.DataFrame, CarryState]:
    if buffers.observed_max_index < 0:
        raise RuntimeError(f"No {symbol} perps core events were observed on {buffers.day}")

    start_index = buffers.observed_min_index
    end_index = buffers.observed_max_index
    full_indices = np.arange(SECONDS_PER_DAY, dtype=np.int64)
    receive_seconds_full = buffers.day_start + full_indices

    # Last trade price state is derived from per-second closes.
    trade_event_ts = buffers.trade_last_ts
    trade_state, trade_state_ts = expand_state(
        buffers.trade_close,
        trade_event_ts,
        incoming.trade_price,
        incoming.trade_ts_recv_ns,
    )
    trade_exchange_state = expand_int_state(
        buffers.trade_last_exchange_ms,
        trade_event_ts,
        incoming.trade_exchange_event_ms,
    )
    trade_time_state = expand_int_state(
        buffers.trade_last_trade_ms,
        trade_event_ts,
        incoming.trade_time_ms,
    )

    mark_state, mark_state_ts = expand_state(
        buffers.mark_event_value,
        buffers.mark_event_ts,
        incoming.mark_price,
        incoming.mark_ts_recv_ns,
    )
    alt_mark_state, _ = expand_state(
        buffers.alt_mark_event_value,
        buffers.mark_event_ts,
        incoming.alternate_mark_price,
        incoming.mark_ts_recv_ns,
    )
    settle_state, _ = expand_state(
        buffers.settle_event_value,
        buffers.mark_event_ts,
        incoming.estimated_settle_price,
        incoming.mark_ts_recv_ns,
    )
    index_state, _ = expand_state(
        buffers.index_event_value,
        buffers.mark_event_ts,
        incoming.index_price,
        incoming.mark_ts_recv_ns,
    )
    funding_state, _ = expand_state(
        buffers.funding_event_value,
        buffers.mark_event_ts,
        incoming.funding_rate,
        incoming.mark_ts_recv_ns,
    )
    mark_exchange_state = expand_int_state(
        buffers.mark_exchange_ms,
        buffers.mark_event_ts,
        incoming.mark_exchange_event_ms,
    )
    next_funding_state = expand_int_state(
        buffers.next_funding_event_ms,
        buffers.mark_event_ts,
        incoming.next_funding_time_ms,
    )

    oi_state, oi_state_ts = expand_state(
        buffers.oi_event_value,
        buffers.oi_event_ts,
        incoming.open_interest,
        incoming.open_interest_ts_recv_ns,
    )
    oi_exchange_state = expand_int_state(
        buffers.oi_exchange_ms,
        buffers.oi_event_ts,
        incoming.open_interest_exchange_time_ms,
    )

    trade_age = state_age_seconds(receive_seconds_full, trade_state_ts)
    mark_age = state_age_seconds(receive_seconds_full, mark_state_ts)
    oi_age = state_age_seconds(receive_seconds_full, oi_state_ts)

    oi_change = np.full(SECONDS_PER_DAY, np.nan)
    oi_change_pct = np.full(SECONDS_PER_DAY, np.nan)
    previous_oi = incoming.open_interest
    for index in np.flatnonzero(buffers.oi_event_ts >= 0):
        current_oi = buffers.oi_event_value[index]
        if previous_oi is not None and np.isfinite(previous_oi):
            oi_change[index] = current_oi - float(previous_oi)
            if float(previous_oi) != 0:
                oi_change_pct[index] = oi_change[index] / float(previous_oi)
        previous_oi = float(current_oi)

    total_liq_base = (
        buffers.long_liq_base + buffers.short_liq_base + buffers.unknown_liq_base
    )
    total_liq_notional = (
        buffers.long_liq_notional
        + buffers.short_liq_notional
        + buffers.unknown_liq_notional
    )
    net_liq_notional = buffers.short_liq_notional - buffers.long_liq_notional

    signed_base_flow = buffers.buy_base_volume - buffers.sell_base_volume
    signed_quote_flow = buffers.buy_quote_volume - buffers.sell_quote_volume
    vwap = np.divide(
        buffers.quote_volume,
        buffers.base_volume,
        out=np.full(SECONDS_PER_DAY, np.nan),
        where=buffers.base_volume > 0,
    )
    mean_trade_base = np.divide(
        buffers.base_volume,
        buffers.trade_count,
        out=np.full(SECONDS_PER_DAY, np.nan),
        where=buffers.trade_count > 0,
    )
    mean_trade_notional = np.divide(
        buffers.quote_volume,
        buffers.trade_count,
        out=np.full(SECONDS_PER_DAY, np.nan),
        where=buffers.trade_count > 0,
    )

    mark_index_basis = mark_state - index_state
    mark_index_basis_bps = np.divide(
        mark_index_basis,
        index_state,
        out=np.full(SECONDS_PER_DAY, np.nan),
        where=np.isfinite(index_state) & (index_state != 0),
    ) * 10_000.0
    oi_notional = oi_state * mark_state
    seconds_to_funding = np.where(
        next_funding_state >= 0,
        (next_funding_state / 1000.0) - (receive_seconds_full + 1.0),
        np.nan,
    )

    trade_fresh = np.isfinite(trade_age) & (trade_age <= trade_max_age_seconds)
    mark_fresh = np.isfinite(mark_age) & (mark_age <= mark_max_age_seconds)
    oi_fresh = np.isfinite(oi_age) & (oi_age <= oi_max_age_seconds)
    core_ready = trade_fresh & mark_fresh & oi_fresh

    any_event = (
        buffers.trade_raw_count
        + buffers.mark_raw_count
        + buffers.oi_raw_count
        + buffers.liq_raw_count
    ) > 0
    input_record_count = (
        buffers.trade_raw_count
        + buffers.mark_raw_count
        + buffers.oi_raw_count
        + buffers.liq_raw_count
    )

    sl = slice(start_index, end_index + 1)
    receive_seconds = receive_seconds_full[sl]
    data: dict[str, Any] = {
        "core_schema_version": np.full(len(receive_seconds), OUTPUT_SCHEMA_VERSION, dtype=np.int16),
        "symbol": np.full(len(receive_seconds), symbol, dtype=object),
        "receive_second": receive_seconds,
        "timestamp_utc": pd.to_datetime(receive_seconds, unit="s", utc=True),
        "available_after_second": receive_seconds + 1,
        "available_after_utc": pd.to_datetime(receive_seconds + 1, unit="s", utc=True),
        "seconds_from_day_start": full_indices[sl].astype(np.int32),
        "any_perps_event_observed": any_event[sl],
        "input_record_count": input_record_count[sl],

        "trade_raw_record_count": buffers.trade_raw_count[sl],
        "trade_count": buffers.trade_count[sl],
        "trade_observed": buffers.trade_count[sl] > 0,
        "trade_open": buffers.trade_open[sl],
        "trade_high": buffers.trade_high[sl],
        "trade_low": buffers.trade_low[sl],
        "trade_close": buffers.trade_close[sl],
        "trade_vwap": vwap[sl],
        "base_volume": buffers.base_volume[sl],
        "quote_volume": buffers.quote_volume[sl],
        "aggressive_buy_trade_count": buffers.buy_trade_count[sl],
        "aggressive_sell_trade_count": buffers.sell_trade_count[sl],
        "aggressor_unknown_trade_count": buffers.unknown_trade_count[sl],
        "aggressive_buy_base_volume": buffers.buy_base_volume[sl],
        "aggressive_sell_base_volume": buffers.sell_base_volume[sl],
        "aggressor_unknown_base_volume": buffers.unknown_base_volume[sl],
        "aggressive_buy_quote_volume": buffers.buy_quote_volume[sl],
        "aggressive_sell_quote_volume": buffers.sell_quote_volume[sl],
        "aggressor_unknown_quote_volume": buffers.unknown_quote_volume[sl],
        "signed_base_flow": signed_base_flow[sl],
        "signed_quote_flow": signed_quote_flow[sl],
        "mean_trade_base": mean_trade_base[sl],
        "mean_trade_notional": mean_trade_notional[sl],
        "max_trade_base": buffers.max_trade_base[sl],
        "max_trade_notional": buffers.max_trade_notional[sl],
        "first_trade_ts_recv_ns": nullable_int_series(buffers.trade_first_ts[sl]),
        "last_trade_ts_recv_ns": nullable_int_series(buffers.trade_last_ts[sl]),
        "first_trade_exchange_event_ms": nullable_int_series(buffers.trade_first_exchange_ms[sl]),
        "last_trade_exchange_event_ms": nullable_int_series(buffers.trade_last_exchange_ms[sl]),
        "first_trade_time_ms": nullable_int_series(buffers.trade_first_trade_ms[sl]),
        "last_trade_time_ms": nullable_int_series(buffers.trade_last_trade_ms[sl]),
        "last_trade_price": trade_state[sl],
        "last_trade_state_ts_recv_ns": nullable_int_series(trade_state_ts[sl]),
        "last_trade_state_exchange_event_ms": nullable_int_series(trade_exchange_state[sl]),
        "last_trade_state_trade_time_ms": nullable_int_series(trade_time_state[sl]),
        "trade_price_age_seconds": trade_age[sl],
        "trade_price_fresh": trade_fresh[sl],

        "mark_raw_record_count": buffers.mark_raw_count[sl],
        "mark_update_count": buffers.mark_update_count[sl],
        "mark_updated": buffers.mark_event_ts[sl] >= 0,
        "mark_price": mark_state[sl],
        "alternate_mark_price": alt_mark_state[sl],
        "estimated_settle_price": settle_state[sl],
        "index_price": index_state[sl],
        "funding_rate": funding_state[sl],
        "next_funding_time_ms": nullable_int_series(next_funding_state[sl]),
        "seconds_to_next_funding": seconds_to_funding[sl],
        "mark_index_basis": mark_index_basis[sl],
        "mark_index_basis_bps": mark_index_basis_bps[sl],
        "mark_state_ts_recv_ns": nullable_int_series(mark_state_ts[sl]),
        "mark_exchange_event_ms": nullable_int_series(mark_exchange_state[sl]),
        "mark_age_seconds": mark_age[sl],
        "mark_fresh": mark_fresh[sl],

        "open_interest_raw_record_count": buffers.oi_raw_count[sl],
        "open_interest_update_count": buffers.oi_update_count[sl],
        "open_interest_updated": buffers.oi_event_ts[sl] >= 0,
        "open_interest": oi_state[sl],
        "open_interest_change": oi_change[sl],
        "open_interest_change_pct": oi_change_pct[sl],
        "open_interest_notional": oi_notional[sl],
        "open_interest_state_ts_recv_ns": nullable_int_series(oi_state_ts[sl]),
        "open_interest_exchange_time_ms": nullable_int_series(oi_exchange_state[sl]),
        "open_interest_age_seconds": oi_age[sl],
        "open_interest_fresh": oi_fresh[sl],

        "liquidation_raw_record_count": buffers.liq_raw_count[sl],
        "liquidation_event_count": buffers.liq_count[sl],
        "liquidation_observed": buffers.liq_count[sl] > 0,
        "long_liquidation_count": buffers.long_liq_count[sl],
        "short_liquidation_count": buffers.short_liq_count[sl],
        "unknown_liquidation_side_count": buffers.unknown_liq_count[sl],
        "long_liquidation_base_volume": buffers.long_liq_base[sl],
        "short_liquidation_base_volume": buffers.short_liq_base[sl],
        "unknown_liquidation_base_volume": buffers.unknown_liq_base[sl],
        "liquidation_base_volume": total_liq_base[sl],
        "long_liquidation_notional": buffers.long_liq_notional[sl],
        "short_liquidation_notional": buffers.short_liq_notional[sl],
        "unknown_liquidation_notional": buffers.unknown_liq_notional[sl],
        "liquidation_notional": total_liq_notional[sl],
        "net_liquidation_notional": net_liq_notional[sl],
        "max_liquidation_notional": buffers.max_liq_notional[sl],

        "core_context_ready": core_ready[sl],
    }
    frame = pd.DataFrame(data)

    final = CarryState(**asdict(incoming))
    last_trade_indices = np.flatnonzero(buffers.trade_last_ts >= 0)
    if last_trade_indices.size:
        index = int(last_trade_indices[-1])
        final.trade_price = float(buffers.trade_close[index])
        final.trade_ts_recv_ns = int(buffers.trade_last_ts[index])
        final.trade_exchange_event_ms = (
            int(buffers.trade_last_exchange_ms[index])
            if buffers.trade_last_exchange_ms[index] >= 0 else None
        )
        final.trade_time_ms = (
            int(buffers.trade_last_trade_ms[index])
            if buffers.trade_last_trade_ms[index] >= 0 else None
        )

    last_mark_indices = np.flatnonzero(buffers.mark_event_ts >= 0)
    if last_mark_indices.size:
        index = int(last_mark_indices[-1])
        final.mark_price = float(buffers.mark_event_value[index])
        final.alternate_mark_price = (
            float(buffers.alt_mark_event_value[index])
            if np.isfinite(buffers.alt_mark_event_value[index]) else None
        )
        final.estimated_settle_price = (
            float(buffers.settle_event_value[index])
            if np.isfinite(buffers.settle_event_value[index]) else None
        )
        final.index_price = (
            float(buffers.index_event_value[index])
            if np.isfinite(buffers.index_event_value[index]) else None
        )
        final.funding_rate = (
            float(buffers.funding_event_value[index])
            if np.isfinite(buffers.funding_event_value[index]) else None
        )
        final.next_funding_time_ms = (
            int(buffers.next_funding_event_ms[index])
            if buffers.next_funding_event_ms[index] >= 0 else None
        )
        final.mark_ts_recv_ns = int(buffers.mark_event_ts[index])
        final.mark_exchange_event_ms = (
            int(buffers.mark_exchange_ms[index])
            if buffers.mark_exchange_ms[index] >= 0 else None
        )

    last_oi_indices = np.flatnonzero(buffers.oi_event_ts >= 0)
    if last_oi_indices.size:
        index = int(last_oi_indices[-1])
        final.open_interest = float(buffers.oi_event_value[index])
        final.open_interest_ts_recv_ns = int(buffers.oi_event_ts[index])
        final.open_interest_exchange_time_ms = (
            int(buffers.oi_exchange_ms[index])
            if buffers.oi_exchange_ms[index] >= 0 else None
        )

    return frame, final


def validate_output(
    frame: pd.DataFrame,
    day: date,
    stats: DayStats,
    buffers: DayBuffers,
) -> dict[str, Any]:
    failures: list[str] = []
    if frame.empty:
        failures.append("output frame is empty")
    if frame["receive_second"].duplicated().any():
        failures.append("receive_second contains duplicates")
    if not frame["receive_second"].is_monotonic_increasing:
        failures.append("receive_second is not strictly increasing")
    if not (frame["available_after_second"] == frame["receive_second"] + 1).all():
        failures.append("available_after_second is not receive_second + 1")
    if not frame["receive_second"].map(receive_day_from_second).eq(day).all():
        failures.append("output row falls outside its date partition")

    trade_rows = frame["trade_observed"]
    if (frame.loc[trade_rows, "trade_high"] < frame.loc[trade_rows, ["trade_open", "trade_close"]].max(axis=1)).any():
        failures.append("trade_high is below open or close")
    if (frame.loc[trade_rows, "trade_low"] > frame.loc[trade_rows, ["trade_open", "trade_close"]].min(axis=1)).any():
        failures.append("trade_low is above open or close")
    if (frame.loc[trade_rows, "trade_low"] > frame.loc[trade_rows, "trade_high"]).any():
        failures.append("trade_low exceeds trade_high")

    tolerance = 1e-8
    base_reconciled = np.isclose(
        frame["base_volume"],
        frame["aggressive_buy_base_volume"]
        + frame["aggressive_sell_base_volume"]
        + frame["aggressor_unknown_base_volume"],
        rtol=1e-10,
        atol=tolerance,
    ).all()
    quote_reconciled = np.isclose(
        frame["quote_volume"],
        frame["aggressive_buy_quote_volume"]
        + frame["aggressive_sell_quote_volume"]
        + frame["aggressor_unknown_quote_volume"],
        rtol=1e-10,
        atol=1e-4,
    ).all()
    count_reconciled = (
        frame["trade_count"]
        == frame["aggressive_buy_trade_count"]
        + frame["aggressive_sell_trade_count"]
        + frame["aggressor_unknown_trade_count"]
    ).all()
    if not base_reconciled:
        failures.append("trade base volume does not reconcile by aggressor side")
    if not quote_reconciled:
        failures.append("trade quote volume does not reconcile by aggressor side")
    if not count_reconciled:
        failures.append("trade counts do not reconcile by aggressor side")

    if int(frame["trade_count"].sum()) != stats.unique_rows["trades"]:
        failures.append("unique trade rows do not reconcile with output trade_count")
    if int(frame["trade_raw_record_count"].sum()) != stats.symbol_rows["trades"]:
        failures.append("raw trade rows do not reconcile")
    if int(frame["mark_raw_record_count"].sum()) != stats.symbol_rows["mark_price"]:
        failures.append("raw mark-price rows do not reconcile")
    if int(frame["open_interest_raw_record_count"].sum()) != stats.symbol_rows["open_interest"]:
        failures.append("raw open-interest rows do not reconcile")
    if int(frame["liquidation_raw_record_count"].sum()) != stats.symbol_rows["liquidations"]:
        failures.append("raw liquidation rows do not reconcile")
    if int(frame["mark_update_count"].sum()) != stats.unique_rows["mark_price"]:
        failures.append("unique mark-price rows do not reconcile")
    if int(frame["open_interest_update_count"].sum()) != stats.unique_rows["open_interest"]:
        failures.append("unique open-interest rows do not reconcile")
    if int(frame["liquidation_event_count"].sum()) != stats.unique_rows["liquidations"]:
        failures.append("unique liquidation rows do not reconcile")

    for state_name, ts_column in (
        ("trade", "last_trade_state_ts_recv_ns"),
        ("mark", "mark_state_ts_recv_ns"),
        ("open_interest", "open_interest_state_ts_recv_ns"),
    ):
        valid = frame[ts_column].notna()
        availability_ns = frame.loc[valid, "available_after_second"].astype("int64") * NANOSECONDS_PER_SECOND
        if (frame.loc[valid, ts_column].astype("int64") >= availability_ns).any():
            failures.append(f"{state_name} state uses an observation not available by second close")

    if failures:
        raise RuntimeError("Output validation failed: " + "; ".join(failures))

    return {
        "output_rows": int(len(frame)),
        "first_receive_second": int(frame["receive_second"].iloc[0]),
        "last_receive_second": int(frame["receive_second"].iloc[-1]),
        "trade_seconds": int(frame["trade_observed"].sum()),
        "mark_update_seconds": int(frame["mark_updated"].sum()),
        "open_interest_update_seconds": int(frame["open_interest_updated"].sum()),
        "liquidation_seconds": int(frame["liquidation_observed"].sum()),
        "core_context_ready_rows": int(frame["core_context_ready"].sum()),
        "input_record_count": int(frame["input_record_count"].sum()),
        "stats": asdict(stats),
    }


OUTPUT_FIELDS: list[pa.Field] = [
    pa.field("core_schema_version", pa.int16(), nullable=False),
    pa.field("symbol", pa.string(), nullable=False),
    pa.field("receive_second", pa.int64(), nullable=False),
    pa.field("timestamp_utc", pa.timestamp("s", tz="UTC"), nullable=False),
    pa.field("available_after_second", pa.int64(), nullable=False),
    pa.field("available_after_utc", pa.timestamp("s", tz="UTC"), nullable=False),
    pa.field("seconds_from_day_start", pa.int32(), nullable=False),
    pa.field("any_perps_event_observed", pa.bool_(), nullable=False),
    pa.field("input_record_count", pa.int64(), nullable=False),
]

for name in (
    "trade_raw_record_count", "trade_count",
):
    OUTPUT_FIELDS.append(pa.field(name, pa.int64(), nullable=False))
OUTPUT_FIELDS.extend([
    pa.field("trade_observed", pa.bool_(), nullable=False),
    *[pa.field(name, pa.float64()) for name in (
        "trade_open", "trade_high", "trade_low", "trade_close", "trade_vwap",
        "base_volume", "quote_volume",
    )],
])
for name in (
    "aggressive_buy_trade_count", "aggressive_sell_trade_count", "aggressor_unknown_trade_count",
):
    OUTPUT_FIELDS.append(pa.field(name, pa.int64(), nullable=False))
for name in (
    "aggressive_buy_base_volume", "aggressive_sell_base_volume", "aggressor_unknown_base_volume",
    "aggressive_buy_quote_volume", "aggressive_sell_quote_volume", "aggressor_unknown_quote_volume",
    "signed_base_flow", "signed_quote_flow", "mean_trade_base", "mean_trade_notional",
    "max_trade_base", "max_trade_notional",
):
    OUTPUT_FIELDS.append(pa.field(name, pa.float64(), nullable=False if name.startswith("max_") or "volume" in name or name.startswith("signed_") else True))
for name in (
    "first_trade_ts_recv_ns", "last_trade_ts_recv_ns",
    "first_trade_exchange_event_ms", "last_trade_exchange_event_ms",
    "first_trade_time_ms", "last_trade_time_ms",
):
    OUTPUT_FIELDS.append(pa.field(name, pa.int64()))
OUTPUT_FIELDS.extend([
    pa.field("last_trade_price", pa.float64()),
    pa.field("last_trade_state_ts_recv_ns", pa.int64()),
    pa.field("last_trade_state_exchange_event_ms", pa.int64()),
    pa.field("last_trade_state_trade_time_ms", pa.int64()),
    pa.field("trade_price_age_seconds", pa.float64()),
    pa.field("trade_price_fresh", pa.bool_(), nullable=False),
])

for name in ("mark_raw_record_count", "mark_update_count"):
    OUTPUT_FIELDS.append(pa.field(name, pa.int64(), nullable=False))
OUTPUT_FIELDS.append(pa.field("mark_updated", pa.bool_(), nullable=False))
for name in (
    "mark_price", "alternate_mark_price", "estimated_settle_price", "index_price", "funding_rate",
):
    OUTPUT_FIELDS.append(pa.field(name, pa.float64()))
OUTPUT_FIELDS.extend([
    pa.field("next_funding_time_ms", pa.int64()),
    pa.field("seconds_to_next_funding", pa.float64()),
    pa.field("mark_index_basis", pa.float64()),
    pa.field("mark_index_basis_bps", pa.float64()),
    pa.field("mark_state_ts_recv_ns", pa.int64()),
    pa.field("mark_exchange_event_ms", pa.int64()),
    pa.field("mark_age_seconds", pa.float64()),
    pa.field("mark_fresh", pa.bool_(), nullable=False),
])

for name in ("open_interest_raw_record_count", "open_interest_update_count"):
    OUTPUT_FIELDS.append(pa.field(name, pa.int64(), nullable=False))
OUTPUT_FIELDS.append(pa.field("open_interest_updated", pa.bool_(), nullable=False))
for name in (
    "open_interest", "open_interest_change", "open_interest_change_pct", "open_interest_notional",
):
    OUTPUT_FIELDS.append(pa.field(name, pa.float64()))
OUTPUT_FIELDS.extend([
    pa.field("open_interest_state_ts_recv_ns", pa.int64()),
    pa.field("open_interest_exchange_time_ms", pa.int64()),
    pa.field("open_interest_age_seconds", pa.float64()),
    pa.field("open_interest_fresh", pa.bool_(), nullable=False),
])

for name in (
    "liquidation_raw_record_count", "liquidation_event_count",
):
    OUTPUT_FIELDS.append(pa.field(name, pa.int64(), nullable=False))
OUTPUT_FIELDS.append(pa.field("liquidation_observed", pa.bool_(), nullable=False))
for name in (
    "long_liquidation_count", "short_liquidation_count", "unknown_liquidation_side_count",
):
    OUTPUT_FIELDS.append(pa.field(name, pa.int64(), nullable=False))
for name in (
    "long_liquidation_base_volume", "short_liquidation_base_volume",
    "unknown_liquidation_base_volume", "liquidation_base_volume",
    "long_liquidation_notional", "short_liquidation_notional",
    "unknown_liquidation_notional", "liquidation_notional",
    "net_liquidation_notional", "max_liquidation_notional",
):
    OUTPUT_FIELDS.append(pa.field(name, pa.float64(), nullable=False))
OUTPUT_FIELDS.append(pa.field("core_context_ready", pa.bool_(), nullable=False))

BASE_OUTPUT_SCHEMA = pa.schema(OUTPUT_FIELDS)


def frame_to_table(frame: pd.DataFrame, metadata: Mapping[str, str]) -> pa.Table:
    schema = BASE_OUTPUT_SCHEMA.with_metadata(
        {str(key).encode(): str(value).encode() for key, value in metadata.items()}
    )
    arrays: list[pa.Array] = []
    for field in schema:
        if field.name not in frame.columns:
            raise RuntimeError(f"Output frame missing schema column {field.name}")
        try:
            arrays.append(pa.array(frame[field.name], type=field.type, from_pandas=True))
        except Exception as exc:
            raise RuntimeError(
                f"Failed converting output column {field.name} to {field.type}: {exc}"
            ) from exc
    return pa.Table.from_arrays(arrays, schema=schema)


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
            canonical_root TEXT NOT NULL,
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

        CREATE INDEX IF NOT EXISTS idx_date_build_status
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
    canonical_root: Path,
    output_root: Path,
    args: argparse.Namespace,
    discovered_dates: int,
) -> None:
    connection.execute(
        """
        INSERT INTO build_runs(
            run_id, started_at_utc, status, canonical_root, output_root,
            arguments_json, discovered_dates
        ) VALUES (?, ?, 'RUNNING', ?, ?, ?, ?)
        """,
        (
            run_id,
            utc_now_iso(),
            str(canonical_root),
            str(output_root),
            compact_json(vars(args)),
            discovered_dates,
        ),
    )
    connection.commit()


def get_date_state(
    connection: sqlite3.Connection,
    symbol: str,
    day: date,
) -> sqlite3.Row | None:
    return connection.execute(
        "SELECT * FROM date_builds WHERE symbol=? AND receive_date=?",
        (symbol, day.isoformat()),
    ).fetchone()


def state_can_skip(
    state: sqlite3.Row | None,
    output_root: Path,
    input_signature: str,
    config_signature: str,
) -> bool:
    if state is None or state["status"] != "COMPLETED":
        return False
    if state["input_signature"] != input_signature:
        return False
    if state["config_signature"] != config_signature:
        return False
    relative = state["output_relative_path"]
    if not relative or not state["final_carry_json"]:
        return False
    output_path = output_root / relative
    if not output_path.is_file():
        return False
    try:
        return read_parquet_metadata_rows(output_path) == int(state["output_rows"])
    except Exception:
        return False


def mark_date_running(
    connection: sqlite3.Connection,
    symbol: str,
    day: date,
    run_id: str,
    config_signature: str,
    input_signature: str,
    carry_signature: str,
    manifest_json: str,
) -> None:
    now = utc_now_iso()
    connection.execute(
        """
        INSERT INTO date_builds(
            symbol, receive_date, status, builder_version,
            output_schema_version, config_signature, input_signature,
            carry_in_signature, input_manifest_json, started_at_utc,
            last_run_id, attempt_count, updated_at_utc
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
            manifest_json,
            now,
            run_id,
            now,
        ),
    )
    connection.commit()


def mark_date_completed(
    connection: sqlite3.Connection,
    symbol: str,
    day: date,
    output_relative_path: str,
    summary: Mapping[str, Any],
    final_carry: CarryState,
) -> None:
    now = utc_now_iso()
    connection.execute(
        """
        UPDATE date_builds SET
            status='COMPLETED',
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
            compact_json(asdict(final_carry)),
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
    *,
    status: str,
    selected_dates: int,
    skipped_dates: int,
    completed_dates: int,
    failed_dates: int,
    output_rows: int,
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
            utc_now_iso(), status, selected_dates, skipped_dates,
            completed_dates, failed_dates, output_rows, error_message, run_id,
        ),
    )
    connection.commit()


def export_state_csvs(connection: sqlite3.Connection, state_dir: Path) -> None:
    for table_name, filename in (
        ("build_runs", "build_runs.csv"),
        ("date_builds", "date_builds.csv"),
    ):
        rows = connection.execute(f"SELECT * FROM {table_name}").fetchall()
        path = state_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            path.write_text("", encoding="utf-8")
            continue
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(dict(row) for row in rows)


# ---------------------------------------------------------------------------
# One-date build
# ---------------------------------------------------------------------------


def build_date(
    *,
    canonical_root: Path,
    output_root: Path,
    staging_root: Path,
    symbol: str,
    day: date,
    paths_by_family: Mapping[str, Sequence[Path]],
    incoming: CarryState,
    input_signature: str,
    config_signature: str,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any], CarryState]:
    buffers = DayBuffers(day)
    stats = DayStats()

    trade_update = aggregate_trades(paths_by_family["trades"], symbol, buffers, stats)
    mark_update = aggregate_mark_price(paths_by_family["mark_price"], symbol, buffers, stats)
    oi_update = aggregate_open_interest(paths_by_family["open_interest"], symbol, buffers, stats)
    aggregate_liquidations(paths_by_family["liquidations"], symbol, buffers, stats)

    frame, final_carry = build_output_frame(
        symbol,
        buffers,
        incoming,
        args.mark_max_age_seconds,
        args.oi_max_age_seconds,
        args.trade_max_age_seconds,
    )
    final_carry = merge_carry(incoming, trade_update, mark_update, oi_update, final_carry)
    summary = validate_output(frame, day, stats, buffers)

    symbol_partition = safe_symbol_partition(symbol)
    relative_path = Path(f"symbol={symbol_partition}") / f"date={day.isoformat()}" / f"part-{day.isoformat()}.parquet"
    final_path = output_root / relative_path
    stage_path = staging_root / relative_path
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "perps_core_schema_version": str(OUTPUT_SCHEMA_VERSION),
        "builder_version": BUILDER_VERSION,
        "symbol": symbol,
        "receive_date": day.isoformat(),
        "input_signature": input_signature,
        "config_signature": config_signature,
        "carry_in_signature": incoming.signature(),
        "row_availability": "end_of_receive_second",
        "aggressor_semantics": "buyer_is_maker_false=aggressive_buy;true=aggressive_sell",
        "liquidation_semantics": "force_order_sell=long_liquidation;buy=short_liquidation",
    }
    table = frame_to_table(frame, metadata)
    try:
        pq.write_table(
            table,
            stage_path,
            compression=args.compression,
            compression_level=args.compression_level,
            row_group_size=args.row_group_size,
            use_dictionary=True,
            write_statistics=True,
        )
        staged_rows = read_parquet_metadata_rows(stage_path)
        if staged_rows != len(frame):
            raise RuntimeError(
                f"Staged Parquet row count {staged_rows:,} != expected {len(frame):,}"
            )
        os.replace(stage_path, final_path)
    finally:
        if stage_path.exists():
            stage_path.unlink()

    # Remove obsolete extra Parquet files from the same output partition only
    # after the replacement file is committed.
    for path in final_path.parent.glob("*.parquet"):
        if path != final_path:
            path.unlink()

    summary = dict(summary)
    summary.update(
        {
            "symbol": symbol,
            "receive_date": day.isoformat(),
            "output_relative_path": relative_path.as_posix(),
            "output_bytes": int(final_path.stat().st_size),
            "input_signature": input_signature,
            "config_signature": config_signature,
            "carry_in_signature": incoming.signature(),
            "final_carry_signature": final_carry.signature(),
        }
    )
    return relative_path.as_posix(), summary, final_carry


# ---------------------------------------------------------------------------
# CLI and orchestration
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the resumable perps one-second core measurement layer."
    )
    parser.add_argument(
        "--canonical-root",
        default="./data/canonical/market_events_v1",
        help="Canonical market_events_v1 root.",
    )
    parser.add_argument(
        "--output-root",
        default="./data/derived/perps/core_1s_v1",
        help="Derived output root.",
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start-date", help="Inclusive receive date YYYY-MM-DD.")
    parser.add_argument("--end-date", help="Inclusive receive date YYYY-MM-DD.")
    parser.add_argument(
        "--families",
        help="Optional comma-separated family subset for diagnostics. Normal builds use all four.",
    )
    parser.add_argument("--trade-max-age-seconds", type=int, default=30)
    parser.add_argument("--mark-max-age-seconds", type=int, default=5)
    parser.add_argument("--oi-max-age-seconds", type=int, default=120)
    parser.add_argument("--state-lookback-days", type=int, default=7)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--compression-level", type=int, default=3)
    parser.add_argument("--row-group-size", type=int, default=86_400)
    parser.add_argument("--min-free-space-gb", type=float, default=10.0)
    parser.add_argument("--force", action="store_true", help="Rebuild selected dates even if unchanged.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to later dates after a failed date. Default stops to preserve carry dependency.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "trade_max_age_seconds", "mark_max_age_seconds", "oi_max_age_seconds",
        "state_lookback_days", "compression_level", "row_group_size",
    ):
        if getattr(args, name) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative")
    if args.row_group_size == 0:
        raise SystemExit("--row-group-size must be greater than zero")
    if args.min_free_space_gb < 0:
        raise SystemExit("--min-free-space-gb must be non-negative")
    requested = parse_csv_values(args.families)
    if requested is not None and requested != set(FAMILIES):
        raise SystemExit(
            "This production builder requires all families. --families is reserved for future diagnostics."
        )


def config_payload(args: argparse.Namespace, symbol: str) -> dict[str, Any]:
    return {
        "builder_version": BUILDER_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "symbol": symbol,
        "trade_max_age_seconds": args.trade_max_age_seconds,
        "mark_max_age_seconds": args.mark_max_age_seconds,
        "oi_max_age_seconds": args.oi_max_age_seconds,
        "compression": args.compression,
        "compression_level": args.compression_level,
        "row_group_size": args.row_group_size,
        "families": list(FAMILIES),
        "availability_contract": "end_of_receive_second",
        "deduplication": {
            "trades": "aggregate_trade_id_then_event_id_then_source",
            "mark_price": "event_id_then_source",
            "open_interest": "event_id_then_source",
            "liquidations": "event_id_then_source",
        },
    }


def write_latest_summary(state_dir: Path, summary: Mapping[str, Any]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "latest_run_summary.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args)

    canonical_root = resolve_path(Path(args.canonical_root))
    output_root = resolve_path(Path(args.output_root))
    symbol = args.symbol.strip().upper()
    safe_symbol_partition(symbol)

    if not canonical_root.is_dir():
        raise SystemExit(f"Canonical root does not exist: {canonical_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    state_dir = output_root / "_build_state"
    state_db_path = state_dir / "build_state.sqlite"
    staging_root = state_dir / "_staging"
    staging_root.mkdir(parents=True, exist_ok=True)

    all_dates = discover_partition_dates(canonical_root)
    if not all_dates:
        raise SystemExit(f"No perps canonical date partitions found under {canonical_root}")

    start_date = parse_iso_date(args.start_date, "--start-date") or all_dates[0]
    end_date = parse_iso_date(args.end_date, "--end-date") or all_dates[-1]
    if start_date > end_date:
        raise SystemExit("--start-date must be on or before --end-date")
    dates = [day for day in all_dates if start_date <= day <= end_date]
    if not dates:
        raise SystemExit("No canonical perps dates fall inside the requested range")

    config = config_payload(args, symbol)
    config_signature = sha256_text(compact_json(config))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]

    connection = open_state_database(state_db_path)
    create_run(connection, run_id, canonical_root, output_root, args, len(dates))

    initial_free = available_bytes(output_root)
    required_free = int(args.min_free_space_gb * (1024 ** 3))
    print("Planning perps core build...")
    print(f"Run ID          : {run_id}")
    print(f"Canonical root  : {canonical_root}")
    print(f"Output root     : {output_root}")
    print(f"State database  : {state_db_path}")
    print(f"Symbol          : {symbol}")
    print(f"Date range      : {dates[0]} to {dates[-1]} ({len(dates):,} observed dates)")
    print(f"Output free     : {format_gib(initial_free)}")
    print(f"Free-space guard: {args.min_free_space_gb:.2f} GiB")

    if not args.dry_run and initial_free < required_free:
        finish_run(
            connection,
            run_id,
            status="FAILED",
            selected_dates=0,
            skipped_dates=0,
            completed_dates=0,
            failed_dates=0,
            output_rows=0,
            error_message="Insufficient free space before build",
        )
        raise SystemExit(
            f"Output drive has {format_gib(initial_free)} free, below the "
            f"{args.min_free_space_gb:.2f} GiB guard"
        )

    carry = find_initial_carry(
        canonical_root, symbol, dates[0], args.state_lookback_days
    )
    print(f"Initial carry   : {carry.signature()[:16]} (lookback={args.state_lookback_days} days)")

    selected_dates = 0
    skipped_dates = 0
    completed_dates = 0
    failed_dates = 0
    total_output_rows = 0
    date_summaries: list[dict[str, Any]] = []

    try:
        for position, day in enumerate(dates, start=1):
            manifest, paths_by_family = input_manifest(canonical_root, day)
            manifest_json = compact_json(manifest)
            signature_payload = {
                "config_signature": config_signature,
                "carry_in": asdict(carry),
                "input_manifest": manifest,
            }
            input_signature = sha256_text(compact_json(signature_payload))
            state = get_date_state(connection, symbol, day)
            can_skip = (
                not args.force
                and state_can_skip(state, output_root, input_signature, config_signature)
            )

            if can_skip:
                skipped_dates += 1
                carry = CarryState.from_json(state["final_carry_json"])
                print(
                    f"SKIPPED   [{position:>3}/{len(dates)}] {day} "
                    f"rows={int(state['output_rows']):,}"
                )
                continue

            selected_dates += 1
            if args.dry_run:
                reason = "forced" if args.force else (
                    "new" if state is None else f"state={state['status']} or signature changed"
                )
                print(f"SELECTED  [{position:>3}/{len(dates)}] {day} ({reason})")
                carry = latest_state_from_date(canonical_root, symbol, day, carry)
                continue

            free_now = available_bytes(output_root)
            if free_now < required_free:
                raise RuntimeError(
                    f"Free-space guard reached before {day}: {format_gib(free_now)} free"
                )

            mark_date_running(
                connection,
                symbol,
                day,
                run_id,
                config_signature,
                input_signature,
                carry.signature(),
                manifest_json,
            )
            try:
                relative_path, summary, final_carry = build_date(
                    canonical_root=canonical_root,
                    output_root=output_root,
                    staging_root=staging_root / run_id,
                    symbol=symbol,
                    day=day,
                    paths_by_family=paths_by_family,
                    incoming=carry,
                    input_signature=input_signature,
                    config_signature=config_signature,
                    args=args,
                )
                mark_date_completed(
                    connection,
                    symbol,
                    day,
                    relative_path,
                    summary,
                    final_carry,
                )
                carry = final_carry
                completed_dates += 1
                total_output_rows += int(summary["output_rows"])
                date_summaries.append(summary)
                stats = summary["stats"]
                print(
                    f"COMPLETED [{position:>3}/{len(dates)}] {day}: "
                    f"rows={summary['output_rows']:,}, "
                    f"trades={stats['unique_rows']['trades']:,}, "
                    f"mark={stats['unique_rows']['mark_price']:,}, "
                    f"oi={stats['unique_rows']['open_interest']:,}, "
                    f"liq={stats['unique_rows']['liquidations']:,}, "
                    f"duplicates_removed={sum(stats['duplicates_removed'].values()):,}"
                )
            except Exception as exc:
                failed_dates += 1
                message = f"{type(exc).__name__}: {exc}"
                mark_date_failed(connection, symbol, day, message)
                print(f"FAILED    [{position:>3}/{len(dates)}] {day}: {message}", file=sys.stderr)
                if not args.continue_on_error:
                    raise

        status = "DRY_RUN" if args.dry_run else ("COMPLETED_WITH_FAILURES" if failed_dates else "COMPLETED")
        finish_run(
            connection,
            run_id,
            status=status,
            selected_dates=selected_dates,
            skipped_dates=skipped_dates,
            completed_dates=completed_dates,
            failed_dates=failed_dates,
            output_rows=total_output_rows,
        )
        summary = {
            "run_id": run_id,
            "builder_version": BUILDER_VERSION,
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
            "status": status,
            "canonical_root": str(canonical_root),
            "output_root": str(output_root),
            "symbol": symbol,
            "start_date": dates[0].isoformat(),
            "end_date": dates[-1].isoformat(),
            "discovered_dates": len(dates),
            "selected_dates": selected_dates,
            "skipped_dates": skipped_dates,
            "completed_dates": completed_dates,
            "failed_dates": failed_dates,
            "output_rows_written_this_run": total_output_rows,
            "config": config,
            "date_summaries": date_summaries,
            "completed_at_utc": utc_now_iso(),
        }
        write_latest_summary(state_dir, summary)
        export_state_csvs(connection, state_dir)

        print("\nPerps core build finished.")
        print(f"Status           : {status}")
        print(f"Selected dates   : {selected_dates:,}")
        print(f"Skipped dates    : {skipped_dates:,}")
        print(f"Completed dates  : {completed_dates:,}")
        print(f"Failed dates     : {failed_dates:,}")
        print(f"Rows written     : {total_output_rows:,}")
        print(f"Summary          : {state_dir / 'latest_run_summary.json'}")
        return 1 if failed_dates else 0

    except Exception as exc:
        finish_run(
            connection,
            run_id,
            status="FAILED",
            selected_dates=selected_dates,
            skipped_dates=skipped_dates,
            completed_dates=completed_dates,
            failed_dates=max(failed_dates, 1),
            output_rows=total_output_rows,
            error_message=f"{type(exc).__name__}: {exc}",
        )
        summary = {
            "run_id": run_id,
            "builder_version": BUILDER_VERSION,
            "status": "FAILED",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "selected_dates": selected_dates,
            "skipped_dates": skipped_dates,
            "completed_dates": completed_dates,
            "failed_dates": max(failed_dates, 1),
            "output_rows_written_this_run": total_output_rows,
            "completed_at_utc": utc_now_iso(),
        }
        write_latest_summary(state_dir, summary)
        export_state_csvs(connection, state_dir)
        print(f"\nBuild failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        connection.close()
        run_staging = staging_root / run_id
        if run_staging.exists():
            shutil.rmtree(run_staging, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
