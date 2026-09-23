#!/usr/bin/env python
"""
Unified, resumable extraction of all Stream bronze WAL market events.

This script is the canonical first stage of the stream-research pipeline.
It scans compressed NDJSON WAL files exactly once per completed source file,
routes every record by its embedded ``market`` and ``etype``, and writes
loss-minimising canonical Parquet shards.

Key guarantees
--------------
* File-level resumability backed by SQLite.
* Atomic per-source output shards: a source is marked COMPLETED only after all
  of its temporary Parquet files have been validated and moved into place.
* Receive-time date partitioning, so records crossing UTC midnight are written
  to the correct date regardless of the raw directory date.
* Full perps depth-delta arrays and full stored book snapshots are preserved.
* Unknown event types are retained in a generic canonical dataset instead of
  being silently discarded.
* Raw records are not deduplicated or aggregated. One canonical row is written
  for every valid source record, enabling exact raw-to-canonical reconciliation.

Expected dependencies
---------------------
    pip install pyarrow zstandard

Typical Git Bash usage
----------------------
    python scripts/01_extract_all_market_events.py \
      --input-root "/c/Users/USER/stream-bronze-download/data/bronze" \
      --output-root "./data/canonical/market_events_v1" \
      --workers 2
"""

from __future__ import annotations

import argparse
import errno
import csv
import hashlib
import io
import json
import os
import re
import shutil
import sqlite3
import sys
import traceback
import uuid
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - user environment check
    raise SystemExit(
        "Missing dependency 'pyarrow'. Activate the project virtual environment "
        "and run: pip install pyarrow zstandard"
    ) from exc

try:
    import zstandard as zstd
except ImportError as exc:  # pragma: no cover - user environment check
    raise SystemExit(
        "Missing dependency 'zstandard'. Activate the project virtual environment "
        "and run: pip install pyarrow zstandard"
    ) from exc


EXTRACTOR_VERSION = "1.2.0"
CANONICAL_SCHEMA_VERSION = 1
DEFAULT_GLOB = "**/*.ndjson*.zst"
KNOWN_TOP_LEVEL_KEYS = {
    "schema",
    "market",
    "ts_recv_ns",
    "mono_ns",
    "etype",
    "symbol",
    "payload",
    "event_id",
    "seq",
}
DATE_COMPONENT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SAFE_PATH_RE = re.compile(r"[^A-Za-z0-9_.-]+")
CAMEL_CASE_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
NON_IDENTIFIER_RE = re.compile(r"[^a-z0-9]+")

# Collector-control records sometimes arrive with etype="unknown" and a
# synthetic symbol beginning with "!". These are operational lifecycle
# events, not unknown market data. Classification remains intentionally narrow:
# both the control-symbol marker and a lifecycle keyword must be present.
COLLECTOR_LIFECYCLE_KEYWORDS = (
    "startup",
    "start",
    "started",
    "shutdown",
    "stop",
    "stopped",
    "connect",
    "connected",
    "disconnect",
    "disconnected",
    "reconnect",
    "reconnected",
    "resync",
    "resynchronization",
    "restart",
    "restarted",
)

LEVEL_STRUCT = pa.struct(
    [
        pa.field("price", pa.string()),
        pa.field("quantity", pa.string()),
        pa.field("extra_json", pa.string()),
    ]
)
LEVEL_LIST = pa.list_(LEVEL_STRUCT)


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compact_json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def safe_component(value: str | None, fallback: str = "unknown") -> str:
    cleaned = SAFE_PATH_RE.sub("_", str(value or fallback)).strip("._")
    return cleaned or fallback


def is_no_space_error(exc: BaseException | None) -> bool:
    """Return True when an exception chain represents ENOSPC/disk-full."""
    seen: set[int] = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, OSError) and getattr(current, "errno", None) == errno.ENOSPC:
            return True
        text = str(current).lower()
        if "no space left on device" in text or "errno 28" in text:
            return True
        current = current.__cause__ or current.__context__
    return False


def free_space_bytes(path: Path) -> int:
    """Return free bytes for the filesystem containing path."""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return int(shutil.disk_usage(probe).free)


def format_gib(value: int) -> str:
    return f"{value / (1024 ** 3):.2f} GiB"


def safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def safe_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def snake_case_identifier(value: str | None) -> str | None:
    """Convert collector labels such as ``serverShutdown`` to snake case."""
    if not value:
        return None
    separated = CAMEL_CASE_BOUNDARY_RE.sub("_", str(value).strip())
    cleaned = NON_IDENTIFIER_RE.sub("_", separated.lower()).strip("_")
    return cleaned or None


def collector_lifecycle_event_type(record: Mapping[str, Any]) -> str | None:
    """Classify synthetic collector lifecycle/control records.

    The collector has emitted records such as::

        etype=unknown, symbol=!SERVERSHUTDOWN, payload.e=serverShutdown

    Unknown records are promoted to system events only when the symbol begins
    with ``!`` and a lifecycle keyword is present. This prevents unfamiliar
    market-data events from being silently reclassified as operational events.
    """
    event_type = (safe_str(record.get("etype")) or "").strip()
    if event_type not in {"unknown", "control", "lifecycle"}:
        return None

    symbol = (safe_str(record.get("symbol")) or "").strip()
    if not symbol.startswith("!"):
        return None

    payload = record.get("payload")
    payload_map = dict(payload) if isinstance(payload, Mapping) else {}
    candidates = (
        safe_str(payload_map.get("e")),
        safe_str(payload_map.get("event")),
        safe_str(payload_map.get("type")),
        symbol.lstrip("!"),
    )
    for candidate in candidates:
        normalized = snake_case_identifier(candidate)
        if not normalized:
            continue
        compact = normalized.replace("_", "")
        if any(keyword in compact for keyword in COLLECTOR_LIFECYCLE_KEYWORDS):
            return normalized
    return None


def safe_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    return None


def receive_date_from_ns(ts_recv_ns: int) -> str:
    if ts_recv_ns <= 0:
        raise ValueError(f"Invalid ts_recv_ns: {ts_recv_ns!r}")
    return datetime.fromtimestamp(ts_recv_ns / 1_000_000_000, tz=timezone.utc).date().isoformat()


def path_date_component(path: Path) -> str | None:
    for part in reversed(path.parts):
        if DATE_COMPONENT_RE.match(part):
            return part
    return None


def parse_csv_set(value: str | None) -> set[str] | None:
    if not value:
        return None
    result = {item.strip() for item in value.split(",") if item.strip()}
    return result or None


def prune_empty_parents(path: Path, stop_at: Path) -> None:
    current = path.parent
    stop = stop_at.resolve()
    while current.exists() and current.resolve() != stop:
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


# ---------------------------------------------------------------------------
# Canonical schemas
# ---------------------------------------------------------------------------


def common_fields() -> list[pa.Field]:
    return [
        pa.field("canonical_schema_version", pa.int16(), nullable=False),
        pa.field("source_schema", pa.string()),
        pa.field("market", pa.string(), nullable=False),
        pa.field("event_type", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("ts_recv_ns", pa.int64(), nullable=False),
        pa.field("receive_second", pa.int64(), nullable=False),
        pa.field("mono_ns", pa.int64()),
        pa.field("event_id", pa.string()),
        pa.field("sequence", pa.int64()),
        pa.field("source_file_id", pa.string(), nullable=False),
        pa.field("source_relative_path", pa.string(), nullable=False),
        pa.field("source_record_number", pa.int64(), nullable=False),
        pa.field("record_extra_json", pa.string()),
        pa.field("payload_extra_json", pa.string()),
    ]


def make_schema(extra_fields: Sequence[pa.Field], schema_name: str) -> pa.Schema:
    schema = pa.schema(common_fields() + list(extra_fields))
    return schema.with_metadata(
        {
            b"canonical_schema_version": str(CANONICAL_SCHEMA_VERSION).encode(),
            b"extractor_version": EXTRACTOR_VERSION.encode(),
            b"canonical_schema_name": schema_name.encode(),
        }
    )


PERPS_TRADE_SCHEMA = make_schema(
    [
        pa.field("event_name", pa.string()),
        pa.field("exchange_event_ms", pa.int64()),
        pa.field("trade_time_ms", pa.int64()),
        pa.field("aggregate_trade_id", pa.int64()),
        pa.field("first_trade_id", pa.int64()),
        pa.field("last_trade_id", pa.int64()),
        pa.field("price", pa.string()),
        pa.field("quantity", pa.string()),
        pa.field("normalized_quantity", pa.string()),
        pa.field("buyer_is_maker", pa.bool_()),
    ],
    "perps.trades.v1",
)

SPOT_TRADE_SCHEMA = make_schema(
    [
        pa.field("event_name", pa.string()),
        pa.field("exchange_event_ms", pa.int64()),
        pa.field("trade_time_ms", pa.int64()),
        pa.field("aggregate_trade_id", pa.int64()),
        pa.field("first_trade_id", pa.int64()),
        pa.field("last_trade_id", pa.int64()),
        pa.field("price", pa.string()),
        pa.field("quantity", pa.string()),
        pa.field("buyer_is_maker", pa.bool_()),
        pa.field("best_price_match", pa.bool_()),
    ],
    "spot.trades.v1",
)

PERPS_DEPTH_SCHEMA = make_schema(
    [
        pa.field("event_name", pa.string()),
        pa.field("exchange_event_ms", pa.int64()),
        pa.field("transaction_time_ms", pa.int64()),
        pa.field("first_update_id", pa.int64()),
        pa.field("final_update_id", pa.int64()),
        pa.field("previous_final_update_id", pa.int64()),
        pa.field("bid_updates", LEVEL_LIST),
        pa.field("ask_updates", LEVEL_LIST),
        pa.field("bid_update_count", pa.int32()),
        pa.field("ask_update_count", pa.int32()),
    ],
    "perps.depth_updates.v1",
)

BOOK_SNAPSHOT_SCHEMA = make_schema(
    [
        pa.field("snapshot_symbol", pa.string()),
        pa.field("last_update_id", pa.int64()),
        pa.field("bids", LEVEL_LIST),
        pa.field("asks", LEVEL_LIST),
        pa.field("bid_level_count", pa.int32()),
        pa.field("ask_level_count", pa.int32()),
    ],
    "book_snapshots.v1",
)

PERPS_MARK_PRICE_SCHEMA = make_schema(
    [
        pa.field("event_name", pa.string()),
        pa.field("exchange_event_ms", pa.int64()),
        pa.field("mark_price", pa.string()),
        pa.field("alternate_mark_price", pa.string()),
        pa.field("estimated_settle_price", pa.string()),
        pa.field("index_price", pa.string()),
        pa.field("funding_rate", pa.string()),
        pa.field("next_funding_time_ms", pa.int64()),
    ],
    "perps.mark_price.v1",
)

PERPS_OPEN_INTEREST_SCHEMA = make_schema(
    [
        pa.field("exchange_time_ms", pa.int64()),
        pa.field("open_interest", pa.string()),
    ],
    "perps.open_interest.v1",
)

PERPS_LIQUIDATION_SCHEMA = make_schema(
    [
        pa.field("event_name", pa.string()),
        pa.field("exchange_event_ms", pa.int64()),
        pa.field("order_symbol", pa.string()),
        pa.field("side", pa.string()),
        pa.field("order_type", pa.string()),
        pa.field("time_in_force", pa.string()),
        pa.field("original_quantity", pa.string()),
        pa.field("price", pa.string()),
        pa.field("average_price", pa.string()),
        pa.field("order_status", pa.string()),
        pa.field("last_filled_quantity", pa.string()),
        pa.field("cumulative_filled_quantity", pa.string()),
        pa.field("trade_time_ms", pa.int64()),
        pa.field("order_extra_json", pa.string()),
    ],
    "perps.liquidations.v1",
)

OKX_BBO_SCHEMA = make_schema(
    [
        pa.field("instrument_id", pa.string()),
        pa.field("channel", pa.string()),
        pa.field("exchange_ts_ms", pa.int64()),
        pa.field("bid_price", pa.string()),
        pa.field("bid_size", pa.string()),
        pa.field("ask_price", pa.string()),
        pa.field("ask_size", pa.string()),
        pa.field("bids0_price", pa.string()),
        pa.field("bids0_size", pa.string()),
        pa.field("asks0_price", pa.string()),
        pa.field("asks0_size", pa.string()),
    ],
    "okx_spot.bbo.v1",
)

OPTIONS_INDEX_PRICE_SCHEMA = make_schema(
    [
        pa.field("event_name", pa.string()),
        pa.field("exchange_event_ms", pa.int64()),
        pa.field("index_price", pa.string()),
    ],
    "options.index_price.v1",
)

OPTIONS_MARK_SCHEMA = make_schema(
    [
        pa.field("event_name", pa.string()),
        pa.field("exchange_event_ms", pa.int64()),
        pa.field("mark_price", pa.string()),
        pa.field("index_price", pa.string()),
        pa.field("estimated_settle_price", pa.string()),
        pa.field("best_bid_price", pa.string()),
        pa.field("best_ask_price", pa.string()),
        pa.field("best_bid_quantity", pa.string()),
        pa.field("best_ask_quantity", pa.string()),
        pa.field("bid_iv", pa.string()),
        pa.field("ask_iv", pa.string()),
        pa.field("high_price_limit", pa.string()),
        pa.field("low_price_limit", pa.string()),
        pa.field("mark_iv", pa.string()),
        pa.field("risk_free_rate", pa.string()),
        pa.field("delta", pa.string()),
        pa.field("theta", pa.string()),
        pa.field("gamma", pa.string()),
        pa.field("vega", pa.string()),
    ],
    "options.mark_price_greeks.v1",
)

OPTIONS_OPEN_INTEREST_SCHEMA = make_schema(
    [
        pa.field("event_name", pa.string()),
        pa.field("exchange_event_ms", pa.int64()),
        pa.field("open_interest", pa.string()),
        pa.field("open_interest_value", pa.string()),
    ],
    "options.open_interest.v1",
)

OPTIONS_TRADE_SCHEMA = make_schema(
    [
        pa.field("event_name", pa.string()),
        pa.field("exchange_event_ms", pa.int64()),
        pa.field("trade_time_ms", pa.int64()),
        pa.field("trade_id", pa.int64()),
        pa.field("price", pa.string()),
        pa.field("quantity", pa.string()),
        pa.field("order_type", pa.string()),
        pa.field("side", pa.string()),
        pa.field("maker", pa.bool_()),
    ],
    "options.trades.v1",
)

SYSTEM_EVENT_SCHEMA = make_schema(
    [pa.field("payload_json", pa.string())],
    "system.events.v1",
)

UNKNOWN_EVENT_SCHEMA = make_schema(
    [
        pa.field("unknown_reason", pa.string()),
        pa.field("payload_json", pa.string()),
        pa.field("raw_record_json", pa.string()),
    ],
    "unknown.events.v1",
)


# ---------------------------------------------------------------------------
# Normalisation and routing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceContext:
    source_file_id: str
    source_relative_path: str
    source_record_number: int


@dataclass(frozen=True)
class Route:
    output_market: str
    family: str
    schema: pa.Schema
    normalizer_name: str
    buffer_rows: int


ROUTES: dict[tuple[str, str], Route] = {
    ("perps", "aggTrade"): Route("perps", "trades", PERPS_TRADE_SCHEMA, "perps_trade", 10_000),
    ("perps", "depth"): Route("perps", "depth_updates", PERPS_DEPTH_SCHEMA, "perps_depth", 100),
    ("perps", "snapshot"): Route("perps", "book_snapshots", BOOK_SNAPSHOT_SCHEMA, "book_snapshot", 250),
    ("perps", "markPrice"): Route("perps", "mark_price", PERPS_MARK_PRICE_SCHEMA, "perps_mark_price", 5_000),
    ("perps", "openInterest"): Route("perps", "open_interest", PERPS_OPEN_INTEREST_SCHEMA, "perps_open_interest", 5_000),
    ("perps", "forceOrder"): Route("perps", "liquidations", PERPS_LIQUIDATION_SCHEMA, "perps_liquidation", 2_000),
    ("spot", "aggTrade"): Route("spot", "trades", SPOT_TRADE_SCHEMA, "spot_trade", 10_000),
    ("spot", "snapshot"): Route("spot", "book_snapshots", BOOK_SNAPSHOT_SCHEMA, "book_snapshot", 250),
    ("okx_spot", "bbo"): Route("okx_spot", "bbo", OKX_BBO_SCHEMA, "okx_bbo", 10_000),
    ("options", "indexPrice"): Route("options", "index_price", OPTIONS_INDEX_PRICE_SCHEMA, "options_index_price", 10_000),
    ("options", "optionMarkPrice"): Route("options", "mark_price_greeks", OPTIONS_MARK_SCHEMA, "options_mark", 5_000),
    ("options", "optionOpenInterest"): Route("options", "open_interest", OPTIONS_OPEN_INTEREST_SCHEMA, "options_open_interest", 10_000),
    ("options", "optionTrade"): Route("options", "trades", OPTIONS_TRADE_SCHEMA, "options_trade", 10_000),
}

NORMALIZER_PAYLOAD_KEYS: dict[str, set[str]] = {
    "perps_trade": {"e", "E", "T", "a", "f", "l", "p", "q", "nq", "m", "s"},
    "spot_trade": {"e", "E", "T", "a", "f", "l", "p", "q", "m", "M", "s"},
    "perps_depth": {"e", "E", "T", "U", "u", "pu", "b", "a", "s"},
    "book_snapshot": {"snapshot"},
    "perps_mark_price": {"e", "E", "p", "ap", "P", "i", "r", "T", "s"},
    "perps_open_interest": {"symbol", "openInterest", "time"},
    "perps_liquidation": {"e", "E", "o"},
    "okx_bbo": {"instId", "channel", "bidPx", "bidSz", "askPx", "askSz", "ts", "bids0", "asks0"},
    "options_index_price": {"e", "E", "p", "s"},
    "options_mark": {"e", "E", "s", "mp", "i", "P", "bo", "ao", "bq", "aq", "b", "a", "hl", "ll", "vo", "rf", "d", "t", "g", "v"},
    "options_open_interest": {"e", "E", "s", "o", "h"},
    "options_trade": {"e", "E", "T", "s", "t", "p", "q", "X", "S", "m"},
}


def parse_levels(value: Any) -> list[dict[str, str | None]]:
    if not isinstance(value, list):
        return []
    levels: list[dict[str, str | None]] = []
    for item in value:
        if isinstance(item, (list, tuple)):
            price = safe_str(item[0]) if len(item) >= 1 else None
            quantity = safe_str(item[1]) if len(item) >= 2 else None
            extra = list(item[2:]) if len(item) > 2 else None
        elif isinstance(item, Mapping):
            price = safe_str(item.get("price", item.get("px")))
            quantity = safe_str(item.get("quantity", item.get("qty", item.get("size"))))
            extra_keys = {
                key: val
                for key, val in item.items()
                if key not in {"price", "px", "quantity", "qty", "size"}
            }
            extra = extra_keys or None
        else:
            price = safe_str(item)
            quantity = None
            extra = None
        levels.append(
            {
                "price": price,
                "quantity": quantity,
                "extra_json": compact_json(extra),
            }
        )
    return levels


def pair_values(value: Any) -> tuple[str | None, str | None]:
    if isinstance(value, (list, tuple)):
        return (
            safe_str(value[0]) if len(value) >= 1 else None,
            safe_str(value[1]) if len(value) >= 2 else None,
        )
    return None, None


def base_row(record: Mapping[str, Any], context: SourceContext) -> dict[str, Any]:
    ts_recv_ns = safe_int(record.get("ts_recv_ns"))
    if ts_recv_ns is None or ts_recv_ns <= 0:
        raise ValueError("Record has no valid positive ts_recv_ns")
    market = safe_str(record.get("market"))
    event_type = safe_str(record.get("etype"))
    symbol = safe_str(record.get("symbol"))
    if not market or not event_type or not symbol:
        raise ValueError("Record is missing market, etype, or symbol")
    record_extras = {key: value for key, value in record.items() if key not in KNOWN_TOP_LEVEL_KEYS}
    return {
        "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
        "source_schema": safe_str(record.get("schema")),
        "market": market,
        "event_type": event_type,
        "symbol": symbol,
        "ts_recv_ns": ts_recv_ns,
        "receive_second": ts_recv_ns // 1_000_000_000,
        "mono_ns": safe_int(record.get("mono_ns")),
        "event_id": safe_str(record.get("event_id")),
        "sequence": safe_int(record.get("seq")),
        "source_file_id": context.source_file_id,
        "source_relative_path": context.source_relative_path,
        "source_record_number": context.source_record_number,
        "record_extra_json": compact_json(record_extras) if record_extras else None,
        "payload_extra_json": None,
    }


def payload_dict(record: Mapping[str, Any]) -> dict[str, Any]:
    value = record.get("payload")
    return dict(value) if isinstance(value, Mapping) else {}


def add_payload_extras(row: dict[str, Any], payload: Mapping[str, Any], normalizer_name: str) -> None:
    known = NORMALIZER_PAYLOAD_KEYS.get(normalizer_name, set())
    extras = {key: value for key, value in payload.items() if key not in known}
    row["payload_extra_json"] = compact_json(extras) if extras else None


def normalize_perps_trade(record: Mapping[str, Any], context: SourceContext) -> dict[str, Any]:
    payload = payload_dict(record)
    row = base_row(record, context)
    row.update(
        {
            "event_name": safe_str(payload.get("e")),
            "exchange_event_ms": safe_int(payload.get("E")),
            "trade_time_ms": safe_int(payload.get("T")),
            "aggregate_trade_id": safe_int(payload.get("a")),
            "first_trade_id": safe_int(payload.get("f")),
            "last_trade_id": safe_int(payload.get("l")),
            "price": safe_str(payload.get("p")),
            "quantity": safe_str(payload.get("q")),
            "normalized_quantity": safe_str(payload.get("nq")),
            "buyer_is_maker": safe_bool(payload.get("m")),
        }
    )
    add_payload_extras(row, payload, "perps_trade")
    return row


def normalize_spot_trade(record: Mapping[str, Any], context: SourceContext) -> dict[str, Any]:
    payload = payload_dict(record)
    row = base_row(record, context)
    row.update(
        {
            "event_name": safe_str(payload.get("e")),
            "exchange_event_ms": safe_int(payload.get("E")),
            "trade_time_ms": safe_int(payload.get("T")),
            "aggregate_trade_id": safe_int(payload.get("a")),
            "first_trade_id": safe_int(payload.get("f")),
            "last_trade_id": safe_int(payload.get("l")),
            "price": safe_str(payload.get("p")),
            "quantity": safe_str(payload.get("q")),
            "buyer_is_maker": safe_bool(payload.get("m")),
            "best_price_match": safe_bool(payload.get("M")),
        }
    )
    add_payload_extras(row, payload, "spot_trade")
    return row


def normalize_perps_depth(record: Mapping[str, Any], context: SourceContext) -> dict[str, Any]:
    payload = payload_dict(record)
    bids = parse_levels(payload.get("b"))
    asks = parse_levels(payload.get("a"))
    row = base_row(record, context)
    row.update(
        {
            "event_name": safe_str(payload.get("e")),
            "exchange_event_ms": safe_int(payload.get("E")),
            "transaction_time_ms": safe_int(payload.get("T")),
            "first_update_id": safe_int(payload.get("U")),
            "final_update_id": safe_int(payload.get("u")),
            "previous_final_update_id": safe_int(payload.get("pu")),
            "bid_updates": bids,
            "ask_updates": asks,
            "bid_update_count": len(bids),
            "ask_update_count": len(asks),
        }
    )
    add_payload_extras(row, payload, "perps_depth")
    return row


def normalize_book_snapshot(record: Mapping[str, Any], context: SourceContext) -> dict[str, Any]:
    payload = payload_dict(record)
    snapshot = payload.get("snapshot")
    snapshot_map = dict(snapshot) if isinstance(snapshot, Mapping) else {}
    bids = parse_levels(snapshot_map.get("bids"))
    asks = parse_levels(snapshot_map.get("asks"))
    row = base_row(record, context)
    row.update(
        {
            "snapshot_symbol": safe_str(snapshot_map.get("symbol")),
            "last_update_id": safe_int(snapshot_map.get("last_u", snapshot_map.get("lastUpdateId"))),
            "bids": bids,
            "asks": asks,
            "bid_level_count": len(bids),
            "ask_level_count": len(asks),
        }
    )
    snapshot_extras = {
        key: value
        for key, value in snapshot_map.items()
        if key not in {"symbol", "last_u", "lastUpdateId", "bids", "asks"}
    }
    payload_extras = {key: value for key, value in payload.items() if key != "snapshot"}
    combined = {}
    if snapshot_extras:
        combined["snapshot_extra"] = snapshot_extras
    if payload_extras:
        combined["payload_extra"] = payload_extras
    row["payload_extra_json"] = compact_json(combined) if combined else None
    return row


def normalize_perps_mark_price(record: Mapping[str, Any], context: SourceContext) -> dict[str, Any]:
    payload = payload_dict(record)
    row = base_row(record, context)
    row.update(
        {
            "event_name": safe_str(payload.get("e")),
            "exchange_event_ms": safe_int(payload.get("E")),
            "mark_price": safe_str(payload.get("p")),
            "alternate_mark_price": safe_str(payload.get("ap")),
            "estimated_settle_price": safe_str(payload.get("P")),
            "index_price": safe_str(payload.get("i")),
            "funding_rate": safe_str(payload.get("r")),
            "next_funding_time_ms": safe_int(payload.get("T")),
        }
    )
    add_payload_extras(row, payload, "perps_mark_price")
    return row


def normalize_perps_open_interest(record: Mapping[str, Any], context: SourceContext) -> dict[str, Any]:
    payload = payload_dict(record)
    row = base_row(record, context)
    row.update(
        {
            "exchange_time_ms": safe_int(payload.get("time")),
            "open_interest": safe_str(payload.get("openInterest")),
        }
    )
    add_payload_extras(row, payload, "perps_open_interest")
    return row


def normalize_perps_liquidation(record: Mapping[str, Any], context: SourceContext) -> dict[str, Any]:
    payload = payload_dict(record)
    order = payload.get("o")
    order_map = dict(order) if isinstance(order, Mapping) else {}
    known_order = {"s", "S", "o", "f", "q", "p", "ap", "X", "l", "z", "T"}
    order_extras = {key: value for key, value in order_map.items() if key not in known_order}
    row = base_row(record, context)
    row.update(
        {
            "event_name": safe_str(payload.get("e")),
            "exchange_event_ms": safe_int(payload.get("E")),
            "order_symbol": safe_str(order_map.get("s")),
            "side": safe_str(order_map.get("S")),
            "order_type": safe_str(order_map.get("o")),
            "time_in_force": safe_str(order_map.get("f")),
            "original_quantity": safe_str(order_map.get("q")),
            "price": safe_str(order_map.get("p")),
            "average_price": safe_str(order_map.get("ap")),
            "order_status": safe_str(order_map.get("X")),
            "last_filled_quantity": safe_str(order_map.get("l")),
            "cumulative_filled_quantity": safe_str(order_map.get("z")),
            "trade_time_ms": safe_int(order_map.get("T")),
            "order_extra_json": compact_json(order_extras) if order_extras else None,
        }
    )
    payload_extras = {key: value for key, value in payload.items() if key not in {"e", "E", "o"}}
    row["payload_extra_json"] = compact_json(payload_extras) if payload_extras else None
    return row


def normalize_okx_bbo(record: Mapping[str, Any], context: SourceContext) -> dict[str, Any]:
    payload = payload_dict(record)
    bids0_price, bids0_size = pair_values(payload.get("bids0"))
    asks0_price, asks0_size = pair_values(payload.get("asks0"))
    row = base_row(record, context)
    row.update(
        {
            "instrument_id": safe_str(payload.get("instId")),
            "channel": safe_str(payload.get("channel")),
            "exchange_ts_ms": safe_int(payload.get("ts")),
            "bid_price": safe_str(payload.get("bidPx")),
            "bid_size": safe_str(payload.get("bidSz")),
            "ask_price": safe_str(payload.get("askPx")),
            "ask_size": safe_str(payload.get("askSz")),
            "bids0_price": bids0_price,
            "bids0_size": bids0_size,
            "asks0_price": asks0_price,
            "asks0_size": asks0_size,
        }
    )
    add_payload_extras(row, payload, "okx_bbo")
    return row


def normalize_options_index_price(record: Mapping[str, Any], context: SourceContext) -> dict[str, Any]:
    payload = payload_dict(record)
    row = base_row(record, context)
    row.update(
        {
            "event_name": safe_str(payload.get("e")),
            "exchange_event_ms": safe_int(payload.get("E")),
            "index_price": safe_str(payload.get("p")),
        }
    )
    add_payload_extras(row, payload, "options_index_price")
    return row


def normalize_options_mark(record: Mapping[str, Any], context: SourceContext) -> dict[str, Any]:
    payload = payload_dict(record)
    row = base_row(record, context)
    row.update(
        {
            "event_name": safe_str(payload.get("e")),
            "exchange_event_ms": safe_int(payload.get("E")),
            "mark_price": safe_str(payload.get("mp")),
            "index_price": safe_str(payload.get("i")),
            "estimated_settle_price": safe_str(payload.get("P")),
            "best_bid_price": safe_str(payload.get("bo")),
            "best_ask_price": safe_str(payload.get("ao")),
            "best_bid_quantity": safe_str(payload.get("bq")),
            "best_ask_quantity": safe_str(payload.get("aq")),
            "bid_iv": safe_str(payload.get("b")),
            "ask_iv": safe_str(payload.get("a")),
            "high_price_limit": safe_str(payload.get("hl")),
            "low_price_limit": safe_str(payload.get("ll")),
            "mark_iv": safe_str(payload.get("vo")),
            "risk_free_rate": safe_str(payload.get("rf")),
            "delta": safe_str(payload.get("d")),
            "theta": safe_str(payload.get("t")),
            "gamma": safe_str(payload.get("g")),
            "vega": safe_str(payload.get("v")),
        }
    )
    add_payload_extras(row, payload, "options_mark")
    return row


def normalize_options_open_interest(record: Mapping[str, Any], context: SourceContext) -> dict[str, Any]:
    payload = payload_dict(record)
    row = base_row(record, context)
    row.update(
        {
            "event_name": safe_str(payload.get("e")),
            "exchange_event_ms": safe_int(payload.get("E")),
            "open_interest": safe_str(payload.get("o")),
            "open_interest_value": safe_str(payload.get("h")),
        }
    )
    add_payload_extras(row, payload, "options_open_interest")
    return row


def normalize_options_trade(record: Mapping[str, Any], context: SourceContext) -> dict[str, Any]:
    payload = payload_dict(record)
    row = base_row(record, context)
    row.update(
        {
            "event_name": safe_str(payload.get("e")),
            "exchange_event_ms": safe_int(payload.get("E")),
            "trade_time_ms": safe_int(payload.get("T")),
            "trade_id": safe_int(payload.get("t")),
            "price": safe_str(payload.get("p")),
            "quantity": safe_str(payload.get("q")),
            "order_type": safe_str(payload.get("X")),
            "side": safe_str(payload.get("S")),
            "maker": safe_bool(payload.get("m")),
        }
    )
    add_payload_extras(row, payload, "options_trade")
    return row


def normalize_system(record: Mapping[str, Any], context: SourceContext) -> dict[str, Any]:
    row = base_row(record, context)
    payload = record.get("payload")
    lifecycle_event_type = collector_lifecycle_event_type(record)
    if lifecycle_event_type is not None:
        source_event_type = row["event_type"]
        row["event_type"] = lifecycle_event_type
        row["payload_extra_json"] = compact_json(
            {
                "classification": "collector_lifecycle",
                "source_event_type": source_event_type,
            }
        )
    else:
        row["payload_extra_json"] = None
    row["payload_json"] = compact_json(payload)
    return row


def normalize_unknown(
    record: Mapping[str, Any], context: SourceContext, reason: str
) -> dict[str, Any]:
    row = base_row(record, context)
    row.update(
        {
            "unknown_reason": reason,
            "payload_json": compact_json(record.get("payload")),
            "raw_record_json": compact_json(record),
        }
    )
    row["payload_extra_json"] = None
    return row


NORMALIZERS: dict[str, Callable[[Mapping[str, Any], SourceContext], dict[str, Any]]] = {
    "perps_trade": normalize_perps_trade,
    "spot_trade": normalize_spot_trade,
    "perps_depth": normalize_perps_depth,
    "book_snapshot": normalize_book_snapshot,
    "perps_mark_price": normalize_perps_mark_price,
    "perps_open_interest": normalize_perps_open_interest,
    "perps_liquidation": normalize_perps_liquidation,
    "okx_bbo": normalize_okx_bbo,
    "options_index_price": normalize_options_index_price,
    "options_mark": normalize_options_mark,
    "options_open_interest": normalize_options_open_interest,
    "options_trade": normalize_options_trade,
}


def resolve_route(record: Mapping[str, Any], include_system_events: bool) -> tuple[Route | None, str | None]:
    market = safe_str(record.get("market")) or "unknown"
    event_type = safe_str(record.get("etype")) or "unknown"
    route = ROUTES.get((market, event_type))
    if route is not None:
        return route, None
    lifecycle_event_type = collector_lifecycle_event_type(record)
    if event_type.startswith("sys.") or lifecycle_event_type is not None:
        if not include_system_events:
            return None, "EXCLUDED_SYSTEM_EVENT"
        return (
            Route("system", safe_component(market), SYSTEM_EVENT_SCHEMA, "system", 5_000),
            None,
        )
    return (
        Route("unknown", safe_component(market), UNKNOWN_EVENT_SCHEMA, "unknown", 2_000),
        f"UNRECOGNISED_ROUTE:{market}:{event_type}",
    )


# ---------------------------------------------------------------------------
# Parquet writing
# ---------------------------------------------------------------------------


class BufferedParquetShard:
    def __init__(
        self,
        temp_path: Path,
        final_path: Path,
        schema: pa.Schema,
        buffer_rows: int,
        compression: str,
        compression_level: int,
    ) -> None:
        self.temp_path = temp_path
        self.final_path = final_path
        self.schema = schema
        self.buffer_rows = max(1, buffer_rows)
        self.compression = compression
        self.compression_level = compression_level
        self.rows: list[dict[str, Any]] = []
        self.total_rows = 0
        self.writer: pq.ParquetWriter | None = None

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.buffer_rows:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        self.temp_path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(self.rows, schema=self.schema)
        if self.writer is None:
            kwargs: dict[str, Any] = {
                "where": str(self.temp_path),
                "schema": self.schema,
                "compression": self.compression,
                "use_dictionary": True,
                "write_statistics": True,
                "version": "2.6",
            }
            if self.compression == "zstd":
                kwargs["compression_level"] = self.compression_level
            self.writer = pq.ParquetWriter(**kwargs)
        self.writer.write_table(table, row_group_size=len(self.rows))
        self.total_rows += len(self.rows)
        self.rows.clear()

    def close_and_validate(self) -> None:
        self.flush()
        if self.writer is None:
            return
        self.writer.close()
        self.writer = None
        metadata_rows = pq.ParquetFile(self.temp_path).metadata.num_rows
        if metadata_rows != self.total_rows:
            raise RuntimeError(
                f"Parquet row-count validation failed for {self.temp_path}: "
                f"expected {self.total_rows}, found {metadata_rows}"
            )

    def abort(self) -> None:
        try:
            if self.writer is not None:
                self.writer.close()
        except Exception:
            pass
        self.writer = None
        self.rows.clear()


# ---------------------------------------------------------------------------
# Source fingerprints and discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceFile:
    absolute_path: str
    relative_path: str
    size_bytes: int
    mtime_ns: int
    fingerprint: str
    fingerprint_mode: str
    source_file_id: str
    source_date_hint: str | None


def quick_fingerprint(path: Path, chunk_bytes: int = 1_048_576) -> str:
    stat = path.stat()
    digest = hashlib.sha256()
    digest.update(str(stat.st_size).encode())
    with path.open("rb") as handle:
        first = handle.read(chunk_bytes)
        digest.update(first)
        if stat.st_size > chunk_bytes:
            offset = max(0, stat.st_size - chunk_bytes)
            handle.seek(offset)
            digest.update(handle.read(chunk_bytes))
    return digest.hexdigest()


def full_fingerprint(path: Path, chunk_bytes: int = 8 * 1_048_576) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_source_file(path: Path, input_root: Path, fingerprint_mode: str) -> SourceFile:
    resolved = path.resolve()
    relative = resolved.relative_to(input_root.resolve()).as_posix()
    stat = resolved.stat()
    if fingerprint_mode == "full":
        fingerprint = full_fingerprint(resolved)
    else:
        fingerprint = quick_fingerprint(resolved)
    source_file_id = hashlib.sha256(
        f"{relative}\0{fingerprint}\0{CANONICAL_SCHEMA_VERSION}".encode()
    ).hexdigest()[:24]
    return SourceFile(
        absolute_path=str(resolved),
        relative_path=relative,
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        fingerprint=fingerprint,
        fingerprint_mode=fingerprint_mode,
        source_file_id=source_file_id,
        source_date_hint=path_date_component(resolved),
    )


def discover_sources(
    input_root: Path,
    source_glob: str,
    fingerprint_mode: str,
    source_date_from: str | None,
    source_date_to: str | None,
    max_files: int | None,
) -> list[SourceFile]:
    paths = sorted(path for path in input_root.glob(source_glob) if path.is_file())
    selected: list[Path] = []
    for path in paths:
        hint = path_date_component(path)
        if source_date_from and hint and hint < source_date_from:
            continue
        if source_date_to and hint and hint > source_date_to:
            continue
        selected.append(path)
    if max_files is not None:
        selected = selected[: max(0, max_files)]
    return [build_source_file(path, input_root, fingerprint_mode) for path in selected]


# ---------------------------------------------------------------------------
# SQLite extraction state
# ---------------------------------------------------------------------------


def connect_state_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def initialise_state_db(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS extraction_runs (
            run_id TEXT PRIMARY KEY,
            started_at_utc TEXT NOT NULL,
            completed_at_utc TEXT,
            status TEXT NOT NULL,
            input_root TEXT NOT NULL,
            output_root TEXT NOT NULL,
            arguments_json TEXT NOT NULL,
            discovered_files INTEGER NOT NULL DEFAULT 0,
            selected_files INTEGER NOT NULL DEFAULT 0,
            completed_files INTEGER NOT NULL DEFAULT 0,
            skipped_files INTEGER NOT NULL DEFAULT 0,
            failed_files INTEGER NOT NULL DEFAULT 0,
            records_read INTEGER NOT NULL DEFAULT 0,
            records_written INTEGER NOT NULL DEFAULT 0,
            unknown_records INTEGER NOT NULL DEFAULT 0,
            malformed_records INTEGER NOT NULL DEFAULT 0,
            error_message TEXT
        );

        CREATE TABLE IF NOT EXISTS source_files (
            source_relative_path TEXT PRIMARY KEY,
            source_file_id TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            fingerprint TEXT NOT NULL,
            fingerprint_mode TEXT NOT NULL,
            canonical_schema_version INTEGER NOT NULL,
            status TEXT NOT NULL,
            embedded_markets_json TEXT,
            event_type_counts_json TEXT,
            family_counts_json TEXT,
            output_paths_json TEXT,
            first_ts_recv_ns INTEGER,
            last_ts_recv_ns INTEGER,
            records_read INTEGER NOT NULL DEFAULT 0,
            records_written INTEGER NOT NULL DEFAULT 0,
            unknown_records INTEGER NOT NULL DEFAULT 0,
            malformed_records INTEGER NOT NULL DEFAULT 0,
            skipped_system_records INTEGER NOT NULL DEFAULT 0,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            started_at_utc TEXT,
            completed_at_utc TEXT,
            last_run_id TEXT,
            error_message TEXT,
            updated_at_utc TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS extraction_attempts (
            attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            source_relative_path TEXT NOT NULL,
            source_file_id TEXT NOT NULL,
            started_at_utc TEXT NOT NULL,
            completed_at_utc TEXT,
            status TEXT NOT NULL,
            records_read INTEGER NOT NULL DEFAULT 0,
            records_written INTEGER NOT NULL DEFAULT 0,
            error_message TEXT,
            FOREIGN KEY(run_id) REFERENCES extraction_runs(run_id)
        );

        CREATE INDEX IF NOT EXISTS idx_source_files_status
            ON source_files(status);
        CREATE INDEX IF NOT EXISTS idx_attempts_run
            ON extraction_attempts(run_id);
        """
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        ("canonical_schema_version", str(CANONICAL_SCHEMA_VERSION)),
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        ("extractor_version", EXTRACTOR_VERSION),
    )
    connection.commit()


def create_run(
    connection: sqlite3.Connection,
    run_id: str,
    input_root: Path,
    output_root: Path,
    args: argparse.Namespace,
    discovered_files: int,
) -> None:
    connection.execute(
        """
        INSERT INTO extraction_runs(
            run_id, started_at_utc, status, input_root, output_root,
            arguments_json, discovered_files
        ) VALUES (?, ?, 'RUNNING', ?, ?, ?, ?)
        """,
        (
            run_id,
            utc_now_iso(),
            str(input_root.resolve()),
            str(output_root.resolve()),
            compact_json(vars(args)) or "{}",
            discovered_files,
        ),
    )
    connection.commit()


def source_state(connection: sqlite3.Connection, relative_path: str) -> sqlite3.Row | None:
    return connection.execute(
        "SELECT * FROM source_files WHERE source_relative_path = ?",
        (relative_path,),
    ).fetchone()


def paths_exist(output_root: Path, paths_json: str | None) -> bool:
    if not paths_json:
        return False
    try:
        paths = json.loads(paths_json)
    except json.JSONDecodeError:
        return False
    return bool(paths) and all((output_root / path).exists() for path in paths)


def should_process_source(
    state: sqlite3.Row | None,
    source: SourceFile,
    output_root: Path,
    force: bool,
    retry_failed: bool,
) -> tuple[bool, str]:
    if force:
        return True, "FORCED"
    if state is None:
        return True, "NEW"
    same_version = state["canonical_schema_version"] == CANONICAL_SCHEMA_VERSION
    same_fingerprint = state["fingerprint"] == source.fingerprint
    if (
        state["status"] == "COMPLETED"
        and same_version
        and same_fingerprint
        and paths_exist(output_root, state["output_paths_json"])
    ):
        return False, "UNCHANGED_COMPLETED"
    if state["status"] == "FAILED" and not retry_failed and same_fingerprint:
        return False, "FAILED_RETRY_DISABLED"
    if state["status"] == "RUNNING":
        return True, "INTERRUPTED_RUNNING"
    if not same_version:
        return True, "SCHEMA_VERSION_CHANGED"
    if not same_fingerprint:
        return True, "SOURCE_CHANGED"
    if state["status"] != "COMPLETED":
        return True, f"STATUS_{state['status']}"
    return True, "OUTPUT_MISSING"


def mark_source_running(
    connection: sqlite3.Connection,
    run_id: str,
    source: SourceFile,
) -> tuple[int, list[str]]:
    old = source_state(connection, source.relative_path)
    old_paths: list[str] = []
    if old and old["output_paths_json"]:
        try:
            old_paths = list(json.loads(old["output_paths_json"]))
        except json.JSONDecodeError:
            old_paths = []
    started = utc_now_iso()
    connection.execute(
        """
        INSERT INTO source_files(
            source_relative_path, source_file_id, size_bytes, mtime_ns,
            fingerprint, fingerprint_mode, canonical_schema_version, status,
            attempt_count, started_at_utc, last_run_id, updated_at_utc,
            output_paths_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'RUNNING', 1, ?, ?, ?, ?)
        ON CONFLICT(source_relative_path) DO UPDATE SET
            source_file_id = excluded.source_file_id,
            size_bytes = excluded.size_bytes,
            mtime_ns = excluded.mtime_ns,
            fingerprint = excluded.fingerprint,
            fingerprint_mode = excluded.fingerprint_mode,
            canonical_schema_version = excluded.canonical_schema_version,
            status = 'RUNNING',
            attempt_count = source_files.attempt_count + 1,
            started_at_utc = excluded.started_at_utc,
            completed_at_utc = NULL,
            last_run_id = excluded.last_run_id,
            error_message = NULL,
            updated_at_utc = excluded.updated_at_utc
        """,
        (
            source.relative_path,
            source.source_file_id,
            source.size_bytes,
            source.mtime_ns,
            source.fingerprint,
            source.fingerprint_mode,
            CANONICAL_SCHEMA_VERSION,
            started,
            run_id,
            started,
            compact_json(old_paths),
        ),
    )
    cursor = connection.execute(
        """
        INSERT INTO extraction_attempts(
            run_id, source_relative_path, source_file_id,
            started_at_utc, status
        ) VALUES (?, ?, ?, ?, 'RUNNING')
        """,
        (run_id, source.relative_path, source.source_file_id, started),
    )
    connection.commit()
    return int(cursor.lastrowid), old_paths


def mark_source_success(
    connection: sqlite3.Connection,
    run_id: str,
    attempt_id: int,
    source: SourceFile,
    result: Mapping[str, Any],
) -> None:
    completed = utc_now_iso()
    connection.execute(
        """
        UPDATE source_files SET
            status = 'COMPLETED',
            embedded_markets_json = ?,
            event_type_counts_json = ?,
            family_counts_json = ?,
            output_paths_json = ?,
            first_ts_recv_ns = ?,
            last_ts_recv_ns = ?,
            records_read = ?,
            records_written = ?,
            unknown_records = ?,
            malformed_records = ?,
            skipped_system_records = ?,
            completed_at_utc = ?,
            last_run_id = ?,
            error_message = NULL,
            updated_at_utc = ?
        WHERE source_relative_path = ?
        """,
        (
            compact_json(result["embedded_markets"]),
            compact_json(result["event_type_counts"]),
            compact_json(result["family_counts"]),
            compact_json(result["output_paths"]),
            result.get("first_ts_recv_ns"),
            result.get("last_ts_recv_ns"),
            result["records_read"],
            result["records_written"],
            result["unknown_records"],
            result["malformed_records"],
            result["skipped_system_records"],
            completed,
            run_id,
            completed,
            source.relative_path,
        ),
    )
    connection.execute(
        """
        UPDATE extraction_attempts SET
            status = 'COMPLETED', completed_at_utc = ?,
            records_read = ?, records_written = ?, error_message = NULL
        WHERE attempt_id = ?
        """,
        (completed, result["records_read"], result["records_written"], attempt_id),
    )
    connection.commit()


def mark_source_failed(
    connection: sqlite3.Connection,
    run_id: str,
    attempt_id: int,
    source: SourceFile,
    result: Mapping[str, Any],
) -> None:
    completed = utc_now_iso()
    error = str(result.get("error") or "Unknown extraction failure")
    connection.execute(
        """
        UPDATE source_files SET
            status = 'FAILED',
            records_read = ?,
            records_written = ?,
            malformed_records = ?,
            completed_at_utc = ?,
            last_run_id = ?,
            error_message = ?,
            updated_at_utc = ?
        WHERE source_relative_path = ?
        """,
        (
            result.get("records_read", 0),
            result.get("records_written", 0),
            result.get("malformed_records", 0),
            completed,
            run_id,
            error,
            completed,
            source.relative_path,
        ),
    )
    connection.execute(
        """
        UPDATE extraction_attempts SET
            status = 'FAILED', completed_at_utc = ?,
            records_read = ?, records_written = ?, error_message = ?
        WHERE attempt_id = ?
        """,
        (
            completed,
            result.get("records_read", 0),
            result.get("records_written", 0),
            error,
            attempt_id,
        ),
    )
    connection.commit()


# ---------------------------------------------------------------------------
# Worker processing
# ---------------------------------------------------------------------------


def output_relative_path(
    output_market: str,
    family: str,
    receive_date: str,
    source_file_id: str,
) -> Path:
    return (
        Path(safe_component(output_market))
        / safe_component(family)
        / f"date={receive_date}"
        / f"part-{source_file_id}.parquet"
    )


def process_source_job(job: Mapping[str, Any]) -> dict[str, Any]:
    source = SourceFile(**job["source"])
    output_root = Path(job["output_root"])
    run_id = str(job["run_id"])
    include_system_events = bool(job["include_system_events"])
    fail_on_unknown = bool(job["fail_on_unknown"])
    skip_malformed = bool(job["skip_malformed"])
    compression = str(job["compression"])
    compression_level = int(job["compression_level"])
    progress_every = int(job["progress_every"])

    stage_root = output_root / "_staging" / run_id / source.source_file_id
    if stage_root.exists():
        shutil.rmtree(stage_root, ignore_errors=True)
    stage_root.mkdir(parents=True, exist_ok=True)

    writers: dict[tuple[str, str, str], BufferedParquetShard] = {}
    event_type_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    embedded_markets: Counter[str] = Counter()
    records_read = 0
    records_written = 0
    unknown_records = 0
    malformed_records = 0
    skipped_system_records = 0
    first_ts_recv_ns: int | None = None
    last_ts_recv_ns: int | None = None
    malformed_handle: io.TextIOWrapper | None = None
    malformed_temp_path = stage_root / "malformed.ndjson"

    try:
        dctx = zstd.ZstdDecompressor()
        with Path(source.absolute_path).open("rb") as compressed:
            with dctx.stream_reader(compressed, read_across_frames=True) as reader:
                text_reader = io.TextIOWrapper(reader, encoding="utf-8", errors="strict")
                for record_number, line in enumerate(text_reader, start=1):
                    records_read += 1
                    if progress_every > 0 and records_read % progress_every == 0:
                        print(
                            f"[{source.relative_path}] read {records_read:,} records",
                            flush=True,
                        )
                    try:
                        record = json.loads(line)
                        if not isinstance(record, Mapping):
                            raise ValueError("JSON record is not an object")
                        ts_recv_ns = safe_int(record.get("ts_recv_ns"))
                        if ts_recv_ns is None or ts_recv_ns <= 0:
                            raise ValueError("Record has invalid ts_recv_ns")
                        market = safe_str(record.get("market")) or "unknown"
                        event_type = safe_str(record.get("etype")) or "unknown"
                        embedded_markets[market] += 1
                        event_type_counts[f"{market}/{event_type}"] += 1
                        first_ts_recv_ns = (
                            ts_recv_ns
                            if first_ts_recv_ns is None
                            else min(first_ts_recv_ns, ts_recv_ns)
                        )
                        last_ts_recv_ns = (
                            ts_recv_ns
                            if last_ts_recv_ns is None
                            else max(last_ts_recv_ns, ts_recv_ns)
                        )
                        receive_date = receive_date_from_ns(ts_recv_ns)
                        route, route_reason = resolve_route(record, include_system_events)
                        if route is None:
                            skipped_system_records += 1
                            continue
                        context = SourceContext(
                            source_file_id=source.source_file_id,
                            source_relative_path=source.relative_path,
                            source_record_number=record_number,
                        )
                        if route.normalizer_name == "system":
                            row = normalize_system(record, context)
                        elif route.normalizer_name == "unknown":
                            unknown_records += 1
                            if fail_on_unknown:
                                raise ValueError(route_reason or "Unknown event route")
                            row = normalize_unknown(
                                record,
                                context,
                                route_reason or "UNRECOGNISED_ROUTE",
                            )
                        else:
                            normalizer = NORMALIZERS[route.normalizer_name]
                            row = normalizer(record, context)

                        key = (route.output_market, route.family, receive_date)
                        shard = writers.get(key)
                        if shard is None:
                            relative_output = output_relative_path(
                                route.output_market,
                                route.family,
                                receive_date,
                                source.source_file_id,
                            )
                            shard = BufferedParquetShard(
                                temp_path=stage_root / relative_output,
                                final_path=output_root / relative_output,
                                schema=route.schema,
                                buffer_rows=route.buffer_rows,
                                compression=compression,
                                compression_level=compression_level,
                            )
                            writers[key] = shard
                        shard.append(row)
                        records_written += 1
                        family_counts[f"{route.output_market}/{route.family}"] += 1
                    except Exception as record_exc:
                        # Storage and other infrastructure failures are file/run failures,
                        # not malformed source records. In particular, ENOSPC must escape
                        # immediately so the parent process can stop dispatching more WALs.
                        if is_no_space_error(record_exc):
                            raise OSError(
                                errno.ENOSPC,
                                "No space left on device while writing canonical output",
                            ) from record_exc
                        malformed_records += 1
                        if not skip_malformed:
                            raise RuntimeError(
                                f"Record {record_number} failed: {record_exc}"
                            ) from record_exc
                        if malformed_handle is None:
                            malformed_temp_path.parent.mkdir(parents=True, exist_ok=True)
                            malformed_handle = malformed_temp_path.open("w", encoding="utf-8")
                        malformed_handle.write(
                            compact_json(
                                {
                                    "source_relative_path": source.relative_path,
                                    "source_file_id": source.source_file_id,
                                    "source_record_number": record_number,
                                    "error": str(record_exc),
                                    "raw_line": line.rstrip("\r\n"),
                                }
                            )
                            + "\n"
                        )

        if malformed_handle is not None:
            malformed_handle.close()
            malformed_handle = None

        for shard in writers.values():
            shard.close_and_validate()

        output_paths: list[str] = []
        for shard in writers.values():
            if shard.total_rows <= 0:
                continue
            shard.final_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(shard.temp_path, shard.final_path)
            output_paths.append(shard.final_path.relative_to(output_root).as_posix())

        if malformed_temp_path.exists():
            malformed_final = (
                output_root
                / "_extraction_state"
                / "malformed"
                / f"{source.source_file_id}.ndjson"
            )
            malformed_final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(malformed_temp_path, malformed_final)
            output_paths.append(malformed_final.relative_to(output_root).as_posix())

        shutil.rmtree(stage_root, ignore_errors=True)
        return {
            "ok": True,
            "source_relative_path": source.relative_path,
            "source_file_id": source.source_file_id,
            "records_read": records_read,
            "records_written": records_written,
            "records_committed": records_written,
            "unknown_records": unknown_records,
            "malformed_records": malformed_records,
            "skipped_system_records": skipped_system_records,
            "first_ts_recv_ns": first_ts_recv_ns,
            "last_ts_recv_ns": last_ts_recv_ns,
            "embedded_markets": dict(sorted(embedded_markets.items())),
            "event_type_counts": dict(sorted(event_type_counts.items())),
            "family_counts": dict(sorted(family_counts.items())),
            "output_paths": sorted(output_paths),
        }
    except Exception as exc:
        if malformed_handle is not None:
            try:
                malformed_handle.close()
            except Exception:
                pass
        for shard in writers.values():
            shard.abort()
        shutil.rmtree(stage_root, ignore_errors=True)
        no_space = is_no_space_error(exc)
        return {
            "ok": False,
            "source_relative_path": source.relative_path,
            "source_file_id": source.source_file_id,
            "records_read": records_read,
            # These rows were buffered only in temporary shards and were aborted.
            # They are not committed canonical rows.
            "records_written": records_written,
            "records_committed": 0,
            "unknown_records": unknown_records,
            "malformed_records": malformed_records,
            "skipped_system_records": skipped_system_records,
            "fatal_error_code": "NO_SPACE_LEFT" if no_space else None,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# Manifests and run summaries
# ---------------------------------------------------------------------------


def schema_description(schema: pa.Schema) -> list[dict[str, Any]]:
    return [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": field.nullable,
        }
        for field in schema
    ]


def write_schema_manifest(output_root: Path) -> None:
    route_rows = []
    for (market, event_type), route in sorted(ROUTES.items()):
        route_rows.append(
            {
                "market": market,
                "event_type": event_type,
                "output_market": route.output_market,
                "family": route.family,
                "normalizer": route.normalizer_name,
                "schema": schema_description(route.schema),
            }
        )
    manifest = {
        "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "created_at_utc": utc_now_iso(),
        "partitioning": "receive-time UTC date",
        "raw_record_policy": "one canonical row per valid raw record; no aggregation or deduplication",
        "known_routes": route_rows,
        "collector_lifecycle_routing": {
            "source_event_types": ["unknown", "control", "lifecycle"],
            "required_symbol_prefix": "!",
            "destination": "system/<market>",
            "event_type_normalization": "snake_case payload.e/event/type, then symbol",
            "lifecycle_keywords": list(COLLECTOR_LIFECYCLE_KEYWORDS),
        },
        "system_schema": schema_description(SYSTEM_EVENT_SCHEMA),
        "unknown_schema": schema_description(UNKNOWN_EVENT_SCHEMA),
    }
    atomic_write_text(
        output_root / "_extraction_state" / "schema_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True),
    )


def append_run_csv(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "started_at_utc",
        "completed_at_utc",
        "status",
        "discovered_files",
        "selected_files",
        "completed_files",
        "skipped_files",
        "failed_files",
        "records_read",
        "records_written",
        "unknown_records",
        "malformed_records",
    ]
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({field: summary.get(field) for field in fieldnames})


def finish_run(
    connection: sqlite3.Connection,
    output_root: Path,
    run_id: str,
    summary: dict[str, Any],
) -> None:
    completed = utc_now_iso()
    summary["completed_at_utc"] = completed
    connection.execute(
        """
        UPDATE extraction_runs SET
            completed_at_utc = ?, status = ?, selected_files = ?,
            completed_files = ?, skipped_files = ?, failed_files = ?,
            records_read = ?, records_written = ?, unknown_records = ?,
            malformed_records = ?, error_message = ?
        WHERE run_id = ?
        """,
        (
            completed,
            summary["status"],
            summary["selected_files"],
            summary["completed_files"],
            summary["skipped_files"],
            summary["failed_files"],
            summary["records_read"],
            summary["records_written"],
            summary["unknown_records"],
            summary["malformed_records"],
            summary.get("error_message"),
            run_id,
        ),
    )
    connection.commit()
    state_root = output_root / "_extraction_state"
    atomic_write_text(
        state_root / "runs" / f"{run_id}.json",
        json.dumps(summary, indent=2, sort_keys=True, default=str),
    )
    atomic_write_text(
        state_root / "latest_run_summary.json",
        json.dumps(summary, indent=2, sort_keys=True, default=str),
    )
    append_run_csv(state_root / "extraction_runs.csv", summary)


# ---------------------------------------------------------------------------
# CLI and main orchestration
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract all Stream bronze WAL market events into resumable canonical Parquet families."
    )
    parser.add_argument("--input-root", required=True, help="Root containing raw market/date WAL files.")
    parser.add_argument(
        "--output-root",
        default="./data/canonical/market_events_v1",
        help="Canonical output root.",
    )
    parser.add_argument(
        "--state-db",
        help="SQLite state path. Defaults to OUTPUT_ROOT/_extraction_state/extraction_state.sqlite.",
    )
    parser.add_argument("--source-glob", default=DEFAULT_GLOB)
    parser.add_argument("--source-date-from", help="Optional raw path date lower bound, YYYY-MM-DD.")
    parser.add_argument("--source-date-to", help="Optional raw path date upper bound, YYYY-MM-DD.")
    parser.add_argument("--max-files", type=int, help="Process at most this many discovered files.")
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Parallel source WAL workers. Two is a safe default for a 4-core/12-GB machine.",
    )
    parser.add_argument(
        "--fingerprint-mode",
        choices=["quick", "full"],
        default="quick",
        help="Quick hashes first/last compressed blocks; full hashes the entire compressed file.",
    )
    parser.add_argument("--compression", choices=["zstd", "snappy", "gzip"], default="zstd")
    parser.add_argument("--compression-level", type=int, default=3)
    parser.add_argument(
        "--include-system-events",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--retry-failed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fail-on-unknown",
        action="store_true",
        help="Fail a source file on an unknown market/etype instead of preserving it under unknown/.",
    )
    parser.add_argument(
        "--skip-malformed",
        action="store_true",
        help="Record malformed rows in a sidecar and continue. Default is to fail and retry the WAL.",
    )
    parser.add_argument("--force", action="store_true", help="Reprocess all selected source files.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--min-free-space-gb",
        type=float,
        default=10.0,
        help=(
            "Stop dispatching new WALs when output-disk free space falls below this "
            "threshold. Default: 10 GiB. Set 0 to disable the guard."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250_000,
        help="Worker progress print interval in source records; use 0 to disable.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.min_free_space_gb < 0:
        parser.error("--min-free-space-gb cannot be negative")
    for label in ("source_date_from", "source_date_to"):
        value = getattr(args, label)
        if value and not DATE_COMPONENT_RE.match(value):
            parser.error(f"--{label.replace('_', '-')} must be YYYY-MM-DD")
    return args


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not input_root.exists() or not input_root.is_dir():
        raise SystemExit(f"Input root does not exist or is not a directory: {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    state_db_path = (
        Path(args.state_db).expanduser().resolve()
        if args.state_db
        else output_root / "_extraction_state" / "extraction_state.sqlite"
    )

    print("Discovering source WAL files and computing fingerprints...", flush=True)
    sources = discover_sources(
        input_root=input_root,
        source_glob=args.source_glob,
        fingerprint_mode=args.fingerprint_mode,
        source_date_from=args.source_date_from,
        source_date_to=args.source_date_to,
        max_files=args.max_files,
    )
    if not sources:
        print(f"No files matched {args.source_glob!r} under {input_root}")
        return 0

    connection = connect_state_db(state_db_path)
    initialise_state_db(connection)
    write_schema_manifest(output_root)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    create_run(connection, run_id, input_root, output_root, args, len(sources))
    started_at = utc_now_iso()

    selected: list[tuple[SourceFile, str]] = []
    skipped: list[tuple[SourceFile, str]] = []
    for source in sources:
        state = source_state(connection, source.relative_path)
        process, reason = should_process_source(
            state,
            source,
            output_root,
            args.force,
            args.retry_failed,
        )
        if process:
            selected.append((source, reason))
        else:
            skipped.append((source, reason))

    print(f"Run ID          : {run_id}")
    print(f"Input root      : {input_root}")
    print(f"Output root     : {output_root}")
    print(f"State database  : {state_db_path}")
    print(f"Discovered files: {len(sources):,}")
    print(f"Selected files  : {len(selected):,}")
    print(f"Skipped files   : {len(skipped):,}")
    print(f"Workers         : {args.workers}")
    print(f"Output free     : {format_gib(free_space_bytes(output_root))}")
    print(f"Free-space guard: {args.min_free_space_gb:.2f} GiB")

    if args.dry_run:
        for source, reason in selected:
            print(f"PROCESS {reason:24s} {source.relative_path}")
        for source, reason in skipped:
            print(f"SKIP    {reason:24s} {source.relative_path}")
        summary = {
            "run_id": run_id,
            "started_at_utc": started_at,
            "status": "DRY_RUN",
            "input_root": str(input_root),
            "output_root": str(output_root),
            "discovered_files": len(sources),
            "selected_files": len(selected),
            "completed_files": 0,
            "skipped_files": len(skipped),
            "failed_files": 0,
            "records_read": 0,
            "records_written": 0,
            "unknown_records": 0,
            "malformed_records": 0,
            "selected_reasons": dict(Counter(reason for _, reason in selected)),
            "skipped_reasons": dict(Counter(reason for _, reason in skipped)),
        }
        finish_run(connection, output_root, run_id, summary)
        connection.close()
        return 0

    totals = Counter()
    failures: list[dict[str, Any]] = []
    stopped_for_low_space = False
    stopped_for_enospc = False
    submitted_files = 0

    def make_job(source: SourceFile, reason: str) -> dict[str, Any]:
        return {
            "source": source.__dict__,
            "output_root": str(output_root),
            "run_id": run_id,
            "include_system_events": args.include_system_events,
            "fail_on_unknown": args.fail_on_unknown,
            "skip_malformed": args.skip_malformed,
            "compression": args.compression,
            "compression_level": args.compression_level,
            "progress_every": args.progress_every,
            "selection_reason": reason,
        }

    def enough_free_space() -> bool:
        if args.min_free_space_gb <= 0:
            return True
        minimum = int(args.min_free_space_gb * (1024 ** 3))
        return free_space_bytes(output_root) >= minimum

    try:
        if selected:
            selected_iter = iter(selected)
            future_map: dict[Any, tuple[dict[str, Any], int, list[str], SourceFile]] = {}
            stop_dispatch = False

            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                def submit_next() -> bool:
                    nonlocal submitted_files, stop_dispatch, stopped_for_low_space
                    if stop_dispatch:
                        return False
                    if not enough_free_space():
                        stopped_for_low_space = True
                        stop_dispatch = True
                        print(
                            "STOPPING: output free space fell below "
                            f"{args.min_free_space_gb:.2f} GiB "
                            f"(currently {format_gib(free_space_bytes(output_root))}).",
                            file=sys.stderr,
                            flush=True,
                        )
                        return False
                    try:
                        source, reason = next(selected_iter)
                    except StopIteration:
                        return False
                    attempt_id, old_paths = mark_source_running(connection, run_id, source)
                    job = make_job(source, reason)
                    future = executor.submit(process_source_job, job)
                    future_map[future] = (job, attempt_id, old_paths, source)
                    submitted_files += 1
                    return True

                for _ in range(min(args.workers, len(selected))):
                    if not submit_next():
                        break

                while future_map:
                    done, _ = wait(tuple(future_map), return_when=FIRST_COMPLETED)
                    completed_slots = 0
                    for future in done:
                        job, attempt_id, old_paths, source = future_map.pop(future)
                        completed_slots += 1
                        try:
                            result = future.result()
                        except Exception as exc:  # unexpected worker-process failure
                            result = {
                                "ok": False,
                                "source_relative_path": source.relative_path,
                                "source_file_id": source.source_file_id,
                                "records_read": 0,
                                "records_written": 0,
                                "records_committed": 0,
                                "malformed_records": 0,
                                "fatal_error_code": (
                                    "NO_SPACE_LEFT" if is_no_space_error(exc) else None
                                ),
                                "error": (
                                    f"Worker process failure: {type(exc).__name__}: {exc}"
                                ),
                            }

                        totals["records_read"] += int(result.get("records_read", 0))

                        if result.get("ok"):
                            committed = int(
                                result.get("records_committed", result.get("records_written", 0))
                            )
                            totals["records_written"] += committed
                            totals["unknown_records"] += int(result.get("unknown_records", 0))
                            totals["malformed_records"] += int(result.get("malformed_records", 0))
                            new_paths = set(result["output_paths"])
                            for old_relative in old_paths:
                                if old_relative in new_paths:
                                    continue
                                old_path = output_root / old_relative
                                if old_path.exists() and old_path.is_file():
                                    old_path.unlink()
                                    prune_empty_parents(old_path, output_root)
                            mark_source_success(
                                connection, run_id, attempt_id, source, result
                            )
                            totals["completed_files"] += 1
                            families = ", ".join(
                                f"{key}={value:,}"
                                for key, value in result["family_counts"].items()
                            )
                            print(
                                f"COMPLETED {source.relative_path}: "
                                f"read={result['records_read']:,}, "
                                f"written={committed:,}"
                                + (f" [{families}]" if families else ""),
                                flush=True,
                            )
                        else:
                            buffered = int(result.get("records_written", 0))
                            totals["records_buffered_in_failed_files"] += buffered
                            # malformed_records is now reserved for genuine parse/schema
                            # errors, never storage failures.
                            totals["malformed_records"] += int(result.get("malformed_records", 0))
                            mark_source_failed(
                                connection, run_id, attempt_id, source, result
                            )
                            totals["failed_files"] += 1
                            failures.append(dict(result))
                            print(
                                f"FAILED    {source.relative_path}: {result.get('error')}",
                                file=sys.stderr,
                                flush=True,
                            )
                            if result.get("fatal_error_code") == "NO_SPACE_LEFT":
                                stopped_for_enospc = True
                                stop_dispatch = True
                                print(
                                    "STOPPING: disk-full error detected. No additional "
                                    "WALs will be dispatched in this run.",
                                    file=sys.stderr,
                                    flush=True,
                                )

                    if not stop_dispatch:
                        for _ in range(completed_slots):
                            if not submit_next():
                                break
    except KeyboardInterrupt:
        print("Interrupted. Files still marked RUNNING will be retried next time.", file=sys.stderr)
        summary = {
            "run_id": run_id,
            "started_at_utc": started_at,
            "status": "INTERRUPTED",
            "input_root": str(input_root),
            "output_root": str(output_root),
            "discovered_files": len(sources),
            "selected_files": len(selected),
            "submitted_files": submitted_files,
            "unsubmitted_files": len(selected) - submitted_files,
            "completed_files": totals["completed_files"],
            "skipped_files": len(skipped),
            "failed_files": totals["failed_files"],
            "records_read": totals["records_read"],
            "records_written": totals["records_written"],
            "records_buffered_in_failed_files": totals["records_buffered_in_failed_files"],
            "unknown_records": totals["unknown_records"],
            "malformed_records": totals["malformed_records"],
            "error_message": "KeyboardInterrupt",
        }
        finish_run(connection, output_root, run_id, summary)
        connection.close()
        return 130

    if stopped_for_enospc:
        status = "STOPPED_NO_SPACE_LEFT"
    elif stopped_for_low_space:
        status = "STOPPED_LOW_FREE_SPACE"
    else:
        status = "COMPLETED" if totals["failed_files"] == 0 else "COMPLETED_WITH_FAILURES"
    summary = {
        "run_id": run_id,
        "started_at_utc": started_at,
        "status": status,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "state_db": str(state_db_path),
        "extractor_version": EXTRACTOR_VERSION,
        "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
        "discovered_files": len(sources),
        "selected_files": len(selected),
        "submitted_files": submitted_files,
        "unsubmitted_files": len(selected) - submitted_files,
        "completed_files": totals["completed_files"],
        "skipped_files": len(skipped),
        "failed_files": totals["failed_files"],
        "records_read": totals["records_read"],
        "records_written": totals["records_written"],
        "records_buffered_in_failed_files": totals["records_buffered_in_failed_files"],
        "unknown_records": totals["unknown_records"],
        "malformed_records": totals["malformed_records"],
        "selected_reasons": dict(Counter(reason for _, reason in selected)),
        "skipped_reasons": dict(Counter(reason for _, reason in skipped)),
        "failed_sources": [
            {
                "source_relative_path": item.get("source_relative_path"),
                "error": item.get("error"),
            }
            for item in failures
        ],
    }
    finish_run(connection, output_root, run_id, summary)
    connection.close()

    print("\nExtraction finished.")
    print(f"Status           : {status}")
    print(f"Completed files  : {totals['completed_files']:,}")
    print(f"Skipped files    : {len(skipped):,}")
    print(f"Submitted files  : {submitted_files:,}")
    print(f"Unsubmitted files: {len(selected) - submitted_files:,}")
    print(f"Failed files     : {totals['failed_files']:,}")
    print(f"Records read     : {totals['records_read']:,}")
    print(f"Records committed: {totals['records_written']:,}")
    print(
        "Rows discarded from failed temporary shards: "
        f"{totals['records_buffered_in_failed_files']:,}"
    )
    print(f"Unknown records  : {totals['unknown_records']:,}")
    print(f"Malformed records: {totals['malformed_records']:,}")
    print(f"Summary          : {output_root / '_extraction_state' / 'latest_run_summary.json'}")

    return 0 if status == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
