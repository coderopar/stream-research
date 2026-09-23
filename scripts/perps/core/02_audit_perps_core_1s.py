#!/usr/bin/env python
"""
Audit the perps one-second core layer produced by
scripts/perps/core/01_build_perps_core_1s.py.

The audit is independent of the builder's in-process validations. It reconciles
build state, current canonical inputs, physical Parquet outputs, row-level
measurements, causal state alignment, and date coverage one partition at a time.
It never loads the full multi-day dataset into memory.

Dependencies
------------
    pip install pandas numpy pyarrow

Typical usage
-------------
    python scripts/perps/core/02_audit_perps_core_1s.py \
      --canonical-root "./data/canonical/market_events_v1" \
      --core-root "./data/derived/perps/core_1s_v1" \
      --symbol BTCUSDT

Reports are written to:
    reports/perps_core_1s_audit/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import traceback
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency. Activate the project virtual environment and run: "
        "pip install pandas numpy pyarrow"
    ) from exc


AUDITOR_VERSION = "1.0.1"
EXPECTED_CORE_SCHEMA_VERSION = 1
SECONDS_PER_DAY = 86_400
NANOSECONDS_PER_SECOND = 1_000_000_000
FAMILIES = ("trades", "mark_price", "open_interest", "liquidations")
DEFAULT_CANONICAL_ROOT = "./data/canonical/market_events_v1"
DEFAULT_CORE_ROOT = "./data/derived/perps/core_1s_v1"
DEFAULT_REPORT_ROOT = "./reports/perps_core_1s_audit"
DATE_RE = re.compile(r"^date=(\d{4}-\d{2}-\d{2})$")
SYMBOL_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
SEVERITY_ORDER = {"INFO": 0, "WARN": 1, "FAIL": 2}


# ---------------------------------------------------------------------------
# Expected output schema (must match builder schema version 1)
# ---------------------------------------------------------------------------


def expected_schema() -> pa.Schema:
    fields: list[pa.Field] = [
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
    for name in ("trade_raw_record_count", "trade_count"):
        fields.append(pa.field(name, pa.int64(), nullable=False))
    fields.extend(
        [
            pa.field("trade_observed", pa.bool_(), nullable=False),
            *[
                pa.field(name, pa.float64())
                for name in (
                    "trade_open",
                    "trade_high",
                    "trade_low",
                    "trade_close",
                    "trade_vwap",
                    "base_volume",
                    "quote_volume",
                )
            ],
        ]
    )
    for name in (
        "aggressive_buy_trade_count",
        "aggressive_sell_trade_count",
        "aggressor_unknown_trade_count",
    ):
        fields.append(pa.field(name, pa.int64(), nullable=False))
    for name in (
        "aggressive_buy_base_volume",
        "aggressive_sell_base_volume",
        "aggressor_unknown_base_volume",
        "aggressive_buy_quote_volume",
        "aggressive_sell_quote_volume",
        "aggressor_unknown_quote_volume",
        "signed_base_flow",
        "signed_quote_flow",
        "mean_trade_base",
        "mean_trade_notional",
        "max_trade_base",
        "max_trade_notional",
    ):
        nullable = not (
            name.startswith("max_")
            or "volume" in name
            or name.startswith("signed_")
        )
        fields.append(pa.field(name, pa.float64(), nullable=nullable))
    for name in (
        "first_trade_ts_recv_ns",
        "last_trade_ts_recv_ns",
        "first_trade_exchange_event_ms",
        "last_trade_exchange_event_ms",
        "first_trade_time_ms",
        "last_trade_time_ms",
    ):
        fields.append(pa.field(name, pa.int64()))
    fields.extend(
        [
            pa.field("last_trade_price", pa.float64()),
            pa.field("last_trade_state_ts_recv_ns", pa.int64()),
            pa.field("last_trade_state_exchange_event_ms", pa.int64()),
            pa.field("last_trade_state_trade_time_ms", pa.int64()),
            pa.field("trade_price_age_seconds", pa.float64()),
            pa.field("trade_price_fresh", pa.bool_(), nullable=False),
        ]
    )
    for name in ("mark_raw_record_count", "mark_update_count"):
        fields.append(pa.field(name, pa.int64(), nullable=False))
    fields.append(pa.field("mark_updated", pa.bool_(), nullable=False))
    for name in (
        "mark_price",
        "alternate_mark_price",
        "estimated_settle_price",
        "index_price",
        "funding_rate",
    ):
        fields.append(pa.field(name, pa.float64()))
    fields.extend(
        [
            pa.field("next_funding_time_ms", pa.int64()),
            pa.field("seconds_to_next_funding", pa.float64()),
            pa.field("mark_index_basis", pa.float64()),
            pa.field("mark_index_basis_bps", pa.float64()),
            pa.field("mark_state_ts_recv_ns", pa.int64()),
            pa.field("mark_exchange_event_ms", pa.int64()),
            pa.field("mark_age_seconds", pa.float64()),
            pa.field("mark_fresh", pa.bool_(), nullable=False),
        ]
    )
    for name in ("open_interest_raw_record_count", "open_interest_update_count"):
        fields.append(pa.field(name, pa.int64(), nullable=False))
    fields.append(pa.field("open_interest_updated", pa.bool_(), nullable=False))
    for name in (
        "open_interest",
        "open_interest_change",
        "open_interest_change_pct",
        "open_interest_notional",
    ):
        fields.append(pa.field(name, pa.float64()))
    fields.extend(
        [
            pa.field("open_interest_state_ts_recv_ns", pa.int64()),
            pa.field("open_interest_exchange_time_ms", pa.int64()),
            pa.field("open_interest_age_seconds", pa.float64()),
            pa.field("open_interest_fresh", pa.bool_(), nullable=False),
        ]
    )
    for name in ("liquidation_raw_record_count", "liquidation_event_count"):
        fields.append(pa.field(name, pa.int64(), nullable=False))
    fields.append(pa.field("liquidation_observed", pa.bool_(), nullable=False))
    for name in (
        "long_liquidation_count",
        "short_liquidation_count",
        "unknown_liquidation_side_count",
    ):
        fields.append(pa.field(name, pa.int64(), nullable=False))
    for name in (
        "long_liquidation_base_volume",
        "short_liquidation_base_volume",
        "unknown_liquidation_base_volume",
        "liquidation_base_volume",
        "long_liquidation_notional",
        "short_liquidation_notional",
        "unknown_liquidation_notional",
        "liquidation_notional",
        "net_liquidation_notional",
        "max_liquidation_notional",
    ):
        fields.append(pa.field(name, pa.float64(), nullable=False))
    fields.append(pa.field("core_context_ready", pa.bool_(), nullable=False))
    return pa.schema(fields)


EXPECTED_SCHEMA = expected_schema()
AUDIT_COLUMNS = EXPECTED_SCHEMA.names
COUNT_COLUMNS = (
    "input_record_count",
    "trade_raw_record_count",
    "trade_count",
    "aggressive_buy_trade_count",
    "aggressive_sell_trade_count",
    "aggressor_unknown_trade_count",
    "mark_raw_record_count",
    "mark_update_count",
    "open_interest_raw_record_count",
    "open_interest_update_count",
    "liquidation_raw_record_count",
    "liquidation_event_count",
    "long_liquidation_count",
    "short_liquidation_count",
    "unknown_liquidation_side_count",
)
NONNEGATIVE_COLUMNS = (
    *COUNT_COLUMNS,
    "base_volume",
    "quote_volume",
    "aggressive_buy_base_volume",
    "aggressive_sell_base_volume",
    "aggressor_unknown_base_volume",
    "aggressive_buy_quote_volume",
    "aggressive_sell_quote_volume",
    "aggressor_unknown_quote_volume",
    "max_trade_base",
    "max_trade_notional",
    "long_liquidation_base_volume",
    "short_liquidation_base_volume",
    "unknown_liquidation_base_volume",
    "liquidation_base_volume",
    "long_liquidation_notional",
    "short_liquidation_notional",
    "unknown_liquidation_notional",
    "liquidation_notional",
    "max_liquidation_notional",
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Issue:
    severity: str
    code: str
    scope: str
    message: str
    receive_date: str | None = None
    output_path: str | None = None
    family: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "scope": self.scope,
            "message": self.message,
            "receive_date": self.receive_date,
            "output_path": self.output_path,
            "family": self.family,
        }


@dataclass
class Invariant:
    name: str
    status: str
    issue_count: int
    details: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "issue_count": self.issue_count,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


def compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_date(value: str | None, name: str) -> date | None:
    if value is None:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SystemExit(f"{name} must use YYYY-MM-DD; received {value!r}") from exc


def safe_symbol(symbol: str) -> str:
    value = SYMBOL_SAFE_RE.sub("_", symbol.strip()).strip("_")
    if not value:
        raise SystemExit("--symbol must contain at least one safe character")
    return value


def day_start_second(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())


def load_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def add_issue(
    issues: list[Issue],
    severity: str,
    code: str,
    scope: str,
    message: str,
    *,
    receive_date: str | None = None,
    output_path: str | None = None,
    family: str | None = None,
) -> None:
    issues.append(
        Issue(
            severity=severity,
            code=code,
            scope=scope,
            message=message,
            receive_date=receive_date,
            output_path=output_path,
            family=family,
        )
    )


def semantic_type_equal(actual: pa.DataType, expected: pa.DataType) -> bool:
    """Compare logical Arrow types while tolerating Parquet timestamp widening.

    Parquet has millisecond/microsecond/nanosecond timestamp logical types but no
    native second-resolution timestamp. A ``timestamp[s, UTC]`` Arrow field can
    therefore be read back as ``timestamp[ms, UTC]`` even though every stored
    value remains exactly aligned to a whole second. Row-level timestamp checks
    below enforce that exact alignment, so a unit-only widening is harmless.
    """
    if pa.types.is_timestamp(actual) and pa.types.is_timestamp(expected):
        return actual.tz == expected.tz
    return actual.equals(expected)


def semantic_schema_equal(actual: pa.Schema, expected: pa.Schema) -> bool:
    actual = actual.remove_metadata()
    expected = expected.remove_metadata()
    if len(actual) != len(expected):
        return False
    for actual_field, expected_field in zip(actual, expected):
        if actual_field.name != expected_field.name:
            return False
        if actual_field.nullable != expected_field.nullable:
            return False
        if not semantic_type_equal(actual_field.type, expected_field.type):
            return False
    return True


def datetime_to_epoch_seconds(series: pd.Series) -> pd.Series:
    """Convert any pandas datetime resolution to exact Unix seconds.

    ``Series.astype("int64")`` uses the series' physical datetime unit in
    recent pandas versions. Dividing that result by 1e9 is wrong when Parquet
    returns ``datetime64[ms]`` or ``datetime64[us]``. Normalize to nanoseconds
    first, then convert to seconds.
    """
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    array = parsed.array
    if hasattr(array, "as_unit"):
        nanoseconds = array.as_unit("ns").asi8
    else:  # pandas compatibility fallback
        nanoseconds = (
            parsed.dt.tz_convert(None)
            .to_numpy(dtype="datetime64[ns]")
            .astype("int64")
        )
    return pd.Series(
        nanoseconds // NANOSECONDS_PER_SECOND,
        index=series.index,
        dtype="int64",
    )


def metadata_text(schema: pa.Schema) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in (schema.metadata or {}).items():
        result[key.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
    return result


def finite_or_null(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    return np.isnan(values) | np.isfinite(values)


def close_mask(left: pd.Series | np.ndarray, right: pd.Series | np.ndarray, *, atol: float, rtol: float = 1e-10) -> np.ndarray:
    a = np.asarray(left, dtype="float64")
    b = np.asarray(right, dtype="float64")
    both_nan = np.isnan(a) & np.isnan(b)
    return both_nan | np.isclose(a, b, atol=atol, rtol=rtol, equal_nan=False)


def parquet_rows(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def discover_dates(root: Path, symbol_partition: str) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    base = root / f"symbol={symbol_partition}"
    if not base.exists():
        return result
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        match = DATE_RE.match(child.name)
        if not match:
            continue
        result[match.group(1)] = sorted(child.glob("*.parquet"))
    return result




def discover_canonical_dates(canonical_root: Path) -> set[str]:
    values: set[str] = set()
    for family in FAMILIES:
        base = canonical_root / "perps" / family
        if not base.exists():
            continue
        for child in base.iterdir():
            if child.is_dir():
                match = DATE_RE.match(child.name)
                if match:
                    values.add(match.group(1))
    return values

def canonical_files(canonical_root: Path, family: str, day_text: str) -> list[Path]:
    folder = canonical_root / "perps" / family / f"date={day_text}"
    return sorted(folder.glob("*.parquet")) if folder.exists() else []


def canonical_manifest(canonical_root: Path, day_text: str) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for family in FAMILIES:
        entries: list[dict[str, Any]] = []
        for path in canonical_files(canonical_root, family, day_text):
            stat = path.stat()
            entries.append(
                {
                    "path": path.relative_to(canonical_root).as_posix(),
                    "size_bytes": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "rows": parquet_rows(path),
                }
            )
        result[family] = entries
    return result


def count_symbol_rows(path: Path, symbol: str) -> tuple[int, int]:
    """Return (all rows, matching symbol rows), using statistics when possible."""
    pf = pq.ParquetFile(path)
    total = int(pf.metadata.num_rows)
    try:
        index = pf.schema_arrow.names.index("symbol")
    except ValueError as exc:
        raise RuntimeError(f"Canonical file lacks symbol column: {path}") from exc

    matching = 0
    for rg_index in range(pf.num_row_groups):
        rg = pf.metadata.row_group(rg_index)
        rows = int(rg.num_rows)
        stats = rg.column(index).statistics
        used_stats = False
        if stats is not None and stats.has_min_max:
            minimum = stats.min.decode() if isinstance(stats.min, bytes) else str(stats.min)
            maximum = stats.max.decode() if isinstance(stats.max, bytes) else str(stats.max)
            if minimum == maximum == symbol:
                matching += rows
                used_stats = True
            elif symbol < minimum or symbol > maximum:
                used_stats = True
        if not used_stats:
            table = pf.read_row_group(rg_index, columns=["symbol"])
            values = table.column("symbol").to_pylist()
            matching += sum(1 for value in values if value == symbol)
    return total, matching


def read_build_rows(connection: sqlite3.Connection, symbol: str) -> list[sqlite3.Row]:
    return connection.execute(
        "SELECT * FROM date_builds WHERE symbol=? ORDER BY receive_date",
        (symbol,),
    ).fetchall()


def read_run_args(connection: sqlite3.Connection, run_identifier: str | None) -> dict[str, Any]:
    if not run_identifier:
        return {}
    row = connection.execute(
        "SELECT arguments_json FROM build_runs WHERE run_id=?",
        (run_identifier,),
    ).fetchone()
    return load_json(row[0] if row else None, {})


# ---------------------------------------------------------------------------
# Date audit
# ---------------------------------------------------------------------------


def audit_date(
    *,
    day_text: str,
    state: sqlite3.Row | None,
    output_paths: list[Path],
    core_root: Path,
    canonical_root: Path,
    symbol: str,
    connection: sqlite3.Connection,
    issues: list[Issue],
    dataset_first_date: str,
    dataset_last_date: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    summary: dict[str, Any] = {
        "receive_date": day_text,
        "state_status": state["status"] if state else None,
        "output_files": len(output_paths),
    }
    gaps: list[dict[str, Any]] = []
    reconciliations: list[dict[str, Any]] = []

    if state is None:
        add_issue(issues, "FAIL", "MISSING_BUILD_STATE", "date", "No date_builds row exists", receive_date=day_text)
        return summary, gaps, reconciliations
    if state["status"] != "COMPLETED":
        add_issue(
            issues,
            "FAIL",
            "DATE_NOT_COMPLETED",
            "date",
            f"Build state is {state['status']!r}, expected COMPLETED",
            receive_date=day_text,
        )
    if len(output_paths) != 1:
        add_issue(
            issues,
            "FAIL",
            "OUTPUT_FILE_COUNT",
            "date",
            f"Found {len(output_paths)} Parquet files; expected exactly one",
            receive_date=day_text,
        )
    expected_relative = state["output_relative_path"]
    expected_path = core_root / expected_relative if expected_relative else None
    if expected_path is None or not expected_path.is_file():
        add_issue(
            issues,
            "FAIL",
            "REGISTERED_OUTPUT_MISSING",
            "date",
            f"Registered output does not exist: {expected_relative!r}",
            receive_date=day_text,
        )
        return summary, gaps, reconciliations
    extra_paths = [path for path in output_paths if path.resolve() != expected_path.resolve()]
    for path in extra_paths:
        add_issue(
            issues,
            "FAIL",
            "EXTRA_OUTPUT_FILE",
            "date",
            "Unexpected extra Parquet file in date partition",
            receive_date=day_text,
            output_path=str(path),
        )

    output_relative = expected_path.relative_to(core_root).as_posix()
    summary["output_relative_path"] = output_relative
    try:
        pf = pq.ParquetFile(expected_path)
    except Exception as exc:
        add_issue(
            issues,
            "FAIL",
            "PARQUET_UNREADABLE",
            "date",
            str(exc),
            receive_date=day_text,
            output_path=output_relative,
        )
        return summary, gaps, reconciliations

    physical_rows = int(pf.metadata.num_rows)
    summary["physical_rows"] = physical_rows
    summary["output_bytes"] = int(expected_path.stat().st_size)
    summary["row_groups"] = int(pf.num_row_groups)
    if physical_rows != int(state["output_rows"]):
        add_issue(
            issues,
            "FAIL",
            "STATE_ROW_COUNT_MISMATCH",
            "date",
            f"Physical rows={physical_rows:,}; state rows={int(state['output_rows']):,}",
            receive_date=day_text,
            output_path=output_relative,
        )

    actual_schema = pf.schema_arrow
    if not semantic_schema_equal(actual_schema, EXPECTED_SCHEMA):
        add_issue(
            issues,
            "FAIL",
            "OUTPUT_SCHEMA_MISMATCH",
            "date",
            f"Actual schema differs from expected core schema v{EXPECTED_CORE_SCHEMA_VERSION}",
            receive_date=day_text,
            output_path=output_relative,
        )
    metadata = metadata_text(actual_schema)
    summary["builder_version"] = metadata.get("builder_version")
    summary["schema_metadata"] = metadata.get("perps_core_schema_version")
    for key, expected in (
        ("perps_core_schema_version", str(EXPECTED_CORE_SCHEMA_VERSION)),
        ("symbol", symbol),
        ("receive_date", day_text),
        ("input_signature", state["input_signature"]),
        ("config_signature", state["config_signature"]),
        ("carry_in_signature", state["carry_in_signature"]),
    ):
        if metadata.get(key) != str(expected):
            add_issue(
                issues,
                "FAIL",
                "OUTPUT_METADATA_MISMATCH",
                "date",
                f"Metadata {key}={metadata.get(key)!r}; expected {str(expected)!r}",
                receive_date=day_text,
                output_path=output_relative,
            )

    # Current canonical inputs must still match the manifest used by the build.
    stored_manifest = load_json(state["input_manifest_json"], {})
    current_manifest = canonical_manifest(canonical_root, day_text)
    if compact_json(stored_manifest) != compact_json(current_manifest):
        add_issue(
            issues,
            "FAIL",
            "CANONICAL_INPUT_CHANGED",
            "date",
            "Current canonical input manifest differs from the completed build manifest",
            receive_date=day_text,
        )

    # Read the one-day output. 86,400 x 90 columns is intentionally bounded.
    try:
        table = pf.read(columns=AUDIT_COLUMNS)
        frame = table.to_pandas(split_blocks=True, self_destruct=True)
    except Exception as exc:
        add_issue(
            issues,
            "FAIL",
            "OUTPUT_READ_FAILED",
            "date",
            str(exc),
            receive_date=day_text,
            output_path=output_relative,
        )
        return summary, gaps, reconciliations

    if frame.empty:
        add_issue(issues, "FAIL", "EMPTY_OUTPUT", "date", "Output is empty", receive_date=day_text)
        return summary, gaps, reconciliations

    day_value = datetime.strptime(day_text, "%Y-%m-%d").date()
    day_start = day_start_second(day_value)
    day_end = day_start + SECONDS_PER_DAY - 1
    receive = pd.to_numeric(frame["receive_second"], errors="coerce").astype("int64")
    first_second = int(receive.iloc[0])
    last_second = int(receive.iloc[-1])
    rows = len(frame)
    span_rows = last_second - first_second + 1
    leading_missing = max(0, first_second - day_start)
    trailing_missing = max(0, day_end - last_second)
    internal_missing = max(0, span_rows - rows)
    summary.update(
        {
            "first_receive_second": first_second,
            "last_receive_second": last_second,
            "first_utc": datetime.fromtimestamp(first_second, tz=timezone.utc).isoformat(),
            "last_utc": datetime.fromtimestamp(last_second, tz=timezone.utc).isoformat(),
            "leading_missing_seconds": leading_missing,
            "trailing_missing_seconds": trailing_missing,
            "internal_missing_seconds": internal_missing,
            "full_day_rows": rows == SECONDS_PER_DAY,
        }
    )
    if first_second != int(state["first_receive_second"]) or last_second != int(state["last_receive_second"]):
        add_issue(
            issues,
            "FAIL",
            "STATE_BOUNDARY_MISMATCH",
            "date",
            "Physical first/last receive seconds differ from build state",
            receive_date=day_text,
        )
    if receive.duplicated().any():
        add_issue(
            issues,
            "FAIL",
            "DUPLICATE_SECONDS",
            "date",
            f"Duplicate receive seconds={int(receive.duplicated().sum()):,}",
            receive_date=day_text,
        )
    diffs = receive.diff().dropna()
    if (diffs <= 0).any():
        add_issue(issues, "FAIL", "SECONDS_NOT_STRICT", "date", "receive_second is not strictly increasing", receive_date=day_text)
    if internal_missing:
        missing = np.setdiff1d(np.arange(first_second, last_second + 1, dtype=np.int64), receive.to_numpy())
        add_issue(
            issues,
            "FAIL",
            "INTERNAL_SECOND_GAPS",
            "date",
            f"Missing {len(missing):,} seconds inside the output span",
            receive_date=day_text,
        )
        for second in missing[:1000]:
            gaps.append(
                {
                    "receive_date": day_text,
                    "gap_type": "INTERNAL",
                    "receive_second": int(second),
                    "timestamp_utc": datetime.fromtimestamp(int(second), tz=timezone.utc).isoformat(),
                }
            )
    if leading_missing:
        leading_severity = "INFO" if day_text == dataset_first_date else "WARN"
        leading_code = "EXPECTED_DATASET_START" if day_text == dataset_first_date else "LEADING_BOUNDARY_GAP"
        add_issue(
            issues,
            leading_severity,
            leading_code,
            "coverage",
            f"Date begins {leading_missing:,} seconds after 00:00:00 UTC",
            receive_date=day_text,
        )
        gaps.append(
            {
                "receive_date": day_text,
                "gap_type": "LEADING_BOUNDARY",
                "start_second": day_start,
                "end_second": first_second - 1,
                "seconds": leading_missing,
                "start_utc": datetime.fromtimestamp(day_start, tz=timezone.utc).isoformat(),
                "end_utc": datetime.fromtimestamp(first_second - 1, tz=timezone.utc).isoformat(),
            }
        )
    if trailing_missing:
        trailing_severity = "INFO" if day_text == dataset_last_date else "WARN"
        trailing_code = "EXPECTED_DATASET_END" if day_text == dataset_last_date else "TRAILING_BOUNDARY_GAP"
        add_issue(
            issues,
            trailing_severity,
            trailing_code,
            "coverage",
            f"Date ends {trailing_missing:,} seconds before 23:59:59 UTC",
            receive_date=day_text,
        )
        gaps.append(
            {
                "receive_date": day_text,
                "gap_type": "TRAILING_BOUNDARY",
                "start_second": last_second + 1,
                "end_second": day_end,
                "seconds": trailing_missing,
                "start_utc": datetime.fromtimestamp(last_second + 1, tz=timezone.utc).isoformat(),
                "end_utc": datetime.fromtimestamp(day_end, tz=timezone.utc).isoformat(),
            }
        )

    def fail_if(condition: bool, code: str, message: str) -> None:
        if condition:
            add_issue(issues, "FAIL", code, "row_invariants", message, receive_date=day_text, output_path=output_relative)

    fail_if(not frame["symbol"].eq(symbol).all(), "SYMBOL_MISMATCH", "Output contains another symbol")
    fail_if(not frame["core_schema_version"].eq(EXPECTED_CORE_SCHEMA_VERSION).all(), "ROW_SCHEMA_VERSION", "Row schema version is not uniformly 1")
    fail_if(not (frame["available_after_second"] == frame["receive_second"] + 1).all(), "AVAILABILITY_SECOND", "available_after_second != receive_second + 1")
    fail_if(not (frame["seconds_from_day_start"] == frame["receive_second"] - day_start).all(), "DAY_OFFSET", "seconds_from_day_start is inconsistent")
    fail_if(not frame["receive_second"].between(day_start, day_end).all(), "PARTITION_DATE", "Rows fall outside their UTC date partition")

    timestamp_seconds = datetime_to_epoch_seconds(frame["timestamp_utc"])
    available_seconds = datetime_to_epoch_seconds(frame["available_after_utc"])
    fail_if(not timestamp_seconds.eq(frame["receive_second"]).all(), "TIMESTAMP_UTC", "timestamp_utc does not match receive_second")
    fail_if(not available_seconds.eq(frame["available_after_second"]).all(), "AVAILABLE_UTC", "available_after_utc does not match available_after_second")

    for column in COUNT_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        fail_if(values.isna().any(), "COUNT_NULL", f"{column} contains null/non-numeric values")
        fail_if((values < 0).any(), "COUNT_NEGATIVE", f"{column} contains negative values")
        fail_if(not np.equal(values.to_numpy(dtype="float64"), np.floor(values.to_numpy(dtype="float64"))).all(), "COUNT_NONINTEGER", f"{column} contains non-integer values")
    for column in NONNEGATIVE_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        fail_if((values.dropna() < -1e-12).any(), "NEGATIVE_MEASUREMENT", f"{column} contains negative values")
    for column in frame.select_dtypes(include=["number"]).columns:
        fail_if(not finite_or_null(frame[column]).all(), "NONFINITE_NUMERIC", f"{column} contains positive or negative infinity")

    raw_total = (
        frame["trade_raw_record_count"]
        + frame["mark_raw_record_count"]
        + frame["open_interest_raw_record_count"]
        + frame["liquidation_raw_record_count"]
    )
    fail_if(not frame["input_record_count"].eq(raw_total).all(), "INPUT_COUNT_RECONCILIATION", "input_record_count does not equal family raw counts")
    fail_if(not frame["any_perps_event_observed"].eq(raw_total > 0).all(), "ANY_EVENT_FLAG", "any_perps_event_observed is inconsistent")

    trade_count_sum = frame["aggressive_buy_trade_count"] + frame["aggressive_sell_trade_count"] + frame["aggressor_unknown_trade_count"]
    fail_if(not frame["trade_count"].eq(trade_count_sum).all(), "TRADE_SIDE_COUNT", "Trade side counts do not reconcile")
    fail_if(not frame["trade_observed"].eq(frame["trade_count"] > 0).all(), "TRADE_OBSERVED_FLAG", "trade_observed is inconsistent")
    fail_if(not close_mask(frame["base_volume"], frame["aggressive_buy_base_volume"] + frame["aggressive_sell_base_volume"] + frame["aggressor_unknown_base_volume"], atol=1e-8).all(), "TRADE_BASE_RECONCILIATION", "Trade base volume does not reconcile")
    fail_if(not close_mask(frame["quote_volume"], frame["aggressive_buy_quote_volume"] + frame["aggressive_sell_quote_volume"] + frame["aggressor_unknown_quote_volume"], atol=1e-4).all(), "TRADE_QUOTE_RECONCILIATION", "Trade quote volume does not reconcile")
    fail_if(not close_mask(frame["signed_base_flow"], frame["aggressive_buy_base_volume"] - frame["aggressive_sell_base_volume"], atol=1e-8).all(), "SIGNED_BASE_FLOW", "signed_base_flow is inconsistent")
    fail_if(not close_mask(frame["signed_quote_flow"], frame["aggressive_buy_quote_volume"] - frame["aggressive_sell_quote_volume"], atol=1e-4).all(), "SIGNED_QUOTE_FLOW", "signed_quote_flow is inconsistent")

    trade_mask = frame["trade_observed"].to_numpy(dtype=bool)
    if trade_mask.any():
        observed = frame.loc[trade_mask]
        fail_if(observed[["trade_open", "trade_high", "trade_low", "trade_close", "trade_vwap"]].isna().any().any(), "TRADE_PRICE_NULL", "Observed trade seconds contain null OHLC/VWAP")
        fail_if((observed["trade_low"] > observed["trade_high"]).any(), "TRADE_LOW_HIGH", "trade_low exceeds trade_high")
        fail_if((observed["trade_high"] < observed[["trade_open", "trade_close"]].max(axis=1)).any(), "TRADE_HIGH_OHLC", "trade_high is below open/close")
        fail_if((observed["trade_low"] > observed[["trade_open", "trade_close"]].min(axis=1)).any(), "TRADE_LOW_OHLC", "trade_low is above open/close")
        fail_if((observed["trade_vwap"] < observed["trade_low"] - 1e-8).any() or (observed["trade_vwap"] > observed["trade_high"] + 1e-8).any(), "VWAP_RANGE", "trade_vwap falls outside low/high")
        fail_if(not close_mask(observed["trade_vwap"], observed["quote_volume"] / observed["base_volume"], atol=1e-8).all(), "VWAP_FORMULA", "trade_vwap != quote_volume/base_volume")
        fail_if(not close_mask(observed["mean_trade_base"], observed["base_volume"] / observed["trade_count"], atol=1e-10).all(), "MEAN_TRADE_BASE", "mean_trade_base formula mismatch")
        fail_if(not close_mask(observed["mean_trade_notional"], observed["quote_volume"] / observed["trade_count"], atol=1e-8).all(), "MEAN_TRADE_NOTIONAL", "mean_trade_notional formula mismatch")
    nontrade = frame.loc[~trade_mask]
    if not nontrade.empty:
        fail_if(not nontrade["trade_count"].eq(0).all(), "NONTRADE_COUNT", "Non-trade seconds have nonzero trade_count")
        fail_if(not nontrade[["base_volume", "quote_volume", "max_trade_base", "max_trade_notional"]].eq(0).all().all(), "NONTRADE_VOLUME", "Non-trade seconds have nonzero volume/maxima")

    fail_if(not frame["mark_updated"].eq(frame["mark_update_count"] > 0).all(), "MARK_UPDATED_FLAG", "mark_updated is inconsistent")
    fail_if(not frame["open_interest_updated"].eq(frame["open_interest_update_count"] > 0).all(), "OI_UPDATED_FLAG", "open_interest_updated is inconsistent")
    fail_if(not frame["liquidation_observed"].eq(frame["liquidation_event_count"] > 0).all(), "LIQ_OBSERVED_FLAG", "liquidation_observed is inconsistent")

    liq_count = frame["long_liquidation_count"] + frame["short_liquidation_count"] + frame["unknown_liquidation_side_count"]
    fail_if(not frame["liquidation_event_count"].eq(liq_count).all(), "LIQ_COUNT_RECONCILIATION", "Liquidation side counts do not reconcile")
    fail_if(not close_mask(frame["liquidation_base_volume"], frame["long_liquidation_base_volume"] + frame["short_liquidation_base_volume"] + frame["unknown_liquidation_base_volume"], atol=1e-8).all(), "LIQ_BASE_RECONCILIATION", "Liquidation base volume does not reconcile")
    fail_if(not close_mask(frame["liquidation_notional"], frame["long_liquidation_notional"] + frame["short_liquidation_notional"] + frame["unknown_liquidation_notional"], atol=1e-4).all(), "LIQ_NOTIONAL_RECONCILIATION", "Liquidation notional does not reconcile")
    fail_if(not close_mask(frame["net_liquidation_notional"], frame["short_liquidation_notional"] - frame["long_liquidation_notional"], atol=1e-4).all(), "LIQ_NET_FORMULA", "net_liquidation_notional formula mismatch")

    basis_valid = frame["mark_price"].notna() & frame["index_price"].notna()
    if basis_valid.any():
        basis = frame.loc[basis_valid, "mark_price"] - frame.loc[basis_valid, "index_price"]
        fail_if(not close_mask(frame.loc[basis_valid, "mark_index_basis"], basis, atol=1e-8).all(), "BASIS_FORMULA", "mark_index_basis formula mismatch")
        nonzero_index = basis_valid & frame["index_price"].ne(0)
        expected_bps = frame.loc[nonzero_index, "mark_index_basis"] / frame.loc[nonzero_index, "index_price"] * 10_000
        fail_if(not close_mask(frame.loc[nonzero_index, "mark_index_basis_bps"], expected_bps, atol=1e-8).all(), "BASIS_BPS_FORMULA", "mark_index_basis_bps formula mismatch")

    oi_notional_valid = frame["open_interest"].notna() & frame["mark_price"].notna()
    if oi_notional_valid.any():
        expected_oi_notional = frame.loc[oi_notional_valid, "open_interest"] * frame.loc[oi_notional_valid, "mark_price"]
        fail_if(not close_mask(frame.loc[oi_notional_valid, "open_interest_notional"], expected_oi_notional, atol=1e-4).all(), "OI_NOTIONAL_FORMULA", "open_interest_notional formula mismatch")
    fail_if(frame.loc[~frame["open_interest_updated"], "open_interest_change"].notna().any(), "OI_CHANGE_WITHOUT_UPDATE", "open_interest_change appears on non-update seconds")
    oi_pct_valid = frame["open_interest_change"].notna() & frame["open_interest_change_pct"].notna() & (frame["open_interest"] - frame["open_interest_change"]).ne(0)
    if oi_pct_valid.any():
        previous = frame.loc[oi_pct_valid, "open_interest"] - frame.loc[oi_pct_valid, "open_interest_change"]
        expected_pct = frame.loc[oi_pct_valid, "open_interest_change"] / previous
        fail_if(not close_mask(frame.loc[oi_pct_valid, "open_interest_change_pct"], expected_pct, atol=1e-12).all(), "OI_CHANGE_PCT", "open_interest_change_pct formula mismatch")

    run_args = read_run_args(connection, state["last_run_id"])
    thresholds = {
        "trade": run_args.get("trade_max_age_seconds"),
        "mark": run_args.get("mark_max_age_seconds"),
        "open_interest": run_args.get("oi_max_age_seconds"),
    }
    for prefix, ts_column, age_column, fresh_column in (
        ("trade", "last_trade_state_ts_recv_ns", "trade_price_age_seconds", "trade_price_fresh"),
        ("mark", "mark_state_ts_recv_ns", "mark_age_seconds", "mark_fresh"),
        ("open_interest", "open_interest_state_ts_recv_ns", "open_interest_age_seconds", "open_interest_fresh"),
    ):
        valid = frame[ts_column].notna()
        if valid.any():
            # Match the builder's end-of-second age calculation exactly. The
            # float operation order matters at epoch-scale nanosecond values.
            receive_values = frame.loc[valid, "receive_second"].to_numpy(dtype="float64")
            state_values = frame.loc[valid, ts_column].to_numpy(dtype="float64")
            expected_age = (
                (receive_values * NANOSECONDS_PER_SECOND) - state_values
            ) / NANOSECONDS_PER_SECOND + 1.0
            expected_age = np.maximum(expected_age, 0.0)
            fail_if((expected_age < -1e-9).any(), "FUTURE_STATE", f"{prefix} state comes from the future")
            fail_if(
                not close_mask(
                    frame.loc[valid, age_column],
                    expected_age,
                    atol=1e-9,
                ).all(),
                "STATE_AGE",
                f"{prefix} age formula mismatch",
            )
        fail_if(frame.loc[~valid, age_column].notna().any(), "AGE_WITHOUT_STATE", f"{prefix} age exists without state timestamp")
        threshold = thresholds[prefix]
        if threshold is None:
            add_issue(issues, "WARN", "MISSING_BUILD_ARGUMENTS", "state", f"Could not recover {prefix} freshness threshold from build_runs", receive_date=day_text)
        else:
            expected_fresh = valid & frame[age_column].le(float(threshold))
            fail_if(not frame[fresh_column].eq(expected_fresh).all(), "FRESHNESS_FLAG", f"{prefix} freshness flag is inconsistent")
    fail_if(not frame["core_context_ready"].eq(frame["trade_price_fresh"] & frame["mark_fresh"] & frame["open_interest_fresh"]).all(), "CORE_READY_FLAG", "core_context_ready is inconsistent")

    funding_valid = frame["next_funding_time_ms"].notna()
    if funding_valid.any():
        expected = frame.loc[funding_valid, "next_funding_time_ms"].astype("float64") / 1000.0 - frame.loc[funding_valid, "available_after_second"].astype("float64")
        fail_if(not close_mask(frame.loc[funding_valid, "seconds_to_next_funding"], expected, atol=1e-9).all(), "FUNDING_COUNTDOWN", "seconds_to_next_funding formula mismatch")

    # Reconcile independently against current canonical inputs.
    canonical_symbol_counts: dict[str, int] = {}
    canonical_all_counts: dict[str, int] = {}
    for family in FAMILIES:
        all_count = 0
        symbol_count = 0
        try:
            for path in canonical_files(canonical_root, family, day_text):
                all_rows, matching_rows = count_symbol_rows(path, symbol)
                all_count += all_rows
                symbol_count += matching_rows
        except Exception as exc:
            add_issue(issues, "FAIL", "CANONICAL_COUNT_FAILED", "canonical", str(exc), receive_date=day_text, family=family)
        canonical_all_counts[family] = all_count
        canonical_symbol_counts[family] = symbol_count

    output_raw = {
        "trades": int(frame["trade_raw_record_count"].sum()),
        "mark_price": int(frame["mark_raw_record_count"].sum()),
        "open_interest": int(frame["open_interest_raw_record_count"].sum()),
        "liquidations": int(frame["liquidation_raw_record_count"].sum()),
    }
    output_unique = {
        "trades": int(frame["trade_count"].sum()),
        "mark_price": int(frame["mark_update_count"].sum()),
        "open_interest": int(frame["open_interest_update_count"].sum()),
        "liquidations": int(frame["liquidation_event_count"].sum()),
    }
    state_summary = load_json(state["summary_json"], {})
    state_stats = state_summary.get("stats", {})
    for family in FAMILIES:
        canonical_symbol = canonical_symbol_counts.get(family, 0)
        raw = output_raw[family]
        unique = output_unique[family]
        duplicate_count = raw - unique
        expected_duplicates = int((state_stats.get("duplicates_removed") or {}).get(family, 0))
        expected_symbol_rows = int((state_stats.get("symbol_rows") or {}).get(family, raw))
        if raw != canonical_symbol:
            add_issue(
                issues,
                "FAIL",
                "CANONICAL_RAW_RECONCILIATION",
                "canonical",
                f"Output raw count={raw:,}; canonical {symbol} rows={canonical_symbol:,}",
                receive_date=day_text,
                family=family,
            )
        if raw != expected_symbol_rows:
            add_issue(
                issues,
                "FAIL",
                "STATE_RAW_RECONCILIATION",
                "state",
                f"Output raw count={raw:,}; state symbol rows={expected_symbol_rows:,}",
                receive_date=day_text,
                family=family,
            )
        if duplicate_count != expected_duplicates:
            add_issue(
                issues,
                "FAIL",
                "DEDUP_RECONCILIATION",
                "state",
                f"Derived raw-unique={duplicate_count:,}; state duplicates_removed={expected_duplicates:,}",
                receive_date=day_text,
                family=family,
            )
        reconciliations.append(
            {
                "receive_date": day_text,
                "family": family,
                "canonical_all_rows": canonical_all_counts.get(family, 0),
                "canonical_symbol_rows": canonical_symbol,
                "output_raw_rows": raw,
                "output_unique_rows": unique,
                "duplicates_removed": duplicate_count,
                "raw_reconciles": raw == canonical_symbol,
            }
        )

    summary.update(
        {
            "input_records": int(frame["input_record_count"].sum()),
            "trade_raw_rows": output_raw["trades"],
            "trade_unique_rows": output_unique["trades"],
            "mark_raw_rows": output_raw["mark_price"],
            "mark_unique_rows": output_unique["mark_price"],
            "oi_raw_rows": output_raw["open_interest"],
            "oi_unique_rows": output_unique["open_interest"],
            "liquidation_raw_rows": output_raw["liquidations"],
            "liquidation_unique_rows": output_unique["liquidations"],
            "trade_seconds": int(frame["trade_observed"].sum()),
            "mark_update_seconds": int(frame["mark_updated"].sum()),
            "oi_update_seconds": int(frame["open_interest_updated"].sum()),
            "liquidation_seconds": int(frame["liquidation_observed"].sum()),
            "core_ready_rows": int(frame["core_context_ready"].sum()),
            "core_ready_pct": float(frame["core_context_ready"].mean() * 100.0),
            "issue_count": sum(1 for issue in issues if issue.receive_date == day_text),
        }
    )
    return summary, gaps, reconciliations


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def build_invariants(issues: Sequence[Issue], summaries: Sequence[Mapping[str, Any]]) -> list[Invariant]:
    groups = [
        ("Build state and output ownership are complete", {"MISSING_BUILD_STATE", "DATE_NOT_COMPLETED", "OUTPUT_FILE_COUNT", "REGISTERED_OUTPUT_MISSING", "EXTRA_OUTPUT_FILE"}),
        ("Output schemas and metadata match core schema v1", {"OUTPUT_SCHEMA_MISMATCH", "OUTPUT_METADATA_MISMATCH", "ROW_SCHEMA_VERSION"}),
        ("Physical row counts and date boundaries match build state", {"STATE_ROW_COUNT_MISMATCH", "STATE_BOUNDARY_MISMATCH", "EMPTY_OUTPUT"}),
        ("Receive-second grids are unique, ordered and gap-free within each span", {"DUPLICATE_SECONDS", "SECONDS_NOT_STRICT", "INTERNAL_SECOND_GAPS"}),
        ("UTC partition and availability timestamps are causal", {"AVAILABILITY_SECOND", "DAY_OFFSET", "PARTITION_DATE", "TIMESTAMP_UTC", "AVAILABLE_UTC", "FUTURE_STATE", "STATE_AGE", "AGE_WITHOUT_STATE"}),
        ("Trade measurements reconcile and satisfy OHLC invariants", {"TRADE_SIDE_COUNT", "TRADE_OBSERVED_FLAG", "TRADE_BASE_RECONCILIATION", "TRADE_QUOTE_RECONCILIATION", "SIGNED_BASE_FLOW", "SIGNED_QUOTE_FLOW", "TRADE_PRICE_NULL", "TRADE_LOW_HIGH", "TRADE_HIGH_OHLC", "TRADE_LOW_OHLC", "VWAP_RANGE", "VWAP_FORMULA", "MEAN_TRADE_BASE", "MEAN_TRADE_NOTIONAL", "NONTRADE_COUNT", "NONTRADE_VOLUME"}),
        ("Mark, basis, funding and freshness fields are internally consistent", {"MARK_UPDATED_FLAG", "BASIS_FORMULA", "BASIS_BPS_FORMULA", "FUNDING_COUNTDOWN", "FRESHNESS_FLAG"}),
        ("Open-interest measurements are internally consistent", {"OI_UPDATED_FLAG", "OI_NOTIONAL_FORMULA", "OI_CHANGE_WITHOUT_UPDATE", "OI_CHANGE_PCT"}),
        ("Liquidation measurements reconcile by side", {"LIQ_OBSERVED_FLAG", "LIQ_COUNT_RECONCILIATION", "LIQ_BASE_RECONCILIATION", "LIQ_NOTIONAL_RECONCILIATION", "LIQ_NET_FORMULA"}),
        ("Derived raw counts reconcile to current canonical inputs", {"CANONICAL_INPUT_CHANGED", "CANONICAL_COUNT_FAILED", "CANONICAL_RAW_RECONCILIATION", "STATE_RAW_RECONCILIATION", "DEDUP_RECONCILIATION"}),
        ("Numeric measurements contain no negatives where prohibited or infinities", {"COUNT_NULL", "COUNT_NEGATIVE", "COUNT_NONINTEGER", "NEGATIVE_MEASUREMENT", "NONFINITE_NUMERIC"}),
        ("Core readiness and event flags are internally consistent", {"INPUT_COUNT_RECONCILIATION", "ANY_EVENT_FLAG", "CORE_READY_FLAG", "SYMBOL_MISMATCH"}),
    ]
    result: list[Invariant] = []
    for name, codes in groups:
        relevant = [issue for issue in issues if issue.code in codes and issue.severity == "FAIL"]
        result.append(
            Invariant(
                name=name,
                status="PASS" if not relevant else "FAIL",
                issue_count=len(relevant),
                details="No failures detected" if not relevant else "; ".join(Counter(issue.code for issue in relevant).keys()),
            )
        )
    boundary_warnings = [issue for issue in issues if issue.code in {"LEADING_BOUNDARY_GAP", "TRAILING_BOUNDARY_GAP"}]
    result.append(
        Invariant(
            name="Date-boundary coverage is documented",
            status="PASS" if not boundary_warnings else "WARN",
            issue_count=len(boundary_warnings),
            details=(
                "Every audited date spans the full UTC day"
                if not boundary_warnings
                else f"{len(boundary_warnings)} boundary gaps recorded; inspect perps_core_gap_seconds.csv"
            ),
        )
    )
    return result


def write_findings(
    path: Path,
    *,
    status: str,
    symbol: str,
    summaries: Sequence[Mapping[str, Any]],
    issues: Sequence[Issue],
    invariants: Sequence[Invariant],
    total_rows: int,
) -> None:
    fail_count = sum(1 for issue in issues if issue.severity == "FAIL")
    warn_count = sum(1 for issue in issues if issue.severity == "WARN")
    info_count = sum(1 for issue in issues if issue.severity == "INFO")
    full_days = sum(bool(row.get("full_day_rows")) for row in summaries)
    boundary_dates = [row for row in summaries if int(row.get("leading_missing_seconds") or 0) or int(row.get("trailing_missing_seconds") or 0)]
    lines = [
        "# Perps Core 1-Second Audit",
        "",
        f"- **Status:** {status}",
        f"- **Auditor version:** {AUDITOR_VERSION}",
        f"- **Symbol:** {symbol}",
        f"- **Dates audited:** {len(summaries):,}",
        f"- **Physical output rows:** {total_rows:,}",
        f"- **Full 86,400-row dates:** {full_days:,}",
        f"- **Dates with boundary gaps:** {len(boundary_dates):,}",
        f"- **Failure issues:** {fail_count:,}",
        f"- **Warning issues:** {warn_count:,}",
        f"- **Informational issues:** {info_count:,}",
        "",
        "## Interpretation",
        "",
    ]
    if status == "PASS":
        lines.append("The perps core one-second layer passed all audited integrity, reconciliation, formula, causality, and schema checks.")
    elif status == "WARN":
        lines.append("No integrity failures were found, but one or more coverage or operational warnings require review before treating every UTC date as fully continuous.")
    else:
        lines.append("One or more integrity failures were found. Do not build downstream rolling features until the failures are resolved and this audit passes.")
    lines.extend(["", "## Invariants", ""])
    for invariant in invariants:
        lines.append(f"- **{invariant.status}** — {invariant.name}: {invariant.details}")
    if boundary_dates:
        lines.extend(["", "## Boundary coverage", ""])
        for row in boundary_dates:
            lines.append(
                f"- {row['receive_date']}: leading={int(row.get('leading_missing_seconds') or 0):,}s, "
                f"trailing={int(row.get('trailing_missing_seconds') or 0):,}s, "
                f"rows={int(row.get('physical_rows') or 0):,}"
            )
    top_issues = Counter((issue.severity, issue.code) for issue in issues)
    if top_issues:
        lines.extend(["", "## Issue summary", ""])
        for (severity, code), count in sorted(top_issues.items(), key=lambda item: (-SEVERITY_ORDER[item[0][0]], -item[1], item[0][1])):
            lines.append(f"- {severity} `{code}`: {count:,}")
    lines.extend(
        [
            "",
            "## Report files",
            "",
            "- `perps_core_audit_summary.json`",
            "- `perps_core_audit_invariants.csv`",
            "- `perps_core_audit_issues.csv`",
            "- `perps_core_date_summary.csv`",
            "- `perps_core_gap_seconds.csv`",
            "- `perps_core_family_reconciliation.csv`",
            "- `perps_core_output_files.csv`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI and orchestration
# ---------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Audit the perps core one-second derived layer.")
    value.add_argument("--canonical-root", default=DEFAULT_CANONICAL_ROOT)
    value.add_argument("--core-root", default=DEFAULT_CORE_ROOT)
    value.add_argument("--reports-root", default=DEFAULT_REPORT_ROOT)
    value.add_argument("--symbol", default="BTCUSDT")
    value.add_argument("--start-date", help="Inclusive UTC date YYYY-MM-DD")
    value.add_argument("--end-date", help="Inclusive UTC date YYYY-MM-DD")
    value.add_argument("--fail-on-warnings", action="store_true")
    value.add_argument("--progress-every", type=int, default=1, help="Print progress every N dates")
    return value


def main() -> int:
    args = parser().parse_args()
    canonical_root = Path(args.canonical_root).expanduser().resolve()
    core_root = Path(args.core_root).expanduser().resolve()
    reports_root = Path(args.reports_root).expanduser().resolve()
    symbol = args.symbol.strip().upper()
    symbol_partition = safe_symbol(symbol)
    start_date = parse_date(args.start_date, "--start-date")
    end_date = parse_date(args.end_date, "--end-date")
    if start_date and end_date and start_date > end_date:
        raise SystemExit("--start-date must be on or before --end-date")
    if args.progress_every <= 0:
        raise SystemExit("--progress-every must be greater than zero")
    if not canonical_root.exists():
        raise SystemExit(f"Canonical root does not exist: {canonical_root}")
    if not core_root.exists():
        raise SystemExit(f"Core root does not exist: {core_root}")
    state_db = core_root / "_build_state" / "build_state.sqlite"
    if not state_db.is_file():
        raise SystemExit(f"Build state database not found: {state_db}")

    identifier = run_id()
    run_reports = reports_root / "runs" / identifier
    run_reports.mkdir(parents=True, exist_ok=True)
    issues: list[Issue] = []
    summaries: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    reconciliations: list[dict[str, Any]] = []

    connection = sqlite3.connect(state_db)
    connection.row_factory = sqlite3.Row
    states = read_build_rows(connection, symbol)
    state_by_date = {str(row["receive_date"]): row for row in states}
    output_by_date = discover_dates(core_root, symbol_partition)
    canonical_date_texts = discover_canonical_dates(canonical_root)
    global_date_texts = sorted(set(state_by_date) | set(output_by_date) | canonical_date_texts)
    all_date_texts = list(global_date_texts)
    if start_date:
        all_date_texts = [value for value in all_date_texts if datetime.strptime(value, "%Y-%m-%d").date() >= start_date]
    if end_date:
        all_date_texts = [value for value in all_date_texts if datetime.strptime(value, "%Y-%m-%d").date() <= end_date]
    if not all_date_texts:
        raise SystemExit("No perps core dates found in the requested range")

    print("Auditing perps core one-second layer...")
    print(f"Run ID          : {identifier}")
    print(f"Canonical root  : {canonical_root}")
    print(f"Core root       : {core_root}")
    print(f"State database  : {state_db}")
    print(f"Reports root    : {run_reports}")
    print(f"Symbol          : {symbol}")
    print(f"Date range      : {all_date_texts[0]} to {all_date_texts[-1]} ({len(all_date_texts):,} dates)")

    try:
        previous_state: sqlite3.Row | None = None
        for position, day_text in enumerate(all_date_texts, start=1):
            state = state_by_date.get(day_text)
            if previous_state is not None and state is not None:
                previous_summary = load_json(previous_state["summary_json"], {})
                expected_carry = previous_summary.get("final_carry_signature")
                if expected_carry and state["carry_in_signature"] != expected_carry:
                    add_issue(
                        issues,
                        "FAIL",
                        "CARRY_CHAIN_MISMATCH",
                        "state",
                        f"carry_in_signature does not match prior date final carry ({expected_carry})",
                        receive_date=day_text,
                    )
            summary, date_gaps, date_reconciliation = audit_date(
                day_text=day_text,
                state=state,
                output_paths=output_by_date.get(day_text, []),
                core_root=core_root,
                canonical_root=canonical_root,
                symbol=symbol,
                connection=connection,
                issues=issues,
                dataset_first_date=global_date_texts[0],
                dataset_last_date=global_date_texts[-1],
            )
            summaries.append(summary)
            gaps.extend(date_gaps)
            reconciliations.extend(date_reconciliation)
            if state is not None:
                previous_state = state
            if position % args.progress_every == 0 or position == len(all_date_texts):
                date_issues = [issue for issue in issues if issue.receive_date == day_text]
                fail_count = sum(issue.severity == "FAIL" for issue in date_issues)
                warn_count = sum(issue.severity == "WARN" for issue in date_issues)
                print(
                    f"AUDITED   [{position:>3}/{len(all_date_texts)}] {day_text}: "
                    f"rows={int(summary.get('physical_rows') or 0):,}, fails={fail_count}, warns={warn_count}"
                )
    except KeyboardInterrupt:
        print("\nAudit interrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        traceback.print_exc()
        add_issue(issues, "FAIL", "AUDITOR_EXCEPTION", "run", f"{type(exc).__name__}: {exc}")

    invariants = build_invariants(issues, summaries)
    fail_count = sum(issue.severity == "FAIL" for issue in issues)
    warn_count = sum(issue.severity == "WARN" for issue in issues)
    status = "FAIL" if fail_count else ("WARN" if warn_count else "PASS")
    total_rows = sum(int(row.get("physical_rows") or 0) for row in summaries)
    total_input = sum(int(row.get("input_records") or 0) for row in summaries)
    full_days = sum(bool(row.get("full_day_rows")) for row in summaries)

    summary_payload = {
        "auditor_version": AUDITOR_VERSION,
        "run_id": identifier,
        "started_or_completed_at_utc": utc_now_iso(),
        "status": status,
        "symbol": symbol,
        "canonical_root": str(canonical_root),
        "core_root": str(core_root),
        "state_database": str(state_db),
        "dates_audited": len(summaries),
        "physical_output_rows": total_rows,
        "input_records_reconciled": total_input,
        "full_day_dates": full_days,
        "boundary_gap_dates": sum(bool(int(row.get("leading_missing_seconds") or 0) or int(row.get("trailing_missing_seconds") or 0)) for row in summaries),
        "failure_issues": fail_count,
        "warning_issues": warn_count,
        "informational_issues": sum(issue.severity == "INFO" for issue in issues),
        "invariants": [item.as_dict() for item in invariants],
    }

    output_files = []
    for day_text, paths in output_by_date.items():
        for path in paths:
            output_files.append(
                {
                    "receive_date": day_text,
                    "relative_path": path.relative_to(core_root).as_posix(),
                    "size_bytes": int(path.stat().st_size),
                    "rows": parquet_rows(path),
                }
            )

    files = {
        "perps_core_audit_summary.json": compact_json(summary_payload) + "\n",
    }
    (run_reports / "perps_core_audit_summary.json").write_text(files["perps_core_audit_summary.json"], encoding="utf-8")
    write_csv(run_reports / "perps_core_audit_invariants.csv", [item.as_dict() for item in invariants])
    write_csv(run_reports / "perps_core_audit_issues.csv", [item.as_dict() for item in issues])
    write_csv(run_reports / "perps_core_date_summary.csv", summaries)
    write_csv(run_reports / "perps_core_gap_seconds.csv", gaps)
    write_csv(run_reports / "perps_core_family_reconciliation.csv", reconciliations)
    write_csv(run_reports / "perps_core_output_files.csv", output_files)
    write_findings(
        run_reports / "perps_core_audit_findings.md",
        status=status,
        symbol=symbol,
        summaries=summaries,
        issues=issues,
        invariants=invariants,
        total_rows=total_rows,
    )

    # Copy latest reports to the stable report root.
    reports_root.mkdir(parents=True, exist_ok=True)
    for path in run_reports.iterdir():
        if path.is_file():
            target = reports_root / path.name
            target.write_bytes(path.read_bytes())

    print("\nPerps core audit finished.")
    print(f"Status           : {status}")
    print(f"Dates audited    : {len(summaries):,}")
    print(f"Output rows      : {total_rows:,}")
    print(f"Input records    : {total_input:,}")
    print(f"Full-day dates   : {full_days:,}")
    print(f"Failure issues   : {fail_count:,}")
    print(f"Warning issues   : {warn_count:,}")
    print(f"Findings         : {run_reports / 'perps_core_audit_findings.md'}")
    print(f"Latest reports   : {reports_root}")

    if fail_count:
        return 1
    if args.fail_on_warnings and warn_count:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
