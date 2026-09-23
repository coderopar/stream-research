#!/usr/bin/env python
"""
Profile causal perps flow-feature behaviour before distinct 50-bps expansion events.

Research question
-----------------
Do audited trade-flow, open-interest, liquidation, volatility and price-impact
features behave differently before a distinct 15-minute / 50-bps expansion event
than they do before chronologically matched ordinary-market control anchors?

This script profiles that question only. It does not train a model, select a
trading direction, modify the causal feature layer, or write future labels into
feature partitions.

Primary event contract
----------------------
The default event source is the selected ``bridge_060s`` segmentation produced
by ``03_profile_perps_expansion_event_segmentation.py``. Event onset is the
first positive decision second in an event. Positive lead values mean seconds
before that onset:

    observation_second = event_start_second - lead_seconds

The default leads are 900, 600, 300, 120, 60, 30 and 0 seconds. All feature
observations are exact existing one-second rows and must have
``flow_feature_context_ready = true``. Missing rows are never filled.

Matched controls
----------------
Control pseudo-onsets come from audited row-label rows with
``expansion_50_15m = false`` at a fixed decision cadence. A control observation
window is rejected when it intersects an event plus the configured embargo.
Controls are unique by default and are selected deterministically through these
match tiers:

1. same UTC date and same UTC hour;
2. same UTC date and adjacent UTC hour;
3. same chronological period and same UTC hour;
4. same chronological period and adjacent UTC hour;
5. any date and same UTC hour;
6. global fallback.

Within a tier, candidates are ordered by calendar distance, time-of-day distance
and epoch second. The tier and match distances are preserved in the output.

Outputs
-------
Default dataset root:
    data/derived/perps/research/pre_expansion_footprints_v1

Default report root:
    reports/perps_pre_expansion_footprints

The output contains event/control observations, the event-to-control match table,
an event catalogue, SQLite build state, CSV/JSON/Markdown profiles, and strict
invariants. All Parquet and report replacements are atomic.

Dependencies
------------
    python -m pip install numpy pandas pyarrow

Typical Git Bash usage
----------------------
    python scripts/perps/participant_behaviour/01_profile_pre_expansion_footprints.py
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import sqlite3
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:  # Optional at import time so --self-test and py_compile remain lightweight.
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - environment specific
    pa = None  # type: ignore[assignment]
    pc = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]


PROFILER_VERSION = "1.0.0"
OUTPUT_SCHEMA_VERSION = 1
EXPECTED_FEATURE_SCHEMA_VERSION = 1
EXPECTED_FEATURE_BUILDER_VERSION = "1.0.2"
EXPECTED_LABEL_SCHEMA_VERSION = 1
EXPECTED_LABEL_BUILDER_VERSION = "1.0.0"
EXPECTED_SEGMENTATION_SCHEMA_VERSION = 1
EXPECTED_SEGMENTATION_PROFILER_VERSION = "1.0.0"
DEFAULT_SEGMENTATION_ID = "bridge_060s"
DEFAULT_LEADS_SECONDS = (900, 600, 300, 120, 60, 30, 0)
DEFAULT_CONTROLS_PER_EVENT = 3
DEFAULT_CONTROL_CADENCE_SECONDS = 30
DEFAULT_CONTROL_EMBARGO_SECONDS = 900
DEFAULT_ADJACENT_HOUR_TOLERANCE = 1
SECONDS_PER_DAY = 86_400
DATE_PREFIX = "date="
SYMBOL_PREFIX = "symbol="
SEGMENTATION_PREFIX = "segmentation="

REQUIRED_FEATURE_COLUMNS = (
    "feature_schema_version",
    "symbol",
    "receive_second",
    "flow_feature_context_ready",
)

REQUIRED_LABEL_COLUMNS = (
    "symbol",
    "decision_second",
    "expansion_50_15m",
)

REQUIRED_EVENT_COLUMNS = (
    "segmentation_schema_version",
    "segmentation_profiler_version",
    "segmentation_id",
    "event_id",
    "symbol",
    "event_start_second",
    "event_end_second",
    "event_duration_seconds",
    "number_of_positive_anchors",
    "maximum_future_swing_15m_bps",
    "maximum_future_upside_15m_bps",
    "maximum_future_downside_15m_bps",
    "path_type",
)

# Metadata, availability and nonstationary price-level columns are excluded from
# the default comparison. CVD levels are read only to derive within-sample causal
# changes from the earliest requested lead.
DEFAULT_EXCLUDED_EXACT_COLUMNS = {
    "feature_schema_version",
    "core_schema_version",
    "symbol",
    "receive_second",
    "timestamp_utc",
    "available_after_second",
    "available_after_utc",
    "seconds_from_day_start",
    "core_context_ready",
    "flow_feature_context_ready",
    "price_for_features",
    "price_source_code",
    "price_fresh",
    "open_interest",
    "cvd_quote",
    "cvd_base",
}

DEFAULT_EXCLUDED_PREFIXES = (
    "window_observation_count_",
    "window_complete_",
    "return_lag_available_",
    "log_return_",
)

FUTURE_OR_LABEL_TOKENS = (
    "future_",
    "target_",
    "label_",
    "expansion_",
    "episode_",
    "event_id",
    "path_end",
    "realization",
)

REPORT_FILENAMES = (
    "perps_pre_expansion_feature_comparison.csv",
    "perps_pre_expansion_matched_effects.csv",
    "perps_pre_expansion_trajectory.csv",
    "perps_pre_expansion_chronological_effects.csv",
    "perps_pre_expansion_stability.csv",
    "perps_pre_expansion_top_features.csv",
    "perps_pre_expansion_control_match_quality.csv",
    "perps_pre_expansion_coverage.csv",
    "perps_pre_expansion_invariants.csv",
    "perps_pre_expansion_summary.json",
    "perps_pre_expansion_config.json",
    "perps_pre_expansion_findings.md",
)


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------


def require_arrow() -> tuple[Any, Any, Any]:
    if pa is None or pc is None or pq is None:
        raise SystemExit(
            "Missing dependency 'pyarrow'. Activate the project virtual environment "
            "and run: python -m pip install numpy pandas pyarrow"
        )
    return pa, pc, pq


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat().replace("+00:00", "Z")


def parse_date(value: str | None, name: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise SystemExit(f"{name} must use YYYY-MM-DD format: {value!r}") from exc


def parse_positive_ints(value: str, name: str, allow_zero: bool = False) -> tuple[int, ...]:
    try:
        values = tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}, reverse=True))
    except ValueError as exc:
        raise SystemExit(f"{name} must be a comma-separated list of integers") from exc
    if not values:
        raise SystemExit(f"{name} must contain at least one value")
    lower_bound = 0 if allow_zero else 1
    if any(item < lower_bound for item in values):
        raise SystemExit(f"{name} values must be >= {lower_bound}")
    return values


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


def replace_with_retry(
    source: Path,
    destination: Path,
    attempts: int = 20,
    initial_delay_seconds: float = 0.05,
) -> None:
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


def unlink_with_retry(path: Path, attempts: int = 10) -> None:
    for attempt in range(1, attempts + 1):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt >= attempts:
                return
            gc.collect()
            time.sleep(min(0.05 * attempt, 0.5))


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        replace_with_retry(temporary, path)
    finally:
        unlink_with_retry(temporary)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        replace_with_retry(temporary, path)
    finally:
        unlink_with_retry(temporary)


def metadata_bytes(values: Mapping[str, Any]) -> dict[bytes, bytes]:
    return {
        str(key).encode("utf-8"): str(value).encode("utf-8")
        for key, value in values.items()
        if value is not None
    }


def decode_metadata(schema: Any) -> dict[str, str]:
    return {
        key.decode("utf-8", errors="replace"): value.decode("utf-8", errors="replace")
        for key, value in (schema.metadata or {}).items()
    }


def atomic_write_parquet(
    path: Path,
    frame: pd.DataFrame,
    metadata: Mapping[str, Any],
    compression_level: int,
) -> None:
    pa_module, _, pq_module = require_arrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    table = pa_module.Table.from_pandas(frame, preserve_index=False)
    table = table.replace_schema_metadata(metadata_bytes(metadata))
    try:
        pq_module.write_table(
            table,
            temporary,
            compression="zstd",
            compression_level=compression_level,
            use_dictionary=True,
            write_statistics=True,
        )
        check = pq_module.ParquetFile(temporary)
        try:
            if int(check.metadata.num_rows) != len(frame):
                raise RuntimeError(
                    f"Atomic Parquet row mismatch for {path}: "
                    f"{check.metadata.num_rows:,} != {len(frame):,}"
                )
            written_metadata = decode_metadata(check.schema_arrow)
            if written_metadata.get("input_signature") != str(metadata.get("input_signature")):
                raise RuntimeError(f"Input-signature metadata mismatch for {path}")
        finally:
            check.close()
        del table
        gc.collect()
        replace_with_retry(temporary, path)
    finally:
        unlink_with_retry(temporary)


def percent(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return float("nan")
    return 100.0 * float(numerator) / float(denominator)


def finite_array(values: Iterable[Any]) -> np.ndarray:
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    return array[np.isfinite(array)]


def safe_stat(values: Iterable[Any], statistic: str) -> float:
    array = finite_array(values)
    if not len(array):
        return float("nan")
    if statistic == "mean":
        return float(np.mean(array))
    if statistic == "median":
        return float(np.median(array))
    if statistic == "std":
        return float(np.std(array, ddof=1)) if len(array) > 1 else float("nan")
    if statistic == "minimum":
        return float(np.min(array))
    if statistic == "maximum":
        return float(np.max(array))
    if statistic.startswith("p"):
        return float(np.quantile(array, float(statistic[1:]) / 100.0, method="linear"))
    raise ValueError(statistic)


def standardized_mean_difference(event_values: Iterable[Any], control_values: Iterable[Any]) -> float:
    event = finite_array(event_values)
    control = finite_array(control_values)
    if len(event) < 2 or len(control) < 2:
        return float("nan")
    event_var = float(np.var(event, ddof=1))
    control_var = float(np.var(control, ddof=1))
    denominator = len(event) + len(control) - 2
    pooled = ((len(event) - 1) * event_var + (len(control) - 1) * control_var) / denominator
    if pooled <= 0 or not math.isfinite(pooled):
        difference = float(np.mean(event) - np.mean(control))
        return 0.0 if math.isclose(difference, 0.0, abs_tol=1e-15) else float("nan")
    return float((np.mean(event) - np.mean(control)) / math.sqrt(pooled))


def cliffs_delta(event_values: Iterable[Any], control_values: Iterable[Any]) -> float:
    """Exact Cliff's delta in O((n+m) log m) without SciPy."""
    event = finite_array(event_values)
    control = np.sort(finite_array(control_values))
    if not len(event) or not len(control):
        return float("nan")
    less = np.searchsorted(control, event, side="left")
    greater = len(control) - np.searchsorted(control, event, side="right")
    return float((np.sum(less) - np.sum(greater)) / (len(event) * len(control)))


def paired_standardized_effect(differences: Iterable[Any]) -> float:
    values = finite_array(differences)
    if len(values) < 2:
        return float("nan")
    std = float(np.std(values, ddof=1))
    if std <= 0 or not math.isfinite(std):
        return 0.0 if math.isclose(float(np.mean(values)), 0.0, abs_tol=1e-15) else float("nan")
    return float(np.mean(values) / std)


def seconds_to_date(value: int) -> date:
    return datetime.fromtimestamp(int(value), tz=timezone.utc).date()


def seconds_to_hour(value: int) -> int:
    return datetime.fromtimestamp(int(value), tz=timezone.utc).hour


def seconds_from_day_start(value: int) -> int:
    return int(value) % SECONDS_PER_DAY


def hour_distance(left: int, right: int) -> int:
    difference = abs(int(left) - int(right))
    return min(difference, 24 - difference)


def time_of_day_distance(left_second: int, right_second: int) -> int:
    left = seconds_from_day_start(left_second)
    right = seconds_from_day_start(right_second)
    difference = abs(left - right)
    return min(difference, SECONDS_PER_DAY - difference)


def feature_family(column: str) -> str:
    if "trade_count" in column or "quote_volume" in column or "base_volume" in column:
        return "participation"
    if "signed_quote_flow" in column or "signed_base_flow" in column or "aggressive_flow" in column:
        return "aggressive_flow"
    if column.startswith("cvd_"):
        return "cvd"
    if "open_interest" in column:
        return "open_interest"
    if "liquidation" in column:
        return "liquidations"
    if "price_response" in column or "price_flow_alignment" in column:
        return "price_flow_interaction"
    if "realized_vol" in column or "return" in column:
        return "price_volatility"
    if "max_trade" in column:
        return "large_trade_proxy"
    return "other"


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParquetPartition:
    day: date
    files: tuple[Path, ...]
    rows: int
    signature: str


def file_signature(path: Path, hash_input_files: bool) -> dict[str, Any]:
    pq_module = require_arrow()[2]
    parquet_file = pq_module.ParquetFile(path)
    try:
        rows = int(parquet_file.metadata.num_rows)
        metadata = decode_metadata(parquet_file.schema_arrow)
    finally:
        parquet_file.close()
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "rows": rows,
        "input_signature": metadata.get("input_signature", ""),
        "dependency_signature": metadata.get("dependency_signature", ""),
    }
    if hash_input_files:
        result["sha256"] = sha256_file(path)
    return result


def extract_partition_day(path: Path) -> date:
    for parent in path.parents:
        if parent.name.startswith(DATE_PREFIX):
            return date.fromisoformat(parent.name[len(DATE_PREFIX) :])
    raise RuntimeError(f"Could not infer date partition from {path}")


def discover_date_partitions(
    root: Path,
    symbol: str,
    role: str,
    hash_input_files: bool,
) -> dict[date, ParquetPartition]:
    if role == "features":
        symbol_root = root / f"{SYMBOL_PREFIX}{symbol}"
    elif role == "labels":
        symbol_root = root / "row_labels" / f"{SYMBOL_PREFIX}{symbol}"
    else:
        raise ValueError(role)
    files = sorted(symbol_root.glob(f"{DATE_PREFIX}*/*.parquet"))
    if not files:
        raise RuntimeError(f"No {role} Parquet files found under {symbol_root}")
    grouped: dict[date, list[Path]] = {}
    for path in files:
        grouped.setdefault(extract_partition_day(path), []).append(path)
    output: dict[date, ParquetPartition] = {}
    for day, paths in sorted(grouped.items()):
        signatures = [file_signature(path, hash_input_files) for path in sorted(paths)]
        output[day] = ParquetPartition(
            day=day,
            files=tuple(sorted(paths)),
            rows=int(sum(int(item["rows"]) for item in signatures)),
            signature=sha256_text(compact_json(signatures)),
        )
    return output


def validate_feature_schema(partitions: Mapping[date, ParquetPartition], symbol: str) -> list[str]:
    pq_module = require_arrow()[2]
    first_file = next(iter(partitions.values())).files[0]
    parquet_file = pq_module.ParquetFile(first_file)
    try:
        names = list(parquet_file.schema_arrow.names)
        missing = sorted(set(REQUIRED_FEATURE_COLUMNS) - set(names))
        if missing:
            raise RuntimeError(f"Feature file {first_file} is missing columns: {missing}")
        metadata = decode_metadata(parquet_file.schema_arrow)
        if metadata.get("perps_flow_feature_builder_version") != EXPECTED_FEATURE_BUILDER_VERSION:
            raise RuntimeError(
                f"Feature builder version is {metadata.get('perps_flow_feature_builder_version')!r}; "
                f"expected {EXPECTED_FEATURE_BUILDER_VERSION!r}"
            )
        if metadata.get("perps_flow_feature_schema_version") != str(EXPECTED_FEATURE_SCHEMA_VERSION):
            raise RuntimeError(
                f"Feature schema version is {metadata.get('perps_flow_feature_schema_version')!r}; "
                f"expected {EXPECTED_FEATURE_SCHEMA_VERSION}"
            )
        file_symbol = metadata.get("symbol")
        if file_symbol and file_symbol.upper() != symbol:
            raise RuntimeError(f"Feature metadata symbol is {file_symbol!r}; expected {symbol}")
        return names
    finally:
        parquet_file.close()


def validate_label_schema(partitions: Mapping[date, ParquetPartition], symbol: str) -> None:
    pq_module = require_arrow()[2]
    first_file = next(iter(partitions.values())).files[0]
    parquet_file = pq_module.ParquetFile(first_file)
    try:
        names = set(parquet_file.schema_arrow.names)
        missing = sorted(set(REQUIRED_LABEL_COLUMNS) - names)
        if missing:
            raise RuntimeError(f"Label file {first_file} is missing columns: {missing}")
        metadata = decode_metadata(parquet_file.schema_arrow)
        if metadata.get("perps_expansion_label_builder_version") != EXPECTED_LABEL_BUILDER_VERSION:
            raise RuntimeError(
                f"Label builder version is {metadata.get('perps_expansion_label_builder_version')!r}; "
                f"expected {EXPECTED_LABEL_BUILDER_VERSION!r}"
            )
        if metadata.get("perps_expansion_label_schema_version") != str(EXPECTED_LABEL_SCHEMA_VERSION):
            raise RuntimeError(
                f"Label schema version is {metadata.get('perps_expansion_label_schema_version')!r}; "
                f"expected {EXPECTED_LABEL_SCHEMA_VERSION}"
            )
        file_symbol = metadata.get("symbol")
        if file_symbol and file_symbol.upper() != symbol:
            raise RuntimeError(f"Label metadata symbol is {file_symbol!r}; expected {symbol}")
    finally:
        parquet_file.close()


def event_file_path(segmentation_root: Path, segmentation_id: str, symbol: str) -> Path:
    return (
        segmentation_root
        / "events"
        / f"{SEGMENTATION_PREFIX}{segmentation_id}"
        / f"{SYMBOL_PREFIX}{symbol}"
        / "part-00000.parquet"
    )


def read_events(
    path: Path,
    symbol: str,
    segmentation_id: str,
    start_date: date | None,
    end_date: date | None,
    max_events: int | None,
) -> pd.DataFrame:
    pq_module = require_arrow()[2]
    if not path.exists():
        raise RuntimeError(f"Segmentation event file does not exist: {path}")
    parquet_file = pq_module.ParquetFile(path)
    try:
        names = set(parquet_file.schema_arrow.names)
        missing = sorted(set(REQUIRED_EVENT_COLUMNS) - names)
        if missing:
            raise RuntimeError(f"Event file {path} is missing columns: {missing}")
        metadata = decode_metadata(parquet_file.schema_arrow)
        if metadata.get("segmentation_profiler_version") != EXPECTED_SEGMENTATION_PROFILER_VERSION:
            raise RuntimeError(
                f"Segmentation profiler version is {metadata.get('segmentation_profiler_version')!r}; "
                f"expected {EXPECTED_SEGMENTATION_PROFILER_VERSION!r}"
            )
        if metadata.get("segmentation_schema_version") != str(EXPECTED_SEGMENTATION_SCHEMA_VERSION):
            raise RuntimeError(
                f"Segmentation schema version is {metadata.get('segmentation_schema_version')!r}; "
                f"expected {EXPECTED_SEGMENTATION_SCHEMA_VERSION}"
            )
        frame = parquet_file.read(columns=list(REQUIRED_EVENT_COLUMNS)).to_pandas()
    finally:
        parquet_file.close()
    if frame.empty:
        raise RuntimeError("Selected event file is empty")
    frame["event_start_second"] = pd.to_numeric(frame["event_start_second"], errors="raise").astype(np.int64)
    frame["event_end_second"] = pd.to_numeric(frame["event_end_second"], errors="raise").astype(np.int64)
    frame = frame[
        frame["symbol"].astype(str).str.upper().eq(symbol)
        & frame["segmentation_id"].astype(str).eq(segmentation_id)
    ].copy()
    frame["event_start_date"] = frame["event_start_second"].map(seconds_to_date)
    if start_date is not None:
        frame = frame[frame["event_start_date"] >= start_date]
    if end_date is not None:
        frame = frame[frame["event_start_date"] <= end_date]
    frame = frame.sort_values("event_start_second", kind="mergesort").reset_index(drop=True)
    if max_events is not None:
        if max_events <= 0:
            raise SystemExit("--max-events must be positive")
        frame = frame.iloc[:max_events].copy()
    if frame.empty:
        raise RuntimeError("No events remain after filtering")
    if frame["event_id"].duplicated().any() or frame["event_start_second"].duplicated().any():
        raise RuntimeError("Event IDs and event-start seconds must be unique")
    return frame


def auto_feature_columns(schema_names: Sequence[str]) -> list[str]:
    columns: list[str] = []
    for column in schema_names:
        lowered = column.lower()
        if column in DEFAULT_EXCLUDED_EXACT_COLUMNS:
            continue
        if any(column.startswith(prefix) for prefix in DEFAULT_EXCLUDED_PREFIXES):
            continue
        if any(token in lowered for token in FUTURE_OR_LABEL_TOKENS):
            continue
        # Retain causal numeric feature groups only. Arrow type validation occurs later.
        include = (
            column == "abs_return_1s_bps"
            or column.startswith("return_") and column.endswith("_bps")
            or column.startswith("realized_vol_")
            or column.startswith("trade_count_")
            or column.startswith("quote_volume_")
            or column.startswith("base_volume_")
            or column.startswith("signed_quote_flow_")
            or column.startswith("signed_base_flow_")
            or column.startswith("aggressive_flow_imbalance_")
            or column.startswith("max_trade_notional_")
            or column.startswith("max_trade_share_")
            or column.startswith("observed_open_interest_change_")
            or column.startswith("open_interest_change_")
            or column.startswith("liquidation_event_count_")
            or column.startswith("liquidation_notional_")
            or column.startswith("net_liquidation_notional_")
            or column.startswith("liquidation_imbalance_")
            or column.startswith("liquidation_to_quote_ratio_")
            or column.startswith("max_liquidation_notional_")
            or column.startswith("price_flow_alignment_")
            or column.startswith("price_response_bps_per_1m_signed_flow_")
            or "_z_" in column
        )
        if include:
            columns.append(column)
    return sorted(set(columns))


def validate_numeric_feature_columns(first_file: Path, columns: Sequence[str]) -> None:
    pa_module, _, pq_module = require_arrow()
    parquet_file = pq_module.ParquetFile(first_file)
    try:
        schema = parquet_file.schema_arrow
        missing = sorted(set(columns) - set(schema.names))
        if missing:
            raise RuntimeError(f"Requested feature columns are missing: {missing}")
        non_numeric = []
        for column in columns:
            dtype = schema.field(column).type
            if not (pa_module.types.is_integer(dtype) or pa_module.types.is_floating(dtype)):
                non_numeric.append(f"{column}:{dtype}")
        if non_numeric:
            raise RuntimeError("Requested feature columns are not numeric: " + ", ".join(non_numeric))
    finally:
        parquet_file.close()


# ---------------------------------------------------------------------------
# Chronological periods and exact-ready seconds
# ---------------------------------------------------------------------------


def assign_chronological_periods(days: Sequence[date]) -> dict[date, str]:
    unique = sorted(set(days))
    if not unique:
        return {}
    chunks = np.array_split(np.asarray(unique, dtype=object), 3)
    labels = ("EARLY", "MIDDLE", "LATE")
    mapping: dict[date, str] = {}
    for label, chunk in zip(labels, chunks):
        for value in chunk.tolist():
            mapping[value] = label
    return mapping


def read_ready_seconds(partitions: Mapping[date, ParquetPartition]) -> np.ndarray:
    pq_module = require_arrow()[2]
    arrays: list[np.ndarray] = []
    for day, partition in sorted(partitions.items()):
        for path in partition.files:
            parquet_file = pq_module.ParquetFile(path)
            try:
                table = parquet_file.read(columns=["receive_second", "flow_feature_context_ready"])
            finally:
                parquet_file.close()
            seconds = table.column("receive_second").to_numpy(zero_copy_only=False).astype(np.int64)
            ready = table.column("flow_feature_context_ready").to_numpy(zero_copy_only=False).astype(bool)
            arrays.append(seconds[ready])
    values = np.concatenate(arrays) if arrays else np.array([], dtype=np.int64)
    if len(values) and np.any(np.diff(values) <= 0):
        raise RuntimeError("Ready feature seconds are not globally strictly increasing")
    return values


def exact_seconds_present(sorted_seconds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if not len(targets):
        return np.array([], dtype=bool)
    positions = np.searchsorted(sorted_seconds, targets, side="left")
    safe = np.minimum(positions, max(len(sorted_seconds) - 1, 0))
    present = positions < len(sorted_seconds)
    if len(sorted_seconds):
        present &= sorted_seconds[safe] == targets
    return present


def valid_anchor_mask(anchors: np.ndarray, leads: Sequence[int], ready_seconds: np.ndarray) -> np.ndarray:
    if not len(anchors):
        return np.array([], dtype=bool)
    matrix = anchors[:, None] - np.asarray(leads, dtype=np.int64)[None, :]
    present = exact_seconds_present(ready_seconds, matrix.reshape(-1)).reshape(matrix.shape)
    return np.all(present, axis=1)


# ---------------------------------------------------------------------------
# Controls and deterministic matching
# ---------------------------------------------------------------------------


def merge_intervals(starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not len(starts):
        return starts.astype(np.int64), ends.astype(np.int64)
    order = np.argsort(starts, kind="mergesort")
    sorted_starts = starts[order].astype(np.int64)
    sorted_ends = ends[order].astype(np.int64)
    merged_starts = [int(sorted_starts[0])]
    merged_ends = [int(sorted_ends[0])]
    for start, end in zip(sorted_starts[1:], sorted_ends[1:]):
        start_i = int(start)
        end_i = int(end)
        if start_i <= merged_ends[-1] + 1:
            merged_ends[-1] = max(merged_ends[-1], end_i)
        else:
            merged_starts.append(start_i)
            merged_ends.append(end_i)
    return np.asarray(merged_starts, dtype=np.int64), np.asarray(merged_ends, dtype=np.int64)


def interval_membership(values: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    if not len(starts):
        return np.zeros(len(values), dtype=bool)
    positions = np.searchsorted(starts, values, side="right") - 1
    valid = positions >= 0
    safe = np.maximum(positions, 0)
    return valid & (values <= ends[safe])


def exclusion_intervals(
    events: pd.DataFrame,
    max_lead_seconds: int,
    embargo_seconds: int,
) -> tuple[np.ndarray, np.ndarray]:
    # A control window [anchor-max_lead, anchor] must not intersect an expanded
    # event interval [event_start-embargo, event_end+embargo]. This is equivalent
    # to excluding anchors in [event_start-embargo, event_end+embargo+max_lead].
    starts = events["event_start_second"].to_numpy(dtype=np.int64) - int(embargo_seconds)
    ends = (
        events["event_end_second"].to_numpy(dtype=np.int64)
        + int(embargo_seconds)
        + int(max_lead_seconds)
    )
    return merge_intervals(starts, ends)


def load_control_candidates(
    label_partitions: Mapping[date, ParquetPartition],
    cadence_seconds: int,
    ready_seconds: np.ndarray,
    leads: Sequence[int],
    exclusion_starts: np.ndarray,
    exclusion_ends: np.ndarray,
    period_by_day: Mapping[date, str],
) -> pd.DataFrame:
    pq_module = require_arrow()[2]
    frames: list[pd.DataFrame] = []
    for day, partition in sorted(label_partitions.items()):
        day_frames: list[pd.DataFrame] = []
        for path in partition.files:
            parquet_file = pq_module.ParquetFile(path)
            try:
                frame = parquet_file.read(columns=list(REQUIRED_LABEL_COLUMNS)).to_pandas()
            finally:
                parquet_file.close()
            day_frames.append(frame)
        frame = pd.concat(day_frames, ignore_index=True) if len(day_frames) > 1 else day_frames[0]
        seconds = pd.to_numeric(frame["decision_second"], errors="raise").to_numpy(dtype=np.int64)
        expansion = frame["expansion_50_15m"].fillna(False).to_numpy(dtype=bool)
        keep = (~expansion) & ((seconds % int(cadence_seconds)) == 0)
        if np.any(keep):
            selected = seconds[keep]
            outside = ~interval_membership(selected, exclusion_starts, exclusion_ends)
            selected = selected[outside]
            if len(selected):
                valid = valid_anchor_mask(selected, leads, ready_seconds)
                selected = selected[valid]
            if len(selected):
                frames.append(
                    pd.DataFrame(
                        {
                            "control_second": selected,
                            "control_date": [seconds_to_date(value) for value in selected],
                            "control_hour_utc": [seconds_to_hour(value) for value in selected],
                            "control_period": [period_by_day.get(seconds_to_date(value), "UNKNOWN") for value in selected],
                        }
                    )
                )
    if not frames:
        raise RuntimeError("No eligible negative control anchors were found")
    output = pd.concat(frames, ignore_index=True)
    output = output.sort_values("control_second", kind="mergesort").reset_index(drop=True)
    if output["control_second"].duplicated().any():
        raise RuntimeError("Control candidate seconds are not unique")
    return output


def tier_for_match(
    event_day: date,
    event_hour: int,
    event_period: str,
    control_day: date,
    control_hour: int,
    control_period: str,
    adjacent_hour_tolerance: int,
) -> tuple[int, str]:
    hdist = hour_distance(event_hour, control_hour)
    if control_day == event_day and hdist == 0:
        return 0, "SAME_DATE_SAME_HOUR"
    if control_day == event_day and hdist <= adjacent_hour_tolerance:
        return 1, "SAME_DATE_ADJACENT_HOUR"
    if control_period == event_period and hdist == 0:
        return 2, "SAME_PERIOD_SAME_HOUR"
    if control_period == event_period and hdist <= adjacent_hour_tolerance:
        return 3, "SAME_PERIOD_ADJACENT_HOUR"
    if hdist == 0:
        return 4, "GLOBAL_SAME_HOUR"
    return 5, "GLOBAL_FALLBACK"


def match_controls(
    events: pd.DataFrame,
    candidates: pd.DataFrame,
    controls_per_event: int,
    adjacent_hour_tolerance: int,
    allow_partial_controls: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match unique controls without scanning the full candidate table per event."""
    candidate_records = candidates.reset_index(drop=True).to_dict("records")
    all_indices = np.arange(len(candidate_records), dtype=np.int64)

    by_date_hour: dict[tuple[date, int], list[int]] = {}
    by_period_hour: dict[tuple[str, int], list[int]] = {}
    by_hour: dict[int, list[int]] = {}
    for index, candidate in enumerate(candidate_records):
        control_day = candidate["control_date"]
        control_hour = int(candidate["control_hour_utc"])
        control_period = str(candidate["control_period"])
        by_date_hour.setdefault((control_day, control_hour), []).append(index)
        by_period_hour.setdefault((control_period, control_hour), []).append(index)
        by_hour.setdefault(control_hour, []).append(index)

    used_seconds: set[int] = set()
    matches: list[dict[str, Any]] = []
    event_catalog_rows: list[dict[str, Any]] = []

    def combined_indices(groups: Iterable[Sequence[int]]) -> np.ndarray:
        values: list[int] = []
        for group in groups:
            values.extend(group)
        if not values:
            return np.array([], dtype=np.int64)
        return np.unique(np.asarray(values, dtype=np.int64))

    for event_row in events.to_dict("records"):
        event_second = int(event_row["event_start_second"])
        event_day = seconds_to_date(event_second)
        event_hour = seconds_to_hour(event_second)
        event_period = str(event_row["chronological_period"])
        adjacent_hours = [
            hour
            for hour in range(24)
            if 0 < hour_distance(event_hour, hour) <= adjacent_hour_tolerance
        ]

        tier_pools: list[tuple[int, str, np.ndarray]] = [
            (
                0,
                "SAME_DATE_SAME_HOUR",
                np.asarray(by_date_hour.get((event_day, event_hour), []), dtype=np.int64),
            ),
            (
                1,
                "SAME_DATE_ADJACENT_HOUR",
                combined_indices(
                    by_date_hour.get((event_day, hour), []) for hour in adjacent_hours
                ),
            ),
            (
                2,
                "SAME_PERIOD_SAME_HOUR",
                np.asarray(
                    by_period_hour.get((event_period, event_hour), []), dtype=np.int64
                ),
            ),
            (
                3,
                "SAME_PERIOD_ADJACENT_HOUR",
                combined_indices(
                    by_period_hour.get((event_period, hour), []) for hour in adjacent_hours
                ),
            ),
            (
                4,
                "GLOBAL_SAME_HOUR",
                np.asarray(by_hour.get(event_hour, []), dtype=np.int64),
            ),
            (5, "GLOBAL_FALLBACK", all_indices),
        ]

        selected: list[tuple[tuple[int, int, int, int], dict[str, Any], str]] = []
        selected_seconds: set[int] = set()
        for tier, tier_name, pool in tier_pools:
            if len(selected) >= controls_per_event:
                break
            scored: list[tuple[tuple[int, int, int, int], dict[str, Any], str]] = []
            for index in pool.tolist():
                candidate = candidate_records[int(index)]
                second = int(candidate["control_second"])
                if second in used_seconds or second in selected_seconds:
                    continue
                control_day = candidate["control_date"]
                # Earlier tiers are intentionally allowed to reappear in wider
                # pools; selected/used filtering prevents duplicate selection.
                score = (
                    tier,
                    abs((control_day - event_day).days),
                    time_of_day_distance(event_second, second),
                    second,
                )
                scored.append((score, candidate, tier_name))
            scored.sort(key=lambda item: item[0])
            needed = controls_per_event - len(selected)
            for item in scored[:needed]:
                selected.append(item)
                selected_seconds.add(int(item[1]["control_second"]))

        if len(selected) < controls_per_event and not allow_partial_controls:
            raise RuntimeError(
                f"Event {event_row['event_id']} received {len(selected)} controls; "
                f"{controls_per_event} required. Use --allow-partial-controls only for diagnostics."
            )
        for rank, (score, candidate, tier_name) in enumerate(selected, start=1):
            control_second = int(candidate["control_second"])
            used_seconds.add(control_second)
            matches.append(
                {
                    "event_id": str(event_row["event_id"]),
                    "event_start_second": event_second,
                    "event_start_timestamp_utc": pd.to_datetime(event_second, unit="s", utc=True),
                    "event_start_date": event_day.isoformat(),
                    "event_hour_utc": event_hour,
                    "event_chronological_period": event_period,
                    "control_id": f"CTRL-{control_second}",
                    "control_rank": rank,
                    "control_second": control_second,
                    "control_timestamp_utc": pd.to_datetime(control_second, unit="s", utc=True),
                    "control_date": candidate["control_date"].isoformat(),
                    "control_hour_utc": int(candidate["control_hour_utc"]),
                    "control_chronological_period": str(candidate["control_period"]),
                    "match_tier": int(score[0]),
                    "match_tier_name": tier_name,
                    "calendar_day_distance": int(score[1]),
                    "time_of_day_distance_seconds": int(score[2]),
                    "hour_distance": hour_distance(
                        event_hour, int(candidate["control_hour_utc"])
                    ),
                }
            )
        event_catalog_rows.append(
            {
                **event_row,
                "matched_control_count": len(selected),
                "worst_match_tier": max(
                    (int(item[0][0]) for item in selected), default=None
                ),
            }
        )
    return pd.DataFrame(matches), pd.DataFrame(event_catalog_rows)


# ---------------------------------------------------------------------------
# Feature observations
# ---------------------------------------------------------------------------


def sampled_seconds_by_day(seconds: Iterable[int]) -> dict[date, np.ndarray]:
    grouped: dict[date, list[int]] = {}
    for value in sorted(set(int(item) for item in seconds)):
        grouped.setdefault(seconds_to_date(value), []).append(value)
    return {day: np.asarray(values, dtype=np.int64) for day, values in grouped.items()}


def read_feature_samples(
    feature_partitions: Mapping[date, ParquetPartition],
    target_seconds: Sequence[int],
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    _, pc_module, pq_module = require_arrow()
    targets_by_day = sampled_seconds_by_day(target_seconds)
    frames: list[pd.DataFrame] = []
    read_columns = [
        "symbol",
        "receive_second",
        "timestamp_utc",
        "flow_feature_context_ready",
        *feature_columns,
    ]
    # CVD levels are loaded only for causal within-sample deltas.
    for optional in ("cvd_quote", "cvd_base"):
        if optional not in read_columns:
            read_columns.append(optional)

    for day, targets in sorted(targets_by_day.items()):
        partition = feature_partitions.get(day)
        if partition is None:
            raise RuntimeError(f"No feature partition exists for sampled date {day}")
        day_frames: list[pd.DataFrame] = []
        target_array = pa.array(targets) if pa is not None else None
        for path in partition.files:
            parquet_file = pq_module.ParquetFile(path)
            try:
                table = parquet_file.read(columns=read_columns)
            finally:
                parquet_file.close()
            mask = pc_module.is_in(table.column("receive_second"), value_set=target_array)
            filtered = table.filter(mask)
            if filtered.num_rows:
                day_frames.append(filtered.to_pandas())
        if day_frames:
            frames.append(pd.concat(day_frames, ignore_index=True))
    if not frames:
        raise RuntimeError("No feature observations matched requested seconds")
    output = pd.concat(frames, ignore_index=True)
    output["receive_second"] = pd.to_numeric(output["receive_second"], errors="raise").astype(np.int64)
    output = output.sort_values("receive_second", kind="mergesort").reset_index(drop=True)
    if output["receive_second"].duplicated().any():
        raise RuntimeError("Sampled feature seconds are duplicated")
    missing = sorted(set(int(value) for value in target_seconds) - set(output["receive_second"].astype(int)))
    if missing:
        raise RuntimeError(f"Feature sampling missed {len(missing)} exact seconds; first={missing[:10]}")
    if not output["flow_feature_context_ready"].fillna(False).all():
        bad = output.loc[~output["flow_feature_context_ready"].fillna(False), "receive_second"].head().tolist()
        raise RuntimeError(f"Sampled feature rows are not context-ready: {bad}")
    return output


def build_observation_requests(
    events: pd.DataFrame,
    matches: pd.DataFrame,
    leads: Sequence[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    event_lookup = events.set_index("event_id", drop=False).to_dict("index")
    for event in events.to_dict("records"):
        anchor = int(event["event_start_second"])
        for lead in leads:
            observation = anchor - int(lead)
            rows.append(
                {
                    "sample_kind": "EVENT",
                    "sample_id": str(event["event_id"]),
                    "source_event_id": str(event["event_id"]),
                    "control_rank": pd.NA,
                    "match_tier": pd.NA,
                    "match_tier_name": None,
                    "anchor_second": anchor,
                    "anchor_timestamp_utc": pd.to_datetime(anchor, unit="s", utc=True),
                    "anchor_date": seconds_to_date(anchor).isoformat(),
                    "anchor_hour_utc": seconds_to_hour(anchor),
                    "chronological_period": str(event["chronological_period"]),
                    "event_context_cohort": str(event["event_context_cohort"]),
                    "lead_seconds": int(lead),
                    "observation_second": observation,
                    "observation_timestamp_utc": pd.to_datetime(observation, unit="s", utc=True),
                    "maximum_future_swing_15m_bps": float(event["maximum_future_swing_15m_bps"]),
                    "event_path_type": str(event["path_type"]),
                }
            )
    for match in matches.to_dict("records"):
        event = event_lookup[str(match["event_id"])]
        anchor = int(match["control_second"])
        for lead in leads:
            observation = anchor - int(lead)
            rows.append(
                {
                    "sample_kind": "CONTROL",
                    "sample_id": str(match["control_id"]),
                    "source_event_id": str(match["event_id"]),
                    "control_rank": int(match["control_rank"]),
                    "match_tier": int(match["match_tier"]),
                    "match_tier_name": str(match["match_tier_name"]),
                    "anchor_second": anchor,
                    "anchor_timestamp_utc": pd.to_datetime(anchor, unit="s", utc=True),
                    "anchor_date": seconds_to_date(anchor).isoformat(),
                    "anchor_hour_utc": seconds_to_hour(anchor),
                    "chronological_period": str(event["chronological_period"]),
                    "event_context_cohort": str(event["event_context_cohort"]),
                    "lead_seconds": int(lead),
                    "observation_second": observation,
                    "observation_timestamp_utc": pd.to_datetime(observation, unit="s", utc=True),
                    "maximum_future_swing_15m_bps": float(event["maximum_future_swing_15m_bps"]),
                    "event_path_type": str(event["path_type"]),
                }
            )
    frame = pd.DataFrame(rows)
    frame["control_rank"] = frame["control_rank"].astype("Int64")
    frame["match_tier"] = frame["match_tier"].astype("Int64")
    return frame


def attach_features(
    requests: pd.DataFrame,
    feature_samples: pd.DataFrame,
    earliest_lead: int,
) -> pd.DataFrame:
    feature_samples = feature_samples.rename(columns={"receive_second": "observation_second"})
    duplicated_metadata = [
        column
        for column in ("timestamp_utc", "symbol", "flow_feature_context_ready")
        if column in feature_samples
    ]
    merged = requests.merge(feature_samples, on="observation_second", how="left", validate="many_to_one")
    if merged["flow_feature_context_ready"].isna().any():
        raise RuntimeError("Some observation requests did not join to feature rows")
    if not merged["flow_feature_context_ready"].astype(bool).all():
        raise RuntimeError("Some joined observations are not flow-feature context-ready")
    baseline = merged[merged["lead_seconds"] == int(earliest_lead)][
        ["sample_id", "cvd_quote", "cvd_base"]
    ].rename(columns={"cvd_quote": "baseline_cvd_quote", "cvd_base": "baseline_cvd_base"})
    if baseline["sample_id"].duplicated().any():
        raise RuntimeError("CVD baseline rows are not unique by sample")
    merged = merged.merge(baseline, on="sample_id", how="left", validate="many_to_one")
    merged["cvd_quote_change_from_earliest"] = (
        pd.to_numeric(merged["cvd_quote"], errors="coerce")
        - pd.to_numeric(merged["baseline_cvd_quote"], errors="coerce")
    )
    merged["cvd_base_change_from_earliest"] = (
        pd.to_numeric(merged["cvd_base"], errors="coerce")
        - pd.to_numeric(merged["baseline_cvd_base"], errors="coerce")
    )
    merged = merged.drop(columns=["baseline_cvd_quote", "baseline_cvd_base"])
    return merged.sort_values(
        ["source_event_id", "sample_kind", "sample_id", "lead_seconds"],
        ascending=[True, True, True, False],
        kind="mergesort",
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


def observation_scopes(observations: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    scopes: list[tuple[str, pd.DataFrame]] = [("ALL_EVENTS", observations)]
    isolated = observations[
        observations["event_context_cohort"].astype(str).eq("ISOLATED_ONSET")
    ]
    if len(isolated):
        scopes.append(("ISOLATED_ONSETS", isolated))
    return scopes


def comparison_profile(observations: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for analysis_scope, scope_frame in observation_scopes(observations):
        for lead in sorted(scope_frame["lead_seconds"].unique(), reverse=True):
            lead_frame = scope_frame[scope_frame["lead_seconds"] == lead]
            event_frame = lead_frame[lead_frame["sample_kind"] == "EVENT"]
            control_frame = lead_frame[lead_frame["sample_kind"] == "CONTROL"]
            for feature in feature_columns:
                event = pd.to_numeric(event_frame[feature], errors="coerce")
                control = pd.to_numeric(control_frame[feature], errors="coerce")
                event_finite = event[np.isfinite(event.to_numpy(dtype=float, na_value=np.nan))]
                control_finite = control[np.isfinite(control.to_numpy(dtype=float, na_value=np.nan))]
                event_median = safe_stat(event_finite, "median")
                control_median = safe_stat(control_finite, "median")
                ratio = (
                    event_median / control_median
                    if math.isfinite(event_median)
                    and math.isfinite(control_median)
                    and not math.isclose(control_median, 0.0, abs_tol=1e-15)
                    else float("nan")
                )
                rows.append(
                    {
                        "analysis_scope": analysis_scope,
                        "feature": feature,
                        "feature_family": feature_family(feature),
                        "lead_seconds": int(lead),
                        "event_rows": len(event_frame),
                        "event_finite_rows": len(event_finite),
                        "event_coverage_pct": percent(len(event_finite), len(event_frame)),
                        "event_mean": safe_stat(event_finite, "mean"),
                        "event_median": event_median,
                        "event_p25": safe_stat(event_finite, "p25"),
                        "event_p75": safe_stat(event_finite, "p75"),
                        "control_rows": len(control_frame),
                        "control_finite_rows": len(control_finite),
                        "control_coverage_pct": percent(len(control_finite), len(control_frame)),
                        "control_mean": safe_stat(control_finite, "mean"),
                        "control_median": control_median,
                        "control_p25": safe_stat(control_finite, "p25"),
                        "control_p75": safe_stat(control_finite, "p75"),
                        "mean_difference": safe_stat(event_finite, "mean") - safe_stat(control_finite, "mean"),
                        "median_difference": event_median - control_median,
                        "event_to_control_median_ratio": ratio,
                        "standardized_mean_difference": standardized_mean_difference(event_finite, control_finite),
                        "cliffs_delta": cliffs_delta(event_finite, control_finite),
                        "event_above_control_median_pct": (
                            percent(
                                int(np.sum(event_finite.to_numpy(dtype=float) > control_median)),
                                len(event_finite),
                            )
                            if math.isfinite(control_median)
                            else float("nan")
                        ),
                    }
                )
    return pd.DataFrame(rows)


def event_level_matched_values(
    observations: pd.DataFrame,
    feature: str,
    lead: int,
) -> pd.DataFrame:
    selected = observations[observations["lead_seconds"] == int(lead)][
        ["source_event_id", "sample_kind", feature]
    ].copy()
    selected[feature] = pd.to_numeric(selected[feature], errors="coerce")
    event = selected[selected["sample_kind"] == "EVENT"].set_index("source_event_id")[[feature]]
    event = event.rename(columns={feature: "event_value"})
    controls = (
        selected[selected["sample_kind"] == "CONTROL"]
        .groupby("source_event_id", sort=False)[feature]
        .mean()
        .to_frame("matched_control_mean")
    )
    result = event.join(controls, how="inner")
    result["paired_difference"] = result["event_value"] - result["matched_control_mean"]
    return result.reset_index()


def matched_effect_profile(observations: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for analysis_scope, scope_frame in observation_scopes(observations):
        for lead in sorted(scope_frame["lead_seconds"].unique(), reverse=True):
            event_total = int(scope_frame.loc[
                (scope_frame["lead_seconds"] == lead)
                & (scope_frame["sample_kind"] == "EVENT"),
                "source_event_id",
            ].nunique())
            for feature in feature_columns:
                paired = event_level_matched_values(scope_frame, feature, int(lead))
                finite = paired[
                    np.isfinite(paired["event_value"].to_numpy(dtype=float, na_value=np.nan))
                    & np.isfinite(paired["matched_control_mean"].to_numpy(dtype=float, na_value=np.nan))
                ]
                difference = finite["paired_difference"]
                rows.append(
                    {
                        "analysis_scope": analysis_scope,
                        "feature": feature,
                        "feature_family": feature_family(feature),
                        "lead_seconds": int(lead),
                        "event_count": event_total,
                        "matched_finite_event_count": len(finite),
                        "matched_coverage_pct": percent(len(finite), event_total),
                        "event_mean": safe_stat(finite["event_value"], "mean"),
                        "matched_control_mean": safe_stat(finite["matched_control_mean"], "mean"),
                        "mean_paired_difference": safe_stat(difference, "mean"),
                        "median_paired_difference": safe_stat(difference, "median"),
                        "paired_difference_p25": safe_stat(difference, "p25"),
                        "paired_difference_p75": safe_stat(difference, "p75"),
                        "positive_paired_difference_pct": percent(
                            int(np.sum(pd.to_numeric(difference, errors="coerce") > 0)), len(finite)
                        ),
                        "paired_standardized_effect": paired_standardized_effect(difference),
                    }
                )
    return pd.DataFrame(rows)


def trajectory_profile(
    observations: pd.DataFrame,
    feature_columns: Sequence[str],
    earliest_lead: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for analysis_scope, scope_frame in observation_scopes(observations):
        baseline = scope_frame[scope_frame["lead_seconds"] == int(earliest_lead)][
            ["sample_id", *feature_columns]
        ].set_index("sample_id")
        for lead in sorted(scope_frame["lead_seconds"].unique(), reverse=True):
            lead_frame = scope_frame[scope_frame["lead_seconds"] == lead].copy()
            for feature in feature_columns:
                values = pd.to_numeric(lead_frame[feature], errors="coerce")
                baseline_values = pd.to_numeric(
                    lead_frame["sample_id"].map(baseline[feature]), errors="coerce"
                )
                lead_frame["_delta"] = values - baseline_values
                event_delta = lead_frame.loc[lead_frame["sample_kind"] == "EVENT", "_delta"]
                control_delta = lead_frame.loc[lead_frame["sample_kind"] == "CONTROL", "_delta"]
                rows.append(
                    {
                        "analysis_scope": analysis_scope,
                        "feature": feature,
                        "feature_family": feature_family(feature),
                        "baseline_lead_seconds": int(earliest_lead),
                        "lead_seconds": int(lead),
                        "event_delta_mean": safe_stat(event_delta, "mean"),
                        "event_delta_median": safe_stat(event_delta, "median"),
                        "control_delta_mean": safe_stat(control_delta, "mean"),
                        "control_delta_median": safe_stat(control_delta, "median"),
                        "difference_in_mean_change": safe_stat(event_delta, "mean") - safe_stat(control_delta, "mean"),
                        "difference_in_median_change": safe_stat(event_delta, "median") - safe_stat(control_delta, "median"),
                        "delta_standardized_mean_difference": standardized_mean_difference(event_delta, control_delta),
                        "delta_cliffs_delta": cliffs_delta(event_delta, control_delta),
                    }
                )
    return pd.DataFrame(rows)


def chronological_effect_profile(
    observations: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for analysis_scope, scope_frame in observation_scopes(observations):
        for period in ("EARLY", "MIDDLE", "LATE"):
            period_frame = scope_frame[scope_frame["chronological_period"] == period]
            for lead in sorted(period_frame["lead_seconds"].unique(), reverse=True):
                lead_frame = period_frame[period_frame["lead_seconds"] == lead]
                event_frame = lead_frame[lead_frame["sample_kind"] == "EVENT"]
                control_frame = lead_frame[lead_frame["sample_kind"] == "CONTROL"]
                for feature in feature_columns:
                    event = pd.to_numeric(event_frame[feature], errors="coerce")
                    control = pd.to_numeric(control_frame[feature], errors="coerce")
                    rows.append(
                        {
                            "analysis_scope": analysis_scope,
                            "chronological_period": period,
                            "feature": feature,
                            "feature_family": feature_family(feature),
                            "lead_seconds": int(lead),
                            "event_count": int(event_frame["source_event_id"].nunique()),
                            "control_count": int(control_frame["sample_id"].nunique()),
                            "event_coverage_pct": percent(int(event.notna().sum()), len(event)),
                            "control_coverage_pct": percent(int(control.notna().sum()), len(control)),
                            "event_median": safe_stat(event, "median"),
                            "control_median": safe_stat(control, "median"),
                            "median_difference": safe_stat(event, "median") - safe_stat(control, "median"),
                            "standardized_mean_difference": standardized_mean_difference(event, control),
                            "cliffs_delta": cliffs_delta(event, control),
                        }
                    )
    return pd.DataFrame(rows)


def stability_profile(
    comparison: pd.DataFrame,
    chronological: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    overall_lookup = comparison.set_index(["analysis_scope", "feature", "lead_seconds"])
    grouped = chronological.groupby(
        ["analysis_scope", "feature", "lead_seconds"], sort=True
    )
    for (analysis_scope, feature, lead), group in grouped:
        effects = pd.to_numeric(group["standardized_mean_difference"], errors="coerce")
        finite = effects[np.isfinite(effects.to_numpy(dtype=float, na_value=np.nan))]
        signs = np.sign(finite.to_numpy(dtype=float))
        nonzero = signs[signs != 0]
        sign_consistent = bool(len(nonzero) > 0 and np.all(nonzero == nonzero[0]))
        overall = overall_lookup.loc[(analysis_scope, feature, int(lead))]
        overall_effect = float(overall["standardized_mean_difference"])
        rows.append(
            {
                "analysis_scope": analysis_scope,
                "feature": feature,
                "feature_family": feature_family(feature),
                "lead_seconds": int(lead),
                "overall_standardized_mean_difference": overall_effect,
                "overall_cliffs_delta": float(overall["cliffs_delta"]),
                "periods_available": len(finite),
                "period_sign_consistent": sign_consistent,
                "periods_abs_smd_at_least_0_2": int(np.sum(np.abs(finite) >= 0.2)),
                "periods_abs_smd_at_least_0_5": int(np.sum(np.abs(finite) >= 0.5)),
                "minimum_absolute_period_smd": float(np.min(np.abs(finite))) if len(finite) else float("nan"),
                "median_absolute_period_smd": float(np.median(np.abs(finite))) if len(finite) else float("nan"),
                "maximum_absolute_period_smd": float(np.max(np.abs(finite))) if len(finite) else float("nan"),
                "period_effect_range": float(np.max(finite) - np.min(finite)) if len(finite) else float("nan"),
                "event_coverage_pct": float(overall["event_coverage_pct"]),
                "control_coverage_pct": float(overall["control_coverage_pct"]),
            }
        )
    return pd.DataFrame(rows)


def top_feature_profile(stability: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    matched_lookup = matched.set_index(["analysis_scope", "feature", "lead_seconds"])
    frame = stability.copy()
    frame["overall_absolute_smd"] = frame["overall_standardized_mean_difference"].abs()
    frame["stable_effect_score"] = np.where(
        frame["period_sign_consistent"]
        & (frame["periods_available"] == 3)
        & (frame["event_coverage_pct"] >= 80.0)
        & (frame["control_coverage_pct"] >= 80.0),
        frame["minimum_absolute_period_smd"],
        0.0,
    )
    paired_effects: list[float] = []
    for row in frame.itertuples(index=False):
        try:
            paired_effects.append(
                float(
                    matched_lookup.loc[
                        (row.analysis_scope, row.feature, row.lead_seconds),
                        "paired_standardized_effect",
                    ]
                )
            )
        except KeyError:
            paired_effects.append(float("nan"))
    frame["paired_standardized_effect"] = paired_effects
    frame["effect_direction"] = np.where(
        frame["overall_standardized_mean_difference"] > 0,
        "HIGHER_BEFORE_EVENTS",
        np.where(
            frame["overall_standardized_mean_difference"] < 0,
            "LOWER_BEFORE_EVENTS",
            "NO_DIFFERENCE",
        ),
    )
    frame = frame.sort_values(
        ["analysis_scope", "lead_seconds", "stable_effect_score", "overall_absolute_smd", "feature"],
        ascending=[True, False, False, False, True],
        kind="mergesort",
    )
    frame["rank_within_lead"] = (
        frame.groupby(["analysis_scope", "lead_seconds"]).cumcount() + 1
    )
    return frame[
        [
            "analysis_scope",
            "lead_seconds",
            "rank_within_lead",
            "feature",
            "feature_family",
            "effect_direction",
            "overall_standardized_mean_difference",
            "overall_cliffs_delta",
            "paired_standardized_effect",
            "period_sign_consistent",
            "periods_abs_smd_at_least_0_2",
            "minimum_absolute_period_smd",
            "median_absolute_period_smd",
            "stable_effect_score",
            "event_coverage_pct",
            "control_coverage_pct",
        ]
    ]


def match_quality_profile(matches: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(matches)
    for (tier, name), group in matches.groupby(["match_tier", "match_tier_name"], sort=True):
        rows.append(
            {
                "match_tier": int(tier),
                "match_tier_name": str(name),
                "matches": len(group),
                "share_pct": percent(len(group), total),
                "median_calendar_day_distance": safe_stat(group["calendar_day_distance"], "median"),
                "p90_calendar_day_distance": safe_stat(group["calendar_day_distance"], "p90"),
                "median_time_of_day_distance_seconds": safe_stat(group["time_of_day_distance_seconds"], "median"),
                "p90_time_of_day_distance_seconds": safe_stat(group["time_of_day_distance_seconds"], "p90"),
            }
        )
    return pd.DataFrame(rows)


def coverage_profile(
    events: pd.DataFrame,
    event_catalog: pd.DataFrame,
    candidates: pd.DataFrame,
    matches: pd.DataFrame,
    observations: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {"metric": "selected_events", "value": len(events), "detail": "event onsets before exact-feature filtering"},
        {"metric": "profiled_events", "value": len(event_catalog), "detail": "events with all requested exact ready lead rows"},
        {"metric": "eligible_control_candidates", "value": len(candidates), "detail": "negative cadence anchors outside embargo intervals"},
        {"metric": "matched_controls", "value": len(matches), "detail": "unique deterministic controls"},
        {"metric": "observation_rows", "value": len(observations), "detail": "event and control lead observations"},
        {"metric": "profiled_features", "value": len(feature_columns), "detail": "causal numeric columns plus derived CVD changes"},
    ]
    for period, group in event_catalog.groupby("chronological_period", sort=True):
        rows.append(
            {
                "metric": f"events_{period.lower()}",
                "value": len(group),
                "detail": "profiled events in chronological period",
            }
        )
    for cohort, group in event_catalog.groupby("event_context_cohort", sort=True):
        rows.append(
            {
                "metric": f"events_{str(cohort).lower()}",
                "value": len(group),
                "detail": "profiled events by pre-window context cohort",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def invariant_frame(
    events: pd.DataFrame,
    event_catalog: pd.DataFrame,
    candidates: pd.DataFrame,
    matches: pd.DataFrame,
    observations: pd.DataFrame,
    leads: Sequence[int],
    controls_per_event: int,
    allow_partial_controls: bool,
    exclusion_starts: np.ndarray,
    exclusion_ends: np.ndarray,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        rows.append({"invariant": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    add(
        "EVENT_IDS_UNIQUE",
        not event_catalog["event_id"].duplicated().any(),
        f"events={len(event_catalog):,}",
    )
    add(
        "CONTROL_SECONDS_UNIQUE",
        not matches["control_second"].duplicated().any(),
        f"controls={len(matches):,}",
    )
    expected_controls = len(event_catalog) * controls_per_event
    add(
        "CONTROL_COUNT_COMPLETE_OR_ALLOWED",
        allow_partial_controls or len(matches) == expected_controls,
        f"actual={len(matches):,}, expected={expected_controls:,}, allow_partial={allow_partial_controls}",
    )
    outside = ~interval_membership(
        matches["control_second"].to_numpy(dtype=np.int64), exclusion_starts, exclusion_ends
    )
    add(
        "CONTROLS_OUTSIDE_EVENT_EMBARGO_WINDOWS",
        bool(np.all(outside)),
        f"outside={int(np.sum(outside)):,}/{len(outside):,}",
    )
    expected_observations = (len(event_catalog) + len(matches)) * len(leads)
    add(
        "OBSERVATION_ROW_COUNT",
        len(observations) == expected_observations,
        f"actual={len(observations):,}, expected={expected_observations:,}",
    )
    exact_offset = (
        observations["anchor_second"].to_numpy(dtype=np.int64)
        - observations["lead_seconds"].to_numpy(dtype=np.int64)
        == observations["observation_second"].to_numpy(dtype=np.int64)
    )
    add(
        "EXACT_LEAD_OFFSETS",
        bool(np.all(exact_offset)),
        f"exact={int(np.sum(exact_offset)):,}/{len(exact_offset):,}",
    )
    add(
        "ALL_OBSERVATIONS_CONTEXT_READY",
        bool(observations["flow_feature_context_ready"].astype(bool).all()),
        f"ready={int(observations['flow_feature_context_ready'].astype(bool).sum()):,}/{len(observations):,}",
    )
    event_onset_rows = observations[
        (observations["sample_kind"] == "EVENT") & (observations["lead_seconds"] == 0)
    ]
    add(
        "ONE_EVENT_ONSET_ROW_PER_EVENT",
        len(event_onset_rows) == len(event_catalog)
        and event_onset_rows["source_event_id"].nunique() == len(event_catalog),
        f"onset_rows={len(event_onset_rows):,}, events={len(event_catalog):,}",
    )
    forbidden = [
        column
        for column in feature_columns
        if any(token in column.lower() for token in FUTURE_OR_LABEL_TOKENS)
    ]
    add(
        "NO_FUTURE_OR_LABEL_FEATURE_COLUMNS",
        not forbidden,
        "none" if not forbidden else ",".join(forbidden),
    )
    add(
        "CONTROL_CANDIDATES_NEGATIVE_BY_CONSTRUCTION",
        len(candidates) >= len(matches),
        f"candidates={len(candidates):,}, selected={len(matches):,}",
    )
    sample_lead_counts = observations.groupby("sample_id")["lead_seconds"].nunique()
    add(
        "EVERY_SAMPLE_HAS_ALL_LEADS",
        bool((sample_lead_counts == len(leads)).all()),
        f"complete={int((sample_lead_counts == len(leads)).sum()):,}/{len(sample_lead_counts):,}",
    )
    add(
        "EVENTS_CHRONOLOGICALLY_ORDERED",
        bool(np.all(np.diff(event_catalog["event_start_second"].to_numpy(dtype=np.int64)) > 0)),
        f"events={len(event_catalog):,}",
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# State and output validation
# ---------------------------------------------------------------------------


def connect_state(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            started_at_utc TEXT NOT NULL,
            completed_at_utc TEXT,
            status TEXT NOT NULL,
            input_signature TEXT NOT NULL,
            config_signature TEXT NOT NULL,
            event_count INTEGER,
            control_count INTEGER,
            observation_rows INTEGER,
            feature_count INTEGER,
            failed_invariants INTEGER,
            configuration_json TEXT NOT NULL,
            summary_json TEXT,
            error_text TEXT
        );

        CREATE TABLE IF NOT EXISTS latest_build (
            symbol TEXT NOT NULL,
            segmentation_id TEXT NOT NULL,
            status TEXT NOT NULL,
            input_signature TEXT,
            config_signature TEXT,
            output_manifest_json TEXT,
            report_manifest_json TEXT,
            summary_json TEXT,
            error_text TEXT,
            PRIMARY KEY (symbol, segmentation_id)
        );
        """
    )
    connection.commit()
    return connection


def validate_output_manifest(manifest: Sequence[Mapping[str, Any]], input_signature: str) -> bool:
    pq_module = require_arrow()[2]
    try:
        for item in manifest:
            path = Path(str(item["path"]))
            if not path.exists():
                return False
            parquet_file = pq_module.ParquetFile(path)
            try:
                if int(parquet_file.metadata.num_rows) != int(item["rows"]):
                    return False
                metadata = decode_metadata(parquet_file.schema_arrow)
                if metadata.get("input_signature") != input_signature:
                    return False
            finally:
                parquet_file.close()
        return True
    except Exception:
        return False


def validate_report_manifest(manifest: Sequence[Mapping[str, Any]]) -> bool:
    try:
        return all(
            Path(str(item["path"])).exists()
            and Path(str(item["path"])).stat().st_size == int(item["size_bytes"])
            for item in manifest
        )
    except Exception:
        return False


def observations_path(output_root: Path, symbol: str) -> Path:
    return output_root / "observations" / f"{SYMBOL_PREFIX}{symbol}" / "part-00000.parquet"


def matches_path(output_root: Path, symbol: str) -> Path:
    return output_root / "matches" / f"{SYMBOL_PREFIX}{symbol}" / "part-00000.parquet"


def event_catalog_path(output_root: Path, symbol: str) -> Path:
    return output_root / "event_catalog" / f"{SYMBOL_PREFIX}{symbol}" / "part-00000.parquet"


def write_reports(
    report_root: Path,
    run_id: str,
    frames: Mapping[str, pd.DataFrame],
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    findings: str,
) -> list[dict[str, Any]]:
    run_root = report_root / "runs" / run_id
    report_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    for filename, frame in frames.items():
        atomic_write_csv(report_root / filename, frame)
        atomic_write_csv(run_root / filename, frame)
    atomic_write_json(report_root / "perps_pre_expansion_summary.json", summary)
    atomic_write_json(run_root / "perps_pre_expansion_summary.json", summary)
    atomic_write_json(report_root / "perps_pre_expansion_config.json", config)
    atomic_write_json(run_root / "perps_pre_expansion_config.json", config)
    atomic_write_text(report_root / "perps_pre_expansion_findings.md", findings)
    atomic_write_text(run_root / "perps_pre_expansion_findings.md", findings)
    manifest: list[dict[str, Any]] = []
    for filename in REPORT_FILENAMES:
        path = report_root / filename
        manifest.append({"path": str(path), "size_bytes": path.stat().st_size})
    return manifest


def markdown_findings(
    summary: Mapping[str, Any],
    top_features: pd.DataFrame,
    match_quality: pd.DataFrame,
) -> str:
    lines = [
        "# Perps Pre-Expansion Footprint Profile",
        "",
        f"Run ID: `{summary['run_id']}`",
        "",
        "## Coverage",
        "",
        f"- Profiled events: **{summary['profiled_events']:,}**",
        f"- Isolated-onset events: **{summary['isolated_onset_events']:,}**",
        f"- Recent-event-context events: **{summary['recent_event_context_events']:,}**",
        f"- Matched controls: **{summary['matched_controls']:,}**",
        f"- Observation rows: **{summary['observation_rows']:,}**",
        f"- Profiled causal features: **{summary['profiled_features']:,}**",
        f"- Failed invariants: **{summary['failed_invariants']:,}**",
        "",
        "## Control matching",
        "",
    ]
    if len(match_quality):
        for row in match_quality.itertuples(index=False):
            lines.append(
                f"- `{row.match_tier_name}`: {int(row.matches):,} controls "
                f"({float(row.share_pct):.2f}%)."
            )
    lines.extend(
        [
            "",
            "## Strongest stable differences by lead",
            "",
            "The table below is descriptive, not model performance. A feature is marked stable only when its effect direction agrees across all available chronological thirds and coverage is at least 80% for events and controls.",
            "",
            "| Lead | Feature | Family | Direction | Overall SMD | Minimum period | Paired effect |",
            "|---:|---|---|---|---:|---:|---:|",
        ]
    )
    preferred_scope = (
        "ISOLATED_ONSETS"
        if "ISOLATED_ONSETS" in set(top_features["analysis_scope"].astype(str))
        else "ALL_EVENTS"
    )
    lines.append(f"Primary table scope: `{preferred_scope}`.")
    lines.append("")
    selected = top_features[
        top_features["analysis_scope"].astype(str).eq(preferred_scope)
        & (top_features["rank_within_lead"] <= 5)
        & (top_features["stable_effect_score"] > 0)
    ]
    for row in selected.itertuples(index=False):
        lines.append(
            f"| {int(row.lead_seconds)}s | `{row.feature}` | {row.feature_family} | "
            f"{row.effect_direction} | {float(row.overall_standardized_mean_difference):.3f} | "
            f"{float(row.minimum_absolute_period_smd):.3f} | "
            f"{float(row.paired_standardized_effect):.3f} |"
        )
    if selected.empty:
        lines.append("| — | No feature passed the strict stability screen | — | — | — | — | — |")
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Event and control identities come from future labels, but every profiled feature value is strictly causal at its observation second.",
            "- `ISOLATED_ONSETS` requires the full requested pre-window plus embargo to be clear of the preceding expansion event; `ALL_EVENTS` is retained as a sensitivity view.",
            "- These are retrospective associations. They do not establish tradability, calibration, precision, recall or warning time.",
            "- Large effects at event onset but not at earlier leads may describe an expansion already becoming visible rather than a useful early footprint.",
            "- Stable effects should be reviewed for redundancy before any modelling table is built.",
            "",
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def run_self_test() -> None:
    starts = np.array([100, 130, 500], dtype=np.int64)
    ends = np.array([120, 160, 520], dtype=np.int64)
    merged_starts, merged_ends = merge_intervals(starts, ends)
    if not np.array_equal(merged_starts, np.array([100, 130, 500])):
        raise AssertionError((merged_starts, merged_ends))
    values = np.array([99, 100, 140, 161, 500, 521], dtype=np.int64)
    expected = np.array([False, True, True, False, True, False])
    if not np.array_equal(interval_membership(values, merged_starts, merged_ends), expected):
        raise AssertionError("interval membership")

    ready = np.arange(0, 2000, dtype=np.int64)
    anchors = np.array([900, 901, 1200], dtype=np.int64)
    valid = valid_anchor_mask(anchors, (900, 300, 0), ready)
    if not np.array_equal(valid, np.array([True, True, True])):
        raise AssertionError(valid)

    tier = tier_for_match(date(2026, 1, 1), 10, "EARLY", date(2026, 1, 1), 10, "EARLY", 1)
    if tier != (0, "SAME_DATE_SAME_HOUR"):
        raise AssertionError(tier)
    tier = tier_for_match(date(2026, 1, 1), 10, "EARLY", date(2026, 1, 2), 10, "EARLY", 1)
    if tier != (2, "SAME_PERIOD_SAME_HOUR"):
        raise AssertionError(tier)

    smd = standardized_mean_difference([2, 3, 4], [0, 1, 2])
    if not math.isfinite(smd) or smd <= 0:
        raise AssertionError(smd)
    delta = cliffs_delta([3, 4], [1, 2])
    if not math.isclose(delta, 1.0):
        raise AssertionError(delta)

    periods = assign_chronological_periods([date(2026, 1, day) for day in range(1, 10)])
    if set(periods.values()) != {"EARLY", "MIDDLE", "LATE"}:
        raise AssertionError(periods)

    print("SELF-TEST PASS")
    print(
        "Validated exact lead availability, interval exclusion, deterministic match tiers, "
        "chronological periods and effect-size calculations."
    )


# ---------------------------------------------------------------------------
# CLI and orchestration
# ---------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Profile causal perps flow-feature footprints before bridge_060s expansion events."
    )
    value.add_argument(
        "--feature-root",
        default="./data/derived/perps/flow_features_1s_v1",
        help="Audited causal perps flow-feature root.",
    )
    value.add_argument(
        "--label-root",
        default="./data/derived/perps/research/expansion_episodes_15m_50bps_v1",
        help="Upstream row-label dataset root.",
    )
    value.add_argument(
        "--segmentation-root",
        default="./data/derived/perps/research/expansion_event_segmentation_sensitivity_v1",
        help="Segmentation-sensitivity dataset root containing events/.",
    )
    value.add_argument(
        "--output-root",
        default="./data/derived/perps/research/pre_expansion_footprints_v1",
        help="Separate footprint observation dataset root.",
    )
    value.add_argument(
        "--report-root",
        default="./reports/perps_pre_expansion_footprints",
        help="CSV, JSON and Markdown report root.",
    )
    value.add_argument("--symbol", default="BTCUSDT")
    value.add_argument("--segmentation-id", default=DEFAULT_SEGMENTATION_ID)
    value.add_argument(
        "--leads-seconds",
        default=",".join(str(value) for value in DEFAULT_LEADS_SECONDS),
        help="Comma-separated seconds before event/control anchor; must include 0.",
    )
    value.add_argument("--controls-per-event", type=int, default=DEFAULT_CONTROLS_PER_EVENT)
    value.add_argument(
        "--control-cadence-seconds",
        type=int,
        default=DEFAULT_CONTROL_CADENCE_SECONDS,
        help="Negative pseudo-onset anchor cadence.",
    )
    value.add_argument(
        "--control-embargo-seconds",
        type=int,
        default=DEFAULT_CONTROL_EMBARGO_SECONDS,
        help="Extra exclusion around every event before control-window intersection checks.",
    )
    value.add_argument(
        "--adjacent-hour-tolerance",
        type=int,
        default=DEFAULT_ADJACENT_HOUR_TOLERANCE,
        help="UTC-hour distance accepted by adjacent-hour match tiers.",
    )
    value.add_argument(
        "--feature-columns",
        help="Optional exact comma-separated causal numeric feature columns. Default auto-selects audited groups.",
    )
    value.add_argument("--start-date", help="Optional inclusive event-onset date YYYY-MM-DD.")
    value.add_argument("--end-date", help="Optional inclusive event-onset date YYYY-MM-DD.")
    value.add_argument(
        "--max-events",
        type=int,
        help="Optional event limit after date filtering; use only with isolated smoke outputs.",
    )
    value.add_argument(
        "--allow-partial-controls",
        action="store_true",
        help="Allow fewer controls than requested for diagnostic runs.",
    )
    value.add_argument(
        "--hash-input-files",
        action="store_true",
        help="Use full SHA-256 file hashes in change detection.",
    )
    value.add_argument("--force", action="store_true")
    value.add_argument("--compression-level", type=int, default=6)
    value.add_argument("--self-test", action="store_true")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    require_arrow()

    if args.controls_per_event <= 0:
        raise SystemExit("--controls-per-event must be positive")
    if args.control_cadence_seconds <= 0:
        raise SystemExit("--control-cadence-seconds must be positive")
    if args.control_embargo_seconds < 0:
        raise SystemExit("--control-embargo-seconds cannot be negative")
    if not 0 <= args.adjacent_hour_tolerance <= 12:
        raise SystemExit("--adjacent-hour-tolerance must be between 0 and 12")
    if not 1 <= args.compression_level <= 22:
        raise SystemExit("--compression-level must be between 1 and 22")

    leads = parse_positive_ints(args.leads_seconds, "--leads-seconds", allow_zero=True)
    if 0 not in leads:
        raise SystemExit("--leads-seconds must include 0")
    start_date = parse_date(args.start_date, "--start-date")
    end_date = parse_date(args.end_date, "--end-date")
    if start_date and end_date and start_date > end_date:
        raise SystemExit("--start-date cannot be after --end-date")

    symbol = args.symbol.strip().upper()
    segmentation_id = args.segmentation_id.strip()
    feature_root = Path(args.feature_root).expanduser().resolve()
    label_root = Path(args.label_root).expanduser().resolve()
    segmentation_root = Path(args.segmentation_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    report_root = Path(args.report_root).expanduser().resolve()
    for root, name in (
        (feature_root, "feature root"),
        (label_root, "label root"),
        (segmentation_root, "segmentation root"),
    ):
        if not root.exists():
            raise SystemExit(f"{name.title()} does not exist: {root}")

    state_path = output_root / "_build_state" / "build_state.sqlite"
    connection = connect_state(state_path)
    run_id = utc_now().strftime("%Y%m%dT%H%M%SZ") + f"-{uuid.uuid4().hex[:8]}"
    started = utc_now_iso()

    try:
        print("Profiling causal pre-expansion feature footprints...")
        print(f"Run ID             : {run_id}")
        print(f"Feature root       : {feature_root}")
        print(f"Label root         : {label_root}")
        print(f"Segmentation root  : {segmentation_root}")
        print(f"Output root        : {output_root}")
        print(f"Report root        : {report_root}")
        print(f"Symbol             : {symbol}")
        print(f"Segmentation       : {segmentation_id}")
        print(f"Lead seconds       : {','.join(map(str, leads))}")

        feature_partitions = discover_date_partitions(
            feature_root, symbol, "features", args.hash_input_files
        )
        label_partitions = discover_date_partitions(
            label_root, symbol, "labels", args.hash_input_files
        )
        feature_schema_names = validate_feature_schema(feature_partitions, symbol)
        validate_label_schema(label_partitions, symbol)

        if args.feature_columns:
            feature_columns = sorted(
                {column.strip() for column in args.feature_columns.split(",") if column.strip()}
            )
        else:
            feature_columns = auto_feature_columns(feature_schema_names)
        if not feature_columns:
            raise RuntimeError("No causal numeric feature columns were selected")
        validate_numeric_feature_columns(next(iter(feature_partitions.values())).files[0], feature_columns)

        event_path = event_file_path(segmentation_root, segmentation_id, symbol)
        all_events = read_events(
            event_path,
            symbol,
            segmentation_id,
            None,
            None,
            None,
        )
        all_events = all_events.sort_values(
            "event_start_second", kind="mergesort"
        ).reset_index(drop=True)
        previous_end = all_events["event_end_second"].shift(1)
        all_events["previous_event_end_second"] = previous_end.astype("Int64")
        all_events["seconds_since_previous_event_recomputed"] = (
            all_events["event_start_second"] - previous_end - 1
        ).astype("Int64")
        isolated_mask = previous_end.isna() | (
            all_events["event_start_second"] - max(leads)
            > previous_end + args.control_embargo_seconds
        )
        all_events["event_context_cohort"] = np.where(
            isolated_mask, "ISOLATED_ONSET", "RECENT_EVENT_CONTEXT"
        )
        events = all_events.copy()
        if start_date is not None:
            events = events[events["event_start_date"] >= start_date]
        if end_date is not None:
            events = events[events["event_start_date"] <= end_date]
        events = events.sort_values("event_start_second", kind="mergesort").reset_index(drop=True)
        if args.max_events is not None:
            if args.max_events <= 0:
                raise SystemExit("--max-events must be positive")
            events = events.iloc[: args.max_events].copy()
        if events.empty:
            raise RuntimeError("No events remain after filtering")
        selected_event_count = len(events)
        event_signature = file_signature(event_path, args.hash_input_files)

        period_by_day = assign_chronological_periods(sorted(feature_partitions))
        all_events["chronological_period"] = all_events["event_start_date"].map(period_by_day)
        if all_events["chronological_period"].isna().any():
            missing_days = sorted(
                set(all_events.loc[all_events["chronological_period"].isna(), "event_start_date"])
            )
            raise RuntimeError(f"Event dates are outside feature coverage: {missing_days}")
        events["chronological_period"] = events["event_start_date"].map(period_by_day)

        config = {
            "profiler_version": PROFILER_VERSION,
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
            "symbol": symbol,
            "segmentation_id": segmentation_id,
            "leads_seconds": list(leads),
            "controls_per_event": args.controls_per_event,
            "control_cadence_seconds": args.control_cadence_seconds,
            "control_embargo_seconds": args.control_embargo_seconds,
            "adjacent_hour_tolerance": args.adjacent_hour_tolerance,
            "feature_columns": feature_columns,
            "derived_feature_columns": [
                "cvd_quote_change_from_earliest",
                "cvd_base_change_from_earliest",
            ],
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "max_events": args.max_events,
            "allow_partial_controls": args.allow_partial_controls,
            "hash_input_files": args.hash_input_files,
        }
        config_signature = sha256_text(compact_json(config))
        input_payload = {
            "event_file": event_signature,
            "feature_partitions": {
                day.isoformat(): item.signature for day, item in sorted(feature_partitions.items())
            },
            "label_partitions": {
                day.isoformat(): item.signature for day, item in sorted(label_partitions.items())
            },
        }
        input_signature = sha256_text(compact_json(input_payload))

        latest = connection.execute(
            "SELECT * FROM latest_build WHERE symbol=? AND segmentation_id=?",
            (symbol, segmentation_id),
        ).fetchone()
        if latest and not args.force:
            output_manifest = json.loads(latest["output_manifest_json"] or "[]")
            report_manifest = json.loads(latest["report_manifest_json"] or "[]")
            if (
                latest["status"] == "COMPLETED"
                and latest["input_signature"] == input_signature
                and latest["config_signature"] == config_signature
                and validate_output_manifest(output_manifest, input_signature)
                and validate_report_manifest(report_manifest)
            ):
                print("SKIP: inputs, configuration, Parquet outputs and reports are unchanged.")
                return 0

        connection.execute(
            """
            INSERT INTO runs (
                run_id, started_at_utc, status, input_signature, config_signature,
                configuration_json
            ) VALUES (?, ?, 'RUNNING', ?, ?, ?)
            """,
            (run_id, started, input_signature, config_signature, compact_json(config)),
        )
        connection.execute(
            """
            INSERT INTO latest_build (
                symbol, segmentation_id, status, input_signature, config_signature, error_text
            ) VALUES (?, ?, 'RUNNING', ?, ?, NULL)
            ON CONFLICT(symbol, segmentation_id) DO UPDATE SET
                status='RUNNING',
                input_signature=excluded.input_signature,
                config_signature=excluded.config_signature,
                error_text=NULL
            """,
            (symbol, segmentation_id, input_signature, config_signature),
        )
        connection.commit()

        print("Loading exact context-ready feature-second index...")
        ready_seconds = read_ready_seconds(feature_partitions)
        event_anchor_seconds = events["event_start_second"].to_numpy(dtype=np.int64)
        event_ready_mask = valid_anchor_mask(event_anchor_seconds, leads, ready_seconds)
        omitted_events = events.loc[~event_ready_mask, ["event_id", "event_start_second"]].copy()
        events = events.loc[event_ready_mask].reset_index(drop=True)
        if events.empty:
            raise RuntimeError("No events have complete exact context-ready lead observations")
        print(
            f"Events retained     : {len(events):,}/{len(event_ready_mask):,} "
            f"(omitted={len(omitted_events):,})"
        )

        exclusion_starts, exclusion_ends = exclusion_intervals(
            all_events,
            max(leads),
            args.control_embargo_seconds,
        )
        print("Building eligible negative control anchors...")
        candidates = load_control_candidates(
            label_partitions,
            args.control_cadence_seconds,
            ready_seconds,
            leads,
            exclusion_starts,
            exclusion_ends,
            period_by_day,
        )
        print(f"Control candidates : {len(candidates):,}")

        print("Matching controls deterministically...")
        matches, event_catalog = match_controls(
            events,
            candidates,
            args.controls_per_event,
            args.adjacent_hour_tolerance,
            args.allow_partial_controls,
        )
        print(f"Matched controls   : {len(matches):,}")

        requests = build_observation_requests(event_catalog, matches, leads)
        print(f"Requested rows     : {len(requests):,}")
        feature_samples = read_feature_samples(
            feature_partitions,
            requests["observation_second"].astype(int).tolist(),
            feature_columns,
        )
        observations = attach_features(requests, feature_samples, max(leads))
        derived_columns = ["cvd_quote_change_from_earliest", "cvd_base_change_from_earliest"]
        profile_features = [*feature_columns, *derived_columns]

        print("Computing descriptive effects and chronological stability...")
        comparison = comparison_profile(observations, profile_features)
        matched_effects = matched_effect_profile(observations, profile_features)
        trajectory = trajectory_profile(observations, profile_features, max(leads))
        chronological = chronological_effect_profile(observations, profile_features)
        stability = stability_profile(comparison, chronological)
        top_features = top_feature_profile(stability, matched_effects)
        match_quality = match_quality_profile(matches)
        coverage = coverage_profile(
            events, event_catalog, candidates, matches, observations, profile_features
        )
        invariants = invariant_frame(
            events,
            event_catalog,
            candidates,
            matches,
            observations,
            leads,
            args.controls_per_event,
            args.allow_partial_controls,
            exclusion_starts,
            exclusion_ends,
            profile_features,
        )
        failed_invariants = int((invariants["status"] == "FAIL").sum())
        if failed_invariants:
            names = invariants.loc[invariants["status"] == "FAIL", "invariant"].tolist()
            raise RuntimeError("Footprint invariants failed: " + ", ".join(names))

        common_metadata = {
            "pre_expansion_footprint_profiler_version": PROFILER_VERSION,
            "pre_expansion_footprint_schema_version": OUTPUT_SCHEMA_VERSION,
            "symbol": symbol,
            "segmentation_id": segmentation_id,
            "source_feature_builder_version": EXPECTED_FEATURE_BUILDER_VERSION,
            "source_label_builder_version": EXPECTED_LABEL_BUILDER_VERSION,
            "source_segmentation_profiler_version": EXPECTED_SEGMENTATION_PROFILER_VERSION,
            "input_signature": input_signature,
            "config_signature": config_signature,
            "created_at_utc": utc_now_iso(),
        }
        observation_file = observations_path(output_root, symbol)
        match_file = matches_path(output_root, symbol)
        catalog_file = event_catalog_path(output_root, symbol)
        atomic_write_parquet(
            observation_file,
            observations,
            {**common_metadata, "dataset_role": "event_and_matched_control_feature_observations"},
            args.compression_level,
        )
        atomic_write_parquet(
            match_file,
            matches,
            {**common_metadata, "dataset_role": "deterministic_event_control_matches"},
            args.compression_level,
        )
        atomic_write_parquet(
            catalog_file,
            event_catalog,
            {**common_metadata, "dataset_role": "profiled_event_catalog"},
            args.compression_level,
        )
        output_manifest = [
            {"path": str(observation_file), "rows": len(observations)},
            {"path": str(match_file), "rows": len(matches)},
            {"path": str(catalog_file), "rows": len(event_catalog)},
        ]

        summary = {
            "status": "COMPLETED",
            "run_id": run_id,
            "profiler_version": PROFILER_VERSION,
            "created_at_utc": utc_now_iso(),
            "symbol": symbol,
            "segmentation_id": segmentation_id,
            "feature_root": str(feature_root),
            "label_root": str(label_root),
            "segmentation_root": str(segmentation_root),
            "output_root": str(output_root),
            "report_root": str(report_root),
            "input_signature": input_signature,
            "config_signature": config_signature,
            "selected_events": selected_event_count,
            "profiled_events": len(event_catalog),
            "isolated_onset_events": int(
                event_catalog["event_context_cohort"].astype(str).eq("ISOLATED_ONSET").sum()
            ),
            "recent_event_context_events": int(
                event_catalog["event_context_cohort"].astype(str).eq("RECENT_EVENT_CONTEXT").sum()
            ),
            "omitted_events_missing_exact_ready_leads": len(omitted_events),
            "eligible_control_candidates": len(candidates),
            "matched_controls": len(matches),
            "observation_rows": len(observations),
            "profiled_features": len(profile_features),
            "failed_invariants": failed_invariants,
            "chronological_event_counts": {
                str(key): int(value)
                for key, value in event_catalog["chronological_period"].value_counts().sort_index().items()
            },
            "output_manifest": output_manifest,
        }
        findings = markdown_findings(summary, top_features, match_quality)
        report_frames = {
            "perps_pre_expansion_feature_comparison.csv": comparison,
            "perps_pre_expansion_matched_effects.csv": matched_effects,
            "perps_pre_expansion_trajectory.csv": trajectory,
            "perps_pre_expansion_chronological_effects.csv": chronological,
            "perps_pre_expansion_stability.csv": stability,
            "perps_pre_expansion_top_features.csv": top_features,
            "perps_pre_expansion_control_match_quality.csv": match_quality,
            "perps_pre_expansion_coverage.csv": coverage,
            "perps_pre_expansion_invariants.csv": invariants,
        }
        report_manifest = write_reports(
            report_root, run_id, report_frames, summary, config, findings
        )
        atomic_write_json(output_root / "_metadata" / "dataset_config.json", config)
        atomic_write_json(output_root / "_metadata" / "latest_run_summary.json", summary)

        connection.execute(
            """
            UPDATE runs SET
                completed_at_utc=?, status='COMPLETED', event_count=?, control_count=?,
                observation_rows=?, feature_count=?, failed_invariants=?, summary_json=?
            WHERE run_id=?
            """,
            (
                utc_now_iso(),
                len(event_catalog),
                len(matches),
                len(observations),
                len(profile_features),
                failed_invariants,
                compact_json(summary),
                run_id,
            ),
        )
        connection.execute(
            """
            UPDATE latest_build SET
                status='COMPLETED', output_manifest_json=?, report_manifest_json=?,
                summary_json=?, error_text=NULL
            WHERE symbol=? AND segmentation_id=?
            """,
            (
                compact_json(output_manifest),
                compact_json(report_manifest),
                compact_json(summary),
                symbol,
                segmentation_id,
            ),
        )
        connection.commit()

        print("COMPLETED")
        print(f"Profiled events    : {len(event_catalog):,}")
        print(f"Matched controls   : {len(matches):,}")
        print(f"Observation rows   : {len(observations):,}")
        print(f"Profiled features  : {len(profile_features):,}")
        print(f"Failed invariants  : {failed_invariants:,}")
        print(f"Findings           : {report_root / 'perps_pre_expansion_findings.md'}")
        return 0

    except Exception as exc:
        error_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        try:
            connection.execute(
                "UPDATE runs SET completed_at_utc=?, status='FAILED', error_text=? WHERE run_id=?",
                (utc_now_iso(), error_text, run_id),
            )
            connection.execute(
                """
                INSERT INTO latest_build (
                    symbol, segmentation_id, status, input_signature, config_signature, error_text
                ) VALUES (?, ?, 'FAILED', '', '', ?)
                ON CONFLICT(symbol, segmentation_id) DO UPDATE SET
                    status='FAILED', error_text=excluded.error_text
                """,
                (symbol, segmentation_id, error_text),
            )
            connection.commit()
        except Exception:
            pass
        print(error_text, file=sys.stderr)
        return 1
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
