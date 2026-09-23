#!/usr/bin/env python
"""
Build and evaluate leakage-safe multi-target expansion baseline models.

Research purpose
----------------
The completed target, segmentation and footprint analyses established a frozen
shortlist of expansion definitions. This script performs the first fair
prediction test:

* sample one causal market snapshot every 30 seconds;
* attach exact future expansion labels from the audited research-label layer;
* train only on earlier dates;
* select alert thresholds on a later chronological calibration block;
* evaluate on still-later unseen dates;
* preserve primary and robustness event definitions for event-level metrics;
* compare identical model families and alert budgets across targets.

The script predicts expansion only. It does not predict direction, simulate a
trade, optimize stops/targets, use random splitting, or modify upstream data.

Frozen default targets
----------------------
Target             Primary events      Robustness events
35 bps / 5m        bridge_030s          bridge_015s
50 bps / 5m        bridge_030s          bridge_015s
50 bps / 10m       bridge_030s          bridge_015s
50 bps / 15m       bridge_060s          bridge_030s
75 bps / 15m       bridge_030s          bridge_015s
75 bps / 20m       bridge_030s          bridge_015s

Causality contract
------------------
For a decision second t, every model feature is read from the audited causal
flow-feature row at t. The target label is read from the separate exact-forward
research layer and is never written into the causal feature dataset.

Outer evaluation uses expanding chronological walk-forward folds. Inside every
outer training period, the most recent calibration days are reserved for alert
threshold selection. The fit/calibration and calibration/test boundaries are
purged by at least the target horizon, preventing future paths on the earlier
side from crossing into the later side.

Event evaluation
----------------
Rolling labels are not treated as independent events. Primary and robustness
event files are loaded from the completed target/segmentation sensitivity layer.
For every event onset, the script reconstructs the exact future price path and
finds the first second at which the target high-low threshold is realised.

A deduplicated model alert can recall an event only when it occurs between that
event's onset and first exact threshold realisation. Alerts and events are
matched one-to-one. Warning time is therefore:

    first_realisation_second - matched_alert_second

This is a real out-of-sample model warning measure, not the mechanical
15-minute oracle-label lead used in earlier descriptive profiling.

Baseline models
---------------
* logistic regression;
* shallow decision tree;
* histogram gradient-boosted trees.

Hyperparameters are fixed globally. No target receives extra tuning effort.

Default outputs
---------------
Dataset root:
    data/derived/perps/research/expansion_baseline_models_v1

Report root:
    reports/perps_expansion_baseline_models

Dependencies
------------
    python -m pip install numpy pandas pyarrow scikit-learn joblib numba

Typical Git Bash usage
----------------------
    python scripts/perps/research/05_build_and_evaluate_expansion_baselines.py
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import re
import sqlite3
import sys
import traceback
import warnings
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args: Any, **kwargs: Any):  # type: ignore[misc]
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(function: Any) -> Any:
            return function

        return decorator


BUILDER_VERSION = "1.0.2"
SCRIPT_RELEASE = "1.0.2-report-fix-1"
OUTPUT_SCHEMA_VERSION = 1
EXPECTED_TARGET_PROFILER_VERSION = "1.0.0"
EXPECTED_TARGET_LABEL_SCHEMA_VERSION = 1
EXPECTED_TARGET_EVENT_SCHEMA_VERSION = 1
SECONDS_PER_DAY = 86_400

DEFAULT_TARGET_PLAN = (
    "H05M_T035BPS=bridge_030s|bridge_015s;"
    "H05M_T050BPS=bridge_030s|bridge_015s;"
    "H10M_T050BPS=bridge_030s|bridge_015s;"
    "H15M_T050BPS=bridge_060s|bridge_030s;"
    "H15M_T075BPS=bridge_030s|bridge_015s;"
    "H20M_T075BPS=bridge_030s|bridge_015s"
)
DEFAULT_MODELS = ("logistic", "shallow_tree", "hist_gradient_boosting")
DEFAULT_ALERT_RATES_PER_DAY = (1.0, 2.0, 4.0, 8.0)
DEFAULT_DECISION_CADENCE_SECONDS = 30
DEFAULT_N_FOLDS = 4
DEFAULT_INITIAL_TRAIN_DAYS = 30
DEFAULT_TEST_DAYS = 8
DEFAULT_CALIBRATION_DAYS = 7
DEFAULT_MIN_FEATURE_COVERAGE = 0.70
DEFAULT_MAX_ABS_CORRELATION = 0.995
DEFAULT_MAX_FEATURES = 192
DEFAULT_THRESHOLD_GRID_SIZE = 161
DEFAULT_RANDOM_STATE = 20260804
DEFAULT_ALERT_COOLDOWN = "HORIZON"

TARGET_PREFIX = "target="
SEGMENTATION_PREFIX = "segmentation="
SYMBOL_PREFIX = "symbol="
DATE_PREFIX = "date="
HORIZON_PREFIX = "horizon="
MODEL_PREFIX = "model="
FOLD_PREFIX = "fold="

_TARGET_RE = re.compile(r"^H(?P<horizon>\d{2,3})M_T(?P<threshold>\d{3}(?:P\d+)?)BPS$")
_SEGMENTATION_RE = re.compile(r"^bridge_(?P<gap>\d{3,4})s$")

REQUIRED_HORIZON_LABEL_COLUMNS = (
    "symbol",
    "decision_second",
    "future_swing_bps",
)
REQUIRED_TARGET_EVENT_COLUMNS = (
    "event_schema_version",
    "profiler_version",
    "event_id",
    "target_id",
    "symbol",
    "horizon_seconds",
    "horizon_minutes",
    "threshold_bps",
    "segmentation_id",
    "event_start_second",
    "event_end_second",
    "event_duration_seconds",
    "number_of_positive_anchors",
    "maximum_future_swing_bps",
    "path_type",
)

REPORT_FILENAMES = (
    "perps_expansion_baseline_folds.csv",
    "perps_expansion_baseline_fold_metrics.csv",
    "perps_expansion_baseline_alert_metrics.csv",
    "perps_expansion_baseline_event_metrics.csv",
    "perps_expansion_baseline_model_summary.csv",
    "perps_expansion_baseline_target_scorecard.csv",
    "perps_expansion_baseline_calibration.csv",
    "perps_expansion_baseline_feature_selection.csv",
    "perps_expansion_baseline_feature_importance.csv",
    "perps_expansion_baseline_invariants.csv",
    "perps_expansion_baseline_summary.json",
    "perps_expansion_baseline_config.json",
    "perps_expansion_baseline_findings.md",
)


# ---------------------------------------------------------------------------
# Shared audited feature methodology
# ---------------------------------------------------------------------------


def load_feature_core() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "participant_behaviour"
        / "01_profile_pre_expansion_footprints.py"
    )
    if not path.exists():
        candidates = (
            Path(__file__).resolve().with_name("01_profile_pre_expansion_footprints.py"),
            Path.cwd() / "01_profile_pre_expansion_footprints.py",
        )
        fallback = next((candidate for candidate in candidates if candidate.exists()), None)
        if fallback is not None:
            path = fallback.resolve()
        else:
            raise RuntimeError(
                "Required feature-methodology script was not found: "
                "scripts/perps/participant_behaviour/01_profile_pre_expansion_footprints.py"
            )
    module_name = "stream_research_expansion_baseline_feature_core"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load feature methodology from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_feature_core()


# ---------------------------------------------------------------------------
# Data classes and parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetPlan:
    target_id: str
    horizon_minutes: int
    horizon_seconds: int
    threshold_bps: float
    primary_segmentation_id: str
    robustness_segmentation_id: str


@dataclass(frozen=True)
class FoldSpec:
    fold_id: str
    fold_ordinal: int
    fit_start_second: int
    fit_end_exclusive_second: int
    calibration_start_second: int
    calibration_end_exclusive_second: int
    test_start_second: int
    test_end_exclusive_second: int
    fit_start_date: date
    fit_end_date: date
    calibration_start_date: date
    calibration_end_date: date
    test_start_date: date
    test_end_date: date
    outer_train_days: int
    fit_days: int
    calibration_days: int
    test_days: int


@dataclass(frozen=True)
class ParquetPartition:
    day: date
    files: tuple[Path, ...]
    rows: int
    signature: str


@dataclass(frozen=True)
class ModelResultPaths:
    predictions: Path
    model: Path
    result_json: Path


def parse_target_id(value: str) -> tuple[int, float]:
    match = _TARGET_RE.fullmatch(value.strip().upper())
    if not match:
        raise ValueError(f"Invalid target ID: {value!r}")
    horizon = int(match.group("horizon"))
    threshold = float(match.group("threshold").replace("P", "."))
    if horizon <= 0 or threshold <= 0:
        raise ValueError(f"Target values must be positive: {value!r}")
    return horizon, threshold


def parse_target_plan(value: str) -> tuple[TargetPlan, ...]:
    plans: list[TargetPlan] = []
    seen: set[str] = set()
    for chunk in value.split(";"):
        cleaned = chunk.strip()
        if not cleaned:
            continue
        if "=" not in cleaned or "|" not in cleaned:
            raise ValueError(
                "Target plan entries must look like "
                "H05M_T035BPS=bridge_030s|bridge_015s"
            )
        target_id, rules = cleaned.split("=", 1)
        primary, robustness = [item.strip() for item in rules.split("|", 1)]
        target_id = target_id.strip().upper()
        horizon, threshold = parse_target_id(target_id)
        for segmentation in (primary, robustness):
            if not _SEGMENTATION_RE.fullmatch(segmentation):
                raise ValueError(f"Invalid bridge segmentation: {segmentation!r}")
        if target_id in seen:
            raise ValueError(f"Duplicate target in plan: {target_id}")
        seen.add(target_id)
        plans.append(
            TargetPlan(
                target_id=target_id,
                horizon_minutes=horizon,
                horizon_seconds=horizon * 60,
                threshold_bps=threshold,
                primary_segmentation_id=primary,
                robustness_segmentation_id=robustness,
            )
        )
    if not plans:
        raise ValueError("Target plan is empty")
    return tuple(plans)


def parse_targets_filter(value: str | None, plans: Sequence[TargetPlan]) -> tuple[TargetPlan, ...]:
    if not value:
        return tuple(plans)
    requested = {item.strip().upper() for item in value.split(",") if item.strip()}
    known = {plan.target_id for plan in plans}
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"Targets are not in the frozen plan: {unknown}")
    output = tuple(plan for plan in plans if plan.target_id in requested)
    if not output:
        raise ValueError("No targets remain after filtering")
    return output


def parse_models(value: str) -> tuple[str, ...]:
    models = tuple(dict.fromkeys(item.strip().lower() for item in value.split(",") if item.strip()))
    invalid = sorted(set(models) - set(DEFAULT_MODELS))
    if invalid:
        raise ValueError(f"Unknown models: {invalid}")
    if not models:
        raise ValueError("At least one model is required")
    return models


def parse_float_list(value: str, name: str) -> tuple[float, ...]:
    output = tuple(sorted({float(item.strip()) for item in value.split(",") if item.strip()}))
    if not output or any(item <= 0 for item in output):
        raise ValueError(f"{name} must contain positive numbers")
    return output


def parse_date(value: str | None, name: str) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise SystemExit(f"{name} must use YYYY-MM-DD") from exc


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def day_start_second(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())


def percent(numerator: float | int, denominator: float | int) -> float:
    return float("nan") if not denominator else 100.0 * float(numerator) / float(denominator)


def safe_mean(values: Iterable[Any]) -> float:
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if len(array) else float("nan")


def safe_median(values: Iterable[Any]) -> float:
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if len(array) else float("nan")


# ---------------------------------------------------------------------------
# Paths and input discovery
# ---------------------------------------------------------------------------


def horizon_label_root(target_root: Path, horizon_minutes: int) -> Path:
    return target_root / "row_labels" / f"{HORIZON_PREFIX}{horizon_minutes:02d}m"


def target_event_path(
    target_root: Path,
    target: TargetPlan,
    segmentation_id: str,
    symbol: str,
) -> Path:
    return (
        target_root
        / "events"
        / f"{TARGET_PREFIX}{target.target_id}"
        / f"{SEGMENTATION_PREFIX}{segmentation_id}"
        / f"{SYMBOL_PREFIX}{symbol}"
        / "part-00000.parquet"
    )


def snapshot_path(output_root: Path, symbol: str, day: date) -> Path:
    return (
        output_root
        / "snapshots"
        / f"{SYMBOL_PREFIX}{symbol}"
        / f"{DATE_PREFIX}{day.isoformat()}"
        / "part-00000.parquet"
    )


def model_result_paths(
    output_root: Path,
    target_id: str,
    model_id: str,
    fold_id: str,
) -> ModelResultPaths:
    root = (
        output_root
        / "experiments"
        / f"{TARGET_PREFIX}{target_id}"
        / f"{MODEL_PREFIX}{model_id}"
        / f"{FOLD_PREFIX}{fold_id}"
    )
    return ModelResultPaths(
        predictions=root / "predictions.parquet",
        model=root / "model.joblib",
        result_json=root / "result.json",
    )


def discover_horizon_label_partitions(
    target_root: Path,
    horizon_minutes: int,
    symbol: str,
    hash_input_files: bool,
) -> dict[date, ParquetPartition]:
    root = horizon_label_root(target_root, horizon_minutes) / f"{SYMBOL_PREFIX}{symbol}"
    files = sorted(root.glob(f"{DATE_PREFIX}*/*.parquet"))
    if not files:
        raise RuntimeError(f"No horizon-label Parquet files found under {root}")
    grouped: dict[date, list[Path]] = {}
    for path in files:
        grouped.setdefault(CORE.extract_partition_day(path), []).append(path)
    output: dict[date, ParquetPartition] = {}
    for day, paths in sorted(grouped.items()):
        signatures = [CORE.file_signature(path, hash_input_files) for path in sorted(paths)]
        output[day] = ParquetPartition(
            day=day,
            files=tuple(sorted(paths)),
            rows=int(sum(int(item["rows"]) for item in signatures)),
            signature=sha256_text(compact_json(signatures)),
        )
    validate_horizon_label_schema(output, horizon_minutes, symbol)
    return output


def validate_horizon_label_schema(
    partitions: Mapping[date, ParquetPartition],
    horizon_minutes: int,
    symbol: str,
) -> None:
    pq_module = CORE.require_arrow()[2]
    first_file = next(iter(partitions.values())).files[0]
    parquet_file = pq_module.ParquetFile(first_file)
    try:
        names = set(parquet_file.schema_arrow.names)
        missing = sorted(set(REQUIRED_HORIZON_LABEL_COLUMNS) - names)
        if missing:
            raise RuntimeError(f"Horizon-label file {first_file} is missing columns: {missing}")
        metadata = CORE.decode_metadata(parquet_file.schema_arrow)
    finally:
        parquet_file.close()
    if metadata.get("perps_expansion_target_profiler_version") != EXPECTED_TARGET_PROFILER_VERSION:
        raise RuntimeError(
            f"Target profiler version in {first_file} is "
            f"{metadata.get('perps_expansion_target_profiler_version')!r}; "
            f"expected {EXPECTED_TARGET_PROFILER_VERSION!r}"
        )
    if metadata.get("perps_expansion_horizon_label_schema_version") != str(
        EXPECTED_TARGET_LABEL_SCHEMA_VERSION
    ):
        raise RuntimeError(
            f"Horizon-label schema version in {first_file} is "
            f"{metadata.get('perps_expansion_horizon_label_schema_version')!r}; "
            f"expected {EXPECTED_TARGET_LABEL_SCHEMA_VERSION}"
        )
    metadata_horizon = metadata.get("horizon_seconds")
    if metadata_horizon and int(metadata_horizon) != horizon_minutes * 60:
        raise RuntimeError(
            f"Horizon metadata in {first_file} is {metadata_horizon}; "
            f"expected {horizon_minutes * 60}"
        )
    metadata_symbol = metadata.get("symbol")
    if metadata_symbol and metadata_symbol.upper() != symbol:
        raise RuntimeError(f"Label metadata symbol is {metadata_symbol!r}; expected {symbol}")


def read_target_events(
    path: Path,
    target: TargetPlan,
    segmentation_id: str,
    symbol: str,
) -> tuple[pd.DataFrame, str]:
    pq_module = CORE.require_arrow()[2]
    if not path.exists():
        raise RuntimeError(f"Target event file does not exist: {path}")
    parquet_file = pq_module.ParquetFile(path)
    try:
        names = set(parquet_file.schema_arrow.names)
        missing = sorted(set(REQUIRED_TARGET_EVENT_COLUMNS) - names)
        if missing:
            raise RuntimeError(f"Target event file {path} is missing columns: {missing}")
        metadata = CORE.decode_metadata(parquet_file.schema_arrow)
        frame = parquet_file.read(columns=list(REQUIRED_TARGET_EVENT_COLUMNS)).to_pandas()
    finally:
        parquet_file.close()
    if metadata.get("perps_expansion_target_profiler_version") != EXPECTED_TARGET_PROFILER_VERSION:
        raise RuntimeError(
            f"Event profiler version in {path} is "
            f"{metadata.get('perps_expansion_target_profiler_version')!r}; "
            f"expected {EXPECTED_TARGET_PROFILER_VERSION!r}"
        )
    if metadata.get("perps_expansion_event_schema_version") != str(
        EXPECTED_TARGET_EVENT_SCHEMA_VERSION
    ):
        raise RuntimeError(
            f"Event schema version in {path} is "
            f"{metadata.get('perps_expansion_event_schema_version')!r}; "
            f"expected {EXPECTED_TARGET_EVENT_SCHEMA_VERSION}"
        )
    frame = frame[
        frame["symbol"].astype(str).str.upper().eq(symbol)
        & frame["target_id"].astype(str).eq(target.target_id)
        & frame["segmentation_id"].astype(str).eq(segmentation_id)
    ].copy()
    if frame.empty:
        raise RuntimeError(f"No rows match {target.target_id}/{segmentation_id} in {path}")
    for column in ("event_start_second", "event_end_second", "horizon_seconds"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.int64)
    frame["threshold_bps"] = pd.to_numeric(frame["threshold_bps"], errors="raise")
    if not frame["horizon_seconds"].eq(target.horizon_seconds).all():
        raise RuntimeError(f"Event rows do not all use horizon {target.horizon_seconds}")
    if not np.allclose(
        frame["threshold_bps"].to_numpy(dtype=float),
        target.threshold_bps,
        atol=1e-8,
        rtol=0.0,
    ):
        raise RuntimeError(f"Event rows do not all use threshold {target.threshold_bps}")
    frame = frame.sort_values("event_start_second", kind="mergesort").reset_index(drop=True)
    if frame["event_id"].duplicated().any() or frame["event_start_second"].duplicated().any():
        raise RuntimeError("Event IDs and event-start seconds must be unique")
    signature = sha256_text(compact_json(CORE.file_signature(path, False)))
    return frame, signature


# ---------------------------------------------------------------------------
# SQLite state and atomic model writes
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
            configuration_json TEXT NOT NULL,
            rebuilt_snapshots INTEGER NOT NULL DEFAULT 0,
            skipped_snapshots INTEGER NOT NULL DEFAULT 0,
            completed_model_folds INTEGER NOT NULL DEFAULT 0,
            skipped_model_folds INTEGER NOT NULL DEFAULT 0,
            failed_model_folds INTEGER NOT NULL DEFAULT 0,
            error_text TEXT
        );

        CREATE TABLE IF NOT EXISTS snapshot_partitions (
            symbol TEXT NOT NULL,
            decision_date TEXT NOT NULL,
            input_signature TEXT NOT NULL,
            output_path TEXT NOT NULL,
            row_count INTEGER,
            status TEXT NOT NULL,
            completed_at_utc TEXT,
            error_text TEXT,
            last_run_id TEXT NOT NULL,
            PRIMARY KEY(symbol, decision_date)
        );

        CREATE TABLE IF NOT EXISTS model_folds (
            target_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            fold_id TEXT NOT NULL,
            input_signature TEXT NOT NULL,
            prediction_path TEXT NOT NULL,
            model_path TEXT NOT NULL,
            result_path TEXT NOT NULL,
            status TEXT NOT NULL,
            completed_at_utc TEXT,
            error_text TEXT,
            last_run_id TEXT NOT NULL,
            PRIMARY KEY(target_id, model_id, fold_id)
        );
        """
    )
    return connection


def replace_with_retry(source: Path, destination: Path, attempts: int = 5) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            import time
            time.sleep(0.2 * (attempt + 1))


def atomic_joblib_dump(path: Path, value: Any) -> None:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        joblib.dump(value, temporary, compress=3)
        if not temporary.exists() or temporary.stat().st_size <= 0:
            raise RuntimeError(f"Temporary model file is empty: {temporary}")
        replace_with_retry(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def validate_parquet_rows(path: Path, expected_rows: int | None = None) -> int:
    pq_module = CORE.require_arrow()[2]
    parquet_file = pq_module.ParquetFile(path)
    try:
        rows = int(parquet_file.metadata.num_rows)
    finally:
        parquet_file.close()
    if expected_rows is not None and rows != expected_rows:
        raise RuntimeError(f"Parquet row mismatch for {path}: {rows:,} != {expected_rows:,}")
    return rows


# ---------------------------------------------------------------------------
# Snapshot feature engineering
# ---------------------------------------------------------------------------


def safe_log_intensity(frame: pd.DataFrame, column: str, seconds: int) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce").astype(float) / float(seconds)
    return np.log1p(np.clip(values, 0.0, None))


def safe_log_nonnegative(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce").astype(float)
    return np.log1p(np.clip(values, 0.0, None))


def add_difference(
    frame: pd.DataFrame,
    output: dict[str, pd.Series],
    name: str,
    left: str,
    right: str,
) -> None:
    if left in frame.columns and right in frame.columns:
        output[name] = (
            pd.to_numeric(frame[left], errors="coerce").astype(float)
            - pd.to_numeric(frame[right], errors="coerce").astype(float)
        )


def collapse_equivalent_duplicate_columns(
    frame: pd.DataFrame,
    context: str,
) -> pd.DataFrame:
    """Collapse only byte/logically equivalent duplicate columns.

    Pandas permits duplicate column names, while PyArrow correctly rejects
    them.  This final defensive boundary keeps the first copy only when all
    copies are equivalent; conflicting duplicates fail loudly.
    """
    if frame.columns.is_unique:
        return frame

    keep_positions: list[int] = []
    seen: set[str] = set()
    for position, column in enumerate(frame.columns):
        name = str(column)
        if name in seen:
            continue
        positions = [
            index for index, candidate in enumerate(frame.columns)
            if str(candidate) == name
        ]
        reference = frame.iloc[:, positions[0]].reset_index(drop=True)
        for duplicate_position in positions[1:]:
            candidate = frame.iloc[:, duplicate_position].reset_index(drop=True)
            try:
                pd.testing.assert_series_equal(
                    reference,
                    candidate,
                    check_dtype=False,
                    check_names=False,
                )
            except AssertionError as exc:
                raise RuntimeError(
                    f"Conflicting duplicate column {name!r} in {context}"
                ) from exc
        keep_positions.append(positions[0])
        seen.add(name)

    collapsed = frame.iloc[:, keep_positions].copy()
    if not collapsed.columns.is_unique:
        duplicates = collapsed.columns[collapsed.columns.duplicated()].tolist()
        raise RuntimeError(
            f"Duplicate columns remain after collapse in {context}: {duplicates}"
        )
    return collapsed


def derive_hypothesis_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    output: dict[str, pd.Series] = {}

    seconds = pd.to_numeric(frame["receive_second"], errors="raise").to_numpy(dtype=np.int64)
    day_fraction = (seconds % SECONDS_PER_DAY) / float(SECONDS_PER_DAY)
    day_index = ((seconds // SECONDS_PER_DAY) + 4) % 7  # 1970-01-01 was Thursday.
    output["utc_time_sin"] = pd.Series(np.sin(2.0 * np.pi * day_fraction), index=frame.index)
    output["utc_time_cos"] = pd.Series(np.cos(2.0 * np.pi * day_fraction), index=frame.index)
    output["utc_weekday_sin"] = pd.Series(np.sin(2.0 * np.pi * day_index / 7.0), index=frame.index)
    output["utc_weekday_cos"] = pd.Series(np.cos(2.0 * np.pi * day_index / 7.0), index=frame.index)

    absolute_prefixes = (
        "return_",
        "signed_quote_flow_",
        "signed_base_flow_",
        "aggressive_flow_imbalance_",
        "net_liquidation_notional_",
        "open_interest_change_",
        "observed_open_interest_change_",
        "price_flow_alignment_",
        "price_response_bps_per_1m_signed_flow_",
    )
    for column in frame.columns:
        if column.startswith(absolute_prefixes) and pd.api.types.is_numeric_dtype(frame[column]):
            output[f"abs_{column}"] = pd.to_numeric(frame[column], errors="coerce").abs()

    for short, long in ((5, 60), (15, 300), (30, 300), (60, 900), (300, 3600)):
        for prefix in ("trade_count", "quote_volume", "liquidation_notional"):
            left = f"{prefix}_{short}s"
            right = f"{prefix}_{long}s"
            if left in frame.columns and right in frame.columns:
                output[f"{prefix}_intensity_logdiff_{short}s_{long}s"] = (
                    safe_log_intensity(frame, left, short)
                    - safe_log_intensity(frame, right, long)
                )
        left_rv = f"realized_vol_{short}s_bps"
        right_rv = f"realized_vol_{long}s_bps"
        if left_rv in frame.columns and right_rv in frame.columns:
            output[f"realized_vol_logratio_{short}s_{long}s"] = (
                safe_log_nonnegative(frame, left_rv)
                - safe_log_nonnegative(frame, right_rv)
            )
        left_imb = f"aggressive_flow_imbalance_{short}s"
        right_imb = f"aggressive_flow_imbalance_{long}s"
        if left_imb in frame.columns and right_imb in frame.columns:
            output[f"abs_flow_imbalance_change_{short}s_{long}s"] = (
                pd.to_numeric(frame[left_imb], errors="coerce").abs()
                - pd.to_numeric(frame[right_imb], errors="coerce").abs()
            )
        left_liq = f"liquidation_to_quote_ratio_{short}s"
        right_liq = f"liquidation_to_quote_ratio_{long}s"
        if left_liq in frame.columns and right_liq in frame.columns:
            output[f"liquidation_ratio_logdiff_{short}s_{long}s"] = (
                safe_log_nonnegative(frame, left_liq)
                - safe_log_nonnegative(frame, right_liq)
            )

    for short, long in ((60, 300), (300, 900), (300, 3600), (900, 3600)):
        left = f"open_interest_change_{short}s_pct"
        right = f"open_interest_change_{long}s_pct"
        add_difference(frame, output, f"oi_pct_short_minus_long_{short}s_{long}s", left, right)
        if left in frame.columns and right in frame.columns:
            short_values = pd.to_numeric(frame[left], errors="coerce").astype(float)
            long_values = pd.to_numeric(frame[right], errors="coerce").astype(float)
            output[f"oi_buildup_then_unwind_{long}s_{short}s"] = long_values - short_values
            output[f"oi_positive_long_negative_short_{long}s_{short}s"] = (
                (long_values > 0.0) & (short_values < 0.0)
            ).astype(float)

    for short, long in ((5, 60), (30, 300), (60, 900)):
        left = f"price_response_bps_per_1m_signed_flow_{short}s"
        right = f"price_response_bps_per_1m_signed_flow_{long}s"
        if left in frame.columns and right in frame.columns:
            output[f"abs_price_response_change_{short}s_{long}s"] = (
                pd.to_numeric(frame[left], errors="coerce").abs()
                - pd.to_numeric(frame[right], errors="coerce").abs()
            )

    # Some audited upstream features already contain an exact version of a
    # hypothesis feature (currently ``abs_return_1s_bps``).  Re-adding that
    # name through ``pd.concat`` creates duplicate DataFrame columns, which
    # PyArrow correctly refuses to serialize.  Reuse the audited upstream
    # column when it is equivalent and materialize only genuinely new names.
    collisions = sorted(set(output).intersection(frame.columns))
    for column in collisions:
        existing = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        candidate = pd.to_numeric(output[column], errors="coerce").to_numpy(dtype=float)
        if not np.allclose(existing, candidate, rtol=1e-10, atol=1e-12, equal_nan=True):
            raise RuntimeError(
                f"Derived feature {column!r} conflicts with the audited upstream column"
            )
        del output[column]

    derived = pd.DataFrame(output, index=frame.index)
    frame = pd.concat([frame, derived], axis=1)
    if not frame.columns.is_unique:
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise RuntimeError(f"Duplicate snapshot feature columns after derivation: {duplicates}")
    return frame, list(derived.columns)


def build_snapshot_partition(
    partition: Any,
    output_path: Path,
    symbol: str,
    raw_features: Sequence[str],
    cadence_seconds: int,
    input_signature: str,
    compression_level: int,
) -> tuple[int, list[str]]:
    pq_module = CORE.require_arrow()[2]
    reserved_columns = {"symbol", "receive_second", "flow_feature_context_ready"}
    snapshot_raw_features = list(
        dict.fromkeys(column for column in raw_features if column not in reserved_columns)
    )
    columns = [
        "symbol",
        "receive_second",
        "flow_feature_context_ready",
        *snapshot_raw_features,
    ]
    frames: list[pd.DataFrame] = []
    for path in partition.files:
        parquet_file = pq_module.ParquetFile(path)
        try:
            frames.append(parquet_file.read(columns=columns).to_pandas())
        finally:
            parquet_file.close()
    frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    frame = frame[
        frame["symbol"].astype(str).str.upper().eq(symbol)
        & frame["flow_feature_context_ready"].fillna(False).astype(bool)
    ].copy()
    frame["receive_second"] = pd.to_numeric(frame["receive_second"], errors="raise").astype(np.int64)
    frame = frame[(frame["receive_second"] % int(cadence_seconds)) == 0].copy()
    frame = frame.sort_values("receive_second", kind="mergesort").reset_index(drop=True)
    if frame["receive_second"].duplicated().any():
        raise RuntimeError(f"Duplicate snapshot seconds for {partition.day}")
    frame, derived_features = derive_hypothesis_features(frame)
    frame.insert(2, "decision_timestamp_utc", pd.to_datetime(frame["receive_second"], unit="s", utc=True))
    frame.insert(3, "decision_date", partition.day.isoformat())
    frame = frame.drop(columns=["flow_feature_context_ready"])
    frame = collapse_equivalent_duplicate_columns(
        frame, context=f"snapshot partition {partition.day}"
    )
    if not frame.columns.is_unique:
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise RuntimeError(
            f"Duplicate snapshot columns for {partition.day}: {duplicates}"
        )
    metadata = {
        "perps_expansion_baseline_builder_version": BUILDER_VERSION,
        "perps_expansion_baseline_snapshot_schema_version": OUTPUT_SCHEMA_VERSION,
        "symbol": symbol,
        "decision_date": partition.day.isoformat(),
        "decision_cadence_seconds": cadence_seconds,
        "input_signature": input_signature,
        "raw_feature_count": len(snapshot_raw_features),
        "derived_feature_count": len(derived_features),
    }
    CORE.atomic_write_parquet(output_path, frame, metadata, compression_level)
    validate_parquet_rows(output_path, len(frame))
    return len(frame), derived_features


def snapshot_is_reusable(path: Path, input_signature: str, expected_rows: int | None) -> bool:
    if not path.exists():
        return False
    pq_module = CORE.require_arrow()[2]
    parquet_file = pq_module.ParquetFile(path)
    try:
        metadata = CORE.decode_metadata(parquet_file.schema_arrow)
        rows = int(parquet_file.metadata.num_rows)
    finally:
        parquet_file.close()
    return (
        metadata.get("perps_expansion_baseline_builder_version") == BUILDER_VERSION
        and metadata.get("input_signature") == input_signature
        and (expected_rows is None or rows == expected_rows)
    )


def load_snapshots(paths: Sequence[Path]) -> pd.DataFrame:
    pq_module = CORE.require_arrow()[2]
    frames: list[pd.DataFrame] = []
    for path in paths:
        parquet_file = pq_module.ParquetFile(path)
        try:
            frames.append(parquet_file.read().to_pandas())
        finally:
            parquet_file.close()
    frame = pd.concat(frames, ignore_index=True)
    frame["receive_second"] = pd.to_numeric(frame["receive_second"], errors="raise").astype(np.int64)
    frame = frame.sort_values("receive_second", kind="mergesort").reset_index(drop=True)
    if frame["receive_second"].duplicated().any():
        raise RuntimeError("Snapshot seconds are not globally unique")
    return frame


# ---------------------------------------------------------------------------
# Labels, exact prices and event realization
# ---------------------------------------------------------------------------


def load_target_labels(
    partitions: Mapping[date, ParquetPartition],
    target: TargetPlan,
    cadence_seconds: int,
    selected_days: set[date],
    symbol: str,
) -> pd.DataFrame:
    pq_module = CORE.require_arrow()[2]
    frames: list[pd.DataFrame] = []
    for day, partition in sorted(partitions.items()):
        if day not in selected_days:
            continue
        for path in partition.files:
            parquet_file = pq_module.ParquetFile(path)
            try:
                frame = parquet_file.read(columns=list(REQUIRED_HORIZON_LABEL_COLUMNS)).to_pandas()
            finally:
                parquet_file.close()
            frames.append(frame)
    if not frames:
        raise RuntimeError(f"No target labels found for {target.target_id}")
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame["symbol"].astype(str).str.upper().eq(symbol)].copy()
    frame["decision_second"] = pd.to_numeric(frame["decision_second"], errors="raise").astype(np.int64)
    frame["future_swing_bps"] = pd.to_numeric(frame["future_swing_bps"], errors="coerce")
    frame = frame[
        np.isfinite(frame["future_swing_bps"].to_numpy(dtype=float))
        & ((frame["decision_second"] % int(cadence_seconds)) == 0)
    ].copy()
    frame["target"] = (
        frame["future_swing_bps"].to_numpy(dtype=float) + 1e-10 >= target.threshold_bps
    ).astype(np.int8)
    frame = frame[["decision_second", "future_swing_bps", "target"]]
    frame = frame.sort_values("decision_second", kind="mergesort").reset_index(drop=True)
    if frame["decision_second"].duplicated().any():
        raise RuntimeError(f"Duplicate label seconds for {target.target_id}")
    return frame


def load_exact_price_series(
    feature_partitions: Mapping[date, Any],
    required_start_second: int,
    required_end_second: int,
    symbol: str,
) -> tuple[np.ndarray, np.ndarray]:
    pq_module = CORE.require_arrow()[2]
    frames: list[pd.DataFrame] = []
    start_day = datetime.fromtimestamp(required_start_second, tz=timezone.utc).date()
    end_day = datetime.fromtimestamp(required_end_second, tz=timezone.utc).date()
    for day, partition in sorted(feature_partitions.items()):
        if day < start_day or day > end_day:
            continue
        for path in partition.files:
            parquet_file = pq_module.ParquetFile(path)
            try:
                frame = parquet_file.read(columns=["symbol", "receive_second", "price_for_features"]).to_pandas()
            finally:
                parquet_file.close()
            frames.append(frame)
    if not frames:
        raise RuntimeError("No exact prices were loaded for event-realization reconstruction")
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame["symbol"].astype(str).str.upper().eq(symbol)].copy()
    frame["receive_second"] = pd.to_numeric(frame["receive_second"], errors="raise").astype(np.int64)
    frame["price_for_features"] = pd.to_numeric(frame["price_for_features"], errors="coerce")
    frame = frame[
        frame["receive_second"].between(required_start_second, required_end_second, inclusive="both")
    ].copy()
    frame = frame.sort_values("receive_second", kind="mergesort").drop_duplicates(
        "receive_second", keep="last"
    )
    return (
        frame["receive_second"].to_numpy(dtype=np.int64),
        frame["price_for_features"].to_numpy(dtype=float),
    )


@njit(cache=True)
def first_threshold_realization_offset(prices: np.ndarray, threshold_bps: float) -> int:
    if len(prices) == 0:
        return -1
    entry = prices[0]
    if not np.isfinite(entry) or entry <= 0.0:
        return -1
    running_high = entry
    running_low = entry
    for index in range(len(prices)):
        value = prices[index]
        if not np.isfinite(value) or value <= 0.0:
            return -1
        if value > running_high:
            running_high = value
        if value < running_low:
            running_low = value
        swing = (running_high - running_low) / entry * 10_000.0
        if swing + 1e-10 >= threshold_bps:
            return index
    return -1


def enrich_event_realizations(
    events: pd.DataFrame,
    target: TargetPlan,
    exact_seconds: np.ndarray,
    exact_prices: np.ndarray,
    cadence_seconds: int,
) -> pd.DataFrame:
    output = events.copy()
    realization_seconds: list[Any] = []
    exact_path_flags: list[bool] = []
    first_model_seconds: list[Any] = []
    scorable_flags: list[bool] = []
    for start_value in output["event_start_second"].to_numpy(dtype=np.int64):
        path_start = int(start_value) + 1
        path_end = path_start + target.horizon_seconds
        left = int(np.searchsorted(exact_seconds, path_start, side="left"))
        right = int(np.searchsorted(exact_seconds, path_end, side="right"))
        expected = target.horizon_seconds + 1
        exact = (
            right - left == expected
            and left < len(exact_seconds)
            and right <= len(exact_seconds)
            and exact_seconds[left] == path_start
            and exact_seconds[right - 1] == path_end
            and (expected <= 1 or np.all(np.diff(exact_seconds[left:right]) == 1))
        )
        exact_path_flags.append(bool(exact))
        if not exact:
            realization_seconds.append(pd.NA)
            first_model_seconds.append(pd.NA)
            scorable_flags.append(False)
            continue
        offset = int(first_threshold_realization_offset(exact_prices[left:right], target.threshold_bps))
        realization = path_start + offset if offset >= 0 else None
        realization_seconds.append(realization if realization is not None else pd.NA)
        first_model = ((int(start_value) + cadence_seconds - 1) // cadence_seconds) * cadence_seconds
        first_model_seconds.append(first_model)
        scorable_flags.append(realization is not None and first_model <= realization)
    output["exact_realization_path"] = exact_path_flags
    output["first_realization_second"] = pd.array(realization_seconds, dtype="Int64")
    output["first_model_decision_second"] = pd.array(first_model_seconds, dtype="Int64")
    output["scorable_event"] = scorable_flags
    output["realization_delay_seconds"] = (
        output["first_realization_second"] - output["event_start_second"]
    ).astype("Int64")
    return output


# ---------------------------------------------------------------------------
# Chronological folds
# ---------------------------------------------------------------------------


def build_folds(
    days: Sequence[date],
    n_folds: int,
    initial_train_days: int,
    test_days: int,
    calibration_days: int,
    horizon_seconds: int,
) -> tuple[FoldSpec, ...]:
    unique_days = sorted(set(days))
    required = initial_train_days + n_folds * test_days
    if len(unique_days) < required:
        raise ValueError(
            f"Need at least {required} dates for {n_folds} folds, but only "
            f"{len(unique_days)} are available. Adjust --n-folds, "
            "--initial-train-days or --test-days."
        )
    if calibration_days <= 0 or calibration_days >= initial_train_days:
        raise ValueError("Calibration days must be positive and smaller than initial training days")
    folds: list[FoldSpec] = []
    for ordinal in range(n_folds):
        test_start_index = initial_train_days + ordinal * test_days
        test_end_index = test_start_index + test_days
        outer_train = unique_days[:test_start_index]
        test = unique_days[test_start_index:test_end_index]
        calibration = outer_train[-calibration_days:]
        fit = outer_train[:-calibration_days]
        calibration_start = day_start_second(calibration[0])
        test_start = day_start_second(test[0])
        fit_end_exclusive = calibration_start - horizon_seconds
        calibration_end_exclusive = test_start - horizon_seconds
        if fit_end_exclusive <= day_start_second(fit[0]):
            raise ValueError(f"Fold {ordinal + 1} has no fit interval after horizon purge")
        if calibration_end_exclusive <= calibration_start:
            raise ValueError(f"Fold {ordinal + 1} has no calibration interval after horizon purge")
        test_end_exclusive = day_start_second(test[-1] + timedelta(days=1))
        folds.append(
            FoldSpec(
                fold_id=f"{ordinal + 1:02d}",
                fold_ordinal=ordinal + 1,
                fit_start_second=day_start_second(fit[0]),
                fit_end_exclusive_second=fit_end_exclusive,
                calibration_start_second=calibration_start,
                calibration_end_exclusive_second=calibration_end_exclusive,
                test_start_second=test_start,
                test_end_exclusive_second=test_end_exclusive,
                fit_start_date=fit[0],
                fit_end_date=datetime.fromtimestamp(fit_end_exclusive - 1, tz=timezone.utc).date(),
                calibration_start_date=calibration[0],
                calibration_end_date=datetime.fromtimestamp(
                    calibration_end_exclusive - 1, tz=timezone.utc
                ).date(),
                test_start_date=test[0],
                test_end_date=test[-1],
                outer_train_days=len(outer_train),
                fit_days=len(fit),
                calibration_days=len(calibration),
                test_days=len(test),
            )
        )
    return tuple(folds)


def fold_frame(folds: Sequence[FoldSpec], target: TargetPlan) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "target_id": target.target_id,
                "horizon_seconds": target.horizon_seconds,
                "fold_id": fold.fold_id,
                "fold_ordinal": fold.fold_ordinal,
                "fit_start_date": fold.fit_start_date.isoformat(),
                "fit_end_date": fold.fit_end_date.isoformat(),
                "calibration_start_date": fold.calibration_start_date.isoformat(),
                "calibration_end_date": fold.calibration_end_date.isoformat(),
                "test_start_date": fold.test_start_date.isoformat(),
                "test_end_date": fold.test_end_date.isoformat(),
                "outer_train_days": fold.outer_train_days,
                "fit_days": fold.fit_days,
                "calibration_days": fold.calibration_days,
                "test_days": fold.test_days,
                "fit_end_exclusive_second": fold.fit_end_exclusive_second,
                "calibration_end_exclusive_second": fold.calibration_end_exclusive_second,
                "test_start_second": fold.test_start_second,
                "test_end_exclusive_second": fold.test_end_exclusive_second,
            }
            for fold in folds
        ]
    )


# ---------------------------------------------------------------------------
# Feature selection and models
# ---------------------------------------------------------------------------


def candidate_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "symbol",
        "receive_second",
        "decision_timestamp_utc",
        "decision_date",
        "future_swing_bps",
        "target",
    }
    columns: list[str] = []
    for column in frame.columns:
        lowered = column.lower()
        if column in excluded:
            continue
        if any(token in lowered for token in CORE.FUTURE_OR_LABEL_TOKENS):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    return sorted(set(columns))


def select_fold_features(
    frame: pd.DataFrame,
    features: Sequence[str],
    minimum_coverage: float,
    maximum_abs_correlation: float,
    max_features: int,
) -> tuple[list[str], pd.DataFrame]:
    if frame.empty:
        raise RuntimeError("Cannot select features from an empty fit frame")
    records: list[dict[str, Any]] = []
    retained: list[str] = []
    for feature in features:
        values = pd.to_numeric(frame[feature], errors="coerce")
        coverage = float(values.notna().mean())
        finite = values[np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))]
        unique = int(finite.nunique(dropna=True)) if len(finite) else 0
        reason = "RETAINED"
        keep = True
        if coverage < minimum_coverage:
            keep = False
            reason = "LOW_COVERAGE"
        elif unique <= 1:
            keep = False
            reason = "CONSTANT"
        if keep:
            retained.append(feature)
        records.append(
            {
                "feature": feature,
                "coverage": coverage,
                "finite_unique_values": unique,
                "pre_correlation_status": reason,
                "selected": keep,
                "dropped_correlated_with": None,
            }
        )
    if not retained:
        raise RuntimeError("No features survived coverage and variance filtering")

    # Deterministic sample and deterministic feature priority. Engineered features
    # are considered first so a redundant raw window does not remove the explicit
    # participant-behaviour hypothesis feature.
    sample = frame[retained]
    if len(sample) > 50_000:
        step = max(1, len(sample) // 50_000)
        sample = sample.iloc[::step].head(50_000)
    medians = sample.median(numeric_only=True)
    sample = sample.fillna(medians).astype(float)
    priority = sorted(
        retained,
        key=lambda column: (
            0 if column.startswith((
                "utc_",
                "abs_",
                "oi_",
                "trade_count_intensity_",
                "quote_volume_intensity_",
                "liquidation_notional_intensity_",
                "liquidation_ratio_",
                "realized_vol_logratio_",
                "abs_flow_imbalance_change_",
                "abs_price_response_change_",
            )) else 1,
            column,
        ),
    )
    correlation_matrix = sample[priority].corr(method="pearson").abs()
    selected: list[str] = []
    dropped_with: dict[str, str] = {}
    for feature in priority:
        if len(selected) >= max_features:
            dropped_with[feature] = "MAX_FEATURES"
            continue
        keep = True
        for chosen in selected:
            correlation = float(correlation_matrix.at[feature, chosen])
            if math.isfinite(correlation) and correlation >= maximum_abs_correlation:
                keep = False
                dropped_with[feature] = chosen
                break
        if keep:
            selected.append(feature)
    selection = pd.DataFrame(records)
    selection["selected"] = selection["feature"].isin(selected)
    selection["dropped_correlated_with"] = selection["feature"].map(dropped_with)
    selection.loc[
        selection["dropped_correlated_with"].eq("MAX_FEATURES"),
        "pre_correlation_status",
    ] = "MAX_FEATURES"
    selection.loc[
        selection["dropped_correlated_with"].notna()
        & ~selection["dropped_correlated_with"].eq("MAX_FEATURES"),
        "pre_correlation_status",
    ] = "HIGH_CORRELATION"
    if not selected:
        raise RuntimeError("No features survived correlation pruning")
    return selected, selection


def require_sklearn() -> dict[str, Any]:
    try:
        import sklearn
        from sklearn.compose import TransformedTargetRegressor  # noqa: F401
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import (
            average_precision_score,
            brier_score_loss,
            log_loss,
            roc_auc_score,
        )
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing scikit-learn. Activate the virtual environment and run: "
            "python -m pip install scikit-learn joblib"
        ) from exc
    return locals()


def build_model(model_id: str, random_state: int) -> Any:
    sk = require_sklearn()
    Pipeline = sk["Pipeline"]
    SimpleImputer = sk["SimpleImputer"]
    if model_id == "logistic":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", sk["StandardScaler"]()),
                (
                    "model",
                    sk["LogisticRegression"](
                        C=1.0,
                        solver="lbfgs",
                        max_iter=1_000,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if model_id == "shallow_tree":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    sk["DecisionTreeClassifier"](
                        max_depth=4,
                        min_samples_leaf=0.005,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if model_id == "hist_gradient_boosting":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    sk["HistGradientBoostingClassifier"](
                        learning_rate=0.05,
                        max_iter=150,
                        max_leaf_nodes=15,
                        max_depth=4,
                        min_samples_leaf=100,
                        l2_regularization=1.0,
                        early_stopping=False,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    raise ValueError(model_id)


def balanced_sample_weights(target: np.ndarray) -> np.ndarray:
    values = np.asarray(target, dtype=np.int8)
    positive = int(np.sum(values == 1))
    negative = int(np.sum(values == 0))
    if positive == 0 or negative == 0:
        raise RuntimeError("Both target classes are required for model fitting")
    total = len(values)
    weights = np.empty(total, dtype=float)
    weights[values == 1] = total / (2.0 * positive)
    weights[values == 0] = total / (2.0 * negative)
    return weights


def fit_model(model: Any, x: pd.DataFrame, y: np.ndarray) -> Any:
    weights = balanced_sample_weights(y)
    model.fit(x, y, model__sample_weight=weights)
    return model


def predict_probability(model: Any, x: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError("Model produced non-finite probabilities")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise RuntimeError("Model probabilities are outside [0,1]")
    return probabilities


# ---------------------------------------------------------------------------
# Metrics, alert thresholds and event matching
# ---------------------------------------------------------------------------


def expected_calibration_error(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    if len(y) == 0:
        return float("nan")
    order = np.argsort(probability, kind="mergesort")
    chunks = np.array_split(order, min(bins, len(order)))
    total = float(len(y))
    error = 0.0
    for chunk in chunks:
        if not len(chunk):
            continue
        error += len(chunk) / total * abs(float(np.mean(probability[chunk])) - float(np.mean(y[chunk])))
    return float(error)


def snapshot_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    sk = require_sklearn()
    y = np.asarray(y, dtype=np.int8)
    probability = np.asarray(probability, dtype=float)
    prevalence = float(np.mean(y)) if len(y) else float("nan")
    metrics: dict[str, Any] = {
        "rows": int(len(y)),
        "positive_rows": int(np.sum(y)),
        "prevalence": prevalence,
        "average_precision": float(sk["average_precision_score"](y, probability))
        if len(np.unique(y)) == 2
        else float("nan"),
        "roc_auc": float(sk["roc_auc_score"](y, probability))
        if len(np.unique(y)) == 2
        else float("nan"),
        "brier_score": float(sk["brier_score_loss"](y, probability)),
        "log_loss": float(sk["log_loss"](y, probability, labels=[0, 1])),
        "expected_calibration_error": expected_calibration_error(y, probability),
    }
    metrics["pr_auc_lift"] = (
        metrics["average_precision"] / prevalence
        if prevalence > 0 and math.isfinite(metrics["average_precision"])
        else float("nan")
    )
    return metrics


@njit(cache=True)
def deduplicated_alert_indices(
    seconds: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    cooldown_seconds: int,
) -> np.ndarray:
    temporary = np.empty(len(seconds), dtype=np.int64)
    count = 0
    last_alert = -9_000_000_000_000_000_000
    for index in range(len(seconds)):
        if probabilities[index] >= threshold and seconds[index] - last_alert >= cooldown_seconds:
            temporary[count] = index
            count += 1
            last_alert = seconds[index]
    return temporary[:count]


def choose_alert_threshold(
    seconds: np.ndarray,
    probabilities: np.ndarray,
    days: float,
    cooldown_seconds: int,
    target_alerts_per_day: float,
    grid_size: int,
) -> dict[str, float]:
    if len(probabilities) == 0 or days <= 0:
        raise RuntimeError("Calibration predictions are required for threshold selection")
    quantiles = np.linspace(0.0, 1.0, grid_size)
    candidates = np.unique(np.quantile(probabilities, quantiles, method="linear"))
    candidates = np.concatenate(
        [
            np.asarray([np.nextafter(float(np.max(probabilities)), math.inf)]),
            candidates[::-1],
            np.asarray([float(np.min(probabilities))]),
        ]
    )
    best: tuple[float, float, float, int] | None = None
    for threshold in candidates:
        indices = deduplicated_alert_indices(
            seconds,
            probabilities,
            float(threshold),
            int(cooldown_seconds),
        )
        rate = len(indices) / days
        difference = abs(rate - target_alerts_per_day)
        # Prefer the closer rate, then the more conservative higher threshold.
        key = (difference, -float(threshold), rate, len(indices))
        if best is None or key < best:
            best = key
    assert best is not None
    difference, negative_threshold, achieved_rate, count = best
    return {
        "threshold": -negative_threshold,
        "calibration_alerts_per_day": achieved_rate,
        "calibration_alert_count": int(count),
        "alert_rate_absolute_error": difference,
    }


def calibration_profile(
    target_id: str,
    model_id: str,
    fold_id: str,
    split: str,
    y: np.ndarray,
    probability: np.ndarray,
    bins: int = 10,
) -> list[dict[str, Any]]:
    order = np.argsort(probability, kind="mergesort")
    rows: list[dict[str, Any]] = []
    for ordinal, chunk in enumerate(np.array_split(order, min(bins, len(order))), start=1):
        if not len(chunk):
            continue
        rows.append(
            {
                "target_id": target_id,
                "model_id": model_id,
                "fold_id": fold_id,
                "split": split,
                "calibration_bin": ordinal,
                "rows": int(len(chunk)),
                "mean_probability": float(np.mean(probability[chunk])),
                "observed_positive_rate": float(np.mean(y[chunk])),
                "minimum_probability": float(np.min(probability[chunk])),
                "maximum_probability": float(np.max(probability[chunk])),
            }
        )
    return rows


def scorable_events_for_test(
    events: pd.DataFrame,
    test_seconds: np.ndarray,
    test_start: int,
    test_end_exclusive: int,
) -> pd.DataFrame:
    selected = events[
        events["event_start_second"].between(test_start, test_end_exclusive - 1, inclusive="both")
        & events["scorable_event"].fillna(False).astype(bool)
        & events["first_realization_second"].notna()
    ].copy()
    if selected.empty:
        return selected
    seconds = np.asarray(test_seconds, dtype=np.int64)
    available: list[bool] = []
    for row in selected.itertuples(index=False):
        left = int(np.searchsorted(seconds, int(row.event_start_second), side="left"))
        available.append(
            left < len(seconds)
            and int(seconds[left]) <= int(row.first_realization_second)
        )
    return selected[np.asarray(available, dtype=bool)].copy().reset_index(drop=True)


def match_alerts_to_events(
    alert_seconds: np.ndarray,
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    alerts = np.asarray(alert_seconds, dtype=np.int64)
    event_rows = events.sort_values(
        ["first_realization_second", "event_start_second"], kind="mergesort"
    ).reset_index(drop=True)
    unmatched = set(range(len(event_rows)))
    matches: list[dict[str, Any]] = []
    false_alerts: list[int] = []
    for alert in alerts:
        candidates = [
            index
            for index in unmatched
            if int(event_rows.at[index, "event_start_second"]) <= int(alert)
            <= int(event_rows.at[index, "first_realization_second"])
        ]
        if not candidates:
            false_alerts.append(int(alert))
            continue
        chosen = min(
            candidates,
            key=lambda index: (
                int(event_rows.at[index, "first_realization_second"]),
                int(event_rows.at[index, "event_start_second"]),
            ),
        )
        unmatched.remove(chosen)
        row = event_rows.loc[chosen]
        matches.append(
            {
                "event_id": str(row["event_id"]),
                "alert_second": int(alert),
                "event_start_second": int(row["event_start_second"]),
                "first_realization_second": int(row["first_realization_second"]),
                "warning_seconds": int(row["first_realization_second"]) - int(alert),
                "seconds_after_event_onset": int(alert) - int(row["event_start_second"]),
            }
        )
    return pd.DataFrame(matches), pd.DataFrame({"alert_second": false_alerts})


def evaluate_alert_budget(
    prediction_frame: pd.DataFrame,
    threshold: float,
    cooldown_seconds: int,
    test_days: float,
    events_by_role: Mapping[str, pd.DataFrame],
    target: TargetPlan,
    model_id: str,
    fold_id: str,
    requested_alerts_per_day: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seconds = prediction_frame["decision_second"].to_numpy(dtype=np.int64)
    probabilities = prediction_frame["probability"].to_numpy(dtype=float)
    y = prediction_frame["target"].to_numpy(dtype=np.int8)
    indices = deduplicated_alert_indices(seconds, probabilities, threshold, cooldown_seconds)
    alert_seconds = seconds[indices]
    alert_y = y[indices]
    common = {
        "target_id": target.target_id,
        "horizon_minutes": target.horizon_minutes,
        "threshold_bps": target.threshold_bps,
        "model_id": model_id,
        "fold_id": fold_id,
        "requested_alerts_per_day": requested_alerts_per_day,
        "probability_threshold": threshold,
        "cooldown_seconds": cooldown_seconds,
        "test_days": test_days,
        "alerts": int(len(indices)),
        "alerts_per_day": len(indices) / test_days if test_days > 0 else float("nan"),
        "positive_alerts": int(np.sum(alert_y)) if len(indices) else 0,
        "snapshot_precision_at_alert": float(np.mean(alert_y)) if len(indices) else float("nan"),
        "mean_alert_probability": float(np.mean(probabilities[indices])) if len(indices) else float("nan"),
    }
    event_records: list[dict[str, Any]] = []
    for role, events in events_by_role.items():
        scorable = scorable_events_for_test(
            events,
            seconds,
            int(seconds[0]) if len(seconds) else 0,
            int(seconds[-1]) + 1 if len(seconds) else 0,
        )
        matches, false_alerts = match_alerts_to_events(alert_seconds, scorable)
        event_records.append(
            {
                **common,
                "segmentation_role": role,
                "segmentation_id": target.primary_segmentation_id
                if role == "PRIMARY"
                else target.robustness_segmentation_id,
                "scorable_events": int(len(scorable)),
                "recalled_events": int(len(matches)),
                "event_recall": len(matches) / len(scorable) if len(scorable) else float("nan"),
                "false_alerts": int(len(false_alerts)),
                "false_alerts_per_day": len(false_alerts) / test_days if test_days > 0 else float("nan"),
                "median_warning_seconds": safe_median(matches.get("warning_seconds", [])),
                "mean_warning_seconds": safe_mean(matches.get("warning_seconds", [])),
                "median_seconds_after_event_onset": safe_median(
                    matches.get("seconds_after_event_onset", [])
                ),
            }
        )
    return common, event_records


def extract_feature_importance(
    model: Any,
    selected_features: Sequence[str],
    target_id: str,
    model_id: str,
    fold_id: str,
) -> list[dict[str, Any]]:
    estimator = model.named_steps.get("model")
    values: np.ndarray | None = None
    kind: str | None = None
    if hasattr(estimator, "coef_"):
        values = np.asarray(estimator.coef_).reshape(-1)
        kind = "coefficient"
    elif hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_).reshape(-1)
        kind = "feature_importance"
    if values is None or len(values) != len(selected_features):
        return []
    return [
        {
            "target_id": target_id,
            "model_id": model_id,
            "fold_id": fold_id,
            "feature": feature,
            "importance_type": kind,
            "importance": float(value),
            "absolute_importance": abs(float(value)),
        }
        for feature, value in zip(selected_features, values)
    ]


# ---------------------------------------------------------------------------
# Fold training
# ---------------------------------------------------------------------------


def split_mask(seconds: np.ndarray, start: int, end_exclusive: int) -> np.ndarray:
    return (seconds >= int(start)) & (seconds < int(end_exclusive))


def model_fold_signature(
    target: TargetPlan,
    model_id: str,
    fold: FoldSpec,
    snapshot_signature: str,
    label_signature: str,
    event_signatures: Mapping[str, str],
    configuration: Mapping[str, Any],
) -> str:
    return sha256_text(
        compact_json(
            {
                "builder_version": BUILDER_VERSION,
                "target": target.__dict__,
                "model_id": model_id,
                "fold": fold.__dict__,
                "snapshot_signature": snapshot_signature,
                "label_signature": label_signature,
                "event_signatures": dict(event_signatures),
                "configuration": configuration,
            }
        )
    )


def reusable_model_fold(
    connection: sqlite3.Connection,
    target_id: str,
    model_id: str,
    fold_id: str,
    input_signature: str,
    paths: ModelResultPaths,
) -> bool:
    row = connection.execute(
        """
        SELECT * FROM model_folds
        WHERE target_id=? AND model_id=? AND fold_id=?
        """,
        (target_id, model_id, fold_id),
    ).fetchone()
    if row is None or row["status"] != "COMPLETED" or row["input_signature"] != input_signature:
        return False
    return paths.predictions.exists() and paths.model.exists() and paths.result_json.exists()


def run_model_fold(
    target_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target: TargetPlan,
    model_id: str,
    fold: FoldSpec,
    events_by_role: Mapping[str, pd.DataFrame],
    alert_rates: Sequence[float],
    cooldown_seconds: int,
    minimum_feature_coverage: float,
    maximum_abs_correlation: float,
    max_features: int,
    threshold_grid_size: int,
    random_state: int,
    preselected_features: Sequence[str] | None = None,
    preselection: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    seconds = target_frame["decision_second"].to_numpy(dtype=np.int64)
    fit_mask = split_mask(seconds, fold.fit_start_second, fold.fit_end_exclusive_second)
    calibration_mask = split_mask(
        seconds, fold.calibration_start_second, fold.calibration_end_exclusive_second
    )
    test_mask = split_mask(seconds, fold.test_start_second, fold.test_end_exclusive_second)
    if not np.any(fit_mask) or not np.any(calibration_mask) or not np.any(test_mask):
        raise RuntimeError(f"Fold {fold.fold_id} contains an empty split")

    fit_frame = target_frame.loc[fit_mask]
    calibration_frame = target_frame.loc[calibration_mask]
    test_frame = target_frame.loc[test_mask]
    for split_name, split in (
        ("FIT", fit_frame),
        ("CALIBRATION", calibration_frame),
        ("TEST", test_frame),
    ):
        if split["target"].nunique() < 2:
            raise RuntimeError(
                f"{target.target_id}/{model_id}/fold {fold.fold_id} {split_name} "
                "does not contain both classes"
            )

    if preselected_features is None or preselection is None:
        selected_features, selection = select_fold_features(
            fit_frame,
            feature_columns,
            minimum_feature_coverage,
            maximum_abs_correlation,
            max_features,
        )
    else:
        selected_features = list(preselected_features)
        selection = preselection.copy()
    model = build_model(model_id, random_state + fold.fold_ordinal)
    model = fit_model(
        model,
        fit_frame[selected_features],
        fit_frame["target"].to_numpy(dtype=np.int8),
    )
    calibration_probability = predict_probability(model, calibration_frame[selected_features])
    test_probability = predict_probability(model, test_frame[selected_features])

    prediction_frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "target_id": target.target_id,
                    "model_id": model_id,
                    "fold_id": fold.fold_id,
                    "split": "CALIBRATION",
                    "decision_second": calibration_frame["decision_second"].to_numpy(dtype=np.int64),
                    "decision_timestamp_utc": pd.to_datetime(
                        calibration_frame["decision_second"], unit="s", utc=True
                    ),
                    "target": calibration_frame["target"].to_numpy(dtype=np.int8),
                    "future_swing_bps": calibration_frame["future_swing_bps"].to_numpy(dtype=float),
                    "probability": calibration_probability,
                }
            ),
            pd.DataFrame(
                {
                    "target_id": target.target_id,
                    "model_id": model_id,
                    "fold_id": fold.fold_id,
                    "split": "TEST",
                    "decision_second": test_frame["decision_second"].to_numpy(dtype=np.int64),
                    "decision_timestamp_utc": pd.to_datetime(
                        test_frame["decision_second"], unit="s", utc=True
                    ),
                    "target": test_frame["target"].to_numpy(dtype=np.int8),
                    "future_swing_bps": test_frame["future_swing_bps"].to_numpy(dtype=float),
                    "probability": test_probability,
                }
            ),
        ],
        ignore_index=True,
    )

    fold_metrics: list[dict[str, Any]] = []
    for split_name, split_frame, probability in (
        ("FIT", fit_frame, predict_probability(model, fit_frame[selected_features])),
        ("CALIBRATION", calibration_frame, calibration_probability),
        ("TEST", test_frame, test_probability),
    ):
        metrics = snapshot_metrics(split_frame["target"].to_numpy(dtype=np.int8), probability)
        fold_metrics.append(
            {
                "target_id": target.target_id,
                "horizon_minutes": target.horizon_minutes,
                "threshold_bps": target.threshold_bps,
                "model_id": model_id,
                "fold_id": fold.fold_id,
                "split": split_name,
                **metrics,
            }
        )

    calibration_days = max(
        1.0,
        (fold.calibration_end_exclusive_second - fold.calibration_start_second)
        / SECONDS_PER_DAY,
    )
    test_days = max(1.0, (fold.test_end_exclusive_second - fold.test_start_second) / SECONDS_PER_DAY)
    thresholds: list[dict[str, Any]] = []
    alert_metrics: list[dict[str, Any]] = []
    event_metrics: list[dict[str, Any]] = []
    test_predictions = prediction_frame[prediction_frame["split"].eq("TEST")].copy()
    calibration_seconds = calibration_frame["decision_second"].to_numpy(dtype=np.int64)
    for alert_rate in alert_rates:
        selected = choose_alert_threshold(
            calibration_seconds,
            calibration_probability,
            calibration_days,
            cooldown_seconds,
            alert_rate,
            threshold_grid_size,
        )
        thresholds.append({"requested_alerts_per_day": alert_rate, **selected})
        alert_record, event_records = evaluate_alert_budget(
            test_predictions,
            selected["threshold"],
            cooldown_seconds,
            test_days,
            events_by_role,
            target,
            model_id,
            fold.fold_id,
            alert_rate,
        )
        alert_record.update(
            {
                "calibration_alerts_per_day": selected["calibration_alerts_per_day"],
                "calibration_alert_count": selected["calibration_alert_count"],
            }
        )
        alert_metrics.append(alert_record)
        for record in event_records:
            record.update(
                {
                    "calibration_alerts_per_day": selected["calibration_alerts_per_day"],
                    "calibration_alert_count": selected["calibration_alert_count"],
                }
            )
        event_metrics.extend(event_records)

    calibration_rows = calibration_profile(
        target.target_id,
        model_id,
        fold.fold_id,
        "CALIBRATION",
        calibration_frame["target"].to_numpy(dtype=np.int8),
        calibration_probability,
    ) + calibration_profile(
        target.target_id,
        model_id,
        fold.fold_id,
        "TEST",
        test_frame["target"].to_numpy(dtype=np.int8),
        test_probability,
    )

    selection.insert(0, "fold_id", fold.fold_id)
    selection.insert(0, "model_id", model_id)
    selection.insert(0, "target_id", target.target_id)
    importance = extract_feature_importance(
        model, selected_features, target.target_id, model_id, fold.fold_id
    )
    result = {
        "target_id": target.target_id,
        "model_id": model_id,
        "fold_id": fold.fold_id,
        "selected_features": selected_features,
        "selected_feature_count": len(selected_features),
        "fold_metrics": fold_metrics,
        "thresholds": thresholds,
        "alert_metrics": alert_metrics,
        "event_metrics": event_metrics,
        "calibration": calibration_rows,
        "feature_selection": selection.to_dict("records"),
        "feature_importance": importance,
    }
    return prediction_frame, {"model": model, "result": result}


# ---------------------------------------------------------------------------
# Reports and validation
# ---------------------------------------------------------------------------


def aggregate_reports(results: Sequence[Mapping[str, Any]]) -> dict[str, pd.DataFrame]:
    fold_metrics = pd.DataFrame(
        [record for result in results for record in result.get("fold_metrics", [])]
    )
    alert_metrics = pd.DataFrame(
        [record for result in results for record in result.get("alert_metrics", [])]
    )
    event_metrics = pd.DataFrame(
        [record for result in results for record in result.get("event_metrics", [])]
    )
    calibration = pd.DataFrame(
        [record for result in results for record in result.get("calibration", [])]
    )
    feature_selection = pd.DataFrame(
        [record for result in results for record in result.get("feature_selection", [])]
    )
    feature_importance = pd.DataFrame(
        [record for result in results for record in result.get("feature_importance", [])]
    )

    test_metrics = fold_metrics[fold_metrics["split"].eq("TEST")].copy()
    model_summary = (
        test_metrics.groupby(["target_id", "horizon_minutes", "threshold_bps", "model_id"], as_index=False)
        .agg(
            folds=("fold_id", "nunique"),
            mean_prevalence=("prevalence", "mean"),
            mean_pr_auc=("average_precision", "mean"),
            std_pr_auc=("average_precision", "std"),
            mean_pr_auc_lift=("pr_auc_lift", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_brier_score=("brier_score", "mean"),
            mean_log_loss=("log_loss", "mean"),
            mean_expected_calibration_error=("expected_calibration_error", "mean"),
        )
        .sort_values(["target_id", "mean_pr_auc"], ascending=[True, False], kind="mergesort")
        .reset_index(drop=True)
    )

    primary_two = event_metrics[
        event_metrics["segmentation_role"].eq("PRIMARY")
        & np.isclose(event_metrics["requested_alerts_per_day"], 2.0)
    ].copy()
    event_two_summary = (
        primary_two.groupby(["target_id", "model_id"], as_index=False)
        .agg(
            mean_primary_event_recall_at_2_alerts=("event_recall", "mean"),
            mean_false_alerts_per_day_at_2_alerts=("false_alerts_per_day", "mean"),
            median_warning_seconds_at_2_alerts=("median_warning_seconds", "median"),
            mean_actual_alerts_per_day_at_2_alerts=("alerts_per_day", "mean"),
        )
    )
    scorecard = model_summary.merge(event_two_summary, on=["target_id", "model_id"], how="left")
    scorecard = scorecard.sort_values(
        ["mean_pr_auc", "mean_primary_event_recall_at_2_alerts"],
        ascending=[False, False],
        kind="mergesort",
    ).reset_index(drop=True)

    return {
        "fold_metrics": fold_metrics,
        "alert_metrics": alert_metrics,
        "event_metrics": event_metrics,
        "model_summary": model_summary,
        "target_scorecard": scorecard,
        "calibration": calibration,
        "feature_selection": feature_selection,
        "feature_importance": feature_importance,
    }


def invariant_record(scope: str, name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {
        "scope": scope,
        "invariant": name,
        "status": "PASS" if passed else "FAIL",
        "detail": detail,
    }


def validate_run(
    snapshots: pd.DataFrame,
    feature_columns: Sequence[str],
    folds_by_target: Mapping[str, Sequence[FoldSpec]],
    reports: Mapping[str, pd.DataFrame],
    cadence_seconds: int,
    expected_results: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    records.append(
        invariant_record(
            "SNAPSHOTS",
            "UNIQUE_SECONDS",
            not snapshots["receive_second"].duplicated().any(),
            f"rows={len(snapshots):,}",
        )
    )
    records.append(
        invariant_record(
            "SNAPSHOTS",
            "EXACT_CADENCE",
            bool((snapshots["receive_second"] % cadence_seconds == 0).all()),
            f"cadence={cadence_seconds}",
        )
    )
    future_features = [
        column
        for column in feature_columns
        if any(token in column.lower() for token in CORE.FUTURE_OR_LABEL_TOKENS)
    ]
    records.append(
        invariant_record(
            "FEATURES",
            "NO_FUTURE_FEATURE_COLUMNS",
            not future_features,
            f"future_columns={future_features}",
        )
    )
    for target_id, folds in folds_by_target.items():
        for fold in folds:
            chronology = (
                fold.fit_end_exclusive_second <= fold.calibration_start_second
                and fold.calibration_end_exclusive_second <= fold.test_start_second
                and fold.test_start_second < fold.test_end_exclusive_second
            )
            records.append(
                invariant_record(
                    target_id,
                    f"FOLD_{fold.fold_id}_CHRONOLOGY_AND_PURGE",
                    chronology,
                    (
                        f"fit_end={fold.fit_end_exclusive_second}, "
                        f"calibration_start={fold.calibration_start_second}, "
                        f"calibration_end={fold.calibration_end_exclusive_second}, "
                        f"test_start={fold.test_start_second}"
                    ),
                )
            )
    fold_metrics = reports["fold_metrics"]
    completed = fold_metrics[
        fold_metrics["split"].eq("TEST")
    ][["target_id", "model_id", "fold_id"]].drop_duplicates()
    records.append(
        invariant_record(
            "MODELS",
            "EXPECTED_MODEL_FOLDS",
            len(completed) == expected_results,
            f"completed={len(completed)}, expected={expected_results}",
        )
    )
    probabilities_valid = True
    if not reports["calibration"].empty:
        probabilities_valid = bool(
            reports["calibration"]["mean_probability"].between(0.0, 1.0, inclusive="both").all()
        )
    records.append(
        invariant_record(
            "MODELS",
            "PROBABILITIES_WITHIN_BOUNDS",
            probabilities_valid,
            "calibration-bin probabilities within [0,1]",
        )
    )
    event_metrics = reports["event_metrics"]
    event_bounds = True
    if not event_metrics.empty:
        finite = pd.to_numeric(event_metrics["event_recall"], errors="coerce").dropna()
        event_bounds = bool(finite.between(0.0, 1.0, inclusive="both").all())
    records.append(
        invariant_record(
            "EVENTS",
            "EVENT_RECALL_WITHIN_BOUNDS",
            event_bounds,
            "event recall within [0,1]",
        )
    )
    return pd.DataFrame(records)


def findings_markdown(
    summary: Mapping[str, Any],
    scorecard: pd.DataFrame,
    model_summary: pd.DataFrame,
) -> str:
    lines = [
        "# Perps Expansion Baseline Findings",
        "",
        "## Run status",
        "",
        f"- Status: **{summary['status']}**",
        f"- Targets: **{summary['targets']}**",
        f"- Models: **{summary['models']}**",
        f"- Walk-forward folds per target: **{summary['folds_per_target']}**",
        f"- Snapshot rows: **{summary['snapshot_rows']:,}**",
        f"- Failed invariants: **{summary['failed_invariants']}**",
        "",
        "## Interpretation contract",
        "",
        "- All model features are causal at the decision second.",
        "- All splits are chronological; no random split is used.",
        "- Alert thresholds are chosen on an inner calibration block, never on the test block.",
        "- Event recall uses one-to-one alerts matched before exact threshold realisation.",
        "- These are expansion forecasts only; no direction or profitability claim is made.",
        "",
        "## Cross-target scorecard",
        "",
    ]
    if scorecard.empty:
        lines.append("No completed scorecard rows were available.")
    else:
        display_columns = [
            "target_id",
            "model_id",
            "mean_pr_auc",
            "mean_pr_auc_lift",
            "mean_primary_event_recall_at_2_alerts",
            "mean_false_alerts_per_day_at_2_alerts",
            "median_warning_seconds_at_2_alerts",
        ]
        lines.append(scorecard[display_columns].head(20).to_markdown(index=False, floatfmt=".4f"))
    lines.extend(
        [
            "",
            "## Model comparison by target",
            "",
        ]
    )
    if model_summary.empty:
        lines.append("No completed model summaries were available.")
    else:
        lines.append(
            model_summary[
                [
                    "target_id",
                    "model_id",
                    "mean_prevalence",
                    "mean_pr_auc",
                    "mean_pr_auc_lift",
                    "mean_brier_score",
                    "mean_expected_calibration_error",
                ]
            ].to_markdown(index=False, floatfmt=".4f")
        )
    lines.extend(
        [
            "",
            "## Next decision",
            "",
            "Do not select a horizon from PR-AUC alone. Compare precision and event recall at the same alert budget, false alerts per day, real warning time, calibration, chronological stability, and primary-versus-robustness segmentation agreement.",
            "",
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def run_self_test() -> None:
    plans = parse_target_plan(DEFAULT_TARGET_PLAN)
    assert len(plans) == 6
    assert plans[3].target_id == "H15M_T050BPS"
    assert plans[3].primary_segmentation_id == "bridge_060s"

    days = [date(2026, 1, 1) + timedelta(days=index) for index in range(30)]
    folds = build_folds(days, 2, 20, 5, 4, 300)
    assert len(folds) == 2
    assert folds[0].fit_end_exclusive_second < folds[0].calibration_start_second
    assert folds[0].calibration_end_exclusive_second < folds[0].test_start_second

    prices = np.asarray([100.0, 100.1, 100.2, 100.6, 100.5], dtype=float)
    assert first_threshold_realization_offset(prices, 50.0) == 3
    assert first_threshold_realization_offset(prices, 75.0) == -1

    seconds = np.asarray([0, 30, 60, 90, 120, 150], dtype=np.int64)
    probabilities = np.asarray([0.9, 0.8, 0.7, 0.95, 0.4, 0.99], dtype=float)
    indices = deduplicated_alert_indices(seconds, probabilities, 0.75, 60)
    assert indices.tolist() == [0, 3, 5]

    events = pd.DataFrame(
        {
            "event_id": ["A", "B"],
            "event_start_second": [0, 100],
            "first_realization_second": [80, 180],
        }
    )
    matches, false_alerts = match_alerts_to_events(np.asarray([30, 130, 250]), events)
    assert len(matches) == 2 and len(false_alerts) == 1
    assert set(matches["event_id"]) == {"A", "B"}

    threshold = choose_alert_threshold(
        seconds,
        probabilities,
        days=1.0,
        cooldown_seconds=60,
        target_alerts_per_day=2.0,
        grid_size=21,
    )
    assert 0.0 <= threshold["threshold"] <= 1.0000001

    # Upstream already supplies abs_return_1s_bps.  The derived-feature layer
    # must reuse it rather than create a second column with the same name.
    collision_frame = pd.DataFrame(
        {
            "receive_second": [0, 1, 2],
            "return_1s_bps": [-2.0, 0.0, 3.0],
            "abs_return_1s_bps": [2.0, 0.0, 3.0],
        }
    )
    collision_result, collision_derived = derive_hypothesis_features(collision_frame)
    assert collision_result.columns.is_unique
    assert "abs_return_1s_bps" not in collision_derived
    assert collision_result["abs_return_1s_bps"].tolist() == [2.0, 0.0, 3.0]

    duplicate_frame = pd.DataFrame(
        [[1.0, 1.0, 2.0], [3.0, 3.0, 4.0]],
        columns=["duplicate", "duplicate", "other"],
    )
    collapsed = collapse_equivalent_duplicate_columns(
        duplicate_frame, context="self-test"
    )
    assert collapsed.columns.tolist() == ["duplicate", "other"]
    assert collapsed.columns.is_unique

    print(
        "SELF-TEST PASS\n"
        "Validated frozen target parsing, purged chronological folds, exact threshold "
        "realisation, duplicate-safe feature derivation and serialization, alert cooldown, one-to-one event "
        "matching and calibration threshold selection."
    )


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Build and evaluate leakage-safe multi-target expansion baselines."
    )
    value.add_argument(
        "--feature-root",
        default="./data/derived/perps/flow_features_1s_v1",
        help="Audited causal flow-feature root.",
    )
    value.add_argument(
        "--target-root",
        default="./data/derived/perps/research/expansion_target_segmentation_sensitivity_v1",
        help="Completed horizon labels and target event datasets.",
    )
    value.add_argument(
        "--output-root",
        default="./data/derived/perps/research/expansion_baseline_models_v1",
        help="Separate modelling snapshots, predictions and model artifacts.",
    )
    value.add_argument(
        "--report-root",
        default="./reports/perps_expansion_baseline_models",
        help="CSV, JSON and Markdown report root.",
    )
    value.add_argument("--symbol", default="BTCUSDT")
    value.add_argument("--target-plan", default=DEFAULT_TARGET_PLAN)
    value.add_argument(
        "--targets",
        help="Optional comma-separated subset of frozen target IDs.",
    )
    value.add_argument("--models", default=",".join(DEFAULT_MODELS))
    value.add_argument(
        "--decision-cadence-seconds",
        type=int,
        default=DEFAULT_DECISION_CADENCE_SECONDS,
    )
    value.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    value.add_argument("--initial-train-days", type=int, default=DEFAULT_INITIAL_TRAIN_DAYS)
    value.add_argument("--test-days", type=int, default=DEFAULT_TEST_DAYS)
    value.add_argument("--calibration-days", type=int, default=DEFAULT_CALIBRATION_DAYS)
    value.add_argument(
        "--alert-rates-per-day",
        default=",".join(str(value) for value in DEFAULT_ALERT_RATES_PER_DAY),
    )
    value.add_argument(
        "--alert-cooldown-seconds",
        default=DEFAULT_ALERT_COOLDOWN,
        help="HORIZON or an exact positive integer.",
    )
    value.add_argument("--minimum-feature-coverage", type=float, default=DEFAULT_MIN_FEATURE_COVERAGE)
    value.add_argument("--maximum-abs-correlation", type=float, default=DEFAULT_MAX_ABS_CORRELATION)
    value.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES)
    value.add_argument("--threshold-grid-size", type=int, default=DEFAULT_THRESHOLD_GRID_SIZE)
    value.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    value.add_argument("--start-date")
    value.add_argument("--end-date")
    value.add_argument("--max-dates", type=int, help="Diagnostic/smoke-test date limit.")
    value.add_argument("--feature-columns", help="Optional exact raw causal feature allowlist.")
    value.add_argument("--hash-input-files", action="store_true")
    value.add_argument("--force", action="store_true")
    value.add_argument("--compression-level", type=int, default=6)
    value.add_argument("--self-test", action="store_true")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    warnings.filterwarnings(
        "ignore",
        message=(
            r"`sklearn\.utils\.parallel\.delayed` should be used with "
            r"`sklearn\.utils\.parallel\.Parallel`.*"
        ),
        category=UserWarning,
        module=r"sklearn\.utils\.parallel",
    )
    if args.self_test:
        run_self_test()
        return 0

    CORE.require_arrow()
    require_sklearn()
    if args.decision_cadence_seconds <= 0:
        raise SystemExit("--decision-cadence-seconds must be positive")
    if args.n_folds <= 0 or args.initial_train_days <= 0 or args.test_days <= 0:
        raise SystemExit("Fold counts and day counts must be positive")
    if not 0.0 < args.minimum_feature_coverage <= 1.0:
        raise SystemExit("--minimum-feature-coverage must be in (0,1]")
    if not 0.0 < args.maximum_abs_correlation <= 1.0:
        raise SystemExit("--maximum-abs-correlation must be in (0,1]")
    if args.max_features <= 0 or args.threshold_grid_size < 11:
        raise SystemExit("--max-features must be positive and threshold grid must be >= 11")
    if not 1 <= args.compression_level <= 22:
        raise SystemExit("--compression-level must be between 1 and 22")

    plans = parse_targets_filter(args.targets, parse_target_plan(args.target_plan))
    models = parse_models(args.models)
    alert_rates = parse_float_list(args.alert_rates_per_day, "--alert-rates-per-day")
    start_date = parse_date(args.start_date, "--start-date")
    end_date = parse_date(args.end_date, "--end-date")
    if start_date and end_date and start_date > end_date:
        raise SystemExit("--start-date cannot be after --end-date")

    feature_root = Path(args.feature_root).resolve()
    target_root = Path(args.target_root).resolve()
    output_root = Path(args.output_root).resolve()
    report_root = Path(args.report_root).resolve()
    for root, name in ((feature_root, "feature root"), (target_root, "target root")):
        if not root.exists():
            raise SystemExit(f"{name.title()} does not exist: {root}")
    output_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    run_id = f"{utc_now().strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    configuration = {
        "builder_version": BUILDER_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "feature_root": str(feature_root),
        "target_root": str(target_root),
        "output_root": str(output_root),
        "report_root": str(report_root),
        "symbol": args.symbol.upper(),
        "target_plan": [plan.__dict__ for plan in plans],
        "models": models,
        "decision_cadence_seconds": args.decision_cadence_seconds,
        "n_folds": args.n_folds,
        "initial_train_days": args.initial_train_days,
        "test_days": args.test_days,
        "calibration_days": args.calibration_days,
        "alert_rates_per_day": alert_rates,
        "alert_cooldown_seconds": args.alert_cooldown_seconds,
        "minimum_feature_coverage": args.minimum_feature_coverage,
        "maximum_abs_correlation": args.maximum_abs_correlation,
        "max_features": args.max_features,
        "threshold_grid_size": args.threshold_grid_size,
        "random_state": args.random_state,
        "start_date": start_date,
        "end_date": end_date,
        "max_dates": args.max_dates,
        "hash_input_files": args.hash_input_files,
    }

    print("Building and evaluating expansion baseline models...")
    print(f"Builder version     : {BUILDER_VERSION}")
    print(f"Script release      : {SCRIPT_RELEASE}")
    print(f"Run ID             : {run_id}")
    print(f"Feature root       : {feature_root}")
    print(f"Target root        : {target_root}")
    print(f"Output root        : {output_root}")
    print(f"Report root        : {report_root}")
    print(f"Targets            : {len(plans)}")
    print(f"Models             : {len(models)}")
    print(f"Cadence            : {args.decision_cadence_seconds}s")

    state_path = output_root / "_build_state" / "build_state.sqlite"
    connection = connect_state(state_path)
    connection.execute(
        "INSERT INTO runs(run_id, started_at_utc, status, configuration_json) VALUES(?,?,?,?)",
        (run_id, iso_utc(utc_now()), "RUNNING", compact_json(configuration)),
    )
    connection.commit()

    rebuilt_snapshots = skipped_snapshots = 0
    completed_model_folds = skipped_model_folds = failed_model_folds = 0
    try:
        symbol = args.symbol.upper()
        feature_partitions = CORE.discover_date_partitions(
            feature_root, symbol, "features", args.hash_input_files
        )
        feature_schema = CORE.validate_feature_schema(feature_partitions, symbol)
        all_days = sorted(
            day
            for day in feature_partitions
            if (start_date is None or day >= start_date)
            and (end_date is None or day <= end_date)
        )
        if args.max_dates is not None:
            if args.max_dates <= 0:
                raise SystemExit("--max-dates must be positive")
            all_days = all_days[: args.max_dates]
        if not all_days:
            raise RuntimeError("No feature dates remain after filtering")
        selected_days = set(all_days)

        if args.feature_columns:
            raw_features = sorted(
                {column.strip() for column in args.feature_columns.split(",") if column.strip()}
            )
        else:
            raw_features = CORE.auto_feature_columns(feature_schema)
        if not raw_features:
            raise RuntimeError("No raw causal features were selected")
        CORE.validate_numeric_feature_columns(
            next(iter(feature_partitions.values())).files[0], raw_features
        )

        snapshot_logic_signature = sha256_text(
            compact_json(
                {
                    "builder_version": BUILDER_VERSION,
                    "cadence": args.decision_cadence_seconds,
                    "raw_features": raw_features,
                    "derived_logic_version": 1,
                }
            )
        )
        snapshot_paths: list[Path] = []
        snapshot_signatures: list[dict[str, str]] = []
        derived_feature_names: list[str] = []
        print("Building/reusing causal 30-second snapshot partitions...")
        for day in all_days:
            partition = feature_partitions[day]
            path = snapshot_path(output_root, symbol, day)
            input_signature = sha256_text(
                compact_json(
                    {
                        "feature_partition_signature": partition.signature,
                        "snapshot_logic_signature": snapshot_logic_signature,
                    }
                )
            )
            state = connection.execute(
                "SELECT * FROM snapshot_partitions WHERE symbol=? AND decision_date=?",
                (symbol, day.isoformat()),
            ).fetchone()
            reusable = (
                not args.force
                and state is not None
                and state["status"] == "COMPLETED"
                and state["input_signature"] == input_signature
                and snapshot_is_reusable(path, input_signature, state["row_count"])
            )
            if reusable:
                skipped_snapshots += 1
            else:
                connection.execute(
                    """
                    INSERT INTO snapshot_partitions(
                        symbol, decision_date, input_signature, output_path, status, last_run_id
                    ) VALUES(?,?,?,?,?,?)
                    ON CONFLICT(symbol, decision_date) DO UPDATE SET
                        input_signature=excluded.input_signature,
                        output_path=excluded.output_path,
                        status=excluded.status,
                        error_text=NULL,
                        last_run_id=excluded.last_run_id
                    """,
                    (symbol, day.isoformat(), input_signature, str(path), "RUNNING", run_id),
                )
                connection.commit()
                try:
                    rows, derived = build_snapshot_partition(
                        partition,
                        path,
                        symbol,
                        raw_features,
                        args.decision_cadence_seconds,
                        input_signature,
                        args.compression_level,
                    )
                    derived_feature_names = derived
                    connection.execute(
                        """
                        UPDATE snapshot_partitions
                        SET row_count=?, status='COMPLETED', completed_at_utc=?, error_text=NULL,
                            last_run_id=?
                        WHERE symbol=? AND decision_date=?
                        """,
                        (rows, iso_utc(utc_now()), run_id, symbol, day.isoformat()),
                    )
                    connection.commit()
                    rebuilt_snapshots += 1
                except Exception:
                    connection.execute(
                        """
                        UPDATE snapshot_partitions
                        SET status='FAILED', error_text=?, last_run_id=?
                        WHERE symbol=? AND decision_date=?
                        """,
                        (traceback.format_exc(), run_id, symbol, day.isoformat()),
                    )
                    connection.commit()
                    raise
            snapshot_paths.append(path)
            snapshot_signatures.append({"date": day.isoformat(), "signature": input_signature})

        snapshots = load_snapshots(snapshot_paths)
        snapshot_signature = sha256_text(compact_json(snapshot_signatures))
        features = candidate_feature_columns(snapshots)
        if not features:
            raise RuntimeError("No model feature columns were found in snapshots")
        if not derived_feature_names:
            derived_feature_names = [column for column in features if column not in raw_features]

        label_partitions_by_horizon: dict[int, dict[date, ParquetPartition]] = {}
        for horizon in sorted({plan.horizon_minutes for plan in plans}):
            label_partitions_by_horizon[horizon] = discover_horizon_label_partitions(
                target_root, horizon, symbol, args.hash_input_files
            )

        # Load all event tables before exact-price reconstruction.
        events_by_target: dict[str, dict[str, pd.DataFrame]] = {}
        event_signatures_by_target: dict[str, dict[str, str]] = {}
        all_event_starts: list[int] = []
        for target in plans:
            role_frames: dict[str, pd.DataFrame] = {}
            role_signatures: dict[str, str] = {}
            for role, segmentation in (
                ("PRIMARY", target.primary_segmentation_id),
                ("ROBUSTNESS", target.robustness_segmentation_id),
            ):
                frame, signature = read_target_events(
                    target_event_path(target_root, target, segmentation, symbol),
                    target,
                    segmentation,
                    symbol,
                )
                frame = frame[
                    frame["event_start_second"].map(
                        lambda value: datetime.fromtimestamp(int(value), tz=timezone.utc).date()
                    ).isin(selected_days)
                ].copy()
                role_frames[role] = frame
                role_signatures[role] = signature
                all_event_starts.extend(frame["event_start_second"].astype(int).tolist())
            events_by_target[target.target_id] = role_frames
            event_signatures_by_target[target.target_id] = role_signatures

        if not all_event_starts:
            raise RuntimeError("No target events remain after date filtering")
        minimum_price_second = min(all_event_starts) + 1
        maximum_price_second = max(
            int(frame["event_start_second"].max()) + target.horizon_seconds + 1
            for target in plans
            for frame in events_by_target[target.target_id].values()
            if not frame.empty
        )
        print("Reconstructing exact event threshold-realization times...")
        exact_seconds, exact_prices = load_exact_price_series(
            feature_partitions,
            minimum_price_second,
            maximum_price_second,
            symbol,
        )
        for target in plans:
            for role in ("PRIMARY", "ROBUSTNESS"):
                events_by_target[target.target_id][role] = enrich_event_realizations(
                    events_by_target[target.target_id][role],
                    target,
                    exact_seconds,
                    exact_prices,
                    args.decision_cadence_seconds,
                )
                event_output = (
                    output_root
                    / "event_catalog"
                    / f"{TARGET_PREFIX}{target.target_id}"
                    / f"{SEGMENTATION_PREFIX}{target.primary_segmentation_id if role == 'PRIMARY' else target.robustness_segmentation_id}"
                    / f"{SYMBOL_PREFIX}{symbol}"
                    / "part-00000.parquet"
                )
                CORE.atomic_write_parquet(
                    event_output,
                    events_by_target[target.target_id][role],
                    {
                        "perps_expansion_baseline_builder_version": BUILDER_VERSION,
                        "perps_expansion_baseline_event_catalog_schema_version": OUTPUT_SCHEMA_VERSION,
                        "target_id": target.target_id,
                        "segmentation_role": role,
                        "symbol": symbol,
                        "input_signature": event_signatures_by_target[target.target_id][role],
                    },
                    args.compression_level,
                )
        del exact_seconds, exact_prices
        gc.collect()

        all_results: list[dict[str, Any]] = []
        all_fold_frames: list[pd.DataFrame] = []
        folds_by_target: dict[str, tuple[FoldSpec, ...]] = {}
        for target in plans:
            label_partitions = label_partitions_by_horizon[target.horizon_minutes]
            labels = load_target_labels(
                label_partitions,
                target,
                args.decision_cadence_seconds,
                selected_days,
                symbol,
            )
            target_frame = snapshots.rename(columns={"receive_second": "decision_second"}).merge(
                labels,
                on="decision_second",
                how="inner",
                validate="one_to_one",
            )
            target_frame = target_frame.sort_values("decision_second", kind="mergesort").reset_index(drop=True)
            if target_frame.empty:
                raise RuntimeError(f"No model rows remain for {target.target_id}")
            label_signature = sha256_text(
                compact_json(
                    [
                        {"day": day.isoformat(), "signature": partition.signature}
                        for day, partition in sorted(label_partitions.items())
                        if day in selected_days
                    ]
                )
            )
            folds = build_folds(
                all_days,
                args.n_folds,
                args.initial_train_days,
                args.test_days,
                args.calibration_days,
                target.horizon_seconds,
            )
            folds_by_target[target.target_id] = folds
            all_fold_frames.append(fold_frame(folds, target))
            cooldown_seconds = (
                target.horizon_seconds
                if str(args.alert_cooldown_seconds).upper() == "HORIZON"
                else int(args.alert_cooldown_seconds)
            )
            if cooldown_seconds <= 0:
                raise SystemExit("--alert-cooldown-seconds must be HORIZON or a positive integer")
            print(
                f"Target {target.target_id}: rows={len(target_frame):,}, "
                f"positive={int(target_frame['target'].sum()):,}, folds={len(folds)}"
            )
            fold_feature_cache: dict[str, tuple[list[str], pd.DataFrame]] = {}
            target_seconds = target_frame["decision_second"].to_numpy(dtype=np.int64)
            for fold in folds:
                fit_mask_for_selection = split_mask(
                    target_seconds, fold.fit_start_second, fold.fit_end_exclusive_second
                )
                fold_feature_cache[fold.fold_id] = select_fold_features(
                    target_frame.loc[fit_mask_for_selection],
                    features,
                    args.minimum_feature_coverage,
                    args.maximum_abs_correlation,
                    args.max_features,
                )
            for model_id in models:
                for fold in folds:
                    paths = model_result_paths(output_root, target.target_id, model_id, fold.fold_id)
                    signature = model_fold_signature(
                        target,
                        model_id,
                        fold,
                        snapshot_signature,
                        label_signature,
                        event_signatures_by_target[target.target_id],
                        {
                            "features": features,
                            "alert_rates": alert_rates,
                            "cooldown_seconds": cooldown_seconds,
                            "minimum_feature_coverage": args.minimum_feature_coverage,
                            "maximum_abs_correlation": args.maximum_abs_correlation,
                            "max_features": args.max_features,
                            "threshold_grid_size": args.threshold_grid_size,
                            "random_state": args.random_state,
                        },
                    )
                    if (
                        not args.force
                        and reusable_model_fold(
                            connection,
                            target.target_id,
                            model_id,
                            fold.fold_id,
                            signature,
                            paths,
                        )
                    ):
                        all_results.append(json.loads(paths.result_json.read_text(encoding="utf-8")))
                        skipped_model_folds += 1
                        continue
                    connection.execute(
                        """
                        INSERT INTO model_folds(
                            target_id, model_id, fold_id, input_signature,
                            prediction_path, model_path, result_path, status, last_run_id
                        ) VALUES(?,?,?,?,?,?,?,?,?)
                        ON CONFLICT(target_id, model_id, fold_id) DO UPDATE SET
                            input_signature=excluded.input_signature,
                            prediction_path=excluded.prediction_path,
                            model_path=excluded.model_path,
                            result_path=excluded.result_path,
                            status=excluded.status,
                            error_text=NULL,
                            last_run_id=excluded.last_run_id
                        """,
                        (
                            target.target_id,
                            model_id,
                            fold.fold_id,
                            signature,
                            str(paths.predictions),
                            str(paths.model),
                            str(paths.result_json),
                            "RUNNING",
                            run_id,
                        ),
                    )
                    connection.commit()
                    try:
                        prediction_frame, payload = run_model_fold(
                            target_frame,
                            features,
                            target,
                            model_id,
                            fold,
                            events_by_target[target.target_id],
                            alert_rates,
                            cooldown_seconds,
                            args.minimum_feature_coverage,
                            args.maximum_abs_correlation,
                            args.max_features,
                            args.threshold_grid_size,
                            args.random_state,
                            preselected_features=fold_feature_cache[fold.fold_id][0],
                            preselection=fold_feature_cache[fold.fold_id][1],
                        )
                        result = payload["result"]
                        result["input_signature"] = signature
                        CORE.atomic_write_parquet(
                            paths.predictions,
                            prediction_frame,
                            {
                                "perps_expansion_baseline_builder_version": BUILDER_VERSION,
                                "perps_expansion_baseline_prediction_schema_version": OUTPUT_SCHEMA_VERSION,
                                "target_id": target.target_id,
                                "model_id": model_id,
                                "fold_id": fold.fold_id,
                                "input_signature": signature,
                            },
                            args.compression_level,
                        )
                        atomic_joblib_dump(paths.model, payload["model"])
                        CORE.atomic_write_json(paths.result_json, result)
                        connection.execute(
                            """
                            UPDATE model_folds
                            SET status='COMPLETED', completed_at_utc=?, error_text=NULL,
                                last_run_id=?
                            WHERE target_id=? AND model_id=? AND fold_id=?
                            """,
                            (
                                iso_utc(utc_now()),
                                run_id,
                                target.target_id,
                                model_id,
                                fold.fold_id,
                            ),
                        )
                        connection.commit()
                        all_results.append(result)
                        completed_model_folds += 1
                        print(
                            f"  completed {model_id} fold {fold.fold_id}: "
                            f"features={result['selected_feature_count']}"
                        )
                    except Exception:
                        failed_model_folds += 1
                        connection.execute(
                            """
                            UPDATE model_folds
                            SET status='FAILED', error_text=?, last_run_id=?
                            WHERE target_id=? AND model_id=? AND fold_id=?
                            """,
                            (
                                traceback.format_exc(),
                                run_id,
                                target.target_id,
                                model_id,
                                fold.fold_id,
                            ),
                        )
                        connection.commit()
                        raise
            del labels, target_frame
            gc.collect()

        reports = aggregate_reports(all_results)
        folds_report = pd.concat(all_fold_frames, ignore_index=True)
        expected_results = len(plans) * len(models) * args.n_folds
        invariants = validate_run(
            snapshots,
            features,
            folds_by_target,
            reports,
            args.decision_cadence_seconds,
            expected_results,
        )
        failed_invariants = int(invariants["status"].eq("FAIL").sum())
        status = "COMPLETED" if failed_invariants == 0 and failed_model_folds == 0 else "FAILED"

        report_map = {
            "perps_expansion_baseline_folds.csv": folds_report,
            "perps_expansion_baseline_fold_metrics.csv": reports["fold_metrics"],
            "perps_expansion_baseline_alert_metrics.csv": reports["alert_metrics"],
            "perps_expansion_baseline_event_metrics.csv": reports["event_metrics"],
            "perps_expansion_baseline_model_summary.csv": reports["model_summary"],
            "perps_expansion_baseline_target_scorecard.csv": reports["target_scorecard"],
            "perps_expansion_baseline_calibration.csv": reports["calibration"],
            "perps_expansion_baseline_feature_selection.csv": reports["feature_selection"],
            "perps_expansion_baseline_feature_importance.csv": reports["feature_importance"],
            "perps_expansion_baseline_invariants.csv": invariants,
        }
        for filename, frame in report_map.items():
            CORE.atomic_write_csv(report_root / filename, frame)

        summary = {
            "status": status,
            "run_id": run_id,
            "builder_version": BUILDER_VERSION,
            "targets": len(plans),
            "models": len(models),
            "folds_per_target": args.n_folds,
            "expected_model_folds": expected_results,
            "completed_model_folds": completed_model_folds,
            "skipped_model_folds": skipped_model_folds,
            "failed_model_folds": failed_model_folds,
            "snapshot_rows": len(snapshots),
            "raw_feature_count": len(raw_features),
            "derived_feature_count": len(derived_feature_names),
            "candidate_model_feature_count": len(features),
            "rebuilt_snapshots": rebuilt_snapshots,
            "skipped_snapshots": skipped_snapshots,
            "validation_checks": len(invariants),
            "failed_invariants": failed_invariants,
        }
        CORE.atomic_write_json(
            report_root / "perps_expansion_baseline_summary.json", summary
        )
        CORE.atomic_write_json(
            report_root / "perps_expansion_baseline_config.json", configuration
        )
        CORE.atomic_write_text(
            report_root / "perps_expansion_baseline_findings.md",
            findings_markdown(summary, reports["target_scorecard"], reports["model_summary"]),
        )
        run_report_root = report_root / "runs" / run_id
        run_report_root.mkdir(parents=True, exist_ok=True)
        for filename in REPORT_FILENAMES:
            source = report_root / filename
            if source.exists():
                destination = run_report_root / filename
                destination.write_bytes(source.read_bytes())

        CORE.atomic_write_json(output_root / "_metadata" / "dataset_config.json", configuration)
        CORE.atomic_write_json(output_root / "_metadata" / "latest_run_summary.json", summary)

        connection.execute(
            """
            UPDATE runs
            SET completed_at_utc=?, status=?, rebuilt_snapshots=?, skipped_snapshots=?,
                completed_model_folds=?, skipped_model_folds=?, failed_model_folds=?,
                error_text=NULL
            WHERE run_id=?
            """,
            (
                iso_utc(utc_now()),
                status,
                rebuilt_snapshots,
                skipped_snapshots,
                completed_model_folds,
                skipped_model_folds,
                failed_model_folds,
                run_id,
            ),
        )
        connection.commit()
        print("Expansion baseline run complete.")
        print(f"Status              : {status}")
        print(f"Snapshot rows       : {len(snapshots):,}")
        print(f"Model folds         : {expected_results}")
        print(f"Completed this run  : {completed_model_folds}")
        print(f"Reused              : {skipped_model_folds}")
        print(f"Failed invariants   : {failed_invariants}")
        print(f"Findings            : {report_root / 'perps_expansion_baseline_findings.md'}")
        return 0 if status == "COMPLETED" else 1
    except Exception:
        error_text = traceback.format_exc()
        connection.execute(
            """
            UPDATE runs
            SET completed_at_utc=?, status='FAILED', rebuilt_snapshots=?,
                skipped_snapshots=?, completed_model_folds=?, skipped_model_folds=?,
                failed_model_folds=?, error_text=?
            WHERE run_id=?
            """,
            (
                iso_utc(utc_now()),
                rebuilt_snapshots,
                skipped_snapshots,
                completed_model_folds,
                skipped_model_folds,
                failed_model_folds,
                error_text,
                run_id,
            ),
        )
        connection.commit()
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
