#!/usr/bin/env python
"""
Audit the existing expansion signal against causal benchmarks and timing nulls.

This script is deliberately forensic. It does not train or refit any model and
it does not search for a new winner. It reuses the saved out-of-sample
CALIBRATION/TEST predictions and emitted alerts produced by Scripts 05 and 06,
then asks whether the selected expansion systems add value beyond matched
random timing, calendar structure, trailing volatility and causal recent range.

Primary scientific question
---------------------------
Do the existing out-of-sample model scores and alerts detect future expansion
events better than matched random timing, calendar structure, trailing
volatility and causal recent-range rules, and are any apparent advantages
stable across folds, days and causal volatility regimes?

Frozen reference systems
------------------------
A  H20M_T075BPS / HGB / rolling percentile / 2 alerts per day / horizon cooldown
B  H20M_T075BPS / HGB / daily trailing / 2 alerts per day / horizon cooldown
C  H15M_T075BPS / HGB / rolling percentile / 2 alerts per day / horizon cooldown
D  H05M_T050BPS / HGB / rolling percentile / 2 alerts per day / horizon cooldown

Important contracts
-------------------
* No model fitting or refitting.
* No future-defined event membership is used to decide whether a row is
  eligible. Every context-ready TEST prediction row remains eligible.
* Retrospective event information is used only for evaluation.
* Benchmark scores at t use only data timestamped at or before t.
* Benchmark alert policies use the same calibration/test periods, requested
  alert rate and cooldown as their frozen model reference.
* Random timing nulls preserve the model's actual daily alert count and enforce
  the same cooldown.
* Event matching is one-to-one and uses the existing Script 05 contract.
* Existing TEST folds are research-development data. This script is an audit,
  not a final holdout confirmation.

Default outputs
---------------
Dataset root:
    data/derived/perps/research/expansion_signal_forensics_v1

Report root:
    reports/perps_expansion_signal_forensics

Dependencies
------------
    python -m pip install numpy pandas pyarrow numba

Typical Git Bash usage
----------------------
    python scripts/perps/research/08_audit_expansion_signal_and_nulls.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sqlite3
import sys
import traceback
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd


AUDITOR_VERSION = "1.0.1"
OUTPUT_SCHEMA_VERSION = 1
EXPECTED_BASELINE_VERSION = "1.0.2"
EXPECTED_POLICY_VERSION = "1.0.2"
SECONDS_PER_DAY = 86_400
DEFAULT_RANDOM_STATE = 20260806
DEFAULT_NULL_REPETITIONS = 500
DEFAULT_BOOTSTRAP_REPETITIONS = 1000
DEFAULT_BLOCK_LENGTHS = (1, 3, 5)
DEFAULT_ALERT_RATE = 2.0
DEFAULT_NULLS = (
    "UNIFORM_MATCHED_DAILY_COUNT",
    "WITHIN_DAY_CIRCULAR_SHIFT",
    "DAY_SCHEDULE_PERMUTATION",
)
DEFAULT_BENCHMARKS = (
    "CALENDAR_PROPENSITY",
    "TRAILING_RV_5M",
    "TRAILING_RV_15M",
    "PAST_RANGE_1M",
    "PAST_RANGE_3M",
    "PAST_RANGE_5M",
    "PAST_RANGE_5M_NORMALIZED_BY_LONG_RV",
    "ABS_RETURN_1M",
    "ABS_RETURN_5M",
)
PRIMARY_ROLE = "PRIMARY"
ROBUSTNESS_ROLE = "ROBUSTNESS"

REPORT_FILENAMES = (
    "expansion_signal_exact_counts.csv",
    "expansion_signal_timing_decomposition.csv",
    "expansion_signal_range_at_alert.csv",
    "expansion_signal_event_concentration.csv",
    "expansion_signal_extreme_day_sensitivity.csv",
    "expansion_signal_volatility_regimes.csv",
    "expansion_signal_score_correlations.csv",
    "expansion_signal_alert_overlap.csv",
    "expansion_signal_incremental_recall.csv",
    "expansion_signal_benchmark_scorecard.csv",
    "expansion_signal_random_null_distribution.csv",
    "expansion_signal_paired_differences.csv",
    "expansion_signal_block_uncertainty.csv",
    "expansion_signal_fold_selection_sensitivity.csv",
    "expansion_signal_invariants.csv",
    "expansion_signal_summary.json",
    "expansion_signal_config.json",
    "expansion_signal_findings.md",
)


# ---------------------------------------------------------------------------
# Existing audited methodology
# ---------------------------------------------------------------------------


def _load_sibling_module(filename: str, module_name: str) -> ModuleType:
    local = Path(__file__).resolve().with_name(filename)
    candidates = (
        local,
        Path.cwd() / "scripts/perps/research" / filename,
        Path.cwd() / filename,
        Path("/mnt/data") / filename,
    )
    path = next((candidate.resolve() for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise RuntimeError(
            f"Required research script was not found: scripts/perps/research/{filename}"
        )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import methodology from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_sibling_module(
    "05_build_and_evaluate_expansion_baselines.py",
    "stream_research_expansion_baseline_forensics_core",
)
POLICY = _load_sibling_module(
    "06_evaluate_expansion_alert_policies.py",
    "stream_research_expansion_policy_forensics_core",
)
CORE = BASE.CORE


# ---------------------------------------------------------------------------
# Frozen configuration and data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceConfiguration:
    config_id: str
    target_id: str
    model_id: str
    policy_id: str
    requested_alerts_per_day: float
    cooldown_id: str
    description: str


FROZEN_CONFIGURATIONS: tuple[ReferenceConfiguration, ...] = (
    ReferenceConfiguration(
        "A",
        "H20M_T075BPS",
        "hist_gradient_boosting",
        "ROLLING_PERCENTILE",
        DEFAULT_ALERT_RATE,
        "HORIZON",
        "Stable early-warning reference",
    ),
    ReferenceConfiguration(
        "B",
        "H20M_T075BPS",
        "hist_gradient_boosting",
        "DAILY_TRAILING_THRESHOLD",
        DEFAULT_ALERT_RATE,
        "HORIZON",
        "More responsive early-warning challenger",
    ),
    ReferenceConfiguration(
        "C",
        "H15M_T075BPS",
        "hist_gradient_boosting",
        "ROLLING_PERCENTILE",
        DEFAULT_ALERT_RATE,
        "HORIZON",
        "Practical 75-bps event reference",
    ),
    ReferenceConfiguration(
        "D",
        "H05M_T050BPS",
        "hist_gradient_boosting",
        "ROLLING_PERCENTILE",
        DEFAULT_ALERT_RATE,
        "HORIZON",
        "Fast confirmation reference",
    ),
)


@dataclass(frozen=True)
class UnitPaths:
    root: Path
    systems: Path
    alerts: Path
    matches: Path
    timing: Path
    daily_components: Path
    regime_metrics: Path
    null_distribution: Path
    result_json: Path


@dataclass(frozen=True)
class UnitContext:
    reference: ReferenceConfiguration
    target: Any
    fold_id: str
    cooldown_seconds: int
    calibration: pd.DataFrame
    test: pd.DataFrame
    snapshots: pd.DataFrame
    events_by_role: Mapping[str, pd.DataFrame]


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------


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


def input_signature(path: Path, hash_content: bool) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Required input does not exist: {path}")
    stat = path.stat()
    value: dict[str, Any] = {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if hash_content:
        value["sha256"] = sha256_file(path)
    return value


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Required JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))


def parse_positive_ints(value: str, name: str) -> tuple[int, ...]:
    parsed = tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError(f"{name} must contain positive integers")
    return parsed


def percent(numerator: float | int, denominator: float | int) -> float:
    return float(numerator) / float(denominator) if denominator else float("nan")


def safe_mean(values: Iterable[Any]) -> float:
    array = pd.to_numeric(pd.Series(list(values), dtype="object"), errors="coerce").dropna()
    return float(array.mean()) if len(array) else float("nan")


def safe_median(values: Iterable[Any]) -> float:
    array = pd.to_numeric(pd.Series(list(values), dtype="object"), errors="coerce").dropna()
    return float(array.median()) if len(array) else float("nan")


def quantile(values: Sequence[float], probability: float) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, probability, method="linear")) if len(array) else float("nan")


def day_string(second: int) -> str:
    return datetime.fromtimestamp(int(second), tz=timezone.utc).date().isoformat()


def elapsed_test_days(seconds: np.ndarray) -> int:
    if not len(seconds):
        return 0
    return int(len(np.unique(seconds // SECONDS_PER_DAY)))


def invariant_record(
    scope: str,
    name: str,
    passed: bool,
    details: str,
    config_id: str | None = None,
    fold_id: str | None = None,
) -> dict[str, Any]:
    return {
        "scope": scope,
        "config_id": config_id,
        "fold_id": fold_id,
        "check_name": name,
        "status": "PASS" if passed else "FAIL",
        "details": details,
    }


# ---------------------------------------------------------------------------
# Windows-safe atomic writers
# ---------------------------------------------------------------------------


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".txt-{uuid.uuid4().hex[:8]}.tmp"
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        CORE.replace_with_retry(temporary, path)
    finally:
        CORE.unlink_with_retry(temporary)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".csv-{uuid.uuid4().hex[:8]}.tmp"
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        CORE.replace_with_retry(temporary, path)
    finally:
        CORE.unlink_with_retry(temporary)


def atomic_write_parquet(
    path: Path,
    frame: pd.DataFrame,
    metadata: Mapping[str, Any],
    compression_level: int,
) -> None:
    pa_module, _, pq_module = CORE.require_arrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".pq-{uuid.uuid4().hex[:8]}.tmp"
    table = pa_module.Table.from_pandas(frame, preserve_index=False)
    table = table.replace_schema_metadata(CORE.metadata_bytes(metadata))
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
            rows = int(check.metadata.num_rows)
            if rows != len(frame):
                raise RuntimeError(f"Parquet row mismatch for {path}: {rows} != {len(frame)}")
        finally:
            check.close()
        CORE.replace_with_retry(temporary, path)
    finally:
        CORE.unlink_with_retry(temporary)


def read_parquet(path: Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    _, _, pq_module = CORE.require_arrow()
    if not path.exists():
        raise RuntimeError(f"Required Parquet file does not exist: {path}")
    parquet_file = pq_module.ParquetFile(path)
    try:
        return parquet_file.read(columns=list(columns) if columns is not None else None).to_pandas()
    finally:
        parquet_file.close()


# ---------------------------------------------------------------------------
# SQLite resumability
# ---------------------------------------------------------------------------


def connect_state(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=FULL;
        CREATE TABLE IF NOT EXISTS runs(
            run_id TEXT PRIMARY KEY,
            started_at_utc TEXT NOT NULL,
            completed_at_utc TEXT,
            status TEXT NOT NULL,
            configuration_json TEXT NOT NULL,
            error_text TEXT
        );
        CREATE TABLE IF NOT EXISTS units(
            config_id TEXT NOT NULL,
            fold_id TEXT NOT NULL,
            input_signature TEXT NOT NULL,
            status TEXT NOT NULL,
            result_path TEXT NOT NULL,
            completed_at_utc TEXT,
            last_run_id TEXT,
            error_text TEXT,
            PRIMARY KEY(config_id, fold_id)
        );
        """
    )
    connection.commit()
    return connection


def unit_paths(output_root: Path, config_id: str, fold_id: str) -> UnitPaths:
    root = output_root / "units" / f"config={config_id}" / f"fold={fold_id}"
    return UnitPaths(
        root=root,
        systems=root / "systems.parquet",
        alerts=root / "alerts.parquet",
        matches=root / "matches.parquet",
        timing=root / "timing.parquet",
        daily_components=root / "daily_components.parquet",
        regime_metrics=root / "regime_metrics.parquet",
        null_distribution=root / "null_distribution.parquet",
        result_json=root / "result.json",
    )


def unit_is_reusable(
    connection: sqlite3.Connection,
    config_id: str,
    fold_id: str,
    signature: str,
    paths: UnitPaths,
    force: bool,
) -> bool:
    if force:
        return False
    row = connection.execute(
        "SELECT * FROM units WHERE config_id=? AND fold_id=?",
        (config_id, fold_id),
    ).fetchone()
    required = (
        paths.systems,
        paths.alerts,
        paths.matches,
        paths.timing,
        paths.daily_components,
        paths.regime_metrics,
        paths.null_distribution,
        paths.result_json,
    )
    return bool(
        row is not None
        and row["status"] == "COMPLETED"
        and row["input_signature"] == signature
        and all(path.exists() for path in required)
    )


# ---------------------------------------------------------------------------
# Paths and source loading
# ---------------------------------------------------------------------------


def target_from_config(item: Mapping[str, Any]) -> Any:
    return BASE.TargetPlan(
        target_id=str(item["target_id"]),
        horizon_minutes=int(item["horizon_minutes"]),
        horizon_seconds=int(item["horizon_seconds"]),
        threshold_bps=float(item["threshold_bps"]),
        primary_segmentation_id=str(item["primary_segmentation_id"]),
        robustness_segmentation_id=str(item["robustness_segmentation_id"]),
    )


def policy_context(reference: ReferenceConfiguration, target: Any, fold_id: str) -> Any:
    cooldown_id, cooldown_seconds = POLICY.resolve_cooldown(
        reference.cooldown_id,
        target.horizon_seconds,
    )
    return POLICY.PolicyContext(
        target=target,
        model_id=reference.model_id,
        fold_id=fold_id,
        cooldown_id=cooldown_id,
        cooldown_seconds=cooldown_seconds,
    )


def prediction_path(baseline_root: Path, reference: ReferenceConfiguration, fold_id: str) -> Path:
    return POLICY.prediction_path(
        baseline_root,
        reference.target_id,
        reference.model_id,
        fold_id,
    )


def source_alert_path(
    policy_root: Path,
    reference: ReferenceConfiguration,
    target: Any,
    fold_id: str,
) -> Path:
    context = policy_context(reference, target, fold_id)
    return POLICY.experiment_paths(policy_root, context, reference.policy_id).alerts


def event_path(baseline_root: Path, target: Any, role: str, symbol: str) -> Path:
    return POLICY.event_catalog_path(baseline_root, target, role, symbol)


def load_full_events(path: Path, target_id: str, role: str) -> pd.DataFrame:
    frame = read_parquet(path)
    required = {
        "event_id",
        "event_start_second",
        "first_realization_second",
        "scorable_event",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Event catalog {path} is missing columns: {missing}")
    if "target_id" in frame and set(frame["target_id"].astype(str)) != {target_id}:
        raise RuntimeError(f"Unexpected target values in {path}")
    frame["event_start_second"] = pd.to_numeric(
        frame["event_start_second"], errors="raise"
    ).astype(np.int64)
    frame["first_realization_second"] = pd.to_numeric(
        frame["first_realization_second"], errors="coerce"
    ).astype("Int64")
    frame["scorable_event"] = frame["scorable_event"].fillna(False).astype(bool)
    frame["segmentation_role"] = role
    frame = frame.sort_values("event_start_second", kind="mergesort").reset_index(drop=True)
    if frame["event_id"].astype(str).duplicated().any():
        raise RuntimeError(f"Duplicate event IDs in {path}")
    return frame


def load_source_alerts(
    path: Path,
    reference: ReferenceConfiguration,
    fold_id: str,
) -> pd.DataFrame:
    frame = read_parquet(path)
    required = {
        "target_id",
        "model_id",
        "fold_id",
        "policy_id",
        "cooldown_id",
        "requested_alerts_per_day",
        "alert_second",
        "score",
        "threshold_history_end_second",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Alert file {path} is missing columns: {missing}")
    mask = (
        frame["target_id"].astype(str).eq(reference.target_id)
        & frame["model_id"].astype(str).eq(reference.model_id)
        & frame["fold_id"].astype(str).str.zfill(2).eq(str(fold_id).zfill(2))
        & frame["policy_id"].astype(str).str.upper().eq(reference.policy_id)
        & frame["cooldown_id"].astype(str).str.upper().eq(reference.cooldown_id)
        & np.isclose(
            pd.to_numeric(frame["requested_alerts_per_day"], errors="raise").to_numpy(float),
            reference.requested_alerts_per_day,
        )
    )
    frame = frame[mask].copy()
    frame["alert_second"] = pd.to_numeric(frame["alert_second"], errors="raise").astype(np.int64)
    frame["score"] = pd.to_numeric(frame["score"], errors="raise").astype(float)
    frame["threshold_history_end_second"] = pd.to_numeric(
        frame["threshold_history_end_second"], errors="raise"
    ).astype(np.int64)
    frame = frame.sort_values("alert_second", kind="mergesort").reset_index(drop=True)
    if frame["alert_second"].duplicated().any():
        raise RuntimeError(f"Duplicate source alerts in {path}")
    return frame


def discover_snapshot_paths(
    baseline_root: Path,
    symbol: str,
    seconds: np.ndarray,
) -> list[Path]:
    days = sorted({day_string(int(second)) for second in seconds})
    paths = [
        baseline_root
        / "snapshots"
        / f"symbol={symbol}"
        / f"date={day}"
        / "part-00000.parquet"
        for day in days
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} snapshot partitions; first={missing[0]}")
    return paths


def load_aligned_snapshots(
    baseline_root: Path,
    symbol: str,
    prediction_seconds: np.ndarray,
) -> tuple[pd.DataFrame, list[Path]]:
    paths = discover_snapshot_paths(baseline_root, symbol, prediction_seconds)
    frames = [read_parquet(path) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    if "receive_second" not in frame:
        raise RuntimeError("Snapshot rows are missing receive_second")
    frame["decision_second"] = pd.to_numeric(
        frame["receive_second"], errors="raise"
    ).astype(np.int64)
    frame = frame[frame["decision_second"].isin(prediction_seconds)].copy()
    frame = frame.sort_values("decision_second", kind="mergesort").reset_index(drop=True)
    if frame["decision_second"].duplicated().any():
        raise RuntimeError("Aligned snapshots contain duplicate decision seconds")
    expected = set(map(int, prediction_seconds))
    actual = set(frame["decision_second"].astype(int))
    if expected != actual:
        raise RuntimeError(
            f"Snapshot/prediction alignment mismatch: missing={len(expected-actual)} extra={len(actual-expected)}"
        )
    return frame, paths


# ---------------------------------------------------------------------------
# Causal benchmark scores
# ---------------------------------------------------------------------------


def _find_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    return next((column for column in candidates if column in frame.columns), None)


def _numeric(frame: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    column = _find_column(frame, candidates)
    if column is None:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").astype(float)


def causal_trailing_range_bps(
    seconds: np.ndarray,
    prices: np.ndarray,
    window_seconds: int,
) -> np.ndarray:
    seconds = np.asarray(seconds, dtype=np.int64)
    prices = np.asarray(prices, dtype=float)
    output = np.full(len(seconds), np.nan, dtype=float)
    left = 0
    for right in range(len(seconds)):
        while left < right and int(seconds[left]) < int(seconds[right]) - int(window_seconds):
            left += 1
        segment = prices[left : right + 1]
        finite = segment[np.isfinite(segment) & (segment > 0)]
        if len(finite) >= 2:
            reference = float(finite[-1])
            output[right] = (float(np.max(finite)) - float(np.min(finite))) / reference * 10_000.0
    return output


def calendar_propensity(calibration: pd.DataFrame, combined: pd.DataFrame) -> np.ndarray:
    cal_seconds = calibration["decision_second"].to_numpy(np.int64)
    cal_y = calibration["target"].to_numpy(np.int8)
    cal_dt = pd.to_datetime(cal_seconds, unit="s", utc=True)
    key = pd.DataFrame(
        {
            "weekday": cal_dt.weekday,
            "hour": cal_dt.hour,
            "target": cal_y,
        }
    )
    global_mean = float((np.sum(cal_y) + 2.0) / (len(cal_y) + 4.0))
    grouped = key.groupby(["weekday", "hour"], as_index=False).agg(
        positives=("target", "sum"),
        rows=("target", "size"),
    )
    grouped["score"] = (grouped["positives"] + 5.0 * global_mean) / (grouped["rows"] + 5.0)
    mapping = {
        (int(row.weekday), int(row.hour)): float(row.score)
        for row in grouped.itertuples(index=False)
    }
    dt = pd.to_datetime(combined["decision_second"].to_numpy(np.int64), unit="s", utc=True)
    return np.asarray(
        [mapping.get((int(weekday), int(hour)), global_mean) for weekday, hour in zip(dt.weekday, dt.hour)],
        dtype=float,
    )


def align_exact_prices_to_decisions(
    decision_seconds: np.ndarray,
    exact_seconds: np.ndarray,
    exact_prices: np.ndarray,
) -> np.ndarray:
    """Align audited one-second causal prices to exact decision timestamps.

    Script 05 snapshots intentionally contain model features rather than the raw
    ``price_for_features`` field. Script 08 already loads that audited one-second
    price series for timing diagnostics, so range benchmarks should use the same
    source instead of expecting a price column inside the snapshot dataset.
    """
    decisions = np.asarray(decision_seconds, dtype=np.int64)
    source_seconds = np.asarray(exact_seconds, dtype=np.int64)
    source_prices = np.asarray(exact_prices, dtype=float)
    if len(source_seconds) != len(source_prices):
        raise RuntimeError("Exact-price seconds and values have different lengths")
    if len(source_seconds) == 0:
        raise RuntimeError("Exact-price series is empty")
    if np.any(source_seconds[1:] <= source_seconds[:-1]):
        raise RuntimeError("Exact-price seconds must be strictly increasing")

    positions = np.searchsorted(source_seconds, decisions, side="left")
    in_bounds = positions < len(source_seconds)
    matched = np.zeros(len(decisions), dtype=bool)
    matched[in_bounds] = source_seconds[positions[in_bounds]] == decisions[in_bounds]
    if not matched.all():
        missing = decisions[~matched]
        preview = ",".join(map(str, missing[:5]))
        raise RuntimeError(
            f"Exact causal prices are missing for {len(missing)} decision seconds; "
            f"first={preview}"
        )

    aligned = source_prices[positions]
    invalid = ~np.isfinite(aligned) | (aligned <= 0.0)
    if invalid.any():
        bad = decisions[invalid]
        preview = ",".join(map(str, bad[:5]))
        raise RuntimeError(
            f"Exact causal prices are invalid for {len(bad)} decision seconds; "
            f"first={preview}"
        )
    return aligned


def build_benchmark_scores(
    calibration_predictions: pd.DataFrame,
    combined: pd.DataFrame,
    exact_seconds: np.ndarray,
    exact_prices: np.ndarray,
) -> pd.DataFrame:
    output = combined[["decision_second", "split", "target"]].copy()
    seconds = output["decision_second"].to_numpy(np.int64)
    prices = align_exact_prices_to_decisions(seconds, exact_seconds, exact_prices)

    output["CALENDAR_PROPENSITY"] = calendar_propensity(calibration_predictions, combined)
    output["TRAILING_RV_5M"] = _numeric(
        combined,
        ("realized_vol_300s_bps", "realized_volatility_300s_bps"),
    )
    output["TRAILING_RV_15M"] = _numeric(
        combined,
        ("realized_vol_900s_bps", "realized_volatility_900s_bps"),
    )
    output["PAST_RANGE_1M"] = causal_trailing_range_bps(seconds, prices, 60)
    output["PAST_RANGE_3M"] = causal_trailing_range_bps(seconds, prices, 180)
    output["PAST_RANGE_5M"] = causal_trailing_range_bps(seconds, prices, 300)
    long_rv = _numeric(
        combined,
        ("realized_vol_3600s_bps", "realized_volatility_3600s_bps"),
    ).to_numpy(float)
    output["PAST_RANGE_5M_NORMALIZED_BY_LONG_RV"] = np.divide(
        output["PAST_RANGE_5M"].to_numpy(float),
        long_rv,
        out=np.full(len(output), np.nan, dtype=float),
        where=np.isfinite(long_rv) & (long_rv > 0),
    )
    output["ABS_RETURN_1M"] = _numeric(
        combined,
        ("abs_return_60s_bps", "return_60s_bps"),
    ).abs()
    output["ABS_RETURN_5M"] = _numeric(
        combined,
        ("abs_return_300s_bps", "return_300s_bps"),
    ).abs()

    for column in DEFAULT_BENCHMARKS:
        values = pd.to_numeric(output[column], errors="coerce").to_numpy(float)
        finite = values[np.isfinite(values)]
        if not len(finite):
            raise RuntimeError(f"Benchmark {column} has no finite scores")
        # Past-only forward fill is permitted; no backward fill is used.
        output[column] = pd.Series(values).ffill().to_numpy(float)
    return output


# ---------------------------------------------------------------------------
# Benchmark policy conversion
# ---------------------------------------------------------------------------


def benchmark_alerts(
    score_frame: pd.DataFrame,
    score_column: str,
    context: UnitContext,
    policy_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = score_frame[["decision_second", "split", "target", score_column]].rename(
        columns={score_column: "probability"}
    )
    calibration = frame[frame["split"].eq("CALIBRATION")].copy()
    test = frame[frame["split"].eq("TEST")].copy()
    calibration = calibration[np.isfinite(calibration["probability"])].copy()
    test = test[np.isfinite(test["probability"])].copy()
    if calibration.empty or test.empty:
        raise RuntimeError(f"Benchmark {score_column} has empty calibration or test scores")

    policy_ctx = policy_context(context.reference, context.target, context.fold_id)
    common = {
        "requested_rate": context.reference.requested_alerts_per_day,
        "lookback_days": int(policy_config["trailing_lookback_days"]),
        "minimum_history_days": int(policy_config["minimum_history_days"]),
        "threshold_grid_size": int(policy_config["threshold_grid_size"]),
    }
    if context.reference.policy_id == "ROLLING_PERCENTILE":
        alerts, thresholds = POLICY.rolling_percentile_policy(
            calibration,
            test,
            policy_ctx,
            common["requested_rate"],
            common["lookback_days"],
            common["minimum_history_days"],
            int(policy_config["rolling_refresh_seconds"]),
            common["threshold_grid_size"],
            capped=False,
        )
    elif context.reference.policy_id == "DAILY_TRAILING_THRESHOLD":
        alerts, thresholds = POLICY.daily_trailing_policy(
            calibration,
            test,
            policy_ctx,
            common["requested_rate"],
            common["lookback_days"],
            common["minimum_history_days"],
            common["threshold_grid_size"],
            capped=False,
        )
    else:
        raise RuntimeError(
            f"Frozen benchmark conversion does not support policy {context.reference.policy_id}"
        )
    alerts = alerts.rename(columns={"score": "system_score"}).copy()
    alerts["system_id"] = score_column
    alerts["system_kind"] = "CAUSAL_BENCHMARK"
    thresholds["system_id"] = score_column
    thresholds["system_kind"] = "CAUSAL_BENCHMARK"
    return alerts, thresholds


# ---------------------------------------------------------------------------
# Event matching, counts and timing
# ---------------------------------------------------------------------------


def scorable_events(context: UnitContext, role: str) -> pd.DataFrame:
    seconds = context.test["decision_second"].to_numpy(np.int64)
    return BASE.scorable_events_for_test(
        context.events_by_role[role],
        seconds,
        int(seconds[0]),
        int(seconds[-1]) + 1,
    )


def match_system_alerts(
    alert_frame: pd.DataFrame,
    context: UnitContext,
    system_id: str,
    system_kind: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    alert_seconds = (
        alert_frame["alert_second"].to_numpy(np.int64)
        if not alert_frame.empty
        else np.empty(0, dtype=np.int64)
    )
    matches_out: list[pd.DataFrame] = []
    false_out: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    test_days = elapsed_test_days(context.test["decision_second"].to_numpy(np.int64))
    for role in (PRIMARY_ROLE, ROBUSTNESS_ROLE):
        events = scorable_events(context, role)
        matches, false_alerts = BASE.match_alerts_to_events(alert_seconds, events)
        if not matches.empty:
            matches.insert(0, "config_id", context.reference.config_id)
            matches.insert(1, "fold_id", context.fold_id)
            matches.insert(2, "system_id", system_id)
            matches.insert(3, "system_kind", system_kind)
            matches.insert(4, "segmentation_role", role)
            matches_out.append(matches)
        if not false_alerts.empty:
            false_alerts.insert(0, "config_id", context.reference.config_id)
            false_alerts.insert(1, "fold_id", context.fold_id)
            false_alerts.insert(2, "system_id", system_id)
            false_alerts.insert(3, "system_kind", system_kind)
            false_alerts.insert(4, "segmentation_role", role)
            false_out.append(false_alerts)
        recalled = int(len(matches))
        alerts = int(len(alert_seconds))
        metrics.append(
            {
                "config_id": context.reference.config_id,
                "target_id": context.reference.target_id,
                "model_id": context.reference.model_id,
                "policy_id": context.reference.policy_id,
                "fold_id": context.fold_id,
                "system_id": system_id,
                "system_kind": system_kind,
                "segmentation_role": role,
                "test_days": test_days,
                "eligible_test_rows": int(len(context.test)),
                "positive_test_rows": int(context.test["target"].sum()),
                "alerts": alerts,
                "alerts_per_day": percent(alerts, test_days),
                "scorable_events": int(len(events)),
                "recalled_events": recalled,
                "event_recall": percent(recalled, len(events)),
                "event_precision": percent(recalled, alerts),
                "false_alerts": int(len(false_alerts)),
                "false_alerts_per_day": percent(len(false_alerts), test_days),
                "median_warning_seconds": safe_median(matches.get("warning_seconds", [])),
                "median_seconds_after_label_start": safe_median(
                    matches.get("seconds_after_event_onset", [])
                ),
            }
        )
    return (
        pd.concat(matches_out, ignore_index=True) if matches_out else pd.DataFrame(),
        pd.concat(false_out, ignore_index=True) if false_out else pd.DataFrame(),
        metrics,
    )


def _price_range_until_alert(
    exact_seconds: np.ndarray,
    exact_prices: np.ndarray,
    event_start: int,
    alert_second: int,
) -> float:
    left = int(np.searchsorted(exact_seconds, int(event_start) + 1, side="left"))
    right = int(np.searchsorted(exact_seconds, int(alert_second), side="right"))
    if left >= right:
        return 0.0
    seconds = exact_seconds[left:right]
    prices = exact_prices[left:right]
    expected = np.arange(int(event_start) + 1, int(alert_second) + 1, dtype=np.int64)
    if len(seconds) != len(expected) or not np.array_equal(seconds, expected):
        return float("nan")
    if not np.isfinite(prices).all() or np.any(prices <= 0):
        return float("nan")
    entry = float(prices[0])
    return (float(np.max(prices)) - float(np.min(prices))) / entry * 10_000.0


def timing_decomposition(
    matches: pd.DataFrame,
    context: UnitContext,
    exact_seconds: np.ndarray,
    exact_prices: np.ndarray,
) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    threshold = float(context.target.threshold_bps)
    for row in matches.itertuples(index=False):
        start = int(row.event_start_second)
        alert = int(row.alert_second)
        realization = int(row.first_realization_second)
        duration = max(1, realization - start)
        after = alert - start
        fraction = after / duration
        if alert < start:
            bucket = "BEFORE_LABEL_EPISODE_START"
        elif alert > realization:
            bucket = "POST_REALIZATION"
        elif fraction < 1.0 / 3.0:
            bucket = "AFTER_LABEL_START_EARLY"
        elif fraction < 2.0 / 3.0:
            bucket = "AFTER_LABEL_START_MIDDLE"
        else:
            bucket = "AFTER_LABEL_START_LATE"
        range_bps = _price_range_until_alert(
            exact_seconds,
            exact_prices,
            start,
            alert,
        )
        records.append(
            {
                "config_id": context.reference.config_id,
                "target_id": context.reference.target_id,
                "fold_id": context.fold_id,
                "system_id": str(row.system_id),
                "system_kind": str(row.system_kind),
                "segmentation_role": str(row.segmentation_role),
                "match_contract": "OFFICIAL_ONE_TO_ONE",
                "event_id": str(row.event_id),
                "event_start_second": start,
                "alert_second": alert,
                "first_realization_second": realization,
                "seconds_after_label_start": after,
                "seconds_to_realization": realization - alert,
                "label_to_realization_seconds": duration,
                "relative_timing_fraction": fraction,
                "timing_bucket": bucket,
                "target_threshold_bps": threshold,
                "range_realized_by_alert_bps": range_bps,
                "fraction_of_target_range_realized": range_bps / threshold
                if math.isfinite(range_bps) and threshold > 0
                else np.nan,
                "remaining_bps_to_target": max(0.0, threshold - range_bps)
                if math.isfinite(range_bps)
                else np.nan,
            }
        )
    return pd.DataFrame(records)


def extended_prelabel_diagnostic(
    alert_frame: pd.DataFrame,
    context: UnitContext,
    role: str,
) -> pd.DataFrame:
    """Diagnostic only: match alerts in [start-horizon, realization].

    This does not alter official event recall and does not use event information
    to select alerts. It only asks retrospectively whether an already-emitted
    alert existed before the label episode start.
    """
    events = scorable_events(context, role).sort_values("first_realization_second")
    alerts = alert_frame["alert_second"].to_numpy(np.int64) if not alert_frame.empty else np.empty(0, np.int64)
    unmatched = set(range(len(events)))
    rows: list[dict[str, Any]] = []
    for alert in alerts:
        candidates = [
            index
            for index in unmatched
            if int(events.iloc[index]["event_start_second"]) - context.target.horizon_seconds
            <= int(alert)
            <= int(events.iloc[index]["first_realization_second"])
        ]
        if not candidates:
            continue
        chosen = min(
            candidates,
            key=lambda idx: (
                int(events.iloc[idx]["first_realization_second"]),
                int(events.iloc[idx]["event_start_second"]),
            ),
        )
        unmatched.remove(chosen)
        event = events.iloc[chosen]
        rows.append(
            {
                "config_id": context.reference.config_id,
                "fold_id": context.fold_id,
                "segmentation_role": role,
                "event_id": str(event["event_id"]),
                "alert_second": int(alert),
                "event_start_second": int(event["event_start_second"]),
                "first_realization_second": int(event["first_realization_second"]),
                "seconds_relative_to_label_start": int(alert) - int(event["event_start_second"]),
                "pre_label_alert": int(alert) < int(event["event_start_second"]),
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows)


def prelabel_timing_decomposition(
    diagnostic: pd.DataFrame,
    context: UnitContext,
    exact_seconds: np.ndarray,
    exact_prices: np.ndarray,
) -> pd.DataFrame:
    if diagnostic.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    threshold = float(context.target.threshold_bps)
    for row in diagnostic.itertuples(index=False):
        start = int(row.event_start_second)
        alert = int(row.alert_second)
        realization = int(row.first_realization_second)
        duration = max(1, realization - start)
        after = alert - start
        fraction = after / duration
        if alert < start:
            bucket = "BEFORE_LABEL_EPISODE_START"
            range_bps = float("nan")
        elif alert > realization:
            bucket = "POST_REALIZATION"
            range_bps = float("nan")
        elif fraction < 1.0 / 3.0:
            bucket = "AFTER_LABEL_START_EARLY"
            range_bps = _price_range_until_alert(
                exact_seconds, exact_prices, start, alert
            )
        elif fraction < 2.0 / 3.0:
            bucket = "AFTER_LABEL_START_MIDDLE"
            range_bps = _price_range_until_alert(
                exact_seconds, exact_prices, start, alert
            )
        else:
            bucket = "AFTER_LABEL_START_LATE"
            range_bps = _price_range_until_alert(
                exact_seconds, exact_prices, start, alert
            )
        records.append(
            {
                "config_id": context.reference.config_id,
                "target_id": context.reference.target_id,
                "fold_id": context.fold_id,
                "system_id": "MODEL",
                "system_kind": "MODEL",
                "segmentation_role": str(row.segmentation_role),
                "match_contract": "EXTENDED_PRELABEL_DIAGNOSTIC",
                "event_id": str(row.event_id),
                "event_start_second": start,
                "alert_second": alert,
                "first_realization_second": realization,
                "seconds_after_label_start": after,
                "seconds_to_realization": realization - alert,
                "label_to_realization_seconds": duration,
                "relative_timing_fraction": fraction,
                "timing_bucket": bucket,
                "target_threshold_bps": threshold,
                "range_realized_by_alert_bps": range_bps,
                "fraction_of_target_range_realized": range_bps / threshold
                if math.isfinite(range_bps) and threshold > 0
                else np.nan,
                "remaining_bps_to_target": max(0.0, threshold - range_bps)
                if math.isfinite(range_bps)
                else np.nan,
            }
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Daily components, concentration and regimes
# ---------------------------------------------------------------------------


def build_daily_components(
    alerts: pd.DataFrame,
    matches: pd.DataFrame,
    false_alerts: pd.DataFrame,
    context: UnitContext,
    system_id: str,
    system_kind: str,
    role: str,
) -> pd.DataFrame:
    test_days = sorted({day_string(value) for value in context.test["decision_second"].astype(int)})
    rows = pd.DataFrame({"test_date": test_days})
    rows["alerts"] = 0
    if not alerts.empty:
        counts = alerts.assign(test_date=alerts["alert_second"].map(day_string)).groupby(
            "test_date"
        ).size()
        rows["alerts"] = rows["test_date"].map(counts).fillna(0).astype(int)

    events = scorable_events(context, role).copy()
    events["test_date"] = events["event_start_second"].map(day_string)
    event_counts = events.groupby("test_date").size()
    rows["scorable_events"] = rows["test_date"].map(event_counts).fillna(0).astype(int)

    role_matches = matches[matches["segmentation_role"].eq(role)].copy() if not matches.empty else pd.DataFrame()
    if not role_matches.empty:
        role_matches["test_date"] = role_matches["event_start_second"].map(day_string)
        recalled = role_matches.groupby("test_date")["event_id"].nunique()
    else:
        recalled = pd.Series(dtype=int)
    rows["recalled_events"] = rows["test_date"].map(recalled).fillna(0).astype(int)

    role_false = false_alerts[false_alerts["segmentation_role"].eq(role)].copy() if not false_alerts.empty else pd.DataFrame()
    if not role_false.empty:
        role_false["test_date"] = role_false["alert_second"].map(day_string)
        false_counts = role_false.groupby("test_date").size()
    else:
        false_counts = pd.Series(dtype=int)
    rows["false_alerts"] = rows["test_date"].map(false_counts).fillna(0).astype(int)
    rows.insert(0, "config_id", context.reference.config_id)
    rows.insert(1, "fold_id", context.fold_id)
    rows.insert(2, "system_id", system_id)
    rows.insert(3, "system_kind", system_kind)
    rows.insert(4, "segmentation_role", role)
    return rows


def derive_regimes(calibration: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    short_col = _find_column(calibration, ("realized_vol_300s_bps", "realized_volatility_300s_bps"))
    long_col = _find_column(calibration, ("realized_vol_3600s_bps", "realized_volatility_3600s_bps"))
    if short_col is None or long_col is None:
        raise RuntimeError("Regime analysis requires 300s and 3600s realized-volatility columns")
    cal_short = pd.to_numeric(calibration[short_col], errors="coerce").to_numpy(float)
    cal_long = pd.to_numeric(calibration[long_col], errors="coerce").to_numpy(float)
    cal_ratio = np.divide(
        cal_short,
        cal_long,
        out=np.full(len(calibration), np.nan),
        where=np.isfinite(cal_long) & (cal_long > 0),
    )
    finite_ratio = cal_ratio[np.isfinite(cal_ratio)]
    finite_long = cal_long[np.isfinite(cal_long)]
    if len(finite_ratio) < 10 or len(finite_long) < 10:
        raise RuntimeError("Insufficient calibration volatility history for regimes")
    thresholds = {
        "ratio_q25": quantile(finite_ratio, 0.25),
        "ratio_q75": quantile(finite_ratio, 0.75),
        "long_q33": quantile(finite_long, 1.0 / 3.0),
        "long_q67": quantile(finite_long, 2.0 / 3.0),
        "history_end_second": int(calibration["decision_second"].max()),
    }
    test_short = pd.to_numeric(test[short_col], errors="coerce").to_numpy(float)
    test_long = pd.to_numeric(test[long_col], errors="coerce").to_numpy(float)
    ratio = np.divide(
        test_short,
        test_long,
        out=np.full(len(test), np.nan),
        where=np.isfinite(test_long) & (test_long > 0),
    )
    regime = np.full(len(test), "UNKNOWN", dtype=object)
    rising = np.isfinite(ratio) & (ratio > thresholds["ratio_q75"])
    falling = np.isfinite(ratio) & (ratio < thresholds["ratio_q25"])
    regime[rising] = "RISING_VOLATILITY"
    regime[falling] = "FALLING_VOLATILITY"
    remaining = ~(rising | falling)
    regime[remaining & (test_long <= thresholds["long_q33"])] = "LOW_VOLATILITY"
    regime[
        remaining
        & (test_long > thresholds["long_q33"])
        & (test_long <= thresholds["long_q67"])
    ] = "MEDIUM_VOLATILITY"
    regime[remaining & (test_long > thresholds["long_q67"])] = "HIGH_VOLATILITY"
    output = test[["decision_second"]].copy()
    output["regime"] = regime
    output["short_rv"] = test_short
    output["long_rv"] = test_long
    output["short_long_ratio"] = ratio
    return output, thresholds


def regime_metrics(
    regimes: pd.DataFrame,
    alerts: pd.DataFrame,
    matches: pd.DataFrame,
    context: UnitContext,
    system_id: str,
    system_kind: str,
    role: str,
) -> pd.DataFrame:
    test_with_regime = context.test[["decision_second"]].merge(regimes, on="decision_second", how="left")
    events = scorable_events(context, role).copy()
    event_regime = pd.merge_asof(
        events.sort_values("event_start_second"),
        regimes.sort_values("decision_second"),
        left_on="event_start_second",
        right_on="decision_second",
        direction="backward",
    )
    alert_regime = alerts.merge(regimes, left_on="alert_second", right_on="decision_second", how="left") if not alerts.empty else pd.DataFrame()
    role_matches = matches[matches["segmentation_role"].eq(role)].copy() if not matches.empty else pd.DataFrame()
    if not role_matches.empty:
        role_matches = role_matches.merge(
            event_regime[["event_id", "regime"]], on="event_id", how="left"
        )
    rows: list[dict[str, Any]] = []
    for regime_name in (
        "LOW_VOLATILITY",
        "MEDIUM_VOLATILITY",
        "HIGH_VOLATILITY",
        "RISING_VOLATILITY",
        "FALLING_VOLATILITY",
    ):
        eligible_rows = int(test_with_regime["regime"].eq(regime_name).sum())
        alert_count = int(alert_regime["regime"].eq(regime_name).sum()) if not alert_regime.empty else 0
        event_count = int(event_regime["regime"].eq(regime_name).sum())
        recalled = int(role_matches["regime"].eq(regime_name).sum()) if not role_matches.empty else 0
        rows.append(
            {
                "config_id": context.reference.config_id,
                "fold_id": context.fold_id,
                "system_id": system_id,
                "system_kind": system_kind,
                "segmentation_role": role,
                "regime": regime_name,
                "eligible_rows": eligible_rows,
                "alerts": alert_count,
                "scorable_events": event_count,
                "recalled_events": recalled,
                "event_recall": percent(recalled, event_count),
                "event_precision": percent(recalled, alert_count),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Matched timing nulls
# ---------------------------------------------------------------------------


def _valid_schedule(seconds: Sequence[int], cooldown_seconds: int) -> bool:
    values = np.sort(np.asarray(seconds, dtype=np.int64))
    return len(values) <= 1 or bool(np.all(np.diff(values) >= int(cooldown_seconds)))


def sample_uniform_schedule(
    eligible_seconds: np.ndarray,
    count: int,
    cooldown_seconds: int,
    rng: np.random.Generator,
    attempts: int = 64,
) -> np.ndarray:
    eligible = np.asarray(eligible_seconds, dtype=np.int64)
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    if count > len(eligible):
        raise RuntimeError("Requested more random alerts than eligible rows")
    for _ in range(attempts):
        order = rng.permutation(len(eligible))
        selected: list[int] = []
        for index in order:
            candidate = int(eligible[index])
            if all(abs(candidate - existing) >= cooldown_seconds for existing in selected):
                selected.append(candidate)
                if len(selected) == count:
                    return np.sort(np.asarray(selected, dtype=np.int64))
    # Deterministic fallback: randomly choose the starting phase, then greedily walk.
    start = int(rng.integers(0, len(eligible)))
    circular = np.concatenate([eligible[start:], eligible[:start] + SECONDS_PER_DAY])
    selected = []
    for candidate in circular:
        normalized = int(candidate)
        if normalized >= eligible[0] + SECONDS_PER_DAY:
            normalized -= SECONDS_PER_DAY
        if all(abs(normalized - existing) >= cooldown_seconds for existing in selected):
            selected.append(normalized)
            if len(selected) == count:
                return np.sort(np.asarray(selected, dtype=np.int64))
    raise RuntimeError(
        f"Could not draw {count} cooldown-valid alerts from {len(eligible)} eligible rows"
    )


def nearest_eligible(value: int, eligible: np.ndarray) -> int:
    position = int(np.searchsorted(eligible, int(value), side="left"))
    candidates: list[int] = []
    if position < len(eligible):
        candidates.append(int(eligible[position]))
    if position > 0:
        candidates.append(int(eligible[position - 1]))
    if not candidates:
        raise RuntimeError("No eligible timestamp exists")
    return min(candidates, key=lambda candidate: (abs(candidate - int(value)), candidate))


def circular_shift_schedule(
    source: np.ndarray,
    eligible: np.ndarray,
    cooldown_seconds: int,
    rng: np.random.Generator,
    attempts: int = 128,
) -> np.ndarray:
    source = np.asarray(source, dtype=np.int64)
    eligible = np.asarray(eligible, dtype=np.int64)
    if not len(source):
        return source.copy()
    day_start = int(eligible[0] // SECONDS_PER_DAY * SECONDS_PER_DAY)
    offsets = source - day_start
    for _ in range(attempts):
        shift = int(rng.integers(0, SECONDS_PER_DAY))
        mapped = np.asarray(
            [nearest_eligible(day_start + int((offset + shift) % SECONDS_PER_DAY), eligible) for offset in offsets],
            dtype=np.int64,
        )
        mapped = np.unique(mapped)
        if len(mapped) == len(source) and _valid_schedule(mapped, cooldown_seconds):
            return np.sort(mapped)
    return sample_uniform_schedule(eligible, len(source), cooldown_seconds, rng)


def permute_day_schedules(
    alerts_by_day: Mapping[str, np.ndarray],
    eligible_by_day: Mapping[str, np.ndarray],
    cooldown_seconds: int,
    rng: np.random.Generator,
) -> np.ndarray:
    days = sorted(eligible_by_day)
    source_days = list(days)
    rng.shuffle(source_days)
    output: list[int] = []
    for destination, source_day in zip(days, source_days):
        source = np.asarray(alerts_by_day.get(source_day, np.empty(0, np.int64)), dtype=np.int64)
        eligible = np.asarray(eligible_by_day[destination], dtype=np.int64)
        if not len(source):
            continue
        source_start = int(source[0] // SECONDS_PER_DAY * SECONDS_PER_DAY)
        destination_start = int(eligible[0] // SECONDS_PER_DAY * SECONDS_PER_DAY)
        offsets = source - source_start
        mapped = np.asarray(
            [nearest_eligible(destination_start + int(offset), eligible) for offset in offsets],
            dtype=np.int64,
        )
        if len(np.unique(mapped)) != len(source) or not _valid_schedule(mapped, cooldown_seconds):
            mapped = sample_uniform_schedule(eligible, len(source), cooldown_seconds, rng)
        output.extend(map(int, mapped))
    return np.sort(np.asarray(output, dtype=np.int64))


def null_alert_seconds(
    null_id: str,
    model_alerts: pd.DataFrame,
    test_seconds: np.ndarray,
    cooldown_seconds: int,
    rng: np.random.Generator,
) -> np.ndarray:
    days = sorted({day_string(value) for value in test_seconds})
    eligible_by_day = {
        day: test_seconds[np.asarray([day_string(value) == day for value in test_seconds])]
        for day in days
    }
    alerts_by_day = {
        day: model_alerts.loc[
            model_alerts["alert_second"].map(day_string).eq(day), "alert_second"
        ].to_numpy(np.int64)
        for day in days
    }
    if null_id == "UNIFORM_MATCHED_DAILY_COUNT":
        values = [
            sample_uniform_schedule(
                eligible_by_day[day],
                len(alerts_by_day[day]),
                cooldown_seconds,
                rng,
            )
            for day in days
        ]
        return np.concatenate(values) if values else np.empty(0, np.int64)
    if null_id == "WITHIN_DAY_CIRCULAR_SHIFT":
        values = [
            circular_shift_schedule(
                alerts_by_day[day],
                eligible_by_day[day],
                cooldown_seconds,
                rng,
            )
            for day in days
        ]
        return np.concatenate(values) if values else np.empty(0, np.int64)
    if null_id == "DAY_SCHEDULE_PERMUTATION":
        return permute_day_schedules(alerts_by_day, eligible_by_day, cooldown_seconds, rng)
    raise RuntimeError(f"Unknown null system: {null_id}")


def evaluate_null_distribution(
    model_alerts: pd.DataFrame,
    context: UnitContext,
    null_ids: Sequence[str],
    repetitions: int,
    random_state: int,
) -> pd.DataFrame:
    test_seconds = context.test["decision_second"].to_numpy(np.int64)
    records: list[dict[str, Any]] = []
    for null_ordinal, null_id in enumerate(null_ids):
        for repetition in range(repetitions):
            seed = int(random_state + 100_003 * int(context.fold_id) + 1_009 * ord(context.reference.config_id) + 17 * null_ordinal + repetition)
            rng = np.random.default_rng(seed)
            alert_seconds = null_alert_seconds(
                null_id,
                model_alerts,
                test_seconds,
                context.cooldown_seconds,
                rng,
            )
            if len(alert_seconds) != len(model_alerts):
                raise RuntimeError(
                    f"{null_id} did not preserve total alert count: {len(alert_seconds)} != {len(model_alerts)}"
                )
            for role in (PRIMARY_ROLE, ROBUSTNESS_ROLE):
                events = scorable_events(context, role)
                matches, false_alerts = BASE.match_alerts_to_events(alert_seconds, events)
                records.append(
                    {
                        "config_id": context.reference.config_id,
                        "target_id": context.reference.target_id,
                        "fold_id": context.fold_id,
                        "null_id": null_id,
                        "repetition": repetition,
                        "seed": seed,
                        "segmentation_role": role,
                        "alerts": int(len(alert_seconds)),
                        "scorable_events": int(len(events)),
                        "recalled_events": int(len(matches)),
                        "event_recall": percent(len(matches), len(events)),
                        "event_precision": percent(len(matches), len(alert_seconds)),
                        "false_alerts": int(len(false_alerts)),
                        "median_warning_seconds": safe_median(matches.get("warning_seconds", [])),
                    }
                )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Unit execution
# ---------------------------------------------------------------------------


def system_alert_frame(
    source: pd.DataFrame,
    system_id: str,
    system_kind: str,
) -> pd.DataFrame:
    output = source.copy()
    if "system_score" not in output:
        if "score" in output:
            output = output.rename(columns={"score": "system_score"})
        else:
            output["system_score"] = np.nan
    output["system_id"] = system_id
    output["system_kind"] = system_kind
    keep = [
        "system_id",
        "system_kind",
        "alert_second",
        "system_score",
        "threshold_used",
        "threshold_history_end_second",
    ]
    for column in keep:
        if column not in output:
            output[column] = pd.NA
    return output[keep].copy()


def execute_unit(
    context: UnitContext,
    source_alerts: pd.DataFrame,
    policy_config: Mapping[str, Any],
    benchmark_ids: Sequence[str],
    null_ids: Sequence[str],
    null_repetitions: int,
    exact_seconds: np.ndarray,
    exact_prices: np.ndarray,
    random_state: int,
) -> dict[str, pd.DataFrame | list[dict[str, Any]] | dict[str, Any]]:
    combined_predictions = pd.concat(
        [context.calibration, context.test], ignore_index=True
    ).sort_values("decision_second", kind="mergesort")
    combined = combined_predictions.merge(
        context.snapshots,
        on="decision_second",
        how="left",
        validate="one_to_one",
        suffixes=("", "_snapshot"),
    )
    benchmark_scores = build_benchmark_scores(
        context.calibration,
        combined,
        exact_seconds,
        exact_prices,
    )

    all_alerts: list[pd.DataFrame] = []
    all_matches: list[pd.DataFrame] = []
    all_false: list[pd.DataFrame] = []
    all_system_metrics: list[dict[str, Any]] = []
    all_timing: list[pd.DataFrame] = []
    all_daily: list[pd.DataFrame] = []
    all_regime: list[pd.DataFrame] = []
    thresholds_out: list[pd.DataFrame] = []

    model_alerts = system_alert_frame(source_alerts, "MODEL", "MODEL")
    all_alerts.append(model_alerts)

    regimes, regime_thresholds = derive_regimes(
        combined[combined["split"].eq("CALIBRATION")],
        combined[combined["split"].eq("TEST")],
    )

    systems: list[tuple[str, str, pd.DataFrame]] = [("MODEL", "MODEL", model_alerts)]
    for benchmark_id in benchmark_ids:
        alerts, thresholds = benchmark_alerts(
            benchmark_scores,
            benchmark_id,
            context,
            policy_config,
        )
        alerts = system_alert_frame(alerts, benchmark_id, "CAUSAL_BENCHMARK")
        systems.append((benchmark_id, "CAUSAL_BENCHMARK", alerts))
        thresholds_out.append(thresholds)
        all_alerts.append(alerts)

    for system_id, system_kind, alerts in systems:
        matches, false_alerts, metrics = match_system_alerts(
            alerts,
            context,
            system_id,
            system_kind,
        )
        if not matches.empty:
            all_matches.append(matches)
            timing = timing_decomposition(matches, context, exact_seconds, exact_prices)
            if not timing.empty:
                all_timing.append(timing)
        if not false_alerts.empty:
            all_false.append(false_alerts)
        all_system_metrics.extend(metrics)
        for role in (PRIMARY_ROLE, ROBUSTNESS_ROLE):
            all_daily.append(
                build_daily_components(
                    alerts,
                    matches,
                    false_alerts,
                    context,
                    system_id,
                    system_kind,
                    role,
                )
            )
            all_regime.append(
                regime_metrics(
                    regimes,
                    alerts,
                    matches,
                    context,
                    system_id,
                    system_kind,
                    role,
                )
            )

    model_prelabel = pd.concat(
        [
            extended_prelabel_diagnostic(model_alerts, context, PRIMARY_ROLE),
            extended_prelabel_diagnostic(model_alerts, context, ROBUSTNESS_ROLE),
        ],
        ignore_index=True,
    )
    prelabel_timing = prelabel_timing_decomposition(
        model_prelabel,
        context,
        exact_seconds,
        exact_prices,
    )

    null_distribution = evaluate_null_distribution(
        model_alerts,
        context,
        null_ids,
        null_repetitions,
        random_state,
    )

    alerts_frame = pd.concat(all_alerts, ignore_index=True) if all_alerts else pd.DataFrame()
    alerts_frame.insert(0, "config_id", context.reference.config_id)
    alerts_frame.insert(1, "fold_id", context.fold_id)
    matches_frame = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()
    false_frame = pd.concat(all_false, ignore_index=True) if all_false else pd.DataFrame()
    timing_frame = pd.concat(all_timing, ignore_index=True) if all_timing else pd.DataFrame()
    if not prelabel_timing.empty:
        timing_frame = pd.concat([timing_frame, prelabel_timing], ignore_index=True)
    daily_frame = pd.concat(all_daily, ignore_index=True) if all_daily else pd.DataFrame()
    regime_frame = pd.concat(all_regime, ignore_index=True) if all_regime else pd.DataFrame()

    checks: list[dict[str, Any]] = []
    test_seconds = set(context.test["decision_second"].astype(int))
    checks.append(
        invariant_record(
            "unit",
            "source_alerts_are_test_rows",
            all(int(value) in test_seconds for value in model_alerts["alert_second"]),
            f"alerts={len(model_alerts)} test_rows={len(test_seconds)}",
            context.reference.config_id,
            context.fold_id,
        )
    )
    checks.append(
        invariant_record(
            "unit",
            "source_threshold_history_is_past_only",
            bool(
                model_alerts.empty
                or np.all(
                    pd.to_numeric(model_alerts["threshold_history_end_second"], errors="raise").to_numpy(np.int64)
                    < model_alerts["alert_second"].to_numpy(np.int64)
                )
            ),
            "Every source alert threshold history must end before the alert",
            context.reference.config_id,
            context.fold_id,
        )
    )
    for system_id, _, alerts in systems:
        checks.append(
            invariant_record(
                "unit",
                f"{system_id}_alert_seconds_unique",
                not alerts["alert_second"].duplicated().any(),
                f"alerts={len(alerts)}",
                context.reference.config_id,
                context.fold_id,
            )
        )
        checks.append(
            invariant_record(
                "unit",
                f"{system_id}_cooldown_respected",
                _valid_schedule(alerts["alert_second"].to_numpy(np.int64), context.cooldown_seconds),
                f"cooldown_seconds={context.cooldown_seconds}",
                context.reference.config_id,
                context.fold_id,
            )
        )
        if system_id != "MODEL":
            history_values = pd.to_numeric(
                alerts["threshold_history_end_second"], errors="coerce"
            ).to_numpy(float)
            alert_values = alerts["alert_second"].to_numpy(np.int64)
            past_only = bool(
                not len(alerts)
                or np.all(np.isfinite(history_values) & (history_values < alert_values))
            )
            checks.append(
                invariant_record(
                    "unit",
                    f"{system_id}_threshold_history_is_past_only",
                    past_only,
                    f"alerts={len(alerts)}",
                    context.reference.config_id,
                    context.fold_id,
                )
            )
    checks.append(
        invariant_record(
            "unit",
            "regime_history_precedes_test",
            int(regime_thresholds["history_end_second"]) < int(context.test["decision_second"].min()),
            f"history_end={regime_thresholds['history_end_second']} test_start={int(context.test['decision_second'].min())}",
            context.reference.config_id,
            context.fold_id,
        )
    )
    checks.append(
        invariant_record(
            "unit",
            "null_repetition_count_reconciles",
            len(null_distribution)
            == len(null_ids) * null_repetitions * 2,
            f"rows={len(null_distribution)} expected={len(null_ids) * null_repetitions * 2}",
            context.reference.config_id,
            context.fold_id,
        )
    )
    checks.append(
        invariant_record(
            "unit",
            "no_oracle_gate_applied",
            len(context.test) == len(context.test[context.test["decision_second"].isin(test_seconds)]),
            "All saved TEST prediction rows remain eligible",
            context.reference.config_id,
            context.fold_id,
        )
    )

    return {
        "systems": pd.DataFrame(all_system_metrics),
        "alerts": alerts_frame,
        "matches": matches_frame,
        "false_alerts": false_frame,
        "timing": timing_frame,
        "daily_components": daily_frame,
        "regime_metrics": regime_frame,
        "null_distribution": null_distribution,
        "thresholds": pd.concat(thresholds_out, ignore_index=True)
        if thresholds_out
        else pd.DataFrame(),
        "prelabel_diagnostic": model_prelabel,
        "regime_thresholds": regime_thresholds,
        "invariants": checks,
    }


# ---------------------------------------------------------------------------
# Aggregation: concentration, null comparison and uncertainty
# ---------------------------------------------------------------------------


def aggregate_exact_counts(systems: pd.DataFrame) -> pd.DataFrame:
    if systems.empty:
        return systems
    group = [
        "config_id",
        "target_id",
        "model_id",
        "policy_id",
        "system_id",
        "system_kind",
        "segmentation_role",
    ]
    return systems.groupby(group, as_index=False).agg(
        folds=("fold_id", "nunique"),
        test_days=("test_days", "sum"),
        eligible_test_rows=("eligible_test_rows", "sum"),
        positive_test_rows=("positive_test_rows", "sum"),
        alerts=("alerts", "sum"),
        scorable_events=("scorable_events", "sum"),
        recalled_events=("recalled_events", "sum"),
        false_alerts=("false_alerts", "sum"),
        median_fold_warning_seconds=("median_warning_seconds", "median"),
    ).assign(
        alerts_per_day=lambda frame: frame["alerts"] / frame["test_days"],
        event_recall=lambda frame: frame["recalled_events"] / frame["scorable_events"],
        event_precision=lambda frame: frame["recalled_events"] / frame["alerts"],
        false_alerts_per_day=lambda frame: frame["false_alerts"] / frame["test_days"],
    )


def aggregate_benchmark_scorecard(systems: pd.DataFrame) -> pd.DataFrame:
    counts = aggregate_exact_counts(systems)
    if counts.empty:
        return counts
    model = counts[counts["system_id"].eq("MODEL")][
        [
            "config_id",
            "segmentation_role",
            "event_recall",
            "event_precision",
            "false_alerts_per_day",
            "alerts_per_day",
        ]
    ].rename(
        columns={
            "event_recall": "model_event_recall",
            "event_precision": "model_event_precision",
            "false_alerts_per_day": "model_false_alerts_per_day",
            "alerts_per_day": "model_alerts_per_day",
        }
    )
    output = counts.merge(model, on=["config_id", "segmentation_role"], how="left")
    output["event_recall_difference_vs_model"] = output["event_recall"] - output["model_event_recall"]
    output["event_precision_difference_vs_model"] = output["event_precision"] - output["model_event_precision"]
    output["false_alerts_per_day_difference_vs_model"] = (
        output["false_alerts_per_day"] - output["model_false_alerts_per_day"]
    )
    output["actual_alert_rate_difference_vs_model"] = output["alerts_per_day"] - output["model_alerts_per_day"]
    return output


def summarize_nulls(nulls: pd.DataFrame, systems: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if nulls.empty:
        return pd.DataFrame(), pd.DataFrame()
    # Pool matching repetition IDs across folds before comparing them with the
    # pooled model. Averaging fold-level recall would weight folds equally even
    # when their scorable-event counts differ.
    pooled = nulls.groupby(
        ["config_id", "null_id", "segmentation_role", "repetition"],
        as_index=False,
    ).agg(
        alerts=("alerts", "sum"),
        scorable_events=("scorable_events", "sum"),
        recalled_events=("recalled_events", "sum"),
        false_alerts=("false_alerts", "sum"),
        median_fold_warning_seconds=("median_warning_seconds", "median"),
    )
    pooled["event_recall"] = pooled["recalled_events"] / pooled["scorable_events"]
    pooled["event_precision"] = pooled["recalled_events"] / pooled["alerts"]
    summary = pooled.groupby(
        ["config_id", "null_id", "segmentation_role"], as_index=False
    ).agg(
        repetitions=("repetition", "nunique"),
        mean_event_recall=("event_recall", "mean"),
        median_event_recall=("event_recall", "median"),
        event_recall_p05=("event_recall", lambda x: quantile(x, 0.05)),
        event_recall_p95=("event_recall", lambda x: quantile(x, 0.95)),
        mean_event_precision=("event_precision", "mean"),
        event_precision_p05=("event_precision", lambda x: quantile(x, 0.05)),
        event_precision_p95=("event_precision", lambda x: quantile(x, 0.95)),
        mean_recalled_events=("recalled_events", "mean"),
    )
    model = aggregate_exact_counts(systems)
    model = model[model["system_id"].eq("MODEL")][
        ["config_id", "segmentation_role", "event_recall", "event_precision"]
    ].rename(
        columns={
            "event_recall": "model_event_recall",
            "event_precision": "model_event_precision",
        }
    )
    paired = summary.merge(model, on=["config_id", "segmentation_role"], how="left")
    paired["model_minus_null_mean_recall"] = paired["model_event_recall"] - paired["mean_event_recall"]
    paired["model_minus_null_mean_precision"] = paired["model_event_precision"] - paired["mean_event_precision"]
    p_values: list[float] = []
    precision_p_values: list[float] = []
    for row in paired.itertuples(index=False):
        subset = pooled[
            pooled["config_id"].eq(row.config_id)
            & pooled["null_id"].eq(row.null_id)
            & pooled["segmentation_role"].eq(row.segmentation_role)
        ]
        p_values.append((1 + int((subset["event_recall"] >= row.model_event_recall).sum())) / (1 + len(subset)))
        precision_p_values.append((1 + int((subset["event_precision"] >= row.model_event_precision).sum())) / (1 + len(subset)))
    paired["monte_carlo_p_recall_ge_model"] = p_values
    paired["monte_carlo_p_precision_ge_model"] = precision_p_values
    return summary, paired

def event_concentration_report(
    all_events: Mapping[tuple[str, str], pd.DataFrame],
    matches: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    concentration_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    model_matches = matches[matches["system_id"].eq("MODEL")].copy() if not matches.empty else pd.DataFrame()
    for (config_id, role), events in all_events.items():
        frame = events.copy()
        frame["event_date"] = frame["event_start_second"].map(day_string)
        magnitude_column = (
            "maximum_future_swing_bps"
            if "maximum_future_swing_bps" in frame
            else None
        )
        daily = frame.groupby("event_date", as_index=False).agg(
            events=("event_id", "nunique"),
            event_magnitude=(magnitude_column, "sum")
            if magnitude_column
            else ("event_id", "size"),
        )
        matched = model_matches[
            model_matches["config_id"].eq(config_id)
            & model_matches["segmentation_role"].eq(role)
        ].copy()
        matched["event_date"] = matched["event_start_second"].map(day_string) if not matched.empty else pd.Series(dtype=str)
        tp_daily = matched.groupby("event_date")["event_id"].nunique() if not matched.empty else pd.Series(dtype=int)
        daily["true_positive_events"] = daily["event_date"].map(tp_daily).fillna(0).astype(int)
        ranked = daily.sort_values(["events", "event_magnitude", "event_date"], ascending=[False, False, True])
        for top_n in (1, 3, 5):
            selected = ranked.head(top_n)
            concentration_rows.append(
                {
                    "config_id": config_id,
                    "segmentation_role": role,
                    "top_n_days": top_n,
                    "selected_dates": "|".join(selected["event_date"]),
                    "share_of_events": percent(selected["events"].sum(), daily["events"].sum()),
                    "share_of_true_positive_events": percent(
                        selected["true_positive_events"].sum(),
                        daily["true_positive_events"].sum(),
                    ),
                    "share_of_event_magnitude": percent(
                        selected["event_magnitude"].sum(), daily["event_magnitude"].sum()
                    ),
                }
            )
            excluded = set(selected["event_date"])
            remaining_events = frame[~frame["event_date"].isin(excluded)]
            remaining_matches = matched[~matched["event_date"].isin(excluded)] if not matched.empty else matched
            total_alerts = int(
                model_matches[
                    model_matches["config_id"].eq(config_id)
                    & model_matches["segmentation_role"].eq(role)
                ]["alert_second"].nunique()
            )
            sensitivity_rows.append(
                {
                    "config_id": config_id,
                    "segmentation_role": role,
                    "excluded_top_n_event_days": top_n,
                    "excluded_dates": "|".join(sorted(excluded)),
                    "remaining_scorable_events": int(remaining_events["event_id"].nunique()),
                    "remaining_recalled_events": int(remaining_matches["event_id"].nunique()) if not remaining_matches.empty else 0,
                    "remaining_event_recall": percent(
                        remaining_matches["event_id"].nunique() if not remaining_matches.empty else 0,
                        remaining_events["event_id"].nunique(),
                    ),
                    "original_matched_alert_count": total_alerts,
                    "note": "Alerts on excluded dates are not reassigned; this is an event-day concentration sensitivity diagnostic.",
                }
            )
    return pd.DataFrame(concentration_rows), pd.DataFrame(sensitivity_rows)


def moving_block_indices(length: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if length <= 0:
        return np.empty(0, np.int64)
    indices: list[int] = []
    while len(indices) < length:
        start = int(rng.integers(0, length))
        indices.extend((start + offset) % length for offset in range(block_length))
    return np.asarray(indices[:length], dtype=np.int64)


def pooled_metrics_from_daily(frame: pd.DataFrame) -> dict[str, float]:
    alerts = float(frame["alerts"].sum())
    events = float(frame["scorable_events"].sum())
    recalled = float(frame["recalled_events"].sum())
    false_alerts = float(frame["false_alerts"].sum())
    days = float(len(frame))
    return {
        "event_recall": percent(recalled, events),
        "event_precision": percent(recalled, alerts),
        "false_alerts_per_day": percent(false_alerts, days),
        "alerts_per_day": percent(alerts, days),
    }


def block_bootstrap_uncertainty(
    daily: pd.DataFrame,
    block_lengths: Sequence[int],
    repetitions: int,
    random_state: int,
) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    comparison_systems = sorted(set(daily["system_id"]) - {"MODEL"})
    keys = sorted(set(zip(daily["config_id"], daily["segmentation_role"])))
    metrics = ("event_recall", "event_precision", "false_alerts_per_day", "alerts_per_day")
    for config_id, role in keys:
        subset = daily[
            daily["config_id"].eq(config_id)
            & daily["segmentation_role"].eq(role)
        ]
        for challenger in comparison_systems:
            available_folds = sorted(set(subset["fold_id"]))
            if not available_folds:
                continue
            for block_length in block_lengths:
                draws: MutableMapping[str, list[float]] = defaultdict(list)
                for repetition in range(repetitions):
                    rng = np.random.default_rng(
                        random_state
                        + repetition
                        + 1009 * block_length
                        + 100_003 * ord(str(config_id)[0])
                    )
                    model_parts: list[pd.DataFrame] = []
                    challenger_parts: list[pd.DataFrame] = []
                    valid = True
                    for fold_id in available_folds:
                        model_fold = subset[
                            subset["fold_id"].eq(fold_id)
                            & subset["system_id"].eq("MODEL")
                        ].sort_values("test_date")
                        challenger_fold = subset[
                            subset["fold_id"].eq(fold_id)
                            & subset["system_id"].eq(challenger)
                        ].sort_values("test_date")
                        if len(model_fold) != len(challenger_fold) or not len(model_fold):
                            valid = False
                            break
                        indices = moving_block_indices(len(model_fold), block_length, rng)
                        model_parts.append(model_fold.iloc[indices])
                        challenger_parts.append(challenger_fold.iloc[indices])
                    if not valid:
                        continue
                    model_metrics = pooled_metrics_from_daily(pd.concat(model_parts, ignore_index=True))
                    challenger_metrics = pooled_metrics_from_daily(pd.concat(challenger_parts, ignore_index=True))
                    for metric in metrics:
                        # Positive means model is better. Lower false alerts is better.
                        difference = (
                            challenger_metrics[metric] - model_metrics[metric]
                            if metric == "false_alerts_per_day"
                            else model_metrics[metric] - challenger_metrics[metric]
                        )
                        draws[metric].append(difference)
                for metric, values in draws.items():
                    records.append(
                        {
                            "config_id": config_id,
                            "segmentation_role": role,
                            "challenger_system_id": challenger,
                            "block_length_days": block_length,
                            "bootstrap_repetitions": len(values),
                            "metric": metric,
                            "difference_definition": "positive_favors_model",
                            "mean_difference": safe_mean(values),
                            "median_difference": safe_median(values),
                            "ci_lower_2p5": quantile(values, 0.025),
                            "ci_upper_97p5": quantile(values, 0.975),
                            "probability_difference_positive": percent(
                                sum(value > 0 for value in values), len(values)
                            ),
                            "conditions_on_fixed_alerts": True,
                            "adaptive_policy_state_rerun": False,
                        }
                    )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Cross-target redundancy and fold-selection sensitivity
# ---------------------------------------------------------------------------


def aligned_score_correlations(
    predictions: Mapping[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    target_order = ("H20M_T075BPS", "H15M_T075BPS", "H05M_T050BPS")
    folds = sorted({fold for _, fold in predictions})
    for fold_id in folds:
        available = [target for target in target_order if (target, fold_id) in predictions]
        for left_index, left in enumerate(available):
            for right in available[left_index + 1 :]:
                left_frame = predictions[(left, fold_id)][
                    ["decision_second", "probability"]
                ].rename(columns={"probability": "left_score"})
                right_frame = predictions[(right, fold_id)][
                    ["decision_second", "probability"]
                ].rename(columns={"probability": "right_score"})
                aligned = left_frame.merge(right_frame, on="decision_second", how="inner")
                records.append(
                    {
                        "fold_id": fold_id,
                        "left_target_id": left,
                        "right_target_id": right,
                        "aligned_rows": len(aligned),
                        "pearson_correlation": aligned["left_score"].corr(aligned["right_score"], method="pearson"),
                        "spearman_correlation": aligned["left_score"].corr(aligned["right_score"], method="spearman"),
                    }
                )
    return pd.DataFrame(records)


def alert_overlap_report(alerts: pd.DataFrame) -> pd.DataFrame:
    model = alerts[alerts["system_id"].eq("MODEL")].copy()
    config_to_target = {item.config_id: item.target_id for item in FROZEN_CONFIGURATIONS}
    records: list[dict[str, Any]] = []
    folds = sorted(set(model["fold_id"]))
    configs = ("A", "C", "D")
    for fold_id in folds:
        for left_index, left in enumerate(configs):
            for right in configs[left_index + 1 :]:
                left_seconds = np.sort(
                    model[model["config_id"].eq(left) & model["fold_id"].eq(fold_id)]["alert_second"].to_numpy(np.int64)
                )
                right_seconds = np.sort(
                    model[model["config_id"].eq(right) & model["fold_id"].eq(fold_id)]["alert_second"].to_numpy(np.int64)
                )
                exact = len(set(map(int, left_seconds)).intersection(map(int, right_seconds)))
                near_left = sum(np.any(np.abs(right_seconds - value) <= 60) for value in left_seconds) if len(right_seconds) else 0
                near_right = sum(np.any(np.abs(left_seconds - value) <= 60) for value in right_seconds) if len(left_seconds) else 0
                records.append(
                    {
                        "fold_id": fold_id,
                        "left_config_id": left,
                        "left_target_id": config_to_target[left],
                        "right_config_id": right,
                        "right_target_id": config_to_target[right],
                        "left_alerts": len(left_seconds),
                        "right_alerts": len(right_seconds),
                        "exact_overlap_alerts": exact,
                        "left_share_within_60s_of_right": percent(near_left, len(left_seconds)),
                        "right_share_within_60s_of_left": percent(near_right, len(right_seconds)),
                    }
                )
    return pd.DataFrame(records)


def incremental_recall_report(
    alerts: pd.DataFrame,
    h20_events_by_fold: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    model = alerts[alerts["system_id"].eq("MODEL")].copy()
    records: list[dict[str, Any]] = []
    config_names = {"A": "H20", "C": "H15", "D": "H05"}
    for fold_id, events in h20_events_by_fold.items():
        recalled_sets: dict[str, set[str]] = {}
        alert_sets: dict[str, np.ndarray] = {}
        for config_id, label in config_names.items():
            seconds = model[
                model["config_id"].eq(config_id) & model["fold_id"].eq(fold_id)
            ]["alert_second"].to_numpy(np.int64)
            alert_sets[label] = seconds
            matches, _ = BASE.match_alerts_to_events(seconds, events)
            recalled_sets[label] = set(matches.get("event_id", pd.Series(dtype=str)).astype(str))
        combinations = {
            "H20": ("H20",),
            "H15": ("H15",),
            "H05": ("H05",),
            "H20_OR_H15": ("H20", "H15"),
            "H20_OR_H05": ("H20", "H05"),
            "ALL_THREE": ("H20", "H15", "H05"),
        }
        base = recalled_sets["H20"]
        for name, members in combinations.items():
            union_alerts = np.unique(np.concatenate([alert_sets[member] for member in members]))
            matches, _ = BASE.match_alerts_to_events(union_alerts, events)
            recalled = set(matches.get("event_id", pd.Series(dtype=str)).astype(str))
            records.append(
                {
                    "fold_id": fold_id,
                    "combination": name,
                    "member_systems": "|".join(members),
                    "alerts": len(union_alerts),
                    "h20_scorable_events": len(events),
                    "h20_recalled_events": len(recalled),
                    "h20_event_recall": percent(len(recalled), len(events)),
                    "incremental_events_beyond_h20_alone": len(recalled - base),
                }
            )
    return pd.DataFrame(records)


def fold_selection_sensitivity(systems: pd.DataFrame) -> pd.DataFrame:
    model = systems[
        systems["system_id"].eq("MODEL")
        & systems["segmentation_role"].eq(PRIMARY_ROLE)
    ].copy()
    if model.empty:
        return pd.DataFrame()
    metrics = {
        "event_recall": False,
        "event_precision": False,
        "false_alerts_per_day": True,
        "alerts_per_day_absolute_error": True,
    }
    model["alerts_per_day_absolute_error"] = (
        model["alerts_per_day"] - DEFAULT_ALERT_RATE
    ).abs()
    records: list[dict[str, Any]] = []
    folds = sorted(set(model["fold_id"]))
    for metric, lower_is_better in metrics.items():
        for held_out in folds:
            development = model[~model["fold_id"].eq(held_out)].groupby(
                "config_id", as_index=False
            )[metric].mean()
            development = development.sort_values(
                [metric, "config_id"],
                ascending=[lower_is_better, True],
                kind="mergesort",
            )
            selected = str(development.iloc[0]["config_id"])
            held = model[
                model["fold_id"].eq(held_out) & model["config_id"].eq(selected)
            ]
            records.append(
                {
                    "metric": metric,
                    "lower_is_better": lower_is_better,
                    "held_out_fold_id": held_out,
                    "selected_config_id": selected,
                    "development_metric": float(development.iloc[0][metric]),
                    "held_out_metric": float(held.iloc[0][metric]) if len(held) else np.nan,
                    "development_folds": "|".join(fold for fold in folds if fold != held_out),
                }
            )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Findings and self-test
# ---------------------------------------------------------------------------


def findings_markdown(
    summary: Mapping[str, Any],
    scorecard: pd.DataFrame,
    null_comparison: pd.DataFrame,
    timing: pd.DataFrame,
    concentration: pd.DataFrame,
) -> str:
    lines = [
        "# Expansion Signal Forensic Audit",
        "",
        f"Run status: **{summary.get('status', 'UNKNOWN')}**",
        "",
        "This audit reused saved out-of-sample predictions and alerts. No model was trained or refit, and no future-informed eligibility gate was applied.",
        "",
        "## Primary interpretation",
        "",
        "The existing folds are research-development data. Results here diagnose signal source, timing, concentration and benchmark difficulty; they are not a final untouched-holdout claim.",
        "",
    ]
    primary = scorecard[scorecard["segmentation_role"].eq(PRIMARY_ROLE)] if not scorecard.empty else pd.DataFrame()
    model = primary[primary["system_id"].eq("MODEL")] if not primary.empty else pd.DataFrame()
    if not model.empty:
        lines.extend(["## Frozen model references", ""])
        for row in model.sort_values("config_id").itertuples(index=False):
            lines.append(
                f"* Configuration {row.config_id}: {row.alerts_per_day:.2f} alerts/day, "
                f"{row.event_recall:.1%} event recall, {row.event_precision:.1%} event precision, "
                f"{row.false_alerts_per_day:.2f} false alerts/day."
            )
        lines.append("")
    if not null_comparison.empty:
        lines.extend(["## Matched timing nulls", ""])
        for row in null_comparison[
            null_comparison["segmentation_role"].eq(PRIMARY_ROLE)
        ].sort_values(["config_id", "null_id"]).itertuples(index=False):
            lines.append(
                f"* {row.config_id} versus {row.null_id}: model-minus-null mean recall "
                f"{row.model_minus_null_mean_recall:+.3f}; Monte Carlo tail probability "
                f"{row.monte_carlo_p_recall_ge_model:.3f}."
            )
        lines.append("")
    if not timing.empty:
        model_timing = timing[
            timing["system_id"].eq("MODEL")
            & timing["segmentation_role"].eq(PRIMARY_ROLE)
            & timing["match_contract"].eq("OFFICIAL_ONE_TO_ONE")
        ]
        if not model_timing.empty:
            lines.extend(["## Timing diagnostic", ""])
            for config_id, group in model_timing.groupby("config_id"):
                lines.append(
                    f"* {config_id}: median fraction of target range already realised by an official true-positive alert was "
                    f"{safe_median(group['fraction_of_target_range_realized']):.1%}."
                )
            lines.append("")
    if not concentration.empty:
        lines.extend(["## Event concentration", ""])
        top_one = concentration[
            concentration["top_n_days"].eq(1)
            & concentration["segmentation_role"].eq(PRIMARY_ROLE)
        ]
        for row in top_one.sort_values("config_id").itertuples(index=False):
            lines.append(
                f"* {row.config_id}: the single most event-heavy day contained {row.share_of_events:.1%} of events and {row.share_of_true_positive_events:.1%} of true positives."
            )
        lines.append("")
    lines.extend(
        [
            "## Decision discipline",
            "",
            "Proceed to fitted volatility benchmarks only when the model survives matched timing nulls and extreme-day removal. Redesign the target when simple recent-range rules match it or official true positives arrive after most of the target range has already occurred. Treat wide dependence-aware intervals as inconclusive and collect more chronological data rather than declaring failure.",
            "",
        ]
    )
    return "\n".join(lines)


def self_test() -> None:
    seconds = np.arange(0, 3600, 30, dtype=np.int64) + 1_800_000_000
    exact_seconds = np.arange(seconds[0] - 300, seconds[-1] + 1, dtype=np.int64)
    exact_prices = 100.0 + np.sin(np.arange(len(exact_seconds)) / 17.0)
    prices = align_exact_prices_to_decisions(seconds, exact_seconds, exact_prices)
    ranges = causal_trailing_range_bps(seconds, prices, 300)
    assert np.isfinite(ranges[10:]).all()
    assert ranges[10] >= 0
    assert np.allclose(prices, exact_prices[np.searchsorted(exact_seconds, seconds)])

    benchmark_combined = pd.DataFrame(
        {
            "decision_second": seconds,
            "split": ["CALIBRATION"] * 60 + ["TEST"] * (len(seconds) - 60),
            "target": (np.arange(len(seconds)) % 7 == 0).astype(np.int8),
            "realized_vol_300s_bps": np.linspace(1.0, 10.0, len(seconds)),
            "realized_vol_900s_bps": np.linspace(2.0, 12.0, len(seconds)),
            "realized_vol_3600s_bps": np.linspace(3.0, 14.0, len(seconds)),
            "return_60s_bps": np.linspace(-3.0, 3.0, len(seconds)),
            "return_300s_bps": np.linspace(-8.0, 8.0, len(seconds)),
        }
    )
    benchmark_calibration = benchmark_combined[
        benchmark_combined["split"].eq("CALIBRATION")
    ].copy()
    benchmark_scores = build_benchmark_scores(
        benchmark_calibration,
        benchmark_combined,
        exact_seconds,
        exact_prices,
    )
    assert set(DEFAULT_BENCHMARKS).issubset(benchmark_scores.columns)
    for benchmark_id in DEFAULT_BENCHMARKS:
        assert np.isfinite(
            pd.to_numeric(benchmark_scores[benchmark_id], errors="coerce").to_numpy(float)
        ).any()

    rng = np.random.default_rng(123)
    selected = sample_uniform_schedule(seconds, 4, 300, rng)
    assert len(selected) == 4
    assert _valid_schedule(selected, 300)

    source = np.asarray([seconds[5], seconds[30], seconds[60]], dtype=np.int64)
    shifted = circular_shift_schedule(source, seconds, 300, np.random.default_rng(456))
    assert len(shifted) == len(source)
    assert _valid_schedule(shifted, 300)

    alerts_by_day = {day_string(seconds[0]): source}
    eligible_by_day = {day_string(seconds[0]): seconds}
    permuted = permute_day_schedules(alerts_by_day, eligible_by_day, 300, np.random.default_rng(789))
    assert len(permuted) == len(source)

    calibration = pd.DataFrame(
        {
            "decision_second": seconds[:60],
            "realized_vol_300s_bps": np.linspace(1, 10, 60),
            "realized_vol_3600s_bps": np.linspace(2, 12, 60),
        }
    )
    test = pd.DataFrame(
        {
            "decision_second": seconds[60:],
            "realized_vol_300s_bps": np.linspace(2, 15, len(seconds) - 60),
            "realized_vol_3600s_bps": np.linspace(3, 14, len(seconds) - 60),
        }
    )
    regimes, thresholds = derive_regimes(calibration, test)
    assert len(regimes) == len(test)
    assert thresholds["history_end_second"] < int(test["decision_second"].min())

    daily = pd.DataFrame(
        {
            "alerts": [1, 2, 1, 3],
            "scorable_events": [2, 2, 1, 3],
            "recalled_events": [1, 1, 0, 2],
            "false_alerts": [0, 1, 1, 1],
        }
    )
    metrics = pooled_metrics_from_daily(daily)
    assert math.isclose(metrics["event_recall"], 4 / 8)
    assert math.isclose(metrics["event_precision"], 4 / 7)

    long_root = Path("/tmp") / ("x" * 120) / ("y" * 120)
    path = long_root / "test.json"
    atomic_write_json(path, {"ok": True})
    assert load_json(path)["ok"] is True
    path.unlink()
    for parent in (long_root, long_root.parent):
        try:
            parent.rmdir()
        except OSError:
            pass

    print("SELF-TEST PASS")
    print(
        "Validated exact-price alignment, causal trailing ranges, cooldown-preserving matched nulls, "
        "past-only volatility regimes, daily count aggregation and short-path atomic writers."
    )


# ---------------------------------------------------------------------------
# CLI and main orchestration
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-root",
        default="data/derived/perps/research/expansion_baseline_models_v1",
    )
    parser.add_argument(
        "--policy-root",
        default="data/derived/perps/research/expansion_alert_policy_evaluation_v1",
    )
    parser.add_argument(
        "--output-root",
        default="data/derived/perps/research/expansion_signal_forensics_v1",
    )
    parser.add_argument(
        "--report-root",
        default="reports/perps_expansion_signal_forensics",
    )
    parser.add_argument(
        "--configurations",
        default="A,B,C,D",
        help="Comma-separated frozen IDs A,B,C,D.",
    )
    parser.add_argument(
        "--folds",
        default="ALL",
        help="Comma-separated fold IDs such as 01,02, or ALL.",
    )
    parser.add_argument(
        "--benchmarks",
        default=",".join(DEFAULT_BENCHMARKS),
    )
    parser.add_argument(
        "--nulls",
        default=",".join(DEFAULT_NULLS),
    )
    parser.add_argument("--null-repetitions", type=int, default=DEFAULT_NULL_REPETITIONS)
    parser.add_argument(
        "--bootstrap-repetitions", type=int, default=DEFAULT_BOOTSTRAP_REPETITIONS
    )
    parser.add_argument(
        "--bootstrap-block-lengths",
        default=",".join(map(str, DEFAULT_BLOCK_LENGTHS)),
    )
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--compression-level", type=int, default=9)
    parser.add_argument("--hash-input-files", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def _resolve_folds(baseline_root: Path, reference: ReferenceConfiguration, requested: str) -> tuple[str, ...]:
    root = baseline_root / "experiments" / f"target={reference.target_id}" / f"model={reference.model_id}"
    available = tuple(
        sorted(
            path.name.split("=", 1)[1].zfill(2)
            for path in root.glob("fold=*")
            if path.is_dir()
        )
    )
    if not available:
        raise RuntimeError(f"No folds found under {root}")
    if requested.upper() == "ALL":
        return available
    selected = tuple(value.zfill(2) for value in parse_csv_strings(requested))
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise RuntimeError(f"Unknown folds for {reference.config_id}: {unknown}")
    return selected


def _load_exact_prices(
    feature_root: Path,
    symbol: str,
    minimum_second: int,
    maximum_second: int,
    hash_input_files: bool,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    partitions = CORE.discover_date_partitions(feature_root, symbol, "features", hash_input_files)
    seconds, prices = BASE.load_exact_price_series(
        partitions,
        minimum_second,
        maximum_second,
        symbol,
    )
    start_day = datetime.fromtimestamp(minimum_second, tz=timezone.utc).date()
    end_day = datetime.fromtimestamp(maximum_second, tz=timezone.utc).date()
    signatures = [
        {
            "date": day.isoformat(),
            "signature": partition.signature,
        }
        for day, partition in sorted(partitions.items())
        if start_day <= day <= end_day
    ]
    return seconds, prices, signatures


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.null_repetitions <= 0 or args.bootstrap_repetitions <= 0:
        raise SystemExit("Null and bootstrap repetitions must be positive")
    if not 1 <= args.compression_level <= 22:
        raise SystemExit("--compression-level must be between 1 and 22")

    requested_configs = parse_csv_strings(args.configurations.upper())
    available_refs = {item.config_id: item for item in FROZEN_CONFIGURATIONS}
    unknown_configs = sorted(set(requested_configs) - set(available_refs))
    if unknown_configs:
        raise SystemExit(f"Unknown frozen configurations: {unknown_configs}")
    references = tuple(available_refs[item] for item in requested_configs)
    benchmark_ids = tuple(item.upper() for item in parse_csv_strings(args.benchmarks))
    unknown_benchmarks = sorted(set(benchmark_ids) - set(DEFAULT_BENCHMARKS))
    if unknown_benchmarks:
        raise SystemExit(f"Unknown benchmarks: {unknown_benchmarks}")
    null_ids = tuple(item.upper() for item in parse_csv_strings(args.nulls))
    unknown_nulls = sorted(set(null_ids) - set(DEFAULT_NULLS))
    if unknown_nulls:
        raise SystemExit(f"Unknown null systems: {unknown_nulls}")
    block_lengths = parse_positive_ints(args.bootstrap_block_lengths, "block lengths")

    baseline_root = Path(args.baseline_root).resolve()
    policy_root = Path(args.policy_root).resolve()
    output_root = Path(args.output_root).resolve()
    report_root = Path(args.report_root).resolve()
    baseline_config_path = baseline_root / "_metadata" / "dataset_config.json"
    baseline_summary_path = baseline_root / "_metadata" / "latest_run_summary.json"
    policy_config_path = policy_root / "_metadata" / "dataset_config.json"
    policy_summary_path = policy_root / "_metadata" / "latest_run_summary.json"
    baseline_config = load_json(baseline_config_path)
    baseline_summary = load_json(baseline_summary_path)
    policy_config = load_json(policy_config_path)
    policy_summary = load_json(policy_summary_path)

    if str(baseline_config.get("builder_version")) != EXPECTED_BASELINE_VERSION:
        raise SystemExit(
            f"Baseline builder must be {EXPECTED_BASELINE_VERSION}; found {baseline_config.get('builder_version')!r}"
        )
    if str(policy_config.get("evaluator_version")) != EXPECTED_POLICY_VERSION:
        raise SystemExit(
            f"Policy evaluator must be {EXPECTED_POLICY_VERSION}; found {policy_config.get('evaluator_version')!r}"
        )
    if baseline_summary.get("status") != "COMPLETED" or policy_summary.get("status") != "COMPLETED":
        raise SystemExit("Baseline and policy datasets must both have COMPLETED status")

    targets = {
        str(item["target_id"]): target_from_config(item)
        for item in baseline_config.get("target_plan", [])
    }
    for reference in references:
        if reference.target_id not in targets:
            raise SystemExit(f"Target {reference.target_id} is absent from baseline config")

    symbol = str(args.symbol or baseline_config.get("symbol", "BTCUSDT")).upper()
    feature_root = Path(str(baseline_config["feature_root"])).resolve()
    if not feature_root.exists():
        raise SystemExit(f"Feature root does not exist: {feature_root}")

    run_id = utc_now().strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_config = {
        "auditor_version": AUDITOR_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "baseline_root": str(baseline_root),
        "policy_root": str(policy_root),
        "feature_root": str(feature_root),
        "output_root": str(output_root),
        "report_root": str(report_root),
        "symbol": symbol,
        "configurations": [asdict(item) for item in references],
        "folds": args.folds,
        "benchmarks": benchmark_ids,
        "nulls": null_ids,
        "null_repetitions": args.null_repetitions,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_block_lengths": block_lengths,
        "random_state": args.random_state,
        "hash_input_files": args.hash_input_files,
        "model_retraining": False,
        "oracle_eligibility_gate": False,
        "benchmark_price_source": "flow_features_1s_v1.price_for_features_at_decision_second",
    }

    print("Auditing saved expansion signal against causal benchmarks and nulls...")
    print(f"Auditor version      : {AUDITOR_VERSION}")
    print(f"Run ID               : {run_id}")
    print(f"Baseline root        : {baseline_root}")
    print(f"Policy root          : {policy_root}")
    print(f"Output root          : {output_root}")
    print(f"Report root          : {report_root}")
    print(f"Frozen configurations: {len(references)}")
    print(f"Causal benchmarks    : {len(benchmark_ids)}")
    print(f"Timing nulls         : {len(null_ids)} x {args.null_repetitions} repetitions")
    print("No model training and no future-informed eligibility gate will be used.")

    connection = connect_state(output_root / "_build_state" / "build_state.sqlite")
    connection.execute(
        "INSERT INTO runs(run_id, started_at_utc, status, configuration_json) VALUES(?,?,?,?)",
        (run_id, iso_utc(utc_now()), "RUNNING", compact_json(run_config)),
    )
    connection.commit()

    all_unit_paths: list[UnitPaths] = []
    all_invariants: list[dict[str, Any]] = []
    prediction_cache: dict[tuple[str, str], pd.DataFrame] = {}
    events_cache_parts: dict[tuple[str, str], list[pd.DataFrame]] = defaultdict(list)
    h20_events_by_fold: dict[str, pd.DataFrame] = {}
    completed = skipped = failed = 0

    try:
        # Determine exact-price range required by all selected units.
        all_test_seconds: list[int] = []
        unit_descriptors: list[tuple[ReferenceConfiguration, Any, str, Path, Path, Path, Path]] = []
        for reference in references:
            target = targets[reference.target_id]
            for fold_id in _resolve_folds(baseline_root, reference, args.folds):
                pred_path = prediction_path(baseline_root, reference, fold_id)
                ctx = policy_context(reference, target, fold_id)
                pred, _ = POLICY.load_predictions(pred_path, ctx)
                prediction_cache[(reference.target_id, fold_id)] = pred[pred["split"].eq("TEST")].copy()
                all_test_seconds.extend(pred["decision_second"].astype(int).tolist())
                alert_path = source_alert_path(policy_root, reference, target, fold_id)
                primary_path = event_path(baseline_root, target, PRIMARY_ROLE, symbol)
                robustness_path = event_path(baseline_root, target, ROBUSTNESS_ROLE, symbol)
                unit_descriptors.append(
                    (reference, target, fold_id, pred_path, alert_path, primary_path, robustness_path)
                )

        minimum_second = min(all_test_seconds) - max(targets[reference.target_id].horizon_seconds for reference in references)
        maximum_second = max(all_test_seconds) + 1
        exact_seconds, exact_prices, feature_signatures = _load_exact_prices(
            feature_root,
            symbol,
            minimum_second,
            maximum_second,
            args.hash_input_files,
        )

        for reference, target, fold_id, pred_path, alert_path, primary_path, robustness_path in unit_descriptors:
            paths = unit_paths(output_root, reference.config_id, fold_id)
            all_unit_paths.append(paths)
            ctx = policy_context(reference, target, fold_id)
            predictions, _ = POLICY.load_predictions(pred_path, ctx)
            calibration = predictions[predictions["split"].eq("CALIBRATION")].copy()
            test = predictions[predictions["split"].eq("TEST")].copy()
            snapshots, snapshot_paths = load_aligned_snapshots(
                baseline_root,
                symbol,
                predictions["decision_second"].to_numpy(np.int64),
            )
            snapshots = snapshots.merge(
                predictions[["decision_second", "split", "target"]],
                on="decision_second",
                how="left",
                validate="one_to_one",
            )
            primary_events = load_full_events(primary_path, reference.target_id, PRIMARY_ROLE)
            robustness_events = load_full_events(robustness_path, reference.target_id, ROBUSTNESS_ROLE)
            source_alerts = load_source_alerts(alert_path, reference, fold_id)
            test_seconds_for_events = test["decision_second"].to_numpy(np.int64)
            for role, event_frame in (
                (PRIMARY_ROLE, primary_events),
                (ROBUSTNESS_ROLE, robustness_events),
            ):
                scorable = BASE.scorable_events_for_test(
                    event_frame,
                    test_seconds_for_events,
                    int(test_seconds_for_events[0]),
                    int(test_seconds_for_events[-1]) + 1,
                ).copy()
                scorable["fold_id"] = fold_id
                events_cache_parts[(reference.config_id, role)].append(scorable)

            signatures = {
                "auditor_version": AUDITOR_VERSION,
                "reference": asdict(reference),
                "fold_id": fold_id,
                "prediction": input_signature(pred_path, args.hash_input_files),
                "source_alerts": input_signature(alert_path, args.hash_input_files),
                "primary_events": input_signature(primary_path, args.hash_input_files),
                "robustness_events": input_signature(robustness_path, args.hash_input_files),
                "snapshots": [input_signature(path, args.hash_input_files) for path in snapshot_paths],
                "feature_partition_signatures": feature_signatures,
                "benchmarks": benchmark_ids,
                "nulls": null_ids,
                "null_repetitions": args.null_repetitions,
                "random_state": args.random_state,
                "policy_config": policy_config,
            }
            signature = sha256_text(compact_json(signatures))
            if unit_is_reusable(
                connection,
                reference.config_id,
                fold_id,
                signature,
                paths,
                args.force,
            ):
                skipped += 1
                continue

            connection.execute(
                """
                INSERT INTO units(config_id, fold_id, input_signature, status, result_path, last_run_id)
                VALUES(?,?,?,?,?,?)
                ON CONFLICT(config_id, fold_id) DO UPDATE SET
                    input_signature=excluded.input_signature,
                    status='RUNNING',
                    result_path=excluded.result_path,
                    last_run_id=excluded.last_run_id,
                    error_text=NULL
                """,
                (
                    reference.config_id,
                    fold_id,
                    signature,
                    "RUNNING",
                    str(paths.result_json),
                    run_id,
                ),
            )
            connection.commit()
            try:
                unit_context = UnitContext(
                    reference=reference,
                    target=target,
                    fold_id=fold_id,
                    cooldown_seconds=int(ctx.cooldown_seconds),
                    calibration=calibration,
                    test=test,
                    snapshots=snapshots,
                    events_by_role={
                        PRIMARY_ROLE: primary_events,
                        ROBUSTNESS_ROLE: robustness_events,
                    },
                )
                result = execute_unit(
                    unit_context,
                    source_alerts,
                    policy_config,
                    benchmark_ids,
                    null_ids,
                    args.null_repetitions,
                    exact_seconds,
                    exact_prices,
                    args.random_state,
                )
                metadata = {
                    "perps_expansion_signal_forensics_auditor_version": AUDITOR_VERSION,
                    "perps_expansion_signal_forensics_schema_version": OUTPUT_SCHEMA_VERSION,
                    "config_id": reference.config_id,
                    "fold_id": fold_id,
                    "input_signature": signature,
                }
                atomic_write_parquet(paths.systems, result["systems"], metadata, args.compression_level)
                atomic_write_parquet(paths.alerts, result["alerts"], metadata, args.compression_level)
                atomic_write_parquet(paths.matches, result["matches"], metadata, args.compression_level)
                atomic_write_parquet(paths.timing, result["timing"], metadata, args.compression_level)
                atomic_write_parquet(paths.daily_components, result["daily_components"], metadata, args.compression_level)
                atomic_write_parquet(paths.regime_metrics, result["regime_metrics"], metadata, args.compression_level)
                atomic_write_parquet(paths.null_distribution, result["null_distribution"], metadata, args.compression_level)
                unit_summary = {
                    "auditor_version": AUDITOR_VERSION,
                    "config_id": reference.config_id,
                    "fold_id": fold_id,
                    "status": "COMPLETED",
                    "input_signature": signature,
                    "system_rows": len(result["systems"]),
                    "alert_rows": len(result["alerts"]),
                    "match_rows": len(result["matches"]),
                    "timing_rows": len(result["timing"]),
                    "null_rows": len(result["null_distribution"]),
                    "regime_thresholds": result["regime_thresholds"],
                    "invariants": result["invariants"],
                }
                atomic_write_json(paths.result_json, unit_summary)
                all_invariants.extend(result["invariants"])
                if any(item["status"] == "FAIL" for item in result["invariants"]):
                    raise RuntimeError(f"Unit {reference.config_id}/{fold_id} failed invariants")
                connection.execute(
                    """
                    UPDATE units SET status='COMPLETED', completed_at_utc=?, error_text=NULL,
                        last_run_id=? WHERE config_id=? AND fold_id=?
                    """,
                    (iso_utc(utc_now()), run_id, reference.config_id, fold_id),
                )
                connection.commit()
                completed += 1
                print(f"Completed config {reference.config_id}, fold {fold_id}")
            except Exception:
                failed += 1
                connection.execute(
                    """
                    UPDATE units SET status='FAILED', error_text=?, last_run_id=?
                    WHERE config_id=? AND fold_id=?
                    """,
                    (traceback.format_exc(), run_id, reference.config_id, fold_id),
                )
                connection.commit()
                raise

        # Aggregate completed/reused units.
        systems = pd.concat([read_parquet(path.systems) for path in all_unit_paths], ignore_index=True)
        alerts = pd.concat([read_parquet(path.alerts) for path in all_unit_paths], ignore_index=True)
        matches_frames = [read_parquet(path.matches) for path in all_unit_paths]
        matches = pd.concat(matches_frames, ignore_index=True) if matches_frames else pd.DataFrame()
        timing_frames = [read_parquet(path.timing) for path in all_unit_paths]
        timing = pd.concat(timing_frames, ignore_index=True) if timing_frames else pd.DataFrame()
        daily = pd.concat([read_parquet(path.daily_components) for path in all_unit_paths], ignore_index=True)
        regimes = pd.concat([read_parquet(path.regime_metrics) for path in all_unit_paths], ignore_index=True)
        nulls = pd.concat([read_parquet(path.null_distribution) for path in all_unit_paths], ignore_index=True)

        for path in all_unit_paths:
            result = load_json(path.result_json)
            all_invariants.extend(result.get("invariants", []))

        exact_counts = aggregate_exact_counts(systems)
        benchmark_scorecard = aggregate_benchmark_scorecard(systems)
        null_summary, null_comparison = summarize_nulls(nulls, systems)
        events_cache = {
            key: pd.concat(parts, ignore_index=True).drop_duplicates(
                subset=["fold_id", "event_id"], keep="first"
            )
            for key, parts in events_cache_parts.items()
        }
        concentration, extreme_sensitivity = event_concentration_report(events_cache, matches)
        uncertainty = block_bootstrap_uncertainty(
            daily,
            block_lengths,
            args.bootstrap_repetitions,
            args.random_state,
        )
        correlations = aligned_score_correlations(prediction_cache)
        overlap = alert_overlap_report(alerts)

        # H20/H15/H05 layered diagnostic requires the three frozen target
        # references. Custom filtered runs that omit one of them simply leave
        # this exploratory report empty.
        if {"A", "C", "D"}.issubset(set(requested_configs)):
            h20_reference = available_refs["A"]
            h20_target = targets[h20_reference.target_id]
            for fold_id in _resolve_folds(baseline_root, h20_reference, args.folds):
                pred = prediction_cache[(h20_reference.target_id, fold_id)]
                events = load_full_events(
                    event_path(baseline_root, h20_target, PRIMARY_ROLE, symbol),
                    h20_reference.target_id,
                    PRIMARY_ROLE,
                )
                seconds = pred["decision_second"].to_numpy(np.int64)
                h20_events_by_fold[fold_id] = BASE.scorable_events_for_test(
                    events,
                    seconds,
                    int(seconds[0]),
                    int(seconds[-1]) + 1,
                )
            incremental = incremental_recall_report(alerts, h20_events_by_fold)
        else:
            incremental = pd.DataFrame()
        selection = fold_selection_sensitivity(systems)

        # Timing report summaries and raw range-at-alert records are separated.
        timing_summary = (
            timing.groupby(
                [
                    "config_id",
                    "target_id",
                    "system_id",
                    "system_kind",
                    "segmentation_role",
                    "match_contract",
                    "timing_bucket",
                ],
                as_index=False,
            ).agg(
                alerts=("event_id", "size"),
                median_seconds_after_label_start=("seconds_after_label_start", "median"),
                median_seconds_to_realization=("seconds_to_realization", "median"),
                median_fraction_target_range_realized=("fraction_of_target_range_realized", "median"),
            )
            if not timing.empty
            else pd.DataFrame()
        )

        # Final global invariants.
        all_invariants.extend(
            [
                invariant_record(
                    "run",
                    "all_units_available",
                    all(path.result_json.exists() for path in all_unit_paths),
                    f"units={len(all_unit_paths)}",
                ),
                invariant_record(
                    "run",
                    "no_post_realization_official_matches",
                    bool(
                        matches.empty
                        or np.all(
                            matches["alert_second"].to_numpy(np.int64)
                            <= matches["first_realization_second"].to_numpy(np.int64)
                        )
                    ),
                    f"matches={len(matches)}",
                ),
                invariant_record(
                    "run",
                    "all_reference_configs_frozen",
                    set(alerts["config_id"].astype(str)).issubset(set(requested_configs)),
                    f"configs={sorted(set(alerts['config_id'].astype(str)))}",
                ),
            ]
        )
        invariant_frame = pd.DataFrame(all_invariants).drop_duplicates()
        failed_invariants = int(invariant_frame["status"].eq("FAIL").sum())

        summary = {
            "auditor_version": AUDITOR_VERSION,
            "run_id": run_id,
            "status": "COMPLETED" if failed_invariants == 0 else "FAILED",
            "completed_units": completed,
            "reused_units": skipped,
            "failed_units": failed,
            "total_units": len(all_unit_paths),
            "reference_configurations": len(references),
            "causal_benchmarks": len(benchmark_ids),
            "timing_nulls": len(null_ids),
            "null_repetitions_per_fold": args.null_repetitions,
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "model_retraining": False,
            "oracle_eligibility_gate": False,
            "invariant_checks": len(invariant_frame),
            "failed_invariants": failed_invariants,
            "research_status": "EXPLORATORY_DEVELOPMENT_FOLDS",
            "completed_at_utc": iso_utc(utc_now()),
        }

        report_root.mkdir(parents=True, exist_ok=True)
        reports: Mapping[str, pd.DataFrame] = {
            "expansion_signal_exact_counts.csv": exact_counts,
            "expansion_signal_timing_decomposition.csv": timing_summary,
            "expansion_signal_range_at_alert.csv": timing,
            "expansion_signal_event_concentration.csv": concentration,
            "expansion_signal_extreme_day_sensitivity.csv": extreme_sensitivity,
            "expansion_signal_volatility_regimes.csv": regimes,
            "expansion_signal_score_correlations.csv": correlations,
            "expansion_signal_alert_overlap.csv": overlap,
            "expansion_signal_incremental_recall.csv": incremental,
            "expansion_signal_benchmark_scorecard.csv": benchmark_scorecard,
            "expansion_signal_random_null_distribution.csv": null_summary,
            "expansion_signal_paired_differences.csv": null_comparison,
            "expansion_signal_block_uncertainty.csv": uncertainty,
            "expansion_signal_fold_selection_sensitivity.csv": selection,
            "expansion_signal_invariants.csv": invariant_frame,
        }
        for filename, frame in reports.items():
            atomic_write_csv(report_root / filename, frame)
        atomic_write_json(report_root / "expansion_signal_summary.json", summary)
        atomic_write_json(report_root / "expansion_signal_config.json", run_config)
        atomic_write_text(
            report_root / "expansion_signal_findings.md",
            findings_markdown(summary, benchmark_scorecard, null_comparison, timing, concentration),
        )

        metadata_root = output_root / "_metadata"
        atomic_write_json(metadata_root / "dataset_config.json", run_config)
        atomic_write_json(metadata_root / "latest_run_summary.json", summary)
        run_report_root = report_root / "runs" / run_id
        run_report_root.mkdir(parents=True, exist_ok=True)
        for filename in REPORT_FILENAMES:
            source = report_root / filename
            destination = run_report_root / filename
            if source.suffix == ".csv":
                atomic_write_text(destination, source.read_text(encoding="utf-8"))
            elif source.suffix == ".json":
                atomic_write_json(destination, load_json(source))
            else:
                atomic_write_text(destination, source.read_text(encoding="utf-8"))

        connection.execute(
            "UPDATE runs SET status=?, completed_at_utc=?, error_text=NULL WHERE run_id=?",
            (summary["status"], iso_utc(utc_now()), run_id),
        )
        connection.commit()
        if failed_invariants:
            raise RuntimeError(f"Forensic audit failed {failed_invariants} invariants")

        print("Forensic audit completed.")
        print(f"Units completed      : {completed}")
        print(f"Units reused         : {skipped}")
        print(f"Invariant failures   : {failed_invariants}")
        print(f"Findings             : {report_root / 'expansion_signal_findings.md'}")
        return 0
    except Exception:
        connection.execute(
            "UPDATE runs SET status='FAILED', completed_at_utc=?, error_text=? WHERE run_id=?",
            (iso_utc(utc_now()), traceback.format_exc(), run_id),
        )
        connection.commit()
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
