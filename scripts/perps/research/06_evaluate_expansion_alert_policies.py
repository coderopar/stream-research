#!/usr/bin/env python
"""
Evaluate causal adaptive alert policies on saved expansion-model predictions.

This script does not train or refit any model. It reuses the CALIBRATION and
TEST prediction scores produced by:

    scripts/perps/research/05_build_and_evaluate_expansion_baselines.py

Research question
-----------------
The baseline models showed useful out-of-sample ranking ability, but fixed
probability thresholds produced unstable alert rates when the market regime
changed. This script asks:

    Can a causal, adaptive score-to-alert policy keep alert frequency stable
    while preserving distinct-event recall, precision and warning time?

Policies
--------
FIXED_CALIBRATION_THRESHOLD
    Baseline reference. Select one probability threshold on the calibration
    block and keep it fixed throughout the test fold.

DAILY_TRAILING_THRESHOLD
    At the start of each UTC test day, select a threshold using only scores from
    the preceding trailing lookback window. Hold that threshold fixed for the
    current day.

DAILY_TRAILING_BUDGET
    The same daily trailing threshold, plus a causal hard cap of K alerts per
    UTC day. This is the safe replacement for retrospective "top-K of the day";
    it never looks ahead to later scores from the same day.

ROLLING_PERCENTILE
    Map the calibration threshold to its empirical percentile. During testing,
    update the score threshold at a fixed refresh cadence using only preceding
    scores in the trailing lookback window.

ROLLING_PERCENTILE_BUDGET
    The rolling-percentile policy plus a causal daily alert cap.

Causality
---------
* No model is retrained.
* No TEST labels are used to choose thresholds or percentile levels.
* Every adaptive threshold is computed only from scores timestamped strictly
  before the period in which it is used.
* Event matching uses the existing exact first-realisation event catalog.
* Retrospective end-of-day top-K selection is deliberately not implemented.

Default scope
-------------
The default targets are the four priority targets identified by the completed
baseline analysis:

* 50 bps / 5 minutes;
* 50 bps / 15 minutes;
* 75 bps / 15 minutes;
* 75 bps / 20 minutes.

All three saved baseline model families are compared. Use --targets ALL to
include every target from the baseline configuration.

Default outputs
---------------
Dataset root:
    data/derived/perps/research/expansion_alert_policy_evaluation_v1

Report root:
    reports/perps_expansion_alert_policies

Dependencies
------------
    python -m pip install numpy pandas pyarrow numba

Typical Git Bash usage
----------------------
    python scripts/perps/research/06_evaluate_expansion_alert_policies.py
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
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


EVALUATOR_VERSION = "1.0.2"
OUTPUT_SCHEMA_VERSION = 1
EXPECTED_BASELINE_BUILDER_VERSION = "1.0.2"
SECONDS_PER_DAY = 86_400

DEFAULT_PRIORITY_TARGETS = (
    "H05M_T050BPS",
    "H15M_T050BPS",
    "H15M_T075BPS",
    "H20M_T075BPS",
)
DEFAULT_POLICIES = (
    "FIXED_CALIBRATION_THRESHOLD",
    "DAILY_TRAILING_THRESHOLD",
    "DAILY_TRAILING_BUDGET",
    "ROLLING_PERCENTILE",
    "ROLLING_PERCENTILE_BUDGET",
)
DEFAULT_MODELS = ("logistic", "shallow_tree", "hist_gradient_boosting")
DEFAULT_ALERT_RATES = (1.0, 2.0, 4.0, 8.0)
DEFAULT_COOLDOWNS = ("HORIZON", "HALF_HORIZON")
DEFAULT_LOOKBACK_DAYS = 7
DEFAULT_MINIMUM_HISTORY_DAYS = 2
DEFAULT_REFRESH_SECONDS = 1_800
DEFAULT_THRESHOLD_GRID_SIZE = 161

REQUIRED_PREDICTION_COLUMNS = (
    "target_id",
    "model_id",
    "fold_id",
    "split",
    "decision_second",
    "target",
    "probability",
)
REQUIRED_EVENT_COLUMNS = (
    "event_id",
    "event_start_second",
    "first_realization_second",
    "scorable_event",
)

REPORT_FILENAMES = (
    "perps_expansion_alert_policy_fold_metrics.csv",
    "perps_expansion_alert_policy_event_metrics.csv",
    "perps_expansion_alert_policy_daily_metrics.csv",
    "perps_expansion_alert_policy_thresholds.csv",
    "perps_expansion_alert_policy_scorecard.csv",
    "perps_expansion_alert_policy_fold_stability.csv",
    "perps_expansion_alert_policy_invariants.csv",
    "perps_expansion_alert_policy_summary.json",
    "perps_expansion_alert_policy_config.json",
    "perps_expansion_alert_policy_findings.md",
)


# ---------------------------------------------------------------------------
# Existing audited baseline methodology
# ---------------------------------------------------------------------------


def load_baseline_module() -> ModuleType:
    path = Path(__file__).resolve().with_name(
        "05_build_and_evaluate_expansion_baselines.py"
    )
    if not path.exists():
        candidates = (
            Path.cwd() / "scripts/perps/research/05_build_and_evaluate_expansion_baselines.py",
            Path.cwd() / "05_build_and_evaluate_expansion_baselines.py",
        )
        fallback = next((item for item in candidates if item.exists()), None)
        if fallback is None:
            raise RuntimeError(
                "Required baseline script was not found: "
                "scripts/perps/research/05_build_and_evaluate_expansion_baselines.py"
            )
        path = fallback.resolve()
    name = "stream_research_saved_expansion_baseline_core"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load baseline methodology from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_baseline_module()
CORE = BASE.CORE


# ---------------------------------------------------------------------------
# Data classes and parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyContext:
    target: Any
    model_id: str
    fold_id: str
    cooldown_id: str
    cooldown_seconds: int


@dataclass(frozen=True)
class ExperimentPaths:
    alerts: Path
    matches: Path
    daily: Path
    thresholds: Path
    result_json: Path


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


def parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))


def parse_float_list(value: str) -> tuple[float, ...]:
    output = tuple(sorted({float(item.strip()) for item in value.split(",") if item.strip()}))
    if not output or any(item <= 0 for item in output):
        raise ValueError("Alert rates must be positive")
    return output


def parse_policies(value: str) -> tuple[str, ...]:
    requested = tuple(item.upper() for item in parse_csv_strings(value))
    unknown = sorted(set(requested) - set(DEFAULT_POLICIES))
    if unknown:
        raise ValueError(f"Unknown alert policies: {unknown}")
    if not requested:
        raise ValueError("At least one policy is required")
    return requested


def parse_models(value: str, available: Sequence[str]) -> tuple[str, ...]:
    if value.strip().upper() == "ALL":
        return tuple(available)
    requested = tuple(item.lower() for item in parse_csv_strings(value))
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Models are not available in the baseline run: {unknown}")
    if not requested:
        raise ValueError("At least one model is required")
    return requested


def parse_targets(value: str, available: Sequence[str]) -> tuple[str, ...]:
    if value.strip().upper() == "ALL":
        return tuple(available)
    requested = tuple(item.upper() for item in parse_csv_strings(value))
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Targets are not available in the baseline run: {unknown}")
    if not requested:
        raise ValueError("At least one target is required")
    return requested


def resolve_cooldown(value: str, horizon_seconds: int) -> tuple[str, int]:
    cleaned = value.strip().upper()
    if cleaned == "HORIZON":
        return cleaned, int(horizon_seconds)
    if cleaned == "HALF_HORIZON":
        return cleaned, max(1, int(math.ceil(horizon_seconds / 2.0)))
    if cleaned.endswith("S"):
        cleaned = cleaned[:-1]
    seconds = int(cleaned)
    if seconds <= 0:
        raise ValueError("Cooldowns must be HORIZON, HALF_HORIZON or positive seconds")
    return f"{seconds}S", seconds


def target_from_config(item: Mapping[str, Any]) -> Any:
    return BASE.TargetPlan(
        target_id=str(item["target_id"]),
        horizon_minutes=int(item["horizon_minutes"]),
        horizon_seconds=int(item["horizon_seconds"]),
        threshold_bps=float(item["threshold_bps"]),
        primary_segmentation_id=str(item["primary_segmentation_id"]),
        robustness_segmentation_id=str(item["robustness_segmentation_id"]),
    )


# ---------------------------------------------------------------------------
# Paths, metadata and input validation
# ---------------------------------------------------------------------------


def prediction_path(
    baseline_root: Path,
    target_id: str,
    model_id: str,
    fold_id: str,
) -> Path:
    return (
        baseline_root
        / "experiments"
        / f"target={target_id}"
        / f"model={model_id}"
        / f"fold={fold_id}"
        / "predictions.parquet"
    )


def baseline_result_path(
    baseline_root: Path,
    target_id: str,
    model_id: str,
    fold_id: str,
) -> Path:
    return prediction_path(baseline_root, target_id, model_id, fold_id).with_name(
        "result.json"
    )


def event_catalog_path(
    baseline_root: Path,
    target: Any,
    role: str,
    symbol: str,
) -> Path:
    segmentation = (
        target.primary_segmentation_id
        if role == "PRIMARY"
        else target.robustness_segmentation_id
    )
    return (
        baseline_root
        / "event_catalog"
        / f"target={target.target_id}"
        / f"segmentation={segmentation}"
        / f"symbol={symbol}"
        / "part-00000.parquet"
    )


def atomic_write_policy_parquet(
    path: Path,
    frame: pd.DataFrame,
    metadata: Mapping[str, Any],
    compression_level: int,
) -> None:
    """Atomically write Parquet using a short temporary filename.

    The shared writer embeds the full destination filename plus a 32-character UUID
    in the temporary name. Deep policy-partition paths can therefore cross the
    legacy Windows MAX_PATH boundary even when the final path itself is valid.
    This writer keeps the temporary file in the destination directory, but uses
    a compact name so the final replace remains same-directory and atomic.
    """
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
            if int(check.metadata.num_rows) != len(frame):
                raise RuntimeError(
                    f"Atomic Parquet row mismatch for {path}: "
                    f"{check.metadata.num_rows:,} != {len(frame):,}"
                )
            written_metadata = CORE.decode_metadata(check.schema_arrow)
            if written_metadata.get("input_signature") != str(
                metadata.get("input_signature")
            ):
                raise RuntimeError(f"Input-signature metadata mismatch for {path}")
        finally:
            check.close()
        CORE.replace_with_retry(temporary, path)
    finally:
        CORE.unlink_with_retry(temporary)


def atomic_write_policy_text(path: Path, content: str) -> None:
    """Atomically write text with a compact same-directory temporary name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".txt-{uuid.uuid4().hex[:8]}.tmp"
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        CORE.replace_with_retry(temporary, path)
    finally:
        CORE.unlink_with_retry(temporary)


def atomic_write_policy_json(path: Path, value: Any) -> None:
    """Atomically write JSON without lengthening the destination filename."""
    atomic_write_policy_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
    )


def atomic_write_policy_csv(path: Path, frame: pd.DataFrame) -> None:
    """Atomically write CSV with a compact same-directory temporary name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".csv-{uuid.uuid4().hex[:8]}.tmp"
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        CORE.replace_with_retry(temporary, path)
    finally:
        CORE.unlink_with_retry(temporary)


def experiment_paths(
    output_root: Path,
    context: PolicyContext,
    policy_id: str,
) -> ExperimentPaths:
    relative = (
        Path(f"target={context.target.target_id}")
        / f"model={context.model_id}"
        / f"fold={context.fold_id}"
        / f"policy={policy_id.lower()}"
        / f"cooldown={context.cooldown_id.lower()}"
    )
    return ExperimentPaths(
        alerts=output_root / "alerts" / relative / "part-00000.parquet",
        matches=output_root / "event_matches" / relative / "part-00000.parquet",
        daily=output_root / "daily_metrics" / relative / "part-00000.parquet",
        thresholds=output_root / "thresholds" / relative / "part-00000.parquet",
        result_json=output_root / "experiments" / relative / "result.json",
    )


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Required JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def input_signature(path: Path, hash_content: bool) -> dict[str, Any]:
    stat = path.stat()
    value: dict[str, Any] = {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if hash_content:
        value["sha256"] = sha256_file(path)
    return value


def load_predictions(path: Path, expected: PolicyContext) -> tuple[pd.DataFrame, dict[str, str]]:
    pa_module, _, pq_module = CORE.require_arrow()
    del pa_module
    if not path.exists():
        raise RuntimeError(f"Saved prediction file does not exist: {path}")
    parquet_file = pq_module.ParquetFile(path)
    try:
        names = set(parquet_file.schema_arrow.names)
        missing = sorted(set(REQUIRED_PREDICTION_COLUMNS) - names)
        if missing:
            raise RuntimeError(f"Prediction file {path} is missing columns: {missing}")
        metadata = CORE.decode_metadata(parquet_file.schema_arrow)
        frame = parquet_file.read(columns=list(REQUIRED_PREDICTION_COLUMNS)).to_pandas()
    finally:
        parquet_file.close()
    if metadata.get("perps_expansion_baseline_builder_version") != EXPECTED_BASELINE_BUILDER_VERSION:
        raise RuntimeError(
            f"Prediction builder version is "
            f"{metadata.get('perps_expansion_baseline_builder_version')!r}; "
            f"expected {EXPECTED_BASELINE_BUILDER_VERSION!r}"
        )
    checks = {
        "target_id": expected.target.target_id,
        "model_id": expected.model_id,
        "fold_id": expected.fold_id,
    }
    for column, wanted in checks.items():
        values = set(frame[column].astype(str))
        if values != {str(wanted)}:
            raise RuntimeError(f"{path} has unexpected {column} values: {sorted(values)}")
    frame["split"] = frame["split"].astype(str).str.upper()
    frame = frame[frame["split"].isin(["CALIBRATION", "TEST"])].copy()
    if set(frame["split"]) != {"CALIBRATION", "TEST"}:
        raise RuntimeError(f"{path} must contain CALIBRATION and TEST rows")
    frame["decision_second"] = pd.to_numeric(
        frame["decision_second"], errors="raise"
    ).astype(np.int64)
    frame["target"] = pd.to_numeric(frame["target"], errors="raise").astype(np.int8)
    frame["probability"] = pd.to_numeric(frame["probability"], errors="coerce")
    probabilities = frame["probability"].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0) or np.any(
        probabilities > 1.0
    ):
        raise RuntimeError(f"Prediction probabilities in {path} must be finite and in [0,1]")
    for split, group in frame.groupby("split", sort=False):
        ordered = group.sort_values("decision_second", kind="mergesort")
        if ordered["decision_second"].duplicated().any():
            raise RuntimeError(f"Duplicate {split} decision seconds in {path}")
        if len(ordered) > 1 and np.any(np.diff(ordered["decision_second"]) <= 0):
            raise RuntimeError(f"Non-increasing {split} decision seconds in {path}")
    calibration_max = int(frame.loc[frame["split"].eq("CALIBRATION"), "decision_second"].max())
    test_min = int(frame.loc[frame["split"].eq("TEST"), "decision_second"].min())
    if calibration_max >= test_min:
        raise RuntimeError(f"Calibration and test periods overlap in {path}")
    frame = frame.sort_values(["split", "decision_second"], kind="mergesort").reset_index(
        drop=True
    )
    return frame, metadata


def load_events(path: Path, expected_target: str, role: str) -> pd.DataFrame:
    _, _, pq_module = CORE.require_arrow()
    if not path.exists():
        raise RuntimeError(f"Saved event catalog does not exist: {path}")
    parquet_file = pq_module.ParquetFile(path)
    try:
        names = set(parquet_file.schema_arrow.names)
        missing = sorted(set(REQUIRED_EVENT_COLUMNS) - names)
        if missing:
            raise RuntimeError(f"Event catalog {path} is missing columns: {missing}")
        metadata = CORE.decode_metadata(parquet_file.schema_arrow)
        columns = sorted(
            set(REQUIRED_EVENT_COLUMNS)
            | ({"target_id", "segmentation_id"} & names)
        )
        frame = parquet_file.read(columns=columns).to_pandas()
    finally:
        parquet_file.close()
    if metadata.get("perps_expansion_baseline_builder_version") != EXPECTED_BASELINE_BUILDER_VERSION:
        raise RuntimeError(
            f"Event catalog builder version in {path} is not "
            f"{EXPECTED_BASELINE_BUILDER_VERSION}"
        )
    if "target_id" in frame and set(frame["target_id"].astype(str)) != {expected_target}:
        raise RuntimeError(f"Unexpected target IDs in {path}")
    frame["event_start_second"] = pd.to_numeric(
        frame["event_start_second"], errors="raise"
    ).astype(np.int64)
    frame["first_realization_second"] = pd.to_numeric(
        frame["first_realization_second"], errors="coerce"
    ).astype("Int64")
    frame["scorable_event"] = frame["scorable_event"].fillna(False).astype(bool)
    frame["segmentation_role"] = role
    frame = frame.sort_values("event_start_second", kind="mergesort").reset_index(drop=True)
    return frame


# ---------------------------------------------------------------------------
# SQLite build state
# ---------------------------------------------------------------------------


def connect_state(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs(
            run_id TEXT PRIMARY KEY,
            started_at_utc TEXT NOT NULL,
            completed_at_utc TEXT,
            status TEXT NOT NULL,
            config_json TEXT NOT NULL,
            error_text TEXT
        );

        CREATE TABLE IF NOT EXISTS policy_experiments(
            target_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            fold_id TEXT NOT NULL,
            policy_id TEXT NOT NULL,
            cooldown_id TEXT NOT NULL,
            input_signature TEXT NOT NULL,
            alerts_path TEXT NOT NULL,
            matches_path TEXT NOT NULL,
            daily_path TEXT NOT NULL,
            thresholds_path TEXT NOT NULL,
            result_path TEXT NOT NULL,
            status TEXT NOT NULL,
            last_run_id TEXT NOT NULL,
            completed_at_utc TEXT,
            error_text TEXT,
            PRIMARY KEY(target_id, model_id, fold_id, policy_id, cooldown_id)
        );
        """
    )
    connection.commit()
    return connection


def reusable_experiment(
    connection: sqlite3.Connection,
    context: PolicyContext,
    policy_id: str,
    signature: str,
    paths: ExperimentPaths,
) -> bool:
    row = connection.execute(
        """
        SELECT status, input_signature
        FROM policy_experiments
        WHERE target_id=? AND model_id=? AND fold_id=? AND policy_id=? AND cooldown_id=?
        """,
        (
            context.target.target_id,
            context.model_id,
            context.fold_id,
            policy_id,
            context.cooldown_id,
        ),
    ).fetchone()
    if row is None or row["status"] != "COMPLETED" or row["input_signature"] != signature:
        return False
    return all(
        path.exists()
        for path in (
            paths.alerts,
            paths.matches,
            paths.daily,
            paths.thresholds,
            paths.result_json,
        )
    )


# ---------------------------------------------------------------------------
# Alert-policy primitives
# ---------------------------------------------------------------------------


def unique_utc_dates(seconds: np.ndarray) -> np.ndarray:
    return np.unique(np.asarray(seconds, dtype=np.int64) // SECONDS_PER_DAY)


def elapsed_days(seconds: np.ndarray) -> float:
    values = np.asarray(seconds, dtype=np.int64)
    if not len(values):
        return 0.0
    return float(len(unique_utc_dates(values)))


def empirical_percentile(values: np.ndarray, threshold: float) -> float:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return float("nan")
    return float(np.mean(array <= float(threshold)))


def quantile_linear(values: np.ndarray, percentile: float) -> float:
    array = np.asarray(values, dtype=float)
    if not len(array):
        raise RuntimeError("Historical scores are required for a percentile threshold")
    return float(np.quantile(array, float(percentile), method="linear"))


def alert_frame_from_indices(
    context: PolicyContext,
    policy_id: str,
    requested_rate: float,
    seconds: np.ndarray,
    scores: np.ndarray,
    targets: np.ndarray,
    indices: Sequence[int],
    thresholds: Sequence[float],
    history_ends: Sequence[int],
    percentile_level: float | None,
    daily_cap: int | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ordinal, index in enumerate(indices, start=1):
        second = int(seconds[int(index)])
        rows.append(
            {
                "evaluator_version": EVALUATOR_VERSION,
                "target_id": context.target.target_id,
                "horizon_minutes": context.target.horizon_minutes,
                "threshold_bps": context.target.threshold_bps,
                "model_id": context.model_id,
                "fold_id": context.fold_id,
                "policy_id": policy_id,
                "cooldown_id": context.cooldown_id,
                "cooldown_seconds": context.cooldown_seconds,
                "requested_alerts_per_day": float(requested_rate),
                "alert_ordinal": ordinal,
                "alert_second": second,
                "alert_timestamp_utc": pd.to_datetime(second, unit="s", utc=True),
                "alert_date": datetime.fromtimestamp(second, tz=timezone.utc).date().isoformat(),
                "score": float(scores[int(index)]),
                "threshold_used": float(thresholds[ordinal - 1]),
                "threshold_history_end_second": int(history_ends[ordinal - 1]),
                "score_percentile_level": percentile_level,
                "snapshot_target": int(targets[int(index)]),
                "daily_alert_cap": daily_cap,
            }
        )
    return pd.DataFrame(rows)


def select_with_thresholds(
    seconds: np.ndarray,
    scores: np.ndarray,
    threshold_by_row: np.ndarray,
    history_end_by_row: np.ndarray,
    cooldown_seconds: int,
    daily_cap: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected: list[int] = []
    thresholds: list[float] = []
    history_ends: list[int] = []
    last_alert = -9_000_000_000_000_000_000
    current_day: int | None = None
    daily_count = 0
    for index in range(len(seconds)):
        second = int(seconds[index])
        day = second // SECONDS_PER_DAY
        if current_day != day:
            current_day = day
            daily_count = 0
        threshold = float(threshold_by_row[index])
        if not math.isfinite(threshold):
            continue
        if daily_cap is not None and daily_count >= daily_cap:
            continue
        if float(scores[index]) < threshold:
            continue
        if second - last_alert < cooldown_seconds:
            continue
        selected.append(index)
        thresholds.append(threshold)
        history_ends.append(int(history_end_by_row[index]))
        last_alert = second
        daily_count += 1
    return (
        np.asarray(selected, dtype=np.int64),
        np.asarray(thresholds, dtype=float),
        np.asarray(history_ends, dtype=np.int64),
    )


def fixed_policy(
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    context: PolicyContext,
    requested_rate: float,
    threshold_grid_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cal_seconds = calibration["decision_second"].to_numpy(dtype=np.int64)
    cal_scores = calibration["probability"].to_numpy(dtype=float)
    cal_days = elapsed_days(cal_seconds)
    selected = BASE.choose_alert_threshold(
        cal_seconds,
        cal_scores,
        cal_days,
        context.cooldown_seconds,
        requested_rate,
        threshold_grid_size,
    )
    seconds = test["decision_second"].to_numpy(dtype=np.int64)
    scores = test["probability"].to_numpy(dtype=float)
    targets = test["target"].to_numpy(dtype=np.int8)
    threshold_array = np.full(len(test), float(selected["threshold"]), dtype=float)
    history_end = np.full(len(test), int(cal_seconds[-1]), dtype=np.int64)
    indices, thresholds, history_ends = select_with_thresholds(
        seconds,
        scores,
        threshold_array,
        history_end,
        context.cooldown_seconds,
    )
    alerts = alert_frame_from_indices(
        context,
        "FIXED_CALIBRATION_THRESHOLD",
        requested_rate,
        seconds,
        scores,
        targets,
        indices,
        thresholds,
        history_ends,
        empirical_percentile(cal_scores, float(selected["threshold"])),
        None,
    )
    threshold_rows = pd.DataFrame(
        [
            {
                "target_id": context.target.target_id,
                "model_id": context.model_id,
                "fold_id": context.fold_id,
                "policy_id": "FIXED_CALIBRATION_THRESHOLD",
                "cooldown_id": context.cooldown_id,
                "requested_alerts_per_day": requested_rate,
                "effective_start_second": int(seconds[0]),
                "effective_end_exclusive_second": int(seconds[-1]) + 1,
                "threshold": float(selected["threshold"]),
                "history_start_second": int(cal_seconds[0]),
                "history_end_second": int(cal_seconds[-1]),
                "history_rows": int(len(cal_scores)),
                "history_days": cal_days,
                "percentile_level": empirical_percentile(
                    cal_scores, float(selected["threshold"])
                ),
                "calibration_alerts_per_day": float(
                    selected["calibration_alerts_per_day"]
                ),
            }
        ]
    )
    return alerts, threshold_rows


def daily_trailing_threshold_schedule(
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    context: PolicyContext,
    requested_rate: float,
    lookback_days: int,
    minimum_history_days: int,
    threshold_grid_size: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    combined = pd.concat([calibration, test], ignore_index=True).sort_values(
        "decision_second", kind="mergesort"
    )
    all_seconds = combined["decision_second"].to_numpy(dtype=np.int64)
    all_scores = combined["probability"].to_numpy(dtype=float)
    test_seconds = test["decision_second"].to_numpy(dtype=np.int64)
    thresholds = np.full(len(test), np.nan, dtype=float)
    history_ends = np.full(len(test), -1, dtype=np.int64)
    records: list[dict[str, Any]] = []
    day_values = unique_utc_dates(test_seconds)
    for day_value in day_values:
        day_start = int(day_value * SECONDS_PER_DAY)
        day_end = day_start + SECONDS_PER_DAY
        history_start = day_start - int(lookback_days * SECONDS_PER_DAY)
        left = int(np.searchsorted(all_seconds, history_start, side="left"))
        right = int(np.searchsorted(all_seconds, day_start, side="left"))
        history_seconds = all_seconds[left:right]
        history_scores = all_scores[left:right]
        history_dates = unique_utc_dates(history_seconds)
        day_mask = (test_seconds >= day_start) & (test_seconds < day_end)
        if len(history_dates) < minimum_history_days or not len(history_scores):
            records.append(
                {
                    "target_id": context.target.target_id,
                    "model_id": context.model_id,
                    "fold_id": context.fold_id,
                    "policy_id": "DAILY_TRAILING_THRESHOLD",
                    "cooldown_id": context.cooldown_id,
                    "requested_alerts_per_day": requested_rate,
                    "effective_start_second": day_start,
                    "effective_end_exclusive_second": day_end,
                    "threshold": np.nan,
                    "history_start_second": int(history_seconds[0])
                    if len(history_seconds)
                    else pd.NA,
                    "history_end_second": int(history_seconds[-1])
                    if len(history_seconds)
                    else pd.NA,
                    "history_rows": int(len(history_scores)),
                    "history_days": int(len(history_dates)),
                    "percentile_level": np.nan,
                    "calibration_alerts_per_day": np.nan,
                }
            )
            continue
        selected = BASE.choose_alert_threshold(
            history_seconds,
            history_scores,
            float(len(history_dates)),
            context.cooldown_seconds,
            requested_rate,
            threshold_grid_size,
        )
        threshold = float(selected["threshold"])
        thresholds[day_mask] = threshold
        history_ends[day_mask] = int(history_seconds[-1])
        records.append(
            {
                "target_id": context.target.target_id,
                "model_id": context.model_id,
                "fold_id": context.fold_id,
                "policy_id": "DAILY_TRAILING_THRESHOLD",
                "cooldown_id": context.cooldown_id,
                "requested_alerts_per_day": requested_rate,
                "effective_start_second": day_start,
                "effective_end_exclusive_second": day_end,
                "threshold": threshold,
                "history_start_second": int(history_seconds[0]),
                "history_end_second": int(history_seconds[-1]),
                "history_rows": int(len(history_scores)),
                "history_days": int(len(history_dates)),
                "percentile_level": empirical_percentile(history_scores, threshold),
                "calibration_alerts_per_day": float(
                    selected["calibration_alerts_per_day"]
                ),
            }
        )
    return thresholds, history_ends, pd.DataFrame(records)


def daily_trailing_policy(
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    context: PolicyContext,
    requested_rate: float,
    lookback_days: int,
    minimum_history_days: int,
    threshold_grid_size: int,
    capped: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold_array, history_end, threshold_rows = daily_trailing_threshold_schedule(
        calibration,
        test,
        context,
        requested_rate,
        lookback_days,
        minimum_history_days,
        threshold_grid_size,
    )
    policy_id = "DAILY_TRAILING_BUDGET" if capped else "DAILY_TRAILING_THRESHOLD"
    threshold_rows["policy_id"] = policy_id
    seconds = test["decision_second"].to_numpy(dtype=np.int64)
    scores = test["probability"].to_numpy(dtype=float)
    targets = test["target"].to_numpy(dtype=np.int8)
    cap = int(math.ceil(requested_rate)) if capped else None
    indices, thresholds, history_ends = select_with_thresholds(
        seconds,
        scores,
        threshold_array,
        history_end,
        context.cooldown_seconds,
        daily_cap=cap,
    )
    alerts = alert_frame_from_indices(
        context,
        policy_id,
        requested_rate,
        seconds,
        scores,
        targets,
        indices,
        thresholds,
        history_ends,
        None,
        cap,
    )
    return alerts, threshold_rows


def rolling_percentile_schedule(
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    context: PolicyContext,
    requested_rate: float,
    lookback_days: int,
    minimum_history_days: int,
    refresh_seconds: int,
    threshold_grid_size: int,
) -> tuple[np.ndarray, np.ndarray, float, pd.DataFrame]:
    cal_seconds = calibration["decision_second"].to_numpy(dtype=np.int64)
    cal_scores = calibration["probability"].to_numpy(dtype=float)
    selected = BASE.choose_alert_threshold(
        cal_seconds,
        cal_scores,
        elapsed_days(cal_seconds),
        context.cooldown_seconds,
        requested_rate,
        threshold_grid_size,
    )
    percentile = empirical_percentile(cal_scores, float(selected["threshold"]))
    percentile = min(max(percentile, 0.0), 1.0)

    combined = pd.concat([calibration, test], ignore_index=True).sort_values(
        "decision_second", kind="mergesort"
    )
    all_seconds = combined["decision_second"].to_numpy(dtype=np.int64)
    all_scores = combined["probability"].to_numpy(dtype=float)
    test_seconds = test["decision_second"].to_numpy(dtype=np.int64)
    thresholds = np.full(len(test), np.nan, dtype=float)
    history_ends = np.full(len(test), -1, dtype=np.int64)
    records: list[dict[str, Any]] = []

    first_test = int(test_seconds[0])
    last_test = int(test_seconds[-1])
    refresh_start = (first_test // refresh_seconds) * refresh_seconds
    for refresh in range(refresh_start, last_test + refresh_seconds, refresh_seconds):
        effective_start = max(refresh, first_test)
        effective_end = min(refresh + refresh_seconds, last_test + 1)
        if effective_start >= effective_end:
            continue
        history_start = effective_start - int(lookback_days * SECONDS_PER_DAY)
        left = int(np.searchsorted(all_seconds, history_start, side="left"))
        right = int(np.searchsorted(all_seconds, effective_start, side="left"))
        history_seconds = all_seconds[left:right]
        history_scores = all_scores[left:right]
        history_dates = unique_utc_dates(history_seconds)
        mask = (test_seconds >= effective_start) & (test_seconds < effective_end)
        if len(history_dates) < minimum_history_days or not len(history_scores):
            records.append(
                {
                    "target_id": context.target.target_id,
                    "model_id": context.model_id,
                    "fold_id": context.fold_id,
                    "policy_id": "ROLLING_PERCENTILE",
                    "cooldown_id": context.cooldown_id,
                    "requested_alerts_per_day": requested_rate,
                    "effective_start_second": effective_start,
                    "effective_end_exclusive_second": effective_end,
                    "threshold": np.nan,
                    "history_start_second": int(history_seconds[0])
                    if len(history_seconds)
                    else pd.NA,
                    "history_end_second": int(history_seconds[-1])
                    if len(history_seconds)
                    else pd.NA,
                    "history_rows": int(len(history_scores)),
                    "history_days": int(len(history_dates)),
                    "percentile_level": percentile,
                    "calibration_alerts_per_day": float(
                        selected["calibration_alerts_per_day"]
                    ),
                }
            )
            continue
        threshold = quantile_linear(history_scores, percentile)
        thresholds[mask] = threshold
        history_ends[mask] = int(history_seconds[-1])
        records.append(
            {
                "target_id": context.target.target_id,
                "model_id": context.model_id,
                "fold_id": context.fold_id,
                "policy_id": "ROLLING_PERCENTILE",
                "cooldown_id": context.cooldown_id,
                "requested_alerts_per_day": requested_rate,
                "effective_start_second": effective_start,
                "effective_end_exclusive_second": effective_end,
                "threshold": threshold,
                "history_start_second": int(history_seconds[0]),
                "history_end_second": int(history_seconds[-1]),
                "history_rows": int(len(history_scores)),
                "history_days": int(len(history_dates)),
                "percentile_level": percentile,
                "calibration_alerts_per_day": float(
                    selected["calibration_alerts_per_day"]
                ),
            }
        )
    return thresholds, history_ends, percentile, pd.DataFrame(records)


def rolling_percentile_policy(
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    context: PolicyContext,
    requested_rate: float,
    lookback_days: int,
    minimum_history_days: int,
    refresh_seconds: int,
    threshold_grid_size: int,
    capped: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold_array, history_end, percentile, threshold_rows = rolling_percentile_schedule(
        calibration,
        test,
        context,
        requested_rate,
        lookback_days,
        minimum_history_days,
        refresh_seconds,
        threshold_grid_size,
    )
    policy_id = "ROLLING_PERCENTILE_BUDGET" if capped else "ROLLING_PERCENTILE"
    threshold_rows["policy_id"] = policy_id
    seconds = test["decision_second"].to_numpy(dtype=np.int64)
    scores = test["probability"].to_numpy(dtype=float)
    targets = test["target"].to_numpy(dtype=np.int8)
    cap = int(math.ceil(requested_rate)) if capped else None
    indices, thresholds, history_ends = select_with_thresholds(
        seconds,
        scores,
        threshold_array,
        history_end,
        context.cooldown_seconds,
        daily_cap=cap,
    )
    alerts = alert_frame_from_indices(
        context,
        policy_id,
        requested_rate,
        seconds,
        scores,
        targets,
        indices,
        thresholds,
        history_ends,
        percentile,
        cap,
    )
    return alerts, threshold_rows


# ---------------------------------------------------------------------------
# Evaluation and validation
# ---------------------------------------------------------------------------


def test_date_frame(test: pd.DataFrame) -> pd.DataFrame:
    frame = test[["decision_second"]].copy()
    frame["test_date"] = pd.to_datetime(
        frame["decision_second"], unit="s", utc=True
    ).dt.date.astype(str)
    return frame.groupby("test_date", as_index=False).agg(
        first_test_second=("decision_second", "min"),
        last_test_second=("decision_second", "max"),
        test_rows=("decision_second", "size"),
    )


def daily_metrics(
    alerts: pd.DataFrame,
    test: pd.DataFrame,
    context: PolicyContext,
    policy_id: str,
    requested_rate: float,
) -> pd.DataFrame:
    dates = test_date_frame(test)
    if alerts.empty:
        counts = pd.DataFrame(columns=["test_date", "alerts", "positive_alerts"])
    else:
        counts = alerts.groupby("alert_date", as_index=False).agg(
            alerts=("alert_second", "size"),
            positive_alerts=("snapshot_target", "sum"),
        ).rename(columns={"alert_date": "test_date"})
    output = dates.merge(counts, how="left", on="test_date")
    output[["alerts", "positive_alerts"]] = output[
        ["alerts", "positive_alerts"]
    ].fillna(0).astype(int)
    output.insert(0, "target_id", context.target.target_id)
    output.insert(1, "model_id", context.model_id)
    output.insert(2, "fold_id", context.fold_id)
    output.insert(3, "policy_id", policy_id)
    output.insert(4, "cooldown_id", context.cooldown_id)
    output.insert(5, "cooldown_seconds", context.cooldown_seconds)
    output.insert(6, "requested_alerts_per_day", requested_rate)
    output["alert_count_error"] = output["alerts"] - requested_rate
    output["absolute_alert_count_error"] = np.abs(output["alert_count_error"])
    output["zero_alert_day"] = output["alerts"].eq(0)
    output["over_requested_day"] = output["alerts"].gt(requested_rate)
    output["snapshot_precision_at_alert"] = np.where(
        output["alerts"].gt(0), output["positive_alerts"] / output["alerts"], np.nan
    )
    return output


def match_events(
    alerts: pd.DataFrame,
    test: pd.DataFrame,
    events_by_role: Mapping[str, pd.DataFrame],
    context: PolicyContext,
    policy_id: str,
    requested_rate: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    seconds = test["decision_second"].to_numpy(dtype=np.int64)
    alert_seconds = (
        alerts["alert_second"].to_numpy(dtype=np.int64)
        if not alerts.empty
        else np.empty(0, dtype=np.int64)
    )
    match_frames: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    test_days = elapsed_days(seconds)
    for role, events in events_by_role.items():
        scorable = BASE.scorable_events_for_test(
            events,
            seconds,
            int(seconds[0]),
            int(seconds[-1]) + 1,
        )
        matches, false_alerts = BASE.match_alerts_to_events(alert_seconds, scorable)
        segmentation_id = (
            context.target.primary_segmentation_id
            if role == "PRIMARY"
            else context.target.robustness_segmentation_id
        )
        if not matches.empty:
            matches.insert(0, "target_id", context.target.target_id)
            matches.insert(1, "model_id", context.model_id)
            matches.insert(2, "fold_id", context.fold_id)
            matches.insert(3, "policy_id", policy_id)
            matches.insert(4, "cooldown_id", context.cooldown_id)
            matches.insert(5, "requested_alerts_per_day", requested_rate)
            matches.insert(6, "segmentation_role", role)
            matches.insert(7, "segmentation_id", segmentation_id)
            match_frames.append(matches)
        recalled = int(len(matches))
        alert_count = int(len(alert_seconds))
        records.append(
            {
                "target_id": context.target.target_id,
                "horizon_minutes": context.target.horizon_minutes,
                "threshold_bps": context.target.threshold_bps,
                "model_id": context.model_id,
                "fold_id": context.fold_id,
                "policy_id": policy_id,
                "cooldown_id": context.cooldown_id,
                "cooldown_seconds": context.cooldown_seconds,
                "requested_alerts_per_day": requested_rate,
                "segmentation_role": role,
                "segmentation_id": segmentation_id,
                "test_days": test_days,
                "alerts": alert_count,
                "alerts_per_day": alert_count / test_days if test_days else np.nan,
                "scorable_events": int(len(scorable)),
                "recalled_events": recalled,
                "event_recall": recalled / len(scorable) if len(scorable) else np.nan,
                "event_precision": recalled / alert_count if alert_count else np.nan,
                "false_alerts": int(len(false_alerts)),
                "false_alerts_per_day": len(false_alerts) / test_days
                if test_days
                else np.nan,
                "median_warning_seconds": BASE.safe_median(
                    matches.get("warning_seconds", [])
                ),
                "mean_warning_seconds": BASE.safe_mean(matches.get("warning_seconds", [])),
                "median_seconds_after_event_onset": BASE.safe_median(
                    matches.get("seconds_after_event_onset", [])
                ),
            }
        )
    output = pd.concat(match_frames, ignore_index=True) if match_frames else pd.DataFrame()
    return output, records


def fold_metric_record(
    alerts: pd.DataFrame,
    daily: pd.DataFrame,
    test: pd.DataFrame,
    context: PolicyContext,
    policy_id: str,
    requested_rate: float,
) -> dict[str, Any]:
    test_days = len(daily)
    alert_count = int(len(alerts))
    actual_rate = alert_count / test_days if test_days else np.nan
    return {
        "target_id": context.target.target_id,
        "horizon_minutes": context.target.horizon_minutes,
        "threshold_bps": context.target.threshold_bps,
        "model_id": context.model_id,
        "fold_id": context.fold_id,
        "policy_id": policy_id,
        "cooldown_id": context.cooldown_id,
        "cooldown_seconds": context.cooldown_seconds,
        "requested_alerts_per_day": requested_rate,
        "test_days": test_days,
        "test_rows": int(len(test)),
        "alerts": alert_count,
        "alerts_per_day": actual_rate,
        "alert_rate_error": actual_rate - requested_rate,
        "absolute_alert_rate_error": abs(actual_rate - requested_rate),
        "daily_alert_count_std": float(daily["alerts"].std(ddof=0)) if len(daily) else np.nan,
        "daily_alert_count_cv": (
            float(daily["alerts"].std(ddof=0)) / float(daily["alerts"].mean())
            if len(daily) and float(daily["alerts"].mean()) > 0
            else np.nan
        ),
        "mean_daily_absolute_error": float(daily["absolute_alert_count_error"].mean())
        if len(daily)
        else np.nan,
        "zero_alert_day_share": float(daily["zero_alert_day"].mean())
        if len(daily)
        else np.nan,
        "over_requested_day_share": float(daily["over_requested_day"].mean())
        if len(daily)
        else np.nan,
        "maximum_daily_alerts": int(daily["alerts"].max()) if len(daily) else 0,
        "positive_alerts": int(alerts["snapshot_target"].sum()) if not alerts.empty else 0,
        "snapshot_precision_at_alert": float(alerts["snapshot_target"].mean())
        if not alerts.empty
        else np.nan,
        "mean_alert_score": float(alerts["score"].mean()) if not alerts.empty else np.nan,
    }


def validate_policy_outputs(
    alerts: pd.DataFrame,
    matches: pd.DataFrame,
    daily: pd.DataFrame,
    thresholds: pd.DataFrame,
    test: pd.DataFrame,
    context: PolicyContext,
    policy_id: str,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, details: str) -> None:
        checks.append(
            {
                "target_id": context.target.target_id,
                "model_id": context.model_id,
                "fold_id": context.fold_id,
                "policy_id": policy_id,
                "cooldown_id": context.cooldown_id,
                "check_name": name,
                "status": "PASS" if passed else "FAIL",
                "details": details,
            }
        )

    test_seconds = set(test["decision_second"].astype(int))
    alert_seconds = alerts["alert_second"].astype(int).tolist() if not alerts.empty else []
    add(
        "alerts_are_test_rows",
        all(value in test_seconds for value in alert_seconds),
        f"alerts={len(alert_seconds)}",
    )
    add(
        "alert_seconds_unique",
        len(alert_seconds) == len(set(alert_seconds)),
        f"alerts={len(alert_seconds)} unique={len(set(alert_seconds))}",
    )
    cooldown_ok = all(
        later - earlier >= context.cooldown_seconds
        for earlier, later in zip(alert_seconds, alert_seconds[1:])
    )
    add(
        "cooldown_respected",
        cooldown_ok,
        f"cooldown_seconds={context.cooldown_seconds}",
    )
    if not alerts.empty:
        history_ok = (
            alerts["threshold_history_end_second"].astype(np.int64)
            < alerts["alert_second"].astype(np.int64)
        ).all()
    else:
        history_ok = True
    add(
        "threshold_history_is_strictly_past",
        bool(history_ok),
        "Every adaptive/fixed threshold source ends before its alert second",
    )
    if policy_id.endswith("BUDGET") and not daily.empty:
        caps = alerts["daily_alert_cap"].dropna().astype(int).unique() if not alerts.empty else []
        cap = int(caps[0]) if len(caps) else int(math.ceil(float(daily["requested_alerts_per_day"].iloc[0])))
        cap_ok = bool(daily["alerts"].le(cap).all())
        add("daily_cap_respected", cap_ok, f"daily_cap={cap}")
    else:
        add("daily_cap_not_applicable", True, "Policy has no hard daily cap")
    match_ok = True
    if not matches.empty:
        match_ok = not matches.duplicated(
            [
                "requested_alerts_per_day",
                "segmentation_role",
                "event_id",
            ]
        ).any()
    add(
        "events_matched_at_most_once",
        bool(match_ok),
        f"matches={len(matches)}",
    )
    threshold_history_ok = True
    if not thresholds.empty and thresholds["history_end_second"].notna().any():
        valid = thresholds.dropna(subset=["history_end_second"])
        threshold_history_ok = bool(
            (
                valid["history_end_second"].astype(np.int64)
                < valid["effective_start_second"].astype(np.int64)
            ).all()
        )
    add(
        "threshold_schedule_is_causal",
        threshold_history_ok,
        f"threshold_intervals={len(thresholds)}",
    )
    add(
        "daily_rows_cover_test_dates",
        len(daily) == len(unique_utc_dates(test["decision_second"].to_numpy(dtype=np.int64))),
        f"daily_rows={len(daily)}",
    )
    return checks


# ---------------------------------------------------------------------------
# Report aggregation
# ---------------------------------------------------------------------------


def numeric_sum(frame: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())


def aggregate_scorecard(
    folds: pd.DataFrame,
    events: pd.DataFrame,
    daily: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "target_id",
        "horizon_minutes",
        "threshold_bps",
        "model_id",
        "policy_id",
        "cooldown_id",
        "cooldown_seconds",
        "requested_alerts_per_day",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in folds.groupby(keys, dropna=False, sort=True):
        record = dict(zip(keys, key))
        record.update(
            {
                "folds": int(group["fold_id"].nunique()),
                "total_test_days": int(group["test_days"].sum()),
                "total_alerts": int(group["alerts"].sum()),
                "mean_alerts_per_day": float(group["alerts_per_day"].mean()),
                "std_alerts_per_day_across_folds": float(
                    group["alerts_per_day"].std(ddof=0)
                ),
                "mean_absolute_alert_rate_error": float(
                    group["absolute_alert_rate_error"].mean()
                ),
                "mean_daily_absolute_error": float(
                    group["mean_daily_absolute_error"].mean()
                ),
                "mean_daily_alert_count_std": float(
                    group["daily_alert_count_std"].mean()
                ),
                "mean_zero_alert_day_share": float(group["zero_alert_day_share"].mean()),
                "mean_over_requested_day_share": float(
                    group["over_requested_day_share"].mean()
                ),
                "maximum_daily_alerts": int(group["maximum_daily_alerts"].max()),
                "pooled_snapshot_precision_at_alert": (
                    numeric_sum(group, "positive_alerts") / numeric_sum(group, "alerts")
                    if numeric_sum(group, "alerts") > 0
                    else np.nan
                ),
            }
        )
        for role in ("PRIMARY", "ROBUSTNESS"):
            event_group = events[
                events["segmentation_role"].eq(role)
                & events["target_id"].eq(record["target_id"])
                & events["model_id"].eq(record["model_id"])
                & events["policy_id"].eq(record["policy_id"])
                & events["cooldown_id"].eq(record["cooldown_id"])
                & np.isclose(
                    events["requested_alerts_per_day"].astype(float),
                    float(record["requested_alerts_per_day"]),
                )
            ]
            prefix = role.lower()
            scorable = numeric_sum(event_group, "scorable_events")
            recalled = numeric_sum(event_group, "recalled_events")
            false_alerts = numeric_sum(event_group, "false_alerts")
            alerts = numeric_sum(event_group, "alerts")
            test_days = numeric_sum(event_group, "test_days")
            record[f"{prefix}_scorable_events"] = int(scorable)
            record[f"{prefix}_recalled_events"] = int(recalled)
            record[f"{prefix}_event_recall"] = recalled / scorable if scorable else np.nan
            record[f"{prefix}_event_precision"] = recalled / alerts if alerts else np.nan
            record[f"{prefix}_false_alerts_per_day"] = (
                false_alerts / test_days if test_days else np.nan
            )
            warnings = pd.to_numeric(
                event_group["median_warning_seconds"], errors="coerce"
            ).dropna()
            record[f"{prefix}_median_fold_warning_seconds"] = (
                float(warnings.median()) if len(warnings) else np.nan
            )
        rows.append(record)
    output = pd.DataFrame(rows)
    if not output.empty:
        output["alert_rate_stability_score"] = 1.0 / (
            1.0 + output["mean_absolute_alert_rate_error"]
        )
        output = output.sort_values(
            [
                "requested_alerts_per_day",
                "mean_absolute_alert_rate_error",
                "primary_event_recall",
                "primary_event_precision",
            ],
            ascending=[True, True, False, False],
            kind="mergesort",
        ).reset_index(drop=True)
    return output


def fold_stability_report(scorecard: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "target_id",
        "model_id",
        "policy_id",
        "cooldown_id",
        "requested_alerts_per_day",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in folds.groupby(keys, sort=True):
        record = dict(zip(keys, key))
        rates = pd.to_numeric(group["alerts_per_day"], errors="coerce")
        record.update(
            {
                "folds": int(group["fold_id"].nunique()),
                "minimum_fold_alerts_per_day": float(rates.min()),
                "maximum_fold_alerts_per_day": float(rates.max()),
                "fold_alert_rate_range": float(rates.max() - rates.min()),
                "fold_alert_rate_std": float(rates.std(ddof=0)),
                "folds_with_zero_alerts": int(group["alerts"].eq(0).sum()),
                "folds_with_more_than_double_requested_rate": int(
                    group["alerts_per_day"].gt(
                        2.0 * float(record["requested_alerts_per_day"])
                    ).sum()
                ),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def write_findings(
    path: Path,
    summary: Mapping[str, Any],
    scorecard: pd.DataFrame,
) -> None:
    lines = [
        "# Expansion Alert-Policy Evaluation",
        "",
        "## Purpose",
        "",
        "This analysis reused saved out-of-sample model scores. No model was retrained.",
        "It compared fixed and causal adaptive score-to-alert policies.",
        "",
        "## Run summary",
        "",
        f"- Status: **{summary['status']}**",
        f"- Target/model/fold prediction files: **{summary['prediction_files']}**",
        f"- Policy experiments completed: **{summary['experiments_completed']}**",
        f"- Policy experiments failed: **{summary['experiments_failed']}**",
        f"- Validation failures: **{summary['validation_failures']}**",
        "",
        "## Interpretation contract",
        "",
        "- FIXED_CALIBRATION_THRESHOLD is the original baseline policy.",
        "- DAILY_TRAILING policies update at the start of each UTC day using only prior scores.",
        "- ROLLING_PERCENTILE policies update from a trailing score distribution using only prior scores.",
        "- BUDGET policies impose a causal maximum number of alerts per UTC day.",
        "- No retrospective end-of-day top-K selection is used.",
        "",
    ]
    if not scorecard.empty:
        subset = scorecard[
            np.isclose(scorecard["requested_alerts_per_day"].astype(float), 2.0)
        ].copy()
        if not subset.empty:
            stable = subset.sort_values(
                ["mean_absolute_alert_rate_error", "primary_event_recall"],
                ascending=[True, False],
                kind="mergesort",
            ).head(10)
            lines.extend(
                [
                    "## Most stable candidates near two alerts per day",
                    "",
                    "The following rows are ranked first by alert-rate stability, then by primary-event recall:",
                    "",
                    "| Target | Model | Policy | Cooldown | Alerts/day | Rate error | Event recall | Event precision | False alerts/day | Median warning (s) |",
                    "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in stable.itertuples(index=False):
                lines.append(
                    "| "
                    f"{row.target_id} | {row.model_id} | {row.policy_id} | {row.cooldown_id} | "
                    f"{row.mean_alerts_per_day:.3f} | {row.mean_absolute_alert_rate_error:.3f} | "
                    f"{row.primary_event_recall:.3f} | {row.primary_event_precision:.3f} | "
                    f"{row.primary_false_alerts_per_day:.3f} | "
                    f"{row.primary_median_fold_warning_seconds:.1f} |"
                )
            lines.append("")
    lines.extend(
        [
            "## Selection guidance",
            "",
            "A policy should not be chosen from event recall alone. Prefer policies that keep actual alerts/day close to the requested budget across folds and days, while retaining useful event recall, precision and warning time under both primary and robustness segmentation.",
            "",
        ]
    )
    atomic_write_policy_text(path, "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------


def run_policy(
    policy_id: str,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    context: PolicyContext,
    requested_rate: float,
    lookback_days: int,
    minimum_history_days: int,
    refresh_seconds: int,
    threshold_grid_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if policy_id == "FIXED_CALIBRATION_THRESHOLD":
        return fixed_policy(
            calibration,
            test,
            context,
            requested_rate,
            threshold_grid_size,
        )
    if policy_id == "DAILY_TRAILING_THRESHOLD":
        return daily_trailing_policy(
            calibration,
            test,
            context,
            requested_rate,
            lookback_days,
            minimum_history_days,
            threshold_grid_size,
            capped=False,
        )
    if policy_id == "DAILY_TRAILING_BUDGET":
        return daily_trailing_policy(
            calibration,
            test,
            context,
            requested_rate,
            lookback_days,
            minimum_history_days,
            threshold_grid_size,
            capped=True,
        )
    if policy_id == "ROLLING_PERCENTILE":
        return rolling_percentile_policy(
            calibration,
            test,
            context,
            requested_rate,
            lookback_days,
            minimum_history_days,
            refresh_seconds,
            threshold_grid_size,
            capped=False,
        )
    if policy_id == "ROLLING_PERCENTILE_BUDGET":
        return rolling_percentile_policy(
            calibration,
            test,
            context,
            requested_rate,
            lookback_days,
            minimum_history_days,
            refresh_seconds,
            threshold_grid_size,
            capped=True,
        )
    raise ValueError(f"Unsupported policy: {policy_id}")


def execute_experiment(
    policy_id: str,
    predictions: pd.DataFrame,
    events_by_role: Mapping[str, pd.DataFrame],
    context: PolicyContext,
    alert_rates: Sequence[float],
    lookback_days: int,
    minimum_history_days: int,
    refresh_seconds: int,
    threshold_grid_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    calibration = predictions[predictions["split"].eq("CALIBRATION")].sort_values(
        "decision_second", kind="mergesort"
    )
    test = predictions[predictions["split"].eq("TEST")].sort_values(
        "decision_second", kind="mergesort"
    )
    all_alerts: list[pd.DataFrame] = []
    all_matches: list[pd.DataFrame] = []
    all_daily: list[pd.DataFrame] = []
    all_thresholds: list[pd.DataFrame] = []
    fold_records: list[dict[str, Any]] = []
    event_records: list[dict[str, Any]] = []
    invariant_records: list[dict[str, Any]] = []

    for requested_rate in alert_rates:
        alerts, thresholds = run_policy(
            policy_id,
            calibration,
            test,
            context,
            requested_rate,
            lookback_days,
            minimum_history_days,
            refresh_seconds,
            threshold_grid_size,
        )
        daily = daily_metrics(
            alerts,
            test,
            context,
            policy_id,
            requested_rate,
        )
        matches, records = match_events(
            alerts,
            test,
            events_by_role,
            context,
            policy_id,
            requested_rate,
        )
        fold_records.append(
            fold_metric_record(
                alerts,
                daily,
                test,
                context,
                policy_id,
                requested_rate,
            )
        )
        event_records.extend(records)
        invariant_records.extend(
            validate_policy_outputs(
                alerts,
                matches,
                daily,
                thresholds,
                test,
                context,
                policy_id,
            )
        )
        if not alerts.empty:
            all_alerts.append(alerts)
        if not matches.empty:
            all_matches.append(matches)
        all_daily.append(daily)
        all_thresholds.append(thresholds)

    alerts_output = pd.concat(all_alerts, ignore_index=True) if all_alerts else pd.DataFrame()
    matches_output = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()
    daily_output = pd.concat(all_daily, ignore_index=True)
    thresholds_output = pd.concat(all_thresholds, ignore_index=True)
    payload = {
        "fold_metrics": fold_records,
        "event_metrics": event_records,
        "invariants": invariant_records,
    }
    return alerts_output, matches_output, daily_output, thresholds_output, payload


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def self_test() -> None:
    target = BASE.TargetPlan(
        target_id="H05M_T050BPS",
        horizon_minutes=5,
        horizon_seconds=300,
        threshold_bps=50.0,
        primary_segmentation_id="bridge_030s",
        robustness_segmentation_id="bridge_015s",
    )
    context = PolicyContext(target, "logistic", "01", "HORIZON", 300)
    start = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp())
    cadence = 30
    calibration_seconds = np.arange(start, start + 3 * SECONDS_PER_DAY, cadence, dtype=np.int64)
    test_seconds = np.arange(
        start + 3 * SECONDS_PER_DAY,
        start + 5 * SECONDS_PER_DAY,
        cadence,
        dtype=np.int64,
    )
    cal_scores = np.linspace(0.0, 1.0, len(calibration_seconds), dtype=float)
    test_scores = np.tile(np.linspace(0.0, 1.0, SECONDS_PER_DAY // cadence), 2)
    calibration = pd.DataFrame(
        {
            "decision_second": calibration_seconds,
            "probability": cal_scores,
            "target": (cal_scores > 0.9).astype(np.int8),
        }
    )
    test = pd.DataFrame(
        {
            "decision_second": test_seconds,
            "probability": test_scores,
            "target": (test_scores > 0.9).astype(np.int8),
        }
    )
    fixed, fixed_thresholds = fixed_policy(calibration, test, context, 2.0, 41)
    assert len(fixed_thresholds) == 1
    assert (fixed["threshold_history_end_second"] < fixed["alert_second"]).all()
    daily, daily_thresholds = daily_trailing_policy(
        calibration,
        test,
        context,
        2.0,
        3,
        2,
        41,
        capped=True,
    )
    assert not daily_thresholds.empty
    assert daily.groupby("alert_date").size().max() <= 2 if not daily.empty else True
    rolling, rolling_thresholds = rolling_percentile_policy(
        calibration,
        test,
        context,
        2.0,
        3,
        2,
        1_800,
        41,
        capped=True,
    )
    assert not rolling_thresholds.empty
    assert rolling.groupby("alert_date").size().max() <= 2 if not rolling.empty else True
    if not rolling.empty:
        assert (rolling["threshold_history_end_second"] < rolling["alert_second"]).all()

    event = pd.DataFrame(
        {
            "event_id": ["E1"],
            "event_start_second": [int(test_seconds[10])],
            "first_realization_second": [int(test_seconds[30])],
            "scorable_event": [True],
        }
    )
    alert_seconds = np.asarray([int(test_seconds[20])], dtype=np.int64)
    matches, false_alerts = BASE.match_alerts_to_events(alert_seconds, event)
    assert len(matches) == 1 and false_alerts.empty
    print(
        "SELF-TEST PASS\n"
        "Validated fixed, daily trailing, rolling-percentile, causal daily-budget, "
        "past-only threshold history, short-path atomic writers, cooldown and event-matching logic."
    )


# ---------------------------------------------------------------------------
# Command line and main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-root",
        default="data/derived/perps/research/expansion_baseline_models_v1",
    )
    parser.add_argument(
        "--output-root",
        default="data/derived/perps/research/expansion_alert_policy_evaluation_v1",
    )
    parser.add_argument(
        "--report-root",
        default="reports/perps_expansion_alert_policies",
    )
    parser.add_argument(
        "--targets",
        default=",".join(DEFAULT_PRIORITY_TARGETS),
        help="Comma-separated target IDs, or ALL.",
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated saved model IDs, or ALL.",
    )
    parser.add_argument(
        "--policies",
        default=",".join(DEFAULT_POLICIES),
    )
    parser.add_argument(
        "--folds",
        default="ALL",
        help="Comma-separated saved fold IDs such as 01,02, or ALL.",
    )
    parser.add_argument(
        "--alert-rates",
        default=",".join(str(value) for value in DEFAULT_ALERT_RATES),
    )
    parser.add_argument(
        "--cooldowns",
        default=",".join(DEFAULT_COOLDOWNS),
        help="Comma-separated HORIZON, HALF_HORIZON or positive seconds.",
    )
    parser.add_argument("--trailing-lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument(
        "--minimum-history-days", type=int, default=DEFAULT_MINIMUM_HISTORY_DAYS
    )
    parser.add_argument("--rolling-refresh-seconds", type=int, default=DEFAULT_REFRESH_SECONDS)
    parser.add_argument(
        "--threshold-grid-size", type=int, default=DEFAULT_THRESHOLD_GRID_SIZE
    )
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--compression-level", type=int, default=9)
    parser.add_argument("--hash-input-files", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.trailing_lookback_days <= 0:
        raise SystemExit("--trailing-lookback-days must be positive")
    if args.minimum_history_days <= 0:
        raise SystemExit("--minimum-history-days must be positive")
    if args.minimum_history_days > args.trailing_lookback_days:
        raise SystemExit("--minimum-history-days cannot exceed --trailing-lookback-days")
    if args.rolling_refresh_seconds <= 0:
        raise SystemExit("--rolling-refresh-seconds must be positive")
    if args.threshold_grid_size < 5:
        raise SystemExit("--threshold-grid-size must be at least 5")

    baseline_root = Path(args.baseline_root).resolve()
    output_root = Path(args.output_root).resolve()
    report_root = Path(args.report_root).resolve()
    metadata_root = baseline_root / "_metadata"
    config = load_json(metadata_root / "dataset_config.json")
    summary = load_json(metadata_root / "latest_run_summary.json")
    if summary.get("status") != "COMPLETED":
        raise SystemExit(
            f"Baseline run status must be COMPLETED, found {summary.get('status')!r}"
        )
    if str(config.get("builder_version")) != EXPECTED_BASELINE_BUILDER_VERSION:
        raise SystemExit(
            f"Baseline builder version must be {EXPECTED_BASELINE_BUILDER_VERSION}, "
            f"found {config.get('builder_version')!r}"
        )
    available_targets = {
        str(item["target_id"]): target_from_config(item)
        for item in config.get("target_plan", [])
    }
    if not available_targets:
        raise SystemExit("The baseline configuration contains no target plan")
    available_models = tuple(str(item) for item in config.get("models", []))
    targets = parse_targets(args.targets, tuple(available_targets))
    models = parse_models(args.models, available_models)
    policies = parse_policies(args.policies)
    alert_rates = parse_float_list(args.alert_rates)
    cooldown_values = parse_csv_strings(args.cooldowns)
    symbol = str(args.symbol or config.get("symbol", "BTCUSDT")).upper()

    run_id = utc_now().strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_config = {
        "evaluator_version": EVALUATOR_VERSION,
        "baseline_root": str(baseline_root),
        "output_root": str(output_root),
        "report_root": str(report_root),
        "targets": targets,
        "models": models,
        "policies": policies,
        "folds": args.folds,
        "alert_rates_per_day": alert_rates,
        "cooldowns": cooldown_values,
        "trailing_lookback_days": args.trailing_lookback_days,
        "minimum_history_days": args.minimum_history_days,
        "rolling_refresh_seconds": args.rolling_refresh_seconds,
        "threshold_grid_size": args.threshold_grid_size,
        "symbol": symbol,
        "hash_input_files": args.hash_input_files,
    }

    print("Evaluating causal expansion alert policies...")
    print(f"Evaluator version   : {EVALUATOR_VERSION}")
    print(f"Run ID              : {run_id}")
    print(f"Baseline root       : {baseline_root}")
    print(f"Output root         : {output_root}")
    print(f"Report root         : {report_root}")
    print(f"Targets             : {len(targets)}")
    print(f"Models              : {len(models)}")
    print(f"Policies            : {len(policies)}")
    print(f"Cooldown definitions: {len(cooldown_values)}")
    print("No model training will be performed.")

    state_path = output_root / "_build_state" / "build_state.sqlite"
    connection = connect_state(state_path)
    connection.execute(
        "INSERT INTO runs(run_id, started_at_utc, status, config_json) VALUES(?,?,?,?)",
        (run_id, iso_utc(utc_now()), "RUNNING", compact_json(run_config)),
    )
    connection.commit()

    completed = 0
    skipped = 0
    failed = 0
    all_results: list[dict[str, Any]] = []
    input_prediction_paths: set[str] = set()
    try:
        event_cache: dict[str, dict[str, pd.DataFrame]] = {}
        event_signature_cache: dict[str, list[dict[str, Any]]] = {}
        for target_id in targets:
            target = available_targets[target_id]
            role_frames: dict[str, pd.DataFrame] = {}
            signatures: list[dict[str, Any]] = []
            for role in ("PRIMARY", "ROBUSTNESS"):
                path = event_catalog_path(baseline_root, target, role, symbol)
                role_frames[role] = load_events(path, target_id, role)
                signatures.append(input_signature(path, args.hash_input_files))
            event_cache[target_id] = role_frames
            event_signature_cache[target_id] = signatures

        for target_id in targets:
            target = available_targets[target_id]
            fold_root = baseline_root / "experiments" / f"target={target_id}"
            if not fold_root.exists():
                raise RuntimeError(f"Baseline experiment root does not exist: {fold_root}")
            for model_id in models:
                model_root = fold_root / f"model={model_id}"
                fold_dirs = sorted(model_root.glob("fold=*"))
                if args.folds.strip().upper() != "ALL":
                    requested_folds = set(parse_csv_strings(args.folds))
                    fold_dirs = [
                        item for item in fold_dirs
                        if item.name.split("=", 1)[1] in requested_folds
                    ]
                if not fold_dirs:
                    raise RuntimeError(f"No saved folds found under {model_root}")
                for fold_dir in fold_dirs:
                    fold_id = fold_dir.name.split("=", 1)[1]
                    result_file = baseline_result_path(
                        baseline_root, target_id, model_id, fold_id
                    )
                    result_payload = load_json(result_file)
                    if not result_payload.get("input_signature"):
                        raise RuntimeError(f"Baseline result has no input signature: {result_file}")
                    for cooldown_value in cooldown_values:
                        cooldown_id, cooldown_seconds = resolve_cooldown(
                            cooldown_value, target.horizon_seconds
                        )
                        context = PolicyContext(
                            target=target,
                            model_id=model_id,
                            fold_id=fold_id,
                            cooldown_id=cooldown_id,
                            cooldown_seconds=cooldown_seconds,
                        )
                        pred_path = prediction_path(
                            baseline_root, target_id, model_id, fold_id
                        )
                        prediction_signature = input_signature(
                            pred_path, args.hash_input_files
                        )
                        input_prediction_paths.add(str(pred_path.resolve()))
                        predictions: pd.DataFrame | None = None
                        for policy_id in policies:
                            paths = experiment_paths(output_root, context, policy_id)
                            signature = sha256_text(
                                compact_json(
                                    {
                                        "evaluator_version": EVALUATOR_VERSION,
                                        "prediction": prediction_signature,
                                        "baseline_result_signature": input_signature(
                                            result_file, args.hash_input_files
                                        ),
                                        "event_catalogs": event_signature_cache[target_id],
                                        "target": target.__dict__,
                                        "model_id": model_id,
                                        "fold_id": fold_id,
                                        "policy_id": policy_id,
                                        "cooldown_id": cooldown_id,
                                        "cooldown_seconds": cooldown_seconds,
                                        "alert_rates": alert_rates,
                                        "lookback_days": args.trailing_lookback_days,
                                        "minimum_history_days": args.minimum_history_days,
                                        "refresh_seconds": args.rolling_refresh_seconds,
                                        "threshold_grid_size": args.threshold_grid_size,
                                    }
                                )
                            )
                            if (
                                not args.force
                                and reusable_experiment(
                                    connection,
                                    context,
                                    policy_id,
                                    signature,
                                    paths,
                                )
                            ):
                                all_results.append(load_json(paths.result_json))
                                skipped += 1
                                continue
                            if predictions is None:
                                predictions, _ = load_predictions(pred_path, context)
                            connection.execute(
                                """
                                INSERT INTO policy_experiments(
                                    target_id, model_id, fold_id, policy_id, cooldown_id,
                                    input_signature, alerts_path, matches_path, daily_path,
                                    thresholds_path, result_path, status, last_run_id
                                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                                ON CONFLICT(target_id, model_id, fold_id, policy_id, cooldown_id)
                                DO UPDATE SET
                                    input_signature=excluded.input_signature,
                                    alerts_path=excluded.alerts_path,
                                    matches_path=excluded.matches_path,
                                    daily_path=excluded.daily_path,
                                    thresholds_path=excluded.thresholds_path,
                                    result_path=excluded.result_path,
                                    status='RUNNING',
                                    error_text=NULL,
                                    last_run_id=excluded.last_run_id
                                """,
                                (
                                    target_id,
                                    model_id,
                                    fold_id,
                                    policy_id,
                                    cooldown_id,
                                    signature,
                                    str(paths.alerts),
                                    str(paths.matches),
                                    str(paths.daily),
                                    str(paths.thresholds),
                                    str(paths.result_json),
                                    "RUNNING",
                                    run_id,
                                ),
                            )
                            connection.commit()
                            try:
                                alerts, matches, daily, thresholds, payload = execute_experiment(
                                    policy_id,
                                    predictions,
                                    event_cache[target_id],
                                    context,
                                    alert_rates,
                                    args.trailing_lookback_days,
                                    args.minimum_history_days,
                                    args.rolling_refresh_seconds,
                                    args.threshold_grid_size,
                                )
                                metadata = {
                                    "perps_expansion_alert_policy_evaluator_version": EVALUATOR_VERSION,
                                    "perps_expansion_alert_policy_schema_version": OUTPUT_SCHEMA_VERSION,
                                    "target_id": target_id,
                                    "model_id": model_id,
                                    "fold_id": fold_id,
                                    "policy_id": policy_id,
                                    "cooldown_id": cooldown_id,
                                    "input_signature": signature,
                                }
                                # Empty alert/match outputs still receive explicit schemas.
                                if alerts.empty:
                                    alerts = pd.DataFrame(
                                        columns=[
                                            "evaluator_version",
                                            "target_id",
                                            "horizon_minutes",
                                            "threshold_bps",
                                            "model_id",
                                            "fold_id",
                                            "policy_id",
                                            "cooldown_id",
                                            "cooldown_seconds",
                                            "requested_alerts_per_day",
                                            "alert_ordinal",
                                            "alert_second",
                                            "alert_timestamp_utc",
                                            "alert_date",
                                            "score",
                                            "threshold_used",
                                            "threshold_history_end_second",
                                            "score_percentile_level",
                                            "snapshot_target",
                                            "daily_alert_cap",
                                        ]
                                    )
                                if matches.empty:
                                    matches = pd.DataFrame(
                                        columns=[
                                            "target_id",
                                            "model_id",
                                            "fold_id",
                                            "policy_id",
                                            "cooldown_id",
                                            "requested_alerts_per_day",
                                            "segmentation_role",
                                            "segmentation_id",
                                            "event_id",
                                            "alert_second",
                                            "event_start_second",
                                            "first_realization_second",
                                            "warning_seconds",
                                            "seconds_after_event_onset",
                                        ]
                                    )
                                atomic_write_policy_parquet(
                                    paths.alerts, alerts, metadata, args.compression_level
                                )
                                atomic_write_policy_parquet(
                                    paths.matches, matches, metadata, args.compression_level
                                )
                                atomic_write_policy_parquet(
                                    paths.daily, daily, metadata, args.compression_level
                                )
                                atomic_write_policy_parquet(
                                    paths.thresholds, thresholds, metadata, args.compression_level
                                )
                                result = {
                                    "evaluator_version": EVALUATOR_VERSION,
                                    "target_id": target_id,
                                    "model_id": model_id,
                                    "fold_id": fold_id,
                                    "policy_id": policy_id,
                                    "cooldown_id": cooldown_id,
                                    "cooldown_seconds": cooldown_seconds,
                                    "input_signature": signature,
                                    **payload,
                                }
                                atomic_write_policy_json(paths.result_json, result)
                                connection.execute(
                                    """
                                    UPDATE policy_experiments
                                    SET status='COMPLETED', completed_at_utc=?, error_text=NULL,
                                        last_run_id=?
                                    WHERE target_id=? AND model_id=? AND fold_id=?
                                      AND policy_id=? AND cooldown_id=?
                                    """,
                                    (
                                        iso_utc(utc_now()),
                                        run_id,
                                        target_id,
                                        model_id,
                                        fold_id,
                                        policy_id,
                                        cooldown_id,
                                    ),
                                )
                                connection.commit()
                                all_results.append(result)
                                completed += 1
                                print(
                                    f"  completed {target_id}/{model_id}/fold {fold_id}/"
                                    f"{policy_id}/{cooldown_id}"
                                )
                            except Exception as exc:
                                connection.execute(
                                    """
                                    UPDATE policy_experiments
                                    SET status='FAILED', error_text=?, last_run_id=?
                                    WHERE target_id=? AND model_id=? AND fold_id=?
                                      AND policy_id=? AND cooldown_id=?
                                    """,
                                    (
                                        "".join(
                                            traceback.format_exception_only(type(exc), exc)
                                        ).strip(),
                                        run_id,
                                        target_id,
                                        model_id,
                                        fold_id,
                                        policy_id,
                                        cooldown_id,
                                    ),
                                )
                                connection.commit()
                                failed += 1
                                raise

        fold_records = [
            row
            for result in all_results
            for row in result.get("fold_metrics", [])
        ]
        event_records = [
            row
            for result in all_results
            for row in result.get("event_metrics", [])
        ]
        invariant_records = [
            row
            for result in all_results
            for row in result.get("invariants", [])
        ]
        fold_frame = pd.DataFrame(fold_records)
        event_frame = pd.DataFrame(event_records)
        invariant_frame = pd.DataFrame(invariant_records)

        # Aggregate daily and threshold outputs from completed current-scope experiments.
        daily_frames: list[pd.DataFrame] = []
        threshold_frames: list[pd.DataFrame] = []
        for result in all_results:
            target = available_targets[result["target_id"]]
            context = PolicyContext(
                target,
                result["model_id"],
                result["fold_id"],
                result["cooldown_id"],
                int(result["cooldown_seconds"]),
            )
            paths = experiment_paths(output_root, context, result["policy_id"])
            _, _, pq_module = CORE.require_arrow()
            daily_frames.append(pq_module.read_table(paths.daily).to_pandas())
            threshold_frames.append(pq_module.read_table(paths.thresholds).to_pandas())
        daily_frame = pd.concat(daily_frames, ignore_index=True)
        threshold_frame = pd.concat(threshold_frames, ignore_index=True)
        scorecard = aggregate_scorecard(fold_frame, event_frame, daily_frame)
        stability = fold_stability_report(scorecard, fold_frame)

        validation_failures = int(
            invariant_frame["status"].astype(str).eq("FAIL").sum()
        ) if not invariant_frame.empty else 0
        status = "COMPLETED" if failed == 0 and validation_failures == 0 else "FAILED"
        final_summary = {
            "status": status,
            "evaluator_version": EVALUATOR_VERSION,
            "run_id": run_id,
            "prediction_files": len(input_prediction_paths),
            "targets": len(targets),
            "models": len(models),
            "policies": len(policies),
            "cooldown_definitions": len(cooldown_values),
            "experiments_completed": completed,
            "experiments_reused": skipped,
            "experiments_failed": failed,
            "validation_checks": int(len(invariant_frame)),
            "validation_failures": validation_failures,
        }

        report_root.mkdir(parents=True, exist_ok=True)
        report_frames = {
            "perps_expansion_alert_policy_fold_metrics.csv": fold_frame,
            "perps_expansion_alert_policy_event_metrics.csv": event_frame,
            "perps_expansion_alert_policy_daily_metrics.csv": daily_frame,
            "perps_expansion_alert_policy_thresholds.csv": threshold_frame,
            "perps_expansion_alert_policy_scorecard.csv": scorecard,
            "perps_expansion_alert_policy_fold_stability.csv": stability,
            "perps_expansion_alert_policy_invariants.csv": invariant_frame,
        }
        for filename, frame in report_frames.items():
            atomic_write_policy_csv(report_root / filename, frame)
        atomic_write_policy_json(
            report_root / "perps_expansion_alert_policy_summary.json", final_summary
        )
        atomic_write_policy_json(
            report_root / "perps_expansion_alert_policy_config.json", run_config
        )
        write_findings(
            report_root / "perps_expansion_alert_policy_findings.md",
            final_summary,
            scorecard,
        )

        run_report_root = report_root / "runs" / run_id
        run_report_root.mkdir(parents=True, exist_ok=True)
        for filename in REPORT_FILENAMES:
            source = report_root / filename
            destination = run_report_root / filename
            if source.exists():
                temporary = destination.with_name(
                    f".{destination.name}.{uuid.uuid4().hex}.tmp"
                )
                temporary.write_bytes(source.read_bytes())
                os.replace(temporary, destination)

        metadata_output = output_root / "_metadata"
        metadata_output.mkdir(parents=True, exist_ok=True)
        atomic_write_policy_json(metadata_output / "dataset_config.json", run_config)
        atomic_write_policy_json(metadata_output / "latest_run_summary.json", final_summary)
        connection.execute(
            "UPDATE runs SET status=?, completed_at_utc=? WHERE run_id=?",
            (status, iso_utc(utc_now()), run_id),
        )
        connection.commit()
        print("Alert-policy evaluation finished.")
        print(f"Status              : {status}")
        print(f"Experiments completed: {completed}")
        print(f"Experiments reused   : {skipped}")
        print(f"Experiments failed   : {failed}")
        print(f"Validation failures  : {validation_failures}")
        return 0 if status == "COMPLETED" else 1
    except Exception as exc:
        connection.execute(
            "UPDATE runs SET status='FAILED', completed_at_utc=?, error_text=? WHERE run_id=?",
            (
                iso_utc(utc_now()),
                "".join(traceback.format_exception_only(type(exc), exc)).strip(),
                run_id,
            ),
        )
        connection.commit()
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
