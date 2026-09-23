#!/usr/bin/env python
"""
Compare causal pre-expansion footprints across a frozen target/horizon shortlist.

Research question
-----------------
Which expansion definitions leave the clearest, earliest and most chronologically
stable causal footprint in the audited perps flow-feature layer?

This script applies exactly the footprint methodology implemented by:

    scripts/perps/participant_behaviour/01_profile_pre_expansion_footprints.py

to a frozen shortlist produced by:

    scripts/perps/research/04_profile_expansion_target_and_segmentation_sensitivity.py

It does not train a model, select a trading direction, modify upstream datasets,
or declare an optimal horizon. It produces comparable descriptive evidence that
will later be evaluated with identical chronological walk-forward models.

Frozen default experiments
--------------------------
Target             Primary event rule    Robustness event rule
35 bps / 5m        bridge_030s            bridge_015s
50 bps / 5m        bridge_030s            bridge_015s
50 bps / 10m       bridge_030s            bridge_015s
50 bps / 15m       bridge_060s            bridge_030s
75 bps / 15m       bridge_030s            bridge_015s
75 bps / 20m       bridge_030s            bridge_015s

Causal observation contract
---------------------------
For event onset second t and positive lead L:

    observation_second = t - L

The default leads are 900, 600, 300, 120, 60, 30 and 0 seconds. Every feature
observation must be an exact existing one-second row with
``flow_feature_context_ready = true``. Missing seconds are never filled.

Controls
--------
Controls are matched separately for every target/segmentation experiment. A
control pseudo-onset must be negative for that exact threshold/horizon target,
outside that target's event and embargo intervals, context-ready at every lead,
and sampled on the configured cadence. Controls are unique within an experiment
and are selected deterministically using the same date/hour/chronological-period
tiers as script 01.

For comparability, the effective control embargo is:

    max(minimum_control_embargo_seconds, target_horizon_seconds)

The default minimum is 900 seconds. This reproduces the existing 50-bps/15m
methodology and avoids using a shorter quiet-context requirement for faster
shortlist targets.

Outputs
-------
Default dataset root:
    data/derived/perps/research/pre_expansion_target_footprints_v1

Default report root:
    reports/perps_pre_expansion_target_footprints

Each target/segmentation has separate event observations, controls, event
catalogue and compact profile Parquet files. SQLite experiment state permits
unchanged experiments to be skipped safely. Global CSV/JSON/Markdown reports
compare all experiments side by side.

Dependencies
------------
    python -m pip install numpy pandas pyarrow

Typical Git Bash usage
----------------------
    python scripts/perps/participant_behaviour/02_compare_pre_expansion_footprints_across_targets.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sqlite3
import sys
import traceback
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


PROFILER_VERSION = "1.0.1"
OUTPUT_SCHEMA_VERSION = 1
EXPECTED_TARGET_PROFILER_VERSION = "1.0.0"
EXPECTED_TARGET_LABEL_SCHEMA_VERSION = 1
EXPECTED_TARGET_EVENT_SCHEMA_VERSION = 1

DEFAULT_TARGET_PLAN = (
    "H05M_T035BPS=bridge_030s|bridge_015s;"
    "H05M_T050BPS=bridge_030s|bridge_015s;"
    "H10M_T050BPS=bridge_030s|bridge_015s;"
    "H15M_T050BPS=bridge_060s|bridge_030s;"
    "H15M_T075BPS=bridge_030s|bridge_015s;"
    "H20M_T075BPS=bridge_030s|bridge_015s"
)
DEFAULT_LEADS_SECONDS = (900, 600, 300, 120, 60, 30, 0)
DEFAULT_SEGMENTATION_ROLES = ("PRIMARY", "ROBUSTNESS")
DEFAULT_CONTROLS_PER_EVENT = 3
DEFAULT_CONTROL_CADENCE_SECONDS = 30
DEFAULT_MINIMUM_CONTROL_EMBARGO_SECONDS = 900
DEFAULT_ADJACENT_HOUR_TOLERANCE = 1

TARGET_PREFIX = "target="
SEGMENTATION_PREFIX = "segmentation="
SYMBOL_PREFIX = "symbol="
DATE_PREFIX = "date="
HORIZON_PREFIX = "horizon="

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
    "maximum_future_upside_bps",
    "maximum_future_downside_bps",
    "path_type",
)

PROFILE_DATASETS = (
    "feature_comparison",
    "matched_effects",
    "trajectory",
    "chronological_effects",
    "stability",
    "top_features",
    "control_match_quality",
    "coverage",
    "invariants",
)

REPORT_FILENAMES = (
    "perps_target_footprint_experiment_summary.csv",
    "perps_target_footprint_feature_comparison.csv",
    "perps_target_footprint_matched_effects.csv",
    "perps_target_footprint_trajectory.csv",
    "perps_target_footprint_chronological_effects.csv",
    "perps_target_footprint_stability.csv",
    "perps_target_footprint_top_features.csv",
    "perps_target_footprint_lead_scorecard.csv",
    "perps_target_footprint_primary_vs_robustness.csv",
    "perps_target_footprint_control_match_quality.csv",
    "perps_target_footprint_coverage.csv",
    "perps_target_footprint_invariants.csv",
    "perps_target_footprint_summary.json",
    "perps_target_footprint_config.json",
    "perps_target_footprint_findings.md",
)

_TARGET_RE = re.compile(r"^H(?P<horizon>\d{2,3})M_T(?P<threshold>\d{3}(?:P\d+)?)BPS$")
_SEGMENTATION_RE = re.compile(r"^bridge_(?P<gap>\d{3,4})s$")


# ---------------------------------------------------------------------------
# Reuse the already-audited single-target methodology without modifying it.
# ---------------------------------------------------------------------------


def load_footprint_core() -> ModuleType:
    path = Path(__file__).resolve().with_name("01_profile_pre_expansion_footprints.py")
    if not path.exists():
        # Supports downloading/running the standalone file from a directory that
        # also contains the previously delivered script.
        fallback = Path.cwd() / "01_profile_pre_expansion_footprints.py"
        if fallback.exists():
            path = fallback.resolve()
        else:
            raise RuntimeError(
                "Required methodology script was not found: "
                "scripts/perps/participant_behaviour/01_profile_pre_expansion_footprints.py"
            )
    module_name = "stream_research_pre_expansion_footprint_core"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load footprint methodology from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_footprint_core()


@dataclass(frozen=True)
class TargetPlan:
    target_id: str
    horizon_minutes: int
    horizon_seconds: int
    threshold_bps: float
    primary_segmentation_id: str
    robustness_segmentation_id: str


@dataclass(frozen=True)
class Experiment:
    target: TargetPlan
    segmentation_role: str
    segmentation_id: str

    @property
    def key(self) -> tuple[str, str]:
        return self.target.target_id, self.segmentation_id


# ---------------------------------------------------------------------------
# Parsing and paths
# ---------------------------------------------------------------------------


def parse_target_id(value: str) -> tuple[int, float]:
    match = _TARGET_RE.fullmatch(value.strip().upper())
    if not match:
        raise ValueError(f"Invalid target ID: {value!r}")
    horizon = int(match.group("horizon"))
    threshold_text = match.group("threshold").replace("P", ".")
    threshold = float(threshold_text)
    if horizon <= 0 or threshold <= 0:
        raise ValueError(f"Target values must be positive: {value!r}")
    return horizon, threshold


def validate_segmentation_id(value: str) -> str:
    cleaned = value.strip()
    if not _SEGMENTATION_RE.fullmatch(cleaned):
        raise ValueError(
            f"Only deterministic bridge segmentations are supported here; got {value!r}"
        )
    return cleaned


def parse_target_plan(value: str) -> tuple[TargetPlan, ...]:
    plans: list[TargetPlan] = []
    seen: set[str] = set()
    for raw_group in value.split(";"):
        group = raw_group.strip()
        if not group:
            continue
        if "=" not in group:
            raise ValueError(f"Target-plan group lacks '=': {group!r}")
        raw_target, raw_segmentations = group.split("=", 1)
        target = raw_target.strip().upper()
        if target in seen:
            raise ValueError(f"Duplicate target in plan: {target}")
        horizon_minutes, threshold_bps = parse_target_id(target)
        segmentations = [item.strip() for item in raw_segmentations.split("|") if item.strip()]
        if len(segmentations) != 2:
            raise ValueError(
                f"Target {target} must define primary|robustness segmentations"
            )
        primary = validate_segmentation_id(segmentations[0])
        robustness = validate_segmentation_id(segmentations[1])
        if primary == robustness:
            raise ValueError(f"Target {target} primary and robustness rules are identical")
        plans.append(
            TargetPlan(
                target_id=target,
                horizon_minutes=horizon_minutes,
                horizon_seconds=horizon_minutes * 60,
                threshold_bps=threshold_bps,
                primary_segmentation_id=primary,
                robustness_segmentation_id=robustness,
            )
        )
        seen.add(target)
    if not plans:
        raise ValueError("Target plan is empty")
    return tuple(plans)


def parse_roles(value: str) -> tuple[str, ...]:
    roles = tuple(dict.fromkeys(item.strip().upper() for item in value.split(",") if item.strip()))
    invalid = sorted(set(roles) - set(DEFAULT_SEGMENTATION_ROLES))
    if invalid:
        raise ValueError(f"Unknown segmentation roles: {invalid}")
    if not roles:
        raise ValueError("At least one segmentation role is required")
    return roles


def experiments_from_plan(plans: Sequence[TargetPlan], roles: Sequence[str]) -> tuple[Experiment, ...]:
    experiments: list[Experiment] = []
    for plan in plans:
        if "PRIMARY" in roles:
            experiments.append(Experiment(plan, "PRIMARY", plan.primary_segmentation_id))
        if "ROBUSTNESS" in roles:
            experiments.append(Experiment(plan, "ROBUSTNESS", plan.robustness_segmentation_id))
    return tuple(experiments)


def horizon_label_root(target_root: Path, horizon_minutes: int) -> Path:
    return target_root / "row_labels" / f"{HORIZON_PREFIX}{horizon_minutes:02d}m"


def target_event_path(target_root: Path, experiment: Experiment, symbol: str) -> Path:
    return (
        target_root
        / "events"
        / f"{TARGET_PREFIX}{experiment.target.target_id}"
        / f"{SEGMENTATION_PREFIX}{experiment.segmentation_id}"
        / f"{SYMBOL_PREFIX}{symbol}"
        / "part-00000.parquet"
    )


def experiment_path(root: Path, role: str, experiment: Experiment, symbol: str) -> Path:
    return (
        root
        / role
        / f"{TARGET_PREFIX}{experiment.target.target_id}"
        / f"{SEGMENTATION_PREFIX}{experiment.segmentation_id}"
        / f"{SYMBOL_PREFIX}{symbol}"
        / "part-00000.parquet"
    )


def profile_path(root: Path, profile_name: str, experiment: Experiment, symbol: str) -> Path:
    return (
        root
        / "profiles"
        / f"{TARGET_PREFIX}{experiment.target.target_id}"
        / f"{SEGMENTATION_PREFIX}{experiment.segmentation_id}"
        / f"{SYMBOL_PREFIX}{symbol}"
        / f"{profile_name}.parquet"
    )


# ---------------------------------------------------------------------------
# Input discovery and validation
# ---------------------------------------------------------------------------


def discover_horizon_label_partitions(
    target_root: Path,
    horizon_minutes: int,
    symbol: str,
    hash_input_files: bool,
) -> dict[date, Any]:
    root = horizon_label_root(target_root, horizon_minutes)
    symbol_root = root / f"{SYMBOL_PREFIX}{symbol}"
    files = sorted(symbol_root.glob(f"{DATE_PREFIX}*/*.parquet"))
    if not files:
        raise RuntimeError(f"No horizon-label Parquet files found under {symbol_root}")
    grouped: dict[date, list[Path]] = {}
    for path in files:
        grouped.setdefault(CORE.extract_partition_day(path), []).append(path)
    output: dict[date, Any] = {}
    for day, paths in sorted(grouped.items()):
        signatures = [CORE.file_signature(path, hash_input_files) for path in sorted(paths)]
        output[day] = CORE.ParquetPartition(
            day=day,
            files=tuple(sorted(paths)),
            rows=int(sum(int(item["rows"]) for item in signatures)),
            signature=CORE.sha256_text(CORE.compact_json(signatures)),
        )
    validate_horizon_label_schema(output, horizon_minutes, symbol)
    return output


def validate_horizon_label_schema(
    partitions: Mapping[date, Any], horizon_minutes: int, symbol: str
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
    experiment: Experiment,
    symbol: str,
) -> pd.DataFrame:
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
            f"Target event profiler version is "
            f"{metadata.get('perps_expansion_target_profiler_version')!r}; "
            f"expected {EXPECTED_TARGET_PROFILER_VERSION!r}"
        )
    if metadata.get("perps_expansion_event_schema_version") != str(
        EXPECTED_TARGET_EVENT_SCHEMA_VERSION
    ):
        raise RuntimeError(
            f"Target event schema version is "
            f"{metadata.get('perps_expansion_event_schema_version')!r}; "
            f"expected {EXPECTED_TARGET_EVENT_SCHEMA_VERSION}"
        )
    metadata_target = metadata.get("target_id")
    if metadata_target and metadata_target != experiment.target.target_id:
        raise RuntimeError(
            f"Event metadata target is {metadata_target!r}; expected {experiment.target.target_id!r}"
        )
    metadata_segmentation = metadata.get("segmentation_id")
    if metadata_segmentation and metadata_segmentation != experiment.segmentation_id:
        raise RuntimeError(
            f"Event metadata segmentation is {metadata_segmentation!r}; "
            f"expected {experiment.segmentation_id!r}"
        )
    if frame.empty:
        raise RuntimeError(f"Target event file is empty: {path}")
    for column in ("event_start_second", "event_end_second", "horizon_seconds"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.int64)
    frame["threshold_bps"] = pd.to_numeric(frame["threshold_bps"], errors="raise").astype(float)
    frame = frame[
        frame["symbol"].astype(str).str.upper().eq(symbol)
        & frame["target_id"].astype(str).eq(experiment.target.target_id)
        & frame["segmentation_id"].astype(str).eq(experiment.segmentation_id)
    ].copy()
    if frame.empty:
        raise RuntimeError(f"No rows match experiment in {path}")
    expected_horizon = experiment.target.horizon_seconds
    if not frame["horizon_seconds"].eq(expected_horizon).all():
        raise RuntimeError(f"Event rows do not all use horizon {expected_horizon}")
    if not np.allclose(
        frame["threshold_bps"].to_numpy(dtype=float),
        experiment.target.threshold_bps,
        rtol=0.0,
        atol=1e-8,
    ):
        raise RuntimeError(
            f"Event rows do not all use threshold {experiment.target.threshold_bps}"
        )
    frame["event_start_date"] = frame["event_start_second"].map(CORE.seconds_to_date)
    frame = frame.sort_values("event_start_second", kind="mergesort").reset_index(drop=True)
    if frame["event_id"].duplicated().any() or frame["event_start_second"].duplicated().any():
        raise RuntimeError("Event IDs and event-start seconds must be unique")
    return frame


# ---------------------------------------------------------------------------
# Target-specific controls and observations
# ---------------------------------------------------------------------------


def load_target_control_candidates(
    label_partitions: Mapping[date, Any],
    threshold_bps: float,
    cadence_seconds: int,
    ready_seconds: np.ndarray,
    leads: Sequence[int],
    exclusion_starts: np.ndarray,
    exclusion_ends: np.ndarray,
    period_by_day: Mapping[date, str],
) -> pd.DataFrame:
    pq_module = CORE.require_arrow()[2]
    frames: list[pd.DataFrame] = []
    for _, partition in sorted(label_partitions.items()):
        day_frames: list[pd.DataFrame] = []
        for path in partition.files:
            parquet_file = pq_module.ParquetFile(path)
            try:
                frame = parquet_file.read(columns=list(REQUIRED_HORIZON_LABEL_COLUMNS)).to_pandas()
            finally:
                parquet_file.close()
            day_frames.append(frame)
        frame = pd.concat(day_frames, ignore_index=True) if len(day_frames) > 1 else day_frames[0]
        seconds = pd.to_numeric(frame["decision_second"], errors="raise").to_numpy(dtype=np.int64)
        swings = pd.to_numeric(frame["future_swing_bps"], errors="coerce").to_numpy(dtype=float)
        negative = np.isfinite(swings) & (swings + 1e-10 < float(threshold_bps))
        keep = negative & ((seconds % int(cadence_seconds)) == 0)
        if not np.any(keep):
            continue
        selected = seconds[keep]
        selected = selected[~CORE.interval_membership(selected, exclusion_starts, exclusion_ends)]
        if len(selected):
            selected = selected[CORE.valid_anchor_mask(selected, leads, ready_seconds)]
        if len(selected):
            days = [CORE.seconds_to_date(value) for value in selected]
            frames.append(
                pd.DataFrame(
                    {
                        "control_second": selected,
                        "control_date": days,
                        "control_hour_utc": [CORE.seconds_to_hour(value) for value in selected],
                        "control_period": [period_by_day.get(day, "UNKNOWN") for day in days],
                    }
                )
            )
    if not frames:
        raise RuntimeError("No eligible target-negative control anchors were found")
    output = pd.concat(frames, ignore_index=True)
    output = output.sort_values("control_second", kind="mergesort").reset_index(drop=True)
    if output["control_second"].duplicated().any():
        raise RuntimeError("Control candidate seconds are not unique")
    return output


def build_observation_requests(
    event_catalog: pd.DataFrame,
    matches: pd.DataFrame,
    leads: Sequence[int],
    experiment: Experiment,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    event_lookup = event_catalog.set_index("event_id", drop=False).to_dict("index")

    def common(event: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "target_id": experiment.target.target_id,
            "horizon_minutes": experiment.target.horizon_minutes,
            "horizon_seconds": experiment.target.horizon_seconds,
            "threshold_bps": experiment.target.threshold_bps,
            "segmentation_role": experiment.segmentation_role,
            "segmentation_id": experiment.segmentation_id,
            "event_context_cohort": str(event["event_context_cohort"]),
            "chronological_period": str(event["chronological_period"]),
            "maximum_future_swing_bps": float(event["maximum_future_swing_bps"]),
            "event_path_type": str(event["path_type"]),
        }

    for event in event_catalog.to_dict("records"):
        anchor = int(event["event_start_second"])
        for lead in leads:
            observation = anchor - int(lead)
            rows.append(
                {
                    **common(event),
                    "sample_kind": "EVENT",
                    "sample_id": str(event["event_id"]),
                    "source_event_id": str(event["event_id"]),
                    "control_rank": pd.NA,
                    "match_tier": pd.NA,
                    "match_tier_name": None,
                    "anchor_second": anchor,
                    "anchor_timestamp_utc": pd.to_datetime(anchor, unit="s", utc=True),
                    "anchor_date": CORE.seconds_to_date(anchor).isoformat(),
                    "anchor_hour_utc": CORE.seconds_to_hour(anchor),
                    "lead_seconds": int(lead),
                    "observation_second": observation,
                    "observation_timestamp_utc": pd.to_datetime(observation, unit="s", utc=True),
                }
            )
    for match in matches.to_dict("records"):
        event = event_lookup[str(match["event_id"])]
        anchor = int(match["control_second"])
        for lead in leads:
            observation = anchor - int(lead)
            rows.append(
                {
                    **common(event),
                    "sample_kind": "CONTROL",
                    "sample_id": str(match["control_id"]),
                    "source_event_id": str(match["event_id"]),
                    "control_rank": int(match["control_rank"]),
                    "match_tier": int(match["match_tier"]),
                    "match_tier_name": str(match["match_tier_name"]),
                    "anchor_second": anchor,
                    "anchor_timestamp_utc": pd.to_datetime(anchor, unit="s", utc=True),
                    "anchor_date": CORE.seconds_to_date(anchor).isoformat(),
                    "anchor_hour_utc": CORE.seconds_to_hour(anchor),
                    "lead_seconds": int(lead),
                    "observation_second": observation,
                    "observation_timestamp_utc": pd.to_datetime(observation, unit="s", utc=True),
                }
            )
    frame = pd.DataFrame(rows)
    frame["control_rank"] = frame["control_rank"].astype("Int64")
    frame["match_tier"] = frame["match_tier"].astype("Int64")
    return frame


def add_experiment_columns(frame: pd.DataFrame, experiment: Experiment) -> pd.DataFrame:
    output = frame.copy()
    values = {
        "target_id": experiment.target.target_id,
        "horizon_minutes": experiment.target.horizon_minutes,
        "horizon_seconds": experiment.target.horizon_seconds,
        "threshold_bps": experiment.target.threshold_bps,
        "segmentation_role": experiment.segmentation_role,
        "segmentation_id": experiment.segmentation_id,
    }
    for column, value in reversed(list(values.items())):
        if column not in output.columns:
            output.insert(0, column, value)
        else:
            output[column] = value
    return output


# ---------------------------------------------------------------------------
# Compact comparison scorecards
# ---------------------------------------------------------------------------


def lead_scorecard(
    stability: pd.DataFrame,
    event_summary: Mapping[str, Any],
    experiment: Experiment,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, scope_frame in stability.groupby("analysis_scope", sort=True):
        for lead, group in scope_frame.groupby("lead_seconds", sort=False):
            covered = group[
                (pd.to_numeric(group["event_coverage_pct"], errors="coerce") >= 80.0)
                & (pd.to_numeric(group["control_coverage_pct"], errors="coerce") >= 80.0)
            ].copy()
            absolute = pd.to_numeric(
                covered["overall_standardized_mean_difference"], errors="coerce"
            ).abs()
            # ``stability_profile`` is the canonical persisted profile and does not
            # include ``stable_effect_score``. That derived field is added only by
            # ``top_feature_profile``. Accept either schema so resumed runs and
            # freshly built runs use the same scorecard logic.
            if "stable_effect_score" in covered.columns:
                stable = pd.to_numeric(covered["stable_effect_score"], errors="coerce")
            else:
                required = {
                    "period_sign_consistent",
                    "periods_available",
                    "minimum_absolute_period_smd",
                }
                missing = sorted(required.difference(covered.columns))
                if missing:
                    raise RuntimeError(
                        "Lead scorecard cannot derive stable effects; missing columns: "
                        + ", ".join(missing)
                    )
                sign_consistent = covered["period_sign_consistent"].fillna(False).astype(bool)
                periods_available = pd.to_numeric(
                    covered["periods_available"], errors="coerce"
                )
                minimum_period_effect = pd.to_numeric(
                    covered["minimum_absolute_period_smd"], errors="coerce"
                )
                stable_mask = sign_consistent & periods_available.eq(3)
                stable = minimum_period_effect.where(stable_mask, 0.0).fillna(0.0)
            rows.append(
                {
                    "target_id": experiment.target.target_id,
                    "horizon_minutes": experiment.target.horizon_minutes,
                    "threshold_bps": experiment.target.threshold_bps,
                    "segmentation_role": experiment.segmentation_role,
                    "segmentation_id": experiment.segmentation_id,
                    "analysis_scope": str(scope),
                    "lead_seconds": int(lead),
                    "profiled_events": int(event_summary["profiled_events"]),
                    "isolated_onset_events": int(event_summary["isolated_onset_events"]),
                    "covered_feature_count": int(absolute.notna().sum()),
                    "median_absolute_smd": float(absolute.median()) if absolute.notna().any() else float("nan"),
                    "p90_absolute_smd": float(absolute.quantile(0.90)) if absolute.notna().any() else float("nan"),
                    "maximum_absolute_smd": float(absolute.max()) if absolute.notna().any() else float("nan"),
                    "stable_feature_count": int((stable > 0).sum()),
                    "stable_feature_count_abs_smd_ge_0_10": int(
                        ((stable > 0) & (absolute >= 0.10)).sum()
                    ),
                    "stable_feature_count_abs_smd_ge_0_20": int(
                        ((stable > 0) & (absolute >= 0.20)).sum()
                    ),
                    "maximum_stable_effect_score": float(stable.max()) if stable.notna().any() else 0.0,
                    "median_positive_stable_effect_score": float(stable[stable > 0].median())
                    if (stable > 0).any()
                    else 0.0,
                }
            )
    return pd.DataFrame(rows)


def primary_vs_robustness(scorecard: pd.DataFrame) -> pd.DataFrame:
    keys = ["target_id", "analysis_scope", "lead_seconds"]
    metrics = [
        "profiled_events",
        "isolated_onset_events",
        "median_absolute_smd",
        "p90_absolute_smd",
        "maximum_absolute_smd",
        "stable_feature_count",
        "stable_feature_count_abs_smd_ge_0_10",
        "stable_feature_count_abs_smd_ge_0_20",
        "maximum_stable_effect_score",
    ]
    primary = scorecard[scorecard["segmentation_role"] == "PRIMARY"][keys + metrics].copy()
    robustness = scorecard[scorecard["segmentation_role"] == "ROBUSTNESS"][keys + metrics].copy()
    primary = primary.rename(columns={column: f"primary_{column}" for column in metrics})
    robustness = robustness.rename(columns={column: f"robustness_{column}" for column in metrics})
    merged = primary.merge(robustness, on=keys, how="outer", validate="one_to_one")
    for metric in metrics:
        left = f"primary_{metric}"
        right = f"robustness_{metric}"
        merged[f"difference_{metric}"] = pd.to_numeric(merged[left], errors="coerce") - pd.to_numeric(
            merged[right], errors="coerce"
        )
    return merged.sort_values(keys, kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# SQLite experiment state and output validation
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
            summary_json TEXT,
            error_text TEXT
        );

        CREATE TABLE IF NOT EXISTS experiment_state (
            symbol TEXT NOT NULL,
            target_id TEXT NOT NULL,
            segmentation_id TEXT NOT NULL,
            segmentation_role TEXT NOT NULL,
            status TEXT NOT NULL,
            input_signature TEXT,
            config_signature TEXT,
            output_manifest_json TEXT,
            summary_json TEXT,
            error_text TEXT,
            updated_at_utc TEXT NOT NULL,
            PRIMARY KEY (symbol, target_id, segmentation_id)
        );
        """
    )
    connection.commit()
    return connection


def validate_manifest(manifest: Sequence[Mapping[str, Any]], input_signature: str) -> bool:
    pq_module = CORE.require_arrow()[2]
    try:
        for item in manifest:
            path = Path(str(item["path"]))
            if not path.exists():
                return False
            parquet_file = pq_module.ParquetFile(path)
            try:
                if int(parquet_file.metadata.num_rows) != int(item["rows"]):
                    return False
                metadata = CORE.decode_metadata(parquet_file.schema_arrow)
                if metadata.get("input_signature") != input_signature:
                    return False
            finally:
                parquet_file.close()
        return True
    except Exception:
        return False


def load_profile_outputs(
    output_root: Path, experiment: Experiment, symbol: str
) -> dict[str, pd.DataFrame]:
    pq_module = CORE.require_arrow()[2]
    output: dict[str, pd.DataFrame] = {}
    for name in PROFILE_DATASETS:
        path = profile_path(output_root, name, experiment, symbol)
        parquet_file = pq_module.ParquetFile(path)
        try:
            output[name] = parquet_file.read().to_pandas()
        finally:
            parquet_file.close()
    return output


def write_experiment_outputs(
    output_root: Path,
    symbol: str,
    experiment: Experiment,
    observations: pd.DataFrame,
    matches: pd.DataFrame,
    event_catalog: pd.DataFrame,
    profiles: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, Any],
    compression_level: int,
) -> list[dict[str, Any]]:
    datasets: list[tuple[Path, pd.DataFrame, str]] = [
        (experiment_path(output_root, "observations", experiment, symbol), observations, "event_and_control_observations"),
        (experiment_path(output_root, "matches", experiment, symbol), matches, "deterministic_control_matches"),
        (experiment_path(output_root, "event_catalog", experiment, symbol), event_catalog, "profiled_event_catalog"),
    ]
    for name, frame in profiles.items():
        datasets.append((profile_path(output_root, name, experiment, symbol), frame, f"profile_{name}"))
    manifest: list[dict[str, Any]] = []
    for path, frame, role in datasets:
        CORE.atomic_write_parquet(
            path,
            frame,
            {**metadata, "dataset_role": role},
            compression_level,
        )
        manifest.append({"path": str(path), "rows": len(frame)})
    return manifest


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def write_reports(
    report_root: Path,
    run_id: str,
    frames: Mapping[str, pd.DataFrame],
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    findings: str,
) -> None:
    run_root = report_root / "runs" / run_id
    report_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    for filename, frame in frames.items():
        CORE.atomic_write_csv(report_root / filename, frame)
        CORE.atomic_write_csv(run_root / filename, frame)
    for root in (report_root, run_root):
        CORE.atomic_write_json(root / "perps_target_footprint_summary.json", summary)
        CORE.atomic_write_json(root / "perps_target_footprint_config.json", config)
        CORE.atomic_write_text(root / "perps_target_footprint_findings.md", findings)


def markdown_findings(
    summary: Mapping[str, Any],
    scorecard: pd.DataFrame,
    experiment_summary: pd.DataFrame,
) -> str:
    lines = [
        "# Multi-Target Pre-Expansion Footprint Comparison",
        "",
        f"Run ID: `{summary['run_id']}`",
        "",
        "## Coverage",
        "",
        f"- Targets: **{summary['target_count']:,}**",
        f"- Experiments: **{summary['experiment_count']:,}**",
        f"- Rebuilt experiments: **{summary['rebuilt_experiments']:,}**",
        f"- Unchanged experiments: **{summary['skipped_experiments']:,}**",
        f"- Profiled causal features: **{summary['profiled_features']:,}**",
        f"- Failed invariants: **{summary['failed_invariants']:,}**",
        "",
        "## Experiment coverage",
        "",
        "| Target | Role | Segmentation | Events | Isolated onsets | Controls |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in experiment_summary.sort_values(
        ["horizon_minutes", "threshold_bps", "segmentation_role"], kind="mergesort"
    ).itertuples(index=False):
        lines.append(
            f"| `{row.target_id}` | {row.segmentation_role} | `{row.segmentation_id}` | "
            f"{int(row.profiled_events):,} | {int(row.isolated_onset_events):,} | "
            f"{int(row.matched_controls):,} |"
        )
    lines.extend(
        [
            "",
            "## Descriptive lead comparison",
            "",
            "The rows below use primary segmentations and isolated onsets. They rank descriptive feature separation only; they are not model performance and do not select an optimal horizon.",
            "",
            "| Lead | Target | Median | P90 | Stable features ≥0.10 | Stable features ≥0.20 |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    selected = scorecard[
        scorecard["segmentation_role"].eq("PRIMARY")
        & scorecard["analysis_scope"].eq("ISOLATED_ONSETS")
    ].copy()
    for lead in sorted(selected["lead_seconds"].dropna().astype(int).unique(), reverse=True):
        group = selected[selected["lead_seconds"] == lead].sort_values(
            ["p90_absolute_smd", "median_absolute_smd"],
            ascending=[False, False],
            kind="mergesort",
        )
        for row in group.itertuples(index=False):
            lines.append(
                f"| {lead}s | `{row.target_id}` | {float(row.median_absolute_smd):.3f} | "
                f"{float(row.p90_absolute_smd):.3f} | "
                f"{int(row.stable_feature_count_abs_smd_ge_0_10)} | "
                f"{int(row.stable_feature_count_abs_smd_ge_0_20)} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Every experiment uses the same causal feature-selection and matched-control methodology.",
            "- Controls are negative for the specific target being evaluated; they are not necessarily negative for every other target.",
            "- Primary and robustness segmentation results must agree before a target relationship is treated as durable.",
            "- A larger descriptive effect does not imply better calibration, event recall, warning time, false-alert rate or profitability.",
            "- Horizon selection remains deferred until identical chronological walk-forward models are evaluated.",
            "",
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def run_self_test() -> None:
    plans = parse_target_plan(DEFAULT_TARGET_PLAN)
    if len(plans) != 6:
        raise AssertionError(len(plans))
    if plans[3].target_id != "H15M_T050BPS":
        raise AssertionError(plans[3])
    if plans[3].primary_segmentation_id != "bridge_060s":
        raise AssertionError(plans[3])
    experiments = experiments_from_plan(plans, DEFAULT_SEGMENTATION_ROLES)
    if len(experiments) != 12:
        raise AssertionError(len(experiments))
    if max(DEFAULT_MINIMUM_CONTROL_EMBARGO_SECONDS, plans[-1].horizon_seconds) != 1200:
        raise AssertionError("effective embargo")

    # Exercise the canonical ``stability_profile`` schema, where
    # ``stable_effect_score`` has not yet been added by ``top_feature_profile``.
    synthetic = pd.DataFrame(
        {
            "analysis_scope": ["ISOLATED_ONSETS"] * 4,
            "lead_seconds": [60] * 4,
            "feature": ["a", "b", "c", "d"],
            "overall_standardized_mean_difference": [0.3, -0.2, 0.05, 0.4],
            "period_sign_consistent": [True, True, False, True],
            "periods_available": [3, 3, 3, 3],
            "minimum_absolute_period_smd": [0.2, 0.1, 0.4, 0.3],
            "event_coverage_pct": [100.0, 100.0, 100.0, 70.0],
            "control_coverage_pct": [100.0, 100.0, 100.0, 100.0],
        }
    )
    summary = {"profiled_events": 100, "isolated_onset_events": 60}
    card = lead_scorecard(synthetic, summary, experiments[0])
    row = card.iloc[0]
    if int(row["covered_feature_count"]) != 3:
        raise AssertionError(row)
    if int(row["stable_feature_count_abs_smd_ge_0_20"]) != 2:
        raise AssertionError(row)

    # Reuse the exact core interval/matching primitives in the test.
    starts, ends = CORE.merge_intervals(
        np.array([100, 110, 500], dtype=np.int64),
        np.array([105, 120, 510], dtype=np.int64),
    )
    if not np.array_equal(starts, np.array([100, 110, 500])):
        raise AssertionError((starts, ends))

    print("SELF-TEST PASS")
    print(
        "Validated target-plan parsing, frozen segmentation roles, target-specific embargo, "
        "cross-target scorecards and reused footprint-core primitives."
    )


# ---------------------------------------------------------------------------
# CLI and orchestration
# ---------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Compare causal pre-expansion footprints across frozen targets and segmentations."
    )
    value.add_argument(
        "--feature-root",
        default="./data/derived/perps/flow_features_1s_v1",
        help="Audited causal perps flow-feature root.",
    )
    value.add_argument(
        "--target-root",
        default="./data/derived/perps/research/expansion_target_segmentation_sensitivity_v1",
        help="Target/horizon label and event dataset produced by research script 04.",
    )
    value.add_argument(
        "--output-root",
        default="./data/derived/perps/research/pre_expansion_target_footprints_v1",
        help="Separate multi-target footprint dataset root.",
    )
    value.add_argument(
        "--report-root",
        default="./reports/perps_pre_expansion_target_footprints",
        help="Cross-target CSV, JSON and Markdown report root.",
    )
    value.add_argument("--symbol", default="BTCUSDT")
    value.add_argument(
        "--target-plan",
        default=DEFAULT_TARGET_PLAN,
        help="Quoted target plan TARGET=PRIMARY|ROBUSTNESS groups separated by semicolons.",
    )
    value.add_argument(
        "--segmentation-roles",
        default=",".join(DEFAULT_SEGMENTATION_ROLES),
        help="Comma-separated PRIMARY,ROBUSTNESS roles.",
    )
    value.add_argument(
        "--leads-seconds",
        default=",".join(str(item) for item in DEFAULT_LEADS_SECONDS),
        help="Comma-separated seconds before event/control anchor; must include 0.",
    )
    value.add_argument("--controls-per-event", type=int, default=DEFAULT_CONTROLS_PER_EVENT)
    value.add_argument(
        "--control-cadence-seconds", type=int, default=DEFAULT_CONTROL_CADENCE_SECONDS
    )
    value.add_argument(
        "--minimum-control-embargo-seconds",
        type=int,
        default=DEFAULT_MINIMUM_CONTROL_EMBARGO_SECONDS,
        help="Effective target embargo is max(this value, target horizon).",
    )
    value.add_argument(
        "--adjacent-hour-tolerance", type=int, default=DEFAULT_ADJACENT_HOUR_TOLERANCE
    )
    value.add_argument(
        "--feature-columns",
        help="Optional exact comma-separated causal numeric feature columns.",
    )
    value.add_argument("--start-date", help="Optional inclusive event-onset date YYYY-MM-DD.")
    value.add_argument("--end-date", help="Optional inclusive event-onset date YYYY-MM-DD.")
    value.add_argument(
        "--max-events-per-experiment",
        type=int,
        help="Optional event limit after date filtering; use only with isolated smoke outputs.",
    )
    value.add_argument(
        "--allow-partial-controls",
        action="store_true",
        help="Allow fewer controls than requested for diagnostic runs.",
    )
    value.add_argument("--hash-input-files", action="store_true")
    value.add_argument("--force", action="store_true")
    value.add_argument("--compression-level", type=int, default=6)
    value.add_argument("--self-test", action="store_true")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    CORE.require_arrow()

    if args.controls_per_event <= 0:
        raise SystemExit("--controls-per-event must be positive")
    if args.control_cadence_seconds <= 0:
        raise SystemExit("--control-cadence-seconds must be positive")
    if args.minimum_control_embargo_seconds < 0:
        raise SystemExit("--minimum-control-embargo-seconds cannot be negative")
    if not 0 <= args.adjacent_hour_tolerance <= 12:
        raise SystemExit("--adjacent-hour-tolerance must be between 0 and 12")
    if not 1 <= args.compression_level <= 22:
        raise SystemExit("--compression-level must be between 1 and 22")

    leads = CORE.parse_positive_ints(args.leads_seconds, "--leads-seconds", allow_zero=True)
    if 0 not in leads:
        raise SystemExit("--leads-seconds must include 0")
    plans = parse_target_plan(args.target_plan)
    roles = parse_roles(args.segmentation_roles)
    experiments = experiments_from_plan(plans, roles)
    start_date = CORE.parse_date(args.start_date, "--start-date")
    end_date = CORE.parse_date(args.end_date, "--end-date")
    if start_date and end_date and start_date > end_date:
        raise SystemExit("--start-date cannot be after --end-date")
    if args.max_events_per_experiment is not None and args.max_events_per_experiment <= 0:
        raise SystemExit("--max-events-per-experiment must be positive")

    symbol = str(args.symbol).strip().upper()
    feature_root = Path(args.feature_root).expanduser().resolve()
    target_root = Path(args.target_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    report_root = Path(args.report_root).expanduser().resolve()
    for root, label in ((feature_root, "Feature root"), (target_root, "Target root")):
        if not root.exists():
            raise SystemExit(f"{label} does not exist: {root}")

    output_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    run_id = CORE.utc_now().strftime("%Y%m%dT%H%M%SZ") + f"-{uuid.uuid4().hex[:8]}"
    state_path = output_root / "_build_state" / "build_state.sqlite"
    connection = connect_state(state_path)

    config: dict[str, Any] = {
        "profiler_version": PROFILER_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "feature_root": str(feature_root),
        "target_root": str(target_root),
        "output_root": str(output_root),
        "report_root": str(report_root),
        "symbol": symbol,
        "target_plan": [plan.__dict__ for plan in plans],
        "segmentation_roles": list(roles),
        "leads_seconds": list(leads),
        "controls_per_event": args.controls_per_event,
        "control_cadence_seconds": args.control_cadence_seconds,
        "minimum_control_embargo_seconds": args.minimum_control_embargo_seconds,
        "adjacent_hour_tolerance": args.adjacent_hour_tolerance,
        "feature_columns": args.feature_columns,
        "start_date": start_date.isoformat() if start_date else None,
        "end_date": end_date.isoformat() if end_date else None,
        "max_events_per_experiment": args.max_events_per_experiment,
        "allow_partial_controls": bool(args.allow_partial_controls),
        "hash_input_files": bool(args.hash_input_files),
    }

    connection.execute(
        "INSERT INTO runs (run_id, started_at_utc, status, configuration_json) VALUES (?, ?, 'RUNNING', ?)",
        (run_id, CORE.utc_now_iso(), CORE.compact_json(config)),
    )
    connection.commit()

    rebuilt = 0
    skipped = 0
    failed_invariants_total = 0
    experiment_summaries: list[dict[str, Any]] = []
    combined_profiles: dict[str, list[pd.DataFrame]] = {name: [] for name in PROFILE_DATASETS}
    scorecards: list[pd.DataFrame] = []

    try:
        print("Comparing causal pre-expansion footprints across targets...")
        print(f"Run ID             : {run_id}")
        print(f"Feature root       : {feature_root}")
        print(f"Target root        : {target_root}")
        print(f"Output root        : {output_root}")
        print(f"Report root        : {report_root}")
        print(f"Targets            : {len(plans):,}")
        print(f"Experiments        : {len(experiments):,}")
        print(f"Lead seconds       : {','.join(map(str, leads))}")

        feature_partitions = CORE.discover_date_partitions(
            feature_root, symbol, "features", args.hash_input_files
        )
        feature_schema_names = CORE.validate_feature_schema(feature_partitions, symbol)
        if args.feature_columns:
            feature_columns = sorted(
                {item.strip() for item in args.feature_columns.split(",") if item.strip()}
            )
        else:
            feature_columns = CORE.auto_feature_columns(feature_schema_names)
        if not feature_columns:
            raise RuntimeError("No causal numeric feature columns were selected")
        CORE.validate_numeric_feature_columns(
            next(iter(feature_partitions.values())).files[0], feature_columns
        )
        profile_features = [
            *feature_columns,
            "cvd_quote_change_from_earliest",
            "cvd_base_change_from_earliest",
        ]
        period_by_day = CORE.assign_chronological_periods(sorted(feature_partitions))
        print("Loading exact context-ready feature-second index...")
        ready_seconds = CORE.read_ready_seconds(feature_partitions)

        labels_by_horizon: dict[int, dict[date, Any]] = {}
        for horizon in sorted({plan.horizon_minutes for plan in plans}):
            labels_by_horizon[horizon] = discover_horizon_label_partitions(
                target_root, horizon, symbol, args.hash_input_files
            )

        feature_signature_payload = {
            day.isoformat(): item.signature for day, item in sorted(feature_partitions.items())
        }

        for index, experiment in enumerate(experiments, start=1):
            target = experiment.target
            effective_embargo = max(
                args.minimum_control_embargo_seconds, target.horizon_seconds
            )
            event_path = target_event_path(target_root, experiment, symbol)
            event_signature = CORE.file_signature(event_path, args.hash_input_files)
            label_partitions = labels_by_horizon[target.horizon_minutes]
            label_signature_payload = {
                day.isoformat(): item.signature for day, item in sorted(label_partitions.items())
            }
            experiment_config = {
                "profiler_version": PROFILER_VERSION,
                "target_id": target.target_id,
                "horizon_seconds": target.horizon_seconds,
                "threshold_bps": target.threshold_bps,
                "segmentation_role": experiment.segmentation_role,
                "segmentation_id": experiment.segmentation_id,
                "leads_seconds": list(leads),
                "controls_per_event": args.controls_per_event,
                "control_cadence_seconds": args.control_cadence_seconds,
                "effective_control_embargo_seconds": effective_embargo,
                "adjacent_hour_tolerance": args.adjacent_hour_tolerance,
                "feature_columns": feature_columns,
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
                "max_events_per_experiment": args.max_events_per_experiment,
                "allow_partial_controls": bool(args.allow_partial_controls),
            }
            config_signature = CORE.sha256_text(CORE.compact_json(experiment_config))
            input_signature = CORE.sha256_text(
                CORE.compact_json(
                    {
                        "event_file": event_signature,
                        "feature_partitions": feature_signature_payload,
                        "label_partitions": label_signature_payload,
                    }
                )
            )
            state = connection.execute(
                "SELECT * FROM experiment_state WHERE symbol=? AND target_id=? AND segmentation_id=?",
                (symbol, target.target_id, experiment.segmentation_id),
            ).fetchone()

            if state is not None and not args.force:
                manifest = json.loads(state["output_manifest_json"] or "[]")
                if (
                    state["status"] == "COMPLETED"
                    and state["input_signature"] == input_signature
                    and state["config_signature"] == config_signature
                    and validate_manifest(manifest, input_signature)
                ):
                    profiles = load_profile_outputs(output_root, experiment, symbol)
                    summary = json.loads(state["summary_json"])
                    for name, frame in profiles.items():
                        combined_profiles[name].append(frame)
                    scorecards.append(lead_scorecard(profiles["stability"], summary, experiment))
                    experiment_summaries.append(summary)
                    skipped += 1
                    print(
                        f"SKIP  [{index:>2}/{len(experiments)}] {target.target_id} "
                        f"{experiment.segmentation_role:<10} {experiment.segmentation_id}"
                    )
                    continue

            connection.execute(
                """
                INSERT INTO experiment_state (
                    symbol, target_id, segmentation_id, segmentation_role, status,
                    input_signature, config_signature, error_text, updated_at_utc
                ) VALUES (?, ?, ?, ?, 'RUNNING', ?, ?, NULL, ?)
                ON CONFLICT(symbol, target_id, segmentation_id) DO UPDATE SET
                    segmentation_role=excluded.segmentation_role,
                    status='RUNNING', input_signature=excluded.input_signature,
                    config_signature=excluded.config_signature, error_text=NULL,
                    updated_at_utc=excluded.updated_at_utc
                """,
                (
                    symbol,
                    target.target_id,
                    experiment.segmentation_id,
                    experiment.segmentation_role,
                    input_signature,
                    config_signature,
                    CORE.utc_now_iso(),
                ),
            )
            connection.commit()

            try:
                all_events = read_target_events(event_path, experiment, symbol)
                previous_end = all_events["event_end_second"].shift(1)
                all_events["previous_event_end_second"] = previous_end.astype("Int64")
                all_events["seconds_since_previous_event_recomputed"] = (
                    all_events["event_start_second"] - previous_end - 1
                ).astype("Int64")
                isolated = previous_end.isna() | (
                    all_events["event_start_second"] - max(leads)
                    > previous_end + effective_embargo
                )
                all_events["event_context_cohort"] = np.where(
                    isolated, "ISOLATED_ONSET", "RECENT_EVENT_CONTEXT"
                )
                all_events["chronological_period"] = all_events["event_start_date"].map(
                    period_by_day
                )
                if all_events["chronological_period"].isna().any():
                    raise RuntimeError("Some event dates fall outside feature coverage")

                events = all_events.copy()
                if start_date is not None:
                    events = events[events["event_start_date"] >= start_date]
                if end_date is not None:
                    events = events[events["event_start_date"] <= end_date]
                events = events.sort_values("event_start_second", kind="mergesort").reset_index(
                    drop=True
                )
                if args.max_events_per_experiment is not None:
                    events = events.iloc[: args.max_events_per_experiment].copy()
                if events.empty:
                    raise RuntimeError("No events remain after filtering")
                selected_event_count = len(events)

                event_anchor_seconds = events["event_start_second"].to_numpy(dtype=np.int64)
                ready_mask = CORE.valid_anchor_mask(event_anchor_seconds, leads, ready_seconds)
                omitted_events = events.loc[~ready_mask, ["event_id", "event_start_second"]]
                events = events.loc[ready_mask].reset_index(drop=True)
                if events.empty:
                    raise RuntimeError("No events have complete exact context-ready lead observations")

                exclusion_starts, exclusion_ends = CORE.exclusion_intervals(
                    all_events, max(leads), effective_embargo
                )
                candidates = load_target_control_candidates(
                    label_partitions,
                    target.threshold_bps,
                    args.control_cadence_seconds,
                    ready_seconds,
                    leads,
                    exclusion_starts,
                    exclusion_ends,
                    period_by_day,
                )
                matches, event_catalog = CORE.match_controls(
                    events,
                    candidates,
                    args.controls_per_event,
                    args.adjacent_hour_tolerance,
                    args.allow_partial_controls,
                )
                requests = build_observation_requests(event_catalog, matches, leads, experiment)
                feature_samples = CORE.read_feature_samples(
                    feature_partitions,
                    requests["observation_second"].astype(int).tolist(),
                    feature_columns,
                )
                observations = CORE.attach_features(requests, feature_samples, max(leads))

                profiles = {
                    "feature_comparison": CORE.comparison_profile(observations, profile_features),
                    "matched_effects": CORE.matched_effect_profile(observations, profile_features),
                    "trajectory": CORE.trajectory_profile(observations, profile_features, max(leads)),
                    "chronological_effects": CORE.chronological_effect_profile(
                        observations, profile_features
                    ),
                }
                profiles["stability"] = CORE.stability_profile(
                    profiles["feature_comparison"], profiles["chronological_effects"]
                )
                profiles["top_features"] = CORE.top_feature_profile(
                    profiles["stability"], profiles["matched_effects"]
                )
                profiles["control_match_quality"] = CORE.match_quality_profile(matches)
                profiles["coverage"] = CORE.coverage_profile(
                    events,
                    event_catalog,
                    candidates,
                    matches,
                    observations,
                    profile_features,
                )
                profiles["invariants"] = CORE.invariant_frame(
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
                failed_invariants = int((profiles["invariants"]["status"] == "FAIL").sum())
                if failed_invariants:
                    names = profiles["invariants"].loc[
                        profiles["invariants"]["status"] == "FAIL", "invariant"
                    ].tolist()
                    raise RuntimeError("Footprint invariants failed: " + ", ".join(names))

                for name in PROFILE_DATASETS:
                    profiles[name] = add_experiment_columns(profiles[name], experiment)
                observations = add_experiment_columns(observations, experiment)
                matches = add_experiment_columns(matches, experiment)
                event_catalog = add_experiment_columns(event_catalog, experiment)

                summary = {
                    "status": "COMPLETED",
                    "run_id": run_id,
                    "profiler_version": PROFILER_VERSION,
                    "symbol": symbol,
                    "target_id": target.target_id,
                    "horizon_minutes": target.horizon_minutes,
                    "horizon_seconds": target.horizon_seconds,
                    "threshold_bps": target.threshold_bps,
                    "segmentation_role": experiment.segmentation_role,
                    "segmentation_id": experiment.segmentation_id,
                    "effective_control_embargo_seconds": effective_embargo,
                    "selected_events": selected_event_count,
                    "profiled_events": len(event_catalog),
                    "isolated_onset_events": int(
                        event_catalog["event_context_cohort"].astype(str).eq("ISOLATED_ONSET").sum()
                    ),
                    "recent_event_context_events": int(
                        event_catalog["event_context_cohort"]
                        .astype(str)
                        .eq("RECENT_EVENT_CONTEXT")
                        .sum()
                    ),
                    "omitted_events_missing_exact_ready_leads": len(omitted_events),
                    "eligible_control_candidates": len(candidates),
                    "matched_controls": len(matches),
                    "observation_rows": len(observations),
                    "profiled_features": len(profile_features),
                    "failed_invariants": failed_invariants,
                    "input_signature": input_signature,
                    "config_signature": config_signature,
                }
                metadata = {
                    "multi_target_footprint_profiler_version": PROFILER_VERSION,
                    "multi_target_footprint_schema_version": OUTPUT_SCHEMA_VERSION,
                    "symbol": symbol,
                    "target_id": target.target_id,
                    "horizon_seconds": target.horizon_seconds,
                    "threshold_bps": target.threshold_bps,
                    "segmentation_role": experiment.segmentation_role,
                    "segmentation_id": experiment.segmentation_id,
                    "source_target_profiler_version": EXPECTED_TARGET_PROFILER_VERSION,
                    "source_feature_builder_version": CORE.EXPECTED_FEATURE_BUILDER_VERSION,
                    "input_signature": input_signature,
                    "config_signature": config_signature,
                    "created_at_utc": CORE.utc_now_iso(),
                }
                manifest = write_experiment_outputs(
                    output_root,
                    symbol,
                    experiment,
                    observations,
                    matches,
                    event_catalog,
                    profiles,
                    metadata,
                    args.compression_level,
                )
                connection.execute(
                    """
                    UPDATE experiment_state SET
                        status='COMPLETED', output_manifest_json=?, summary_json=?,
                        error_text=NULL, updated_at_utc=?
                    WHERE symbol=? AND target_id=? AND segmentation_id=?
                    """,
                    (
                        CORE.compact_json(manifest),
                        CORE.compact_json(summary),
                        CORE.utc_now_iso(),
                        symbol,
                        target.target_id,
                        experiment.segmentation_id,
                    ),
                )
                connection.commit()

                for name, frame in profiles.items():
                    combined_profiles[name].append(frame)
                scorecards.append(lead_scorecard(profiles["stability"], summary, experiment))
                experiment_summaries.append(summary)
                rebuilt += 1
                print(
                    f"BUILD [{index:>2}/{len(experiments)}] {target.target_id} "
                    f"{experiment.segmentation_role:<10} {experiment.segmentation_id} "
                    f"events={len(event_catalog):,} controls={len(matches):,}"
                )
            except Exception:
                error_text = traceback.format_exc()
                connection.execute(
                    """
                    UPDATE experiment_state SET status='FAILED', error_text=?, updated_at_utc=?
                    WHERE symbol=? AND target_id=? AND segmentation_id=?
                    """,
                    (
                        error_text,
                        CORE.utc_now_iso(),
                        symbol,
                        target.target_id,
                        experiment.segmentation_id,
                    ),
                )
                connection.commit()
                raise

        combined = {
            name: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            for name, frames in combined_profiles.items()
        }
        experiment_summary = pd.DataFrame(experiment_summaries).sort_values(
            ["horizon_minutes", "threshold_bps", "segmentation_role"], kind="mergesort"
        )
        scorecard = pd.concat(scorecards, ignore_index=True).sort_values(
            ["analysis_scope", "lead_seconds", "horizon_minutes", "threshold_bps", "segmentation_role"],
            ascending=[True, False, True, True, True],
            kind="mergesort",
        )
        robustness = primary_vs_robustness(scorecard) if set(roles) == set(DEFAULT_SEGMENTATION_ROLES) else pd.DataFrame()
        failed_invariants_total = int(
            (combined["invariants"]["status"].astype(str) == "FAIL").sum()
        )
        if failed_invariants_total:
            raise RuntimeError(f"Combined invariant report has {failed_invariants_total} failures")

        summary = {
            "status": "COMPLETED",
            "run_id": run_id,
            "profiler_version": PROFILER_VERSION,
            "created_at_utc": CORE.utc_now_iso(),
            "symbol": symbol,
            "target_count": len(plans),
            "experiment_count": len(experiments),
            "rebuilt_experiments": rebuilt,
            "skipped_experiments": skipped,
            "profiled_features": len(profile_features),
            "failed_invariants": failed_invariants_total,
            "total_profiled_events_across_experiments": int(
                experiment_summary["profiled_events"].sum()
            ),
            "total_isolated_onsets_across_experiments": int(
                experiment_summary["isolated_onset_events"].sum()
            ),
            "total_matched_controls_across_experiments": int(
                experiment_summary["matched_controls"].sum()
            ),
            "output_root": str(output_root),
            "report_root": str(report_root),
        }
        findings = markdown_findings(summary, scorecard, experiment_summary)
        report_frames = {
            "perps_target_footprint_experiment_summary.csv": experiment_summary,
            "perps_target_footprint_feature_comparison.csv": combined["feature_comparison"],
            "perps_target_footprint_matched_effects.csv": combined["matched_effects"],
            "perps_target_footprint_trajectory.csv": combined["trajectory"],
            "perps_target_footprint_chronological_effects.csv": combined[
                "chronological_effects"
            ],
            "perps_target_footprint_stability.csv": combined["stability"],
            "perps_target_footprint_top_features.csv": combined["top_features"],
            "perps_target_footprint_lead_scorecard.csv": scorecard,
            "perps_target_footprint_primary_vs_robustness.csv": robustness,
            "perps_target_footprint_control_match_quality.csv": combined[
                "control_match_quality"
            ],
            "perps_target_footprint_coverage.csv": combined["coverage"],
            "perps_target_footprint_invariants.csv": combined["invariants"],
        }
        write_reports(report_root, run_id, report_frames, summary, config, findings)
        CORE.atomic_write_json(output_root / "_metadata" / "dataset_config.json", config)
        CORE.atomic_write_json(output_root / "_metadata" / "latest_run_summary.json", summary)

        connection.execute(
            "UPDATE runs SET completed_at_utc=?, status='COMPLETED', summary_json=? WHERE run_id=?",
            (CORE.utc_now_iso(), CORE.compact_json(summary), run_id),
        )
        connection.commit()

        print("COMPLETED")
        print(f"Targets             : {len(plans):,}")
        print(f"Experiments         : {len(experiments):,}")
        print(f"Rebuilt             : {rebuilt:,}")
        print(f"Skipped unchanged   : {skipped:,}")
        print(f"Profiled features   : {len(profile_features):,}")
        print(f"Failed invariants   : {failed_invariants_total:,}")
        print(f"Findings            : {report_root / 'perps_target_footprint_findings.md'}")
        return 0

    except Exception as exc:
        error_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        try:
            connection.execute(
                "UPDATE runs SET completed_at_utc=?, status='FAILED', error_text=? WHERE run_id=?",
                (CORE.utc_now_iso(), error_text, run_id),
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
