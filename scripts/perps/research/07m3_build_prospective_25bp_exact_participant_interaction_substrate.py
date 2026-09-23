#!/usr/bin/env python3
"""
Phase 7M.3 — Prospective 25-bp exact participant-interaction substrate
=====================================================================

PLAIN-ENGLISH PURPOSE
---------------------
Phase 7M.2 established a genuinely prospective 25-bp candidate population:

    5-bp reset
        -> 10-bp release
        -> 25-bp candidate
        -> 60-second same-direction re-arm

and, crucially, that population includes attempts that later fail before 35 bps.

Phase 7M.2 also showed that unusually strong opposing-side withdrawal during
candidate+1..+3 seconds remains associated with 75-bp continuation after:
    * adding failed-before-35 attempts,
    * controlling for immediate momentum and aggressive pressure,
    * passing discovery and chronological validation.

Phase 7M.3 therefore earns the expensive test:

    At a LIVE prospective 25-bp candidate, what is the exact queue-level
    mechanism over the next 1 / 3 / 5 / 10 seconds?

This phase is a SUBSTRATE BUILDER only.  It performs the expensive raw replay
once and saves reusable causal mechanics.  It does NOT test which exact feature
predicts future 35/50/75/100 continuation.  That is intentionally deferred to
a cheap downstream Phase 7M.4 analysis.

WHY REUSE PHASE 7I.4?
---------------------
Phase 7I.4 v1.0.1 is the already-audited exact depth/trade replay engine used by
Phase 7L.1.  Reimplementing sequence-sensitive Binance book reconstruction here
would create unnecessary scientific and engineering risk.

This wrapper:
    1. validates frozen Phase 7M.2 v1.0.0;
    2. selects ONLY the independent prospective 25-bp policy;
    3. validates the exact 5 / 10 / 25-bp + 60s causal contract;
    4. freezes the 25-bp candidate-ID set;
    5. validates frozen Phase 7I.4 v1.0.1 and its replay interface;
    6. supplies the 25-bp candidates to that engine without regenerating them;
    7. replays exact authoritative depth/trades at candidate_second;
    8. preserves the existing atomic per-candidate-date checkpoint system;
    9. adds durable, human-visible progress/heartbeat reporting;
   10. normalizes legacy 35-bp interface labels in final aggregate outputs.

POPULATION
----------
Primary prospective policy:
    EARLY_25BP_RESET5_RELEASE10_REARM60

Expected full population from reviewed Phase 7M.2:
    7,067 candidates
    62 candidate dates

The replay MUST include:
    25 -> fails before 35
    25 -> reaches 35 but stalls
    25 -> reaches 50
    25 -> reaches 75
    25 -> reaches 100

No future outcome is allowed to filter this replay population.

EXACT MECHANICS
---------------
The frozen Phase 7I.4 engine measures the same audited families used at 35 bps:

BOOK / QUEUE RESPONSE
    * exact initial SUPPORT / OPPOSING frozen price levels within 5 bps
    * exact same-price queue depletion
    * retained initial liquidity
    * durable unrecovered initial-queue loss
    * peak initial queue loss
    * same-price refill after depletion
    * durable refill fraction
    * refill latency
    * near-touch restoration / noninitial additions
    * support-vs-opposing asymmetry

EXECUTED TRADES
    * directional / counter-direction aggressive quote
    * aggressive imbalance
    * trade counts
    * average / maximum aggressive trade quote
    * trade concentration / HHI
    * pre-window versus forward flow acceleration

INTERACTIONS
    * queue depletion relative to directional aggression
    * unrecovered loss per unit directional aggression
    * refill/restoration under pressure
    * price response per unit net aggressive flow

Observation horizons:
    1 / 3 / 5 / 10 seconds
Primary downstream horizon:
    3 seconds

CAUSAL / FUTURE SEPARATION
--------------------------
CAUSAL:
    candidate_inputs.parquet
    candidate_mechanics.parquet

FUTURE LABELS — physically separate:
    candidate_decision_outcomes.parquet
    candidate_7m2_outcomes.parquet

The expensive substrate itself makes NO continuation-effect claim.

RUNTIME VISIBILITY
------------------
This phase may be expensive, so progress is intentionally visible.

During each candidate date the console emits:

    [7M.3 START]
    [7M.3 HEARTBEAT] every --heartbeat-sec (default 30s)
    [7M.3 PROGRESS] when the date finishes

The heartbeat shows:
    * current date and date index
    * candidates on that date
    * completed dates / candidates
    * cumulative windows / valid windows
    * current-date elapsed seconds
    * total invocation elapsed seconds

It deliberately does NOT fabricate an ETA.

Durable live progress is written atomically to:

    data/derived/perps/research/
        prospective_25bp_exact_participant_interaction_v1/
        metadata/progress.json

and a per-date ledger is written to:

    reports/perps_prospective_25bp_exact_participant_interaction/
        replay_progress.csv

The Phase 7I.4 date checkpoints remain authoritative for resumability.
If the run is interrupted, rerun the same command.  Matching completed date
checkpoints are reused.

OUTPUTS
-------
data/derived/perps/research/prospective_25bp_exact_participant_interaction_v1/
    candidate_inputs.parquet
    candidate_mechanics.parquet
    candidate_decision_outcomes.parquet
    candidate_7m2_outcomes.parquet
    candidate_windows/candidate_date=.../part.parquet
    level_audit/candidate_date=.../part.parquet
    decision_outcomes/candidate_date=.../part.parquet
    metadata/progress.json
    metadata/latest_summary.json
    metadata/run_config.json
    metadata/frozen_candidate_manifest.json

reports/perps_prospective_25bp_exact_participant_interaction/
    replay_progress.csv
    prospective_25bp_exact_replay_contract.csv
    prospective_25bp_exact_replay_invariants.csv
    participant_interaction_coverage.csv
    participant_interaction_data_quality_by_day.csv
    participant_interaction_schema.csv
    participant_interaction_invariants.csv
    participant_interaction_findings.md

IMPORTANT INTERPRETATION LIMITS
-------------------------------
* This is historical evaluation over a rich window we have already studied.
  It is prospective in candidate construction, not an untouched future holdout.
* Exact quantity loss is observed; cancellation/order identity is not.
* Same-price refill does not identify the participant/order providing it.
* No composite failure score is built.
* No ML is fit.
* No profitability claim is made.
* Phase 7M.4 must compare exact mechanics against outcomes using controls and
  predeclared statistical rules; do not infer that result from this substrate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

BUILDER_VERSION = "1.0.0"
SCHEMA_VERSION = 1
SCHEMA_NAME = "prospective_25bp_exact_participant_interaction_v1"

FROZEN_PHASE7M2_VERSION = "1.0.0"
FROZEN_PHASE7M2_SCHEMA = "prospective_early_candidate_audit_v1"
FROZEN_PHASE7I4_VERSION = "1.0.1"

FROZEN_POLICY = "EARLY_25BP_RESET5_RELEASE10_REARM60"
EXPECTED_CANDIDATES = 7_067
EXPECTED_CANDIDATE_DAYS = 62

FROZEN_RESET_BPS = 5.0
FROZEN_RELEASE_BPS = 10.0
FROZEN_CANDIDATE_BPS = 25.0
FROZEN_COOLDOWN_SEC = 60

FIXED_HORIZONS_SEC = (1, 3, 5, 10)
PRIMARY_HORIZON_SEC = 3

PRICE_SCALE = 10.0
MAX_BOOK_AGE_MS = 2_000
DEFAULT_HEARTBEAT_SEC = 30.0

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = (
    SCRIPT_PATH.parents[3]
    if len(SCRIPT_PATH.parents) >= 4
    else Path.cwd()
)

DEFAULT_PHASE7I4_SCRIPT = (
    REPO_ROOT / "scripts" / "perps" / "research" /
    "07i4_build_causal_candidate_participant_interaction_dataset.py"
)
DEFAULT_PHASE6A_SCRIPT = (
    REPO_ROOT / "scripts" / "perps" / "research" /
    "06a_build_execution_foundation.py"
)
DEFAULT_PHASE7A_SCRIPT = (
    REPO_ROOT / "scripts" / "perps" / "research" /
    "07a_build_large_move_event_catalog.py"
)
DEFAULT_PHASE7E_SCRIPT = (
    REPO_ROOT / "scripts" / "perps" / "research" /
    "07e_build_raw_depth_continuation_mechanics.py"
)
DEFAULT_PHASE7I1_SCRIPT = (
    REPO_ROOT / "scripts" / "perps" / "research" /
    "07i1_build_causal_large_move_candidate_catalog.py"
)

DEFAULT_CANONICAL_ROOT = (
    REPO_ROOT / "data" / "canonical" / "market_events_v1"
)
DEFAULT_EXECUTION_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" /
    "execution_foundation_v1"
)
DEFAULT_PHASE7M2_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" /
    FROZEN_PHASE7M2_SCHEMA
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" /
    SCHEMA_NAME
)
DEFAULT_REPORT_ROOT = (
    REPO_ROOT / "reports" /
    "perps_prospective_25bp_exact_participant_interaction"
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat()


def safe_float(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        return math.nan
    return x if math.isfinite(x) else math.nan


def require_pandas():
    try:
        import pandas as pd
    except Exception as exc:
        raise SystemExit(
            "Phase 7M.3 requires pandas in the project .venv."
        ) from exc
    return pd


def require_stack():
    pd = require_pandas()
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise SystemExit(
            "Phase 7M.3 requires pyarrow in the project .venv."
        ) from exc
    return pd, pq


def load_module(path: Path, name: str):
    if not path.exists():
        raise RuntimeError(f"Missing required script: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix="tmp7m3-",
        suffix=path.suffix,
        dir=str(path.parent),
    )
    os.close(fd)
    try:
        Path(tmp).write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(
        path,
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=str,
        ) + "\n",
    )


def atomic_csv(frame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix="tmp7m3-",
        suffix=".csv",
        dir=str(path.parent),
    )
    os.close(fd)
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def candidate_hash(frame) -> str:
    ids = sorted(frame["candidate_id"].astype(str).tolist())
    return hashlib.sha256(
        "\n".join(ids).encode("utf-8")
    ).hexdigest()


def unique_numeric(frame, column: str) -> list[float]:
    pd = require_pandas()
    return sorted(
        set(
            pd.to_numeric(
                frame[column],
                errors="coerce",
            )
            .dropna()
            .astype(float)
            .tolist()
        )
    )


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {
        "true", "1", "yes", "y"
    }


def validate_and_adapt_25bp_candidates(
    all_candidates,
    summary: dict[str, Any],
    *,
    full_scope_required: bool = True,
):
    """
    Freeze ONLY the causal prospective 25-bp population.

    No future label is used to select a candidate.
    """
    pd = require_pandas()

    if str(summary.get("builder_version")) != FROZEN_PHASE7M2_VERSION:
        raise RuntimeError(
            f"Phase 7M.3 requires Phase 7M.2 "
            f"v{FROZEN_PHASE7M2_VERSION}; found "
            f"{summary.get('builder_version')!r}"
        )
    if str(summary.get("schema_name")) != FROZEN_PHASE7M2_SCHEMA:
        raise RuntimeError(
            f"Phase 7M.2 schema mismatch: "
            f"{summary.get('schema_name')!r}"
        )
    if str(summary.get("status")) != "PASS":
        raise RuntimeError(
            "Phase 7M.2 latest_summary must be PASS; "
            f"found {summary.get('status')!r}"
        )

    required_summary = {
        "primary_threshold_bps": 25,
        "reset_bps": FROZEN_RESET_BPS,
        "release_bps": FROZEN_RELEASE_BPS,
        "rearm_sec": FROZEN_COOLDOWN_SEC,
    }
    for key, expected in required_summary.items():
        observed = summary.get(key)
        if safe_float(observed) != safe_float(expected):
            raise RuntimeError(
                f"Phase 7M.2 summary contract mismatch for {key}: "
                f"observed={observed!r}, expected={expected!r}"
            )

    if not _bool_value(
        summary.get(
            "prospective_population_includes_failed_before35",
            False,
        )
    ):
        raise RuntimeError(
            "Phase 7M.2 does not declare that the prospective "
            "population includes failed-before-35 attempts."
        )

    count_map = summary.get("candidate_count_by_threshold", {})
    expected_25 = None
    if isinstance(count_map, dict):
        expected_25 = count_map.get("25", count_map.get(25))
    if (
        full_scope_required
        and int(expected_25 or -1) != EXPECTED_CANDIDATES
    ):
        raise RuntimeError(
            "Phase 7M.2 metadata does not confirm the reviewed "
            f"{EXPECTED_CANDIDATES:,}-candidate 25-bp population; "
            f"found {expected_25!r}"
        )

    required = {
        "candidate_id",
        "policy_id",
        "candidate_date",
        "direction",
        "direction_label",
        "origin_second",
        "origin_price",
        "release_second",
        "candidate_second",
        "candidate_price",
        "candidate_displacement_bps",
        "reset_threshold_bps",
        "release_threshold_bps",
        "candidate_threshold_bps",
        "cooldown_sec",
        "early_threshold_bps",
    }
    missing = required - set(all_candidates.columns)
    if missing:
        raise RuntimeError(
            "Phase 7M.2 early candidate table is missing required "
            f"causal fields: {sorted(missing)}"
        )

    selected = all_candidates[
        (all_candidates["policy_id"].astype(str) == FROZEN_POLICY)
        & (
            pd.to_numeric(
                all_candidates["early_threshold_bps"],
                errors="coerce",
            ) == FROZEN_CANDIDATE_BPS
        )
    ].copy()

    if selected.empty:
        raise RuntimeError(
            f"No candidates found for {FROZEN_POLICY}"
        )
    if selected["candidate_id"].duplicated().any():
        raise RuntimeError(
            "Selected 25-bp candidates contain duplicate candidate_id."
        )

    observed_contract = {
        "reset_bps": unique_numeric(
            selected, "reset_threshold_bps"
        ),
        "release_bps": unique_numeric(
            selected, "release_threshold_bps"
        ),
        "candidate_bps": unique_numeric(
            selected, "candidate_threshold_bps"
        ),
        "early_threshold_bps": unique_numeric(
            selected, "early_threshold_bps"
        ),
        "cooldown_sec": unique_numeric(
            selected, "cooldown_sec"
        ),
    }
    expected_contract = {
        "reset_bps": [FROZEN_RESET_BPS],
        "release_bps": [FROZEN_RELEASE_BPS],
        "candidate_bps": [FROZEN_CANDIDATE_BPS],
        "early_threshold_bps": [FROZEN_CANDIDATE_BPS],
        "cooldown_sec": [float(FROZEN_COOLDOWN_SEC)],
    }
    if observed_contract != expected_contract:
        raise RuntimeError(
            "Prospective 25-bp candidate mechanics disagree with "
            "the approved contract.\n"
            f"Observed: {observed_contract}\n"
            f"Expected: {expected_contract}"
        )

    days = int(
        selected["candidate_date"].astype(str).nunique()
    )
    if full_scope_required:
        if len(selected) != EXPECTED_CANDIDATES:
            raise RuntimeError(
                f"Approved Phase 7M.2 25-bp universe requires "
                f"{EXPECTED_CANDIDATES:,} candidates; "
                f"found {len(selected):,}"
            )
        if days != EXPECTED_CANDIDATE_DAYS:
            raise RuntimeError(
                f"Approved Phase 7M.2 25-bp universe requires "
                f"{EXPECTED_CANDIDATE_DAYS} candidate dates; "
                f"found {days}"
            )

    # Whitelist only fields that are causal/current-state metadata.
    # This prevents accidental carry-over if Phase 7M.2 later gains more
    # outcome/report convenience columns.
    causal_keep = [
        "candidate_id",
        "policy_id",
        "candidate_date",
        "direction",
        "direction_label",
        "origin_second",
        "origin_price",
        "release_second",
        "candidate_second",
        "candidate_price",
        "candidate_displacement_bps",
        "reset_threshold_bps",
        "release_threshold_bps",
        "candidate_threshold_bps",
        "cooldown_sec",
        "early_threshold_bps",
    ]
    for optional in (
        "origin_kind",
        "is_primary_early_threshold",
        "is_reaccel_context",
        "release_to_candidate_sec",
        "seconds_since_prior_same_direction_candidate",
    ):
        if optional in selected.columns:
            causal_keep.append(optional)

    selected = selected[causal_keep].copy()

    # The frozen 7I.4 replay engine needs an identifier called episode_id.
    # It is only an interface identifier here; it does NOT reintroduce the old
    # one-candidate directional episode ontology.
    selected["episode_id"] = (
        "PROSPECTIVE_25BP::"
        + selected["candidate_id"].astype(str)
    )

    if "release_to_candidate_sec" in selected.columns:
        rtc = pd.to_numeric(
            selected["release_to_candidate_sec"],
            errors="coerce",
        )
    else:
        rtc = pd.Series(
            math.nan,
            index=selected.index,
            dtype=float,
        )
    derived_rtc = (
        pd.to_numeric(
            selected["candidate_second"],
            errors="coerce",
        )
        - pd.to_numeric(
            selected["release_second"],
            errors="coerce",
        )
    )
    selected["seconds_release_to_candidate"] = (
        rtc.where(rtc.notna(), derived_rtc)
        .astype("int64")
    )

    # Geometry/order checks.
    origin = pd.to_numeric(
        selected["origin_second"], errors="coerce"
    )
    release = pd.to_numeric(
        selected["release_second"], errors="coerce"
    )
    candidate = pd.to_numeric(
        selected["candidate_second"], errors="coerce"
    )
    disp = pd.to_numeric(
        selected["candidate_displacement_bps"],
        errors="coerce",
    )

    if not bool((origin <= release).all()):
        raise RuntimeError(
            "25-bp origin_second must be <= release_second."
        )
    if not bool((release <= candidate).all()):
        raise RuntimeError(
            "25-bp release_second must be <= candidate_second."
        )
    if not bool(
        (disp + 1e-9 >= FROZEN_CANDIDATE_BPS).all()
    ):
        raise RuntimeError(
            "25-bp candidate displacement is below the frozen threshold."
        )

    leakage_tokens = (
        "reach_",
        "future",
        "outcome",
        "mfe_",
        "mae_",
        "seconds_to_35",
        "seconds_to_50",
        "seconds_to_75",
        "seconds_to_100",
    )
    leakage_cols = [
        c
        for c in selected.columns
        if any(tok in c.lower() for tok in leakage_tokens)
    ]
    if leakage_cols:
        raise RuntimeError(
            "Future-label names leaked into the frozen 25-bp causal "
            f"candidate interface: {leakage_cols}"
        )

    selected = (
        selected.sort_values(
            ["candidate_second", "candidate_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    return selected, observed_contract


def load_phase7m2_freeze(root: Path):
    pd, pq = require_stack()

    candidates_path = root / "early_candidates.parquet"
    outcomes_path = root / "early_candidate_outcomes.parquet"
    summary_path = root / "metadata" / "latest_summary.json"

    for path in (
        candidates_path,
        outcomes_path,
        summary_path,
    ):
        if not path.exists():
            raise RuntimeError(
                f"Missing frozen Phase 7M.2 artifact: {path}"
            )

    all_candidates = pq.read_table(
        candidates_path
    ).to_pandas()
    all_outcomes = pq.read_table(
        outcomes_path
    ).to_pandas()
    summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )

    selected, contract = validate_and_adapt_25bp_candidates(
        all_candidates,
        summary,
        full_scope_required=True,
    )
    ids = set(selected["candidate_id"].astype(str))

    if "early_threshold_bps" in all_outcomes.columns:
        selected_outcomes = all_outcomes[
            (
                all_outcomes["candidate_id"]
                .astype(str)
                .isin(ids)
            )
            & (
                pd.to_numeric(
                    all_outcomes["early_threshold_bps"],
                    errors="coerce",
                )
                == FROZEN_CANDIDATE_BPS
            )
        ].copy()
    else:
        selected_outcomes = all_outcomes[
            all_outcomes["candidate_id"]
            .astype(str)
            .isin(ids)
        ].copy()

    if selected_outcomes["candidate_id"].duplicated().any():
        raise RuntimeError(
            "Phase 7M.2 selected outcome table is not one-to-one."
        )
    outcome_ids = set(
        selected_outcomes["candidate_id"].astype(str)
    )
    if outcome_ids != ids:
        missing = ids - outcome_ids
        extra = outcome_ids - ids
        raise RuntimeError(
            "Phase 7M.2 selected outcome coverage mismatch: "
            f"missing={len(missing)}, extra={len(extra)}"
        )

    h = candidate_hash(selected)

    freeze_summary = {
        "builder_version": FROZEN_PHASE7M2_VERSION,
        "schema_name": FROZEN_PHASE7M2_SCHEMA,
        "status": "PASS",
        "selected_policy": FROZEN_POLICY,
        "candidate_threshold_bps": FROZEN_CANDIDATE_BPS,
        "candidate_count": int(len(selected)),
        "candidate_days": int(
            selected["candidate_date"]
            .astype(str)
            .nunique()
        ),
        "candidate_id_set_sha256": h,
        "prospective_population_includes_failed_before35": True,
        "contract": contract,
    }
    freeze_manifest = {
        "freeze_status": (
            "FROZEN_FOR_PHASE_7M3_EXACT_25BP_REPLAY"
        ),
        **freeze_summary,
    }
    return (
        selected,
        selected_outcomes,
        summary,
        freeze_summary,
        freeze_manifest,
        h,
    )


def validate_7i4_replay_interface(engine) -> None:
    required_callables = (
        "main",
        "load_frozen_candidates",
        "process_candidate_date",
        "build_decision_outcomes",
        "build_coverage",
        "build_day_quality",
        "build_schema_report",
        "check_invariants",
        "write_parquet_atomic",
    )
    missing_callables = [
        name
        for name in required_callables
        if not callable(getattr(engine, name, None))
    ]

    required_constants = {
        "FIXED_HORIZONS_SEC": FIXED_HORIZONS_SEC,
        "PRIMARY_HORIZON_SEC": PRIMARY_HORIZON_SEC,
        "PRICE_SCALE": PRICE_SCALE,
        "MAX_BOOK_AGE_MS": MAX_BOOK_AGE_MS,
    }
    bad_constants: dict[str, Any] = {}
    for name, expected in required_constants.items():
        found = getattr(engine, name, None)
        if name == "FIXED_HORIZONS_SEC":
            try:
                found_cmp = tuple(int(x) for x in found)
            except Exception:
                found_cmp = found
            if found_cmp != tuple(expected):
                bad_constants[name] = (
                    found,
                    expected,
                )
        elif safe_float(found) != safe_float(expected):
            bad_constants[name] = (
                found,
                expected,
            )

    if missing_callables or bad_constants:
        raise RuntimeError(
            "Phase 7I.4 v1.0.1 replay interface is incompatible "
            "with Phase 7M.3. "
            f"Missing callables={missing_callables}; "
            f"bad constants={bad_constants}"
        )


class ProgressTracker:
    """
    Human-visible and durable progress for the expensive replay.

    Phase 7I.4 remains responsible for scientific per-date checkpointing.
    This class only observes and reports progress.
    """

    def __init__(
        self,
        *,
        output_root: Path,
        report_root: Path,
        scope_dates: list[str],
        scope_candidates,
        heartbeat_sec: float,
    ):
        self.pd = require_pandas()
        self.output_root = output_root
        self.report_root = report_root
        self.progress_json = (
            output_root / "metadata" / "progress.json"
        )
        self.progress_csv = (
            report_root / "replay_progress.csv"
        )
        self.scope_dates = list(scope_dates)
        self.date_index = {
            d: i + 1
            for i, d in enumerate(self.scope_dates)
        }
        self.total_dates = len(self.scope_dates)
        self.candidate_count_by_date = {
            str(k): int(v)
            for k, v in (
                scope_candidates.groupby(
                    scope_candidates[
                        "candidate_date"
                    ].astype(str)
                ).size()
                .to_dict()
                .items()
            )
        }
        self.total_candidates = int(len(scope_candidates))
        self.heartbeat_sec = max(
            0.0,
            float(heartbeat_sec),
        )

        self.run_started = time.perf_counter()
        self.run_started_utc = utc_now_iso()
        self.current_date: str | None = None
        self.current_date_started: float | None = None
        self.completed_dates = 0
        self.completed_candidates = 0
        self.cumulative_windows = 0
        self.cumulative_valid_windows = 0
        self.reused_dates = 0
        self.built_dates = 0
        self.rows: list[dict[str, Any]] = []

        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        # The ledger describes THIS invocation only. Replay checkpoint
        # resumability is independent and remains in Phase 7I.4.
        self._write_json(
            status="STARTING",
            note=(
                "Replay checkpoints are independently resumable; "
                "this progress ledger covers the current invocation."
            ),
        )
        atomic_csv(
            self.pd.DataFrame(
                columns=[
                    "candidate_date",
                    "date_index",
                    "total_dates",
                    "candidates",
                    "windows",
                    "valid_windows",
                    "checkpoint_status",
                    "date_elapsed_sec",
                    "run_elapsed_sec",
                    "completed_candidates",
                    "total_candidates",
                    "trade_aggregation_seconds",
                    "raw_replay_seconds",
                    "outcome_seconds",
                    "depth_rows_decoded",
                    "tracked_depth_messages",
                    "completed_at_utc",
                ]
            ),
            self.progress_csv,
        )

    def _snapshot(
        self,
        status: str,
        *,
        note: str | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        now_perf = time.perf_counter()
        current_elapsed = (
            now_perf - self.current_date_started
            if self.current_date_started is not None
            else 0.0
        )
        return {
            "builder_version": BUILDER_VERSION,
            "schema_name": SCHEMA_NAME,
            "status": status,
            "run_started_utc": self.run_started_utc,
            "last_update_utc": utc_now_iso(),
            "current_date": self.current_date,
            "date_index": (
                self.date_index.get(
                    str(self.current_date)
                )
                if self.current_date is not None
                else None
            ),
            "total_dates": self.total_dates,
            "current_date_candidates": (
                self.candidate_count_by_date.get(
                    str(self.current_date),
                    0,
                )
                if self.current_date is not None
                else 0
            ),
            "completed_dates": self.completed_dates,
            "completed_candidates": self.completed_candidates,
            "total_candidates": self.total_candidates,
            "cumulative_windows": self.cumulative_windows,
            "cumulative_valid_windows": (
                self.cumulative_valid_windows
            ),
            "reused_dates": self.reused_dates,
            "built_dates": self.built_dates,
            "current_date_elapsed_sec": round(
                current_elapsed,
                1,
            ),
            "run_elapsed_sec": round(
                now_perf - self.run_started,
                1,
            ),
            "heartbeat_sec": self.heartbeat_sec,
            "note": note,
            "error": error,
        }

    def _write_json(
        self,
        *,
        status: str,
        note: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            atomic_json(
                self.progress_json,
                self._snapshot(
                    status,
                    note=note,
                    error=error,
                ),
            )

    def begin_date(
        self,
        candidate_date: str,
    ) -> None:
        self.stop_heartbeat()
        self.current_date = str(candidate_date)
        self.current_date_started = time.perf_counter()

        idx = self.date_index.get(
            self.current_date,
            0,
        )
        n = self.candidate_count_by_date.get(
            self.current_date,
            0,
        )
        print(
            f"[7M.3 START] [{idx:02d}/{self.total_dates:02d}] "
            f"{self.current_date}: candidates={n:,} | "
            f"completed={self.completed_dates}/{self.total_dates} dates, "
            f"{self.completed_candidates:,}/{self.total_candidates:,} candidates"
        )
        self._write_json(status="PROCESSING_DATE")

        if self.heartbeat_sec > 0:
            self._stop_event = threading.Event()
            self._thread = threading.Thread(
                target=self._heartbeat_loop,
                name="phase7m3-heartbeat",
                daemon=True,
            )
            self._thread.start()

    def _heartbeat_loop(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.wait(
            self.heartbeat_sec
        ):
            idx = self.date_index.get(
                str(self.current_date),
                0,
            )
            n = self.candidate_count_by_date.get(
                str(self.current_date),
                0,
            )
            elapsed = (
                time.perf_counter()
                - (self.current_date_started or time.perf_counter())
            )
            run_elapsed = (
                time.perf_counter() - self.run_started
            )
            print(
                f"[7M.3 HEARTBEAT] "
                f"[{idx:02d}/{self.total_dates:02d}] "
                f"{self.current_date}: "
                f"date_candidates={n:,} "
                f"date_elapsed={elapsed:.0f}s "
                f"run_elapsed={run_elapsed:.0f}s | "
                f"completed_dates={self.completed_dates}/{self.total_dates} "
                f"completed_candidates={self.completed_candidates:,}/{self.total_candidates:,} "
                f"windows={self.cumulative_windows:,} "
                f"valid={self.cumulative_valid_windows:,}",
                flush=True,
            )
            self._write_json(status="PROCESSING_DATE")

    def stop_heartbeat(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if (
            self._thread is not None
            and self._thread.is_alive()
            and self._thread is not threading.current_thread()
        ):
            self._thread.join(timeout=2.0)
        self._stop_event = None
        self._thread = None

    def finish_date(
        self,
        *,
        candidate_date: str,
        candidate_count: int,
        mechanics,
        reused: bool,
        diag: dict[str, Any],
    ) -> None:
        self.stop_heartbeat()
        now_perf = time.perf_counter()
        day_elapsed = (
            now_perf - self.current_date_started
            if self.current_date_started is not None
            else 0.0
        )

        windows = int(len(mechanics))
        valid = (
            int(
                mechanics["window_valid"]
                .fillna(False)
                .astype(bool)
                .sum()
            )
            if (
                len(mechanics)
                and "window_valid" in mechanics.columns
            )
            else 0
        )

        self.completed_dates += 1
        self.completed_candidates += int(
            candidate_count
        )
        self.cumulative_windows += windows
        self.cumulative_valid_windows += valid
        self.reused_dates += int(bool(reused))
        self.built_dates += int(not bool(reused))

        row = {
            "candidate_date": str(candidate_date),
            "date_index": self.date_index.get(
                str(candidate_date),
                0,
            ),
            "total_dates": self.total_dates,
            "candidates": int(candidate_count),
            "windows": windows,
            "valid_windows": valid,
            "checkpoint_status": (
                "REUSED" if reused else "BUILT"
            ),
            "date_elapsed_sec": round(
                day_elapsed,
                3,
            ),
            "run_elapsed_sec": round(
                now_perf - self.run_started,
                3,
            ),
            "completed_candidates": (
                self.completed_candidates
            ),
            "total_candidates": self.total_candidates,
            "trade_aggregation_seconds": safe_float(
                diag.get("trade_aggregation_seconds")
            ),
            "raw_replay_seconds": safe_float(
                diag.get("raw_replay_seconds")
            ),
            "outcome_seconds": safe_float(
                diag.get("outcome_seconds")
            ),
            "depth_rows_decoded": int(
                diag.get("depth_rows_decoded") or 0
            ),
            "tracked_depth_messages": int(
                diag.get("tracked_depth_messages") or 0
            ),
            "completed_at_utc": utc_now_iso(),
        }
        self.rows.append(row)
        atomic_csv(
            self.pd.DataFrame(self.rows),
            self.progress_csv,
        )

        idx = self.date_index.get(
            str(candidate_date),
            0,
        )
        print(
            f"[7M.3 PROGRESS] "
            f"[{idx:02d}/{self.total_dates:02d}] "
            f"{candidate_date}: "
            f"{'REUSED' if reused else 'BUILT'} "
            f"candidates={candidate_count:,} "
            f"windows={windows:,} valid={valid:,} | "
            f"cumulative_candidates="
            f"{self.completed_candidates:,}/{self.total_candidates:,} "
            f"cumulative_windows={self.cumulative_windows:,} "
            f"valid={self.cumulative_valid_windows:,} "
            f"date_elapsed={day_elapsed:.1f}s "
            f"run_elapsed={now_perf-self.run_started:.1f}s",
            flush=True,
        )

        self.current_date = None
        self.current_date_started = None
        self._write_json(status="RUNNING_BETWEEN_DATES")

    def fail_date(
        self,
        error: BaseException,
    ) -> None:
        self.stop_heartbeat()
        self._write_json(
            status="ERROR",
            error=f"{type(error).__name__}: {error}",
        )

    def complete(
        self,
        status: str,
    ) -> None:
        self.stop_heartbeat()
        self.current_date = None
        self.current_date_started = None
        self._write_json(
            status=status,
            note=(
                "All dates in this invocation finished. "
                "See latest_summary.json for scientific integrity status."
            ),
        )


def patch_engine(
    engine,
    *,
    selected,
    selected_outcomes,
    freeze_summary: dict[str, Any],
    freeze_manifest: dict[str, Any],
    candidate_id_hash: str,
    tracker: ProgressTracker,
):
    """
    Patch ONLY the candidate manifest/schema globals and add an observing
    progress wrapper around the frozen replay function.

    The underlying order-book/trade replay implementation is untouched.
    """
    pd = require_pandas()

    if str(
        getattr(engine, "BUILDER_VERSION", "")
    ) != FROZEN_PHASE7I4_VERSION:
        raise RuntimeError(
            f"Phase 7M.3 requires frozen Phase 7I.4 "
            f"v{FROZEN_PHASE7I4_VERSION}; found "
            f"{getattr(engine, 'BUILDER_VERSION', None)!r}"
        )

    validate_7i4_replay_interface(engine)

    up = int(
        (
            pd.to_numeric(
                selected["direction"],
                errors="coerce",
            ) > 0
        ).sum()
    )
    down = int(
        (
            pd.to_numeric(
                selected["direction"],
                errors="coerce",
            ) < 0
        ).sum()
    )

    original_process = engine.process_candidate_date

    # A unique builder/schema name makes Phase7I.4's input signatures and
    # checkpoints specific to this 25-bp substrate.
    engine.BUILDER_VERSION = BUILDER_VERSION
    engine.SCHEMA_VERSION = SCHEMA_VERSION
    engine.SCHEMA_NAME = SCHEMA_NAME

    engine.EXPECTED_FULL_CANDIDATES = int(
        len(selected)
    )
    engine.EXPECTED_FULL_UP = up
    engine.EXPECTED_FULL_DOWN = down
    engine.EXPECTED_FULL_CANDIDATE_DAYS = int(
        selected["candidate_date"].astype(str).nunique()
    )

    def selected_loader(_args):
        return (
            selected.copy(),
            selected_outcomes.copy(),
            dict(freeze_summary),
            dict(freeze_manifest),
            str(candidate_id_hash),
        )

    def progress_process(
        candidate_date,
        day_candidates,
        approved_hash,
        phase6a,
        phase7a,
        phase7e,
        phase7i1,
        family_maps,
        args,
    ):
        tracker.begin_date(str(candidate_date))
        try:
            result = original_process(
                candidate_date,
                day_candidates,
                approved_hash,
                phase6a,
                phase7a,
                phase7e,
                phase7i1,
                family_maps,
                args,
            )
            (
                mech_day,
                level_day,
                out_day,
                reused,
                diag,
            ) = result
            tracker.finish_date(
                candidate_date=str(candidate_date),
                candidate_count=int(
                    len(day_candidates)
                ),
                mechanics=mech_day,
                reused=bool(reused),
                diag=dict(diag or {}),
            )
            return result
        except BaseException as exc:
            tracker.fail_date(exc)
            raise

    engine.load_frozen_candidates = selected_loader
    engine.process_candidate_date = progress_process

    return up, down


def normalize_legacy_35bp_interface(frame):
    """
    Phase 7I.4 has two harmless legacy 35-bp labels:
      * stage='CAUSAL_35BP'
      * extra_displacement_since_35bp_candidate_bps

    Its math uses candidate_second/candidate_displacement_bps, so it is valid
    at 25 bps.  Normalize names in the FINAL aggregate outputs so downstream
    Phase 7M.4 cannot accidentally interpret the label as a 35-bp anchor.

    Per-date raw replay checkpoints remain engine-internal and untouched.
    """
    if frame is None:
        return frame
    out = frame.copy()

    if "stage" in out.columns:
        out["stage"] = (
            out["stage"]
            .astype(str)
            .replace(
                {"CAUSAL_35BP": "PROSPECTIVE_25BP"}
            )
        )

    legacy = (
        "extra_displacement_since_35bp_candidate_bps"
    )
    modern = "extra_displacement_since_candidate_bps"
    if legacy in out.columns:
        if modern not in out.columns:
            out[modern] = out[legacy]
        out = out.drop(columns=[legacy])

    out["candidate_threshold_bps"] = (
        FROZEN_CANDIDATE_BPS
    )
    out["candidate_policy_id"] = FROZEN_POLICY
    out["prospective_25bp_population"] = True
    return out


def normalize_final_aggregate_outputs(
    *,
    engine,
    output_root: Path,
    report_root: Path,
) -> dict[str, int]:
    pd, pq = require_stack()

    stats = {
        "candidate_inputs_rows": 0,
        "mechanics_rows": 0,
        "valid_mechanics_rows": 0,
        "decision_outcome_rows": 0,
    }

    paths = (
        (
            output_root / "candidate_inputs.parquet",
            "candidate_inputs_rows",
        ),
        (
            output_root / "candidate_mechanics.parquet",
            "mechanics_rows",
        ),
        (
            output_root / "candidate_decision_outcomes.parquet",
            "decision_outcome_rows",
        ),
    )

    normalized: dict[str, Any] = {}
    for path, key in paths:
        if not path.exists():
            continue
        frame = pq.read_table(path).to_pandas()
        frame = normalize_legacy_35bp_interface(
            frame
        )
        engine.write_parquet_atomic(frame, path)
        stats[key] = int(len(frame))
        normalized[path.name] = frame

    mechanics = normalized.get(
        "candidate_mechanics.parquet"
    )
    outcomes = normalized.get(
        "candidate_decision_outcomes.parquet"
    )
    if mechanics is not None:
        stats["valid_mechanics_rows"] = (
            int(
                mechanics["window_valid"]
                .fillna(False)
                .astype(bool)
                .sum()
            )
            if "window_valid" in mechanics.columns
            else 0
        )

    # Rebuild schema report after semantic normalization when possible.
    if (
        mechanics is not None
        and outcomes is not None
        and callable(
            getattr(engine, "build_schema_report", None)
        )
    ):
        schema = engine.build_schema_report(
            mechanics,
            outcomes,
        )
        if schema is not None:
            engine.write_csv_atomic(
                schema,
                report_root /
                "participant_interaction_schema.csv",
            )

    return stats


def build_custom_invariants(
    *,
    output_root: Path,
    candidate_hash_expected: str,
    full_scope: bool,
    expected_scope_candidates: int,
):
    pd, pq = require_stack()
    rows: list[dict[str, Any]] = []

    def add(
        check: str,
        passed: bool,
        observed: Any,
        expected: Any,
    ) -> None:
        rows.append(
            {
                "check": check,
                "status": (
                    "PASS" if bool(passed) else "FAIL"
                ),
                "observed": observed,
                "expected": expected,
            }
        )

    cp = output_root / "candidate_inputs.parquet"
    mp = output_root / "candidate_mechanics.parquet"
    op = output_root / "candidate_decision_outcomes.parquet"
    up = output_root / "candidate_7m2_outcomes.parquet"

    for path, label in (
        (cp, "candidate_inputs_exists"),
        (mp, "candidate_mechanics_exists"),
        (op, "decision_outcomes_exists"),
        (up, "phase7m2_outcomes_exists"),
    ):
        add(label, path.exists(), path.exists(), True)

    if not all(p.exists() for p in (cp, mp, op)):
        return pd.DataFrame(rows)

    candidates = pq.read_table(cp).to_pandas()
    mechanics = pq.read_table(mp).to_pandas()
    outcomes = pq.read_table(op).to_pandas()

    add(
        "scope_candidate_count",
        len(candidates) == expected_scope_candidates,
        len(candidates),
        expected_scope_candidates,
    )
    add(
        "candidate_ids_unique",
        not candidates["candidate_id"].duplicated().any(),
        int(candidates["candidate_id"].duplicated().sum()),
        0,
    )
    if full_scope:
        add(
            "full_candidate_hash_preserved",
            candidate_hash(candidates)
            == candidate_hash_expected,
            candidate_hash(candidates),
            candidate_hash_expected,
        )

    contract_ok = (
        unique_numeric(
            candidates, "candidate_threshold_bps"
        ) == [FROZEN_CANDIDATE_BPS]
        and unique_numeric(
            candidates, "reset_threshold_bps"
        ) == [FROZEN_RESET_BPS]
        and unique_numeric(
            candidates, "release_threshold_bps"
        ) == [FROZEN_RELEASE_BPS]
        and unique_numeric(
            candidates, "cooldown_sec"
        ) == [float(FROZEN_COOLDOWN_SEC)]
    )
    add(
        "prospective_5_10_25_60_contract",
        contract_ok,
        {
            "reset": unique_numeric(
                candidates, "reset_threshold_bps"
            ),
            "release": unique_numeric(
                candidates, "release_threshold_bps"
            ),
            "candidate": unique_numeric(
                candidates, "candidate_threshold_bps"
            ),
            "cooldown": unique_numeric(
                candidates, "cooldown_sec"
            ),
        },
        "reset=5 release=10 candidate=25 cooldown=60",
    )

    causal_leak_tokens = (
        "reach_",
        "future",
        "outcome",
        "mfe_",
        "mae_",
        "seconds_to_35",
        "seconds_to_50",
        "seconds_to_75",
        "seconds_to_100",
    )
    input_leaks = [
        c
        for c in candidates.columns
        if any(
            tok in c.lower()
            for tok in causal_leak_tokens
        )
    ]
    add(
        "candidate_inputs_future_label_clean",
        len(input_leaks) == 0,
        input_leaks,
        [],
    )

    mechanics_leaks = [
        c
        for c in mechanics.columns
        if (
            c.lower().startswith("h300_")
            or c.lower().startswith("h600_")
            or c.lower().startswith("h900_")
            or c.lower().startswith("h1800_")
            or "origin_reach_" in c.lower()
            or "first_favorable_" in c.lower()
            or "first_adverse_" in c.lower()
        )
    ]
    add(
        "candidate_mechanics_future_label_clean",
        len(mechanics_leaks) == 0,
        mechanics_leaks,
        [],
    )

    expected_rows = (
        len(candidates) * len(FIXED_HORIZONS_SEC)
    )
    add(
        "mechanics_has_four_horizons_per_candidate",
        len(mechanics) == expected_rows,
        len(mechanics),
        expected_rows,
    )
    add(
        "decision_outcomes_has_four_horizons_per_candidate",
        len(outcomes) == expected_rows,
        len(outcomes),
        expected_rows,
    )

    mech_keys = (
        mechanics[
            ["candidate_id", "observation_horizon_sec"]
        ]
        .astype(
            {
                "candidate_id": str,
            }
        )
    )
    add(
        "mechanics_keys_unique",
        not mech_keys.duplicated().any(),
        int(mech_keys.duplicated().sum()),
        0,
    )

    add(
        "legacy_35bp_stage_label_removed_from_aggregate",
        (
            "stage" not in mechanics.columns
            or not (
                mechanics["stage"].astype(str)
                == "CAUSAL_35BP"
            ).any()
        ),
        (
            int(
                (
                    mechanics["stage"].astype(str)
                    == "CAUSAL_35BP"
                ).sum()
            )
            if "stage" in mechanics.columns
            else 0
        ),
        0,
    )
    add(
        "legacy_35bp_extra_displacement_name_removed",
        (
            "extra_displacement_since_35bp_candidate_bps"
            not in mechanics.columns
            and
            "extra_displacement_since_35bp_candidate_bps"
            not in outcomes.columns
        ),
        (
            "extra_displacement_since_35bp_candidate_bps"
            in mechanics.columns
            or
            "extra_displacement_since_35bp_candidate_bps"
            in outcomes.columns
        ),
        False,
    )

    primary = mechanics[
        pd.to_numeric(
            mechanics["observation_horizon_sec"],
            errors="coerce",
        ) == PRIMARY_HORIZON_SEC
    ]
    valid_primary = (
        float(
            primary["window_valid"]
            .fillna(False)
            .astype(bool)
            .mean()
        )
        if (
            len(primary)
            and "window_valid" in primary.columns
        )
        else math.nan
    )
    add(
        "primary_3s_exact_book_coverage_at_least_95pct",
        (
            math.isfinite(valid_primary)
            and valid_primary >= 0.95
        ),
        valid_primary,
        ">=0.95",
    )

    return pd.DataFrame(rows)


def rewrite_metadata_and_reports(
    *,
    engine,
    output_root: Path,
    report_root: Path,
    phase7m2_root: Path,
    phase7i4_script: Path,
    selected,
    selected_outcomes,
    m2_summary: dict[str, Any],
    freeze_summary: dict[str, Any],
    freeze_manifest: dict[str, Any],
    candidate_id_hash: str,
    up: int,
    down: int,
    engine_status: int,
    max_candidate_dates: int | None,
    tracker: ProgressTracker,
) -> int:
    pd, pq = require_stack()

    latest_path = (
        output_root / "metadata" / "latest_summary.json"
    )
    engine_summary: dict[str, Any] = {}
    if latest_path.exists():
        engine_summary = json.loads(
            latest_path.read_text(encoding="utf-8")
        )

    # The frozen engine writes its convenience upstream future-label table
    # with a legacy 7I.1 name. Rename it so its provenance is unambiguous.
    old_future = (
        output_root / "candidate_7i1_outcomes.parquet"
    )
    new_future = (
        output_root / "candidate_7m2_outcomes.parquet"
    )
    if old_future.exists():
        if new_future.exists():
            new_future.unlink()
        os.replace(old_future, new_future)

    stats = normalize_final_aggregate_outputs(
        engine=engine,
        output_root=output_root,
        report_root=report_root,
    )

    all_dates = sorted(
        selected["candidate_date"]
        .astype(str)
        .unique()
        .tolist()
    )
    scope_dates = list(all_dates)
    if max_candidate_dates is not None:
        scope_dates = scope_dates[
            : max(0, int(max_candidate_dates))
        ]
    scope = selected[
        selected["candidate_date"]
        .astype(str)
        .isin(scope_dates)
    ].copy()
    partial_scope = len(scope_dates) < len(all_dates)

    custom_inv = build_custom_invariants(
        output_root=output_root,
        candidate_hash_expected=candidate_id_hash,
        full_scope=not partial_scope,
        expected_scope_candidates=int(len(scope)),
    )
    custom_failed = custom_inv[
        custom_inv["status"] == "FAIL"
    ]
    atomic_csv(
        custom_inv,
        report_root /
        "prospective_25bp_exact_replay_invariants.csv",
    )

    contract_rows = [
        {
            "field": "upstream_phase",
            "value": "Phase 7M.2 v1.0.0 PASS",
        },
        {
            "field": "policy_id",
            "value": FROZEN_POLICY,
        },
        {
            "field": "candidate_threshold_bps",
            "value": FROZEN_CANDIDATE_BPS,
        },
        {
            "field": "reset_bps",
            "value": FROZEN_RESET_BPS,
        },
        {
            "field": "release_bps",
            "value": FROZEN_RELEASE_BPS,
        },
        {
            "field": "rearm_sec",
            "value": FROZEN_COOLDOWN_SEC,
        },
        {
            "field": "full_candidate_count",
            "value": len(selected),
        },
        {
            "field": "full_candidate_days",
            "value": len(all_dates),
        },
        {
            "field": "scope_candidate_count",
            "value": len(scope),
        },
        {
            "field": "scope_candidate_days",
            "value": len(scope_dates),
        },
        {
            "field": "candidate_id_set_sha256",
            "value": candidate_id_hash,
        },
        {
            "field": "future35_failures_included",
            "value": True,
        },
        {
            "field": "fixed_horizons_sec",
            "value": ",".join(
                str(x)
                for x in FIXED_HORIZONS_SEC
            ),
        },
        {
            "field": "primary_horizon_sec",
            "value": PRIMARY_HORIZON_SEC,
        },
        {
            "field": "raw_replay_engine",
            "value": (
                "Phase 7I.4 v1.0.1 exact "
                "depth/trade engine"
            ),
        },
        {
            "field": "scientific_role",
            "value": (
                "EXACT_CAUSAL_SUBSTRATE_ONLY_"
                "NO_OUTCOME_EFFECT_TEST"
            ),
        },
    ]
    atomic_csv(
        pd.DataFrame(contract_rows),
        report_root /
        "prospective_25bp_exact_replay_contract.csv",
    )

    engine_status_name = str(
        engine_summary.get("status", "")
    )
    if not engine_status_name:
        engine_status_name = (
            "PASS" if engine_status == 0 else "FAIL"
        )

    if custom_failed.empty and engine_status == 0:
        status = (
            "SMOKE_PASS"
            if partial_scope
            else "PASS"
        )
    else:
        status = "FAIL"

    summary = {
        "builder_version": BUILDER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "schema_name": SCHEMA_NAME,
        "status": status,
        "run_scope": (
            "PARTIAL_SMOKE"
            if partial_scope
            else "FULL"
        ),
        "symbol": engine_summary.get(
            "symbol",
            "BTCUSDT",
        ),
        "upstream_phase7m2_builder": (
            FROZEN_PHASE7M2_VERSION
        ),
        "upstream_phase7m2_schema": (
            FROZEN_PHASE7M2_SCHEMA
        ),
        "raw_replay_engine_builder": (
            FROZEN_PHASE7I4_VERSION
        ),
        "selected_policy": FROZEN_POLICY,
        "reset_bps": FROZEN_RESET_BPS,
        "release_bps": FROZEN_RELEASE_BPS,
        "candidate_threshold_bps": (
            FROZEN_CANDIDATE_BPS
        ),
        "rearm_sec": FROZEN_COOLDOWN_SEC,
        "full_candidate_count": int(len(selected)),
        "candidate_count": int(len(scope)),
        "candidate_dates": int(len(scope_dates)),
        "full_candidate_dates": int(len(all_dates)),
        "up_candidates_full": int(up),
        "down_candidates_full": int(down),
        "candidate_id_set_sha256": (
            candidate_id_hash
        ),
        "fixed_horizons_sec": list(
            FIXED_HORIZONS_SEC
        ),
        "primary_horizon_sec": (
            PRIMARY_HORIZON_SEC
        ),
        "prospective_population_includes_failed_before35": True,
        "causal_mechanics_future_labels_separate": True,
        "candidate_inputs_rows": (
            stats["candidate_inputs_rows"]
        ),
        "mechanics_rows": stats["mechanics_rows"],
        "valid_mechanics_rows": (
            stats["valid_mechanics_rows"]
        ),
        "decision_outcome_rows": (
            stats["decision_outcome_rows"]
        ),
        "reused_candidate_date_checkpoints": int(
            tracker.reused_dates
        ),
        "built_candidate_date_checkpoints": int(
            tracker.built_dates
        ),
        "failed_custom_invariants": int(
            len(custom_failed)
        ),
        "engine_status_before_wrapper_rewrite": (
            engine_status_name
        ),
        "generated_at": utc_now_iso(),
        "scientific_state": (
            "EXACT_25BP_SUBSTRATE_ONLY_"
            "AWAIT_PHASE_7M4_MECHANISM_ANALYSIS"
        ),
        "interpretation": (
            "No exact book feature has been declared predictive "
            "by this substrate. Phase 7M.4 must perform the "
            "outcome-controlled analysis."
        ),
    }

    run_config = {
        **summary,
        "phase7m2_root": str(
            phase7m2_root.resolve()
        ),
        "phase7i4_script": str(
            phase7i4_script.resolve()
        ),
        "raw_engine_contract": {
            "price_scale": PRICE_SCALE,
            "max_book_age_ms": MAX_BOOK_AGE_MS,
            "fixed_horizons_sec": list(
                FIXED_HORIZONS_SEC
            ),
            "primary_horizon_sec": (
                PRIMARY_HORIZON_SEC
            ),
            "frozen_levels_within_bps": 5.0,
        },
        "progress_contract": {
            "progress_json": str(
                (
                    output_root /
                    "metadata" /
                    "progress.json"
                ).resolve()
            ),
            "progress_csv": str(
                (
                    report_root /
                    "replay_progress.csv"
                ).resolve()
            ),
            "heartbeat_sec": (
                tracker.heartbeat_sec
            ),
            "eta_policy": (
                "NO_ETA_REPORTED_RUNTIME_VARIES_BY_DATE"
            ),
        },
        "data_separation": {
            "causal_inputs": (
                "candidate_inputs.parquet"
            ),
            "causal_mechanics": (
                "candidate_mechanics.parquet"
            ),
            "future_decision_outcomes": (
                "candidate_decision_outcomes.parquet"
            ),
            "upstream_phase7m2_future_labels": (
                "candidate_7m2_outcomes.parquet"
            ),
        },
        "legacy_interface_note": (
            "Frozen Phase 7I.4 internally names its stage "
            "CAUSAL_35BP and one displacement column with "
            "'35bp'. The math is candidate-relative. Final "
            "aggregate Phase7M.3 outputs normalize those names "
            "to PROSPECTIVE_25BP / candidate-relative wording. "
            "Per-date replay checkpoints remain engine-internal."
        ),
        "upstream_phase7m2_review_gate": {
            "primary_25bp_precompute_feasible": (
                m2_summary.get(
                    "primary_25bp_precompute_feasible"
                )
            ),
            "primary_25bp_validation_effect_pp": (
                m2_summary.get(
                    "primary_25bp_validation_effect_pp"
                )
            ),
            "primary_25bp_validation_ci_low_pp": (
                m2_summary.get(
                    "primary_25bp_validation_ci_low_pp"
                )
            ),
        },
    }

    frozen_manifest = {
        **freeze_manifest,
        "builder_version": BUILDER_VERSION,
        "schema_name": SCHEMA_NAME,
        "freeze_status": (
            "FROZEN_PHASE_7M3_EXACT_REPLAY_POPULATION"
        ),
        "raw_replay_engine_builder": (
            FROZEN_PHASE7I4_VERSION
        ),
    }

    atomic_json(
        output_root /
        "metadata" /
        "latest_summary.json",
        summary,
    )
    atomic_json(
        output_root /
        "metadata" /
        "run_config.json",
        run_config,
    )
    atomic_json(
        output_root /
        "metadata" /
        "frozen_candidate_manifest.json",
        frozen_manifest,
    )

    findings = f"""# Phase 7M.3 — Prospective 25-bp exact participant-interaction substrate

- Builder: **{BUILDER_VERSION}**
- Status: **{status}**
- Run scope: **{'PARTIAL SMOKE' if partial_scope else 'FULL'}**
- Prospective policy: **{FROZEN_POLICY}**
- Candidate threshold: **25 bps**
- Candidates in scope: **{len(scope):,}**
- Candidate dates in scope: **{len(scope_dates)}**
- Exact mechanics rows: **{stats['mechanics_rows']:,}**
- Valid exact mechanics rows: **{stats['valid_mechanics_rows']:,}**
- Reused candidate-date checkpoints this invocation: **{tracker.reused_dates}**
- Built candidate-date checkpoints this invocation: **{tracker.built_dates}**

## What this phase establishes

This is the reusable exact causal measurement substrate for the genuinely
prospective 25-bp candidate population established in Phase 7M.2.

The candidate universe includes attempts that later fail before 35 bps.
No future 35/50/75/100 outcome was used to select the replay population.

For each candidate the frozen Phase 7I.4 exact book/trade engine measures
1/3/5/10-second queue and participant-flow mechanics at the 25-bp candidate
second.

## What this phase does NOT establish

This substrate does **not** establish that exact queue failure predicts
35/50/75/100 continuation.

It does not:
- choose a winning exact book feature;
- build a composite failure score;
- fit ML;
- optimize an entry threshold;
- establish profitability.

Those questions are intentionally deferred to cheap Phase 7M.4 analysis so the
expensive replay remains reusable if statistical analysis changes.

## Data separation

Causal:
- `candidate_inputs.parquet`
- `candidate_mechanics.parquet`

Future labels:
- `candidate_decision_outcomes.parquet`
- `candidate_7m2_outcomes.parquet`

## Runtime / resumability

Per-candidate-date raw replay checkpoints are atomic and signature-validated by
the frozen Phase 7I.4 engine.

Live progress:
- `metadata/progress.json`
- `replay_progress.csv`

An interrupted run can be restarted with the same command. Matching completed
date checkpoints are reused.

## Next scientific step

After a full PASS, Phase 7M.4 should ask whether the predeclared exact mechanism
families — opposing initial queue destruction, retained liquidity, durable
unrecovered loss, refill durability, support resilience and their interaction
with aggression — add information for prospective 75-bp continuation after
appropriate controls.
"""
    atomic_text(
        report_root /
        "participant_interaction_findings.md",
        findings,
    )

    final_progress_status = (
        "SMOKE_COMPLETE"
        if status == "SMOKE_PASS"
        else (
            "COMPLETE"
            if status == "PASS"
            else "FAILED_INTEGRITY"
        )
    )
    tracker.complete(final_progress_status)

    return 0 if status in {"PASS", "SMOKE_PASS"} else 2


def self_test() -> int:
    pd = require_pandas()

    rows = []
    for threshold in (15, 25):
        for i in range(4):
            rows.append(
                {
                    "candidate_id": (
                        f"{threshold}-c{i}"
                    ),
                    "policy_id": (
                        FROZEN_POLICY
                        if threshold == 25
                        else "EARLY_15BP_RESET5_RELEASE10_REARM60"
                    ),
                    "candidate_date": (
                        "2026-06-01"
                    ),
                    "direction": (
                        1 if i % 2 == 0 else -1
                    ),
                    "direction_label": (
                        "UP" if i % 2 == 0 else "DOWN"
                    ),
                    "origin_kind": (
                        "REARM_RESET"
                        if i
                        else "INITIAL_EXTREME"
                    ),
                    "origin_second": (
                        1000 + i * 100
                    ),
                    "origin_price": 100.0,
                    "release_second": (
                        1010 + i * 100
                    ),
                    "candidate_second": (
                        1030 + i * 100
                    ),
                    "candidate_price": (
                        100.25
                        if threshold == 25
                        else 100.15
                    ),
                    "candidate_displacement_bps": (
                        float(threshold)
                    ),
                    "reset_threshold_bps": 5.0,
                    "release_threshold_bps": 10.0,
                    "candidate_threshold_bps": (
                        float(threshold)
                    ),
                    "cooldown_sec": 60,
                    "early_threshold_bps": (
                        int(threshold)
                    ),
                    "is_reaccel_context": (
                        i >= 2
                    ),
                    # Synthetic future field deliberately exists upstream.
                    # The whitelist MUST remove it.
                    "h900_origin_reach_75": (
                        i % 2 == 0
                    ),
                }
            )

    summary = {
        "builder_version": FROZEN_PHASE7M2_VERSION,
        "schema_name": FROZEN_PHASE7M2_SCHEMA,
        "status": "PASS",
        "primary_threshold_bps": 25,
        "reset_bps": 5.0,
        "release_bps": 10.0,
        "rearm_sec": 60,
        "prospective_population_includes_failed_before35": True,
        "candidate_count_by_threshold": {
            "25": 4,
        },
    }

    selected, contract = validate_and_adapt_25bp_candidates(
        pd.DataFrame(rows),
        summary,
        full_scope_required=False,
    )
    assert len(selected) == 4
    assert (
        selected["candidate_threshold_bps"]
        .astype(float)
        == 25.0
    ).all()
    assert selected["episode_id"].str.startswith(
        "PROSPECTIVE_25BP::"
    ).all()
    assert (
        selected["seconds_release_to_candidate"]
        .astype(int)
        == 20
    ).all()
    assert "h900_origin_reach_75" not in selected.columns
    assert contract == {
        "reset_bps": [5.0],
        "release_bps": [10.0],
        "candidate_bps": [25.0],
        "early_threshold_bps": [25.0],
        "cooldown_sec": [60.0],
    }

    legacy = pd.DataFrame(
        {
            "candidate_id": ["c1"],
            "stage": ["CAUSAL_35BP"],
            "extra_displacement_since_35bp_candidate_bps": [
                2.5
            ],
        }
    )
    normalized = normalize_legacy_35bp_interface(
        legacy
    )
    assert (
        normalized.loc[0, "stage"]
        == "PROSPECTIVE_25BP"
    )
    assert (
        "extra_displacement_since_35bp_candidate_bps"
        not in normalized.columns
    )
    assert math.isclose(
        float(
            normalized.loc[
                0,
                "extra_displacement_since_candidate_bps",
            ]
        ),
        2.5,
    )
    assert (
        normalized.loc[
            0, "candidate_threshold_bps"
        ]
        == 25.0
    )

    with tempfile.TemporaryDirectory(
        prefix="phase7m3-selftest-"
    ) as tmp:
        root = Path(tmp)
        tracker = ProgressTracker(
            output_root=root / "data",
            report_root=root / "reports",
            scope_dates=["2026-06-01"],
            scope_candidates=selected,
            heartbeat_sec=0,
        )
        tracker.begin_date("2026-06-01")
        mechanics = pd.DataFrame(
            {
                "window_valid": [
                    True,
                    True,
                    False,
                    True,
                ]
            }
        )
        tracker.finish_date(
            candidate_date="2026-06-01",
            candidate_count=4,
            mechanics=mechanics,
            reused=False,
            diag={
                "raw_replay_seconds": 12.3,
                "tracked_depth_messages": 1000,
            },
        )
        tracker.complete("SMOKE_COMPLETE")

        progress = json.loads(
            (
                root /
                "data" /
                "metadata" /
                "progress.json"
            ).read_text(encoding="utf-8")
        )
        assert (
            progress["status"]
            == "SMOKE_COMPLETE"
        )
        ledger = pd.read_csv(
            root /
            "reports" /
            "replay_progress.csv"
        )
        assert len(ledger) == 1
        assert (
            int(ledger.iloc[0]["valid_windows"])
            == 3
        )

    print("SELF-TEST PASS")
    print(
        "Validated prospective 25-bp policy selection, "
        "5/10/25bp + 60s contract, causal-field whitelisting, "
        "legacy 35-bp interface normalization, and durable "
        "progress/heartbeat ledger behavior."
    )
    return 0


def parse_args(
    argv: Sequence[str] | None = None,
):
    p = argparse.ArgumentParser(
        description=(
            "Phase 7M.3 prospective 25-bp exact "
            "participant-interaction substrate"
        )
    )
    p.add_argument(
        "--symbol",
        default="BTCUSDT",
    )
    p.add_argument(
        "--phase7m2-root",
        type=Path,
        default=DEFAULT_PHASE7M2_ROOT,
    )
    p.add_argument(
        "--phase7i4-script",
        type=Path,
        default=DEFAULT_PHASE7I4_SCRIPT,
    )
    p.add_argument(
        "--canonical-root",
        type=Path,
        default=DEFAULT_CANONICAL_ROOT,
    )
    p.add_argument(
        "--execution-root",
        type=Path,
        default=DEFAULT_EXECUTION_ROOT,
    )
    p.add_argument(
        "--phase6a-script",
        type=Path,
        default=DEFAULT_PHASE6A_SCRIPT,
    )
    p.add_argument(
        "--phase7a-script",
        type=Path,
        default=DEFAULT_PHASE7A_SCRIPT,
    )
    p.add_argument(
        "--phase7e-script",
        type=Path,
        default=DEFAULT_PHASE7E_SCRIPT,
    )
    p.add_argument(
        "--phase7i1-script",
        type=Path,
        default=DEFAULT_PHASE7I1_SCRIPT,
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    p.add_argument(
        "--report-root",
        type=Path,
        default=DEFAULT_REPORT_ROOT,
    )
    p.add_argument(
        "--heartbeat-sec",
        type=float,
        default=DEFAULT_HEARTBEAT_SEC,
        help=(
            "Console/progress.json heartbeat interval while a "
            "candidate date is being replayed. 0 disables."
        ),
    )
    p.add_argument(
        "--force-day",
        action="store_true",
        help=(
            "Force rebuilding per-candidate-date checkpoints "
            "even when frozen signatures match."
        ),
    )
    p.add_argument(
        "--max-candidate-dates",
        type=int,
        default=None,
        help=(
            "Smoke-test only: process first N candidate dates. "
            "Those valid date checkpoints are reusable by the "
            "later full run."
        ),
    )
    p.add_argument(
        "--self-test",
        action="store_true",
    )
    return p.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = parse_args(argv)

    if args.self_test:
        return self_test()

    if args.symbol != "BTCUSDT":
        raise SystemExit(
            "Phase 7M.3 is frozen to BTCUSDT."
        )
    if args.heartbeat_sec < 0:
        raise SystemExit(
            "--heartbeat-sec must be >= 0."
        )
    if (
        args.max_candidate_dates is not None
        and args.max_candidate_dates <= 0
    ):
        raise SystemExit(
            "--max-candidate-dates must be positive."
        )

    pd, _ = require_stack()

    print(
        "Building Phase 7M.3 prospective 25-bp "
        "exact participant-interaction substrate..."
    )
    print(f"Builder version       : {BUILDER_VERSION}")
    print(f"Prospective policy    : {FROZEN_POLICY}")
    print("Candidate geometry    : 5bp reset -> 10bp release -> 25bp candidate -> 60s rearm")
    print("Observation horizons  : 1,3,5,10s (primary downstream=3s)")
    print("Population             : ALL prospective 25bp attempts, including fail-before35")
    print("Raw replay             : EXACT DEPTH + EXECUTED TRADES")
    print(
        f"Progress heartbeat     : "
        f"{args.heartbeat_sec:g}s"
    )
    print("Scientific role        : SUBSTRATE ONLY; exact feature/outcome analysis deferred")

    print(
        "[1/6] Loading and freezing reviewed "
        "Phase 7M.2 25-bp population..."
    )
    (
        selected,
        selected_outcomes,
        m2_summary,
        freeze_summary,
        freeze_manifest,
        h,
    ) = load_phase7m2_freeze(
        args.phase7m2_root
    )

    up = int(
        (
            pd.to_numeric(
                selected["direction"],
                errors="coerce",
            ) > 0
        ).sum()
    )
    down = int(
        (
            pd.to_numeric(
                selected["direction"],
                errors="coerce",
            ) < 0
        ).sum()
    )
    all_dates = sorted(
        selected["candidate_date"]
        .astype(str)
        .unique()
        .tolist()
    )
    scope_dates = list(all_dates)
    if args.max_candidate_dates is not None:
        scope_dates = scope_dates[
            : max(
                0,
                int(args.max_candidate_dates),
            )
        ]
    scope = selected[
        selected["candidate_date"]
        .astype(str)
        .isin(scope_dates)
    ].copy()

    print(
        f"      full candidates={len(selected):,} "
        f"({up:,} UP / {down:,} DOWN), "
        f"dates={len(all_dates)}"
    )
    if len(scope_dates) < len(all_dates):
        print(
            f"      SMOKE scope={len(scope):,} candidates "
            f"across first {len(scope_dates)} date(s)"
        )
    print(
        f"      candidate hash={h[:16]}..."
    )

    print(
        "[2/6] Loading and validating frozen "
        "Phase 7I.4 exact replay engine..."
    )
    engine = load_module(
        args.phase7i4_script,
        "phase7i4_frozen_for_7m3",
    )

    tracker = ProgressTracker(
        output_root=args.output_root,
        report_root=args.report_root,
        scope_dates=scope_dates,
        scope_candidates=scope,
        heartbeat_sec=args.heartbeat_sec,
    )

    up, down = patch_engine(
        engine,
        selected=selected,
        selected_outcomes=selected_outcomes,
        freeze_summary=freeze_summary,
        freeze_manifest=freeze_manifest,
        candidate_id_hash=h,
        tracker=tracker,
    )

    print(
        "[3/6] Starting exact per-date replay. "
        "Watch [7M.3 HEARTBEAT] lines for live progress..."
    )
    engine_argv = [
        "--symbol",
        args.symbol,
        "--candidate-root",
        str(args.phase7m2_root),
        "--canonical-root",
        str(args.canonical_root),
        "--execution-root",
        str(args.execution_root),
        "--phase6a-script",
        str(args.phase6a_script),
        "--phase7a-script",
        str(args.phase7a_script),
        "--phase7e-script",
        str(args.phase7e_script),
        "--phase7i1-script",
        str(args.phase7i1_script),
        "--output-root",
        str(args.output_root),
        "--report-root",
        str(args.report_root),
        "--price-scale",
        str(PRICE_SCALE),
        "--max-book-age-ms",
        str(MAX_BOOK_AGE_MS),
    ]
    if args.force_day:
        engine_argv.append("--force-day")
    if args.max_candidate_dates is not None:
        engine_argv.extend(
            [
                "--max-candidate-dates",
                str(args.max_candidate_dates),
            ]
        )

    try:
        engine_status = int(
            engine.main(engine_argv) or 0
        )
    except BaseException as exc:
        tracker.fail_date(exc)
        raise

    print(
        "[4/6] Normalizing final aggregate "
        "25-bp semantics..."
    )
    print(
        "[5/6] Running Phase 7M.3-specific "
        "integrity/data-separation checks..."
    )
    final_status = rewrite_metadata_and_reports(
        engine=engine,
        output_root=args.output_root,
        report_root=args.report_root,
        phase7m2_root=args.phase7m2_root,
        phase7i4_script=args.phase7i4_script,
        selected=selected,
        selected_outcomes=selected_outcomes,
        m2_summary=m2_summary,
        freeze_summary=freeze_summary,
        freeze_manifest=freeze_manifest,
        candidate_id_hash=h,
        up=up,
        down=down,
        engine_status=engine_status,
        max_candidate_dates=args.max_candidate_dates,
        tracker=tracker,
    )

    print("[6/6] Phase 7M.3 summary")
    latest = json.loads(
        (
            args.output_root /
            "metadata" /
            "latest_summary.json"
        ).read_text(encoding="utf-8")
    )
    print(
        f"      status={latest.get('status')}"
    )
    print(
        f"      scope candidates="
        f"{int(latest.get('candidate_count') or 0):,}"
    )
    print(
        f"      mechanics rows="
        f"{int(latest.get('mechanics_rows') or 0):,}"
    )
    print(
        f"      valid mechanics="
        f"{int(latest.get('valid_mechanics_rows') or 0):,}"
    )
    print(
        f"      BUILT dates="
        f"{int(latest.get('built_candidate_date_checkpoints') or 0)}, "
        f"REUSED dates="
        f"{int(latest.get('reused_candidate_date_checkpoints') or 0)}"
    )
    print(
        f"      progress={args.output_root / 'metadata' / 'progress.json'}"
    )
    print(
        f"      reports={args.report_root}"
    )
    print(
        "      next scientific step="
        "cheap Phase 7M.4 exact-mechanism analysis; "
        "do not infer exact feature efficacy from this substrate alone."
    )

    return final_status


if __name__ == "__main__":
    raise SystemExit(main())
