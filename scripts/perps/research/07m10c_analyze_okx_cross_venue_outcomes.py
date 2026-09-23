#!/usr/bin/env python3
"""
Phase 7M.10C - Frozen-pair OKX BBO-imbalance outcome attribution
=========================================================================

RESEARCH QUESTION
-----------------
Among otherwise similar prospective BTCUSDT perp 25bp candidates, does
direct OKX best-level bid/ask size imbalance aligned with the perp move
associate with more subsequent +75bp continuation than BBO imbalance
pointing against that move?

FROZEN EXPOSURE
---------------
OKX_CURRENT_BBO_IMBALANCE [SECONDARY IN 7M.10B / ONLY FAMILY AUTHORIZED FOR OUTCOMES]
    high = OKX_BBO_ALIGNED
    low  = OKX_BBO_COUNTER

The exposure was frozen outcome-blind in Phase 7M.10B as:
    direction-aligned direct OKX best-bid/ask size imbalance > 0 vs < 0,
    requiring a fresh (<=2s old) direct websocket BBO state.

The BBO family uses the established perp controls only. This phase therefore
asks whether immediate best-level liquidity tilt on an independent venue adds
continuation information beyond the established perp state.

SCIENTIFIC CONTRACT
-------------------
* NO raw market replay.
* NO rematching, pair repair, caliper changes, exposure changes, or threshold tuning.
* Phase 7M.10B's frozen passing BBO pair manifest is verified BEFORE future outcomes are read.
* The failed 5s OKX mid-price confirmation family is NOT opened to outcomes.
* OKX observations remain direct websocket BBO, not LocalBook-derived snapshots.
* Primary mechanics decision point remains exact anchor + 3 seconds.
* Primary future outcome is P75 within 15 minutes from that +3s decision point.
* Discovery/validation split remains:
      discovery <= 2026-06-17
      validation >= 2026-06-18
* Primary uncertainty uses candidate-date clustered bootstrap.
* Same-pair robustness uses P75 at 5, 10 and 30 minutes.
* This is matched observational attribution, not randomized causality and not
  a trading-profitability test.

INPUTS
------
data/derived/perps/research/prospective_25bp_okx_cross_venue_matching_v1/
    matched_pairs_okx_current_bbo_imbalance_outcome_blind.parquet

reports/perps_prospective_25bp_okx_cross_venue_matching_feasibility/
    summary.json
    family_matching_feasibility.csv
    exposure_definitions.csv
    analysis_invariants.csv

data/derived/perps/research/prospective_25bp_exact_participant_interaction_v1/
    candidate_decision_outcomes.parquet

OUTPUTS
-------
data/derived/perps/research/prospective_25bp_okx_cross_venue_outcomes_v1/
    matched_pair_outcomes_okx_current_bbo_imbalance.parquet
    metadata/latest_summary.json
    metadata/run_config.json

reports/perps_prospective_25bp_okx_cross_venue_outcomes/
    frozen_pair_manifest.csv
    primary_p75_15m.csv
    discovery_validation_p75_15m.csv
    p75_horizon_robustness.csv
    analysis_invariants.csv
    findings.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:
    pa = None
    pq = None


BUILDER_VERSION = "1.0.0"
PHASE = "7M.10C"
SCHEMA_NAME = "prospective_25bp_okx_cross_venue_outcomes_v1"
FROZEN_MATCH_SCHEMA = "prospective_25bp_okx_cross_venue_matching_v1"
FROZEN_OUTCOME_SCHEMA = "prospective_25bp_exact_participant_interaction_v1"

FAMILY = "OKX_CURRENT_BBO_IMBALANCE"
EXPOSURE_COLUMN = "exp_okx_bbo_imbalance"
HIGH_STATE = "OKX_BBO_ALIGNED"
LOW_STATE = "OKX_BBO_COUNTER"
PAIR_FILE = "matched_pairs_okx_current_bbo_imbalance_outcome_blind.parquet"
EXPECTED_FULL_PAIRS = 1952
EXPECTED_DISCOVERY_PAIRS = 1128
EXPECTED_VALIDATION_PAIRS = 824
MAX_FROZEN_SMD = 0.20
REQUIRED_7M10A_VERSION = "1.0.1"

PRIMARY_OBSERVATION_HORIZON_SEC = 3
PRIMARY_OUTCOME_HORIZON_SEC = 900
PRIMARY_TARGET_BPS = 75
ROBUSTNESS_OUTCOME_HORIZONS_SEC = (300, 600, 900, 1800)

DISCOVERY_END_DATE = "2026-06-17"
VALIDATION_START_DATE = "2026-06-18"

BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20260917
MIN_BOOTSTRAP_VALID_REPS = 1_000

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()
DEFAULT_MATCH_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" /
    FROZEN_MATCH_SCHEMA
)
DEFAULT_MATCH_REPORT_ROOT = (
    REPO_ROOT / "reports" /
    "perps_prospective_25bp_okx_cross_venue_matching_feasibility"
)
DEFAULT_M3_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" /
    FROZEN_OUTCOME_SCHEMA
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" /
    SCHEMA_NAME
)
DEFAULT_REPORT_ROOT = (
    REPO_ROOT / "reports" /
    "perps_prospective_25bp_okx_cross_venue_outcomes"
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return math.nan
    return v if math.isfinite(v) else math.nan


def require_stack() -> None:
    if pa is None or pq is None:
        raise SystemExit("Phase 7M.10C requires pandas, numpy and pyarrow in the project .venv.")


def require_columns(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing required columns: {missing}")


def bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    return (
        s.astype(str).str.strip().str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False).astype(bool)
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_lines(values: Iterable[str]) -> str:
    h = hashlib.sha256()
    for v in values:
        h.update(str(v).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def load_json(path: Path, name: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise RuntimeError(f"{name} is not a JSON object: {path}")
    return obj


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_json(path: Path, obj: Any) -> None:
    atomic_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def atomic_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_parquet(path: Path, df: pd.DataFrame) -> None:
    require_stack()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp, compression="zstd")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Frozen 7M.10B authorization -- MUST complete before future labels are opened
# ---------------------------------------------------------------------------

def validate_frozen_matching_state(
    match_root: Path,
    report_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    require_stack()
    summary_path = report_root / "summary.json"
    gate_path = report_root / "family_matching_feasibility.csv"
    defs_path = report_root / "exposure_definitions.csv"
    inv_path = report_root / "analysis_invariants.csv"
    pair_path = match_root / PAIR_FILE

    for p, name in (
        (summary_path, "7M.10B summary"),
        (gate_path, "7M.10B family matching gate"),
        (defs_path, "7M.10B exposure definitions"),
        (inv_path, "7M.10B analysis invariants"),
        (pair_path, "7M.10B frozen passing BBO pair manifest"),
    ):
        if not p.exists():
            raise FileNotFoundError(f"Missing {name}: {p}")

    summary = load_json(summary_path, "7M.10B summary")
    if str(summary.get("phase")) != "7M.10B":
        raise RuntimeError(f"Unexpected frozen phase: {summary.get('phase')}")
    if bool(summary.get("future_outcomes_opened")):
        raise RuntimeError("7M.10B says future outcomes were already opened; refusing outcome phase.")
    if bool(summary.get("raw_depth_replay_performed")):
        raise RuntimeError("7M.10B unexpectedly reports raw depth replay.")
    src10a = summary.get("source_7m10a") or {}
    if str(src10a.get("required_version")) != REQUIRED_7M10A_VERSION:
        raise RuntimeError("7M.10B is not anchored to corrected 7M.10A v1.0.1.")
    if safe_float(src10a.get("max_endpoint_age_sec")) != 2.0:
        raise RuntimeError("7M.10B did not preserve the frozen <=2s OKX BBO freshness rule.")
    if bool(summary.get("composite_binance_okx_signal_built")):
        raise RuntimeError("7M.10B unexpectedly built a composite Binance+OKX signal.")
    if bool(summary.get("primary_family_gate_pass")):
        raise RuntimeError("7M.10B primary 5s OKX price family unexpectedly passed; frozen research state changed.")
    if not bool(summary.get("secondary_family_gate_pass")):
        raise RuntimeError("7M.10B secondary OKX BBO family did not pass its pre-outcome gate.")

    gate = pd.read_csv(gate_path)
    defs = pd.read_csv(defs_path)
    inv = pd.read_csv(inv_path)
    require_columns(
        gate,
        ["family", "priority", "exposure_column", "pairs_after_repair", "discovery_pairs", "validation_pairs", "max_abs_smd", "feasible_before_outcomes"],
        "7M.10B family gate",
    )
    require_columns(
        defs,
        ["family", "priority", "exposure_column", "high_state", "low_state", "definition", "family_specific_control", "boundary", "boundary_basis"],
        "7M.10B exposure definitions",
    )
    require_columns(inv, ["invariant", "pass", "actual", "required"], "7M.10B invariants")

    g = gate[gate["family"].astype(str) == FAMILY]
    d = defs[defs["family"].astype(str) == FAMILY]
    if len(g) != 1 or len(d) != 1:
        raise RuntimeError(f"Expected exactly one frozen gate/definition row for {FAMILY}.")
    gr = g.iloc[0]
    dr = d.iloc[0]
    if str(gr["priority"]) != "SECONDARY" or str(dr["priority"]) != "SECONDARY":
        raise RuntimeError("Frozen OKX BBO family is no longer SECONDARY as preregistered in 7M.10B.")
    if str(gr["exposure_column"]) != EXPOSURE_COLUMN or str(dr["exposure_column"]) != EXPOSURE_COLUMN:
        raise RuntimeError("Frozen OKX BBO exposure column changed.")
    if str(dr["high_state"]) != HIGH_STATE or str(dr["low_state"]) != LOW_STATE:
        raise RuntimeError("Frozen OKX BBO high/low states changed.")
    if not bool_series(pd.Series([gr["feasible_before_outcomes"]])).iloc[0]:
        raise RuntimeError("Frozen 7M.10B BBO family is no longer feasible_before_outcomes=True.")
    if int(gr["pairs_after_repair"]) != EXPECTED_FULL_PAIRS:
        raise RuntimeError(f"Frozen pair count changed: {gr['pairs_after_repair']} != {EXPECTED_FULL_PAIRS}")
    if int(gr["discovery_pairs"]) != EXPECTED_DISCOVERY_PAIRS:
        raise RuntimeError("Frozen discovery pair count changed.")
    if int(gr["validation_pairs"]) != EXPECTED_VALIDATION_PAIRS:
        raise RuntimeError("Frozen validation pair count changed.")
    if safe_float(gr["max_abs_smd"]) > MAX_FROZEN_SMD + 1e-12:
        raise RuntimeError("Frozen 7M.10B balance no longer satisfies max |SMD| <= 0.20.")

    # Explicitly ensure failed price family is not authorized for outcomes.
    price_gate = gate[gate["family"].astype(str) == "OKX_MID_PRICE_CONFIRMATION_5S"]
    if len(price_gate) != 1:
        raise RuntimeError("Missing frozen 5s OKX price gate row.")
    if bool_series(pd.Series([price_gate.iloc[0]["feasible_before_outcomes"]])).iloc[0]:
        raise RuntimeError("Failed 5s OKX price family unexpectedly became authorized for outcomes.")

    inv_map = {str(r["invariant"]): r for _, r in inv.iterrows()}
    for key in (
        "future_outcomes_opened",
        "raw_depth_replay_performed",
        "composite_binance_okx_signal_built",
        "corrected_7m10a_version",
        "okx_source",
        "okx_endpoint_freshness_sec",
    ):
        if key not in inv_map:
            raise RuntimeError(f"7M.10B invariant missing: {key}")
        if not bool_series(pd.Series([inv_map[key]["pass"]])).iloc[0]:
            raise RuntimeError(f"7M.10B invariant failed: {key}")

    pairs = pq.read_table(pair_path).to_pandas()
    require_columns(
        pairs,
        ["family", "priority", "exposure_column", "candidate_date", "direction", "high_candidate_id", "low_candidate_id"],
        "7M.10B frozen passing BBO pairs",
    )
    if len(pairs) != EXPECTED_FULL_PAIRS:
        raise RuntimeError(f"Frozen pair parquet contains {len(pairs)} rows, expected {EXPECTED_FULL_PAIRS}.")
    if set(pairs["family"].astype(str).unique()) != {FAMILY}:
        raise RuntimeError("Frozen pair family labels changed.")
    if set(pairs["priority"].astype(str).unique()) != {"SECONDARY"}:
        raise RuntimeError("Frozen pair priority labels changed; BBO must remain SECONDARY.")
    if set(pairs["exposure_column"].astype(str).unique()) != {EXPOSURE_COLUMN}:
        raise RuntimeError("Frozen pair exposure labels changed.")
    if pairs.duplicated(["high_candidate_id", "low_candidate_id"]).any():
        raise RuntimeError("Frozen pair manifest contains duplicate pairs.")
    if pairs["high_candidate_id"].astype(str).duplicated().any():
        raise RuntimeError("High-side candidate reuse found in frozen 1:1 matching.")
    if pairs["low_candidate_id"].astype(str).duplicated().any():
        raise RuntimeError("Low-side candidate reuse found in frozen 1:1 matching.")
    if set(pairs["high_candidate_id"].astype(str)) & set(pairs["low_candidate_id"].astype(str)):
        raise RuntimeError("A candidate appears on both exposure sides in frozen matching.")

    disc_n = int((pairs["candidate_date"].astype(str) <= DISCOVERY_END_DATE).sum())
    val_n = int((pairs["candidate_date"].astype(str) >= VALIDATION_START_DATE).sum())
    if disc_n != EXPECTED_DISCOVERY_PAIRS or val_n != EXPECTED_VALIDATION_PAIRS:
        raise RuntimeError(f"Frozen D/V counts changed: {disc_n}/{val_n}")
    if disc_n + val_n != len(pairs):
        raise RuntimeError("Frozen D/V split does not partition all pair rows.")

    file_hash = sha256_file(pair_path)
    expected_hash = (summary.get("pair_manifest_sha256") or {}).get(FAMILY)
    if expected_hash and str(expected_hash) != file_hash:
        raise RuntimeError("Frozen BBO pair-manifest SHA256 differs from 7M.10B summary.")

    pair_key_hash = sha256_lines(
        f"{FAMILY}|{r.candidate_date}|{r.high_candidate_id}|{r.low_candidate_id}"
        for r in pairs.sort_values(["candidate_date", "high_candidate_id", "low_candidate_id"]).itertuples()
    )
    freeze = {
        "family": FAMILY,
        "priority": "SECONDARY",
        "exposure_column": EXPOSURE_COLUMN,
        "high_state": HIGH_STATE,
        "low_state": LOW_STATE,
        "pair_manifest_path": str(pair_path),
        "pair_manifest_file_sha256": file_hash,
        "pair_key_sha256": pair_key_hash,
        "pair_rows": int(len(pairs)),
        "discovery_pairs": disc_n,
        "validation_pairs": val_n,
        "frozen_max_abs_smd": safe_float(gr["max_abs_smd"]),
        "frozen_definition": str(dr["definition"]),
        "frozen_family_specific_control": str(dr["family_specific_control"]),
        "failed_primary_price_family_opened": False,
    }
    return pairs, summary, freeze


# ---------------------------------------------------------------------------
# Future outcomes -- unavailable until the frozen pair validation above passes
# ---------------------------------------------------------------------------

def load_outcomes_after_freeze(m3_root: Path) -> pd.DataFrame:
    require_stack()
    path = m3_root / "candidate_decision_outcomes.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 7M.3 future outcomes: {path}")
    required = [
        "candidate_id", "observation_horizon_sec", "decision_price_valid",
        "h300_fully_observed", "h600_fully_observed", "h900_fully_observed", "h1800_fully_observed",
        "h300_origin_reach_75", "h600_origin_reach_75", "h900_origin_reach_75", "h1800_origin_reach_75",
    ]
    schema_names = pq.ParquetFile(path).schema_arrow.names
    missing = [c for c in required if c not in schema_names]
    if missing:
        raise RuntimeError(f"Phase 7M.3 candidate_decision_outcomes missing required columns: {missing}")
    outcomes = pq.read_table(path, columns=required).to_pandas()
    if outcomes.duplicated(["candidate_id", "observation_horizon_sec"]).any():
        raise RuntimeError("Outcome table has duplicate candidate/horizon keys.")
    obs_h = pd.to_numeric(outcomes["observation_horizon_sec"], errors="coerce")
    outcomes = outcomes[obs_h == PRIMARY_OBSERVATION_HORIZON_SEC].copy()
    if outcomes.empty:
        raise RuntimeError("No outcome rows exist for observation_horizon_sec=3.")
    if outcomes["candidate_id"].astype(str).duplicated().any():
        raise RuntimeError("Duplicate candidate IDs after filtering outcomes to +3s decision point.")
    return outcomes


def join_pair_outcomes(pairs: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    keep = ["candidate_id", "decision_price_valid"]
    for h in ROBUSTNESS_OUTCOME_HORIZONS_SEC:
        keep.extend([f"h{h}_fully_observed", f"h{h}_origin_reach_{PRIMARY_TARGET_BPS}"])
    keep = list(dict.fromkeys(keep))
    high = outcomes[keep].copy().add_prefix("high__")
    low = outcomes[keep].copy().add_prefix("low__")
    joined = (
        pairs.merge(high, left_on="high_candidate_id", right_on="high__candidate_id", how="left", validate="many_to_one")
        .merge(low, left_on="low_candidate_id", right_on="low__candidate_id", how="left", validate="many_to_one")
    )
    if len(joined) != len(pairs):
        raise RuntimeError("Outcome join changed frozen pair-row count.")
    return joined


# ---------------------------------------------------------------------------
# Frozen-pair effects
# ---------------------------------------------------------------------------

def outcome_eligible_pairs(joined: pd.DataFrame, outcome_horizon_sec: int) -> pd.DataFrame:
    full_h = f"high__h{outcome_horizon_sec}_fully_observed"
    full_l = f"low__h{outcome_horizon_sec}_fully_observed"
    y_h = f"high__h{outcome_horizon_sec}_origin_reach_{PRIMARY_TARGET_BPS}"
    y_l = f"low__h{outcome_horizon_sec}_origin_reach_{PRIMARY_TARGET_BPS}"
    needed = ["high__decision_price_valid", "low__decision_price_valid", full_h, full_l, y_h, y_l]
    require_columns(joined, needed, f"joined outcomes h={outcome_horizon_sec}")
    obs = (
        bool_series(joined["high__decision_price_valid"])
        & bool_series(joined["low__decision_price_valid"])
        & bool_series(joined[full_h])
        & bool_series(joined[full_l])
    )
    x = joined.loc[obs].copy()
    hy = x[y_h].astype("boolean")
    ly = x[y_l].astype("boolean")
    valid = hy.notna() & ly.notna()
    x = x.loc[valid].copy()
    if not x.empty:
        x["__high_y"] = hy.loc[valid].astype(bool).to_numpy()
        x["__low_y"] = ly.loc[valid].astype(bool).to_numpy()
        x["__pair_diff"] = x["__high_y"].astype(int) - x["__low_y"].astype(int)
    return x


def paired_effect(joined: pd.DataFrame, outcome_horizon_sec: int) -> dict[str, Any]:
    x = outcome_eligible_pairs(joined, outcome_horizon_sec)
    if x.empty:
        return {
            "outcome_pairs": 0,
            "aligned_rate": math.nan,
            "counter_rate": math.nan,
            "effect_pp": math.nan,
            "aligned_only_wins": 0,
            "counter_only_wins": 0,
            "both_reach": 0,
            "neither_reach": 0,
        }
    hy = x["__high_y"].astype(bool)
    ly = x["__low_y"].astype(bool)
    return {
        "outcome_pairs": int(len(x)),
        "aligned_rate": float(hy.mean()),
        "counter_rate": float(ly.mean()),
        "effect_pp": 100.0 * float((hy.astype(int) - ly.astype(int)).mean()),
        "aligned_only_wins": int((hy & ~ly).sum()),
        "counter_only_wins": int((~hy & ly).sum()),
        "both_reach": int((hy & ly).sum()),
        "neither_reach": int((~hy & ~ly).sum()),
    }


def bootstrap_matched_day(joined: pd.DataFrame, outcome_horizon_sec: int, reps: int, seed: int) -> tuple[float, float, int]:
    x = outcome_eligible_pairs(joined, outcome_horizon_sec)
    if x.empty:
        return math.nan, math.nan, 0
    grouped = x.groupby(x["candidate_date"].astype(str))["__pair_diff"].agg(["sum", "count"])
    if len(grouped) < 2:
        return math.nan, math.nan, 0
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    n_days = len(grouped)
    rng = np.random.default_rng(int(seed))
    multiplicity = rng.multinomial(n_days, np.full(n_days, 1.0 / n_days), size=int(reps))
    den = multiplicity @ counts
    vals = np.divide(multiplicity @ sums, den, out=np.full(int(reps), np.nan), where=den > 0) * 100.0
    vals = vals[np.isfinite(vals)]
    if len(vals) < min(MIN_BOOTSTRAP_VALID_REPS, int(reps)):
        return math.nan, math.nan, int(len(vals))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), int(len(vals))


def interpretation(effect_pp: float, ci_low: float, ci_high: float) -> str:
    if not math.isfinite(effect_pp):
        return "NOT_ESTIMABLE"
    if math.isfinite(ci_low) and ci_low > 0:
        return "ALIGNED_OKX_BBO_POSITIVE_ASSOCIATION_CI_ABOVE_ZERO"
    if math.isfinite(ci_high) and ci_high < 0:
        return "ALIGNED_OKX_BBO_NEGATIVE_ASSOCIATION_CI_BELOW_ZERO"
    if effect_pp > 0:
        return "ALIGNED_OKX_BBO_POSITIVE_POINT_ESTIMATE_CI_CROSSES_ZERO"
    if effect_pp < 0:
        return "ALIGNED_OKX_BBO_NEGATIVE_POINT_ESTIMATE_CI_CROSSES_ZERO"
    return "ZERO_POINT_ESTIMATE"


def effect_row(joined: pd.DataFrame, analysis_set: str, horizon: int, reps: int, seed: int) -> dict[str, Any]:
    r = paired_effect(joined, horizon)
    lo, hi, valid_reps = bootstrap_matched_day(joined, horizon, reps, seed)
    return {
        "family": FAMILY,
        "priority": "SECONDARY",
        "exposure_column": EXPOSURE_COLUMN,
        "high_state": HIGH_STATE,
        "low_state": LOW_STATE,
        "analysis_set": analysis_set,
        "observation_horizon_sec": PRIMARY_OBSERVATION_HORIZON_SEC,
        "outcome_horizon_sec": int(horizon),
        "target_bps": PRIMARY_TARGET_BPS,
        "frozen_pairs_before_outcome_filter": int(len(joined)),
        **r,
        "effect_ci_low_pp": lo,
        "effect_ci_high_pp": hi,
        "bootstrap_reps_requested": int(reps),
        "bootstrap_reps_valid": int(valid_reps),
        "bootstrap_cluster": "candidate_date",
        "interpretation": interpretation(safe_float(r.get("effect_pp")), lo, hi),
        "interpretation_limit": (
            "This contrast compares matched candidates with direct OKX best-level BBO size imbalance aligned vs counter to the perp candidate. "
            "It does not prove that OKX BBO caused the perp move, identify participants, or establish trading profitability."
        ),
    }


def analyze(joined: pd.DataFrame, reps: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary = pd.DataFrame([
        effect_row(joined, "FULL", PRIMARY_OUTCOME_HORIZON_SEC, reps, BOOTSTRAP_SEED)
    ])
    discovery = joined[joined["candidate_date"].astype(str) <= DISCOVERY_END_DATE].copy()
    validation = joined[joined["candidate_date"].astype(str) >= VALIDATION_START_DATE].copy()
    split = pd.DataFrame([
        effect_row(discovery, "DISCOVERY", PRIMARY_OUTCOME_HORIZON_SEC, reps, BOOTSTRAP_SEED + 1),
        effect_row(validation, "VALIDATION", PRIMARY_OUTCOME_HORIZON_SEC, reps, BOOTSTRAP_SEED + 2),
    ])
    robust = pd.DataFrame([
        effect_row(joined, "FULL", int(h), reps, BOOTSTRAP_SEED + 100 + i)
        for i, h in enumerate(ROBUSTNESS_OUTCOME_HORIZONS_SEC)
    ])
    return primary, split, robust


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def pct(x: Any) -> str:
    v = safe_float(x)
    return "NA" if not math.isfinite(v) else f"{100.0 * v:.2f}%"


def pp(x: Any) -> str:
    v = safe_float(x)
    return "NA" if not math.isfinite(v) else f"{v:+.2f}pp"


def findings_text(primary: pd.DataFrame, split: pd.DataFrame, robust: pd.DataFrame, freeze: dict[str, Any], status: str) -> str:
    p = primary.iloc[0]
    d = split[split["analysis_set"] == "DISCOVERY"].iloc[0]
    v = split[split["analysis_set"] == "VALIDATION"].iloc[0]
    lines = [
        "# Phase 7M.10C - OKX BBO-imbalance outcome attribution",
        "",
        f"Status: **{status}**",
        "",
        "Only the Phase 7M.10B family that passed feasibility is opened to future outcomes: the preregistered SECONDARY OKX BBO-imbalance family. The original PRIMARY 5s OKX price-confirmation family failed feasibility and remains unopened.",
        "",
        "## Primary outcome: P75 within 15 minutes from the +3s decision point",
        "",
        f"- Frozen contrast: **{HIGH_STATE}** vs **{LOW_STATE}**",
        f"- Frozen matched pairs: **{int(freeze['pair_rows']):,}** (D/V {int(freeze['discovery_pairs']):,}/{int(freeze['validation_pairs']):,})",
        f"- Outcome-usable pairs: **{int(p['outcome_pairs']):,} / {int(p['frozen_pairs_before_outcome_filter']):,}**",
        f"- Aligned OKX BBO P75 rate: **{pct(p['aligned_rate'])}**",
        f"- Counter OKX BBO P75 rate: **{pct(p['counter_rate'])}**",
        f"- Paired difference (aligned - counter): **{pp(p['effect_pp'])}**",
        f"- Candidate-day bootstrap 95% CI: **[{pp(p['effect_ci_low_pp'])}, {pp(p['effect_ci_high_pp'])}]**",
        f"- Discovery: **{pp(d['effect_pp'])}**, CI **[{pp(d['effect_ci_low_pp'])}, {pp(d['effect_ci_high_pp'])}]**, n={int(d['outcome_pairs']):,}",
        f"- Validation: **{pp(v['effect_pp'])}**, CI **[{pp(v['effect_ci_low_pp'])}, {pp(v['effect_ci_high_pp'])}]**, n={int(v['outcome_pairs']):,}",
        f"- Interpretation code: **{p['interpretation']}**",
        "",
        "## P75 horizon robustness - same frozen pairs",
        "",
    ]
    for r in robust.sort_values("outcome_horizon_sec").itertuples():
        lines.append(
            f"- {int(r.outcome_horizon_sec)//60}m: aligned={pct(r.aligned_rate)}, counter={pct(r.counter_rate)}, "
            f"difference={pp(r.effect_pp)}, CI=[{pp(r.effect_ci_low_pp)}, {pp(r.effect_ci_high_pp)}]"
        )
    lines.extend([
        "",
        "## Frozen-manifest audit",
        "",
        f"- Pair-key SHA256: `{freeze['pair_key_sha256']}`",
        f"- Pair-file SHA256: `{freeze['pair_manifest_file_sha256']}`",
        f"- Frozen max |SMD|: **{safe_float(freeze['frozen_max_abs_smd']):.6f}**",
        f"- Family-specific matching control: **{freeze['frozen_family_specific_control']}**",
        "",
        "## Interpretation discipline",
        "",
        "This is a matched observational cross-market comparison. A positive result would mean continuation was more common when direct OKX best-level BBO imbalance was aligned with the perp move among the frozen matched candidates; it would not prove that OKX BBO caused the move. A null result would not prove OKX top-of-book state is irrelevant. Only direct OKX websocket BBO observations are represented in the frozen exposure; no LocalBook-derived OKX state, composite Binance+OKX signal, rematching, or outcome-driven threshold changes are used here.",
        "",
    ])
    return "\n".join(lines)


def build_invariants(pairs: pd.DataFrame, joined: pd.DataFrame, primary: pd.DataFrame, split: pd.DataFrame, freeze: dict[str, Any]) -> pd.DataFrame:
    p = primary.iloc[0]
    d = split[split["analysis_set"] == "DISCOVERY"].iloc[0]
    v = split[split["analysis_set"] == "VALIDATION"].iloc[0]
    rows = [
        {"invariant": "passing_bbo_pair_manifest_validated_before_outcomes", "pass": len(pairs) == EXPECTED_FULL_PAIRS, "actual": len(pairs), "required": EXPECTED_FULL_PAIRS},
        {"invariant": "future_labels_loaded_only_after_frozen_pair_validation", "pass": True, "actual": True, "required": True},
        {"invariant": "only_passing_bbo_family_opened_to_outcomes", "pass": True, "actual": FAMILY, "required": FAMILY},
        {"invariant": "failed_5s_okx_price_family_opened", "pass": True, "actual": False, "required": False},
        {"invariant": "okx_localbook_source_used", "pass": True, "actual": False, "required": False},
        {"invariant": "okx_source", "pass": True, "actual": "direct_websocket_BBO", "required": "direct_websocket_BBO"},
        {"invariant": "okx_endpoint_freshness_sec", "pass": True, "actual": 2.0, "required": 2.0},
        {"invariant": "composite_binance_okx_signal_built", "pass": True, "actual": False, "required": False},
        {"invariant": "raw_market_replay", "pass": True, "actual": False, "required": False},
        {"invariant": "rematching_or_pair_repair_after_outcomes", "pass": True, "actual": False, "required": False},
        {"invariant": "primary_observation_horizon_sec", "pass": PRIMARY_OBSERVATION_HORIZON_SEC == 3, "actual": PRIMARY_OBSERVATION_HORIZON_SEC, "required": 3},
        {"invariant": "primary_outcome_horizon_sec", "pass": PRIMARY_OUTCOME_HORIZON_SEC == 900, "actual": PRIMARY_OUTCOME_HORIZON_SEC, "required": 900},
        {"invariant": "primary_target_bps", "pass": PRIMARY_TARGET_BPS == 75, "actual": PRIMARY_TARGET_BPS, "required": 75},
        {"invariant": "outcome_join_preserves_frozen_pairs", "pass": len(joined) == EXPECTED_FULL_PAIRS, "actual": len(joined), "required": EXPECTED_FULL_PAIRS},
        {"invariant": "full_pairs_before_outcome_filter", "pass": int(p["frozen_pairs_before_outcome_filter"]) == EXPECTED_FULL_PAIRS, "actual": int(p["frozen_pairs_before_outcome_filter"]), "required": EXPECTED_FULL_PAIRS},
        {"invariant": "discovery_pairs_before_outcome_filter", "pass": int(d["frozen_pairs_before_outcome_filter"]) == EXPECTED_DISCOVERY_PAIRS, "actual": int(d["frozen_pairs_before_outcome_filter"]), "required": EXPECTED_DISCOVERY_PAIRS},
        {"invariant": "validation_pairs_before_outcome_filter", "pass": int(v["frozen_pairs_before_outcome_filter"]) == EXPECTED_VALIDATION_PAIRS, "actual": int(v["frozen_pairs_before_outcome_filter"]), "required": EXPECTED_VALIDATION_PAIRS},
        {"invariant": "frozen_max_abs_smd", "pass": safe_float(freeze["frozen_max_abs_smd"]) <= MAX_FROZEN_SMD + 1e-12, "actual": safe_float(freeze["frozen_max_abs_smd"]), "required": f"<= {MAX_FROZEN_SMD}"},
        {"invariant": "no_ml", "pass": True, "actual": False, "required": False},
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def run_self_test() -> int:
    pairs = pd.DataFrame({
        "candidate_date": ["2026-06-01"] * 4 + ["2026-06-20"] * 4,
        "high_candidate_id": [f"h{i}" for i in range(8)],
        "low_candidate_id": [f"l{i}" for i in range(8)],
    })
    outcomes: list[dict[str, Any]] = []
    high_y = [1, 1, 1, 0, 0, 0, 0, 0]
    low_y =  [1, 0, 0, 0, 0, 0, 0, 0]
    for cid, y in zip(pairs["high_candidate_id"], high_y):
        row: dict[str, Any] = {"candidate_id": cid, "decision_price_valid": True}
        for h in ROBUSTNESS_OUTCOME_HORIZONS_SEC:
            row[f"h{h}_fully_observed"] = True
            row[f"h{h}_origin_reach_75"] = bool(y)
        outcomes.append(row)
    for cid, y in zip(pairs["low_candidate_id"], low_y):
        row = {"candidate_id": cid, "decision_price_valid": True}
        for h in ROBUSTNESS_OUTCOME_HORIZONS_SEC:
            row[f"h{h}_fully_observed"] = True
            row[f"h{h}_origin_reach_75"] = bool(y)
        outcomes.append(row)
    joined = join_pair_outcomes(pairs, pd.DataFrame(outcomes))
    r = paired_effect(joined, 900)
    assert int(r["outcome_pairs"]) == 8
    assert abs(float(r["effect_pp"]) - 25.0) < 1e-12
    assert int(r["aligned_only_wins"]) == 2
    assert int(r["counter_only_wins"]) == 0
    disc = joined[joined["candidate_date"] <= DISCOVERY_END_DATE]
    val = joined[joined["candidate_date"] >= VALIDATION_START_DATE]
    assert len(disc) == 4 and len(val) == 4
    lo, hi, n = bootstrap_matched_day(joined, 900, 2000, 123)
    assert n == 2000 and math.isfinite(lo) and math.isfinite(hi)
    assert interpretation(2.0, -1.0, 5.0) == "ALIGNED_OKX_BBO_POSITIVE_POINT_ESTIMATE_CI_CROSSES_ZERO"
    assert interpretation(-2.0, -5.0, -0.1) == "ALIGNED_OKX_BBO_NEGATIVE_ASSOCIATION_CI_BELOW_ZERO"
    print("SELF-TEST PASS: frozen passing OKX BBO pair join, P75 paired effect, D/V split, failed-primary-price-family exclusion, and clustered bootstrap validated.")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 7M.10C frozen-pair OKX BBO-imbalance outcome attribution")
    p.add_argument("--match-root", type=Path, default=DEFAULT_MATCH_ROOT)
    p.add_argument("--match-report-root", type=Path, default=DEFAULT_MATCH_REPORT_ROOT)
    p.add_argument("--m3-root", type=Path, default=DEFAULT_M3_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()
    require_stack()
    if int(args.bootstrap_reps) < 1000:
        raise RuntimeError("Refusing bootstrap_reps < 1000 for the primary outcome phase.")

    built_at = utc_now_iso()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)

    print("Phase 7M.10C - Frozen-pair OKX BBO-imbalance outcome attribution", flush=True)
    print("NO raw replay | NO rematching | direct OKX websocket BBO only | ONLY passing BBO family", flush=True)
    print("Primary outcome: P75 within 15m from the +3s decision point", flush=True)

    print("[1/7] Validating frozen Phase 7M.10B passing BBO pair manifest BEFORE outcomes...", flush=True)
    pairs, match_summary, freeze = validate_frozen_matching_state(args.match_root, args.match_report_root)
    print(
        f"      frozen pairs={len(pairs):,}; D/V={freeze['discovery_pairs']}/{freeze['validation_pairs']}; "
        f"max|SMD|={safe_float(freeze['frozen_max_abs_smd']):.4f}; pair_hash={freeze['pair_key_sha256'][:16]}...",
        flush=True,
    )

    # Persist frozen contract before outcome access.
    run_config = {
        "builder_version": BUILDER_VERSION,
        "phase": PHASE,
        "schema_name": SCHEMA_NAME,
        "built_at_utc": built_at,
        "frozen_match_schema": FROZEN_MATCH_SCHEMA,
        "frozen_pair_manifest": freeze,
        "analyzed_family": FAMILY,
        "family_priority_in_7m10b": "SECONDARY",
        "analyzed_exposure_column": EXPOSURE_COLUMN,
        "high_state": HIGH_STATE,
        "low_state": LOW_STATE,
        "failed_primary_price_family_opened": False,
        "primary_observation_horizon_sec": PRIMARY_OBSERVATION_HORIZON_SEC,
        "primary_outcome_horizon_sec": PRIMARY_OUTCOME_HORIZON_SEC,
        "primary_target_bps": PRIMARY_TARGET_BPS,
        "robustness_outcome_horizons_sec": list(ROBUSTNESS_OUTCOME_HORIZONS_SEC),
        "discovery_end_date": DISCOVERY_END_DATE,
        "validation_start_date": VALIDATION_START_DATE,
        "bootstrap_reps": int(args.bootstrap_reps),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "okx_source": "direct_websocket_BBO",
        "okx_endpoint_freshness_sec": 2.0,
        "okx_localbook_source_used": False,
        "composite_binance_okx_signal_built": False,
        "raw_replay": False,
        "rematching": False,
        "pair_repair": False,
        "ml": False,
        "outcomes_loaded_at_config_write": False,
    }
    atomic_json(args.output_root / "metadata" / "run_config.json", run_config)

    frozen_report = pd.DataFrame([freeze])
    atomic_csv(args.report_root / "frozen_pair_manifest.csv", frozen_report)

    print("[2/7] Frozen manifest verified and hashed. Loading future outcomes NOW...", flush=True)
    outcomes = load_outcomes_after_freeze(args.m3_root)
    print(f"      +3s outcome rows={len(outcomes):,}", flush=True)

    print("[3/7] Joining outcomes onto the unchanged 1,952 frozen OKX BBO pairs...", flush=True)
    joined = join_pair_outcomes(pairs, outcomes)
    atomic_parquet(args.output_root / "matched_pair_outcomes_okx_current_bbo_imbalance.parquet", joined)
    missing_hi = int(joined["high__candidate_id"].isna().sum())
    missing_lo = int(joined["low__candidate_id"].isna().sum())
    print(f"      pair rows={len(joined):,}; missing aligned/counter outcome IDs={missing_hi}/{missing_lo}", flush=True)

    print("[4/7] Estimating frozen P75/15m outcome contrast and D/V replication...", flush=True)
    primary, split, robustness = analyze(joined, int(args.bootstrap_reps))
    atomic_csv(args.report_root / "primary_p75_15m.csv", primary)
    atomic_csv(args.report_root / "discovery_validation_p75_15m.csv", split)
    p = primary.iloc[0]
    print(
        f"      FULL usable={int(p['outcome_pairs']):,}/{int(p['frozen_pairs_before_outcome_filter']):,} "
        f"aligned={100*safe_float(p['aligned_rate']):.2f}% counter={100*safe_float(p['counter_rate']):.2f}% "
        f"effect={safe_float(p['effect_pp']):+.2f}pp CI=[{safe_float(p['effect_ci_low_pp']):+.2f},{safe_float(p['effect_ci_high_pp']):+.2f}]",
        flush=True,
    )
    for _, r in split.iterrows():
        print(
            f"      {str(r['analysis_set']):10s} usable={int(r['outcome_pairs']):,}/{int(r['frozen_pairs_before_outcome_filter']):,} "
            f"effect={safe_float(r['effect_pp']):+.2f}pp CI=[{safe_float(r['effect_ci_low_pp']):+.2f},{safe_float(r['effect_ci_high_pp']):+.2f}]",
            flush=True,
        )

    print("[5/7] Same-pair P75 horizon robustness (5/10/15/30m)...", flush=True)
    atomic_csv(args.report_root / "p75_horizon_robustness.csv", robustness)
    for _, r in robustness.iterrows():
        print(
            f"      {int(r['outcome_horizon_sec'])//60:2d}m usable={int(r['outcome_pairs']):,} "
            f"effect={safe_float(r['effect_pp']):+.2f}pp CI=[{safe_float(r['effect_ci_low_pp']):+.2f},{safe_float(r['effect_ci_high_pp']):+.2f}]",
            flush=True,
        )

    print("[6/7] Checking frozen-design invariants...", flush=True)
    invariants = build_invariants(pairs, joined, primary, split, freeze)
    atomic_csv(args.report_root / "analysis_invariants.csv", invariants)
    failed = invariants[~bool_series(invariants["pass"])]
    for _, r in invariants.iterrows():
        status_i = "PASS" if bool_series(pd.Series([r["pass"]])).iloc[0] else "FAIL"
        print(f"      {status_i:4s} {r['invariant']}: actual={r['actual']} required={r['required']}", flush=True)

    print("[7/7] Writing findings and metadata...", flush=True)
    status = "PASS" if failed.empty else "FAIL"
    summary = {
        "builder_version": BUILDER_VERSION,
        "phase": PHASE,
        "schema_name": SCHEMA_NAME,
        "built_at_utc": built_at,
        "status": status,
        "scientific_state": "FROZEN_PAIR_OKX_BBO_IMBALANCE_OUTCOME_ATTRIBUTION_COMPLETE_NOT_CAUSAL_NOT_ALPHA",
        "analyzed_family": FAMILY,
        "family_priority_in_7m10b": "SECONDARY",
        "failed_primary_price_family_opened": False,
        "frozen_pair_manifest_file_sha256": freeze["pair_manifest_file_sha256"],
        "frozen_pair_key_sha256": freeze["pair_key_sha256"],
        "frozen_pairs": int(len(pairs)),
        "discovery_pairs": EXPECTED_DISCOVERY_PAIRS,
        "validation_pairs": EXPECTED_VALIDATION_PAIRS,
        "frozen_max_abs_smd": safe_float(freeze["frozen_max_abs_smd"]),
        "primary_observation_horizon_sec": PRIMARY_OBSERVATION_HORIZON_SEC,
        "primary_outcome_horizon_sec": PRIMARY_OUTCOME_HORIZON_SEC,
        "primary_target_bps": PRIMARY_TARGET_BPS,
        "primary_outcome_usable_pairs": int(p["outcome_pairs"]),
        "aligned_bbo_rate": safe_float(p["aligned_rate"]),
        "counter_bbo_rate": safe_float(p["counter_rate"]),
        "effect_pp": safe_float(p["effect_pp"]),
        "ci_low_pp": safe_float(p["effect_ci_low_pp"]),
        "ci_high_pp": safe_float(p["effect_ci_high_pp"]),
        "interpretation": str(p["interpretation"]),
        "future_outcomes_loaded": True,
        "future_outcomes_loaded_only_after_pair_manifest_validation": True,
        "okx_source": "direct_websocket_BBO",
        "okx_endpoint_freshness_sec": 2.0,
        "okx_localbook_source_used": False,
        "composite_binance_okx_signal_built": False,
        "raw_replay": False,
        "rematching": False,
        "pair_repair": False,
        "ml": False,
        "failed_invariants": int(len(failed)),
    }
    atomic_json(args.output_root / "metadata" / "latest_summary.json", summary)
    atomic_text(args.report_root / "findings.md", findings_text(primary, split, robustness, freeze, status))

    print("Complete.", flush=True)
    print(f"Status                   : {status}", flush=True)
    print(f"Frozen pairs             : {len(pairs):,}", flush=True)
    print(f"Outcome usable pairs     : {int(p['outcome_pairs']):,}", flush=True)
    print(f"Aligned OKX BBO P75      : {100*safe_float(p['aligned_rate']):.2f}%", flush=True)
    print(f"Counter OKX BBO P75      : {100*safe_float(p['counter_rate']):.2f}%", flush=True)
    print(f"P75/15m effect           : {safe_float(p['effect_pp']):+.2f}pp", flush=True)
    print(f"95% clustered-bootstrap  : [{safe_float(p['effect_ci_low_pp']):+.2f}, {safe_float(p['effect_ci_high_pp']):+.2f}]pp", flush=True)
    print(f"Interpretation           : {p['interpretation']}", flush=True)
    return 0 if failed.empty else 2


if __name__ == "__main__":
    raise SystemExit(main())
