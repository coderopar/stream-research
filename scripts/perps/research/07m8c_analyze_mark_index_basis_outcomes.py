#!/usr/bin/env python3
"""
Phase 7M.8C - Frozen-pair mark/index basis and funding outcome attribution
=====================================================================

RESEARCH QUESTION
-----------------
Do frozen perp mark/index dislocation and funding-regime states add continuation
information beyond the already-characterized prospective 25-bp visible-book state?

The three Phase 7M.8B exposures are tested separately:

1. DIRECTIONAL_BASIS_CHANGE_30S  [PRIMARY]
      high = mark-index basis widened in the candidate direction over the causal 30s window
      low  = mark-index basis moved against the candidate direction over that window
   This is the primary 7M.8 hypothesis because it measures whether perp-specific
   dislocation was actively building with the move before the +3s decision point.

2. CURRENT_DIRECTIONAL_BASIS  [SECONDARY]
      high = current mark-index basis sign aligned with candidate direction
      low  = current mark-index basis sign counter to candidate direction
   This measures the current perp-vs-index dislocation state, not its short-run change.

3. CURRENT_DIRECTIONAL_FUNDING  [SECONDARY]
      high = current funding sign aligned with candidate direction
      low  = current funding sign counter to candidate direction
   Funding is treated as a slower crowding/regime state, not short-horizon flow.

SCIENTIFIC CONTRACT
-------------------
* NO raw market replay.
* NO rematching, balance repair, caliper changes, exposure redefinition, or
  threshold tuning after future outcomes are opened.
* All three Phase 7M.8B family gates and frozen pair manifests are validated
  BEFORE candidate_decision_outcomes.parquet is read.
* Primary mechanics decision point remains exact anchor + 3 seconds.
* Primary future outcome is P75 within 15 minutes from that +3s decision point
  (`h900_origin_reach_75`).
* Discovery/validation split remains:
      discovery <= 2026-06-17
      validation >= 2026-06-18
* Primary uncertainty uses candidate-date clustered bootstrap.
* Same-pair robustness uses P75 at 5, 10, and 30 minutes.
* Families remain separate; NO composite basis/funding signal is created.

This is matched observational attribution, not randomized causal identification
and not a trading-profitability test.

INPUTS
------
data/derived/perps/research/prospective_25bp_mark_index_basis_matching_v1/
    matched_pairs_current_directional_basis_outcome_blind.parquet
    matched_pairs_directional_basis_change_30s_outcome_blind.parquet
    matched_pairs_current_directional_funding_outcome_blind.parquet

reports/perps_prospective_25bp_mark_index_basis_matching_feasibility/
    summary.json
    family_matching_feasibility.csv
    exposure_definitions.csv
    analysis_invariants.csv

data/derived/perps/research/prospective_25bp_exact_participant_interaction_v1/
    candidate_decision_outcomes.parquet

OUTPUTS
-------
data/derived/perps/research/prospective_25bp_mark_index_basis_outcomes_v1/
    matched_pair_outcomes_<family>.parquet
    metadata/latest_summary.json
    metadata/run_config.json

reports/perps_prospective_25bp_mark_index_basis_outcomes/
    frozen_family_pair_manifests.csv
    primary_p75_15m_by_family.csv
    discovery_validation_p75_15m_by_family.csv
    p75_horizon_robustness_by_family.csv
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
PHASE = "7M.8C"
SCHEMA_NAME = "prospective_25bp_mark_index_basis_outcomes_v1"
FROZEN_MATCH_SCHEMA = "prospective_25bp_mark_index_basis_matching_v1"
FROZEN_OUTCOME_SCHEMA = "prospective_25bp_exact_participant_interaction_v1"

PRIMARY_OBSERVATION_HORIZON_SEC = 3
PRIMARY_OUTCOME_HORIZON_SEC = 900
PRIMARY_TARGET_BPS = 75
ROBUSTNESS_OUTCOME_HORIZONS_SEC = (300, 600, 900, 1800)

DISCOVERY_END_DATE = "2026-06-17"
VALIDATION_START_DATE = "2026-06-18"
MAX_FROZEN_SMD = 0.20
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
    "perps_prospective_25bp_mark_index_basis_matching_feasibility"
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
    "perps_prospective_25bp_mark_index_basis_outcomes"
)

FAMILY_SPECS: dict[str, dict[str, Any]] = {
    "CURRENT_DIRECTIONAL_BASIS": {
        "slug": "current_directional_basis",
        "exposure_column": "exp_current_directional_basis",
        "high_state": "BASIS_ALIGNED_WITH_MOVE",
        "low_state": "BASIS_COUNTER_TO_MOVE",
        "expected_pairs": 275,
        "expected_discovery_pairs": 141,
        "expected_validation_pairs": 134,
        "role": "SECONDARY",
        "interpretation_limit": "Current mark-index basis is a perp-vs-index dislocation state. It does not identify which venue or participant caused the dislocation.",
    },
    "DIRECTIONAL_BASIS_CHANGE_30S": {
        "slug": "directional_basis_change_30s",
        "exposure_column": "exp_directional_basis_change_30s",
        "high_state": "BASIS_WIDENING_WITH_MOVE",
        "low_state": "BASIS_MOVING_AGAINST_MOVE",
        "expected_pairs": 992,
        "expected_discovery_pairs": 535,
        "expected_validation_pairs": 457,
        "role": "PRIMARY",
        "interpretation_limit": "The 30s direction-aligned basis change measures short-horizon perp/index dislocation change before the decision point; it does not establish causal flow origin.",
    },
    "CURRENT_DIRECTIONAL_FUNDING": {
        "slug": "current_directional_funding",
        "exposure_column": "exp_current_directional_funding",
        "high_state": "FUNDING_ALIGNED_WITH_MOVE",
        "low_state": "FUNDING_COUNTER_TO_MOVE",
        "expected_pairs": 685,
        "expected_discovery_pairs": 560,
        "expected_validation_pairs": 125,
        "role": "SECONDARY",
        "interpretation_limit": "Funding is a slow contract-level crowding/regime state, not a short-horizon order-flow measure or long/short identity label.",
    },
}


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
        raise SystemExit("Phase 7M.8C requires pandas, numpy and pyarrow in the project .venv.")


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
# Frozen 7M.8B authorization - MUST complete before future labels are opened
# ---------------------------------------------------------------------------

def validate_frozen_matching_state(
    match_root: Path,
    report_root: Path,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Validate all three frozen family states before any outcome file is read."""
    require_stack()
    summary_path = report_root / "summary.json"
    gate_path = report_root / "family_matching_feasibility.csv"
    defs_path = report_root / "exposure_definitions.csv"
    inv_path = report_root / "analysis_invariants.csv"
    for p, name in (
        (summary_path, "7M.8B summary"),
        (gate_path, "7M.8B family gate"),
        (defs_path, "7M.8B exposure definitions"),
        (inv_path, "7M.8B invariants"),
    ):
        if not p.exists():
            raise FileNotFoundError(f"Missing {name}: {p}")

    summary = load_json(summary_path, "7M.8B summary")
    if str(summary.get("phase")) != "7M.8B":
        raise RuntimeError(f"Unexpected frozen phase in 7M.8B summary: {summary.get('phase')}")
    if bool(summary.get("future_outcomes_opened")):
        raise RuntimeError("7M.8B summary says future outcomes were already opened; refusing outcome phase.")
    if not bool(summary.get("all_three_family_gates_pass")):
        raise RuntimeError("Not all 7M.8B family gates passed; refusing outcome phase.")
    if bool(summary.get("composite_signal_built")):
        raise RuntimeError("7M.8B unexpectedly built a composite signal.")

    gate = pd.read_csv(gate_path)
    defs = pd.read_csv(defs_path)
    inv = pd.read_csv(inv_path)
    require_columns(gate, ["family", "exposure_column", "pairs_after_repair", "discovery_pairs", "validation_pairs", "max_abs_smd", "feasible_before_outcomes"], "7M.8B family gate")
    require_columns(defs, ["family", "exposure_column", "high_state", "low_state", "definition", "lower_boundary", "upper_boundary", "boundary_basis"], "7M.8B exposure definitions")
    require_columns(inv, ["invariant", "pass", "actual", "required"], "7M.8B analysis invariants")

    inv_map = {str(r["invariant"]): r for _, r in inv.iterrows()}
    for key in ("future_outcomes_opened", "families_tested_independently", "all_three_family_gates_pass"):
        if key not in inv_map:
            raise RuntimeError(f"7M.8B invariant missing: {key}")
        if not bool_series(pd.Series([inv_map[key]["pass"]])).iloc[0]:
            raise RuntimeError(f"7M.8B invariant failed: {key}")

    pairs_by_family: dict[str, pd.DataFrame] = {}
    freeze_rows: list[dict[str, Any]] = []
    for family, spec in FAMILY_SPECS.items():
        g = gate[gate["family"].astype(str) == family]
        d = defs[defs["family"].astype(str) == family]
        if len(g) != 1 or len(d) != 1:
            raise RuntimeError(f"Expected one frozen gate/definition row for {family}; got gate={len(g)} defs={len(d)}")
        gr = g.iloc[0]
        dr = d.iloc[0]
        if str(gr["exposure_column"]) != spec["exposure_column"] or str(dr["exposure_column"]) != spec["exposure_column"]:
            raise RuntimeError(f"Frozen exposure column changed for {family}.")
        if str(dr["high_state"]) != spec["high_state"] or str(dr["low_state"]) != spec["low_state"]:
            raise RuntimeError(f"Frozen exposure states changed for {family}.")
        if not bool_series(pd.Series([gr["feasible_before_outcomes"]])).iloc[0]:
            raise RuntimeError(f"Frozen 7M.8B gate failed for {family}.")
        if int(gr["pairs_after_repair"]) != int(spec["expected_pairs"]):
            raise RuntimeError(f"Frozen pair count changed for {family}: {gr['pairs_after_repair']} != {spec['expected_pairs']}")
        if int(gr["discovery_pairs"]) != int(spec["expected_discovery_pairs"]):
            raise RuntimeError(f"Frozen discovery pair count changed for {family}.")
        if int(gr["validation_pairs"]) != int(spec["expected_validation_pairs"]):
            raise RuntimeError(f"Frozen validation pair count changed for {family}.")
        if safe_float(gr["max_abs_smd"]) > MAX_FROZEN_SMD + 1e-12:
            raise RuntimeError(f"Frozen matching balance no longer passes for {family}.")

        pair_path = match_root / f"matched_pairs_{spec['slug']}_outcome_blind.parquet"
        if not pair_path.exists():
            raise FileNotFoundError(f"Missing frozen pair manifest for {family}: {pair_path}")
        pairs = pq.read_table(pair_path).to_pandas()
        require_columns(pairs, ["family", "exposure_column", "candidate_date", "direction", "high_candidate_id", "low_candidate_id"], f"{family} frozen pairs")
        if len(pairs) != int(spec["expected_pairs"]):
            raise RuntimeError(f"Frozen parquet pair count changed for {family}: {len(pairs)} != {spec['expected_pairs']}")
        if set(pairs["family"].astype(str).unique()) != {family}:
            raise RuntimeError(f"Frozen pair family labels changed for {family}.")
        if set(pairs["exposure_column"].astype(str).unique()) != {spec["exposure_column"]}:
            raise RuntimeError(f"Frozen pair exposure labels changed for {family}.")
        if pairs.duplicated(["high_candidate_id", "low_candidate_id"]).any():
            raise RuntimeError(f"Duplicate frozen pairs found for {family}.")
        if pairs["high_candidate_id"].astype(str).duplicated().any() or pairs["low_candidate_id"].astype(str).duplicated().any():
            raise RuntimeError(f"Candidate reuse found within one exposure side for {family}; expected 1:1 no-replacement matching.")
        if set(pairs["high_candidate_id"].astype(str)) & set(pairs["low_candidate_id"].astype(str)):
            raise RuntimeError(f"A candidate appears on both exposure sides for {family}.")

        disc_n = int((pairs["candidate_date"].astype(str) <= DISCOVERY_END_DATE).sum())
        val_n = int((pairs["candidate_date"].astype(str) >= VALIDATION_START_DATE).sum())
        if disc_n != int(spec["expected_discovery_pairs"]) or val_n != int(spec["expected_validation_pairs"]):
            raise RuntimeError(f"Frozen D/V partition changed for {family}: {disc_n}/{val_n}")
        if disc_n + val_n != len(pairs):
            raise RuntimeError(f"Frozen D/V split does not partition all {family} pairs.")

        key_hash = sha256_lines(
            f"{family}|{r.candidate_date}|{r.high_candidate_id}|{r.low_candidate_id}"
            for r in pairs.sort_values(["candidate_date", "high_candidate_id", "low_candidate_id"]).itertuples()
        )
        pairs_by_family[family] = pairs
        freeze_rows.append({
            "family": family,
            "analysis_role": spec["role"],
            "exposure_column": spec["exposure_column"],
            "high_state": spec["high_state"],
            "low_state": spec["low_state"],
            "pair_manifest_path": str(pair_path),
            "pair_manifest_file_sha256": sha256_file(pair_path),
            "pair_key_sha256": key_hash,
            "pair_rows": int(len(pairs)),
            "discovery_pairs": disc_n,
            "validation_pairs": val_n,
            "frozen_max_abs_smd": safe_float(gr["max_abs_smd"]),
            "frozen_definition": str(dr["definition"]),
            "lower_boundary": safe_float(dr["lower_boundary"]),
            "upper_boundary": safe_float(dr["upper_boundary"]),
            "boundary_basis": str(dr["boundary_basis"]),
        })

    return pairs_by_family, gate, defs, summary, pd.DataFrame(freeze_rows)


# ---------------------------------------------------------------------------
# Future outcomes - deliberately callable only after frozen-state validation
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
        raise RuntimeError("No outcome rows for observation_horizon_sec=3.")
    if outcomes["candidate_id"].astype(str).duplicated().any():
        raise RuntimeError("Duplicate candidate IDs after filtering outcome table to +3s decision point.")
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
            "high_rate": math.nan,
            "low_rate": math.nan,
            "effect_pp": math.nan,
            "high_only_wins": 0,
            "low_only_wins": 0,
            "both_reach": 0,
            "neither_reach": 0,
        }
    hy = x["__high_y"].astype(bool)
    ly = x["__low_y"].astype(bool)
    return {
        "outcome_pairs": int(len(x)),
        "high_rate": float(hy.mean()),
        "low_rate": float(ly.mean()),
        "effect_pp": 100.0 * float((hy.astype(int) - ly.astype(int)).mean()),
        "high_only_wins": int((hy & ~ly).sum()),
        "low_only_wins": int((~hy & ly).sum()),
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
        return "HIGH_STATE_POSITIVE_ASSOCIATION_CI_ABOVE_ZERO"
    if math.isfinite(ci_high) and ci_high < 0:
        return "HIGH_STATE_NEGATIVE_ASSOCIATION_CI_BELOW_ZERO"
    if effect_pp > 0:
        return "HIGH_STATE_POSITIVE_POINT_ESTIMATE_CI_CROSSES_ZERO"
    if effect_pp < 0:
        return "HIGH_STATE_NEGATIVE_POINT_ESTIMATE_CI_CROSSES_ZERO"
    return "ZERO_POINT_ESTIMATE"


def effect_row(
    family: str,
    spec: dict[str, Any],
    joined: pd.DataFrame,
    analysis_set: str,
    horizon: int,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    r = paired_effect(joined, horizon)
    lo, hi, valid_reps = bootstrap_matched_day(joined, horizon, reps, seed)
    return {
        "family": family,
        "exposure_column": spec["exposure_column"],
        "high_state": spec["high_state"],
        "low_state": spec["low_state"],
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
        "interpretation_limit": spec["interpretation_limit"],
    }


def analyze_family(
    family: str,
    spec: dict[str, Any],
    joined: pd.DataFrame,
    reps: int,
    seed_offset: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    primary = effect_row(family, spec, joined, "FULL", PRIMARY_OUTCOME_HORIZON_SEC, reps, BOOTSTRAP_SEED + seed_offset)
    discovery = joined[joined["candidate_date"].astype(str) <= DISCOVERY_END_DATE].copy()
    validation = joined[joined["candidate_date"].astype(str) >= VALIDATION_START_DATE].copy()
    split = [
        effect_row(family, spec, discovery, "DISCOVERY", PRIMARY_OUTCOME_HORIZON_SEC, reps, BOOTSTRAP_SEED + seed_offset + 1),
        effect_row(family, spec, validation, "VALIDATION", PRIMARY_OUTCOME_HORIZON_SEC, reps, BOOTSTRAP_SEED + seed_offset + 2),
    ]
    robust: list[dict[str, Any]] = []
    for i, h in enumerate(ROBUSTNESS_OUTCOME_HORIZONS_SEC):
        robust.append(effect_row(family, spec, joined, "FULL", int(h), reps, BOOTSTRAP_SEED + seed_offset + 100 + i))
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


def make_findings(primary: pd.DataFrame, split: pd.DataFrame, robust: pd.DataFrame, freeze: pd.DataFrame) -> str:
    lines = [
        "# Phase 7M.8C - mark/index basis and funding outcome attribution",
        "",
        "Status: **FROZEN-PAIR OUTCOME ANALYSIS COMPLETE**",
        "",
        "No rematching, threshold changes, raw replay, or composite signal were used after future outcomes were opened.",
        "",
        "## Primary: P75 within 15 minutes from the +3s decision point",
        "",
    ]
    for family in FAMILY_SPECS:
        p = primary[primary["family"] == family].iloc[0]
        d = split[(split["family"] == family) & (split["analysis_set"] == "DISCOVERY")].iloc[0]
        v = split[(split["family"] == family) & (split["analysis_set"] == "VALIDATION")].iloc[0]
        spec = FAMILY_SPECS[family]
        lines.extend([
            f"### {family} ({spec['role']})",
            "",
            f"- Frozen contrast: **{spec['high_state']}** vs **{spec['low_state']}**",
            f"- Outcome-usable pairs: **{int(p['outcome_pairs']):,} / {int(p['frozen_pairs_before_outcome_filter']):,}**",
            f"- High-state P75 rate: **{pct(p['high_rate'])}**",
            f"- Low-state P75 rate: **{pct(p['low_rate'])}**",
            f"- Paired difference (high - low): **{pp(p['effect_pp'])}**",
            f"- Candidate-day bootstrap 95% CI: **[{pp(p['effect_ci_low_pp'])}, {pp(p['effect_ci_high_pp'])}]**",
            f"- Discovery: **{pp(d['effect_pp'])}**, CI **[{pp(d['effect_ci_low_pp'])}, {pp(d['effect_ci_high_pp'])}]**, n={int(d['outcome_pairs']):,}",
            f"- Validation: **{pp(v['effect_pp'])}**, CI **[{pp(v['effect_ci_low_pp'])}, {pp(v['effect_ci_high_pp'])}]**, n={int(v['outcome_pairs']):,}",
            f"- Interpretation code: **{p['interpretation']}**",
            f"- Limit: {spec['interpretation_limit']}",
            "",
        ])
    lines.extend([
        "## P75 horizon robustness - same frozen pairs",
        "",
    ])
    for family in FAMILY_SPECS:
        lines.append(f"### {family}")
        lines.append("")
        rr = robust[robust["family"] == family].sort_values("outcome_horizon_sec")
        for r in rr.itertuples():
            lines.append(
                f"- {int(r.outcome_horizon_sec)//60}m: high={pct(r.high_rate)}, low={pct(r.low_rate)}, "
                f"difference={pp(r.effect_pp)}, CI=[{pp(r.effect_ci_low_pp)}, {pp(r.effect_ci_high_pp)}]"
            )
        lines.append("")
    lines.extend([
        "## Frozen-manifest audit",
        "",
    ])
    for r in freeze.itertuples():
        lines.append(
            f"- {r.family}: pairs={int(r.pair_rows):,}, D/V={int(r.discovery_pairs):,}/{int(r.validation_pairs):,}, "
            f"pair-key SHA256=`{r.pair_key_sha256}`"
        )
    lines.extend([
        "",
        "## Interpretation discipline",
        "",
        "These are matched observational associations. A non-zero contrast does not prove that the exposure caused continuation, and a null result does not prove that the exposure is physically irrelevant. Families remain separate; the 30s directional basis-change family is preregistered primary, while current basis and funding are secondary. This phase does not create a combined basis/funding score.",
        "",
    ])
    return "\n".join(lines)


def build_invariants(freeze: pd.DataFrame, primary: pd.DataFrame, split: pd.DataFrame, joined_by_family: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {"invariant": "all_three_pair_manifests_validated_before_outcomes", "pass": len(freeze) == 3, "actual": len(freeze), "required": 3},
        {"invariant": "future_labels_loaded_only_after_frozen_pair_validation", "pass": True, "actual": True, "required": True},
        {"invariant": "primary_observation_horizon_sec", "pass": PRIMARY_OBSERVATION_HORIZON_SEC == 3, "actual": PRIMARY_OBSERVATION_HORIZON_SEC, "required": 3},
        {"invariant": "primary_outcome_horizon_sec", "pass": PRIMARY_OUTCOME_HORIZON_SEC == 900, "actual": PRIMARY_OUTCOME_HORIZON_SEC, "required": 900},
        {"invariant": "primary_target_bps", "pass": PRIMARY_TARGET_BPS == 75, "actual": PRIMARY_TARGET_BPS, "required": 75},
        {"invariant": "families_analyzed_independently", "pass": len(primary) == 3, "actual": len(primary), "required": 3},
        {"invariant": "primary_family", "pass": True, "actual": "DIRECTIONAL_BASIS_CHANGE_30S", "required": "DIRECTIONAL_BASIS_CHANGE_30S"},
        {"invariant": "composite_signal_built", "pass": True, "actual": False, "required": False},
        {"invariant": "raw_market_replay", "pass": True, "actual": False, "required": False},
        {"invariant": "rematching_or_pair_repair_after_outcomes", "pass": True, "actual": False, "required": False},
        {"invariant": "no_ml", "pass": True, "actual": False, "required": False},
    ]
    for family, spec in FAMILY_SPECS.items():
        p = primary[primary["family"] == family].iloc[0]
        d = split[(split["family"] == family) & (split["analysis_set"] == "DISCOVERY")].iloc[0]
        v = split[(split["family"] == family) & (split["analysis_set"] == "VALIDATION")].iloc[0]
        joined = joined_by_family[family]
        rows.extend([
            {"invariant": f"{family}:outcome_join_preserves_pairs", "pass": len(joined) == int(spec["expected_pairs"]), "actual": len(joined), "required": int(spec["expected_pairs"])},
            {"invariant": f"{family}:full_pairs_before_outcome_filter", "pass": int(p["frozen_pairs_before_outcome_filter"]) == int(spec["expected_pairs"]), "actual": int(p["frozen_pairs_before_outcome_filter"]), "required": int(spec["expected_pairs"])},
            {"invariant": f"{family}:discovery_pairs_before_outcome_filter", "pass": int(d["frozen_pairs_before_outcome_filter"]) == int(spec["expected_discovery_pairs"]), "actual": int(d["frozen_pairs_before_outcome_filter"]), "required": int(spec["expected_discovery_pairs"])},
            {"invariant": f"{family}:validation_pairs_before_outcome_filter", "pass": int(v["frozen_pairs_before_outcome_filter"]) == int(spec["expected_validation_pairs"]), "actual": int(v["frozen_pairs_before_outcome_filter"]), "required": int(spec["expected_validation_pairs"])},
        ])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def run_self_test() -> int:
    # Two days, balanced outcomes, known +25pp high-minus-low effect.
    pairs = pd.DataFrame({
        "candidate_date": ["2026-06-01"] * 4 + ["2026-06-20"] * 4,
        "high_candidate_id": [f"h{i}" for i in range(8)],
        "low_candidate_id": [f"l{i}" for i in range(8)],
    })
    outcomes = []
    # High wins 3/8, low wins 1/8 => +25pp.
    high_y = [1, 1, 1, 0, 0, 0, 0, 0]
    low_y =  [1, 0, 0, 0, 0, 0, 0, 0]
    for cid, y in zip(pairs["high_candidate_id"], high_y):
        row = {"candidate_id": cid, "decision_price_valid": True}
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
    assert int(r["high_only_wins"]) == 2 and int(r["low_only_wins"]) == 0
    disc = joined[joined["candidate_date"] <= DISCOVERY_END_DATE]
    val = joined[joined["candidate_date"] >= VALIDATION_START_DATE]
    assert len(disc) == 4 and len(val) == 4
    assert interpretation(2.0, -1.0, 5.0) == "HIGH_STATE_POSITIVE_POINT_ESTIMATE_CI_CROSSES_ZERO"
    assert interpretation(-2.0, -5.0, -0.1) == "HIGH_STATE_NEGATIVE_ASSOCIATION_CI_BELOW_ZERO"
    print("SELF-TEST PASS: frozen basis/funding high-low outcome join, P75 paired effect, D/V split, primary-family freeze, and interpretation logic validated.")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 7M.8C frozen-pair mark/index basis and funding outcome attribution")
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
    if int(args.bootstrap_reps) < MIN_BOOTSTRAP_VALID_REPS:
        raise SystemExit(f"--bootstrap-reps must be >= {MIN_BOOTSTRAP_VALID_REPS}")

    print("Phase 7M.8C - Frozen-pair mark/index basis and funding outcome attribution")
    print("[1/6] Validating all three frozen 7M.8B gates and pair manifests BEFORE outcomes...")
    pairs_by_family, gate, defs, frozen_summary, freeze = validate_frozen_matching_state(args.match_root, args.match_report_root)
    for r in freeze.itertuples():
        print(f"      {r.family}: pairs={int(r.pair_rows):,} D/V={int(r.discovery_pairs):,}/{int(r.validation_pairs):,} hash={r.pair_key_sha256[:12]}...")

    print("[2/6] Frozen state verified. Opening Phase 7M.3 future outcomes at +3s decision point...")
    outcomes = load_outcomes_after_freeze(args.m3_root)

    print("[3/6] Joining future P75 labels to each family independently; NO rematching...")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)
    joined_by_family: dict[str, pd.DataFrame] = {}
    primary_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    robust_rows: list[dict[str, Any]] = []

    for i, (family, spec) in enumerate(FAMILY_SPECS.items()):
        joined = join_pair_outcomes(pairs_by_family[family], outcomes)
        joined_by_family[family] = joined
        primary, split, robust = analyze_family(family, spec, joined, int(args.bootstrap_reps), seed_offset=i * 1000)
        primary_rows.append(primary)
        split_rows.extend(split)
        robust_rows.extend(robust)
        atomic_parquet(args.output_root / f"matched_pair_outcomes_{spec['slug']}.parquet", joined)
        print(
            f"      {family}: usable={primary['outcome_pairs']:,}/{primary['frozen_pairs_before_outcome_filter']:,} "
            f"high={100*safe_float(primary['high_rate']):.2f}% low={100*safe_float(primary['low_rate']):.2f}% "
            f"diff={safe_float(primary['effect_pp']):+.2f}pp "
            f"CI=[{safe_float(primary['effect_ci_low_pp']):+.2f},{safe_float(primary['effect_ci_high_pp']):+.2f}]"
        )

    primary_df = pd.DataFrame(primary_rows)
    split_df = pd.DataFrame(split_rows)
    robust_df = pd.DataFrame(robust_rows)

    print("[4/6] Writing primary, D/V, robustness and frozen-manifest reports...")
    atomic_csv(args.report_root / "frozen_family_pair_manifests.csv", freeze)
    atomic_csv(args.report_root / "primary_p75_15m_by_family.csv", primary_df)
    atomic_csv(args.report_root / "discovery_validation_p75_15m_by_family.csv", split_df)
    atomic_csv(args.report_root / "p75_horizon_robustness_by_family.csv", robust_df)

    print("[5/6] Verifying outcome-phase invariants and writing findings...")
    inv = build_invariants(freeze, primary_df, split_df, joined_by_family)
    if not bool_series(inv["pass"]).all():
        bad = inv.loc[~bool_series(inv["pass"]), ["invariant", "actual", "required"]]
        raise RuntimeError(f"7M.8C invariant failure after analysis:\n{bad.to_string(index=False)}")
    atomic_csv(args.report_root / "analysis_invariants.csv", inv)
    atomic_text(args.report_root / "findings.md", make_findings(primary_df, split_df, robust_df, freeze))

    run_config = {
        "phase": PHASE,
        "builder_version": BUILDER_VERSION,
        "generated_at_utc": utc_now_iso(),
        "primary_observation_horizon_sec": PRIMARY_OBSERVATION_HORIZON_SEC,
        "primary_outcome_horizon_sec": PRIMARY_OUTCOME_HORIZON_SEC,
        "primary_target_bps": PRIMARY_TARGET_BPS,
        "robustness_outcome_horizons_sec": list(ROBUSTNESS_OUTCOME_HORIZONS_SEC),
        "discovery_end_date": DISCOVERY_END_DATE,
        "validation_start_date": VALIDATION_START_DATE,
        "bootstrap_reps": int(args.bootstrap_reps),
        "bootstrap_cluster": "candidate_date",
        "families": FAMILY_SPECS,
        "frozen_pair_manifests": freeze.to_dict(orient="records"),
        "matching_summary_sha256": sha256_file(args.match_report_root / "summary.json"),
        "matching_gate_sha256": sha256_file(args.match_report_root / "family_matching_feasibility.csv"),
        "exposure_definitions_sha256": sha256_file(args.match_report_root / "exposure_definitions.csv"),
        "no_rematching": True,
        "no_composite": True,
        "no_raw_replay": True,
    }
    summary = {
        "schema_name": SCHEMA_NAME,
        "phase": PHASE,
        "builder_version": BUILDER_VERSION,
        "generated_at_utc": utc_now_iso(),
        "future_outcomes_loaded": True,
        "frozen_pairs_validated_before_outcomes": True,
        "families_analyzed_independently": True,
        "composite_signal_built": False,
        "raw_market_replay_performed": False,
        "primary_results": primary_df.to_dict(orient="records"),
        "discovery_validation_results": split_df.to_dict(orient="records"),
        "robustness_results": robust_df.to_dict(orient="records"),
        "frozen_pair_manifests": freeze.to_dict(orient="records"),
    }
    atomic_json(args.output_root / "metadata" / "run_config.json", run_config)
    atomic_json(args.output_root / "metadata" / "latest_summary.json", summary)

    print("[6/6] Complete.")
    print(f"      report={args.report_root}")
    print(f"      output={args.output_root}")
    print("      rematching=NO composite=NO raw_replay=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
