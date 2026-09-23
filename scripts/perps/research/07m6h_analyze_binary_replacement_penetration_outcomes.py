#!/usr/bin/env python3
"""
Phase 7M.6H — Frozen-pair replacement-penetration outcome attribution
=====================================================================

RESEARCH QUESTION
-----------------
Among the already-frozen Phase 7M.6F matched pairs, is conservative evidence
that aggressive execution penetrated newly replenished opposing liquidity
associated with a higher probability of subsequent +75bp continuation?

SCIENTIFIC CONTRACT
-------------------
* NO raw replay.
* NO candidate regeneration.
* NO rematching, balance repair, caliper changes, exposure changes, or threshold tuning.
* The authoritative exposure pair manifest is the Phase 7M.6F
  `full_matched_pairs_outcome_blind.parquet` written before future labels.
* Exposure is frozen as:
      replacement_penetrated =
          opposing__replacement_execution_lower_bound_quote > 0
* Primary mechanics decision point is frozen at anchor + 3 seconds.
* Primary future outcome is frozen as P75 within 15 minutes from that +3s
  decision point (`h900_origin_reach_75`).
* Discovery/validation split is frozen at:
      discovery <= 2026-06-17
      validation >= 2026-06-18
* Primary uncertainty uses candidate-date clustered bootstrap, matching the
  earlier 7M attribution analyses.
* Secondary robustness checks use the SAME frozen pairs and P75 at 5, 10,
  and 30 minutes. They do not create new feature definitions or pair sets.

IMPORTANT INTERPRETATION LIMITS
-------------------------------
This is a matched observational mechanism comparison, not randomized causal
identification and not a profitability/alpha test. A positive contrast means
that, among otherwise similar frozen matched candidates under the specified
controls, demonstrable replacement penetration was associated with more P75
continuation. It does not prove participant identity, FIFO execution, or a
tradable strategy.

INPUTS
------
data/derived/perps/research/prospective_25bp_refill_redepletion_sequence_v2/
    full_matched_pairs_outcome_blind.parquet
    metadata/latest_summary.json

reports/perps_prospective_25bp_refill_redepletion_sequence_v2/
    full_preoutcome_gate.csv
    analysis_invariants.csv

data/derived/perps/research/prospective_25bp_exact_participant_interaction_v1/
    candidate_decision_outcomes.parquet

OUTPUTS
-------
data/derived/perps/research/prospective_25bp_refill_redepletion_outcomes_v1/
    matched_pair_outcomes.parquet
    metadata/latest_summary.json
    metadata/run_config.json

reports/perps_prospective_25bp_refill_redepletion_outcomes/
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
SCHEMA_VERSION = 1
SCHEMA_NAME = "prospective_25bp_refill_redepletion_outcomes_v1"

FROZEN_7M6F_SCHEMA = "prospective_25bp_refill_redepletion_sequence_v2"
FROZEN_7M3_SCHEMA = "prospective_25bp_exact_participant_interaction_v1"

PRIMARY_OBSERVATION_HORIZON_SEC = 3
PRIMARY_OUTCOME_HORIZON_SEC = 900
PRIMARY_TARGET_BPS = 75
ROBUSTNESS_OUTCOME_HORIZONS_SEC = (300, 600, 900, 1800)

DISCOVERY_END_DATE = "2026-06-17"
VALIDATION_START_DATE = "2026-06-18"

EXPECTED_FULL_PAIRS = 695
EXPECTED_DISCOVERY_PAIRS = 390
EXPECTED_VALIDATION_PAIRS = 305
MAX_FROZEN_SMD = 0.20

BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20260917
MIN_BOOTSTRAP_VALID_REPS = 1_000

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()
DEFAULT_M6_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / FROZEN_7M6F_SCHEMA
DEFAULT_M6_REPORT_ROOT = REPO_ROOT / "reports" / "perps_prospective_25bp_refill_redepletion_sequence_v2"
DEFAULT_M3_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / FROZEN_7M3_SCHEMA
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / SCHEMA_NAME
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "perps_prospective_25bp_refill_redepletion_outcomes"


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
        raise SystemExit("Phase 7M.6H requires pandas, numpy and pyarrow in the project .venv.")


def require_columns(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing required columns: {missing}")


def bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
        .astype(bool)
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_lines(values: Iterable[str]) -> str:
    h = hashlib.sha256()
    for v in values:
        h.update(str(v).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


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


def load_json(path: Path, name: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise RuntimeError(f"{name} is not a JSON object: {path}")
    return obj


# ---------------------------------------------------------------------------
# Frozen pre-outcome authorization
# ---------------------------------------------------------------------------

def validate_frozen_preoutcome_state(m6_root: Path, m6_report_root: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Validate the already-frozen causal pair manifest BEFORE any outcome file is read."""
    require_stack()
    pair_path = m6_root / "full_matched_pairs_outcome_blind.parquet"
    summary_path = m6_root / "metadata" / "latest_summary.json"
    gate_path = m6_report_root / "full_preoutcome_gate.csv"
    inv_path = m6_report_root / "analysis_invariants.csv"

    for p, name in (
        (pair_path, "7M.6F frozen pair manifest"),
        (summary_path, "7M.6F summary"),
        (gate_path, "7M.6F pre-outcome gate"),
        (inv_path, "7M.6F invariants"),
    ):
        if not p.exists():
            raise FileNotFoundError(f"Missing {name}: {p}")

    summary = load_json(summary_path, "7M.6F summary")
    if str(summary.get("schema_name")) != FROZEN_7M6F_SCHEMA:
        raise RuntimeError(f"Unexpected 7M.6F schema: {summary.get('schema_name')}")
    if bool(summary.get("future_outcomes_loaded")):
        raise RuntimeError("7M.6F summary says future outcomes were already loaded; refusing frozen-pair outcome phase.")
    if not bool(summary.get("primary_preoutcome_gate_pass")):
        raise RuntimeError("7M.6F primary pre-outcome gate did not pass; refusing outcome loading.")
    if int(summary.get("full_matched_pairs", -1)) != EXPECTED_FULL_PAIRS:
        raise RuntimeError(f"Frozen full-pair count changed: {summary.get('full_matched_pairs')} != {EXPECTED_FULL_PAIRS}")
    if int(summary.get("discovery_pairs", -1)) != EXPECTED_DISCOVERY_PAIRS:
        raise RuntimeError("Frozen discovery pair count changed.")
    if int(summary.get("validation_pairs", -1)) != EXPECTED_VALIDATION_PAIRS:
        raise RuntimeError("Frozen validation pair count changed.")

    gate = pd.read_csv(gate_path)
    if len(gate) != 1:
        raise RuntimeError(f"Expected exactly one primary 7M.6F gate row, found {len(gate)}")
    g = gate.iloc[0]
    if str(g.get("feature")) != "replacement_penetrated":
        raise RuntimeError("7M.6F gate feature is no longer replacement_penetrated.")
    if not bool_series(pd.Series([g.get("feasible_before_outcomes")])).iloc[0]:
        raise RuntimeError("7M.6F gate is not feasible_before_outcomes=True.")
    if int(g.get("pairs_after_repair", -1)) != EXPECTED_FULL_PAIRS:
        raise RuntimeError("7M.6F gate pair count changed.")
    if safe_float(g.get("max_abs_smd")) > MAX_FROZEN_SMD + 1e-12:
        raise RuntimeError("7M.6F frozen matching balance no longer satisfies max |SMD| <= 0.20.")

    inv = pd.read_csv(inv_path)
    require_columns(inv, ["invariant", "pass", "actual", "required"], "7M.6F analysis_invariants")
    inv_map = {str(r["invariant"]): r for _, r in inv.iterrows()}
    for key in ("future_outcomes_loaded", "full_preoutcome_common_support_gate", "pair_manifest_frozen_before_outcomes"):
        if key not in inv_map:
            raise RuntimeError(f"7M.6F invariant missing: {key}")
        if not bool_series(pd.Series([inv_map[key]["pass"]])).iloc[0]:
            raise RuntimeError(f"7M.6F invariant failed: {key}")

    pairs = pq.read_table(pair_path).to_pandas()
    require_columns(
        pairs,
        [
            "candidate_date", "direction", "penetrated_candidate_id", "not_penetrated_candidate_id",
            "high_candidate_id", "low_candidate_id", "high__replacement_penetrated", "low__replacement_penetrated",
        ],
        "7M.6F frozen pair manifest",
    )
    if len(pairs) != EXPECTED_FULL_PAIRS:
        raise RuntimeError(f"Frozen pair manifest contains {len(pairs)} rows, expected {EXPECTED_FULL_PAIRS}.")
    if pairs.duplicated(["penetrated_candidate_id", "not_penetrated_candidate_id"]).any():
        raise RuntimeError("Frozen pair manifest contains duplicate pair IDs.")
    if not pairs["penetrated_candidate_id"].astype(str).equals(pairs["high_candidate_id"].astype(str)):
        raise RuntimeError("Frozen high_candidate_id is no longer the penetrated candidate.")
    if not pairs["not_penetrated_candidate_id"].astype(str).equals(pairs["low_candidate_id"].astype(str)):
        raise RuntimeError("Frozen low_candidate_id is no longer the non-penetrated candidate.")
    if not (pd.to_numeric(pairs["high__replacement_penetrated"], errors="coerce") == 1).all():
        raise RuntimeError("Frozen high exposure is not uniformly replacement_penetrated=1.")
    if not (pd.to_numeric(pairs["low__replacement_penetrated"], errors="coerce") == 0).all():
        raise RuntimeError("Frozen low exposure is not uniformly replacement_penetrated=0.")

    disc_n = int((pairs["candidate_date"].astype(str) <= DISCOVERY_END_DATE).sum())
    val_n = int((pairs["candidate_date"].astype(str) >= VALIDATION_START_DATE).sum())
    if disc_n != EXPECTED_DISCOVERY_PAIRS or val_n != EXPECTED_VALIDATION_PAIRS:
        raise RuntimeError(f"Frozen D/V counts changed: {disc_n}/{val_n} != {EXPECTED_DISCOVERY_PAIRS}/{EXPECTED_VALIDATION_PAIRS}")
    if disc_n + val_n != len(pairs):
        raise RuntimeError("Frozen D/V split does not partition all pair rows.")

    pair_key_hash = sha256_lines(
        f"{r.candidate_date}|{r.penetrated_candidate_id}|{r.not_penetrated_candidate_id}"
        for r in pairs.sort_values(["candidate_date", "penetrated_candidate_id", "not_penetrated_candidate_id"]).itertuples()
    )
    freeze = {
        "pair_manifest_path": str(pair_path),
        "pair_manifest_file_sha256": sha256_file(pair_path),
        "pair_key_sha256": pair_key_hash,
        "pair_rows": int(len(pairs)),
        "discovery_pairs": disc_n,
        "validation_pairs": val_n,
        "frozen_max_abs_smd": safe_float(g.get("max_abs_smd")),
        "frozen_feature": str(g.get("feature")),
        "frozen_feature_definition": str(g.get("feature_definition")),
        "frozen_control_set": str(g.get("control_set")),
    }
    return pairs, summary, freeze


# ---------------------------------------------------------------------------
# Outcome loading — deliberately unavailable until freeze validation completes
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
        raise RuntimeError("No Phase 7M.3 outcome rows exist for observation_horizon_sec=3.")
    if outcomes["candidate_id"].astype(str).duplicated().any():
        raise RuntimeError("Outcome table has duplicate candidate IDs after filtering to +3s decision point.")
    return outcomes


def join_pair_outcomes(pairs: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    keep = ["candidate_id", "decision_price_valid"]
    for h in ROBUSTNESS_OUTCOME_HORIZONS_SEC:
        keep.extend([f"h{h}_fully_observed", f"h{h}_origin_reach_{PRIMARY_TARGET_BPS}"])
    keep = list(dict.fromkeys(keep))

    high = outcomes[keep].copy().add_prefix("penetrated__")
    low = outcomes[keep].copy().add_prefix("not_penetrated__")

    joined = (
        pairs.merge(
            high,
            left_on="penetrated_candidate_id",
            right_on="penetrated__candidate_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            low,
            left_on="not_penetrated_candidate_id",
            right_on="not_penetrated__candidate_id",
            how="left",
            validate="many_to_one",
        )
    )
    if len(joined) != len(pairs):
        raise RuntimeError("Outcome join changed the frozen pair-row count.")
    return joined


# ---------------------------------------------------------------------------
# Frozen-pair effects
# ---------------------------------------------------------------------------

def outcome_eligible_pairs(joined: pd.DataFrame, outcome_horizon_sec: int) -> pd.DataFrame:
    full_h = f"penetrated__h{outcome_horizon_sec}_fully_observed"
    full_l = f"not_penetrated__h{outcome_horizon_sec}_fully_observed"
    y_h = f"penetrated__h{outcome_horizon_sec}_origin_reach_{PRIMARY_TARGET_BPS}"
    y_l = f"not_penetrated__h{outcome_horizon_sec}_origin_reach_{PRIMARY_TARGET_BPS}"
    needed = [
        "penetrated__decision_price_valid", "not_penetrated__decision_price_valid",
        full_h, full_l, y_h, y_l,
    ]
    require_columns(joined, needed, f"joined outcomes h={outcome_horizon_sec}")

    obs = (
        bool_series(joined["penetrated__decision_price_valid"])
        & bool_series(joined["not_penetrated__decision_price_valid"])
        & bool_series(joined[full_h])
        & bool_series(joined[full_l])
    )
    x = joined.loc[obs].copy()
    hy = x[y_h].astype("boolean")
    ly = x[y_l].astype("boolean")
    valid = hy.notna() & ly.notna()
    x = x.loc[valid].copy()
    if not x.empty:
        x["__penetrated_y"] = hy.loc[valid].astype(bool).to_numpy()
        x["__not_penetrated_y"] = ly.loc[valid].astype(bool).to_numpy()
        x["__pair_diff"] = x["__penetrated_y"].astype(int) - x["__not_penetrated_y"].astype(int)
    return x


def paired_effect(joined: pd.DataFrame, outcome_horizon_sec: int) -> dict[str, Any]:
    x = outcome_eligible_pairs(joined, outcome_horizon_sec)
    if x.empty:
        return {
            "outcome_pairs": 0,
            "penetrated_rate": math.nan,
            "not_penetrated_rate": math.nan,
            "effect_pp": math.nan,
            "penetrated_only_wins": 0,
            "not_penetrated_only_wins": 0,
            "both_reach": 0,
            "neither_reach": 0,
        }
    hy = x["__penetrated_y"].astype(bool)
    ly = x["__not_penetrated_y"].astype(bool)
    return {
        "outcome_pairs": int(len(x)),
        "penetrated_rate": float(hy.mean()),
        "not_penetrated_rate": float(ly.mean()),
        "effect_pp": 100.0 * float((hy.astype(int) - ly.astype(int)).mean()),
        "penetrated_only_wins": int((hy & ~ly).sum()),
        "not_penetrated_only_wins": int((~hy & ly).sum()),
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
    vals = np.divide(
        multiplicity @ sums,
        den,
        out=np.full(int(reps), np.nan, dtype=float),
        where=den > 0,
    ) * 100.0
    vals = vals[np.isfinite(vals)]
    if len(vals) < min(MIN_BOOTSTRAP_VALID_REPS, int(reps)):
        return math.nan, math.nan, int(len(vals))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), int(len(vals))


def interpretation(effect_pp: float, ci_low: float, ci_high: float) -> str:
    if not math.isfinite(effect_pp):
        return "NOT_ESTIMABLE"
    if math.isfinite(ci_low) and ci_low > 0:
        return "POSITIVE_ASSOCIATION_CI_ABOVE_ZERO"
    if math.isfinite(ci_high) and ci_high < 0:
        return "NEGATIVE_ASSOCIATION_CI_BELOW_ZERO"
    if effect_pp > 0:
        return "POSITIVE_POINT_ESTIMATE_CI_CROSSES_ZERO"
    if effect_pp < 0:
        return "NEGATIVE_POINT_ESTIMATE_CI_CROSSES_ZERO"
    return "ZERO_POINT_ESTIMATE"


def effect_row(joined: pd.DataFrame, label: str, horizon: int, reps: int, seed: int) -> dict[str, Any]:
    r = paired_effect(joined, horizon)
    lo, hi, valid_reps = bootstrap_matched_day(joined, horizon, reps, seed)
    return {
        "analysis_set": label,
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
    }


def analyze_primary(joined: pd.DataFrame, reps: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full = pd.DataFrame([
        effect_row(joined, "FULL", PRIMARY_OUTCOME_HORIZON_SEC, reps, BOOTSTRAP_SEED)
    ])

    discovery = joined[joined["candidate_date"].astype(str) <= DISCOVERY_END_DATE].copy()
    validation = joined[joined["candidate_date"].astype(str) >= VALIDATION_START_DATE].copy()
    split = pd.DataFrame([
        effect_row(discovery, "DISCOVERY", PRIMARY_OUTCOME_HORIZON_SEC, reps, BOOTSTRAP_SEED + 1),
        effect_row(validation, "VALIDATION", PRIMARY_OUTCOME_HORIZON_SEC, reps, BOOTSTRAP_SEED + 2),
    ])

    robust_rows = []
    for i, h in enumerate(ROBUSTNESS_OUTCOME_HORIZONS_SEC):
        robust_rows.append(effect_row(joined, "FULL", int(h), reps, BOOTSTRAP_SEED + 100 + i))
    robustness = pd.DataFrame(robust_rows)
    return full, split, robustness


# ---------------------------------------------------------------------------
# Reporting / invariants
# ---------------------------------------------------------------------------

def build_invariants(
    pairs: pd.DataFrame,
    joined: pd.DataFrame,
    primary: pd.DataFrame,
    split: pd.DataFrame,
    freeze: dict[str, Any],
) -> pd.DataFrame:
    p = primary.iloc[0]
    disc = split[split["analysis_set"] == "DISCOVERY"].iloc[0]
    val = split[split["analysis_set"] == "VALIDATION"].iloc[0]
    rows = [
        {"invariant": "frozen_pair_manifest_used_without_rematching", "pass": len(pairs) == EXPECTED_FULL_PAIRS, "actual": len(pairs), "required": EXPECTED_FULL_PAIRS},
        {"invariant": "frozen_pair_key_hash_recorded_before_outcomes", "pass": bool(freeze.get("pair_key_sha256")), "actual": freeze.get("pair_key_sha256"), "required": "non-empty SHA256"},
        {"invariant": "primary_observation_horizon_sec", "pass": PRIMARY_OBSERVATION_HORIZON_SEC == 3, "actual": PRIMARY_OBSERVATION_HORIZON_SEC, "required": 3},
        {"invariant": "primary_outcome_horizon_sec", "pass": PRIMARY_OUTCOME_HORIZON_SEC == 900, "actual": PRIMARY_OUTCOME_HORIZON_SEC, "required": 900},
        {"invariant": "primary_target_bps", "pass": PRIMARY_TARGET_BPS == 75, "actual": PRIMARY_TARGET_BPS, "required": 75},
        {"invariant": "outcome_join_preserves_pair_rows", "pass": len(joined) == len(pairs), "actual": len(joined), "required": len(pairs)},
        {"invariant": "full_pair_count_before_outcome_filter", "pass": int(p["frozen_pairs_before_outcome_filter"]) == EXPECTED_FULL_PAIRS, "actual": int(p["frozen_pairs_before_outcome_filter"]), "required": EXPECTED_FULL_PAIRS},
        {"invariant": "discovery_pair_count_before_outcome_filter", "pass": int(disc["frozen_pairs_before_outcome_filter"]) == EXPECTED_DISCOVERY_PAIRS, "actual": int(disc["frozen_pairs_before_outcome_filter"]), "required": EXPECTED_DISCOVERY_PAIRS},
        {"invariant": "validation_pair_count_before_outcome_filter", "pass": int(val["frozen_pairs_before_outcome_filter"]) == EXPECTED_VALIDATION_PAIRS, "actual": int(val["frozen_pairs_before_outcome_filter"]), "required": EXPECTED_VALIDATION_PAIRS},
        {"invariant": "future_labels_loaded_only_after_frozen_pair_validation", "pass": True, "actual": True, "required": True},
        {"invariant": "no_raw_replay", "pass": True, "actual": False, "required": False},
        {"invariant": "no_rematching_or_pair_repair", "pass": True, "actual": False, "required": False},
        {"invariant": "no_ml", "pass": True, "actual": False, "required": False},
    ]
    return pd.DataFrame(rows)


def pct(x: Any) -> str:
    v = safe_float(x)
    return "NA" if not math.isfinite(v) else f"{100.0*v:.2f}%"


def pp(x: Any) -> str:
    v = safe_float(x)
    return "NA" if not math.isfinite(v) else f"{v:+.2f}pp"


def findings_text(primary: pd.DataFrame, split: pd.DataFrame, robustness: pd.DataFrame, freeze: dict[str, Any], status: str) -> str:
    p = primary.iloc[0]
    disc = split[split["analysis_set"] == "DISCOVERY"].iloc[0]
    val = split[split["analysis_set"] == "VALIDATION"].iloc[0]
    lines = [
        "# Phase 7M.6H — Replacement-penetration outcome attribution",
        "",
        f"- Status: **{status}**",
        f"- Frozen matched pairs: **{freeze['pair_rows']:,}**",
        f"- Frozen pair-key SHA256: `{freeze['pair_key_sha256']}`",
        "- Rematching after outcome load: **NO**",
        "",
        "## Primary: P75 within 15 minutes from the +3s decision point",
        "",
        f"- Outcome-usable pairs: **{int(p['outcome_pairs']):,} / {int(p['frozen_pairs_before_outcome_filter']):,}**",
        f"- Replacement penetrated P75 rate: **{pct(p['penetrated_rate'])}**",
        f"- Not demonstrably penetrated P75 rate: **{pct(p['not_penetrated_rate'])}**",
        f"- Paired difference: **{pp(p['effect_pp'])}**",
        f"- Candidate-day bootstrap 95% CI: **[{pp(p['effect_ci_low_pp'])}, {pp(p['effect_ci_high_pp'])}]**",
        f"- Mechanical interpretation: **{p['interpretation']}**",
        "",
        "## Discovery / validation",
        "",
        f"- Discovery: {int(disc['outcome_pairs']):,} usable pairs; effect **{pp(disc['effect_pp'])}**; 95% CI **[{pp(disc['effect_ci_low_pp'])}, {pp(disc['effect_ci_high_pp'])}]**",
        f"- Validation: {int(val['outcome_pairs']):,} usable pairs; effect **{pp(val['effect_pp'])}**; 95% CI **[{pp(val['effect_ci_low_pp'])}, {pp(val['effect_ci_high_pp'])}]**",
        "",
        "## P75 horizon robustness — same frozen pairs",
        "",
    ]
    for _, r in robustness.iterrows():
        lines.append(
            f"- {int(r['outcome_horizon_sec'])//60}m: {int(r['outcome_pairs']):,} usable pairs; "
            f"penetrated {pct(r['penetrated_rate'])} vs non-penetrated {pct(r['not_penetrated_rate'])}; "
            f"difference **{pp(r['effect_pp'])}**; 95% CI **[{pp(r['effect_ci_low_pp'])}, {pp(r['effect_ci_high_pp'])}]**"
        )
    lines.extend([
        "",
        "## Interpretation limit",
        "",
        "This is a matched observational mechanism comparison. It can support or weaken the repeated-consumption interpretation, but it does not prove participant identity, FIFO execution, causal treatment effects, or profitability.",
        "",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def run_self_test() -> None:
    # Synthetic frozen pairs across four dates. Outcome rows are intentionally
    # duplicated across exposure states only by candidate ID, never by pair row.
    pairs = pd.DataFrame({
        "candidate_date": ["2026-06-01", "2026-06-02", "2026-06-18", "2026-06-19"],
        "direction": [1, -1, 1, -1],
        "penetrated_candidate_id": ["H1", "H2", "H3", "H4"],
        "not_penetrated_candidate_id": ["L1", "L2", "L3", "L4"],
        "high_candidate_id": ["H1", "H2", "H3", "H4"],
        "low_candidate_id": ["L1", "L2", "L3", "L4"],
        "high__replacement_penetrated": [1, 1, 1, 1],
        "low__replacement_penetrated": [0, 0, 0, 0],
    })
    ids = ["H1", "H2", "H3", "H4", "L1", "L2", "L3", "L4"]
    outcomes = pd.DataFrame({
        "candidate_id": ids,
        "decision_price_valid": [True] * 8,
        "h300_fully_observed": [True] * 8,
        "h600_fully_observed": [True] * 8,
        "h900_fully_observed": [True] * 8,
        "h1800_fully_observed": [True] * 8,
        "h300_origin_reach_75": [True, False, True, True, False, False, False, True],
        "h600_origin_reach_75": [True, True, True, True, False, False, True, True],
        "h900_origin_reach_75": [True, True, False, True, False, False, False, True],
        "h1800_origin_reach_75": [True, True, True, True, False, True, False, True],
    })
    joined = join_pair_outcomes(pairs, outcomes)
    r = paired_effect(joined, 900)
    assert r["outcome_pairs"] == 4
    assert abs(r["penetrated_rate"] - 0.75) < 1e-12
    assert abs(r["not_penetrated_rate"] - 0.25) < 1e-12
    assert abs(r["effect_pp"] - 50.0) < 1e-12
    assert r["penetrated_only_wins"] == 2
    assert r["not_penetrated_only_wins"] == 0
    lo, hi, n = bootstrap_matched_day(joined, 900, 2000, 123)
    assert n == 2000 and math.isfinite(lo) and math.isfinite(hi)
    assert interpretation(5.0, 1.0, 9.0) == "POSITIVE_ASSOCIATION_CI_ABOVE_ZERO"
    assert interpretation(-5.0, -9.0, -1.0) == "NEGATIVE_ASSOCIATION_CI_BELOW_ZERO"
    assert len(joined) == len(pairs)
    print("SELF-TEST PASS: frozen pair join, paired P75 effect, clustered bootstrap, and interpretation logic validated.")


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 7M.6H frozen-pair replacement-penetration outcome attribution")
    p.add_argument("--m6-root", type=Path, default=DEFAULT_M6_ROOT)
    p.add_argument("--m6-report-root", type=Path, default=DEFAULT_M6_REPORT_ROOT)
    p.add_argument("--m3-root", type=Path, default=DEFAULT_M3_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    require_stack()
    if int(args.bootstrap_reps) < 1000:
        raise RuntimeError("Refusing bootstrap_reps < 1000 for the primary outcome phase.")

    built_at = utc_now_iso()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)

    print("Phase 7M.6H - Frozen-pair replacement-penetration outcome attribution", flush=True)
    print("NO raw replay | NO rematching | NO ML | SAME frozen 695 pair manifest", flush=True)
    print("Primary: P75 within 15m from the +3s decision point", flush=True)

    print("[1/7] Validating Phase 7M.6F frozen pre-outcome authorization...", flush=True)
    pairs, m6_summary, freeze = validate_frozen_preoutcome_state(args.m6_root, args.m6_report_root)
    print(
        f"      frozen pairs={len(pairs):,}; D/V={freeze['discovery_pairs']}/{freeze['validation_pairs']}; "
        f"pair_hash={freeze['pair_key_sha256'][:16]}...",
        flush=True,
    )

    # Persist run config containing only frozen causal/future-separation contract
    # before opening the outcome file.
    config = {
        "builder_version": BUILDER_VERSION,
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "built_at_utc": built_at,
        "frozen_7m6f_schema": FROZEN_7M6F_SCHEMA,
        "frozen_pair_manifest": freeze,
        "primary_observation_horizon_sec": PRIMARY_OBSERVATION_HORIZON_SEC,
        "primary_outcome_horizon_sec": PRIMARY_OUTCOME_HORIZON_SEC,
        "primary_target_bps": PRIMARY_TARGET_BPS,
        "robustness_outcome_horizons_sec": list(ROBUSTNESS_OUTCOME_HORIZONS_SEC),
        "discovery_end_date": DISCOVERY_END_DATE,
        "validation_start_date": VALIDATION_START_DATE,
        "bootstrap_reps": int(args.bootstrap_reps),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "raw_replay": False,
        "rematching": False,
        "pair_repair": False,
        "ml": False,
        "outcomes_loaded_at_config_write": False,
    }
    atomic_json(args.output_root / "metadata" / "run_config.json", config)

    print("[2/7] Frozen pair manifest verified and hashed. Loading future outcomes NOW...", flush=True)
    outcomes = load_outcomes_after_freeze(args.m3_root)
    print(f"      +3s outcome rows={len(outcomes):,}", flush=True)

    print("[3/7] Joining outcomes onto the unchanged frozen pairs...", flush=True)
    joined = join_pair_outcomes(pairs, outcomes)
    atomic_parquet(args.output_root / "matched_pair_outcomes.parquet", joined)
    missing_hi = int(joined["penetrated__candidate_id"].isna().sum())
    missing_lo = int(joined["not_penetrated__candidate_id"].isna().sum())
    print(f"      joined pair rows={len(joined):,}; missing penetrated/non-penetrated outcome IDs={missing_hi}/{missing_lo}", flush=True)

    print("[4/7] Estimating frozen primary P75/15m contrast and D/V replication...", flush=True)
    primary, split, robustness = analyze_primary(joined, int(args.bootstrap_reps))
    atomic_csv(args.report_root / "primary_p75_15m.csv", primary)
    atomic_csv(args.report_root / "discovery_validation_p75_15m.csv", split)
    p = primary.iloc[0]
    print(
        f"      FULL usable={int(p['outcome_pairs']):,}/{int(p['frozen_pairs_before_outcome_filter']):,} "
        f"penetrated={100*safe_float(p['penetrated_rate']):.2f}% non-penetrated={100*safe_float(p['not_penetrated_rate']):.2f}% "
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
        status = "PASS" if bool_series(pd.Series([r["pass"]])).iloc[0] else "FAIL"
        print(f"      {status:4s} {r['invariant']}: actual={r['actual']} required={r['required']}", flush=True)

    print("[7/7] Writing mechanical findings and metadata...", flush=True)
    status = "PASS" if failed.empty else "FAIL"
    summary = {
        "builder_version": BUILDER_VERSION,
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "built_at_utc": built_at,
        "status": status,
        "scientific_state": "FROZEN_PAIR_REPLACEMENT_PENETRATION_OUTCOME_ATTRIBUTION_COMPLETE_NOT_CAUSAL_NOT_ALPHA",
        "frozen_pair_manifest_file_sha256": freeze["pair_manifest_file_sha256"],
        "frozen_pair_key_sha256": freeze["pair_key_sha256"],
        "frozen_pairs": int(len(pairs)),
        "discovery_pairs": EXPECTED_DISCOVERY_PAIRS,
        "validation_pairs": EXPECTED_VALIDATION_PAIRS,
        "primary_observation_horizon_sec": PRIMARY_OBSERVATION_HORIZON_SEC,
        "primary_outcome_horizon_sec": PRIMARY_OUTCOME_HORIZON_SEC,
        "primary_target_bps": PRIMARY_TARGET_BPS,
        "primary_outcome_usable_pairs": int(p["outcome_pairs"]),
        "primary_penetrated_rate": safe_float(p["penetrated_rate"]),
        "primary_not_penetrated_rate": safe_float(p["not_penetrated_rate"]),
        "primary_effect_pp": safe_float(p["effect_pp"]),
        "primary_ci_low_pp": safe_float(p["effect_ci_low_pp"]),
        "primary_ci_high_pp": safe_float(p["effect_ci_high_pp"]),
        "primary_interpretation": str(p["interpretation"]),
        "future_outcomes_loaded": True,
        "future_outcomes_loaded_only_after_pair_manifest_validation": True,
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
    print(f"Primary usable pairs     : {int(p['outcome_pairs']):,}", flush=True)
    print(f"P75/15m effect           : {safe_float(p['effect_pp']):+.2f}pp", flush=True)
    print(f"95% clustered-bootstrap  : [{safe_float(p['effect_ci_low_pp']):+.2f}, {safe_float(p['effect_ci_high_pp']):+.2f}]pp", flush=True)
    print(f"Interpretation           : {p['interpretation']}", flush=True)
    return 0 if failed.empty else 2


if __name__ == "__main__":
    raise SystemExit(main())
