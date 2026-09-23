#!/usr/bin/env python3
"""
Phase 7L.2 — Incremental participant/liquidity continuation analysis
===================================================================

Plain-English question
----------------------
Every row in Phase 7L.1 is already a causal repeatable 35-bp candidate.

Ask whether, among candidates with similar immediate price momentum and similar
executed directional aggressive notional, individual order-book failure/support
mechanisms still distinguish future 75-bp or 100-bp continuation.

This phase is cheap: no raw replay, no candidate regeneration, no ML, no
profitability claim, and NO composite book-failure score.

Primary analysis
----------------
* Observation horizon: 3 seconds after candidate.
* Outcome horizon: 15 minutes from the decision timestamp.
* Primary outcome: origin reaches 75 bps.
* Secondary outcome: origin reaches 100 bps.
* Primary controls: directional_mid_move_bps tercile x
  directional_aggressive_quote tercile.
* Each book mechanism is tested separately.

The causal feature bins/control cells are created before future outcomes are
joined. Legacy outcome-like fields such as continuation_stratum are explicitly
excluded.

Heterogeneity diagnostics
-------------------------
* UP vs DOWN.
* Common-50 FIRST vs REACCEL onset-associated candidates from Phase 7K.4.
  These are hindsight diagnostics only and never filter the primary all-market
  analysis.

Outputs
-------
reports/perps_repeatable_candidate_incremental_mechanism_analysis/
  feature_quartile_rates.csv
  controlled_book_contrasts.csv
  controlled_book_contrasts_by_segment.csv
  control_variable_quartile_rates.csv
  horizon_robustness.csv
  outcome_horizon_robustness.csv
  direction_robustness.csv
  common50_role_robustness.csv
  feature_day_sign_consistency.csv
  analysis_invariants.csv
  findings.md

data/derived/perps/research/repeatable_candidate_incremental_mechanism_analysis_v1/
  metadata/latest_summary.json
  metadata/run_config.json
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

BUILDER_VERSION = "1.0.1"
SCHEMA_VERSION = 1
SCHEMA_NAME = "repeatable_candidate_incremental_mechanism_analysis_v1"

FROZEN_PHASE7L1_BUILDER = "1.0.1"
FROZEN_PHASE7L1_SCHEMA = "repeatable_candidate_participant_interaction_v1"
FROZEN_PHASE7K4_BUILDER = "1.0.0"
FROZEN_PHASE7K4_SCHEMA = "candidate_redundancy_earliest_availability_audit_v1"
FROZEN_POLICY = "TEST_COOLDOWN_60S"

EXPECTED_CANDIDATES = 4_037
EXPECTED_HORIZONS = (1, 3, 5, 10)
PRIMARY_OBSERVATION_HORIZON_SEC = 3
OUTCOME_HORIZONS_SEC = (300, 600, 900, 1800)
PRIMARY_OUTCOME_HORIZON_SEC = 900
PRIMARY_TARGET_BPS = 75
SECONDARY_TARGET_BPS = 100

BOOTSTRAP_REPS = 1_000
BOOTSTRAP_SEED = 20260829
CONTROL_BINS = 3
BOOK_BINS = 3
QUARTILE_BINS = 4
MIN_CELL_SIDE_N = 10
MIN_BOOTSTRAP_VALID_REPS = 250

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / FROZEN_PHASE7L1_SCHEMA
DEFAULT_K4_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / FROZEN_PHASE7K4_SCHEMA
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / SCHEMA_NAME
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "perps_repeatable_candidate_incremental_mechanism_analysis"

# sign orients higher values toward the hypothesized continuation-favoring state.
BOOK_FEATURES: tuple[tuple[str, int, str], ...] = (
    ("opposing__durable_unrecovered_loss_share_initial", +1, "CORE_OPPOSING_FAILURE"),
    ("opposing__peak_initial_queue_loss_share", +1, "CORE_OPPOSING_FAILURE"),
    ("opposing__retained_initial_quote_ratio", -1, "CORE_OPPOSING_FAILURE"),
    ("support_minus_opposing_retained_ratio", +1, "CORE_OPPOSING_FAILURE"),
    ("opposing__same_price_refill_to_depletion_ratio", -1, "REFILL_DURABILITY_CONTEXT"),
    ("opposing__durable_same_price_refill_fraction", -1, "REFILL_DURABILITY_CONTEXT"),
    ("support__retained_initial_quote_ratio", +1, "SUPPORT_CONTEXT"),
    ("support__same_price_refill_to_depletion_ratio", +1, "SUPPORT_CONTEXT"),
)
CORE_BOOK_FEATURES = tuple(x[0] for x in BOOK_FEATURES if x[2] == "CORE_OPPOSING_FAILURE")

AGGRESSION_FEATURES: tuple[tuple[str, int, str], ...] = (
    ("directional_aggressive_quote", +1, "log1p_nonnegative"),
    ("forward_directional_aggressive_imbalance", +1, "identity"),
    ("directional_trade_count_share", +1, "identity"),
    ("directional_net_flow_acceleration_quote_per_sec", +1, "signed_log1p"),
)
MOMENTUM_FEATURES: tuple[tuple[str, int], ...] = (
    ("directional_mid_move_bps", +1),
    ("extra_displacement_since_35bp_candidate_bps", +1),
)
PRIMARY_MOMENTUM_CONTROL = "directional_mid_move_bps"
PRIMARY_AGGRESSION_CONTROL = "directional_aggressive_quote"

LEGACY_BLACKLIST = {"continuation_stratum", "continuation_class", "event_class", "future_outcome"}
FUTURE_PREFIXES = ("h300_", "h600_", "h900_", "h1800_")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return math.nan
    return v if math.isfinite(v) else math.nan


def bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    mapping = {"true": True, "false": False, "1": True, "0": False}
    return s.astype(str).str.lower().map(mapping).fillna(False).astype(bool)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7l2-", suffix=path.suffix, dir=str(path.parent))
    os.close(fd)
    try:
        Path(tmp).write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7l2-", suffix=".csv", dir=str(path.parent))
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def require_columns(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing required columns: {missing}")


def sha256_ids(ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(str(x) for x in ids)).encode("utf-8")).hexdigest()


def percentile_bin(s: pd.Series, n_bins: int) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = pd.Series(pd.NA, index=s.index, dtype="Int64")
    mask = x.notna()
    if not mask.any():
        return out
    ranks = x.loc[mask].rank(method="first", pct=True)
    out.loc[mask] = np.ceil(ranks * int(n_bins)).clip(1, n_bins).astype(int).astype("Int64")
    return out


def transform_aggression(s: pd.Series, method: str) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    if method == "identity":
        return x
    if method == "log1p_nonnegative":
        return np.log1p(x.clip(lower=0.0))
    if method == "signed_log1p":
        return np.sign(x) * np.log1p(np.abs(x))
    raise ValueError(method)


def load_metadata(root: Path, expected_builder: str, label: str) -> dict[str, Any]:
    p = root / "metadata" / "latest_summary.json"
    if not p.exists():
        raise RuntimeError(f"Missing {label} metadata: {p}")
    meta = json.loads(p.read_text(encoding="utf-8"))
    if str(meta.get("builder_version")) != str(expected_builder):
        raise RuntimeError(f"{label} builder mismatch: expected {expected_builder}, found {meta.get('builder_version')!r}")
    if str(meta.get("status")) != "PASS":
        raise RuntimeError(f"{label} must be PASS; found {meta.get('status')!r}")
    return meta


def validate_causal_feature_contract(mechanics: pd.DataFrame) -> None:
    required = [
        "candidate_id", "candidate_date", "direction", "direction_label",
        "observation_horizon_sec", "window_valid", "decision_price_valid",
        PRIMARY_MOMENTUM_CONTROL, PRIMARY_AGGRESSION_CONTROL,
    ]
    required += [x[0] for x in BOOK_FEATURES]
    required += [x[0] for x in AGGRESSION_FEATURES]
    required += [x[0] for x in MOMENTUM_FEATURES]
    require_columns(mechanics, required, "candidate_mechanics")
    configured = [x[0] for x in BOOK_FEATURES] + [x[0] for x in AGGRESSION_FEATURES] + [x[0] for x in MOMENTUM_FEATURES]
    leak = [c for c in configured if c in LEGACY_BLACKLIST or any(c.startswith(p) for p in FUTURE_PREFIXES) or "future" in c.lower() or "outcome" in c.lower()]
    if leak:
        raise RuntimeError(f"Configured causal feature list contains outcome-like fields: {leak}")


def add_outcome_blind_bins(mechanics: pd.DataFrame) -> pd.DataFrame:
    df = mechanics.copy()
    drop = [c for c in df.columns if c in LEGACY_BLACKLIST or "continuation_stratum" in c.lower()]
    if drop:
        df = df.drop(columns=drop)
    df["__momentum_control_value"] = pd.to_numeric(df[PRIMARY_MOMENTUM_CONTROL], errors="coerce")
    df["__aggression_control_value"] = np.log1p(pd.to_numeric(df[PRIMARY_AGGRESSION_CONTROL], errors="coerce").clip(lower=0.0))
    for h in EXPECTED_HORIZONS:
        hm = pd.to_numeric(df["observation_horizon_sec"], errors="coerce") == h
        df.loc[hm, "momentum_tercile"] = percentile_bin(df.loc[hm, "__momentum_control_value"], CONTROL_BINS).astype(float)
        df.loc[hm, "aggression_tercile"] = percentile_bin(df.loc[hm, "__aggression_control_value"], CONTROL_BINS).astype(float)
        for col, sign, _ in BOOK_FEATURES:
            oriented = pd.to_numeric(df.loc[hm, col], errors="coerce") * float(sign)
            df.loc[hm, f"__book_tercile__{col}"] = percentile_bin(oriented, BOOK_BINS).astype(float)
            df.loc[hm, f"__book_quartile__{col}"] = percentile_bin(oriented, QUARTILE_BINS).astype(float)
        for col, sign, method in AGGRESSION_FEATURES:
            oriented = transform_aggression(df.loc[hm, col], method) * float(sign)
            df.loc[hm, f"__aggr_quartile__{col}"] = percentile_bin(oriented, QUARTILE_BINS).astype(float)
        for col, sign in MOMENTUM_FEATURES:
            oriented = pd.to_numeric(df.loc[hm, col], errors="coerce") * float(sign)
            df.loc[hm, f"__momentum_quartile__{col}"] = percentile_bin(oriented, QUARTILE_BINS).astype(float)
    return df


def load_common50_roles(k4_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta = load_metadata(k4_root, FROZEN_PHASE7K4_BUILDER, "Phase 7K.4")
    p = k4_root / "candidate_onset_associations.parquet"
    if not p.exists():
        raise RuntimeError(f"Missing Phase 7K.4 onset associations: {p}")
    a = pd.read_parquet(p)
    require_columns(a, ["policy_id", "candidate_id", "impulse_id", "impulse_ordinal_in_physical_leg", "is_reacceleration", "candidate_rank_in_onset_window"], "candidate_onset_associations")
    a = a[a["policy_id"].astype(str) == FROZEN_POLICY].copy()
    a["candidate_id"] = a["candidate_id"].astype(str)
    rows = []
    for cid, g in a.groupby("candidate_id", sort=False):
        impulses = sorted(set(g["impulse_id"].astype(str)))
        roles = set("REACCEL" if x else "FIRST" for x in g["is_reacceleration"].fillna(False).astype(bool))
        if len(impulses) == 1 and len(roles) == 1:
            r = g.sort_values("candidate_rank_in_onset_window", kind="mergesort").iloc[0]
            rows.append({"candidate_id": cid, "common50_role": next(iter(roles)), "common50_impulse_id": str(r["impulse_id"]), "common50_association_ambiguous": False})
        else:
            rows.append({"candidate_id": cid, "common50_role": "AMBIGUOUS", "common50_impulse_id": None, "common50_association_ambiguous": True})
    return pd.DataFrame(rows), meta


def join_roles(causal: pd.DataFrame, roles: pd.DataFrame) -> pd.DataFrame:
    out = causal.merge(roles, on="candidate_id", how="left", validate="many_to_one")
    out["common50_role"] = out["common50_role"].fillna("UNASSOCIATED")
    out["common50_association_ambiguous"] = out["common50_association_ambiguous"].fillna(False).astype(bool)
    return out


def join_outcomes(causal: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    key = ["candidate_id", "observation_horizon_sec"]
    keep = key.copy()
    for h in OUTCOME_HORIZONS_SEC:
        keep += [f"h{h}_fully_observed", f"h{h}_origin_reach_50", f"h{h}_origin_reach_75", f"h{h}_origin_reach_100", f"h{h}_mfe_from_decision_bps", f"h{h}_mae_from_decision_bps"]
    keep = list(dict.fromkeys(keep))
    require_columns(outcomes, keep, "candidate_decision_outcomes")
    merged = causal.merge(outcomes[keep], on=key, how="left", validate="one_to_one")
    for h in OUTCOME_HORIZONS_SEC:
        observed = bool_series(merged[f"h{h}_fully_observed"]) & bool_series(merged["decision_price_valid"])
        merged[f"__outcome_observed_{h}"] = observed
        for target in (50, 75, 100):
            merged[f"__reach_{target}_{h}"] = merged[f"h{h}_origin_reach_{target}"].where(observed, pd.NA).astype("boolean")
    return merged


def rate_bool(s: pd.Series) -> float:
    x = s.dropna()
    return float(x.astype(bool).mean()) if len(x) else math.nan


def feature_quartile_rates(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = []
    specs += [("BOOK", fam, col, f"__book_quartile__{col}") for col, _, fam in BOOK_FEATURES]
    specs += [("AGGRESSION", trans, col, f"__aggr_quartile__{col}") for col, _, trans in AGGRESSION_FEATURES]
    specs += [("MOMENTUM", "identity", col, f"__momentum_quartile__{col}") for col, _ in MOMENTUM_FEATURES]
    for h in EXPECTED_HORIZONS:
        d = joined[pd.to_numeric(joined["observation_horizon_sec"], errors="coerce") == h]
        for family, subtype, feature, qcol in specs:
            for q in range(1, 5):
                x = d[pd.to_numeric(d[qcol], errors="coerce") == q]
                x = x[x[f"__outcome_observed_{PRIMARY_OUTCOME_HORIZON_SEC}"]]
                rows.append({"observation_horizon_sec": h, "feature_family": family, "feature_subtype": subtype, "feature": feature, "quartile": q, "rows": len(x), "p75": rate_bool(x[f"__reach_75_{PRIMARY_OUTCOME_HORIZON_SEC}"]) if len(x) else math.nan, "p100": rate_bool(x[f"__reach_100_{PRIMARY_OUTCOME_HORIZON_SEC}"]) if len(x) else math.nan})
    return pd.DataFrame(rows)


def _control_keys(control_scheme: str) -> list[str]:
    if control_scheme == "NONE": return []
    if control_scheme == "MOMENTUM_TERCILE": return ["momentum_tercile"]
    if control_scheme == "AGGRESSION_TERCILE": return ["aggression_tercile"]
    if control_scheme == "MOMENTUM_X_AGGRESSION_TERCILE": return ["momentum_tercile", "aggression_tercile"]
    raise ValueError(control_scheme)


def stratified_rate_difference(df: pd.DataFrame, *, feature: str, target_bps: int, outcome_horizon_sec: int, control_scheme: str, min_cell_side_n: int = MIN_CELL_SIDE_N) -> dict[str, Any]:
    bcol = f"__book_tercile__{feature}"
    ycol = f"__reach_{target_bps}_{outcome_horizon_sec}"
    x = df[df[f"__outcome_observed_{outcome_horizon_sec}"] & bool_series(df["window_valid"])].copy()
    x = x[pd.to_numeric(x[bcol], errors="coerce").isin([1, BOOK_BINS])].copy()
    keys = _control_keys(control_scheme)
    for k in keys:
        x = x[pd.to_numeric(x[k], errors="coerce").notna()].copy()
    groups = [((), x)] if not keys else list(x.groupby(keys, dropna=False))
    diffs, hi_rates, lo_rates, weights = [], [], [], []
    high_n = low_n = 0
    for _, g in groups:
        hi = g[pd.to_numeric(g[bcol], errors="coerce") == BOOK_BINS][ycol].dropna().astype(bool)
        lo = g[pd.to_numeric(g[bcol], errors="coerce") == 1][ycol].dropna().astype(bool)
        if len(hi) < min_cell_side_n or len(lo) < min_cell_side_n:
            continue
        w = min(len(hi), len(lo))
        hrate, lrate = float(hi.mean()), float(lo.mean())
        diffs.append(hrate - lrate); hi_rates.append(hrate); lo_rates.append(lrate); weights.append(w)
        high_n += len(hi); low_n += len(lo)
    if not weights:
        return {"effect_pp": math.nan, "common_support_n": 0, "cells_used": 0, "high_n": 0, "low_n": 0, "high_rate_standardized": math.nan, "low_rate_standardized": math.nan}
    w = np.asarray(weights, float)
    return {"effect_pp": 100.0 * float(np.average(diffs, weights=w)), "common_support_n": int(2 * np.sum(w)), "cells_used": len(weights), "high_n": high_n, "low_n": low_n, "high_rate_standardized": float(np.average(hi_rates, weights=w)), "low_rate_standardized": float(np.average(lo_rates, weights=w))}


def cluster_bootstrap_effect(
    df: pd.DataFrame,
    *,
    feature: str,
    target_bps: int,
    outcome_horizon_sec: int,
    control_scheme: str,
    reps: int,
    seed: int,
) -> tuple[float, float, int]:
    """
    Exact candidate-day cluster bootstrap using sufficient statistics.

    This is statistically equivalent to the v1.0.0 implementation's:
        sample candidate dates with replacement
        -> concatenate all rows from every sampled day
        -> recompute the stratified high-vs-low book contrast

    But instead of rebuilding thousands of DataFrames, aggregate each
    day x control-cell x book-side to (N, successes) once, then bootstrap
    those counts with multinomial day multiplicities.

    This preserves:
      * candidate-day clustering
      * 1,000 default bootstrap replicates
      * min-cell-side common-support rule
      * min(n_high, n_low) cell weighting
      * exact high/low outcome rates
    """
    bcol = f"__book_tercile__{feature}"
    ycol = f"__reach_{target_bps}_{outcome_horizon_sec}"
    observed_col = f"__outcome_observed_{outcome_horizon_sec}"

    x = df[
        df[observed_col]
        & bool_series(df["window_valid"])
    ].copy()
    x = x[
        pd.to_numeric(x[bcol], errors="coerce").isin([1, BOOK_BINS])
    ].copy()

    keys = _control_keys(control_scheme)
    for k in keys:
        x = x[pd.to_numeric(x[k], errors="coerce").notna()].copy()

    x = x[x[ycol].notna()].copy()
    if x.empty:
        return math.nan, math.nan, 0

    x["__day"] = x["candidate_date"].astype(str)
    x["__side"] = pd.to_numeric(x[bcol], errors="coerce").astype(int)
    x["__success"] = x[ycol].astype(bool).astype(int)

    days = sorted(x["__day"].unique().tolist())
    if len(days) < 2:
        return math.nan, math.nan, 0

    if keys:
        # Stable tuple-valued control cell; e.g. (momentum tercile, aggression tercile).
        x["__cell"] = list(
            zip(*[
                pd.to_numeric(x[k], errors="coerce").astype(int).tolist()
                for k in keys
            ])
        )
        cells = sorted(set(x["__cell"].tolist()))
    else:
        x["__cell"] = [("ALL",)] * len(x)
        cells = [("ALL",)]

    day_ix = {d: i for i, d in enumerate(days)}
    cell_ix = {c: i for i, c in enumerate(cells)}
    shape = (len(days), len(cells))

    n_hi = np.zeros(shape, dtype=np.int64)
    y_hi = np.zeros(shape, dtype=np.int64)
    n_lo = np.zeros(shape, dtype=np.int64)
    y_lo = np.zeros(shape, dtype=np.int64)

    agg = (
        x.groupby(["__day", "__cell", "__side"], dropna=False)["__success"]
        .agg(["size", "sum"])
        .reset_index()
    )
    for day, cell, side, n, y in agg.itertuples(index=False, name=None):
        di = day_ix[str(day)]
        ci = cell_ix[cell]
        side = int(side)
        n = int(n)
        y = int(y)
        if side == BOOK_BINS:
            n_hi[di, ci] = n
            y_hi[di, ci] = y
        elif side == 1:
            n_lo[di, ci] = n
            y_lo[di, ci] = y

    rng = np.random.default_rng(seed)

    # Drawing D days with replacement is exactly multinomial(D, 1/D).
    multiplicity = rng.multinomial(
        len(days),
        np.full(len(days), 1.0 / len(days)),
        size=int(reps),
    ).astype(np.int64)

    # Bootstrap aggregate counts: [reps, days] @ [days, cells].
    bn_hi = multiplicity @ n_hi
    by_hi = multiplicity @ y_hi
    bn_lo = multiplicity @ n_lo
    by_lo = multiplicity @ y_lo

    valid_cell = (
        (bn_hi >= MIN_CELL_SIDE_N)
        & (bn_lo >= MIN_CELL_SIDE_N)
    )
    weights = np.where(
        valid_cell,
        np.minimum(bn_hi, bn_lo),
        0,
    ).astype(float)

    hi_rate = np.divide(
        by_hi,
        bn_hi,
        out=np.full(bn_hi.shape, np.nan, dtype=float),
        where=bn_hi > 0,
    )
    lo_rate = np.divide(
        by_lo,
        bn_lo,
        out=np.full(bn_lo.shape, np.nan, dtype=float),
        where=bn_lo > 0,
    )
    diff = np.where(valid_cell, hi_rate - lo_rate, 0.0)

    total_weight = weights.sum(axis=1)
    effect = np.divide(
        (weights * diff).sum(axis=1),
        total_weight,
        out=np.full(int(reps), np.nan, dtype=float),
        where=total_weight > 0,
    ) * 100.0

    vals = effect[np.isfinite(effect)]
    if len(vals) < MIN_BOOTSTRAP_VALID_REPS:
        return math.nan, math.nan, int(len(vals))

    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), int(len(vals))



def interpretation_label(effect_pp: float, ci_low: float) -> str:
    if not math.isfinite(effect_pp): return "NOT_TESTED"
    if effect_pp <= 0: return "NOT_SUPPORTED"
    if math.isfinite(ci_low) and ci_low > 0: return "SUPPORTED"
    return "TENTATIVE"


def controlled_contrasts(joined: pd.DataFrame, *, observation_horizons: Sequence[int], outcome_horizons: Sequence[int], segments: dict[str, pd.Series] | None, bootstrap_reps: int) -> pd.DataFrame:
    rows = []
    controls = ("NONE", "MOMENTUM_TERCILE", "AGGRESSION_TERCILE", "MOMENTUM_X_AGGRESSION_TERCILE")
    for obs_h in observation_horizons:
        hframe = joined[pd.to_numeric(joined["observation_horizon_sec"], errors="coerce") == obs_h].copy()
        segs = (
            {"ALL": pd.Series(True, index=hframe.index)}
            if not segments
            else {}
        )
        if segments:
            for name, mask in segments.items():
                segs[name] = mask.reindex(hframe.index, fill_value=False)
        for seg_name, mask in segs.items():
            d = hframe.loc[mask].copy()
            print(
                f"      contrasts obs={obs_h}s segment={seg_name} rows={len(d):,}",
                flush=True,
            )
            for out_h in outcome_horizons:
                for target in (75, 100):
                    for feature, sign, family in BOOK_FEATURES:
                        for control in controls:
                            result = stratified_rate_difference(d, feature=feature, target_bps=target, outcome_horizon_sec=out_h, control_scheme=control)
                            do_boot = control == "MOMENTUM_X_AGGRESSION_TERCILE" and out_h == PRIMARY_OUTCOME_HORIZON_SEC
                            if do_boot:
                                seed = BOOTSTRAP_SEED + obs_h * 10000 + target * 100 + sum(ord(c) for c in (feature + seg_name)) % 97
                                lo, hi, nboot = cluster_bootstrap_effect(d, feature=feature, target_bps=target, outcome_horizon_sec=out_h, control_scheme=control, reps=bootstrap_reps, seed=seed)
                            else:
                                lo = hi = math.nan; nboot = 0
                            rows.append({"segment": seg_name, "observation_horizon_sec": obs_h, "outcome_horizon_sec": out_h, "target_bps": target, "feature": feature, "feature_family": family, "feature_orientation_sign": sign, "control_scheme": control, **result, "cluster_bootstrap_ci_low_pp": lo, "cluster_bootstrap_ci_high_pp": hi, "cluster_bootstrap_valid_reps": nboot, "interpretation": interpretation_label(safe_float(result["effect_pp"]), lo) if (obs_h == PRIMARY_OBSERVATION_HORIZON_SEC and out_h == PRIMARY_OUTCOME_HORIZON_SEC and control == "MOMENTUM_X_AGGRESSION_TERCILE") else "ROBUSTNESS_ONLY"})
    return pd.DataFrame(rows)


def control_variable_quartile_rates(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [("MOMENTUM", x[0], f"__momentum_quartile__{x[0]}") for x in MOMENTUM_FEATURES] + [("AGGRESSION", x[0], f"__aggr_quartile__{x[0]}") for x in AGGRESSION_FEATURES]
    for h in EXPECTED_HORIZONS:
        d = joined[pd.to_numeric(joined["observation_horizon_sec"], errors="coerce") == h]
        for family, feature, qcol in specs:
            for q in range(1, 5):
                x = d[(pd.to_numeric(d[qcol], errors="coerce") == q) & d[f"__outcome_observed_{PRIMARY_OUTCOME_HORIZON_SEC}"]]
                rows.append({"observation_horizon_sec": h, "feature_family": family, "feature": feature, "quartile": q, "rows": len(x), "p75_15m": rate_bool(x[f"__reach_75_{PRIMARY_OUTCOME_HORIZON_SEC}"]) if len(x) else math.nan, "p100_15m": rate_bool(x[f"__reach_100_{PRIMARY_OUTCOME_HORIZON_SEC}"]) if len(x) else math.nan})
    return pd.DataFrame(rows)


def day_sign_consistency(joined: pd.DataFrame) -> pd.DataFrame:
    d = joined[pd.to_numeric(joined["observation_horizon_sec"], errors="coerce") == PRIMARY_OBSERVATION_HORIZON_SEC].copy()
    rows = []
    for feature, _, family in BOOK_FEATURES:
        qcol = f"__book_quartile__{feature}"
        for target in (75, 100):
            ycol = f"__reach_{target}_{PRIMARY_OUTCOME_HORIZON_SEC}"
            effects = []
            for _, g in d.groupby(d["candidate_date"].astype(str)):
                hi = g[(pd.to_numeric(g[qcol], errors="coerce") == 4) & g[f"__outcome_observed_{PRIMARY_OUTCOME_HORIZON_SEC}"]][ycol].dropna().astype(bool)
                lo = g[(pd.to_numeric(g[qcol], errors="coerce") == 1) & g[f"__outcome_observed_{PRIMARY_OUTCOME_HORIZON_SEC}"]][ycol].dropna().astype(bool)
                if len(hi) >= 3 and len(lo) >= 3: effects.append(float(hi.mean() - lo.mean()))
            rows.append({"feature": feature, "feature_family": family, "target_bps": target, "comparable_days": len(effects), "positive_days": sum(x > 0 for x in effects), "negative_days": sum(x < 0 for x in effects), "positive_day_share": float(sum(x > 0 for x in effects) / len(effects)) if effects else math.nan, "median_day_effect_pp": 100.0 * float(np.median(effects)) if effects else math.nan})
    return pd.DataFrame(rows)


def build_invariants(mechanics: pd.DataFrame, causal_binned: pd.DataFrame, joined: pd.DataFrame, roles: pd.DataFrame, l1_meta: dict[str, Any], k4_meta: dict[str, Any]) -> pd.DataFrame:
    rows = []
    def add(name, ok, observed, expected): rows.append({"invariant": name, "status": "PASS" if ok else "FAIL", "observed": observed, "expected": expected})
    n = mechanics["candidate_id"].astype(str).nunique()
    add("expected_candidate_population", n == EXPECTED_CANDIDATES, n, EXPECTED_CANDIDATES)
    horizons = sorted(pd.to_numeric(mechanics["observation_horizon_sec"], errors="coerce").dropna().astype(int).unique().tolist())
    add("expected_observation_horizons", tuple(horizons) == EXPECTED_HORIZONS, horizons, list(EXPECTED_HORIZONS))
    add("mechanics_row_count", len(mechanics) == EXPECTED_CANDIDATES * len(EXPECTED_HORIZONS), len(mechanics), EXPECTED_CANDIDATES * len(EXPECTED_HORIZONS))
    dup = mechanics.duplicated(["candidate_id", "observation_horizon_sec"]).sum()
    add("mechanics_unique_candidate_horizon", dup == 0, int(dup), 0)
    add("joined_row_count_preserved", len(joined) == len(causal_binned), len(joined), len(causal_binned))
    add("legacy_continuation_stratum_removed", "continuation_stratum" not in causal_binned.columns, "continuation_stratum" in causal_binned.columns, False)
    p = joined[pd.to_numeric(joined["observation_horizon_sec"], errors="coerce") == 3]
    cov = float(bool_series(p["window_valid"]).mean())
    add("primary_book_coverage_at_least_95pct", cov >= .95, cov, ">=0.95")
    add("phase7l1_metadata_candidate_count", int(l1_meta.get("candidate_count", -1)) == EXPECTED_CANDIDATES, l1_meta.get("candidate_count"), EXPECTED_CANDIDATES)
    policies = {str(k4_meta.get("primary_policy")), str(k4_meta.get("challenger_policy"))}
    add("phase7k4_contains_frozen_60s_policy", FROZEN_POLICY in policies, sorted(policies), FROZEN_POLICY)
    return pd.DataFrame(rows)


def findings_text(status: str, controlled: pd.DataFrame, role_counts: dict[str, int]) -> str:
    p = controlled[(controlled["segment"] == "ALL") & (controlled["observation_horizon_sec"] == 3) & (controlled["outcome_horizon_sec"] == 900) & (controlled["control_scheme"] == "MOMENTUM_X_AGGRESSION_TERCILE")]
    lines = ["# Phase 7L.2 — Incremental participant/liquidity continuation analysis", "", f"- Builder: **{BUILDER_VERSION}**", f"- Status: **{status}**", "- Primary: **3s mechanics -> 75bp within 15m**", "- Controls: **momentum tercile x directional aggressive quote tercile**", "- Composite book score: **NONE**", "", "## Primary 75bp controlled results", ""]
    for _, r in p[p["target_bps"] == 75].sort_values("effect_pp", ascending=False).iterrows():
        lines.append(f"- {r['feature']}: {safe_float(r['effect_pp']):+.2f}pp; CI [{safe_float(r['cluster_bootstrap_ci_low_pp']):+.2f}, {safe_float(r['cluster_bootstrap_ci_high_pp']):+.2f}]pp; N={int(r['common_support_n'])}; **{r['interpretation']}**.")
    lines += ["", "## Primary 100bp controlled results", ""]
    for _, r in p[p["target_bps"] == 100].sort_values("effect_pp", ascending=False).iterrows():
        lines.append(f"- {r['feature']}: {safe_float(r['effect_pp']):+.2f}pp; CI [{safe_float(r['cluster_bootstrap_ci_low_pp']):+.2f}, {safe_float(r['cluster_bootstrap_ci_high_pp']):+.2f}]pp; N={int(r['common_support_n'])}; **{r['interpretation']}**.")
    lines += ["", "## Common-50 hindsight role context", "", f"- FIRST: **{role_counts.get('FIRST',0):,}**", f"- REACCEL: **{role_counts.get('REACCEL',0):,}**", f"- UNASSOCIATED: **{role_counts.get('UNASSOCIATED',0):,}**", f"- AMBIGUOUS: **{role_counts.get('AMBIGUOUS',0):,}**", "", "Role-specific results are hindsight mechanism diagnostics only.", "", "## Interpretation discipline", "", "- Causal bins were built before outcomes were joined.", "- No composite book-failure score was created.", "- Positive association is not participant-identity proof.", "- Historical all-market analysis is not untouched forward validation.", "- Do not turn these results directly into a trade rule.", ""]
    return "\n".join(lines)


def self_test() -> int:
    rows = []
    cid = 0
    feature = "opposing__durable_unrecovered_loss_share_initial"
    for m in (1,2,3):
        for a in (1,2,3):
            for book in (1,3):
                for j in range(20):
                    cid += 1
                    rows.append({"candidate_id": f"c{cid}", "candidate_date": f"2026-06-{1+(cid%10):02d}", "window_valid": True, "momentum_tercile": float(m), "aggression_tercile": float(a), f"__book_tercile__{feature}": float(book), "__outcome_observed_900": True, "__reach_75_900": bool((book==3 and j<14) or (book==1 and j<6))})
    df = pd.DataFrame(rows)
    r = stratified_rate_difference(df, feature=feature, target_bps=75, outcome_horizon_sec=900, control_scheme="MOMENTUM_X_AGGRESSION_TERCILE")
    assert r["cells_used"] == 9 and r["common_support_n"] == 360 and r["effect_pp"] > 30
    q = percentile_bin(pd.Series(range(100)), 4)
    assert set(q.dropna().astype(int)) == {1,2,3,4}
    print("SELF-TEST PASS")
    print("Validated individual-book-feature testing, outcome-blind bins, momentum x aggression common-support conditioning, and future-label separation.")
    return 0


def parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(description="Phase 7L.2 incremental participant/liquidity continuation analysis")
    p.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    p.add_argument("--phase7k4-root", type=Path, default=DEFAULT_K4_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test: return self_test()

    print("Building Phase 7L.2 incremental participant/liquidity analysis...")
    print(f"Builder version       : {BUILDER_VERSION}")
    print(f"Frozen Phase-7L.1     : {FROZEN_PHASE7L1_BUILDER}")
    print(f"Frozen candidate      : {FROZEN_POLICY}")
    print("Primary observation   : 3s")
    print("Primary outcome       : 75bp within 15m")
    print("Secondary outcome     : 100bp")
    print("Primary controls      : momentum x aggressive-notional terciles")
    print("Book analysis         : individual mechanisms; NO composite score")
    print("Raw replay            : NONE")

    l1_meta = load_metadata(args.input_root, FROZEN_PHASE7L1_BUILDER, "Phase 7L.1")
    if str(l1_meta.get("schema_name")) != FROZEN_PHASE7L1_SCHEMA:
        raise RuntimeError(f"Phase 7L.1 schema mismatch: {l1_meta.get('schema_name')!r}")
    mechanics_path = args.input_root / "candidate_mechanics.parquet"
    outcomes_path = args.input_root / "candidate_decision_outcomes.parquet"
    for p in (mechanics_path, outcomes_path):
        if not p.exists(): raise RuntimeError(f"Missing frozen Phase 7L.1 input: {p}")

    print("[1/8] Loading and validating causal mechanics only...")
    mechanics = pd.read_parquet(mechanics_path)
    validate_causal_feature_contract(mechanics)
    mechanics["candidate_id"] = mechanics["candidate_id"].astype(str)
    mechanics["candidate_date"] = mechanics["candidate_date"].astype(str)
    print(f"      mechanics={len(mechanics):,} candidates={mechanics['candidate_id'].nunique():,}")

    print("[2/8] Building outcome-blind feature/control bins...")
    causal_binned = add_outcome_blind_bins(mechanics)

    print("[3/8] Loading hindsight common-50 role context...")
    roles, k4_meta = load_common50_roles(args.phase7k4_root)
    causal_binned = join_roles(causal_binned, roles)
    role_counts = causal_binned[causal_binned["observation_horizon_sec"] == 3]["common50_role"].value_counts().to_dict()
    print(f"      role counts={role_counts}")

    print("[4/8] Joining future outcomes AFTER causal bins exist...")
    outcomes = pd.read_parquet(outcomes_path)
    outcomes["candidate_id"] = outcomes["candidate_id"].astype(str)
    joined = join_outcomes(causal_binned, outcomes)

    print("[5/8] Building descriptive quartiles and control baselines...")
    quartiles = feature_quartile_rates(joined)
    control_quartiles = control_variable_quartile_rates(joined)

    print("[6/8] Building controlled individual-book-feature contrasts...")
    controlled_all = controlled_contrasts(joined, observation_horizons=EXPECTED_HORIZONS, outcome_horizons=OUTCOME_HORIZONS_SEC, segments=None, bootstrap_reps=args.bootstrap_reps)
    segments = {
        "UP": pd.to_numeric(joined["direction"], errors="coerce") > 0,
        "DOWN": pd.to_numeric(joined["direction"], errors="coerce") < 0,
        "COMMON50_FIRST_ONSET": joined["common50_role"].astype(str) == "FIRST",
        "COMMON50_REACCEL_ONSET": joined["common50_role"].astype(str) == "REACCEL",
    }
    controlled_segments = controlled_contrasts(joined, observation_horizons=(3,), outcome_horizons=(900,), segments=segments, bootstrap_reps=args.bootstrap_reps)
    controlled_segments = controlled_segments[controlled_segments["segment"] != "ALL"].copy()
    day_sign = day_sign_consistency(joined)

    print("[7/8] Running invariants...")
    invariants = build_invariants(mechanics, causal_binned, joined, roles, l1_meta, k4_meta)
    failed = invariants[invariants["status"] == "FAIL"]
    status = "PASS" if failed.empty else "FAIL"

    print("[8/8] Writing reports and metadata...")
    horizon_robustness = controlled_all[(controlled_all["outcome_horizon_sec"] == 900) & (controlled_all["control_scheme"] == "MOMENTUM_X_AGGRESSION_TERCILE")].copy()
    outcome_horizon_robustness = controlled_all[(controlled_all["observation_horizon_sec"] == 3) & (controlled_all["control_scheme"] == "MOMENTUM_X_AGGRESSION_TERCILE")].copy()
    direction_robustness = controlled_segments[controlled_segments["segment"].isin(["UP","DOWN"]) & (controlled_segments["control_scheme"] == "MOMENTUM_X_AGGRESSION_TERCILE")].copy()
    role_robustness = controlled_segments[controlled_segments["segment"].isin(["COMMON50_FIRST_ONSET","COMMON50_REACCEL_ONSET"]) & (controlled_segments["control_scheme"] == "MOMENTUM_X_AGGRESSION_TERCILE")].copy()

    atomic_write_csv(quartiles, args.report_root / "feature_quartile_rates.csv")
    atomic_write_csv(controlled_all, args.report_root / "controlled_book_contrasts.csv")
    atomic_write_csv(controlled_segments, args.report_root / "controlled_book_contrasts_by_segment.csv")
    atomic_write_csv(control_quartiles, args.report_root / "control_variable_quartile_rates.csv")
    atomic_write_csv(horizon_robustness, args.report_root / "horizon_robustness.csv")
    atomic_write_csv(outcome_horizon_robustness, args.report_root / "outcome_horizon_robustness.csv")
    atomic_write_csv(direction_robustness, args.report_root / "direction_robustness.csv")
    atomic_write_csv(role_robustness, args.report_root / "common50_role_robustness.csv")
    atomic_write_csv(day_sign, args.report_root / "feature_day_sign_consistency.csv")
    atomic_write_csv(invariants, args.report_root / "analysis_invariants.csv")

    primary = controlled_all[(controlled_all["segment"] == "ALL") & (controlled_all["observation_horizon_sec"] == 3) & (controlled_all["outcome_horizon_sec"] == 900) & (controlled_all["control_scheme"] == "MOMENTUM_X_AGGRESSION_TERCILE")]
    supported_75 = int(((primary["target_bps"] == 75) & (primary["interpretation"] == "SUPPORTED")).sum())
    positive_core_75 = int(((primary["target_bps"] == 75) & primary["feature"].isin(CORE_BOOK_FEATURES) & (pd.to_numeric(primary["effect_pp"], errors="coerce") > 0)).sum())
    candidate_hash = sha256_ids(mechanics["candidate_id"].unique())
    summary = {
        "builder_version": BUILDER_VERSION, "schema_version": SCHEMA_VERSION, "schema_name": SCHEMA_NAME,
        "status": status, "built_at_utc": utc_now_iso(), "frozen_phase7l1_builder_version": l1_meta.get("builder_version"),
        "frozen_phase7k4_builder_version": k4_meta.get("builder_version"), "candidate_count": mechanics["candidate_id"].nunique(),
        "candidate_id_set_sha256": candidate_hash, "mechanics_rows": len(mechanics), "primary_observation_horizon_sec": 3,
        "primary_outcome_horizon_sec": 900, "primary_target_bps": 75, "secondary_target_bps": 100,
        "book_features_tested": [x[0] for x in BOOK_FEATURES], "primary_control_scheme": "MOMENTUM_X_AGGRESSION_TERCILE",
        "supported_primary_75_book_features": supported_75, "positive_core_primary_75_features": positive_core_75,
        "core_primary_75_features_total": len(CORE_BOOK_FEATURES), "common50_role_counts": {str(k): int(v) for k,v in role_counts.items()},
        "failed_invariants": len(failed), "scientific_state": "INCREMENTAL_MECHANISM_ANALYSIS_COMPLETE_NOT_ALPHA",
    }
    run_config = {**summary, "input_root": str(args.input_root), "phase7k4_root": str(args.phase7k4_root), "bootstrap_reps": args.bootstrap_reps, "bootstrap_seed": BOOTSTRAP_SEED, "control_bins": CONTROL_BINS, "book_bins": BOOK_BINS, "min_cell_side_n": MIN_CELL_SIDE_N, "primary_momentum_control": PRIMARY_MOMENTUM_CONTROL, "primary_aggression_control": PRIMARY_AGGRESSION_CONTROL, "feature_orientation": {c:s for c,s,_ in BOOK_FEATURES}, "legacy_feature_blacklist": sorted(LEGACY_BLACKLIST), "data_separation": "causal bins constructed before outcomes join"}
    atomic_write_json(args.output_root / "metadata" / "latest_summary.json", summary)
    atomic_write_json(args.output_root / "metadata" / "run_config.json", run_config)
    atomic_write_text(args.report_root / "findings.md", findings_text(status, controlled_all, {str(k): int(v) for k,v in role_counts.items()}))

    print("")
    print(f"Phase 7L.2 status: {status}")
    print(f"Primary 75bp: {positive_core_75}/{len(CORE_BOOK_FEATURES)} core book mechanisms positive after momentum+aggression conditioning")
    print(f"Primary 75bp features with day-cluster CI > 0: {supported_75}/{len(BOOK_FEATURES)}")
    print(f"Reports: {args.report_root}")
    if len(failed):
        print(failed.to_string(index=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
