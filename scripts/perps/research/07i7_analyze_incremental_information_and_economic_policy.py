#!/usr/bin/env python3
"""
Phase 7I.7 — Incremental-information and economic-policy gate.

Scientific role
---------------
Cheap downstream analysis over the frozen Phase-7I.6 analysis parquet.
NO raw-depth replay. NO trade replay. NO candidate regeneration. NO ML.

Primary questions
-----------------
1. Incremental information:
   Does exact opposing-book failure still add continuation information after
   conditioning on two obvious causal alternatives measured over the same
   observation window:
       - immediate directional price continuation, and
       - directional aggressive executed notional?

2. Economic plausibility:
   From the *actual causal decision timestamp*, do any transparent target/stop/
   horizon policies look plausible after a range of round-trip cost assumptions?

Important accounting limitation
--------------------------------
The frozen 7I.4 outcome substrate contains first-passage, MFE and MAE, but not
an exact mark-to-market return at timeout. Therefore this script does NOT claim
an exact backtest P&L. It reports:
    - target-first / stop-first / timeout rates,
    - resolved-only expectancy,
    - a clearly labelled flat-timeout P&L proxy (timeout gross return = 0), and
    - a worst-timeout proxy (timeout treated like a stop loss).

No thresholds are tuned against outcomes. Existing 7I.6 book-failure quartiles
are frozen. Momentum and directional-notional quartiles are constructed without
using future labels, separately at each observation horizon.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

BUILDER_VERSION = "1.0.0"
SCHEMA_NAME = "causal_candidate_incremental_economic_gate_v1"
SCHEMA_VERSION = 1
FROZEN_PHASE7I6_BUILDER = "1.0.0"
FROZEN_PHASE7I6_SCHEMA = "causal_candidate_participant_interaction_analysis_v1"
EXPECTED_CANDIDATES = 2_729
EXPECTED_HORIZONS = (1, 3, 5, 10)
PRIMARY_DECISION_HORIZON_SEC = 3
PRIMARY_OUTCOME_HORIZON_SEC = 1_800

# Outcome-blind adjustment design.
ADJUSTMENT_BINS = 4
MIN_STRATUM_PER_BOOK_GROUP = 8
MIN_INCREMENTAL_COMMON_SUPPORT_N = 250
BOOTSTRAP_REPS = 2_000
BOOTSTRAP_SEED = 20260826

# Economic grid. Costs are total round-trip bps and are deliberately broad.
TARGETS_BPS = (10, 20, 40)
STOPS_BPS = (10, 15, 20)
HOLDING_HORIZONS_SEC = (300, 600, 900, 1_800)
COST_GRID_BPS = (0.0, 4.0, 8.0, 12.0, 16.0)
PRIMARY_ROUND_TRIP_COST_BPS = 12.0
PRIMARY_POLICY_TARGET_BPS = 40
PRIMARY_POLICY_STOP_BPS = 15
PRIMARY_POLICY_HORIZON_SEC = 1_800
PRIMARY_ECONOMIC_GROUPS = ("BOOK_FAILURE_Q4", "PRESSURE_PLUS_FAILURE")
MIN_POLICY_N = 100
MIN_DAY_N = 5

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / FROZEN_PHASE7I6_SCHEMA
DEFAULT_INPUT_PATH = DEFAULT_INPUT_ROOT / "participant_interaction_analysis.parquet"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / SCHEMA_NAME
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "perps_causal_candidate_incremental_economic_gate"


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
    return s.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False}).fillna(False).astype(bool)


def require_columns(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing required columns: {missing}")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def deterministic_quantile_bin(values: pd.Series, q: int) -> pd.Series:
    """Outcome-blind deterministic approximately equal-sized bins, 1..q."""
    x = pd.to_numeric(values, errors="coerce")
    out = pd.Series(pd.NA, index=values.index, dtype="Int64")
    valid = x.notna()
    if not valid.any():
        return out
    # rank(method='first') makes ties deterministic in existing row order.
    ranks = x.loc[valid].rank(method="first", pct=True)
    bins = np.ceil(ranks.to_numpy(float) * q).astype(int)
    bins = np.clip(bins, 1, q)
    out.loc[valid] = bins
    return out


def add_outcome_blind_condition_bins(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    baseline_rows: list[dict[str, Any]] = []
    out["momentum_quartile"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["directional_notional_quartile"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    for h in EXPECTED_HORIZONS:
        hm = pd.to_numeric(out["observation_horizon_sec"], errors="coerce") == h
        # The actual displacement consumed between the causal 35-bp timestamp and
        # the decision time is the cleanest directly causal momentum measure.
        mom = pd.to_numeric(out.loc[hm, "extra_displacement_since_35bp_candidate_bps"], errors="coerce")
        notion = pd.to_numeric(out.loc[hm, "directional_aggressive_quote"], errors="coerce")
        out.loc[hm, "momentum_quartile"] = deterministic_quantile_bin(mom, ADJUSTMENT_BINS)
        out.loc[hm, "directional_notional_quartile"] = deterministic_quantile_bin(notion, ADJUSTMENT_BINS)
        for name, s in (("MOMENTUM", mom), ("DIRECTIONAL_NOTIONAL", notion)):
            xs = s.dropna()
            baseline_rows.append({
                "observation_horizon_sec": h,
                "feature": name,
                "n": int(len(xs)),
                "median": float(xs.median()) if len(xs) else math.nan,
                "q25": float(xs.quantile(0.25)) if len(xs) else math.nan,
                "q75": float(xs.quantile(0.75)) if len(xs) else math.nan,
            })
    out["momentum_x_notional_stratum"] = (
        out["momentum_quartile"].astype("string") + "x" + out["directional_notional_quartile"].astype("string")
    )
    return out, pd.DataFrame(baseline_rows)


def outcome_rate(s: pd.Series) -> float:
    x = s.dropna()
    return float(x.astype(bool).mean()) if len(x) else math.nan


def observed_primary(df: pd.DataFrame) -> pd.DataFrame:
    return df[bool_series(df["primary_outcome_observed"])].copy()


def adjustment_stratum_col(adjustment: str) -> str:
    return {
        "MOMENTUM_ONLY": "momentum_quartile",
        "NOTIONAL_ONLY": "directional_notional_quartile",
        "MOMENTUM_X_NOTIONAL": "momentum_x_notional_stratum",
    }[adjustment]


def standardized_book_failure_difference(
    df: pd.DataFrame,
    outcome_col: str,
    adjustment: str,
    min_each: int = MIN_STRATUM_PER_BOOK_GROUP,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Direct standardization of book Q4 vs Q1 within common-support strata."""
    d = observed_primary(df)
    d = d[pd.to_numeric(d["book_quartile"], errors="coerce").isin([1, 4])].copy()
    scol = adjustment_stratum_col(adjustment)
    rows: list[dict[str, Any]] = []
    for stratum, x in d.groupby(scol, dropna=True, sort=True):
        hi = x[pd.to_numeric(x["book_quartile"], errors="coerce") == 4]
        lo = x[pd.to_numeric(x["book_quartile"], errors="coerce") == 1]
        if len(hi) < min_each or len(lo) < min_each:
            continue
        ph = outcome_rate(hi[outcome_col])
        pl = outcome_rate(lo[outcome_col])
        if not (math.isfinite(ph) and math.isfinite(pl)):
            continue
        rows.append({
            "adjustment": adjustment,
            "stratum": str(stratum),
            "high_failure_n": int(len(hi)),
            "low_failure_n": int(len(lo)),
            "combined_n": int(len(hi) + len(lo)),
            "high_failure_rate": ph,
            "low_failure_rate": pl,
            "difference": ph - pl,
        })
    detail = pd.DataFrame(rows)
    if detail.empty:
        return {
            "adjustment": adjustment,
            "outcome": outcome_col,
            "common_support_strata": 0,
            "common_support_n": 0,
            "standardized_high_failure_rate": math.nan,
            "standardized_low_failure_rate": math.nan,
            "standardized_difference": math.nan,
        }, detail
    w = detail["combined_n"].astype(float)
    w = w / w.sum()
    ph = float(np.sum(w * detail["high_failure_rate"].astype(float)))
    pl = float(np.sum(w * detail["low_failure_rate"].astype(float)))
    return {
        "adjustment": adjustment,
        "outcome": outcome_col,
        "common_support_strata": int(len(detail)),
        "common_support_n": int(detail["combined_n"].sum()),
        "standardized_high_failure_rate": ph,
        "standardized_low_failure_rate": pl,
        "standardized_difference": ph - pl,
    }, detail


def bootstrap_standardized_difference(
    df: pd.DataFrame,
    outcome_col: str,
    adjustment: str,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    d = observed_primary(df)
    days = sorted(d["candidate_date"].astype(str).unique().tolist())
    if len(days) < 2:
        return {"bootstrap_reps": 0, "estimate": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    estimate, _ = standardized_book_failure_difference(d, outcome_col, adjustment)
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    day_groups = {day: d[d["candidate_date"].astype(str) == day] for day in days}
    for _ in range(int(reps)):
        sampled = rng.choice(days, size=len(days), replace=True)
        chunks = []
        for i, day in enumerate(sampled):
            x = day_groups[str(day)].copy()
            # Artificial bootstrap day key preserves multiplicity while retaining
            # all candidates from a sampled cluster/day.
            x["candidate_date"] = f"B{i:03d}_{day}"
            chunks.append(x)
        b = pd.concat(chunks, ignore_index=True)
        r, _ = standardized_book_failure_difference(b, outcome_col, adjustment)
        v = safe_float(r.get("standardized_difference"))
        if math.isfinite(v):
            vals.append(v)
    if not vals:
        return {"bootstrap_reps": 0, "estimate": safe_float(estimate.get("standardized_difference")), "ci_low": math.nan, "ci_high": math.nan}
    arr = np.asarray(vals, dtype=float)
    return {
        "bootstrap_reps": int(len(arr)),
        "estimate": safe_float(estimate.get("standardized_difference")),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
    }


def chronological_eras(dates: pd.Series, n: int = 4) -> pd.Series:
    unique = sorted(pd.Series(dates.astype(str).unique()).tolist())
    mapping: dict[str, str] = {}
    chunks = np.array_split(np.array(unique, dtype=object), n)
    for i, chunk in enumerate(chunks, start=1):
        for day in chunk.tolist():
            mapping[str(day)] = f"ERA_{i}"
    return dates.astype(str).map(mapping)


def build_incremental_reports(df: pd.DataFrame, reps: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = df[pd.to_numeric(df["observation_horizon_sec"], errors="coerce") == PRIMARY_DECISION_HORIZON_SEC].copy()
    adjustments = ("MOMENTUM_ONLY", "NOTIONAL_ONLY", "MOMENTUM_X_NOTIONAL")
    outcomes = ("reach_75_30m", "reach_100_30m")
    summary_rows: list[dict[str, Any]] = []
    strata_parts: list[pd.DataFrame] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for adjustment in adjustments:
        for outcome in outcomes:
            r, detail = standardized_book_failure_difference(p, outcome, adjustment)
            summary_rows.append(r)
            if not detail.empty:
                detail = detail.copy()
                detail["outcome"] = outcome
                strata_parts.append(detail)
            b = bootstrap_standardized_difference(p, outcome, adjustment, reps, BOOTSTRAP_SEED + len(bootstrap_rows))
            bootstrap_rows.append({"adjustment": adjustment, "outcome": outcome, **b})

    # Primary direction robustness: strongest 4x4 adjustment and P75.
    dir_rows: list[dict[str, Any]] = []
    for direction, x in p.groupby("direction_label", dropna=False):
        r, _ = standardized_book_failure_difference(x, "reach_75_30m", "MOMENTUM_X_NOTIONAL")
        dir_rows.append({"direction_label": str(direction), **r})

    # Four chronological blocks, deliberately descriptive rather than "new holdout".
    p["analysis_era"] = chronological_eras(p["candidate_date"], 4)
    era_rows: list[dict[str, Any]] = []
    for era, x in p.groupby("analysis_era", dropna=False, sort=True):
        r, _ = standardized_book_failure_difference(x, "reach_75_30m", "MOMENTUM_X_NOTIONAL")
        era_rows.append({"analysis_era": str(era), **r})

    summary = pd.DataFrame(summary_rows)
    strata = pd.concat(strata_parts, ignore_index=True) if strata_parts else pd.DataFrame()
    bootstrap = pd.DataFrame(bootstrap_rows)
    direction = pd.DataFrame(dir_rows)
    eras = pd.DataFrame(era_rows)
    return summary, strata, bootstrap, direction, eras


def build_incremental_gate(summary: pd.DataFrame, bootstrap: pd.DataFrame, direction: pd.DataFrame, eras: pd.DataFrame) -> pd.DataFrame:
    s = summary[(summary["adjustment"] == "MOMENTUM_X_NOTIONAL") & (summary["outcome"] == "reach_75_30m")]
    b = bootstrap[(bootstrap["adjustment"] == "MOMENTUM_X_NOTIONAL") & (bootstrap["outcome"] == "reach_75_30m")]
    if s.empty or b.empty:
        return pd.DataFrame([{"gate": "BOOK_FAILURE_INCREMENTAL_AFTER_MOMENTUM_AND_NOTIONAL", "status": "FAIL", "reason": "MISSING_PRIMARY_RESULT"}])
    sr = s.iloc[0]
    br = b.iloc[0]
    dir_positive = int((pd.to_numeric(direction["standardized_difference"], errors="coerce") > 0).sum()) if len(direction) else 0
    era_positive = int((pd.to_numeric(eras["standardized_difference"], errors="coerce") > 0).sum()) if len(eras) else 0
    checks = {
        "common_support_n_ge_250": int(sr["common_support_n"]) >= MIN_INCREMENTAL_COMMON_SUPPORT_N,
        "adjusted_p75_lift_ge_10pp": safe_float(sr["standardized_difference"]) >= 0.10,
        "day_cluster_ci_low_gt_0": safe_float(br["ci_low"]) > 0,
        "both_directions_positive": dir_positive >= 2,
        "three_of_four_eras_positive": era_positive >= 3,
    }
    return pd.DataFrame([{
        "gate": "BOOK_FAILURE_INCREMENTAL_AFTER_MOMENTUM_AND_NOTIONAL",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "common_support_n": int(sr["common_support_n"]),
        "adjusted_p75_high_failure": safe_float(sr["standardized_high_failure_rate"]),
        "adjusted_p75_low_failure": safe_float(sr["standardized_low_failure_rate"]),
        "adjusted_p75_difference": safe_float(sr["standardized_difference"]),
        "bootstrap_ci_low": safe_float(br["ci_low"]),
        "bootstrap_ci_high": safe_float(br["ci_high"]),
        "positive_directions": dir_positive,
        "positive_eras": era_positive,
        **checks,
    }])


def group_mask(df: pd.DataFrame, group: str) -> pd.Series:
    if group == "ALL_CANDIDATES":
        return pd.Series(True, index=df.index)
    if group == "BOOK_FAILURE_Q4":
        return pd.to_numeric(df["book_quartile"], errors="coerce") == 4
    if group in ("PRESSURE_PLUS_FAILURE", "HIGH_PRESSURE_ABSORBED", "FAILURE_WITHOUT_HIGH_PRESSURE"):
        return df["state"].astype(str) == group
    raise KeyError(group)


def first_passage_status(row: pd.Series, target: int, stop: int, holding_sec: int) -> str:
    if not bool(row.get("decision_price_valid", False)):
        return "CENSORED"
    if not bool(row.get(f"h{holding_sec}_fully_observed", False)):
        return "CENSORED"
    fh = safe_float(row.get(f"h{holding_sec}_decision_favorable_{target}_hit_second"))
    ah = safe_float(row.get(f"h{holding_sec}_decision_adverse_{stop}_hit_second"))
    if math.isfinite(fh) and math.isfinite(ah):
        if fh < ah:
            return "WIN"
        if ah < fh:
            return "LOSS"
        return "AMBIGUOUS"
    if math.isfinite(fh):
        return "WIN"
    if math.isfinite(ah):
        return "LOSS"
    return "TIMEOUT"


def policy_metrics_from_statuses(
    statuses: pd.Series,
    target: float,
    stop: float,
    cost: float,
) -> dict[str, Any]:
    s = statuses.astype(str)
    n = int(len(s))
    wins = int((s == "WIN").sum())
    losses = int((s == "LOSS").sum())
    timeouts = int((s == "TIMEOUT").sum())
    ambiguous = int((s == "AMBIGUOUS").sum())
    censored = int((s == "CENSORED").sum())
    usable = wins + losses + timeouts
    resolved = wins + losses
    win_net = target - cost
    loss_net = -(stop + cost)
    timeout_flat = -cost
    timeout_worst = -(stop + cost)
    flat_exp = (wins * win_net + losses * loss_net + timeouts * timeout_flat) / usable if usable else math.nan
    worst_exp = (wins * win_net + losses * loss_net + timeouts * timeout_worst) / usable if usable else math.nan
    resolved_exp = (wins * win_net + losses * loss_net) / resolved if resolved else math.nan
    resolved_win_rate = wins / resolved if resolved else math.nan
    breakeven_resolved = (stop + cost) / (target + stop) if target > cost else 1.0
    rr = win_net / (stop + cost) if target > cost else math.nan
    return {
        "rows": n,
        "usable_rows": usable,
        "resolved_rows": resolved,
        "wins": wins,
        "losses": losses,
        "timeouts": timeouts,
        "ambiguous": ambiguous,
        "censored": censored,
        "target_first_rate": wins / usable if usable else math.nan,
        "stop_first_rate": losses / usable if usable else math.nan,
        "timeout_rate": timeouts / usable if usable else math.nan,
        "resolved_win_rate": resolved_win_rate,
        "resolved_breakeven_win_rate": breakeven_resolved,
        "net_reward_risk": rr,
        "resolved_only_expectancy_bps": resolved_exp,
        "flat_timeout_proxy_expectancy_bps": flat_exp,
        "worst_timeout_proxy_expectancy_bps": worst_exp,
    }


def build_policy_grid(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = ("ALL_CANDIDATES", "BOOK_FAILURE_Q4", "PRESSURE_PLUS_FAILURE", "HIGH_PRESSURE_ABSORBED", "FAILURE_WITHOUT_HIGH_PRESSURE")
    for dh in EXPECTED_HORIZONS:
        d = df[pd.to_numeric(df["observation_horizon_sec"], errors="coerce") == dh].copy()
        for group in groups:
            x = d[group_mask(d, group)].copy()
            for holding in HOLDING_HORIZONS_SEC:
                for target in TARGETS_BPS:
                    for stop in STOPS_BPS:
                        statuses = x.apply(lambda r: first_passage_status(r, target, stop, holding), axis=1)
                        for cost in COST_GRID_BPS:
                            m = policy_metrics_from_statuses(statuses, float(target), float(stop), float(cost))
                            rows.append({
                                "decision_horizon_sec": dh,
                                "group": group,
                                "holding_horizon_sec": holding,
                                "target_bps": target,
                                "stop_bps": stop,
                                "round_trip_cost_bps": cost,
                                **m,
                            })
    return pd.DataFrame(rows)


def candidate_proxy_pnl(row: pd.Series, target: int, stop: int, holding: int, cost: float, timeout_mode: str) -> float | None:
    status = first_passage_status(row, target, stop, holding)
    if status in ("CENSORED", "AMBIGUOUS"):
        return None
    if status == "WIN":
        return float(target - cost)
    if status == "LOSS":
        return float(-(stop + cost))
    if timeout_mode == "FLAT":
        return float(-cost)
    if timeout_mode == "WORST":
        return float(-(stop + cost))
    raise KeyError(timeout_mode)


def bootstrap_day_proxy_expectancy(df: pd.DataFrame, group: str, reps: int, seed: int) -> dict[str, Any]:
    d = df[
        (pd.to_numeric(df["observation_horizon_sec"], errors="coerce") == PRIMARY_DECISION_HORIZON_SEC)
        & group_mask(df, group)
    ].copy()
    d["proxy_pnl"] = d.apply(
        lambda r: candidate_proxy_pnl(
            r,
            PRIMARY_POLICY_TARGET_BPS,
            PRIMARY_POLICY_STOP_BPS,
            PRIMARY_POLICY_HORIZON_SEC,
            PRIMARY_ROUND_TRIP_COST_BPS,
            "FLAT",
        ),
        axis=1,
    )
    d = d[pd.to_numeric(d["proxy_pnl"], errors="coerce").notna()].copy()
    days = sorted(d["candidate_date"].astype(str).unique().tolist())
    if len(days) < 2 or d.empty:
        return {"group": group, "estimate": math.nan, "ci_low": math.nan, "ci_high": math.nan, "bootstrap_reps": 0}
    estimate = float(pd.to_numeric(d["proxy_pnl"], errors="coerce").mean())
    groups = {day: d[d["candidate_date"].astype(str) == day] for day in days}
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(int(reps)):
        sampled = rng.choice(days, size=len(days), replace=True)
        chunks = [groups[str(day)] for day in sampled]
        b = pd.concat(chunks, ignore_index=True)
        vals.append(float(pd.to_numeric(b["proxy_pnl"], errors="coerce").mean()))
    arr = np.asarray(vals, dtype=float)
    return {
        "group": group,
        "estimate": estimate,
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
        "bootstrap_reps": int(len(arr)),
    }


def build_primary_economic_gate(df: pd.DataFrame, policy_grid: pd.DataFrame, reps: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    key = (
        (policy_grid["decision_horizon_sec"] == PRIMARY_DECISION_HORIZON_SEC)
        & (policy_grid["holding_horizon_sec"] == PRIMARY_POLICY_HORIZON_SEC)
        & (policy_grid["target_bps"] == PRIMARY_POLICY_TARGET_BPS)
        & (policy_grid["stop_bps"] == PRIMARY_POLICY_STOP_BPS)
        & (policy_grid["round_trip_cost_bps"] == PRIMARY_ROUND_TRIP_COST_BPS)
        & (policy_grid["group"].isin(PRIMARY_ECONOMIC_GROUPS))
    )
    base = policy_grid[key].copy()
    boot_rows = [bootstrap_day_proxy_expectancy(df, g, reps, BOOTSTRAP_SEED + 100 + i) for i, g in enumerate(PRIMARY_ECONOMIC_GROUPS)]
    boot = pd.DataFrame(boot_rows)

    # Direction and chronology use the flat-timeout proxy only; this is a
    # robustness diagnostic, not an exact realized P&L backtest.
    d = df[pd.to_numeric(df["observation_horizon_sec"], errors="coerce") == PRIMARY_DECISION_HORIZON_SEC].copy()
    d["analysis_era"] = chronological_eras(d["candidate_date"], 4)
    dir_rows: list[dict[str, Any]] = []
    era_rows: list[dict[str, Any]] = []
    for group in PRIMARY_ECONOMIC_GROUPS:
        gx = d[group_mask(d, group)].copy()
        for direction, x in gx.groupby("direction_label", dropna=False):
            pnl = x.apply(lambda r: candidate_proxy_pnl(r, PRIMARY_POLICY_TARGET_BPS, PRIMARY_POLICY_STOP_BPS, PRIMARY_POLICY_HORIZON_SEC, PRIMARY_ROUND_TRIP_COST_BPS, "FLAT"), axis=1)
            vals = pd.to_numeric(pnl, errors="coerce").dropna()
            dir_rows.append({"group": group, "direction_label": str(direction), "n": int(len(vals)), "flat_timeout_proxy_expectancy_bps": float(vals.mean()) if len(vals) else math.nan})
        for era, x in gx.groupby("analysis_era", dropna=False, sort=True):
            pnl = x.apply(lambda r: candidate_proxy_pnl(r, PRIMARY_POLICY_TARGET_BPS, PRIMARY_POLICY_STOP_BPS, PRIMARY_POLICY_HORIZON_SEC, PRIMARY_ROUND_TRIP_COST_BPS, "FLAT"), axis=1)
            vals = pd.to_numeric(pnl, errors="coerce").dropna()
            era_rows.append({"group": group, "analysis_era": str(era), "n": int(len(vals)), "flat_timeout_proxy_expectancy_bps": float(vals.mean()) if len(vals) else math.nan})
    direction = pd.DataFrame(dir_rows)
    eras = pd.DataFrame(era_rows)

    gate_rows: list[dict[str, Any]] = []
    for group in PRIMARY_ECONOMIC_GROUPS:
        r = base[base["group"] == group]
        b = boot[boot["group"] == group]
        if r.empty or b.empty:
            gate_rows.append({"group": group, "status": "FAIL", "reason": "MISSING_PRIMARY_POLICY"})
            continue
        rr = r.iloc[0]
        bb = b.iloc[0]
        dp = direction[direction["group"] == group]
        ep = eras[eras["group"] == group]
        positive_dirs = int((pd.to_numeric(dp["flat_timeout_proxy_expectancy_bps"], errors="coerce") > 0).sum()) if len(dp) else 0
        positive_eras = int((pd.to_numeric(ep["flat_timeout_proxy_expectancy_bps"], errors="coerce") > 0).sum()) if len(ep) else 0
        checks = {
            "usable_n_ge_100": int(rr["usable_rows"]) >= MIN_POLICY_N,
            "flat_timeout_proxy_positive": safe_float(rr["flat_timeout_proxy_expectancy_bps"]) > 0,
            "day_cluster_ci_low_gt_0": safe_float(bb["ci_low"]) > 0,
            "both_directions_positive": positive_dirs >= 2,
            "three_of_four_eras_positive": positive_eras >= 3,
        }
        gate_rows.append({
            "group": group,
            "status": "PASS" if all(checks.values()) else "FAIL",
            "decision_horizon_sec": PRIMARY_DECISION_HORIZON_SEC,
            "holding_horizon_sec": PRIMARY_POLICY_HORIZON_SEC,
            "target_bps": PRIMARY_POLICY_TARGET_BPS,
            "stop_bps": PRIMARY_POLICY_STOP_BPS,
            "round_trip_cost_bps": PRIMARY_ROUND_TRIP_COST_BPS,
            "usable_rows": int(rr["usable_rows"]),
            "target_first_rate": safe_float(rr["target_first_rate"]),
            "stop_first_rate": safe_float(rr["stop_first_rate"]),
            "timeout_rate": safe_float(rr["timeout_rate"]),
            "net_reward_risk": safe_float(rr["net_reward_risk"]),
            "flat_timeout_proxy_expectancy_bps": safe_float(rr["flat_timeout_proxy_expectancy_bps"]),
            "worst_timeout_proxy_expectancy_bps": safe_float(rr["worst_timeout_proxy_expectancy_bps"]),
            "bootstrap_ci_low": safe_float(bb["ci_low"]),
            "bootstrap_ci_high": safe_float(bb["ci_high"]),
            "positive_directions": positive_dirs,
            "positive_eras": positive_eras,
            **checks,
        })
    return pd.DataFrame(gate_rows), boot, direction, eras


def build_invariants(src: pd.DataFrame, conditioned: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    def add(check: str, ok: bool, observed: Any, expected: Any) -> None:
        rows.append({"check": check, "status": "PASS" if bool(ok) else "FAIL", "observed": observed, "expected": expected})

    add("candidate_count_frozen", src["candidate_id"].astype(str).nunique() == EXPECTED_CANDIDATES, src["candidate_id"].astype(str).nunique(), EXPECTED_CANDIDATES)
    hs = sorted(pd.to_numeric(src["observation_horizon_sec"], errors="coerce").dropna().astype(int).unique().tolist())
    add("decision_horizons_frozen", hs == list(EXPECTED_HORIZONS), hs, list(EXPECTED_HORIZONS))
    add("one_row_per_candidate_horizon", not src.duplicated(["candidate_id", "observation_horizon_sec"]).any(), int(src.duplicated(["candidate_id", "observation_horizon_sec"]).sum()), 0)
    add("frozen_7i6_builder_present", "analysis_builder_version" in src.columns and set(src["analysis_builder_version"].dropna().astype(str).unique()) == {FROZEN_PHASE7I6_BUILDER}, sorted(src.get("analysis_builder_version", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()), [FROZEN_PHASE7I6_BUILDER])
    add("book_quartiles_present", set(pd.to_numeric(src["book_quartile"], errors="coerce").dropna().astype(int).unique()).issuperset({1,2,3,4}), sorted(pd.to_numeric(src["book_quartile"], errors="coerce").dropna().astype(int).unique().tolist()), [1,2,3,4])
    add("condition_bins_outcome_blind_complete", conditioned["momentum_quartile"].notna().mean() > 0.99 and conditioned["directional_notional_quartile"].notna().mean() > 0.99, f"momentum={conditioned['momentum_quartile'].notna().mean():.4f}; notional={conditioned['directional_notional_quartile'].notna().mean():.4f}", ">=0.99")
    needed_passages = []
    for h in HOLDING_HORIZONS_SEC:
        needed_passages.append(f"h{h}_fully_observed")
        for t in TARGETS_BPS:
            needed_passages.append(f"h{h}_decision_favorable_{t}_hit_second")
        for s in STOPS_BPS:
            needed_passages.append(f"h{h}_decision_adverse_{s}_hit_second")
    missing = [c for c in needed_passages if c not in conditioned.columns]
    add("economic_first_passage_columns_present", not missing, missing, [])
    return pd.DataFrame(rows)


def findings_markdown(summary: dict[str, Any], incremental_gate: pd.DataFrame, econ_gate: pd.DataFrame) -> str:
    ig = incremental_gate.iloc[0].to_dict() if len(incremental_gate) else {}
    lines = [
        "# Phase 7I.7 — Incremental Information & Economic Policy Gate",
        "",
        "## Status",
        "",
        f"- Builder: `{BUILDER_VERSION}`",
        f"- Engineering status: **{summary['status']}**",
        f"- Frozen Phase-7I.6 input: `{FROZEN_PHASE7I6_BUILDER}`",
        f"- Candidate count: **{summary['candidate_count']:,}**",
        f"- Primary decision horizon: **{PRIMARY_DECISION_HORIZON_SEC}s**",
        "- Raw replay: **NONE**",
        "",
        "## Incremental-information gate",
        "",
        "This gate asks whether book-failure Q4 still beats book-failure Q1 after direct standardization across the same immediate-momentum and directional-aggressive-notional quartiles.",
        "",
        f"- Gate: **{ig.get('status', 'N/A')}**",
        f"- Standardized P(75) high failure: **{safe_float(ig.get('adjusted_p75_high_failure')):.3f}**",
        f"- Standardized P(75) low failure: **{safe_float(ig.get('adjusted_p75_low_failure')):.3f}**",
        f"- Adjusted difference: **{safe_float(ig.get('adjusted_p75_difference')):.3f}**",
        f"- Day-cluster 95% CI: **[{safe_float(ig.get('bootstrap_ci_low')):.3f}, {safe_float(ig.get('bootstrap_ci_high')):.3f}]**",
        "",
        "## Economic gate",
        "",
        f"Primary policy diagnostic: **+{PRIMARY_POLICY_TARGET_BPS} / -{PRIMARY_POLICY_STOP_BPS} bps**, {PRIMARY_POLICY_HORIZON_SEC // 60}m horizon, **{PRIMARY_ROUND_TRIP_COST_BPS:.0f} bps round-trip cost**, entered at the actual {PRIMARY_DECISION_HORIZON_SEC}s decision timestamp.",
        "",
        "`flat_timeout_proxy_expectancy_bps` assumes a timeout exits at zero gross price return and still pays costs. `worst_timeout_proxy_expectancy_bps` treats every timeout as a stop. Neither is an exact timeout-close backtest because terminal timeout return was not frozen in 7I.4.",
        "",
    ]
    if len(econ_gate):
        for r in econ_gate.to_dict("records"):
            lines.extend([
                f"### {r.get('group')}",
                f"- Gate: **{r.get('status')}**",
                f"- Usable N: **{int(r.get('usable_rows', 0)):,}**",
                f"- Target first: **{safe_float(r.get('target_first_rate')):.3f}**",
                f"- Stop first: **{safe_float(r.get('stop_first_rate')):.3f}**",
                f"- Timeout: **{safe_float(r.get('timeout_rate')):.3f}**",
                f"- Flat-timeout proxy expectancy: **{safe_float(r.get('flat_timeout_proxy_expectancy_bps')):.2f} bps**",
                f"- Worst-timeout proxy expectancy: **{safe_float(r.get('worst_timeout_proxy_expectancy_bps')):.2f} bps**",
                f"- Day-cluster CI for flat-timeout proxy: **[{safe_float(r.get('bootstrap_ci_low')):.2f}, {safe_float(r.get('bootstrap_ci_high')):.2f}] bps**",
                "",
            ])
    lines.extend([
        "## Interpretation discipline",
        "",
        "A PASS on the incremental-information gate means book failure carries retrospective all-market information beyond coarse immediate momentum and directional aggressive notional. It is not a prospective forward-validation claim.",
        "",
        "A PASS on an economic proxy gate is only evidence that the frozen first-passage geometry is promising under the stated timeout proxy and cost assumption. Exact realized P&L still requires a terminal timeout exit price and explicit execution/slippage modelling.",
    ])
    return "\n".join(lines) + "\n"


def self_test() -> int:
    # Deterministic quartiles.
    s = pd.Series(range(1, 17), dtype=float)
    q = deterministic_quantile_bin(s, 4)
    assert q.tolist() == [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4]
    # First-passage ordering.
    row = pd.Series({
        "decision_price_valid": True,
        "h1800_fully_observed": True,
        "h1800_decision_favorable_40_hit_second": 120,
        "h1800_decision_adverse_15_hit_second": 140,
    })
    assert first_passage_status(row, 40, 15, 1800) == "WIN"
    m = policy_metrics_from_statuses(pd.Series(["WIN", "LOSS", "TIMEOUT", "WIN"]), 40, 15, 12)
    assert m["wins"] == 2 and m["losses"] == 1 and m["timeouts"] == 1
    # Flat-timeout arithmetic: (28 -27 -12 +28) / 4 = 4.25
    assert abs(float(m["flat_timeout_proxy_expectancy_bps"]) - 4.25) < 1e-9
    print("SELF-TEST PASS")
    print("Validated outcome-blind conditioning bins, standardized book-failure contrasts, first-passage ordering, and economic proxy arithmetic.")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    if not args.input.exists():
        raise FileNotFoundError(
            "Frozen Phase-7I.6 analysis parquet is missing:\n"
            f"  {args.input}\n"
            "Do not replay raw depth. Complete/fix Phase 7I.6 instead."
        )

    print("Building Phase 7I.7 incremental-information and economic-policy gate...")
    print(f"Builder version      : {BUILDER_VERSION}")
    print(f"Frozen Phase-7I.6    : {FROZEN_PHASE7I6_BUILDER}")
    print(f"Primary decision     : {PRIMARY_DECISION_HORIZON_SEC}s")
    print(f"Primary economic diag: +{PRIMARY_POLICY_TARGET_BPS}/-{PRIMARY_POLICY_STOP_BPS} bps, {PRIMARY_POLICY_HORIZON_SEC//60}m, cost={PRIMARY_ROUND_TRIP_COST_BPS:.0f}bps")
    print("Research role        : cheap retrospective incremental/economic gate; no replay, no ML")

    print("[1/7] Loading frozen Phase-7I.6 analysis substrate...")
    src = pd.read_parquet(args.input)
    require_columns(src, [
        "candidate_id", "candidate_date", "direction_label", "observation_horizon_sec",
        "analysis_builder_version", "book_quartile", "book_failure_score", "state",
        "primary_outcome_observed", "reach_75_30m", "reach_100_30m",
        "extra_displacement_since_35bp_candidate_bps", "directional_aggressive_quote",
        "decision_price_valid",
    ], "participant_interaction_analysis")
    print(f"      rows={len(src):,} candidates={src['candidate_id'].astype(str).nunique():,}")

    print("[2/7] Building outcome-blind momentum/notional conditioning bins...")
    conditioned, condition_baselines = add_outcome_blind_condition_bins(src)

    print("[3/7] Testing incremental book information after momentum + executed notional...")
    incremental_summary, incremental_strata, incremental_bootstrap, incremental_direction, incremental_eras = build_incremental_reports(conditioned, max(100, int(args.bootstrap_reps)))
    incremental_gate = build_incremental_gate(incremental_summary, incremental_bootstrap, incremental_direction, incremental_eras)

    print("[4/7] Evaluating transparent first-passage policy grid and cost sensitivity...")
    policy_grid = build_policy_grid(conditioned)
    economic_gate, economic_bootstrap, economic_direction, economic_eras = build_primary_economic_gate(conditioned, policy_grid, max(100, int(args.bootstrap_reps)))

    print("[5/7] Checking frozen-substrate invariants...")
    invariants = build_invariants(src, conditioned)
    for r in invariants.to_dict("records"):
        print(f"      {r['status']:4s} {r['check']}: observed={r['observed']} expected={r['expected']}")
    engineering_pass = bool(len(invariants) and (invariants["status"] == "PASS").all())
    if not engineering_pass:
        raise RuntimeError("Phase 7I.7 invariants failed; do not interpret reports.")

    print("[6/7] Writing analysis-only dataset and reports...")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)
    frozen = conditioned[[
        "candidate_id", "candidate_date", "direction_label", "observation_horizon_sec",
        "book_failure_score", "book_quartile", "state",
        "extra_displacement_since_35bp_candidate_bps", "directional_aggressive_quote",
        "momentum_quartile", "directional_notional_quartile", "momentum_x_notional_stratum",
        "primary_outcome_observed", "reach_75_30m", "reach_100_30m",
    ]].copy()
    frozen["analysis_builder_version"] = BUILDER_VERSION
    frozen["analysis_schema_version"] = SCHEMA_VERSION
    atomic_write_parquet(frozen, args.output_root / "incremental_conditioned_analysis.parquet")

    reports = {
        "incremental_condition_baselines.csv": condition_baselines,
        "incremental_book_failure_summary.csv": incremental_summary,
        "incremental_book_failure_strata.csv": incremental_strata,
        "incremental_day_cluster_bootstrap.csv": incremental_bootstrap,
        "incremental_direction_robustness.csv": incremental_direction,
        "incremental_chronological_robustness.csv": incremental_eras,
        "incremental_information_gate.csv": incremental_gate,
        "economic_policy_grid.csv": policy_grid,
        "economic_policy_primary_gate.csv": economic_gate,
        "economic_policy_day_cluster_bootstrap.csv": economic_bootstrap,
        "economic_policy_direction_robustness.csv": economic_direction,
        "economic_policy_chronological_robustness.csv": economic_eras,
        "incremental_economic_invariants.csv": invariants,
    }
    for name, frame in reports.items():
        atomic_write_csv(frame, args.report_root / name)

    summary = {
        "builder_version": BUILDER_VERSION,
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": "PASS",
        "candidate_count": int(src["candidate_id"].astype(str).nunique()),
        "rows": int(len(src)),
        "incremental_gate": incremental_gate.iloc[0].to_dict() if len(incremental_gate) else {},
        "economic_gates": economic_gate.to_dict("records"),
        "primary_decision_horizon_sec": PRIMARY_DECISION_HORIZON_SEC,
        "primary_policy": {
            "target_bps": PRIMARY_POLICY_TARGET_BPS,
            "stop_bps": PRIMARY_POLICY_STOP_BPS,
            "holding_horizon_sec": PRIMARY_POLICY_HORIZON_SEC,
            "round_trip_cost_bps": PRIMARY_ROUND_TRIP_COST_BPS,
        },
        "cost_grid_bps": list(COST_GRID_BPS),
        "pnl_limitation": "No exact timeout terminal return is frozen. Flat-timeout and worst-timeout figures are explicit proxies, not exact backtest P&L.",
    }
    atomic_write_json(args.output_root / "metadata" / "latest_summary.json", summary)
    atomic_write_text(args.report_root / "incremental_economic_findings.md", findings_markdown(summary, incremental_gate, economic_gate))

    print("[7/7] Phase 7I.7 summary")
    ig = incremental_gate.iloc[0] if len(incremental_gate) else None
    if ig is not None:
        print(f"      incremental book gate={ig['status']} adjusted P75 diff={safe_float(ig.get('adjusted_p75_difference')):.3f}")
    for r in economic_gate.to_dict("records"):
        print(f"      economic proxy {r.get('group')}: {r.get('status')} flat-timeout={safe_float(r.get('flat_timeout_proxy_expectancy_bps')):.2f}bps")
    print(f"      status=PASS reports={args.report_root}")
    print("Interpret incremental information before economic proxy results. Do not call the timeout proxy an exact backtest P&L.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
