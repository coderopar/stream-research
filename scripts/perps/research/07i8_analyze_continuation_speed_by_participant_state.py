#!/usr/bin/env python3
"""
Phase 7I.8 — Continuation-speed decomposition by causal participant state.

Research question
-----------------
The earlier all-market work used 30 minutes as the primary realization horizon.
That mixes fast continuation with moves that only reach the same physical target
much later.  Phase 7I.8 asks whether the frozen perps participant-interaction
states are especially informative for FAST continuation.

This script performs NO raw-depth replay, NO trade replay, NO candidate
regeneration, and NO model fitting.  It reads the frozen Phase-7I.6 analysis
Parquet, whose mechanics/states are causal by their row decision timestamp and
whose future labels were physically isolated upstream before being joined into
an analysis-only table.

Primary lens
------------
- decision timestamp: +3 seconds after the causal first-35-bp candidate
- realization horizons from that actual decision timestamp: 5m / 10m / 15m / 30m
- primary target: origin reaches 75 bps
- secondary target: origin reaches 100 bps

The script reports both cumulative reach probabilities and mutually exclusive
speed buckets:
    <=5m, 5-10m, 10-15m, 15-30m, NOT_WITHIN_30M.

Interpretation discipline
-------------------------
This is an all-market causal retrospective analysis over the already-inspected
May-July 2026 research window.  It is not untouched forward validation and not
a trading-policy backtest.  The purpose is to determine whether the established
book-failure anatomy preferentially selects fast large moves or merely eventual
30-minute completions.
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
from typing import Any, Iterable

import numpy as np
import pandas as pd

BUILDER_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0.0"
SCHEMA_NAME = "causal_candidate_continuation_speed_analysis_v1"
FROZEN_PHASE7I6_BUILDER = "1.0.0"
FROZEN_PHASE7I6_SCHEMA = "causal_candidate_participant_interaction_analysis_v1"

EXPECTED_FULL_CANDIDATES = 2_729
EXPECTED_DECISION_HORIZONS = (1, 3, 5, 10)
PRIMARY_DECISION_HORIZON_SEC = 3
REALIZATION_HORIZONS_SEC = (300, 600, 900, 1800)
TARGETS_BPS = (75, 100)
BOOTSTRAP_REPS_DEFAULT = 2000
BOOTSTRAP_SEED = 20260827
MIN_DAY_GROUP_N = 3

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / FROZEN_PHASE7I6_SCHEMA
DEFAULT_INPUT_PATH = DEFAULT_INPUT_ROOT / "participant_interaction_analysis.parquet"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / SCHEMA_NAME
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "perps_causal_candidate_continuation_speed"

PRIMARY_GROUPS = (
    "ALL_CANDIDATES",
    "BOOK_FAILURE_Q4",
    "BOOK_HEALTHY_Q1",
    "AGGRESSION_Q4",
    "PRESSURE_PLUS_FAILURE",
    "HIGH_PRESSURE_ABSORBED",
    "FAILURE_WITHOUT_HIGH_PRESSURE",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_float(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


def bool_series(s: pd.Series) -> pd.Series:
    if str(s.dtype) == "boolean" or s.dtype == bool:
        return s.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(float) != 0
    return s.astype(str).str.lower().isin({"true", "1", "yes"})


def rate(s: pd.Series) -> float:
    x = s.dropna()
    return float(x.astype(bool).mean()) if len(x) else math.nan


def median_num(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(x.median()) if len(x) else math.nan


def require_columns(df: pd.DataFrame, cols: Iterable[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{label} missing required columns: {missing}")


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def atomic_write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True, default=str)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def sha256_ids(ids: Iterable[str]) -> str:
    h = hashlib.sha256()
    for x in sorted(str(v) for v in ids):
        h.update(x.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def chronology_blocks(dates: pd.Series, n: int = 4) -> dict[str, str]:
    unique = sorted(pd.Series(dates.astype(str).unique()).tolist())
    chunks = np.array_split(np.array(unique, dtype=object), n)
    out: dict[str, str] = {}
    for i, arr in enumerate(chunks, 1):
        for d in arr.tolist():
            out[str(d)] = f"ERA_{i}"
    return out


def group_mask(df: pd.DataFrame, group: str) -> pd.Series:
    if group == "ALL_CANDIDATES":
        return pd.Series(True, index=df.index)
    if group == "BOOK_FAILURE_Q4":
        return pd.to_numeric(df["book_quartile"], errors="coerce") == 4
    if group == "BOOK_HEALTHY_Q1":
        return pd.to_numeric(df["book_quartile"], errors="coerce") == 1
    if group == "AGGRESSION_Q4":
        return pd.to_numeric(df["aggression_quartile"], errors="coerce") == 4
    if group in {"PRESSURE_PLUS_FAILURE", "HIGH_PRESSURE_ABSORBED", "FAILURE_WITHOUT_HIGH_PRESSURE"}:
        return df["state"].astype(str) == group
    raise KeyError(group)


def add_speed_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for target in TARGETS_BPS:
        # Cumulative outcomes at each realization horizon, from actual decision time.
        for h in REALIZATION_HORIZONS_SEC:
            fully = bool_series(out[f"h{h}_fully_observed"]) & bool_series(out["decision_price_valid"])
            src = out[f"h{h}_origin_reach_{target}"]
            out[f"reach_{target}_{h}s"] = src.where(fully, pd.NA).astype("boolean")

        full30 = bool_series(out["h1800_fully_observed"]) & bool_series(out["decision_price_valid"])
        sec = pd.to_numeric(out[f"h1800_seconds_to_origin_{target}_from_decision"], errors="coerce")
        reached = out[f"reach_{target}_1800s"].fillna(False).astype(bool)
        bucket = pd.Series(pd.NA, index=out.index, dtype="string")
        bucket.loc[full30 & ~reached] = "NOT_WITHIN_30M"
        bucket.loc[full30 & reached & (sec <= 300)] = "LE_5M"
        bucket.loc[full30 & reached & (sec > 300) & (sec <= 600)] = "GT5_LE10M"
        bucket.loc[full30 & reached & (sec > 600) & (sec <= 900)] = "GT10_LE15M"
        bucket.loc[full30 & reached & (sec > 900) & (sec <= 1800)] = "GT15_LE30M"
        out[f"speed_{target}_bucket"] = bucket
    return out


def summarize_state_rates(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for decision_h in EXPECTED_DECISION_HORIZONS:
        d = df[pd.to_numeric(df["observation_horizon_sec"], errors="coerce") == decision_h].copy()
        for group in PRIMARY_GROUPS:
            g = d[group_mask(d, group)].copy()
            row: dict[str, Any] = {
                "decision_horizon_sec": decision_h,
                "group": group,
                "rows": int(len(g)),
                "book_valid_rows": int(bool_series(g["window_valid"]).sum()) if len(g) else 0,
                "median_extra_displacement_at_decision_bps": median_num(g["extra_displacement_since_35bp_candidate_bps"]),
            }
            for target in TARGETS_BPS:
                for h in REALIZATION_HORIZONS_SEC:
                    observed = g[g[f"reach_{target}_{h}s"].notna()]
                    row[f"p{target}_{h//60}m"] = rate(observed[f"reach_{target}_{h}s"])
                    row[f"n{target}_{h//60}m"] = int(len(observed))
                success30 = g[g[f"speed_{target}_bucket"].isin(["LE_5M", "GT5_LE10M", "GT10_LE15M", "GT15_LE30M"])]
                row[f"n{target}_success_30m"] = int(len(success30))
                if len(success30):
                    buckets = success30[f"speed_{target}_bucket"].value_counts(normalize=True)
                    row[f"share_{target}_success_le5m"] = float(buckets.get("LE_5M", 0.0))
                    row[f"share_{target}_success_le10m"] = float(buckets.get("LE_5M", 0.0) + buckets.get("GT5_LE10M", 0.0))
                    row[f"share_{target}_success_le15m"] = float(buckets.get("LE_5M", 0.0) + buckets.get("GT5_LE10M", 0.0) + buckets.get("GT10_LE15M", 0.0))
                    row[f"share_{target}_success_15to30m"] = float(buckets.get("GT15_LE30M", 0.0))
                    row[f"median_seconds_to_{target}"] = median_num(success30[f"h1800_seconds_to_origin_{target}_from_decision"])
                else:
                    for c in (f"share_{target}_success_le5m", f"share_{target}_success_le10m", f"share_{target}_success_le15m", f"share_{target}_success_15to30m", f"median_seconds_to_{target}"):
                        row[c] = math.nan
            rows.append(row)
    out = pd.DataFrame(rows)

    # Add lift versus all-candidate baseline at same decision horizon.
    bases = out[out["group"] == "ALL_CANDIDATES"].set_index("decision_horizon_sec")
    for target in TARGETS_BPS:
        for h in REALIZATION_HORIZONS_SEC:
            col = f"p{target}_{h//60}m"
            out[f"{col}_abs_lift_vs_base"] = out.apply(
                lambda r: safe_float(r[col]) - safe_float(bases.loc[int(r["decision_horizon_sec"]), col]), axis=1
            )
            out[f"{col}_rel_lift_vs_base"] = out.apply(
                lambda r: safe_float(r[col]) / safe_float(bases.loc[int(r["decision_horizon_sec"]), col])
                if safe_float(bases.loc[int(r["decision_horizon_sec"]), col]) > 0 else math.nan,
                axis=1,
            )
    return out


def build_speed_bucket_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    d = df[pd.to_numeric(df["observation_horizon_sec"], errors="coerce") == PRIMARY_DECISION_HORIZON_SEC].copy()
    order = ["LE_5M", "GT5_LE10M", "GT10_LE15M", "GT15_LE30M", "NOT_WITHIN_30M"]
    for target in TARGETS_BPS:
        for group in PRIMARY_GROUPS:
            g = d[group_mask(d, group)].copy()
            x = g[g[f"speed_{target}_bucket"].notna()]
            denom = len(x)
            for bucket in order:
                n = int((x[f"speed_{target}_bucket"] == bucket).sum())
                rows.append({
                    "target_bps": target,
                    "group": group,
                    "speed_bucket": bucket,
                    "n": n,
                    "denominator_30m_observed": int(denom),
                    "fraction_of_all_observed": float(n / denom) if denom else math.nan,
                })
    return pd.DataFrame(rows)


def state_diff(df: pd.DataFrame, group_a: str, group_b: str, target: int, h: int) -> dict[str, Any]:
    a = df[group_mask(df, group_a)]
    b = df[group_mask(df, group_b)]
    ca = f"reach_{target}_{h}s"
    aa = a[a[ca].notna()]
    bb = b[b[ca].notna()]
    pa, pb = rate(aa[ca]), rate(bb[ca])
    return {
        "group_a": group_a,
        "group_b": group_b,
        "target_bps": target,
        "realization_horizon_sec": h,
        "group_a_n": int(len(aa)),
        "group_b_n": int(len(bb)),
        "group_a_rate": pa,
        "group_b_rate": pb,
        "difference": pa - pb if math.isfinite(pa) and math.isfinite(pb) else math.nan,
    }


def build_primary_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    d = df[pd.to_numeric(df["observation_horizon_sec"], errors="coerce") == PRIMARY_DECISION_HORIZON_SEC].copy()
    pairs = (
        ("BOOK_FAILURE_Q4", "BOOK_HEALTHY_Q1"),
        ("PRESSURE_PLUS_FAILURE", "HIGH_PRESSURE_ABSORBED"),
        ("BOOK_FAILURE_Q4", "ALL_CANDIDATES"),
    )
    rows = []
    for target in TARGETS_BPS:
        for h in REALIZATION_HORIZONS_SEC:
            for ga, gb in pairs:
                rows.append(state_diff(d, ga, gb, target, h))
    return pd.DataFrame(rows)


def build_direction_robustness(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    d = df[pd.to_numeric(df["observation_horizon_sec"], errors="coerce") == PRIMARY_DECISION_HORIZON_SEC].copy()
    for direction in ("UP", "DOWN"):
        x = d[d["direction_label"].astype(str) == direction]
        for h in REALIZATION_HORIZONS_SEC:
            r = state_diff(x, "BOOK_FAILURE_Q4", "BOOK_HEALTHY_Q1", 75, h)
            r["subset"] = direction
            rows.append(r)
    return pd.DataFrame(rows)


def build_chronological_robustness(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    d = df[pd.to_numeric(df["observation_horizon_sec"], errors="coerce") == PRIMARY_DECISION_HORIZON_SEC].copy()
    cmap = chronology_blocks(d["candidate_date"], 4)
    d["chronology_era"] = d["candidate_date"].astype(str).map(cmap)
    for era, x in d.groupby("chronology_era", sort=True):
        for h in REALIZATION_HORIZONS_SEC:
            r = state_diff(x, "BOOK_FAILURE_Q4", "BOOK_HEALTHY_Q1", 75, h)
            r.update({
                "chronology_era": era,
                "start_date": min(x["candidate_date"].astype(str)),
                "end_date": max(x["candidate_date"].astype(str)),
                "candidates": int(x["candidate_id"].nunique()),
            })
            rows.append(r)
    return pd.DataFrame(rows)


def day_cluster_bootstrap(df: pd.DataFrame, target: int, h: int, reps: int, seed: int) -> dict[str, Any]:
    d = df[pd.to_numeric(df["observation_horizon_sec"], errors="coerce") == PRIMARY_DECISION_HORIZON_SEC].copy()
    col = f"reach_{target}_{h}s"
    d = d[d[col].notna()]
    days = sorted(d["candidate_date"].astype(str).unique().tolist())
    if not days:
        return {"target_bps": target, "realization_horizon_sec": h, "observed_diff": math.nan, "bootstrap_low": math.nan, "bootstrap_high": math.nan, "bootstrap_reps": 0}

    def diff(x: pd.DataFrame) -> float:
        a = x[group_mask(x, "BOOK_FAILURE_Q4")]
        b = x[group_mask(x, "BOOK_HEALTHY_Q1")]
        pa, pb = rate(a[col]), rate(b[col])
        return pa - pb if math.isfinite(pa) and math.isfinite(pb) else math.nan

    observed = diff(d)
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    day_frames = {day: d[d["candidate_date"].astype(str) == day] for day in days}
    for _ in range(max(100, int(reps))):
        sampled = rng.choice(days, size=len(days), replace=True)
        pieces = [day_frames[str(day)] for day in sampled]
        x = pd.concat(pieces, ignore_index=True)
        v = diff(x)
        if math.isfinite(v):
            vals.append(v)
    if vals:
        lo, hi = np.percentile(np.asarray(vals, dtype=float), [2.5, 97.5])
    else:
        lo = hi = math.nan
    return {
        "target_bps": target,
        "realization_horizon_sec": h,
        "observed_diff": observed,
        "bootstrap_low": float(lo),
        "bootstrap_high": float(hi),
        "bootstrap_reps": int(len(vals)),
        "days": int(len(days)),
    }


def build_invariants(src: pd.DataFrame) -> pd.DataFrame:
    rows = []
    def add(check: str, ok: bool, observed: Any, expected: Any) -> None:
        rows.append({"check": check, "status": "PASS" if ok else "FAIL", "observed": observed, "expected": expected})

    hs = sorted(pd.to_numeric(src["observation_horizon_sec"], errors="coerce").dropna().astype(int).unique().tolist())
    add("decision_horizons_frozen", hs == list(EXPECTED_DECISION_HORIZONS), hs, list(EXPECTED_DECISION_HORIZONS))
    add("one_row_per_candidate_horizon", not src.duplicated(["candidate_id", "observation_horizon_sec"]).any(), int(src.duplicated(["candidate_id", "observation_horizon_sec"]).sum()), 0)
    add("candidate_population_preserved", int(src["candidate_id"].nunique()) == EXPECTED_FULL_CANDIDATES, int(src["candidate_id"].nunique()), EXPECTED_FULL_CANDIDATES)
    primary = src[pd.to_numeric(src["observation_horizon_sec"], errors="coerce") == PRIMARY_DECISION_HORIZON_SEC]
    add("primary_one_row_per_candidate", int(primary["candidate_id"].nunique()) == EXPECTED_FULL_CANDIDATES and len(primary) == EXPECTED_FULL_CANDIDATES, len(primary), EXPECTED_FULL_CANDIDATES)

    # Horizon nesting: if a target is reached by an earlier horizon it must remain reached later.
    violations = 0
    for target in TARGETS_BPS:
        for h1, h2 in zip(REALIZATION_HORIZONS_SEC[:-1], REALIZATION_HORIZONS_SEC[1:]):
            a = src[f"reach_{target}_{h1}s"]
            b = src[f"reach_{target}_{h2}s"]
            violations += int((a.fillna(False).astype(bool) & b.notna() & ~b.fillna(False).astype(bool)).sum())
    add("continuation_horizons_nested", violations == 0, violations, 0)

    # Target nesting at every realization horizon: 100 => 75.
    target_viol = 0
    for h in REALIZATION_HORIZONS_SEC:
        p75 = src[f"reach_75_{h}s"]
        p100 = src[f"reach_100_{h}s"]
        target_viol += int((p100.fillna(False).astype(bool) & p75.notna() & ~p75.fillna(False).astype(bool)).sum())
    add("target_nesting_100_implies_75", target_viol == 0, target_viol, 0)

    # Speed bucket consistency with cumulative reach labels on primary decision horizon.
    bucket_viol = 0
    p = primary
    for target in TARGETS_BPS:
        b = p[f"speed_{target}_bucket"].astype("string")
        checks = [
            (b == "LE_5M", p[f"reach_{target}_300s"].fillna(False).astype(bool)),
            (b == "GT5_LE10M", (~p[f"reach_{target}_300s"].fillna(False).astype(bool)) & p[f"reach_{target}_600s"].fillna(False).astype(bool)),
            (b == "GT10_LE15M", (~p[f"reach_{target}_600s"].fillna(False).astype(bool)) & p[f"reach_{target}_900s"].fillna(False).astype(bool)),
            (b == "GT15_LE30M", (~p[f"reach_{target}_900s"].fillna(False).astype(bool)) & p[f"reach_{target}_1800s"].fillna(False).astype(bool)),
            (b == "NOT_WITHIN_30M", ~p[f"reach_{target}_1800s"].fillna(False).astype(bool)),
        ]
        for left, right in checks:
            bucket_viol += int((left & ~right).sum())
    add("speed_buckets_consistent", bucket_viol == 0, bucket_viol, 0)
    return pd.DataFrame(rows)


def build_findings(state_rates: pd.DataFrame, contrasts: pd.DataFrame, boot: pd.DataFrame) -> str:
    p = state_rates[state_rates["decision_horizon_sec"] == PRIMARY_DECISION_HORIZON_SEC].set_index("group")
    def g(group: str, col: str) -> float:
        return safe_float(p.loc[group, col]) if group in p.index and col in p.columns else math.nan

    lines = [
        "# Phase 7I.8 — Continuation-Speed Decomposition",
        "",
        f"- Builder: **{BUILDER_VERSION}**",
        f"- Frozen Phase-7I.6 schema: **{FROZEN_PHASE7I6_SCHEMA}**",
        f"- Primary decision timestamp: **+{PRIMARY_DECISION_HORIZON_SEC}s after causal first-35-bp detection**",
        "- Realization horizons: **5m / 10m / 15m / 30m from the actual decision timestamp**",
        "- Primary target: **75 bps from the frozen causal episode origin**",
        "- Research role: **all-market causal retrospective continuation-speed audit; no raw replay; not a trading backtest**",
        "",
        "## Primary 3-second speed curve",
        "",
    ]
    for group in ("ALL_CANDIDATES", "BOOK_FAILURE_Q4", "BOOK_HEALTHY_Q1", "PRESSURE_PLUS_FAILURE", "HIGH_PRESSURE_ABSORBED"):
        lines.append(
            f"- {group}: P75 <=5m **{g(group,'p75_5m'):.3f}**, <=10m **{g(group,'p75_10m'):.3f}**, <=15m **{g(group,'p75_15m'):.3f}**, <=30m **{g(group,'p75_30m'):.3f}**."
        )
    lines += ["", "## How to interpret", ""]
    lines.append(
        "The key question is whether book-failure lift is largest at short horizons (suggesting a fast-continuation mechanism) or only accumulates by 30 minutes (suggesting eventual continuation with substantial path noise)."
    )
    lines.append(
        "Also inspect the composition of 30-minute successes: if a high fraction of book-failure successes are already complete by 5-10 minutes, the earlier 30-minute label was hiding a faster process."
    )
    lines.append(
        "Do not equate a faster origin-target completion probability with tradable target-before-stop expectancy; Phase 7I.7 already showed those are different questions."
    )
    lines += ["", "## Day-cluster book-failure vs healthy-book differences", ""]
    for r in boot.sort_values(["target_bps", "realization_horizon_sec"]).to_dict("records"):
        lines.append(
            f"- target {int(r['target_bps'])}, {int(r['realization_horizon_sec'])//60}m: diff **{safe_float(r['observed_diff']):+.3f}**, 95% day-cluster interval **[{safe_float(r['bootstrap_low']):+.3f}, {safe_float(r['bootstrap_high']):+.3f}]**."
        )
    lines += ["", "Start with `continuation_speed_state_rates.csv`, then `continuation_speed_bucket_distribution.csv`, primary contrasts, direction/chronology, and bootstrap robustness."]
    return "\n".join(lines) + "\n"


def self_test() -> int:
    rows = []
    for cid, sec75, sec100, bookq, aggrq, state in [
        ("A", 200, 700, 4, 3, "PRESSURE_PLUS_FAILURE"),
        ("B", 500, math.nan, 1, 3, "HIGH_PRESSURE_ABSORBED"),
        ("C", 1100, math.nan, 4, 2, "FAILURE_WITHOUT_HIGH_PRESSURE"),
        ("D", math.nan, math.nan, 1, 1, "OTHER"),
    ]:
        for dh in EXPECTED_DECISION_HORIZONS:
            r = {
                "candidate_id": cid,
                "candidate_date": "2026-06-01",
                "direction_label": "UP" if cid in {"A", "C"} else "DOWN",
                "observation_horizon_sec": dh,
                "window_valid": True,
                "decision_price_valid": True,
                "extra_displacement_since_35bp_candidate_bps": 2.0,
                "book_quartile": bookq,
                "aggression_quartile": aggrq,
                "state": state,
            }
            for h in REALIZATION_HORIZONS_SEC:
                r[f"h{h}_fully_observed"] = True
                for target, sec in ((75, sec75), (100, sec100)):
                    reached = math.isfinite(sec) and sec <= h
                    r[f"h{h}_origin_reach_{target}"] = reached
                    r[f"h{h}_seconds_to_origin_{target}_from_decision"] = sec if reached else math.nan
            # h1800 seconds-to should retain exact hit time if reached within 30m.
            r["h1800_seconds_to_origin_75_from_decision"] = sec75
            r["h1800_seconds_to_origin_100_from_decision"] = sec100
            rows.append(r)
    x = add_speed_labels(pd.DataFrame(rows))
    p = x[(x["candidate_id"] == "A") & (x["observation_horizon_sec"] == 3)].iloc[0]
    assert p["speed_75_bucket"] == "LE_5M"
    assert p["speed_100_bucket"] == "GT10_LE15M"
    q = x[(x["candidate_id"] == "B") & (x["observation_horizon_sec"] == 3)].iloc[0]
    assert q["speed_75_bucket"] == "GT5_LE10M"
    d = x[(x["candidate_id"] == "D") & (x["observation_horizon_sec"] == 3)].iloc[0]
    assert d["speed_75_bucket"] == "NOT_WITHIN_30M"
    inv = build_invariants(x)
    # Population-size invariant intentionally fails in tiny self-test; all arithmetic invariants must pass.
    must = inv[~inv["check"].isin(["candidate_population_preserved", "primary_one_row_per_candidate"])]
    assert (must["status"] == "PASS").all(), must.to_dict("records")
    print("SELF-TEST PASS")
    print("Validated 5/10/15/30-minute nested continuation labels, mutually exclusive speed buckets, target nesting, and causal group summaries.")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 7I.8 continuation-speed decomposition from frozen 7I.6 analysis data.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS_DEFAULT)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    if not args.input.exists():
        raise FileNotFoundError(
            f"Frozen Phase-7I.6 analysis dataset missing: {args.input}\n"
            "Do not replay raw data. Complete/fix Phase 7I.6/7I.4 instead."
        )

    print("Building Phase 7I.8 continuation-speed decomposition...")
    print(f"Builder version      : {BUILDER_VERSION}")
    print(f"Frozen Phase-7I.6    : {FROZEN_PHASE7I6_BUILDER}")
    print(f"Primary decision     : {PRIMARY_DECISION_HORIZON_SEC}s")
    print("Realization horizons : 5m,10m,15m,30m from actual decision time")
    print("Targets              : 75,100 bps from frozen causal episode origin")
    print("Research role        : all-market causal retrospective speed audit; no raw replay")

    print("[1/7] Loading frozen Phase-7I.6 participant-interaction analysis substrate...")
    src = pd.read_parquet(args.input)
    required = [
        "candidate_id", "candidate_date", "direction_label", "observation_horizon_sec",
        "window_valid", "decision_price_valid", "extra_displacement_since_35bp_candidate_bps",
        "book_quartile", "aggression_quartile", "state",
    ]
    for h in REALIZATION_HORIZONS_SEC:
        required.append(f"h{h}_fully_observed")
        for target in TARGETS_BPS:
            required.extend([
                f"h{h}_origin_reach_{target}",
                f"h{h}_seconds_to_origin_{target}_from_decision",
            ])
    require_columns(src, required, "participant_interaction_analysis")
    print(f"      rows={len(src):,} candidates={src['candidate_id'].nunique():,}")

    print("[2/7] Building nested 5/10/15/30-minute continuation labels and exact speed buckets...")
    analysis = add_speed_labels(src)

    print("[3/7] Computing state speed curves, 30-minute success composition, and primary contrasts...")
    state_rates = summarize_state_rates(analysis)
    buckets = build_speed_bucket_distribution(analysis)
    contrasts = build_primary_contrasts(analysis)

    print("[4/7] Auditing direction, chronology, and day-cluster robustness...")
    direction = build_direction_robustness(analysis)
    chronology = build_chronological_robustness(analysis)
    boot_rows = []
    for target in TARGETS_BPS:
        for h in REALIZATION_HORIZONS_SEC:
            boot_rows.append(day_cluster_bootstrap(analysis, target, h, max(100, int(args.bootstrap_reps)), BOOTSTRAP_SEED + target + h))
    bootstrap = pd.DataFrame(boot_rows)

    print("[5/7] Checking speed-study invariants...")
    invariants = build_invariants(analysis)
    engineering_pass = bool(len(invariants) and (invariants["status"] == "PASS").all())
    for r in invariants.to_dict("records"):
        print(f"      {r['status']:4s} {r['check']}: observed={r['observed']} expected={r['expected']}")
    if not engineering_pass:
        raise RuntimeError("Phase 7I.8 engineering invariants failed; do not interpret reports.")

    print("[6/7] Writing reports and frozen cheap-analysis output...")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)
    # Analysis-only output: contains causal state plus future speed labels.
    out = analysis.copy()
    out["speed_analysis_builder_version"] = BUILDER_VERSION
    out["speed_analysis_schema_version"] = SCHEMA_VERSION
    out.to_parquet(args.output_root / "continuation_speed_analysis.parquet", index=False)
    atomic_write_csv(invariants, args.report_root / "continuation_speed_invariants.csv")
    atomic_write_csv(state_rates, args.report_root / "continuation_speed_state_rates.csv")
    atomic_write_csv(buckets, args.report_root / "continuation_speed_bucket_distribution.csv")
    atomic_write_csv(contrasts, args.report_root / "continuation_speed_primary_contrasts.csv")
    atomic_write_csv(direction, args.report_root / "continuation_speed_direction_robustness.csv")
    atomic_write_csv(chronology, args.report_root / "continuation_speed_chronological_robustness.csv")
    atomic_write_csv(bootstrap, args.report_root / "continuation_speed_day_cluster_bootstrap.csv")
    atomic_write_text(build_findings(state_rates, contrasts, bootstrap), args.report_root / "continuation_speed_findings.md")

    primary = state_rates[state_rates["decision_horizon_sec"] == PRIMARY_DECISION_HORIZON_SEC].set_index("group")
    summary = {
        "builder_version": BUILDER_VERSION,
        "schema_name": SCHEMA_NAME,
        "generated_at": utc_now_iso(),
        "input_path": str(args.input),
        "input_candidate_hash": sha256_ids(src["candidate_id"].unique()),
        "candidates": int(src["candidate_id"].nunique()),
        "primary_decision_horizon_sec": PRIMARY_DECISION_HORIZON_SEC,
        "realization_horizons_sec": list(REALIZATION_HORIZONS_SEC),
        "targets_bps": list(TARGETS_BPS),
        "all_candidates_p75": {str(h): safe_float(primary.loc["ALL_CANDIDATES", f"p75_{h//60}m"]) for h in REALIZATION_HORIZONS_SEC},
        "book_failure_q4_p75": {str(h): safe_float(primary.loc["BOOK_FAILURE_Q4", f"p75_{h//60}m"]) for h in REALIZATION_HORIZONS_SEC},
        "book_healthy_q1_p75": {str(h): safe_float(primary.loc["BOOK_HEALTHY_Q1", f"p75_{h//60}m"]) for h in REALIZATION_HORIZONS_SEC},
        "research_role": "all-market causal retrospective continuation-speed decomposition; analysis-only labels; no raw replay; not untouched forward validation",
    }
    atomic_write_json(summary, args.output_root / "latest_summary.json")

    print("[7/7] Primary 3-second continuation-speed summary")
    for group in ("ALL_CANDIDATES", "BOOK_FAILURE_Q4", "BOOK_HEALTHY_Q1", "PRESSURE_PLUS_FAILURE", "HIGH_PRESSURE_ABSORBED"):
        if group not in primary.index:
            continue
        r = primary.loc[group]
        print(
            f"      {group:28s} "
            f"P75[5m]={safe_float(r['p75_5m']):.3f} "
            f"10m={safe_float(r['p75_10m']):.3f} "
            f"15m={safe_float(r['p75_15m']):.3f} "
            f"30m={safe_float(r['p75_30m']):.3f}"
        )
    print(f"      reports={args.report_root}")
    print("      No raw replay was performed. Interpret fast-vs-late completion composition before changing holding horizons or introducing options context.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
