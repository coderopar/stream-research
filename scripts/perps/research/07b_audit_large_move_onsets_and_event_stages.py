#!/usr/bin/env python3
"""
Phase 7B — Large-move onset and event-stage audit
=================================================

Purpose
-------
Audit the Phase-7A physical-event catalog before any order-book precursor
feature is inspected. Phase 7B separates two concepts that must not be
confused:

1. PHYSICAL ALIGNMENT — a retrospective timestamp used to align historical
   large moves for top-down precursor discovery.
2. DETECTABLE STAGE — a rule whose inputs are available online and which may
   later be prospectively scanned across *all* market seconds.

Phase 7B does not create a trading signal. In particular, even the
PRIOR_60S_RANGE_BREAK occurrence in this file is still selected *inside a
hindsight-known physical event leg*. It is therefore only a candidate online
rule, not prospective evidence.

Inputs
------
data/derived/perps/research/large_move_event_catalog_v1/
  physical_events.parquet
  onset_candidates.parquet
  target_membership.parquet

Outputs
-------
data/derived/perps/research/large_move_onset_stage_audit_v1/
  event_stage_audit.parquet
  onset_target_audit.parquet
  continuation_strata.parquet
  continuation_latencies.parquet
  metadata/latest_summary.json
  metadata/run_config.json

reports/perps_large_move_onset_stage_audit/
  large_move_onset_by_target.csv
  large_move_onset_by_target_horizon.csv
  large_move_event_stage_timeline.csv
  large_move_onset_ordering.csv
  large_move_continuation_strata.csv
  large_move_continuation_latencies.csv
  large_move_direction_symmetry.csv
  large_move_onset_decision_matrix.csv
  large_move_onset_stage_invariants.csv
  large_move_onset_stage_findings.md

Scientific contract
-------------------
* Phase-7A physical events and candidate onsets are frozen inputs.
* No precursor feature is inspected in Phase 7B.
* Target-conditioned onset timing is descriptive only.
* Retrospective alignment and future prospective detection are reported as
  separate roles.
* PRIOR_60S_RANGE_BREAK is "causal-input-computable" but its occurrence here
  remains hindsight event-conditioned.
* No alpha, strategy, or profitability claim is made by Phase 7B.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

BUILDER_VERSION = "1.0.0"
SCHEMA_VERSION = 1
SCHEMA_NAME = "large_move_onset_stage_audit_v1"
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_CATALOG_ROOT = Path("data/derived/perps/research/large_move_event_catalog_v1")
DEFAULT_OUTPUT_ROOT = Path("data/derived/perps/research/large_move_onset_stage_audit_v1")
DEFAULT_REPORT_ROOT = Path("reports/perps_large_move_onset_stage_audit")
DEFAULT_TARGETS_BPS = (35, 50, 75, 100)
DEFAULT_HORIZONS_SEC = (300, 600, 900, 1800)
DEFAULT_PRIMARY_HORIZON_SEC = 1800

METHOD_ORDER = (
    "EXTREMUM_HINDSIGHT",
    "VOL_SCALED_DEPARTURE",
    "FIXED_10BP_DEPARTURE",
    "PRIOR_60S_RANGE_BREAK",
)
METHOD_LABELS = {
    "EXTREMUM_HINDSIGHT": "PHYSICAL_EXTREMUM",
    "VOL_SCALED_DEPARTURE": "VOL_SCALED_DEPARTURE",
    "FIXED_10BP_DEPARTURE": "FIXED_10BP_DEPARTURE",
    "PRIOR_60S_RANGE_BREAK": "PRIOR_60S_RANGE_BREAK",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7b-", suffix=path.suffix, dir=str(path.parent))
    os.close(fd)
    try:
        Path(tmp).write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def require_numeric_stack():
    try:
        import numpy as np
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Phase 7B requires numpy and pandas. Install with:\n"
            "  python -m pip install numpy pandas"
        ) from exc
    return np, pd


def require_parquet_stack():
    np, pd = require_numeric_stack()
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Phase 7B Parquet I/O requires pyarrow. Install with:\n"
            "  python -m pip install pyarrow"
        ) from exc
    return np, pd, pa, pq


def write_parquet_atomic(df, path: Path) -> None:
    _, _, pa, pq = require_parquet_stack()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7b-", suffix=".parquet", dir=str(path.parent))
    os.close(fd)
    try:
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def q(series, p: float) -> float:
    vals = series.dropna()
    return float(vals.quantile(p)) if len(vals) else math.nan


def med(series) -> float:
    vals = series.dropna()
    return float(vals.median()) if len(vals) else math.nan


def mean(series) -> float:
    vals = series.dropna()
    return float(vals.mean()) if len(vals) else math.nan


def pct(x: bool) -> str:
    return "YES" if bool(x) else "NO"


def load_inputs(catalog_root: Path):
    _, pd, _, _ = require_parquet_stack()
    paths = {
        "events": catalog_root / "physical_events.parquet",
        "onsets": catalog_root / "onset_candidates.parquet",
        "membership": catalog_root / "target_membership.parquet",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing Phase-7A inputs:\n  " + "\n  ".join(missing))
    events = pd.read_parquet(paths["events"])
    onsets = pd.read_parquet(paths["onsets"])
    membership = pd.read_parquet(paths["membership"])
    return events, onsets, membership


def target_hit_mask(events, target: int, horizon_sec: int):
    _, pd = require_numeric_stack()
    col = f"time_to_{target}_sec"
    if col not in events.columns:
        return pd.Series(False, index=events.index)
    s = pd.to_numeric(events[col], errors="coerce")
    return s.notna() & (s >= 0) & (s <= int(horizon_sec))


def classify_continuation_strata(events, primary_horizon_sec: int):
    _, pd = require_numeric_stack()
    out = events.copy()
    h35 = target_hit_mask(out, 35, primary_horizon_sec)
    h50 = target_hit_mask(out, 50, primary_horizon_sec)
    h75 = target_hit_mask(out, 75, primary_horizon_sec)
    h100 = target_hit_mask(out, 100, primary_horizon_sec)
    labels = pd.Series("BELOW_35_WITHIN_HORIZON", index=out.index, dtype="object")
    labels.loc[h35 & ~h50] = "35_ONLY"
    labels.loc[h50 & ~h75] = "50_ONLY"
    labels.loc[h75 & ~h100] = "75_ONLY"
    labels.loc[h100] = "100_PLUS"
    out["continuation_stratum"] = labels
    return out


def build_event_stage_audit(events, onsets, primary_horizon_sec: int):
    np, pd = require_numeric_stack()
    base_cols = [
        "event_id", "symbol", "onset_date", "direction", "direction_label",
        "onset_second", "onset_price", "base_hit_second", "time_to_35_sec",
        "time_to_50_sec", "time_to_75_sec", "time_to_100_sec",
    ]
    e = events[[c for c in base_cols if c in events.columns]].copy()
    e = classify_continuation_strata(e, primary_horizon_sec)

    rows: list[dict[str, Any]] = []
    for r in e.itertuples(index=False):
        event_id = r.event_id
        target_times = {
            35: getattr(r, "time_to_35_sec", None),
            50: getattr(r, "time_to_50_sec", None),
            75: getattr(r, "time_to_75_sec", None),
            100: getattr(r, "time_to_100_sec", None),
        }
        rows.append({
            "event_id": event_id,
            "symbol": getattr(r, "symbol", DEFAULT_SYMBOL),
            "onset_date": r.onset_date,
            "direction": int(r.direction),
            "direction_label": r.direction_label,
            "continuation_stratum": r.continuation_stratum,
            "stage_name": "PHYSICAL_EXTREMUM",
            "stage_source": "EXTREMUM_HINDSIGHT",
            "stage_found": True,
            "seconds_after_physical_extremum": 0.0,
            "displacement_from_physical_extremum_bps": 0.0,
            "causal_input_computable": False,
            "hindsight_event_conditioned": True,
        })
        oe = onsets[onsets["event_id"] == event_id]
        for m in METHOD_ORDER[1:]:
            g = oe[oe["onset_method"] == m]
            if g.empty:
                continue
            x = g.iloc[0]
            found = bool(x.get("candidate_found", False))
            rows.append({
                "event_id": event_id,
                "symbol": getattr(r, "symbol", DEFAULT_SYMBOL),
                "onset_date": r.onset_date,
                "direction": int(r.direction),
                "direction_label": r.direction_label,
                "continuation_stratum": r.continuation_stratum,
                "stage_name": METHOD_LABELS[m],
                "stage_source": m,
                "stage_found": found,
                "seconds_after_physical_extremum": float(x["seconds_after_event_extremum"]) if found and pd.notna(x["seconds_after_event_extremum"]) else math.nan,
                "displacement_from_physical_extremum_bps": float(x["displacement_from_event_extremum_bps"]) if found and pd.notna(x["displacement_from_event_extremum_bps"]) else math.nan,
                "causal_input_computable": bool(x.get("causal_rule_candidate", False)),
                "hindsight_event_conditioned": True,
            })
        for target, tt in target_times.items():
            if tt is None or (isinstance(tt, float) and math.isnan(tt)):
                continue
            rows.append({
                "event_id": event_id,
                "symbol": getattr(r, "symbol", DEFAULT_SYMBOL),
                "onset_date": r.onset_date,
                "direction": int(r.direction),
                "direction_label": r.direction_label,
                "continuation_stratum": r.continuation_stratum,
                "stage_name": f"HIT_{target}BP",
                "stage_source": f"TARGET_{target}",
                "stage_found": True,
                "seconds_after_physical_extremum": float(tt),
                "displacement_from_physical_extremum_bps": float(target),
                "causal_input_computable": False,
                "hindsight_event_conditioned": True,
            })
    out = pd.DataFrame(rows)
    return out


def build_onset_target_audit(events, onsets, targets: Sequence[int], horizons: Sequence[int]):
    np, pd = require_numeric_stack()
    merged = onsets.merge(
        events[["event_id", "direction_label"] + [f"time_to_{t}_sec" for t in targets if f"time_to_{t}_sec" in events.columns]],
        on="event_id",
        how="left",
        suffixes=("", "_event"),
    )
    rows: list[dict[str, Any]] = []
    for target in targets:
        tt_col = f"time_to_{target}_sec"
        if tt_col not in merged.columns:
            continue
        tt = pd.to_numeric(merged[tt_col], errors="coerce")
        for horizon in horizons:
            cohort_mask = tt.notna() & (tt >= 0) & (tt <= int(horizon))
            cohort_ids = set(merged.loc[cohort_mask, "event_id"].tolist())
            denom = len(cohort_ids)
            if not denom:
                continue
            for method in METHOD_ORDER:
                g0 = merged[(merged["onset_method"] == method) & (merged["event_id"].isin(cohort_ids))].copy()
                found = g0[g0["candidate_found"].astype(bool)].copy()
                if len(found):
                    found["target_bps"] = float(target)
                    found["seconds_before_target_hit"] = pd.to_numeric(found[tt_col], errors="coerce") - pd.to_numeric(found["seconds_after_event_extremum"], errors="coerce")
                    found["fraction_target_realized"] = pd.to_numeric(found["displacement_from_event_extremum_bps"], errors="coerce") / float(target)
                    found["remaining_target_bps"] = float(target) - pd.to_numeric(found["displacement_from_event_extremum_bps"], errors="coerce")
                rows.append({
                    "target_bps": int(target),
                    "horizon_sec": int(horizon),
                    "onset_method": method,
                    "eligible_events": int(denom),
                    "candidate_found": int(len(found)),
                    "coverage_rate": float(len(found) / denom) if denom else math.nan,
                    "median_seconds_after_extremum": med(found["seconds_after_event_extremum"]) if len(found) else math.nan,
                    "p90_seconds_after_extremum": q(found["seconds_after_event_extremum"], 0.90) if len(found) else math.nan,
                    "median_seconds_before_target_hit": med(found["seconds_before_target_hit"]) if len(found) else math.nan,
                    "p10_seconds_before_target_hit": q(found["seconds_before_target_hit"], 0.10) if len(found) else math.nan,
                    "median_displacement_bps": med(found["displacement_from_event_extremum_bps"]) if len(found) else math.nan,
                    "median_fraction_target_realized": med(found["fraction_target_realized"]) if len(found) else math.nan,
                    "p25_fraction_target_realized": q(found["fraction_target_realized"], 0.25) if len(found) else math.nan,
                    "p75_fraction_target_realized": q(found["fraction_target_realized"], 0.75) if len(found) else math.nan,
                    "p90_fraction_target_realized": q(found["fraction_target_realized"], 0.90) if len(found) else math.nan,
                    "median_remaining_target_bps": med(found["remaining_target_bps"]) if len(found) else math.nan,
                    "p10_remaining_target_bps": q(found["remaining_target_bps"], 0.10) if len(found) else math.nan,
                    "causal_rule_inputs_available_online": bool(g0["causal_rule_candidate"].fillna(False).all()) if len(g0) else False,
                    "uses_hindsight_event_extremum": bool(g0["uses_hindsight_event_extremum"].fillna(False).any()) if len(g0) else False,
                    "event_conditioned_occurrence_is_hindsight_selected": True,
                })
    return pd.DataFrame(rows)


def build_continuation_strata_report(events, onsets, primary_horizon_sec: int):
    np, pd = require_numeric_stack()
    e = classify_continuation_strata(events, primary_horizon_sec)
    rows: list[dict[str, Any]] = []
    for stratum, g in e.groupby("continuation_stratum", sort=False):
        if stratum == "BELOW_35_WITHIN_HORIZON":
            continue
        row: dict[str, Any] = {
            "continuation_stratum": stratum,
            "events": int(len(g)),
            "up_events": int((g["direction"] > 0).sum()),
            "down_events": int((g["direction"] < 0).sum()),
            "median_time_to_35_sec": med(g["time_to_35_sec"]),
            "median_time_to_50_sec": med(g["time_to_50_sec"]),
            "median_time_to_75_sec": med(g["time_to_75_sec"]),
            "median_time_to_100_sec": med(g["time_to_100_sec"]),
        }
        ids = set(g["event_id"].tolist())
        og = onsets[onsets["event_id"].isin(ids) & onsets["candidate_found"].astype(bool)]
        for method in METHOD_ORDER:
            mg = og[og["onset_method"] == method]
            key = method.lower()
            row[f"{key}_coverage"] = float(len(mg) / len(g)) if len(g) else math.nan
            row[f"{key}_median_sec_after_extremum"] = med(mg["seconds_after_event_extremum"])
            row[f"{key}_median_displacement_bps"] = med(mg["displacement_from_event_extremum_bps"])
        rows.append(row)
    return pd.DataFrame(rows), e


def build_continuation_latencies(events, primary_horizon_sec: int):
    _, pd = require_numeric_stack()
    rows: list[dict[str, Any]] = []
    stages = [("EXTREMUM_TO_35", 0, 35), ("35_TO_50", 35, 50), ("50_TO_75", 50, 75), ("75_TO_100", 75, 100)]
    for label, lo, hi in stages:
        hi_col = f"time_to_{hi}_sec"
        if hi_col not in events.columns:
            continue
        hi_s = pd.to_numeric(events[hi_col], errors="coerce")
        mask = hi_s.notna() & (hi_s >= 0) & (hi_s <= primary_horizon_sec)
        g = events.loc[mask].copy()
        if lo == 0:
            vals = pd.to_numeric(g[hi_col], errors="coerce")
        else:
            lo_col = f"time_to_{lo}_sec"
            vals = pd.to_numeric(g[hi_col], errors="coerce") - pd.to_numeric(g[lo_col], errors="coerce")
        vals = vals.dropna()
        rows.append({
            "stage_transition": label,
            "events": int(len(vals)),
            "median_latency_sec": med(vals),
            "p25_latency_sec": q(vals, 0.25),
            "p75_latency_sec": q(vals, 0.75),
            "p90_latency_sec": q(vals, 0.90),
            "p95_latency_sec": q(vals, 0.95),
        })
    return pd.DataFrame(rows)


def build_stage_timeline(stage_df, events, targets: Sequence[int], primary_horizon_sec: int):
    np, pd = require_numeric_stack()
    rows: list[dict[str, Any]] = []
    for target in targets:
        mask = target_hit_mask(events, target, primary_horizon_sec)
        ids = set(events.loc[mask, "event_id"].tolist())
        if not ids:
            continue
        sg = stage_df[stage_df["event_id"].isin(ids) & stage_df["stage_found"].astype(bool)]
        for stage, g in sg.groupby("stage_name", sort=False):
            disp = pd.to_numeric(g["displacement_from_physical_extremum_bps"], errors="coerce")
            secs = pd.to_numeric(g["seconds_after_physical_extremum"], errors="coerce")
            rows.append({
                "target_bps": int(target),
                "primary_horizon_sec": int(primary_horizon_sec),
                "stage_name": stage,
                "events": int(len(g)),
                "coverage_rate": float(len(g) / len(ids)),
                "median_seconds_after_extremum": med(secs),
                "p25_seconds_after_extremum": q(secs, 0.25),
                "p75_seconds_after_extremum": q(secs, 0.75),
                "median_displacement_bps": med(disp),
                "median_fraction_target_realized": med(disp / float(target)),
                "p90_fraction_target_realized": q(disp / float(target), 0.90),
            })
    return pd.DataFrame(rows)


def build_onset_ordering(onsets, events, primary_horizon_sec: int):
    np, pd = require_numeric_stack()
    primary_ids = set(events.loc[target_hit_mask(events, 35, primary_horizon_sec), "event_id"].tolist())
    o = onsets[onsets["event_id"].isin(primary_ids) & onsets["candidate_found"].astype(bool)].copy()
    wide = o.pivot_table(index="event_id", columns="onset_method", values="seconds_after_event_extremum", aggfunc="first")
    rows: list[dict[str, Any]] = []
    for a in METHOD_ORDER:
        for b in METHOD_ORDER:
            if METHOD_ORDER.index(b) <= METHOD_ORDER.index(a):
                continue
            if a not in wide.columns or b not in wide.columns:
                continue
            both = wide[[a, b]].dropna()
            if not len(both):
                continue
            diff = both[b] - both[a]
            rows.append({
                "method_a": a,
                "method_b": b,
                "events_with_both": int(len(both)),
                "share_a_before_b": float((diff > 0).mean()),
                "share_same_second": float((diff == 0).mean()),
                "share_b_before_a": float((diff < 0).mean()),
                "median_b_minus_a_sec": med(diff),
                "p10_b_minus_a_sec": q(diff, 0.10),
                "p90_b_minus_a_sec": q(diff, 0.90),
            })
    return pd.DataFrame(rows)


def build_direction_symmetry(events, onsets, targets: Sequence[int], primary_horizon_sec: int):
    _, pd = require_numeric_stack()
    rows: list[dict[str, Any]] = []
    for target in targets:
        ids = set(events.loc[target_hit_mask(events, target, primary_horizon_sec), "event_id"].tolist())
        for direction_label in ("UP", "DOWN"):
            dir_ids = set(events[(events["event_id"].isin(ids)) & (events["direction_label"] == direction_label)]["event_id"].tolist())
            for method in METHOD_ORDER:
                g0 = onsets[(onsets["event_id"].isin(dir_ids)) & (onsets["onset_method"] == method)]
                g = g0[g0["candidate_found"].astype(bool)]
                rows.append({
                    "target_bps": int(target),
                    "direction_label": direction_label,
                    "onset_method": method,
                    "eligible_events": int(len(dir_ids)),
                    "coverage_rate": float(len(g) / len(dir_ids)) if len(dir_ids) else math.nan,
                    "median_seconds_after_extremum": med(g["seconds_after_event_extremum"]),
                    "median_displacement_bps": med(g["displacement_from_event_extremum_bps"]),
                    "median_fraction_target_realized": med(pd.to_numeric(g["displacement_from_event_extremum_bps"], errors="coerce") / float(target)) if len(g) else math.nan,
                })
    return pd.DataFrame(rows)


def build_decision_matrix(onset_target_audit, primary_target_bps: int, primary_horizon_sec: int):
    _, pd = require_numeric_stack()
    rows: list[dict[str, Any]] = []
    g = onset_target_audit[
        (onset_target_audit["target_bps"] == int(primary_target_bps))
        & (onset_target_audit["horizon_sec"] == int(primary_horizon_sec))
    ]
    for method in METHOD_ORDER:
        x = g[g["onset_method"] == method]
        if x.empty:
            continue
        r = x.iloc[0]
        if method == "EXTREMUM_HINDSIGHT":
            role = "RETROSPECTIVE_PHYSICAL_ALIGNMENT"
            recommendation = "USE_FOR_TOP_DOWN_ALIGNMENT"
        elif method == "PRIOR_60S_RANGE_BREAK":
            role = "PROSPECTIVE_RULE_CANDIDATE"
            recommendation = "TAKE_TO_LATER_ALL-SECONDS_PROSPECTIVE_VALIDATION"
        else:
            role = "RETROSPECTIVE_EVENT_STAGE_MARKER"
            recommendation = "USE_AS_STAGE_MARKER_NOT_SIGNAL"
        # Descriptive screen only; it does not certify a signal.
        screen = bool(
            float(r["coverage_rate"]) >= 0.90
            and float(r["median_fraction_target_realized"]) <= 0.35
            and float(r["p90_fraction_target_realized"]) <= 0.60
        ) if pd.notna(r["median_fraction_target_realized"]) and pd.notna(r["p90_fraction_target_realized"]) else False
        rows.append({
            "onset_method": method,
            "role": role,
            "recommendation": recommendation,
            "coverage_rate": float(r["coverage_rate"]),
            "median_fraction_primary_target_realized": float(r["median_fraction_target_realized"]),
            "p90_fraction_primary_target_realized": float(r["p90_fraction_target_realized"]),
            "median_remaining_primary_target_bps": float(r["median_remaining_target_bps"]),
            "causal_rule_inputs_available_online": bool(r["causal_rule_inputs_available_online"]),
            "uses_hindsight_event_extremum": bool(r["uses_hindsight_event_extremum"]),
            "event_conditioned_occurrence_is_hindsight_selected": True,
            "descriptive_early_coverage_screen": screen,
            "screen_definition": "coverage>=90%; median target realized<=35%; p90<=60%",
        })
    return pd.DataFrame(rows)


def check_invariants(events, onsets, membership, stage_df, onset_target_audit, targets: Sequence[int], horizons: Sequence[int]):
    rows: list[dict[str, Any]] = []

    def add(name: str, ok: bool, observed: Any, expected: Any) -> None:
        rows.append({"check": name, "status": "PASS" if ok else "FAIL", "observed": observed, "expected": expected})

    add("phase7a_event_ids_unique", bool(events["event_id"].is_unique), int(events["event_id"].nunique()), len(events))
    expected_onset = len(events) * len(METHOD_ORDER)
    add("four_onset_methods_per_event", len(onsets) == expected_onset, len(onsets), expected_onset)
    add("onset_methods_exact", set(onsets["onset_method"].unique()) == set(METHOD_ORDER), sorted(onsets["onset_method"].unique()), list(METHOD_ORDER))
    expected_membership = len(events) * len(set(int(x) for x in targets)) * len(set(int(x) for x in horizons))
    add("target_membership_complete", len(membership) == expected_membership, len(membership), expected_membership)
    found = onsets[onsets["candidate_found"].astype(bool)]
    add("found_candidate_nonnegative_delay", bool((found["seconds_after_event_extremum"] >= 0).all()), None, True)
    add("found_candidate_before_35_hit", bool((found["seconds_before_35bp_hit"] >= 0).all()), None, True)
    # All stage rows for event onsets and target hits must be at/after physical extremum.
    sf = stage_df[stage_df["stage_found"].astype(bool)]
    add("stage_offsets_nonnegative", bool((sf["seconds_after_physical_extremum"] >= 0).all()), None, True)
    # Onset audit should contain every method x target x horizon combination with a nonempty target cohort.
    expected_rows = 0
    for t in targets:
        for h in horizons:
            if target_hit_mask(events, int(t), int(h)).any():
                expected_rows += len(METHOD_ORDER)
    add("onset_target_audit_grid_complete", len(onset_target_audit) == expected_rows, len(onset_target_audit), expected_rows)
    # Explicit distinction: only the prior-range-break rule advertises causal inputs.
    causal_methods = set(onsets.loc[onsets["causal_rule_candidate"].fillna(False), "onset_method"].unique())
    add("only_prior_range_break_marked_causal_input_rule", causal_methods == {"PRIOR_60S_RANGE_BREAK"}, sorted(causal_methods), ["PRIOR_60S_RANGE_BREAK"])
    return rows


def build_findings(report_root: Path, args, summary: dict[str, Any], decision, strata, latencies) -> None:
    _, pd = require_numeric_stack()
    lines = [
        "# Phase 7B — Large-Move Onset and Event-Stage Audit",
        "",
        f"- Builder: **{BUILDER_VERSION}**",
        f"- Symbol: **{args.symbol}**",
        f"- Phase-7A physical events: **{summary['physical_event_count']:,}**",
        f"- Primary horizon: **{args.primary_horizon_sec // 60} minutes**",
        f"- Primary decision target: **{args.primary_target_bps} bps**",
        f"- Failed invariants: **{summary['failed_invariants']}**",
        "",
        "## Interpretation contract",
        "",
        "Phase 7B deliberately separates retrospective physical alignment from future",
        "prospective detection. EXTREMUM_HINDSIGHT may be used to align top-down",
        "historical precursor trajectories, but it is not an online signal. The",
        "PRIOR_60S_RANGE_BREAK rule uses only trailing information at its timestamp,",
        "but its occurrence in this audit is still selected inside hindsight-known",
        "large-move legs and must later be scanned across all market seconds.",
        "",
        "## Method roles",
        "",
    ]
    for r in decision.itertuples(index=False):
        lines.append(
            f"- **{r.onset_method}** — {r.role}; {r.recommendation}. "
            f"Coverage {r.coverage_rate:.1%}; median {r.median_fraction_primary_target_realized:.1%} "
            f"of the {args.primary_target_bps}-bps target realized; P90 {r.p90_fraction_primary_target_realized:.1%}."
        )
    lines += ["", "## Continuation strata", ""]
    for r in strata.itertuples(index=False):
        lines.append(f"- {r.continuation_stratum}: **{int(r.events):,}** events")
    lines += ["", "## Median continuation latencies", ""]
    for r in latencies.itertuples(index=False):
        lines.append(f"- {r.stage_transition}: **{r.median_latency_sec:.0f}s** median")
    lines += [
        "",
        "## Phase-7B conclusion",
        "",
        "This audit does not certify a prospective onset signal. Its purpose is to",
        "freeze the event-stage vocabulary that Phase 7C will use when examining",
        "pre-event participant behaviour. No alpha, strategy, or profitability claim",
        "is made by Phase 7B.",
    ]
    atomic_write_text(report_root / "large_move_onset_stage_findings.md", "\n".join(lines) + "\n")


def self_test() -> None:
    _, pd = require_numeric_stack()
    events = pd.DataFrame([
        {"event_id": "e1", "symbol": "BTCUSDT", "onset_date": "2026-01-01", "direction": 1, "direction_label": "UP", "onset_second": 100, "onset_price": 100.0, "base_hit_second": 200, "time_to_35_sec": 100, "time_to_50_sec": 160, "time_to_75_sec": 240, "time_to_100_sec": None},
        {"event_id": "e2", "symbol": "BTCUSDT", "onset_date": "2026-01-01", "direction": -1, "direction_label": "DOWN", "onset_second": 300, "onset_price": 100.0, "base_hit_second": 390, "time_to_35_sec": 90, "time_to_50_sec": None, "time_to_75_sec": None, "time_to_100_sec": None},
        {"event_id": "e3", "symbol": "BTCUSDT", "onset_date": "2026-01-02", "direction": 1, "direction_label": "UP", "onset_second": 500, "onset_price": 100.0, "base_hit_second": 580, "time_to_35_sec": 80, "time_to_50_sec": 120, "time_to_75_sec": 180, "time_to_100_sec": 260},
    ])
    onset_rows = []
    specs = {
        "EXTREMUM_HINDSIGHT": (0, 0.0, False, True),
        "VOL_SCALED_DEPARTURE": (10, 6.0, False, True),
        "FIXED_10BP_DEPARTURE": (25, 10.5, False, True),
        "PRIOR_60S_RANGE_BREAK": (30, 11.0, True, False),
    }
    for ev in events.itertuples(index=False):
        for m, (sec_after, disp, causal, hind) in specs.items():
            onset_rows.append({
                "event_id": ev.event_id,
                "onset_date": ev.onset_date,
                "direction": ev.direction,
                "direction_label": ev.direction_label,
                "onset_method": m,
                "candidate_found": True,
                "candidate_second": ev.onset_second + sec_after,
                "candidate_time": None,
                "rule_threshold_bps": disp,
                "displacement_from_event_extremum_bps": disp,
                "seconds_after_event_extremum": sec_after,
                "seconds_before_35bp_hit": ev.time_to_35_sec - sec_after,
                "fraction_35_realized": disp / 35.0,
                "fraction_50_realized": disp / 50.0,
                "fraction_75_realized": disp / 75.0,
                "fraction_100_realized": disp / 100.0,
                "causal_rule_candidate": causal,
                "uses_hindsight_event_extremum": hind,
            })
    onsets = pd.DataFrame(onset_rows)
    membership = []
    for ev in events.itertuples(index=False):
        for t in DEFAULT_TARGETS_BPS:
            tt = getattr(ev, f"time_to_{t}_sec")
            for h in DEFAULT_HORIZONS_SEC:
                membership.append({"event_id": ev.event_id, "target_bps": t, "horizon_sec": h, "target_hit": tt is not None and tt <= h})
    membership = pd.DataFrame(membership)

    stage = build_event_stage_audit(events, onsets, 1800)
    audit = build_onset_target_audit(events, onsets, DEFAULT_TARGETS_BPS, DEFAULT_HORIZONS_SEC)
    strata, classified = build_continuation_strata_report(events, onsets, 1800)
    lat = build_continuation_latencies(events, 1800)
    timeline = build_stage_timeline(stage, events, DEFAULT_TARGETS_BPS, 1800)
    ordering = build_onset_ordering(onsets, events, 1800)
    direction = build_direction_symmetry(events, onsets, DEFAULT_TARGETS_BPS, 1800)
    decision = build_decision_matrix(audit, 50, 1800)
    checks = check_invariants(events, onsets, membership, stage, audit, DEFAULT_TARGETS_BPS, DEFAULT_HORIZONS_SEC)
    assert not [r for r in checks if r["status"] == "FAIL"], checks
    assert set(classified["continuation_stratum"]) == {"35_ONLY", "75_ONLY", "100_PLUS"}
    r = audit[(audit.target_bps == 50) & (audit.horizon_sec == 1800) & (audit.onset_method == "FIXED_10BP_DEPARTURE")].iloc[0]
    assert abs(float(r.median_fraction_target_realized) - 0.21) < 1e-12
    rr = decision[decision.onset_method == "PRIOR_60S_RANGE_BREAK"].iloc[0]
    assert rr.role == "PROSPECTIVE_RULE_CANDIDATE"
    assert bool(rr.causal_rule_inputs_available_online)
    assert len(timeline) > 0 and len(ordering) > 0 and len(direction) > 0 and len(lat) == 4
    print("SELF-TEST PASS")
    print("Validated target-conditioned onset timing, continuation strata, stage latencies,")
    print("onset ordering, direction symmetry, and the separation between retrospective")
    print("physical alignment and event-conditioned causal-input-computable stage rules.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit Phase-7A large-move onset candidates and event stages")
    p.add_argument("--catalog-root", type=Path, default=DEFAULT_CATALOG_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--symbol", default=DEFAULT_SYMBOL)
    p.add_argument("--targets-bps", type=int, nargs="+", default=list(DEFAULT_TARGETS_BPS))
    p.add_argument("--horizons-sec", type=int, nargs="+", default=list(DEFAULT_HORIZONS_SEC))
    p.add_argument("--primary-target-bps", type=int, default=50)
    p.add_argument("--primary-horizon-sec", type=int, default=DEFAULT_PRIMARY_HORIZON_SEC)
    p.add_argument("--self-test", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        self_test()
        return 0

    np, pd, _, _ = require_parquet_stack()
    targets = sorted(set(int(x) for x in args.targets_bps))
    horizons = sorted(set(int(x) for x in args.horizons_sec))
    if args.primary_target_bps not in targets:
        raise SystemExit("--primary-target-bps must be present in --targets-bps")
    if args.primary_horizon_sec not in horizons:
        raise SystemExit("--primary-horizon-sec must be present in --horizons-sec")

    print("Building Phase-7B large-move onset and event-stage audit...")
    print(f"Builder version       : {BUILDER_VERSION}")
    print(f"Symbol                : {args.symbol}")
    print(f"Phase-7A catalog root : {args.catalog_root.resolve()}")
    print(f"Primary target        : {args.primary_target_bps} bps")
    print(f"Primary horizon       : {args.primary_horizon_sec // 60} minutes")
    print("Research role         : retrospective alignment + event-stage audit")
    print("No precursor feature or trading signal is evaluated in Phase 7B.")

    print("[1/6] Loading frozen Phase-7A catalog...")
    events, onsets, membership = load_inputs(args.catalog_root)
    if "symbol" in events.columns:
        events = events[events["symbol"] == args.symbol].reset_index(drop=True)
    if "symbol" in onsets.columns:
        onsets = onsets[onsets["symbol"] == args.symbol].reset_index(drop=True)
    if "symbol" in membership.columns:
        membership = membership[membership["symbol"] == args.symbol].reset_index(drop=True)
    print(f"      physical events : {len(events):,}")
    print(f"      onset rows      : {len(onsets):,}")

    print("[2/6] Building event-stage timeline and continuation strata...")
    stage_df = build_event_stage_audit(events, onsets, args.primary_horizon_sec)
    strata_report, classified_events = build_continuation_strata_report(events, onsets, args.primary_horizon_sec)
    latencies = build_continuation_latencies(events, args.primary_horizon_sec)

    print("[3/6] Auditing onset timing by target and horizon...")
    onset_target_audit = build_onset_target_audit(events, onsets, targets, horizons)
    timeline_report = build_stage_timeline(stage_df, events, targets, args.primary_horizon_sec)
    ordering = build_onset_ordering(onsets, events, args.primary_horizon_sec)
    direction = build_direction_symmetry(events, onsets, targets, args.primary_horizon_sec)
    decision = build_decision_matrix(onset_target_audit, args.primary_target_bps, args.primary_horizon_sec)

    print("[4/6] Checking onset/stage invariants...")
    checks = check_invariants(events, onsets, membership, stage_df, onset_target_audit, targets, horizons)
    failed = [x for x in checks if x["status"] == "FAIL"]

    print("[5/6] Writing audit datasets...")
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_parquet_atomic(stage_df, args.output_root / "event_stage_audit.parquet")
    write_parquet_atomic(onset_target_audit, args.output_root / "onset_target_audit.parquet")
    write_parquet_atomic(classified_events, args.output_root / "continuation_strata.parquet")
    write_parquet_atomic(latencies, args.output_root / "continuation_latencies.parquet")

    summary = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "builder_version": BUILDER_VERSION,
        "generated_at": utc_now_iso(),
        "symbol": args.symbol,
        "physical_event_count": int(len(events)),
        "onset_candidate_rows": int(len(onsets)),
        "event_stage_rows": int(len(stage_df)),
        "primary_target_bps": int(args.primary_target_bps),
        "primary_horizon_sec": int(args.primary_horizon_sec),
        "targets_bps": targets,
        "horizons_sec": horizons,
        "failed_invariants": int(len(failed)),
        "phase7a_builder_versions": sorted(str(x) for x in events.get("builder_version", pd.Series(dtype=str)).dropna().unique()),
        "physical_event_ontology": sorted(str(x) for x in events.get("physical_event_ontology", pd.Series(dtype=str)).dropna().unique()),
    }
    atomic_write_json(args.output_root / "metadata" / "latest_summary.json", summary)
    atomic_write_json(args.output_root / "metadata" / "run_config.json", {
        "builder_version": BUILDER_VERSION,
        "catalog_root": str(args.catalog_root),
        "symbol": args.symbol,
        "targets_bps": targets,
        "horizons_sec": horizons,
        "primary_target_bps": args.primary_target_bps,
        "primary_horizon_sec": args.primary_horizon_sec,
        "method_order": list(METHOD_ORDER),
        "descriptive_early_coverage_screen": "coverage>=90%; median target realized<=35%; p90<=60%",
        "scientific_note": "PRIOR_60S_RANGE_BREAK inputs are causal at timestamp, but occurrences here are hindsight event-conditioned.",
    })

    print("[6/6] Writing reports...")
    args.report_root.mkdir(parents=True, exist_ok=True)
    onset_target_audit.to_csv(args.report_root / "large_move_onset_by_target_horizon.csv", index=False)
    onset_target_audit[onset_target_audit["horizon_sec"] == args.primary_horizon_sec].to_csv(args.report_root / "large_move_onset_by_target.csv", index=False)
    timeline_report.to_csv(args.report_root / "large_move_event_stage_timeline.csv", index=False)
    ordering.to_csv(args.report_root / "large_move_onset_ordering.csv", index=False)
    strata_report.to_csv(args.report_root / "large_move_continuation_strata.csv", index=False)
    latencies.to_csv(args.report_root / "large_move_continuation_latencies.csv", index=False)
    direction.to_csv(args.report_root / "large_move_direction_symmetry.csv", index=False)
    decision.to_csv(args.report_root / "large_move_onset_decision_matrix.csv", index=False)
    pd.DataFrame(checks).to_csv(args.report_root / "large_move_onset_stage_invariants.csv", index=False)
    build_findings(args.report_root, args, summary, decision, strata_report, latencies)

    print()
    print("Phase-7B onset/stage audit summary")
    print("----------------------------------")
    print(f"Status                    : {'PASS' if not failed else 'FAIL'}")
    print(f"Physical events           : {len(events):,}")
    print(f"Event-stage rows          : {len(stage_df):,}")
    print(f"Onset target/horizon rows : {len(onset_target_audit):,}")
    print(f"Failed invariants          : {len(failed):,}")
    print(f"Findings                   : {(args.report_root / 'large_move_onset_stage_findings.md').resolve()}")
    print("Phase 7B does not certify a prospective onset signal.")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
