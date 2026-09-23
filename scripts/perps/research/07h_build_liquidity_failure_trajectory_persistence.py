#!/usr/bin/env python3
"""
Phase 7H — Liquidity-failure trajectory and persistence audit.

Scientific role
---------------
Retrospective matched-event trajectory/persistence audit.  This phase does NOT
replay raw depth or trades.  It consumes the frozen dense 10..35-bp / 3-second
exact-depth panel produced by Phase 7G and asks whether the *path* of liquidity
failure — severity, persistence, recovery and late escalation — distinguishes
75+ continuations from matched 35_ONLY physical moves better than first binary
state occurrence did.

Important interpretation discipline
-----------------------------------
The trajectory hypothesis was formulated after Phase 7G's original validation
half had already been inspected.  Therefore Phase 7H does NOT relabel that same
period as an untouched validation set.  Robustness is assessed with four
chronological eras, UP/DOWN symmetry, and the larger frozen 75+ matched sample.
A passing gate means RETROSPECTIVELY ROBUST, not prospective alpha or a fresh
out-of-sample validation.

Measurement provenance
----------------------
* Exact-depth measurements: frozen Phase-7G outputs only.
* Matched parent/control IDs: frozen Phase-7D.1 control_assignments.parquet,
  used solely as pairing metadata.  No new market measurements are read from
  Phase 7D.1.
* No Phase-6A reconstruction, canonical depth, trades, or raw market data are
  opened by this script.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

BUILDER_VERSION = "1.0.0"
SCHEMA_NAME = "large_move_liquidity_failure_trajectory_persistence_v1"
SCHEMA_VERSION = 1
FROZEN_PHASE7G_VERSION = "1.0.0"
FROZEN_PHASE7D1_VERSION = "1.0.0"
OBSERVATION_HORIZON_SEC = 3
STAGE_MIN_BPS = 10
STAGE_MAX_BPS = 35
CHECKPOINT_BPS = (20, 25, 30, 35)
LEGACY_DISCOVERY_END_DATE = date(2026, 6, 17)

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()
DEFAULT_PHASE7G_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" /
    "large_move_liquidity_state_transition_timing_v1"
)
DEFAULT_PHASE7D1_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" /
    "large_move_common_support_matched_controls_v1"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" / SCHEMA_NAME
)
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "perps_liquidity_failure_trajectory_persistence"

ANALYSIS_SPECS = (
    ("PRIMARY", "75_PLUS|THIRTYFIVE_ONLY|REGIME_MATCHED|PRIMARY"),
    ("ROBUST_75_RELAXED", "75_PLUS|THIRTYFIVE_ONLY|REGIME_MATCHED|RELAXED"),
    ("ROBUST_50_STRICT", "50_PLUS|THIRTYFIVE_ONLY|STRICT_STATE_MATCHED|RELAXED"),
    ("ROBUST_50_REGIME", "50_PLUS|THIRTYFIVE_ONLY|REGIME_MATCHED|PRIMARY"),
)
PRIMARY_ANALYSIS_ID = ANALYSIS_SPECS[0][1]
ROBUST_75_ANALYSIS_ID = ANALYSIS_SPECS[1][1]
ANALYSIS_ROLE = {aid: role for role, aid in ANALYSIS_SPECS}
EXPECTED_PARENT_COUNTS = {
    "75_PLUS|THIRTYFIVE_ONLY|REGIME_MATCHED|PRIMARY": 52,
    "75_PLUS|THIRTYFIVE_ONLY|REGIME_MATCHED|RELAXED": 122,
    "50_PLUS|THIRTYFIVE_ONLY|STRICT_STATE_MATCHED|RELAXED": 81,
    "50_PLUS|THIRTYFIVE_ONLY|REGIME_MATCHED|PRIMARY": 131,
}

# These directions are frozen from the mechanism interpretation, not selected
# by whichever sign looks favorable in Phase 7H.
CORE_TRAJECTORY_FEATURES = (
    ("positive_severity_auc_per_bp", +1),
    ("late_5bp_mean_score", +1),
    ("post_first_candidate_persistence_fraction", +1),
    ("longest_candidate_run_bps_span", +1),
    ("terminal_recovery_from_peak_score", -1),
)

CONTEXT_TRAJECTORY_FEATURES = (
    ("severity_mean", +1),
    ("terminal_score", +1),
    ("score_slope_per_10bp", +1),
    ("candidate_stage_fraction", +1),
    ("candidate_distinct_observation_fraction", +1),
    ("longest_candidate_run_observations", +1),
    ("recovery_transition_count", -1),
    ("reentry_count", +1),
    ("opposing_retained_slope_per_10bp", -1),
    ("opposing_unrecovered_slope_per_10bp", +1),
    ("retention_asymmetry_slope_per_10bp", +1),
)

ALL_TRAJECTORY_FEATURES = tuple(dict(CORE_TRAJECTORY_FEATURES + CONTEXT_TRAJECTORY_FEATURES).keys())
EXPECTED_SIGN = dict(CORE_TRAJECTORY_FEATURES + CONTEXT_TRAJECTORY_FEATURES)
COMPOSITE_FEATURE = "trajectory_composite_score"

# Fixed calendar eras.  These are robustness partitions, not untouched
# validation folds.  The boundary at 2026-06-17 preserves the old 7G split,
# while subdividing each side to make temporal instability visible.
TEMPORAL_ERAS = (
    ("ERA_1", date(2026, 5, 18), date(2026, 6, 2)),
    ("ERA_2", date(2026, 6, 3), date(2026, 6, 17)),
    ("ERA_3", date(2026, 6, 18), date(2026, 7, 2)),
    ("ERA_4", date(2026, 7, 3), date(2026, 7, 18)),
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def require_stack():
    try:
        import numpy as np
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Phase 7H requires numpy and pandas") from exc
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception:  # self-test/analysis helpers do not require parquet I/O
        pa = None
        pq = None
    return np, pd, pa, pq


def safe_float(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        return math.nan
    return x if math.isfinite(x) else math.nan


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7h-", suffix=".txt", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def write_parquet_atomic(df, path: Path) -> None:
    _, _, pa, pq = require_stack()
    if pa is None or pq is None:
        raise RuntimeError("Phase 7H parquet I/O requires pyarrow")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7h-", suffix=".parquet", dir=str(path.parent))
    os.close(fd)
    try:
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_csv_atomic(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7h-", suffix=".csv", dir=str(path.parent))
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def robust_scale(values: Sequence[float]) -> float:
    np, _, _, _ = require_stack()
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return math.nan
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med))) * 1.4826
    if math.isfinite(mad) and mad > 1e-12:
        return mad
    q25, q75 = np.quantile(a, [0.25, 0.75])
    iqr = float((q75 - q25) / 1.349)
    if math.isfinite(iqr) and iqr > 1e-12:
        return iqr
    sd = float(np.std(a, ddof=1))
    return sd if math.isfinite(sd) and sd > 1e-12 else math.nan


def robust_location_scale(values: Sequence[float]) -> tuple[float, float]:
    np, _, _, _ = require_stack()
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return math.nan, math.nan
    return float(np.median(a)), robust_scale(a)


def pooled_robust_scale(a: Sequence[float], b: Sequence[float]) -> float:
    np, _, _, _ = require_stack()
    return robust_scale(np.concatenate([np.asarray(a, dtype=float), np.asarray(b, dtype=float)]))


def linear_slope_per_10bp(x: Sequence[float], y: Sequence[float]) -> float:
    np, _, _, _ = require_stack()
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    ok = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[ok]
    yy = yy[ok]
    if len(xx) < 2 or float(np.var(xx)) <= 1e-12:
        return math.nan
    xm = float(np.mean(xx))
    ym = float(np.mean(yy))
    slope = float(np.sum((xx - xm) * (yy - ym)) / np.sum((xx - xm) ** 2))
    return slope * 10.0


def trapezoid_area(x: Sequence[float], y: Sequence[float]) -> float:
    np, _, _, _ = require_stack()
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    ok = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[ok]
    yy = yy[ok]
    if len(xx) < 2:
        return math.nan
    order = np.argsort(xx, kind="mergesort")
    xx = xx[order]
    yy = yy[order]
    dx = np.diff(xx)
    return float(np.sum(dx * (yy[:-1] + yy[1:]) * 0.5))


def era_for_date(d: date) -> str | None:
    for label, start, end in TEMPORAL_ERAS:
        if start <= d <= end:
            return label
    return None


def load_inputs(args):
    _, pd, _, pq = require_stack()
    if pq is None:
        raise RuntimeError("Phase 7H parquet input requires pyarrow")
    meta_path = args.phase7g_root / "metadata.json"
    scored_path = args.phase7g_root / "transition_scored_stage_profiles.parquet"
    stages_path = args.phase7g_root / "selected_transition_event_stages.parquet"
    assignments_path = args.phase7d1_root / "control_assignments.parquet"
    phase7d1_summary_path = args.phase7d1_root / "metadata" / "latest_summary.json"
    for path in (meta_path, scored_path, stages_path, assignments_path, phase7d1_summary_path):
        if not path.exists():
            raise FileNotFoundError(f"Required frozen input is missing: {path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if str(meta.get("builder_version")) != FROZEN_PHASE7G_VERSION:
        raise RuntimeError(
            f"Frozen Phase-7G builder mismatch: expected {FROZEN_PHASE7G_VERSION}, "
            f"found {meta.get('builder_version')}"
        )
    if str(meta.get("status")) != "PASS" or bool(meta.get("partial_scope")):
        raise RuntimeError("Phase 7H requires a full PASS Phase-7G run, not a partial/smoke output")
    if int(meta.get("observation_horizon_sec", -1)) != OBSERVATION_HORIZON_SEC:
        raise RuntimeError("Phase-7G observation horizon no longer matches frozen Phase-7H contract")
    if list(meta.get("stage_bps") or []) != list(range(STAGE_MIN_BPS, STAGE_MAX_BPS + 1)):
        raise RuntimeError("Phase-7G stage grid no longer matches frozen 10..35-bp contract")

    phase7d1_meta = json.loads(phase7d1_summary_path.read_text(encoding="utf-8"))
    if str(phase7d1_meta.get("builder_version")) != FROZEN_PHASE7D1_VERSION or str(phase7d1_meta.get("status")) != "PASS":
        raise RuntimeError(
            "Phase 7H requires frozen Phase-7D.1 v1.0.0 with PASS status for pairing metadata; "
            f"found version={phase7d1_meta.get('builder_version')} status={phase7d1_meta.get('status')}"
        )

    scored = pq.read_table(scored_path).to_pandas()
    stages = pq.read_table(stages_path).to_pandas()
    assignments = pq.read_table(assignments_path).to_pandas()
    scored["event_id"] = scored["event_id"].astype(str)
    stages["event_id"] = stages["event_id"].astype(str)
    assignments["parent_event_id"] = assignments["parent_event_id"].astype(str)
    assignments.loc[assignments["control_event_id"].notna(), "control_event_id"] = (
        assignments.loc[assignments["control_event_id"].notna(), "control_event_id"].astype(str)
    )
    assignments = assignments[
        assignments["analysis_id"].astype(str).isin([aid for _, aid in ANALYSIS_SPECS])
        & assignments["accepted_parent"].fillna(False).astype(bool)
        & (assignments["control_family"].astype(str) == "THIRTYFIVE_ONLY")
    ].copy()
    return meta, scored, stages, assignments


def collapse_distinct_observations(g):
    # Multiple 1-bp thresholds can be crossed in the same second.  For
    # persistence/recovery counts they are one live observation, not multiple
    # independent confirmations.  Keep the highest displacement in that second.
    return (
        g.sort_values(["stage_second", "stage_bps"], kind="mergesort")
        .drop_duplicates("stage_second", keep="last")
        .reset_index(drop=True)
    )


def longest_true_run(values: Sequence[bool], bps: Sequence[float]) -> tuple[int, float]:
    best_n = 0
    best_span = 0.0
    cur_start = None
    cur_n = 0
    for i, flag in enumerate(values):
        if bool(flag):
            if cur_start is None:
                cur_start = i
                cur_n = 1
            else:
                cur_n += 1
            span = float(bps[i]) - float(bps[cur_start]) if cur_n > 1 else 0.0
            if cur_n > best_n or (cur_n == best_n and span > best_span):
                best_n = cur_n
                best_span = span
        else:
            cur_start = None
            cur_n = 0
    return int(best_n), float(best_span)


def recovery_counts(candidate: Sequence[bool]) -> tuple[int, int]:
    vals = [bool(x) for x in candidate]
    recoveries = 0
    reentries = 0
    seen_candidate = False
    for prev, cur in zip(vals[:-1], vals[1:]):
        if prev and not cur:
            recoveries += 1
            seen_candidate = True
        elif (not prev) and cur and seen_candidate:
            reentries += 1
    return recoveries, reentries


def build_event_checkpoint_features(scored):
    np, pd, _, _ = require_stack()
    required = {
        "event_id", "onset_date", "direction", "direction_label", "continuation_stratum",
        "stage_bps", "stage_second", "window_valid", "state_candidate", "liquidity_failure_score",
        "opposing__retained_initial_quote_ratio",
        "opposing__durable_unrecovered_loss_share_initial",
        "liquidity_retention_asymmetry",
        "state_observable_second", "observable_directional_displacement_approx_bps",
    }
    missing = sorted(required - set(scored.columns))
    if missing:
        raise RuntimeError(f"Phase-7G scored profile schema is missing columns: {missing}")

    p = scored[
        scored["window_valid"].fillna(False).astype(bool)
        & scored["stage_bps"].between(STAGE_MIN_BPS, STAGE_MAX_BPS)
    ].copy()
    rows: list[dict[str, Any]] = []

    for event_id, eg in p.groupby("event_id", sort=False):
        eg = eg.sort_values("stage_bps", kind="mergesort").reset_index(drop=True)
        for checkpoint in CHECKPOINT_BPS:
            g = eg[eg["stage_bps"].astype(int) <= checkpoint].copy()
            terminal = eg[eg["stage_bps"].astype(int) == checkpoint]
            if g.empty or len(terminal) != 1:
                continue
            score = pd.to_numeric(g["liquidity_failure_score"], errors="coerce").to_numpy(float)
            bps = pd.to_numeric(g["stage_bps"], errors="coerce").to_numpy(float)
            finite = np.isfinite(score) & np.isfinite(bps)
            if finite.sum() < 2:
                continue
            score_f = score[finite]
            bps_f = bps[finite]
            positive = np.maximum(score_f, 0.0)
            area = trapezoid_area(bps_f, positive)
            width = float(checkpoint - STAGE_MIN_BPS)
            positive_auc_per_bp = area / width if width > 0 and math.isfinite(area) else math.nan

            late = g[g["stage_bps"].astype(int) >= checkpoint - 4]
            late_score = pd.to_numeric(late["liquidity_failure_score"], errors="coerce")
            term = terminal.iloc[0]
            terminal_score = safe_float(term["liquidity_failure_score"])
            max_score = float(np.nanmax(score_f)) if len(score_f) else math.nan
            terminal_recovery = max_score - terminal_score if math.isfinite(max_score) and math.isfinite(terminal_score) else math.nan

            d = collapse_distinct_observations(g)
            d_candidate = d["state_candidate"].fillna(False).astype(bool).to_numpy()
            d_bps = pd.to_numeric(d["stage_bps"], errors="coerce").to_numpy(float)
            longest_n, longest_span = longest_true_run(d_candidate, d_bps)
            cand_idx = np.flatnonzero(d_candidate)
            if len(cand_idx):
                first = int(cand_idx[0])
                persistence = float(np.mean(d_candidate[first:]))
            else:
                persistence = 0.0
            recoveries, reentries = recovery_counts(d_candidate)

            opp_ret = pd.to_numeric(g["opposing__retained_initial_quote_ratio"], errors="coerce").to_numpy(float)
            opp_unrec = pd.to_numeric(g["opposing__durable_unrecovered_loss_share_initial"], errors="coerce").to_numpy(float)
            asym = pd.to_numeric(g["liquidity_retention_asymmetry"], errors="coerce").to_numpy(float)

            rows.append({
                "event_id": str(event_id),
                "onset_date": str(term["onset_date"]),
                "direction": int(term["direction"]),
                "direction_label": str(term["direction_label"]),
                "continuation_stratum": str(term["continuation_stratum"]),
                "checkpoint_bps": int(checkpoint),
                "checkpoint_stage_second": int(term["stage_second"]),
                "checkpoint_observable_second": int(term["state_observable_second"]),
                "observable_directional_displacement_approx_bps": safe_float(term["observable_directional_displacement_approx_bps"]),
                "stage_points": int(len(g)),
                "distinct_anchor_observations": int(len(d)),
                "severity_mean": float(np.nanmean(score_f)),
                "positive_severity_auc_per_bp": float(positive_auc_per_bp),
                "late_5bp_mean_score": float(late_score.mean()) if late_score.notna().any() else math.nan,
                "terminal_score": terminal_score,
                "score_slope_per_10bp": linear_slope_per_10bp(bps_f, score_f),
                "candidate_stage_fraction": float(g["state_candidate"].fillna(False).astype(bool).mean()),
                "candidate_distinct_observation_fraction": float(np.mean(d_candidate)) if len(d_candidate) else math.nan,
                "post_first_candidate_persistence_fraction": persistence,
                "longest_candidate_run_observations": longest_n,
                "longest_candidate_run_bps_span": longest_span,
                "terminal_recovery_from_peak_score": float(terminal_recovery),
                "recovery_transition_count": int(recoveries),
                "reentry_count": int(reentries),
                "opposing_retained_slope_per_10bp": linear_slope_per_10bp(bps, opp_ret),
                "opposing_unrecovered_slope_per_10bp": linear_slope_per_10bp(bps, opp_unrec),
                "retention_asymmetry_slope_per_10bp": linear_slope_per_10bp(bps, asym),
            })
    return pd.DataFrame(rows)


def fit_trajectory_baselines(event_features, assignments):
    _, pd, _, _ = require_stack()
    primary = assignments[assignments["analysis_id"].astype(str) == PRIMARY_ANALYSIS_ID]
    control_ids = set(primary["control_event_id"].dropna().astype(str))
    p = event_features[
        event_features["event_id"].astype(str).isin(control_ids)
        & (pd.to_datetime(event_features["onset_date"]).dt.date <= LEGACY_DISCOVERY_END_DATE)
    ].copy()
    rows = []
    for checkpoint in CHECKPOINT_BPS:
        g = p[p["checkpoint_bps"].astype(int) == checkpoint]
        for feat, sign in CORE_TRAJECTORY_FEATURES:
            vals = pd.to_numeric(g[feat], errors="coerce").dropna().to_numpy(float)
            med, scale = robust_location_scale(vals)
            rows.append({
                "checkpoint_bps": int(checkpoint),
                "feature": feat,
                "expected_sign": int(sign),
                "control_n": int(len(vals)),
                "control_median": med,
                "control_robust_scale": scale,
                "baseline_scope": (
                    "PRIMARY unique 35_ONLY controls from frozen 7G measurements; "
                    f"onset_date <= {LEGACY_DISCOVERY_END_DATE.isoformat()}"
                ),
            })
    return pd.DataFrame(rows)


def add_trajectory_composite(event_features, baselines):
    np, pd, _, _ = require_stack()
    out = event_features.copy()
    base = baselines.set_index(["checkpoint_bps", "feature"])
    zcols = []
    for feat, sign in CORE_TRAJECTORY_FEATURES:
        zc = "signed_z__" + feat
        zcols.append(zc)
        vals = pd.to_numeric(out[feat], errors="coerce").to_numpy(float)
        cps = out["checkpoint_bps"].astype(int).to_numpy()
        z = np.full(len(out), np.nan, dtype=float)
        for i, (cp, value) in enumerate(zip(cps, vals)):
            if not math.isfinite(value) or (cp, feat) not in base.index:
                continue
            r = base.loc[(cp, feat)]
            med = safe_float(r["control_median"])
            scale = safe_float(r["control_robust_scale"])
            if math.isfinite(med) and math.isfinite(scale) and scale > 1e-12:
                z[i] = int(sign) * (value - med) / scale
        out[zc] = z
    zmat = out[zcols].to_numpy(float)
    finite = np.isfinite(zmat)
    out["trajectory_composite_valid_features"] = finite.sum(axis=1)
    out[COMPOSITE_FEATURE] = np.nanmedian(zmat, axis=1)
    out.loc[out["trajectory_composite_valid_features"] != len(CORE_TRAJECTORY_FEATURES), COMPOSITE_FEATURE] = math.nan
    return out


def build_pair_rows(assignments, event_features):
    _, pd, _, _ = require_stack()
    ef = event_features.set_index(["event_id", "checkpoint_bps"])
    features = list(ALL_TRAJECTORY_FEATURES) + [COMPOSITE_FEATURE]
    rows = []
    for role, aid in ANALYSIS_SPECS:
        a = assignments[assignments["analysis_id"].astype(str) == aid]
        for pid, pg in a.groupby("parent_event_id", sort=False):
            pid = str(pid)
            cids_all = sorted({str(x) for x in pg["control_event_id"].dropna().astype(str)})
            for checkpoint in CHECKPOINT_BPS:
                if (pid, checkpoint) not in ef.index:
                    continue
                cids = [cid for cid in cids_all if (cid, checkpoint) in ef.index]
                if len(cids) < 2:
                    continue
                er = ef.loc[(pid, checkpoint)]
                controls = ef.loc[[(cid, checkpoint) for cid in cids]]
                for feat in features:
                    ev = safe_float(er[feat])
                    co_vals = pd.to_numeric(controls[feat], errors="coerce").dropna()
                    co = float(co_vals.mean()) if len(co_vals) else math.nan
                    if not math.isfinite(ev) or not math.isfinite(co):
                        continue
                    rows.append({
                        "analysis_id": aid,
                        "analysis_role": role,
                        "checkpoint_bps": int(checkpoint),
                        "feature": feat,
                        "parent_event_id": pid,
                        "parent_event_date": str(er["onset_date"]),
                        "direction": int(er["direction"]),
                        "direction_label": str(er["direction_label"]),
                        "control_count": int(len(cids)),
                        "event_value": ev,
                        "control_value": co,
                        "paired_difference": float(ev - co),
                    })
    return pd.DataFrame(rows)


def summarize_pair_rows(pair_rows):
    np, pd, _, _ = require_stack()
    rows = []
    if pair_rows.empty:
        return pd.DataFrame()
    for (aid, role, checkpoint, feat), g in pair_rows.groupby(
        ["analysis_id", "analysis_role", "checkpoint_bps", "feature"], sort=True
    ):
        ev = pd.to_numeric(g["event_value"], errors="coerce")
        co = pd.to_numeric(g["control_value"], errors="coerce")
        ok = ev.notna() & co.notna()
        evv = ev[ok].to_numpy(float)
        cov = co[ok].to_numpy(float)
        if len(evv) == 0:
            continue
        diff = evv - cov
        scale = pooled_robust_scale(evv, cov)
        rows.append({
            "analysis_id": aid,
            "analysis_role": role,
            "checkpoint_bps": int(checkpoint),
            "feature": feat,
            "parent_pairs": int(len(evv)),
            "event_median": float(np.median(evv)),
            "control_median": float(np.median(cov)),
            "median_paired_difference": float(np.median(diff)),
            "paired_robust_standardized_difference": (
                float(np.median(diff) / scale) if math.isfinite(scale) and scale > 1e-12 else math.nan
            ),
            "paired_positive_share": float(np.mean(diff > 0)),
            "expected_sign": int(EXPECTED_SIGN.get(feat, +1)) if feat != COMPOSITE_FEATURE else +1,
        })
    return pd.DataFrame(rows)


def build_direction_robustness(pair_rows):
    _, pd, _, _ = require_stack()
    parts = []
    for direction, label in ((1, "UP"), (-1, "DOWN")):
        sub = pair_rows[pair_rows["direction"].astype(int) == direction].copy()
        s = summarize_pair_rows(sub)
        if not s.empty:
            s.insert(0, "subset_type", "DIRECTION")
            s.insert(1, "subset_value", label)
            parts.append(s)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_temporal_robustness(pair_rows):
    _, pd, _, _ = require_stack()
    work = pair_rows.copy()
    work["temporal_era"] = work["parent_event_date"].astype(str).map(
        lambda x: era_for_date(date.fromisoformat(x))
    )
    parts = []
    for era, g in work.groupby("temporal_era", dropna=True, sort=True):
        s = summarize_pair_rows(g)
        if not s.empty:
            s.insert(0, "subset_type", "TEMPORAL_ERA")
            s.insert(1, "subset_value", str(era))
            parts.append(s)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_daily_consistency(pair_rows):
    np, pd, _, _ = require_stack()
    rows = []
    for keys, g in pair_rows.groupby(["analysis_id", "checkpoint_bps", "feature"], sort=True):
        day_vals = []
        for day, dg in g.groupby("parent_event_date", sort=True):
            d = pd.to_numeric(dg["paired_difference"], errors="coerce").dropna()
            if len(d) >= 2:
                day_vals.append((str(day), float(d.median())))
        if not day_vals:
            continue
        overall = pd.to_numeric(g["paired_difference"], errors="coerce").dropna()
        med = float(overall.median()) if len(overall) else math.nan
        sign = 1 if med > 0 else (-1 if med < 0 else 0)
        vals = np.asarray([x[1] for x in day_vals], dtype=float)
        rows.append({
            "analysis_id": keys[0],
            "checkpoint_bps": int(keys[1]),
            "feature": keys[2],
            "days_comparable": int(len(day_vals)),
            "overall_parent_median_difference": med,
            "same_sign_day_share": float(np.mean(np.sign(vals) == sign)) if sign else math.nan,
            "first_day": day_vals[0][0],
            "last_day": day_vals[-1][0],
        })
    return pd.DataFrame(rows)


def build_checkpoint_gate(summary, direction, temporal):
    _, pd, _, _ = require_stack()
    rows = []
    for checkpoint in CHECKPOINT_BPS:
        prim = summary[
            (summary["analysis_id"] == PRIMARY_ANALYSIS_ID)
            & (summary["checkpoint_bps"].astype(int) == checkpoint)
        ]
        robust75 = summary[
            (summary["analysis_id"] == ROBUST_75_ANALYSIS_ID)
            & (summary["checkpoint_bps"].astype(int) == checkpoint)
        ]
        pcomp = prim[prim["feature"] == COMPOSITE_FEATURE]
        rcomp = robust75[robust75["feature"] == COMPOSITE_FEATURE]
        if len(pcomp):
            pr = pcomp.iloc[0]
            primary_n = int(pr["parent_pairs"])
            primary_diff = safe_float(pr["median_paired_difference"])
            primary_rsd = safe_float(pr["paired_robust_standardized_difference"])
        else:
            primary_n = 0; primary_diff = primary_rsd = math.nan
        if len(rcomp):
            rr = rcomp.iloc[0]
            robust_n = int(rr["parent_pairs"])
            robust_diff = safe_float(rr["median_paired_difference"])
            robust_rsd = safe_float(rr["paired_robust_standardized_difference"])
        else:
            robust_n = 0; robust_diff = robust_rsd = math.nan

        core_supported = 0
        core_rows = []
        for feat, expected in CORE_TRAJECTORY_FEATURES:
            q = prim[prim["feature"] == feat]
            if not len(q):
                continue
            r = q.iloc[0]
            diff = safe_float(r["median_paired_difference"])
            rsd = safe_float(r["paired_robust_standardized_difference"])
            ok = math.isfinite(diff) and math.isfinite(rsd) and expected * diff > 0 and abs(rsd) >= 0.25
            core_supported += int(ok)
            core_rows.append((feat, diff, rsd, ok))

        dsub = direction[
            (direction["analysis_id"] == PRIMARY_ANALYSIS_ID)
            & (direction["checkpoint_bps"].astype(int) == checkpoint)
            & (direction["feature"] == COMPOSITE_FEATURE)
        ]
        direction_ok = True
        direction_details = {}
        for label in ("UP", "DOWN"):
            q = dsub[dsub["subset_value"] == label]
            if not len(q):
                direction_ok = False
                direction_details[label] = (0, math.nan)
                continue
            r = q.iloc[0]
            n = int(r["parent_pairs"])
            diff = safe_float(r["median_paired_difference"])
            direction_details[label] = (n, diff)
            if n < 8 or not math.isfinite(diff) or diff <= 0:
                direction_ok = False

        tsub = temporal[
            (temporal["analysis_id"] == PRIMARY_ANALYSIS_ID)
            & (temporal["checkpoint_bps"].astype(int) == checkpoint)
            & (temporal["feature"] == COMPOSITE_FEATURE)
        ]
        eligible_eras = 0
        positive_eras = 0
        era_details = {}
        for era in [x[0] for x in TEMPORAL_ERAS]:
            q = tsub[tsub["subset_value"] == era]
            if not len(q):
                era_details[era] = (0, math.nan)
                continue
            r = q.iloc[0]
            n = int(r["parent_pairs"])
            diff = safe_float(r["median_paired_difference"])
            era_details[era] = (n, diff)
            if n >= 5:
                eligible_eras += 1
                positive_eras += int(math.isfinite(diff) and diff > 0)
        temporal_ok = eligible_eras >= 3 and positive_eras / eligible_eras >= 0.75

        primary_ok = primary_n >= 30 and math.isfinite(primary_diff) and primary_diff > 0 and math.isfinite(primary_rsd) and primary_rsd >= 0.30
        robust75_ok = robust_n >= 60 and math.isfinite(robust_diff) and robust_diff > 0 and math.isfinite(robust_rsd) and robust_rsd >= 0.25
        core_ok = core_supported >= 3
        passed = bool(primary_ok and robust75_ok and core_ok and direction_ok and temporal_ok)

        rows.append({
            "gate": "RETROSPECTIVE_TRAJECTORY_ROBUSTNESS",
            "checkpoint_bps": int(checkpoint),
            "passed": passed,
            "primary_parent_pairs": primary_n,
            "primary_composite_median_difference": primary_diff,
            "primary_composite_rsd": primary_rsd,
            "robust75_parent_pairs": robust_n,
            "robust75_composite_median_difference": robust_diff,
            "robust75_composite_rsd": robust_rsd,
            "core_features_supported": int(core_supported),
            "core_features_required": 3,
            "up_parent_pairs": direction_details.get("UP", (0, math.nan))[0],
            "up_composite_median_difference": direction_details.get("UP", (0, math.nan))[1],
            "down_parent_pairs": direction_details.get("DOWN", (0, math.nan))[0],
            "down_composite_median_difference": direction_details.get("DOWN", (0, math.nan))[1],
            "eligible_temporal_eras": int(eligible_eras),
            "positive_temporal_eras": int(positive_eras),
            "temporal_positive_share": float(positive_eras / eligible_eras) if eligible_eras else math.nan,
            "primary_ok": primary_ok,
            "robust75_ok": robust75_ok,
            "core_ok": core_ok,
            "direction_ok": direction_ok,
            "temporal_ok": temporal_ok,
            "rule": (
                "primary composite n>=30, diff>0, RSD>=0.30; larger 75+ composite n>=60, diff>0, RSD>=0.25; "
                ">=3/5 core trajectory features expected-direction with |RSD|>=0.25; UP/DOWN n>=8 and composite diff>0; "
                ">=3 eras with n>=5 and >=75% positive composite differences"
            ),
            "interpretation": (
                "Retrospective robustness only. The trajectory hypothesis was formulated after inspecting Phase-7G validation, "
                "so this is not an untouched out-of-sample validation or prospective alpha gate."
            ),
        })
    return pd.DataFrame(rows)


def build_economic_timing(event_features, stages, assignments):
    np, pd, _, _ = require_stack()
    primary_parent_ids = set(
        assignments[assignments["analysis_id"].astype(str) == PRIMARY_ANALYSIS_ID]["parent_event_id"].astype(str)
    )
    st = stages.set_index("event_id")
    rows = []
    e = event_features[event_features["event_id"].astype(str).isin(primary_parent_ids)].copy()
    for checkpoint in CHECKPOINT_BPS:
        g = e[e["checkpoint_bps"].astype(int) == checkpoint].copy()
        seconds75 = []
        bps75 = []
        seconds100 = []
        bps100 = []
        before75 = []
        for r in g.to_dict("records"):
            eid = str(r["event_id"])
            if eid not in st.index:
                continue
            sr = st.loc[eid]
            obs = int(r["checkpoint_observable_second"])
            disp = safe_float(r["observable_directional_displacement_approx_bps"])
            h75 = safe_float(sr.get("hit_75_second"))
            h100 = safe_float(sr.get("hit_100_second"))
            if math.isfinite(h75):
                seconds75.append(h75 - obs)
                before75.append(obs < h75)
                if math.isfinite(disp):
                    bps75.append(75.0 - disp)
            if math.isfinite(h100):
                seconds100.append(h100 - obs)
                if math.isfinite(disp):
                    bps100.append(100.0 - disp)
        rows.append({
            "checkpoint_bps": int(checkpoint),
            "eligible_primary_75plus_parents": int(len(g)),
            "median_observable_displacement_approx_bps": float(pd.to_numeric(g["observable_directional_displacement_approx_bps"], errors="coerce").median()) if len(g) else math.nan,
            "median_seconds_remaining_to_75bp": float(np.median(seconds75)) if seconds75 else math.nan,
            "median_approx_bps_remaining_to_75": float(np.median(bps75)) if bps75 else math.nan,
            "share_observable_before_75bp": float(np.mean(before75)) if before75 else math.nan,
            "100plus_parents_with_hit": int(len(seconds100)),
            "median_seconds_remaining_to_100bp": float(np.median(seconds100)) if seconds100 else math.nan,
            "median_approx_bps_remaining_to_100": float(np.median(bps100)) if bps100 else math.nan,
        })
    return pd.DataFrame(rows)


def check_invariants(meta, scored, stages, assignments, event_features, baselines, pair_rows, gate):
    _, pd, _, _ = require_stack()
    rows = []

    def add(name: str, passed: bool, observed: Any, expected: Any):
        rows.append({"check": name, "passed": bool(passed), "observed": observed, "expected": expected})

    add("frozen_phase7g_version", str(meta.get("builder_version")) == FROZEN_PHASE7G_VERSION, meta.get("builder_version"), FROZEN_PHASE7G_VERSION)
    add("frozen_phase7g_full_pass", str(meta.get("status")) == "PASS" and not bool(meta.get("partial_scope")), f"status={meta.get('status')} partial={meta.get('partial_scope')}", "PASS full scope")
    add("frozen_phase7g_selected_events", int(meta.get("selected_events", -1)) == 565, meta.get("selected_events"), 565)
    add("frozen_phase7g_profile_rows", int(meta.get("profile_rows", -1)) == 14690 and int(meta.get("valid_profile_rows", -1)) == 14690, f"rows={meta.get('profile_rows')} valid={meta.get('valid_profile_rows')}", "14690 / 14690")
    dup_profiles = int(scored.duplicated(["event_id", "stage_bps"]).sum())
    add("scored_stage_profiles_unique", dup_profiles == 0, dup_profiles, 0)
    stage_counts = scored.groupby("event_id")["stage_bps"].nunique()
    add("dense_10_35_stage_grid_complete", len(stage_counts) == 565 and int(stage_counts.min()) == 26 and int(stage_counts.max()) == 26, f"events={len(stage_counts)} min={int(stage_counts.min()) if len(stage_counts) else 0} max={int(stage_counts.max()) if len(stage_counts) else 0}", "565 events, exactly 26 stages")
    add("event_stage_rows_unique", not stages["event_id"].duplicated().any() and len(stages) == 565, f"rows={len(stages)} dup={int(stages['event_id'].duplicated().sum())}", "565 unique")

    observed_counts = assignments.groupby("analysis_id")["parent_event_id"].nunique().to_dict()
    for aid, expected in EXPECTED_PARENT_COUNTS.items():
        add(f"frozen_pairing_parent_count__{ANALYSIS_ROLE[aid]}", int(observed_counts.get(aid, 0)) == expected, int(observed_counts.get(aid, 0)), expected)

    dup_ef = int(event_features.duplicated(["event_id", "checkpoint_bps"]).sum())
    add("event_checkpoint_features_unique", dup_ef == 0, dup_ef, 0)
    expected_ef = 565 * len(CHECKPOINT_BPS)
    add("event_checkpoint_features_complete", len(event_features) == expected_ef, len(event_features), expected_ef)
    min_baseline = int(baselines["control_n"].min()) if len(baselines) else 0
    bad_scale = int((pd.to_numeric(baselines["control_robust_scale"], errors="coerce") <= 0).sum()) if len(baselines) else 1
    add("trajectory_baselines_sufficient", len(baselines) == len(CHECKPOINT_BPS) * len(CORE_TRAJECTORY_FEATURES) and min_baseline >= 20 and bad_scale == 0, f"rows={len(baselines)} min_n={min_baseline} bad_scale={bad_scale}", f"{len(CHECKPOINT_BPS) * len(CORE_TRAJECTORY_FEATURES)} rows, min_n>=20, positive scales")
    bad_composite = int((event_features["trajectory_composite_valid_features"] != len(CORE_TRAJECTORY_FEATURES)).sum())
    add("trajectory_composite_uses_all_core_features", bad_composite == 0, bad_composite, 0)

    pair_counts = pair_rows[pair_rows["feature"] == COMPOSITE_FEATURE].groupby(["analysis_id", "checkpoint_bps"])["parent_event_id"].nunique().to_dict()
    for aid, expected in EXPECTED_PARENT_COUNTS.items():
        for cp in CHECKPOINT_BPS:
            observed = int(pair_counts.get((aid, cp), 0))
            add(f"pairable_parents__{ANALYSIS_ROLE[aid]}__{cp}bp", observed == expected, observed, expected)
    add("checkpoint_gate_rows_present", len(gate) == len(CHECKPOINT_BPS), len(gate), len(CHECKPOINT_BPS))
    return pd.DataFrame(rows)


def build_findings(meta: Mapping[str, Any], gate, timing) -> str:
    _, pd, _, _ = require_stack()
    lines = [
        "# Phase 7H — Liquidity-Failure Trajectory & Persistence Audit",
        "",
        f"- Builder: **{BUILDER_VERSION}**",
        f"- Frozen Phase 7G: **{meta.get('builder_version')} / {meta.get('status')}**",
        "- Raw depth/trade replay: **NONE**",
        "- Scientific role: **retrospective trajectory/persistence robustness; not prospective alpha**",
        "",
        "## Interpretation discipline",
        "",
        "Phase 7H was conceived after Phase 7G's former validation half had already been inspected. "
        "Therefore the four chronological eras below are robustness checks, not a newly untouched holdout.",
        "",
        "## Retrospective robustness gate",
        "",
    ]
    if gate.empty:
        lines.append("No gate rows were generated.")
    else:
        for r in gate.sort_values("checkpoint_bps").to_dict("records"):
            lines.append(
                f"- **{int(r['checkpoint_bps'])} bp:** {'PASS' if bool(r['passed']) else 'FAIL'} — "
                f"primary composite RSD={safe_float(r['primary_composite_rsd']):+.3f}; "
                f"robust-75 RSD={safe_float(r['robust75_composite_rsd']):+.3f}; "
                f"core supported={int(r['core_features_supported'])}/5; "
                f"positive eras={int(r['positive_temporal_eras'])}/{int(r['eligible_temporal_eras'])}."
            )
        passed = gate[gate["passed"].astype(bool)]
        if len(passed):
            earliest = int(passed["checkpoint_bps"].min())
            lines += ["", f"Earliest retrospectively robust trajectory checkpoint: **{earliest} bp**."]
        else:
            lines += ["", "No checkpoint passed the full retrospective trajectory-robustness gate."]
    lines += ["", "## Economic timing", ""]
    for r in timing.sort_values("checkpoint_bps").to_dict("records"):
        lines.append(
            f"- **{int(r['checkpoint_bps'])} bp checkpoint:** median observable displacement "
            f"{safe_float(r['median_observable_displacement_approx_bps']):.1f} bp; median remaining to 75 "
            f"{safe_float(r['median_approx_bps_remaining_to_75']):.1f} bp / "
            f"{safe_float(r['median_seconds_remaining_to_75bp']):.1f}s."
        )
    lines += [
        "",
        "A PASS here does not establish a live trading signal.  The next prospective/all-market phase must include failed causal candidates.",
        "",
    ]
    return "\n".join(lines)


def self_test() -> None:
    np, pd, _, _ = require_stack()
    rows = []
    # Synthetic event with increasing stress.  11 and 12 bp share an anchor
    # second so they must not count as two persistence observations.
    for bps in range(10, 21):
        sec = 100 + (bps - 10)
        if bps == 12:
            sec = 101
        score = (bps - 10) / 10.0
        rows.append({
            "event_id": "E1", "onset_date": "2026-05-20", "direction": 1, "direction_label": "UP",
            "continuation_stratum": "75_ONLY", "stage_bps": bps, "stage_second": sec,
            "window_valid": True, "state_candidate": bps >= 15, "liquidity_failure_score": score,
            "opposing__retained_initial_quote_ratio": 1.0 - 0.02 * (bps - 10),
            "opposing__durable_unrecovered_loss_share_initial": 0.02 * (bps - 10),
            "liquidity_retention_asymmetry": 0.01 * (bps - 10),
            "state_observable_second": sec + 3,
            "observable_directional_displacement_approx_bps": float(bps) + 1.0,
        })
    df = pd.DataFrame(rows)
    out = build_event_checkpoint_features(df)
    r = out[out["checkpoint_bps"] == 20].iloc[0]
    assert int(r["distinct_anchor_observations"]) == 10, r["distinct_anchor_observations"]
    assert safe_float(r["positive_severity_auc_per_bp"]) > 0.45
    assert safe_float(r["score_slope_per_10bp"]) > 0.9
    assert safe_float(r["post_first_candidate_persistence_fraction"]) == 1.0
    assert safe_float(r["terminal_recovery_from_peak_score"]) == 0.0
    assert safe_float(r["opposing_retained_slope_per_10bp"]) < 0
    print("SELF-TEST PASS")
    print("Validated no-replay trajectory summaries, same-second persistence collapse, recovery arithmetic and expected trajectory directions.")


def parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(description="Phase 7H liquidity-failure trajectory/persistence audit from frozen Phase-7G outputs")
    p.add_argument("--phase7g-root", type=Path, default=DEFAULT_PHASE7G_ROOT)
    p.add_argument("--phase7d1-root", type=Path, default=DEFAULT_PHASE7D1_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    np, pd, pa, pq = require_stack()

    print("Building Phase 7H liquidity-failure trajectory/persistence audit...")
    print(f"Builder version      : {BUILDER_VERSION}")
    print(f"Frozen Phase-7G      : {FROZEN_PHASE7G_VERSION}")
    print(f"Frozen Phase-7D.1    : {FROZEN_PHASE7D1_VERSION} pairing metadata only")
    print(f"Physical panel       : 10..35 bps, frozen 3s exact-depth mechanics")
    print(f"Trajectory checkpoints: {','.join(map(str, CHECKPOINT_BPS))} bps")
    print(f"Primary analysis     : {PRIMARY_ANALYSIS_ID}")
    print("Raw replay           : NONE")
    print("Research role        : retrospective trajectory/persistence robustness; not fresh validation or prospective alpha")

    print("[1/8] Loading frozen Phase-7G measurements and frozen pairing metadata...")
    meta, scored, stages, assignments = load_inputs(args)
    print(f"      Phase-7G selected events={int(meta['selected_events']):,} profiles={int(meta['profile_rows']):,} valid={int(meta['valid_profile_rows']):,}")
    for role, aid in ANALYSIS_SPECS:
        n = int(assignments[assignments["analysis_id"] == aid]["parent_event_id"].nunique())
        c = int(assignments[assignments["analysis_id"] == aid]["control_event_id"].nunique())
        print(f"      {role:<18} parents={n:3d} unique_controls={c:3d}")

    print("[2/8] Deriving event-level trajectory, persistence and recovery summaries...")
    event_features = build_event_checkpoint_features(scored)
    print(f"      event-checkpoint rows={len(event_features):,} ({event_features['event_id'].nunique():,} events x {event_features['checkpoint_bps'].nunique()} checkpoints)")

    print("[3/8] Freezing trajectory-feature baselines and building continuous composite...")
    baselines = fit_trajectory_baselines(event_features, assignments)
    event_features = add_trajectory_composite(event_features, baselines)
    print(f"      baseline rows={len(baselines):,} min control n={int(baselines['control_n'].min()) if len(baselines) else 0}")

    print("[4/8] Building matched parent/control trajectory contrasts...")
    pair_rows = build_pair_rows(assignments, event_features)
    summary = summarize_pair_rows(pair_rows)
    direction = build_direction_robustness(pair_rows)
    temporal = build_temporal_robustness(pair_rows)
    daily = build_daily_consistency(pair_rows)
    print(f"      pair rows={len(pair_rows):,} summary rows={len(summary):,}")

    print("[5/8] Evaluating predeclared retrospective robustness gate and economic timing...")
    gate = build_checkpoint_gate(summary, direction, temporal)
    timing = build_economic_timing(event_features, stages, assignments)
    for r in gate.sort_values("checkpoint_bps").to_dict("records"):
        print(
            f"      {int(r['checkpoint_bps']):2d}bp {'PASS' if bool(r['passed']) else 'FAIL'} | "
            f"primary RSD={safe_float(r['primary_composite_rsd']):+.3f} "
            f"robust75 RSD={safe_float(r['robust75_composite_rsd']):+.3f} "
            f"core={int(r['core_features_supported'])}/5 eras={int(r['positive_temporal_eras'])}/{int(r['eligible_temporal_eras'])}"
        )

    print("[6/8] Checking trajectory-study invariants...")
    checks = check_invariants(meta, scored, stages, assignments, event_features, baselines, pair_rows, gate)
    for r in checks.to_dict("records"):
        print(f"      {'PASS' if bool(r['passed']) else 'FAIL'} {r['check']}: observed={r['observed']} expected={r['expected']}")
    failed = checks[~checks["passed"].astype(bool)]
    status = "PASS" if failed.empty else "FAIL"

    print("[7/8] Writing frozen trajectory outputs and reports...")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)
    write_parquet_atomic(event_features, args.output_root / "trajectory_event_checkpoint_features.parquet")
    write_parquet_atomic(pair_rows, args.output_root / "trajectory_pair_rows.parquet")
    write_csv_atomic(baselines, args.report_root / "trajectory_feature_baselines.csv")
    write_csv_atomic(summary, args.report_root / "trajectory_contrasts.csv")
    write_csv_atomic(direction, args.report_root / "trajectory_direction_robustness.csv")
    write_csv_atomic(temporal, args.report_root / "trajectory_temporal_robustness.csv")
    write_csv_atomic(daily, args.report_root / "trajectory_daily_consistency.csv")
    write_csv_atomic(gate, args.report_root / "trajectory_checkpoint_gate.csv")
    write_csv_atomic(timing, args.report_root / "trajectory_economic_timing.csv")
    write_csv_atomic(checks, args.report_root / "trajectory_invariants.csv")

    passed = gate[gate["passed"].astype(bool)]
    earliest = int(passed["checkpoint_bps"].min()) if len(passed) else None
    meta_out = {
        "builder_version": BUILDER_VERSION,
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "generated_at": utc_now_iso(),
        "frozen_phase7g_builder_version": meta.get("builder_version"),
        "frozen_phase7g_status": meta.get("status"),
        "frozen_phase7g_profile_rows": int(meta.get("profile_rows")),
        "frozen_phase7d1_builder_version_expected": FROZEN_PHASE7D1_VERSION,
        "raw_market_replay": False,
        "trajectory_checkpoints_bps": list(CHECKPOINT_BPS),
        "core_trajectory_features": [x[0] for x in CORE_TRAJECTORY_FEATURES],
        "primary_analysis_id": PRIMARY_ANALYSIS_ID,
        "event_checkpoint_rows": int(len(event_features)),
        "pair_rows": int(len(pair_rows)),
        "earliest_retrospectively_robust_checkpoint_bps": earliest,
        "interpretation": (
            "Retrospective matched-event trajectory/persistence robustness only.  Because this hypothesis was formulated "
            "after Phase-7G validation was inspected, temporal eras are not a fresh holdout.  No prospective/all-market "
            "signal or trading edge is established."
        ),
    }
    atomic_write_json(args.output_root / "metadata.json", meta_out)
    atomic_write_text(args.report_root / "trajectory_findings.md", build_findings(meta, gate, timing))

    print("[8/8] Phase 7H summary")
    print(f"      status={status}")
    print(f"      event-checkpoint rows={len(event_features):,}")
    print(f"      paired trajectory rows={len(pair_rows):,}")
    print(f"      earliest retrospectively robust checkpoint={earliest if earliest is not None else 'NONE'}")
    print(f"      reports={args.report_root}")
    if status == "FAIL":
        print("ERROR: one or more Phase-7H invariants failed; do not interpret trajectory reports.")
        return 2
    print("Interpret the continuous trajectory contrasts, temporal eras and UP/DOWN robustness before any prospective translation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
