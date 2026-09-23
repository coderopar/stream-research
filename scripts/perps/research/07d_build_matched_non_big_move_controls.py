#!/usr/bin/env python3
"""Phase 7D: matched non-big-move controls for large-move precursor discovery.

Research role
-------------
Phase 7C found a retrospective anatomy around large BTCUSDT physical moves:
higher background activity/volatility, thinner visible near-touch liquidity,
higher visible-band turnover, and a possible price-displacement-vs-aggressive-flow
relationship.  Phase 7D is the falsification step.

This script does NOT train a predictor and does NOT claim prospective alpha.  It:

1. freezes Phase-7A/7C physical-event labels,
2. builds ordinary control anchors that do NOT realize 35 bps in either
   direction over the next 30 minutes,
3. matches nested 50+/75+/100+ physical-event extrema to controls using two
   causal-background matching tiers,
4. profiles the same Phase-7C preparation windows for the matched controls,
5. reports parent-paired event-minus-control mechanism contrasts.

Two matching tiers are intentionally different:

REGIME_MATCHED
    Matches time-of-day, volatility/activity, the directional geometry of the
    recent move into the anchor, and OI context when available.  Near-touch
    depth/spread are NOT matched away, so static liquidity differences remain
    testable.

STRICT_STATE_MATCHED
    Adds spread plus direction-normalized supporting/opposing 5-bps depth.
    This asks whether dynamic liquidity/flow mechanics survive after static
    visible-book state is also conditioned on.

Controls are outcome-selected retrospectively (future <35 bps in either
 direction for 30m).  Their matching inputs are causal at the anchor.  This is
still a retrospective matched-control study, not a deployable detector.
"""

from __future__ import annotations

import argparse
import importlib.util
import hashlib
import json
import math
import os
import tempfile
import time
from collections import deque
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

BUILDER_VERSION = "1.0.0"
SCHEMA_VERSION = 1
SCHEMA_NAME = "large_move_matched_controls_v1"
DEFAULT_SYMBOL = "BTCUSDT"
PRIMARY_HORIZON_SEC = 1800
CONTROL_NEGATIVE_THRESHOLD_BPS = 35.0
CONTROL_CADENCE_SEC = 5
CONTROLS_PER_EVENT = 3
CONTROL_REUSE_CAP = 5
CONTROL_EVENT_EMBARGO_SEC = 300

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()
DEFAULT_CATALOG_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / "large_move_event_catalog_v1"
DEFAULT_ATLAS_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / "large_move_participant_behavior_atlas_v1"
DEFAULT_PRIMITIVE_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / "liquidity_event_primitives_v1"
DEFAULT_FLOW_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "flow_features_1s_v1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / SCHEMA_NAME
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "perps_large_move_matched_controls"
DEFAULT_PHASE7C_SCRIPT = REPO_ROOT / "scripts" / "perps" / "research" / "07c_build_large_move_participant_behavior_atlas.py"

COHORT_TARGETS = (
    ("50_PLUS", 50),
    ("75_PLUS", 75),
    ("100_PLUS", 100),
)
MATCH_TIERS = ("REGIME_MATCHED", "STRICT_STATE_MATCHED")

# Match-feature names are derived, not raw source-column names.
REGIME_MATCH_FEATURES_CORE = (
    "m_log1p_realized_vol_60s_bps",
    "m_log1p_trade_count_60s",
    "m_log1p_quote_volume_60s",
    "m_dir_return_15s_bps",
    "m_dir_return_60s_bps",
)
REGIME_MATCH_FEATURE_OPTIONAL = "m_open_interest_change_300s_pct"
STRICT_ADDITIONAL_MATCH_FEATURES = (
    "m_log1p_spread_bps",
    "m_log1p_support_depth_quote_5bps",
    "m_log1p_opposing_depth_quote_5bps",
)


def require_numeric_stack():
    try:
        import numpy as np
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Phase 7D requires numpy and pandas. Activate the project .venv and run:\n"
            "  python -m pip install numpy pandas"
        ) from exc
    return np, pd


def require_stack():
    np, pd = require_numeric_stack()
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Phase 7D Parquet I/O requires pyarrow. Activate the project .venv and run:\n"
            "  python -m pip install pyarrow"
        ) from exc
    return np, pd, pa, pq


def load_phase7c_module(path: Path):
    if not path.exists():
        raise RuntimeError(f"Missing frozen Phase-7C script: {path}")
    spec = importlib.util.spec_from_file_location("phase7c_frozen", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import Phase-7C script: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if getattr(mod, "SCHEMA_NAME", None) != "large_move_participant_behavior_atlas_v1":
        raise RuntimeError("Unexpected Phase-7C schema contract")
    return mod


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7d-", suffix=path.suffix, dir=str(path.parent))
    os.close(fd)
    try:
        Path(tmp).write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def write_parquet_atomic(df, path: Path) -> None:
    _, _, pa, pq = require_stack()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7d-", suffix=".parquet", dir=str(path.parent))
    os.close(fd)
    try:
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def parse_date(text: str) -> date:
    return date.fromisoformat(text)


def daterange(start: date, end: date) -> list[date]:
    out: list[date] = []
    d = start
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return out


def safe_float(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return math.nan
    return x if math.isfinite(x) else math.nan


def cohort_mask(events, target_bps: int, horizon_sec: int = PRIMARY_HORIZON_SEC):
    _, pd = require_numeric_stack()
    col = f"time_to_{target_bps}_sec"
    x = pd.to_numeric(events[col], errors="coerce")
    return x.notna() & (x <= horizon_sec)


def forward_rolling_extreme(values, window: int, want_max: bool):
    """Forward inclusive rolling max/min over row windows in O(n).

    Returns the extreme over values[i:i+window]. NaNs are ignored.  The caller
    separately checks timestamp completeness before treating the result as a
    valid 30-minute outcome label.
    """
    np, _ = require_numeric_stack()
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)
    q: deque[int] = deque()
    better = (lambda a, b: a >= b) if want_max else (lambda a, b: a <= b)
    # Scan from right to left. q holds indices in [i, i+window-1] with the
    # current best at q[0].
    for i in range(n - 1, -1, -1):
        right = i + window
        while q and q[0] >= right:
            q.popleft()
        v = arr[i]
        if math.isfinite(v):
            while q and better(v, arr[q[-1]]):
                q.pop()
            q.append(i)
        out[i] = arr[q[0]] if q else math.nan
    return out


def future_no_big_move_metrics(seconds, price, horizon_sec: int, threshold_bps: float):
    np, _ = require_numeric_stack()
    s = np.asarray(seconds, dtype=np.int64)
    p = np.asarray(price, dtype=float)
    n = len(s)
    rows = horizon_sec + 1
    fmax = forward_rolling_extreme(p, rows, True)
    fmin = forward_rolling_extreme(p, rows, False)
    complete = np.zeros(n, dtype=bool)
    idx = np.arange(n)
    j = idx + horizon_sec
    ok = j < n
    complete[ok] = (s[j[ok]] - s[idx[ok]]) == horizon_sec
    # Outcome-negative controls must have a fully observed future price path;
    # otherwise a missing print could hide a threshold crossing.
    finite = np.isfinite(p).astype(np.int64)
    cs = np.concatenate(([0], np.cumsum(finite)))
    end_idx = np.minimum(n, idx + rows)
    finite_count = cs[end_idx] - cs[idx]
    complete &= finite_count == rows
    with np.errstate(divide="ignore", invalid="ignore"):
        up = (fmax / p - 1.0) * 1e4
        down = (1.0 - fmin / p) * 1e4
    eligible = complete & np.isfinite(p) & (p > 0) & np.isfinite(up) & np.isfinite(down)
    eligible &= (up < float(threshold_bps)) & (down < float(threshold_bps))
    return up, down, complete, eligible


def merge_intervals(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    raw = sorted((int(a), int(b)) for a, b in intervals)
    out: list[list[int]] = []
    for a, b in raw:
        if not out or a > out[-1][1] + 1:
            out.append([a, b])
        else:
            out[-1][1] = max(out[-1][1], b)
    return [(a, b) for a, b in out]


def interval_mask(seconds, intervals: Sequence[tuple[int, int]]):
    np, _ = require_numeric_stack()
    s = np.asarray(seconds, dtype=np.int64)
    if not intervals:
        return np.zeros(len(s), dtype=bool)
    merged = merge_intervals(intervals)
    starts = np.asarray([x[0] for x in merged], dtype=np.int64)
    ends = np.asarray([x[1] for x in merged], dtype=np.int64)
    pos = np.searchsorted(starts, s, side="right") - 1
    valid = pos >= 0
    safe = np.clip(pos, 0, len(ends) - 1)
    return valid & (s <= ends[safe])


def build_event_contamination_intervals(events, embargo_sec: int) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for r in events.to_dict("records"):
        t35 = safe_float(r.get("time_to_35_sec"))
        if not math.isfinite(t35) or t35 > PRIMARY_HORIZON_SEC:
            continue
        start = int(r["onset_second"]) - int(embargo_sec)
        end_candidates = [safe_float(r.get("reversal_confirm_second")), safe_float(r.get("extreme_second")), safe_float(r.get("hit_35_second"))]
        finite = [int(x) for x in end_candidates if math.isfinite(x)]
        end = (max(finite) if finite else int(r["onset_second"])) + int(embargo_sec)
        intervals.append((start, end))
    return merge_intervals(intervals)


def build_match_frame(frame, direction: int):
    """Direction-aware causal matching state at each anchor second."""
    np, pd = require_numeric_stack()
    d = 1 if int(direction) > 0 else -1
    out = pd.DataFrame({"receive_second": pd.to_numeric(frame["receive_second"], errors="coerce")})
    out["direction"] = d
    out["anchor_date"] = pd.to_datetime(out["receive_second"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    out["utc_hour"] = pd.to_datetime(out["receive_second"], unit="s", utc=True).dt.hour.astype("int16")

    def num(c: str):
        return pd.to_numeric(frame[c], errors="coerce") if c in frame.columns else pd.Series(np.nan, index=frame.index)

    rv = num("realized_vol_60s_bps").clip(lower=0)
    tc = num("trade_count_60s").clip(lower=0)
    qv = num("quote_volume_60s").clip(lower=0)
    out["m_log1p_realized_vol_60s_bps"] = np.log1p(rv)
    out["m_log1p_trade_count_60s"] = np.log1p(tc)
    out["m_log1p_quote_volume_60s"] = np.log1p(qv)
    out["m_dir_return_15s_bps"] = d * num("return_15s_bps")
    out["m_dir_return_60s_bps"] = d * num("return_60s_bps")
    out["m_open_interest_change_300s_pct"] = num("open_interest_change_300s_pct")
    out["m_log1p_spread_bps"] = np.log1p(num("spread_bps").clip(lower=0))
    if d > 0:
        support = num("bid_depth_quote_5bps")
        oppose = num("ask_depth_quote_5bps")
    else:
        support = num("ask_depth_quote_5bps")
        oppose = num("bid_depth_quote_5bps")
    out["m_log1p_support_depth_quote_5bps"] = np.log1p(support.clip(lower=0))
    out["m_log1p_opposing_depth_quote_5bps"] = np.log1p(oppose.clip(lower=0))
    if "book_valid" in frame.columns:
        out["book_valid"] = frame["book_valid"].fillna(False).astype(bool).to_numpy()
    else:
        out["book_valid"] = False
    if "flow_feature_context_ready" in frame.columns:
        out["flow_feature_context_ready"] = frame["flow_feature_context_ready"].fillna(False).astype(bool).to_numpy()
    else:
        # Flow schemas predating this flag are accepted if the core columns are finite.
        out["flow_feature_context_ready"] = True
    return out


def resolve_match_features(candidates) -> tuple[list[str], list[str]]:
    np, pd = require_numeric_stack()
    regime = list(REGIME_MATCH_FEATURES_CORE)
    missing = [c for c in regime if c not in candidates.columns]
    if missing:
        raise RuntimeError(f"Missing required Phase-7D regime matching features: {missing}")
    # OI is useful but optional: include it only if the candidate pool has broad coverage.
    if REGIME_MATCH_FEATURE_OPTIONAL in candidates.columns:
        cov = float(pd.to_numeric(candidates[REGIME_MATCH_FEATURE_OPTIONAL], errors="coerce").notna().mean())
        if cov >= 0.80:
            regime.append(REGIME_MATCH_FEATURE_OPTIONAL)
    strict = regime + list(STRICT_ADDITIONAL_MATCH_FEATURES)
    missing_strict = [c for c in STRICT_ADDITIONAL_MATCH_FEATURES if c not in candidates.columns]
    if missing_strict:
        raise RuntimeError(f"Missing required Phase-7D strict-state matching features: {missing_strict}")
    return regime, strict


def robust_scale(values) -> float:
    np, pd = require_numeric_stack()
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, copy=False)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    if math.isfinite(mad) and mad > 1e-12:
        return mad
    sd = float(np.std(x, ddof=0))
    return sd if math.isfinite(sd) and sd > 1e-12 else 1.0


def candidate_tiers(event_row: Mapping[str, Any], groups, all_by_hour):
    np, _ = require_numeric_stack()
    day = str(event_row["anchor_date"])
    h = int(event_row["utc_hour"])
    direction = int(event_row["direction"])
    same = groups.get((direction, day, h), np.array([], dtype=np.int64))
    adj_parts = [groups.get((direction, day, hh), np.array([], dtype=np.int64)) for hh in (h - 1, h + 1) if 0 <= hh <= 23]
    adjacent = np.unique(np.concatenate(adj_parts)) if adj_parts and any(len(x) for x in adj_parts) else np.array([], dtype=np.int64)
    any_same = all_by_hour.get((direction, h), np.array([], dtype=np.int64))
    return (
        ("SAME_DATE_SAME_HOUR", same),
        ("SAME_DATE_ADJACENT_HOUR", adjacent),
        ("ANY_DATE_SAME_HOUR", any_same),
    )


def match_one_analysis(events, candidates, match_features: Sequence[str], controls_per_event: int, reuse_cap: int, cohort: str, match_tier: str):
    np, pd = require_numeric_stack()
    if events.empty or candidates.empty:
        return pd.DataFrame()
    cand = candidates.copy().reset_index(drop=True)
    cand = cand.dropna(subset=list(match_features)).reset_index(drop=True)
    if cand.empty:
        return pd.DataFrame()
    scales = {c: robust_scale(cand[c]) for c in match_features}
    groups = {(int(k[0]), str(k[1]), int(k[2])): np.asarray(list(idx), dtype=np.int64)
              for k, idx in cand.groupby(["direction", "anchor_date", "utc_hour"], sort=False).groups.items()}
    all_by_hour = {(int(k[0]), int(k[1])): np.asarray(list(idx), dtype=np.int64)
                   for k, idx in cand.groupby(["direction", "utc_hour"], sort=False).groups.items()}
    usage = np.zeros(len(cand), dtype=np.int16)
    out: list[dict[str, Any]] = []

    for ev in events.sort_values("anchor_second", kind="mergesort").to_dict("records"):
        if any(not math.isfinite(safe_float(ev.get(c))) for c in match_features):
            continue
        selected: list[tuple[int, str, float]] = []
        seen: set[int] = set()
        needed = int(controls_per_event)
        for fallback, idxs in candidate_tiers(ev, groups, all_by_hour):
            if needed <= 0:
                break
            if len(idxs) == 0:
                continue
            valid_idxs = idxs[usage[idxs] < int(reuse_cap)]
            if len(valid_idxs) == 0:
                continue
            sub = cand.iloc[valid_idxs]
            dist = np.zeros(len(sub), dtype=float)
            valid = np.ones(len(sub), dtype=bool)
            for c in match_features:
                evv = safe_float(ev.get(c))
                vals = pd.to_numeric(sub[c], errors="coerce").to_numpy(float)
                valid &= np.isfinite(vals)
                dist += np.abs(vals - evv) / scales[c]
            # Very small deterministic temporal-distance tiebreaker.
            dist += np.abs(sub["receive_second"].to_numpy(np.int64) - int(ev["anchor_second"])) / (86400.0 * 10000.0)
            order = np.argsort(np.where(valid, dist, np.inf), kind="mergesort")
            for pos in order:
                if needed <= 0:
                    break
                dd = float(dist[pos])
                if not math.isfinite(dd):
                    break
                idx = int(valid_idxs[pos])
                if idx in seen or usage[idx] >= int(reuse_cap):
                    continue
                selected.append((idx, fallback, dd))
                seen.add(idx)
                usage[idx] += 1
                needed -= 1

        for ordinal, (idx, fallback, distance) in enumerate(selected, 1):
            c = cand.iloc[idx]
            row = {
                "cohort": cohort,
                "match_tier": match_tier,
                "parent_event_id": ev["event_id"],
                "parent_event_date": ev["anchor_date"],
                "direction": int(ev["direction"]),
                "control_ordinal": int(ordinal),
                "control_key": f"{int(ev['direction']):+d}:{int(c['receive_second'])}",
                "control_second": int(c["receive_second"]),
                "control_date": str(c["anchor_date"]),
                "control_utc_hour": int(c["utc_hour"]),
                "match_fallback": fallback,
                "match_distance": distance,
                "control_forward_up_bps_30m": safe_float(c.get("forward_up_bps_30m")),
                "control_forward_down_bps_30m": safe_float(c.get("forward_down_bps_30m")),
            }
            for f in match_features:
                row[f"event__{f}"] = safe_float(ev.get(f))
                row[f"control__{f}"] = safe_float(c.get(f))
            out.append(row)
    return pd.DataFrame(out)


def standardized_mean_difference(a, b) -> float:
    np, pd = require_numeric_stack()
    aa = pd.to_numeric(a, errors="coerce").dropna().to_numpy(float)
    bb = pd.to_numeric(b, errors="coerce").dropna().to_numpy(float)
    if len(aa) < 2 or len(bb) < 2:
        return math.nan
    den = math.sqrt((float(np.var(aa, ddof=1)) + float(np.var(bb, ddof=1))) / 2.0)
    if den <= 1e-12:
        return 0.0 if math.isclose(float(np.mean(aa)), float(np.mean(bb)), abs_tol=1e-12) else math.nan
    return float((np.mean(aa) - np.mean(bb)) / den)


def control_balance_report(assignments, event_states, match_feature_map: Mapping[str, Sequence[str]]):
    np, pd = require_numeric_stack()
    rows: list[dict[str, Any]] = []
    if assignments.empty:
        return pd.DataFrame()
    for (cohort, tier), g in assignments.groupby(["cohort", "match_tier"], sort=True):
        feats = list(match_feature_map[tier])
        parent_ids = g["parent_event_id"].drop_duplicates()
        ev = event_states[event_states["event_id"].isin(parent_ids)].set_index("event_id")
        for f in feats:
            # Equal parent weighting: compare event value to the mean of its matched controls.
            cg = g.groupby("parent_event_id")[f"control__{f}"].mean()
            common = ev.index.intersection(cg.index)
            a = pd.to_numeric(ev.loc[common, f], errors="coerce")
            b = pd.to_numeric(cg.loc[common], errors="coerce")
            rows.append({
                "cohort": cohort, "match_tier": tier, "feature": f,
                "matched_parents": int(len(common)),
                "event_mean": float(a.mean()) if len(a) else math.nan,
                "control_parent_mean": float(b.mean()) if len(b) else math.nan,
                "standardized_mean_difference": standardized_mean_difference(a, b),
                "mean_paired_difference": float((a.to_numpy(float) - b.to_numpy(float)).mean()) if len(common) else math.nan,
            })
    return pd.DataFrame(rows)


def control_coverage_report(assignments, cohort_events, controls_per_event: int):
    _, pd = require_numeric_stack()
    rows = []
    for cohort, _target in COHORT_TARGETS:
        expected = int(len(cohort_events.get(cohort, [])))
        for tier in MATCH_TIERS:
            g = assignments[(assignments["cohort"] == cohort) & (assignments["match_tier"] == tier)] if not assignments.empty else pd.DataFrame()
            counts = g.groupby("parent_event_id").size() if not g.empty else pd.Series(dtype=int)
            rows.append({
                "cohort": cohort, "match_tier": tier,
                "events_eligible": expected,
                "events_with_any_control": int((counts >= 1).sum()),
                "events_with_full_controls": int((counts >= controls_per_event).sum()),
                "full_control_coverage": float((counts >= controls_per_event).sum() / max(1, expected)),
                "controls": int(len(g)),
                "unique_control_keys": int(g["control_key"].nunique()) if not g.empty else 0,
                "max_observed_reuse": int(g["control_key"].value_counts().max()) if not g.empty else 0,
                "same_date_same_hour_share": float((g["match_fallback"] == "SAME_DATE_SAME_HOUR").mean()) if not g.empty else math.nan,
                "same_date_adjacent_hour_share": float((g["match_fallback"] == "SAME_DATE_ADJACENT_HOUR").mean()) if not g.empty else math.nan,
                "any_date_same_hour_share": float((g["match_fallback"] == "ANY_DATE_SAME_HOUR").mean()) if not g.empty else math.nan,
            })
    return pd.DataFrame(rows)


def profile_controls(assignments, phase7c, primitive_root: Path, flow_root: Path, symbol: str, start: date, end: date, features: Sequence[str]):
    np, pd, _, _ = require_stack()
    if assignments.empty:
        return pd.DataFrame()
    unique = assignments[["control_key", "control_second", "control_date", "direction"]].drop_duplicates().copy()
    rows: list[dict[str, Any]] = []
    days = daterange(start, end)
    by_day = {d: g for d, g in unique.groupby("control_date")}
    for i, d in enumerate(days, 1):
        dg = by_day.get(d.isoformat())
        if dg is None or dg.empty:
            continue
        ctx_days = [d - timedelta(days=1), d]
        prim_files = [phase7c.primitive_day_path(primitive_root, symbol, x) for x in ctx_days]
        prim_files = [p for p in prim_files if p.exists()]
        if not phase7c.primitive_day_path(primitive_root, symbol, d).exists():
            raise RuntimeError(f"Missing Phase-6B primitive partition for control profiling {d}")
        prim = phase7c.read_parquet_columns(prim_files, phase7c.PRIMITIVE_COLUMNS, required=("receive_second", "book_valid"))
        flow_files: list[Path] = []
        for x in ctx_days:
            flow_files.extend(phase7c.flow_day_files(flow_root, symbol, x))
        flow = phase7c.read_parquet_columns(flow_files, phase7c.FLOW_OPTIONAL_COLUMNS, required=("receive_second",)) if flow_files else pd.DataFrame(columns=["receive_second"])
        overlap = [c for c in flow.columns if c != "receive_second" and c in prim.columns]
        flow = flow.drop(columns=overlap, errors="ignore")
        context = prim.merge(flow, on="receive_second", how="left", validate="one_to_one").sort_values("receive_second", kind="mergesort").reset_index(drop=True)
        t0 = time.perf_counter()
        nday = 0
        for direction in (+1, -1):
            sub = dg[dg["direction"] == direction]
            if sub.empty:
                continue
            f = phase7c.build_direction_normalized_frame(context, direction)
            seconds, arrays = phase7c.prepare_feature_arrays(f, features)
            for cr in sub.to_dict("records"):
                ev_meta = {
                    "event_id": cr["control_key"],
                    "onset_date": cr["control_date"],
                    "direction": int(cr["direction"]),
                    "direction_label": "UP" if int(cr["direction"]) > 0 else "DOWN",
                    "continuation_stratum": "NON_BIG_CONTROL",
                    "onset_second": int(cr["control_second"]),
                }
                for label, lo, hi in phase7c.PREPARATION_WINDOWS:
                    rr = phase7c.aggregate_window_feature_rows(ev_meta, seconds, arrays, label, lo, hi, features)
                    for r in rr:
                        r["control_key"] = cr["control_key"]
                        r["control_date"] = cr["control_date"]
                    rows.extend(rr)
                nday += 1
        print(f"  [{i:02d}/{len(days):02d}] {d}: unique controls={nday:,}, runtime={time.perf_counter()-t0:.2f}s")
    return pd.DataFrame(rows)


def paired_mechanism_contrasts(event_prep, control_prep, assignments, feature_family: Mapping[str, str], match_feature_map: Mapping[str, Sequence[str]]):
    np, pd = require_numeric_stack()
    if assignments.empty or control_prep.empty:
        return pd.DataFrame()
    ev = event_prep[["event_id", "event_date", "window_label", "feature", "median", "coverage_rate"]].copy()
    cp = control_prep[["control_key", "window_label", "feature", "median", "coverage_rate"]].copy()
    cp = cp.rename(columns={"median": "control_value", "coverage_rate": "control_coverage"})
    rows: list[dict[str, Any]] = []

    for (cohort, tier), ag in assignments.groupby(["cohort", "match_tier"], sort=True):
        parents = ag["parent_event_id"].drop_duplicates()
        e = ev[ev["event_id"].isin(parents)]
        ac = ag[["parent_event_id", "control_key"]].merge(cp, on="control_key", how="left")
        cmean = ac.groupby(["parent_event_id", "window_label", "feature"], as_index=False).agg(
            control_parent_mean=("control_value", "mean"),
            control_parent_median=("control_value", "median"),
            controls_available=("control_value", "count"),
            control_mean_coverage=("control_coverage", "mean"),
        )
        merged = e.merge(cmean, left_on=["event_id", "window_label", "feature"], right_on=["parent_event_id", "window_label", "feature"], how="inner")
        merged["event_value"] = pd.to_numeric(merged["median"], errors="coerce")
        merged["control_parent_mean"] = pd.to_numeric(merged["control_parent_mean"], errors="coerce")
        merged["paired_difference"] = merged["event_value"] - merged["control_parent_mean"]

        for (loc, feat), g in merged.groupby(["window_label", "feature"], sort=True):
            gg = g.replace([np.inf, -np.inf], np.nan).dropna(subset=["event_value", "control_parent_mean", "paired_difference"])
            if len(gg) < 5:
                continue
            event_vals = gg["event_value"].to_numpy(float)
            control_vals = gg["control_parent_mean"].to_numpy(float)
            diffs = gg["paired_difference"].to_numpy(float)
            scale = robust_scale(pd.Series(np.concatenate([event_vals, control_vals])))
            meddiff = float(np.median(diffs))
            family = feature_family.get(str(feat), "UNKNOWN")
            # The raw atlas feature names differ from derived matching-state names;
            # this flag is about scientific role rather than string equality.
            role = "TEST_MECHANISM"
            if tier == "STRICT_STATE_MATCHED" and feat in {"spread_bps", "support_depth_quote_5bps", "opposing_depth_quote_5bps"}:
                role = "STATIC_STATE_MATCHED_AWAY_DIAGNOSTIC"
            rows.append({
                "cohort": cohort, "match_tier": tier, "window_label": loc, "feature": feat, "feature_family": family,
                "feature_role": role,
                "matched_parents": int(len(gg)),
                "median_event_value": float(np.median(event_vals)),
                "median_control_parent_mean": float(np.median(control_vals)),
                "median_paired_difference": meddiff,
                "mean_paired_difference": float(np.mean(diffs)),
                "pooled_robust_scale": scale,
                "paired_robust_standardized_difference": meddiff / scale if math.isfinite(scale) and scale > 0 else math.nan,
                "positive_parent_share": float(np.mean(diffs > 0)),
                "negative_parent_share": float(np.mean(diffs < 0)),
                "median_event_window_coverage": float(pd.to_numeric(gg["coverage_rate"], errors="coerce").median()),
                "median_control_window_coverage": float(pd.to_numeric(gg["control_mean_coverage"], errors="coerce").median()),
                "interpretation": "RETROSPECTIVE_MATCHED_CONTROL_DISCOVERY_NOT_PROSPECTIVE_ALPHA",
            })
    return pd.DataFrame(rows)


def daily_paired_consistency(event_prep, control_prep, assignments):
    np, pd = require_numeric_stack()
    if assignments.empty or control_prep.empty:
        return pd.DataFrame()
    ev = event_prep[["event_id", "event_date", "window_label", "feature", "median"]].copy()
    cp = control_prep[["control_key", "window_label", "feature", "median"]].rename(columns={"median": "control_value"})
    rows: list[dict[str, Any]] = []
    for (cohort, tier), ag in assignments.groupby(["cohort", "match_tier"], sort=True):
        parents = ag["parent_event_id"].drop_duplicates()
        e = ev[ev["event_id"].isin(parents)]
        ac = ag[["parent_event_id", "control_key"]].merge(cp, on="control_key", how="left")
        cmean = ac.groupby(["parent_event_id", "window_label", "feature"], as_index=False)["control_value"].mean()
        m = e.merge(cmean, left_on=["event_id", "window_label", "feature"], right_on=["parent_event_id", "window_label", "feature"], how="inner")
        m["diff"] = pd.to_numeric(m["median"], errors="coerce") - pd.to_numeric(m["control_value"], errors="coerce")
        for (loc, feat), g in m.groupby(["window_label", "feature"], sort=True):
            daydiff = []
            for day, dg in g.groupby("event_date"):
                x = pd.to_numeric(dg["diff"], errors="coerce").dropna()
                if len(x) >= 2:
                    daydiff.append(float(x.median()))
            if not daydiff:
                continue
            arr = np.asarray(daydiff, dtype=float)
            rows.append({
                "cohort": cohort, "match_tier": tier, "window_label": loc, "feature": feat,
                "days_comparable": int(len(arr)),
                "positive_days": int((arr > 0).sum()), "negative_days": int((arr < 0).sum()), "zero_days": int((arr == 0).sum()),
                "positive_day_share": float((arr > 0).mean()),
                "negative_day_share": float((arr < 0).mean()),
                "median_daily_paired_difference": float(np.median(arr)),
            })
    return pd.DataFrame(rows)


def cross_tier_survival(contrasts, daily):
    np, pd = require_numeric_stack()
    if contrasts.empty:
        return pd.DataFrame()
    key = ["cohort", "window_label", "feature", "feature_family"]
    reg = contrasts[contrasts["match_tier"] == "REGIME_MATCHED"].copy()
    strict = contrasts[contrasts["match_tier"] == "STRICT_STATE_MATCHED"].copy()
    cols = key + ["matched_parents", "paired_robust_standardized_difference", "median_paired_difference", "positive_parent_share", "negative_parent_share"]
    reg = reg[cols].rename(columns={c: f"regime__{c}" for c in cols if c not in key})
    strict = strict[cols].rename(columns={c: f"strict__{c}" for c in cols if c not in key})
    out = reg.merge(strict, on=key, how="inner")
    if daily is not None and not daily.empty:
        dd = daily.copy()
        dd["same_sign_day_share"] = dd[["positive_day_share", "negative_day_share"]].max(axis=1)
        dr = dd[dd["match_tier"] == "REGIME_MATCHED"][key[:-1] + ["days_comparable", "same_sign_day_share"]].rename(columns={"days_comparable":"regime__days_comparable","same_sign_day_share":"regime__same_sign_day_share"})
        ds = dd[dd["match_tier"] == "STRICT_STATE_MATCHED"][key[:-1] + ["days_comparable", "same_sign_day_share"]].rename(columns={"days_comparable":"strict__days_comparable","same_sign_day_share":"strict__same_sign_day_share"})
        out = out.merge(dr, on=key[:-1], how="left").merge(ds, on=key[:-1], how="left")
    r = pd.to_numeric(out["regime__paired_robust_standardized_difference"], errors="coerce")
    s = pd.to_numeric(out["strict__paired_robust_standardized_difference"], errors="coerce")
    out["same_effect_direction"] = np.sign(r) == np.sign(s)
    out["min_abs_rsd_across_tiers"] = np.minimum(np.abs(r), np.abs(s))
    out["survival_grade"] = "WEAK_OR_INCONSISTENT"
    strong = out["same_effect_direction"] & (out["min_abs_rsd_across_tiers"] >= 0.20)
    if "regime__same_sign_day_share" in out.columns:
        strong &= pd.to_numeric(out["regime__same_sign_day_share"], errors="coerce").fillna(0) >= 0.65
        strong &= pd.to_numeric(out["strict__same_sign_day_share"], errors="coerce").fillna(0) >= 0.65
    out.loc[strong, "survival_grade"] = "SURVIVES_BOTH_MATCHING_TIERS_DESCRIPTIVELY"
    return out.sort_values(["survival_grade", "min_abs_rsd_across_tiers"], ascending=[True, False], kind="mergesort")


def feature_family_map(phase7c, features: Sequence[str]) -> dict[str, str]:
    return {f: str(phase7c.FEATURE_DICTIONARY.get(f, ("RICH_CAUSAL_FLOW_FEATURE", ""))[0]) for f in features}


def build_findings(summary: Mapping[str, Any], coverage, balance, survival) -> str:
    lines = [
        "# Phase 7D — Matched non-big-move controls",
        "",
        f"- Status: **{summary['status']}**",
        f"- Builder: **{summary['builder_version']}**",
        f"- Candidate controls: **{summary['candidate_controls']:,}** direction-specific anchors",
        f"- Control assignments: **{summary['control_assignments']:,}**",
        f"- Unique profiled control keys: **{summary['unique_profiled_controls']:,}**",
        "- Research role: retrospective matched-control falsification; **not prospective alpha validation**.",
        "- Negative-control outcome rule: neither direction reaches 35 bps over the following 30 minutes.",
        "",
        "## Matching design",
        "",
        "- `REGIME_MATCHED`: time-of-day + volatility/activity + recent direction-normalized return geometry + OI when sufficiently covered.",
        "- `STRICT_STATE_MATCHED`: regime match plus spread and direction-normalized supporting/opposing 5-bps visible depth.",
        "- Candidate mechanisms such as visible-band churn and flow/price response are deliberately not matched away.",
        "",
        "## Coverage / balance",
        "",
    ]
    if coverage is not None and not coverage.empty:
        for r in coverage.to_dict("records"):
            lines.append(f"- {r['cohort']} / {r['match_tier']}: full 3-control coverage **{100*safe_float(r['full_control_coverage']):.1f}%**, controls={int(r['controls']):,}.")
    if balance is not None and not balance.empty:
        mb = balance.assign(abs_smd=balance["standardized_mean_difference"].abs()).groupby(["cohort","match_tier"])["abs_smd"].max().reset_index()
        lines += ["", "Maximum absolute matching-feature SMD (equal parent weighting):"]
        for r in mb.to_dict("records"):
            lines.append(f"- {r['cohort']} / {r['match_tier']}: **{safe_float(r['abs_smd']):.3f}**")
    lines += ["", "## Cross-tier mechanism survival", ""]
    if survival is None or survival.empty:
        lines.append("- No cross-tier mechanism rows were available.")
    else:
        top = survival[survival["survival_grade"] == "SURVIVES_BOTH_MATCHING_TIERS_DESCRIPTIVELY"].head(20)
        if top.empty:
            lines.append("- No feature/window combination met the descriptive cross-tier survival screen (same sign, |RSD|>=0.20 in both tiers, and >=65% same-sign comparable days where available).")
        else:
            lines.append("The following are **discovery survivors only**, not predictive signals:")
            for r in top.to_dict("records"):
                lines.append(
                    f"- {r['cohort']} / {r['window_label']} / `{r['feature']}` ({r['feature_family']}): "
                    f"regime RSD={safe_float(r['regime__paired_robust_standardized_difference']):+.3f}, "
                    f"strict RSD={safe_float(r['strict__paired_robust_standardized_difference']):+.3f}."
                )
    lines += [
        "",
        "## Interpretation contract",
        "",
        "A surviving feature says that its event-control difference remains visible after matching on the specified retrospective causal background state. It does **not** establish that the physical extremum or the control-selection rule is available prospectively. Any surviving motif must be frozen and scanned across all market seconds in a later causal validation phase.",
        "",
    ]
    return "\n".join(lines)


def run_scan_candidate_pool(args, phase7c, events):
    np, pd, _, _ = require_stack()
    days = daterange(parse_date(args.start_date), parse_date(args.end_date))
    contamination = build_event_contamination_intervals(events, args.control_event_embargo_sec)
    cand_parts: list[Any] = []
    event_state_parts: list[Any] = []
    events_by_date = {d: g for d, g in events.groupby("onset_date")}
    print("[2/7] Scanning causal anchor state and building clean non-big-move candidate pool...")
    for i, d in enumerate(days, 1):
        t0 = time.perf_counter()
        curp = phase7c.primitive_day_path(args.primitive_root, args.symbol, d)
        if not curp.exists():
            raise RuntimeError(f"Missing Phase-6B primitive partition: {curp}")
        future_days = [d, d + timedelta(days=1)]
        prim_future_files = [phase7c.primitive_day_path(args.primitive_root, args.symbol, x) for x in future_days]
        prim_future_files = [p for p in prim_future_files if p.exists()]
        pf = phase7c.read_parquet_columns(prim_future_files, ("receive_second", "last_trade_price"), required=("receive_second",))
        pf = pf.sort_values("receive_second", kind="mergesort").reset_index(drop=True)
        fup, fdn, fcomplete, feligible = future_no_big_move_metrics(
            pf["receive_second"].to_numpy(np.int64),
            pd.to_numeric(pf["last_trade_price"], errors="coerce").to_numpy(float),
            args.horizon_sec,
            args.negative_threshold_bps,
        )
        outcome = pd.DataFrame({
            "receive_second": pf["receive_second"].to_numpy(np.int64),
            "forward_up_bps_30m": fup,
            "forward_down_bps_30m": fdn,
            "future_complete_30m": fcomplete,
            "negative_outcome_eligible": feligible,
        })
        day_lo = int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())
        day_hi = day_lo + 86399
        outcome = outcome[(outcome["receive_second"] >= day_lo) & (outcome["receive_second"] <= day_hi)].copy()

        # Current-day causal state.
        prim = phase7c.read_parquet_columns([curp], phase7c.PRIMITIVE_COLUMNS, required=("receive_second", "book_valid"))
        flow_files = phase7c.flow_day_files(args.flow_root, args.symbol, d)
        flow = phase7c.read_parquet_columns(flow_files, phase7c.FLOW_OPTIONAL_COLUMNS, required=("receive_second",)) if flow_files else pd.DataFrame(columns=["receive_second"])
        overlap = [c for c in flow.columns if c != "receive_second" and c in prim.columns]
        flow = flow.drop(columns=overlap, errors="ignore")
        context = prim.merge(flow, on="receive_second", how="left", validate="one_to_one").sort_values("receive_second", kind="mergesort").reset_index(drop=True)
        context = context.merge(outcome, on="receive_second", how="left", validate="one_to_one")
        contaminated = interval_mask(context["receive_second"].to_numpy(np.int64), contamination)
        context["contaminated_by_large_event_interval"] = contaminated

        # Event anchor states are exact-second causal states, orientation-normalized by event direction.
        dg = events_by_date.get(d.isoformat())
        if dg is not None and not dg.empty:
            for direction in (+1, -1):
                eg = dg[dg["direction"] == direction]
                if eg.empty:
                    continue
                mf = build_match_frame(context, direction).set_index("receive_second")
                for er in eg.to_dict("records"):
                    sec = int(er["onset_second"])
                    if sec not in mf.index:
                        continue
                    sr = mf.loc[sec]
                    row = {"event_id": er["event_id"], "anchor_second": sec, "anchor_date": d.isoformat(), "direction": int(direction), "utc_hour": int(sr["utc_hour"])}
                    for c in list(REGIME_MATCH_FEATURES_CORE) + [REGIME_MATCH_FEATURE_OPTIONAL] + list(STRICT_ADDITIONAL_MATCH_FEATURES):
                        row[c] = safe_float(sr.get(c))
                    row["book_valid"] = bool(sr.get("book_valid"))
                    row["flow_feature_context_ready"] = bool(sr.get("flow_feature_context_ready"))
                    event_state_parts.append(row)

        base_eligible = context["negative_outcome_eligible"].fillna(False).astype(bool)
        base_eligible &= context["book_valid"].fillna(False).astype(bool)
        if "flow_feature_context_ready" in context.columns:
            base_eligible &= context["flow_feature_context_ready"].fillna(False).astype(bool)
        base_eligible &= ~context["contaminated_by_large_event_interval"].fillna(True).astype(bool)
        base_eligible &= (context["receive_second"].astype("int64") % int(args.control_cadence_sec) == 0)
        # Require the complete 300-second preparation history inside the requested sample.
        global_start_sec = int(datetime(parse_date(args.start_date).year, parse_date(args.start_date).month, parse_date(args.start_date).day, tzinfo=timezone.utc).timestamp())
        base_eligible &= context["receive_second"].astype("int64") >= global_start_sec + 300
        cb = context.loc[base_eligible].copy()
        for direction in (+1, -1):
            mf = build_match_frame(cb, direction)
            mf["forward_up_bps_30m"] = pd.to_numeric(cb["forward_up_bps_30m"], errors="coerce").to_numpy(float)
            mf["forward_down_bps_30m"] = pd.to_numeric(cb["forward_down_bps_30m"], errors="coerce").to_numpy(float)
            cand_parts.append(mf)
        print(f"  [{i:02d}/{len(days):02d}] {d}: ordinary anchors={len(cb):,} ({2*len(cb):,} direction-specific), runtime={time.perf_counter()-t0:.2f}s")

    candidates = pd.concat(cand_parts, ignore_index=True) if cand_parts else pd.DataFrame()
    event_states = pd.DataFrame(event_state_parts)
    if not candidates.empty:
        candidates["receive_second"] = candidates["receive_second"].astype("int64")
    return candidates, event_states


def check_invariants(events, candidates, event_states, assignments, control_prep, args):
    np, pd = require_numeric_stack()
    rows: list[dict[str, Any]] = []
    def add(name: str, ok: bool, observed: Any, expected: Any):
        rows.append({"check": name, "status": "PASS" if ok else "FAIL", "observed": observed, "expected": expected})
    add("event_anchor_states_unique", not event_states.duplicated(["event_id"]).any(), int(event_states.duplicated(["event_id"]).sum()), 0)
    if not candidates.empty:
        max_future = float(candidates[["forward_up_bps_30m", "forward_down_bps_30m"]].max().max())
        add("candidate_controls_are_strictly_below_35bps_both_directions", max_future < args.negative_threshold_bps + 1e-9, max_future, f"<{args.negative_threshold_bps}")
        add("candidate_cadence", bool((candidates["receive_second"].astype("int64") % args.control_cadence_sec == 0).all()), None, True)
    if not assignments.empty:
        add("assignment_controls_below_threshold", bool((pd.to_numeric(assignments["control_forward_up_bps_30m"], errors="coerce") < args.negative_threshold_bps).all() and (pd.to_numeric(assignments["control_forward_down_bps_30m"], errors="coerce") < args.negative_threshold_bps).all()), None, True)
        add("assignment_no_duplicate_parent_tier_control", not assignments.duplicated(["cohort","match_tier","parent_event_id","control_key"]).any(), int(assignments.duplicated(["cohort","match_tier","parent_event_id","control_key"]).sum()), 0)
        max_reuse = int(assignments.groupby(["cohort","match_tier","control_key"]).size().max())
        add("reuse_cap_respected_within_analysis", max_reuse <= args.control_reuse_cap, max_reuse, f"<={args.control_reuse_cap}")
    if not control_prep.empty:
        expected = int(assignments["control_key"].nunique() * len(load_phase7c_module(args.phase7c_script).PREPARATION_WINDOWS) * control_prep["feature"].nunique())
        add("control_preparation_unique_keys", not control_prep.duplicated(["control_key","window_label","feature"]).any(), int(control_prep.duplicated(["control_key","window_label","feature"]).sum()), 0)
        add("control_preparation_row_count", len(control_prep) == expected, len(control_prep), expected)
    return pd.DataFrame(rows)


def self_test(phase7c_script: Path | None = None) -> None:
    np, pd = require_numeric_stack()
    # Future-window outcome labeling.
    s = np.arange(1000, 1010, dtype=np.int64)
    p = np.array([100,100,100,100,100,100.2,100.2,100.2,100.2,100.2], dtype=float)
    up, dn, comp, elig = future_no_big_move_metrics(s, p, 3, 35.0)
    assert comp[0]
    assert elig[0]
    p2 = p.copy(); p2[2] = 100.5
    up2, dn2, comp2, elig2 = future_no_big_move_metrics(s, p2, 3, 35.0)
    assert up2[0] >= 50.0 - 1e-9 and not elig2[0]

    # Interval embargo.
    m = interval_mask(np.array([90,100,110,120]), [(95,115)])
    assert m.tolist() == [False, True, True, False]

    # Matching determinism and cap.
    ev = pd.DataFrame([
        {"event_id":"E1","anchor_second":1000,"anchor_date":"2026-01-01","utc_hour":0,"direction":1,
         **{c:1.0 for c in REGIME_MATCH_FEATURES_CORE}},
        {"event_id":"E2","anchor_second":1010,"anchor_date":"2026-01-01","utc_hour":0,"direction":1,
         **{c:1.1 for c in REGIME_MATCH_FEATURES_CORE}},
    ])
    cand_rows=[]
    for i in range(10):
        cand_rows.append({"receive_second":2000+i*5,"anchor_date":"2026-01-01","utc_hour":0,"direction":1,
                          "forward_up_bps_30m":5.0,"forward_down_bps_30m":4.0,
                          **{c:1.0+i*0.01 for c in REGIME_MATCH_FEATURES_CORE}})
    # Add deceptively closer opposite-direction rows; they must never be selected.
    for i in range(3):
        cand_rows.append({"receive_second":3000+i*5,"anchor_date":"2026-01-01","utc_hour":0,"direction":-1,
                          "forward_up_bps_30m":1.0,"forward_down_bps_30m":1.0,
                          **{c:1.0 for c in REGIME_MATCH_FEATURES_CORE}})
    cand=pd.DataFrame(cand_rows)
    a1=match_one_analysis(ev,cand,REGIME_MATCH_FEATURES_CORE,3,1,"100_PLUS","REGIME_MATCHED")
    a2=match_one_analysis(ev,cand,REGIME_MATCH_FEATURES_CORE,3,1,"100_PLUS","REGIME_MATCHED")
    assert len(a1)==6 and a1.equals(a2)
    assert a1["control_key"].value_counts().max()==1
    assert all(str(x).startswith("+1:") for x in a1["control_key"])

    # 7C import contract and direction-aware state derivation if script exists.
    path = phase7c_script or DEFAULT_PHASE7C_SCRIPT
    if path.exists():
        mod = load_phase7c_module(path)
        assert getattr(mod, "SCHEMA_NAME") == "large_move_participant_behavior_atlas_v1"
    frame = pd.DataFrame({
        "receive_second":[1000], "realized_vol_60s_bps":[5.0], "trade_count_60s":[100.0], "quote_volume_60s":[1_000_000.0],
        "return_15s_bps":[-3.0], "return_60s_bps":[-8.0], "open_interest_change_300s_pct":[0.1], "spread_bps":[0.2],
        "bid_depth_quote_5bps":[2_000_000.0], "ask_depth_quote_5bps":[1_000_000.0], "book_valid":[True], "flow_feature_context_ready":[True],
    })
    upf=build_match_frame(frame,+1); dnf=build_match_frame(frame,-1)
    assert float(upf.loc[0,"m_dir_return_60s_bps"])==-8.0
    assert float(dnf.loc[0,"m_dir_return_60s_bps"])==8.0
    assert float(upf.loc[0,"m_log1p_support_depth_quote_5bps"]) > float(upf.loc[0,"m_log1p_opposing_depth_quote_5bps"])
    assert float(dnf.loc[0,"m_log1p_support_depth_quote_5bps"]) < float(dnf.loc[0,"m_log1p_opposing_depth_quote_5bps"])
    print("SELF-TEST PASS")
    print("Validated forward 30-minute non-big-move labeling, event-interval embargoes, direction-aware")
    print("causal match-state transforms, deterministic bounded-reuse nearest-neighbor matching, and")
    print("the frozen Phase-7C atlas import contract without claiming prospective alpha.")


def stable_control_key_checksum(keys: Sequence[str]) -> str:
    h = hashlib.sha256()
    for key in keys:
        h.update(str(key).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build Phase-7D matched non-big-move controls for large-move precursor discovery")
    p.add_argument("--start-date", default="2026-05-18")
    p.add_argument("--end-date", default="2026-07-18")
    p.add_argument("--symbol", default=DEFAULT_SYMBOL)
    p.add_argument("--catalog-root", type=Path, default=DEFAULT_CATALOG_ROOT)
    p.add_argument("--atlas-root", type=Path, default=DEFAULT_ATLAS_ROOT)
    p.add_argument("--primitive-root", type=Path, default=DEFAULT_PRIMITIVE_ROOT)
    p.add_argument("--flow-root", type=Path, default=DEFAULT_FLOW_ROOT)
    p.add_argument("--phase7c-script", type=Path, default=DEFAULT_PHASE7C_SCRIPT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--horizon-sec", type=int, default=PRIMARY_HORIZON_SEC)
    p.add_argument("--negative-threshold-bps", type=float, default=CONTROL_NEGATIVE_THRESHOLD_BPS)
    p.add_argument("--control-cadence-sec", type=int, default=CONTROL_CADENCE_SEC)
    p.add_argument("--controls-per-event", type=int, default=CONTROLS_PER_EVENT)
    p.add_argument("--control-reuse-cap", type=int, default=CONTROL_REUSE_CAP)
    p.add_argument("--control-event-embargo-sec", type=int, default=CONTROL_EVENT_EMBARGO_SEC)
    p.add_argument("--force", action="store_true", help="Rebuild the expensive candidate pool even if a compatible checkpoint exists")
    p.add_argument("--self-test", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        self_test(args.phase7c_script)
        return 0
    np, pd, pa, pq = require_stack()
    start, end = parse_date(args.start_date), parse_date(args.end_date)
    if end < start:
        raise SystemExit("--end-date must be >= --start-date")
    phase7c = load_phase7c_module(args.phase7c_script)

    print("Building Phase-7D matched non-big-move controls...")
    print(f"Builder version       : {BUILDER_VERSION}")
    print(f"Symbol                : {args.symbol}")
    print(f"Dates                 : {start} to {end}")
    print(f"Negative outcome      : <{args.negative_threshold_bps:g} bps in either direction over next {args.horizon_sec}s")
    print(f"Candidate cadence     : {args.control_cadence_sec}s")
    print(f"Controls/event/tier   : {args.controls_per_event}")
    print("Research role         : retrospective matched-control falsification")
    print("Prospective signal    : NONE; physical extrema remain hindsight event alignments")

    print("[1/7] Loading frozen Phase-7C event metadata and preparation trajectories...")
    ep = args.atlas_root / "event_metadata.parquet"
    pp = args.atlas_root / "preparation_event_features.parquet"
    if not ep.exists() or not pp.exists():
        raise RuntimeError(f"Missing frozen Phase-7C outputs: {ep} / {pp}")
    events = pq.read_table(ep).to_pandas()
    event_prep = pq.read_table(pp).to_pandas()
    events = events[(events["onset_date"] >= start.isoformat()) & (events["onset_date"] <= end.isoformat())].copy()
    if "symbol" in events.columns:
        events = events[events["symbol"] == args.symbol].copy()
    event_prep = event_prep[event_prep["event_id"].isin(events["event_id"])].copy()
    features = sorted(event_prep["feature"].astype(str).unique())
    print(f"      events={len(events):,}, atlas features={len(features):,}")
    cohort_events: dict[str, Any] = {}
    for name, target in COHORT_TARGETS:
        ce = events[cohort_mask(events, target, args.horizon_sec)].copy()
        cohort_events[name] = ce
        print(f"      {name:<10}: {len(ce):,}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output_root / "candidate_controls.parquet"
    event_state_path = args.output_root / "event_anchor_match_states.parquet"
    checkpoint_meta = args.output_root / "metadata" / "candidate_checkpoint.json"
    compatible = False
    if candidate_path.exists() and event_state_path.exists() and checkpoint_meta.exists() and not args.force:
        try:
            meta = json.loads(checkpoint_meta.read_text(encoding="utf-8"))
            compatible = (
                meta.get("builder_version") == BUILDER_VERSION and meta.get("symbol") == args.symbol and
                meta.get("start_date") == start.isoformat() and meta.get("end_date") == end.isoformat() and
                int(meta.get("horizon_sec")) == args.horizon_sec and float(meta.get("negative_threshold_bps")) == float(args.negative_threshold_bps) and
                int(meta.get("control_cadence_sec")) == args.control_cadence_sec and int(meta.get("control_event_embargo_sec")) == args.control_event_embargo_sec
            )
        except Exception:
            compatible = False
    if compatible:
        print("[2/7] Reusing compatible candidate-control checkpoint...")
        candidates = pq.read_table(candidate_path).to_pandas()
        event_states = pq.read_table(event_state_path).to_pandas()
        print(f"      candidate direction-specific anchors={len(candidates):,}, event anchor states={len(event_states):,}")
    else:
        candidates, event_states = run_scan_candidate_pool(args, phase7c, events)
        write_parquet_atomic(candidates, candidate_path)
        write_parquet_atomic(event_states, event_state_path)
        atomic_write_json(checkpoint_meta, {
            "builder_version": BUILDER_VERSION, "symbol": args.symbol, "start_date": start.isoformat(), "end_date": end.isoformat(),
            "horizon_sec": args.horizon_sec, "negative_threshold_bps": args.negative_threshold_bps,
            "control_cadence_sec": args.control_cadence_sec, "control_event_embargo_sec": args.control_event_embargo_sec,
            "candidate_rows": int(len(candidates)), "event_anchor_states": int(len(event_states)),
        })
        print(f"      checkpointed candidate pool: {len(candidates):,} direction-specific anchors")

    regime_features, strict_features = resolve_match_features(candidates)
    match_feature_map = {"REGIME_MATCHED": regime_features, "STRICT_STATE_MATCHED": strict_features}
    print("[3/7] Matching nested large-move cohorts to ordinary controls...")
    print(f"      regime features: {', '.join(regime_features)}")
    print(f"      strict adds     : {', '.join([x for x in strict_features if x not in regime_features])}")
    assignment_parts = []
    valid_events = event_states[event_states["book_valid"].fillna(False).astype(bool) & event_states["flow_feature_context_ready"].fillna(False).astype(bool)].copy()
    for cohort, _target in COHORT_TARGETS:
        ids = set(cohort_events[cohort]["event_id"].astype(str))
        es = valid_events[valid_events["event_id"].astype(str).isin(ids)].copy()
        for tier in MATCH_TIERS:
            feats = match_feature_map[tier]
            cp = candidates[candidates["direction"].isin(es["direction"].unique())].copy()
            part = match_one_analysis(es, cp, feats, args.controls_per_event, args.control_reuse_cap, cohort, tier)
            assignment_parts.append(part)
            full = int((part.groupby("parent_event_id").size() >= args.controls_per_event).sum()) if not part.empty else 0
            print(f"      {cohort:<10} {tier:<20}: events={len(es):,}, controls={len(part):,}, full parents={full:,}")
    assignments = pd.concat(assignment_parts, ignore_index=True) if assignment_parts else pd.DataFrame()
    write_parquet_atomic(assignments, args.output_root / "control_assignments.parquet")

    print("[4/7] Profiling Phase-7C preparation features for unique matched controls...")
    control_prep_path = args.output_root / "control_preparation_features.parquet"
    control_signature = args.output_root / "metadata" / "control_profile_checkpoint.json"
    wanted_keys = sorted(assignments["control_key"].drop_duplicates().astype(str).tolist()) if not assignments.empty else []
    reuse_profiles = False
    if control_prep_path.exists() and control_signature.exists() and not args.force:
        try:
            meta = json.loads(control_signature.read_text(encoding="utf-8"))
            reuse_profiles = meta.get("builder_version") == BUILDER_VERSION and int(meta.get("unique_controls")) == len(wanted_keys) and meta.get("control_key_checksum") == stable_control_key_checksum(wanted_keys)
        except Exception:
            reuse_profiles = False
    if reuse_profiles:
        control_prep = pq.read_table(control_prep_path).to_pandas()
        print(f"      reused {control_prep['control_key'].nunique():,} profiled control keys")
    else:
        control_prep = profile_controls(assignments, phase7c, args.primitive_root, args.flow_root, args.symbol, start, end, features)
        write_parquet_atomic(control_prep, control_prep_path)
        atomic_write_json(control_signature, {
            "builder_version": BUILDER_VERSION, "unique_controls": len(wanted_keys), "control_key_checksum": stable_control_key_checksum(wanted_keys),
            "rows": int(len(control_prep)), "features": len(features),
        })
        print(f"      checkpointed control trajectories: keys={control_prep['control_key'].nunique():,}, rows={len(control_prep):,}")

    print("[5/7] Computing balance, parent-paired mechanism contrasts, and cross-tier survival...")
    balance = control_balance_report(assignments, event_states, match_feature_map)
    coverage = control_coverage_report(assignments, {k: list(v["event_id"]) for k,v in cohort_events.items()}, args.controls_per_event)
    fam = feature_family_map(phase7c, features)
    contrasts = paired_mechanism_contrasts(event_prep, control_prep, assignments, fam, match_feature_map)
    daily = daily_paired_consistency(event_prep, control_prep, assignments)
    survival = cross_tier_survival(contrasts, daily)

    print("[6/7] Checking structural invariants...")
    checks = check_invariants(events, candidates, event_states, assignments, control_prep, args)
    failed = checks[checks["status"] == "FAIL"]
    print(f"      invariants={len(checks):,}, failed={len(failed):,}")

    print("[7/7] Writing reports and metadata...")
    args.report_root.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(args.report_root / "large_move_matched_control_coverage.csv", index=False)
    balance.to_csv(args.report_root / "large_move_matched_control_balance.csv", index=False)
    contrasts.to_csv(args.report_root / "large_move_matched_mechanism_contrasts.csv", index=False)
    daily.to_csv(args.report_root / "large_move_matched_daily_consistency.csv", index=False)
    survival.to_csv(args.report_root / "large_move_matched_cross_tier_survival.csv", index=False)
    checks.to_csv(args.report_root / "large_move_matched_control_invariants.csv", index=False)

    summary = {
        "status": "PASS" if failed.empty else "FAIL",
        "builder_version": BUILDER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "symbol": args.symbol,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "candidate_controls": int(len(candidates)),
        "event_anchor_states": int(len(event_states)),
        "control_assignments": int(len(assignments)),
        "unique_profiled_controls": int(control_prep["control_key"].nunique()) if not control_prep.empty else 0,
        "control_preparation_rows": int(len(control_prep)),
        "features": int(len(features)),
        "failed_invariants": int(len(failed)),
        "cohort_counts": {name: int(len(df)) for name, df in cohort_events.items()},
        "matching_features": match_feature_map,
        "research_role": "RETROSPECTIVE_MATCHED_CONTROL_FALSIFICATION_NOT_PROSPECTIVE_ALPHA",
        "control_outcome_contract": f"NO_35BPS_EITHER_DIRECTION_WITHIN_{args.horizon_sec}S",
    }
    atomic_write_json(args.output_root / "metadata" / "latest_summary.json", summary)
    atomic_write_json(args.output_root / "metadata" / "run_config.json", {
        "builder_version": BUILDER_VERSION, "catalog_root": str(args.catalog_root), "atlas_root": str(args.atlas_root),
        "primitive_root": str(args.primitive_root), "flow_root": str(args.flow_root), "phase7c_script": str(args.phase7c_script),
        "negative_threshold_bps": args.negative_threshold_bps, "horizon_sec": args.horizon_sec,
        "control_cadence_sec": args.control_cadence_sec, "controls_per_event": args.controls_per_event,
        "control_reuse_cap": args.control_reuse_cap, "control_event_embargo_sec": args.control_event_embargo_sec,
        "cohorts": list(COHORT_TARGETS), "match_tiers": list(MATCH_TIERS), "matching_features": match_feature_map,
    })
    atomic_write_text(args.report_root / "large_move_matched_control_findings.md", build_findings(summary, coverage, balance, survival))

    print(f"Phase 7D status       : {summary['status']}")
    print(f"Candidate controls    : {summary['candidate_controls']:,}")
    print(f"Control assignments   : {summary['control_assignments']:,}")
    print(f"Unique controls       : {summary['unique_profiled_controls']:,}")
    print(f"Reports               : {args.report_root.resolve()}")
    if not failed.empty:
        print(failed.to_string(index=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
