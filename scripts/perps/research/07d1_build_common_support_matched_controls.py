#!/usr/bin/env python3
"""Phase 7D.1: common-support corrected matched controls for large-move anatomy.

Research role
-------------
Phase 7D v1.0.0 was structurally valid but failed its central balance objective:
large-event anchors remained much more volatile/active than their nominally
matched ordinary controls.  Phase 7D.1 preserves that run as an audit artifact
and asks a narrower, stricter question:

    Among large-move events for which genuinely comparable states exist, do
    bilateral near-touch thinness and dynamic liquidity behavior remain
    different before the hindsight physical extremum?

This script deliberately prefers dropout over poor matches.  It consumes the
frozen Phase-7D candidate-control checkpoint rather than rebuilding it, applies
prespecified per-feature common-support calipers, reports coverage-vs-balance
sensitivity, and never calls an empirical mechanism result interpretable unless
the corresponding match analysis passes an explicit aggregate balance gate.

Two retrospective control families are studied:

ORDINARY_LT35
    Ordinary causal anchors whose next 30 minutes remain strictly below 35 bps
    in BOTH directions.  These come from the frozen Phase-7D candidate pool.

THIRTYFIVE_ONLY
    Physical events that reach 35 bps within 30 minutes but do not reach 50 bps.
    These provide a closer continuation control: why does one real release stop
    near 35 bps while another reaches 50/75/100+ bps?  Because both sides are
    physical events, Phase-7C event-stage features can also be compared after
    matching the pre-extremum state.

Matching tiers remain frozen from Phase 7D:

REGIME_MATCHED
    Volatility/activity, recent direction-normalized return geometry and OI
    context when broadly available.

STRICT_STATE_MATCHED
    REGIME_MATCHED plus spread and direction-normalized supporting/opposing
    5-bps visible depth.

Three frozen caliper policies expose the common-support tradeoff instead of
silently tuning to the observed results:

TIGHT     max pairwise robust-z <= 0.50 and mean <= 0.30
PRIMARY   max pairwise robust-z <= 0.75 and mean <= 0.40
RELAXED   max pairwise robust-z <= 1.00 and mean <= 0.55

The robust-z denominator is the candidate-pool MAD scale used consistently
within each control-family/matching-tier analysis.  Calipers constrain every
match feature individually; aggregate parent-paired SMD is reported separately
because pairwise calipers do not guarantee aggregate balance after reuse.

Scientific interpretation gate
------------------------------
An analysis is tagged INTERPRETABLE_BALANCED only when:

* max absolute parent-paired SMD across matching variables <= 0.20, and
* at least 30 parent events have the required minimum number of controls.

Balance 0.20-0.30 is tagged MARGINAL_BALANCE and >0.30 POOR_BALANCE.  These are
research diagnostics, not p-value thresholds and not causal-identification
claims.

All event directions, physical extrema, continuation strata and control outcome
labels remain hindsight-defined.  Matching inputs themselves are causal at the
anchor.  This script is retrospective mechanism research, NOT a prospective
signal test and NOT evidence of a tradable edge.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

BUILDER_VERSION = "1.0.0"
SCHEMA_VERSION = 1
SCHEMA_NAME = "large_move_common_support_matched_controls_v1"
DEFAULT_SYMBOL = "BTCUSDT"
PRIMARY_HORIZON_SEC = 1800
NEGATIVE_THRESHOLD_BPS = 35.0
CONTROLS_PER_EVENT = 3
MIN_CONTROLS_PER_EVENT = 2
CONTROL_REUSE_CAP = 3
BALANCE_GOOD_MAX_ABS_SMD = 0.20
BALANCE_MARGINAL_MAX_ABS_SMD = 0.30
MIN_INTERPRETABLE_PARENTS = 30
EVENT_EMBARGO_SEC = 300

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()
DEFAULT_PHASE7D_SCRIPT = REPO_ROOT / "scripts" / "perps" / "research" / "07d_build_matched_non_big_move_controls.py"
DEFAULT_PHASE7C_SCRIPT = REPO_ROOT / "scripts" / "perps" / "research" / "07c_build_large_move_participant_behavior_atlas.py"
DEFAULT_PHASE7D_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / "large_move_matched_controls_v1"
DEFAULT_PHASE7C_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / "large_move_participant_behavior_atlas_v1"
DEFAULT_CATALOG_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / "large_move_event_catalog_v1"
DEFAULT_PRIMITIVE_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / "liquidity_event_primitives_v1"
DEFAULT_FLOW_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "flow_features_1s_v1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / SCHEMA_NAME
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "perps_large_move_common_support"

TARGET_COHORTS = (
    ("50_PLUS", 50),
    ("75_PLUS", 75),
    ("100_PLUS", 100),
)
CONTROL_FAMILIES = ("ORDINARY_LT35", "THIRTYFIVE_ONLY")
MATCH_TIERS = ("REGIME_MATCHED", "STRICT_STATE_MATCHED")


@dataclass(frozen=True)
class CaliperPolicy:
    name: str
    max_abs_robust_z: float
    mean_abs_robust_z: float


CALIPER_POLICIES = (
    CaliperPolicy("TIGHT", 0.50, 0.30),
    CaliperPolicy("PRIMARY", 0.75, 0.40),
    CaliperPolicy("RELAXED", 1.00, 0.55),
)
MAX_CALIPER = max(CALIPER_POLICIES, key=lambda x: x.max_abs_robust_z)


def require_numeric_stack():
    try:
        import numpy as np
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Phase 7D.1 requires numpy and pandas. Activate the project .venv and run:\n"
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
            "Phase 7D.1 Parquet I/O requires pyarrow. Activate the project .venv and run:\n"
            "  python -m pip install pyarrow"
        ) from exc
    return np, pd, pa, pq


def load_module(path: Path, module_name: str):
    if not path.exists():
        raise RuntimeError(f"Missing required frozen script: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import frozen script: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_phase_modules(phase7c_script: Path, phase7d_script: Path):
    phase7c = load_module(phase7c_script, "phase7c_frozen_for_7d1")
    phase7d = load_module(phase7d_script, "phase7d_frozen_for_7d1")
    if getattr(phase7c, "SCHEMA_NAME", None) != "large_move_participant_behavior_atlas_v1":
        raise RuntimeError("Unexpected Phase-7C schema contract")
    if getattr(phase7d, "SCHEMA_NAME", None) != "large_move_matched_controls_v1":
        raise RuntimeError("Unexpected Phase-7D schema contract")
    if getattr(phase7d, "BUILDER_VERSION", None) != "1.0.0":
        raise RuntimeError(
            f"Phase 7D.1 is frozen against Phase-7D builder 1.0.0; found {getattr(phase7d, 'BUILDER_VERSION', None)!r}"
        )
    return phase7c, phase7d


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7d1-", suffix=path.suffix, dir=str(path.parent))
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
    fd, tmp = tempfile.mkstemp(prefix="tmp7d1-", suffix=".parquet", dir=str(path.parent))
    os.close(fd)
    try:
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def safe_float(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return math.nan
    return x if math.isfinite(x) else math.nan


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


def pooled_robust_scale(a, b) -> float:
    np, pd = require_numeric_stack()
    aa = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float, copy=False)
    bb = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float, copy=False)
    x = np.concatenate([aa[np.isfinite(aa)], bb[np.isfinite(bb)]])
    if x.size < 2:
        return math.nan
    q25, q75 = np.quantile(x, [0.25, 0.75])
    rs = float((q75 - q25) / 1.349)
    if math.isfinite(rs) and rs > 1e-12:
        return rs
    sd = float(np.std(x, ddof=0))
    return sd if math.isfinite(sd) and sd > 1e-12 else math.nan


def standardized_mean_difference(a, b) -> float:
    np, pd = require_numeric_stack()
    aa = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float, copy=False)
    bb = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float, copy=False)
    aa = aa[np.isfinite(aa)]
    bb = bb[np.isfinite(bb)]
    if len(aa) < 2 or len(bb) < 2:
        return math.nan
    pooled = math.sqrt((float(np.var(aa, ddof=1)) + float(np.var(bb, ddof=1))) / 2.0)
    if not math.isfinite(pooled) or pooled <= 1e-12:
        return 0.0 if abs(float(np.mean(aa)) - float(np.mean(bb))) <= 1e-12 else math.nan
    return float((np.mean(aa) - np.mean(bb)) / pooled)


def cohort_mask(events, target_bps: int, horizon_sec: int):
    _, pd = require_numeric_stack()
    col = f"time_to_{target_bps}_sec"
    x = pd.to_numeric(events[col], errors="coerce")
    return x.notna() & (x <= int(horizon_sec))


def event_interval(row: Mapping[str, Any], embargo_sec: int = EVENT_EMBARGO_SEC) -> tuple[int, int]:
    start = int(row["onset_second"]) - int(embargo_sec)
    vals = [safe_float(row.get("reversal_confirm_second")), safe_float(row.get("extreme_second")), safe_float(row.get("hit_35_second"))]
    finite = [int(v) for v in vals if math.isfinite(v)]
    end = (max(finite) if finite else int(row["onset_second"])) + int(embargo_sec)
    return start, end


def intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def stable_checksum(values: Iterable[str]) -> str:
    h = hashlib.sha256()
    for v in sorted(str(x) for x in values):
        h.update(v.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def analysis_key(cohort: str, control_family: str, match_tier: str, caliper: str) -> str:
    return f"{cohort}|{control_family}|{match_tier}|{caliper}"


def balance_grade(max_abs_smd: float, n_parents: int) -> str:
    if not math.isfinite(max_abs_smd):
        return "NOT_EVALUABLE"
    if max_abs_smd <= BALANCE_GOOD_MAX_ABS_SMD:
        if int(n_parents) >= MIN_INTERPRETABLE_PARENTS:
            return "INTERPRETABLE_BALANCED"
        return "BALANCED_LIMITED_N"
    if max_abs_smd <= BALANCE_MARGINAL_MAX_ABS_SMD:
        return "MARGINAL_BALANCE"
    return "POOR_BALANCE"


def candidate_priority_groups(cand, first_feature: str):
    """Build deterministic first-feature-sorted indices for fast caliper lookup."""
    np, pd = require_numeric_stack()
    groups: dict[tuple[int, str, int], tuple[Any, Any]] = {}
    all_by_hour: dict[tuple[int, int], tuple[Any, Any]] = {}

    def build(idxs):
        idx = np.asarray(list(idxs), dtype=np.int64)
        vals = pd.to_numeric(cand.iloc[idx][first_feature], errors="coerce").to_numpy(float)
        good = np.isfinite(vals)
        idx = idx[good]
        vals = vals[good]
        order = np.argsort(vals, kind="mergesort")
        return idx[order], vals[order]

    for k, idxs in cand.groupby(["direction", "anchor_date", "utc_hour"], sort=False).groups.items():
        groups[(int(k[0]), str(k[1]), int(k[2]))] = build(idxs)
    for k, idxs in cand.groupby(["direction", "utc_hour"], sort=False).groups.items():
        all_by_hour[(int(k[0]), int(k[1]))] = build(idxs)
    return groups, all_by_hour


def _range_indices(sorted_pair, center: float, radius: float):
    np, _ = require_numeric_stack()
    if sorted_pair is None:
        return np.array([], dtype=np.int64)
    idxs, vals = sorted_pair
    lo = int(np.searchsorted(vals, center - radius, side="left"))
    hi = int(np.searchsorted(vals, center + radius, side="right"))
    return idxs[lo:hi]


def fallback_candidate_indices(ev: Mapping[str, Any], groups, all_by_hour, first_scale: float, first_feature: str):
    np, _ = require_numeric_stack()
    d = int(ev["direction"])
    day = str(ev["anchor_date"])
    h = int(ev["utc_hour"])
    center = safe_float(ev[first_feature])
    radius = MAX_CALIPER.max_abs_robust_z * float(first_scale)
    same = _range_indices(groups.get((d, day, h)), center, radius)
    adjacent_parts = []
    for hh in (h - 1, h + 1):
        if 0 <= hh <= 23:
            x = _range_indices(groups.get((d, day, hh)), center, radius)
            if len(x):
                adjacent_parts.append(x)
    adjacent = np.unique(np.concatenate(adjacent_parts)) if adjacent_parts else np.array([], dtype=np.int64)
    any_same = _range_indices(all_by_hour.get((d, h)), center, radius)
    return (
        ("SAME_DATE_SAME_HOUR", 0, same),
        ("SAME_DATE_ADJACENT_HOUR", 1, adjacent),
        ("ANY_DATE_SAME_HOUR", 2, any_same),
    )


def build_feasible_cache(events, candidates, match_features: Sequence[str], control_family: str, parent_intervals: Mapping[str, tuple[int, int]], control_intervals: Mapping[str, tuple[int, int]]):
    """Compute candidate pair distances once using the most relaxed frozen caliper.

    Returned options remain grouped by fallback rank.  Tighter policies simply
    filter this deterministic cache; empirical outcomes never tune the caliper.
    """
    np, pd = require_numeric_stack()
    cand = candidates.copy().reset_index(drop=True)
    cand = cand.dropna(subset=list(match_features)).reset_index(drop=True)
    scales = {f: robust_scale(cand[f]) for f in match_features}
    first = match_features[0]
    groups, all_by_hour = candidate_priority_groups(cand, first)
    feature_arrays = {f: pd.to_numeric(cand[f], errors="coerce").to_numpy(float) for f in match_features}
    sec_arr = pd.to_numeric(cand["receive_second"], errors="coerce").to_numpy(np.int64)
    control_event_ids = cand["control_event_id"].astype(object).to_numpy() if "control_event_id" in cand.columns else np.array([None] * len(cand), dtype=object)

    cache: dict[str, list[dict[str, Any]]] = {}
    diagnostics: list[dict[str, Any]] = []
    for ev in events.sort_values(["anchor_second", "event_id"], kind="mergesort").to_dict("records"):
        eid = str(ev["event_id"])
        if any(not math.isfinite(safe_float(ev.get(f))) for f in match_features):
            cache[eid] = []
            diagnostics.append({"event_id": eid, "feasible_relaxed": 0, "reason": "MISSING_EVENT_MATCH_STATE"})
            continue
        ev_interval = parent_intervals.get(eid)
        seen: set[int] = set()
        options: list[dict[str, Any]] = []
        for fallback, rank, idxs in fallback_candidate_indices(ev, groups, all_by_hour, scales[first], first):
            if len(idxs) == 0:
                continue
            idxs = np.asarray([int(i) for i in idxs if int(i) not in seen], dtype=np.int64)
            if len(idxs) == 0:
                continue
            for i in idxs:
                seen.add(int(i))
            # Never pair overlapping physical-event windows in the 35_ONLY family.
            if control_family == "THIRTYFIVE_ONLY" and ev_interval is not None:
                keep = []
                for i in idxs:
                    ceid = str(control_event_ids[int(i)])
                    ci = control_intervals.get(ceid)
                    keep.append(ci is not None and not intervals_overlap(ev_interval, ci))
                idxs = idxs[np.asarray(keep, dtype=bool)]
                if len(idxs) == 0:
                    continue
            zmat = []
            valid = np.ones(len(idxs), dtype=bool)
            for f in match_features:
                vals = feature_arrays[f][idxs]
                valid &= np.isfinite(vals)
                zmat.append(np.abs(vals - safe_float(ev[f])) / scales[f])
            if not zmat:
                continue
            z = np.column_stack(zmat)
            valid &= np.all(np.isfinite(z), axis=1)
            maxz = np.max(z, axis=1)
            meanz = np.mean(z, axis=1)
            valid &= maxz <= MAX_CALIPER.max_abs_robust_z + 1e-12
            valid &= meanz <= MAX_CALIPER.mean_abs_robust_z + 1e-12
            for pos in np.flatnonzero(valid):
                i = int(idxs[pos])
                row = {
                    "candidate_index": i,
                    "fallback": fallback,
                    "fallback_rank": int(rank),
                    "max_abs_robust_z": float(maxz[pos]),
                    "mean_abs_robust_z": float(meanz[pos]),
                    "temporal_tiebreak": abs(int(sec_arr[i]) - int(ev["anchor_second"])) / (86400.0 * 10000.0),
                }
                for j, f in enumerate(match_features):
                    row[f"z__{f}"] = float(z[pos, j])
                options.append(row)
        options.sort(key=lambda r: (r["fallback_rank"], r["mean_abs_robust_z"], r["max_abs_robust_z"], r["temporal_tiebreak"], r["candidate_index"]))
        cache[eid] = options
        diagnostics.append({"event_id": eid, "feasible_relaxed": len(options), "reason": "OK" if options else "NO_COMMON_SUPPORT_RELAXED"})
    return cand, scales, cache, pd.DataFrame(diagnostics)


def match_from_cache(events, cand, cache: Mapping[str, list[dict[str, Any]]], policy: CaliperPolicy, match_features: Sequence[str], cohort: str, control_family: str, match_tier: str, controls_per_event: int, min_controls: int, reuse_cap: int):
    np, pd = require_numeric_stack()
    # Scarce-support parents first reduces deterministic greedy loss from reuse.
    event_records = []
    for ev in events.to_dict("records"):
        opts = [o for o in cache.get(str(ev["event_id"]), []) if o["max_abs_robust_z"] <= policy.max_abs_robust_z + 1e-12 and o["mean_abs_robust_z"] <= policy.mean_abs_robust_z + 1e-12]
        event_records.append((len(opts), int(ev["anchor_second"]), str(ev["event_id"]), ev, opts))
    event_records.sort(key=lambda x: (x[0], x[1], x[2]))

    usage = np.zeros(len(cand), dtype=np.int16)
    assignments: list[dict[str, Any]] = []
    parent_rows: list[dict[str, Any]] = []
    for feasible_count, _sec, eid, ev, opts in event_records:
        selected: list[dict[str, Any]] = []
        selected_idx: set[int] = set()
        for fallback_rank in (0, 1, 2):
            if len(selected) >= controls_per_event:
                break
            bucket = [o for o in opts if o["fallback_rank"] == fallback_rank]
            for o in bucket:
                if len(selected) >= controls_per_event:
                    break
                idx = int(o["candidate_index"])
                if idx in selected_idx or int(usage[idx]) >= int(reuse_cap):
                    continue
                usage[idx] += 1
                selected_idx.add(idx)
                selected.append(o)
        nsel = len(selected)
        accepted = nsel >= min_controls
        if not accepted:
            # A parent that fails the minimum-control gate is not part of the
            # matched sample, so its tentative controls must not consume reuse
            # capacity that could support another parent.
            for o in selected:
                usage[int(o["candidate_index"])] -= 1
        if feasible_count == 0:
            status = "NO_COMMON_SUPPORT"
        elif nsel == 0:
            status = "REUSE_EXHAUSTED"
        elif nsel < min_controls:
            status = "INSUFFICIENT_CONTROLS"
        elif nsel < controls_per_event:
            status = "ACCEPTED_PARTIAL"
        else:
            status = "ACCEPTED_FULL"
        parent_rows.append({
            "analysis_id": analysis_key(cohort, control_family, match_tier, policy.name),
            "cohort": cohort,
            "control_family": control_family,
            "match_tier": match_tier,
            "caliper_policy": policy.name,
            "parent_event_id": eid,
            "parent_event_date": str(ev["anchor_date"]),
            "direction": int(ev["direction"]),
            "feasible_candidates_pre_reuse": int(feasible_count),
            "attempted_controls_before_min_gate": int(nsel),
            "selected_controls": int(nsel if accepted else 0),
            "accepted_for_contrasts": bool(accepted),
            "parent_match_status": status,
        })
        if not accepted:
            continue
        for ordinal, o in enumerate(selected, 1):
            c = cand.iloc[int(o["candidate_index"])]
            row = {
                "analysis_id": analysis_key(cohort, control_family, match_tier, policy.name),
                "cohort": cohort,
                "control_family": control_family,
                "match_tier": match_tier,
                "caliper_policy": policy.name,
                "parent_event_id": eid,
                "parent_event_date": str(ev["anchor_date"]),
                "direction": int(ev["direction"]),
                "control_ordinal": int(ordinal),
                "control_key": str(c["control_key"]),
                "control_event_id": None if pd.isna(c.get("control_event_id")) else str(c.get("control_event_id")),
                "control_second": int(c["receive_second"]),
                "control_date": str(c["anchor_date"]),
                "control_utc_hour": int(c["utc_hour"]),
                "match_fallback": str(o["fallback"]),
                "max_abs_robust_z": float(o["max_abs_robust_z"]),
                "mean_abs_robust_z": float(o["mean_abs_robust_z"]),
                "accepted_parent": bool(nsel >= min_controls),
            }
            if "forward_up_bps_30m" in c.index:
                row["control_forward_up_bps_30m"] = safe_float(c.get("forward_up_bps_30m"))
                row["control_forward_down_bps_30m"] = safe_float(c.get("forward_down_bps_30m"))
            for f in match_features:
                row[f"event__{f}"] = safe_float(ev.get(f))
                row[f"control__{f}"] = safe_float(c.get(f))
                row[f"z__{f}"] = float(o[f"z__{f}"])
            assignments.append(row)
    return pd.DataFrame(assignments), pd.DataFrame(parent_rows)


def build_control_pools(candidates, event_states, event_meta):
    _, pd = require_numeric_stack()
    ordinary = candidates.copy()
    ordinary["control_family"] = "ORDINARY_LT35"
    ordinary["control_key"] = ordinary.apply(lambda r: f"{int(r['direction']):+d}:{int(r['receive_second'])}", axis=1)
    ordinary["control_event_id"] = None

    states = event_states.merge(event_meta[["event_id", "continuation_stratum"]], on="event_id", how="left", validate="one_to_one")
    thirtyfive = states[states["continuation_stratum"] == "35_ONLY"].copy()
    thirtyfive = thirtyfive[thirtyfive["book_valid"].fillna(False).astype(bool) & thirtyfive["flow_feature_context_ready"].fillna(False).astype(bool)].copy()
    thirtyfive = thirtyfive.rename(columns={"anchor_second": "receive_second"})
    thirtyfive["control_family"] = "THIRTYFIVE_ONLY"
    thirtyfive["control_event_id"] = thirtyfive["event_id"].astype(str)
    thirtyfive["control_key"] = "EVENT:" + thirtyfive["event_id"].astype(str)
    # Future columns are intentionally absent: the family is defined by the frozen continuation stratum.
    return ordinary.reset_index(drop=True), thirtyfive.reset_index(drop=True), states


def build_parent_balance(assignments, parent_status, match_feature_map: Mapping[str, Sequence[str]], controls_per_event: int):
    np, pd = require_numeric_stack()
    rows = []
    analysis_rows = []
    if parent_status.empty:
        return pd.DataFrame(), pd.DataFrame()
    accepted = assignments.copy() if not assignments.empty else pd.DataFrame()
    for aid, ps in parent_status.groupby("analysis_id", sort=True):
        pmeta = ps.iloc[0]
        ag = accepted[accepted["analysis_id"] == aid].copy() if not accepted.empty else pd.DataFrame()
        feats = list(match_feature_map[str(pmeta["match_tier"])])
        max_abs_smd = math.nan
        if not ag.empty:
            per_parent = []
            for pid, pg in ag.groupby("parent_event_id", sort=False):
                rec = {"parent_event_id": pid}
                for f in feats:
                    evv = safe_float(pg[f"event__{f}"].iloc[0])
                    cv = pd.to_numeric(pg[f"control__{f}"], errors="coerce")
                    rec[f"event__{f}"] = evv
                    rec[f"control__{f}"] = float(cv.mean()) if cv.notna().any() else math.nan
                per_parent.append(rec)
            pp = pd.DataFrame(per_parent)
            observed_abs_smd = []
            for f in feats:
                evv = pd.to_numeric(pp[f"event__{f}"], errors="coerce")
                cvv = pd.to_numeric(pp[f"control__{f}"], errors="coerce")
                ok = evv.notna() & cvv.notna()
                a = evv[ok]
                b = cvv[ok]
                smd = standardized_mean_difference(a, b)
                rs = pooled_robust_scale(a, b)
                med_delta = float(np.median(a.to_numpy(float) - b.to_numpy(float))) if len(a) else math.nan
                rmd = med_delta / rs if math.isfinite(rs) and rs > 1e-12 else math.nan
                if math.isfinite(smd):
                    observed_abs_smd.append(abs(smd))
                rows.append({
                    "analysis_id": aid,
                    "cohort": pmeta["cohort"],
                    "control_family": pmeta["control_family"],
                    "match_tier": pmeta["match_tier"],
                    "caliper_policy": pmeta["caliper_policy"],
                    "feature": f,
                    "parent_pairs": int(len(a)),
                    "event_mean": float(a.mean()) if len(a) else math.nan,
                    "control_mean": float(b.mean()) if len(b) else math.nan,
                    "event_median": float(a.median()) if len(a) else math.nan,
                    "control_median": float(b.median()) if len(b) else math.nan,
                    "standardized_mean_difference": smd,
                    "median_paired_difference": med_delta,
                    "paired_robust_standardized_difference": rmd,
                })
            max_abs_smd = max(observed_abs_smd) if observed_abs_smd else math.nan
        eligible = int(len(ps))
        accepted_parents = int(ps["accepted_for_contrasts"].fillna(False).sum())
        full = int((ps["selected_controls"] >= controls_per_event).sum())
        analysis_rows.append({
            "analysis_id": aid,
            "cohort": pmeta["cohort"],
            "control_family": pmeta["control_family"],
            "match_tier": pmeta["match_tier"],
            "caliper_policy": pmeta["caliper_policy"],
            "eligible_parents": eligible,
            "accepted_parents": accepted_parents,
            "accepted_parent_share": accepted_parents / eligible if eligible else math.nan,
            "full_requested_control_parents": full,
            "full_requested_control_share": full / eligible if eligible else math.nan,
            "controls": int(len(ag)),
            "unique_controls": int(ag["control_key"].nunique()) if not ag.empty else 0,
            "max_abs_smd": max_abs_smd,
            "balance_grade": balance_grade(max_abs_smd, accepted_parents),
            "median_pair_max_abs_robust_z": float(pd.to_numeric(ag["max_abs_robust_z"], errors="coerce").median()) if not ag.empty else math.nan,
            "p95_pair_max_abs_robust_z": float(pd.to_numeric(ag["max_abs_robust_z"], errors="coerce").quantile(0.95)) if not ag.empty else math.nan,
            "same_date_same_hour_share": float((ag["match_fallback"] == "SAME_DATE_SAME_HOUR").mean()) if not ag.empty else math.nan,
            "same_date_adjacent_hour_share": float((ag["match_fallback"] == "SAME_DATE_ADJACENT_HOUR").mean()) if not ag.empty else math.nan,
            "any_date_same_hour_share": float((ag["match_fallback"] == "ANY_DATE_SAME_HOUR").mean()) if not ag.empty else math.nan,
            "max_observed_reuse": int(ag["control_key"].value_counts().max()) if not ag.empty else 0,
        })
    return pd.DataFrame(rows), pd.DataFrame(analysis_rows)


def prepare_ordinary_control_profiles(assignments, phase7d, phase7c, primitive_root: Path, flow_root: Path, symbol: str, start: date, end: date, features: Sequence[str], output_root: Path, force: bool):
    _, pd, _, pq = require_stack()
    accepted = assignments[(assignments["control_family"] == "ORDINARY_LT35") & assignments["accepted_parent"].fillna(False).astype(bool)].copy()
    unique = accepted[["control_key", "control_second", "control_date", "direction"]].drop_duplicates() if not accepted.empty else pd.DataFrame()
    path = output_root / "ordinary_control_preparation_features.parquet"
    meta_path = output_root / "metadata" / "ordinary_control_profile_checkpoint.json"
    keys = sorted(unique["control_key"].astype(str).tolist()) if not unique.empty else []
    checksum = stable_checksum(keys)
    if path.exists() and meta_path.exists() and not force:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("builder_version") == BUILDER_VERSION and meta.get("control_key_checksum") == checksum and int(meta.get("unique_controls")) == len(keys):
                return pq.read_table(path).to_pandas(), True
        except Exception:
            pass
    if unique.empty:
        out = pd.DataFrame()
    else:
        out = phase7d.profile_controls(unique, phase7c, primitive_root, flow_root, symbol, start, end, features)
    write_parquet_atomic(out, path)
    atomic_write_json(meta_path, {
        "builder_version": BUILDER_VERSION,
        "unique_controls": len(keys),
        "control_key_checksum": checksum,
        "rows": int(len(out)),
        "features": int(len(features)),
    })
    return out, False


def parent_pair_rows_preparation(analysis_assignments, event_prep, ordinary_prep):
    np, pd = require_numeric_stack()
    if analysis_assignments.empty:
        return pd.DataFrame()
    ag = analysis_assignments[analysis_assignments["accepted_parent"].fillna(False).astype(bool)].copy()
    if ag.empty:
        return pd.DataFrame()
    family = str(ag["control_family"].iloc[0])
    event_vals = event_prep[["event_id", "event_date", "direction", "window_label", "feature", "median"]].rename(columns={"median": "event_value"})
    parent_ids = set(ag["parent_event_id"].astype(str))
    event_vals = event_vals[event_vals["event_id"].astype(str).isin(parent_ids)].copy()
    if family == "ORDINARY_LT35":
        if ordinary_prep.empty:
            return pd.DataFrame()
        cp = ordinary_prep[["control_key", "window_label", "feature", "median"]].rename(columns={"median": "control_value"})
        x = ag[["parent_event_id", "control_key"]].merge(cp, on="control_key", how="left")
    else:
        cp = event_prep[["event_id", "window_label", "feature", "median"]].rename(columns={"event_id": "control_event_id", "median": "control_value"})
        x = ag[["parent_event_id", "control_event_id"]].merge(cp, on="control_event_id", how="left")
    control_mean = x.groupby(["parent_event_id", "window_label", "feature"], sort=False, as_index=False)["control_value"].mean()
    out = event_vals.rename(columns={"event_id": "parent_event_id", "event_date": "parent_event_date"}).merge(control_mean, on=["parent_event_id", "window_label", "feature"], how="inner")
    out["paired_difference"] = pd.to_numeric(out["event_value"], errors="coerce") - pd.to_numeric(out["control_value"], errors="coerce")
    return out


def parent_pair_rows_stage_35only(analysis_assignments, event_stage):
    _, pd = require_numeric_stack()
    if analysis_assignments.empty or str(analysis_assignments["control_family"].iloc[0]) != "THIRTYFIVE_ONLY":
        return pd.DataFrame()
    ag = analysis_assignments[analysis_assignments["accepted_parent"].fillna(False).astype(bool)].copy()
    if ag.empty:
        return pd.DataFrame()
    event_vals = event_stage[["event_id", "event_date", "direction", "stage", "feature", "value"]].rename(columns={"value": "event_value"})
    parent_ids = set(ag["parent_event_id"].astype(str))
    event_vals = event_vals[event_vals["event_id"].astype(str).isin(parent_ids)].copy()
    cp = event_stage[["event_id", "stage", "feature", "value"]].rename(columns={"event_id": "control_event_id", "value": "control_value"})
    x = ag[["parent_event_id", "control_event_id"]].merge(cp, on="control_event_id", how="left")
    control_mean = x.groupby(["parent_event_id", "stage", "feature"], sort=False, as_index=False)["control_value"].mean()
    out = event_vals.rename(columns={"event_id": "parent_event_id", "event_date": "parent_event_date"}).merge(control_mean, on=["parent_event_id", "stage", "feature"], how="inner")
    out["paired_difference"] = pd.to_numeric(out["event_value"], errors="coerce") - pd.to_numeric(out["control_value"], errors="coerce")
    return out


def summarize_pair_rows(pair_rows, location_col: str, feature_family: Mapping[str, str], analysis_meta: Mapping[str, Any]):
    np, pd = require_numeric_stack()
    rows = []
    if pair_rows.empty:
        return pd.DataFrame()
    for (loc, feat), g in pair_rows.groupby([location_col, "feature"], sort=True):
        ev = pd.to_numeric(g["event_value"], errors="coerce")
        co = pd.to_numeric(g["control_value"], errors="coerce")
        ok = ev.notna() & co.notna()
        ev = ev[ok]
        co = co[ok]
        if len(ev) == 0:
            continue
        diff = ev.to_numpy(float) - co.to_numpy(float)
        rs = pooled_robust_scale(ev, co)
        rmd = float(np.median(diff) / rs) if math.isfinite(rs) and rs > 1e-12 else math.nan
        row = dict(analysis_meta)
        row.update({
            location_col: loc,
            "feature": feat,
            "feature_family": feature_family.get(str(feat), "OTHER"),
            "parent_pairs": int(len(ev)),
            "event_median": float(ev.median()),
            "control_median": float(co.median()),
            "median_paired_difference": float(np.median(diff)),
            "paired_robust_standardized_difference": rmd,
            "paired_positive_share": float(np.mean(diff > 0)),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def daily_consistency(pair_rows, location_col: str, analysis_meta: Mapping[str, Any]):
    np, pd = require_numeric_stack()
    rows = []
    if pair_rows.empty:
        return pd.DataFrame()
    for (loc, feat), g in pair_rows.groupby([location_col, "feature"], sort=True):
        daily = []
        for day, dg in g.groupby("parent_event_date", sort=True):
            d = pd.to_numeric(dg["paired_difference"], errors="coerce").dropna()
            if len(d) < 2:
                continue
            daily.append((str(day), float(d.median())))
        if not daily:
            continue
        overall = pd.to_numeric(g["paired_difference"], errors="coerce").dropna()
        overall_med = float(overall.median()) if len(overall) else math.nan
        sign = 1 if overall_med > 0 else (-1 if overall_med < 0 else 0)
        vals = np.asarray([x[1] for x in daily], dtype=float)
        same = float(np.mean(np.sign(vals) == sign)) if sign != 0 else math.nan
        row = dict(analysis_meta)
        row.update({
            location_col: loc,
            "feature": feat,
            "days_comparable": int(len(daily)),
            "overall_parent_median_difference": overall_med,
            "same_sign_day_share": same,
            "first_day": daily[0][0],
            "last_day": daily[-1][0],
        })
        rows.append(row)
    return pd.DataFrame(rows)


def subset_robustness(pair_rows, location_col: str, analysis_meta: Mapping[str, Any], start: date, end: date):
    np, pd = require_numeric_stack()
    rows = []
    if pair_rows.empty:
        return pd.DataFrame()
    midpoint = start + timedelta(days=(end - start).days // 2)
    work = pair_rows.copy()
    work["sample_half"] = work["parent_event_date"].astype(str).map(lambda x: "FIRST_HALF" if date.fromisoformat(x) <= midpoint else "SECOND_HALF")
    for subset_type, col in (("DIRECTION", "direction"), ("SAMPLE_HALF", "sample_half")):
        for subset_value, sg in work.groupby(col, sort=True):
            for (loc, feat), g in sg.groupby([location_col, "feature"], sort=True):
                ev = pd.to_numeric(g["event_value"], errors="coerce")
                co = pd.to_numeric(g["control_value"], errors="coerce")
                ok = ev.notna() & co.notna()
                ev = ev[ok]
                co = co[ok]
                if len(ev) == 0:
                    continue
                diff = ev.to_numpy(float) - co.to_numpy(float)
                rs = pooled_robust_scale(ev, co)
                row = dict(analysis_meta)
                row.update({
                    "subset_type": subset_type,
                    "subset_value": "UP" if subset_type == "DIRECTION" and int(subset_value) > 0 else ("DOWN" if subset_type == "DIRECTION" else str(subset_value)),
                    location_col: loc,
                    "feature": feat,
                    "parent_pairs": int(len(ev)),
                    "median_paired_difference": float(np.median(diff)),
                    "paired_robust_standardized_difference": float(np.median(diff) / rs) if math.isfinite(rs) and rs > 1e-12 else math.nan,
                })
                rows.append(row)
    return pd.DataFrame(rows)


def build_interpretable_screen(mechanism, daily, analysis_quality, location_col: str):
    np, pd = require_numeric_stack()
    if mechanism.empty:
        return pd.DataFrame()
    q = analysis_quality[["analysis_id", "balance_grade", "max_abs_smd", "accepted_parents"]].copy()
    out = mechanism.merge(q, on="analysis_id", how="left", suffixes=("", "__quality"))
    if not daily.empty:
        key = ["analysis_id", location_col, "feature"]
        out = out.merge(daily[key + ["days_comparable", "same_sign_day_share"]], on=key, how="left")
    rsd = pd.to_numeric(out["paired_robust_standardized_difference"], errors="coerce")
    day = pd.to_numeric(out.get("same_sign_day_share"), errors="coerce") if "same_sign_day_share" in out.columns else pd.Series(np.nan, index=out.index)
    out["descriptive_residual_screen"] = np.where(
        (out["balance_grade"] == "INTERPRETABLE_BALANCED") & (rsd.abs() >= 0.20) & ((day.isna()) | (day >= 0.65)),
        "BALANCED_DESCRIPTIVE_RESIDUAL",
        np.where(out["balance_grade"] == "INTERPRETABLE_BALANCED", "BALANCED_WEAK_OR_INCONSISTENT", "DO_NOT_INTERPRET_BALANCE"),
    )
    return out.sort_values(["descriptive_residual_screen", "analysis_id", location_col, "feature"], kind="mergesort")


def build_coverage(parent_status, controls_per_event: int):
    _, pd = require_numeric_stack()
    rows = []
    if parent_status.empty:
        return pd.DataFrame()
    for aid, g in parent_status.groupby("analysis_id", sort=True):
        r = g.iloc[0]
        n = len(g)
        rows.append({
            "analysis_id": aid,
            "cohort": r["cohort"],
            "control_family": r["control_family"],
            "match_tier": r["match_tier"],
            "caliper_policy": r["caliper_policy"],
            "eligible_parents": int(n),
            "parents_with_any_feasible_candidate": int((g["feasible_candidates_pre_reuse"] > 0).sum()),
            "common_support_share_pre_reuse": float((g["feasible_candidates_pre_reuse"] > 0).mean()) if n else math.nan,
            "accepted_parents": int(g["accepted_for_contrasts"].fillna(False).sum()),
            "accepted_parent_share": float(g["accepted_for_contrasts"].fillna(False).mean()) if n else math.nan,
            "full_requested_control_parents": int((g["selected_controls"] >= controls_per_event).sum()),
            "full_requested_control_share": float((g["selected_controls"] >= controls_per_event).mean()) if n else math.nan,
            "median_feasible_candidates_pre_reuse": float(pd.to_numeric(g["feasible_candidates_pre_reuse"], errors="coerce").median()),
            "zero_support_parents": int((g["parent_match_status"] == "NO_COMMON_SUPPORT").sum()),
            "reuse_exhausted_parents": int((g["parent_match_status"] == "REUSE_EXHAUSTED").sum()),
            "insufficient_control_parents": int((g["parent_match_status"] == "INSUFFICIENT_CONTROLS").sum()),
        })
    return pd.DataFrame(rows)


def check_invariants(candidates, event_meta, assignments, parent_status, ordinary_profiles, match_feature_map: Mapping[str, Sequence[str]], phase7d, args):
    np, pd = require_numeric_stack()
    rows = []

    def add(name: str, ok: bool, observed: Any, expected: Any):
        rows.append({"check": name, "status": "PASS" if ok else "FAIL", "observed": observed, "expected": expected})

    if not candidates.empty:
        max_future = float(candidates[["forward_up_bps_30m", "forward_down_bps_30m"]].max().max())
        add("frozen_ordinary_candidates_below_35bps_both_directions", max_future < args.negative_threshold_bps + 1e-9, max_future, f"<{args.negative_threshold_bps}")
        contamination = phase7d.build_event_contamination_intervals(event_meta, args.event_embargo_sec)
        contaminated = phase7d.interval_mask(candidates["receive_second"].to_numpy(np.int64), contamination)
        add("frozen_ordinary_candidates_outside_large_event_embargo", not bool(contaminated.any()), int(contaminated.sum()), 0)
    if not assignments.empty:
        add("no_duplicate_parent_analysis_control", not assignments.duplicated(["analysis_id", "parent_event_id", "control_key"]).any(), int(assignments.duplicated(["analysis_id", "parent_event_id", "control_key"]).sum()), 0)
        max_reuse = int(assignments.groupby(["analysis_id", "control_key"]).size().max())
        add("reuse_cap_respected_within_analysis", max_reuse <= args.control_reuse_cap, max_reuse, f"<={args.control_reuse_cap}")
        accepted = assignments[assignments["accepted_parent"].fillna(False).astype(bool)]
        if not accepted.empty:
            parent_counts = accepted.groupby(["analysis_id", "parent_event_id"]).size()
            add("accepted_parents_have_minimum_controls", bool((parent_counts >= args.min_controls_per_event).all()), int(parent_counts.min()), f">={args.min_controls_per_event}")
        ordinary = assignments[assignments["control_family"] == "ORDINARY_LT35"]
        if not ordinary.empty:
            ok = (pd.to_numeric(ordinary["control_forward_up_bps_30m"], errors="coerce") < args.negative_threshold_bps).all() and (pd.to_numeric(ordinary["control_forward_down_bps_30m"], errors="coerce") < args.negative_threshold_bps).all()
            add("assigned_ordinary_controls_below_threshold", bool(ok), None, True)
        thirtyfive_ids = set(event_meta[event_meta["continuation_stratum"] == "35_ONLY"]["event_id"].astype(str))
        evc = assignments[assignments["control_family"] == "THIRTYFIVE_ONLY"]
        if not evc.empty:
            observed = set(evc["control_event_id"].dropna().astype(str))
            add("assigned_event_controls_are_35_only", observed.issubset(thirtyfive_ids), len(observed - thirtyfive_ids), 0)
            intervals = {str(r["event_id"]): event_interval(r, args.event_embargo_sec) for r in event_meta.to_dict("records")}
            overlap_count = 0
            for r in evc.to_dict("records"):
                a = intervals.get(str(r["parent_event_id"]))
                b = intervals.get(str(r["control_event_id"]))
                if a is not None and b is not None and intervals_overlap(a, b):
                    overlap_count += 1
            add("35_only_control_event_windows_do_not_overlap_parent", overlap_count == 0, overlap_count, 0)
        for policy in CALIPER_POLICIES:
            sub = assignments[assignments["caliper_policy"] == policy.name]
            if sub.empty:
                continue
            maxz = float(pd.to_numeric(sub["max_abs_robust_z"], errors="coerce").max())
            meanz = float(pd.to_numeric(sub["mean_abs_robust_z"], errors="coerce").max())
            add(f"{policy.name.lower()}_max_feature_caliper_respected", maxz <= policy.max_abs_robust_z + 1e-9, maxz, f"<={policy.max_abs_robust_z}")
            add(f"{policy.name.lower()}_mean_feature_caliper_respected", meanz <= policy.mean_abs_robust_z + 1e-9, meanz, f"<={policy.mean_abs_robust_z}")
    if not parent_status.empty:
        add("parent_status_unique", not parent_status.duplicated(["analysis_id", "parent_event_id"]).any(), int(parent_status.duplicated(["analysis_id", "parent_event_id"]).sum()), 0)
    if not ordinary_profiles.empty:
        add("ordinary_control_profiles_unique", not ordinary_profiles.duplicated(["control_key", "window_label", "feature"]).any(), int(ordinary_profiles.duplicated(["control_key", "window_label", "feature"]).sum()), 0)
    return pd.DataFrame(rows)


def build_findings(summary: Mapping[str, Any], quality, coverage, prep_screen, stage_screen) -> str:
    lines = [
        "# Phase 7D.1 common-support matched-control findings",
        "",
        f"- Builder version: `{summary['builder_version']}`",
        f"- Status: **{summary['status']}**",
        f"- Research role: `{summary['research_role']}`",
        f"- Frozen Phase-7D source version: `{summary['phase7d_builder_version']}`",
        f"- Ordinary candidate anchors reused: **{summary['ordinary_candidate_controls']:,}**",
        f"- 35_ONLY event-control candidates: **{summary['thirtyfive_only_candidates']:,}**",
        f"- Total accepted control assignments across sensitivity analyses: **{summary['control_assignments']:,}**",
        "",
        "## Interpretation rule",
        "",
        "This report intentionally treats low coverage as a valid empirical result. A mechanism row is eligible for interpretation only when its matching analysis is tagged `INTERPRETABLE_BALANCED`; all other residuals remain diagnostics rather than evidence that a Phase-7C mechanism survived matching.",
        "",
        "## Coverage and balance",
        "",
    ]
    if quality.empty:
        lines.append("- No matched analyses were available.")
    else:
        q = quality.sort_values(["cohort", "control_family", "match_tier", "caliper_policy"], kind="mergesort")
        for r in q.to_dict("records"):
            lines.append(
                f"- {r['cohort']} / {r['control_family']} / {r['match_tier']} / {r['caliper_policy']}: "
                f"accepted parents={int(r['accepted_parents'])}/{int(r['eligible_parents'])} "
                f"({100*safe_float(r['accepted_parent_share']):.1f}%), max |SMD|={safe_float(r['max_abs_smd']):.3f}, "
                f"grade=`{r['balance_grade']}`."
            )
    lines += ["", "## Balanced descriptive preparation residuals", ""]
    if prep_screen.empty:
        lines.append("- None available.")
    else:
        good = prep_screen[prep_screen["descriptive_residual_screen"] == "BALANCED_DESCRIPTIVE_RESIDUAL"].copy()
        if good.empty:
            lines.append("- No preparation feature/window passed both the aggregate balance gate and the descriptive residual screen.")
        else:
            good["abs_rsd"] = good["paired_robust_standardized_difference"].abs()
            for r in good.sort_values("abs_rsd", ascending=False).head(20).to_dict("records"):
                lines.append(
                    f"- {r['analysis_id']} / {r['window_label']} / `{r['feature']}`: "
                    f"paired RSD={safe_float(r['paired_robust_standardized_difference']):+.3f}, "
                    f"days={int(r['days_comparable']) if math.isfinite(safe_float(r.get('days_comparable'))) else 0}, "
                    f"same-sign-day share={safe_float(r.get('same_sign_day_share')):.3f}."
                )
    lines += ["", "## Balanced 35_ONLY event-stage residuals", ""]
    if stage_screen.empty:
        lines.append("- None available.")
    else:
        good = stage_screen[stage_screen["descriptive_residual_screen"] == "BALANCED_DESCRIPTIVE_RESIDUAL"].copy()
        if good.empty:
            lines.append("- No event-stage feature passed both the aggregate balance gate and the descriptive residual screen.")
        else:
            good["abs_rsd"] = good["paired_robust_standardized_difference"].abs()
            for r in good.sort_values("abs_rsd", ascending=False).head(20).to_dict("records"):
                lines.append(
                    f"- {r['analysis_id']} / {r['stage']} / `{r['feature']}`: "
                    f"paired RSD={safe_float(r['paired_robust_standardized_difference']):+.3f}, "
                    f"days={int(r['days_comparable']) if math.isfinite(safe_float(r.get('days_comparable'))) else 0}."
                )
    lines += [
        "",
        "## Scientific contract",
        "",
        "`INTERPRETABLE_BALANCED` means only that the observed matching variables are well balanced under this retrospective design and sample-size floor. It does not remove unobserved confounding, make hindsight event direction/extrema prospective, establish a detector, or establish a tradable edge. The next phase must be chosen from the actual balanced residuals, not from raw effect size alone.",
        "",
    ]
    return "\n".join(lines)


def self_test() -> None:
    np, pd = require_numeric_stack()
    # Deterministic caliper matching with a deliberately bad candidate excluded.
    feats = ["f1", "f2"]
    events = pd.DataFrame([
        {"event_id": "E1", "anchor_second": 1000, "anchor_date": "2026-01-01", "utc_hour": 0, "direction": 1, "f1": 1.0, "f2": 2.0},
        {"event_id": "E2", "anchor_second": 1010, "anchor_date": "2026-01-01", "utc_hour": 0, "direction": 1, "f1": 1.1, "f2": 2.1},
    ])
    cand = pd.DataFrame([
        {"receive_second": 2000, "anchor_date": "2026-01-01", "utc_hour": 0, "direction": 1, "control_key": "+1:2000", "control_event_id": None, "f1": 1.02, "f2": 2.01},
        {"receive_second": 2005, "anchor_date": "2026-01-01", "utc_hour": 0, "direction": 1, "control_key": "+1:2005", "control_event_id": None, "f1": 1.08, "f2": 2.08},
        {"receive_second": 2010, "anchor_date": "2026-01-01", "utc_hour": 0, "direction": 1, "control_key": "+1:2010", "control_event_id": None, "f1": 1.15, "f2": 2.15},
        {"receive_second": 2015, "anchor_date": "2026-01-01", "utc_hour": 0, "direction": 1, "control_key": "+1:2015", "control_event_id": None, "f1": 10.0, "f2": 20.0},
    ])
    parent_intervals = {"E1": (900, 1100), "E2": (900, 1100)}
    cc, _scales, cache, _diag = build_feasible_cache(events, cand, feats, "ORDINARY_LT35", parent_intervals, {})
    a1, p1 = match_from_cache(events, cc, cache, CALIPER_POLICIES[-1], feats, "100_PLUS", "ORDINARY_LT35", "REGIME_MATCHED", 2, 1, 2)
    a2, p2 = match_from_cache(events, cc, cache, CALIPER_POLICIES[-1], feats, "100_PLUS", "ORDINARY_LT35", "REGIME_MATCHED", 2, 1, 2)
    assert a1.equals(a2) and p1.equals(p2)
    assert not (a1["control_key"] == "+1:2015").any()
    assert (a1["max_abs_robust_z"] <= CALIPER_POLICIES[-1].max_abs_robust_z + 1e-12).all()

    # Physical-event overlap exclusion.
    ecand = pd.DataFrame([
        {"receive_second": 1005, "anchor_date": "2026-01-01", "utc_hour": 0, "direction": 1, "control_key": "EVENT:C1", "control_event_id": "C1", "f1": 1.0, "f2": 2.0},
        {"receive_second": 5000, "anchor_date": "2026-01-01", "utc_hour": 1, "direction": 1, "control_key": "EVENT:C2", "control_event_id": "C2", "f1": 1.0, "f2": 2.0},
    ])
    ctrl_intervals = {"C1": (950, 1200), "C2": (4900, 5200)}
    ecc, _, ecache, _ = build_feasible_cache(events.iloc[[0]], ecand, feats, "THIRTYFIVE_ONLY", parent_intervals, ctrl_intervals)
    ids = {str(ecc.iloc[o["candidate_index"]]["control_event_id"]) for o in ecache["E1"]}
    assert "C1" not in ids

    # Balance grading is intentionally separate from structural PASS/FAIL.
    assert balance_grade(0.19, 40) == "INTERPRETABLE_BALANCED"
    assert balance_grade(0.19, 10) == "BALANCED_LIMITED_N"
    assert balance_grade(0.25, 100) == "MARGINAL_BALANCE"
    assert balance_grade(0.50, 100) == "POOR_BALANCE"
    print("SELF-TEST PASS")
    print("Validated deterministic caliper exclusion, bounded reuse, physical-event overlap exclusion,")
    print("and the explicit separation between structural integrity and empirical balance quality.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 7D.1 common-support corrected matched-control audit")
    p.add_argument("--symbol", default=DEFAULT_SYMBOL)
    p.add_argument("--phase7d-script", type=Path, default=DEFAULT_PHASE7D_SCRIPT)
    p.add_argument("--phase7c-script", type=Path, default=DEFAULT_PHASE7C_SCRIPT)
    p.add_argument("--phase7d-root", type=Path, default=DEFAULT_PHASE7D_ROOT)
    p.add_argument("--phase7c-root", type=Path, default=DEFAULT_PHASE7C_ROOT)
    p.add_argument("--catalog-root", type=Path, default=DEFAULT_CATALOG_ROOT)
    p.add_argument("--primitive-root", type=Path, default=DEFAULT_PRIMITIVE_ROOT)
    p.add_argument("--flow-root", type=Path, default=DEFAULT_FLOW_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--horizon-sec", type=int, default=PRIMARY_HORIZON_SEC)
    p.add_argument("--negative-threshold-bps", type=float, default=NEGATIVE_THRESHOLD_BPS)
    p.add_argument("--controls-per-event", type=int, default=CONTROLS_PER_EVENT)
    p.add_argument("--min-controls-per-event", type=int, default=MIN_CONTROLS_PER_EVENT)
    p.add_argument("--control-reuse-cap", type=int, default=CONTROL_REUSE_CAP)
    p.add_argument("--event-embargo-sec", type=int, default=EVENT_EMBARGO_SEC)
    p.add_argument("--force-control-profiles", action="store_true", help="Rebuild only Phase-7D.1 ordinary-control trajectory checkpoint")
    p.add_argument("--self-test", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.controls_per_event < 1:
        raise SystemExit("--controls-per-event must be >=1")
    if args.min_controls_per_event < 1 or args.min_controls_per_event > args.controls_per_event:
        raise SystemExit("--min-controls-per-event must be between 1 and --controls-per-event")
    if args.control_reuse_cap < 1:
        raise SystemExit("--control-reuse-cap must be >=1")

    np, pd, _, pq = require_stack()
    phase7c, phase7d = load_phase_modules(args.phase7c_script, args.phase7d_script)

    print("Building Phase 7D.1 common-support corrected matched controls...")
    print(f"Builder version          : {BUILDER_VERSION}")
    print(f"Frozen Phase-7D version  : {phase7d.BUILDER_VERSION}")
    print(f"Frozen Phase-7C version  : {getattr(phase7c, 'BUILDER_VERSION', 'unknown')}")
    print(f"Symbol                   : {args.symbol}")
    print(f"Controls/event           : up to {args.controls_per_event}; require >= {args.min_controls_per_event}")
    print(f"Reuse cap                : {args.control_reuse_cap} within each analysis")
    print("Research role            : retrospective common-support correction; no prospective alpha claim")

    print("[1/8] Loading frozen Phase-7A/7C/7D artifacts...")
    candidate_path = args.phase7d_root / "candidate_controls.parquet"
    event_state_path = args.phase7d_root / "event_anchor_match_states.parquet"
    event_path = args.catalog_root / "physical_events.parquet"
    prep_path = args.phase7c_root / "preparation_event_features.parquet"
    stage_path = args.phase7c_root / "stage_event_features.parquet"
    required = [candidate_path, event_state_path, event_path, prep_path, stage_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError("Missing frozen upstream artifacts. Phase 7D.1 does not regenerate audited upstream data:\n  " + "\n  ".join(missing))
    candidates = pq.read_table(candidate_path).to_pandas()
    event_states = pq.read_table(event_state_path).to_pandas()
    events = pq.read_table(event_path).to_pandas()
    event_prep = pq.read_table(prep_path).to_pandas()
    event_stage = pq.read_table(stage_path).to_pandas()
    if "symbol" in events.columns:
        events = events[events["symbol"] == args.symbol].copy()
    event_meta = phase7c.build_event_metadata(events).reset_index(drop=True)
    event_states = event_states[event_states["event_id"].isin(event_meta["event_id"])].copy()
    event_prep = event_prep[event_prep["event_id"].isin(event_meta["event_id"])].copy()
    event_stage = event_stage[event_stage["event_id"].isin(event_meta["event_id"])].copy()
    start = date.fromisoformat(str(event_meta["onset_date"].min()))
    end = date.fromisoformat(str(event_meta["onset_date"].max()))
    print(f"      dates={start} to {end}, physical events={len(event_meta):,}, ordinary candidates={len(candidates):,}")

    print("[2/8] Resolving frozen matching variables and control families...")
    regime_features, strict_features = phase7d.resolve_match_features(candidates)
    match_feature_map = {"REGIME_MATCHED": list(regime_features), "STRICT_STATE_MATCHED": list(strict_features)}
    ordinary_pool, thirtyfive_pool, states_enriched = build_control_pools(candidates, event_states, event_meta)
    valid_states = states_enriched[states_enriched["book_valid"].fillna(False).astype(bool) & states_enriched["flow_feature_context_ready"].fillna(False).astype(bool)].copy()
    target_union_ids = set(event_meta[cohort_mask(event_meta, 50, args.horizon_sec)]["event_id"].astype(str))
    target_states = valid_states[valid_states["event_id"].astype(str).isin(target_union_ids)].copy()
    parent_intervals = {str(r["event_id"]): event_interval(r, args.event_embargo_sec) for r in event_meta.to_dict("records")}
    control_intervals = parent_intervals
    print(f"      valid 50+ target anchor states={len(target_states):,}")
    print(f"      35_ONLY control-event states={len(thirtyfive_pool):,}")
    print(f"      regime features={', '.join(regime_features)}")
    print(f"      strict adds={', '.join([x for x in strict_features if x not in regime_features])}")

    print("[3/8] Building relaxed common-support caches once per control family/tier...")
    pools = {"ORDINARY_LT35": ordinary_pool, "THIRTYFIVE_ONLY": thirtyfive_pool}
    caches: dict[tuple[str, str], Any] = {}
    prepared_pools: dict[tuple[str, str], Any] = {}
    scale_meta = []
    cache_diag = []
    for family in CONTROL_FAMILIES:
        for tier in MATCH_TIERS:
            feats = match_feature_map[tier]
            cp = pools[family]
            cc, scales, cache, diag = build_feasible_cache(target_states, cp, feats, family, parent_intervals, control_intervals)
            prepared_pools[(family, tier)] = cc
            caches[(family, tier)] = cache
            for f, s in scales.items():
                scale_meta.append({"control_family": family, "match_tier": tier, "feature": f, "candidate_robust_scale": s})
            diag["control_family"] = family
            diag["match_tier"] = tier
            cache_diag.append(diag)
            supported = int((diag["feasible_relaxed"] > 0).sum()) if not diag.empty else 0
            print(f"      {family:<18} {tier:<20}: candidates={len(cc):,}, 50+ relaxed-support parents={supported:,}/{len(target_states):,}")

    print("[4/8] Running prespecified caliper sensitivity matches...")
    assignment_parts = []
    parent_parts = []
    for cohort, target in TARGET_COHORTS:
        ids = set(event_meta[cohort_mask(event_meta, target, args.horizon_sec)]["event_id"].astype(str))
        es = target_states[target_states["event_id"].astype(str).isin(ids)].copy()
        for family in CONTROL_FAMILIES:
            for tier in MATCH_TIERS:
                cp = prepared_pools[(family, tier)]
                cache = caches[(family, tier)]
                feats = match_feature_map[tier]
                for policy in CALIPER_POLICIES:
                    a, ps = match_from_cache(es, cp, cache, policy, feats, cohort, family, tier, args.controls_per_event, args.min_controls_per_event, args.control_reuse_cap)
                    assignment_parts.append(a)
                    parent_parts.append(ps)
                    accepted = int(ps["accepted_for_contrasts"].sum()) if not ps.empty else 0
                    print(f"      {cohort:<8} {family:<18} {tier:<20} {policy.name:<7}: accepted={accepted:,}/{len(es):,}, controls={len(a):,}")
    assignments = pd.concat(assignment_parts, ignore_index=True) if assignment_parts else pd.DataFrame()
    parent_status = pd.concat(parent_parts, ignore_index=True) if parent_parts else pd.DataFrame()
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_parquet_atomic(assignments, args.output_root / "control_assignments.parquet")
    write_parquet_atomic(parent_status, args.output_root / "parent_match_status.parquet")

    print("[5/8] Computing parent-paired matching balance before mechanism interpretation...")
    balance, quality = build_parent_balance(assignments, parent_status, match_feature_map, args.controls_per_event)
    coverage = build_coverage(parent_status, args.controls_per_event)
    if not quality.empty:
        for r in quality.sort_values(["cohort", "control_family", "match_tier", "caliper_policy"]).to_dict("records"):
            print(f"      {r['analysis_id']}: coverage={100*safe_float(r['accepted_parent_share']):5.1f}% max|SMD|={safe_float(r['max_abs_smd']):.3f} {r['balance_grade']}")

    print("[6/8] Profiling only accepted ordinary controls and computing mechanism trajectories...")
    features = sorted(event_prep["feature"].astype(str).unique())
    feature_family = phase7d.feature_family_map(phase7c, features)
    ordinary_profiles, reused = prepare_ordinary_control_profiles(assignments, phase7d, phase7c, args.primitive_root, args.flow_root, args.symbol, start, end, features, args.output_root, args.force_control_profiles)
    print(f"      ordinary control profiles={'reused' if reused else 'built'}, unique keys={ordinary_profiles['control_key'].nunique() if not ordinary_profiles.empty else 0:,}")

    prep_summary_parts = []
    prep_daily_parts = []
    prep_robust_parts = []
    stage_summary_parts = []
    stage_daily_parts = []
    stage_robust_parts = []
    if not assignments.empty:
        quality_map = quality.set_index("analysis_id").to_dict("index") if not quality.empty else {}
        for aid, ag in assignments.groupby("analysis_id", sort=True):
            q = quality_map.get(aid, {})
            meta = {
                "analysis_id": aid,
                "cohort": str(ag["cohort"].iloc[0]),
                "control_family": str(ag["control_family"].iloc[0]),
                "match_tier": str(ag["match_tier"].iloc[0]),
                "caliper_policy": str(ag["caliper_policy"].iloc[0]),
                "analysis_balance_grade": q.get("balance_grade", "NOT_EVALUABLE"),
                "analysis_max_abs_smd": q.get("max_abs_smd", math.nan),
            }
            pp = parent_pair_rows_preparation(ag, event_prep, ordinary_profiles)
            if not pp.empty:
                prep_summary_parts.append(summarize_pair_rows(pp, "window_label", feature_family, meta))
                prep_daily_parts.append(daily_consistency(pp, "window_label", meta))
                prep_robust_parts.append(subset_robustness(pp, "window_label", meta, start, end))
            if meta["control_family"] == "THIRTYFIVE_ONLY":
                sp = parent_pair_rows_stage_35only(ag, event_stage)
                if not sp.empty:
                    stage_summary_parts.append(summarize_pair_rows(sp, "stage", feature_family, meta))
                    stage_daily_parts.append(daily_consistency(sp, "stage", meta))
                    stage_robust_parts.append(subset_robustness(sp, "stage", meta, start, end))
    prep_summary = pd.concat(prep_summary_parts, ignore_index=True) if prep_summary_parts else pd.DataFrame()
    prep_daily = pd.concat(prep_daily_parts, ignore_index=True) if prep_daily_parts else pd.DataFrame()
    prep_robust = pd.concat(prep_robust_parts, ignore_index=True) if prep_robust_parts else pd.DataFrame()
    stage_summary = pd.concat(stage_summary_parts, ignore_index=True) if stage_summary_parts else pd.DataFrame()
    stage_daily = pd.concat(stage_daily_parts, ignore_index=True) if stage_daily_parts else pd.DataFrame()
    stage_robust = pd.concat(stage_robust_parts, ignore_index=True) if stage_robust_parts else pd.DataFrame()
    prep_screen = build_interpretable_screen(prep_summary, prep_daily, quality, "window_label")
    stage_screen = build_interpretable_screen(stage_summary, stage_daily, quality, "stage")

    print("[7/8] Checking structural and anti-leakage invariants...")
    checks = check_invariants(candidates, event_meta, assignments, parent_status, ordinary_profiles, match_feature_map, phase7d, args)
    failed = checks[checks["status"] == "FAIL"]
    print(f"      invariants={len(checks):,}, failed={len(failed):,}")

    print("[8/8] Writing reports and frozen run metadata...")
    args.report_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scale_meta).to_csv(args.report_root / "common_support_candidate_scales.csv", index=False)
    pd.concat(cache_diag, ignore_index=True).to_csv(args.report_root / "common_support_relaxed_feasibility.csv", index=False)
    coverage.to_csv(args.report_root / "common_support_coverage.csv", index=False)
    parent_status.to_csv(args.report_root / "common_support_parent_dropout.csv", index=False)
    balance.to_csv(args.report_root / "common_support_balance.csv", index=False)
    quality.to_csv(args.report_root / "common_support_analysis_quality.csv", index=False)
    prep_summary.to_csv(args.report_root / "common_support_preparation_contrasts.csv", index=False)
    prep_daily.to_csv(args.report_root / "common_support_preparation_daily_consistency.csv", index=False)
    prep_robust.to_csv(args.report_root / "common_support_preparation_subset_robustness.csv", index=False)
    prep_screen.to_csv(args.report_root / "common_support_preparation_interpretable_screen.csv", index=False)
    stage_summary.to_csv(args.report_root / "common_support_35only_stage_contrasts.csv", index=False)
    stage_daily.to_csv(args.report_root / "common_support_35only_stage_daily_consistency.csv", index=False)
    stage_robust.to_csv(args.report_root / "common_support_35only_stage_subset_robustness.csv", index=False)
    stage_screen.to_csv(args.report_root / "common_support_35only_stage_interpretable_screen.csv", index=False)
    checks.to_csv(args.report_root / "common_support_invariants.csv", index=False)

    interpretable_analyses = int((quality["balance_grade"] == "INTERPRETABLE_BALANCED").sum()) if not quality.empty else 0
    summary = {
        "status": "PASS" if failed.empty else "FAIL",
        "builder_version": BUILDER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "schema_name": SCHEMA_NAME,
        "symbol": args.symbol,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "phase7d_builder_version": str(phase7d.BUILDER_VERSION),
        "phase7c_builder_version": str(getattr(phase7c, "BUILDER_VERSION", "unknown")),
        "ordinary_candidate_controls": int(len(candidates)),
        "thirtyfive_only_candidates": int(len(thirtyfive_pool)),
        "target_50plus_anchor_states": int(len(target_states)),
        "control_assignments": int(len(assignments)),
        "accepted_control_assignments": int(assignments["accepted_parent"].fillna(False).sum()) if not assignments.empty else 0,
        "interpretable_balanced_analyses": interpretable_analyses,
        "failed_invariants": int(len(failed)),
        "matching_features": match_feature_map,
        "caliper_policies": [vars(x) for x in CALIPER_POLICIES],
        "research_role": "RETROSPECTIVE_COMMON_SUPPORT_CORRECTION_NOT_PROSPECTIVE_ALPHA",
    }
    atomic_write_json(args.output_root / "metadata" / "latest_summary.json", summary)
    atomic_write_json(args.output_root / "metadata" / "run_config.json", {
        "builder_version": BUILDER_VERSION,
        "phase7d_script": str(args.phase7d_script),
        "phase7c_script": str(args.phase7c_script),
        "phase7d_root": str(args.phase7d_root),
        "phase7c_root": str(args.phase7c_root),
        "catalog_root": str(args.catalog_root),
        "primitive_root": str(args.primitive_root),
        "flow_root": str(args.flow_root),
        "horizon_sec": args.horizon_sec,
        "negative_threshold_bps": args.negative_threshold_bps,
        "controls_per_event": args.controls_per_event,
        "min_controls_per_event": args.min_controls_per_event,
        "control_reuse_cap": args.control_reuse_cap,
        "event_embargo_sec": args.event_embargo_sec,
        "target_cohorts": list(TARGET_COHORTS),
        "control_families": list(CONTROL_FAMILIES),
        "match_tiers": list(MATCH_TIERS),
        "matching_features": match_feature_map,
        "caliper_policies": [vars(x) for x in CALIPER_POLICIES],
        "balance_good_max_abs_smd": BALANCE_GOOD_MAX_ABS_SMD,
        "balance_marginal_max_abs_smd": BALANCE_MARGINAL_MAX_ABS_SMD,
        "min_interpretable_parents": MIN_INTERPRETABLE_PARENTS,
    })
    atomic_write_text(args.report_root / "common_support_findings.md", build_findings(summary, quality, coverage, prep_screen, stage_screen))

    print(f"Phase 7D.1 status       : {summary['status']}")
    print(f"Interpretable analyses  : {summary['interpretable_balanced_analyses']}")
    print(f"Reports                 : {args.report_root.resolve()}")
    if not failed.empty:
        print(failed.to_string(index=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
