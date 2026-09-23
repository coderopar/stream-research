#!/usr/bin/env python3
"""Phase 7G — liquidity-state transition timing.

Scientific role
---------------
Phase 7F showed that no single earlier fixed displacement (10/15/20/25/30 bps)
replicated cleanly enough to replace 35 bps.  Phase 7G therefore stops trying
to optimize a universal bps breakpoint and instead asks when the exact
liquidity-failure STATE first becomes observable within each physical move.

The experiment preserves the frozen Phase-7D.1 matched physical populations
and the exact raw-depth mechanics established in Phase 7E.  It samples every
integer physical displacement from 10 through 35 bps.  At each stage it freezes
the exact <=5-bp queues and follows them for a fixed 3-second observation
horizon.

A liquidity-failure state is defined BEFORE inspecting Phase-7G outcomes using
four Phase-7E core mechanics:

    opposing peak initial-queue loss                 higher = failure
    opposing durable unrecovered loss                higher = failure
    opposing retained initial liquidity              lower  = failure
    support-minus-opposing retained-liquidity gap     higher = failure

Each feature is robust-standardized against the discovery-period PRIMARY
35_ONLY control distribution at the same physical stage.  A stage is a
candidate state when all four features are measurable, at least 3/4 point in
the expected direction, at least 2/4 exceed +0.50 robust-z, and the median
signed robust-z is >= +0.35.

The PRIMARY transition is CONFIRMED only when candidate states occur at two
consecutive distinct stage observations.  Because each state itself requires a
3-second forward raw-depth observation, the live-observable confirmation time
is the confirmation stage second + 3 seconds.  The first single-stage candidate
is retained separately as an exploratory earlier timestamp.

The detector rule is NOT tuned to 75+ outcomes.  Robust baselines are estimated
only from discovery-period 35_ONLY controls (onset date <= 2026-06-17), then
frozen and applied unchanged to discovery and validation events.

This remains retrospective mechanism/timing research:
  * physical extremum and direction are hindsight-defined;
  * 75+/50+ continuation labels are hindsight-defined;
  * matched 35_ONLY controls are hindsight-defined;
  * the resulting state rule is causal-input-computable once an anchor exists,
    but it is NOT yet an all-market prospective signal or a tradable edge.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

BUILDER_VERSION = "1.0.0"
SCHEMA_VERSION = 1
SCHEMA_NAME = "large_move_liquidity_state_transition_timing_v1"
DEFAULT_SYMBOL = "BTCUSDT"

STAGE_BPS = tuple(range(10, 36))
STAGES = tuple(f"HIT_{x}BP" for x in STAGE_BPS)
OBSERVATION_HORIZON_SEC = 3
WINDOW_LABEL = f"FWD_{OBSERVATION_HORIZON_SEC}S"
DISCOVERY_END_DATE = date(2026, 6, 17)
VALIDATION_START_DATE = date(2026, 6, 18)

# Frozen transition-state rule.  Do not tune these on Phase-7G outcomes.
STRONG_SIGNED_Z = 0.50
COMPOSITE_SIGNED_Z = 0.35
MIN_ALIGNED_FEATURES = 3
MIN_STRONG_FEATURES = 2
MIN_BASELINE_N = 20
UNDETECTED_SENTINEL_BPS = 36.0

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()
DEFAULT_PHASE6A_SCRIPT = REPO_ROOT / "scripts" / "perps" / "research" / "06a_build_execution_foundation.py"
DEFAULT_PHASE7A_SCRIPT = REPO_ROOT / "scripts" / "perps" / "research" / "07a_build_large_move_event_catalog.py"
DEFAULT_PHASE7C_SCRIPT = REPO_ROOT / "scripts" / "perps" / "research" / "07c_build_large_move_participant_behavior_atlas.py"
DEFAULT_PHASE7D1_SCRIPT = REPO_ROOT / "scripts" / "perps" / "research" / "07d1_build_common_support_matched_controls.py"
DEFAULT_PHASE7E_SCRIPT = REPO_ROOT / "scripts" / "perps" / "research" / "07e_build_raw_depth_continuation_mechanics.py"
DEFAULT_PHASE7F_SCRIPT = REPO_ROOT / "scripts" / "perps" / "research" / "07f_build_continuation_threshold_liquidity_calibration.py"
DEFAULT_EXECUTION_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "execution_foundation_v1"
DEFAULT_CATALOG_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / "large_move_event_catalog_v1"
DEFAULT_PHASE7D1_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / "large_move_common_support_matched_controls_v1"
DEFAULT_CANONICAL_ROOT = REPO_ROOT / "data" / "canonical" / "market_events_v1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / SCHEMA_NAME
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "perps_liquidity_state_transition_timing"

ANALYSIS_SPECS = (
    ("PRIMARY", "75_PLUS|THIRTYFIVE_ONLY|REGIME_MATCHED|PRIMARY"),
    ("ROBUST_75_RELAXED", "75_PLUS|THIRTYFIVE_ONLY|REGIME_MATCHED|RELAXED"),
    ("ROBUST_50_STRICT", "50_PLUS|THIRTYFIVE_ONLY|STRICT_STATE_MATCHED|RELAXED"),
    ("ROBUST_50_REGIME", "50_PLUS|THIRTYFIVE_ONLY|REGIME_MATCHED|PRIMARY"),
)
PRIMARY_ANALYSIS_ID = ANALYSIS_SPECS[0][1]
ANALYSIS_ROLE = {aid: role for role, aid in ANALYSIS_SPECS}

CORE_FEATURES = (
    ("opposing__peak_initial_queue_loss_share", +1),
    ("opposing__durable_unrecovered_loss_share_initial", +1),
    ("opposing__retained_initial_quote_ratio", -1),
    ("liquidity_retention_asymmetry", +1),
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_module(path: Path, name: str):
    import importlib.util
    import sys
    if not path.exists():
        raise RuntimeError(f"Missing required frozen script: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def require_numeric_stack():
    try:
        import numpy as np
        import pandas as pd
    except Exception as exc:
        raise SystemExit("Phase 7G requires numpy and pandas. Activate the project .venv first.") from exc
    return np, pd


def require_stack():
    np, pd = require_numeric_stack()
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:
        raise SystemExit("Phase 7G Parquet I/O requires pyarrow. Activate the project .venv first.") from exc
    return np, pd, pa, pq


def safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else math.nan
    except Exception:
        return math.nan


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7g-", suffix=path.suffix, dir=str(path.parent))
    os.close(fd)
    try:
        Path(tmp).write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def write_parquet_atomic(df, path: Path) -> None:
    _, _, pa, pq = require_stack()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7g-", suffix=".parquet", dir=str(path.parent))
    os.close(fd)
    try:
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp, compression="zstd")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def file_signature(paths: Sequence[Path]) -> list[dict[str, Any]]:
    out = []
    for p in sorted(set(Path(x) for x in paths), key=lambda x: str(x)):
        st = p.stat()
        out.append({"path": str(p.resolve()), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)})
    return out


def stage_label(bps: int) -> str:
    return f"HIT_{int(bps)}BP"


def stage_bps_from_label(label: str) -> int:
    return int(str(label).replace("HIT_", "").replace("BP", ""))


def load_frozen_stack(args):
    phase7f = load_module(args.phase7f_script, "phase7f_frozen_for_7g")
    if getattr(phase7f, "BUILDER_VERSION", None) != "1.0.0":
        raise RuntimeError(f"Phase 7G is frozen against Phase-7F builder 1.0.0; found {getattr(phase7f, 'BUILDER_VERSION', None)!r}")
    phase6a, phase7a, phase7c, phase7d1, phase7e = phase7f.load_frozen_modules(args)
    if getattr(phase7e, "BUILDER_VERSION", None) != "1.0.3":
        raise RuntimeError("Phase 7G requires frozen Phase-7E builder 1.0.3")
    return phase6a, phase7a, phase7c, phase7d1, phase7e, phase7f


def load_assignments_events(args, phase7c, phase7e, phase7f):
    assignments, quality = phase7f.load_assignments(args, phase7e)
    events, fixed10_map = phase7f.load_selected_events(args, phase7c, assignments)
    return assignments, quality, events, fixed10_map


def build_dense_stage_table(args, phase7a, phase7f, events, fixed10_map):
    np, pd = require_numeric_stack()
    rows = []
    mismatch35 = 0
    mismatch10 = 0
    for i, (onset_date, g) in enumerate(events.groupby("onset_date", sort=True), 1):
        base = date.fromisoformat(str(onset_date))
        max_hit = int(pd.to_numeric(g["hit_35_second"], errors="coerce").dropna().max())
        last_day = datetime.fromtimestamp(max_hit, tz=timezone.utc).date()
        days = []
        prev = base - timedelta(days=1)
        if phase7a.execution_day_path(args.execution_root, args.symbol, prev).exists():
            days.append(prev)
        days.append(base)
        if last_day > base:
            days.append(base + timedelta(days=1))
        frame = phase7f.read_execution_days(args, phase7a, days)
        sec, price, source_label = phase7a.build_catalog_price(frame, args.max_trade_stale_sec, "last")
        for ev in g.to_dict("records"):
            onset = int(ev["onset_second"])
            hit35 = int(ev["hit_35_second"])
            li = int(np.searchsorted(sec, onset, side="left"))
            ri = int(np.searchsorted(sec, hit35, side="right"))
            if li >= len(sec) or int(sec[li]) != onset:
                raise RuntimeError(f"Missing exact onset second in execution price for event {ev['event_id']}")
            path_sec = sec[li:ri]
            path_px = price[li:ri]
            row = dict(ev)
            row["catalog_price_source"] = source_label
            for bps in STAGE_BPS:
                hs = phase7f.first_passage_second(path_sec, path_px, float(ev["onset_price"]), int(ev["direction"]), float(bps))
                row[f"stage__{stage_label(bps)}"] = hs
            if row["stage__HIT_35BP"] != hit35:
                mismatch35 += 1
            expected10 = fixed10_map.get(str(ev["event_id"]))
            if expected10 is not None and row["stage__HIT_10BP"] != expected10:
                mismatch10 += 1
            rows.append(row)
        print(f"      stage geometry [{i:02d}] {onset_date}: events={len(g):3d}")
    out = pd.DataFrame(rows)
    if out["event_id"].duplicated().any():
        raise RuntimeError("Dense transition-stage table is not unique by event_id")
    for st in STAGES:
        if out[f"stage__{st}"].isna().any():
            raise RuntimeError(f"Selected 35+ event missing transition stage {st}")
    return out.sort_values(["onset_date", "onset_second", "event_id"], kind="mergesort").reset_index(drop=True), mismatch35, mismatch10


def day_partition_paths(output_root: Path, onset_date: str) -> tuple[Path, Path]:
    root = output_root / "event_windows" / f"onset_date={onset_date}"
    return root / "part.parquet", root / "manifest.json"


def day_input_signature(day_events, day_files, onset_date: str) -> str:
    payload = {
        "builder_version": BUILDER_VERSION,
        "schema_name": SCHEMA_NAME,
        "onset_date": onset_date,
        "events": day_events.sort_values("event_id", kind="mergesort").to_dict("records"),
        "canonical_files": file_signature(day_files.depth + day_files.snapshots + day_files.trades),
        "stage_bps": STAGE_BPS,
        "observation_horizon_sec": OBSERVATION_HORIZON_SEC,
        "frozen_near_bps": 5.0,
    }
    return sha256_json(payload)


def level_summary_add(summary: dict[str, Any], tr) -> None:
    for role, rt in tr.roles.items():
        summary["roles"].add(str(role))
        for st in rt.initial_levels.values():
            summary["row_count"] += 1
            if not (math.isfinite(float(st.initial_qty)) and float(st.initial_qty) > 0):
                summary["nonpositive_initial_qty"] += 1


def make_primary_tracker(phase7e, ev: Mapping[str, Any], stage: str, stage_second: int, book: Any, max_age_ms: int, price_scale: float):
    anchor_end_ns = (int(stage_second) + 1) * 1_000_000_000 - 1
    end_second = int(stage_second) + OBSERVATION_HORIZON_SEC
    ep = phase7e.Endpoint(
        WINDOW_LABEL,
        anchor_end_ns + OBSERVATION_HORIZON_SEC * 1_000_000_000,
        end_second,
        "FIXED_HORIZON",
    )
    age_ms = (
        (anchor_end_ns - int(book.last_state_ts_ns)) / 1_000_000.0
        if getattr(book, "last_state_ts_ns", None) is not None else math.inf
    )
    mid0 = phase7e.book_mid(book)
    if not bool(getattr(book, "valid", False)) or bool(getattr(book, "bridge_pending", False)) or age_ms > max_age_ms or not math.isfinite(mid0):
        return None, [{
            "builder_version": BUILDER_VERSION,
            "schema_version": SCHEMA_VERSION,
            "event_id": str(ev["event_id"]),
            "onset_date": str(ev["onset_date"]),
            "direction": int(ev["direction"]),
            "direction_label": "UP" if int(ev["direction"]) > 0 else "DOWN",
            "continuation_stratum": str(ev["continuation_stratum"]),
            "stage": stage,
            "stage_second": int(stage_second),
            "window_label": WINDOW_LABEL,
            "window_type": "FIXED_HORIZON",
            "transition_end_stage": None,
            "window_end_second": end_second,
            "elapsed_sec": float(OBSERVATION_HORIZON_SEC),
            "anchor_mid": mid0,
            "anchor_spread_bps": phase7e.spread_bps(book),
            "window_valid": False,
            "invalid_reason": "ANCHOR_BOOK_INVALID_OR_STALE",
        }]
    roles = {}
    for role in ("OPPOSING", "SUPPORT"):
        side_name = phase7e.role_side(int(ev["direction"]), role)
        levels = phase7e.frozen_levels(book, int(ev["direction"]), role, mid0, price_scale, phase7e.FROZEN_NEAR_BPS)
        states = {tick: phase7e.LevelState(tick, tick / price_scale, q, q) for tick, q in levels.items()}
        initial_quote = sum(st.price * st.initial_qty for st in states.values())
        roles[role] = phase7e.RoleTracker(role, side_name, states, float(initial_quote))
    tr = phase7e.MechanicsTracker(
        event_id=str(ev["event_id"]), onset_date=str(ev["onset_date"]), direction=int(ev["direction"]),
        continuation_stratum=str(ev["continuation_stratum"]), stage=stage, stage_second=int(stage_second),
        anchor_end_ns=anchor_end_ns, mid0=mid0, spread_bps0=phase7e.spread_bps(book), endpoints=[ep],
        roles=roles, price_scale=price_scale, max_book_age_ms=int(max_age_ms),
    )
    return tr, []


def process_onset_date(onset_date: str, day_events, phase6a, phase7e, family_maps, args):
    _, pd, _, pq = require_stack()
    base = date.fromisoformat(onset_date)
    stage_seconds = [int(r[f"stage__{st}"]) for r in day_events.to_dict("records") for st in STAGES]
    max_end_second = max(stage_seconds) + OBSERVATION_HORIZON_SEC
    end_date = datetime.fromtimestamp(max_end_second, tz=timezone.utc).date()
    scan_days = [base]
    if end_date > base:
        scan_days.append(base + timedelta(days=1))
    day_files = phase7e.build_day_files(phase6a, family_maps, scan_days)
    if not day_files.depth or not day_files.snapshots:
        raise RuntimeError(f"Missing canonical depth/snapshot files for onset date {onset_date}")

    profile_path, manifest_path = day_partition_paths(args.output_root, onset_date)
    sig = day_input_signature(day_events, day_files, onset_date)
    if not args.force_day and profile_path.exists() and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
        if manifest.get("builder_version") == BUILDER_VERSION and manifest.get("input_signature") == sig:
            profiles = pq.read_table(profile_path).to_pandas()
            return profiles, True, manifest

    trade_t0 = time.perf_counter()
    trades = phase6a.aggregate_trades(day_files, args.symbol)
    trade_context = phase7e.TradeContext(trades)
    trade_seconds = time.perf_counter() - trade_t0

    anchors = []
    for ev in day_events.to_dict("records"):
        for st in STAGES:
            sec = int(ev[f"stage__{st}"])
            anchors.append(((sec + 1) * 1_000_000_000 - 1, st, ev))
    anchors.sort(key=lambda x: (x[0], str(x[2]["event_id"]), stage_bps_from_label(x[1])))

    book = phase6a.LocalBook(price_scale=float(args.price_scale))
    fast_stats = phase6a.FastDepthStats()
    gap_records = []
    active = []
    profile_rows: list[dict[str, Any]] = []
    anchor_idx = 0
    stop_ns = (max_end_second + 1) * 1_000_000_000 - 1
    tracked_depth_messages = 0
    tracked_exact_changes = 0
    level_summary = {"row_count": 0, "nonpositive_initial_qty": 0, "roles": set()}
    replay_t0 = time.perf_counter()

    def collect_finished(tr) -> None:
        profile_rows.extend(tr.rows)
        level_summary_add(level_summary, tr)

    def instantiate_before(ts_ns: int) -> None:
        nonlocal anchor_idx
        while anchor_idx < len(anchors) and anchors[anchor_idx][0] < ts_ns:
            _, st, ev = anchors[anchor_idx]
            sec = int(ev[f"stage__{st}"])
            tr, invalid = make_primary_tracker(phase7e, ev, st, sec, book, int(args.max_book_age_ms), float(args.price_scale))
            profile_rows.extend(invalid)
            if tr is not None:
                active.append(tr)
            anchor_idx += 1

    for rec in phase6a.unified_book_events(day_files, args.symbol, float(args.price_scale), fast_stats):
        ts_ns = int(rec["ts_recv_ns"])
        if ts_ns > stop_ns:
            instantiate_before(ts_ns)
            for tr in active:
                tr.finalize_due_before(ts_ns, book, trade_context)
            remain = []
            for tr in active:
                if tr.done:
                    collect_finished(tr)
                else:
                    remain.append(tr)
            active[:] = remain
            break

        instantiate_before(ts_ns)
        for tr in active:
            tr.finalize_due_before(ts_ns, book, trade_context)

        kind = rec.get("_kind")
        if kind == "SNAPSHOT":
            book.apply_snapshot(rec)
        elif kind == "DEPTH":
            applicable = phase7e.depth_row_is_applicable(book, rec)
            track_changes = bool(active) and any(tr.censored_ns is None and not tr.done and ts_ns > tr.anchor_end_ns for tr in active)
            changes = phase7e.extract_row_changes(book, rec) if applicable and track_changes else []
            if applicable and track_changes:
                tracked_depth_messages += 1
                tracked_exact_changes += len(changes)
            was_valid = bool(book.valid)
            book.apply_depth(rec, gap_records, onset_date)
            still_valid = bool(book.valid)
            if applicable and was_valid and still_valid:
                for tr in active:
                    tr.on_changes(ts_ns, changes)
            elif was_valid and not still_valid:
                reason = gap_records[-1].reason if gap_records else "BOOK_INVALIDATED"
                for tr in active:
                    if ts_ns > tr.anchor_end_ns and not tr.done:
                        tr.censor(ts_ns, reason)

        for tr in active:
            tr.finalize_due_through(ts_ns, book, trade_context)
        remain = []
        for tr in active:
            if tr.done:
                collect_finished(tr)
            else:
                remain.append(tr)
        active[:] = remain

    instantiate_before(stop_ns + 1)
    for tr in active:
        tr.finalize_due_through(stop_ns, book, trade_context)
        if not tr.done:
            tr.finalize_all(book, trade_context)
        collect_finished(tr)
    active.clear()
    replay_seconds = time.perf_counter() - replay_t0

    profiles = pd.DataFrame(profile_rows)
    if not profiles.empty:
        profiles = profiles.rename(columns={"builder_version": "raw_mechanics_builder_version"})
        profiles["builder_version"] = BUILDER_VERSION
        profiles["transition_schema_version"] = SCHEMA_VERSION
        profiles["day_input_signature"] = sig
        profiles["stage_bps"] = profiles["stage"].map(stage_bps_from_label).astype(int)
        profiles["liquidity_retention_asymmetry"] = (
            pd.to_numeric(profiles.get("support__retained_initial_quote_ratio"), errors="coerce")
            - pd.to_numeric(profiles.get("opposing__retained_initial_quote_ratio"), errors="coerce")
        )
    write_parquet_atomic(profiles, profile_path)
    manifest = {
        "builder_version": BUILDER_VERSION, "schema_name": SCHEMA_NAME, "input_signature": sig,
        "onset_date": onset_date, "event_count": int(day_events["event_id"].nunique()),
        "profile_rows": int(len(profiles)), "level_rows": int(level_summary["row_count"]),
        "nonpositive_initial_qty": int(level_summary["nonpositive_initial_qty"]),
        "level_roles": sorted(level_summary["roles"]), "trade_aggregation_seconds": float(trade_seconds),
        "raw_replay_seconds": float(replay_seconds), "depth_rows_decoded": int(getattr(fast_stats, "rows_decoded", 0)),
        "depth_levels_decoded": int(getattr(fast_stats, "levels_decoded", 0)),
        "tracked_depth_messages": int(tracked_depth_messages), "tracked_exact_changes": int(tracked_exact_changes),
        "generated_at": utc_now_iso(),
    }
    atomic_write_json(manifest_path, manifest)
    return profiles, False, manifest


def robust_baseline_scale(values) -> tuple[float, float]:
    np, _ = require_numeric_stack()
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return math.nan, math.nan
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med))) * 1.4826
    if math.isfinite(mad) and mad > 1e-12:
        return med, mad
    q25, q75 = np.quantile(a, [0.25, 0.75])
    iqr_scale = float((q75 - q25) / 1.349)
    if math.isfinite(iqr_scale) and iqr_scale > 1e-12:
        return med, iqr_scale
    sd = float(np.std(a, ddof=1))
    return med, sd if math.isfinite(sd) and sd > 1e-12 else math.nan


def fit_discovery_control_baselines(profiles, assignments):
    _, pd = require_numeric_stack()
    primary = assignments[assignments["analysis_id"] == PRIMARY_ANALYSIS_ID]
    control_ids = set(primary["control_event_id"].astype(str))
    p = profiles[
        profiles["event_id"].astype(str).isin(control_ids)
        & profiles["window_valid"].fillna(False).astype(bool)
        & (pd.to_datetime(profiles["onset_date"]).dt.date <= DISCOVERY_END_DATE)
    ].copy()
    rows = []
    for bps in STAGE_BPS:
        sg = p[p["stage_bps"] == bps]
        for feat, sign in CORE_FEATURES:
            vals = pd.to_numeric(sg.get(feat), errors="coerce").dropna().to_numpy(float)
            med, scale = robust_baseline_scale(vals)
            rows.append({
                "stage_bps": int(bps), "stage": stage_label(bps), "feature": feat, "expected_sign": int(sign),
                "control_n": int(len(vals)), "control_median": med, "control_robust_scale": scale,
                "baseline_scope": f"PRIMARY unique 35_ONLY controls with onset_date <= {DISCOVERY_END_DATE.isoformat()}",
            })
    return pd.DataFrame(rows)


def score_profiles(profiles, baselines):
    np, pd = require_numeric_stack()
    p = profiles.copy()
    p = p[p["window_valid"].fillna(False).astype(bool)].copy()
    base = baselines.set_index(["stage_bps", "feature"])
    zcols = []
    for feat, sign in CORE_FEATURES:
        zc = "signed_z__" + feat
        zcols.append(zc)
        vals = pd.to_numeric(p.get(feat), errors="coerce").to_numpy(float)
        zs = np.full(len(p), np.nan, dtype=float)
        for i, (bps, value) in enumerate(zip(p["stage_bps"].astype(int).to_numpy(), vals)):
            if not math.isfinite(value) or (bps, feat) not in base.index:
                continue
            r = base.loc[(bps, feat)]
            med = safe_float(r["control_median"]); scale = safe_float(r["control_robust_scale"])
            if math.isfinite(med) and math.isfinite(scale) and scale > 1e-12:
                zs[i] = int(sign) * (value - med) / scale
        p[zc] = zs
    zmat = p[zcols].to_numpy(float)
    finite = np.isfinite(zmat)
    p["state_valid_core_features"] = finite.sum(axis=1)
    p["state_aligned_features"] = np.sum(finite & (zmat > 0.0), axis=1)
    p["state_strong_features"] = np.sum(finite & (zmat >= STRONG_SIGNED_Z), axis=1)
    p["liquidity_failure_score"] = np.nanmedian(zmat, axis=1)
    p["state_candidate"] = (
        (p["state_valid_core_features"] == len(CORE_FEATURES))
        & (p["state_aligned_features"] >= MIN_ALIGNED_FEATURES)
        & (p["state_strong_features"] >= MIN_STRONG_FEATURES)
        & (p["liquidity_failure_score"] >= COMPOSITE_SIGNED_Z)
    )
    p["state_observable_second"] = pd.to_numeric(p["stage_second"], errors="coerce") + OBSERVATION_HORIZON_SEC
    p["observable_directional_displacement_approx_bps"] = (
        pd.to_numeric(p["stage_bps"], errors="coerce")
        + pd.to_numeric(p.get("directional_mid_move_bps"), errors="coerce").fillna(0.0)
    )
    return p


def build_event_transition_summary(scored_profiles, event_stages):
    np, pd = require_numeric_stack()
    meta = event_stages.set_index("event_id")
    rows = []
    for event_id, g in scored_profiles.groupby("event_id", sort=False):
        # Multiple 1-bp thresholds can be crossed in one second.  They are one
        # live observation, not independent persistence evidence.  Keep the
        # highest displacement reached in each distinct anchor second.
        gg = g.sort_values(["stage_second", "stage_bps"], kind="mergesort").drop_duplicates("stage_second", keep="last").reset_index(drop=True)
        candidate_idx = [i for i, x in enumerate(gg["state_candidate"].astype(bool)) if x]
        first_cand = candidate_idx[0] if candidate_idx else None
        confirmed = None
        for i in range(1, len(gg)):
            if bool(gg.iloc[i - 1]["state_candidate"]) and bool(gg.iloc[i]["state_candidate"]):
                if int(gg.iloc[i]["stage_second"]) > int(gg.iloc[i - 1]["stage_second"]):
                    confirmed = i
                    break
        m = meta.loc[str(event_id)] if str(event_id) in meta.index else None
        base = {
            "event_id": str(event_id),
            "onset_date": str(gg.iloc[0]["onset_date"]),
            "direction": int(gg.iloc[0]["direction"]),
            "direction_label": str(gg.iloc[0]["direction_label"]),
            "continuation_stratum": str(gg.iloc[0]["continuation_stratum"]),
            "stage_observations": int(len(gg)),
            "candidate_detected": bool(first_cand is not None),
            "confirmed_transition_detected": bool(confirmed is not None),
        }
        if first_cand is not None:
            r = gg.iloc[first_cand]
            base.update({
                "first_candidate_bps": float(r["stage_bps"]),
                "first_candidate_stage_second": int(r["stage_second"]),
                "first_candidate_observable_second": int(r["state_observable_second"]),
                "first_candidate_score": safe_float(r["liquidity_failure_score"]),
            })
        else:
            base.update({"first_candidate_bps": math.nan, "first_candidate_stage_second": math.nan, "first_candidate_observable_second": math.nan, "first_candidate_score": math.nan})
        if confirmed is not None:
            r = gg.iloc[confirmed]
            prev = gg.iloc[confirmed - 1]
            obs_sec = int(r["state_observable_second"])
            approx_disp = safe_float(r["observable_directional_displacement_approx_bps"])
            base.update({
                "confirmation_prior_candidate_bps": float(prev["stage_bps"]),
                "first_confirmed_bps": float(r["stage_bps"]),
                "first_confirmed_stage_second": int(r["stage_second"]),
                "first_confirmed_observable_second": obs_sec,
                "first_confirmed_score": safe_float(r["liquidity_failure_score"]),
                "observable_directional_displacement_approx_bps": approx_disp,
            })
            for target in (50, 75, 100):
                hit = safe_float(m.get(f"hit_{target}_second")) if m is not None else math.nan
                base[f"seconds_remaining_to_{target}bp_from_observable"] = hit - obs_sec if math.isfinite(hit) else math.nan
                base[f"observable_before_{target}bp"] = bool(math.isfinite(hit) and obs_sec < hit)
                base[f"approx_bps_remaining_to_{target}"] = float(target - approx_disp) if math.isfinite(approx_disp) else math.nan
        else:
            base.update({
                "confirmation_prior_candidate_bps": math.nan, "first_confirmed_bps": math.nan,
                "first_confirmed_stage_second": math.nan, "first_confirmed_observable_second": math.nan,
                "first_confirmed_score": math.nan, "observable_directional_displacement_approx_bps": math.nan,
            })
            for target in (50, 75, 100):
                base[f"seconds_remaining_to_{target}bp_from_observable"] = math.nan
                base[f"observable_before_{target}bp"] = False
                base[f"approx_bps_remaining_to_{target}"] = math.nan
        for bps in (15, 20, 25, 30, 35):
            base[f"confirmed_by_{bps}bp"] = bool(confirmed is not None and float(base["first_confirmed_bps"]) <= bps)
            base[f"candidate_by_{bps}bp"] = bool(first_cand is not None and float(base["first_candidate_bps"]) <= bps)
        rows.append(base)
    return pd.DataFrame(rows)


def build_transition_pair_rows(assignments, event_summary):
    _, pd = require_numeric_stack()
    em = event_summary.set_index("event_id")
    metrics = [
        "candidate_detected", "confirmed_transition_detected",
        "candidate_by_20bp", "candidate_by_25bp", "candidate_by_30bp", "candidate_by_35bp",
        "confirmed_by_20bp", "confirmed_by_25bp", "confirmed_by_30bp", "confirmed_by_35bp",
    ]
    rows = []
    for role, aid in ANALYSIS_SPECS:
        a = assignments[assignments["analysis_id"] == aid]
        for pid, pg in a.groupby("parent_event_id", sort=False):
            pid = str(pid)
            if pid not in em.index:
                continue
            control_ids = [str(x) for x in pg["control_event_id"] if str(x) in em.index]
            if len(set(control_ids)) < 2:
                continue
            er = em.loc[pid]
            controls = em.loc[sorted(set(control_ids))]
            parent_date = str(er["onset_date"])
            direction = int(er["direction"])
            for feat in metrics:
                ev = float(bool(er[feat]))
                co = pd.to_numeric(controls[feat], errors="coerce").astype(float).mean()
                rows.append({
                    "analysis_id": aid, "analysis_role": role, "stage": "LIQUIDITY_STATE_TRANSITION",
                    "window_label": WINDOW_LABEL, "window_type": "EVENT_TRANSITION_SUMMARY", "feature": feat,
                    "parent_event_id": pid, "parent_event_date": parent_date, "direction": direction,
                    "event_value": ev, "control_value": float(co), "paired_difference": float(ev - co),
                })
            # Earlier first transition is better.  Undetected events receive a
            # transparent 36-bp sentinel so incidence and timing are not mixed silently.
            for feat in ("first_candidate_bps", "first_confirmed_bps"):
                evraw = safe_float(er[feat]); ev = evraw if math.isfinite(evraw) else UNDETECTED_SENTINEL_BPS
                cvals = pd.to_numeric(controls[feat], errors="coerce").fillna(UNDETECTED_SENTINEL_BPS).to_numpy(float)
                co = float(cvals.mean())
                rows.append({
                    "analysis_id": aid, "analysis_role": role, "stage": "LIQUIDITY_STATE_TRANSITION",
                    "window_label": WINDOW_LABEL, "window_type": "EVENT_TRANSITION_SUMMARY", "feature": feat + "_or_36",
                    "parent_event_id": pid, "parent_event_date": parent_date, "direction": direction,
                    "event_value": float(ev), "control_value": co, "paired_difference": float(ev - co),
                })
    return pd.DataFrame(rows)


def summarize_pair_subset(pair_rows, phase7e, label: str, mask):
    sub = pair_rows[mask].copy()
    if sub.empty:
        return sub
    z = phase7e.summarize_pair_rows(sub)
    z.insert(0, "sample_subset", label)
    return z


def build_discovery_validation(pair_rows, phase7e):
    _, pd = require_numeric_stack()
    d = pd.to_datetime(pair_rows["parent_event_date"]).dt.date
    parts = [
        summarize_pair_subset(pair_rows, phase7e, "DISCOVERY", d <= DISCOVERY_END_DATE),
        summarize_pair_subset(pair_rows, phase7e, "VALIDATION", d >= VALIDATION_START_DATE),
        summarize_pair_subset(pair_rows, phase7e, "FULL", pd.Series([True] * len(pair_rows), index=pair_rows.index)),
    ]
    return pd.concat([x for x in parts if x is not None and not x.empty], ignore_index=True) if any(x is not None and not x.empty for x in parts) else pd.DataFrame()


def build_direction_robustness(pair_rows, phase7e):
    _, pd = require_numeric_stack()
    parts = []
    for direction, label in ((1, "UP"), (-1, "DOWN")):
        x = summarize_pair_subset(pair_rows, phase7e, label, pair_rows["direction"].astype(int) == direction)
        if x is not None and not x.empty:
            parts.append(x)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_detection_curve(assignments, event_summary):
    _, pd = require_numeric_stack()
    em = event_summary.set_index("event_id")
    rows = []
    for role, aid in ANALYSIS_SPECS:
        a = assignments[assignments["analysis_id"] == aid]
        for split, pred in (
            ("DISCOVERY", lambda d: d <= DISCOVERY_END_DATE),
            ("VALIDATION", lambda d: d >= VALIDATION_START_DATE),
            ("FULL", lambda d: True),
        ):
            parent_rows = []
            for pid, pg in a.groupby("parent_event_id", sort=False):
                pid = str(pid)
                if pid not in em.index:
                    continue
                er = em.loc[pid]
                pdate = date.fromisoformat(str(er["onset_date"]))
                if not pred(pdate):
                    continue
                cids = sorted({str(x) for x in pg["control_event_id"] if str(x) in em.index})
                if len(cids) < 2:
                    continue
                controls = em.loc[cids]
                parent_rows.append((er, controls))
            for bps in STAGE_BPS:
                ev = []
                co = []
                cand_ev = []
                cand_co = []
                for er, controls in parent_rows:
                    fc = safe_float(er["first_confirmed_bps"])
                    fca = safe_float(er["first_candidate_bps"])
                    ev.append(float(math.isfinite(fc) and fc <= bps))
                    cand_ev.append(float(math.isfinite(fca) and fca <= bps))
                    cc = pd.to_numeric(controls["first_confirmed_bps"], errors="coerce")
                    ca = pd.to_numeric(controls["first_candidate_bps"], errors="coerce")
                    co.append(float(((cc.notna()) & (cc <= bps)).mean()))
                    cand_co.append(float(((ca.notna()) & (ca <= bps)).mean()))
                rows.append({
                    "analysis_id": aid, "analysis_role": role, "sample_split": split, "stage_bps": int(bps),
                    "parent_pairs": int(len(ev)),
                    "event_confirmed_cdf": float(sum(ev) / len(ev)) if ev else math.nan,
                    "matched_control_confirmed_cdf": float(sum(co) / len(co)) if co else math.nan,
                    "confirmed_cdf_difference": float(sum(ev) / len(ev) - sum(co) / len(co)) if ev else math.nan,
                    "event_candidate_cdf": float(sum(cand_ev) / len(cand_ev)) if cand_ev else math.nan,
                    "matched_control_candidate_cdf": float(sum(cand_co) / len(cand_co)) if cand_co else math.nan,
                    "candidate_cdf_difference": float(sum(cand_ev) / len(cand_ev) - sum(cand_co) / len(cand_co)) if cand_ev else math.nan,
                })
    return pd.DataFrame(rows)


def build_economic_timing(event_summary, assignments):
    np, pd = require_numeric_stack()
    primary_parent_ids = set(assignments[assignments["analysis_id"] == PRIMARY_ANALYSIS_ID]["parent_event_id"].astype(str))
    e = event_summary[event_summary["event_id"].astype(str).isin(primary_parent_ids)].copy()
    rows = []
    for split, mask in (
        ("DISCOVERY", pd.to_datetime(e["onset_date"]).dt.date <= DISCOVERY_END_DATE),
        ("VALIDATION", pd.to_datetime(e["onset_date"]).dt.date >= VALIDATION_START_DATE),
        ("FULL", pd.Series([True] * len(e), index=e.index)),
    ):
        g = e[mask].copy()
        det = g[g["confirmed_transition_detected"].astype(bool)].copy()
        row = {
            "sample_split": split, "eligible_75plus_parents": int(len(g)), "confirmed_transitions": int(len(det)),
            "confirmed_transition_rate": float(len(det) / len(g)) if len(g) else math.nan,
            "median_first_confirmed_bps": float(pd.to_numeric(det["first_confirmed_bps"], errors="coerce").median()) if len(det) else math.nan,
            "median_observable_displacement_approx_bps": float(pd.to_numeric(det["observable_directional_displacement_approx_bps"], errors="coerce").median()) if len(det) else math.nan,
            "median_seconds_remaining_to_75bp": float(pd.to_numeric(det["seconds_remaining_to_75bp_from_observable"], errors="coerce").median()) if len(det) else math.nan,
            "median_approx_bps_remaining_to_75": float(pd.to_numeric(det["approx_bps_remaining_to_75"], errors="coerce").median()) if len(det) else math.nan,
            "share_observable_before_75bp": float(det["observable_before_75bp"].astype(bool).mean()) if len(det) else math.nan,
        }
        h100 = det[pd.to_numeric(det["seconds_remaining_to_100bp_from_observable"], errors="coerce").notna()].copy()
        row["detected_100plus_events"] = int(len(h100))
        row["median_seconds_remaining_to_100bp"] = float(pd.to_numeric(h100["seconds_remaining_to_100bp_from_observable"], errors="coerce").median()) if len(h100) else math.nan
        row["median_approx_bps_remaining_to_100"] = float(pd.to_numeric(h100["approx_bps_remaining_to_100"], errors="coerce").median()) if len(h100) else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_transition_gate(discovery_validation, detection_curve, direction):
    _, pd = require_numeric_stack()
    rows = []
    # Predeclared matched-event validation gate; this is NOT an all-market alpha gate.
    # Require later-sample event transition incidence to exceed matched controls by 15pp,
    # at least half of 75+ parents to confirm by 35bp, and both directions to have a
    # positive matched incidence difference in the full sample.
    val = detection_curve[(detection_curve["analysis_id"] == PRIMARY_ANALYSIS_ID) & (detection_curve["sample_split"] == "VALIDATION") & (detection_curve["stage_bps"] == 35)]
    if len(val):
        r = val.iloc[0]
        rate = safe_float(r["event_confirmed_cdf"]); ctrl = safe_float(r["matched_control_confirmed_cdf"]); diff = safe_float(r["confirmed_cdf_difference"]); n = int(r["parent_pairs"])
    else:
        rate = ctrl = diff = math.nan; n = 0
    dsub = direction[(direction["analysis_id"] == PRIMARY_ANALYSIS_ID) & (direction["feature"] == "confirmed_transition_detected")]
    dir_diffs = {str(x["sample_subset"]): safe_float(x["median_paired_difference"]) for _, x in dsub.iterrows()}
    direction_ok = all(k in dir_diffs and dir_diffs[k] > 0 for k in ("UP", "DOWN"))
    passed = bool(n >= 15 and math.isfinite(rate) and rate >= 0.50 and math.isfinite(diff) and diff >= 0.15 and direction_ok)
    rows.append({
        "gate": "MATCHED_EVENT_TRANSITION_VALIDATION", "passed": passed,
        "validation_parent_pairs": n, "validation_event_confirmed_by_35": rate,
        "validation_matched_control_confirmed_by_35": ctrl, "validation_rate_difference": diff,
        "up_full_median_paired_difference": dir_diffs.get("UP", math.nan),
        "down_full_median_paired_difference": dir_diffs.get("DOWN", math.nan),
        "rule": "validation n>=15; event confirmed_by_35>=0.50; event-control difference>=0.15; UP and DOWN matched incidence differences >0",
        "interpretation": "Matched-event mechanism validation only; not prospective/all-market alpha.",
    })
    return pd.DataFrame(rows)


def check_invariants(event_stages, profiles, scored, baselines, event_summary, assignments, quality, mismatch35, mismatch10, level_summary, partial_scope):
    _, pd = require_numeric_stack()
    rows = []
    def add(name, passed, observed, expected, note=""):
        rows.append({"check": name, "passed": bool(passed), "observed": str(observed), "expected": str(expected), "note": note})
    add("transition_stage_ids_unique", not event_stages["event_id"].duplicated().any(), int(event_stages["event_id"].nunique()), len(event_stages))
    add("derived_35bp_matches_frozen_catalog", mismatch35 == 0, mismatch35, 0)
    add("derived_10bp_matches_frozen_fixed10", mismatch10 == 0, mismatch10, 0)
    monotonic_bad = 0
    for r in event_stages.to_dict("records"):
        vals = [int(r[f"stage__{stage_label(b)}"]) for b in STAGE_BPS]
        monotonic_bad += int(any(vals[i] > vals[i + 1] for i in range(len(vals) - 1)))
    add("dense_threshold_times_monotonic", monotonic_bad == 0, monotonic_bad, 0)
    dup = int(profiles.duplicated(["event_id", "stage", "window_label"]).sum()) if not profiles.empty else 0
    add("profiles_unique", dup == 0, dup, 0)
    valid = profiles[profiles["window_valid"].fillna(False).astype(bool)] if not profiles.empty else profiles
    add("valid_profiles_present", len(valid) > 0, len(valid), ">0")
    add("frozen_level_initial_qty_positive", int(level_summary["nonpositive_initial_qty"]) == 0, int(level_summary["nonpositive_initial_qty"]), 0)
    add("level_roles_known", set(level_summary["roles"]).issubset({"SUPPORT", "OPPOSING"}), sorted(level_summary["roles"]), "SUPPORT/OPPOSING")
    pr = quality[quality["analysis_id"] == PRIMARY_ANALYSIS_ID]
    good_balance = len(pr) == 1 and int(pr.iloc[0]["accepted_parents"]) >= 30 and safe_float(pr.iloc[0]["max_abs_smd_recomputed"]) <= 0.2
    add("primary_frozen_match_balance_good", good_balance, f"N={int(pr.iloc[0]['accepted_parents']) if len(pr) else 0} max|SMD|={safe_float(pr.iloc[0]['max_abs_smd_recomputed']) if len(pr) else math.nan:.4f}", "N>=30, max|SMD|<=0.2")
    min_bn = int(pd.to_numeric(baselines["control_n"], errors="coerce").min()) if len(baselines) else 0
    scales_good = bool(len(baselines) == len(STAGE_BPS) * len(CORE_FEATURES) and (pd.to_numeric(baselines["control_robust_scale"], errors="coerce") > 0).all())
    if partial_scope:
        add("discovery_control_baselines_sufficient", True, f"PARTIAL_SCOPE min_n={min_bn} rows={len(baselines)}", "SKIPPED until full matched-control scope", "Partial smoke runs validate raw replay/plumbing only.")
        add("state_scores_use_all_core_features", True, "PARTIAL_SCOPE", "SKIPPED until full matched-control scope")
    else:
        add("discovery_control_baselines_sufficient", min_bn >= MIN_BASELINE_N and scales_good, f"min_n={min_bn} rows={len(baselines)}", f"min_n>={MIN_BASELINE_N}; rows={len(STAGE_BPS)*len(CORE_FEATURES)}; positive scales")
        if not scored.empty:
            add("state_scores_use_all_core_features", int((scored["state_valid_core_features"] != len(CORE_FEATURES)).sum()) == 0, int((scored["state_valid_core_features"] != len(CORE_FEATURES)).sum()), 0)
    bad_confirm = 0
    for r in event_summary.to_dict("records"):
        if bool(r["confirmed_transition_detected"]):
            if not bool(r["candidate_detected"]) or safe_float(r["first_confirmed_bps"]) < safe_float(r["first_candidate_bps"]):
                bad_confirm += 1
            if safe_float(r["first_confirmed_observable_second"]) != safe_float(r["first_confirmed_stage_second"]) + OBSERVATION_HORIZON_SEC:
                bad_confirm += 1
    add("confirmed_transition_after_candidate_and_observable_delay", bad_confirm == 0, bad_confirm, 0)
    if not partial_scope:
        needed = set(assignments[assignments["analysis_id"] == PRIMARY_ANALYSIS_ID]["parent_event_id"].astype(str))
        have = set(event_summary["event_id"].astype(str))
        add("primary_parent_transition_summaries_present", needed.issubset(have), len(needed & have), len(needed))
    return pd.DataFrame(rows)


def build_findings(meta: Mapping[str, Any], gate, timing) -> str:
    lines = [
        "# Phase 7G — Liquidity-State Transition Timing", "",
        f"- Builder: **{BUILDER_VERSION}**", f"- Status: **{meta['status']}**",
        f"- Primary analysis: `{PRIMARY_ANALYSIS_ID}`", f"- Physical scan: **10–35 bps in 1-bp steps**",
        f"- Exact-depth observation horizon: **{OBSERVATION_HORIZON_SEC}s**", "",
        "## Frozen state rule", "",
        f"Candidate state: all four core mechanics measurable; >= {MIN_ALIGNED_FEATURES}/4 expected-direction; >= {MIN_STRONG_FEATURES}/4 signed robust-z >= {STRONG_SIGNED_Z:.2f}; median signed robust-z >= {COMPOSITE_SIGNED_Z:.2f}.",
        "Confirmed transition: candidate state at two consecutive distinct stage observations.  Live-observable confirmation time is confirmation stage + 3 seconds.",
        "Discovery-period PRIMARY 35_ONLY controls calibrate the stage-specific robust baselines; 75+ outcomes are not used to tune the rule.", "",
        "## Interpretation discipline", "",
        "This phase remains retrospective matched-event mechanism/timing research.  A successful matched-event validation gate does not establish all-market precision, false-positive rate, or tradable alpha.", "",
    ]
    if gate is not None and len(gate):
        r = gate.iloc[0]
        lines += ["## Matched-event validation gate", "", f"- Result: **{'PASS' if bool(r['passed']) else 'FAIL'}**", f"- Validation confirmed-by-35 rate: **{safe_float(r['validation_event_confirmed_by_35']):.3f}** vs matched controls **{safe_float(r['validation_matched_control_confirmed_by_35']):.3f}**", ""]
    if timing is not None and len(timing):
        f = timing[timing["sample_split"] == "FULL"]
        if len(f):
            r = f.iloc[0]
            lines += ["## Primary 75+ economic timing", "", f"- Confirmed transition rate: **{safe_float(r['confirmed_transition_rate']):.3f}**", f"- Median confirmed physical stage: **{safe_float(r['median_first_confirmed_bps']):.1f} bps**", f"- Median approximate displacement when observable: **{safe_float(r['median_observable_displacement_approx_bps']):.1f} bps**", f"- Median time remaining to 75 bps: **{safe_float(r['median_seconds_remaining_to_75bp']):.1f}s**", ""]
    return "\n".join(lines) + "\n"


def self_test() -> None:
    np, pd = require_numeric_stack()
    # Baseline/scoring test: event-like state should align all four signed z values.
    base_rows = []
    for b in STAGE_BPS:
        for feat, sign in CORE_FEATURES:
            base_rows.append({"stage_bps": b, "stage": stage_label(b), "feature": feat, "expected_sign": sign, "control_n": 50, "control_median": 0.5, "control_robust_scale": 0.1, "baseline_scope": "test"})
    baselines = pd.DataFrame(base_rows)
    r = {"event_id": "E", "onset_date": "2026-06-01", "direction": 1, "direction_label": "UP", "continuation_stratum": "75_ONLY", "stage": "HIT_20BP", "stage_bps": 20, "stage_second": 100, "window_label": WINDOW_LABEL, "window_valid": True, "directional_mid_move_bps": 1.0}
    for feat, sign in CORE_FEATURES:
        r[feat] = 0.65 if sign > 0 else 0.35
    r["support__retained_initial_quote_ratio"] = 0.8
    s = score_profiles(pd.DataFrame([r]), baselines)
    assert bool(s.iloc[0]["state_candidate"])
    assert int(s.iloc[0]["state_aligned_features"]) == 4
    assert int(s.iloc[0]["state_strong_features"]) == 4
    assert int(s.iloc[0]["state_observable_second"]) == 103
    print("SELF-TEST PASS")
    print("Validated frozen signed-z state rule, 3-second observability delay, and transition-score directionality.")


def parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(description="Phase 7G liquidity-state transition timing")
    p.add_argument("--symbol", default=DEFAULT_SYMBOL)
    p.add_argument("--phase6a-script", type=Path, default=DEFAULT_PHASE6A_SCRIPT)
    p.add_argument("--phase7a-script", type=Path, default=DEFAULT_PHASE7A_SCRIPT)
    p.add_argument("--phase7c-script", type=Path, default=DEFAULT_PHASE7C_SCRIPT)
    p.add_argument("--phase7d1-script", type=Path, default=DEFAULT_PHASE7D1_SCRIPT)
    p.add_argument("--phase7e-script", type=Path, default=DEFAULT_PHASE7E_SCRIPT)
    p.add_argument("--phase7f-script", type=Path, default=DEFAULT_PHASE7F_SCRIPT)
    p.add_argument("--execution-root", type=Path, default=DEFAULT_EXECUTION_ROOT)
    p.add_argument("--catalog-root", type=Path, default=DEFAULT_CATALOG_ROOT)
    p.add_argument("--phase7d1-root", type=Path, default=DEFAULT_PHASE7D1_ROOT)
    p.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--price-scale", type=float, default=10.0)
    p.add_argument("--max-book-age-ms", type=int, default=2000)
    p.add_argument("--max-trade-stale-sec", type=int, default=5)
    p.add_argument("--max-onset-dates", type=int, default=None, help="Partial smoke scope only; full transition study requires all onset dates")
    p.add_argument("--force-day", action="store_true", help="Rebuild matching per-day Phase-7G raw replay checkpoints")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test(); return 0
    np, pd, pa, pq = require_stack()
    phase6a, phase7a, phase7c, phase7d1, phase7e, phase7f = load_frozen_stack(args)

    print("Building Phase 7G liquidity-state transition timing...")
    print(f"Builder version      : {BUILDER_VERSION}")
    print(f"Frozen Phase-6A      : {phase6a.BUILDER_VERSION}")
    print(f"Frozen Phase-7A      : {phase7a.BUILDER_VERSION}")
    print(f"Frozen Phase-7C      : {phase7c.BUILDER_VERSION}")
    print(f"Frozen Phase-7D.1    : {phase7d1.BUILDER_VERSION}")
    print(f"Frozen Phase-7E      : {phase7e.BUILDER_VERSION}")
    print(f"Frozen Phase-7F      : {phase7f.BUILDER_VERSION}")
    print(f"Physical scan        : {STAGE_BPS[0]}..{STAGE_BPS[-1]} bps in 1-bp steps")
    print(f"Observation horizon  : {OBSERVATION_HORIZON_SEC}s")
    print(f"Primary analysis     : {PRIMARY_ANALYSIS_ID}")
    print("Research role        : retrospective matched-event state-transition timing; not prospective alpha")

    print("[1/10] Loading frozen matched assignments and balance...")
    assignments, quality, events, fixed10_map = load_assignments_events(args, phase7c, phase7e, phase7f)
    for r in quality.to_dict("records"):
        print(f"      {r['analysis_role']:<18} parents={int(r['accepted_parents']):3d} max|SMD|={safe_float(r['max_abs_smd_recomputed']):.3f} {r['balance_grade_recomputed']}")

    print("[2/10] Reconstructing dense 10..35-bp physical first-passage stages...")
    event_stages, mismatch35, mismatch10 = build_dense_stage_table(args, phase7a, phase7f, events, fixed10_map)
    print(f"      selected physical events={len(event_stages):,}")
    print(f"      35bp catalog mismatches={mismatch35} | 10bp fixed-departure mismatches={mismatch10}")

    print("[3/10] Discovering canonical depth/trade partitions...")
    family_maps = {fam: phase6a.discover_family_days(args.canonical_root, fam) for fam in ("depth_updates", "book_snapshots", "trades")}
    all_dates = sorted(event_stages["onset_date"].astype(str).unique())
    selected_dates = list(all_dates)
    if args.max_onset_dates is not None:
        selected_dates = selected_dates[: max(0, int(args.max_onset_dates))]
    partial_scope = len(selected_dates) < len(all_dates)
    print(f"      onset dates={len(selected_dates)}" + (f" of {len(all_dates)} (PARTIAL SMOKE SCOPE)" if partial_scope else ""))

    print("[4/10] Replaying authoritative depth and materializing 3-second exact mechanics...")
    profile_parts = []
    level_summary = {"row_count": 0, "nonpositive_initial_qty": 0, "roles": set()}
    reused_days = 0
    for i, d in enumerate(selected_dates, 1):
        de = event_stages[event_stages["onset_date"] == d].copy()
        prof, reused, diag = process_onset_date(d, de, phase6a, phase7e, family_maps, args)
        profile_parts.append(prof); reused_days += int(reused)
        level_summary["row_count"] += int(diag.get("level_rows") or 0)
        level_summary["nonpositive_initial_qty"] += int(diag.get("nonpositive_initial_qty") or 0)
        level_summary["roles"].update(diag.get("level_roles") or [])
        valid = int(prof["window_valid"].fillna(False).sum()) if not prof.empty else 0
        line = f"      [{i:02d}/{len(selected_dates):02d}] {d}: events={len(de):3d} windows={len(prof):5d} valid={valid:5d} {'REUSED' if reused else 'BUILT'}"
        if not reused:
            line += f" | trade={safe_float(diag.get('trade_aggregation_seconds')):.1f}s replay={safe_float(diag.get('raw_replay_seconds')):.1f}s tracked_msgs={int(diag.get('tracked_depth_messages') or 0):,}"
        print(line)
        del de
        gc.collect()
        try: pa.default_memory_pool().release_unused()
        except Exception: pass

    profiles = pd.concat(profile_parts, ignore_index=True) if profile_parts else pd.DataFrame()
    processed_events = event_stages[event_stages["onset_date"].isin(selected_dates)].copy()
    processed_ids = set(processed_events["event_id"].astype(str))
    a_proc = assignments[
        assignments["parent_event_id"].astype(str).isin(processed_ids)
        & assignments["control_event_id"].astype(str).isin(processed_ids)
    ].copy()
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_parquet_atomic(processed_events, args.output_root / "selected_transition_event_stages.parquet")
    write_parquet_atomic(profiles, args.output_root / "transition_stage_profiles.parquet")

    print("[5/10] Fitting frozen discovery-control state baselines and scoring every stage...")
    # For partial smoke runs, baseline fitting can be incomplete by design; the
    # raw replay is still validated, but transition interpretation is withheld.
    baselines = fit_discovery_control_baselines(profiles, a_proc)
    scored = score_profiles(profiles, baselines) if len(baselines) else pd.DataFrame()
    event_summary = build_event_transition_summary(scored, processed_events) if not scored.empty else pd.DataFrame()
    write_parquet_atomic(scored, args.output_root / "transition_scored_stage_profiles.parquet")
    write_parquet_atomic(event_summary, args.output_root / "event_transition_summary.parquet")

    print("[6/10] Building matched transition incidence/timing contrasts...")
    pair_rows = build_transition_pair_rows(a_proc, event_summary) if not event_summary.empty else pd.DataFrame()
    if not pair_rows.empty:
        write_parquet_atomic(pair_rows, args.output_root / "transition_pair_rows.parquet")
    summary = phase7e.summarize_pair_rows(pair_rows) if not pair_rows.empty else pd.DataFrame()
    dv = build_discovery_validation(pair_rows, phase7e) if not pair_rows.empty else pd.DataFrame()
    direction = build_direction_robustness(pair_rows, phase7e) if not pair_rows.empty else pd.DataFrame()
    curve = build_detection_curve(a_proc, event_summary) if not event_summary.empty else pd.DataFrame()
    timing = build_economic_timing(event_summary, a_proc) if not event_summary.empty else pd.DataFrame()
    gate = build_transition_gate(dv, curve, direction) if not partial_scope and not dv.empty else pd.DataFrame()

    print("[7/10] Checking transition-study invariants...")
    checks = check_invariants(processed_events, profiles, scored, baselines, event_summary, a_proc, quality, mismatch35, mismatch10, level_summary, partial_scope)
    for r in checks.to_dict("records"):
        print(f"      {'PASS' if r['passed'] else 'FAIL'} {r['check']}: observed={r['observed']} expected={r['expected']}")
    failed = checks[~checks["passed"].astype(bool)]
    status = "SMOKE_PASS" if partial_scope and failed.empty else ("PASS" if failed.empty else "FAIL")

    print("[8/10] Writing reports...")
    args.report_root.mkdir(parents=True, exist_ok=True)
    quality.to_csv(args.report_root / "transition_analysis_quality.csv", index=False)
    baselines.to_csv(args.report_root / "transition_discovery_control_baselines.csv", index=False)
    summary.to_csv(args.report_root / "transition_matched_contrasts.csv", index=False)
    dv.to_csv(args.report_root / "transition_discovery_validation.csv", index=False)
    direction.to_csv(args.report_root / "transition_direction_robustness.csv", index=False)
    curve.to_csv(args.report_root / "transition_detection_curve.csv", index=False)
    timing.to_csv(args.report_root / "transition_economic_timing.csv", index=False)
    gate.to_csv(args.report_root / "transition_validation_gate.csv", index=False)
    checks.to_csv(args.report_root / "transition_invariants.csv", index=False)

    meta = {
        "builder_version": BUILDER_VERSION, "schema_name": SCHEMA_NAME, "status": status,
        "generated_at": utc_now_iso(), "symbol": args.symbol, "stage_bps": list(STAGE_BPS),
        "observation_horizon_sec": OBSERVATION_HORIZON_SEC, "primary_analysis_id": PRIMARY_ANALYSIS_ID,
        "discovery_end_date": DISCOVERY_END_DATE.isoformat(), "validation_start_date": VALIDATION_START_DATE.isoformat(),
        "state_rule": {"strong_signed_z": STRONG_SIGNED_Z, "composite_signed_z": COMPOSITE_SIGNED_Z,
                       "min_aligned_features": MIN_ALIGNED_FEATURES, "min_strong_features": MIN_STRONG_FEATURES,
                       "confirmation": "two consecutive distinct stage observations"},
        "selected_events": int(processed_events["event_id"].nunique()), "profile_rows": int(len(profiles)),
        "valid_profile_rows": int(profiles["window_valid"].fillna(False).sum()) if not profiles.empty else 0,
        "event_transition_rows": int(len(event_summary)), "reused_onset_date_checkpoints": int(reused_days),
        "level_rows_audited": int(level_summary["row_count"]), "partial_scope": bool(partial_scope),
        "interpretation": "Retrospective matched-event state-transition timing only; not all-market prospective alpha or a trading rule.",
    }
    atomic_write_json(args.output_root / "metadata.json", meta)
    atomic_write_text(args.report_root / "transition_findings.md", build_findings(meta, gate, timing))

    print("[9/10] Transition summary")
    if not partial_scope and len(gate):
        print(f"      matched-event validation gate={'PASS' if bool(gate.iloc[0]['passed']) else 'FAIL'}")
        print(f"      validation confirmed_by_35: event={safe_float(gate.iloc[0]['validation_event_confirmed_by_35']):.3f} matched_control={safe_float(gate.iloc[0]['validation_matched_control_confirmed_by_35']):.3f}")
    if len(timing):
        full = timing[timing["sample_split"] == "FULL"]
        if len(full):
            r = full.iloc[0]
            print(f"      primary confirmed rate={safe_float(r['confirmed_transition_rate']):.3f} median stage={safe_float(r['median_first_confirmed_bps']):.1f}bps median remaining-to-75={safe_float(r['median_seconds_remaining_to_75bp']):.1f}s")
    print(f"      status={status} selected events={meta['selected_events']:,} profiles={meta['profile_rows']:,} valid={meta['valid_profile_rows']:,}")
    print(f"      reports={args.report_root}")

    print("[10/10] Interpretation gate")
    if status == "FAIL":
        print("ERROR: one or more Phase-7G invariants failed; do not interpret transition reports.")
        return 2
    if partial_scope:
        print("Partial smoke scope completed. Do not interpret state-transition timing until the full onset-date run is complete.")
    else:
        print("Full Phase 7G completed. Interpret validation, direction robustness and economic timing before any prospective translation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
