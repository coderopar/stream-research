#!/usr/bin/env python3
"""Phase 7F — continuation-threshold and liquidity-failure calibration.

Scientific role
---------------
Phase 7E established an exact raw-depth continuation mechanism most clearly
around the hindsight HIT_35BP stage.  Phase 7F asks whether the same mechanism
is already observable earlier, and whether 35 bps is a scientifically justified
breaking point rather than an arbitrary research threshold.

The experiment deliberately preserves the frozen Phase-7D.1 matched physical
comparisons used in Phase 7E:

    PRIMARY            75_PLUS vs matched 35_ONLY / REGIME_MATCHED / PRIMARY
    ROBUST_75_RELAXED  75_PLUS vs matched 35_ONLY / REGIME_MATCHED / RELAXED
    ROBUST_50_STRICT   50_PLUS vs matched 35_ONLY / STRICT_STATE_MATCHED / RELAXED
    ROBUST_50_REGIME   50_PLUS vs matched 35_ONLY / REGIME_MATCHED / PRIMARY

Every parent and every 35_ONLY control reaches all calibration stages, so the
comparison population remains fixed while displacement advances through:

    10, 15, 20, 25, 30, 35 bps

At each stage the exact price levels visible within 5 bps are frozen and the
frozen Phase-7E mechanics are followed for 1/3/5/10 seconds.  The primary
calibration horizon is fixed at 3 seconds before inspecting any Phase-7F
result; other horizons are robustness only.

This remains retrospective mechanism/timing research.  The physical extremum,
direction, continuation labels, and 35_ONLY controls are hindsight-defined.
The script does NOT claim a prospective signal or tradable edge.

A chronological discovery/validation split is frozen at the midpoint of the
rich-data interval:

    DISCOVERY   parent onset date <= 2026-06-17
    VALIDATION  parent onset date >= 2026-06-18

The script reports an explicit, rule-based discovery gate and then evaluates
the same stage on the validation half.  It does not optimize a classifier.
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
SCHEMA_NAME = "large_move_continuation_threshold_liquidity_calibration_v1"
DEFAULT_SYMBOL = "BTCUSDT"

STAGE_BPS = (10, 15, 20, 25, 30, 35)
STAGES = tuple(f"HIT_{x}BP" for x in STAGE_BPS)
PRIMARY_HORIZON_SEC = 3
FIXED_HORIZONS_SEC = (1, 3, 5, 10)
PRIMARY_WINDOW_LABEL = f"FWD_{PRIMARY_HORIZON_SEC}S"
DISCOVERY_END_DATE = date(2026, 6, 17)
VALIDATION_START_DATE = date(2026, 6, 18)
MIN_DISCOVERY_PAIRS = 20
MIN_VALIDATION_PAIRS = 15
DISCOVERY_STRONG_RSD = 0.35
VALIDATION_STRONG_RSD = 0.25
MIN_EXPECTED_SIGN_FEATURES = 3
MIN_STRONG_FEATURES = 2
MIN_ALIGNED_PAIR_SHARE = 0.60

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()
DEFAULT_PHASE6A_SCRIPT = REPO_ROOT / "scripts" / "perps" / "research" / "06a_build_execution_foundation.py"
DEFAULT_PHASE7A_SCRIPT = REPO_ROOT / "scripts" / "perps" / "research" / "07a_build_large_move_event_catalog.py"
DEFAULT_PHASE7C_SCRIPT = REPO_ROOT / "scripts" / "perps" / "research" / "07c_build_large_move_participant_behavior_atlas.py"
DEFAULT_PHASE7D1_SCRIPT = REPO_ROOT / "scripts" / "perps" / "research" / "07d1_build_common_support_matched_controls.py"
DEFAULT_PHASE7E_SCRIPT = REPO_ROOT / "scripts" / "perps" / "research" / "07e_build_raw_depth_continuation_mechanics.py"
DEFAULT_EXECUTION_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "execution_foundation_v1"
DEFAULT_CATALOG_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / "large_move_event_catalog_v1"
DEFAULT_PHASE7D1_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / "large_move_common_support_matched_controls_v1"
DEFAULT_CANONICAL_ROOT = REPO_ROOT / "data" / "canonical" / "market_events_v1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / SCHEMA_NAME
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "perps_continuation_threshold_liquidity_calibration"

# Frozen from Phase 7E.  Do not tune these choices on Phase-7F outcomes.
ANALYSIS_SPECS = (
    ("PRIMARY", "75_PLUS|THIRTYFIVE_ONLY|REGIME_MATCHED|PRIMARY"),
    ("ROBUST_75_RELAXED", "75_PLUS|THIRTYFIVE_ONLY|REGIME_MATCHED|RELAXED"),
    ("ROBUST_50_STRICT", "50_PLUS|THIRTYFIVE_ONLY|STRICT_STATE_MATCHED|RELAXED"),
    ("ROBUST_50_REGIME", "50_PLUS|THIRTYFIVE_ONLY|REGIME_MATCHED|PRIMARY"),
)
PRIMARY_ANALYSIS_ID = ANALYSIS_SPECS[0][1]

# Small, frozen mechanism set.  expected_sign is event minus matched controls.
CORE_FEATURES = (
    ("opposing__peak_initial_queue_loss_share", +1),
    ("opposing__durable_unrecovered_loss_share_initial", +1),
    ("opposing__retained_initial_quote_ratio", -1),
    ("liquidity_retention_asymmetry", +1),
)
CONTEXT_FEATURES = (
    "support__same_price_refill_to_depletion_ratio",
    "support__retained_initial_quote_ratio",
    "forward_directional_aggressive_imbalance",
    "directional_aggressive_quote",
    "directional_mid_move_bps",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def require_numeric_stack():
    try:
        import numpy as np
        import pandas as pd
    except Exception as exc:
        raise SystemExit("Phase 7F requires numpy and pandas. Activate the project .venv first.") from exc
    return np, pd


def require_stack():
    np, pd = require_numeric_stack()
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:
        raise SystemExit("Phase 7F Parquet I/O requires pyarrow. Activate the project .venv first.") from exc
    return np, pd, pa, pq


def load_module(path: Path, name: str):
    import importlib.util
    if not path.exists():
        raise RuntimeError(f"Missing required frozen script: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7f-", suffix=path.suffix, dir=str(path.parent))
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
    fd, tmp = tempfile.mkstemp(prefix="tmp7f-", suffix=".parquet", dir=str(path.parent))
    os.close(fd)
    try:
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp, compression="zstd")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def safe_float(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


def sha256_json(payload: Any) -> str:
    txt = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def file_signature(paths: Sequence[Path]) -> list[dict[str, Any]]:
    out = []
    for p in sorted(paths):
        st = p.stat()
        out.append({"path": str(p.resolve()), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)})
    return out


def stage_label(bps: int) -> str:
    return f"HIT_{int(bps)}BP"


def stage_bps_from_label(stage: str) -> int:
    return int(str(stage).replace("HIT_", "").replace("BP", ""))


def directional_moves_bps(prices, anchor: float, direction: int):
    np, _ = require_numeric_stack()
    p = np.asarray(prices, dtype=float)
    if int(direction) > 0:
        return (p / float(anchor) - 1.0) * 10_000.0
    return (1.0 - p / float(anchor)) * 10_000.0


def first_passage_second(seconds, prices, anchor: float, direction: int, threshold_bps: float) -> int | None:
    np, _ = require_numeric_stack()
    sec = np.asarray(seconds, dtype=np.int64)
    px = np.asarray(prices, dtype=float)
    move = directional_moves_bps(px, anchor, direction)
    ok = np.isfinite(move) & (move + 1e-12 >= float(threshold_bps))
    idx = np.flatnonzero(ok)
    return int(sec[int(idx[0])]) if len(idx) else None


def load_frozen_modules(args):
    phase6a = load_module(args.phase6a_script, "phase6a_frozen_for_7f")
    phase7a = load_module(args.phase7a_script, "phase7a_frozen_for_7f")
    phase7c = load_module(args.phase7c_script, "phase7c_frozen_for_7f")
    phase7d1 = load_module(args.phase7d1_script, "phase7d1_frozen_for_7f")
    phase7e = load_module(args.phase7e_script, "phase7e_frozen_for_7f")
    expected = {
        "Phase-6A": (getattr(phase6a, "BUILDER_VERSION", None), "1.0.4"),
        "Phase-7A": (getattr(phase7a, "BUILDER_VERSION", None), "1.0.0"),
        "Phase-7C": (getattr(phase7c, "BUILDER_VERSION", None), "1.0.2"),
        "Phase-7D.1": (getattr(phase7d1, "BUILDER_VERSION", None), "1.0.0"),
        "Phase-7E": (getattr(phase7e, "BUILDER_VERSION", None), "1.0.3"),
    }
    bad = [f"{k}: found {found!r}, expected {want}" for k, (found, want) in expected.items() if found != want]
    if bad:
        raise RuntimeError("Frozen builder mismatch:\n  " + "\n  ".join(bad))
    if getattr(phase6a, "OUTPUT_SCHEMA_NAME", None) != "perps_execution_foundation_v1":
        raise RuntimeError("Unexpected Phase-6A output schema")
    return phase6a, phase7a, phase7c, phase7d1, phase7e


def load_assignments(args, phase7e):
    # Reuse the audited 7E selection and balance logic exactly.
    proxy = argparse.Namespace(phase7d1_root=args.phase7d1_root)
    a = phase7e.load_selected_assignments(proxy)
    q = phase7e.recompute_analysis_quality(a)
    return a, q


def load_selected_events(args, phase7c, assignments):
    _, pd, _, pq = require_stack()
    ep = args.catalog_root / "physical_events.parquet"
    op = args.catalog_root / "onset_candidates.parquet"
    if not ep.exists() or not op.exists():
        raise RuntimeError(f"Missing Phase-7A catalog artifacts: {ep} / {op}")
    events = pq.read_table(ep).to_pandas()
    onsets = pq.read_table(op).to_pandas()
    if "symbol" in events.columns:
        events = events[events["symbol"].astype(str) == args.symbol].copy()
    em = phase7c.build_event_metadata(events)
    em["event_id"] = em["event_id"].astype(str)
    needed = set(assignments["parent_event_id"].astype(str)) | set(assignments["control_event_id"].astype(str))
    em = em[em["event_id"].isin(needed)].copy()
    missing = needed - set(em["event_id"])
    if missing:
        raise RuntimeError(f"Missing selected catalog events; first={sorted(missing)[:5]}")
    onsets["event_id"] = onsets["event_id"].astype(str)
    fixed10 = onsets[(onsets["event_id"].isin(needed)) & (onsets["onset_method"].astype(str) == "FIXED_10BP_DEPARTURE")].copy()
    fixed10_map = {
        str(r["event_id"]): int(r["candidate_second"])
        for r in fixed10.to_dict("records")
        if bool(r.get("candidate_found")) and r.get("candidate_second") is not None and math.isfinite(safe_float(r.get("candidate_second")))
    }
    keep = [
        "event_id", "onset_date", "direction", "direction_label", "continuation_stratum",
        "onset_second", "onset_price", "hit_35_second", "hit_50_second", "hit_75_second", "hit_100_second",
        "time_to_35_sec", "time_to_50_sec", "time_to_75_sec", "time_to_100_sec",
    ]
    for c in keep:
        if c not in em.columns:
            em[c] = math.nan
    return em[keep].copy(), fixed10_map


def read_execution_days(args, phase7a, days: Sequence[date]):
    np, pd, _, pq = require_stack()
    cols = ["receive_second", "last_trade_price", "trade_price_ffill", "mid_price", "book_valid"]
    parts = []
    for d in days:
        p = phase7a.execution_day_path(args.execution_root, args.symbol, d)
        if not p.exists():
            raise RuntimeError(f"Missing Phase-6A execution partition: {p}")
        names = set(pq.read_schema(p).names)
        use = [c for c in cols if c in names]
        f = pq.read_table(p, columns=use).to_pandas()
        for c in cols:
            if c not in f.columns:
                f[c] = False if c == "book_valid" else np.nan
        parts.append(f[cols])
    frame = pd.concat(parts, ignore_index=True)
    frame = frame.sort_values("receive_second", kind="mergesort").drop_duplicates("receive_second", keep="last").reset_index(drop=True)
    return frame


def build_threshold_stage_table(args, phase7a, events, fixed10_map):
    np, pd, pa, _ = require_stack()
    rows = []
    mismatch35 = 0
    mismatch10 = 0
    for i, (onset_date, g) in enumerate(events.groupby("onset_date", sort=True), 1):
        base = date.fromisoformat(str(onset_date))
        max_hit = int(pd.to_numeric(g["hit_35_second"], errors="coerce").dropna().max())
        last_day = datetime.fromtimestamp(max_hit, tz=timezone.utc).date()
        # Include the previous execution day when available so the bounded
        # last-trade ffill at the UTC boundary reproduces the catalog's
        # cross-day price context.  The first rich-data day simply starts at
        # its own boundary when no prior execution partition exists.
        days = []
        prev = base - timedelta(days=1)
        if phase7a.execution_day_path(args.execution_root, args.symbol, prev).exists():
            days.append(prev)
        days.append(base)
        if last_day > base:
            days.append(base + timedelta(days=1))
        frame = read_execution_days(args, phase7a, days)
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
                hs = first_passage_second(path_sec, path_px, float(ev["onset_price"]), int(ev["direction"]), float(bps))
                row[f"stage__{stage_label(bps)}"] = hs
            derived35 = row["stage__HIT_35BP"]
            if derived35 != hit35:
                mismatch35 += 1
            expected10 = fixed10_map.get(str(ev["event_id"]))
            if expected10 is not None and row["stage__HIT_10BP"] != expected10:
                mismatch10 += 1
            rows.append(row)
        print(f"      stage geometry [{i:02d}] {onset_date}: events={len(g):3d}")
        del frame, sec, price
        gc.collect()
        try:
            pa.default_memory_pool().release_unused()
        except Exception:
            pass
    out = pd.DataFrame(rows)
    if out["event_id"].duplicated().any():
        raise RuntimeError("Threshold stage table is not unique by event_id")
    for st in STAGES:
        if out[f"stage__{st}"].isna().any():
            raise RuntimeError(f"Selected 35+ event missing calibration stage {st}")
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
        "fixed_horizons_sec": FIXED_HORIZONS_SEC,
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


def process_onset_date(onset_date: str, day_events, phase6a, phase7e, family_maps, args):
    _, pd, _, pq = require_stack()
    base = date.fromisoformat(onset_date)
    stage_seconds = [int(r[f"stage__{st}"]) for r in day_events.to_dict("records") for st in STAGES]
    max_end_second = max(stage_seconds) + max(FIXED_HORIZONS_SEC)
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
            diag = {k: manifest.get(k) for k in (
                "trade_aggregation_seconds", "raw_replay_seconds", "depth_rows_decoded",
                "depth_levels_decoded", "tracked_depth_messages", "tracked_exact_changes",
                "level_rows", "nonpositive_initial_qty", "level_roles",
            )}
            return profiles, True, diag

    trade_t0 = time.perf_counter()
    trades = phase6a.aggregate_trades(day_files, args.symbol)
    trade_context = phase7e.TradeContext(trades)
    trade_seconds = time.perf_counter() - trade_t0

    anchors = []
    for ev in day_events.to_dict("records"):
        for st in STAGES:
            sec = int(ev[f"stage__{st}"])
            anchors.append(((sec + 1) * 1_000_000_000 - 1, st, ev))
    anchors.sort(key=lambda x: (x[0], str(x[2]["event_id"]), x[1]))

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
            tr, invalid = phase7e.make_tracker(ev, st, sec, book, trade_context, args.max_book_age_ms, float(args.price_scale))
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
            track_changes = bool(active) and any(
                tr.censored_ns is None and not tr.done and ts_ns > tr.anchor_end_ns for tr in active
            )
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
        profiles["calibration_schema_version"] = SCHEMA_VERSION
        profiles["day_input_signature"] = sig
        # Intuitive derived asymmetries: positive means the book favors continuation direction.
        profiles["liquidity_retention_asymmetry"] = (
            pd.to_numeric(profiles.get("support__retained_initial_quote_ratio"), errors="coerce")
            - pd.to_numeric(profiles.get("opposing__retained_initial_quote_ratio"), errors="coerce")
        )
        profiles["liquidity_unrecovered_loss_asymmetry"] = (
            pd.to_numeric(profiles.get("opposing__durable_unrecovered_loss_share_initial"), errors="coerce")
            - pd.to_numeric(profiles.get("support__durable_unrecovered_loss_share_initial"), errors="coerce")
        )
    write_parquet_atomic(profiles, profile_path)
    manifest = {
        "builder_version": BUILDER_VERSION,
        "schema_name": SCHEMA_NAME,
        "input_signature": sig,
        "onset_date": onset_date,
        "event_count": int(day_events["event_id"].nunique()),
        "profile_rows": int(len(profiles)),
        "level_rows": int(level_summary["row_count"]),
        "nonpositive_initial_qty": int(level_summary["nonpositive_initial_qty"]),
        "level_roles": sorted(level_summary["roles"]),
        "trade_aggregation_seconds": float(trade_seconds),
        "raw_replay_seconds": float(replay_seconds),
        "depth_rows_decoded": int(getattr(fast_stats, "rows_decoded", 0)),
        "depth_levels_decoded": int(getattr(fast_stats, "levels_decoded", 0)),
        "tracked_depth_messages": int(tracked_depth_messages),
        "tracked_exact_changes": int(tracked_exact_changes),
        "generated_at": utc_now_iso(),
    }
    atomic_write_json(manifest_path, manifest)
    return profiles, False, manifest


def build_pair_rows(assignments, profiles, phase7e):
    features = phase7e.numeric_profile_features(profiles)
    for extra in ("liquidity_retention_asymmetry", "liquidity_unrecovered_loss_asymmetry"):
        if extra in profiles.columns:
            features.append(extra)
    features = sorted(set(features))
    parts = [phase7e.build_pair_rows(assignments, profiles, aid, features) for _, aid in ANALYSIS_SPECS]
    parts = [x for x in parts if x is not None and not x.empty]
    if not parts:
        _, pd, _, _ = require_stack()
        return pd.DataFrame(), features
    _, pd, _, _ = require_stack()
    return pd.concat(parts, ignore_index=True), features


def summarize_subset(g, phase7e):
    np, pd, _, _ = require_stack()
    ev = pd.to_numeric(g["event_value"], errors="coerce")
    co = pd.to_numeric(g["control_value"], errors="coerce")
    ok = ev.notna() & co.notna()
    ev = ev[ok].to_numpy(float)
    co = co[ok].to_numpy(float)
    if len(ev) == 0:
        return None
    diff = ev - co
    scale = phase7e.pooled_robust_scale(ev, co)
    return {
        "parent_pairs": int(len(ev)),
        "event_median": float(np.median(ev)),
        "control_median": float(np.median(co)),
        "median_paired_difference": float(np.median(diff)),
        "paired_robust_standardized_difference": float(np.median(diff) / scale) if math.isfinite(scale) and scale > 1e-12 else math.nan,
        "paired_positive_share": float(np.mean(diff > 0)),
    }


def discovery_validation_table(pair_rows, phase7e):
    _, pd, _, _ = require_stack()
    if pair_rows.empty:
        return pd.DataFrame()
    work = pair_rows.copy()
    work["parent_date"] = work["parent_event_date"].astype(str).map(date.fromisoformat)
    work["sample_split"] = work["parent_date"].map(lambda d: "DISCOVERY" if d <= DISCOVERY_END_DATE else "VALIDATION")
    rows = []
    for keys, g in work.groupby(["sample_split", "analysis_id", "analysis_role", "stage", "window_label", "feature"], sort=True):
        rec = summarize_subset(g, phase7e)
        if rec is None:
            continue
        rows.append({
            "sample_split": keys[0], "analysis_id": keys[1], "analysis_role": keys[2],
            "stage": keys[3], "stage_bps": stage_bps_from_label(keys[3]),
            "window_label": keys[4], "feature": keys[5], **rec,
        })
    return pd.DataFrame(rows)


def direction_robustness(pair_rows, phase7e):
    _, pd, _, _ = require_stack()
    rows = []
    if pair_rows.empty:
        return pd.DataFrame()
    for keys, g in pair_rows.groupby(["analysis_id", "analysis_role", "stage", "window_label", "feature", "direction"], sort=True):
        rec = summarize_subset(g, phase7e)
        if rec is None:
            continue
        rows.append({
            "analysis_id": keys[0], "analysis_role": keys[1], "stage": keys[2],
            "stage_bps": stage_bps_from_label(keys[2]), "window_label": keys[3], "feature": keys[4],
            "direction": int(keys[5]), "direction_label": "UP" if int(keys[5]) > 0 else "DOWN", **rec,
        })
    return pd.DataFrame(rows)


def core_gate_table(dv):
    _, pd, _, _ = require_stack()
    rows = []
    if dv.empty:
        return pd.DataFrame()
    expected = dict(CORE_FEATURES)
    for split in ("DISCOVERY", "VALIDATION"):
        min_pairs = MIN_DISCOVERY_PAIRS if split == "DISCOVERY" else MIN_VALIDATION_PAIRS
        strong_rsd = DISCOVERY_STRONG_RSD if split == "DISCOVERY" else VALIDATION_STRONG_RSD
        sub = dv[(dv["sample_split"] == split) & (dv["analysis_id"] == PRIMARY_ANALYSIS_ID) & (dv["window_label"] == PRIMARY_WINDOW_LABEL)]
        for bps in STAGE_BPS:
            sg = sub[sub["stage_bps"] == bps]
            aligned = 0
            strong = 0
            share_aligned = 0
            min_n = math.inf
            details = []
            for feat, sign in CORE_FEATURES:
                fg = sg[sg["feature"] == feat]
                if len(fg) != 1:
                    details.append(f"{feat}:MISSING")
                    continue
                r = fg.iloc[0]
                rsd = safe_float(r["paired_robust_standardized_difference"])
                pos = safe_float(r["paired_positive_share"])
                n = int(r["parent_pairs"])
                min_n = min(min_n, n)
                sign_ok = math.isfinite(rsd) and rsd * sign > 0
                strong_ok = math.isfinite(rsd) and rsd * sign >= strong_rsd
                aligned_share = (pos >= MIN_ALIGNED_PAIR_SHARE) if sign > 0 else ((1.0 - pos) >= MIN_ALIGNED_PAIR_SHARE)
                aligned += int(sign_ok)
                strong += int(strong_ok)
                share_aligned += int(aligned_share)
                details.append(f"{feat}:RSD={rsd:.3f},n={n}")
            min_n = 0 if math.isinf(min_n) else int(min_n)
            gate = (
                min_n >= min_pairs
                and aligned >= MIN_EXPECTED_SIGN_FEATURES
                and strong >= MIN_STRONG_FEATURES
                and share_aligned >= MIN_EXPECTED_SIGN_FEATURES
            )
            rows.append({
                "sample_split": split, "stage_bps": int(bps), "stage": stage_label(bps),
                "primary_window_label": PRIMARY_WINDOW_LABEL, "minimum_parent_pairs": int(min_n),
                "expected_sign_features": int(aligned), "strong_features": int(strong),
                "aligned_pair_share_features": int(share_aligned), "gate_pass": bool(gate),
                "gate_rule": f"n>={min_pairs}; expected_sign>={MIN_EXPECTED_SIGN_FEATURES}; strong_RSD>={strong_rsd:g} for >={MIN_STRONG_FEATURES}; aligned_pair_share>={MIN_ALIGNED_PAIR_SHARE:g} for >={MIN_EXPECTED_SIGN_FEATURES}",
                "details": " | ".join(details),
            })
    return pd.DataFrame(rows)


def economic_timing(assignments, event_stages):
    np, pd, _, _ = require_stack()
    rows = []
    meta = event_stages.set_index("event_id")
    for role, aid in ANALYSIS_SPECS:
        a = assignments[assignments["analysis_id"] == aid]
        parents = sorted(set(a["parent_event_id"].astype(str)))
        p = meta.loc[[x for x in parents if x in meta.index]].copy()
        cohort = aid.split("|")[0]
        target = 75 if cohort == "75_PLUS" else 50
        target_col = f"hit_{target}_second"
        for bps in STAGE_BPS:
            stage_col = f"stage__{stage_label(bps)}"
            t = pd.to_numeric(p[target_col], errors="coerce") - pd.to_numeric(p[stage_col], errors="coerce")
            t = t.replace([np.inf, -np.inf], np.nan).dropna()
            rows.append({
                "analysis_id": aid, "analysis_role": role, "cohort_target_bps": int(target),
                "stage_bps": int(bps), "nominal_bps_remaining_to_cohort_target": float(target - bps),
                "parents_with_target_timing": int(len(t)),
                "median_seconds_remaining_to_cohort_target": float(t.median()) if len(t) else math.nan,
                "p25_seconds_remaining": float(t.quantile(0.25)) if len(t) else math.nan,
                "p75_seconds_remaining": float(t.quantile(0.75)) if len(t) else math.nan,
            })
    return pd.DataFrame(rows)


def build_coverage(assignments, profiles):
    _, pd, _, _ = require_stack()
    rows = []
    valid = profiles[profiles["window_valid"].fillna(False).astype(bool)]
    for role, aid in ANALYSIS_SPECS:
        a = assignments[assignments["analysis_id"] == aid]
        parent_ids = sorted(set(a["parent_event_id"].astype(str)))
        for bps in STAGE_BPS:
            st = stage_label(bps)
            for h in FIXED_HORIZONS_SEC:
                win = f"FWD_{h}S"
                pv = set(valid[(valid["stage"] == st) & (valid["window_label"] == win)]["event_id"].astype(str))
                pairable = 0
                for pid, pg in a.groupby("parent_event_id", sort=False):
                    controls = set(pg["control_event_id"].astype(str))
                    if str(pid) in pv and len(controls & pv) >= 2:
                        pairable += 1
                rows.append({
                    "analysis_id": aid, "analysis_role": role, "stage": st, "stage_bps": int(bps),
                    "window_label": win, "eligible_parents": int(len(parent_ids)), "pairable_parents": int(pairable),
                    "pairable_rate": float(pairable / max(1, len(parent_ids))),
                })
    return pd.DataFrame(rows)


def check_invariants(event_stages, profiles, assignments, quality, coverage, mismatch35: int, mismatch10: int, level_summary, partial_scope: bool):
    _, pd, _, _ = require_stack()
    rows = []
    def add(name, passed, observed, expected, note=""):
        rows.append({"check": name, "passed": bool(passed), "observed": str(observed), "expected": str(expected), "note": note})
    add("threshold_stage_ids_unique", not event_stages["event_id"].duplicated().any(), int(event_stages["event_id"].nunique()), len(event_stages))
    add("derived_35bp_matches_frozen_catalog", mismatch35 == 0, mismatch35, 0)
    add("derived_10bp_matches_frozen_fixed10", mismatch10 == 0, mismatch10, 0)
    monotonic_bad = 0
    for r in event_stages.to_dict("records"):
        vals = [int(r[f"stage__{stage_label(b)}"]) for b in STAGE_BPS]
        monotonic_bad += int(any(vals[i] > vals[i + 1] for i in range(len(vals) - 1)))
    add("threshold_times_monotonic", monotonic_bad == 0, monotonic_bad, 0)
    add("profiles_unique", not profiles.duplicated(["event_id", "stage", "window_label"]).any(), int(profiles.duplicated(["event_id", "stage", "window_label"]).sum()), 0)
    valid = profiles[profiles["window_valid"].fillna(False).astype(bool)]
    add("valid_profiles_present", len(valid) > 0, len(valid), ">0")
    add("frozen_level_initial_qty_positive", int(level_summary["nonpositive_initial_qty"]) == 0, int(level_summary["nonpositive_initial_qty"]), 0)
    add("level_roles_known", set(level_summary["roles"]).issubset({"SUPPORT", "OPPOSING"}), sorted(level_summary["roles"]), "SUPPORT/OPPOSING")
    pr = quality[quality["analysis_id"] == PRIMARY_ANALYSIS_ID].iloc[0]
    add("primary_frozen_match_balance_good", pr["balance_grade_recomputed"] == "INTERPRETABLE_BALANCED", f"N={int(pr['accepted_parents'])} max|SMD|={safe_float(pr['max_abs_smd_recomputed']):.4f}", "INTERPRETABLE_BALANCED")
    primary_cov = coverage[(coverage["analysis_id"] == PRIMARY_ANALYSIS_ID) & (coverage["window_label"] == PRIMARY_WINDOW_LABEL)]
    if partial_scope:
        add("primary_pairable_each_stage", True, "not required", "partial smoke scope")
    else:
        bad = int((primary_cov["pairable_parents"] <= 0).sum())
        add("primary_pairable_each_stage", bad == 0, bad, 0)
    return pd.DataFrame(rows)


def build_findings(summary: Mapping[str, Any], gate, timing, quality) -> str:
    lines = [
        "# Phase 7F — Continuation Threshold & Liquidity-Failure Calibration", "",
        f"- Builder: **{BUILDER_VERSION}**", f"- Status: **{summary['status']}**",
        f"- Research role: **RETROSPECTIVE_PHYSICAL_STAGE_CALIBRATION_NOT_PROSPECTIVE_ALPHA**",
        f"- Primary analysis: `{PRIMARY_ANALYSIS_ID}`", f"- Primary observation horizon: **{PRIMARY_HORIZON_SEC}s**", "",
        "## Design", "",
        "The same frozen matched physical-event comparisons used in Phase 7E are followed through 10/15/20/25/30/35-bp first-passage stages. Exact <=5-bp price levels are frozen independently at each stage and followed for 1/3/5/10 seconds.",
        "", "The 35_ONLY controls reach every calibration stage, so changing the displacement threshold does not change the control ontology. 35 bps is therefore calibrated against earlier stages rather than assumed to be special.", "",
        "## Discovery / validation gate", "",
    ]
    if gate.empty:
        lines += ["No gate rows were available.", ""]
    else:
        d = gate[gate["sample_split"] == "DISCOVERY"].sort_values("stage_bps")
        passed = d[d["gate_pass"].astype(bool)]
        if len(passed):
            earliest = int(passed.iloc[0]["stage_bps"])
            lines += [f"The earliest stage passing the predeclared discovery gate is **{earliest} bps**."]
            v = gate[(gate["sample_split"] == "VALIDATION") & (gate["stage_bps"] == earliest)]
            if len(v):
                lines += [f"The same stage {'**passes**' if bool(v.iloc[0]['gate_pass']) else '**does not pass**'} the validation-half gate."]
        else:
            lines += ["No stage passes the predeclared discovery gate. Do not force a calibrated breaking point from this run."]
        lines += [""]
    lines += [
        "## Interpretation discipline", "",
        "A stage gate is only a retrospective calibration marker inside hindsight-known 35+ physical moves. It is not a live entry rule. A later causal-anchor/all-market study must include failed candidate detections before any predictive or trading claim.", "",
        "Variable-duration stage-to-stage totals are intentionally excluded from threshold selection; the calibration uses fixed 3-second mechanics so faster continuations do not mechanically accumulate less depth activity merely because they spend less time between stages.", "",
    ]
    return "\n".join(lines) + "\n"


def self_test() -> None:
    import numpy as np
    sec = np.arange(100, 106)
    px = np.array([100.0, 100.05, 100.1001, 100.16, 100.30, 100.40])
    assert first_passage_second(sec, px, 100.0, +1, 10.0) == 102
    assert first_passage_second(sec, px, 100.0, +1, 25.0) == 104
    assert first_passage_second(sec, px, 100.0, -1, 10.0) is None
    assert stage_bps_from_label(stage_label(35)) == 35
    expected = dict(CORE_FEATURES)
    assert expected["opposing__retained_initial_quote_ratio"] == -1
    print("SELF-TEST PASS")
    print("Validated physical first-passage threshold geometry, frozen stage labels, and calibration feature directions.")


def parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(description="Phase 7F continuation threshold & exact-liquidity calibration")
    p.add_argument("--symbol", default=DEFAULT_SYMBOL)
    p.add_argument("--phase6a-script", type=Path, default=DEFAULT_PHASE6A_SCRIPT)
    p.add_argument("--phase7a-script", type=Path, default=DEFAULT_PHASE7A_SCRIPT)
    p.add_argument("--phase7c-script", type=Path, default=DEFAULT_PHASE7C_SCRIPT)
    p.add_argument("--phase7d1-script", type=Path, default=DEFAULT_PHASE7D1_SCRIPT)
    p.add_argument("--phase7e-script", type=Path, default=DEFAULT_PHASE7E_SCRIPT)
    p.add_argument("--execution-root", type=Path, default=DEFAULT_EXECUTION_ROOT)
    p.add_argument("--catalog-root", type=Path, default=DEFAULT_CATALOG_ROOT)
    p.add_argument("--phase7d1-root", type=Path, default=DEFAULT_PHASE7D1_ROOT)
    p.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--price-scale", type=float, default=10.0)
    p.add_argument("--max-book-age-ms", type=int, default=2000)
    p.add_argument("--max-trade-stale-sec", type=int, default=5)
    p.add_argument("--max-onset-dates", type=int, default=None, help="Partial smoke scope only; full calibration requires all onset dates")
    p.add_argument("--force-day", action="store_true", help="Rebuild matching per-day 7F raw replay checkpoints")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    np, pd, pa, pq = require_stack()
    phase6a, phase7a, phase7c, phase7d1, phase7e = load_frozen_modules(args)

    print("Building Phase 7F continuation-threshold liquidity calibration...")
    print(f"Builder version      : {BUILDER_VERSION}")
    print(f"Frozen Phase-6A      : {phase6a.BUILDER_VERSION}")
    print(f"Frozen Phase-7A      : {phase7a.BUILDER_VERSION}")
    print(f"Frozen Phase-7C      : {phase7c.BUILDER_VERSION}")
    print(f"Frozen Phase-7D.1    : {phase7d1.BUILDER_VERSION}")
    print(f"Frozen Phase-7E      : {phase7e.BUILDER_VERSION}")
    print(f"Stages               : {','.join(str(x) for x in STAGE_BPS)} bps")
    print(f"Primary horizon      : {PRIMARY_HORIZON_SEC}s")
    print(f"Primary analysis     : {PRIMARY_ANALYSIS_ID}")
    print("Research role        : retrospective physical-stage calibration; not prospective alpha")

    print("[1/9] Loading frozen matched assignments and balance...")
    assignments, quality = load_assignments(args, phase7e)
    for r in quality.to_dict("records"):
        print(f"      {r['analysis_role']:<18} parents={int(r['accepted_parents']):3d} max|SMD|={safe_float(r['max_abs_smd_recomputed']):.3f} {r['balance_grade_recomputed']}")

    print("[2/9] Reconstructing 10/15/20/25/30/35-bp physical first-passage stages...")
    events, fixed10_map = load_selected_events(args, phase7c, assignments)
    event_stages, mismatch35, mismatch10 = build_threshold_stage_table(args, phase7a, events, fixed10_map)
    print(f"      selected physical events={len(event_stages):,}")
    print(f"      35bp catalog mismatches={mismatch35} | 10bp fixed-departure mismatches={mismatch10}")

    print("[3/9] Discovering canonical depth/trade partitions...")
    family_maps = {fam: phase6a.discover_family_days(args.canonical_root, fam) for fam in ("depth_updates", "book_snapshots", "trades")}
    all_dates = sorted(event_stages["onset_date"].astype(str).unique())
    selected_dates = list(all_dates)
    if args.max_onset_dates is not None:
        selected_dates = selected_dates[: max(0, int(args.max_onset_dates))]
    partial_scope = len(selected_dates) < len(all_dates)
    print(f"      onset dates={len(selected_dates)}" + (f" of {len(all_dates)} (PARTIAL SMOKE SCOPE)" if partial_scope else ""))

    print("[4/9] Replaying authoritative depth and materializing exact mechanics...")
    profile_parts = []
    level_summary = {"row_count": 0, "nonpositive_initial_qty": 0, "roles": set()}
    reused_days = 0
    for i, d in enumerate(selected_dates, 1):
        de = event_stages[event_stages["onset_date"] == d].copy()
        prof, reused, diag = process_onset_date(d, de, phase6a, phase7e, family_maps, args)
        profile_parts.append(prof)
        reused_days += int(reused)
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
        try:
            pa.default_memory_pool().release_unused()
        except Exception:
            pass

    profiles = pd.concat(profile_parts, ignore_index=True) if profile_parts else pd.DataFrame()
    processed_events = event_stages[event_stages["onset_date"].isin(selected_dates)].copy()
    processed_ids = set(processed_events["event_id"].astype(str))
    a_proc = assignments[
        assignments["parent_event_id"].astype(str).isin(processed_ids)
        & assignments["control_event_id"].astype(str).isin(processed_ids)
    ].copy()

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_parquet_atomic(processed_events, args.output_root / "selected_threshold_event_stages.parquet")
    write_parquet_atomic(profiles, args.output_root / "threshold_stage_profiles.parquet")

    print("[5/9] Building matched stage-by-stage calibration contrasts...")
    pair_rows, features = build_pair_rows(a_proc, profiles, phase7e)
    summary = phase7e.summarize_pair_rows(pair_rows) if not pair_rows.empty else pd.DataFrame()
    discovery_validation = discovery_validation_table(pair_rows, phase7e)
    direction = direction_robustness(pair_rows, phase7e)
    coverage = build_coverage(a_proc, profiles)
    gate = core_gate_table(discovery_validation)
    timing = economic_timing(a_proc, processed_events)
    if not pair_rows.empty:
        write_parquet_atomic(pair_rows, args.output_root / "threshold_pair_rows.parquet")

    print("[6/9] Checking calibration invariants...")
    checks = check_invariants(processed_events, profiles, a_proc, quality, coverage, mismatch35, mismatch10, level_summary, partial_scope)
    for r in checks.to_dict("records"):
        print(f"      {'PASS' if r['passed'] else 'FAIL'} {r['check']}: observed={r['observed']} expected={r['expected']}")
    failed = checks[~checks["passed"].astype(bool)]
    status = "SMOKE_PASS" if partial_scope and failed.empty else ("PASS" if failed.empty else "FAIL")

    print("[7/9] Writing reports...")
    args.report_root.mkdir(parents=True, exist_ok=True)
    quality.to_csv(args.report_root / "threshold_calibration_analysis_quality.csv", index=False)
    coverage.to_csv(args.report_root / "threshold_calibration_coverage.csv", index=False)
    summary.to_csv(args.report_root / "threshold_calibration_contrasts.csv", index=False)
    discovery_validation.to_csv(args.report_root / "threshold_calibration_discovery_validation.csv", index=False)
    direction.to_csv(args.report_root / "threshold_calibration_direction_robustness.csv", index=False)
    gate.to_csv(args.report_root / "threshold_calibration_gate.csv", index=False)
    timing.to_csv(args.report_root / "threshold_calibration_economic_timing.csv", index=False)
    checks.to_csv(args.report_root / "threshold_calibration_invariants.csv", index=False)

    summary_meta = {
        "builder_version": BUILDER_VERSION, "schema_name": SCHEMA_NAME, "status": status,
        "generated_at": utc_now_iso(), "symbol": args.symbol, "stage_bps": STAGE_BPS,
        "primary_horizon_sec": PRIMARY_HORIZON_SEC, "primary_analysis_id": PRIMARY_ANALYSIS_ID,
        "discovery_end_date": DISCOVERY_END_DATE.isoformat(), "validation_start_date": VALIDATION_START_DATE.isoformat(),
        "selected_events": int(processed_events["event_id"].nunique()), "profile_rows": int(len(profiles)),
        "valid_profile_rows": int(profiles["window_valid"].fillna(False).sum()) if not profiles.empty else 0,
        "level_rows_audited": int(level_summary["row_count"]), "reused_onset_date_checkpoints": int(reused_days),
        "partial_scope": bool(partial_scope), "feature_count": int(len(features)),
        "interpretation": "Retrospective physical-stage calibration only; not prospective alpha or a trading rule.",
    }
    atomic_write_json(args.output_root / "metadata.json", summary_meta)
    atomic_write_text(args.report_root / "threshold_calibration_findings.md", build_findings(summary_meta, gate, timing, quality))

    print("[8/9] Calibration summary")
    if not gate.empty:
        dpass = gate[(gate["sample_split"] == "DISCOVERY") & gate["gate_pass"].astype(bool)].sort_values("stage_bps")
        if len(dpass):
            earliest = int(dpass.iloc[0]["stage_bps"])
            v = gate[(gate["sample_split"] == "VALIDATION") & (gate["stage_bps"] == earliest)]
            print(f"      earliest discovery-gate stage={earliest} bps | validation={'PASS' if len(v) and bool(v.iloc[0]['gate_pass']) else 'FAIL'}")
        else:
            print("      no displacement stage passed the frozen discovery gate")
    print(f"      status={status}")
    print(f"      selected events={summary_meta['selected_events']:,} profiles={summary_meta['profile_rows']:,} valid={summary_meta['valid_profile_rows']:,}")
    print(f"      reports={args.report_root}")

    print("[9/9] Interpretation gate")
    if status == "FAIL":
        print("ERROR: one or more Phase-7F invariants failed; do not interpret calibration reports.")
        return 2
    if partial_scope:
        print("Partial smoke scope completed. Do not interpret threshold calibration until the full onset-date run is complete.")
    else:
        print("Full Phase 7F completed. Interpret the discovery gate first, then validation and direction robustness; do not treat the selected stage as prospective alpha.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
