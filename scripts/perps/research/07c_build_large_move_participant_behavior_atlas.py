#!/usr/bin/env python3
"""
Phase 7C — Large-move participant-behavior trajectory atlas
============================================================

Purpose
-------
Build a descriptive, event-centered atlas of order-book state, visible-liquidity
change, aggressive flow, price response, open-interest context, liquidation
observations, and mark/last dislocation around the frozen Phase-7A/7B large-move
physical events.

This phase is TOP-DOWN DISCOVERY.  It does not build a predictive model and it
has no non-event control group yet.  Phase 7D must add matched non-big-move
controls before any feature is called a precursor rather than an event-stratum
fingerprint.

Frozen event vocabulary
-----------------------
* Physical alignment: Phase-7A ``EXTREMUM_HINDSIGHT`` / event ``onset_second``.
* Event stages: ``EXTREMUM_HINDSIGHT``, ``VOL_SCALED_DEPARTURE``,
  ``FIXED_10BP_DEPARTURE``, ``PRIOR_60S_RANGE_BREAK``, then 35/50/75/100-bp
  first-passage hits.
* Primary continuation strata (within 30 minutes):
    35_ONLY, 50_ONLY, 75_ONLY, 100_PLUS.

Primary discovery questions
---------------------------
1. What changes during preparation before the retrospective physical extremum?
2. After the extremum, what differs between legs that stop near 35/50 bps and
   legs that continue to 75/100+ bps?
3. Are the same directional-normalized patterns present for UP and DOWN legs?
4. Which patterns replicate across many event days rather than a few extremes?

Data contracts
--------------
* Phase-7A event labels are hindsight objects used only for discovery alignment.
* Phase-6B primitive rows are causal one-second state and carry Phase-6A book
  validity.  Invalid book rows never contribute book-derived atlas features.
* The richer flow-feature layer is causal and keyed by receive_second.
* forceOrder observed size remains selected/censored; it is never interpreted
  as total liquidation volume.
* ``visible_band_replenishment_proxy`` is NOT exact same-price refill.  It is a
  state-change proxy derived from visible quote depth inside 5 bps and can be
  affected by price movement / liquidity migration across the band boundary.

Outputs
-------
data/derived/perps/research/large_move_participant_behavior_atlas_v1/
  event_metadata.parquet
  preparation_event_features.parquet
  stage_event_features.parquet
  metadata/latest_summary.json
  metadata/run_config.json

reports/perps_large_move_participant_behavior_atlas/
  large_move_behavior_feature_dictionary.csv
  large_move_behavior_preparation_trajectory.csv
  large_move_behavior_stage_trajectory.csv
  large_move_behavior_continuation_contrast.csv
  large_move_behavior_daily_sign_consistency.csv
  large_move_behavior_direction_symmetry.csv
  large_move_behavior_coverage.csv
  large_move_behavior_invariants.csv
  large_move_behavior_findings.md

No alpha, strategy, or profitability claim is made by Phase 7C.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

BUILDER_VERSION = "1.0.2"
SCHEMA_VERSION = 1
SCHEMA_NAME = "large_move_participant_behavior_atlas_v1"
DEFAULT_SYMBOL = "BTCUSDT"
SECONDS_PER_DAY = 86_400

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()
DEFAULT_CATALOG_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / "large_move_event_catalog_v1"
DEFAULT_PRIMITIVE_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / "liquidity_event_primitives_v1"
DEFAULT_FLOW_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "flow_features_1s_v1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / SCHEMA_NAME
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "perps_large_move_participant_behavior_atlas"

PRIMARY_HORIZON_SEC = 1800
PREPARATION_WINDOWS = (
    ("PRE_300_121", -300, -121),
    ("PRE_120_61", -120, -61),
    ("PRE_60_31", -60, -31),
    ("PRE_30_11", -30, -11),
    ("PRE_10_6", -10, -6),
    ("PRE_5_1", -5, -1),
)
STAGE_METHODS = (
    "EXTREMUM_HINDSIGHT",
    "VOL_SCALED_DEPARTURE",
    "FIXED_10BP_DEPARTURE",
    "PRIOR_60S_RANGE_BREAK",
    "HIT_35BP",
)

# Phase-6B columns that are already audited and useful for the atlas.
PRIMITIVE_COLUMNS = (
    "receive_second", "book_valid", "state_age_ms", "mid_price", "spread_bps",
    "bid_depth_quote_5bps", "ask_depth_quote_5bps",
    "bid_depth_quote_10bps", "ask_depth_quote_10bps",
    "bid_depth_quote_25bps", "ask_depth_quote_25bps",
    "depth_imbalance_5bps", "bid_withdrawal_frac_1s", "ask_withdrawal_frac_1s",
    "bid_withdrawal_z_900s", "ask_withdrawal_z_900s",
    "realized_vol_60s_bps", "quote_volume_5s", "quote_volume_60s",
    "signed_quote_flow_5s", "aggressive_flow_imbalance_5s",
    "signed_quote_flow_5s_z_900s", "quote_volume_5s_z_900s",
    "open_interest_change_60s_pct", "open_interest_change_300s_pct",
    "last_trade_price", "trade_price_fresh", "mark_price", "mark_fresh",
    "mark_last_dislocation_bps", "mark_last_dislocation_z_900s",
    "force_order_count", "force_sell_count", "force_buy_count",
    "force_selected_notional_sum",
)

# Optional richer causal features.  The script discovers which ones exist.
FLOW_OPTIONAL_COLUMNS = (
    "receive_second", "flow_feature_context_ready",
    "return_1s_bps", "return_5s_bps", "return_15s_bps", "return_30s_bps", "return_60s_bps",
    "realized_vol_5s_bps", "realized_vol_15s_bps", "realized_vol_30s_bps", "realized_vol_60s_bps",
    "trade_count_1s", "trade_count_5s", "trade_count_15s", "trade_count_30s", "trade_count_60s",
    "quote_volume_1s", "quote_volume_5s", "quote_volume_15s", "quote_volume_30s", "quote_volume_60s",
    "signed_quote_flow_1s", "signed_quote_flow_5s", "signed_quote_flow_15s", "signed_quote_flow_30s", "signed_quote_flow_60s",
    "aggressive_flow_imbalance_1s", "aggressive_flow_imbalance_5s", "aggressive_flow_imbalance_15s",
    "aggressive_flow_imbalance_30s", "aggressive_flow_imbalance_60s",
    "price_response_bps_per_1m_signed_flow_5s", "price_response_bps_per_1m_signed_flow_15s",
    "price_response_bps_per_1m_signed_flow_30s", "price_response_bps_per_1m_signed_flow_60s",
    "price_flow_alignment_5s", "price_flow_alignment_15s", "price_flow_alignment_30s", "price_flow_alignment_60s",
    "open_interest_change_60s_pct", "open_interest_change_300s_pct",
    "liquidation_notional_1s", "liquidation_notional_5s", "liquidation_notional_15s", "liquidation_notional_30s", "liquidation_notional_60s",
    "net_liquidation_notional_1s", "net_liquidation_notional_5s", "net_liquidation_notional_15s",
    "net_liquidation_notional_30s", "net_liquidation_notional_60s",
    "liquidation_imbalance_5s", "liquidation_imbalance_15s", "liquidation_imbalance_30s", "liquidation_imbalance_60s",
)


def require_numeric_stack():
    try:
        import numpy as np
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Phase 7C requires numpy and pandas. Activate the project .venv and run:\n"
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
            "Phase 7C Parquet I/O requires pyarrow. Activate the project .venv and run:\n"
            "  python -m pip install pyarrow"
        ) from exc
    return np, pd, pa, pq


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7c-", suffix=path.suffix, dir=str(path.parent))
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
    fd, tmp = tempfile.mkstemp(prefix="tmp7c-", suffix=".parquet", dir=str(path.parent))
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


def primitive_day_path(root: Path, symbol: str, d: date) -> Path:
    return root / "primitives" / f"symbol={symbol}" / f"date={d.isoformat()}" / f"part-{d.isoformat()}.parquet"


def flow_day_files(root: Path, symbol: str, d: date) -> list[Path]:
    # Support both the common symbol/date partition and date-only historical layouts.
    patterns = [
        root / f"symbol={symbol}" / f"date={d.isoformat()}",
        root / f"date={d.isoformat()}",
    ]
    out: list[Path] = []
    for p in patterns:
        if p.exists():
            out.extend(sorted(p.glob("*.parquet")))
    return sorted(set(out))


def read_parquet_columns(files: Sequence[Path], desired: Sequence[str], *, required: Sequence[str] = ()):
    np, pd, _, pq = require_stack()
    frames = []
    for p in files:
        pf = pq.ParquetFile(p)
        try:
            names = set(pf.schema_arrow.names)
            missing = [c for c in required if c not in names]
            if missing:
                raise RuntimeError(f"{p} missing required columns: {missing}")
            use = [c for c in desired if c in names]
            if not use:
                continue
            frames.append(pf.read(columns=use).to_pandas())
        finally:
            pf.close()
    if not frames:
        return pd.DataFrame(columns=list(desired))
    out = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if "receive_second" in out.columns:
        out["receive_second"] = pd.to_numeric(out["receive_second"], errors="coerce")
        out = out.dropna(subset=["receive_second"])
        out["receive_second"] = out["receive_second"].astype("int64")
        out = out.sort_values("receive_second", kind="mergesort").drop_duplicates("receive_second", keep="last")
    return out.reset_index(drop=True)


def classify_stratum(row: Mapping[str, Any], horizon: int = PRIMARY_HORIZON_SEC) -> str:
    def hit(name: str) -> bool:
        v = row.get(name)
        try:
            x = float(v)
        except (TypeError, ValueError):
            return False
        return math.isfinite(x) and x <= horizon

    if hit("time_to_100_sec"):
        return "100_PLUS"
    if hit("time_to_75_sec"):
        return "75_ONLY"
    if hit("time_to_50_sec"):
        return "50_ONLY"
    if hit("time_to_35_sec"):
        return "35_ONLY"
    return "BASE_NOT_35_WITHIN_30M"


def build_event_metadata(events):
    _, pd, _, _ = require_stack()
    out = events.copy()
    out["continuation_stratum"] = [classify_stratum(r) for r in out.to_dict("records")]
    return out


def stage_seconds_for_event(ev: Mapping[str, Any], onset_map: Mapping[str, Mapping[str, Any]]) -> dict[str, int | None]:
    stages: dict[str, int | None] = {"EXTREMUM_HINDSIGHT": int(ev["onset_second"])}
    for method in ("VOL_SCALED_DEPARTURE", "FIXED_10BP_DEPARTURE", "PRIOR_60S_RANGE_BREAK"):
        r = onset_map.get(method)
        if r and bool(r.get("candidate_found")) and r.get("candidate_second") is not None:
            stages[method] = int(r["candidate_second"])
        else:
            stages[method] = None
    hit35 = ev.get("hit_35_second")
    try:
        stages["HIT_35BP"] = int(hit35) if hit35 is not None and not (isinstance(hit35, float) and math.isnan(hit35)) else None
    except Exception:
        stages["HIT_35BP"] = None
    return stages


def numeric(series, pd):
    return pd.to_numeric(series, errors="coerce")


def build_direction_normalized_frame(frame, direction: int):
    """Build one direction-normalized context without repeated column inserts.

    v1.0.1 is scientifically identical to v1.0.0 but materializes numeric/base
    columns and all derived atlas columns in bulk.  This avoids Pandas block
    fragmentation and allows one normalized frame to be reused by all events
    of the same direction on a given UTC day.
    """
    np, pd = require_numeric_stack()
    d = 1 if int(direction) > 0 else -1

    # Convert numeric source columns in one block instead of assigning columns
    # one-by-one, which fragmented the DataFrame in v1.0.0.
    excluded = {"symbol", "receive_date", "book_valid", "flow_feature_context_ready", "receive_second"}
    numeric_cols = [c for c in frame.columns if c not in excluded]
    base_parts = [frame.drop(columns=numeric_cols, errors="ignore").copy()]
    if numeric_cols:
        base_parts.append(frame[numeric_cols].apply(pd.to_numeric, errors="coerce"))
    f = pd.concat(base_parts, axis=1)
    derived: dict[str, Any] = {}

    # Direction-normalized participant flow / price response.
    for w in (1, 5, 15, 30, 60):
        c = f"return_{w}s_bps"
        if c in f:
            derived[f"dir_return_{w}s_bps"] = d * f[c]
        c = f"signed_quote_flow_{w}s"
        if c in f:
            derived[f"dir_signed_quote_flow_{w}s"] = d * f[c]
        c = f"aggressive_flow_imbalance_{w}s"
        if c in f:
            derived[f"dir_aggressive_flow_imbalance_{w}s"] = d * f[c]
        c = f"net_liquidation_notional_{w}s"
        if c in f:
            derived[f"dir_net_liquidation_notional_{w}s"] = d * f[c]
        c = f"liquidation_imbalance_{w}s"
        if c in f:
            derived[f"dir_liquidation_imbalance_{w}s"] = d * f[c]

    if "mark_last_dislocation_bps" in f:
        derived["dir_mark_last_dislocation_bps"] = d * f["mark_last_dislocation_bps"]
    if "mark_last_dislocation_z_900s" in f:
        derived["dir_mark_last_dislocation_z_900s"] = d * f["mark_last_dislocation_z_900s"]
    if "depth_imbalance_5bps" in f:
        derived["dir_depth_imbalance_5bps"] = d * f["depth_imbalance_5bps"]

    # Direction-normalized book sides: supporting side trails the move; opposing side is ahead of it.
    if d > 0:
        support_prefix, oppose_prefix = "bid", "ask"
        force_same, force_opp = "force_buy_count", "force_sell_count"
    else:
        support_prefix, oppose_prefix = "ask", "bid"
        force_same, force_opp = "force_sell_count", "force_buy_count"

    for band in (5, 10, 25):
        sc = f"{support_prefix}_depth_quote_{band}bps"
        oc = f"{oppose_prefix}_depth_quote_{band}bps"
        if sc in f:
            derived[f"support_depth_quote_{band}bps"] = f[sc]
        if oc in f:
            derived[f"opposing_depth_quote_{band}bps"] = f[oc]

    sw = f"{support_prefix}_withdrawal_frac_1s"
    ow = f"{oppose_prefix}_withdrawal_frac_1s"
    swz = f"{support_prefix}_withdrawal_z_900s"
    owz = f"{oppose_prefix}_withdrawal_z_900s"
    if sw in f:
        derived["support_withdrawal_frac_1s"] = f[sw]
    if ow in f:
        derived["opposing_withdrawal_frac_1s"] = f[ow]
    if swz in f:
        derived["support_withdrawal_z_900s"] = f[swz]
    if owz in f:
        derived["opposing_withdrawal_z_900s"] = f[owz]
    if force_same in f:
        derived["force_with_direction_count"] = f[force_same]
    if force_opp in f:
        derived["force_against_direction_count"] = f[force_opp]

    # Helper that can read source or not-yet-concatenated derived series.
    def col(name: str):
        if name in derived:
            return derived[name]
        return f[name]

    # Visible-band liquidity distance / migration proxy: fraction of 25-bp depth outside 5 bps.
    for side in ("support", "opposing"):
        c5 = f"{side}_depth_quote_5bps"
        c25 = f"{side}_depth_quote_25bps"
        if c5 in derived and c25 in derived:
            denom = col(c25).replace(0.0, np.nan)
            derived[f"{side}_outer_depth_share_5to25bps"] = ((col(c25) - col(c5)).clip(lower=0.0) / denom).clip(0.0, 1.0)

    # Exact-second visible-band state-change proxies. These are NOT raw add/remove events.
    sec = pd.to_numeric(f["receive_second"], errors="coerce").to_numpy(dtype=np.int64)
    exact_prev = np.zeros(len(f), dtype=bool)
    if len(f) > 1:
        exact_prev[1:] = np.diff(sec) == 1
    if "book_valid" in f:
        bv = f["book_valid"].fillna(False).astype(bool).to_numpy()
        exact_prev &= np.r_[False, bv[:-1]] & bv

    for side in ("support", "opposing"):
        c = f"{side}_depth_quote_5bps"
        if c not in derived:
            continue
        x = pd.to_numeric(col(c), errors="coerce")
        prev = x.shift(1)
        delta = (x / prev.replace(0.0, np.nan) - 1.0).where(exact_prev).clip(lower=-1.0, upper=5.0)
        add = delta.clip(lower=0.0)
        rem = (-delta).clip(lower=0.0)
        add5 = add.rolling(5, min_periods=1).sum()
        rem5 = rem.rolling(5, min_periods=1).sum()
        derived[f"{side}_depth_delta_frac_1s"] = delta
        derived[f"{side}_visible_add_frac_1s_proxy"] = add
        derived[f"{side}_visible_remove_frac_1s_proxy"] = rem
        derived[f"{side}_visible_add_5s_proxy"] = add5
        derived[f"{side}_visible_remove_5s_proxy"] = rem5
        derived[f"{side}_visible_replenishment_ratio_5s_proxy"] = (add5 / rem5.replace(0.0, np.nan)).clip(upper=10.0)

    # A simple event-direction price-response/flow interaction. Existing audited response fields are retained too.
    if "dir_return_5s_bps" in derived and "dir_signed_quote_flow_5s" in derived:
        q = derived["dir_signed_quote_flow_5s"] / 1_000_000.0
        derived["dir_price_response_bps_per_1m_flow_5s_proxy"] = (derived["dir_return_5s_bps"] / q.replace(0.0, np.nan)).where(q.abs() >= 0.01)

    if derived:
        f = pd.concat([f, pd.DataFrame(derived, index=f.index)], axis=1)
    # One copy consolidates blocks after the bulk concat and gives downstream
    # slicing a compact frame.
    return f.copy()


FEATURE_DICTIONARY = {
    "spread_bps": ("BOOK_STATE", "Phase-6A reconstructed spread; lower means tighter book."),
    "support_depth_quote_5bps": ("BOOK_STATE", "Visible 5-bps quote depth on the side supporting the event direction."),
    "opposing_depth_quote_5bps": ("BOOK_STATE", "Visible 5-bps quote depth ahead of the event direction."),
    "support_depth_quote_10bps": ("BOOK_STATE", "Visible 10-bps supporting-side quote depth."),
    "opposing_depth_quote_10bps": ("BOOK_STATE", "Visible 10-bps opposing-side quote depth."),
    "support_depth_quote_25bps": ("BOOK_STATE", "Visible 25-bps supporting-side quote depth."),
    "opposing_depth_quote_25bps": ("BOOK_STATE", "Visible 25-bps opposing-side quote depth."),
    "dir_depth_imbalance_5bps": ("BOOK_STATE", "Direction-normalized 5-bps depth imbalance; positive means support side deeper than opposing side."),
    "support_withdrawal_frac_1s": ("BOOK_CHANGE", "1-second withdrawal fraction on supporting side."),
    "opposing_withdrawal_frac_1s": ("BOOK_CHANGE", "1-second withdrawal fraction on opposing side."),
    "support_withdrawal_z_900s": ("BOOK_CHANGE", "Causal withdrawal surprise on supporting side."),
    "opposing_withdrawal_z_900s": ("BOOK_CHANGE", "Causal withdrawal surprise on opposing side."),
    "support_outer_depth_share_5to25bps": ("LIQUIDITY_MIGRATION_PROXY", "Share of supporting 25-bps depth outside the inner 5 bps."),
    "opposing_outer_depth_share_5to25bps": ("LIQUIDITY_MIGRATION_PROXY", "Share of opposing 25-bps depth outside the inner 5 bps."),
    "support_visible_add_5s_proxy": ("REPLENISHMENT_PROXY", "Positive 5-bps visible-depth state changes summed over trailing 5s; not exact order additions."),
    "opposing_visible_add_5s_proxy": ("REPLENISHMENT_PROXY", "Opposing-side visible-depth addition proxy over 5s."),
    "support_visible_remove_5s_proxy": ("REPLENISHMENT_PROXY", "Supporting-side visible-depth removal proxy over 5s."),
    "opposing_visible_remove_5s_proxy": ("REPLENISHMENT_PROXY", "Opposing-side visible-depth removal proxy over 5s."),
    "support_visible_replenishment_ratio_5s_proxy": ("REPLENISHMENT_PROXY", "Visible add/remove ratio on supporting side; band-state proxy, not exact same-price refill."),
    "opposing_visible_replenishment_ratio_5s_proxy": ("REPLENISHMENT_PROXY", "Visible add/remove ratio on opposing side; band-state proxy, not exact same-price refill."),
    "realized_vol_60s_bps": ("VOLATILITY_CONTEXT", "Trailing realized volatility over 60 seconds."),
    "quote_volume_60s": ("TRADE_ACTIVITY", "Trailing 60-second quote volume."),
    "dir_signed_quote_flow_5s": ("AGGRESSIVE_FLOW", "5-second signed aggressive quote flow normalized into event direction."),
    "dir_aggressive_flow_imbalance_5s": ("AGGRESSIVE_FLOW", "5-second aggressive-flow imbalance normalized into event direction."),
    "dir_return_5s_bps": ("PRICE_RESPONSE", "5-second return normalized into event direction."),
    "dir_price_response_bps_per_1m_flow_5s_proxy": ("PRICE_FLOW_INTERACTION", "Direction-normalized 5-second return per $1m direction-normalized signed flow; atlas proxy."),
    "price_response_bps_per_1m_signed_flow_5s": ("PRICE_FLOW_INTERACTION", "Audited causal price response per signed flow from flow-feature layer."),
    "price_flow_alignment_5s": ("PRICE_FLOW_INTERACTION", "Audited causal price/flow alignment."),
    "open_interest_change_60s_pct": ("OI_CONTEXT", "Observed open-interest percentage change over 60s; regime/context only."),
    "open_interest_change_300s_pct": ("OI_CONTEXT", "Observed open-interest percentage change over 300s; regime/context only."),
    "dir_mark_last_dislocation_bps": ("MARK_LAST", "Mark/last dislocation normalized into event direction."),
    "force_with_direction_count": ("FORCE_ORDER", "Observed selected/censored forceOrder count in event direction."),
    "force_against_direction_count": ("FORCE_ORDER", "Observed selected/censored forceOrder count opposite event direction."),
    "force_selected_notional_sum": ("FORCE_ORDER", "Descriptive sum of selected/censored forceOrder snapshots; NOT total liquidation volume."),
}


def atlas_features(frame) -> list[str]:
    # Use the explicit dictionary first, then add a small richer causal set if available.
    cols = [c for c in FEATURE_DICTIONARY if c in frame.columns]
    richer = [
        "dir_signed_quote_flow_15s", "dir_signed_quote_flow_30s", "dir_signed_quote_flow_60s",
        "dir_aggressive_flow_imbalance_15s", "dir_aggressive_flow_imbalance_30s", "dir_aggressive_flow_imbalance_60s",
        "price_response_bps_per_1m_signed_flow_15s", "price_response_bps_per_1m_signed_flow_30s", "price_response_bps_per_1m_signed_flow_60s",
        "price_flow_alignment_15s", "price_flow_alignment_30s", "price_flow_alignment_60s",
        "trade_count_60s", "quote_volume_15s", "quote_volume_30s",
        "dir_net_liquidation_notional_5s", "dir_net_liquidation_notional_15s", "dir_liquidation_imbalance_5s", "dir_liquidation_imbalance_15s",
    ]
    for c in richer:
        if c in frame.columns and c not in cols:
            cols.append(c)
    return cols


def prepare_feature_arrays(f, features: Sequence[str]):
    """Materialize sorted seconds and numeric feature arrays once per direction/day."""
    np, pd = require_numeric_stack()
    seconds = pd.to_numeric(f["receive_second"], errors="coerce").to_numpy(dtype=np.int64, copy=False)
    arrays: dict[str, Any] = {}
    for feat in features:
        if feat in f.columns:
            arrays[feat] = pd.to_numeric(f[feat], errors="coerce").to_numpy(dtype=float, copy=False)
        else:
            arrays[feat] = np.full(len(f), np.nan, dtype=float)
    return seconds, arrays


def aggregate_window_feature_rows(event_meta: Mapping[str, Any], seconds, feature_arrays: Mapping[str, Any], window_label: str, lo: int, hi: int, features: Sequence[str]):
    """Aggregate one event window using NumPy only.

    Semantics match v1.0.0: non-finite values are excluded, coverage uses the
    nominal second count, and ``last`` means the last finite observation in the
    window.  ``np.quantile`` replaces Pandas Series.quantile to avoid hundreds
    of thousands of tiny DataFrame/Series constructions.
    """
    np, _ = require_numeric_stack()
    onset = int(event_meta["onset_second"])
    left = int(np.searchsorted(seconds, onset + lo, side="left"))
    right = int(np.searchsorted(seconds, onset + hi, side="right"))
    expected = hi - lo + 1
    rows = []
    for feat in features:
        raw = feature_arrays[feat][left:right]
        x = raw[np.isfinite(raw)]
        n = int(x.size)
        if n:
            q25, med, q75 = np.quantile(x, [0.25, 0.50, 0.75])
            mean = float(np.mean(x))
            last = float(x[-1])
            p25, median, p75 = float(q25), float(med), float(q75)
        else:
            mean = median = p25 = p75 = last = math.nan
        rows.append({
            "event_id": event_meta["event_id"],
            "event_date": event_meta["onset_date"],
            "direction": int(event_meta["direction"]),
            "direction_label": event_meta.get("direction_label"),
            "continuation_stratum": event_meta["continuation_stratum"],
            "window_label": window_label,
            "window_start_rel_sec": lo,
            "window_end_rel_sec": hi,
            "feature": feat,
            "n_valid": n,
            "n_expected": int(expected),
            "coverage_rate": float(n / expected) if expected > 0 else math.nan,
            "mean": mean,
            "median": median,
            "p25": p25,
            "p75": p75,
            "last": last,
        })
    return rows


def stage_feature_rows(event_meta: Mapping[str, Any], stages: Mapping[str, int | None], seconds, feature_arrays: Mapping[str, Any], features: Sequence[str]):
    np, _ = require_numeric_stack()
    rows = []
    onset = int(event_meta["onset_second"])
    for stage in STAGE_METHODS:
        sec = stages.get(stage)
        idx = -1
        if sec is not None:
            pos = int(np.searchsorted(seconds, int(sec), side="left"))
            if pos < len(seconds) and int(seconds[pos]) == int(sec):
                idx = pos
        for feat in features:
            if idx >= 0:
                val = float(feature_arrays[feat][idx])
                ok = bool(math.isfinite(val))
            else:
                val = math.nan
                ok = False
            rows.append({
                "event_id": event_meta["event_id"], "event_date": event_meta["onset_date"],
                "direction": int(event_meta["direction"]), "direction_label": event_meta.get("direction_label"),
                "continuation_stratum": event_meta["continuation_stratum"], "stage": stage,
                "stage_second": None if sec is None else int(sec),
                "seconds_after_extremum": None if sec is None else int(sec - onset),
                "feature": feat, "value": val if ok else math.nan, "available": ok,
            })
    return rows



def robust_scale(series):
    """Robust pooled scale used by descriptive continuation contrasts.

    Uses IQR/1.349 when available, falling back to population standard
    deviation for degenerate IQRs. NumPy quantiles avoid the small-Series
    Pandas quantile hot path that previously dominated event-window work.
    """
    np, pd = require_numeric_stack()
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, copy=False)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return math.nan
    q25, q75 = np.quantile(arr, [0.25, 0.75])
    iqr = float(q75 - q25)
    if iqr > 0:
        return iqr / 1.349
    return float(np.std(arr, ddof=0))


def summarize_preparation(prep):
    np, pd = require_numeric_stack()
    rows = []
    group_cols = ["continuation_stratum", "window_label", "feature"]
    for keys, g in prep.groupby(group_cols, sort=True, dropna=False):
        vals = pd.to_numeric(g["median"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        cov = pd.to_numeric(g["coverage_rate"], errors="coerce").dropna()
        rows.append({
            "continuation_stratum": keys[0], "window_label": keys[1], "feature": keys[2],
            "events": int(g["event_id"].nunique()), "valid_events": int(len(vals)),
            "median_event_value": float(vals.median()) if len(vals) else math.nan,
            "p25_event_value": float(vals.quantile(0.25)) if len(vals) else math.nan,
            "p75_event_value": float(vals.quantile(0.75)) if len(vals) else math.nan,
            "mean_window_coverage": float(cov.mean()) if len(cov) else math.nan,
        })
    return pd.DataFrame(rows)


def summarize_stages(stage):
    np, pd = require_numeric_stack()
    rows = []
    for keys, g in stage.groupby(["continuation_stratum", "stage", "feature"], sort=True, dropna=False):
        vals = pd.to_numeric(g["value"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        rows.append({
            "continuation_stratum": keys[0], "stage": keys[1], "feature": keys[2],
            "events": int(g["event_id"].nunique()), "valid_events": int(len(vals)),
            "coverage_rate": float(len(vals) / max(1, g["event_id"].nunique())),
            "median_event_value": float(vals.median()) if len(vals) else math.nan,
            "p25_event_value": float(vals.quantile(0.25)) if len(vals) else math.nan,
            "p75_event_value": float(vals.quantile(0.75)) if len(vals) else math.nan,
        })
    return pd.DataFrame(rows)


def build_contrasts(prep, stage):
    np, pd = require_numeric_stack()
    rows = []
    # Event-level 100_PLUS versus 35_ONLY contrasts; robust standardized difference uses pooled robust scale.
    for domain, df, loc_col, val_col in (
        ("PREPARATION", prep, "window_label", "median"),
        ("STAGE", stage, "stage", "value"),
    ):
        for (loc, feat), g in df.groupby([loc_col, "feature"], sort=True):
            a = pd.to_numeric(g.loc[g["continuation_stratum"] == "100_PLUS", val_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            b = pd.to_numeric(g.loc[g["continuation_stratum"] == "35_ONLY", val_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(a) < 5 or len(b) < 5:
                continue
            pooled = pd.concat([a, b], ignore_index=True)
            scale = robust_scale(pooled)
            diff = float(a.median() - b.median())
            rows.append({
                "domain": domain, "location": loc, "feature": feat,
                "n_100_plus": int(len(a)), "n_35_only": int(len(b)),
                "median_100_plus": float(a.median()), "median_35_only": float(b.median()),
                "median_difference_100_plus_minus_35_only": diff,
                "pooled_robust_scale": scale,
                "robust_standardized_difference": diff / scale if math.isfinite(scale) and scale > 0 else math.nan,
                "interpretation": "DESCRIPTIVE_DISCOVERY_NOT_CONTROL_ADJUSTED",
            })
    return pd.DataFrame(rows)


def daily_sign_consistency(prep, stage):
    np, pd = require_numeric_stack()
    rows = []
    for domain, df, loc_col, val_col in (
        ("PREPARATION", prep, "window_label", "median"),
        ("STAGE", stage, "stage", "value"),
    ):
        for (loc, feat), g in df.groupby([loc_col, "feature"], sort=True):
            day_diffs = []
            for day, dg in g.groupby("event_date"):
                a = pd.to_numeric(dg.loc[dg["continuation_stratum"] == "100_PLUS", val_col], errors="coerce").dropna()
                b = pd.to_numeric(dg.loc[dg["continuation_stratum"] == "35_ONLY", val_col], errors="coerce").dropna()
                if len(a) >= 2 and len(b) >= 2:
                    day_diffs.append(float(a.median() - b.median()))
            if not day_diffs:
                continue
            arr = np.asarray(day_diffs, dtype=float)
            rows.append({
                "domain": domain, "location": loc, "feature": feat,
                "days_comparable": int(len(arr)),
                "positive_days": int((arr > 0).sum()), "negative_days": int((arr < 0).sum()), "zero_days": int((arr == 0).sum()),
                "positive_day_share": float((arr > 0).mean()),
                "median_daily_difference_100_plus_minus_35_only": float(np.median(arr)),
            })
    return pd.DataFrame(rows)


def direction_symmetry(prep, stage):
    np, pd = require_numeric_stack()
    rows = []
    for domain, df, loc_col, val_col in (
        ("PREPARATION", prep, "window_label", "median"),
        ("STAGE", stage, "stage", "value"),
    ):
        for (stratum, loc, feat), g in df.groupby(["continuation_stratum", loc_col, "feature"], sort=True):
            up = pd.to_numeric(g.loc[g["direction"] > 0, val_col], errors="coerce").dropna()
            dn = pd.to_numeric(g.loc[g["direction"] < 0, val_col], errors="coerce").dropna()
            if len(up) < 3 or len(dn) < 3:
                continue
            rows.append({
                "domain": domain, "continuation_stratum": stratum, "location": loc, "feature": feat,
                "n_up": int(len(up)), "n_down": int(len(dn)),
                "median_up": float(up.median()), "median_down": float(dn.median()),
                "direction_normalized_median_difference_up_minus_down": float(up.median() - dn.median()),
            })
    return pd.DataFrame(rows)


def coverage_report(prep, stage):
    np, pd = require_numeric_stack()
    rows = []
    for (loc, feat), g in prep.groupby(["window_label", "feature"], sort=True):
        cov = pd.to_numeric(g["coverage_rate"], errors="coerce").dropna()
        rows.append({"domain": "PREPARATION", "location": loc, "feature": feat,
                     "events": int(g["event_id"].nunique()), "mean_coverage": float(cov.mean()) if len(cov) else math.nan,
                     "share_events_ge_95pct": float((cov >= 0.95).mean()) if len(cov) else math.nan})
    for (loc, feat), g in stage.groupby(["stage", "feature"], sort=True):
        a = g["available"].fillna(False).astype(bool)
        rows.append({"domain": "STAGE", "location": loc, "feature": feat,
                     "events": int(g["event_id"].nunique()), "mean_coverage": float(a.mean()) if len(a) else math.nan,
                     "share_events_ge_95pct": math.nan})
    return pd.DataFrame(rows)


def check_invariants(event_meta, prep, stage, feature_names):
    rows = []
    def add(name: str, ok: bool, observed: Any, expected: Any):
        rows.append({"check": name, "status": "PASS" if ok else "FAIL", "observed": observed, "expected": expected})
    add("event_ids_unique", bool(event_meta["event_id"].is_unique), int(event_meta["event_id"].nunique()), len(event_meta))
    expected_prep = len(event_meta) * len(PREPARATION_WINDOWS) * len(feature_names)
    add("preparation_row_count", len(prep) == expected_prep, len(prep), expected_prep)
    expected_stage = len(event_meta) * len(STAGE_METHODS) * len(feature_names)
    add("stage_row_count", len(stage) == expected_stage, len(stage), expected_stage)
    add("preparation_no_duplicate_keys", not prep.duplicated(["event_id", "window_label", "feature"]).any(),
        int(prep.duplicated(["event_id", "window_label", "feature"]).sum()), 0)
    add("stage_no_duplicate_keys", not stage.duplicated(["event_id", "stage", "feature"]).any(),
        int(stage.duplicated(["event_id", "stage", "feature"]).sum()), 0)
    add("strata_known", set(event_meta["continuation_stratum"].unique()).issubset({"35_ONLY", "50_ONLY", "75_ONLY", "100_PLUS", "BASE_NOT_35_WITHIN_30M"}),
        sorted(event_meta["continuation_stratum"].unique()), "known strata")
    return rows


def feature_dictionary_frame(features: Sequence[str]):
    _, pd = require_numeric_stack()
    rows = []
    for feat in features:
        fam, desc = FEATURE_DICTIONARY.get(feat, ("RICH_CAUSAL_FLOW_FEATURE", "Audited causal flow-feature column retained for descriptive trajectory analysis."))
        rows.append({"feature": feat, "family": fam, "description": desc,
                     "exact_orderbook_event_semantics": False if "proxy" in feat else None})
    return pd.DataFrame(rows)


def build_findings(summary: Mapping[str, Any], contrast, coverage) -> str:
    lines = [
        "# Phase 7C — Large-move participant-behavior trajectory atlas",
        "",
        f"- Builder version: `{BUILDER_VERSION}`",
        f"- Status: **{summary['status']}**",
        f"- Events: **{summary['events']:,}**",
        f"- Atlas features: **{summary['features']}**",
        "",
        "## Scientific role",
        "",
        "Phase 7C is retrospective top-down discovery. It compares behavior across large-move continuation strata and event stages. It has no non-big-move control group, so differences are not yet called predictive precursors.",
        "",
        "`EXTREMUM_HINDSIGHT` is retained as the physical alignment point. Phase-7B stage markers remain event-conditioned audit objects; no stage occurrence is claimed prospectively predictive here.",
        "",
        "## Order-book proxy warning",
        "",
        "Visible-band addition/removal/replenishment and outer-depth-share fields are state-change proxies derived from reconstructed 5/25-bps depth. They are not exact same-price order additions/refills and can be affected by price movement through the measurement band. Exact replenishment requires a later raw depth-event analysis.",
        "",
        "## Primary descriptive contrast",
        "",
        "The primary atlas contrast is `100_PLUS` versus `35_ONLY`, direction-normalized. The contrast report ranks preparation windows and event stages by robust standardized median differences, with day-level sign consistency reported separately.",
        "",
        "## Next step",
        "",
        "Phase 7D should construct matched non-big-move controls on causal pre-event regime variables before interpreting any discovered trajectory difference as a precursor rather than generic high-volatility/event geometry.",
        "",
    ]
    if not contrast.empty:
        top = contrast.dropna(subset=["robust_standardized_difference"]).copy()
        if len(top):
            top["abs_effect"] = top["robust_standardized_difference"].abs()
            top = top.sort_values("abs_effect", ascending=False).head(10)
            lines += ["## Largest descriptive 100_PLUS vs 35_ONLY differences", ""]
            for r in top.itertuples(index=False):
                lines.append(f"- `{r.domain}/{r.location}/{r.feature}`: robust standardized median difference **{r.robust_standardized_difference:.3f}**.")
            lines.append("")
    return "\n".join(lines)


def self_test() -> None:
    np, pd = require_numeric_stack()
    sec = np.arange(1000, 1010, dtype=np.int64)
    base = pd.DataFrame({
        "receive_second": sec, "book_valid": True,
        "bid_depth_quote_5bps": [100,100,100,90,80,70,80,90,100,110],
        "ask_depth_quote_5bps": [100,100,90,80,70,60,50,55,60,65],
        "bid_depth_quote_10bps": 200.0, "ask_depth_quote_10bps": 200.0,
        "bid_depth_quote_25bps": 400.0, "ask_depth_quote_25bps": 400.0,
        "depth_imbalance_5bps": 0.0,
        "bid_withdrawal_frac_1s": 0.0, "ask_withdrawal_frac_1s": 0.0,
        "bid_withdrawal_z_900s": 0.0, "ask_withdrawal_z_900s": 0.0,
        "signed_quote_flow_5s": np.arange(10, dtype=float) * 1000,
        "aggressive_flow_imbalance_5s": np.linspace(0, .9, 10),
        "return_5s_bps": np.linspace(0, 5, 10),
        "spread_bps": .5, "realized_vol_60s_bps": 5.0, "quote_volume_60s": 1e6,
        "mark_last_dislocation_bps": 0.2, "mark_last_dislocation_z_900s": 0.0,
        "force_buy_count": 0, "force_sell_count": 0, "force_selected_notional_sum": 0,
    })
    f = build_direction_normalized_frame(base, +1)
    assert np.allclose(f["support_depth_quote_5bps"], base["bid_depth_quote_5bps"])
    assert np.allclose(f["opposing_depth_quote_5bps"], base["ask_depth_quote_5bps"])
    assert float(f.loc[4, "opposing_visible_remove_frac_1s_proxy"]) > 0
    assert float(f.loc[7, "opposing_visible_add_frac_1s_proxy"]) > 0
    assert np.allclose(f["dir_return_5s_bps"], base["return_5s_bps"])
    fd = build_direction_normalized_frame(base, -1)
    assert np.allclose(fd["support_depth_quote_5bps"], base["ask_depth_quote_5bps"])
    assert np.allclose(fd["opposing_depth_quote_5bps"], base["bid_depth_quote_5bps"])
    assert np.allclose(fd["dir_return_5s_bps"], -base["return_5s_bps"])
    assert classify_stratum({"time_to_35_sec":100,"time_to_50_sec":200,"time_to_75_sec":300,"time_to_100_sec":400}) == "100_PLUS"
    assert classify_stratum({"time_to_35_sec":100,"time_to_50_sec":200,"time_to_75_sec":None,"time_to_100_sec":None}) == "50_ONLY"
    feats = ["dir_return_5s_bps", "opposing_depth_quote_5bps"]
    seconds, arrays = prepare_feature_arrays(f, feats)
    ev = {"event_id":"T", "onset_date":"2026-01-01", "direction":1, "direction_label":"UP", "continuation_stratum":"100_PLUS", "onset_second":1005}
    rows = aggregate_window_feature_rows(ev, seconds, arrays, "TEST", -2, 2, feats)
    got = {r["feature"]: r for r in rows}
    expected = f[(f["receive_second"] >= 1003) & (f["receive_second"] <= 1007)]["dir_return_5s_bps"].to_numpy(float)
    assert math.isclose(got["dir_return_5s_bps"]["median"], float(np.median(expected)), rel_tol=0, abs_tol=1e-12)
    assert math.isclose(got["dir_return_5s_bps"]["p25"], float(np.quantile(expected, 0.25)), rel_tol=0, abs_tol=1e-12)
    stages = {k: 1005 for k in STAGE_METHODS}
    srows = stage_feature_rows(ev, stages, seconds, arrays, feats)
    assert all(r["available"] for r in srows)

    # Regression: v1.0.1 omitted robust_scale(), so the full 62-day scan
    # completed but continuation-contrast aggregation failed afterward.
    prep_test = pd.DataFrame([
        *[{"continuation_stratum":"100_PLUS", "window_label":"TEST", "feature":"f", "median":v} for v in (1.0, 2.0, 3.0, 4.0, 5.0)],
        *[{"continuation_stratum":"35_ONLY", "window_label":"TEST", "feature":"f", "median":v} for v in (0.0, 0.5, 1.0, 1.5, 2.0)],
    ])
    stage_test = pd.DataFrame(columns=["continuation_stratum", "stage", "feature", "value"])
    ctest = build_contrasts(prep_test, stage_test)
    assert len(ctest) == 1
    assert math.isfinite(float(ctest.iloc[0]["pooled_robust_scale"]))
    print("SELF-TEST PASS")
    print("Validated continuation strata, UP/DOWN direction normalization, bulk feature construction, NumPy window statistics, contrast aggregation/robust scaling, visible-band")
    print("depth-change/replenishment proxies, liquidity-distance proxies, and event-stage")
    print("atlas plumbing without claiming exact order-level replenishment or predictive alpha.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build Phase-7C large-move participant-behavior trajectory atlas")
    p.add_argument("--start-date", default="2026-05-18")
    p.add_argument("--end-date", default="2026-07-18")
    p.add_argument("--symbol", default=DEFAULT_SYMBOL)
    p.add_argument("--catalog-root", type=Path, default=DEFAULT_CATALOG_ROOT)
    p.add_argument("--primitive-root", type=Path, default=DEFAULT_PRIMITIVE_ROOT)
    p.add_argument("--flow-root", type=Path, default=DEFAULT_FLOW_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--primary-horizon-sec", type=int, default=PRIMARY_HORIZON_SEC)
    p.add_argument("--self-test", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        self_test()
        return 0

    np, pd, pa, pq = require_stack()
    start, end = parse_date(args.start_date), parse_date(args.end_date)
    if end < start:
        raise SystemExit("--end-date must be >= --start-date")

    print("Building Phase-7C participant-behavior trajectory atlas...")
    print(f"Builder version       : {BUILDER_VERSION}")
    print(f"Symbol                : {args.symbol}")
    print(f"Dates                 : {start} to {end}")
    print(f"Catalog root          : {args.catalog_root.resolve()}")
    print(f"Phase-6B primitive root: {args.primitive_root.resolve()}")
    print(f"Flow-feature root     : {args.flow_root.resolve()}")
    print("Research role         : retrospective participant-behavior discovery")
    print("Controls              : NONE in 7C; Phase 7D must add matched non-big-move controls")

    print("[1/6] Loading frozen Phase-7A physical events and onset candidates...")
    ep = args.catalog_root / "physical_events.parquet"
    op = args.catalog_root / "onset_candidates.parquet"
    if not ep.exists() or not op.exists():
        raise RuntimeError(f"Missing Phase-7A inputs: {ep} / {op}")
    events = pq.read_table(ep).to_pandas()
    onsets = pq.read_table(op).to_pandas()
    events = events[(events["onset_date"] >= start.isoformat()) & (events["onset_date"] <= end.isoformat())].copy()
    if "symbol" in events.columns:
        events = events[events["symbol"] == args.symbol].copy()
    onsets = onsets[onsets["event_id"].isin(events["event_id"])].copy()
    event_meta = build_event_metadata(events).reset_index(drop=True)
    print(f"      physical events: {len(event_meta):,}")
    print("      strata:")
    for k, v in event_meta["continuation_stratum"].value_counts().sort_index().items():
        print(f"        {k:<24} {int(v):,}")

    onset_by_event: dict[str, dict[str, dict[str, Any]]] = {}
    for eid, g in onsets.groupby("event_id"):
        onset_by_event[str(eid)] = {str(r["onset_method"]): r for r in g.to_dict("records")}

    print("[2/6] Scanning daily one-second behavior context and materializing event trajectories...")
    prep_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    feature_names: list[str] | None = None
    days = daterange(start, end)
    events_by_date = {d: g.copy() for d, g in event_meta.groupby("onset_date")}

    for i, d in enumerate(days, 1):
        day_events = events_by_date.get(d.isoformat())
        if day_events is None or day_events.empty:
            print(f"  [{i:02d}/{len(days):02d}] {d}: no events")
            continue
        # Load prior/current/next days to cover -300s prep and post-extremum stages across midnight.
        ctx_days = [d - timedelta(days=1), d, d + timedelta(days=1)]
        prim_files = [primitive_day_path(args.primitive_root, args.symbol, x) for x in ctx_days]
        prim_files = [p for p in prim_files if p.exists()]
        if not primitive_day_path(args.primitive_root, args.symbol, d).exists():
            raise RuntimeError(f"Missing Phase-6B primitive partition for {d}")
        prim = read_parquet_columns(prim_files, PRIMITIVE_COLUMNS, required=("receive_second", "book_valid"))

        flow_files: list[Path] = []
        for x in ctx_days:
            flow_files.extend(flow_day_files(args.flow_root, args.symbol, x))
        flow = read_parquet_columns(flow_files, FLOW_OPTIONAL_COLUMNS, required=("receive_second",)) if flow_files else pd.DataFrame(columns=["receive_second"])
        # Avoid duplicate overlapping columns; Phase-6B versions remain authoritative for their shared fields.
        overlap = [c for c in flow.columns if c != "receive_second" and c in prim.columns]
        flow = flow.drop(columns=overlap, errors="ignore")
        context = prim.merge(flow, on="receive_second", how="left", validate="one_to_one")
        context = context.sort_values("receive_second", kind="mergesort").reset_index(drop=True)

        event_count = 0
        day_t0 = time.perf_counter()
        # The normalized context depends only on event direction, not event ID.
        # v1.0.0 rebuilt the full 3-day frame once per event; v1.0.1 builds it
        # at most twice per day and reuses compact NumPy feature arrays.
        for direction in (+1, -1):
            direction_events = day_events[day_events["direction"] == direction]
            if direction_events.empty:
                continue
            norm_t0 = time.perf_counter()
            dframe = build_direction_normalized_frame(context, direction)
            if feature_names is None:
                feature_names = atlas_features(dframe)
                if not feature_names:
                    raise RuntimeError("No Phase-7C atlas features resolved from one-second inputs")
                print(f"      resolved atlas features: {len(feature_names)}")
            seconds, feature_arrays = prepare_feature_arrays(dframe, feature_names)
            norm_sec = time.perf_counter() - norm_t0
            ev_t0 = time.perf_counter()
            for ev in direction_events.to_dict("records"):
                for label, lo, hi in PREPARATION_WINDOWS:
                    prep_rows.extend(aggregate_window_feature_rows(ev, seconds, feature_arrays, label, lo, hi, feature_names))
                stages = stage_seconds_for_event(ev, onset_by_event.get(str(ev["event_id"]), {}))
                stage_rows.extend(stage_feature_rows(ev, stages, seconds, feature_arrays, feature_names))
                event_count += 1
            ev_sec = time.perf_counter() - ev_t0
            print(f"      dir={direction:+d}: events={len(direction_events):,}, normalize={norm_sec:.2f}s, atlas={ev_sec:.2f}s")
        print(f"  [{i:02d}/{len(days):02d}] {d}: events={event_count:,}, runtime={time.perf_counter()-day_t0:.2f}s")

    if feature_names is None:
        raise RuntimeError("No event trajectories were built")
    prep = pd.DataFrame(prep_rows)
    stage = pd.DataFrame(stage_rows)

    # Persist the expensive event-level materialization before any report-layer
    # aggregation. If a downstream summary/report bug occurs, the 62-day scan
    # is preserved and can be post-processed without reconstructing trajectories.
    print("      checkpointing event-level Parquet outputs before aggregation...")
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_parquet_atomic(event_meta, args.output_root / "event_metadata.parquet")
    write_parquet_atomic(prep, args.output_root / "preparation_event_features.parquet")
    write_parquet_atomic(stage, args.output_root / "stage_event_features.parquet")

    print("[3/6] Aggregating continuation-strata trajectories and direction symmetry...")
    prep_summary = summarize_preparation(prep)
    stage_summary = summarize_stages(stage)
    contrast = build_contrasts(prep, stage)
    daily = daily_sign_consistency(prep, stage)
    symmetry = direction_symmetry(prep, stage)
    coverage = coverage_report(prep, stage)
    feature_dict = feature_dictionary_frame(feature_names)

    print("[4/6] Checking atlas invariants...")
    checks = check_invariants(event_meta, prep, stage, feature_names)
    failed = [r for r in checks if r["status"] == "FAIL"]

    print("[5/6] Writing metadata for checkpointed event-level outputs...")

    summary = {
        "status": "PASS" if not failed else "FAIL",
        "builder_version": BUILDER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "symbol": args.symbol,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "events": int(len(event_meta)),
        "features": int(len(feature_names)),
        "preparation_rows": int(len(prep)),
        "stage_rows": int(len(stage)),
        "failed_invariants": int(len(failed)),
        "stratum_counts": {str(k): int(v) for k, v in event_meta["continuation_stratum"].value_counts().to_dict().items()},
        "research_role": "RETROSPECTIVE_DISCOVERY_NOT_PREDICTIVE_VALIDATION",
        "replenishment_contract": "VISIBLE_5BPS_BAND_STATE_CHANGE_PROXY_NOT_EXACT_SAME_PRICE_REFILL",
    }
    atomic_write_json(args.output_root / "metadata" / "latest_summary.json", summary)
    atomic_write_json(args.output_root / "metadata" / "run_config.json", {
        "builder_version": BUILDER_VERSION,
        "catalog_root": str(args.catalog_root), "primitive_root": str(args.primitive_root), "flow_root": str(args.flow_root),
        "preparation_windows": list(PREPARATION_WINDOWS), "stages": list(STAGE_METHODS),
        "primary_horizon_sec": int(args.primary_horizon_sec), "feature_names": feature_names,
        "scientific_note": "No non-big-move controls in 7C; Phase 7D required before precursor claims.",
    })

    print("[6/6] Writing reports...")
    args.report_root.mkdir(parents=True, exist_ok=True)
    feature_dict.to_csv(args.report_root / "large_move_behavior_feature_dictionary.csv", index=False)
    prep_summary.to_csv(args.report_root / "large_move_behavior_preparation_trajectory.csv", index=False)
    stage_summary.to_csv(args.report_root / "large_move_behavior_stage_trajectory.csv", index=False)
    contrast.to_csv(args.report_root / "large_move_behavior_continuation_contrast.csv", index=False)
    daily.to_csv(args.report_root / "large_move_behavior_daily_sign_consistency.csv", index=False)
    symmetry.to_csv(args.report_root / "large_move_behavior_direction_symmetry.csv", index=False)
    coverage.to_csv(args.report_root / "large_move_behavior_coverage.csv", index=False)
    pd.DataFrame(checks).to_csv(args.report_root / "large_move_behavior_invariants.csv", index=False)
    atomic_write_text(args.report_root / "large_move_behavior_findings.md", build_findings(summary, contrast, coverage))

    print()
    print("Phase-7C participant-behavior atlas summary")
    print("---------------------------------------------")
    print(f"Status                    : {summary['status']}")
    print(f"Physical events           : {summary['events']:,}")
    print(f"Atlas features            : {summary['features']}")
    print(f"Preparation rows          : {summary['preparation_rows']:,}")
    print(f"Stage rows                : {summary['stage_rows']:,}")
    print(f"Failed invariants         : {summary['failed_invariants']}")
    print(f"Findings                  : {(args.report_root / 'large_move_behavior_findings.md').resolve()}")
    print("No non-event control, alpha, strategy, or profitability claim is made by Phase 7C.")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
