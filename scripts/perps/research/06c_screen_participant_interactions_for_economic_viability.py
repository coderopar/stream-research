#!/usr/bin/env python3
"""Phase 6C: first-passage economic viability of causal participant interactions.

Purpose
-------
Phase 6B established causal participant-behaviour primitives and matched controls.
This script asks a harder, explicitly economic question:

    After the interaction becomes causally tradeable, does the executable price
    path reach a *large* favourable move before a realistic adverse move often
    enough to support the post-cost reward:risk ratio?

This replaces tiny fixed-horizon markout hurdles as the primary endpoint.  The
interaction cohorts remain deliberately small and preregistered; primitive
thresholds are not tuned here.

Causal / execution contract
---------------------------
* A Phase-6B event detected in second t is tradeable no earlier than t+1.
* Companion primitives may only be detected in the trailing interaction window;
  future information never defines cohort membership.
* Long paths enter at the Phase-6A BUY VWAP and are marked/exited at future SELL
  VWAPs.  Short paths enter at SELL VWAP and are marked/exited at future BUY
  VWAPs.
* Therefore spread/crossing/slippage represented by Phase 6A are embedded in the
  executable pre-fee PnL path itself.  Fee scenarios are subtracted separately.
* Barrier targets/stops are defined on executable PRE-FEE PnL.  This intentionally
  exposes cost asymmetry.  Example: +10/-10 bps barriers with 2.5 bps/leg fees
  imply approximately +5 bps net reward versus -15 bps net risk, requiring a
  75% break-even hit rate before considering timeouts.
* If book/execution validity disappears before either barrier is hit, the path is
  censored as PATH_INVALIDATED.  We never look through a data-quality gap.
* If neither barrier is hit by max_path_seconds, the path is TIMEOUT and the
  last valid executable PnL at the horizon is used only for the explicitly
  labelled timeout-exit EV calculation.
* Phase-6A fixed-notional VWAPs are fixed-QUOTE executions, so path PnL is a close
  execution proxy rather than an exact same-base-quantity round trip.  This is
  stated in every report and must be revisited before deployment-grade sizing.

Default first-passage grid
--------------------------
Favourable targets (bps): 25, 35, 50, 75, 100
Adverse stops (bps):      10, 15, 20, 25, 35, 50
Maximum path:             1800 seconds (30 minutes)
Fee sensitivity / leg:    0, 1.5, 3.0, 4.5 bps
Notionals:                 $1k, $5k, $10k, $25k

Primary outputs
---------------
For each cohort / notional / target / stop / fee scenario:
* target-first, stop-first, timeout, and invalid-path rates;
* target-before-stop probability among barrier-resolved paths;
* nominal post-fee reward, risk, R:R, and required break-even hit rate;
* realised mean/median net PnL including timeout exits;
* realised winner/loser magnitudes and realised R:R;
* target-hit-rate margin versus break-even;
* matched-control target-first and expected-PnL differences;
* MFE/MAE and time-to-extreme summaries;
* time-to-target / time-to-stop distributions.

This is an event-study/economic screen, not a deployable strategy backtest and
not an alpha claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


BUILDER_VERSION = "1.1.2"
OUTPUT_SCHEMA_NAME = "perps_participant_interaction_first_passage_v1"
PRIMITIVE_SCHEMA_NAME = "perps_liquidity_event_primitives_v1"

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()
DEFAULT_PRIMITIVE_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / "liquidity_event_primitives_v1"
DEFAULT_EXECUTION_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "execution_foundation_v1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / "participant_interaction_first_passage_v1"
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "perps_participant_interaction_first_passage"

PRIMITIVES = (
    "AGGRESSIVE_FLOW",
    "DEPTH_WITHDRAWAL",
    "FORCE_ORDER",
    "MARK_LAST_DISLOCATION",
)
NON_FORCE_PRIMITIVES = tuple(x for x in PRIMITIVES if x != "FORCE_ORDER")

DEFAULT_TARGETS_BPS = (25.0, 35.0, 50.0, 75.0, 100.0)
DEFAULT_STOPS_BPS = (10.0, 15.0, 20.0, 25.0, 35.0, 50.0)
DEFAULT_NOTIONALS = (1000.0, 5000.0, 10000.0, 25000.0)
DEFAULT_FEE_SCENARIOS_BPS = (0.0, 1.5, 3.0, 4.5)
DEFAULT_MAX_PATH_SECONDS = 1800
DEFAULT_CHUNK_SIZE = 256


@dataclass(frozen=True)
class Config:
    symbol: str = "BTCUSDT"
    interaction_lookback_seconds: int = 5
    min_cohort_events: int = 30
    targets_bps: tuple[float, ...] = DEFAULT_TARGETS_BPS
    stops_bps: tuple[float, ...] = DEFAULT_STOPS_BPS
    notionals: tuple[float, ...] = DEFAULT_NOTIONALS
    fee_scenarios_bps: tuple[float, ...] = DEFAULT_FEE_SCENARIOS_BPS
    max_path_seconds: int = DEFAULT_MAX_PATH_SECONDS
    path_chunk_size: int = DEFAULT_CHUNK_SIZE
    min_path_coverage_rate: float = 0.90
    random_seed: int = 20260815


def stable_id(*parts: Any, n: int = 16) -> str:
    raw = "|".join(str(x) for x in parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:n]


def parse_csv_numbers(text: str, caster=float) -> tuple:
    vals = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            vals.append(caster(token))
    if not vals:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated value")
    return tuple(vals)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="._tmp_", suffix=path.suffix, dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, default=str, sort_keys=True) + "\n")


def atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="._tmp_", suffix=".csv", dir=str(path.parent))
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="._tmp_", suffix=".parquet", dir=str(path.parent))
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
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


def read_parquet(path: Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path, columns=list(columns) if columns else None)

def read_phase6b_partitioned(
    root: Path, dataset: str, symbol: str, start_date: date | None, end_date: date | None,
) -> pd.DataFrame:
    """Read v1.0.3 partitioned Phase-6B shards, with legacy-file fallback."""
    base = root / dataset / f"symbol={symbol}"
    files: list[Path] = []
    if base.exists():
        for child in sorted(base.glob("date=????-??-??")):
            try:
                d = date.fromisoformat(child.name.split("=", 1)[1])
            except Exception:
                continue
            if start_date and d < start_date:
                continue
            if end_date and d > end_date:
                continue
            files.extend(sorted(child.glob("*.parquet")))
    if files:
        frames = [pd.read_parquet(p) for p in files]
        return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    legacy = root / dataset / f"{dataset}.parquet"
    if legacy.exists():
        return read_parquet(legacy)
    raise RuntimeError(f"No Phase-6B {dataset} dataset found under {root}")



def filter_dates(df: pd.DataFrame, start: date | None, end: date | None, seconds_col: str) -> pd.DataFrame:
    if df.empty or (start is None and end is None):
        return df.copy()
    sec = pd.to_numeric(df[seconds_col], errors="coerce")
    keep = pd.Series(True, index=df.index)
    if start is not None:
        keep &= sec >= int(datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc).timestamp())
    if end is not None:
        end_exclusive = int(datetime.combine(end + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc).timestamp())
        keep &= sec < end_exclusive
    return df.loc[keep].copy()


def validate_phase6b(events: pd.DataFrame, controls: pd.DataFrame) -> None:
    required_events = {
        "event_id", "symbol", "primitive", "direction", "detected", "detection_second",
        "tradeable_second", "detection_lag_seconds", "move_realized_before_detection_bps",
    }
    required_controls = {
        "control_id", "parent_event_id", "primitive", "direction", "anchor_second", "tradeable_second",
    }
    missing_e = required_events - set(events.columns)
    missing_c = required_controls - set(controls.columns)
    if missing_e:
        raise RuntimeError(f"Phase-6B events missing columns: {sorted(missing_e)}")
    if missing_c:
        raise RuntimeError(f"Phase-6B controls missing columns: {sorted(missing_c)}")


# ---------------------------------------------------------------------------
# Causal interaction state
# ---------------------------------------------------------------------------

def latest_detections_by_family(detected: pd.DataFrame) -> dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Return primitive -> direction -> (seconds,event_ids), sorted by second."""
    out: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for primitive in PRIMITIVES:
        out[primitive] = {}
        for direction in (-1, 1):
            g = detected.loc[(detected["primitive"] == primitive) & (detected["direction"] == direction)].sort_values("detection_second")
            out[primitive][direction] = (
                pd.to_numeric(g["detection_second"], errors="raise").to_numpy(np.int64),
                g["event_id"].astype(str).to_numpy(),
            )
    return out


def recent_detection(
    index: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]],
    primitive: str,
    direction: int,
    anchor_second: int,
    lookback_seconds: int,
    exclude_event_id: str | None = None,
) -> tuple[bool, float, str | None]:
    seconds, ids = index[primitive][int(direction)]
    if len(seconds) == 0:
        return False, math.nan, None
    pos = int(np.searchsorted(seconds, anchor_second, side="right") - 1)
    while pos >= 0:
        lag = anchor_second - int(seconds[pos])
        if lag > lookback_seconds:
            break
        eid = str(ids[pos])
        if exclude_event_id is None or eid != exclude_event_id:
            return True, float(lag), eid
        pos -= 1
    return False, math.nan, None


def build_interaction_features(events: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    detected = events.loc[
        events["detected"].fillna(False).astype(bool)
        & events["direction"].isin([-1, 1])
    ].copy()
    detected = detected.sort_values(["detection_second", "primitive", "event_id"], kind="mergesort").reset_index(drop=True)
    idx = latest_detections_by_family(detected)
    rows: list[dict[str, Any]] = []

    for ev in detected.itertuples(index=False):
        dsec = int(ev.detection_second)
        edir = int(ev.direction)
        eid = str(ev.event_id)
        primitive = str(ev.primitive)
        row = ev._asdict()
        same_count = 0
        opp_count = 0
        same_families: list[str] = []
        opp_families: list[str] = []

        for fam in PRIMITIVES:
            if fam == primitive:
                # Do not allow the anchor event to count itself as confirmation.
                exclude = eid
            else:
                exclude = None
            present_same, lag_same, id_same = recent_detection(
                idx, fam, edir, dsec, cfg.interaction_lookback_seconds, exclude_event_id=exclude
            )
            present_opp, lag_opp, id_opp = recent_detection(
                idx, fam, -edir, dsec, cfg.interaction_lookback_seconds, exclude_event_id=None
            )
            row[f"recent_same_{fam.lower()}"] = bool(present_same)
            row[f"recent_same_{fam.lower()}_lag_s"] = lag_same
            row[f"recent_same_{fam.lower()}_event_id"] = id_same
            row[f"recent_opp_{fam.lower()}"] = bool(present_opp)
            row[f"recent_opp_{fam.lower()}_lag_s"] = lag_opp
            row[f"recent_opp_{fam.lower()}_event_id"] = id_opp
            if fam != primitive and present_same:
                same_count += 1
                same_families.append(fam)
            if fam != primitive and present_opp:
                opp_count += 1
                opp_families.append(fam)

        row["same_direction_other_family_count"] = same_count
        row["opposite_direction_other_family_count"] = opp_count
        row["same_direction_other_families"] = "+".join(sorted(same_families))
        row["opposite_direction_other_families"] = "+".join(sorted(opp_families))
        rows.append(row)

    return pd.DataFrame(rows)


def add_membership(
    out: list[dict[str, Any]],
    ev: pd.Series,
    cohort: str,
    orientation: str,
    trade_direction: int,
    rationale: str,
) -> None:
    out.append({
        "membership_id": f"CM-{stable_id(ev['event_id'], cohort, orientation)}",
        "event_id": str(ev["event_id"]),
        "symbol": str(ev["symbol"]),
        "primitive": str(ev["primitive"]),
        "event_direction": int(ev["direction"]),
        "trade_direction": int(trade_direction),
        "orientation": orientation,
        "cohort": cohort,
        "cohort_rationale": rationale,
        "detection_second": int(ev["detection_second"]),
        "tradeable_second": int(ev["tradeable_second"]),
        "detection_lag_seconds": safe_float(ev.get("detection_lag_seconds")),
        "move_realized_before_detection_bps": safe_float(ev.get("move_realized_before_detection_bps")),
        "same_direction_other_family_count": int(ev.get("same_direction_other_family_count", 0)),
        "opposite_direction_other_family_count": int(ev.get("opposite_direction_other_family_count", 0)),
        "same_direction_other_families": str(ev.get("same_direction_other_families", "")),
        "opposite_direction_other_families": str(ev.get("opposite_direction_other_families", "")),
        "event_date": datetime.fromtimestamp(int(ev["detection_second"]), tz=timezone.utc).date().isoformat(),
    })


def build_cohort_membership(features: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out: list[dict[str, Any]] = []
    for _, ev in features.iterrows():
        primitive = str(ev["primitive"])
        direction = int(ev["direction"])
        same_n = int(ev["same_direction_other_family_count"])
        opp_n = int(ev["opposite_direction_other_family_count"])

        if primitive != "FORCE_ORDER":
            add_membership(out, ev, f"BASE::{primitive}::CONT", "CONTINUATION", direction,
                           "Base detected primitive, traded in primitive direction")
            if same_n >= 1:
                add_membership(out, ev, f"CONFIRM_ANY::{primitive}::CONT", "CONTINUATION", direction,
                               ">=1 other same-direction primitive detected causally in trailing window")
            if same_n >= 2:
                add_membership(out, ev, f"CONFIRM_2PLUS::{primitive}::CONT", "CONTINUATION", direction,
                               ">=2 other same-direction primitive families detected causally in trailing window")
            if opp_n >= 1:
                add_membership(out, ev, f"CONFLICT_ANY::{primitive}::CONT", "CONTINUATION", direction,
                               ">=1 other opposite-direction primitive detected causally in trailing window")

        if primitive == "DEPTH_WITHDRAWAL":
            flow = bool(ev.get("recent_same_aggressive_flow", False))
            mark = bool(ev.get("recent_same_mark_last_dislocation", False))
            if flow:
                add_membership(out, ev, "DW_FLOW_CONT", "CONTINUATION", direction,
                               "Depth withdrawal plus same-direction aggressive flow")
            if mark:
                add_membership(out, ev, "DW_MARK_CONT", "CONTINUATION", direction,
                               "Depth withdrawal plus same-direction mark-last dislocation")
            if flow and mark:
                add_membership(out, ev, "DW_FLOW_MARK_CONT", "CONTINUATION", direction,
                               "Depth withdrawal plus same-direction aggressive flow and mark-last dislocation")

        if primitive == "FORCE_ORDER":
            # Reversal trade uses opposite direction and is recomputed from execution books.
            rev = -direction
            add_membership(out, ev, "FORCE_REV_BASE", "REVERSAL", rev,
                           "Observed forceOrder, traded opposite the forceOrder direction")
            flow = bool(ev.get("recent_same_aggressive_flow", False))
            depth = bool(ev.get("recent_same_depth_withdrawal", False))
            mark = bool(ev.get("recent_same_mark_last_dislocation", False))
            if flow:
                add_membership(out, ev, "FORCE_REV_FLOW", "REVERSAL", rev,
                               "ForceOrder preceded/accompanied by same-direction aggressive flow; trade reversal")
            if depth:
                add_membership(out, ev, "FORCE_REV_DEPTH", "REVERSAL", rev,
                               "ForceOrder preceded/accompanied by same-direction depth withdrawal; trade reversal")
            if mark:
                add_membership(out, ev, "FORCE_REV_MARK", "REVERSAL", rev,
                               "ForceOrder preceded/accompanied by same-direction mark dislocation; trade reversal")
            if flow and depth:
                add_membership(out, ev, "FORCE_REV_FLOW_DEPTH", "REVERSAL", rev,
                               "ForceOrder plus same-direction aggressive flow and depth withdrawal; trade reversal")

    df = pd.DataFrame(out)
    if df.empty:
        return df
    return df.sort_values(["cohort", "detection_second", "event_id"], kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Exact execution-side markouts for arbitrary continuation/reversal orientation
# ---------------------------------------------------------------------------

def execution_path(root: Path, symbol: str, day: date) -> Path:
    return root / f"symbol={symbol}" / f"date={day.isoformat()}" / f"part-{day.isoformat()}.parquet"


def execution_columns(notionals: Sequence[float]) -> list[str]:
    cols = ["receive_second", "book_valid", "mid_price"]
    for n in notionals:
        tag = str(int(round(n)))
        for side in ("buy", "sell"):
            cols.extend([
                f"{side}_vwap_{tag}",
                f"{side}_complete_{tag}",
                f"{side}_crossing_cost_bps_{tag}",
            ])
    return cols


def read_execution_days(root: Path, symbol: str, days: Iterable[date], notionals: Sequence[float]) -> pd.DataFrame:
    frames = []
    cols = execution_columns(notionals)
    for d in sorted(set(days)):
        p = execution_path(root, symbol, d)
        if p.exists():
            frames.append(pd.read_parquet(p, columns=cols))
    if not frames:
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("receive_second", kind="mergesort").drop_duplicates("receive_second", keep="last").reset_index(drop=True)


def lookup_positions(sorted_seconds: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = np.searchsorted(sorted_seconds, targets, side="left")
    valid = pos < len(sorted_seconds)
    safe = np.clip(pos, 0, max(len(sorted_seconds) - 1, 0))
    if len(sorted_seconds):
        valid &= sorted_seconds[safe] == targets
    return safe.astype(np.int64), valid


def build_anchor_table(membership: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if membership.empty:
        return pd.DataFrame()

    ev = membership.copy()
    rows.append(pd.DataFrame({
        "cohort": ev["cohort"].astype(str),
        "membership_id": ev["membership_id"].astype(str),
        "anchor_id": ev["event_id"].astype(str),
        "parent_event_id": ev["event_id"].astype(str),
        "sample_type": "EVENT",
        "trade_direction": pd.to_numeric(ev["trade_direction"], errors="raise").astype(np.int8),
        "anchor_second": pd.to_numeric(ev["detection_second"], errors="raise").astype(np.int64),
        "tradeable_second": pd.to_numeric(ev["tradeable_second"], errors="raise").astype(np.int64),
        "event_date": ev["event_date"].astype(str),
    }))

    controls_by_parent = {k: g for k, g in controls.groupby("parent_event_id", sort=False)}
    control_rows: list[dict[str, Any]] = []
    for m in membership.itertuples(index=False):
        cg = controls_by_parent.get(str(m.event_id))
        if cg is None:
            continue
        for c in cg.itertuples(index=False):
            control_rows.append({
                "cohort": str(m.cohort),
                "membership_id": str(m.membership_id),
                "anchor_id": str(c.control_id),
                "parent_event_id": str(m.event_id),
                "sample_type": "CONTROL",
                "trade_direction": int(m.trade_direction),
                "anchor_second": int(c.anchor_second),
                "tradeable_second": int(c.tradeable_second),
                "event_date": str(m.event_date),
            })
    if control_rows:
        rows.append(pd.DataFrame(control_rows))
    return pd.concat(rows, ignore_index=True)



# ---------------------------------------------------------------------------
# Executable first-passage path profiles
# ---------------------------------------------------------------------------

def bps_tag(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else str(x).replace(".", "p")


def anchor_path_key(sample_type: str, anchor_id: str, direction: int, tradeable_second: int) -> str:
    return f"PK-{stable_id(sample_type, anchor_id, int(direction), int(tradeable_second))}"


def add_path_keys(anchors: pd.DataFrame) -> pd.DataFrame:
    out = anchors.copy()
    out["path_key"] = [
        anchor_path_key(st, aid, int(d), int(ts))
        for st, aid, d, ts in zip(
            out["sample_type"].astype(str), out["anchor_id"].astype(str),
            out["trade_direction"], out["tradeable_second"]
        )
    ]
    out["anchor_date"] = pd.to_datetime(out["tradeable_second"], unit="s", utc=True).dt.date.astype(str)
    return out


def _first_true_offset(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return first 1-based column offset and any-hit flag for a 2D bool mask."""
    any_hit = np.any(mask, axis=1)
    first = np.argmax(mask, axis=1).astype(np.int32) + 1
    first[~any_hit] = -1
    return first, any_hit


def build_path_profiles(
    anchors: pd.DataFrame,
    execution_root: Path,
    symbol: str,
    cfg: Config,
) -> pd.DataFrame:
    """Compute each unique anchor/direction executable path once per notional.

    The path starts at tradeable_second.  Barrier evaluation starts one second
    later and stops immediately before the first invalid/unexecutable second.
    """
    if anchors.empty:
        return pd.DataFrame()
    a = add_path_keys(anchors)
    uniq = a[["path_key", "anchor_id", "sample_type", "trade_direction", "tradeable_second", "anchor_date"]].drop_duplicates("path_key")
    uniq["entry_date_obj"] = pd.to_datetime(uniq["tradeable_second"], unit="s", utc=True).dt.date
    targets = np.asarray(cfg.targets_bps, dtype=float)
    stops = np.asarray(cfg.stops_bps, dtype=float)
    offsets = np.arange(1, cfg.max_path_seconds + 1, dtype=np.int64)
    rows: list[dict[str, Any]] = []

    for entry_date, gday in uniq.groupby("entry_date_obj", sort=True):
        execution = read_execution_days(
            execution_root, symbol, [entry_date, entry_date + timedelta(days=1)], cfg.notionals
        )
        if execution.empty:
            for r in gday.itertuples(index=False):
                for n in cfg.notionals:
                    rows.append({
                        "path_key": r.path_key, "anchor_id": r.anchor_id, "sample_type": r.sample_type,
                        "trade_direction": int(r.trade_direction), "tradeable_second": int(r.tradeable_second),
                        "anchor_date": r.anchor_date, "quote_notional": float(n),
                        "entry_found": False, "entry_book_valid": False, "entry_complete": False,
                        "path_invalidated": True, "first_invalid_offset_s": 0,
                        "path_seconds_observed": 0, "path_full_horizon": False,
                    })
            continue

        ex_sec = pd.to_numeric(execution["receive_second"], errors="raise").to_numpy(np.int64)
        book_valid = execution["book_valid"].fillna(False).astype(bool).to_numpy()
        entry_secs_all = pd.to_numeric(gday["tradeable_second"], errors="raise").to_numpy(np.int64)
        entry_pos_all, entry_found_all = lookup_positions(ex_sec, entry_secs_all)
        day_records = gday.reset_index(drop=True)

        for notional in cfg.notionals:
            tag = str(int(round(notional)))
            buy_vwap = pd.to_numeric(execution[f"buy_vwap_{tag}"], errors="coerce").to_numpy(float)
            sell_vwap = pd.to_numeric(execution[f"sell_vwap_{tag}"], errors="coerce").to_numpy(float)
            buy_complete = execution[f"buy_complete_{tag}"].fillna(False).astype(bool).to_numpy()
            sell_complete = execution[f"sell_complete_{tag}"].fillna(False).astype(bool).to_numpy()

            for lo in range(0, len(day_records), cfg.path_chunk_size):
                hi = min(lo + cfg.path_chunk_size, len(day_records))
                gg = day_records.iloc[lo:hi].reset_index(drop=True)
                m = len(gg)
                entry_sec = pd.to_numeric(gg["tradeable_second"], errors="raise").to_numpy(np.int64)
                direction = pd.to_numeric(gg["trade_direction"], errors="raise").to_numpy(np.int8)
                ep = entry_pos_all[lo:hi]
                efound = entry_found_all[lo:hi]
                long = direction > 0

                entry_price = np.where(long, buy_vwap[ep], sell_vwap[ep])
                entry_complete = np.where(long, buy_complete[ep], sell_complete[ep])
                entry_good = efound & book_valid[ep] & entry_complete & np.isfinite(entry_price) & (entry_price > 0)

                pos = ep[:, None] + offsets[None, :]
                in_bounds = pos < len(ex_sec)
                safe_pos = np.minimum(pos, max(len(ex_sec) - 1, 0))
                expected_sec = entry_sec[:, None] + offsets[None, :]
                exact_sec = in_bounds & (ex_sec[safe_pos] == expected_sec)
                exit_price = np.where(long[:, None], sell_vwap[safe_pos], buy_vwap[safe_pos])
                exit_complete = np.where(long[:, None], sell_complete[safe_pos], buy_complete[safe_pos])
                valid = exact_sec & book_valid[safe_pos] & exit_complete & np.isfinite(exit_price) & (exit_price > 0)
                valid &= entry_good[:, None]

                # Conservative censoring: after the first invalid second, nothing later is observable.
                invalid_any = np.any(~valid, axis=1)
                first_invalid_idx = np.argmax(~valid, axis=1)
                first_invalid_offset = np.where(invalid_any, first_invalid_idx + 1, cfg.max_path_seconds + 1).astype(np.int32)
                prefix = offsets[None, :] < first_invalid_offset[:, None]
                prefix &= valid

                pnl = np.where(
                    long[:, None],
                    (exit_price / entry_price[:, None] - 1.0) * 1e4,
                    (entry_price[:, None] / exit_price - 1.0) * 1e4,
                )
                pnl[~prefix] = np.nan
                observed = np.sum(prefix, axis=1).astype(np.int32)
                any_obs = observed > 0
                terminal_col = np.maximum(observed - 1, 0)
                terminal_pnl = pnl[np.arange(m), terminal_col]
                terminal_pnl[~any_obs] = np.nan

                mfe = np.full(m, np.nan)
                mae = np.full(m, np.nan)
                time_mfe = np.full(m, -1, dtype=np.int32)
                time_mae = np.full(m, -1, dtype=np.int32)
                if np.any(any_obs):
                    idxs = np.flatnonzero(any_obs)
                    sub = pnl[idxs]
                    mfe[idxs] = np.nanmax(sub, axis=1)
                    mae[idxs] = np.nanmin(sub, axis=1)
                    # Replace NaN with extremes only for arg funcs.
                    time_mfe[idxs] = np.nanargmax(sub, axis=1).astype(np.int32) + 1
                    time_mae[idxs] = np.nanargmin(sub, axis=1).astype(np.int32) + 1

                target_first: dict[float, tuple[np.ndarray, np.ndarray]] = {}
                for t in targets:
                    hit_mask = prefix & (pnl >= t)
                    first, hit = _first_true_offset(hit_mask)
                    hit_pnl = np.full(m, np.nan)
                    if np.any(hit):
                        ii = np.flatnonzero(hit)
                        hit_pnl[ii] = pnl[ii, first[ii] - 1]
                    target_first[float(t)] = (first, hit_pnl)

                stop_first: dict[float, tuple[np.ndarray, np.ndarray]] = {}
                for s in stops:
                    hit_mask = prefix & (pnl <= -s)
                    first, hit = _first_true_offset(hit_mask)
                    hit_pnl = np.full(m, np.nan)
                    if np.any(hit):
                        ii = np.flatnonzero(hit)
                        hit_pnl[ii] = pnl[ii, first[ii] - 1]
                    stop_first[float(s)] = (first, hit_pnl)

                for i, r in enumerate(gg.itertuples(index=False)):
                    rec: dict[str, Any] = {
                        "path_key": str(r.path_key), "anchor_id": str(r.anchor_id),
                        "sample_type": str(r.sample_type), "trade_direction": int(r.trade_direction),
                        "tradeable_second": int(r.tradeable_second), "anchor_date": str(r.anchor_date),
                        "quote_notional": float(notional), "entry_found": bool(efound[i]),
                        "entry_book_valid": bool(efound[i] and book_valid[ep[i]]),
                        "entry_complete": bool(entry_complete[i]) if efound[i] else False,
                        "entry_vwap": float(entry_price[i]) if entry_good[i] else math.nan,
                        "path_invalidated": bool(first_invalid_offset[i] <= cfg.max_path_seconds),
                        "first_invalid_offset_s": int(first_invalid_offset[i]) if first_invalid_offset[i] <= cfg.max_path_seconds else -1,
                        "path_seconds_observed": int(observed[i]),
                        "path_full_horizon": bool(observed[i] == cfg.max_path_seconds),
                        "terminal_exec_pre_fee_pnl_bps": float(terminal_pnl[i]) if math.isfinite(float(terminal_pnl[i])) else math.nan,
                        "mfe_exec_pre_fee_bps": float(mfe[i]) if math.isfinite(float(mfe[i])) else math.nan,
                        "mae_exec_pre_fee_bps": float(mae[i]) if math.isfinite(float(mae[i])) else math.nan,
                        "time_to_mfe_s": int(time_mfe[i]), "time_to_mae_s": int(time_mae[i]),
                        "execution_contract": "PHASE6A_FIXED_QUOTE_VWAP_PROXY_NOT_EXACT_SAME_BASE_QTY",
                    }
                    for t in targets:
                        ft, hp = target_first[float(t)]
                        tg = bps_tag(float(t))
                        rec[f"target_{tg}_first_hit_s"] = int(ft[i])
                        rec[f"target_{tg}_hit_exec_pre_fee_pnl_bps"] = float(hp[i]) if math.isfinite(float(hp[i])) else math.nan
                    for s in stops:
                        fs, hp = stop_first[float(s)]
                        sg = bps_tag(float(s))
                        rec[f"stop_{sg}_first_hit_s"] = int(fs[i])
                        rec[f"stop_{sg}_hit_exec_pre_fee_pnl_bps"] = float(hp[i]) if math.isfinite(float(hp[i])) else math.nan
                    rows.append(rec)

    return pd.DataFrame(rows)


def derive_outcomes(frame: pd.DataFrame, target: float, stop: float) -> pd.DataFrame:
    """Derive pre-fee first-passage outcomes without fee expansion."""
    tg, sg = bps_tag(target), bps_tag(stop)
    tsec = pd.to_numeric(frame[f"target_{tg}_first_hit_s"], errors="coerce").fillna(-1).to_numpy(np.int32)
    ssec = pd.to_numeric(frame[f"stop_{sg}_first_hit_s"], errors="coerce").fillna(-1).to_numpy(np.int32)
    thit = tsec > 0
    shit = ssec > 0
    target_first = thit & (~shit | (tsec < ssec))
    stop_first = shit & (~thit | (ssec < tsec))
    path_invalid = frame["path_invalidated"].fillna(True).astype(bool).to_numpy()
    terminal = pd.to_numeric(frame["terminal_exec_pre_fee_pnl_bps"], errors="coerce").to_numpy(float)
    t_pnl = pd.to_numeric(frame[f"target_{tg}_hit_exec_pre_fee_pnl_bps"], errors="coerce").to_numpy(float)
    s_pnl = pd.to_numeric(frame[f"stop_{sg}_hit_exec_pre_fee_pnl_bps"], errors="coerce").to_numpy(float)

    outcome = np.full(len(frame), "TIMEOUT", dtype=object)
    outcome[target_first] = "TARGET_FIRST"
    outcome[stop_first] = "STOP_FIRST"
    unresolved = ~(target_first | stop_first)
    outcome[unresolved & path_invalid] = "PATH_INVALIDATED"
    no_entry = ~(frame["entry_found"].fillna(False).astype(bool).to_numpy()
                 & frame["entry_book_valid"].fillna(False).astype(bool).to_numpy()
                 & frame["entry_complete"].fillna(False).astype(bool).to_numpy())
    outcome[no_entry] = "NO_VALID_ENTRY"

    realized = terminal.copy()
    realized[target_first] = t_pnl[target_first]
    realized[stop_first] = s_pnl[stop_first]
    realized[np.isin(outcome, ["PATH_INVALIDATED", "NO_VALID_ENTRY"])] = np.nan
    hit_time = np.full(len(frame), -1, dtype=np.int32)
    hit_time[target_first] = tsec[target_first]
    hit_time[stop_first] = ssec[stop_first]
    hit_time[outcome == "TIMEOUT"] = pd.to_numeric(frame.loc[outcome == "TIMEOUT", "path_seconds_observed"], errors="coerce").fillna(0).to_numpy(np.int32)

    out = frame[["path_key", "anchor_id", "sample_type", "trade_direction", "tradeable_second", "anchor_date", "quote_notional"]].copy()
    out["target_bps"] = float(target)
    out["stop_bps"] = float(stop)
    out["outcome"] = outcome
    out["outcome_time_s"] = hit_time
    out["realized_exec_pre_fee_pnl_bps"] = realized
    out["mfe_exec_pre_fee_bps"] = pd.to_numeric(frame["mfe_exec_pre_fee_bps"], errors="coerce")
    out["mae_exec_pre_fee_bps"] = pd.to_numeric(frame["mae_exec_pre_fee_bps"], errors="coerce")
    out["path_full_horizon"] = frame["path_full_horizon"].fillna(False).astype(bool)
    return out


# ---------------------------------------------------------------------------
# Cohort-level economic summaries
# ---------------------------------------------------------------------------

def _safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else math.nan


def _safe_median(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else math.nan


def _rate(mask: np.ndarray) -> float:
    return float(np.mean(mask)) if len(mask) else math.nan


def summarize_one_sample(outcomes: pd.DataFrame, fee: float) -> dict[str, Any]:
    o = outcomes["outcome"].astype(str).to_numpy()
    usable = ~np.isin(o, ["PATH_INVALIDATED", "NO_VALID_ENTRY"])
    target = o == "TARGET_FIRST"
    stop = o == "STOP_FIRST"
    timeout = o == "TIMEOUT"
    resolved = target | stop
    pre = pd.to_numeric(outcomes["realized_exec_pre_fee_pnl_bps"], errors="coerce").to_numpy(float)
    net = pre - 2.0 * float(fee)
    valid_net = net[usable & np.isfinite(net)]
    wins = net[target & np.isfinite(net)]
    losses = net[stop & np.isfinite(net)]
    time = pd.to_numeric(outcomes["outcome_time_s"], errors="coerce").to_numpy(float)
    mfe = pd.to_numeric(outcomes["mfe_exec_pre_fee_bps"], errors="coerce").to_numpy(float)
    mae = pd.to_numeric(outcomes["mae_exec_pre_fee_bps"], errors="coerce").to_numpy(float)
    mean_win = _safe_mean(wins)
    mean_loss = _safe_mean(losses)
    rr = mean_win / abs(mean_loss) if math.isfinite(mean_win) and math.isfinite(mean_loss) and mean_win > 0 and mean_loss < 0 else math.nan
    be = abs(mean_loss) / (mean_win + abs(mean_loss)) if math.isfinite(rr) else math.nan
    resolved_hit = float(np.sum(target) / np.sum(resolved)) if np.sum(resolved) else math.nan
    return {
        "n": int(len(outcomes)),
        "usable_n": int(np.sum(usable)),
        "usable_rate": _rate(usable),
        "target_first_rate_all": _rate(target & usable),
        "stop_first_rate_all": _rate(stop & usable),
        "timeout_rate_all": _rate(timeout & usable),
        "path_invalid_rate": _rate(o == "PATH_INVALIDATED"),
        "no_valid_entry_rate": _rate(o == "NO_VALID_ENTRY"),
        "target_first_rate_resolved": resolved_hit,
        "mean_net_pnl_bps_timeout_exit": _safe_mean(valid_net),
        "median_net_pnl_bps_timeout_exit": _safe_median(valid_net),
        "share_net_positive_timeout_exit": _rate(valid_net > 0) if len(valid_net) else math.nan,
        "mean_target_win_net_bps": mean_win,
        "mean_stop_loss_net_bps": mean_loss,
        "realized_reward_risk": rr,
        "realized_break_even_hit_rate": be,
        "resolved_hit_rate_minus_realized_break_even": resolved_hit - be if math.isfinite(resolved_hit) and math.isfinite(be) else math.nan,
        "median_time_to_target_s": _safe_median(time[target]),
        "median_time_to_stop_s": _safe_median(time[stop]),
        "median_mfe_pre_fee_bps": _safe_median(mfe[usable]),
        "median_mae_pre_fee_bps": _safe_median(mae[usable]),
    }


def first_passage_summary(
    anchors_with_profiles: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    if anchors_with_profiles.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = ["cohort", "quote_notional"]
    for (cohort, notional), g in anchors_with_profiles.groupby(group_cols, sort=True):
        ev = g.loc[g["sample_type"] == "EVENT"].copy()
        ct = g.loc[g["sample_type"] == "CONTROL"].copy()
        if ev.empty:
            continue
        for target in cfg.targets_bps:
            for stop in cfg.stops_bps:
                ev_o = derive_outcomes(ev, target, stop)
                ct_o = derive_outcomes(ct, target, stop) if not ct.empty else pd.DataFrame()
                for fee in cfg.fee_scenarios_bps:
                    es = summarize_one_sample(ev_o, fee)
                    cs = summarize_one_sample(ct_o, fee) if not ct_o.empty else {}
                    nominal_reward = float(target) - 2.0 * float(fee)
                    nominal_risk = float(stop) + 2.0 * float(fee)
                    nominal_rr = nominal_reward / nominal_risk if nominal_reward > 0 and nominal_risk > 0 else math.nan
                    nominal_be = nominal_risk / (nominal_reward + nominal_risk) if nominal_reward > 0 else 1.0

                    # Parent-paired comparison of terminal/barrier net PnL and target-first indicator.
                    paired_mean_pnl = math.nan
                    paired_target_rate = math.nan
                    if not ct_o.empty:
                        e_tmp = ev_o[["anchor_id", "outcome", "realized_exec_pre_fee_pnl_bps"]].copy()
                        e_tmp["net"] = pd.to_numeric(e_tmp["realized_exec_pre_fee_pnl_bps"], errors="coerce") - 2.0 * fee
                        e_usable = ~e_tmp["outcome"].isin(["PATH_INVALIDATED", "NO_VALID_ENTRY"])
                        e_tmp["target"] = np.where(e_usable, (e_tmp["outcome"] == "TARGET_FIRST").astype(float), np.nan)
                        e_tmp = e_tmp.rename(columns={"anchor_id": "parent_event_id"}).drop_duplicates("parent_event_id")
                        c_tmp = ct_o[["anchor_id", "outcome", "realized_exec_pre_fee_pnl_bps"]].copy()
                        # Map control anchor back to parent through the already merged column if present.
                        if "parent_event_id" in ct.columns:
                            parent_map = ct[["anchor_id", "parent_event_id"]].drop_duplicates("anchor_id")
                            c_tmp = c_tmp.merge(parent_map, on="anchor_id", how="left")
                            c_tmp["net"] = pd.to_numeric(c_tmp["realized_exec_pre_fee_pnl_bps"], errors="coerce") - 2.0 * fee
                            c_usable = ~c_tmp["outcome"].isin(["PATH_INVALIDATED", "NO_VALID_ENTRY"])
                            c_tmp["target"] = np.where(c_usable, (c_tmp["outcome"] == "TARGET_FIRST").astype(float), np.nan)
                            cg = c_tmp.groupby("parent_event_id", sort=False).agg(control_net=("net", "mean"), control_target=("target", "mean"))
                            eg = e_tmp.set_index("parent_event_id")
                            common = eg.index.intersection(cg.index)
                            if len(common):
                                paired_mean_pnl = float(np.nanmean(eg.loc[common, "net"] - cg.loc[common, "control_net"]))
                                paired_target_rate = float(np.nanmean(eg.loc[common, "target"] - cg.loc[common, "control_target"]))

                    row = {
                        "cohort": str(cohort), "quote_notional": float(notional),
                        "target_bps_pre_fee_executable": float(target), "stop_bps_pre_fee_executable": float(stop),
                        "fee_bps_per_leg": float(fee), "max_path_seconds": int(cfg.max_path_seconds),
                        "nominal_net_reward_bps": nominal_reward, "nominal_net_risk_bps": nominal_risk,
                        "nominal_post_fee_reward_risk": nominal_rr,
                        "nominal_break_even_hit_rate": nominal_be,
                        "large_move_target_ge_50bps": bool(target >= 50.0),
                        "fee_contract": "SENSITIVITY_ONLY_NOT_ACCOUNT_FEE",
                        "execution_contract": "PHASE6A_FIXED_QUOTE_VWAP_PROXY_NOT_EXACT_SAME_BASE_QTY",
                        "event_minus_control_mean_net_pnl_bps": paired_mean_pnl,
                        "event_minus_control_target_first_rate_pp": paired_target_rate * 100.0 if math.isfinite(paired_target_rate) else math.nan,
                    }
                    row.update({f"event_{k}": v for k, v in es.items()})
                    row.update({f"control_{k}": v for k, v in cs.items()})
                    event_resolved = es.get("target_first_rate_resolved", math.nan)
                    row["event_resolved_hit_rate_minus_nominal_break_even"] = (
                        event_resolved - nominal_be if math.isfinite(event_resolved) else math.nan
                    )
                    rows.append(row)
    return pd.DataFrame(rows)


def mfe_mae_summary(anchors_with_profiles: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if anchors_with_profiles.empty:
        return pd.DataFrame()
    rows = []
    for (cohort, notional, sample_type), g in anchors_with_profiles.groupby(["cohort", "quote_notional", "sample_type"], sort=True):
        mfe = pd.to_numeric(g["mfe_exec_pre_fee_bps"], errors="coerce").to_numpy(float)
        mae = pd.to_numeric(g["mae_exec_pre_fee_bps"], errors="coerce").to_numpy(float)
        valid = np.isfinite(mfe) & np.isfinite(mae)
        base = {
            "cohort": cohort, "quote_notional": float(notional), "sample_type": sample_type,
            "n": int(len(g)), "usable_n": int(np.sum(valid)), "usable_rate": _rate(valid),
            "median_mfe_pre_fee_bps": _safe_median(mfe[valid]), "p75_mfe_pre_fee_bps": float(np.nanquantile(mfe[valid], .75)) if np.sum(valid) else math.nan,
            "p90_mfe_pre_fee_bps": float(np.nanquantile(mfe[valid], .90)) if np.sum(valid) else math.nan,
            "median_mae_pre_fee_bps": _safe_median(mae[valid]), "p25_mae_pre_fee_bps": float(np.nanquantile(mae[valid], .25)) if np.sum(valid) else math.nan,
        }
        for t in cfg.targets_bps:
            base[f"share_mfe_ge_{bps_tag(t)}bps"] = _rate(mfe[valid] >= t) if np.sum(valid) else math.nan
        for s in cfg.stops_bps:
            base[f"share_mae_le_minus_{bps_tag(s)}bps"] = _rate(mae[valid] <= -s) if np.sum(valid) else math.nan
        rows.append(base)
    return pd.DataFrame(rows)


def daily_summary(anchors_with_profiles: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if anchors_with_profiles.empty:
        return pd.DataFrame()
    rows = []
    # Restrict daily robustness to economically central 50bps target and 20/25bps stops,
    # all fee scenarios, avoiding an unnecessarily huge report.
    targets = [t for t in cfg.targets_bps if t >= 50.0]
    stops = [s for s in cfg.stops_bps if s in (20.0, 25.0, 35.0)]
    for (cohort, notional, day), g in anchors_with_profiles.groupby(["cohort", "quote_notional", "event_date"], sort=True):
        ev = g.loc[g["sample_type"] == "EVENT"]
        if ev.empty:
            continue
        for target in targets:
            for stop in stops:
                o = derive_outcomes(ev, target, stop)
                for fee in cfg.fee_scenarios_bps:
                    ss = summarize_one_sample(o, fee)
                    rows.append({
                        "cohort": cohort, "quote_notional": float(notional), "event_date": day,
                        "target_bps": float(target), "stop_bps": float(stop), "fee_bps_per_leg": float(fee),
                        "event_n": ss["n"], "usable_rate": ss["usable_rate"],
                        "target_first_rate_all": ss["target_first_rate_all"],
                        "target_first_rate_resolved": ss["target_first_rate_resolved"],
                        "mean_net_pnl_bps_timeout_exit": ss["mean_net_pnl_bps_timeout_exit"],
                        "realized_reward_risk": ss["realized_reward_risk"],
                    })
    return pd.DataFrame(rows)


def frontier(summary: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    x = summary.loc[summary["large_move_target_ge_50bps"]].copy()
    if x.empty:
        return x
    x["resolved_hitrate_clears_nominal_be"] = x["event_resolved_hit_rate_minus_nominal_break_even"] > 0
    x["positive_timeout_exit_ev"] = x["event_mean_net_pnl_bps_timeout_exit"] > 0
    x["positive_vs_controls"] = x["event_minus_control_mean_net_pnl_bps"] > 0
    x["screen_pass_descriptive"] = (
        (x["event_usable_rate"] >= cfg.min_path_coverage_rate)
        & x["resolved_hitrate_clears_nominal_be"]
        & x["positive_timeout_exit_ev"]
        & x["positive_vs_controls"]
    )
    return x.sort_values(
        ["screen_pass_descriptive", "event_mean_net_pnl_bps_timeout_exit", "event_resolved_hit_rate_minus_nominal_break_even"],
        ascending=[False, False, False], kind="mergesort"
    )


def cohort_counts(membership: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if membership.empty:
        return pd.DataFrame()
    x = membership.groupby(["cohort", "orientation", "primitive"], dropna=False).agg(
        memberships=("membership_id", "count"), unique_events=("event_id", "nunique")
    ).reset_index()
    x["economically_screened"] = x["unique_events"] >= cfg.min_cohort_events
    return x


def timing_summary(membership: pd.DataFrame) -> pd.DataFrame:
    if membership.empty:
        return pd.DataFrame()
    rows=[]
    for cohort,g in membership.groupby("cohort",sort=True):
        lag=pd.to_numeric(g["detection_lag_seconds"],errors="coerce")
        pre=pd.to_numeric(g["move_realized_before_detection_bps"],errors="coerce")
        rows.append({"cohort":cohort,"n":len(g),"median_detection_lag_s":float(lag.median()),"p90_detection_lag_s":float(lag.quantile(.9)),"median_pre_detection_move_bps":float(pre.median()),"p90_pre_detection_move_bps":float(pre.quantile(.9))})
    return pd.DataFrame(rows)


def invariant_report(membership: pd.DataFrame, prof: pd.DataFrame, summary: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    checks=[]
    def add(name, ok, detail): checks.append({"invariant":name,"status":"PASS" if ok else "FAIL","detail":detail})
    add("trade_direction_directional", membership.empty or membership["trade_direction"].isin([-1,1]).all(), "All cohort trades must be directional")
    add("tradeable_after_detection", membership.empty or (pd.to_numeric(membership["tradeable_second"]) >= pd.to_numeric(membership["detection_second"])+1).all(), "Earliest tradeable second must be detection+1 or later")
    add("path_no_negative_observed_seconds", prof.empty or (pd.to_numeric(prof["path_seconds_observed"],errors="coerce").fillna(0)>=0).all(), "Observed path seconds cannot be negative")
    add("mfe_ge_mae", prof.empty or ((pd.to_numeric(prof["mfe_exec_pre_fee_bps"],errors="coerce") >= pd.to_numeric(prof["mae_exec_pre_fee_bps"],errors="coerce")) | prof["mfe_exec_pre_fee_bps"].isna()).all(), "MFE must not be below MAE")
    add("summary_nonempty", not summary.empty, "At least one economically screened cohort must produce first-passage summaries")
    if not summary.empty:
        rates=[c for c in summary.columns if c.endswith("_rate_all") or c.endswith("_usable_rate")]
        ok=True
        for c in rates:
            v=pd.to_numeric(summary[c],errors="coerce").dropna()
            ok &= bool(((v>=-1e-12)&(v<=1+1e-12)).all())
        add("probabilities_in_unit_interval",ok,"All reported rates must lie in [0,1]")
    return pd.DataFrame(checks)


def findings_text(meta: dict[str, Any], counts: pd.DataFrame, front: pd.DataFrame, cfg: Config) -> str:
    lines=[
        "# Phase 6C first-passage economic screen", "",
        f"- Builder: {BUILDER_VERSION}",
        f"- Cohort memberships: {meta['cohort_memberships']:,}",
        f"- Unique executable path profiles: {meta['path_profiles']:,}",
        f"- Max path: {cfg.max_path_seconds}s",
        f"- Targets: {', '.join(bps_tag(x) for x in cfg.targets_bps)} bps",
        f"- Stops: {', '.join(bps_tag(x) for x in cfg.stops_bps)} bps", "",
        "## Interpretation contract", "",
        "Targets/stops are first-passage barriers on Phase-6A executable PRE-FEE VWAP PnL. Fees are then subtracted per scenario, so post-cost reward and risk are asymmetric exactly as they would be economically.",
        "Paths are censored at the first invalid/unexecutable second. A timeout exits at the final valid second of the configured horizon only for the explicitly labelled timeout-exit EV statistic.",
        "Phase-6A uses fixed-quote notionals; this is not yet exact same-base-quantity round-trip accounting.", "",
        "## Large-move frontier", "",
    ]
    if front.empty:
        lines.append("No >=50 bps target rows were available.")
    else:
        shown=front.head(20)
        for r in shown.itertuples(index=False):
            lines.append(
                f"- {r.cohort} | ${int(r.quote_notional):,} | +{bps_tag(r.target_bps_pre_fee_executable)}/-{bps_tag(r.stop_bps_pre_fee_executable)} bps | fee {r.fee_bps_per_leg:g}/leg | "
                f"target-first resolved={r.event_target_first_rate_resolved:.3f} vs nominal BE={r.nominal_break_even_hit_rate:.3f} | "
                f"timeout-exit mean net={r.event_mean_net_pnl_bps_timeout_exit:.3f} bps | descriptive pass={bool(r.screen_pass_descriptive)}"
            )
    lines += ["", "No alpha or strategy claim is made by this screen."]
    return "\n".join(lines)+"\n"



# ---------------------------------------------------------------------------
# Partitioned / resumable full-window execution
# ---------------------------------------------------------------------------

def iter_dates(start: date, end: date) -> list[date]:
    out: list[date] = []
    d = start
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return out


def utc_day_bounds(day: date) -> tuple[int, int]:
    lo = int(datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc).timestamp())
    return lo, lo + 86400


def phase6b_shard_path(root: Path, dataset: str, symbol: str, day: date) -> Path:
    base = root / dataset / f"symbol={symbol}" / f"date={day.isoformat()}"
    preferred = base / f"part-{day.isoformat()}.parquet"
    if preferred.exists():
        return preferred
    matches = sorted(base.glob("*.parquet")) if base.exists() else []
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Missing Phase-6B {dataset} shard for {symbol} {day}: {base}")
    raise RuntimeError(f"Ambiguous Phase-6B {dataset} shards for {symbol} {day}: {matches}")


def read_phase6b_day(root: Path, dataset: str, symbol: str, day: date) -> pd.DataFrame:
    try:
        return pd.read_parquet(phase6b_shard_path(root, dataset, symbol, day))
    except FileNotFoundError:
        # Legacy fallback for old one-file smoke outputs.
        legacy = root / dataset / f"{dataset}.parquet"
        if not legacy.exists():
            raise
        df = pd.read_parquet(legacy)
        sec_col = "detection_second" if dataset == "events" else "anchor_second"
        lo, hi = utc_day_bounds(day)
        sec = pd.to_numeric(df[sec_col], errors="coerce")
        return df.loc[(sec >= lo) & (sec < hi)].copy()


def membership_shard_path(root: Path, symbol: str, day: date) -> Path:
    return root / "cohort_membership" / f"symbol={symbol}" / f"date={day.isoformat()}" / f"part-{day.isoformat()}.parquet"


def profiled_shard_path(root: Path, symbol: str, day: date) -> Path:
    return root / "profiled_anchors" / f"symbol={symbol}" / f"date={day.isoformat()}" / f"part-{day.isoformat()}.parquet"


def daily_state_path(root: Path, symbol: str, day: date) -> Path:
    return root / "metadata" / "daily" / f"symbol={symbol}" / f"date={day.isoformat()}.json"


def config_payload(cfg: Config, start_date: date, end_date: date) -> dict[str, Any]:
    return {
        "builder_version": BUILDER_VERSION,
        "schema_name": OUTPUT_SCHEMA_NAME,
        "symbol": cfg.symbol,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "config": asdict(cfg),
    }


def config_signature(cfg: Config, start_date: date, end_date: date) -> str:
    raw = json.dumps(config_payload(cfg, start_date, end_date), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_daily_state(
    root: Path,
    symbol: str,
    day: date,
    signature: str,
    *,
    membership_complete: bool,
    profile_complete: bool,
    membership_rows: int,
    profiled_rows: int,
    runtime_membership_seconds: float | None = None,
    runtime_profile_seconds: float | None = None,
) -> None:
    prev = read_state(daily_state_path(root, symbol, day)) or {}
    payload = {
        "builder_version": BUILDER_VERSION,
        "schema_name": OUTPUT_SCHEMA_NAME,
        "symbol": symbol,
        "date": day.isoformat(),
        "config_signature": signature,
        "membership_complete": bool(membership_complete),
        "profile_complete": bool(profile_complete),
        "membership_rows": int(membership_rows),
        "profiled_rows": int(profiled_rows),
        "runtime_membership_seconds": (
            float(runtime_membership_seconds)
            if runtime_membership_seconds is not None
            else prev.get("runtime_membership_seconds")
        ),
        "runtime_profile_seconds": (
            float(runtime_profile_seconds)
            if runtime_profile_seconds is not None
            else prev.get("runtime_profile_seconds")
        ),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(daily_state_path(root, symbol, day), payload)


def state_reusable(
    root: Path,
    symbol: str,
    day: date,
    signature: str,
    stage: str,
) -> bool:
    st = read_state(daily_state_path(root, symbol, day))
    if not st or st.get("config_signature") != signature:
        return False
    if stage == "membership":
        return bool(st.get("membership_complete")) and membership_shard_path(root, symbol, day).exists()
    if stage == "profile":
        return bool(st.get("profile_complete")) and profiled_shard_path(root, symbol, day).exists()
    raise ValueError(stage)


def build_membership_for_day(
    primitive_root: Path,
    symbol: str,
    day: date,
    cfg: Config,
) -> pd.DataFrame:
    """Build target-day memberships with only trailing cross-midnight context.

    Companion detections use at most cfg.interaction_lookback_seconds of history,
    so previous-day events are sufficient.  Future-day events are deliberately
    excluded because interaction membership is trailing-only.
    """
    frames: list[pd.DataFrame] = []
    prev_day = day - timedelta(days=1)
    try:
        frames.append(read_phase6b_day(primitive_root, "events", symbol, prev_day))
    except FileNotFoundError:
        pass
    frames.append(read_phase6b_day(primitive_root, "events", symbol, day))
    events = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    validate_phase6b(events, pd.DataFrame(columns=[
        "control_id", "parent_event_id", "primitive", "direction", "anchor_second", "tradeable_second"
    ]))
    features = build_interaction_features(events, cfg)
    if features.empty:
        return pd.DataFrame()
    lo, hi = utc_day_bounds(day)
    sec = pd.to_numeric(features["detection_second"], errors="coerce")
    features = features.loc[(sec >= lo) & (sec < hi)].copy()
    return build_cohort_membership(features, cfg)


def _daily_count_rows(membership: pd.DataFrame) -> pd.DataFrame:
    if membership.empty:
        return pd.DataFrame(columns=["cohort", "orientation", "primitive", "memberships", "unique_events"])
    return membership.groupby(["cohort", "orientation", "primitive"], dropna=False).agg(
        memberships=("membership_id", "count"), unique_events=("event_id", "nunique")
    ).reset_index()


def global_cohort_counts_from_shards(
    output_root: Path,
    symbol: str,
    days: Sequence[date],
    cfg: Config,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for day in days:
        p = membership_shard_path(output_root, symbol, day)
        if not p.exists():
            continue
        parts.append(_daily_count_rows(pd.read_parquet(p)))
    if not parts:
        return pd.DataFrame()
    x = pd.concat(parts, ignore_index=True)
    # Membership IDs/event IDs are unique to an anchor event/cohort, so summing
    # daily distinct counts is exact across disjoint UTC event dates.
    g = x.groupby(["cohort", "orientation", "primitive"], dropna=False).agg(
        memberships=("memberships", "sum"), unique_events=("unique_events", "sum")
    ).reset_index()
    g["economically_screened"] = g["unique_events"] >= cfg.min_cohort_events
    return g


def build_profiled_for_day(
    primitive_root: Path,
    execution_root: Path,
    symbol: str,
    day: date,
    membership: pd.DataFrame,
    screened: set[str],
    cfg: Config,
) -> pd.DataFrame:
    screen_mem = membership.loc[membership["cohort"].isin(screened)].copy() if not membership.empty else membership
    if screen_mem.empty:
        return pd.DataFrame()
    controls = read_phase6b_day(primitive_root, "controls", symbol, day)
    validate_phase6b(
        pd.DataFrame(columns=[
            "event_id", "symbol", "primitive", "direction", "detected", "detection_second",
            "tradeable_second", "detection_lag_seconds", "move_realized_before_detection_bps"
        ]),
        controls,
    )
    parent_ids = set(screen_mem["event_id"].astype(str))
    controls = controls.loc[controls["parent_event_id"].astype(str).isin(parent_ids)].copy()
    anchors = add_path_keys(build_anchor_table(screen_mem, controls))
    if anchors.empty:
        return pd.DataFrame()
    profiles = build_path_profiles(anchors, execution_root, symbol, cfg)
    if profiles.empty:
        return pd.DataFrame()
    return anchors.merge(
        profiles,
        on=["path_key", "anchor_id", "sample_type", "trade_direction", "tradeable_second", "anchor_date"],
        how="left",
        validate="many_to_many",
    )


def load_membership_cohort(
    output_root: Path, symbol: str, days: Sequence[date], cohort: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for day in days:
        p = membership_shard_path(output_root, symbol, day)
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        x = df.loc[df["cohort"].astype(str) == cohort]
        if not x.empty:
            frames.append(x)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_profiled_cohort(
    output_root: Path, symbol: str, days: Sequence[date], cohort: str,
    quote_notional: float | None = None,
) -> pd.DataFrame:
    """Load one pooled cohort slice, optionally one notional, across daily shards.

    The notional slice is the important memory bound for large base cohorts: pooled
    reports never need all four notionals resident simultaneously.
    """
    frames: list[pd.DataFrame] = []
    filters: list[tuple[str,str,Any]] = [("cohort", "==", cohort)]
    if quote_notional is not None:
        filters.append(("quote_notional", "==", float(quote_notional)))
    for day in days:
        p = profiled_shard_path(output_root, symbol, day)
        if not p.exists():
            continue
        try:
            x = pd.read_parquet(p, filters=filters)
        except Exception:
            df = pd.read_parquet(p)
            mask = df["cohort"].astype(str) == cohort
            if quote_notional is not None:
                mask &= np.isclose(pd.to_numeric(df["quote_notional"], errors="coerce"), float(quote_notional), rtol=0, atol=1e-9)
            x = df.loc[mask]
        if not x.empty:
            frames.append(x)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate_timing_from_shards(
    output_root: Path, symbol: str, days: Sequence[date], cohorts: Sequence[str],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for cohort in cohorts:
        m = load_membership_cohort(output_root, symbol, days, cohort)
        if not m.empty:
            rows.append(timing_summary(m))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def aggregate_pooled_reports(
    output_root: Path,
    report_root: Path,
    symbol: str,
    days: Sequence[date],
    counts: pd.DataFrame,
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    screened = sorted(counts.loc[counts["economically_screened"], "cohort"].astype(str).unique()) if not counts.empty else []
    fp_parts: list[pd.DataFrame] = []
    mm_parts: list[pd.DataFrame] = []
    daily_parts: list[pd.DataFrame] = []
    invariant_parts: list[pd.DataFrame] = []

    print("Aggregating pooled reports one cohort/notional slice at a time...")
    total_slices=max(1,len(screened)*len(cfg.notionals)); slice_i=0
    for cohort in screened:
        mem = load_membership_cohort(output_root, symbol, days, cohort)
        cohort_had_profile=False
        for notional in cfg.notionals:
            slice_i += 1
            t0 = time.perf_counter()
            prof = load_profiled_cohort(output_root, symbol, days, cohort, float(notional))
            if prof.empty:
                print(f"  [{slice_i:02d}/{total_slices:02d}] {cohort} ${int(notional):,}: EMPTY")
                continue
            cohort_had_profile=True
            fp = first_passage_summary(prof, cfg)
            mm = mfe_mae_summary(prof, cfg)
            if not fp.empty:
                fp_parts.append(fp)
            if not mm.empty:
                mm_parts.append(mm)
            # Structural invariants are identical across notionals except path
            # values, so check the first available notional once per cohort.
            if not any((not x.empty and "cohort_scope" in x.columns and (x["cohort_scope"] == cohort).any()) for x in invariant_parts):
                inv = invariant_report(mem, prof, fp, cfg)
                if not inv.empty:
                    inv = inv.copy(); inv["cohort_scope"] = cohort
                    invariant_parts.append(inv)
            print(f"  [{slice_i:02d}/{total_slices:02d}] {cohort} ${int(notional):,}: rows={len(prof):,}, runtime={time.perf_counter()-t0:.1f}s")
        if mem.empty or not cohort_had_profile:
            invariant_parts.append(pd.DataFrame([{
                "invariant": f"cohort_profile_present::{cohort}",
                "status": "FAIL",
                "detail": "Screened cohort missing membership or profiled rows",
                "cohort_scope": cohort,
            }]))

    # Daily robustness is naturally day-local and can be aggregated without
    # retaining multiple days of anchors in memory.
    print("Aggregating daily robustness...")
    for i, day in enumerate(days, 1):
        p = profiled_shard_path(output_root, symbol, day)
        if not p.exists():
            continue
        prof = pd.read_parquet(p)
        if prof.empty:
            continue
        d = daily_summary(prof, cfg)
        if not d.empty:
            daily_parts.append(d)
        if i == 1 or i % 10 == 0 or i == len(days):
            print(f"  daily [{i:02d}/{len(days):02d}] {day}")

    fp_all = pd.concat(fp_parts, ignore_index=True) if fp_parts else pd.DataFrame()
    mm_all = pd.concat(mm_parts, ignore_index=True) if mm_parts else pd.DataFrame()
    daily_all = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()
    timing_all = aggregate_timing_from_shards(output_root, symbol, days, screened)
    inv_all = pd.concat(invariant_parts, ignore_index=True) if invariant_parts else pd.DataFrame()
    return fp_all, mm_all, daily_all, timing_all, inv_all


def partitioned_self_test() -> None:
    """Small deterministic equivalence check for day partitioning/context."""
    cfg = Config(interaction_lookback_seconds=5, min_cohort_events=1)
    base = int(datetime(2026, 6, 12, 23, 59, 58, tzinfo=timezone.utc).timestamp())
    # Aggressive-flow confirmation lands just before midnight; a depth event two
    # seconds after midnight must see it in trailing context when partitioned.
    events = pd.DataFrame([
        {"event_id":"A","symbol":"BTCUSDT","primitive":"AGGRESSIVE_FLOW","direction":1,"detected":True,
         "detection_second":base,"tradeable_second":base+1,"detection_lag_seconds":0,"move_realized_before_detection_bps":0.0},
        {"event_id":"D","symbol":"BTCUSDT","primitive":"DEPTH_WITHDRAWAL","direction":1,"detected":True,
         "detection_second":base+4,"tradeable_second":base+5,"detection_lag_seconds":0,"move_realized_before_detection_bps":0.0},
    ])
    one = build_cohort_membership(build_interaction_features(events, cfg), cfg)
    target_day = datetime.fromtimestamp(base+4, tz=timezone.utc).date().isoformat()
    one = one.loc[one["event_date"] == target_day].sort_values("cohort").reset_index(drop=True)
    # Equivalent partitioned context is simply previous + current day events.
    prev = events.loc[pd.to_datetime(events["detection_second"], unit="s", utc=True).dt.date.astype(str) != target_day]
    cur = events.loc[pd.to_datetime(events["detection_second"], unit="s", utc=True).dt.date.astype(str) == target_day]
    two = build_cohort_membership(build_interaction_features(pd.concat([prev, cur], ignore_index=True), cfg), cfg)
    two = two.loc[two["event_date"] == target_day].sort_values("cohort").reset_index(drop=True)
    assert one["cohort"].tolist() == two["cohort"].tolist()
    assert "DW_FLOW_CONT" in set(two["cohort"])


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test() -> None:
    cfg=Config(targets_bps=(10.0,50.0),stops_bps=(10.0,20.0),notionals=(1000.0,),fee_scenarios_bps=(0.0,2.5),max_path_seconds=6,path_chunk_size=8,min_cohort_events=1)
    base=int(datetime(2026,6,12,tzinfo=timezone.utc).timestamp())
    anchors=pd.DataFrame([
        {"cohort":"T","membership_id":"M1","anchor_id":"E1","parent_event_id":"E1","sample_type":"EVENT","trade_direction":1,"anchor_second":base,"tradeable_second":base+1,"event_date":"2026-06-12"},
        {"cohort":"R","membership_id":"M2","anchor_id":"E2","parent_event_id":"E2","sample_type":"EVENT","trade_direction":-1,"anchor_second":base,"tradeable_second":base+1,"event_date":"2026-06-12"},
    ])
    anchors=add_path_keys(anchors)
    secs=np.arange(base,base+10,dtype=np.int64)
    # Long entry BUY=100.00; future SELL path reaches +55 bps at t+3.
    sell=np.array([99.99,99.99,100.10,100.30,100.55,100.40,100.20,100.10,100.00,99.90])
    buy=np.array([100.00,100.00,100.11,100.31,100.56,100.41,100.21,100.11,100.01,99.91])
    fake=pd.DataFrame({"receive_second":secs,"book_valid":True,"mid_price":100.0,
                       "buy_vwap_1000":buy,"buy_complete_1000":True,"buy_crossing_cost_bps_1000":0.0,
                       "sell_vwap_1000":sell,"sell_complete_1000":True,"sell_crossing_cost_bps_1000":0.0})
    orig=globals()["read_execution_days"]
    try:
        globals()["read_execution_days"]=lambda *a,**k: fake.copy()
        prof=build_path_profiles(anchors,Path('.'),"BTCUSDT",cfg)
    finally:
        globals()["read_execution_days"]=orig
    longp=prof.loc[(prof["anchor_id"]=="E1") & (prof["quote_notional"]==1000)].iloc[0]
    assert int(longp["target_50_first_hit_s"])==3, longp.to_dict()
    assert float(longp["mfe_exec_pre_fee_bps"])>=50
    # Cost asymmetry exactly matches user's motivating example.
    dummy=pd.DataFrame([{**longp.to_dict(),"path_key":"P","anchor_id":"E","sample_type":"EVENT","trade_direction":1,"tradeable_second":base+1,"anchor_date":"2026-06-12","quote_notional":1000.0,
                         "target_10_first_hit_s":1,"target_10_hit_exec_pre_fee_pnl_bps":10.0,
                         "stop_10_first_hit_s":-1,"stop_10_hit_exec_pre_fee_pnl_bps":math.nan,
                         "terminal_exec_pre_fee_pnl_bps":10.0,"path_invalidated":False,"entry_found":True,"entry_book_valid":True,"entry_complete":True,"path_seconds_observed":6}])
    sm=summarize_one_sample(derive_outcomes(dummy,10.0,10.0),2.5)
    nominal_reward=10-5
    nominal_risk=10+5
    be=nominal_risk/(nominal_reward+nominal_risk)
    assert abs(nominal_reward-5)<1e-12 and abs(nominal_risk-15)<1e-12 and abs(be-.75)<1e-12
    # Reversal direction must use opposite executable side: short entry SELL, future BUY exit.
    shortp=prof.loc[prof["anchor_id"]=="E2"].iloc[0]
    assert int(shortp["trade_direction"])==-1 and math.isfinite(float(shortp["entry_vwap"]))

    # A data-quality hole must censor the path; a later target cannot be credited.
    invalid_fake = fake.copy()
    invalid_fake.loc[invalid_fake["receive_second"] == base + 3, "book_valid"] = False
    one_anchor = anchors.loc[anchors["anchor_id"] == "E1"].copy()
    try:
        globals()["read_execution_days"] = lambda *a, **k: invalid_fake.copy()
        censored = build_path_profiles(one_anchor, Path('.'), "BTCUSDT", cfg)
    finally:
        globals()["read_execution_days"] = orig
    cp = censored.iloc[0]
    assert int(cp["first_invalid_offset_s"]) == 2
    assert int(cp["target_50_first_hit_s"]) == -1, "must not look through an invalid second to a later target"

    partitioned_self_test()
    print("SELF-TEST PASS")
    print("Validated executable first-passage paths, target-before-stop ordering, conservative path censoring, opposite-side reversal pricing, MFE/MAE, and +10/-10 with 2.5bps/leg => +5/-15 net and 75% break-even hit rate.")


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description="Evaluate causal participant interactions with resumable executable target-before-stop first-passage economics.")
    p.add_argument("--primitive-root",type=Path,default=DEFAULT_PRIMITIVE_ROOT)
    p.add_argument("--execution-root",type=Path,default=DEFAULT_EXECUTION_ROOT)
    p.add_argument("--output-root",type=Path,default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root",type=Path,default=DEFAULT_REPORT_ROOT)
    p.add_argument("--symbol",default="BTCUSDT")
    p.add_argument("--start-date",type=date.fromisoformat,default=None)
    p.add_argument("--end-date",type=date.fromisoformat,default=None)
    p.add_argument("--interaction-lookback-seconds",type=int,default=5)
    p.add_argument("--min-cohort-events",type=int,default=30)
    p.add_argument("--targets-bps",default=",".join(bps_tag(x) for x in DEFAULT_TARGETS_BPS))
    p.add_argument("--stops-bps",default=",".join(bps_tag(x) for x in DEFAULT_STOPS_BPS))
    p.add_argument("--notionals",default=",".join(str(int(x)) for x in DEFAULT_NOTIONALS))
    p.add_argument("--fee-scenarios-bps",default=",".join(map(str,DEFAULT_FEE_SCENARIOS_BPS)),help="Per-leg fee sensitivity only; not an account-tier claim")
    p.add_argument("--max-path-seconds",type=int,default=DEFAULT_MAX_PATH_SECONDS)
    p.add_argument("--path-chunk-size",type=int,default=DEFAULT_CHUNK_SIZE)
    p.add_argument("--min-path-coverage-rate",type=float,default=.90)
    p.add_argument("--force",action="store_true",help="Rebuild daily Phase-6C shards even when matching state exists")
    p.add_argument("--self-test",action="store_true")
    return p.parse_args()


def _resolve_requested_dates(args: argparse.Namespace) -> tuple[date, date, list[date]]:
    if args.start_date is None or args.end_date is None:
        raise SystemExit("Phase 6C v1.1.2 partitioned mode requires both --start-date and --end-date")
    if args.end_date < args.start_date:
        raise SystemExit("--end-date must be >= --start-date")
    return args.start_date, args.end_date, iter_dates(args.start_date,args.end_date)


def main() -> int:
    args=parse_args()
    if args.self_test:
        self_test(); return 0

    start_date,end_date,days=_resolve_requested_dates(args)
    cfg=Config(symbol=args.symbol,interaction_lookback_seconds=args.interaction_lookback_seconds,min_cohort_events=args.min_cohort_events,
               targets_bps=tuple(float(x) for x in parse_csv_numbers(args.targets_bps,float)),
               stops_bps=tuple(float(x) for x in parse_csv_numbers(args.stops_bps,float)),
               notionals=tuple(float(x) for x in parse_csv_numbers(args.notionals,float)),
               fee_scenarios_bps=tuple(float(x) for x in parse_csv_numbers(args.fee_scenarios_bps,float)),
               max_path_seconds=args.max_path_seconds,path_chunk_size=args.path_chunk_size,min_path_coverage_rate=args.min_path_coverage_rate)
    if any(x<=0 for x in cfg.targets_bps+cfg.stops_bps): raise SystemExit("Targets/stops must be positive")
    if cfg.max_path_seconds<1: raise SystemExit("--max-path-seconds must be >=1")

    signature=config_signature(cfg,start_date,end_date)
    args.output_root.mkdir(parents=True,exist_ok=True)
    args.report_root.mkdir(parents=True,exist_ok=True)

    print("Building Phase-6C partitioned executable first-passage economic screen...")
    print(f"Builder version       : {BUILDER_VERSION}")
    print(f"Symbol                : {cfg.symbol}")
    print(f"Dates                 : {len(days)} ({start_date} to {end_date})")
    print(f"Targets (pre-fee exec): {', '.join(bps_tag(x) for x in cfg.targets_bps)} bps")
    print(f"Stops   (pre-fee exec): {', '.join(bps_tag(x) for x in cfg.stops_bps)} bps")
    print(f"Max path              : {cfg.max_path_seconds}s")
    print(f"Fee scenarios / leg   : {', '.join(str(x) for x in cfg.fee_scenarios_bps)} bps (sensitivity only)")
    print("Execution mode         : daily membership -> global screen -> daily path shards -> cohort-wise pooled reports")
    print(f"Config signature       : {signature[:16]}")

    performance_rows: list[dict[str,Any]]=[]
    total_start=time.perf_counter()

    # ------------------------------------------------------------------
    # Pass 1: causal memberships by UTC event day.
    # ------------------------------------------------------------------
    print("\n[1/4] Building/reusing daily causal cohort memberships...")
    for i,day in enumerate(days,1):
        p=membership_shard_path(args.output_root,cfg.symbol,day)
        if (not args.force) and state_reusable(args.output_root,cfg.symbol,day,signature,"membership"):
            rows=int((read_state(daily_state_path(args.output_root,cfg.symbol,day)) or {}).get("membership_rows",0))
            print(f"  [{i:02d}/{len(days):02d}] {day}: MEMBERSHIP_REUSED ({rows:,} rows)")
            continue
        t0=time.perf_counter()
        membership=build_membership_for_day(args.primitive_root,cfg.symbol,day,cfg)
        atomic_write_parquet(p,membership)
        elapsed=time.perf_counter()-t0
        prev=read_state(daily_state_path(args.output_root,cfg.symbol,day)) or {}
        write_daily_state(
            args.output_root,cfg.symbol,day,signature,
            membership_complete=True,profile_complete=False,
            membership_rows=len(membership),profiled_rows=0,
            runtime_membership_seconds=elapsed,
        )
        performance_rows.append({"date":day.isoformat(),"stage":"membership","runtime_seconds":elapsed,"rows":len(membership)})
        print(f"  [{i:02d}/{len(days):02d}] {day}: MEMBERSHIP_BUILT rows={len(membership):,}, runtime={elapsed:.1f}s")

    counts=global_cohort_counts_from_shards(args.output_root,cfg.symbol,days,cfg)
    screened=set(counts.loc[counts["economically_screened"],"cohort"].astype(str)) if not counts.empty else set()
    if not screened:
        raise RuntimeError("No cohorts met the global --min-cohort-events screen")
    print(f"Global cohort families : {len(counts):,}")
    print(f"Screened cohorts       : {len(screened):,} (global n >= {cfg.min_cohort_events})")

    # The global screen is part of the profile signature.  If it changes while
    # all other config is identical, profile shards must rebuild.
    profile_signature=hashlib.sha256((signature+"|"+"|".join(sorted(screened))).encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Pass 2: daily anchors + unique executable paths, immediately persisted.
    # ------------------------------------------------------------------
    print("\n[2/4] Building/reusing daily executable first-passage shards...")
    total_profiled_rows=0
    total_unique_paths=0
    for i,day in enumerate(days,1):
        st=read_state(daily_state_path(args.output_root,cfg.symbol,day)) or {}
        profile_reusable=(
            (not args.force)
            and bool(st.get("profile_complete"))
            and st.get("profile_signature") == profile_signature
            and profiled_shard_path(args.output_root,cfg.symbol,day).exists()
        )
        if profile_reusable:
            n=int(st.get("profiled_rows",0)); total_profiled_rows+=n
            p=profiled_shard_path(args.output_root,cfg.symbol,day)
            try:
                tmp=pd.read_parquet(p,columns=["path_key","quote_notional"])
                total_unique_paths += int(len(tmp.drop_duplicates(["path_key","quote_notional"])))
            except Exception:
                pass
            print(f"  [{i:02d}/{len(days):02d}] {day}: PROFILE_REUSED ({n:,} rows)")
            continue

        t0=time.perf_counter()
        membership=pd.read_parquet(membership_shard_path(args.output_root,cfg.symbol,day))
        profiled=build_profiled_for_day(
            args.primitive_root,args.execution_root,cfg.symbol,day,membership,screened,cfg
        )
        atomic_write_parquet(profiled_shard_path(args.output_root,cfg.symbol,day),profiled)
        elapsed=time.perf_counter()-t0
        unique_paths=(
            int(len(profiled[["path_key","quote_notional"]].drop_duplicates()))
            if not profiled.empty and {"path_key","quote_notional"}.issubset(profiled.columns) else 0
        )
        total_profiled_rows+=len(profiled); total_unique_paths+=unique_paths
        state=read_state(daily_state_path(args.output_root,cfg.symbol,day)) or {}
        payload={
            **state,
            "builder_version":BUILDER_VERSION,"schema_name":OUTPUT_SCHEMA_NAME,
            "symbol":cfg.symbol,"date":day.isoformat(),"config_signature":signature,
            "profile_signature":profile_signature,"membership_complete":True,"profile_complete":True,
            "membership_rows":int(state.get("membership_rows",len(membership))),
            "profiled_rows":int(len(profiled)),"unique_path_notional_rows":unique_paths,
            "runtime_profile_seconds":elapsed,"screened_cohort_count":len(screened),
            "updated_at_utc":datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(daily_state_path(args.output_root,cfg.symbol,day),payload)
        performance_rows.append({"date":day.isoformat(),"stage":"profile","runtime_seconds":elapsed,"rows":len(profiled)})
        ev_n=int((profiled["sample_type"]=="EVENT").sum()) if not profiled.empty else 0
        ctrl_n=int((profiled["sample_type"]=="CONTROL").sum()) if not profiled.empty else 0
        print(f"  [{i:02d}/{len(days):02d}] {day}: PROFILE_BUILT events={ev_n:,}, controls={ctrl_n:,}, profiled={len(profiled):,}, unique_paths={unique_paths:,}, runtime={elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Pass 3: exact pooled reports, one cohort at a time (bounded memory).
    # ------------------------------------------------------------------
    print("\n[3/4] Aggregating exact pooled first-passage reports...")
    report_start=time.perf_counter()
    fp,mm,daily,timing,checks=aggregate_pooled_reports(
        args.output_root,args.report_root,cfg.symbol,days,counts,cfg
    )
    front=frontier(fp,cfg)

    # Add partition-level invariants.
    part_checks=[]
    def add_part(name: str, ok: bool, detail: str) -> None:
        part_checks.append({"invariant":name,"status":"PASS" if ok else "FAIL","detail":detail,"cohort_scope":"GLOBAL"})
    mem_missing=[d.isoformat() for d in days if not membership_shard_path(args.output_root,cfg.symbol,d).exists()]
    prof_missing=[d.isoformat() for d in days if not profiled_shard_path(args.output_root,cfg.symbol,d).exists()]
    add_part("all_membership_dates_present",not mem_missing,f"missing={mem_missing}")
    add_part("all_profile_dates_present",not prof_missing,f"missing={prof_missing}")
    daily_days=int(daily["event_date"].nunique()) if not daily.empty else 0
    add_part("daily_robustness_spans_requested_window",daily_days==len(days),
             f"days={daily_days}; requested={len(days)}")
    checks=pd.concat([checks,pd.DataFrame(part_checks)],ignore_index=True) if not checks.empty else pd.DataFrame(part_checks)
    failed=checks.loc[checks["status"]=="FAIL"] if not checks.empty else pd.DataFrame()
    report_elapsed=time.perf_counter()-report_start
    performance_rows.append({"date":"ALL","stage":"pooled_reports","runtime_seconds":report_elapsed,"rows":len(fp)})

    # ------------------------------------------------------------------
    # Pass 4: reports + metadata.  No giant monolithic path parquet is written.
    # ------------------------------------------------------------------
    print("\n[4/4] Writing reports and run metadata...")
    atomic_write_csv(args.report_root/"participant_interaction_cohort_counts.csv",counts)
    atomic_write_csv(args.report_root/"participant_interaction_timing.csv",timing)
    atomic_write_csv(args.report_root/"participant_interaction_first_passage_summary.csv",fp)
    atomic_write_csv(args.report_root/"participant_interaction_mfe_mae_summary.csv",mm)
    atomic_write_csv(args.report_root/"participant_interaction_daily_robustness.csv",daily)
    atomic_write_csv(args.report_root/"participant_interaction_large_move_frontier.csv",front)
    atomic_write_csv(args.report_root/"participant_interaction_invariants.csv",checks)
    if performance_rows:
        atomic_write_csv(args.report_root/"participant_interaction_partition_performance.csv",pd.DataFrame(performance_rows))

    membership_rows=sum(int((read_state(daily_state_path(args.output_root,cfg.symbol,d)) or {}).get("membership_rows",0)) for d in days)
    # Recompute totals from state so a fully reused rerun reports correctly.
    profiled_rows=sum(int((read_state(daily_state_path(args.output_root,cfg.symbol,d)) or {}).get("profiled_rows",0)) for d in days)
    unique_path_rows=sum(int((read_state(daily_state_path(args.output_root,cfg.symbol,d)) or {}).get("unique_path_notional_rows",0)) for d in days)
    total_elapsed=time.perf_counter()-total_start
    meta={
        "builder_version":BUILDER_VERSION,"schema_name":OUTPUT_SCHEMA_NAME,"symbol":cfg.symbol,
        "start_date":start_date.isoformat(),"end_date":end_date.isoformat(),"dates":len(days),
        "cohort_memberships":membership_rows,"screened_cohorts":len(screened),
        "profiled_anchor_rows":profiled_rows,"path_profiles":unique_path_rows,
        "summary_rows":int(len(fp)),"daily_robustness_rows":int(len(daily)),
        "failed_invariants":int(len(failed)),"config_signature":signature,"profile_signature":profile_signature,
        "total_runtime_seconds":total_elapsed,"partition_contract":"UTC_EVENT_DAY_RESUMABLE_V1",
        "config":asdict(cfg),
    }
    atomic_write_json(args.output_root/"metadata"/"latest_summary.json",meta)
    atomic_write_json(args.output_root/"metadata"/"run_config.json",config_payload(cfg,start_date,end_date))
    atomic_write_text(args.report_root/"participant_interaction_findings.md",findings_text(meta,counts,front,cfg))

    print(); print("Phase-6C partitioned first-passage summary"); print("--------------------------------------------")
    print(f"Dates                   : {len(days)}")
    print(f"Cohort memberships      : {membership_rows:,}")
    print(f"Screened cohorts        : {len(screened):,}")
    print(f"Profiled anchor rows    : {profiled_rows:,}")
    print(f"Unique path/notional    : {unique_path_rows:,}")
    print(f"First-passage rows      : {len(fp):,}")
    print(f"Daily robustness rows   : {len(daily):,}")
    print(f"Failed invariants       : {len(failed):,}")
    print(f"Total runtime           : {total_elapsed:.1f}s")
    print(f"Reports                 : {args.report_root}")
    print("No alpha or strategy claim is made by Phase 6C.")
    return 1 if len(failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
