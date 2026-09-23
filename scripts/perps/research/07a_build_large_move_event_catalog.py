#!/usr/bin/env python3
"""
Phase 7A — Large-move physical-event catalog
============================================

Purpose
-------
Build a deterministic, top-down catalog of economically meaningful BTCUSDT
perpetual price expansions before inspecting any order-book precursor feature.

The catalog uses a 35 bps directional-change ontology to de-duplicate thousands
of overlapping forward windows into physical directional legs. Higher targets
(50/75/100 bps) and realization horizons (5/10/15/30 minutes) are then nested
inside those physical events.

This is a DISCOVERY catalog, not a trading signal. Candidate onset definitions
are audited retrospectively and are not assumed tradeable. Phase 7B must decide
which onset definition, if any, can be frozen as a causal mechanical onset.

Primary price source
--------------------
The default catalog price is the last traded BTCUSDT perp price, forward-filled
for at most 5 seconds across the loaded range. This keeps large-move labels
independent of Phase-6A book-valid cold starts. Phase-6A book_valid is retained
only as a coverage diagnostic for future precursor studies.

Outputs
-------
data/derived/perps/research/large_move_event_catalog_v1/
  events/symbol=BTCUSDT/onset_date=YYYY-MM-DD/part-YYYY-MM-DD.parquet
  onset_candidates/symbol=BTCUSDT/onset_date=YYYY-MM-DD/part-YYYY-MM-DD.parquet
  physical_events.parquet
  onset_candidates.parquet
  target_membership.parquet
  metadata/latest_summary.json
  metadata/run_config.json

reports/perps_large_move_event_catalog/
  large_move_event_counts.csv
  large_move_horizon_matrix.csv
  large_move_target_nesting.csv
  large_move_duration_profile.csv
  large_move_onset_audit.csv
  large_move_book_coverage.csv
  large_move_invariants.csv
  large_move_findings.md

Scientific contract
-------------------
* Physical event ontology: 35-bps directional-change legs.
* A new UP leg starts at the prior local trough when price first reaches +35 bps.
  It ends at the subsequent local peak once a 35-bps drawdown confirms reversal.
  DOWN is symmetric.
* Higher thresholds are first-passage hits from the SAME physical-leg extremum.
* Horizon membership is based on onset -> first-target-hit latency.
* Invalid/missing catalog price resets segmentation; events are never bridged
  through unknown price paths.
* Candidate onset methods are descriptive/audit objects only.
* No alpha, strategy, or profitability claim is made by Phase 7A.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

BUILDER_VERSION = "1.0.0"
SCHEMA_VERSION = 1
SCHEMA_NAME = "large_move_event_catalog_v1"
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_EXEC_ROOT = Path("data/derived/perps/execution_foundation_v1")
DEFAULT_OUTPUT_ROOT = Path("data/derived/perps/research/large_move_event_catalog_v1")
DEFAULT_REPORT_ROOT = Path("reports/perps_large_move_event_catalog")
DEFAULT_BASE_THRESHOLD_BPS = 35.0
DEFAULT_TARGETS_BPS = (35.0, 50.0, 75.0, 100.0)
DEFAULT_HORIZONS_SEC = (300, 600, 900, 1800)
DEFAULT_MAX_TRADE_STALE_SEC = 5
DEFAULT_RANGE_LOOKBACK_SEC = 60
DEFAULT_RANGE_BREAK_BUFFER_BPS = 1.0
DEFAULT_FIXED_DEPARTURE_BPS = 10.0
DEFAULT_VOL_LOOKBACK_SEC = 60
DEFAULT_VOL_MULTIPLIER = 0.50
DEFAULT_VOL_DEPARTURE_MIN_BPS = 5.0
DEFAULT_VOL_DEPARTURE_MAX_BPS = 15.0
PRE_BOOK_WINDOWS_SEC = (5, 10, 30, 60, 120, 300)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_date(text: str) -> date:
    return date.fromisoformat(text)


def daterange(start: date, end: date) -> list[date]:
    out: list[date] = []
    d = start
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return out


def stable_id(*parts: Any) -> str:
    payload = "|".join(str(x) for x in parts).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:20]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7a-", suffix=path.suffix, dir=str(path.parent))
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
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "Phase 7A requires numpy and pandas. Install with:\n"
            "  python -m pip install numpy pandas"
        ) from exc
    return np, pd


def require_parquet_stack():
    np, pd = require_numeric_stack()
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "Phase 7A Parquet I/O requires pyarrow. Install with:\n"
            "  python -m pip install pyarrow"
        ) from exc
    return np, pd, pa, pq


def execution_day_path(root: Path, symbol: str, d: date) -> Path:
    return root / f"symbol={symbol}" / f"date={d.isoformat()}" / f"part-{d.isoformat()}.parquet"


def write_parquet_atomic(df, path: Path) -> None:
    _, _, pa, pq = require_parquet_stack()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7a-", suffix=".parquet", dir=str(path.parent))
    os.close(fd)
    try:
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@dataclass
class PhysicalEvent:
    event_id: str
    symbol: str
    direction: int
    direction_label: str
    onset_second: int
    onset_time: str
    onset_date: str
    onset_price: float
    base_threshold_bps: float
    base_hit_second: int
    base_hit_time: str
    base_time_to_hit_sec: int
    extreme_second: int
    extreme_time: str
    extreme_price: float
    reversal_confirm_second: int | None
    reversal_confirm_time: str | None
    closure_reason: str
    leg_duration_to_extreme_sec: int
    leg_max_excursion_bps: float
    hit_35_second: int | None
    hit_50_second: int | None
    hit_75_second: int | None
    hit_100_second: int | None
    time_to_35_sec: int | None
    time_to_50_sec: int | None
    time_to_75_sec: int | None
    time_to_100_sec: int | None
    forward_30m_mfe_bps: float | None
    forward_30m_mae_bps: float | None
    pre_book_valid_5s: float | None
    pre_book_valid_10s: float | None
    pre_book_valid_30s: float | None
    pre_book_valid_60s: float | None
    pre_book_valid_120s: float | None
    pre_book_valid_300s: float | None
    primary_catalog_eligible: bool


def directional_bps(price: float, anchor: float, direction: int) -> float:
    if not (math.isfinite(price) and math.isfinite(anchor) and anchor > 0):
        return math.nan
    if direction > 0:
        return (price / anchor - 1.0) * 10_000.0
    return (1.0 - price / anchor) * 10_000.0


def bps_return(p1: float, p0: float) -> float:
    if not (math.isfinite(p1) and math.isfinite(p0) and p0 > 0):
        return math.nan
    return (p1 / p0 - 1.0) * 10_000.0


def load_execution_range(exec_root: Path, symbol: str, days: Sequence[date]):
    np, pd, _, pq = require_parquet_stack()
    cols = ["receive_second", "last_trade_price", "trade_price_ffill", "mid_price", "book_valid"]
    parts = []
    missing: list[str] = []
    for i, d in enumerate(days, 1):
        p = execution_day_path(exec_root, symbol, d)
        if not p.exists():
            missing.append(str(p))
            continue
        schema_names = set(pq.read_schema(p).names)
        use = [c for c in cols if c in schema_names]
        if "receive_second" not in use:
            raise RuntimeError(f"Missing receive_second in {p}")
        table = pq.read_table(p, columns=use)
        f = table.to_pandas()
        for c in cols:
            if c not in f.columns:
                f[c] = np.nan if c != "book_valid" else False
        parts.append(f[cols])
        if i == 1 or i % 10 == 0 or i == len(days):
            print(f"  loading execution [{i:02d}/{len(days):02d}] {d} rows={len(f):,}")
    if missing:
        raise RuntimeError("Missing Phase-6A execution partitions:\n  " + "\n  ".join(missing[:20]))
    if not parts:
        raise RuntimeError("No Phase-6A execution rows loaded")
    frame = pd.concat(parts, ignore_index=True)
    frame = frame.sort_values("receive_second", kind="mergesort").drop_duplicates("receive_second", keep="last").reset_index(drop=True)
    return frame


def build_catalog_price(frame, max_stale_sec: int, price_source: str):
    np, pd = require_numeric_stack()
    sec = pd.to_numeric(frame["receive_second"], errors="coerce").to_numpy(dtype=np.int64)
    n = len(frame)
    if price_source == "mid":
        raw = pd.to_numeric(frame["mid_price"], errors="coerce").to_numpy(dtype=float)
        bv = frame["book_valid"].fillna(False).astype(bool).to_numpy()
        price = np.where(bv & np.isfinite(raw) & (raw > 0), raw, np.nan)
        source_label = "PHASE6A_VALID_MID"
    else:
        last = pd.to_numeric(frame["last_trade_price"], errors="coerce").to_numpy(dtype=float)
        fallback = pd.to_numeric(frame["trade_price_ffill"], errors="coerce").to_numpy(dtype=float)
        price = np.full(n, np.nan, dtype=float)
        last_seen_price = math.nan
        last_seen_sec = -10**18
        for i in range(n):
            if math.isfinite(last[i]) and last[i] > 0:
                last_seen_price = float(last[i])
                last_seen_sec = int(sec[i])
            elif not math.isfinite(last_seen_price) and math.isfinite(fallback[i]) and fallback[i] > 0:
                # Only bootstrap from the Phase-6A per-day ffill if no actual trade has
                # yet been observed in the loaded range. Subsequent staleness is bounded.
                last_seen_price = float(fallback[i])
                last_seen_sec = int(sec[i])
            if math.isfinite(last_seen_price) and int(sec[i]) - last_seen_sec <= max_stale_sec:
                price[i] = last_seen_price
        source_label = f"LAST_TRADE_FFILL_MAX_{max_stale_sec}S"
    return sec, price, source_label


def trailing_rv_bps(price, sec, lookback: int):
    np, _ = require_numeric_stack()
    n = len(price)
    ret2 = np.zeros(n, dtype=float)
    valid_step = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if sec[i] == sec[i - 1] + 1 and math.isfinite(price[i]) and math.isfinite(price[i - 1]) and price[i - 1] > 0:
            r = (price[i] / price[i - 1] - 1.0) * 10_000.0
            ret2[i] = r * r
            valid_step[i] = True
    csum = np.concatenate(([0.0], np.cumsum(ret2)))
    cvalid = np.concatenate(([0], np.cumsum(valid_step.astype(np.int64))))
    rv = np.full(n, np.nan, dtype=float)
    for i in range(n):
        lo = max(1, i - lookback + 1)
        count = int(cvalid[i + 1] - cvalid[lo])
        if count >= max(5, lookback // 3):
            rv[i] = math.sqrt(max(0.0, float(csum[i + 1] - csum[lo])))
    return rv


def _target_key(t: float) -> str:
    return str(int(round(t)))


def segment_directional_change(
    sec,
    price,
    symbol: str,
    base_threshold_bps: float,
    targets_bps: Sequence[float],
):
    """Return physical 35-bps directional-change legs.

    This is intentionally retrospective. The onset is the local extremum from
    which the base threshold was first reached. Higher thresholds are nested
    first-passage hits within the same directional leg.
    """
    targets = sorted(set(float(x) for x in targets_bps))
    if base_threshold_bps not in targets:
        targets = sorted(set(targets + [float(base_threshold_bps)]))

    records: list[dict[str, Any]] = []
    state = "SEEK"
    run_min = math.inf
    run_min_i = -1
    run_max = -math.inf
    run_max_i = -1
    open_event: dict[str, Any] | None = None

    def is_valid(i: int) -> bool:
        return math.isfinite(float(price[i])) and float(price[i]) > 0

    def initialize_seek(i: int) -> None:
        nonlocal run_min, run_min_i, run_max, run_max_i
        p = float(price[i])
        run_min = p
        run_min_i = i
        run_max = p
        run_max_i = i

    def new_event(direction: int, onset_i: int, trigger_i: int) -> dict[str, Any]:
        onset_price = float(price[onset_i])
        hits = {float(t): None for t in targets}
        for t in targets:
            if directional_bps(float(price[trigger_i]), onset_price, direction) + 1e-12 >= t:
                hits[float(t)] = trigger_i
        return {
            "direction": int(direction),
            "onset_i": int(onset_i),
            "onset_price": onset_price,
            "base_hit_i": int(trigger_i),
            "extreme_i": int(trigger_i),
            "extreme_price": float(price[trigger_i]),
            "reversal_confirm_i": None,
            "closure_reason": None,
            "hits": hits,
        }

    def update_event_hits(ev: dict[str, Any], i: int) -> None:
        move = directional_bps(float(price[i]), float(ev["onset_price"]), int(ev["direction"]))
        for t in targets:
            if ev["hits"][float(t)] is None and move + 1e-12 >= t:
                ev["hits"][float(t)] = int(i)

    def finalize(ev: dict[str, Any], reason: str, confirm_i: int | None) -> None:
        ev = dict(ev)
        ev["closure_reason"] = reason
        ev["reversal_confirm_i"] = confirm_i
        records.append(ev)

    prev_valid_i: int | None = None
    for i in range(len(price)):
        if not is_valid(i) or (prev_valid_i is not None and int(sec[i]) != int(sec[prev_valid_i]) + 1):
            if open_event is not None:
                finalize(open_event, "GAP_CENSORED", None)
                open_event = None
            state = "SEEK"
            run_min = math.inf
            run_min_i = -1
            run_max = -math.inf
            run_max_i = -1
            prev_valid_i = None if not is_valid(i) else i
            if is_valid(i):
                initialize_seek(i)
            continue

        p = float(price[i])
        if state == "SEEK":
            if run_min_i < 0:
                initialize_seek(i)
            if p < run_min:
                run_min = p
                run_min_i = i
            if p > run_max:
                run_max = p
                run_max_i = i
            up = directional_bps(p, run_min, +1)
            down = directional_bps(p, run_max, -1)
            if up + 1e-12 >= base_threshold_bps:
                open_event = new_event(+1, run_min_i, i)
                state = "UP"
            elif down + 1e-12 >= base_threshold_bps:
                open_event = new_event(-1, run_max_i, i)
                state = "DOWN"
        elif state == "UP":
            assert open_event is not None
            if p > float(open_event["extreme_price"]):
                open_event["extreme_price"] = p
                open_event["extreme_i"] = i
            update_event_hits(open_event, i)
            dd = (1.0 - p / float(open_event["extreme_price"])) * 10_000.0 if float(open_event["extreme_price"]) > 0 else math.nan
            if dd + 1e-12 >= base_threshold_bps:
                old = open_event
                peak_i = int(old["extreme_i"])
                peak_p = float(old["extreme_price"])
                finalize(old, "OPPOSITE_35BP_CONFIRMED", i)
                open_event = new_event(-1, peak_i, i)
                # Preserve exact peak price even if current trigger initialization
                # would otherwise recalculate from the same index.
                open_event["onset_price"] = peak_p
                update_event_hits(open_event, i)
                state = "DOWN"
        elif state == "DOWN":
            assert open_event is not None
            if p < float(open_event["extreme_price"]):
                open_event["extreme_price"] = p
                open_event["extreme_i"] = i
            update_event_hits(open_event, i)
            rebound = (p / float(open_event["extreme_price"]) - 1.0) * 10_000.0 if float(open_event["extreme_price"]) > 0 else math.nan
            if rebound + 1e-12 >= base_threshold_bps:
                old = open_event
                trough_i = int(old["extreme_i"])
                trough_p = float(old["extreme_price"])
                finalize(old, "OPPOSITE_35BP_CONFIRMED", i)
                open_event = new_event(+1, trough_i, i)
                open_event["onset_price"] = trough_p
                update_event_hits(open_event, i)
                state = "UP"
        prev_valid_i = i

    if open_event is not None:
        finalize(open_event, "RANGE_END_CENSORED", None)
    return records


def coverage_fraction(csum, lo: int, hi: int) -> float:
    if hi <= lo:
        return math.nan
    return float((csum[hi] - csum[lo]) / (hi - lo))


def first_fixed_departure(price, onset_i: int, end_i: int, direction: int, threshold_bps: float) -> int | None:
    anchor = float(price[onset_i])
    for i in range(onset_i, end_i + 1):
        if directional_bps(float(price[i]), anchor, direction) + 1e-12 >= threshold_bps:
            return i
    return None


def first_vol_departure(
    price,
    rv60,
    onset_i: int,
    end_i: int,
    direction: int,
    mult: float,
    lo_bps: float,
    hi_bps: float,
) -> tuple[int | None, float | None]:
    anchor = float(price[onset_i])
    for i in range(onset_i, end_i + 1):
        rv = float(rv60[i]) if math.isfinite(float(rv60[i])) else math.nan
        if not math.isfinite(rv):
            continue
        threshold = min(hi_bps, max(lo_bps, mult * rv))
        if directional_bps(float(price[i]), anchor, direction) + 1e-12 >= threshold:
            return i, threshold
    return None, None


def first_prior_range_break(
    price,
    sec,
    onset_i: int,
    end_i: int,
    direction: int,
    lookback_sec: int,
    buffer_bps: float,
) -> int | None:
    for i in range(onset_i, end_i + 1):
        lo = max(0, i - lookback_sec)
        # Require exact contiguous second history for the chosen trailing window.
        if i - lo < max(5, lookback_sec // 3):
            continue
        hist = price[lo:i]
        if len(hist) == 0:
            continue
        if not all(math.isfinite(float(x)) and float(x) > 0 for x in hist):
            continue
        p = float(price[i])
        if direction > 0:
            ref = max(float(x) for x in hist)
            if bps_return(p, ref) + 1e-12 >= buffer_bps:
                return i
        else:
            ref = min(float(x) for x in hist)
            if directional_bps(p, ref, -1) + 1e-12 >= buffer_bps:
                return i
    return None


def materialize_events(
    raw_events: Sequence[dict[str, Any]],
    sec,
    price,
    book_valid,
    base_threshold_bps: float,
    targets_bps: Sequence[float],
    horizons_sec: Sequence[int],
):
    np, pd = require_numeric_stack()
    valid_int = np.asarray(book_valid, dtype=bool).astype(np.int64)
    bv_csum = np.concatenate(([0], np.cumsum(valid_int)))
    max_h = int(max(horizons_sec))
    target_set = sorted(set(float(x) for x in targets_bps))
    events: list[PhysicalEvent] = []
    memberships: list[dict[str, Any]] = []

    for ev in raw_events:
        oi = int(ev["onset_i"])
        direction = int(ev["direction"])
        onset_price = float(ev["onset_price"])
        base_hit_i = int(ev["base_hit_i"])
        extreme_i = int(ev["extreme_i"])
        onset_sec = int(sec[oi])
        eid = stable_id(SCHEMA_NAME, DEFAULT_SYMBOL, onset_sec, direction, base_threshold_bps)
        hits_by_target: dict[float, int | None] = {float(t): ev["hits"].get(float(t)) for t in target_set}
        max_forward_i = min(len(price) - 1, oi + max_h)
        # Restrict 30-minute excursion to contiguous valid catalog-price seconds.
        end_i = oi
        for j in range(oi, max_forward_i + 1):
            if j > oi and int(sec[j]) != int(sec[j - 1]) + 1:
                break
            if not (math.isfinite(float(price[j])) and float(price[j]) > 0):
                break
            end_i = j
        path = price[oi : end_i + 1]
        dir_moves = [directional_bps(float(x), onset_price, direction) for x in path]
        opp_moves = [directional_bps(float(x), onset_price, -direction) for x in path]
        mfe = max(dir_moves) if dir_moves else math.nan
        mae = -max(opp_moves) if opp_moves else math.nan

        pre_cov: dict[int, float] = {}
        for w in PRE_BOOK_WINDOWS_SEC:
            lo = max(0, oi - w)
            # Only call it a full trailing window if second stamps are contiguous.
            if oi - lo <= 0 or int(sec[oi]) - int(sec[lo]) != oi - lo:
                pre_cov[w] = math.nan
            else:
                pre_cov[w] = coverage_fraction(bv_csum, lo, oi)

        hit_vals: dict[int, int | None] = {}
        tt_vals: dict[int, int | None] = {}
        for t in target_set:
            key = int(round(t))
            hi = hits_by_target[t]
            hit_vals[key] = int(sec[hi]) if hi is not None else None
            tt_vals[key] = int(sec[hi] - onset_sec) if hi is not None else None
            for h in horizons_sec:
                memberships.append(
                    {
                        "event_id": eid,
                        "symbol": DEFAULT_SYMBOL,
                        "onset_date": datetime.fromtimestamp(onset_sec, tz=timezone.utc).date().isoformat(),
                        "direction": direction,
                        "direction_label": "UP" if direction > 0 else "DOWN",
                        "target_bps": float(t),
                        "horizon_sec": int(h),
                        "target_hit": bool(hi is not None and int(sec[hi] - onset_sec) <= int(h)),
                        "time_to_target_sec": int(sec[hi] - onset_sec) if hi is not None else None,
                    }
                )

        base_tt = int(sec[base_hit_i] - onset_sec)
        extreme_sec = int(sec[extreme_i])
        rev_i = ev.get("reversal_confirm_i")
        rev_sec = int(sec[int(rev_i)]) if rev_i is not None else None
        leg_exc = directional_bps(float(ev["extreme_price"]), onset_price, direction)
        events.append(
            PhysicalEvent(
                event_id=eid,
                symbol=DEFAULT_SYMBOL,
                direction=direction,
                direction_label="UP" if direction > 0 else "DOWN",
                onset_second=onset_sec,
                onset_time=datetime.fromtimestamp(onset_sec, tz=timezone.utc).isoformat(),
                onset_date=datetime.fromtimestamp(onset_sec, tz=timezone.utc).date().isoformat(),
                onset_price=onset_price,
                base_threshold_bps=float(base_threshold_bps),
                base_hit_second=int(sec[base_hit_i]),
                base_hit_time=datetime.fromtimestamp(int(sec[base_hit_i]), tz=timezone.utc).isoformat(),
                base_time_to_hit_sec=base_tt,
                extreme_second=extreme_sec,
                extreme_time=datetime.fromtimestamp(extreme_sec, tz=timezone.utc).isoformat(),
                extreme_price=float(ev["extreme_price"]),
                reversal_confirm_second=rev_sec,
                reversal_confirm_time=datetime.fromtimestamp(rev_sec, tz=timezone.utc).isoformat() if rev_sec is not None else None,
                closure_reason=str(ev["closure_reason"]),
                leg_duration_to_extreme_sec=int(extreme_sec - onset_sec),
                leg_max_excursion_bps=float(leg_exc),
                hit_35_second=hit_vals.get(35),
                hit_50_second=hit_vals.get(50),
                hit_75_second=hit_vals.get(75),
                hit_100_second=hit_vals.get(100),
                time_to_35_sec=tt_vals.get(35),
                time_to_50_sec=tt_vals.get(50),
                time_to_75_sec=tt_vals.get(75),
                time_to_100_sec=tt_vals.get(100),
                forward_30m_mfe_bps=float(mfe) if math.isfinite(mfe) else None,
                forward_30m_mae_bps=float(mae) if math.isfinite(mae) else None,
                pre_book_valid_5s=pre_cov[5],
                pre_book_valid_10s=pre_cov[10],
                pre_book_valid_30s=pre_cov[30],
                pre_book_valid_60s=pre_cov[60],
                pre_book_valid_120s=pre_cov[120],
                pre_book_valid_300s=pre_cov[300],
                primary_catalog_eligible=bool(base_tt <= max_h),
            )
        )
    return pd.DataFrame([asdict(x) for x in events]), pd.DataFrame(memberships)


def build_onset_candidates(
    events_df,
    sec,
    price,
    rv60,
    fixed_departure_bps: float,
    range_lookback_sec: int,
    range_break_buffer_bps: float,
    vol_mult: float,
    vol_lo: float,
    vol_hi: float,
):
    _, pd = require_numeric_stack()
    sec_to_i = {int(s): i for i, s in enumerate(sec)}
    rows: list[dict[str, Any]] = []
    for ev in events_df.itertuples(index=False):
        oi = sec_to_i.get(int(ev.onset_second))
        hi = sec_to_i.get(int(ev.base_hit_second))
        if oi is None or hi is None or hi < oi:
            continue
        methods: list[tuple[str, int | None, float | None, bool, bool]] = []
        methods.append(("EXTREMUM_HINDSIGHT", oi, 0.0, False, True))
        fi = first_fixed_departure(price, oi, hi, int(ev.direction), fixed_departure_bps)
        methods.append(("FIXED_10BP_DEPARTURE", fi, fixed_departure_bps if fi is not None else None, False, True))
        vi, vthr = first_vol_departure(price, rv60, oi, hi, int(ev.direction), vol_mult, vol_lo, vol_hi)
        methods.append(("VOL_SCALED_DEPARTURE", vi, vthr, False, True))
        ri = first_prior_range_break(price, sec, oi, hi, int(ev.direction), range_lookback_sec, range_break_buffer_bps)
        methods.append(("PRIOR_60S_RANGE_BREAK", ri, range_break_buffer_bps if ri is not None else None, True, False))

        for method, idx, rule_threshold, causal_rule_candidate, uses_hindsight_extremum in methods:
            if idx is None:
                rows.append(
                    {
                        "event_id": ev.event_id,
                        "onset_date": ev.onset_date,
                        "direction": int(ev.direction),
                        "direction_label": ev.direction_label,
                        "onset_method": method,
                        "candidate_found": False,
                        "candidate_second": None,
                        "candidate_time": None,
                        "rule_threshold_bps": rule_threshold,
                        "displacement_from_event_extremum_bps": None,
                        "seconds_after_event_extremum": None,
                        "seconds_before_35bp_hit": None,
                        "fraction_35_realized": None,
                        "fraction_50_realized": None,
                        "fraction_75_realized": None,
                        "fraction_100_realized": None,
                        "causal_rule_candidate": causal_rule_candidate,
                        "uses_hindsight_event_extremum": uses_hindsight_extremum,
                    }
                )
                continue
            disp = directional_bps(float(price[idx]), float(ev.onset_price), int(ev.direction))
            rows.append(
                {
                    "event_id": ev.event_id,
                    "onset_date": ev.onset_date,
                    "direction": int(ev.direction),
                    "direction_label": ev.direction_label,
                    "onset_method": method,
                    "candidate_found": True,
                    "candidate_second": int(sec[idx]),
                    "candidate_time": datetime.fromtimestamp(int(sec[idx]), tz=timezone.utc).isoformat(),
                    "rule_threshold_bps": float(rule_threshold) if rule_threshold is not None else None,
                    "displacement_from_event_extremum_bps": float(disp),
                    "seconds_after_event_extremum": int(sec[idx] - int(ev.onset_second)),
                    "seconds_before_35bp_hit": int(int(ev.base_hit_second) - sec[idx]),
                    "fraction_35_realized": float(disp / 35.0),
                    "fraction_50_realized": float(disp / 50.0),
                    "fraction_75_realized": float(disp / 75.0),
                    "fraction_100_realized": float(disp / 100.0),
                    "causal_rule_candidate": causal_rule_candidate,
                    "uses_hindsight_event_extremum": uses_hindsight_extremum,
                }
            )
    return pd.DataFrame(rows)


def check_invariants(events, memberships, onsets, targets: Sequence[float], horizons: Sequence[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(name: str, ok: bool, observed: Any, expected: Any) -> None:
        rows.append({"check": name, "status": "PASS" if ok else "FAIL", "observed": observed, "expected": expected})

    add("event_ids_unique", bool(events["event_id"].is_unique), int(events["event_id"].nunique()), len(events))
    if len(events):
        dirs = set(events["direction"].dropna().astype(int).unique())
        add("directions_are_plus_minus_one", dirs.issubset({-1, 1}), sorted(dirs), [-1, 1])
        add("base_hit_not_before_onset", bool((events["base_hit_second"] >= events["onset_second"]).all()), None, True)
        add("extreme_not_before_onset", bool((events["extreme_second"] >= events["onset_second"]).all()), None, True)
        # Nested hit ordering.
        ok_nested = True
        for r in events.itertuples(index=False):
            vals = [r.hit_35_second, r.hit_50_second, r.hit_75_second, r.hit_100_second]
            seen_none = False
            prev = None
            for v in vals:
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    seen_none = True
                    continue
                if seen_none:
                    ok_nested = False
                    break
                if prev is not None and int(v) < int(prev):
                    ok_nested = False
                    break
                prev = v
            if not ok_nested:
                break
        add("target_hits_nested_and_ordered", ok_nested, None, True)
    expected_memberships = len(events) * len(set(float(x) for x in targets)) * len(set(int(x) for x in horizons))
    add("membership_grid_complete", len(memberships) == expected_memberships, len(memberships), expected_memberships)
    add("onset_methods_four_per_event", len(onsets) == len(events) * 4, len(onsets), len(events) * 4)
    if len(onsets):
        found = onsets[onsets["candidate_found"].astype(bool)]
        add(
            "candidate_onsets_not_after_35bp_hit",
            bool((found["seconds_before_35bp_hit"] >= 0).all()) if len(found) else True,
            None,
            True,
        )
    # Physical legs should alternate direction when sorted by onset within each continuous segment.
    ordered = events.sort_values(["onset_second", "base_hit_second"], kind="mergesort")
    duplicate_onsets = int(ordered.duplicated(["onset_second", "direction"]).sum()) if len(ordered) else 0
    add("no_duplicate_same_direction_physical_onset", duplicate_onsets == 0, duplicate_onsets, 0)
    return rows


def build_reports(events, memberships, onsets, checks, report_root: Path, args, summary: dict[str, Any]):
    np, pd = require_numeric_stack()
    report_root.mkdir(parents=True, exist_ok=True)

    # Counts by target/horizon/direction.
    counts = (
        memberships[memberships["target_hit"].astype(bool)]
        .groupby(["target_bps", "horizon_sec", "direction_label"], dropna=False)
        .size()
        .reset_index(name="physical_event_count")
    )
    counts.to_csv(report_root / "large_move_event_counts.csv", index=False)

    matrix = (
        memberships.groupby(["target_bps", "horizon_sec"], dropna=False)["target_hit"]
        .agg([("physical_event_count", "sum"), ("eligible_physical_events", "size")])
        .reset_index()
    )
    matrix["share_of_base_physical_events"] = np.where(
        matrix["eligible_physical_events"] > 0,
        matrix["physical_event_count"] / matrix["eligible_physical_events"],
        np.nan,
    )
    matrix.to_csv(report_root / "large_move_horizon_matrix.csv", index=False)

    # Target nesting using 30-minute hits.
    nest_rows: list[dict[str, Any]] = []
    primary = events[events["primary_catalog_eligible"].astype(bool)].copy()
    for a in (35, 50, 75, 100):
        ca = primary[f"time_to_{a}_sec"].notna() & (primary[f"time_to_{a}_sec"] <= max(args.horizons_sec))
        n_a = int(ca.sum())
        for b in (35, 50, 75, 100):
            if b <= a:
                continue
            cb = primary[f"time_to_{b}_sec"].notna() & (primary[f"time_to_{b}_sec"] <= max(args.horizons_sec))
            both = int((ca & cb).sum())
            nest_rows.append(
                {
                    "base_target_bps": a,
                    "higher_target_bps": b,
                    "base_event_count": n_a,
                    "also_hits_higher_count": both,
                    "conditional_share": both / n_a if n_a else math.nan,
                }
            )
    pd.DataFrame(nest_rows).to_csv(report_root / "large_move_target_nesting.csv", index=False)

    dur_rows: list[dict[str, Any]] = []
    for target in (35, 50, 75, 100):
        col = f"time_to_{target}_sec"
        vals = pd.to_numeric(primary[col], errors="coerce").dropna()
        if len(vals):
            vals = vals[vals <= max(args.horizons_sec)]
        if len(vals):
            dur_rows.append(
                {
                    "target_bps": target,
                    "n": len(vals),
                    "median_time_to_target_sec": float(vals.median()),
                    "p25_sec": float(vals.quantile(0.25)),
                    "p75_sec": float(vals.quantile(0.75)),
                    "p90_sec": float(vals.quantile(0.90)),
                    "p95_sec": float(vals.quantile(0.95)),
                }
            )
    pd.DataFrame(dur_rows).to_csv(report_root / "large_move_duration_profile.csv", index=False)

    onset_rows: list[dict[str, Any]] = []
    o = onsets[onsets["candidate_found"].astype(bool)].copy()
    for method, g in o.groupby("onset_method"):
        onset_rows.append(
            {
                "onset_method": method,
                "n_found": len(g),
                "event_coverage_rate": len(g) / len(events) if len(events) else math.nan,
                "median_seconds_after_extremum": float(g["seconds_after_event_extremum"].median()),
                "median_seconds_before_35bp_hit": float(g["seconds_before_35bp_hit"].median()),
                "median_displacement_bps": float(g["displacement_from_event_extremum_bps"].median()),
                "median_fraction_35_realized": float(g["fraction_35_realized"].median()),
                "median_fraction_50_realized": float(g["fraction_50_realized"].median()),
                "p90_fraction_50_realized": float(g["fraction_50_realized"].quantile(0.90)),
                "causal_rule_candidate": bool(g["causal_rule_candidate"].all()),
                "uses_hindsight_event_extremum": bool(g["uses_hindsight_event_extremum"].any()),
            }
        )
    pd.DataFrame(onset_rows).to_csv(report_root / "large_move_onset_audit.csv", index=False)

    cov_rows: list[dict[str, Any]] = []
    for target in (35, 50, 75, 100):
        mask = primary[f"time_to_{target}_sec"].notna() & (primary[f"time_to_{target}_sec"] <= max(args.horizons_sec))
        g = primary[mask]
        if not len(g):
            continue
        for w in PRE_BOOK_WINDOWS_SEC:
            col = f"pre_book_valid_{w}s"
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            cov_rows.append(
                {
                    "target_bps": target,
                    "pre_window_sec": w,
                    "events": len(g),
                    "coverage_observations": len(vals),
                    "median_book_valid_rate": float(vals.median()) if len(vals) else math.nan,
                    "share_full_book_coverage": float((vals >= 0.999999).mean()) if len(vals) else math.nan,
                    "share_ge_95pct_book_coverage": float((vals >= 0.95).mean()) if len(vals) else math.nan,
                }
            )
    pd.DataFrame(cov_rows).to_csv(report_root / "large_move_book_coverage.csv", index=False)

    pd.DataFrame(checks).to_csv(report_root / "large_move_invariants.csv", index=False)

    # Human-readable findings intentionally descriptive only.
    lines = [
        "# Phase 7A — Large-Move Physical-Event Catalog",
        "",
        f"- Builder: **{BUILDER_VERSION}**",
        f"- Range: **{args.start_date} → {args.end_date}**",
        f"- Symbol: **{args.symbol}**",
        f"- Price source: **{summary['price_source']}**",
        f"- Physical-event ontology: **{summary['physical_event_ontology']}**",
        f"- Physical events: **{summary['physical_event_count']:,}**",
        f"- Primary ≤30m base events: **{summary['primary_event_count']:,}**",
        "",
        "## Target counts within 30 minutes",
        "",
    ]
    for target in (35, 50, 75, 100):
        lines.append(f"- {target}+ bps: **{summary.get(f'hit_{target}_within_30m', 0):,}**")
    lines += [
        "",
        "## Interpretation",
        "",
        "Phase 7A is a retrospective event ontology, not a predictor. The 35-bps",
        "directional-change extremum is intentionally hindsight-defined so overlapping",
        "forward windows collapse into physical directional legs. Candidate onset rules",
        "are audited separately and must be challenged in Phase 7B before any causal",
        "precursor study begins.",
        "",
        "No alpha, strategy, or profitability claim is made by Phase 7A.",
    ]
    atomic_write_text(report_root / "large_move_findings.md", "\n".join(lines) + "\n")


def partition_write(df, root: Path, symbol: str, date_col: str, leaf_name: str) -> None:
    if df.empty:
        return
    for day_text, g in df.groupby(date_col, sort=True):
        path = root / f"symbol={symbol}" / f"onset_date={day_text}" / f"part-{day_text}.parquet"
        write_parquet_atomic(g.reset_index(drop=True), path)


def self_test() -> None:
    np, pd = require_numeric_stack()
    # 1-second synthetic path: quiet, +60 bps physical up leg, 35+bps reversal,
    # then a down leg. Introduce no price gap so event nesting can be checked.
    n = 900
    sec = np.arange(1_700_000_000, 1_700_000_000 + n, dtype=np.int64)
    p = np.full(n, 100.0, dtype=float)
    # Small deterministic pre-noise.
    for i in range(1, 100):
        p[i] = 100.0 + 0.001 * math.sin(i / 7.0)
    # Local trough at 100, then rise 60 bps over 60 sec.
    for k, i in enumerate(range(100, 161)):
        p[i] = 100.0 * (1.0 + (k / 60.0) * 0.0060)
    # Hold peak then fall >35 bps from peak.
    p[161:170] = p[160]
    for k, i in enumerate(range(170, 211)):
        p[i] = p[160] * (1.0 - (k / 40.0) * 0.0040)
    p[211:] = p[210]
    book_valid = np.ones(n, dtype=bool)
    raw = segment_directional_change(sec, p, DEFAULT_SYMBOL, 35.0, (35.0, 50.0, 75.0, 100.0))
    assert len(raw) >= 2, raw
    events, memberships = materialize_events(raw, sec, p, book_valid, 35.0, (35.0, 50.0, 75.0, 100.0), (300, 600, 900, 1800))
    first = events.iloc[0]
    assert int(first.direction) == 1
    assert pd.notna(first.hit_35_second)
    assert pd.notna(first.hit_50_second)
    assert pd.isna(first.hit_75_second)
    assert int(first.hit_35_second) <= int(first.hit_50_second)
    rv = trailing_rv_bps(p, sec, 60)
    onsets = build_onset_candidates(events, sec, p, rv, 10.0, 60, 1.0, 0.5, 5.0, 15.0)
    checks = check_invariants(events, memberships, onsets, (35.0, 50.0, 75.0, 100.0), (300, 600, 900, 1800))
    failed = [x for x in checks if x["status"] == "FAIL"]
    assert not failed, failed
    # Gap reset test: no event may bridge a NaN price gap.
    p2 = p.copy()
    p2[130:135] = np.nan
    raw2 = segment_directional_change(sec, p2, DEFAULT_SYMBOL, 35.0, (35.0, 50.0))
    for ev in raw2:
        oi, bi = ev["onset_i"], ev["base_hit_i"]
        assert not (oi < 130 and bi >= 135), ev
    # +10 bps departure must occur after/equal hindsight extremum and before base hit.
    e0 = events.iloc[0]
    oo = onsets[(onsets.event_id == e0.event_id) & (onsets.onset_method == "FIXED_10BP_DEPARTURE")].iloc[0]
    assert bool(oo.candidate_found)
    assert 0 <= int(oo.seconds_after_event_extremum) <= int(e0.base_time_to_hit_sec)
    assert abs(float(oo.fraction_50_realized) - 0.20) < 0.03
    print("SELF-TEST PASS")
    print("Validated 35-bps physical-leg de-duplication, nested 50/75/100 first hits,")
    print("horizon membership, gap censoring, book-coverage plumbing, and deterministic")
    print("candidate-onset audits without treating retrospective extrema as causal signals.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build Phase-7A large-move physical-event catalog")
    p.add_argument("--start-date", default="2026-05-18")
    p.add_argument("--end-date", default="2026-07-18")
    p.add_argument("--symbol", default=DEFAULT_SYMBOL)
    p.add_argument("--execution-root", type=Path, default=DEFAULT_EXEC_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--price-source", choices=["last_trade", "mid"], default="last_trade")
    p.add_argument("--max-trade-stale-sec", type=int, default=DEFAULT_MAX_TRADE_STALE_SEC)
    p.add_argument("--base-threshold-bps", type=float, default=DEFAULT_BASE_THRESHOLD_BPS)
    p.add_argument("--targets-bps", type=float, nargs="+", default=list(DEFAULT_TARGETS_BPS))
    p.add_argument("--horizons-sec", type=int, nargs="+", default=list(DEFAULT_HORIZONS_SEC))
    p.add_argument("--fixed-departure-bps", type=float, default=DEFAULT_FIXED_DEPARTURE_BPS)
    p.add_argument("--range-lookback-sec", type=int, default=DEFAULT_RANGE_LOOKBACK_SEC)
    p.add_argument("--range-break-buffer-bps", type=float, default=DEFAULT_RANGE_BREAK_BUFFER_BPS)
    p.add_argument("--vol-lookback-sec", type=int, default=DEFAULT_VOL_LOOKBACK_SEC)
    p.add_argument("--vol-multiplier", type=float, default=DEFAULT_VOL_MULTIPLIER)
    p.add_argument("--vol-departure-min-bps", type=float, default=DEFAULT_VOL_DEPARTURE_MIN_BPS)
    p.add_argument("--vol-departure-max-bps", type=float, default=DEFAULT_VOL_DEPARTURE_MAX_BPS)
    p.add_argument("--self-test", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        self_test()
        return 0

    np, pd, _, _ = require_parquet_stack()
    start = parse_date(args.start_date)
    end = parse_date(args.end_date)
    if end < start:
        raise SystemExit("--end-date must be >= --start-date")
    targets = sorted(set(float(x) for x in args.targets_bps))
    if args.base_threshold_bps not in targets:
        targets = sorted(set(targets + [float(args.base_threshold_bps)]))
    horizons = sorted(set(int(x) for x in args.horizons_sec))
    if max(horizons) > 1800:
        print("WARNING: design intent is ≤30-minute target realization; continuing with requested horizon.")

    days = daterange(start, end)
    print("Building Phase-7A large-move physical-event catalog...")
    print(f"Builder version       : {BUILDER_VERSION}")
    print(f"Symbol                : {args.symbol}")
    print(f"Dates                 : {len(days)} ({start} to {end})")
    print(f"Execution root        : {args.execution_root.resolve()}")
    print(f"Output root           : {args.output_root.resolve()}")
    print(f"Report root           : {args.report_root.resolve()}")
    print(f"Base physical threshold: {args.base_threshold_bps:g} bps")
    print(f"Nested targets        : {', '.join(f'{x:g}' for x in targets)} bps")
    print(f"Realization horizons  : {', '.join(str(x//60) for x in horizons)} minutes")
    print(f"Price source          : {args.price_source}")
    print("Physical ontology     : 35-bps directional-change legs; retrospective de-duplication")
    print("No precursor or alpha claim is made by Phase 7A.")

    frame = load_execution_range(args.execution_root, args.symbol, days)
    sec, price, price_source_label = build_catalog_price(frame, args.max_trade_stale_sec, args.price_source)
    book_valid = frame["book_valid"].fillna(False).astype(bool).to_numpy()
    valid_price = np.isfinite(price) & (price > 0)
    print(f"Loaded seconds        : {len(frame):,}")
    print(f"Catalog-price coverage: {valid_price.mean()*100:.4f}%")

    print("[1/5] Segmenting de-duplicated physical directional legs...")
    raw = segment_directional_change(sec, price, args.symbol, args.base_threshold_bps, targets)
    print(f"      physical legs: {len(raw):,}")

    print("[2/5] Materializing nested target/horizon membership...")
    events, memberships = materialize_events(raw, sec, price, book_valid, args.base_threshold_bps, targets, horizons)
    events["symbol"] = args.symbol
    memberships["symbol"] = args.symbol

    print("[3/5] Building deterministic candidate-onset audit...")
    rv = trailing_rv_bps(price, sec, args.vol_lookback_sec)
    onsets = build_onset_candidates(
        events,
        sec,
        price,
        rv,
        args.fixed_departure_bps,
        args.range_lookback_sec,
        args.range_break_buffer_bps,
        args.vol_multiplier,
        args.vol_departure_min_bps,
        args.vol_departure_max_bps,
    )
    onsets["symbol"] = args.symbol

    print("[4/5] Checking event ontology invariants...")
    checks = check_invariants(events, memberships, onsets, targets, horizons)
    failed = [x for x in checks if x["status"] == "FAIL"]

    # Metadata columns.
    for df in (events, memberships, onsets):
        df["catalog_schema_version"] = SCHEMA_VERSION
        df["builder_version"] = BUILDER_VERSION
        df["price_source"] = price_source_label
        df["physical_event_ontology"] = f"DIRECTIONAL_CHANGE_{args.base_threshold_bps:g}BPS_V1"

    print("[5/5] Writing catalog partitions and reports...")
    args.output_root.mkdir(parents=True, exist_ok=True)
    partition_write(events, args.output_root / "events", args.symbol, "onset_date", "events")
    partition_write(onsets, args.output_root / "onset_candidates", args.symbol, "onset_date", "onsets")
    write_parquet_atomic(events, args.output_root / "physical_events.parquet")
    write_parquet_atomic(onsets, args.output_root / "onset_candidates.parquet")
    write_parquet_atomic(memberships, args.output_root / "target_membership.parquet")

    max_h = max(horizons)
    primary = events[events["primary_catalog_eligible"].astype(bool)]
    summary: dict[str, Any] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "builder_version": BUILDER_VERSION,
        "generated_at": utc_now_iso(),
        "symbol": args.symbol,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "loaded_seconds": int(len(frame)),
        "catalog_price_valid_seconds": int(valid_price.sum()),
        "catalog_price_coverage_rate": float(valid_price.mean()),
        "price_source": price_source_label,
        "physical_event_ontology": f"DIRECTIONAL_CHANGE_{args.base_threshold_bps:g}BPS_V1",
        "physical_event_count": int(len(events)),
        "primary_event_count": int(len(primary)),
        "failed_invariants": int(len(failed)),
        "targets_bps": targets,
        "horizons_sec": horizons,
    }
    for t in (35, 50, 75, 100):
        col = f"time_to_{t}_sec"
        if col in events.columns:
            summary[f"hit_{t}_within_30m"] = int((events[col].notna() & (events[col] <= max_h)).sum())

    config = {
        "builder_version": BUILDER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "symbol": args.symbol,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "execution_root": str(args.execution_root),
        "price_source": args.price_source,
        "max_trade_stale_sec": args.max_trade_stale_sec,
        "base_threshold_bps": args.base_threshold_bps,
        "targets_bps": targets,
        "horizons_sec": horizons,
        "fixed_departure_bps": args.fixed_departure_bps,
        "range_lookback_sec": args.range_lookback_sec,
        "range_break_buffer_bps": args.range_break_buffer_bps,
        "vol_lookback_sec": args.vol_lookback_sec,
        "vol_multiplier": args.vol_multiplier,
        "vol_departure_min_bps": args.vol_departure_min_bps,
        "vol_departure_max_bps": args.vol_departure_max_bps,
        "physical_event_ontology": summary["physical_event_ontology"],
    }
    atomic_write_json(args.output_root / "metadata" / "latest_summary.json", summary)
    atomic_write_json(args.output_root / "metadata" / "run_config.json", config)
    build_reports(events, memberships, onsets, checks, args.report_root, args, summary)

    print()
    print("Phase-7A large-move catalog summary")
    print("-----------------------------------")
    print(f"Status                    : {'PASS' if not failed else 'FAIL'}")
    print(f"Physical events           : {len(events):,}")
    print(f"Primary ≤30m base events  : {len(primary):,}")
    for t in (35, 50, 75, 100):
        print(f"{t:>3}+ bps within 30m      : {summary.get(f'hit_{t}_within_30m', 0):,}")
    print(f"Candidate onset rows       : {len(onsets):,}")
    print(f"Failed invariants          : {len(failed):,}")
    print(f"Summary                    : {(args.output_root / 'metadata' / 'latest_summary.json').resolve()}")
    print(f"Findings                   : {(args.report_root / 'large_move_findings.md').resolve()}")
    print("No alpha or strategy claim is made by Phase 7A.")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
