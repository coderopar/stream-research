#!/usr/bin/env python3
"""Evaluate causal BTCUSDT reactions around deterministic price levels versus controls.

Pass 4B of the historical-context pipeline.

Research purpose
----------------
Before rebuilding a fitted multi-year statistical level engine, establish a zero/few-parameter
benchmark: when BTCUSDT approaches an obvious deterministic reference level, is that price
harder to penetrate than comparable control prices?

This script deliberately does NOT use participant-flow features and does NOT optimize a
level model. It operates only on the independently audited Binance USD-M BTCUSDT 1-minute
history.

Primary pre-registered contract
-------------------------------
* Analysis window by default: 2020-02-01 through 2025-12-31 UTC. January 2020 is warm-up.
* Level families are evaluated separately; they are never pooled into one "key level" score.
* Primary interaction tolerance: 0.10 x trailing 24h realized sigma, clipped to 10..75 bps.
* Primary endpoint: maximum 60-minute penetration beyond the approached level, normalized
  by trailing 24h realized sigma known strictly before the event minute.
* Primary contrast: REAL minus CONTROL. Negative values mean real levels were harder to pass.
* Planned minimum-detectable-effect target: 0.15 SD at alpha=0.05, power=0.80.
* Secondary outcomes and tolerance sweeps are explicitly exploratory/sensitivity analyses.

Deterministic level families
----------------------------
    PREV_DAY_HIGH / PREV_DAY_LOW
    PREV_WEEK_HIGH / PREV_WEEK_LOW
    PREV_MONTH_HIGH / PREV_MONTH_LOW
    ROLLING_7D_HIGH / ROLLING_7D_LOW
    ROLLING_30D_HIGH / ROLLING_30D_LOW
    ROUND_1K
    ROUND_5K
    ATH_TO_DATE   (emitted only from 2021-01-01 by default to avoid early-history truncation)

Controls
--------
1. SHIFTED: multiplicative price shifts at +/-150 and +/-300 bps by default.
2. WRONG_DAY: same deterministic family as it existed 5, 10 and 20 days earlier.
3. MATCHED_RANDOM: causal price-bin control selected inside +/-25% of day-start price to
   approximately match signed distance and trailing-30d close-price exposure, while avoiding
   collisions with real deterministic levels.

Approach-event grammar
----------------------
For a level/day/tolerance, an event is the first 1m bar whose range comes within tolerance
of the level after the level has previously been "armed": price closed at least 3*tolerance
away for 60 consecutive minutes. At most one event per level per UTC day is emitted.

Causality
---------
* Day-level references use only completed prior UTC periods.
* Trailing volatility is shifted one minute; the event bar cannot influence its own tolerance.
* Rolling 7d/30d extrema exclude the current day.
* ATH_TO_DATE excludes the current day.
* Wrong-day and matched-random controls use only information available before the current day.
* Outcome windows begin at the event minute and are never used to construct levels/controls.

Default audited source
----------------------
    D:\\stream-data\\historical\\binance\\futures_um\\BTCUSDT\\klines_1m

Default output
--------------
    D:\\stream-data\\historical\\binance\\futures_um\\BTCUSDT\\deterministic_level_reactions_v1

Reports under repository
------------------------
    reports/binance_um_deterministic_level_reactions/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sqlite3
import sys
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

EVALUATOR_VERSION = "1.0.1"
SCHEMA_VERSION = "deterministic_level_reactions_v1"
DEFAULT_HISTORICAL_ROOT = Path(r"D:\stream-data\historical\binance\futures_um\BTCUSDT\klines_1m")
DEFAULT_OUTPUT_ROOT = Path(r"D:\stream-data\historical\binance\futures_um\BTCUSDT\deterministic_level_reactions_v1")
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_ANALYSIS_START = "2020-02-01"
DEFAULT_ANALYSIS_END = "2025-12-31"
DEFAULT_ATH_VALID_FROM = "2021-01-01"

PRIMARY_TOLERANCE_ID = "VOL_0P10"
PRIMARY_ENDPOINT = "penetration_vol_60m"
PLANNED_MDE_TARGET_SD = 0.15
ALPHA = 0.05
POWER = 0.80
Z_ALPHA_2 = 1.959963984540054
Z_POWER_80 = 0.8416212335729143
Z_MDE = Z_ALPHA_2 + Z_POWER_80

TRAILING_VOL_MINUTES = 1440
TRAILING_VOL_MIN_PERIODS = 720
ARM_DISTANCE_MULTIPLIER = 3.0
ARM_CONSECUTIVE_MINUTES = 60
MAX_EVENTS_PER_LEVEL_DAY = 1
MATCH_EXPOSURE_DAYS = 30
EXPOSURE_BIN_BPS = 25.0
MATCH_ACTIVE_WINDOW_PCT = 0.25
ROUND_ACTIVE_WINDOW_PCT = 0.50
ROUND_GRID_MAX_DEFAULT = 250_000.0
CONTROL_COLLISION_BPS = 20.0
CONFLUENCE_FLOOR_BPS = 25.0
RNG_SEED = 20260808

SHIFT_OFFSETS_BPS = (-300.0, -150.0, 150.0, 300.0)
WRONG_DAY_LAGS = (5, 10, 20)

# Primary plus explicitly labelled sensitivity definitions.
TOLERANCE_SPECS: dict[str, dict[str, Any]] = {
    "VOL_0P05": {"kind": "VOL", "value": 0.05, "min_bps": 5.0, "max_bps": 50.0, "primary": False},
    "VOL_0P10": {"kind": "VOL", "value": 0.10, "min_bps": 10.0, "max_bps": 75.0, "primary": True},
    "VOL_0P20": {"kind": "VOL", "value": 0.20, "min_bps": 15.0, "max_bps": 125.0, "primary": False},
    "FIXED_10": {"kind": "FIXED", "value": 10.0, "primary": False},
    "FIXED_20": {"kind": "FIXED", "value": 20.0, "primary": False},
    "FIXED_30": {"kind": "FIXED", "value": 30.0, "primary": False},
}

TIME_DERIVED_FAMILIES = {
    "PREV_DAY_HIGH", "PREV_DAY_LOW",
    "PREV_WEEK_HIGH", "PREV_WEEK_LOW",
    "PREV_MONTH_HIGH", "PREV_MONTH_LOW",
    "ROLLING_7D_HIGH", "ROLLING_7D_LOW",
    "ROLLING_30D_HIGH", "ROLLING_30D_LOW",
}

REQUIRED_SOURCE_COLUMNS = ["open_time", "open", "high", "low", "close", "quote_volume", "trade_count"]

EVENT_COLUMNS = [
    "run_id", "schema_version", "symbol", "event_id", "event_time", "event_date", "event_year",
    "tolerance_id", "tolerance_primary", "tolerance_kind", "tolerance_parameter", "tau_bps",
    "level_family", "level_side", "level_kind", "control_type", "control_variant",
    "level_id", "source_level_id", "level_price", "source_level_price", "level_available_time", "level_age_hours",
    "approach_from", "event_close", "event_open", "event_high", "event_low",
    "trailing_sigma_bps", "approach_return_15m_bps", "approach_velocity_toward_bps_15m",
    "day_start_price", "signed_distance_from_day_start_bps", "abs_distance_from_day_start_bps",
    "source_abs_distance_from_day_start_bps",
    "recent_exposure_30d", "source_recent_exposure_30d",
    "confluence_count", "confluence_families", "is_confluence",
    "penetration_bps_15m", "penetration_vol_15m", "rejection_bps_15m", "rejection_vol_15m",
    "penetration_bps_60m", "penetration_vol_60m", "rejection_bps_60m", "rejection_vol_60m",
    "penetration_bps_240m", "penetration_vol_240m", "rejection_bps_240m", "rejection_vol_240m",
    "beyond_close_fraction_60m", "near_level_minutes_60m", "final_side_60m",
    "exploratory_resolution_60m", "retest_within_240m",
    "outcome_complete_15m", "outcome_complete_60m", "outcome_complete_240m",
]


@dataclass(frozen=True)
class Level:
    family: str
    side: str
    price: float
    available_time: datetime | None
    level_id: str
    kind: str = "REAL"
    control_type: str = "REAL"
    control_variant: str = "REAL"
    source_level_id: str = ""
    source_level_price: float | None = None
    recent_exposure_30d: int | None = None
    source_recent_exposure_30d: int | None = None


@dataclass
class Check:
    scope: str
    check: str
    status: str
    observed: Any
    expected: Any
    detail: str = ""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def find_repo_root(script_path: Path) -> Path:
    current = script_path.resolve().parent
    for candidate in [current, *current.parents]:
        if (candidate / "scripts").exists() and (candidate / "reports").exists():
            return candidate
    return script_path.resolve().parent


def short_tmp(path: Path, prefix: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.parent / f".{prefix}-{uuid.uuid4().hex[:8]}.tmp"


def atomic_write_text(path: Path, content: str) -> None:
    tmp = short_tmp(path, "tx")
    try:
        tmp.write_text(content, encoding="utf-8", newline="\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def atomic_write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    tmp = short_tmp(path, "cs")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def atomic_write_parquet(path: Path, frame) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    tmp = short_tmp(path, "pq")
    try:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        pq.write_table(table, tmp, compression="zstd", use_dictionary=True)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def atomic_write_dataframe_csv(path: Path, frame) -> None:
    tmp = short_tmp(path, "df")
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def require_runtime_libraries() -> None:
    missing = []
    for name in ("numpy", "pandas", "pyarrow"):
        try:
            __import__(name)
        except Exception:
            missing.append(name)
    if missing:
        raise RuntimeError("Missing required Python packages: " + ", ".join(missing))


def stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def bps_log_ratio(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return math.nan
    return math.log(a / b) * 10_000.0


def exp_bps(price: float, bps: float) -> float:
    return price * math.exp(bps / 10_000.0)


def exposure_bin_idx(price: float) -> int:
    step = math.log1p(EXPOSURE_BIN_BPS / 10_000.0)
    return int(math.floor(math.log(price) / step))


def exposure_bin_center(idx: int) -> float:
    step = math.log1p(EXPOSURE_BIN_BPS / 10_000.0)
    return math.exp((idx + 0.5) * step)


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def iso_week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())


def prev_month_bounds(d: date) -> tuple[date, date]:
    first = d.replace(day=1)
    prev_end = first - timedelta(days=1)
    prev_start = prev_end.replace(day=1)
    return prev_start, prev_end


def level_id(family: str, price: float, available_time: datetime | None, suffix: str = "") -> str:
    stamp = available_time.isoformat() if available_time else "STATIC"
    base = f"{family}|{stamp}|{price:.8f}"
    if suffix:
        base += f"|{suffix}"
    return base


def verify_source_audit(root: Path) -> dict[str, Any]:
    summary_path = root / "metadata" / "latest_audit_summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"Pass-3 audit summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    status = str(summary.get("status", "")).upper()
    if status != "PASS":
        raise RuntimeError(f"Pass-3 historical audit must be PASS, found {status or 'UNKNOWN'}")
    return summary


def discover_parquet_files(root: Path, load_start: date, load_end: date) -> list[Path]:
    files = sorted((root / "parquet" / f"symbol={DEFAULT_SYMBOL}").glob("year=*/month=*/part-00000.parquet"))
    selected: list[Path] = []
    for path in files:
        try:
            year = int(path.parent.parent.name.split("=", 1)[1])
            month = int(path.parent.name.split("=", 1)[1])
            month_start = date(year, month, 1)
            if month == 12:
                month_end = date(year + 1, 1, 1) - timedelta(days=1)
            else:
                month_end = date(year, month + 1, 1) - timedelta(days=1)
            if month_end >= load_start and month_start <= load_end:
                selected.append(path)
        except Exception:
            continue
    if not selected:
        raise RuntimeError("No source Parquet files matched the requested load window")
    return selected


def source_signature(files: Sequence[Path], audit_summary: dict[str, Any], config: dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(EVALUATOR_VERSION.encode())
    h.update(SCHEMA_VERSION.encode())
    h.update(json.dumps(config, sort_keys=True, default=str).encode())
    h.update(json.dumps(audit_summary, sort_keys=True, default=str).encode())
    for path in files:
        st = path.stat()
        h.update(str(path).encode())
        h.update(str(st.st_size).encode())
        h.update(str(st.st_mtime_ns).encode())
    return h.hexdigest()


def init_state_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS yearly_runs (
            year INTEGER PRIMARY KEY,
            input_signature TEXT NOT NULL,
            status TEXT NOT NULL,
            event_rows INTEGER NOT NULL,
            run_id TEXT NOT NULL,
            completed_at TEXT NOT NULL,
            output_path TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def load_source(files: Sequence[Path]):
    import pandas as pd
    frames = []
    for i, path in enumerate(files, 1):
        frame = pd.read_parquet(path, columns=REQUIRED_SOURCE_COLUMNS)
        missing = [c for c in REQUIRED_SOURCE_COLUMNS if c not in frame.columns]
        if missing:
            raise RuntimeError(f"{path} missing columns: {missing}")
        frames.append(frame)
        if i % 12 == 0 or i == len(files):
            print(f"Loaded source months : {i}/{len(files)}")
    df = pd.concat(frames, ignore_index=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.sort_values("open_time", kind="mergesort").drop_duplicates("open_time", keep="first").reset_index(drop=True)
    for c in ("open", "high", "low", "close", "quote_volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df["trade_count"] = pd.to_numeric(df["trade_count"], errors="coerce")
    return df


def prepare_market_arrays(df):
    import numpy as np
    import pandas as pd

    close = df["close"].to_numpy(dtype=float)
    logret = pd.Series(np.log(close)).diff()
    trailing_sigma_bps = (
        logret.rolling(TRAILING_VOL_MINUTES, min_periods=TRAILING_VOL_MIN_PERIODS).std(ddof=0)
        .shift(1)
        .to_numpy(dtype=float)
        * math.sqrt(TRAILING_VOL_MINUTES)
        * 10_000.0
    )
    times = df["open_time"]
    day_starts = times.dt.floor("D")
    df = df.copy()
    df["_day"] = day_starts
    return df, trailing_sigma_bps


def build_daily_stats(df):
    import pandas as pd
    grouped = df.groupby("_day", sort=True)
    daily = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        quote_volume=("quote_volume", "sum"),
        trade_count=("trade_count", "sum"),
        rows=("close", "size"),
    ).reset_index().rename(columns={"_day": "day"})
    daily["day"] = pd.to_datetime(daily["day"], utc=True)
    daily["date"] = daily["day"].dt.date
    return daily


def build_day_index(df) -> dict[date, tuple[int, int]]:
    result: dict[date, tuple[int, int]] = {}
    days = df["_day"].to_numpy()
    if len(days) == 0:
        return result
    start = 0
    for i in range(1, len(days) + 1):
        if i == len(days) or days[i] != days[start]:
            d = df["open_time"].iloc[start].date()
            result[d] = (start, i)
            start = i
    return result


def build_daily_exposure_counts(df, day_index: dict[date, tuple[int, int]]) -> dict[date, Counter]:
    result: dict[date, Counter] = {}
    closes = df["close"].to_numpy(dtype=float)
    for d, (s, e) in day_index.items():
        c = Counter()
        for price in closes[s:e]:
            if math.isfinite(price) and price > 0:
                c[exposure_bin_idx(float(price))] += 1
        result[d] = c
    return result


def build_real_levels_by_day(daily, analysis_start: date, analysis_end: date, ath_valid_from: date,
                             round_grid_max: float) -> dict[date, list[Level]]:
    import numpy as np
    import pandas as pd

    daily = daily.copy().sort_values("date").reset_index(drop=True)
    by_date = {r.date: r for r in daily.itertuples(index=False)}
    dates = list(daily["date"])

    # Causal rolling extrema excluding current day.
    highs = daily["high"].astype(float)
    lows = daily["low"].astype(float)
    daily["roll7_high"] = highs.shift(1).rolling(7, min_periods=7).max()
    daily["roll7_low"] = lows.shift(1).rolling(7, min_periods=7).min()
    daily["roll30_high"] = highs.shift(1).rolling(30, min_periods=30).max()
    daily["roll30_low"] = lows.shift(1).rolling(30, min_periods=30).min()
    daily["ath_prior"] = highs.shift(1).cummax()
    # Day on which the prior all-time-to-date high was established. We use completed
    # days only, so ATH availability is conservatively the following UTC day-start.
    ath_origins: list[date | None] = []
    running_high = -math.inf
    running_origin: date | None = None
    for r in daily.itertuples(index=False):
        ath_origins.append(running_origin)
        if math.isfinite(float(r.high)) and float(r.high) > running_high:
            running_high = float(r.high)
            running_origin = r.date
    daily["ath_prior_origin"] = ath_origins

    roll_map = {r.date: r for r in daily.itertuples(index=False)}

    # Completed prior ISO-week and calendar-month aggregates.
    tmp = daily.copy()
    tmp["week_start"] = tmp["date"].map(iso_week_start)
    weekly = tmp.groupby("week_start", sort=True).agg(high=("high", "max"), low=("low", "min")).reset_index()
    week_map = {r.week_start: r for r in weekly.itertuples(index=False)}

    tmp["month_key"] = tmp["date"].map(lambda x: f"{x.year:04d}-{x.month:02d}")
    monthly = tmp.groupby("month_key", sort=True).agg(high=("high", "max"), low=("low", "min")).reset_index()
    month_map = {r.month_key: r for r in monthly.itertuples(index=False)}

    result: dict[date, list[Level]] = {}
    d = analysis_start
    while d <= analysis_end:
        if d not in by_date:
            d += timedelta(days=1)
            continue
        levels: list[Level] = []
        day_start_dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)

        prev_d = d - timedelta(days=1)
        if prev_d in by_date:
            r = by_date[prev_d]
            for family, side, price in (
                ("PREV_DAY_HIGH", "HIGH", float(r.high)),
                ("PREV_DAY_LOW", "LOW", float(r.low)),
            ):
                levels.append(Level(family, side, price, day_start_dt,
                                    level_id(family, price, day_start_dt)))

        ws = iso_week_start(d)
        prev_ws = ws - timedelta(days=7)
        if prev_ws in week_map:
            r = week_map[prev_ws]
            available = datetime(ws.year, ws.month, ws.day, tzinfo=timezone.utc)
            for family, side, price in (
                ("PREV_WEEK_HIGH", "HIGH", float(r.high)),
                ("PREV_WEEK_LOW", "LOW", float(r.low)),
            ):
                levels.append(Level(family, side, price, available,
                                    level_id(family, price, available)))

        pm_start, _ = prev_month_bounds(d)
        pm_key = f"{pm_start.year:04d}-{pm_start.month:02d}"
        if pm_key in month_map:
            r = month_map[pm_key]
            month_start = d.replace(day=1)
            available = datetime(month_start.year, month_start.month, 1, tzinfo=timezone.utc)
            for family, side, price in (
                ("PREV_MONTH_HIGH", "HIGH", float(r.high)),
                ("PREV_MONTH_LOW", "LOW", float(r.low)),
            ):
                levels.append(Level(family, side, price, available,
                                    level_id(family, price, available)))

        rr = roll_map.get(d)
        if rr is not None:
            for family, side, value in (
                ("ROLLING_7D_HIGH", "HIGH", rr.roll7_high),
                ("ROLLING_7D_LOW", "LOW", rr.roll7_low),
                ("ROLLING_30D_HIGH", "HIGH", rr.roll30_high),
                ("ROLLING_30D_LOW", "LOW", rr.roll30_low),
            ):
                if value is not None and np.isfinite(value):
                    price = float(value)
                    levels.append(Level(family, side, price, day_start_dt,
                                        level_id(family, price, day_start_dt)))

            if d >= ath_valid_from and rr.ath_prior is not None and np.isfinite(rr.ath_prior) and rr.ath_prior_origin is not None:
                price = float(rr.ath_prior)
                origin = rr.ath_prior_origin
                available = datetime(origin.year, origin.month, origin.day, tzinfo=timezone.utc) + timedelta(days=1)
                levels.append(Level("ATH_TO_DATE", "HIGH", price, available,
                                    level_id("ATH_TO_DATE", price, available)))

        # Round-number families are static, but only levels inside a causal +/-50% day-start
        # active view are materialized for efficiency. The grid itself is deterministic.
        prev_close = float(by_date[prev_d].close) if prev_d in by_date else float(by_date[d].open)
        low_active = max(1.0, prev_close * (1.0 - ROUND_ACTIVE_WINDOW_PCT))
        high_active = min(round_grid_max, prev_close * (1.0 + ROUND_ACTIVE_WINDOW_PCT))
        for step, family in ((1000.0, "ROUND_1K"), (5000.0, "ROUND_5K")):
            first = max(step, math.ceil(low_active / step) * step)
            p = first
            while p <= high_active + 1e-9:
                levels.append(Level(family, "ROUND", float(p), None,
                                    level_id(family, float(p), None)))
                p += step

        result[d] = levels
        d += timedelta(days=1)
    return result


def trailing_exposure_counter(d: date, daily_counts: dict[date, Counter], days: int = MATCH_EXPOSURE_DAYS) -> Counter:
    c = Counter()
    for k in range(1, days + 1):
        c.update(daily_counts.get(d - timedelta(days=k), Counter()))
    return c


def is_collision(price: float, real_prices: Sequence[float], collision_bps: float = CONTROL_COLLISION_BPS,
                 ignore_price: float | None = None) -> bool:
    for p in real_prices:
        if ignore_price is not None and abs(bps_log_ratio(p, ignore_price)) < 1e-9:
            continue
        if abs(bps_log_ratio(price, p)) <= collision_bps:
            return True
    return False


def matched_random_price(d: date, source: Level, day_start_price: float, exposure: Counter,
                         real_prices: Sequence[float], reserved_bins: set[int] | None = None) -> tuple[float | None, int | None, int | None]:
    if day_start_price <= 0 or source.price <= 0:
        return None, None, None
    source_bin = exposure_bin_idx(source.price)
    source_exp = int(exposure.get(source_bin, 0))
    target_signed = bps_log_ratio(source.price, day_start_price)
    target_abs = abs(target_signed)
    lo_price = day_start_price * (1.0 - MATCH_ACTIVE_WINDOW_PCT)
    hi_price = day_start_price * (1.0 + MATCH_ACTIVE_WINDOW_PCT)
    lo_idx = exposure_bin_idx(max(1.0, lo_price))
    hi_idx = exposure_bin_idx(hi_price)
    candidates: list[tuple[float, int, int]] = []
    reserved_bins = reserved_bins or set()
    for idx in range(lo_idx, hi_idx + 1):
        if idx in reserved_bins:
            continue
        price = exposure_bin_center(idx)
        if price <= 0:
            continue
        signed = bps_log_ratio(price, day_start_price)
        if target_abs >= 50 and signed * target_signed <= 0:
            continue
        if abs(bps_log_ratio(price, source.price)) <= max(2 * CONTROL_COLLISION_BPS, 100.0):
            continue
        if is_collision(price, real_prices):
            continue
        dist_abs = abs(signed)
        dist_term = abs(dist_abs - target_abs) / max(100.0, target_abs)
        exp_count = int(exposure.get(idx, 0))
        exp_term = abs(math.log1p(exp_count) - math.log1p(source_exp)) / max(1.0, math.log1p(source_exp + 1))
        score = dist_term + exp_term
        if dist_term <= 1.0:  # Prevent absurd geometry matches.
            candidates.append((score, idx, exp_count))
    if not candidates:
        return None, source_exp, None
    candidates.sort(key=lambda x: (x[0], x[1]))
    top = candidates[: min(10, len(candidates))]
    seed = RNG_SEED ^ stable_hash(f"{d.isoformat()}|{source.level_id}|MATCHED_RANDOM")
    pick = top[seed % len(top)]
    return exposure_bin_center(pick[1]), source_exp, int(pick[2])


def make_controls_for_day(d: date, real_levels: list[Level], levels_by_day: dict[date, list[Level]],
                          day_start_price: float, exposure: Counter) -> list[Level]:
    controls: list[Level] = []
    real_prices = [x.price for x in real_levels]
    # Multiple real levels from the same family (notably ROUND_1K/ROUND_5K) can
    # otherwise select the same matched-random exposure bin. Reserve each selected
    # physical control location within family/day so it can generate at most one
    # control interaction instead of pseudo-replicating the same price event.
    matched_reserved_bins: dict[str, set[int]] = defaultdict(set)
    old_maps: dict[int, dict[str, Level]] = {}
    for lag in WRONG_DAY_LAGS:
        old_maps[lag] = {x.family: x for x in levels_by_day.get(d - timedelta(days=lag), [])
                         if x.family in TIME_DERIVED_FAMILIES}

    for src in real_levels:
        src_exp = int(exposure.get(exposure_bin_idx(src.price), 0))
        for offset in SHIFT_OFFSETS_BPS:
            p = exp_bps(src.price, offset)
            if p <= 0 or is_collision(p, real_prices):
                continue
            available = src.available_time
            suffix = f"SHIFT_{offset:+.0f}BPS"
            controls.append(Level(
                src.family, src.side, p, available,
                level_id(src.family, p, available, suffix),
                kind="CONTROL", control_type="SHIFTED", control_variant=suffix,
                source_level_id=src.level_id, source_level_price=src.price,
                recent_exposure_30d=int(exposure.get(exposure_bin_idx(p), 0)),
                source_recent_exposure_30d=src_exp,
            ))

        if src.family in TIME_DERIVED_FAMILIES:
            for lag in WRONG_DAY_LAGS:
                old = old_maps[lag].get(src.family)
                if old is None:
                    continue
                p = old.price
                if p <= 0 or is_collision(p, real_prices):
                    continue
                suffix = f"T_MINUS_{lag}D"
                controls.append(Level(
                    src.family, src.side, p, old.available_time,
                    level_id(src.family, p, old.available_time, suffix),
                    kind="CONTROL", control_type="WRONG_DAY", control_variant=suffix,
                    source_level_id=src.level_id, source_level_price=src.price,
                    recent_exposure_30d=int(exposure.get(exposure_bin_idx(p), 0)),
                    source_recent_exposure_30d=src_exp,
                ))

        reserved = matched_reserved_bins[src.family]
        p, source_exp, match_exp = matched_random_price(
            d, src, day_start_price, exposure, real_prices, reserved_bins=reserved
        )
        if p is not None:
            reserved.add(exposure_bin_idx(p))
            suffix = "MATCHED_RANDOM"
            controls.append(Level(
                src.family, src.side, p, datetime(d.year, d.month, d.day, tzinfo=timezone.utc),
                level_id(src.family, p, datetime(d.year, d.month, d.day, tzinfo=timezone.utc), suffix),
                kind="CONTROL", control_type="MATCHED_RANDOM", control_variant=suffix,
                source_level_id=src.level_id, source_level_price=src.price,
                recent_exposure_30d=match_exp,
                source_recent_exposure_30d=source_exp,
            ))

    return controls


def tau_array(spec: dict[str, Any], sigma_bps):
    import numpy as np
    if spec["kind"] == "FIXED":
        return np.full(len(sigma_bps), float(spec["value"]), dtype=float)
    arr = sigma_bps * float(spec["value"])
    return np.clip(arr, float(spec["min_bps"]), float(spec["max_bps"]))


def range_distance_bps(high, low, levels):
    """Return minute-bar distance from each point level in bps, shape [minutes, levels]."""
    import numpy as np
    H = high[:, None]
    L = low[:, None]
    P = levels[None, :]
    below = P < L
    above = P > H
    out = np.zeros((len(high), len(levels)), dtype=float)
    out[below] = np.log((L / P)[below]) * 10_000.0
    out[above] = np.log((P / H)[above]) * 10_000.0
    return out


def first_approach_indices(close, high, low, level_prices, tau, prefix_rows: int = 0, allow_prefix=None):
    """Vectorized first event index per level for one UTC day.

    A level becomes armed after 60 consecutive closes >= 3*tau away. Once armed, the
    first bar whose range is <= tau from the point level fires. Returns -1 for no event.
    """
    import numpy as np
    n = len(close)
    m = len(level_prices)
    if n == 0 or m == 0:
        return np.full(m, -1, dtype=int)
    logdist = np.abs(np.log(close[:, None] / level_prices[None, :])) * 10_000.0
    t = tau[:, None]
    finite = np.isfinite(t) & np.isfinite(logdist)
    far = finite & (logdist >= ARM_DISTANCE_MULTIPLIER * t)
    near = finite & (range_distance_bps(high, low, level_prices) <= t)
    if prefix_rows > 0:
        # Prefix bars can arm only levels that were already known before current day-start.
        # No event may fire in the prefix itself.
        near[:prefix_rows, :] = False
        if allow_prefix is None:
            allow_prefix = np.ones(m, dtype=bool)
        allow_prefix = np.asarray(allow_prefix, dtype=bool)
        if len(allow_prefix) != m:
            raise ValueError("allow_prefix length must match level_prices")
        blocked = ~allow_prefix
        if np.any(blocked):
            far[:prefix_rows, blocked] = False

    run60 = np.zeros_like(far, dtype=bool)
    if n >= ARM_CONSECUTIVE_MINUTES:
        cs = np.cumsum(far.astype(np.int16), axis=0)
        sums = cs[ARM_CONSECUTIVE_MINUTES - 1 :].copy()
        if ARM_CONSECUTIVE_MINUTES < n:
            sums[1:] -= cs[:-ARM_CONSECUTIVE_MINUTES]
        run60[ARM_CONSECUTIVE_MINUTES - 1 :] = sums == ARM_CONSECUTIVE_MINUTES
    armed_seen = np.maximum.accumulate(run60, axis=0)
    # The firing bar itself cannot be what completes the away run.
    armed_before = np.vstack([np.zeros((1, m), dtype=bool), armed_seen[:-1]])
    trigger = near & armed_before
    any_trigger = trigger.any(axis=0)
    first = np.argmax(trigger, axis=0)
    return np.where(any_trigger, first, -1).astype(int)


def outcome_for_event(global_idx: int, level_price: float, approach_from: str, sigma_bps: float,
                      tau_bps_value: float, high_all, low_all, close_all) -> dict[str, Any]:
    import numpy as np
    result: dict[str, Any] = {}

    def horizon(h: int) -> tuple[float, float, bool, np.ndarray]:
        end = min(len(close_all), global_idx + h)
        complete = end - global_idx >= h
        if end <= global_idx:
            return math.nan, math.nan, False, np.asarray([], dtype=float)
        hs = high_all[global_idx:end]
        ls = low_all[global_idx:end]
        cs = close_all[global_idx:end]
        if approach_from == "BELOW":
            pen = max(0.0, bps_log_ratio(float(np.nanmax(hs)), level_price))
            rej = max(0.0, bps_log_ratio(level_price, float(np.nanmin(ls))))
        else:
            pen = max(0.0, bps_log_ratio(level_price, float(np.nanmin(ls))))
            rej = max(0.0, bps_log_ratio(float(np.nanmax(hs)), level_price))
        return pen, rej, complete, cs

    windows = {}
    for h in (15, 60, 240):
        pen, rej, complete, cs = horizon(h)
        windows[h] = (pen, rej, complete, cs)
        result[f"penetration_bps_{h}m"] = pen
        result[f"penetration_vol_{h}m"] = pen / sigma_bps if sigma_bps > 0 and math.isfinite(pen) else math.nan
        result[f"rejection_bps_{h}m"] = rej
        result[f"rejection_vol_{h}m"] = rej / sigma_bps if sigma_bps > 0 and math.isfinite(rej) else math.nan
        result[f"outcome_complete_{h}m"] = bool(complete)

    cs60 = windows[60][3]
    if len(cs60):
        if approach_from == "BELOW":
            beyond = cs60 > level_price
            final_side = "BEYOND" if cs60[-1] > level_price else "ORIGIN"
        else:
            beyond = cs60 < level_price
            final_side = "BEYOND" if cs60[-1] < level_price else "ORIGIN"
        near = np.abs(np.log(cs60 / level_price)) * 10_000.0 <= tau_bps_value
        result["beyond_close_fraction_60m"] = float(np.mean(beyond))
        result["near_level_minutes_60m"] = int(np.sum(near))
        result["final_side_60m"] = final_side
    else:
        result["beyond_close_fraction_60m"] = math.nan
        result["near_level_minutes_60m"] = 0
        result["final_side_60m"] = "UNKNOWN"

    penv = result["penetration_vol_60m"]
    rejv = result["rejection_vol_60m"]
    frac = result["beyond_close_fraction_60m"]
    if not all(math.isfinite(x) for x in (penv, rejv, frac)):
        resolution = "INCOMPLETE"
    elif penv < 0.15 and rejv >= 0.25 and frac <= 0.25:
        resolution = "REJECTION"
    elif penv >= 0.25 and frac >= 0.50:
        resolution = "CLEAN_BREAK"
    elif penv >= 0.15 and result["final_side_60m"] == "ORIGIN":
        resolution = "PENETRATE_RETURN"
    else:
        resolution = "STALL_OTHER"
    result["exploratory_resolution_60m"] = resolution

    # Exploratory retest: after moving >=0.25 sigma beyond the level, does close return
    # within tau during the remainder of the next 240 minutes?
    cs240 = windows[240][3]
    retest = False
    if len(cs240) and sigma_bps > 0:
        threshold_price = exp_bps(level_price, 0.25 * sigma_bps if approach_from == "BELOW" else -0.25 * sigma_bps)
        if approach_from == "BELOW":
            beyond_idx = np.where(cs240 >= threshold_price)[0]
        else:
            beyond_idx = np.where(cs240 <= threshold_price)[0]
        if len(beyond_idx):
            j = int(beyond_idx[0]) + 1
            if j < len(cs240):
                dist = np.abs(np.log(cs240[j:] / level_price)) * 10_000.0
                retest = bool(np.any(dist <= tau_bps_value))
    result["retest_within_240m"] = retest
    return result


def confluence_for(level: Level, real_levels: Sequence[Level], tau_bps_value: float) -> tuple[int, str]:
    threshold = max(CONFLUENCE_FLOOR_BPS, tau_bps_value)
    fams = sorted({x.family for x in real_levels if abs(bps_log_ratio(x.price, level.price)) <= threshold})
    return len(fams), ";".join(fams)


def process_year(year: int, df, sigma_all, day_index: dict[date, tuple[int, int]],
                 real_levels_by_day: dict[date, list[Level]], daily_exposure: dict[date, Counter],
                 tolerance_ids: Sequence[str], analysis_start: date, analysis_end: date,
                 run_id: str) -> tuple[Any, list[dict[str, Any]]]:
    import numpy as np
    import pandas as pd

    opens_all = df["open"].to_numpy(dtype=float)
    highs_all = df["high"].to_numpy(dtype=float)
    lows_all = df["low"].to_numpy(dtype=float)
    closes_all = df["close"].to_numpy(dtype=float)
    times_all = df["open_time"].to_numpy()

    rows: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    start = max(analysis_start, date(year, 1, 1))
    end = min(analysis_end, date(year, 12, 31))
    d = start
    processed_days = 0
    while d <= end:
        if d not in day_index or d not in real_levels_by_day:
            d += timedelta(days=1)
            continue
        s, e = day_index[d]
        if s <= 0:
            d += timedelta(days=1)
            continue
        day_start_price = float(closes_all[s - 1])
        exposure = trailing_exposure_counter(d, daily_exposure)
        real_levels0 = real_levels_by_day[d]
        controls = make_controls_for_day(d, real_levels0, real_levels_by_day, day_start_price, exposure)
        # Attach exposure to real levels without mutating frozen dataclasses.
        real_levels = [Level(
            x.family, x.side, x.price, x.available_time, x.level_id,
            kind="REAL", control_type="REAL", control_variant="REAL", source_level_id=x.level_id, source_level_price=x.price,
            recent_exposure_30d=int(exposure.get(exposure_bin_idx(x.price), 0)),
            source_recent_exposure_30d=int(exposure.get(exposure_bin_idx(x.price), 0)),
        ) for x in real_levels0]
        candidates = real_levels + controls
        if not candidates:
            d += timedelta(days=1)
            continue

        # Inventory is independent of tolerance; used for approach-hit-rate diagnostics.
        inv_count = Counter((x.family, x.control_type, x.control_variant) for x in candidates)
        for (family, ctype, variant), n in sorted(inv_count.items()):
            inventory.append({
                "year": year, "date": d.isoformat(), "level_family": family,
                "control_type": ctype, "control_variant": variant, "candidate_levels": int(n),
            })

        prefix_s = max(0, s - ARM_CONSECUTIVE_MINUTES)
        prefix_rows = s - prefix_s
        close = closes_all[prefix_s:e]
        high = highs_all[prefix_s:e]
        low = lows_all[prefix_s:e]
        level_prices = np.asarray([x.price for x in candidates], dtype=float)
        sig = sigma_all[prefix_s:e]
        day_start_dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        allow_prefix = np.asarray([x.available_time is None or x.available_time < day_start_dt for x in candidates], dtype=bool)

        for tol_id in tolerance_ids:
            spec = TOLERANCE_SPECS[tol_id]
            tau = tau_array(spec, sig)
            first = first_approach_indices(close, high, low, level_prices, tau, prefix_rows=prefix_rows, allow_prefix=allow_prefix)
            for j, local_idx in enumerate(first):
                if local_idx < 0:
                    continue
                g = prefix_s + int(local_idx)
                if g < s or g >= e:
                    continue
                sigma = float(sigma_all[g])
                tau_v = float(tau[int(local_idx)])
                if not (math.isfinite(sigma) and sigma > 0 and math.isfinite(tau_v) and tau_v > 0):
                    continue
                lev = candidates[j]
                prev_idx = max(s, g - 1)
                approach_from = "BELOW" if closes_all[prev_idx] < lev.price else "ABOVE"
                if g >= 16:
                    ar15 = bps_log_ratio(float(closes_all[g - 1]), float(closes_all[g - 16]))
                else:
                    ar15 = math.nan
                toward = ar15 if approach_from == "BELOW" else -ar15 if math.isfinite(ar15) else math.nan
                conf_count, conf_fams = confluence_for(lev, real_levels, tau_v)
                available = lev.available_time
                event_dt = pd.Timestamp(df["open_time"].iloc[g]).to_pydatetime()
                age_h = ((event_dt - available).total_seconds() / 3600.0) if available else math.nan
                out = outcome_for_event(g, lev.price, approach_from, sigma, tau_v,
                                        highs_all, lows_all, closes_all)
                eid_text = f"{lev.level_id}|{event_dt.isoformat()}|{tol_id}|{lev.control_type}|{lev.control_variant}"
                row = {
                    "run_id": run_id,
                    "schema_version": SCHEMA_VERSION,
                    "symbol": DEFAULT_SYMBOL,
                    "event_id": hashlib.sha256(eid_text.encode()).hexdigest()[:24],
                    "event_time": event_dt,
                    "event_date": d.isoformat(),
                    "event_year": year,
                    "tolerance_id": tol_id,
                    "tolerance_primary": bool(spec.get("primary", False)),
                    "tolerance_kind": spec["kind"],
                    "tolerance_parameter": float(spec["value"]),
                    "tau_bps": tau_v,
                    "level_family": lev.family,
                    "level_side": lev.side,
                    "level_kind": lev.kind,
                    "control_type": lev.control_type,
                    "control_variant": lev.control_variant,
                    "level_id": lev.level_id,
                    "source_level_id": lev.source_level_id or lev.level_id,
                    "level_price": lev.price,
                    "source_level_price": lev.source_level_price if lev.source_level_price is not None else lev.price,
                    "level_available_time": available,
                    "level_age_hours": age_h,
                    "approach_from": approach_from,
                    "event_close": float(closes_all[g]),
                    "event_open": float(opens_all[g]),
                    "event_high": float(highs_all[g]),
                    "event_low": float(lows_all[g]),
                    "trailing_sigma_bps": sigma,
                    "approach_return_15m_bps": ar15,
                    "approach_velocity_toward_bps_15m": toward,
                    "day_start_price": day_start_price,
                    "signed_distance_from_day_start_bps": bps_log_ratio(lev.price, day_start_price),
                    "abs_distance_from_day_start_bps": abs(bps_log_ratio(lev.price, day_start_price)),
                    "source_abs_distance_from_day_start_bps": abs(bps_log_ratio(lev.source_level_price if lev.source_level_price is not None else lev.price, day_start_price)),
                    "recent_exposure_30d": lev.recent_exposure_30d,
                    "source_recent_exposure_30d": lev.source_recent_exposure_30d,
                    "confluence_count": conf_count,
                    "confluence_families": conf_fams,
                    "is_confluence": conf_count >= 2,
                }
                row.update(out)
                rows.append(row)

        processed_days += 1
        if processed_days % 60 == 0:
            print(f"  {year}: processed {processed_days} days, events so far {len(rows):,}")
        d += timedelta(days=1)

    events = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    return events, inventory


def one_way_cluster_cov(X, residuals, groups):
    import numpy as np
    X = np.asarray(X, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    groups = np.asarray(groups)
    n, k = X.shape
    uniq = np.unique(groups)
    if len(uniq) < 2:
        return np.zeros((k, k), dtype=float)
    meat = np.zeros((k, k), dtype=float)
    for g in uniq:
        mask = groups == g
        score = X[mask].T @ residuals[mask]
        meat += np.outer(score, score)
    bread = np.linalg.pinv(X.T @ X)
    correction = (len(uniq) / (len(uniq) - 1.0)) * ((n - 1.0) / max(1.0, n - k))
    return correction * (bread @ meat @ bread)


def two_way_cluster_effect(real, control, endpoint: str) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    a = real[[endpoint, "event_date", "source_level_id"]].copy()
    a["is_real"] = 1.0
    b = control[[endpoint, "event_date", "source_level_id"]].copy()
    b["is_real"] = 0.0
    z = pd.concat([a, b], ignore_index=True)
    z[endpoint] = pd.to_numeric(z[endpoint], errors="coerce")
    z = z[np.isfinite(z[endpoint].to_numpy(dtype=float))].copy()
    nr = int((z["is_real"] == 1).sum())
    nc = int((z["is_real"] == 0).sum())
    if nr < 2 or nc < 2:
        return {"n_real": nr, "n_control": nc, "status": "INSUFFICIENT"}

    y = z[endpoint].to_numpy(dtype=float)
    x1 = z["is_real"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(z)), x1])
    beta = np.linalg.pinv(X.T @ X) @ (X.T @ y)
    resid = y - X @ beta
    day = z["event_date"].astype(str).to_numpy()
    level = z["source_level_id"].astype(str).to_numpy()
    inter = np.asarray([f"{d}|{l}" for d, l in zip(day, level)], dtype=object)
    V = one_way_cluster_cov(X, resid, day) + one_way_cluster_cov(X, resid, level) - one_way_cluster_cov(X, resid, inter)
    var = max(0.0, float(V[1, 1]))
    se = math.sqrt(var)
    effect = float(beta[1])
    zstat = effect / se if se > 0 else math.nan
    p = math.erfc(abs(zstat) / math.sqrt(2.0)) if math.isfinite(zstat) else math.nan

    yr = a[endpoint].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    yc = b[endpoint].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    mean_r = float(np.mean(yr)) if len(yr) else math.nan
    mean_c = float(np.mean(yc)) if len(yc) else math.nan
    sd_r = float(np.std(yr, ddof=1)) if len(yr) > 1 else math.nan
    sd_c = float(np.std(yc, ddof=1)) if len(yc) > 1 else math.nan
    pooled_var_num = max(0, (len(yr) - 1)) * (sd_r ** 2 if math.isfinite(sd_r) else 0) + max(0, (len(yc) - 1)) * (sd_c ** 2 if math.isfinite(sd_c) else 0)
    pooled_den = max(1, len(yr) + len(yc) - 2)
    pooled_sd = math.sqrt(pooled_var_num / pooled_den) if pooled_var_num > 0 else math.nan
    effect_sd = effect / pooled_sd if math.isfinite(pooled_sd) and pooled_sd > 0 else math.nan
    cluster_mde = Z_MDE * se
    cluster_mde_sd = cluster_mde / pooled_sd if math.isfinite(pooled_sd) and pooled_sd > 0 else math.nan
    independent_mde_sd = Z_MDE * math.sqrt(1.0 / nr + 1.0 / nc)
    conservative_mde_sd = independent_mde_sd * math.sqrt(2.0)
    return {
        "status": "OK",
        "n_real": nr,
        "n_control": nc,
        "mean_real": mean_r,
        "mean_control": mean_c,
        "effect_real_minus_control": effect,
        "effect_sd": effect_sd,
        "cluster_se": se,
        "ci95_low": effect - Z_ALPHA_2 * se,
        "ci95_high": effect + Z_ALPHA_2 * se,
        "z": zstat,
        "p_normal": p,
        "clusters_day": int(len(np.unique(day))),
        "clusters_level": int(len(np.unique(level))),
        "mde_80pct_cluster_units": cluster_mde,
        "mde_80pct_cluster_sd": cluster_mde_sd,
        "mde_80pct_independent_sd": independent_mde_sd,
        "mde_80pct_conservative_2x_sd": conservative_mde_sd,
        "planned_target_sd": PLANNED_MDE_TARGET_SD,
        "target_met_independent": independent_mde_sd <= PLANNED_MDE_TARGET_SD,
        "target_met_conservative_2x": conservative_mde_sd <= PLANNED_MDE_TARGET_SD,
    }


def build_effect_tables(events):
    import pandas as pd
    rows_pooled: list[dict[str, Any]] = []
    rows_yearly: list[dict[str, Any]] = []
    rows_variants: list[dict[str, Any]] = []

    families = sorted(events["level_family"].dropna().unique())
    tolerances = sorted(events["tolerance_id"].dropna().unique())
    for tol in tolerances:
        et = events[events["tolerance_id"] == tol]
        for family in families:
            ef = et[et["level_family"] == family]
            real = ef[ef["control_type"] == "REAL"]
            if real.empty:
                continue
            for ctype in ("SHIFTED", "WRONG_DAY", "MATCHED_RANDOM"):
                ctrl = ef[ef["control_type"] == ctype]
                if ctrl.empty:
                    continue
                stat = two_way_cluster_effect(real, ctrl, PRIMARY_ENDPOINT)
                rows_pooled.append({
                    "scope": "POOLED", "year": "ALL", "tolerance_id": tol,
                    "tolerance_primary": tol == PRIMARY_TOLERANCE_ID,
                    "level_family": family, "control_type": ctype, "control_variant": "POOLED",
                    "endpoint": PRIMARY_ENDPOINT, **stat,
                })
                for variant in sorted(ctrl["control_variant"].dropna().unique()):
                    cv = ctrl[ctrl["control_variant"] == variant]
                    sv = two_way_cluster_effect(real, cv, PRIMARY_ENDPOINT)
                    rows_variants.append({
                        "scope": "VARIANT", "year": "ALL", "tolerance_id": tol,
                        "tolerance_primary": tol == PRIMARY_TOLERANCE_ID,
                        "level_family": family, "control_type": ctype, "control_variant": variant,
                        "endpoint": PRIMARY_ENDPOINT, **sv,
                    })
                for year in sorted(ef["event_year"].dropna().unique()):
                    ry = real[real["event_year"] == year]
                    cy = ctrl[ctrl["event_year"] == year]
                    if ry.empty or cy.empty:
                        continue
                    sy = two_way_cluster_effect(ry, cy, PRIMARY_ENDPOINT)
                    rows_yearly.append({
                        "scope": "YEAR", "year": int(year), "tolerance_id": tol,
                        "tolerance_primary": tol == PRIMARY_TOLERANCE_ID,
                        "level_family": family, "control_type": ctype, "control_variant": "POOLED",
                        "endpoint": PRIMARY_ENDPOINT, **sy,
                    })
    return pd.DataFrame(rows_pooled), pd.DataFrame(rows_yearly), pd.DataFrame(rows_variants)


def secondary_summary(events):
    import numpy as np
    import pandas as pd
    metrics = [
        "penetration_vol_15m", "penetration_vol_60m", "penetration_vol_240m",
        "rejection_vol_15m", "rejection_vol_60m", "rejection_vol_240m",
        "beyond_close_fraction_60m", "near_level_minutes_60m",
    ]
    rows = []
    for keys, g in events.groupby(["tolerance_id", "level_family", "control_type"], dropna=False):
        tol, family, ctype = keys
        row = {
            "tolerance_id": tol, "tolerance_primary": tol == PRIMARY_TOLERANCE_ID,
            "level_family": family, "control_type": ctype, "events": len(g),
        }
        for metric in metrics:
            vals = pd.to_numeric(g[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else math.nan
            row[f"{metric}_median"] = float(vals.median()) if len(vals) else math.nan
        for label in ("REJECTION", "CLEAN_BREAK", "PENETRATE_RETURN", "STALL_OTHER"):
            row[f"resolution_{label.lower()}_rate"] = float((g["exploratory_resolution_60m"] == label).mean()) if len(g) else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_event_count_table(events, inventory_df):
    import pandas as pd
    if inventory_df.empty:
        return pd.DataFrame()
    inv = inventory_df.groupby(["year", "level_family", "control_type", "control_variant"], as_index=False)["candidate_levels"].sum()
    ev = events.groupby(["event_year", "level_family", "control_type", "control_variant", "tolerance_id"], as_index=False).size().rename(columns={"event_year": "year", "size": "events"})
    out = ev.merge(inv, on=["year", "level_family", "control_type", "control_variant"], how="left")
    out["event_rate_per_candidate_day"] = out["events"] / out["candidate_levels"].clip(lower=1)
    return out


def control_match_quality(events):
    import numpy as np
    import pandas as pd
    c = events[(events["control_type"] == "MATCHED_RANDOM") & (events["tolerance_id"] == PRIMARY_TOLERANCE_ID)].copy()
    if c.empty:
        return pd.DataFrame()
    c["exposure_log_abs_diff"] = np.abs(np.log1p(c["recent_exposure_30d"].astype(float)) - np.log1p(c["source_recent_exposure_30d"].astype(float)))
    c["distance_abs_diff_bps"] = np.abs(c["abs_distance_from_day_start_bps"].astype(float) - c["source_abs_distance_from_day_start_bps"].astype(float))
    rows = []
    for family, g in c.groupby("level_family"):
        rows.append({
            "level_family": family,
            "events": len(g),
            "median_control_exposure": float(g["recent_exposure_30d"].median()),
            "median_source_exposure": float(g["source_recent_exposure_30d"].median()),
            "median_log_exposure_abs_diff": float(g["exposure_log_abs_diff"].median()),
            "median_abs_distance_from_day_start_bps": float(g["abs_distance_from_day_start_bps"].median()),
            "median_source_abs_distance_from_day_start_bps": float(g["source_abs_distance_from_day_start_bps"].median()),
            "median_distance_abs_diff_bps": float(g["distance_abs_diff_bps"].median()),
        })
    return pd.DataFrame(rows)


def invariants(events, analysis_start: date, analysis_end: date) -> list[Check]:
    import numpy as np
    checks: list[Check] = []
    checks.append(Check("events", "unique_event_id", "PASS" if events["event_id"].is_unique else "FAIL",
                        int(events["event_id"].nunique()), int(len(events))))
    t = events["event_time"]
    dates_ok = bool(((t.dt.date >= analysis_start) & (t.dt.date <= analysis_end)).all()) if len(events) else True
    checks.append(Check("events", "event_date_within_analysis_window", "PASS" if dates_ok else "FAIL", dates_ok, True))
    avail = events["level_available_time"]
    mask = avail.notna()
    causal = bool((avail[mask] <= t[mask]).all()) if mask.any() else True
    checks.append(Check("causality", "level_available_no_later_than_event", "PASS" if causal else "FAIL", causal, True))
    sigma_ok = bool((events["trailing_sigma_bps"].dropna() > 0).all()) if len(events) else True
    checks.append(Check("causality", "positive_trailing_sigma", "PASS" if sigma_ok else "FAIL", sigma_ok, True))
    tau_ok = bool((events["tau_bps"].dropna() > 0).all()) if len(events) else True
    checks.append(Check("events", "positive_tolerance", "PASS" if tau_ok else "FAIL", tau_ok, True))
    primary_exists = bool((events["tolerance_id"] == PRIMARY_TOLERANCE_ID).any())
    checks.append(Check("contract", "primary_tolerance_present", "PASS" if primary_exists else "FAIL", primary_exists, True))
    finite_primary = events.loc[events["tolerance_id"] == PRIMARY_TOLERANCE_ID, PRIMARY_ENDPOINT].replace([np.inf, -np.inf], np.nan).notna().mean() if len(events) else 0
    checks.append(Check("outcomes", "primary_endpoint_completeness", "PASS" if finite_primary >= 0.99 else "WARN",
                        float(finite_primary), ">=0.99"))
    cap = events.groupby(["event_date", "tolerance_id", "level_id", "control_type", "control_variant"]).size().max() if len(events) else 0
    checks.append(Check("events", "one_event_per_level_day_tolerance", "PASS" if cap <= MAX_EVENTS_PER_LEVEL_DAY else "FAIL", int(cap), MAX_EVENTS_PER_LEVEL_DAY))
    return checks


def write_findings(path: Path, summary: dict[str, Any], pooled, yearly, checks) -> None:
    import pandas as pd
    lines = [
        "# Deterministic price-level reaction benchmark", "",
        f"- Auditor version: `{EVALUATOR_VERSION}`",
        f"- Run ID: `{summary['run_id']}`",
        f"- Status: **{summary['status']}**",
        f"- Analysis window: `{summary['analysis_start']}` through `{summary['analysis_end']}`",
        f"- Events: **{summary['event_rows']:,}**",
        f"- Primary tolerance: `{PRIMARY_TOLERANCE_ID}`",
        f"- Primary endpoint: `{PRIMARY_ENDPOINT}`",
        f"- Planned MDE target: **{PLANNED_MDE_TARGET_SD:.2f} SD** at alpha={ALPHA}, power={POWER}", "",
        "## Interpretation contract", "",
        "`effect_real_minus_control < 0` means the real deterministic level was harder to penetrate than its control.",
        "The primary endpoint/tolerance were fixed in code before the first empirical run. Alternative tolerances and resolution labels are sensitivity/exploratory outputs.", "",
        "## Primary pooled effects", "",
    ]
    primary = pooled[pooled["tolerance_id"] == PRIMARY_TOLERANCE_ID].copy() if len(pooled) else pd.DataFrame()
    if primary.empty:
        lines.append("No primary pooled comparisons were available.")
    else:
        cols = ["level_family", "control_type", "n_real", "n_control", "effect_real_minus_control", "effect_sd", "ci95_low", "ci95_high", "mde_80pct_conservative_2x_sd"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for r in primary.sort_values(["level_family", "control_type"]).itertuples(index=False):
            values = []
            for c in cols:
                v = getattr(r, c, "")
                if isinstance(v, float):
                    values.append(f"{v:.4f}" if math.isfinite(v) else "")
                else:
                    values.append(str(v))
            lines.append("| " + " | ".join(values) + " |")
    lines += ["", "## Cross-year sign consistency", ""]
    if len(yearly):
        yp = yearly[yearly["tolerance_id"] == PRIMARY_TOLERANCE_ID].copy()
        for (family, ctype), g in yp.groupby(["level_family", "control_type"]):
            ok = g[g["status"] == "OK"]
            if len(ok):
                neg = int((ok["effect_real_minus_control"] < 0).sum())
                lines.append(f"- `{family}` vs `{ctype}`: negative effect in **{neg}/{len(ok)}** years.")
    else:
        lines.append("No yearly comparisons available.")
    lines += ["", "## Invariants", ""]
    for c in checks:
        lines.append(f"- **{c.status}** `{c.scope}/{c.check}` — observed `{c.observed}`, expected `{c.expected}`.")
    lines += ["", "## Guardrail", "", "This pass establishes a benchmark only. It does not validate statistical historical reaction levels and it does not use participant-flow features.", ""]
    atomic_write_text(path, "\n".join(lines))


def self_test() -> None:
    import numpy as np
    import pandas as pd

    # 1) Event grammar: 60 far minutes, gradual approach, touch/rejection.
    n = 180
    level = 100.0
    close = np.full(n, 96.0)
    close[70:100] = np.linspace(96.0, 99.95, 30)
    close[100:] = 98.5
    high = close + 0.02
    low = close - 0.02
    high[99] = 100.01
    tau = np.full(n, 20.0)
    first = first_approach_indices(close, high, low, np.asarray([level]), tau)
    assert first[0] == 98, first
    # Prefix arming: an old/static level may fire near minute 0 of a new day, while a
    # newly available day-start level may not borrow pre-day arming evidence.
    close2 = np.concatenate([np.full(60, 96.0), np.asarray([99.95, 98.0, 98.0])])
    high2 = close2 + 0.02; low2 = close2 - 0.02; high2[60] = 100.0
    tau2 = np.full(len(close2), 20.0)
    old_first = first_approach_indices(close2, high2, low2, np.asarray([100.0]), tau2, prefix_rows=60, allow_prefix=np.asarray([True]))
    new_first = first_approach_indices(close2, high2, low2, np.asarray([100.0]), tau2, prefix_rows=60, allow_prefix=np.asarray([False]))
    assert old_first[0] == 60 and new_first[0] == -1, (old_first, new_first)

    # 2) Outcome directionality.
    highs = high.copy(); lows = low.copy()
    out = outcome_for_event(98, level, "BELOW", 200.0, 20.0, highs, lows, close)
    assert out["penetration_bps_60m"] < 5.0
    assert out["rejection_bps_60m"] > 100.0

    # 3) Cluster-effect sign.
    real = pd.DataFrame({PRIMARY_ENDPOINT: [0.1, 0.2, 0.15, 0.2], "event_date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"], "source_level_id": ["a", "b", "c", "d"]})
    ctl = pd.DataFrame({PRIMARY_ENDPOINT: [0.5, 0.6, 0.55, 0.65], "event_date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"], "source_level_id": ["a", "b", "c", "d"]})
    stat = two_way_cluster_effect(real, ctl, PRIMARY_ENDPOINT)
    assert stat["effect_real_minus_control"] < 0

    # 4) Exposure-bin roundtrip approximate containment.
    p = 63_250.0
    idx = exposure_bin_idx(p)
    center = exposure_bin_center(idx)
    assert abs(bps_log_ratio(center, p)) <= EXPOSURE_BIN_BPS

    # 5) Matched-random control locations are reserved within family/day. This
    # prevents two round-number source levels from emitting the same physical
    # matched-control interaction and duplicating an event ID.
    d0 = date(2020, 2, 1)
    day_start = 10_000.0
    exposure = Counter({i: 100 for i in range(exposure_bin_idx(7_500.0), exposure_bin_idx(12_500.0) + 1)})
    src = Level("ROUND_1K", "ROUND", 9_000.0, None, level_id("ROUND_1K", 9_000.0, None))
    real_prices = [9_000.0, 10_000.0, 11_000.0]
    p1, _, _ = matched_random_price(d0, src, day_start, exposure, real_prices, reserved_bins=set())
    assert p1 is not None
    reserved = {exposure_bin_idx(p1)}
    p2, _, _ = matched_random_price(d0, src, day_start, exposure, real_prices, reserved_bins=reserved)
    assert p2 is not None and exposure_bin_idx(p2) != exposure_bin_idx(p1), (p1, p2)

    print("SELF-TEST PASS")
    print("Validated approach arming/trigger logic, penetration directionality, clustered effect sign, exposure-bin geometry, and matched-control reservation.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--historical-root", type=Path, default=DEFAULT_HISTORICAL_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=None)
    p.add_argument("--start-date", default=DEFAULT_ANALYSIS_START)
    p.add_argument("--end-date", default=DEFAULT_ANALYSIS_END)
    p.add_argument("--ath-valid-from", default=DEFAULT_ATH_VALID_FROM)
    p.add_argument("--round-grid-max", type=float, default=ROUND_GRID_MAX_DEFAULT)
    p.add_argument("--tolerances", default="all", help="all, primary, or comma-separated tolerance IDs")
    p.add_argument("--force", action="store_true", help="Recompute yearly event parts even if matching state exists")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0

    require_runtime_libraries()
    import numpy as np
    import pandas as pd

    script_path = Path(__file__)
    repo_root = find_repo_root(script_path)
    report_root = args.report_root or (repo_root / "reports" / "binance_um_deterministic_level_reactions")
    report_root.mkdir(parents=True, exist_ok=True)
    output_root: Path = args.output_root
    metadata_root = output_root / "metadata"
    events_root = output_root / "events"
    metadata_root.mkdir(parents=True, exist_ok=True)
    events_root.mkdir(parents=True, exist_ok=True)

    analysis_start = parse_date(args.start_date)
    analysis_end = parse_date(args.end_date)
    ath_valid_from = parse_date(args.ath_valid_from)
    if analysis_end < analysis_start:
        raise RuntimeError("end-date precedes start-date")

    if args.tolerances == "all":
        tolerance_ids = list(TOLERANCE_SPECS)
    elif args.tolerances == "primary":
        tolerance_ids = [PRIMARY_TOLERANCE_ID]
    else:
        tolerance_ids = [x.strip() for x in args.tolerances.split(",") if x.strip()]
        unknown = [x for x in tolerance_ids if x not in TOLERANCE_SPECS]
        if unknown:
            raise RuntimeError(f"Unknown tolerance IDs: {unknown}")
        if PRIMARY_TOLERANCE_ID not in tolerance_ids:
            print(f"WARNING: requested tolerances do not include primary {PRIMARY_TOLERANCE_ID}")

    # Load one warm-up month before analysis and one day after end for 240m outcomes.
    load_start = analysis_start - timedelta(days=31)
    load_end = analysis_end + timedelta(days=1)
    audit_summary = verify_source_audit(args.historical_root)
    files = discover_parquet_files(args.historical_root, load_start, load_end)
    config = {
        "auditor_version": EVALUATOR_VERSION,
        "schema_version": SCHEMA_VERSION,
        "symbol": DEFAULT_SYMBOL,
        "analysis_start": analysis_start.isoformat(),
        "analysis_end": analysis_end.isoformat(),
        "ath_valid_from": ath_valid_from.isoformat(),
        "primary_tolerance": PRIMARY_TOLERANCE_ID,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "planned_mde_target_sd": PLANNED_MDE_TARGET_SD,
        "alpha": ALPHA,
        "power": POWER,
        "tolerance_ids": tolerance_ids,
        "tolerance_specs": {k: TOLERANCE_SPECS[k] for k in tolerance_ids},
        "shift_offsets_bps": SHIFT_OFFSETS_BPS,
        "wrong_day_lags": WRONG_DAY_LAGS,
        "arm_distance_multiplier": ARM_DISTANCE_MULTIPLIER,
        "arm_consecutive_minutes": ARM_CONSECUTIVE_MINUTES,
        "match_exposure_days": MATCH_EXPOSURE_DAYS,
        "exposure_bin_bps": EXPOSURE_BIN_BPS,
        "match_active_window_pct": MATCH_ACTIVE_WINDOW_PCT,
        "round_active_window_pct": ROUND_ACTIVE_WINDOW_PCT,
        "round_grid_max": args.round_grid_max,
        "control_collision_bps": CONTROL_COLLISION_BPS,
        "rng_seed": RNG_SEED,
    }
    sig = source_signature(files, audit_summary, config)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]

    print("Evaluating deterministic BTCUSDT price-level reactions...")
    print(f"Auditor version   : {EVALUATOR_VERSION}")
    print(f"Run ID            : {run_id}")
    print(f"Analysis window   : {analysis_start} to {analysis_end}")
    print(f"Historical root   : {args.historical_root}")
    print(f"Output root       : {output_root}")
    print(f"Report root       : {report_root}")
    print(f"Primary tolerance : {PRIMARY_TOLERANCE_ID}")
    print(f"Primary endpoint  : {PRIMARY_ENDPOINT}")
    print(f"MDE target        : {PLANNED_MDE_TARGET_SD:.2f} SD")
    print("Level families are evaluated separately; no scalar key-level score is created.")

    df = load_source(files)
    df, sigma_all = prepare_market_arrays(df)
    daily = build_daily_stats(df)
    day_index = build_day_index(df)
    daily_exposure = build_daily_exposure_counts(df, day_index)
    real_levels_by_day = build_real_levels_by_day(daily, analysis_start, analysis_end, ath_valid_from, args.round_grid_max)

    # Input continuity sanity inside loaded data.
    dt = df["open_time"].diff().dropna().dt.total_seconds().to_numpy()
    if len(dt) and not np.all(dt == 60):
        raise RuntimeError("Loaded source is not strictly 60-second continuous; Pass-3 audit/source selection mismatch")

    state_conn = init_state_db(metadata_root / "state.sqlite")
    all_events = []
    inventory_rows: list[dict[str, Any]] = []
    years = list(range(analysis_start.year, analysis_end.year + 1))
    for yi, year in enumerate(years, 1):
        part = events_root / f"year={year}" / "part-00000.parquet"
        row = state_conn.execute("SELECT input_signature, status, event_rows, output_path FROM yearly_runs WHERE year=?", (year,)).fetchone()
        if (not args.force and row and row[0] == sig and row[1] == "COMPLETED" and part.exists()):
            ev = pd.read_parquet(part)
            all_events.append(ev)
            print(f"[{yi:02d}/{len(years):02d}] {year}: COMPLETED_REUSED ({len(ev):,} events)")
            # Inventory must still be rebuilt cheaply for reports.
            for d in sorted(x for x in real_levels_by_day if x.year == year):
                if d not in day_index:
                    continue
                s, _ = day_index[d]
                if s <= 0:
                    continue
                exposure = trailing_exposure_counter(d, daily_exposure)
                day_start_price = float(df["close"].iloc[s - 1])
                real0 = real_levels_by_day[d]
                controls = make_controls_for_day(d, real0, real_levels_by_day, day_start_price, exposure)
                cnt = Counter((x.family, x.control_type, x.control_variant) for x in [*real0, *controls])
                for (family, ctype, variant), n in cnt.items():
                    inventory_rows.append({"year": year, "date": d.isoformat(), "level_family": family, "control_type": ctype, "control_variant": variant, "candidate_levels": n})
            continue
        print(f"[{yi:02d}/{len(years):02d}] {year}: processing...")
        ev, inv = process_year(year, df, sigma_all, day_index, real_levels_by_day, daily_exposure,
                               tolerance_ids, analysis_start, analysis_end, run_id)
        part.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_parquet(part, ev)
        all_events.append(ev)
        inventory_rows.extend(inv)
        state_conn.execute(
            "INSERT INTO yearly_runs(year,input_signature,status,event_rows,run_id,completed_at,output_path) VALUES(?,?,?,?,?,?,?) "
            "ON CONFLICT(year) DO UPDATE SET input_signature=excluded.input_signature,status=excluded.status,event_rows=excluded.event_rows,run_id=excluded.run_id,completed_at=excluded.completed_at,output_path=excluded.output_path",
            (year, sig, "COMPLETED", len(ev), run_id, utc_now_iso(), str(part)),
        )
        state_conn.commit()
        print(f"[{yi:02d}/{len(years):02d}] {year}: COMPLETED ({len(ev):,} events)")
    state_conn.close()

    events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame(columns=EVENT_COLUMNS)
    if not events.empty:
        events["event_time"] = pd.to_datetime(events["event_time"], utc=True)
        events["level_available_time"] = pd.to_datetime(events["level_available_time"], utc=True, errors="coerce")
    inventory_df = pd.DataFrame(inventory_rows)

    pooled, yearly, variants = build_effect_tables(events) if len(events) else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    secondary = secondary_summary(events) if len(events) else pd.DataFrame()
    counts = build_event_count_table(events, inventory_df) if len(events) else pd.DataFrame()
    matchq = control_match_quality(events) if len(events) else pd.DataFrame()
    checks = invariants(events, analysis_start, analysis_end) if len(events) else [Check("events", "nonempty", "FAIL", 0, ">0")]
    failed = [c for c in checks if c.status == "FAIL"]
    status = "PASS" if not failed else "FAIL"

    # Reports.
    if len(pooled): atomic_write_dataframe_csv(report_root / "deterministic_level_primary_effects.csv", pooled)
    if len(yearly): atomic_write_dataframe_csv(report_root / "deterministic_level_yearly_effects.csv", yearly)
    if len(variants): atomic_write_dataframe_csv(report_root / "deterministic_level_control_variants.csv", variants)
    if len(secondary): atomic_write_dataframe_csv(report_root / "deterministic_level_secondary_outcomes.csv", secondary)
    if len(counts): atomic_write_dataframe_csv(report_root / "deterministic_level_event_counts.csv", counts)
    if len(matchq): atomic_write_dataframe_csv(report_root / "deterministic_level_control_match_quality.csv", matchq)
    if len(inventory_df): atomic_write_dataframe_csv(report_root / "deterministic_level_candidate_inventory.csv", inventory_df)
    atomic_write_csv(report_root / "deterministic_level_invariants.csv", [asdict(x) for x in checks])
    atomic_write_json(report_root / "deterministic_level_config.json", config)

    primary_events = events[events["tolerance_id"] == PRIMARY_TOLERANCE_ID] if len(events) else events
    summary = {
        "status": status,
        "auditor_version": EVALUATOR_VERSION,
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "completed_at": utc_now_iso(),
        "analysis_start": analysis_start.isoformat(),
        "analysis_end": analysis_end.isoformat(),
        "source_files": len(files),
        "source_rows_loaded": int(len(df)),
        "event_rows": int(len(events)),
        "primary_event_rows": int(len(primary_events)),
        "real_primary_events": int((primary_events["control_type"] == "REAL").sum()) if len(primary_events) else 0,
        "control_primary_events": int((primary_events["control_type"] != "REAL").sum()) if len(primary_events) else 0,
        "level_families": sorted(events["level_family"].dropna().unique().tolist()) if len(events) else [],
        "tolerance_ids": tolerance_ids,
        "primary_tolerance": PRIMARY_TOLERANCE_ID,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "planned_mde_target_sd": PLANNED_MDE_TARGET_SD,
        "failed_invariants": len(failed),
        "input_signature": sig,
        "output_root": str(output_root),
        "report_root": str(report_root),
    }
    atomic_write_json(metadata_root / "latest_summary.json", summary)
    atomic_write_json(metadata_root / "run_config.json", config)
    atomic_write_json(report_root / "deterministic_level_summary.json", summary)
    write_findings(report_root / "deterministic_level_findings.md", summary, pooled, yearly, checks)

    print("\nDeterministic level benchmark summary")
    print("------------------------------------")
    print(f"Status                 : {status}")
    print(f"Events total           : {len(events):,}")
    print(f"Primary events         : {len(primary_events):,}")
    print(f"Primary real events    : {summary['real_primary_events']:,}")
    print(f"Primary control events : {summary['control_primary_events']:,}")
    print(f"Failed invariants      : {len(failed)}")
    print(f"Summary                : {metadata_root / 'latest_summary.json'}")
    print(f"Findings               : {report_root / 'deterministic_level_findings.md'}")
    print("Negative REAL-minus-CONTROL primary effects mean the real level was harder to penetrate.")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
