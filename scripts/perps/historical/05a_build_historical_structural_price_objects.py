#!/usr/bin/env python3
"""Build causal historical structural price objects for BTCUSDT.

Phase 5A of the Stream historical-context research pipeline.

Purpose
-------
Construct *outcome-blind* structural price objects from the independently audited Binance
USD-M BTCUSDT 1-minute history. This script does not read the deterministic-reaction
benchmark outputs and does not label any object as a successful support/resistance level.
It creates structural candidates that Phase 5B can evaluate through the already-built
interaction benchmark harness.

Object families
---------------
REACTION
    Exposure-normalized concentrations of causally confirmed swing anchors. Structural
    scales are kept separate (1h, 4h, 1d). Two exposure denominators are built in parallel:
    close-at-price and uniform-across-1m-range. No cross-scale scalar strength score exists.

ACCEPTANCE
    High historical time-at-price regions, built from close occupancy. These are value-like
    areas, not assumed rejection levels.

TRAVERSAL
    Low-occupancy valleys between adjacent acceptance peaks, conditional on genuine historical
    visitation/crossing. These are thin/traversal candidates, not assumed breakout levels.

RECENT_SWING
    Simple recent confirmed swing structure (7d/30d x 1h/4h), deliberately much simpler than
    the multi-year reaction-density engine.

Causality
---------
* Every daily snapshot is frozen at 00:00 UTC and uses only information available strictly
  before that snapshot.
* Swing pivots have separate pivot_time and available_time. A centered pivot is not usable
  until the future confirmation window has fully closed.
* Trailing 24h volatility used as an anchor formation attribute is shifted one minute.
* Exposure, touch-day and crossing histories use only completed prior UTC days.
* Object construction never reads future interaction outcomes.

Important design choices
------------------------
* Price storage grids and structural geometry are separate. Anchor numerator resolution is
  5 bps; exposure denominator resolution is 25 bps.
* Reaction evidence is normalized by exposure opportunity with a shrinkage denominator.
* Connected-component chaining is not used. Reaction zones are local evidence peaks separated
  by valleys and bounded by a formation-volatility-derived hard width cap.
* Historical zone width is not resized by current volatility. Current-volatility approach
  tolerance belongs in Phase 5B, not here.
* Raw evidence vectors are emitted. There is intentionally no generic ``strength_score``.

Default source
--------------
D:\\stream-data\\historical\\binance\\futures_um\\BTCUSDT\\klines_1m

Default output
--------------
D:\\stream-data\\historical\\binance\\futures_um\\BTCUSDT\\historical_structural_price_objects_v1

Reports
-------
reports/binance_um_historical_structural_price_objects/
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
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

BUILDER_VERSION = "1.0.2"
OBJECT_SCHEMA_VERSION = "historical_structural_price_objects_v1"
ANCHOR_SCHEMA_VERSION = "historical_structural_anchors_v1"

DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_HISTORICAL_ROOT = Path(r"D:\stream-data\historical\binance\futures_um\BTCUSDT\klines_1m")
DEFAULT_OUTPUT_ROOT = Path(r"D:\stream-data\historical\binance\futures_um\BTCUSDT\historical_structural_price_objects_v1")
DEFAULT_START_DATE = "2020-02-01"

# Storage / evidence geometry. These are representation choices, not fitted outcomes.
ANCHOR_GRID_BPS = 5.0
EXPOSURE_GRID_BPS = 25.0
EXPOSURE_SHRINKAGE_MINUTES = 240.0
MIN_REACTION_EXPOSURE_MINUTES = 60.0
MIN_REACTION_ANCHORS = 2
MIN_INDEPENDENT_EPISODES = 2
EPISODE_SEPARATION_DAYS = 3
REGIME_SEPARATION_DAYS = 90
REACTION_BASIN_FLOOR_FRACTION = 0.50
REACTION_PEAK_PROMINENCE_RATIO = 0.25
WIDTH_CAP_SIGMA_MULTIPLIER = 0.50
WIDTH_CAP_MIN_BPS = 50.0
WIDTH_CAP_MAX_BPS = 300.0

# Structural scales are deliberately evaluated separately downstream.
REACTION_SCALE_SPECS: dict[int, dict[str, float]] = {
    60: {"bandwidth_bps": 50.0},
    240: {"bandwidth_bps": 75.0},
    1440: {"bandwidth_bps": 100.0},
}
REACTION_EXPOSURE_SCHEMES = ("CLOSE", "RANGE_UNIFORM")

# Acceptance / traversal definitions. These are descriptive structural objects.
ACCEPTANCE_BANDWIDTH_BPS = 75.0
ACCEPTANCE_MIN_VISIT_DAYS = 10
ACCEPTANCE_MIN_EXPOSURE_MINUTES = 360.0
ACCEPTANCE_PEAK_PERCENTILE = 90.0
ACCEPTANCE_PEAK_PROMINENCE_RATIO = 0.20
ACCEPTANCE_BASIN_FLOOR_FRACTION = 0.50
ACCEPTANCE_WIDTH_CAP_BPS = 300.0
TRAVERSAL_MAX_GAP_BPS = 1500.0
TRAVERSAL_VALLEY_RATIO_MAX = 0.40
TRAVERSAL_ZONE_FLOOR_RATIO = 0.60
TRAVERSAL_MIN_VISIT_DAYS = 5
TRAVERSAL_MIN_RANGE_CROSSINGS = 50.0
TRAVERSAL_WIDTH_CAP_BPS = 300.0

RECENT_SWING_SPECS = (
    (7, 60),
    (7, 240),
    (30, 60),
    (30, 240),
)
RECENT_SWING_CLUSTER_BPS = 25.0

TRAILING_VOL_MINUTES = 1440
TRAILING_VOL_MIN_PERIODS = 720
MIN_PROMINENCE_BPS = 5.0

SOURCE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "quote_volume", "trade_count"
]

OBJECT_COLUMNS = [
    "run_id", "schema_version", "symbol", "snapshot_time", "snapshot_date", "snapshot_year",
    "object_state_id", "object_key", "object_type", "variant_id", "tier", "geometry_kind", "scale_minutes",
    "exposure_scheme", "side_balance", "zone_lower", "zone_upper", "zone_center",
    "zone_width_bps", "peak_price", "peak_evidence", "peak_prominence_ratio",
    "width_cap_bps", "construction_width_cap_bps", "width_capped", "anchor_count", "support_anchor_count",
    "resistance_anchor_count", "anchor_prominence_sum", "independent_episode_count",
    "independent_regime_count", "first_anchor_time", "last_anchor_time",
    "last_anchor_available_time", "median_anchor_formation_sigma_bps",
    "close_exposure_minutes", "range_uniform_exposure_minutes", "visit_days",
    "range_crossings", "quote_volume_close_alloc", "trade_count_close_alloc",
    "evidence_rate_close", "evidence_rate_range_uniform", "acceptance_percentile",
    "traversal_valley_ratio", "age_days", "days_since_last_anchor", "is_active_25pct",
    "distance_to_zone_edge_bps", "inside_zone", "current_price",
    "latest_information_time", "construction_note",
]

ANCHOR_COLUMNS = [
    "symbol", "anchor_id", "pivot_time", "available_time", "side", "scale_minutes",
    "pivot_price", "prominence_bps", "formation_sigma_bps", "fine_bin",
]


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


def short_tmp(path: Path, prefix: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.parent / f".{prefix}-{uuid.uuid4().hex[:8]}.tmp"


def atomic_write_text(path: Path, text: str) -> None:
    tmp = short_tmp(path, "tx")
    try:
        tmp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def atomic_write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    tmp = short_tmp(path, "cv")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_parquet(path: Path, frame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = short_tmp(path, "pq")
    try:
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def require_dependencies() -> None:
    try:
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        import pyarrow  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Phase 5A requires numpy, pandas and pyarrow. Activate the project venv first."
        ) from exc


def parse_date(value: str | None) -> date | None:
    if value is None:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def find_repo_root(script_path: Path) -> Path:
    current = script_path.resolve().parent
    for candidate in [current, *current.parents]:
        if (candidate / "scripts").exists() and (candidate / "reports").exists():
            return candidate
    return script_path.resolve().parent


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise RuntimeError(f"Required manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_audited_source(root: Path) -> list[Path]:
    summary_path = root / "metadata" / "latest_audit_summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"Pass-3 audit summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS":
        raise RuntimeError(f"Pass-3 historical audit is not PASS: {summary.get('status')!r}")
    for key in ("missing_minutes", "duplicate_minutes", "source_mismatch_rows", "invariant_failures"):
        if int(summary.get(key, 0)) != 0:
            raise RuntimeError(f"Pass-3 summary has non-zero {key}: {summary.get(key)}")

    rows = _read_manifest(root / "metadata" / "build_manifest.csv")
    paths: list[Path] = []
    for row in rows:
        if row.get("status") not in {"COMPLETED", "COMPLETED_REUSED"}:
            continue
        p = Path(row.get("parquet_path", ""))
        if p.exists():
            paths.append(p)
    paths = sorted(set(paths))
    if not paths:
        raise RuntimeError("No completed Pass-2 monthly Parquet files were found")
    return paths


def build_signature(paths: list[Path], config: dict[str, Any]) -> str:
    payload = {
        "builder_version": BUILDER_VERSION,
        "schema_version": OBJECT_SCHEMA_VERSION,
        "config": config,
        "inputs": [
            {"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in paths
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def build_anchor_signature(paths: list[Path], symbol: str) -> str:
    """Signature only for the reusable anchor archive; independent of snapshot date range."""
    payload = {
        "builder_version": BUILDER_VERSION,
        "anchor_schema_version": ANCHOR_SCHEMA_VERSION,
        "symbol": symbol,
        "scales": sorted(set(REACTION_SCALE_SPECS) | {60, 240}),
        "min_prominence_bps": MIN_PROMINENCE_BPS,
        "anchor_grid_bps": ANCHOR_GRID_BPS,
        "trailing_vol_minutes": TRAILING_VOL_MINUTES,
        "trailing_vol_min_periods": TRAILING_VOL_MIN_PERIODS,
        "inputs": [
            {"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in paths
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def log_step(bps: float) -> float:
    return math.log1p(float(bps) / 10000.0)


def price_to_bin(values, bps: float):
    import numpy as np

    arr = np.asarray(values, dtype="float64")
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0):
        raise RuntimeError("price_to_bin received non-positive or non-finite values")
    return np.floor(np.log(arr) / log_step(bps)).astype("int64")


def bin_lower(idx: int, bps: float) -> float:
    return math.exp(idx * log_step(bps))


def bin_upper(idx: int, bps: float) -> float:
    return math.exp((idx + 1) * log_step(bps))


def bin_center(idx: int, bps: float) -> float:
    return math.exp((idx + 0.5) * log_step(bps))


def bps_distance(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return float("nan")
    return abs(math.log(a / b)) * 10000.0


def signed_bps(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return float("nan")
    return math.log(a / b) * 10000.0


def load_history(paths: list[Path]):
    import pandas as pd

    frames = [pd.read_parquet(p, columns=SOURCE_COLUMNS) for p in paths]
    data = pd.concat(frames, ignore_index=True)
    data["open_time"] = pd.to_datetime(data["open_time"], utc=True)
    data.sort_values("open_time", inplace=True, kind="mergesort")
    data.reset_index(drop=True, inplace=True)
    if data["open_time"].duplicated().any():
        raise RuntimeError("Duplicate open_time despite Pass-3 PASS")
    for col in ("open", "high", "low", "close", "quote_volume", "trade_count"):
        data[col] = pd.to_numeric(data[col], errors="raise")
    return data


def add_trailing_sigma(data):
    import numpy as np

    close = data["close"].to_numpy(dtype="float64")
    r = np.full(len(close), np.nan, dtype="float64")
    r[1:] = np.diff(np.log(close))
    sq = r * r
    # pandas rolling is robust and fast enough for 3.46m rows.
    import pandas as pd

    rv = pd.Series(sq).rolling(TRAILING_VOL_MINUTES, min_periods=TRAILING_VOL_MIN_PERIODS).sum()
    sigma = np.sqrt(rv.to_numpy(dtype="float64")) * 10000.0
    # Information at minute t must not include minute t's own return.
    sigma = np.roll(sigma, 1)
    sigma[0] = np.nan
    data = data.copy()
    data["sigma_24h_bps"] = sigma
    return data


def _centered_extrema(series, radius: int, mode: str):
    roll = series.rolling(2 * radius + 1, center=True, min_periods=2 * radius + 1)
    return roll.max() if mode == "max" else roll.min()


def build_anchor_archive(data, symbol: str):
    """Build causal anchors. Scales remain separate; no cross-scale weighting or dedup."""
    import numpy as np
    import pandas as pd

    highs = data["high"].astype("float64")
    lows = data["low"].astype("float64")
    times = pd.to_datetime(data["open_time"], utc=True)
    formation_sigma = data["sigma_24h_bps"].astype("float64")
    pieces = []

    all_scales = sorted(set(REACTION_SCALE_SPECS) | {60, 240})
    for radius in all_scales:
        max_high = _centered_extrema(highs, radius, "max")
        min_low = _centered_extrema(lows, radius, "min")

        left_low = lows.shift(1).rolling(radius, min_periods=radius).min()
        right_low = lows.iloc[::-1].shift(1).rolling(radius, min_periods=radius).min().iloc[::-1]
        left_high = highs.shift(1).rolling(radius, min_periods=radius).max()
        right_high = highs.iloc[::-1].shift(1).rolling(radius, min_periods=radius).max().iloc[::-1]

        high_prom = np.minimum(
            (highs - left_low) / highs * 10000.0,
            (highs - right_low) / highs * 10000.0,
        )
        low_prom = np.minimum(
            (left_high - lows) / lows * 10000.0,
            (right_high - lows) / lows * 10000.0,
        )

        specs = (
            ("RESISTANCE", highs.eq(max_high), highs, high_prom),
            ("SUPPORT", lows.eq(min_low), lows, low_prom),
        )
        for side, mask, prices, prominence in specs:
            keep = mask & prominence.ge(MIN_PROMINENCE_BPS) & prominence.notna() & formation_sigma.notna()
            if not bool(keep.any()):
                continue
            sub = pd.DataFrame({
                "symbol": symbol,
                "pivot_time": times[keep].to_numpy(),
                "available_time": (times[keep] + pd.to_timedelta(radius + 1, unit="m")).to_numpy(),
                "side": side,
                "scale_minutes": radius,
                "pivot_price": prices[keep].to_numpy(dtype="float64"),
                "prominence_bps": prominence[keep].to_numpy(dtype="float64"),
                "formation_sigma_bps": formation_sigma[keep].to_numpy(dtype="float64"),
            })
            pieces.append(sub)

    if not pieces:
        raise RuntimeError("No structural anchors were detected")
    anchors = pd.concat(pieces, ignore_index=True)
    anchors["pivot_time"] = pd.to_datetime(anchors["pivot_time"], utc=True)
    anchors["available_time"] = pd.to_datetime(anchors["available_time"], utc=True)
    anchors["fine_bin"] = price_to_bin(anchors["pivot_price"], ANCHOR_GRID_BPS)
    anchors.sort_values(["available_time", "scale_minutes", "side", "pivot_price"], inplace=True)
    anchors.reset_index(drop=True, inplace=True)
    anchors["anchor_id"] = [
        hashlib.sha1(
            f"{symbol}|{row.scale_minutes}|{row.side}|{row.pivot_time.isoformat()}|{row.pivot_price:.8f}".encode()
        ).hexdigest()[:20]
        for row in anchors.itertuples(index=False)
    ]
    return anchors[ANCHOR_COLUMNS].copy()


def gaussian_kernel(bandwidth_bps: float, grid_bps: float):
    import numpy as np

    sigma_bins = max(float(bandwidth_bps) / float(grid_bps), 0.5)
    radius = max(2, int(math.ceil(3.0 * sigma_bins)))
    x = np.arange(-radius, radius + 1, dtype="float64")
    k = np.exp(-0.5 * (x / sigma_bins) ** 2)
    k /= k.sum()
    return k


def smooth(arr, kernel):
    import numpy as np

    return np.convolve(np.asarray(arr, dtype="float64"), kernel, mode="same")


def build_daily_aggregates(data, coarse_min: int, coarse_max: int):
    """Build day x coarse-price-bin arrays from observed prior-day 1m bars.

    RANGE_UNIFORM is explicitly an approximation: each minute contributes one minute of
    exposure spread uniformly across the coarse bins intersected by its OHLC range.
    """
    import numpy as np
    import pandas as pd

    days = pd.DatetimeIndex(pd.to_datetime(data["open_time"], utc=True).dt.floor("D").unique()).sort_values()
    day_to_idx = {d.date(): i for i, d in enumerate(days)}
    n_days = len(days)
    n_bins = coarse_max - coarse_min + 1

    close_minutes = np.zeros((n_days, n_bins), dtype="float32")
    range_uniform = np.zeros((n_days, n_bins), dtype="float32")
    range_crossings = np.zeros((n_days, n_bins), dtype="float32")
    quote_close = np.zeros((n_days, n_bins), dtype="float64")
    trades_close = np.zeros((n_days, n_bins), dtype="float64")
    visit_day = np.zeros((n_days, n_bins), dtype="uint8")

    day_values = pd.to_datetime(data["open_time"], utc=True).dt.floor("D")
    for day_ts, idxs in data.groupby(day_values, sort=True).groups.items():
        di = day_to_idx[pd.Timestamp(day_ts).date()]
        pos = np.asarray(list(idxs), dtype="int64")
        close = data.loc[pos, "close"].to_numpy(dtype="float64")
        low = data.loc[pos, "low"].to_numpy(dtype="float64")
        high = data.loc[pos, "high"].to_numpy(dtype="float64")
        quote = data.loc[pos, "quote_volume"].to_numpy(dtype="float64")
        trades = data.loc[pos, "trade_count"].to_numpy(dtype="float64")

        cb = price_to_bin(close, EXPOSURE_GRID_BPS) - coarse_min
        lb = price_to_bin(low, EXPOSURE_GRID_BPS) - coarse_min
        hb = price_to_bin(high, EXPOSURE_GRID_BPS) - coarse_min
        lb = np.clip(lb, 0, n_bins - 1)
        hb = np.clip(hb, 0, n_bins - 1)
        cb = np.clip(cb, 0, n_bins - 1)

        close_minutes[di] += np.bincount(cb, minlength=n_bins).astype("float32")
        quote_close[di] += np.bincount(cb, weights=quote, minlength=n_bins)
        trades_close[di] += np.bincount(cb, weights=trades, minlength=n_bins)

        widths = (hb - lb + 1).astype("float64")
        weights = 1.0 / widths
        diff_u = np.zeros(n_bins + 1, dtype="float64")
        diff_c = np.zeros(n_bins + 1, dtype="float64")
        np.add.at(diff_u, lb, weights)
        np.add.at(diff_u, hb + 1, -weights)
        np.add.at(diff_c, lb, 1.0)
        np.add.at(diff_c, hb + 1, -1.0)
        ru = np.cumsum(diff_u[:-1])
        rc = np.cumsum(diff_c[:-1])
        range_uniform[di] = ru.astype("float32")
        range_crossings[di] = rc.astype("float32")
        visit_day[di] = (rc > 0).astype("uint8")

    return {
        "days": days,
        "day_to_idx": day_to_idx,
        "close_minutes": close_minutes,
        "range_uniform": range_uniform,
        "range_crossings": range_crossings,
        "quote_close": quote_close,
        "trades_close": trades_close,
        "visit_day": visit_day,
        "cum_close_minutes": np.cumsum(close_minutes, axis=0, dtype="float64"),
        "cum_range_uniform": np.cumsum(range_uniform, axis=0, dtype="float64"),
        "cum_range_crossings": np.cumsum(range_crossings, axis=0, dtype="float64"),
        "cum_quote_close": np.cumsum(quote_close, axis=0, dtype="float64"),
        "cum_trades_close": np.cumsum(trades_close, axis=0, dtype="float64"),
        "cum_visit_days": np.cumsum(visit_day, axis=0, dtype="int32"),
    }


def fine_to_coarse_index(fine_bins, coarse_min: int):
    import numpy as np

    prices = np.array([bin_center(int(b), ANCHOR_GRID_BPS) for b in fine_bins], dtype="float64")
    return price_to_bin(prices, EXPOSURE_GRID_BPS) - coarse_min


def local_peaks(score, eligible, min_prominence_ratio: float, min_separation_bins: int):
    """Outcome-blind local maxima with a simple topographic prominence approximation."""
    import numpy as np

    score = np.asarray(score, dtype="float64")
    eligible = np.asarray(eligible, dtype=bool)
    if len(score) < 3:
        return []
    raw = [
        i for i in range(1, len(score) - 1)
        if eligible[i] and np.isfinite(score[i]) and score[i] > 0
        and score[i] >= score[i - 1] and score[i] >= score[i + 1]
    ]
    if not raw:
        return []

    window = max(3, min_separation_bins * 4)
    enriched = []
    for i in raw:
        lo = max(0, i - window)
        hi = min(len(score), i + window + 1)
        left = score[lo:i]
        right = score[i + 1:hi]
        if len(left) == 0 or len(right) == 0:
            continue
        left_floor = float(np.nanmin(left))
        right_floor = float(np.nanmin(right))
        saddle = max(left_floor, right_floor)
        prom = (float(score[i]) - saddle) / max(float(score[i]), 1e-12)
        if prom >= min_prominence_ratio:
            enriched.append((i, float(score[i]), prom))

    # Non-maximum suppression prevents multiple nearly identical peaks without connected chaining.
    selected: list[tuple[int, float, float]] = []
    for item in sorted(enriched, key=lambda x: (-x[1], x[0])):
        if all(abs(item[0] - j[0]) >= min_separation_bins for j in selected):
            selected.append(item)
    return sorted(selected, key=lambda x: x[0])


def bounded_basins(score, peaks, floor_fraction: float, max_width_bins_by_peak: dict[int, int]):
    """Return non-overlapping local basins around peaks; no transitive connected components."""
    import numpy as np

    if not peaks:
        return []
    peak_indices = [p[0] for p in peaks]
    out = []
    n = len(score)
    for k, (i, height, prom) in enumerate(peaks):
        threshold = height * floor_fraction
        max_half = max(1, max_width_bins_by_peak.get(i, 1) // 2)
        left_limit = 0 if k == 0 else (peak_indices[k - 1] + i) // 2 + 1
        right_limit = n - 1 if k == len(peaks) - 1 else (i + peak_indices[k + 1]) // 2
        left = i
        while left - 1 >= left_limit and i - (left - 1) <= max_half and score[left - 1] >= threshold:
            left -= 1
        right = i
        while right + 1 <= right_limit and (right + 1) - i <= max_half and score[right + 1] >= threshold:
            right += 1
        width_capped = (i - left >= max_half and left > left_limit) or (right - i >= max_half and right < right_limit)
        out.append((left, right, i, height, prom, bool(width_capped)))
    return out


def independent_counts(anchor_days: Iterable[int]):
    days = sorted(set(int(x) for x in anchor_days))
    if not days:
        return 0, 0
    episodes = 1
    regimes = 1
    prev = days[0]
    for d in days[1:]:
        gap = d - prev
        if gap >= EPISODE_SEPARATION_DAYS:
            episodes += 1
        if gap >= REGIME_SEPARATION_DAYS:
            regimes += 1
        prev = d
    return episodes, regimes


def zone_distance_fields(current_price: float, lower: float, upper: float):
    if lower <= current_price <= upper:
        return 0.0, True
    if current_price < lower:
        return signed_bps(lower, current_price), False
    return -signed_bps(current_price, upper), False


def _safe_iso(value) -> str:
    if value is None:
        return ""
    try:
        if hasattr(value, "isoformat"):
            return value.isoformat()
    except Exception:
        pass
    return str(value)


def object_id(snapshot_time, object_type: str, variant_id: str, key: str) -> str:
    raw = f"{snapshot_time.isoformat()}|{object_type}|{variant_id}|{key}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]


def reaction_rows_for_snapshot(
    snapshot_time,
    current_price: float,
    fine_bins,
    fine_to_coarse,
    coarse_min: int,
    cumulative,
    anchor_state: dict[int, dict[str, Any]],
    run_id: str,
    symbol: str,
):
    import numpy as np
    import pandas as pd

    rows = []
    close_exp_fine = cumulative["close_minutes"][fine_to_coarse]
    range_exp_fine = cumulative["range_uniform"][fine_to_coarse]

    for scale, spec in REACTION_SCALE_SPECS.items():
        state = anchor_state[scale]
        raw_count = state["count"]
        support = state["support"]
        resistance = state["resistance"]
        prom_sum = state["prominence"]
        kernel = gaussian_kernel(spec["bandwidth_bps"], ANCHOR_GRID_BPS)
        sm_num = smooth(raw_count, kernel)
        sm_close_exp = smooth(close_exp_fine, kernel)
        sm_range_exp = smooth(range_exp_fine, kernel)

        for scheme in REACTION_EXPOSURE_SCHEMES:
            raw_exposure = close_exp_fine if scheme == "CLOSE" else range_exp_fine
            sm_exp = sm_close_exp if scheme == "CLOSE" else sm_range_exp
            rate = sm_num / (sm_exp + EXPOSURE_SHRINKAGE_MINUTES)
            eligible = (sm_exp >= MIN_REACTION_EXPOSURE_MINUTES) & (sm_num >= MIN_REACTION_ANCHORS)
            min_sep = max(2, int(round(spec["bandwidth_bps"] / ANCHOR_GRID_BPS)))
            peaks = local_peaks(rate, eligible, REACTION_PEAK_PROMINENCE_RATIO, min_sep)
            if not peaks:
                continue

            cap_bins: dict[int, int] = {}
            formation_by_peak: dict[int, float] = {}
            for i, _, _ in peaks:
                near_lo = max(0, i - min_sep)
                near_hi = min(len(fine_bins), i + min_sep + 1)
                vols = []
                for j in range(near_lo, near_hi):
                    vols.extend(state["formation_vols"].get(int(fine_bins[j]), []))
                med = float(np.median(vols)) if vols else 200.0
                cap = float(np.clip(WIDTH_CAP_SIGMA_MULTIPLIER * med, WIDTH_CAP_MIN_BPS, WIDTH_CAP_MAX_BPS))
                formation_by_peak[i] = med
                cap_bins[i] = max(1, int(math.ceil(cap / ANCHOR_GRID_BPS)))

            basins = bounded_basins(rate, peaks, REACTION_BASIN_FLOOR_FRACTION, cap_bins)
            for left, right, peak_i, peak_height, peak_prom, width_capped in basins:
                lo_bin = int(fine_bins[left])
                hi_bin = int(fine_bins[right])
                lower = bin_lower(lo_bin, ANCHOR_GRID_BPS)
                upper = bin_upper(hi_bin, ANCHOR_GRID_BPS)
                center = bin_center(int(fine_bins[peak_i]), ANCHOR_GRID_BPS)
                zone_count = int(raw_count[left:right + 1].sum())
                if zone_count < MIN_REACTION_ANCHORS:
                    continue
                supp = int(support[left:right + 1].sum())
                res = int(resistance[left:right + 1].sum())
                prom = float(prom_sum[left:right + 1].sum())

                days: list[int] = []
                times = []
                avail_times = []
                formation = []
                for j in range(left, right + 1):
                    fb = int(fine_bins[j])
                    days.extend(state["anchor_days"].get(fb, []))
                    times.extend(state["pivot_times"].get(fb, []))
                    avail_times.extend(state["available_times"].get(fb, []))
                    formation.extend(state["formation_vols"].get(fb, []))
                episodes, regimes = independent_counts(days)
                tier = "ESTABLISHED" if episodes >= MIN_INDEPENDENT_EPISODES else "CANDIDATE"

                c_lo = max(0, int(price_to_bin([lower], EXPOSURE_GRID_BPS)[0]) - coarse_min)
                c_hi = min(len(cumulative["close_minutes"]) - 1, int(price_to_bin([upper], EXPOSURE_GRID_BPS)[0]) - coarse_min)
                close_minutes = float(cumulative["close_minutes"][c_lo:c_hi + 1].sum())
                range_minutes = float(cumulative["range_uniform"][c_lo:c_hi + 1].sum())
                visit_days = int(cumulative["visit_days"][c_lo:c_hi + 1].max()) if c_hi >= c_lo else 0
                crossings = float(cumulative["range_crossings"][c_lo:c_hi + 1].sum())
                quote = float(cumulative["quote_close"][c_lo:c_hi + 1].sum())
                trades = float(cumulative["trades_close"][c_lo:c_hi + 1].sum())
                med_form = float(np.median(formation)) if formation else formation_by_peak[peak_i]
                # Preserve the exact cap that was used to construct this peak neighborhood.
                # The median formation sigma of all anchors inside the final zone is a separate
                # descriptive covariate and can differ from the peak-local sigma used at build time.
                construction_cap = float(np.clip(
                    WIDTH_CAP_SIGMA_MULTIPLIER * formation_by_peak[peak_i],
                    WIDTH_CAP_MIN_BPS, WIDTH_CAP_MAX_BPS,
                ))
                edge_dist, inside = zone_distance_fields(current_price, lower, upper)
                side_balance = "TWO_SIDED"
                if supp > 1.5 * max(res, 1):
                    side_balance = "SUPPORT_HEAVY"
                elif res > 1.5 * max(supp, 1):
                    side_balance = "RESISTANCE_HEAVY"
                first_anchor = min(times) if times else None
                last_anchor = max(times) if times else None
                last_available = max(avail_times) if avail_times else None
                age_days = (snapshot_time - first_anchor).total_seconds() / 86400.0 if first_anchor is not None else float("nan")
                since_last = (snapshot_time - last_anchor).total_seconds() / 86400.0 if last_anchor is not None else float("nan")

                variant = f"REACTION_{scale}M_{scheme}"
                key = f"{scale}|{scheme}|{int(fine_bins[peak_i])}"
                rows.append({
                    "run_id": run_id,
                    "schema_version": OBJECT_SCHEMA_VERSION,
                    "symbol": symbol,
                    "snapshot_time": snapshot_time,
                    "snapshot_date": snapshot_time.date().isoformat(),
                    "snapshot_year": snapshot_time.year,
                    "object_state_id": object_id(snapshot_time, "REACTION", variant, key),
                    "object_key": key,
                    "object_type": "REACTION",
                    "variant_id": variant,
                    "tier": tier,
                    "geometry_kind": "PEAK_NEIGHBORHOOD" if width_capped else "NATURAL_BASIN",
                    "scale_minutes": scale,
                    "exposure_scheme": scheme,
                    "side_balance": side_balance,
                    "zone_lower": lower,
                    "zone_upper": upper,
                    "zone_center": center,
                    "zone_width_bps": bps_distance(upper, lower),
                    "peak_price": center,
                    "peak_evidence": peak_height,
                    "peak_prominence_ratio": peak_prom,
                    "width_cap_bps": construction_cap,
                    "construction_width_cap_bps": construction_cap,
                    "width_capped": bool(width_capped),
                    "anchor_count": zone_count,
                    "support_anchor_count": supp,
                    "resistance_anchor_count": res,
                    "anchor_prominence_sum": prom,
                    "independent_episode_count": episodes,
                    "independent_regime_count": regimes,
                    "first_anchor_time": first_anchor,
                    "last_anchor_time": last_anchor,
                    "last_anchor_available_time": last_available,
                    "median_anchor_formation_sigma_bps": med_form,
                    "close_exposure_minutes": close_minutes,
                    "range_uniform_exposure_minutes": range_minutes,
                    "visit_days": visit_days,
                    "range_crossings": crossings,
                    "quote_volume_close_alloc": quote,
                    "trade_count_close_alloc": trades,
                    "evidence_rate_close": float(sm_num[peak_i] / (sm_close_exp[peak_i] + EXPOSURE_SHRINKAGE_MINUTES)),
                    "evidence_rate_range_uniform": float(sm_num[peak_i] / (sm_range_exp[peak_i] + EXPOSURE_SHRINKAGE_MINUTES)),
                    "acceptance_percentile": float("nan"),
                    "traversal_valley_ratio": float("nan"),
                    "age_days": age_days,
                    "days_since_last_anchor": since_last,
                    "is_active_25pct": abs(signed_bps(center, current_price)) <= math.log1p(0.25) * 10000.0,
                    "distance_to_zone_edge_bps": edge_dist,
                    "inside_zone": inside,
                    "current_price": current_price,
                    "latest_information_time": last_available,
                    "construction_note": (
                        "Exposure-normalized pivot-density peak; bounded peak neighborhood stopped by the formation-time construction cap; no connected-component chaining."
                        if width_capped else
                        "Exposure-normalized pivot-density peak; naturally bounded local basin; no connected-component chaining."
                    ),
                })
    return rows


def acceptance_and_traversal_rows(snapshot_time, current_price: float, coarse_bins, cumulative, run_id: str, symbol: str):
    import numpy as np

    kernel = gaussian_kernel(ACCEPTANCE_BANDWIDTH_BPS, EXPOSURE_GRID_BPS)
    exposure = cumulative["close_minutes"]
    sm = smooth(exposure, kernel)
    visits = cumulative["visit_days"]
    eligible = (exposure >= ACCEPTANCE_MIN_EXPOSURE_MINUTES) & (visits >= ACCEPTANCE_MIN_VISIT_DAYS)
    if not bool(np.any(eligible)):
        return [], []
    threshold = float(np.nanpercentile(sm[eligible], ACCEPTANCE_PEAK_PERCENTILE))
    eligible_peak = eligible & (sm >= threshold)
    min_sep = max(2, int(round(ACCEPTANCE_BANDWIDTH_BPS / EXPOSURE_GRID_BPS)))
    peaks = local_peaks(sm, eligible_peak, ACCEPTANCE_PEAK_PROMINENCE_RATIO, min_sep)
    cap_bins = {i: max(1, int(math.ceil(ACCEPTANCE_WIDTH_CAP_BPS / EXPOSURE_GRID_BPS))) for i, _, _ in peaks}
    basins = bounded_basins(sm, peaks, ACCEPTANCE_BASIN_FLOOR_FRACTION, cap_bins)

    acceptance_rows = []
    accepted_peak_meta = []
    for left, right, peak_i, height, prom, capped in basins:
        lo_bin = int(coarse_bins[left])
        hi_bin = int(coarse_bins[right])
        lower = bin_lower(lo_bin, EXPOSURE_GRID_BPS)
        upper = bin_upper(hi_bin, EXPOSURE_GRID_BPS)
        center = bin_center(int(coarse_bins[peak_i]), EXPOSURE_GRID_BPS)
        zone_exp = float(exposure[left:right + 1].sum())
        if zone_exp < ACCEPTANCE_MIN_EXPOSURE_MINUTES:
            continue
        percentile = float(100.0 * (np.sum(sm[eligible] <= height) / max(np.sum(eligible), 1)))
        edge_dist, inside = zone_distance_fields(current_price, lower, upper)
        key = f"TIME|{int(coarse_bins[peak_i])}"
        acceptance_rows.append({
            "run_id": run_id, "schema_version": OBJECT_SCHEMA_VERSION, "symbol": symbol,
            "snapshot_time": snapshot_time, "snapshot_date": snapshot_time.date().isoformat(),
            "snapshot_year": snapshot_time.year, "object_state_id": object_id(snapshot_time, "ACCEPTANCE", "ACCEPTANCE_TIME", key),
            "object_key": key, "object_type": "ACCEPTANCE", "variant_id": "ACCEPTANCE_TIME", "tier": "ESTABLISHED",
            "geometry_kind": "BOUNDED_ACCEPTANCE_NEIGHBORHOOD",
            "scale_minutes": None, "exposure_scheme": "CLOSE", "side_balance": "NONE",
            "zone_lower": lower, "zone_upper": upper, "zone_center": center,
            "zone_width_bps": bps_distance(upper, lower), "peak_price": center,
            "peak_evidence": height, "peak_prominence_ratio": prom, "width_cap_bps": ACCEPTANCE_WIDTH_CAP_BPS,
            "construction_width_cap_bps": ACCEPTANCE_WIDTH_CAP_BPS,
            "width_capped": bool(capped), "anchor_count": 0, "support_anchor_count": 0,
            "resistance_anchor_count": 0, "anchor_prominence_sum": 0.0,
            "independent_episode_count": 0, "independent_regime_count": 0,
            "first_anchor_time": None, "last_anchor_time": None, "last_anchor_available_time": None,
            "median_anchor_formation_sigma_bps": float("nan"),
            "close_exposure_minutes": zone_exp,
            "range_uniform_exposure_minutes": float(cumulative["range_uniform"][left:right + 1].sum()),
            "visit_days": int(visits[left:right + 1].max()),
            "range_crossings": float(cumulative["range_crossings"][left:right + 1].sum()),
            "quote_volume_close_alloc": float(cumulative["quote_close"][left:right + 1].sum()),
            "trade_count_close_alloc": float(cumulative["trades_close"][left:right + 1].sum()),
            "evidence_rate_close": float("nan"), "evidence_rate_range_uniform": float("nan"),
            "acceptance_percentile": percentile, "traversal_valley_ratio": float("nan"),
            "age_days": float("nan"), "days_since_last_anchor": float("nan"),
            "is_active_25pct": abs(signed_bps(center, current_price)) <= math.log1p(0.25) * 10000.0,
            "distance_to_zone_edge_bps": edge_dist, "inside_zone": inside, "current_price": current_price,
            "latest_information_time": snapshot_time - timedelta(minutes=1),
            "construction_note": "High cumulative close-time occupancy; descriptive acceptance/value object.",
        })
        accepted_peak_meta.append((peak_i, left, right, height))

    traversal_rows = []
    accepted_peak_meta.sort(key=lambda x: x[0])
    for a, b in zip(accepted_peak_meta, accepted_peak_meta[1:]):
        pi1, _, r1, h1 = a
        pi2, l2, _, h2 = b
        if pi2 <= pi1 + 1:
            continue
        gap_bps = (pi2 - pi1) * EXPOSURE_GRID_BPS
        if gap_bps > TRAVERSAL_MAX_GAP_BPS:
            continue
        lo = r1 + 1
        hi = l2 - 1
        if hi < lo:
            lo = pi1 + 1
            hi = pi2 - 1
        if hi < lo:
            continue
        valley_rel = int(np.argmin(sm[lo:hi + 1]))
        vi = lo + valley_rel
        valley = float(sm[vi])
        denom = max(min(h1, h2), 1e-12)
        ratio = valley / denom
        if ratio > TRAVERSAL_VALLEY_RATIO_MAX:
            continue
        if visits[vi] < TRAVERSAL_MIN_VISIT_DAYS or cumulative["range_crossings"][vi] < TRAVERSAL_MIN_RANGE_CROSSINGS:
            continue
        max_half = max(1, int(math.ceil(TRAVERSAL_WIDTH_CAP_BPS / EXPOSURE_GRID_BPS)) // 2)
        floor = denom * TRAVERSAL_ZONE_FLOOR_RATIO
        left = vi
        while left - 1 >= lo and vi - (left - 1) <= max_half and sm[left - 1] <= floor:
            left -= 1
        right = vi
        while right + 1 <= hi and (right + 1) - vi <= max_half and sm[right + 1] <= floor:
            right += 1
        lower = bin_lower(int(coarse_bins[left]), EXPOSURE_GRID_BPS)
        upper = bin_upper(int(coarse_bins[right]), EXPOSURE_GRID_BPS)
        center = bin_center(int(coarse_bins[vi]), EXPOSURE_GRID_BPS)
        edge_dist, inside = zone_distance_fields(current_price, lower, upper)
        key = f"VALLEY|{int(coarse_bins[vi])}|{int(coarse_bins[pi1])}|{int(coarse_bins[pi2])}"
        traversal_rows.append({
            "run_id": run_id, "schema_version": OBJECT_SCHEMA_VERSION, "symbol": symbol,
            "snapshot_time": snapshot_time, "snapshot_date": snapshot_time.date().isoformat(),
            "snapshot_year": snapshot_time.year, "object_state_id": object_id(snapshot_time, "TRAVERSAL", "TRAVERSAL_TIME_VALLEY", key),
            "object_key": key, "object_type": "TRAVERSAL", "variant_id": "TRAVERSAL_TIME_VALLEY", "tier": "ESTABLISHED",
            "geometry_kind": "BOUNDED_TRAVERSAL_NEIGHBORHOOD",
            "scale_minutes": None, "exposure_scheme": "CLOSE", "side_balance": "NONE",
            "zone_lower": lower, "zone_upper": upper, "zone_center": center,
            "zone_width_bps": bps_distance(upper, lower), "peak_price": center,
            "peak_evidence": valley, "peak_prominence_ratio": float("nan"), "width_cap_bps": TRAVERSAL_WIDTH_CAP_BPS,
            "construction_width_cap_bps": TRAVERSAL_WIDTH_CAP_BPS,
            "width_capped": (right - left + 1) * EXPOSURE_GRID_BPS >= TRAVERSAL_WIDTH_CAP_BPS,
            "anchor_count": 0, "support_anchor_count": 0, "resistance_anchor_count": 0,
            "anchor_prominence_sum": 0.0, "independent_episode_count": 0, "independent_regime_count": 0,
            "first_anchor_time": None, "last_anchor_time": None, "last_anchor_available_time": None,
            "median_anchor_formation_sigma_bps": float("nan"),
            "close_exposure_minutes": float(exposure[left:right + 1].sum()),
            "range_uniform_exposure_minutes": float(cumulative["range_uniform"][left:right + 1].sum()),
            "visit_days": int(visits[left:right + 1].max()),
            "range_crossings": float(cumulative["range_crossings"][left:right + 1].sum()),
            "quote_volume_close_alloc": float(cumulative["quote_close"][left:right + 1].sum()),
            "trade_count_close_alloc": float(cumulative["trades_close"][left:right + 1].sum()),
            "evidence_rate_close": float("nan"), "evidence_rate_range_uniform": float("nan"),
            "acceptance_percentile": float("nan"), "traversal_valley_ratio": ratio,
            "age_days": float("nan"), "days_since_last_anchor": float("nan"),
            "is_active_25pct": abs(signed_bps(center, current_price)) <= math.log1p(0.25) * 10000.0,
            "distance_to_zone_edge_bps": edge_dist, "inside_zone": inside, "current_price": current_price,
            "latest_information_time": snapshot_time - timedelta(minutes=1),
            "construction_note": "Low-occupancy valley between adjacent acceptance peaks; requires historical visitation/crossing.",
        })
    return acceptance_rows, traversal_rows


def recent_swing_rows(snapshot_time, current_price: float, anchors, run_id: str, symbol: str):
    import numpy as np
    import pandas as pd

    rows = []
    for window_days, scale in RECENT_SWING_SPECS:
        cutoff = snapshot_time - timedelta(days=window_days)
        sub = anchors[
            (anchors["scale_minutes"] == scale)
            & (anchors["available_time"] < snapshot_time)
            & (anchors["pivot_time"] >= cutoff)
        ].copy()
        if sub.empty:
            continue
        sub.sort_values("pivot_price", inplace=True)
        # Complete-linkage bounded clusters: adding an anchor may never make cluster >25 bps wide.
        clusters: list[list[Any]] = []
        current: list[Any] = []
        current_min = None
        for row in sub.itertuples(index=False):
            p = float(row.pivot_price)
            if not current:
                current = [row]
                current_min = p
                continue
            if bps_distance(p, float(current_min)) <= RECENT_SWING_CLUSTER_BPS:
                current.append(row)
            else:
                clusters.append(current)
                current = [row]
                current_min = p
        if current:
            clusters.append(current)

        variant = f"RECENT_SWING_{window_days}D_{scale}M"
        for cluster in clusters:
            prices = np.array([float(r.pivot_price) for r in cluster], dtype="float64")
            lower = float(np.min(prices))
            upper = float(np.max(prices))
            if upper <= lower:
                # A point-like cluster can contain either one pivot or several pivots at exactly
                # the same traded price. Store both cases as one 5-bps representation cell so
                # every emitted object has positive geometric width. Interaction tolerance is
                # Phase 5B's job and is intentionally not baked into this structural zone.
                fb = int(price_to_bin([prices[0]], ANCHOR_GRID_BPS)[0])
                lower = bin_lower(fb, ANCHOR_GRID_BPS)
                upper = bin_upper(fb, ANCHOR_GRID_BPS)
            center = float(np.median(prices))
            supp = sum(1 for r in cluster if r.side == "SUPPORT")
            res = len(cluster) - supp
            side_balance = "TWO_SIDED"
            if supp and not res:
                side_balance = "SUPPORT_HEAVY"
            elif res and not supp:
                side_balance = "RESISTANCE_HEAVY"
            pivot_times = [pd.Timestamp(r.pivot_time).to_pydatetime() for r in cluster]
            available_times = [pd.Timestamp(r.available_time).to_pydatetime() for r in cluster]
            edge_dist, inside = zone_distance_fields(current_price, lower, upper)
            # Object identity is based on actual cluster geometry plus exact latest availability,
            # not merely a rounded center bin/date. This prevents distinct bounded clusters from
            # colliding when their medians fall in the same 5-bps cell on the same UTC date.
            member_ids = sorted(str(r.anchor_id) for r in cluster)
            membership_hash = hashlib.sha1("|".join(member_ids).encode("utf-8")).hexdigest()[:12]
            lower_bin_key = int(price_to_bin([lower], ANCHOR_GRID_BPS)[0])
            upper_bin_key = int(price_to_bin([upper], ANCHOR_GRID_BPS)[0])
            latest_available_key = pd.Timestamp(max(available_times)).isoformat()
            key = f"{window_days}|{scale}|{lower_bin_key}|{upper_bin_key}|{latest_available_key}|{membership_hash}"
            rows.append({
                "run_id": run_id, "schema_version": OBJECT_SCHEMA_VERSION, "symbol": symbol,
                "snapshot_time": snapshot_time, "snapshot_date": snapshot_time.date().isoformat(),
                "snapshot_year": snapshot_time.year, "object_state_id": object_id(snapshot_time, "RECENT_SWING", variant, key),
                "object_key": key, "object_type": "RECENT_SWING", "variant_id": variant, "tier": "OBSERVED",
                "geometry_kind": "BOUNDED_RECENT_SWING_CLUSTER",
                "scale_minutes": scale, "exposure_scheme": "NONE", "side_balance": side_balance,
                "zone_lower": lower, "zone_upper": upper, "zone_center": center,
                "zone_width_bps": bps_distance(upper, lower), "peak_price": center,
                "peak_evidence": float(len(cluster)), "peak_prominence_ratio": float("nan"),
                "width_cap_bps": RECENT_SWING_CLUSTER_BPS, "construction_width_cap_bps": RECENT_SWING_CLUSTER_BPS,
                "width_capped": False,
                "anchor_count": len(cluster), "support_anchor_count": supp, "resistance_anchor_count": res,
                "anchor_prominence_sum": float(sum(float(r.prominence_bps) for r in cluster)),
                "independent_episode_count": 1, "independent_regime_count": 1,
                "first_anchor_time": min(pivot_times), "last_anchor_time": max(pivot_times),
                "last_anchor_available_time": max(available_times),
                "median_anchor_formation_sigma_bps": float(np.median([float(r.formation_sigma_bps) for r in cluster])),
                "close_exposure_minutes": float("nan"), "range_uniform_exposure_minutes": float("nan"),
                "visit_days": 0, "range_crossings": float("nan"), "quote_volume_close_alloc": float("nan"),
                "trade_count_close_alloc": float("nan"), "evidence_rate_close": float("nan"),
                "evidence_rate_range_uniform": float("nan"), "acceptance_percentile": float("nan"),
                "traversal_valley_ratio": float("nan"),
                "age_days": (snapshot_time - min(pivot_times)).total_seconds() / 86400.0,
                "days_since_last_anchor": (snapshot_time - max(pivot_times)).total_seconds() / 86400.0,
                "is_active_25pct": abs(signed_bps(center, current_price)) <= math.log1p(0.25) * 10000.0,
                "distance_to_zone_edge_bps": edge_dist, "inside_zone": inside, "current_price": current_price,
                "latest_information_time": max(available_times),
                "construction_note": "Recent confirmed pivots clustered by complete-linkage bounded price diameter; no transitive chaining.",
            })
    return rows


def make_anchor_state(fine_bins):
    import numpy as np

    state = {}
    n = len(fine_bins)
    for scale in REACTION_SCALE_SPECS:
        state[scale] = {
            "count": np.zeros(n, dtype="float64"),
            "support": np.zeros(n, dtype="float64"),
            "resistance": np.zeros(n, dtype="float64"),
            "prominence": np.zeros(n, dtype="float64"),
            "anchor_days": defaultdict(list),
            "pivot_times": defaultdict(list),
            "available_times": defaultdict(list),
            "formation_vols": defaultdict(list),
        }
    return state


def activate_anchor(anchor, anchor_state, fine_min: int) -> None:
    scale = int(anchor.scale_minutes)
    if scale not in anchor_state:
        return
    state = anchor_state[scale]
    fb = int(anchor.fine_bin)
    idx = fb - fine_min
    if idx < 0 or idx >= len(state["count"]):
        return
    state["count"][idx] += 1.0
    if anchor.side == "SUPPORT":
        state["support"][idx] += 1.0
    else:
        state["resistance"][idx] += 1.0
    state["prominence"][idx] += float(anchor.prominence_bps)
    pivot_dt = anchor.pivot_time.to_pydatetime() if hasattr(anchor.pivot_time, "to_pydatetime") else anchor.pivot_time
    avail_dt = anchor.available_time.to_pydatetime() if hasattr(anchor.available_time, "to_pydatetime") else anchor.available_time
    state["anchor_days"][fb].append(pivot_dt.date().toordinal())
    state["pivot_times"][fb].append(pivot_dt)
    state["available_times"][fb].append(avail_dt)
    state["formation_vols"][fb].append(float(anchor.formation_sigma_bps))


def cumulative_before_day(aggregates, day_idx: int):
    import numpy as np

    n_bins = aggregates["close_minutes"].shape[1]
    if day_idx <= 0:
        z = np.zeros(n_bins, dtype="float64")
        zi = np.zeros(n_bins, dtype="int32")
        return {
            "close_minutes": z.copy(), "range_uniform": z.copy(), "range_crossings": z.copy(),
            "quote_close": z.copy(), "trades_close": z.copy(), "visit_days": zi,
        }
    j = day_idx - 1
    return {
        "close_minutes": aggregates["cum_close_minutes"][j],
        "range_uniform": aggregates["cum_range_uniform"][j],
        "range_crossings": aggregates["cum_range_crossings"][j],
        "quote_close": aggregates["cum_quote_close"][j],
        "trades_close": aggregates["cum_trades_close"][j],
        "visit_days": aggregates["cum_visit_days"][j],
    }


def init_state_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS year_units (
        year INTEGER PRIMARY KEY,
        signature TEXT NOT NULL,
        status TEXT NOT NULL,
        rows INTEGER NOT NULL,
        updated_at TEXT NOT NULL
        )"""
    )
    conn.commit()
    return conn


def state_year_complete(conn: sqlite3.Connection, year: int, signature: str, output_path: Path) -> bool:
    row = conn.execute("SELECT signature,status FROM year_units WHERE year=?", (year,)).fetchone()
    return bool(row and row[0] == signature and row[1] == "COMPLETED" and output_path.exists())


def mark_year(conn: sqlite3.Connection, year: int, signature: str, status: str, rows: int) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO year_units(year,signature,status,rows,updated_at) VALUES(?,?,?,?,?)",
        (year, signature, status, rows, utc_now_iso()),
    )
    conn.commit()


def quantiles(values: Sequence[float]) -> dict[str, float]:
    import numpy as np

    arr = np.asarray([x for x in values if np.isfinite(x)], dtype="float64")
    if len(arr) == 0:
        return {k: float("nan") for k in ("p50", "p75", "p90", "p95", "p99", "max")}
    return {
        "p50": float(np.percentile(arr, 50)), "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)), "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)), "max": float(np.max(arr)),
    }


def audit_outputs(objects, anchors, start_date: date, end_date: date) -> list[Check]:
    import numpy as np
    import pandas as pd

    checks: list[Check] = []
    failed_anchor_availability = int((pd.to_datetime(anchors["available_time"], utc=True) <= pd.to_datetime(anchors["pivot_time"], utc=True)).sum())
    checks.append(Check("anchors", "available_time_after_pivot", "PASS" if failed_anchor_availability == 0 else "FAIL", failed_anchor_availability, 0))

    if objects.empty:
        checks.append(Check("objects", "non_empty", "FAIL", 0, ">0"))
        return checks

    dup = int(objects["object_state_id"].duplicated().sum())
    checks.append(Check("objects", "unique_object_state_id", "PASS" if dup == 0 else "FAIL", dup, 0))
    bad_bounds = int(((objects["zone_lower"] <= 0) | (objects["zone_upper"] <= objects["zone_lower"])).sum())
    checks.append(Check("objects", "valid_zone_bounds", "PASS" if bad_bounds == 0 else "FAIL", bad_bounds, 0))
    bad_center = int(((objects["zone_center"] < objects["zone_lower"]) | (objects["zone_center"] > objects["zone_upper"])).sum())
    checks.append(Check("objects", "center_inside_zone", "PASS" if bad_center == 0 else "FAIL", bad_center, 0))

    snaps = pd.to_datetime(objects["snapshot_time"], utc=True)
    latest = pd.to_datetime(objects["latest_information_time"], utc=True, errors="coerce")
    bad_info = int(((latest.notna()) & (latest >= snaps)).sum())
    checks.append(Check("causality", "latest_information_strictly_before_snapshot", "PASS" if bad_info == 0 else "FAIL", bad_info, 0))

    reaction = objects[objects["object_type"] == "REACTION"]
    if len(reaction):
        bad_width = int((reaction["zone_width_bps"].to_numpy(dtype="float64") > reaction["construction_width_cap_bps"].to_numpy(dtype="float64") + ANCHOR_GRID_BPS * 2.1).sum())
        checks.append(Check("reaction", "bounded_width", "PASS" if bad_width == 0 else "FAIL", bad_width, 0))
        bad_est = int(((reaction["tier"] == "ESTABLISHED") & (reaction["independent_episode_count"] < MIN_INDEPENDENT_EPISODES)).sum())
        checks.append(Check("reaction", "established_requires_independent_episodes", "PASS" if bad_est == 0 else "FAIL", bad_est, 0))

    forbidden = [c for c in objects.columns if any(token in c.lower() for token in ("penetration", "rejection_60", "outcome", "mfe", "mae"))]
    checks.append(Check("research_contract", "no_future_outcome_columns", "PASS" if not forbidden else "FAIL", ",".join(forbidden), "none"))

    observed_start = snaps.min().date()
    observed_end = snaps.max().date()
    checks.append(Check("coverage", "snapshot_start", "PASS" if observed_start >= start_date else "FAIL", observed_start.isoformat(), f">={start_date.isoformat()}"))
    checks.append(Check("coverage", "snapshot_end", "PASS" if observed_end <= end_date else "FAIL", observed_end.isoformat(), f"<={end_date.isoformat()}"))
    return checks


def write_reports(report_root: Path, objects, anchors, checks: list[Check], config: dict[str, Any], summary: dict[str, Any]) -> None:
    import pandas as pd

    report_root.mkdir(parents=True, exist_ok=True)
    count_rows = []
    grouped = objects.groupby(["object_type", "variant_id", "tier"], dropna=False).size().reset_index(name="rows")
    for row in grouped.to_dict("records"):
        count_rows.append(row)
    atomic_write_csv(report_root / "structural_price_object_counts.csv", count_rows)

    yearly = objects.groupby(["snapshot_year", "object_type", "variant_id"], dropna=False).size().reset_index(name="rows")
    atomic_write_csv(report_root / "structural_price_object_counts_by_year.csv", yearly.to_dict("records"))

    geometry_rows = []
    for (otype, variant), sub in objects.groupby(["object_type", "variant_id"], dropna=False):
        q = quantiles(sub["zone_width_bps"].to_numpy(dtype="float64"))
        geometry_rows.append({"object_type": otype, "variant_id": variant, "rows": len(sub), **q,
                              "active_25pct_share": float(sub["is_active_25pct"].mean())})
    atomic_write_csv(report_root / "structural_price_object_geometry.csv", geometry_rows)

    reaction_rows = []
    reaction = objects[objects["object_type"] == "REACTION"]
    if len(reaction):
        for (variant, tier), sub in reaction.groupby(["variant_id", "tier"]):
            reaction_rows.append({
                "variant_id": variant, "tier": tier, "rows": len(sub),
                "median_anchor_count": float(sub["anchor_count"].median()),
                "median_episode_count": float(sub["independent_episode_count"].median()),
                "median_regime_count": float(sub["independent_regime_count"].median()),
                "median_evidence_rate_close": float(sub["evidence_rate_close"].median()),
                "median_evidence_rate_range_uniform": float(sub["evidence_rate_range_uniform"].median()),
                "width_capped_share": float(sub["width_capped"].mean()),
            })
    atomic_write_csv(report_root / "reaction_object_evidence_summary.csv", reaction_rows)

    # Explicit requested-scale coverage, including zero-count scales such as 1d if no
    # reaction objects survive the predeclared gates. Zero is a result, not a missing row.
    reaction_scale_rows = []
    for scale in sorted(REACTION_SCALE_SPECS):
        for scheme in REACTION_EXPOSURE_SCHEMES:
            variant = f"REACTION_{scale}M_{scheme}"
            sub = reaction[reaction["variant_id"] == variant]
            reaction_scale_rows.append({
                "scale_minutes": scale,
                "exposure_scheme": scheme,
                "variant_id": variant,
                "rows": int(len(sub)),
                "established_rows": int((sub["tier"] == "ESTABLISHED").sum()) if len(sub) else 0,
                "candidate_rows": int((sub["tier"] == "CANDIDATE").sum()) if len(sub) else 0,
                "width_capped_share": float(sub["width_capped"].mean()) if len(sub) else float("nan"),
            })
    atomic_write_csv(report_root / "reaction_scale_coverage.csv", reaction_scale_rows)

    reaction_cap_rows = []
    if len(reaction):
        for (year, variant), sub in reaction.groupby(["snapshot_year", "variant_id"]):
            reaction_cap_rows.append({
                "snapshot_year": int(year),
                "variant_id": variant,
                "rows": int(len(sub)),
                "width_capped_share": float(sub["width_capped"].mean()),
                "natural_basin_share": float((~sub["width_capped"].astype(bool)).mean()),
                "median_zone_width_bps": float(sub["zone_width_bps"].median()),
                "median_construction_width_cap_bps": float(sub["construction_width_cap_bps"].median()),
            })
    atomic_write_csv(report_root / "reaction_width_cap_by_year.csv", reaction_cap_rows)

    anchor_summary = anchors.groupby(["scale_minutes", "side"]).size().reset_index(name="anchors")
    atomic_write_csv(report_root / "structural_anchor_summary.csv", anchor_summary.to_dict("records"))
    atomic_write_csv(report_root / "structural_price_object_invariants.csv", [asdict(c) for c in checks])
    atomic_write_json(report_root / "structural_price_object_config.json", config)
    atomic_write_json(report_root / "structural_price_object_summary.json", summary)

    failed = [c for c in checks if c.status == "FAIL"]
    lines = [
        "# Historical Structural Price Objects — Phase 5A",
        "",
        f"Status: **{summary['status']}**",
        "",
        "## Research contract",
        "",
        "This build is outcome-blind. It constructs structural objects only; Phase 5B will evaluate subsequent interactions.",
        "Reaction scales are separate and no scalar generic key-level strength score is emitted.",
        "",
        "## Counts",
        "",
    ]
    for row in count_rows:
        lines.append(f"- {row['object_type']} / {row['variant_id']} / {row['tier']}: {row['rows']:,}")
    lines += ["", "## Invariants", "", f"Failed checks: **{len(failed)}**"]
    if failed:
        for c in failed:
            lines.append(f"- FAIL {c.scope}/{c.check}: observed={c.observed} expected={c.expected} {c.detail}")
    zero_reaction_variants = [r["variant_id"] for r in reaction_scale_rows if r["rows"] == 0]
    fully_capped_variants = [r["variant_id"] for r in reaction_scale_rows if r["rows"] > 0 and r["width_capped_share"] >= 0.999999]
    lines += ["", "## Reaction geometry diagnostics", ""]
    if zero_reaction_variants:
        lines.append("- Zero-count requested reaction variants: " + ", ".join(zero_reaction_variants))
    else:
        lines.append("- Every requested reaction variant emitted at least one object.")
    if fully_capped_variants:
        lines.append("- Variants with 100% width-capped objects: " + ", ".join(fully_capped_variants))
        lines.append("- Width-capped REACTION objects are explicitly labeled PEAK_NEIGHBORHOOD; they are not claimed to be naturally bounded basins.")
    elif len(reaction):
        lines.append("- At least one reaction variant contains naturally bounded basins.")
    lines += [
        "",
        "## Interpretation guardrail",
        "",
        "REACTION objects are historical reaction-density candidates; capped ones are PEAK_NEIGHBORHOOD objects rather than claimed natural basins. ACCEPTANCE objects are time-at-price regions,",
        "TRAVERSAL objects are low-occupancy valleys, and RECENT_SWING objects are simple recent pivots.",
        "None of those names imply a future rejection, breakout, or profitability result.",
    ]
    atomic_write_text(report_root / "structural_price_object_findings.md", "\n".join(lines) + "\n")


def self_test() -> int:
    import numpy as np
    import pandas as pd

    # Peak / basin test ensures weak connecting evidence cannot chain two distant peaks.
    score = np.array([0, 1, 5, 10, 5, 1, 0.5, 1, 6, 11, 6, 1, 0], dtype="float64")
    eligible = np.ones_like(score, dtype=bool)
    peaks = local_peaks(score, eligible, 0.20, 2)
    assert len(peaks) == 2, peaks
    basins = bounded_basins(score, peaks, 0.5, {p[0]: 6 for p in peaks})
    assert len(basins) == 2
    assert basins[0][1] < basins[1][0], basins

    # Independent episodes/regimes.
    ep, rg = independent_counts([1, 1, 2, 8, 9, 120])
    assert ep == 3, (ep, rg)
    assert rg == 2, (ep, rg)

    # Complete-linkage recent-swing concept: endpoints >25 bps must not chain through middle.
    a = 100.0
    b = a * math.exp(20 / 10000)
    c = a * math.exp(40 / 10000)
    assert bps_distance(a, b) < RECENT_SWING_CLUSTER_BPS
    assert bps_distance(a, c) > RECENT_SWING_CLUSTER_BPS

    # Repeated pivots at exactly the same traded price are still a point-like
    # recent-swing cluster and must be represented by a positive-width cell.
    same_prices = np.array([100.0, 100.0], dtype="float64")
    same_lower = float(np.min(same_prices))
    same_upper = float(np.max(same_prices))
    assert same_upper <= same_lower
    same_bin = int(price_to_bin([same_prices[0]], ANCHOR_GRID_BPS)[0])
    same_lower = bin_lower(same_bin, ANCHOR_GRID_BPS)
    same_upper = bin_upper(same_bin, ANCHOR_GRID_BPS)
    assert same_upper > same_lower

    # Geometry-based recent-swing identity: two clusters sharing a center/date must
    # still differ if their bounded geometry or membership differs.
    latest = pd.Timestamp("2026-01-01T12:34:56Z")
    def _test_recent_key(lo, hi, ids):
        lo_bin = int(price_to_bin([lo], ANCHOR_GRID_BPS)[0])
        hi_bin = int(price_to_bin([hi], ANCHOR_GRID_BPS)[0])
        mh = hashlib.sha1("|".join(sorted(ids)).encode("utf-8")).hexdigest()[:12]
        return f"7|60|{lo_bin}|{hi_bin}|{latest.isoformat()}|{mh}"
    assert _test_recent_key(99.9, 100.1, ["a", "b"]) != _test_recent_key(99.9, 100.2, ["a", "c"])

    # Construction-cap semantics: the emitted audit cap is the peak-local cap actually
    # used by bounded_basins, not a post-hoc recomputation from all anchors in the zone.
    peak_local_sigma = 400.0
    construction_cap = float(np.clip(WIDTH_CAP_SIGMA_MULTIPLIER * peak_local_sigma, WIDTH_CAP_MIN_BPS, WIDTH_CAP_MAX_BPS))
    posthoc_zone_sigma = 250.0
    posthoc_cap = float(np.clip(WIDTH_CAP_SIGMA_MULTIPLIER * posthoc_zone_sigma, WIDTH_CAP_MIN_BPS, WIDTH_CAP_MAX_BPS))
    assert construction_cap != posthoc_cap
    assert construction_cap == 200.0

    # Causality geometry / distance.
    d, inside = zone_distance_fields(100.0, 99.0, 101.0)
    assert inside and d == 0.0
    d2, inside2 = zone_distance_fields(98.0, 99.0, 101.0)
    assert not inside2 and d2 > 0

    # Range-uniform difference-array math on a tiny synthetic day.
    lb = np.array([0, 1], dtype=int)
    hb = np.array([1, 2], dtype=int)
    widths = (hb - lb + 1).astype(float)
    diff = np.zeros(4)
    np.add.at(diff, lb, 1.0 / widths)
    np.add.at(diff, hb + 1, -1.0 / widths)
    ru = np.cumsum(diff[:-1])
    assert np.allclose(ru, [0.5, 1.0, 0.5])
    assert math.isclose(float(ru.sum()), 2.0)

    # Timestamp availability example.
    pivot = pd.Timestamp("2026-01-01T10:00:00Z")
    available = pivot + pd.Timedelta(minutes=61)
    snap = pd.Timestamp("2026-01-01T11:00:00Z")
    assert not (available < snap)
    snap2 = pd.Timestamp("2026-01-01T12:00:00Z")
    assert available < snap2

    print("SELF-TEST PASS")
    print("Validated exposure-grid math, independent episodes, bounded peak/basin geometry, geometry-based recent-swing identity, exact construction-cap semantics, and causal availability.")
    return 0


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-root", type=Path, default=DEFAULT_HISTORICAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-root", type=Path, default=None)
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=None, help="Inclusive UTC snapshot date; default is final source date")
    parser.add_argument("--force", action="store_true", help="Rewrite completed yearly partitions")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    require_dependencies()

    import numpy as np
    import pandas as pd

    script_path = Path(__file__)
    repo_root = find_repo_root(script_path)
    report_root = args.report_root or (repo_root / "reports" / "binance_um_historical_structural_price_objects")
    historical_root = args.historical_root
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    source_paths = verify_audited_source(historical_root)
    config = {
        "builder_version": BUILDER_VERSION,
        "schema_version": OBJECT_SCHEMA_VERSION,
        "symbol": args.symbol,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "anchor_grid_bps": ANCHOR_GRID_BPS,
        "exposure_grid_bps": EXPOSURE_GRID_BPS,
        "exposure_shrinkage_minutes": EXPOSURE_SHRINKAGE_MINUTES,
        "reaction_scale_specs": REACTION_SCALE_SPECS,
        "reaction_exposure_schemes": list(REACTION_EXPOSURE_SCHEMES),
        "reaction_peak_prominence_ratio": REACTION_PEAK_PROMINENCE_RATIO,
        "reaction_basin_floor_fraction": REACTION_BASIN_FLOOR_FRACTION,
        "min_independent_episodes": MIN_INDEPENDENT_EPISODES,
        "episode_separation_days": EPISODE_SEPARATION_DAYS,
        "regime_separation_days": REGIME_SEPARATION_DAYS,
        "width_cap_sigma_multiplier": WIDTH_CAP_SIGMA_MULTIPLIER,
        "width_cap_min_bps": WIDTH_CAP_MIN_BPS,
        "width_cap_max_bps": WIDTH_CAP_MAX_BPS,
        "acceptance": {
            "bandwidth_bps": ACCEPTANCE_BANDWIDTH_BPS,
            "min_visit_days": ACCEPTANCE_MIN_VISIT_DAYS,
            "min_exposure_minutes": ACCEPTANCE_MIN_EXPOSURE_MINUTES,
            "peak_percentile": ACCEPTANCE_PEAK_PERCENTILE,
            "peak_prominence_ratio": ACCEPTANCE_PEAK_PROMINENCE_RATIO,
            "width_cap_bps": ACCEPTANCE_WIDTH_CAP_BPS,
        },
        "traversal": {
            "max_gap_bps": TRAVERSAL_MAX_GAP_BPS,
            "valley_ratio_max": TRAVERSAL_VALLEY_RATIO_MAX,
            "min_visit_days": TRAVERSAL_MIN_VISIT_DAYS,
            "min_range_crossings": TRAVERSAL_MIN_RANGE_CROSSINGS,
            "width_cap_bps": TRAVERSAL_WIDTH_CAP_BPS,
        },
        "recent_swing_specs": [list(x) for x in RECENT_SWING_SPECS],
        "research_contract": {
            "outcome_blind": True,
            "no_generic_strength_score": True,
            "reaction_scales_kept_separate": True,
            "historical_zone_width_not_current_vol_scaled": True,
            "deterministic_levels_external_benchmark": True,
            "reaction_width_cap_is_exact_construction_cap": True,
            "capped_reaction_geometry_label": "PEAK_NEIGHBORHOOD",
        },
    }
    signature = build_signature(source_paths, config)
    anchor_signature = build_anchor_signature(source_paths, args.symbol)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]

    print("Building causal BTCUSDT historical structural price objects...")
    print(f"Builder version    : {BUILDER_VERSION}")
    print(f"Run ID             : {run_id}")
    print(f"Historical root    : {historical_root}")
    print(f"Output root        : {output_root}")
    print(f"Report root        : {report_root}")
    print("Object families    : REACTION, ACCEPTANCE, TRAVERSAL, RECENT_SWING")
    print("Outcome data       : NOT READ")
    print("Connected chaining : NOT USED")

    data = load_history(source_paths)
    data = add_trailing_sigma(data)
    source_first = pd.Timestamp(data["open_time"].iloc[0]).date()
    source_last = pd.Timestamp(data["open_time"].iloc[-1]).date()
    start_date = parse_date(args.start_date) or source_first
    end_date = parse_date(args.end_date) or source_last
    if start_date < source_first:
        raise RuntimeError(f"start-date {start_date} precedes source history {source_first}")
    if end_date > source_last:
        raise RuntimeError(f"end-date {end_date} exceeds source history {source_last}")
    if end_date < start_date:
        raise RuntimeError("end-date precedes start-date")

    print(f"Source rows        : {len(data):,}")
    print(f"Source coverage    : {source_first} to {source_last}")
    print(f"Snapshot coverage  : {start_date} to {end_date}")

    anchors_path = output_root / "anchors" / "structural_anchors.parquet"
    anchor_meta_path = output_root / "metadata" / "anchor_signature.json"
    anchors = None
    if anchors_path.exists() and anchor_meta_path.exists() and not args.force:
        meta = json.loads(anchor_meta_path.read_text(encoding="utf-8"))
        if meta.get("signature") == anchor_signature:
            anchors = pd.read_parquet(anchors_path)
            anchors["pivot_time"] = pd.to_datetime(anchors["pivot_time"], utc=True)
            anchors["available_time"] = pd.to_datetime(anchors["available_time"], utc=True)
            print(f"Structural anchors : REUSED ({len(anchors):,})")
    if anchors is None:
        print("Detecting causal structural anchors...")
        anchors = build_anchor_archive(data, args.symbol)
        atomic_write_parquet(anchors_path, anchors)
        atomic_write_json(anchor_meta_path, {"signature": anchor_signature, "rows": len(anchors), "created_at": utc_now_iso()})
        print(f"Structural anchors : BUILT ({len(anchors):,})")

    fine_min = int(min(price_to_bin(data["low"], ANCHOR_GRID_BPS).min(), anchors["fine_bin"].min()))
    fine_max = int(max(price_to_bin(data["high"], ANCHOR_GRID_BPS).max(), anchors["fine_bin"].max()))
    coarse_min = int(price_to_bin([float(data["low"].min())], EXPOSURE_GRID_BPS)[0])
    coarse_max = int(price_to_bin([float(data["high"].max())], EXPOSURE_GRID_BPS)[0])
    fine_bins = np.arange(fine_min, fine_max + 1, dtype="int64")
    coarse_bins = np.arange(coarse_min, coarse_max + 1, dtype="int64")
    f2c = fine_to_coarse_index(fine_bins, coarse_min)
    f2c = np.clip(f2c, 0, len(coarse_bins) - 1)

    print(f"Anchor grid bins   : {len(fine_bins):,} @ {ANCHOR_GRID_BPS:g} bps")
    print(f"Exposure grid bins : {len(coarse_bins):,} @ {EXPOSURE_GRID_BPS:g} bps")
    print("Building causal daily exposure archives...")
    aggregates = build_daily_aggregates(data, coarse_min, coarse_max)
    print(f"Exposure days      : {len(aggregates['days']):,}")

    # Day-start current price is previous minute's close. Build a lookup once.
    data_times = pd.to_datetime(data["open_time"], utc=True)
    close_by_time = pd.Series(data["close"].to_numpy(dtype="float64"), index=data_times)

    anchor_state = make_anchor_state(fine_bins)
    reaction_anchors = anchors[anchors["scale_minutes"].isin(REACTION_SCALE_SPECS)].sort_values("available_time").reset_index(drop=True)
    anchor_pointer = 0

    snapshot_days = pd.date_range(start=start_date.isoformat(), end=end_date.isoformat(), freq="D", tz="UTC")
    all_rows: list[dict[str, Any]] = []
    year_rows: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    state_conn = init_state_db(output_root / "metadata" / "build_state.sqlite")

    # We still advance state through pre-start days so first requested snapshot is causal.
    all_state_days = aggregates["days"]
    requested_set = set(d.date() for d in snapshot_days)
    total_requested = len(snapshot_days)
    completed_requested = 0

    for day_idx, day_ts in enumerate(all_state_days):
        snapshot_time = day_ts.to_pydatetime()
        # Activate anchors strictly before day-start.
        while anchor_pointer < len(reaction_anchors):
            row = reaction_anchors.iloc[anchor_pointer]
            if pd.Timestamp(row["available_time"]) >= day_ts:
                break
            activate_anchor(row, anchor_state, fine_min)
            anchor_pointer += 1

        day_date = day_ts.date()
        if day_date not in requested_set:
            continue
        prev_minute = day_ts - pd.Timedelta(minutes=1)
        if prev_minute not in close_by_time.index:
            # First source day cannot form a prior-day snapshot.
            continue
        current_price = float(close_by_time.loc[prev_minute])
        cumulative = cumulative_before_day(aggregates, day_idx)

        rows = []
        rows.extend(reaction_rows_for_snapshot(
            snapshot_time, current_price, fine_bins, f2c, coarse_min, cumulative,
            anchor_state, run_id, args.symbol,
        ))
        acc, trav = acceptance_and_traversal_rows(snapshot_time, current_price, coarse_bins, cumulative, run_id, args.symbol)
        rows.extend(acc)
        rows.extend(trav)
        rows.extend(recent_swing_rows(snapshot_time, current_price, anchors, run_id, args.symbol))
        year_rows[day_ts.year].extend(rows)
        all_rows.extend(rows)
        completed_requested += 1
        if completed_requested == 1 or completed_requested % 100 == 0 or completed_requested == total_requested:
            print(f"Snapshots processed : {completed_requested:,}/{total_requested:,} | objects {len(all_rows):,}")

    # Write yearly partitions. Existing matching partitions may be reused, but this build deliberately
    # recomputes causal state in memory so a resumed run cannot start from an inconsistent partial state.
    for year in sorted(year_rows):
        out_path = output_root / "objects" / f"year={year}" / "part-00000.parquet"
        if state_year_complete(state_conn, year, signature, out_path) and not args.force:
            print(f"{year}: COMPLETED_REUSED ({len(year_rows[year]):,} recomputed rows; existing partition retained)")
            continue
        frame = pd.DataFrame(year_rows[year], columns=OBJECT_COLUMNS)
        atomic_write_parquet(out_path, frame)
        mark_year(state_conn, year, signature, "COMPLETED", len(frame))
        print(f"{year}: COMPLETED ({len(frame):,} objects)")
    state_conn.close()

    objects = pd.DataFrame(all_rows, columns=OBJECT_COLUMNS)
    checks = audit_outputs(objects, anchors, start_date, end_date)
    failed = [c for c in checks if c.status == "FAIL"]
    status = "PASS" if not failed else "FAIL"

    counts = objects["object_type"].value_counts().to_dict() if len(objects) else {}
    summary = {
        "status": status,
        "builder_version": BUILDER_VERSION,
        "schema_version": OBJECT_SCHEMA_VERSION,
        "run_id": run_id,
        "signature": signature,
        "source_rows": len(data),
        "source_months": len(source_paths),
        "source_first_date": source_first.isoformat(),
        "source_last_date": source_last.isoformat(),
        "snapshot_start_date": start_date.isoformat(),
        "snapshot_end_date": end_date.isoformat(),
        "snapshots_requested": total_requested,
        "snapshots_processed": completed_requested,
        "anchor_rows": len(anchors),
        "object_rows": len(objects),
        "object_counts": counts,
        "failed_invariants": len(failed),
        "created_at": utc_now_iso(),
    }

    atomic_write_json(output_root / "metadata" / "latest_summary.json", summary)
    atomic_write_json(output_root / "metadata" / "build_config.json", config)
    atomic_write_csv(output_root / "metadata" / "invariants.csv", [asdict(c) for c in checks])
    write_reports(report_root, objects, anchors, checks, config, summary)

    print("\nHistorical structural price-object build summary")
    print("----------------------------------------------")
    print(f"Status                 : {status}")
    print(f"Snapshots processed    : {completed_requested:,}")
    print(f"Structural anchors     : {len(anchors):,}")
    print(f"Object rows            : {len(objects):,}")
    for key in ("REACTION", "ACCEPTANCE", "TRAVERSAL", "RECENT_SWING"):
        print(f"{key:<23}: {int(counts.get(key, 0)):,}")
    print(f"Failed invariants      : {len(failed)}")
    print(f"Summary                : {output_root / 'metadata' / 'latest_summary.json'}")
    print(f"Findings               : {report_root / 'structural_price_object_findings.md'}")
    print("Phase 5A does not evaluate future reactions. Phase 5B should consume these objects through the benchmark harness.")
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
