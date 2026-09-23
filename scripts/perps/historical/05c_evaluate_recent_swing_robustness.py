#!/usr/bin/env python3
"""Phase 5C robustness study for recent BTCUSDT swing structure.

Purpose
-------
Stress-test the promising Phase-5B RECENT_SWING result without changing Phase-5A
structural definitions or the frozen primary outcome. This evaluator is deliberately
narrow: it asks whether recent confirmed swing zones remain harder to penetrate when
compared with genuine but stale swing zones and with much tighter geometry/exposure-
matched random zones.

Frozen primary contract
-----------------------
* Primary structural hypothesis: RECENT_SWING_7D_60M.
* Primary tolerance: VOL_0P10, exactly as Phase 4B / 5B.
* Primary endpoint: 60-minute maximum penetration beyond the approached near edge,
  normalized by trailing 24h realized volatility known before the event minute.
* Primary control: genuine 1h swing structure whose pivots are 8-14 days old at the
  snapshot, matched to the source zone on side of day-start price, distance, width,
  and trailing-30d close exposure. Primary inference is source-matched: only real
  interactions whose own matched control also produced a valid interaction enter the
  contrast. Unmatched real interactions remain descriptive but cannot enter the effect.
* Secondary stale control for 7d objects: 15-30 days old.
* 30d objects use 31-60-day stale controls. 4h variants are sensitivity hypotheses.
* Strict matched-random controls are accepted only when explicit distance/exposure
  gates pass; missing controls are reported, never relaxed post hoc.
* Phase-5A objects are frozen and read-only. Outcome data never feeds construction.
* Default analysis window: 2020-02-01 through 2025-12-31 UTC.
* Planned MDE target remains 0.15 SD (alpha=.05, power=.80).

Stale-swing control idea
------------------------
The stale control is the recent-swing analogue of the wrong-day control used in
Pass 4B. It is still a genuine confirmed turning-price cluster, but its temporal
relevance has been deliberately broken. Stale clusters that overlap any currently
active RECENT_SWING zone are rejected so a supposedly stale level cannot secretly
also be current structure.

Confluence
----------
Overlapping Phase-5A recent-swing objects remain separate hypotheses, but real
interactions share a physical_event_id. Reports include both membership-level
clustered effects and a physical-event-deduplicated robustness table so dense
confluence cannot manufacture sample size.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sqlite3
import uuid
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

EVALUATOR_VERSION = "1.0.2"
SCHEMA_VERSION = "recent_swing_robustness_v1"
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_HISTORICAL_ROOT = Path(r"D:\stream-data\historical\binance\futures_um\BTCUSDT\klines_1m")
DEFAULT_OBJECT_ROOT = Path(r"D:\stream-data\historical\binance\futures_um\BTCUSDT\historical_structural_price_objects_v1")
DEFAULT_OUTPUT_ROOT = Path(r"D:\stream-data\historical\binance\futures_um\BTCUSDT\recent_swing_robustness_v1")
DEFAULT_ANALYSIS_START = "2020-02-01"
DEFAULT_ANALYSIS_END = "2025-12-31"

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
MAX_EVENTS_PER_OBJECT_DAY = 1
MATCH_EXPOSURE_DAYS = 30
EXPOSURE_BIN_BPS = 25.0
MATCH_ACTIVE_WINDOW_PCT = 0.25
CONTROL_COLLISION_BUFFER_BPS = 20.0
CONTROL_RANDOM_CANDIDATES = 256
CONFLUENCE_BUFFER_FLOOR_BPS = 10.0
RNG_SEED = 20260811
RECENT_SWING_CLUSTER_BPS = 25.0
PRIMARY_VARIANT = "RECENT_SWING_7D_60M"
RECENT_VARIANTS = ("RECENT_SWING_7D_60M", "RECENT_SWING_7D_240M", "RECENT_SWING_30D_60M", "RECENT_SWING_30D_240M")
STALE_BANDS_BY_WINDOW = {
    7: (("STALE_8_14D", 8, 14), ("STALE_15_30D", 15, 30)),
    30: (("STALE_31_60D", 31, 60),),
}
STALE_DISTANCE_MAX_BPS = 125.0
STALE_EXPOSURE_LOG1P_MAX = 0.75
STALE_WIDTH_DELTA_MAX_BPS = 25.0
STRICT_RANDOM_DISTANCE_MAX_BPS = 100.0
STRICT_RANDOM_EXPOSURE_LOG1P_MAX = 0.50
STRICT_RANDOM_WIDTH_DELTA_MAX_BPS = 5.0
STRICT_RANDOM_COLLISION_BUFFER_BPS = 10.0
STALE_COLLISION_BUFFER_BPS = 10.0

# Width-preserving shifted controls. Actual bps is max(floor, width_multiplier * source width).
SHIFT_CONTROL_SPECS = (
    ("SHIFT_NEG_NEAR", -1.0, 500.0, 2.0),
    ("SHIFT_POS_NEAR", +1.0, 500.0, 2.0),
    ("SHIFT_NEG_FAR", -1.0, 1000.0, 4.0),
    ("SHIFT_POS_FAR", +1.0, 1000.0, 4.0),
)

TOLERANCE_SPECS: dict[str, dict[str, Any]] = {
    "VOL_0P05": {"kind": "VOL", "value": 0.05, "min_bps": 5.0, "max_bps": 50.0, "primary": False},
    "VOL_0P10": {"kind": "VOL", "value": 0.10, "min_bps": 10.0, "max_bps": 75.0, "primary": True},
    "VOL_0P20": {"kind": "VOL", "value": 0.20, "min_bps": 15.0, "max_bps": 125.0, "primary": False},
    "FIXED_10": {"kind": "FIXED", "value": 10.0, "primary": False},
    "FIXED_20": {"kind": "FIXED", "value": 20.0, "primary": False},
    "FIXED_30": {"kind": "FIXED", "value": 30.0, "primary": False},
}

REQUIRED_SOURCE_COLUMNS = ["open_time", "open", "high", "low", "close", "quote_volume", "trade_count"]
REQUIRED_OBJECT_COLUMNS = [
    "snapshot_time", "snapshot_date", "object_state_id", "object_key", "object_type", "variant_id",
    "tier", "scale_minutes", "exposure_scheme", "side_balance", "zone_lower", "zone_upper",
    "zone_center", "zone_width_bps", "width_cap_bps", "width_capped", "anchor_count",
    "independent_episode_count", "independent_regime_count", "age_days", "days_since_last_anchor",
    "latest_information_time", "construction_note",
]
REQUIRED_ANCHOR_COLUMNS = [
    "anchor_id", "pivot_time", "available_time", "side", "scale_minutes",
    "pivot_price", "prominence_bps", "formation_sigma_bps",
]

EVENT_COLUMNS = [
    "run_id", "schema_version", "symbol", "event_id", "physical_event_id", "event_time", "event_date", "event_year",
    "tolerance_id", "tolerance_primary", "tolerance_kind", "tolerance_parameter", "tau_bps",
    "object_type", "variant_id", "tier", "scale_minutes", "exposure_scheme", "side_balance",
    "control_type", "control_variant", "object_state_id", "source_object_state_id", "source_object_key",
    "zone_lower", "zone_upper", "zone_center", "zone_width_bps", "source_zone_lower", "source_zone_upper",
    "source_zone_center", "source_zone_width_bps", "object_latest_information_time", "object_age_days",
    "width_capped", "construction_note", "anchor_count", "independent_episode_count", "independent_regime_count",
    "approach_from", "entry_edge", "far_edge", "event_open", "event_high", "event_low", "event_close",
    "trailing_sigma_bps", "approach_return_15m_bps", "approach_velocity_toward_bps_15m",
    "day_start_price", "signed_distance_from_day_start_bps", "abs_distance_from_day_start_bps",
    "source_abs_distance_from_day_start_bps", "recent_exposure_30d", "source_recent_exposure_30d",
    "confluence_count", "confluence_object_types", "confluence_variants", "is_confluence",
    "penetration_bps_15m", "penetration_vol_15m", "rejection_bps_15m", "rejection_vol_15m",
    "far_edge_penetration_bps_15m", "far_edge_penetration_vol_15m", "zone_traversal_fraction_15m",
    "penetration_bps_60m", "penetration_vol_60m", "rejection_bps_60m", "rejection_vol_60m",
    "far_edge_penetration_bps_60m", "far_edge_penetration_vol_60m", "zone_traversal_fraction_60m",
    "penetration_bps_240m", "penetration_vol_240m", "rejection_bps_240m", "rejection_vol_240m",
    "far_edge_penetration_bps_240m", "far_edge_penetration_vol_240m", "zone_traversal_fraction_240m",
    "inside_zone_close_fraction_60m", "beyond_far_edge_close_fraction_60m", "near_zone_minutes_60m",
    "final_state_60m", "exploratory_resolution_60m", "retest_within_240m",
    "outcome_complete_15m", "outcome_complete_60m", "outcome_complete_240m",
]


@dataclass(frozen=True)
class Zone:
    object_type: str
    variant_id: str
    tier: str
    lower: float
    upper: float
    center: float
    width_bps: float
    object_state_id: str
    object_key: str
    latest_information_time: datetime | None
    age_days: float | None
    scale_minutes: int | None
    exposure_scheme: str
    side_balance: str
    width_capped: bool
    construction_note: str
    anchor_count: int
    independent_episode_count: int
    independent_regime_count: int
    control_type: str = "REAL"
    control_variant: str = "REAL"
    source_object_state_id: str = ""
    source_object_key: str = ""
    source_lower: float | None = None
    source_upper: float | None = None
    source_center: float | None = None
    source_width_bps: float | None = None
    recent_exposure_30d: int | None = None
    source_recent_exposure_30d: int | None = None
    allow_prefix: bool = False


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


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def find_repo_root(script_path: Path) -> Path:
    p = script_path.resolve()
    for parent in [p.parent] + list(p.parents):
        if (parent / "scripts").exists() and (parent / "reports").exists():
            return parent
    # Expected location scripts/perps/historical/file.py.
    try:
        return p.parents[3]
    except Exception:
        return Path.cwd()


def short_tmp(path: Path, prefix: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.parent / f".{prefix}-{uuid.uuid4().hex[:8]}.tmp"


def atomic_write_text(path: Path, text: str) -> None:
    tmp = short_tmp(path, "tx")
    try:
        tmp.write_text(text, encoding="utf-8")
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


def atomic_write_dataframe_csv(path: Path, frame) -> None:
    tmp = short_tmp(path, "df")
    try:
        frame.to_csv(tmp, index=False)
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


def zone_width_bps(lower: float, upper: float) -> float:
    return bps_log_ratio(upper, lower)


def zone_center(lower: float, upper: float) -> float:
    return math.sqrt(lower * upper)


def shift_zone(lower: float, upper: float, shift_bps: float) -> tuple[float, float, float]:
    lo = exp_bps(lower, shift_bps)
    hi = exp_bps(upper, shift_bps)
    return lo, hi, zone_center(lo, hi)


def zone_from_center(center: float, width_bps_value: float) -> tuple[float, float]:
    half = max(width_bps_value, 0.01) / 2.0
    return exp_bps(center, -half), exp_bps(center, half)


def exposure_bin_idx(price: float) -> int:
    step = math.log1p(EXPOSURE_BIN_BPS / 10_000.0)
    return int(math.floor(math.log(price) / step))


def exposure_bin_center(idx: int) -> float:
    step = math.log1p(EXPOSURE_BIN_BPS / 10_000.0)
    return math.exp((idx + 0.5) * step)


def zone_gap_bps(a_lo: float, a_hi: float, b_lo: float, b_hi: float) -> float:
    if a_hi >= b_lo and b_hi >= a_lo:
        return 0.0
    if a_hi < b_lo:
        return max(0.0, bps_log_ratio(b_lo, a_hi))
    return max(0.0, bps_log_ratio(a_lo, b_hi))


def zones_collide(a_lo: float, a_hi: float, b_lo: float, b_hi: float, buffer_bps: float = 0.0) -> bool:
    return zone_gap_bps(a_lo, a_hi, b_lo, b_hi) <= buffer_bps


def close_distance_to_zone_bps(close, lowers, uppers):
    """Vectorized unsigned log-bps distance from closes to zones; zero inside."""
    import numpy as np
    c = np.asarray(close, dtype=float)[:, None]
    lo = np.asarray(lowers, dtype=float)[None, :]
    hi = np.asarray(uppers, dtype=float)[None, :]
    below = c < lo
    above = c > hi
    out = np.zeros((len(c), len(lowers)), dtype=float)
    if out.size == 0:
        return out
    out[below] = np.log((lo / c)[below]) * 10_000.0
    out[above] = np.log((c / hi)[above]) * 10_000.0
    return out


def bar_distance_to_zone_bps(high, low, lowers, uppers):
    """Vectorized unsigned distance from candle ranges to zones; zero if ranges overlap."""
    import numpy as np
    h = np.asarray(high, dtype=float)[:, None]
    l = np.asarray(low, dtype=float)[:, None]
    lo = np.asarray(lowers, dtype=float)[None, :]
    hi = np.asarray(uppers, dtype=float)[None, :]
    below = h < lo
    above = l > hi
    out = np.zeros((len(h), len(lowers)), dtype=float)
    if out.size == 0:
        return out
    out[below] = np.log((lo / h)[below]) * 10_000.0
    out[above] = np.log((l / hi)[above]) * 10_000.0
    return out


def range_hits_zone_with_tau(bar_high: float, bar_low: float, lower: float, upper: float, tau_bps_value: float) -> bool:
    lo = exp_bps(lower, -tau_bps_value)
    hi = exp_bps(upper, +tau_bps_value)
    return bar_high >= lo and bar_low <= hi


def verify_source_audit(root: Path) -> dict[str, Any]:
    path = root / "metadata" / "latest_audit_summary.json"
    if not path.exists():
        raise RuntimeError(f"Missing Pass-3 audit summary: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    hist = str(obj.get("historical_status", obj.get("status", ""))).upper()
    if hist != "PASS":
        raise RuntimeError(f"Historical audit is not PASS: {hist}")
    if int(obj.get("missing_minutes", 0)) != 0 or int(obj.get("duplicate_minutes", 0)) != 0:
        raise RuntimeError("Historical audit reports missing/duplicate minutes")
    return obj


def verify_object_build(root: Path) -> dict[str, Any]:
    path = root / "metadata" / "latest_summary.json"
    if not path.exists():
        raise RuntimeError(f"Missing Phase-5A summary: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if str(obj.get("status", "")).upper() != "PASS":
        raise RuntimeError(f"Phase 5A build is not PASS: {obj.get('status')}")
    failed = int(obj.get("failed_invariants", 0))
    if failed != 0:
        raise RuntimeError(f"Phase 5A reports {failed} failed invariants")
    version = str(obj.get("builder_version", ""))
    if version and tuple(int(x) for x in version.split(".")[:3]) < (1, 0, 2):
        raise RuntimeError(f"Phase 5A version {version} is older than required 1.0.2")
    return obj


def discover_market_files(root: Path, start: date, end: date) -> list[Path]:
    out = []
    d = date(start.year, start.month, 1)
    last = date(end.year, end.month, 1)
    while d <= last:
        p = root / "parquet" / f"symbol={DEFAULT_SYMBOL}" / f"year={d.year:04d}" / f"month={d.month:02d}" / "part-00000.parquet"
        if not p.exists():
            raise RuntimeError(f"Missing historical parquet month: {p}")
        out.append(p)
        if d.month == 12:
            d = date(d.year + 1, 1, 1)
        else:
            d = date(d.year, d.month + 1, 1)
    return out


def discover_object_files(root: Path, start: date, end: date) -> list[Path]:
    files = []
    for y in range(start.year, end.year + 1):
        p = root / "objects" / f"year={y}" / "part-00000.parquet"
        if not p.exists():
            raise RuntimeError(f"Missing Phase-5A object partition: {p}")
        files.append(p)
    return files


def source_signature(files: Sequence[Path], object_files: Sequence[Path], config: dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(config, sort_keys=True, default=str).encode())
    for p in list(files) + list(object_files):
        st = p.stat()
        h.update(str(p).encode())
        h.update(str(st.st_size).encode())
        h.update(str(st.st_mtime_ns).encode())
    return h.hexdigest()


def load_market(files: Sequence[Path], start: date, end: date):
    import pandas as pd
    parts = []
    for p in files:
        part = pd.read_parquet(p, columns=REQUIRED_SOURCE_COLUMNS)
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end + timedelta(days=1), tz="UTC")
    df = df[(df["open_time"] >= s) & (df["open_time"] < e)].copy()
    df = df.sort_values("open_time", kind="mergesort").reset_index(drop=True)
    return df


def load_objects(files: Sequence[Path], start: date, end: date):
    import pandas as pd
    parts = []
    for p in files:
        part = pd.read_parquet(p)
        missing = sorted(set(REQUIRED_OBJECT_COLUMNS) - set(part.columns))
        if missing:
            raise RuntimeError(f"Phase-5A object file {p} missing columns: {missing}")
        parts.append(part)
    obj = pd.concat(parts, ignore_index=True)
    obj["snapshot_time"] = pd.to_datetime(obj["snapshot_time"], utc=True)
    obj["latest_information_time"] = pd.to_datetime(obj["latest_information_time"], utc=True, errors="coerce")
    obj["snapshot_date"] = obj["snapshot_time"].dt.date
    obj = obj[(obj["snapshot_date"] >= start) & (obj["snapshot_date"] <= end)].copy()
    for c in ("zone_lower", "zone_upper", "zone_center", "zone_width_bps", "age_days"):
        obj[c] = obj[c].astype(float)
    return obj.sort_values(["snapshot_time", "object_type", "variant_id", "zone_center"], kind="mergesort").reset_index(drop=True)



def load_anchor_archive(object_root: Path):
    import pandas as pd
    path = object_root / "anchors" / "structural_anchors.parquet"
    if not path.exists():
        raise RuntimeError(f"Missing Phase-5A anchor archive: {path}")
    anchors = pd.read_parquet(path)
    missing = sorted(set(REQUIRED_ANCHOR_COLUMNS) - set(anchors.columns))
    if missing:
        raise RuntimeError(f"Anchor archive missing columns: {missing}")
    anchors = anchors[REQUIRED_ANCHOR_COLUMNS].copy()
    anchors["pivot_time"] = pd.to_datetime(anchors["pivot_time"], utc=True)
    anchors["available_time"] = pd.to_datetime(anchors["available_time"], utc=True)
    anchors["pivot_price"] = anchors["pivot_price"].astype(float)
    anchors["scale_minutes"] = anchors["scale_minutes"].astype(int)
    return anchors.sort_values(["scale_minutes", "pivot_time", "pivot_price"], kind="mergesort").reset_index(drop=True)


def parse_recent_variant(variant_id: str) -> tuple[int, int]:
    m = re.fullmatch(r"RECENT_SWING_(\d+)D_(\d+)M", str(variant_id))
    if not m:
        raise ValueError(f"Not a RECENT_SWING variant: {variant_id}")
    return int(m.group(1)), int(m.group(2))


def complete_linkage_clusters(anchor_rows, max_width_bps: float = RECENT_SWING_CLUSTER_BPS):
    """Return anchor-row clusters sorted by price; cluster diameter never exceeds max_width_bps."""
    if anchor_rows.empty:
        return []
    sub = anchor_rows.sort_values("pivot_price", kind="mergesort")
    clusters = []
    current = []
    current_min = None
    for row in sub.itertuples(index=False):
        p = float(row.pivot_price)
        if not current:
            current = [row]; current_min = p; continue
        if abs(bps_log_ratio(p, float(current_min))) <= max_width_bps + 1e-9:
            current.append(row)
        else:
            clusters.append(current); current = [row]; current_min = p
    if current:
        clusters.append(current)
    return clusters


def stale_zone_pool(snapshot_day: date, scale: int, band_label: str, age_lo: int, age_hi: int,
                    anchors, exposure: Counter) -> list[Zone]:
    import numpy as np
    import pandas as pd
    snap = pd.Timestamp(datetime.combine(snapshot_day, datetime.min.time(), tzinfo=timezone.utc))
    # age_lo=8, age_hi=14 means pivot_time in [T-14d, T-8d]. Both bounds are historical.
    lo_time = snap - pd.Timedelta(days=age_hi)
    hi_time = snap - pd.Timedelta(days=age_lo)
    sub = anchors[(anchors["scale_minutes"] == scale)
                  & (anchors["available_time"] < snap)
                  & (anchors["pivot_time"] >= lo_time)
                  & (anchors["pivot_time"] < hi_time)].copy()
    out = []
    for cluster in complete_linkage_clusters(sub):
        prices = np.asarray([float(r.pivot_price) for r in cluster], dtype=float)
        lower = float(np.min(prices)); upper = float(np.max(prices))
        center = float(np.median(prices))
        if upper <= lower:
            lower, upper = zone_from_center(center, 5.0)
        supp = sum(1 for r in cluster if str(r.side) == "SUPPORT")
        res = len(cluster) - supp
        side_balance = "TWO_SIDED"
        if supp and not res: side_balance = "SUPPORT_HEAVY"
        elif res and not supp: side_balance = "RESISTANCE_HEAVY"
        member_ids = sorted(str(r.anchor_id) for r in cluster)
        mh = hashlib.sha1("|".join(member_ids).encode()).hexdigest()[:12]
        oid = hashlib.sha1(f"STALE|{snapshot_day}|{band_label}|{scale}|{lower:.8f}|{upper:.8f}|{mh}".encode()).hexdigest()[:24]
        latest = max(pd.Timestamp(r.available_time).to_pydatetime() for r in cluster)
        first_pivot = min(pd.Timestamp(r.pivot_time).to_pydatetime() for r in cluster)
        out.append(Zone(
            object_type="RECENT_SWING", variant_id=f"STALE_POOL_{band_label}_{scale}M", tier="CONTROL",
            lower=lower, upper=upper, center=center, width_bps=zone_width_bps(lower, upper),
            object_state_id=oid, object_key=f"{band_label}|{scale}|{mh}", latest_information_time=latest,
            age_days=(datetime.combine(snapshot_day, datetime.min.time(), tzinfo=timezone.utc)-first_pivot).total_seconds()/86400.0,
            scale_minutes=scale, exposure_scheme="NONE", side_balance=side_balance, width_capped=False,
            construction_note=f"Genuine stale confirmed swing cluster: {band_label}", anchor_count=len(cluster),
            independent_episode_count=1, independent_regime_count=1,
            control_type="STALE_SWING_ZONE", control_variant=band_label,
            recent_exposure_30d=int(exposure.get(exposure_bin_idx(center), 0)), allow_prefix=True,
        ))
    return out


def same_side(source_center: float, candidate_center: float, day_start_price: float) -> bool:
    s = bps_log_ratio(source_center, day_start_price)
    c = bps_log_ratio(candidate_center, day_start_price)
    if abs(s) < EXPOSURE_BIN_BPS:
        return abs(c) < 2 * EXPOSURE_BIN_BPS
    return (s < 0 and c < 0) or (s > 0 and c > 0)


def match_score(source: Zone, candidate: Zone, day_start_price: float, exposure: Counter):
    src_dist = bps_log_ratio(source.center, day_start_price)
    ctl_dist = bps_log_ratio(candidate.center, day_start_price)
    src_exp = int(exposure.get(exposure_bin_idx(source.center), 0))
    ctl_exp = int(exposure.get(exposure_bin_idx(candidate.center), 0))
    dist_delta = abs(ctl_dist - src_dist)
    exp_delta = abs(math.log1p(ctl_exp) - math.log1p(src_exp))
    width_delta = abs(candidate.width_bps - source.width_bps)
    score = dist_delta / 50.0 + 2.0 * exp_delta + width_delta / 25.0
    return score, dist_delta, exp_delta, width_delta, src_exp, ctl_exp


def stale_collision_reference(source: Zone, real_zones: Sequence[Zone]) -> list[Zone]:
    """Only same-hypothesis active zones invalidate a stale control.

    Example: an 8-14d-old 1h swing is intentionally stale relative to the 7d x 1h
    hypothesis even though it is still contained in the broader 30d x 1h archive.
    Rejecting it for overlapping any 30d object makes the preregistered stale test
    impossible by construction. Other-variant overlap is preserved as confluence.
    """
    return [z for z in real_zones if z.variant_id == source.variant_id]


def choose_stale_control(source: Zone, pool: Sequence[Zone], real_zones: Sequence[Zone], reserved,
                         day_start_price: float, exposure: Counter, band_label: str) -> Zone | None:
    eligible = []
    for cand in pool:
        if not same_side(source.center, cand.center, day_start_price):
            continue
        if is_collision(cand.lower, cand.upper, real_zones, reserved):
            continue
        score, dd, ed, wd, se, ce = match_score(source, cand, day_start_price, exposure)
        if dd > STALE_DISTANCE_MAX_BPS or ed > STALE_EXPOSURE_LOG1P_MAX or wd > STALE_WIDTH_DELTA_MAX_BPS:
            continue
        eligible.append((score, cand, se, ce))
    if not eligible:
        return None
    _, cand, src_exp, ctl_exp = min(eligible, key=lambda x: (x[0], x[1].center))
    reserved.append((cand.lower, cand.upper))
    oid = hashlib.sha1(f"STALEMATCH|{source.object_state_id}|{band_label}|{cand.object_state_id}".encode()).hexdigest()[:24]
    return Zone(
        object_type=source.object_type, variant_id=source.variant_id, tier=source.tier,
        lower=cand.lower, upper=cand.upper, center=cand.center, width_bps=cand.width_bps,
        object_state_id=oid, object_key=f"{band_label}|{source.object_key}|{cand.object_key}",
        latest_information_time=cand.latest_information_time, age_days=cand.age_days,
        scale_minutes=source.scale_minutes, exposure_scheme="NONE", side_balance=cand.side_balance,
        width_capped=False, construction_note=cand.construction_note, anchor_count=cand.anchor_count,
        independent_episode_count=1, independent_regime_count=1,
        control_type="STALE_SWING_ZONE", control_variant=band_label,
        source_object_state_id=source.object_state_id, source_object_key=source.object_key,
        source_lower=source.lower, source_upper=source.upper, source_center=source.center,
        source_width_bps=source.width_bps, recent_exposure_30d=ctl_exp,
        source_recent_exposure_30d=src_exp, allow_prefix=True,
    )


def strict_random_control(source: Zone, d: date, day_start_price: float, exposure: Counter,
                          real_zones: Sequence[Zone], reserved) -> Zone | None:
    src_dist = bps_log_ratio(source.center, day_start_price)
    src_exp = int(exposure.get(exposure_bin_idx(source.center), 0))
    # Exhaustively inspect the small distance-matched band at 25-bps storage resolution.
    target_price = source.center
    idx0 = exposure_bin_idx(exp_bps(target_price, -STRICT_RANDOM_DISTANCE_MAX_BPS))
    idx1 = exposure_bin_idx(exp_bps(target_price, +STRICT_RANDOM_DISTANCE_MAX_BPS))
    candidates = []
    for idx in range(min(idx0, idx1), max(idx0, idx1) + 1):
        center = exposure_bin_center(idx)
        if not same_side(source.center, center, day_start_price):
            continue
        lower, upper = zone_from_center(center, source.width_bps)
        if lower <= 0 or zones_collide(lower, upper, source.lower, source.upper, STRICT_RANDOM_COLLISION_BUFFER_BPS):
            continue
        if is_collision(lower, upper, real_zones, reserved):
            continue
        ctl_dist = bps_log_ratio(center, day_start_price)
        ctl_exp = int(exposure.get(idx, 0))
        dd = abs(ctl_dist - src_dist)
        ed = abs(math.log1p(ctl_exp) - math.log1p(src_exp))
        wd = abs(zone_width_bps(lower, upper) - source.width_bps)
        if dd > STRICT_RANDOM_DISTANCE_MAX_BPS or ed > STRICT_RANDOM_EXPOSURE_LOG1P_MAX or wd > STRICT_RANDOM_WIDTH_DELTA_MAX_BPS:
            continue
        # Hash tie-break avoids always choosing one side while staying deterministic.
        tie = stable_hash(f"{d}|{source.object_state_id}|{idx}") / 2**64
        score = dd / 25.0 + 3.0 * ed + tie * 1e-6
        candidates.append((score, lower, upper, center, ctl_exp))
    if not candidates:
        return None
    _, lower, upper, center, ctl_exp = min(candidates, key=lambda x: x[0])
    reserved.append((lower, upper))
    oid = hashlib.sha1(f"STRICTRAND|{d}|{source.object_state_id}|{lower:.8f}|{upper:.8f}".encode()).hexdigest()[:24]
    return Zone(
        object_type=source.object_type, variant_id=source.variant_id, tier=source.tier,
        lower=lower, upper=upper, center=center, width_bps=source.width_bps,
        object_state_id=oid, object_key=f"STRICTRAND|{source.object_key}", latest_information_time=None,
        age_days=source.age_days, scale_minutes=source.scale_minutes, exposure_scheme="NONE",
        side_balance=source.side_balance, width_capped=False, construction_note="STRICT_MATCHED_RANDOM_ZONE",
        anchor_count=source.anchor_count, independent_episode_count=1, independent_regime_count=1,
        control_type="STRICT_MATCHED_RANDOM_ZONE", control_variant="STRICT_MATCHED_RANDOM_ZONE",
        source_object_state_id=source.object_state_id, source_object_key=source.object_key,
        source_lower=source.lower, source_upper=source.upper, source_center=source.center,
        source_width_bps=source.width_bps, recent_exposure_30d=ctl_exp,
        source_recent_exposure_30d=src_exp, allow_prefix=False,
    )


def prepare_market_arrays(df):
    import numpy as np
    import pandas as pd
    close = df["close"].to_numpy(dtype=float)
    logret = np.full(len(df), np.nan, dtype=float)
    if len(close) > 1:
        logret[1:] = np.diff(np.log(close))
    sq = pd.Series(logret * logret)
    # shift(1): event minute cannot affect its own sigma.
    sigma = np.sqrt(sq.rolling(TRAILING_VOL_MINUTES, min_periods=TRAILING_VOL_MIN_PERIODS).sum()).shift(1).to_numpy(dtype=float) * 10_000.0
    return sigma


def build_day_index(df) -> dict[date, tuple[int, int]]:
    import pandas as pd
    day = pd.to_datetime(df["open_time"], utc=True).dt.date.to_numpy()
    out: dict[date, tuple[int, int]] = {}
    if not len(day):
        return out
    start = 0
    current = day[0]
    for i in range(1, len(day) + 1):
        if i == len(day) or day[i] != current:
            out[current] = (start, i)
            if i < len(day):
                start = i
                current = day[i]
    return out


def build_daily_exposure_counts(df) -> dict[date, Counter]:
    import pandas as pd
    tmp = pd.DataFrame({"day": pd.to_datetime(df["open_time"], utc=True).dt.date, "close": df["close"].astype(float)})
    out: dict[date, Counter] = {}
    for d, g in tmp.groupby("day", sort=True):
        out[d] = Counter(exposure_bin_idx(float(x)) for x in g["close"].to_numpy())
    return out


def trailing_exposure_counter(d: date, daily: dict[date, Counter]) -> Counter:
    out: Counter = Counter()
    for k in range(1, MATCH_EXPOSURE_DAYS + 1):
        out.update(daily.get(d - timedelta(days=k), {}))
    return out


def object_rows_by_day(objects) -> dict[date, list[Zone]]:
    import pandas as pd
    out: dict[date, list[Zone]] = {}
    prev_by_identity: dict[tuple[str, str, str], Zone] = {}
    for d, g in objects.groupby("snapshot_date", sort=True):
        zones: list[Zone] = []
        for r in g.itertuples(index=False):
            latest = r.latest_information_time
            if pd.isna(latest):
                latest_dt = None
            else:
                latest_dt = pd.Timestamp(latest).to_pydatetime()
            ident = (str(r.object_type), str(r.variant_id), str(r.object_key))
            prev = prev_by_identity.get(ident)
            allow_prefix = False
            if prev is not None:
                # Persisting object can borrow the prior 60-minute arming state only if geometry
                # remained materially the same. This avoids granting pre-history to new zones.
                inter_lo = max(float(r.zone_lower), prev.lower)
                inter_hi = min(float(r.zone_upper), prev.upper)
                if inter_hi > inter_lo:
                    inter = zone_width_bps(inter_lo, inter_hi)
                    union = zone_width_bps(min(float(r.zone_lower), prev.lower), max(float(r.zone_upper), prev.upper))
                    iou = inter / union if union > 0 else 0.0
                    allow_prefix = iou >= 0.50
                else:
                    center_delta = abs(bps_log_ratio(float(r.zone_center), prev.center))
                    allow_prefix = center_delta <= 10.0
            z = Zone(
                object_type=str(r.object_type), variant_id=str(r.variant_id), tier=str(r.tier),
                lower=float(r.zone_lower), upper=float(r.zone_upper), center=float(r.zone_center),
                width_bps=float(r.zone_width_bps), object_state_id=str(r.object_state_id), object_key=str(r.object_key),
                latest_information_time=latest_dt, age_days=float(r.age_days) if math.isfinite(float(r.age_days)) else None,
                scale_minutes=int(r.scale_minutes) if not pd.isna(r.scale_minutes) else None,
                exposure_scheme=str(r.exposure_scheme) if not pd.isna(r.exposure_scheme) else "",
                side_balance=str(r.side_balance) if not pd.isna(r.side_balance) else "",
                width_capped=bool(r.width_capped), construction_note=str(r.construction_note),
                anchor_count=int(r.anchor_count) if not pd.isna(r.anchor_count) else 0,
                independent_episode_count=int(r.independent_episode_count) if not pd.isna(r.independent_episode_count) else 0,
                independent_regime_count=int(r.independent_regime_count) if not pd.isna(r.independent_regime_count) else 0,
                source_object_state_id=str(r.object_state_id), source_object_key=str(r.object_key),
                source_lower=float(r.zone_lower), source_upper=float(r.zone_upper), source_center=float(r.zone_center),
                source_width_bps=float(r.zone_width_bps), allow_prefix=allow_prefix,
            )
            zones.append(z)
            prev_by_identity[ident] = z
        out[d] = zones
    return out


def tau_array(spec: dict[str, Any], sigma_bps):
    import numpy as np
    if spec["kind"] == "FIXED":
        return np.full(len(sigma_bps), float(spec["value"]), dtype=float)
    tau = np.asarray(sigma_bps, dtype=float) * float(spec["value"])
    return np.clip(tau, float(spec["min_bps"]), float(spec["max_bps"]))


def is_collision(lower: float, upper: float, real_zones: Sequence[Zone], reserved: Sequence[tuple[float, float]] = ()) -> bool:
    for z in real_zones:
        if zones_collide(lower, upper, z.lower, z.upper, CONTROL_COLLISION_BUFFER_BPS):
            return True
    for lo, hi in reserved:
        if zones_collide(lower, upper, lo, hi, CONTROL_COLLISION_BUFFER_BPS):
            return True
    return False


def matched_random_zone(source: Zone, d: date, day_start_price: float, exposure: Counter,
                        real_zones: Sequence[Zone], reserved: list[tuple[float, float]]) -> Zone | None:
    import random
    if day_start_price <= 0:
        return None
    rng = random.Random(RNG_SEED ^ stable_hash(f"{source.object_state_id}|{d.isoformat()}"))
    low_price = day_start_price * (1.0 - MATCH_ACTIVE_WINDOW_PCT)
    high_price = day_start_price * (1.0 + MATCH_ACTIVE_WINDOW_PCT)
    lo_idx = exposure_bin_idx(low_price)
    hi_idx = exposure_bin_idx(high_price)
    src_dist = bps_log_ratio(source.center, day_start_price)
    src_exp = int(exposure.get(exposure_bin_idx(source.center), 0))
    src_sign = 0 if abs(src_dist) < EXPOSURE_BIN_BPS else (-1 if src_dist < 0 else 1)
    candidates = []
    for _ in range(CONTROL_RANDOM_CANDIDATES):
        idx = rng.randint(lo_idx, hi_idx)
        center = exposure_bin_center(idx)
        dist = bps_log_ratio(center, day_start_price)
        if src_sign != 0 and ((dist < 0 and src_sign > 0) or (dist > 0 and src_sign < 0)):
            continue
        lower, upper = zone_from_center(center, source.width_bps)
        if lower <= 0 or is_collision(lower, upper, real_zones, reserved):
            continue
        exp_count = int(exposure.get(idx, 0))
        dist_score = abs(dist - src_dist) / 50.0
        exp_score = abs(math.log1p(exp_count) - math.log1p(src_exp))
        score = dist_score + exp_score
        candidates.append((score, lower, upper, center, exp_count))
    if not candidates:
        return None
    _, lower, upper, center, exp_count = min(candidates, key=lambda x: x[0])
    reserved.append((lower, upper))
    oid = hashlib.sha1(f"MATCHED|{d}|{source.object_state_id}|{lower:.8f}|{upper:.8f}".encode()).hexdigest()[:24]
    return Zone(
        object_type=source.object_type, variant_id=source.variant_id, tier=source.tier,
        lower=lower, upper=upper, center=center, width_bps=source.width_bps,
        object_state_id=oid, object_key=f"MATCHED|{source.object_key}", latest_information_time=None,
        age_days=source.age_days, scale_minutes=source.scale_minutes, exposure_scheme=source.exposure_scheme,
        side_balance=source.side_balance, width_capped=source.width_capped,
        construction_note="MATCHED_RANDOM_ZONE", anchor_count=source.anchor_count,
        independent_episode_count=source.independent_episode_count,
        independent_regime_count=source.independent_regime_count,
        control_type="MATCHED_RANDOM_ZONE", control_variant="MATCHED_RANDOM_ZONE",
        source_object_state_id=source.object_state_id, source_object_key=source.object_key,
        source_lower=source.lower, source_upper=source.upper, source_center=source.center,
        source_width_bps=source.width_bps, recent_exposure_30d=exp_count,
        source_recent_exposure_30d=src_exp, allow_prefix=False,
    )


def controls_for_day(d: date, real_zones: Sequence[Zone], day_start_price: float, exposure: Counter,
                     anchors) -> list[Zone]:
    controls: list[Zone] = []
    pools: dict[tuple[int, str], list[Zone]] = {}
    reserved_by_variant: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for src in real_zones:
        window_days, scale = parse_recent_variant(src.variant_id)
        src_exp = int(exposure.get(exposure_bin_idx(src.center), 0))
        src = Zone(**{**asdict(src), "recent_exposure_30d": src_exp, "source_recent_exposure_30d": src_exp})
        reserved = reserved_by_variant[(src.variant_id, "ALL")]
        for band_label, age_lo, age_hi in STALE_BANDS_BY_WINDOW[window_days]:
            key = (scale, band_label)
            if key not in pools:
                pools[key] = stale_zone_pool(d, scale, band_label, age_lo, age_hi, anchors, exposure)
            ctl = choose_stale_control(src, pools[key], stale_collision_reference(src, real_zones), reserved, day_start_price, exposure, band_label)
            if ctl is not None:
                controls.append(ctl)
        rand = strict_random_control(src, d, day_start_price, exposure, real_zones, reserved)
        if rand is not None:
            controls.append(rand)
    return controls


def first_approach_indices(close, high, low, lowers, uppers, tau, prefix_rows: int = 0, allow_prefix=None):
    """First zone approach after 60 consecutive far closes. Returns local row index per zone."""
    import numpy as np
    m = len(lowers)
    if m == 0:
        return np.asarray([], dtype=int)
    dist_close = close_distance_to_zone_bps(close, lowers, uppers)
    dist_bar = bar_distance_to_zone_bps(high, low, lowers, uppers)
    tau2 = np.asarray(tau, dtype=float)[:, None]
    far = dist_close >= ARM_DISTANCE_MULTIPLIER * tau2
    near = dist_bar <= tau2
    n = len(close)
    run60 = np.zeros((n, m), dtype=bool)
    if n >= ARM_CONSECUTIVE_MINUTES:
        # Rolling all-true via cumulative integer sums.
        cs = np.cumsum(far.astype(np.int16), axis=0)
        sums = cs[ARM_CONSECUTIVE_MINUTES - 1:].copy()
        if n > ARM_CONSECUTIVE_MINUTES:
            sums[1:] -= cs[:-ARM_CONSECUTIVE_MINUTES]
        run60[ARM_CONSECUTIVE_MINUTES - 1:] = sums == ARM_CONSECUTIVE_MINUTES
    armed_seen = np.maximum.accumulate(run60, axis=0)
    armed_before = np.vstack([np.zeros((1, m), dtype=bool), armed_seen[:-1]])
    if prefix_rows > 0 and allow_prefix is not None:
        allow_prefix = np.asarray(allow_prefix, dtype=bool)
        armed_before[:prefix_rows, :] = False
        # After day-start, prefix evidence only counts for persisting objects/controls.
        for j in range(m):
            if not allow_prefix[j]:
                armed_before[:prefix_rows + ARM_CONSECUTIVE_MINUTES, j] = False
    trigger = near & armed_before
    if prefix_rows:
        trigger[:prefix_rows] = False
    first = np.full(m, -1, dtype=int)
    any_hit = trigger.any(axis=0)
    if np.any(any_hit):
        first[any_hit] = trigger[:, any_hit].argmax(axis=0)
    return first


def approach_side(prev_close: float, lower: float, upper: float, center: float) -> str:
    if prev_close < lower:
        return "BELOW"
    if prev_close > upper:
        return "ABOVE"
    return "BELOW" if prev_close <= center else "ABOVE"


def outcome_for_event(global_idx: int, zone: Zone, approach_from: str, sigma_bps: float,
                      tau_bps_value: float, highs, lows, closes) -> dict[str, Any]:
    import numpy as np
    entry = zone.lower if approach_from == "BELOW" else zone.upper
    far = zone.upper if approach_from == "BELOW" else zone.lower
    width = max(zone.width_bps, 1e-9)
    result: dict[str, Any] = {}
    windows: dict[int, tuple[Any, Any, Any]] = {}
    for horizon in (15, 60, 240):
        end = min(len(closes), global_idx + horizon)
        hh = np.asarray(highs[global_idx:end], dtype=float)
        ll = np.asarray(lows[global_idx:end], dtype=float)
        cc = np.asarray(closes[global_idx:end], dtype=float)
        complete = len(cc) == horizon
        result[f"outcome_complete_{horizon}m"] = bool(complete)
        windows[horizon] = (hh, ll, cc)
        if len(cc) == 0:
            for stem in ("penetration_bps", "penetration_vol", "rejection_bps", "rejection_vol",
                         "far_edge_penetration_bps", "far_edge_penetration_vol", "zone_traversal_fraction"):
                result[f"{stem}_{horizon}m"] = math.nan
            continue
        if approach_from == "BELOW":
            pen = max(0.0, bps_log_ratio(float(np.nanmax(hh)), entry))
            rej = max(0.0, bps_log_ratio(entry, float(np.nanmin(ll))))
            far_pen = max(0.0, bps_log_ratio(float(np.nanmax(hh)), far))
        else:
            pen = max(0.0, bps_log_ratio(entry, float(np.nanmin(ll))))
            rej = max(0.0, bps_log_ratio(float(np.nanmax(hh)), entry))
            far_pen = max(0.0, bps_log_ratio(far, float(np.nanmin(ll))))
        result[f"penetration_bps_{horizon}m"] = pen
        result[f"penetration_vol_{horizon}m"] = pen / sigma_bps if sigma_bps > 0 else math.nan
        result[f"rejection_bps_{horizon}m"] = rej
        result[f"rejection_vol_{horizon}m"] = rej / sigma_bps if sigma_bps > 0 else math.nan
        result[f"far_edge_penetration_bps_{horizon}m"] = far_pen
        result[f"far_edge_penetration_vol_{horizon}m"] = far_pen / sigma_bps if sigma_bps > 0 else math.nan
        result[f"zone_traversal_fraction_{horizon}m"] = min(5.0, pen / width)

    hh60, ll60, cc60 = windows[60]
    if len(cc60):
        inside = (cc60 >= zone.lower) & (cc60 <= zone.upper)
        if approach_from == "BELOW":
            beyond = cc60 > zone.upper
            final_state = "BEYOND" if cc60[-1] > zone.upper else "INSIDE" if zone.lower <= cc60[-1] <= zone.upper else "ORIGIN"
        else:
            beyond = cc60 < zone.lower
            final_state = "BEYOND" if cc60[-1] < zone.lower else "INSIDE" if zone.lower <= cc60[-1] <= zone.upper else "ORIGIN"
        dist = close_distance_to_zone_bps(cc60, [zone.lower], [zone.upper])[:, 0]
        result["inside_zone_close_fraction_60m"] = float(np.mean(inside))
        result["beyond_far_edge_close_fraction_60m"] = float(np.mean(beyond))
        result["near_zone_minutes_60m"] = int(np.sum(dist <= tau_bps_value))
        result["final_state_60m"] = final_state
    else:
        result["inside_zone_close_fraction_60m"] = math.nan
        result["beyond_far_edge_close_fraction_60m"] = math.nan
        result["near_zone_minutes_60m"] = 0
        result["final_state_60m"] = "UNKNOWN"

    penv = result.get("penetration_vol_60m", math.nan)
    rejv = result.get("rejection_vol_60m", math.nan)
    farv = result.get("far_edge_penetration_vol_60m", math.nan)
    beyond_frac = result.get("beyond_far_edge_close_fraction_60m", math.nan)
    inside_frac = result.get("inside_zone_close_fraction_60m", math.nan)
    if not all(math.isfinite(x) for x in (penv, rejv, farv, beyond_frac, inside_frac)):
        resolution = "INCOMPLETE"
    elif penv < 0.15 and rejv >= 0.25 and inside_frac <= 0.25:
        resolution = "REJECTION"
    elif farv >= 0.25 and beyond_frac >= 0.50:
        resolution = "CLEAN_TRAVERSAL"
    elif penv >= 0.15 and result["final_state_60m"] == "ORIGIN":
        resolution = "ENTER_RETURN"
    elif inside_frac >= 0.50:
        resolution = "ACCEPTANCE_DWELL"
    else:
        resolution = "STALL_OTHER"
    result["exploratory_resolution_60m"] = resolution

    # Retest after first decisive far-edge break.
    _, _, cc240 = windows[240]
    retest = False
    if len(cc240) and sigma_bps > 0:
        if approach_from == "BELOW":
            threshold = exp_bps(zone.upper, 0.25 * sigma_bps)
            beyond_idx = (cc240 >= threshold).nonzero()[0]
        else:
            threshold = exp_bps(zone.lower, -0.25 * sigma_bps)
            beyond_idx = (cc240 <= threshold).nonzero()[0]
        if len(beyond_idx):
            j = int(beyond_idx[0]) + 1
            for x in cc240[j:]:
                if zone.lower <= x <= zone.upper or min(abs(bps_log_ratio(x, zone.lower)), abs(bps_log_ratio(x, zone.upper))) <= tau_bps_value:
                    retest = True
                    break
    result["retest_within_240m"] = bool(retest)
    return result


def confluence_members(event_idx: int, tau_bps_value: float, high: float, low: float,
                       real_zones: Sequence[Zone]) -> list[Zone]:
    _ = event_idx
    members = []
    buffer_bps = max(CONFLUENCE_BUFFER_FLOOR_BPS, tau_bps_value)
    for z in real_zones:
        if range_hits_zone_with_tau(high, low, z.lower, z.upper, buffer_bps):
            members.append(z)
    return members


def candidate_near_day_range(z: Zone, day_low: float, day_high: float, max_tau_bps: float) -> bool:
    expanded_low = exp_bps(day_low, -max_tau_bps)
    expanded_high = exp_bps(day_high, +max_tau_bps)
    return z.upper >= expanded_low and z.lower <= expanded_high


def process_year(year: int, df, sigma_all, day_index: dict[date, tuple[int, int]],
                 zones_by_day: dict[date, list[Zone]], daily_exposure: dict[date, Counter], anchors,
                 tolerance_ids: Sequence[str], analysis_start: date, analysis_end: date,
                 run_id: str):
    import numpy as np
    import pandas as pd

    opens_all = df["open"].to_numpy(dtype=float)
    highs_all = df["high"].to_numpy(dtype=float)
    lows_all = df["low"].to_numpy(dtype=float)
    closes_all = df["close"].to_numpy(dtype=float)

    rows: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    start = max(analysis_start, date(year, 1, 1))
    end = min(analysis_end, date(year, 12, 31))
    d = start
    processed = 0
    while d <= end:
        if d not in day_index or d not in zones_by_day:
            d += timedelta(days=1)
            continue
        s, e = day_index[d]
        if s <= 0 or e <= s:
            d += timedelta(days=1)
            continue
        day_start_price = float(closes_all[s - 1])
        exposure = trailing_exposure_counter(d, daily_exposure)
        real0 = zones_by_day[d]
        # Attach causal trailing exposure to frozen objects.
        real_zones = [Zone(**{**asdict(z),
            "recent_exposure_30d": int(exposure.get(exposure_bin_idx(z.center), 0)),
            "source_recent_exposure_30d": int(exposure.get(exposure_bin_idx(z.center), 0)),
        }) for z in real0]
        controls = controls_for_day(d, real_zones, day_start_price, exposure, anchors)
        all_candidates = real_zones + controls

        inv_count = Counter((z.object_type, z.variant_id, z.control_type, z.control_variant) for z in all_candidates)
        for (otype, variant, ctype, cvariant), n in sorted(inv_count.items()):
            inventory.append({
                "year": year, "date": d.isoformat(), "object_type": otype, "variant_id": variant,
                "control_type": ctype, "control_variant": cvariant, "candidate_objects": int(n),
            })

        if not all_candidates:
            d += timedelta(days=1)
            continue
        day_low = float(np.nanmin(lows_all[s:e]))
        day_high = float(np.nanmax(highs_all[s:e]))
        max_tau = max(float(TOLERANCE_SPECS[t].get("max_bps", TOLERANCE_SPECS[t]["value"])) for t in tolerance_ids)
        candidates = [z for z in all_candidates if candidate_near_day_range(z, day_low, day_high, max_tau)]
        if not candidates:
            d += timedelta(days=1)
            continue

        prefix_s = max(0, s - ARM_CONSECUTIVE_MINUTES)
        prefix_rows = s - prefix_s
        close = closes_all[prefix_s:e]
        high = highs_all[prefix_s:e]
        low = lows_all[prefix_s:e]
        sig = sigma_all[prefix_s:e]
        lowers = np.asarray([z.lower for z in candidates], dtype=float)
        uppers = np.asarray([z.upper for z in candidates], dtype=float)
        allow_prefix = np.asarray([z.allow_prefix for z in candidates], dtype=bool)

        for tol_id in tolerance_ids:
            spec = TOLERANCE_SPECS[tol_id]
            tau = tau_array(spec, sig)
            first = first_approach_indices(close, high, low, lowers, uppers, tau, prefix_rows, allow_prefix)
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
                z = candidates[j]
                prev_idx = max(s, g - 1)
                app = approach_side(float(closes_all[prev_idx]), z.lower, z.upper, z.center)
                entry = z.lower if app == "BELOW" else z.upper
                far = z.upper if app == "BELOW" else z.lower
                if g >= 16:
                    ar15 = bps_log_ratio(float(closes_all[g - 1]), float(closes_all[g - 16]))
                else:
                    ar15 = math.nan
                toward = ar15 if app == "BELOW" else (-ar15 if math.isfinite(ar15) else math.nan)
                event_dt = pd.Timestamp(df["open_time"].iloc[g]).to_pydatetime()

                if z.control_type == "REAL":
                    members = confluence_members(g, tau_v, float(highs_all[g]), float(lows_all[g]), real_zones)
                    member_ids = sorted(x.object_state_id for x in members)
                    types = sorted({x.object_type for x in members})
                    variants = sorted({x.variant_id for x in members})
                    physical_raw = f"{event_dt.isoformat()}|{tol_id}|{'|'.join(member_ids)}"
                else:
                    members = []
                    types = []
                    variants = []
                    physical_raw = f"CONTROL|{event_dt.isoformat()}|{tol_id}|{z.object_state_id}"
                physical_id = hashlib.sha256(physical_raw.encode()).hexdigest()[:24]

                out = outcome_for_event(g, z, app, sigma, tau_v, highs_all, lows_all, closes_all)
                source_center = z.source_center if z.source_center is not None else z.center
                src_abs = abs(bps_log_ratio(source_center, day_start_price))
                eid_text = f"{z.object_state_id}|{event_dt.isoformat()}|{tol_id}|{z.control_type}|{z.control_variant}"
                latest = z.latest_information_time
                row = {
                    "run_id": run_id, "schema_version": SCHEMA_VERSION, "symbol": DEFAULT_SYMBOL,
                    "event_id": hashlib.sha256(eid_text.encode()).hexdigest()[:24],
                    "physical_event_id": physical_id, "event_time": event_dt, "event_date": d.isoformat(), "event_year": year,
                    "tolerance_id": tol_id, "tolerance_primary": bool(spec.get("primary", False)),
                    "tolerance_kind": spec["kind"], "tolerance_parameter": float(spec["value"]), "tau_bps": tau_v,
                    "object_type": z.object_type, "variant_id": z.variant_id, "tier": z.tier,
                    "scale_minutes": z.scale_minutes, "exposure_scheme": z.exposure_scheme, "side_balance": z.side_balance,
                    "control_type": z.control_type, "control_variant": z.control_variant,
                    "object_state_id": z.object_state_id, "source_object_state_id": z.source_object_state_id or z.object_state_id,
                    "source_object_key": z.source_object_key or z.object_key,
                    "zone_lower": z.lower, "zone_upper": z.upper, "zone_center": z.center, "zone_width_bps": z.width_bps,
                    "source_zone_lower": z.source_lower if z.source_lower is not None else z.lower,
                    "source_zone_upper": z.source_upper if z.source_upper is not None else z.upper,
                    "source_zone_center": source_center,
                    "source_zone_width_bps": z.source_width_bps if z.source_width_bps is not None else z.width_bps,
                    "object_latest_information_time": latest, "object_age_days": z.age_days,
                    "width_capped": z.width_capped, "construction_note": z.construction_note,
                    "anchor_count": z.anchor_count, "independent_episode_count": z.independent_episode_count,
                    "independent_regime_count": z.independent_regime_count,
                    "approach_from": app, "entry_edge": entry, "far_edge": far,
                    "event_open": float(opens_all[g]), "event_high": float(highs_all[g]), "event_low": float(lows_all[g]),
                    "event_close": float(closes_all[g]), "trailing_sigma_bps": sigma,
                    "approach_return_15m_bps": ar15, "approach_velocity_toward_bps_15m": toward,
                    "day_start_price": day_start_price,
                    "signed_distance_from_day_start_bps": bps_log_ratio(z.center, day_start_price),
                    "abs_distance_from_day_start_bps": abs(bps_log_ratio(z.center, day_start_price)),
                    "source_abs_distance_from_day_start_bps": src_abs,
                    "recent_exposure_30d": z.recent_exposure_30d,
                    "source_recent_exposure_30d": z.source_recent_exposure_30d,
                    "confluence_count": len(members) if z.control_type == "REAL" else 0,
                    "confluence_object_types": ";".join(types), "confluence_variants": ";".join(variants),
                    "is_confluence": bool(len(members) >= 2) if z.control_type == "REAL" else False,
                }
                row.update(out)
                rows.append(row)
        processed += 1
        if processed % 60 == 0:
            print(f"  {year}: processed {processed} days, events so far {len(rows):,}")
        d += timedelta(days=1)

    events = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    return events, inventory


def one_way_cluster_cov(X, residuals, groups):
    import numpy as np
    X = np.asarray(X, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    groups = np.asarray(groups)
    k = X.shape[1]
    meat = np.zeros((k, k), dtype=float)
    for g in np.unique(groups):
        idx = groups == g
        score = X[idx].T @ residuals[idx]
        meat += np.outer(score, score)
    return meat


def two_way_cluster_effect(real, control, endpoint: str) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    a = real[[endpoint, "event_date", "physical_event_id", "source_object_state_id"]].copy()
    b = control[[endpoint, "event_date", "physical_event_id", "source_object_state_id"]].copy()
    a["is_real"] = 1.0
    b["is_real"] = 0.0
    x = pd.concat([a, b], ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna(subset=[endpoint])
    nr = int((x["is_real"] == 1).sum()); nc = int((x["is_real"] == 0).sum())
    if nr < 20 or nc < 20:
        return {"n_real": nr, "n_control": nc, "status": "INSUFFICIENT"}
    y = x[endpoint].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(x)), x["is_real"].to_numpy(dtype=float)])
    bread = np.linalg.pinv(X.T @ X)
    beta = bread @ X.T @ y
    resid = y - X @ beta
    day = x["event_date"].astype(str).to_numpy()
    # Physical market interaction is the appropriate within-event dependence unit. Controls
    # are each their own physical IDs; repeated memberships of real confluence share one ID.
    physical = x["physical_event_id"].astype(str).to_numpy()
    inter = np.asarray([f"{d}|{p}" for d, p in zip(day, physical)], dtype=object)
    meat = one_way_cluster_cov(X, resid, day) + one_way_cluster_cov(X, resid, physical) - one_way_cluster_cov(X, resid, inter)
    V = bread @ meat @ bread
    se = float(math.sqrt(max(V[1, 1], 0.0)))
    effect = float(beta[1])
    zstat = effect / se if se > 0 else math.nan
    p = math.erfc(abs(zstat) / math.sqrt(2.0)) if math.isfinite(zstat) else math.nan
    mean_r = float(a[endpoint].replace([np.inf, -np.inf], np.nan).mean())
    mean_c = float(b[endpoint].replace([np.inf, -np.inf], np.nan).mean())
    sd_r = float(a[endpoint].std(ddof=1)); sd_c = float(b[endpoint].std(ddof=1))
    pooled_sd = math.sqrt(((nr - 1) * sd_r * sd_r + (nc - 1) * sd_c * sd_c) / max(nr + nc - 2, 1)) if nr > 1 and nc > 1 else math.nan
    effect_sd = effect / pooled_sd if math.isfinite(pooled_sd) and pooled_sd > 0 else math.nan
    cluster_mde = Z_MDE * se
    # Conservative simple independent-groups MDE too; useful when cluster estimate is unstable.
    conservative_sd = Z_MDE * math.sqrt(2.0 * (1.0 / nr + 1.0 / nc))
    return {
        "status": "OK", "n_real": nr, "n_control": nc, "mean_real": mean_r, "mean_control": mean_c,
        "effect_real_minus_control": effect, "effect_sd": effect_sd, "cluster_se": se,
        "ci95_low": effect - Z_ALPHA_2 * se, "ci95_high": effect + Z_ALPHA_2 * se, "p_two_sided": p,
        "clusters_day": int(len(set(day))), "clusters_physical_event": int(len(set(physical))),
        "mde_80pct_cluster_units": cluster_mde,
        "mde_80pct_cluster_sd": cluster_mde / pooled_sd if math.isfinite(pooled_sd) and pooled_sd > 0 else math.nan,
        "mde_80pct_conservative_2x_sd": conservative_sd,
    }



def source_matched_pair_frame(real, control, endpoint: str):
    """Inner-match REAL/control interactions by source object state.

    Controls are built conditionally on a source zone. When control coverage is incomplete,
    comparing those controls with the full REAL event population is selection-biased. Only
    source states where both the REAL zone and its matched control generated an interaction
    enter the robustness contrast.
    """
    import pandas as pd
    cols = [endpoint, "event_date", "physical_event_id", "source_object_state_id"]
    r = real[cols].replace([float("inf"), float("-inf")], float("nan")).dropna(subset=[endpoint, "source_object_state_id"]).copy()
    c = control[cols].replace([float("inf"), float("-inf")], float("nan")).dropna(subset=[endpoint, "source_object_state_id"]).copy()
    if r.empty or c.empty:
        return pd.DataFrame()
    rg = r.groupby("source_object_state_id", as_index=False).agg(
        real_value=(endpoint, "mean"), event_date=("event_date", "first"),
        real_physical_event_id=("physical_event_id", "first"),
    )
    cg = c.groupby("source_object_state_id", as_index=False).agg(
        control_value=(endpoint, "mean"), control_event_date=("event_date", "first"),
        control_physical_event_id=("physical_event_id", "first"),
    )
    m = rg.merge(cg, on="source_object_state_id", how="inner", validate="one_to_one")
    if m.empty:
        return m
    m["difference"] = m["real_value"].astype(float) - m["control_value"].astype(float)
    return m


def paired_cluster_effect(real, control, endpoint: str, physical_dedup: bool = False) -> dict[str, Any]:
    """Source-matched REAL-minus-control effect with day/REAL-physical-event clustering."""
    import numpy as np
    m = source_matched_pair_frame(real, control, endpoint)
    raw_real_n = int(len(real)); raw_control_n = int(len(control))
    if m.empty:
        return {
            "status": "INSUFFICIENT", "n_real": 0, "n_control": 0, "n_pairs": 0,
            "raw_real_events": raw_real_n, "raw_control_events": raw_control_n,
            "real_pair_coverage": 0.0, "control_pair_coverage": 0.0,
        }
    if physical_dedup:
        m = m.groupby(["event_date", "real_physical_event_id"], as_index=False).agg(
            real_value=("real_value", "mean"), control_value=("control_value", "mean"),
            difference=("difference", "mean"),
        )
        m["source_object_state_id"] = m["real_physical_event_id"]
    n = int(len(m))
    coverage_r = n / raw_real_n if raw_real_n else 0.0
    coverage_c = n / raw_control_n if raw_control_n else 0.0
    if n < 20:
        return {
            "status": "INSUFFICIENT", "n_real": n, "n_control": n, "n_pairs": n,
            "raw_real_events": raw_real_n, "raw_control_events": raw_control_n,
            "real_pair_coverage": coverage_r, "control_pair_coverage": coverage_c,
        }
    diff = m["difference"].to_numpy(dtype=float)
    X = np.ones((n, 1), dtype=float)
    bread = np.linalg.pinv(X.T @ X)
    beta = bread @ X.T @ diff
    resid = diff - X @ beta
    day = m["event_date"].astype(str).to_numpy()
    physical = m["real_physical_event_id"].astype(str).to_numpy()
    inter = np.asarray([f"{d}|{q}" for d, q in zip(day, physical)], dtype=object)
    meat = one_way_cluster_cov(X, resid, day) + one_way_cluster_cov(X, resid, physical) - one_way_cluster_cov(X, resid, inter)
    V = bread @ meat @ bread
    se = float(math.sqrt(max(V[0, 0], 0.0)))
    effect = float(beta[0])
    zstat = effect / se if se > 0 else math.nan
    pval = math.erfc(abs(zstat) / math.sqrt(2.0)) if math.isfinite(zstat) else math.nan
    mean_r = float(m["real_value"].mean()); mean_c = float(m["control_value"].mean())
    sd_r = float(m["real_value"].std(ddof=1)); sd_c = float(m["control_value"].std(ddof=1))
    pooled_sd = math.sqrt(((n - 1) * sd_r * sd_r + (n - 1) * sd_c * sd_c) / max(2*n - 2, 1)) if n > 1 else math.nan
    effect_sd = effect / pooled_sd if math.isfinite(pooled_sd) and pooled_sd > 0 else math.nan
    cluster_mde = Z_MDE * se
    diff_sd = float(m["difference"].std(ddof=1)) if n > 1 else math.nan
    conservative_units = Z_MDE * diff_sd / math.sqrt(n) if math.isfinite(diff_sd) else math.nan
    conservative_sd = conservative_units / pooled_sd if math.isfinite(conservative_units) and math.isfinite(pooled_sd) and pooled_sd > 0 else math.nan
    return {
        "status": "OK", "n_real": n, "n_control": n, "n_pairs": n,
        "raw_real_events": raw_real_n, "raw_control_events": raw_control_n,
        "real_pair_coverage": coverage_r, "control_pair_coverage": coverage_c,
        "mean_real": mean_r, "mean_control": mean_c,
        "effect_real_minus_control": effect, "effect_sd": effect_sd, "cluster_se": se,
        "ci95_low": effect - Z_ALPHA_2 * se, "ci95_high": effect + Z_ALPHA_2 * se,
        "p_two_sided": pval, "clusters_day": int(len(set(day))),
        "clusters_physical_event": int(len(set(physical))),
        "mde_80pct_cluster_units": cluster_mde,
        "mde_80pct_cluster_sd": cluster_mde / pooled_sd if math.isfinite(pooled_sd) and pooled_sd > 0 else math.nan,
        "mde_80pct_conservative_2x_sd": conservative_sd,
    }


def source_matched_pair_coverage(events):
    import pandas as pd
    if events.empty:
        return pd.DataFrame()
    rows = []
    for tol in sorted(events["tolerance_id"].dropna().unique()):
        et = events[events["tolerance_id"] == tol]
        for variant, g in et.groupby("variant_id"):
            real = g[g["control_type"] == "REAL"]
            real_ids = set(real["source_object_state_id"].astype(str))
            for (ctype, cvariant), c in g[g["control_type"] != "REAL"].groupby(["control_type", "control_variant"]):
                ctl_ids = set(c["source_object_state_id"].astype(str))
                paired = real_ids & ctl_ids
                rows.append({
                    "tolerance_id": tol, "tolerance_primary": tol == PRIMARY_TOLERANCE_ID,
                    "variant_id": variant, "control_type": ctype, "control_variant": cvariant,
                    "real_events": int(len(real)), "control_events": int(len(c)),
                    "paired_source_events": int(len(paired)),
                    "real_event_pair_coverage": len(paired) / len(real_ids) if real_ids else 0.0,
                    "control_event_pair_coverage": len(paired) / len(ctl_ids) if ctl_ids else 0.0,
                })
    return pd.DataFrame(rows)

def build_effect_tables(events):
    import pandas as pd
    rows = []
    if events.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for tol in sorted(events["tolerance_id"].dropna().unique()):
        et = events[events["tolerance_id"] == tol]
        for (otype, variant), ef in et.groupby(["object_type", "variant_id"], dropna=False):
            real = ef[ef["control_type"] == "REAL"]
            if real.empty:
                continue
            for ctype in sorted(x for x in ef["control_type"].dropna().unique() if x != "REAL"):
                ctrl = ef[ef["control_type"] == ctype]
                stat = paired_cluster_effect(real, ctrl, PRIMARY_ENDPOINT)
                rows.append({
                    "scope": "POOLED", "year": "ALL", "tolerance_id": tol,
                    "tolerance_primary": tol == PRIMARY_TOLERANCE_ID, "object_type": otype, "variant_id": variant,
                    "control_type": ctype, "control_variant": "POOLED", "endpoint": PRIMARY_ENDPOINT, **stat,
                })
                for cv in sorted(ctrl["control_variant"].dropna().unique()):
                    sv = paired_cluster_effect(real, ctrl[ctrl["control_variant"] == cv], PRIMARY_ENDPOINT)
                    rows.append({
                        "scope": "VARIANT", "year": "ALL", "tolerance_id": tol,
                        "tolerance_primary": tol == PRIMARY_TOLERANCE_ID, "object_type": otype, "variant_id": variant,
                        "control_type": ctype, "control_variant": cv, "endpoint": PRIMARY_ENDPOINT, **sv,
                    })
                for y in sorted(ef["event_year"].dropna().unique()):
                    ry = real[real["event_year"] == y]
                    cy = ctrl[ctrl["event_year"] == y]
                    sy = paired_cluster_effect(ry, cy, PRIMARY_ENDPOINT)
                    rows.append({
                        "scope": "YEAR", "year": int(y), "tolerance_id": tol,
                        "tolerance_primary": tol == PRIMARY_TOLERANCE_ID, "object_type": otype, "variant_id": variant,
                        "control_type": ctype, "control_variant": "POOLED", "endpoint": PRIMARY_ENDPOINT, **sy,
                    })
    out = pd.DataFrame(rows)
    pooled = out[out["scope"] == "POOLED"].copy() if len(out) else pd.DataFrame()
    yearly = out[out["scope"] == "YEAR"].copy() if len(out) else pd.DataFrame()
    variants = out[out["scope"] == "VARIANT"].copy() if len(out) else pd.DataFrame()
    return pooled, yearly, variants


def secondary_summary(events):
    import pandas as pd
    if events.empty:
        return pd.DataFrame()
    metrics = [
        "penetration_vol_15m", "penetration_vol_60m", "penetration_vol_240m",
        "rejection_vol_15m", "rejection_vol_60m", "rejection_vol_240m",
        "far_edge_penetration_vol_60m", "zone_traversal_fraction_60m",
        "inside_zone_close_fraction_60m", "beyond_far_edge_close_fraction_60m",
        "near_zone_minutes_60m",
    ]
    rows = []
    for keys, g in events.groupby(["tolerance_id", "object_type", "variant_id", "control_type"], dropna=False):
        tol, otype, variant, ctype = keys
        row = {
            "tolerance_id": tol, "tolerance_primary": tol == PRIMARY_TOLERANCE_ID,
            "object_type": otype, "variant_id": variant, "control_type": ctype, "events": len(g),
        }
        for m in metrics:
            row[f"mean_{m}"] = float(g[m].mean())
            row[f"median_{m}"] = float(g[m].median())
        for label in ("REJECTION", "CLEAN_TRAVERSAL", "ENTER_RETURN", "ACCEPTANCE_DWELL", "STALL_OTHER"):
            row[f"share_{label.lower()}"] = float((g["exploratory_resolution_60m"] == label).mean())
        row["retest_share"] = float(g["retest_within_240m"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def confluence_summary(events):
    import pandas as pd
    if events.empty:
        return pd.DataFrame()
    real = events[(events["control_type"] == "REAL") & (events["tolerance_id"] == PRIMARY_TOLERANCE_ID)].copy()
    if real.empty:
        return pd.DataFrame()
    rows = []
    for (otype, variant), g in real.groupby(["object_type", "variant_id"]):
        rows.append({
            "object_type": otype, "variant_id": variant, "events": len(g),
            "physical_events": int(g["physical_event_id"].nunique()),
            "confluence_event_share": float(g["is_confluence"].mean()),
            "median_confluence_count": float(g["confluence_count"].median()),
            "p95_confluence_count": float(g["confluence_count"].quantile(0.95)),
        })
    return pd.DataFrame(rows)


def build_event_count_table(events, inventory_df):
    import pandas as pd
    if inventory_df.empty:
        return pd.DataFrame()
    inv = inventory_df.groupby(["year", "object_type", "variant_id", "control_type", "control_variant"], as_index=False)["candidate_objects"].sum()
    if events.empty:
        inv["tolerance_id"] = PRIMARY_TOLERANCE_ID
        inv["events"] = 0
        inv["event_rate_per_candidate"] = 0.0
        return inv
    ev = events.groupby(["event_year", "object_type", "variant_id", "control_type", "control_variant", "tolerance_id"], as_index=False).size().rename(columns={"event_year": "year", "size": "events"})
    out = ev.merge(inv, on=["year", "object_type", "variant_id", "control_type", "control_variant"], how="left")
    out["event_rate_per_candidate"] = out["events"] / out["candidate_objects"].where(out["candidate_objects"] > 0)
    return out



def candidate_control_coverage(inventory):
    import pandas as pd
    if inventory.empty:
        return pd.DataFrame()
    real = inventory[inventory["control_type"] == "REAL"].groupby(["year", "variant_id"], as_index=False)["candidate_objects"].sum().rename(columns={"candidate_objects":"real_candidates"})
    ctl = inventory[inventory["control_type"] != "REAL"].groupby(["year", "variant_id", "control_type", "control_variant"], as_index=False)["candidate_objects"].sum().rename(columns={"candidate_objects":"control_candidates"})
    if ctl.empty:
        return pd.DataFrame()
    out = ctl.merge(real, on=["year","variant_id"], how="left")
    out["candidate_match_coverage"] = out["control_candidates"] / out["real_candidates"].where(out["real_candidates"] > 0)
    pooled_ctl = ctl.groupby(["variant_id","control_type","control_variant"], as_index=False)["control_candidates"].sum()
    pooled_real = real.groupby(["variant_id"], as_index=False)["real_candidates"].sum()
    pooled = pooled_ctl.merge(pooled_real, on="variant_id", how="left")
    pooled["candidate_match_coverage"] = pooled["control_candidates"] / pooled["real_candidates"].where(pooled["real_candidates"] > 0)
    pooled.insert(0, "year", "ALL")
    return pd.concat([pooled, out], ignore_index=True)


def control_match_quality(events):
    import pandas as pd
    c = events[(events["control_type"].isin(["STALE_SWING_ZONE", "STRICT_MATCHED_RANDOM_ZONE"]))
               & (events["tolerance_id"] == PRIMARY_TOLERANCE_ID)].copy()
    if c.empty:
        return pd.DataFrame()
    rows = []
    for (variant, ctype, cvariant), g in c.groupby(["variant_id", "control_type", "control_variant"]):
        src_exp = g["source_recent_exposure_30d"].fillna(0).astype(float)
        ctl_exp = g["recent_exposure_30d"].fillna(0).astype(float)
        rows.append({
            "variant_id": variant, "control_type": ctype, "control_variant": cvariant, "events": len(g),
            "median_source_abs_distance_bps": float(g["source_abs_distance_from_day_start_bps"].median()),
            "median_control_abs_distance_bps": float(g["abs_distance_from_day_start_bps"].median()),
            "median_abs_distance_error_bps": float((g["abs_distance_from_day_start_bps"] - g["source_abs_distance_from_day_start_bps"]).abs().median()),
            "p95_abs_distance_error_bps": float((g["abs_distance_from_day_start_bps"] - g["source_abs_distance_from_day_start_bps"]).abs().quantile(.95)),
            "median_source_exposure": float(src_exp.median()), "median_control_exposure": float(ctl_exp.median()),
            "median_log_exposure_error": float((ctl_exp.map(math.log1p) - src_exp.map(math.log1p)).abs().median()),
            "p95_log_exposure_error": float((ctl_exp.map(math.log1p) - src_exp.map(math.log1p)).abs().quantile(.95)),
        })
    return pd.DataFrame(rows)




def preregistered_primary_tables(events):
    """Exact predeclared 7d×1h vs stale 8-14d contrast; never pools stale bands."""
    import pandas as pd
    if events.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    rows = []
    for tol in sorted(events["tolerance_id"].dropna().unique()):
        g = events[(events["variant_id"] == PRIMARY_VARIANT) & (events["tolerance_id"] == tol)]
        real = g[g["control_type"] == "REAL"]
        ctl = g[(g["control_type"] == "STALE_SWING_ZONE") & (g["control_variant"] == "STALE_8_14D")]
        stat = paired_cluster_effect(real, ctl, PRIMARY_ENDPOINT)
        rows.append({"scope":"POOLED","year":"ALL","tolerance_id":tol,"tolerance_primary":tol==PRIMARY_TOLERANCE_ID,
                     "variant_id":PRIMARY_VARIANT,"control_type":"STALE_SWING_ZONE","control_variant":"STALE_8_14D",
                     "endpoint":PRIMARY_ENDPOINT, **stat})
        for y in sorted(g["event_year"].dropna().unique()):
            sy = paired_cluster_effect(real[real["event_year"] == y], ctl[ctl["event_year"] == y], PRIMARY_ENDPOINT)
            rows.append({"scope":"YEAR","year":int(y),"tolerance_id":tol,"tolerance_primary":tol==PRIMARY_TOLERANCE_ID,
                         "variant_id":PRIMARY_VARIANT,"control_type":"STALE_SWING_ZONE","control_variant":"STALE_8_14D",
                         "endpoint":PRIMARY_ENDPOINT, **sy})
    out = pd.DataFrame(rows)
    pooled = out[(out["scope"] == "POOLED") & (out["tolerance_id"] == PRIMARY_TOLERANCE_ID)].copy()
    yearly = out[(out["scope"] == "YEAR") & (out["tolerance_id"] == PRIMARY_TOLERANCE_ID)].copy()
    sensitivity = out[out["scope"] == "POOLED"].copy()

    # Physical-deduplicated version for the exact primary tolerance/control.
    g = events[(events["variant_id"] == PRIMARY_VARIANT) & (events["tolerance_id"] == PRIMARY_TOLERANCE_ID)]
    real = g[g["control_type"] == "REAL"].copy()
    ctl = g[(g["control_type"] == "STALE_SWING_ZONE") & (g["control_variant"] == "STALE_8_14D")].copy()
    dstat = paired_cluster_effect(real, ctl, PRIMARY_ENDPOINT, physical_dedup=True) if len(real) or len(ctl) else {"status":"INSUFFICIENT","n_real":0,"n_control":0,"n_pairs":0}
    dedup = pd.DataFrame([{"variant_id":PRIMARY_VARIANT,"tolerance_id":PRIMARY_TOLERANCE_ID,
                           "control_type":"STALE_SWING_ZONE","control_variant":"STALE_8_14D", **dstat}])
    return pooled, yearly, sensitivity, dedup


def physical_dedup_effect_table(events):
    """One contribution per REAL physical episode/variant; controls remain one row per event."""
    import pandas as pd
    if events.empty:
        return pd.DataFrame()
    rows = []
    et = events[events["tolerance_id"] == PRIMARY_TOLERANCE_ID].copy()
    for variant, g in et.groupby("variant_id"):
        real = g[g["control_type"] == "REAL"].copy()
        if real.empty:
            continue
        # Pair first; paired_cluster_effect handles REAL physical-event deduplication.
        for ctype in ("STALE_SWING_ZONE", "STRICT_MATCHED_RANDOM_ZONE"):
            ctl = g[g["control_type"] == ctype].copy()
            if ctl.empty:
                continue
            # Controls are already unique physical events.
            stat = paired_cluster_effect(real, ctl, PRIMARY_ENDPOINT, physical_dedup=True)
            rows.append({"variant_id": variant, "control_type": ctype, "control_variant": "POOLED",
                         "tolerance_id": PRIMARY_TOLERANCE_ID, **stat})
            for cv in sorted(ctl["control_variant"].dropna().unique()):
                s = paired_cluster_effect(real, ctl[ctl["control_variant"] == cv], PRIMARY_ENDPOINT, physical_dedup=True)
                rows.append({"variant_id": variant, "control_type": ctype, "control_variant": cv,
                             "tolerance_id": PRIMARY_TOLERANCE_ID, **s})
    return pd.DataFrame(rows)


def confluence_strata(events):
    import pandas as pd
    r = events[(events["control_type"] == "REAL") & (events["tolerance_id"] == PRIMARY_TOLERANCE_ID)].copy()
    if r.empty:
        return pd.DataFrame()
    def band(x):
        x = int(x)
        if x <= 3: return "01_1_TO_3"
        if x <= 7: return "02_4_TO_7"
        if x <= 15: return "03_8_TO_15"
        return "04_16_PLUS"
    r["confluence_band"] = r["confluence_count"].map(band)
    rows = []
    for (variant, b), g in r.groupby(["variant_id", "confluence_band"]):
        phys = g.groupby("physical_event_id", as_index=False)[PRIMARY_ENDPOINT].mean()
        rows.append({"variant_id": variant, "confluence_band": b, "membership_rows": len(g),
                     "physical_events": len(phys), "mean_penetration_vol_60m": float(phys[PRIMARY_ENDPOINT].mean()),
                     "median_penetration_vol_60m": float(phys[PRIMARY_ENDPOINT].median())})
    return pd.DataFrame(rows)


def invariants(events, objects, analysis_start: date, analysis_end: date) -> list[Check]:
    import numpy as np
    import pandas as pd
    checks: list[Check] = []
    dup = int(events["event_id"].duplicated().sum()) if len(events) else 0
    checks.append(Check("events", "unique_event_id", "PASS" if dup == 0 else "FAIL", dup, 0))
    bad_bounds = int(((events["zone_lower"] <= 0) | (events["zone_upper"] <= events["zone_lower"])).sum()) if len(events) else 0
    checks.append(Check("events", "valid_zone_bounds", "PASS" if bad_bounds == 0 else "FAIL", bad_bounds, 0))
    if len(events):
        latest = pd.to_datetime(events["object_latest_information_time"], utc=True, errors="coerce")
        evt = pd.to_datetime(events["event_time"], utc=True)
        real_mask = events["control_type"] == "REAL"
        bad_causal = int((latest[real_mask].notna() & (latest[real_mask] >= evt[real_mask])).sum())
    else:
        bad_causal = 0
    checks.append(Check("causality", "object_information_before_event", "PASS" if bad_causal == 0 else "FAIL", bad_causal, 0))
    sigma_ok = bool((events["trailing_sigma_bps"].dropna() > 0).all()) if len(events) else True
    checks.append(Check("causality", "positive_trailing_sigma", "PASS" if sigma_ok else "FAIL", sigma_ok, True))
    tau_ok = bool((events["tau_bps"].dropna() > 0).all()) if len(events) else True
    checks.append(Check("events", "positive_tolerance", "PASS" if tau_ok else "FAIL", tau_ok, True))
    primary_exists = bool((events["tolerance_id"] == PRIMARY_TOLERANCE_ID).any()) if len(events) else False
    checks.append(Check("contract", "primary_tolerance_present", "PASS" if primary_exists else "FAIL", primary_exists, True))
    finite = events.loc[events["tolerance_id"] == PRIMARY_TOLERANCE_ID, PRIMARY_ENDPOINT].replace([np.inf, -np.inf], np.nan).notna().mean() if len(events) else 0.0
    checks.append(Check("outcomes", "primary_endpoint_completeness", "PASS" if finite >= 0.99 else "WARN", float(finite), ">=0.99"))
    cap = events.groupby(["event_date", "tolerance_id", "object_state_id", "control_type", "control_variant"]).size().max() if len(events) else 0
    checks.append(Check("events", "one_event_per_object_day_tolerance", "PASS" if cap <= MAX_EVENTS_PER_OBJECT_DAY else "FAIL", int(cap), MAX_EVENTS_PER_OBJECT_DAY))
    # Every real event must refer to a Phase-5A state loaded for that date.
    expected_ids = set(objects["object_state_id"].astype(str)) if len(objects) else set()
    unknown = int((~events.loc[events["control_type"] == "REAL", "object_state_id"].astype(str).isin(expected_ids)).sum()) if len(events) else 0
    checks.append(Check("objects", "real_events_reference_phase5a_state", "PASS" if unknown == 0 else "FAIL", unknown, 0))
    non_recent = int((events.loc[events["control_type"] == "REAL", "object_type"] != "RECENT_SWING").sum()) if len(events) else 0
    checks.append(Check("contract", "real_events_recent_swing_only", "PASS" if non_recent == 0 else "FAIL", non_recent, 0))
    if len(events):
        stale = events[events["control_type"] == "STALE_SWING_ZONE"]
        sl = pd.to_datetime(stale["object_latest_information_time"], utc=True, errors="coerce")
        se = pd.to_datetime(stale["event_time"], utc=True, errors="coerce")
        bad_stale = int((sl.notna() & (sl >= se)).sum())
    else:
        bad_stale = 0
    checks.append(Check("causality", "stale_control_information_before_event", "PASS" if bad_stale == 0 else "FAIL", bad_stale, 0))
    if len(events):
        dmin = min(parse_date(x) for x in events["event_date"].astype(str).unique())
        dmax = max(parse_date(x) for x in events["event_date"].astype(str).unique())
        checks.append(Check("coverage", "event_start_not_before_analysis", "PASS" if dmin >= analysis_start else "FAIL", dmin.isoformat(), f">={analysis_start}"))
        checks.append(Check("coverage", "event_end_not_after_analysis", "PASS" if dmax <= analysis_end else "FAIL", dmax.isoformat(), f"<={analysis_end}"))
    return checks


def init_state_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS year_units (
            year INTEGER PRIMARY KEY,
            signature TEXT NOT NULL,
            status TEXT NOT NULL,
            event_rows INTEGER NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def state_year_complete(conn: sqlite3.Connection, year: int, signature: str, event_path: Path, inventory_path: Path) -> bool:
    row = conn.execute("SELECT signature,status FROM year_units WHERE year=?", (year,)).fetchone()
    return bool(row and row[0] == signature and row[1] == "COMPLETED" and event_path.exists() and inventory_path.exists())


def mark_year(conn: sqlite3.Connection, year: int, signature: str, status: str, event_rows: int) -> None:
    conn.execute("""
        INSERT INTO year_units(year,signature,status,event_rows,updated_at)
        VALUES(?,?,?,?,?)
        ON CONFLICT(year) DO UPDATE SET signature=excluded.signature,status=excluded.status,
            event_rows=excluded.event_rows,updated_at=excluded.updated_at
    """, (year, signature, status, int(event_rows), utc_now_iso()))
    conn.commit()


def write_findings(path: Path, summary: dict[str, Any], pooled, yearly, confluence, checks) -> None:
    import pandas as pd
    lines = [
        "# Recent-swing robustness evaluation", "",
        f"- Evaluator version: `{EVALUATOR_VERSION}`",
        f"- Run ID: `{summary['run_id']}`",
        f"- Status: **{summary['status']}**",
        f"- Analysis: `{summary['analysis_start']}` through `{summary['analysis_end']}`",
        f"- Events: **{summary['event_rows']:,}**",
        f"- Primary tolerance: `{PRIMARY_TOLERANCE_ID}`",
        f"- Primary endpoint: `{PRIMARY_ENDPOINT}`",
        f"- Planned MDE target: **{PLANNED_MDE_TARGET_SD:.2f} SD**", "",
        "## Interpretation contract", "",
        "`effect_real_minus_control < 0` means the current recent-swing zone was harder to penetrate than the specified stale-swing or strict-matched control.",
        "`effect_real_minus_control > 0` means the current recent-swing zone was easier to penetrate than that control.",
        "The primary hypothesis is `RECENT_SWING_7D_60M` versus genuine 8-14-day-old 1h swing zones; other stale bands/variants are sensitivity checks.",
        "Object variants are evaluated separately and Phase 5C never changes Phase-5A structural definitions.", "",
        "## Primary pooled effects", "",
    ]
    primary = pooled[pooled["tolerance_id"] == PRIMARY_TOLERANCE_ID].copy() if len(pooled) else pd.DataFrame()
    cols = ["object_type", "variant_id", "control_type", "n_real", "n_control", "effect_real_minus_control", "effect_sd", "ci95_low", "ci95_high", "mde_80pct_conservative_2x_sd"]
    if primary.empty:
        lines.append("No primary pooled effects were available.")
    else:
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for r in primary.sort_values(["object_type", "variant_id", "control_type"]).itertuples(index=False):
            vals = []
            for c in cols:
                v = getattr(r, c, "")
                vals.append(f"{v:.4f}" if isinstance(v, float) and math.isfinite(v) else str(v))
            lines.append("| " + " | ".join(vals) + " |")
    lines += ["", "## Cross-year sign consistency", ""]
    if len(yearly):
        yp = yearly[yearly["tolerance_id"] == PRIMARY_TOLERANCE_ID]
        for (otype, variant, ctype), g in yp.groupby(["object_type", "variant_id", "control_type"]):
            ok = g[g["status"] == "OK"]
            if len(ok):
                neg = int((ok["effect_real_minus_control"] < 0).sum())
                lines.append(f"- `{variant}` vs `{ctype}`: negative penetration effect in **{neg}/{len(ok)}** years.")
    lines += ["", "## Confluence", ""]
    if len(confluence):
        for r in confluence.sort_values(["object_type", "variant_id"]).itertuples(index=False):
            lines.append(f"- `{r.variant_id}`: {int(r.physical_events):,} physical events; confluence share {r.confluence_event_share:.1%}.")
    else:
        lines.append("No primary real confluence events were available.")
    lines += ["", "## Invariants", ""]
    for c in checks:
        lines.append(f"- **{c.status}** `{c.scope}/{c.check}` — observed `{c.observed}`, expected `{c.expected}`.")
    lines += [
        "", "## Guardrail", "",
        "This robustness pass evaluates recent swing structure only. It does not use trade-flow, OI, liquidation, or L2 participant features. Participant-motif interaction testing comes after the structural benchmark is understood.", "",
    ]
    atomic_write_text(path, "\n".join(lines))


def self_test() -> None:
    import numpy as np
    import pandas as pd
    # Variant parsing and bounded clustering.
    assert parse_recent_variant("RECENT_SWING_7D_60M") == (7, 60)
    a = pd.DataFrame({
        "anchor_id": ["a","b","c"], "pivot_time": pd.to_datetime(["2020-01-01","2020-01-02","2020-01-03"], utc=True),
        "available_time": pd.to_datetime(["2020-01-01 01:00","2020-01-02 01:00","2020-01-03 01:00"], utc=True),
        "side": ["SUPPORT","SUPPORT","RESISTANCE"], "scale_minutes": [60,60,60],
        "pivot_price": [100.0,100.10,101.0], "prominence_bps": [10.,12.,11.], "formation_sigma_bps": [100.,100.,100.],
    })
    cl = complete_linkage_clusters(a)
    assert len(cl) >= 2
    # Match score must reward identical geometry/exposure.
    s = Zone("RECENT_SWING","RECENT_SWING_7D_60M","OBSERVED",99.9,100.1,100.0,zone_width_bps(99.9,100.1),
             "s","s",None,1.0,60,"NONE","TWO_SIDED",False,"",2,1,1)
    c = Zone("RECENT_SWING","X","CONTROL",99.9,100.1,100.0,s.width_bps,"c","c",None,10.0,60,"NONE","TWO_SIDED",False,"",2,1,1)
    exposure = Counter({exposure_bin_idx(100.0): 100})
    sc = match_score(s,c,100.0,exposure)
    assert sc[1] < 1e-9 and sc[2] < 1e-9
    # A stale 8-14d control for 7d x 1h must not be invalidated merely because
    # the same price remains active in a broader 30d hypothesis. Only the same
    # current hypothesis can make the control non-stale.
    same = Zone("RECENT_SWING","RECENT_SWING_7D_60M","OBSERVED",99.8,100.2,100.0,zone_width_bps(99.8,100.2),
                "same","same",None,1.0,60,"NONE","TWO_SIDED",False,"",2,1,1)
    broader = Zone("RECENT_SWING","RECENT_SWING_30D_60M","OBSERVED",99.8,100.2,100.0,zone_width_bps(99.8,100.2),
                   "broad","broad",None,1.0,60,"NONE","TWO_SIDED",False,"",2,1,1)
    refs = stale_collision_reference(s, [same, broader])
    assert [z.variant_id for z in refs] == ["RECENT_SWING_7D_60M"]
    # Clustered effect sign and physical dedup threshold.
    days = [f"2020-02-{(i%28)+1:02d}" for i in range(30)]
    real = pd.DataFrame({PRIMARY_ENDPOINT:[.1]*30,"event_date":days,"physical_event_id":[f"r{i}" for i in range(30)],"source_object_state_id":[f"s{i}" for i in range(30)]})
    ctl = pd.DataFrame({PRIMARY_ENDPOINT:[.4]*30,"event_date":days,"physical_event_id":[f"c{i}" for i in range(30)],"source_object_state_id":[f"s{i}" for i in range(30)]})
    stat = paired_cluster_effect(real, ctl, PRIMARY_ENDPOINT)
    assert stat["status"] == "OK" and stat["effect_real_minus_control"] < 0 and stat["n_pairs"] == 30
    extra = pd.DataFrame({PRIMARY_ENDPOINT:[9.0]*50,"event_date":["2020-03-01"]*50,
                          "physical_event_id":[f"x{i}" for i in range(50)],
                          "source_object_state_id":[f"unmatched{i}" for i in range(50)]})
    stat2 = paired_cluster_effect(pd.concat([real, extra], ignore_index=True), ctl, PRIMARY_ENDPOINT)
    assert stat2["n_pairs"] == 30 and abs(stat2["effect_real_minus_control"] - stat["effect_real_minus_control"]) < 1e-12
    print("SELF-TEST PASS")
    print("Validated stale-window parsing, same-hypothesis stale collision semantics, strict match geometry/exposure, source-matched pairing, unmatched-REAL immunity, and clustered effect sign.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--historical-root", type=Path, default=DEFAULT_HISTORICAL_ROOT)
    p.add_argument("--object-root", type=Path, default=DEFAULT_OBJECT_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=None)
    p.add_argument("--start-date", default=DEFAULT_ANALYSIS_START)
    p.add_argument("--end-date", default=DEFAULT_ANALYSIS_END)
    p.add_argument("--tolerances", default="all", help="all, primary, or comma-separated tolerance IDs")
    p.add_argument("--variants", default="all", help="all, primary, or comma-separated RECENT_SWING variant IDs")
    p.add_argument("--force", action="store_true", help="Recompute yearly parts even if matching state exists")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0

    require_runtime_libraries()
    import pandas as pd

    analysis_start = parse_date(args.start_date)
    analysis_end = parse_date(args.end_date)
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

    if args.variants == "all":
        selected_variants = list(RECENT_VARIANTS)
    elif args.variants == "primary":
        selected_variants = [PRIMARY_VARIANT]
    else:
        selected_variants = [x.strip() for x in args.variants.split(",") if x.strip()]
        bad = sorted(set(selected_variants) - set(RECENT_VARIANTS))
        if bad:
            raise RuntimeError(f"Unknown recent-swing variants: {bad}")

    selected_types = ["RECENT_SWING"]

    script_path = Path(__file__)
    repo_root = find_repo_root(script_path)
    report_root = args.report_root or (repo_root / "reports" / "binance_um_recent_swing_robustness")
    report_root.mkdir(parents=True, exist_ok=True)
    output_root: Path = args.output_root
    events_root = output_root / "events"
    inventory_root = output_root / "inventory"
    metadata_root = output_root / "metadata"
    events_root.mkdir(parents=True, exist_ok=True); inventory_root.mkdir(parents=True, exist_ok=True); metadata_root.mkdir(parents=True, exist_ok=True)

    source_audit = verify_source_audit(args.historical_root)
    object_summary = verify_object_build(args.object_root)
    # Warm-up: 31 days for 30d exposure + trailing sigma. Outcomes need 240m after final event day.
    load_start = analysis_start - timedelta(days=31)
    load_end = analysis_end + timedelta(days=1)
    market_files = discover_market_files(args.historical_root, load_start, load_end)
    object_files = discover_object_files(args.object_root, analysis_start, analysis_end)

    config = {
        "evaluator_version": EVALUATOR_VERSION, "schema_version": SCHEMA_VERSION, "symbol": DEFAULT_SYMBOL,
        "analysis_start": analysis_start.isoformat(), "analysis_end": analysis_end.isoformat(),
        "primary_tolerance": PRIMARY_TOLERANCE_ID, "primary_endpoint": PRIMARY_ENDPOINT,
        "planned_mde_target_sd": PLANNED_MDE_TARGET_SD, "alpha": ALPHA, "power": POWER,
        "effect_estimator": "SOURCE_MATCHED_PAIRED",
        "tolerance_ids": tolerance_ids, "tolerance_specs": {k: TOLERANCE_SPECS[k] for k in tolerance_ids},
        "selected_object_types": selected_types, "selected_variants": selected_variants, "primary_variant": PRIMARY_VARIANT,
        "stale_bands_by_window": STALE_BANDS_BY_WINDOW,
        "stale_distance_max_bps": STALE_DISTANCE_MAX_BPS, "stale_exposure_log1p_max": STALE_EXPOSURE_LOG1P_MAX,
        "strict_random_distance_max_bps": STRICT_RANDOM_DISTANCE_MAX_BPS, "strict_random_exposure_log1p_max": STRICT_RANDOM_EXPOSURE_LOG1P_MAX,
        "arm_distance_multiplier": ARM_DISTANCE_MULTIPLIER, "arm_consecutive_minutes": ARM_CONSECUTIVE_MINUTES,
        "match_exposure_days": MATCH_EXPOSURE_DAYS, "match_active_window_pct": MATCH_ACTIVE_WINDOW_PCT,
        "source_audit_run_id": source_audit.get("run_id"), "phase5a_run_id": object_summary.get("run_id"),
        "phase5a_builder_version": object_summary.get("builder_version"),
        "anchor_archive_size": (args.object_root / "anchors" / "structural_anchors.parquet").stat().st_size,
        "anchor_archive_mtime_ns": (args.object_root / "anchors" / "structural_anchors.parquet").stat().st_mtime_ns,
    }
    signature = source_signature(market_files, object_files, config)
    config["signature"] = signature
    atomic_write_json(metadata_root / "run_config.json", config)

    print("Evaluating BTCUSDT recent-swing robustness...")
    print(f"Evaluator version : {EVALUATOR_VERSION}")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    print(f"Run ID            : {run_id}")
    print(f"Analysis window   : {analysis_start} to {analysis_end}")
    print(f"Historical root   : {args.historical_root}")
    print(f"Object root       : {args.object_root}")
    print(f"Output root       : {output_root}")
    print(f"Report root       : {report_root}")
    print(f"Object types      : {', '.join(selected_types)}")
    print(f"Variants          : {', '.join(selected_variants)}")
    print(f"Primary tolerance : {PRIMARY_TOLERANCE_ID}")
    print(f"Primary endpoint  : {PRIMARY_ENDPOINT}")
    print(f"Primary variant   : {PRIMARY_VARIANT}")
    print("Primary control   : STALE_8_14D genuine stale 1h swings")
    print("Phase-5A recent-swing definitions are frozen. Primary robustness contrast: source-matched 7d x 1h vs stale 8-14d swings.")

    market = load_market(market_files, load_start, load_end)
    objects = load_objects(object_files, analysis_start, analysis_end)
    objects = objects[(objects["object_type"].isin(selected_types)) & (objects["variant_id"].isin(selected_variants))].copy()
    anchors = load_anchor_archive(args.object_root)
    print(f"Loaded market rows : {len(market):,}")
    print(f"Loaded object states: {len(objects):,}")
    print(f"Loaded structural anchors: {len(anchors):,}")

    sigma = prepare_market_arrays(market)
    day_index = build_day_index(market)
    daily_exposure = build_daily_exposure_counts(market)
    zones_by_day = object_rows_by_day(objects)

    conn = init_state_db(metadata_root / "state.sqlite")
    all_events = []
    all_inventory = []
    years = list(range(analysis_start.year, analysis_end.year + 1))
    for pos, year in enumerate(years, 1):
        ev_path = events_root / f"year={year}" / "part-00000.parquet"
        inv_path = inventory_root / f"year={year}" / "part-00000.parquet"
        if not args.force and state_year_complete(conn, year, signature, ev_path, inv_path):
            ev = pd.read_parquet(ev_path)
            inv = pd.read_parquet(inv_path)
            print(f"[{pos:02d}/{len(years):02d}] {year}: COMPLETED_REUSED ({len(ev):,} events)")
        else:
            print(f"[{pos:02d}/{len(years):02d}] {year}: processing...")
            ev, inv_rows = process_year(year, market, sigma, day_index, zones_by_day, daily_exposure, anchors,
                                        tolerance_ids, analysis_start, analysis_end, run_id)
            inv = pd.DataFrame(inv_rows)
            atomic_write_parquet(ev_path, ev)
            atomic_write_parquet(inv_path, inv)
            mark_year(conn, year, signature, "COMPLETED", len(ev))
            print(f"[{pos:02d}/{len(years):02d}] {year}: COMPLETED ({len(ev):,} events)")
        all_events.append(ev)
        all_inventory.append(inv)
    conn.close()

    events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame(columns=EVENT_COLUMNS)
    inventory = pd.concat(all_inventory, ignore_index=True) if all_inventory else pd.DataFrame()
    pooled, yearly, variants = build_effect_tables(events)
    secondary = secondary_summary(events)
    confluence = confluence_summary(events)
    counts = build_event_count_table(events, inventory)
    matchq = control_match_quality(events)
    coverage = candidate_control_coverage(inventory)
    pair_coverage = source_matched_pair_coverage(events)
    primary_exact, primary_yearly, primary_sensitivity, primary_exact_dedup = preregistered_primary_tables(events)
    physical_dedup = physical_dedup_effect_table(events)
    confluence_strat = confluence_strata(events)
    checks = invariants(events, objects, analysis_start, analysis_end)
    failed = sum(c.status == "FAIL" for c in checks)
    status = "PASS" if failed == 0 else "FAIL"

    # Primary object-type behavior summary: raw REAL outcomes, separate from control effects.
    primary_real = events[(events["control_type"] == "REAL") & (events["tolerance_id"] == PRIMARY_TOLERANCE_ID)].copy()
    behavior_rows = []
    if len(primary_real):
        for (otype, variant), g in primary_real.groupby(["object_type", "variant_id"]):
            behavior_rows.append({
                "object_type": otype, "variant_id": variant, "events": len(g), "physical_events": int(g["physical_event_id"].nunique()),
                "mean_penetration_vol_60m": float(g["penetration_vol_60m"].mean()),
                "median_penetration_vol_60m": float(g["penetration_vol_60m"].median()),
                "mean_far_edge_penetration_vol_60m": float(g["far_edge_penetration_vol_60m"].mean()),
                "mean_rejection_vol_60m": float(g["rejection_vol_60m"].mean()),
                "mean_inside_zone_close_fraction_60m": float(g["inside_zone_close_fraction_60m"].mean()),
                "mean_beyond_far_edge_close_fraction_60m": float(g["beyond_far_edge_close_fraction_60m"].mean()),
                "rejection_share": float((g["exploratory_resolution_60m"] == "REJECTION").mean()),
                "clean_traversal_share": float((g["exploratory_resolution_60m"] == "CLEAN_TRAVERSAL").mean()),
                "acceptance_dwell_share": float((g["exploratory_resolution_60m"] == "ACCEPTANCE_DWELL").mean()),
            })
    behavior = pd.DataFrame(behavior_rows)

    atomic_write_dataframe_csv(report_root / "recent_swing_preregistered_primary.csv", primary_exact)
    atomic_write_dataframe_csv(report_root / "recent_swing_preregistered_primary_by_year.csv", primary_yearly)
    atomic_write_dataframe_csv(report_root / "recent_swing_preregistered_primary_tolerance_sensitivity.csv", primary_sensitivity)
    atomic_write_dataframe_csv(report_root / "recent_swing_preregistered_primary_physical_dedup.csv", primary_exact_dedup)
    atomic_write_dataframe_csv(report_root / "recent_swing_primary_effects.csv", pooled)
    atomic_write_dataframe_csv(report_root / "recent_swing_yearly_effects.csv", yearly)
    atomic_write_dataframe_csv(report_root / "recent_swing_control_variants.csv", variants)
    atomic_write_dataframe_csv(report_root / "recent_swing_behavior_summary.csv", behavior)
    atomic_write_dataframe_csv(report_root / "recent_swing_secondary_outcomes.csv", secondary)
    atomic_write_dataframe_csv(report_root / "recent_swing_confluence_summary.csv", confluence)
    atomic_write_dataframe_csv(report_root / "recent_swing_event_counts.csv", counts)
    atomic_write_dataframe_csv(report_root / "recent_swing_control_match_quality.csv", matchq)
    atomic_write_dataframe_csv(report_root / "recent_swing_candidate_control_coverage.csv", coverage)
    atomic_write_dataframe_csv(report_root / "recent_swing_source_matched_pair_coverage.csv", pair_coverage)
    atomic_write_dataframe_csv(report_root / "recent_swing_physical_dedup_effects.csv", physical_dedup)
    atomic_write_dataframe_csv(report_root / "recent_swing_confluence_strata.csv", confluence_strat)
    atomic_write_csv(report_root / "recent_swing_invariants.csv", [asdict(c) for c in checks])

    exact_primary_record = primary_exact.iloc[0].to_dict() if len(primary_exact) else {}
    exact_primary_status = str(exact_primary_record.get("status", "INSUFFICIENT"))
    exact_primary_effect_sd = exact_primary_record.get("effect_sd")
    exact_primary_n_real = int(exact_primary_record.get("n_real", 0) or 0)
    exact_primary_n_control = int(exact_primary_record.get("n_control", 0) or 0)

    summary = {
        "evaluator_version": EVALUATOR_VERSION, "schema_version": SCHEMA_VERSION, "run_id": run_id,
        "status": status, "analysis_start": analysis_start.isoformat(), "analysis_end": analysis_end.isoformat(),
        "market_rows": len(market), "phase5a_object_states": len(objects), "event_rows": len(events),
        "primary_events": int((events["tolerance_id"] == PRIMARY_TOLERANCE_ID).sum()) if len(events) else 0,
        "primary_real_events": int(((events["tolerance_id"] == PRIMARY_TOLERANCE_ID) & (events["control_type"] == "REAL")).sum()) if len(events) else 0,
        "primary_control_events": int(((events["tolerance_id"] == PRIMARY_TOLERANCE_ID) & (events["control_type"] != "REAL")).sum()) if len(events) else 0,
        "primary_physical_real_events": int(primary_real["physical_event_id"].nunique()) if len(primary_real) else 0,
        "failed_invariants": failed, "primary_tolerance": PRIMARY_TOLERANCE_ID, "primary_endpoint": PRIMARY_ENDPOINT,
        "planned_mde_target_sd": PLANNED_MDE_TARGET_SD, "selected_object_types": selected_types, "selected_variants": selected_variants, "primary_variant": PRIMARY_VARIANT,
        "phase5a_run_id": object_summary.get("run_id"), "phase5a_builder_version": object_summary.get("builder_version"),
        "primary_control_type": "STALE_SWING_ZONE", "primary_control_variant": "STALE_8_14D",
        "preregistered_primary_status": exact_primary_status, "preregistered_primary_effect_sd": exact_primary_effect_sd,
        "preregistered_primary_n_real": exact_primary_n_real, "preregistered_primary_n_control": exact_primary_n_control,
        "preregistered_primary_n_pairs": int(exact_primary_record.get("n_pairs", 0) or 0),
        "preregistered_primary_real_pair_coverage": exact_primary_record.get("real_pair_coverage"),
        "signature": signature,
    }
    atomic_write_json(metadata_root / "latest_summary.json", summary)
    atomic_write_json(report_root / "recent_swing_robustness_summary.json", summary)
    atomic_write_json(report_root / "recent_swing_robustness_config.json", config)
    write_findings(report_root / "recent_swing_robustness_findings.md", summary, pooled, yearly, confluence, checks)

    print("\nRecent-swing robustness summary")
    print("-------------------------------")
    print(f"Status                 : {status}")
    print(f"Phase-5A recent-swing states: {len(objects):,}")
    print(f"Events total           : {len(events):,}")
    print(f"Primary events         : {summary['primary_events']:,}")
    print(f"Primary real events    : {summary['primary_real_events']:,}")
    print(f"Primary control events : {summary['primary_control_events']:,}")
    print(f"Primary physical events: {summary['primary_physical_real_events']:,}")
    print(f"Preregistered primary : {exact_primary_status} | real {exact_primary_n_real:,} | stale {exact_primary_n_control:,} | effect SD {exact_primary_effect_sd}")
    print(f"Failed invariants      : {failed}")
    print(f"Summary                : {metadata_root / 'latest_summary.json'}")
    print(f"Findings               : {report_root / 'recent_swing_robustness_findings.md'}")
    print("Negative REAL-minus-CONTROL effects mean recent swings were harder to penetrate than the stale/strict-matched control.")
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
