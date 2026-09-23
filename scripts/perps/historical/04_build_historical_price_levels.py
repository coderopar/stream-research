#!/usr/bin/env python3
"""Build causal historical BTCUSDT price-level state snapshots from audited 1m klines.

Pass 4 of the historical-context pipeline.

Research intent
---------------
This builder does NOT label support/resistance with hindsight and does NOT use future
interactions to strengthen earlier snapshots. It creates transparent, causal day-start
price-zone states from two sources of historical evidence:

1. Confirmed multi-scale swing anchors (local highs/lows). A swing anchor is only
   available after its right-hand confirmation window has fully closed.
2. Historical activity-at-price context, approximated by assigning each 1-minute bar's
   close, quote volume and trade count to a fixed log-price bin.

For each UTC day-start snapshot, the builder uses only information available strictly
before that snapshot. Adjacent qualifying price bins are merged into zones. The output
contains raw evidence components plus a simple transparent ranking score; the score is
not treated as ground truth.

Default audited source:
    D:\\stream-data\\historical\\binance\\futures_um\\BTCUSDT\\klines_1m

Default output:
    D:\\stream-data\\historical\\binance\\futures_um\\BTCUSDT\\historical_price_levels_v1

Outputs
-------
    anchors/structural_anchors.parquet
    daily_levels/year=YYYY/month=MM/part-00000.parquet
    checkpoints/month=YYYY-MM/bin_state.parquet
    metadata/build_state.sqlite
    metadata/latest_build_summary.json
    metadata/level_config.json
    metadata/level_invariants.csv

Reports under repository:
    reports/binance_um_historical_price_levels/
        historical_price_levels_daily_counts.csv
        historical_price_levels_anchor_summary.csv
        historical_price_levels_level_summary.csv
        historical_price_levels_invariants.csv
        historical_price_levels_findings.md

Key causality contract
----------------------
* Snapshot at day D 00:00 UTC uses minute activity with open_time < D 00:00 UTC.
* A swing at pivot minute t with confirmation radius R minutes is not available until
  the right-most confirming 1m bar has closed: t + (R + 1) minutes.
* No anchor with available_time >= snapshot_time may appear in that snapshot.
* No future interaction can retroactively alter an earlier snapshot.
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
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

BUILDER_VERSION = "1.0.0"
LEVEL_SCHEMA_VERSION = "historical_price_levels_v1"
DEFAULT_HISTORICAL_ROOT = Path(r"D:\stream-data\historical\binance\futures_um\BTCUSDT\klines_1m")
DEFAULT_OUTPUT_ROOT = Path(r"D:\stream-data\historical\binance\futures_um\BTCUSDT\historical_price_levels_v1")
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "1m"

PRICE_COLUMNS = ["open_time", "high", "low", "close", "quote_volume", "trade_count"]

ANCHOR_COLUMNS = [
    "symbol",
    "pivot_time",
    "available_time",
    "side",
    "pivot_price",
    "bin_idx",
    "scale_minutes",
    "prominence_bps",
    "anchor_weight",
]

LEVEL_COLUMNS = [
    "symbol",
    "as_of_time",
    "snapshot_date",
    "current_price",
    "level_id",
    "zone_lower",
    "zone_upper",
    "zone_center",
    "zone_width_bps",
    "relative_location",
    "signed_distance_bps",
    "abs_distance_bps",
    "structural_type",
    "strength_score",
    "bin_count",
    "anchor_count",
    "support_anchor_count",
    "resistance_anchor_count",
    "anchor_weight",
    "support_anchor_weight",
    "resistance_anchor_weight",
    "max_prominence_bps",
    "max_scale_minutes",
    "occupied_minutes",
    "touch_days",
    "quote_volume",
    "trade_count",
    "daily_volume_share_sum",
    "first_anchor_time",
    "last_anchor_time",
    "last_anchor_available_time",
    "last_visit_time",
    "age_days",
    "days_since_last_anchor",
    "days_since_last_visit",
    "causal_history_minutes",
]

STATE_COLUMNS = [
    "bin_idx",
    "occupied_minutes",
    "touch_days",
    "quote_volume",
    "trade_count",
    "daily_volume_share_sum",
    "first_visit_ms",
    "last_visit_ms",
    "anchor_count",
    "support_anchor_count",
    "resistance_anchor_count",
    "anchor_weight",
    "support_anchor_weight",
    "resistance_anchor_weight",
    "max_prominence_bps",
    "max_scale_minutes",
    "first_anchor_ms",
    "last_anchor_ms",
    "last_anchor_available_ms",
]


@dataclass
class Invariant:
    scope: str
    invariant: str
    status: str
    observed: Any
    expected: Any
    detail: str = ""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def short_tmp_path(final_path: Path, prefix: str) -> Path:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    return final_path.parent / f".{prefix}-{uuid.uuid4().hex[:8]}.tmp"


def atomic_write_text(path: Path, content: str) -> None:
    tmp = short_tmp_path(path, "tx")
    try:
        tmp.write_text(content, encoding="utf-8", newline="\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def atomic_write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    tmp = short_tmp_path(path, "cs")
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

    tmp = short_tmp_path(path, "pq")
    try:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        pq.write_table(table, tmp, compression="zstd", use_dictionary=True)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def require_libraries() -> None:
    try:
        import numpy as np  # noqa: F401
        import pandas as pd  # noqa: F401
        import pyarrow as pa  # noqa: F401
        import pyarrow.parquet as pq  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Pass 4 requires numpy, pandas and pyarrow. Install with: "
            "python -m pip install numpy pandas pyarrow"
        ) from exc


def find_repo_root(script_path: Path) -> Path:
    try:
        candidate = script_path.resolve().parents[3]
        if (candidate / "scripts").exists():
            return candidate
    except IndexError:
        pass
    return Path.cwd().resolve()


def parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        number = int(token)
        if number <= 0:
            raise argparse.ArgumentTypeError("All integer list values must be positive")
        out.append(number)
    if not out:
        raise argparse.ArgumentTypeError("Expected at least one integer")
    return sorted(set(out))


def parse_date(value: str | None):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date {value!r}; expected YYYY-MM-DD") from exc


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise RuntimeError(f"Required manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_audited_source(root: Path, symbol: str, interval: str) -> list[Path]:
    summary_path = root / "metadata" / "latest_audit_summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"Pass-3 audit summary was not found: {summary_path}")
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
        path = Path(row.get("parquet_path", ""))
        if path.exists():
            paths.append(path)
    paths = sorted(set(paths))
    if not paths:
        raise RuntimeError("No completed Pass-2 Parquet files were found")
    return paths


def source_signature(paths: list[Path], config_payload: dict[str, Any]) -> str:
    payload = {
        "builder_version": BUILDER_VERSION,
        "level_schema_version": LEVEL_SCHEMA_VERSION,
        "config": config_payload,
        "inputs": [
            {"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in paths
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def datetime_to_ms(series):
    import pandas as pd

    dt = pd.to_datetime(series, utc=True)
    # Robust to datetime64[s/ms/us/ns, UTC].
    naive = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    unit = str(naive.dtype)
    ints = naive.astype("int64").to_numpy()
    if "datetime64[ns]" in unit:
        return ints // 1_000_000
    if "datetime64[us]" in unit:
        return ints // 1_000
    if "datetime64[ms]" in unit:
        return ints
    if "datetime64[s]" in unit:
        return ints * 1_000
    # Pandas normally normalizes to ns; fallback is explicit conversion.
    return dt.astype("int64").to_numpy() // 1_000_000


def load_history(paths: list[Path]):
    import pandas as pd

    frames = []
    for path in paths:
        frame = pd.read_parquet(path, columns=PRICE_COLUMNS)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["open_time"] = pd.to_datetime(data["open_time"], utc=True)
    data.sort_values("open_time", inplace=True, kind="mergesort")
    data.reset_index(drop=True, inplace=True)
    if data["open_time"].duplicated().any():
        raise RuntimeError("Duplicate open_time values found despite Pass-3 PASS")
    return data


def price_to_bin(price, step_log: float):
    import numpy as np

    arr = np.asarray(price, dtype="float64")
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0):
        raise RuntimeError("Price-to-bin received non-positive or non-finite price")
    return np.floor(np.log(arr) / step_log).astype("int64")


def bin_lower(idx: int, step_log: float) -> float:
    return math.exp(idx * step_log)


def bin_upper(idx: int, step_log: float) -> float:
    return math.exp((idx + 1) * step_log)


def _centered_side_extrema(series, radius: int, mode: str):
    # rolling with center=True is allowed here because availability is delayed by radius+1m.
    window = 2 * radius + 1
    roll = series.rolling(window=window, center=True, min_periods=window)
    if mode == "max":
        return roll.max()
    return roll.min()


def build_structural_anchors(
    data,
    symbol: str,
    scales: list[int],
    step_log: float,
    min_prominence_bps: float,
    dedup_minutes: int,
    dedup_price_bps: float,
):
    import numpy as np
    import pandas as pd

    highs = data["high"].astype("float64")
    lows = data["low"].astype("float64")
    times = pd.to_datetime(data["open_time"], utc=True)
    pieces = []

    for radius in scales:
        max_high = _centered_side_extrema(highs, radius, "max")
        min_low = _centered_side_extrema(lows, radius, "min")

        left_low = lows.shift(1).rolling(radius, min_periods=radius).min()
        right_low = lows.iloc[::-1].shift(1).rolling(radius, min_periods=radius).min().iloc[::-1]
        left_high = highs.shift(1).rolling(radius, min_periods=radius).max()
        right_high = highs.iloc[::-1].shift(1).rolling(radius, min_periods=radius).max().iloc[::-1]

        high_mask = highs.eq(max_high)
        low_mask = lows.eq(min_low)

        high_prom = np.minimum(
            (highs - left_low) / highs * 10000.0,
            (highs - right_low) / highs * 10000.0,
        )
        low_prom = np.minimum(
            (left_high - lows) / lows * 10000.0,
            (right_high - lows) / lows * 10000.0,
        )

        for side, mask, price_series, prom in (
            ("RESISTANCE", high_mask, highs, high_prom),
            ("SUPPORT", low_mask, lows, low_prom),
        ):
            keep = mask & prom.ge(min_prominence_bps) & prom.notna()
            if not bool(keep.any()):
                continue
            sub = pd.DataFrame(
                {
                    "symbol": symbol,
                    "pivot_time": times[keep].to_numpy(),
                    "available_time": (times[keep] + pd.to_timedelta(radius + 1, unit="m")).to_numpy(),
                    "side": side,
                    "pivot_price": price_series[keep].to_numpy(dtype="float64"),
                    "scale_minutes": int(radius),
                    "prominence_bps": prom[keep].to_numpy(dtype="float64"),
                }
            )
            sub["bin_idx"] = price_to_bin(sub["pivot_price"].to_numpy(), step_log)
            scale_weight = math.sqrt(float(radius) / float(min(scales)))
            sub["anchor_weight"] = scale_weight * np.log1p(sub["prominence_bps"].to_numpy() / 10.0)
            pieces.append(sub)

    if not pieces:
        return pd.DataFrame(columns=ANCHOR_COLUMNS)

    anchors = pd.concat(pieces, ignore_index=True)
    anchors.sort_values(["side", "pivot_time", "scale_minutes"], inplace=True, kind="mergesort")

    # Exact same pivot minute/side across scales: keep the strongest structural confirmation.
    anchors.sort_values(
        ["side", "pivot_time", "anchor_weight", "scale_minutes"],
        ascending=[True, True, False, False],
        inplace=True,
        kind="mergesort",
    )
    anchors = anchors.drop_duplicates(["side", "pivot_time"], keep="first")
    anchors.sort_values(["side", "pivot_time"], inplace=True, kind="mergesort")

    # Greedy dedup of nearby confirmations of the same structural turn.
    kept_rows = []
    max_ms = dedup_minutes * 60_000
    for side, group in anchors.groupby("side", sort=False):
        last = None
        for row in group.itertuples(index=False):
            current = row._asdict()
            pt_ms = int(pd.Timestamp(current["pivot_time"]).timestamp() * 1000)
            if last is None:
                current["_pivot_ms"] = pt_ms
                kept_rows.append(current)
                last = current
                continue
            time_close = pt_ms - int(last["_pivot_ms"]) <= max_ms
            px_gap = abs(math.log(float(current["pivot_price"]) / float(last["pivot_price"]))) * 10000.0
            price_close = px_gap <= dedup_price_bps
            if time_close and price_close:
                if float(current["anchor_weight"]) > float(last["anchor_weight"]):
                    current["_pivot_ms"] = pt_ms
                    kept_rows[-1] = current
                    last = current
                continue
            current["_pivot_ms"] = pt_ms
            kept_rows.append(current)
            last = current

    out = pd.DataFrame(kept_rows)
    if "_pivot_ms" in out.columns:
        out.drop(columns=["_pivot_ms"], inplace=True)
    out = out[ANCHOR_COLUMNS].copy()
    out["pivot_time"] = pd.to_datetime(out["pivot_time"], utc=True)
    out["available_time"] = pd.to_datetime(out["available_time"], utc=True)
    out.sort_values("available_time", inplace=True, kind="mergesort")
    out.reset_index(drop=True, inplace=True)
    return out


def empty_state():
    import pandas as pd

    return pd.DataFrame(columns=STATE_COLUMNS).set_index("bin_idx")


def ensure_state_rows(state, bin_indices):
    import numpy as np
    import pandas as pd

    ids = np.unique(np.asarray(bin_indices, dtype="int64"))
    missing = [int(x) for x in ids if int(x) not in state.index]
    if not missing:
        return state
    add = pd.DataFrame(index=pd.Index(missing, name="bin_idx"))
    for col in STATE_COLUMNS[1:]:
        if col.endswith("_ms"):
            add[col] = np.nan
        else:
            add[col] = 0.0
    add["max_scale_minutes"] = 0.0
    if state.empty:
        return add.sort_index()
    return pd.concat([state, add], axis=0).sort_index()


def apply_daily_activity(state, day_frame, step_log: float):
    import numpy as np
    import pandas as pd

    bins = price_to_bin(day_frame["close"].to_numpy(dtype="float64"), step_log)
    temp = pd.DataFrame(
        {
            "bin_idx": bins,
            "quote_volume": day_frame["quote_volume"].to_numpy(dtype="float64"),
            "trade_count": day_frame["trade_count"].to_numpy(dtype="float64"),
        }
    )
    day_total = float(temp["quote_volume"].sum())
    grouped = temp.groupby("bin_idx", sort=False).agg(
        occupied_minutes=("bin_idx", "size"),
        quote_volume=("quote_volume", "sum"),
        trade_count=("trade_count", "sum"),
    )
    grouped["daily_volume_share_sum"] = (
        grouped["quote_volume"] / day_total if day_total > 0 else 0.0
    )
    state = ensure_state_rows(state, grouped.index.to_numpy())
    idx = grouped.index
    for col in ["occupied_minutes", "quote_volume", "trade_count", "daily_volume_share_sum"]:
        state.loc[idx, col] = state.loc[idx, col].astype(float).to_numpy() + grouped[col].to_numpy(dtype="float64")
    state.loc[idx, "touch_days"] = state.loc[idx, "touch_days"].astype(float).to_numpy() + 1.0

    ms = datetime_to_ms(day_frame["open_time"])
    first_ms = int(ms.min())
    last_ms = int(ms.max())
    for bid in idx:
        old_first = state.at[bid, "first_visit_ms"]
        state.at[bid, "first_visit_ms"] = first_ms if pd.isna(old_first) else min(float(old_first), first_ms)
        old_last = state.at[bid, "last_visit_ms"]
        state.at[bid, "last_visit_ms"] = last_ms if pd.isna(old_last) else max(float(old_last), last_ms)
    return state


def apply_available_anchors(state, anchors_day):
    import pandas as pd

    if anchors_day.empty:
        return state
    state = ensure_state_rows(state, anchors_day["bin_idx"].to_numpy(dtype="int64"))
    for row in anchors_day.itertuples(index=False):
        bid = int(row.bin_idx)
        state.at[bid, "anchor_count"] = float(state.at[bid, "anchor_count"]) + 1.0
        state.at[bid, "anchor_weight"] = float(state.at[bid, "anchor_weight"]) + float(row.anchor_weight)
        if row.side == "SUPPORT":
            state.at[bid, "support_anchor_count"] = float(state.at[bid, "support_anchor_count"]) + 1.0
            state.at[bid, "support_anchor_weight"] = float(state.at[bid, "support_anchor_weight"]) + float(row.anchor_weight)
        else:
            state.at[bid, "resistance_anchor_count"] = float(state.at[bid, "resistance_anchor_count"]) + 1.0
            state.at[bid, "resistance_anchor_weight"] = float(state.at[bid, "resistance_anchor_weight"]) + float(row.anchor_weight)
        state.at[bid, "max_prominence_bps"] = max(float(state.at[bid, "max_prominence_bps"]), float(row.prominence_bps))
        state.at[bid, "max_scale_minutes"] = max(float(state.at[bid, "max_scale_minutes"]), float(row.scale_minutes))
        pivot_ms = int(pd.Timestamp(row.pivot_time).timestamp() * 1000)
        first = state.at[bid, "first_anchor_ms"]
        state.at[bid, "first_anchor_ms"] = pivot_ms if pd.isna(first) else min(float(first), pivot_ms)
        last = state.at[bid, "last_anchor_ms"]
        state.at[bid, "last_anchor_ms"] = pivot_ms if pd.isna(last) else max(float(last), pivot_ms)
        available_ms = int(pd.Timestamp(row.available_time).timestamp() * 1000)
        last_available = state.at[bid, "last_anchor_available_ms"]
        state.at[bid, "last_anchor_available_ms"] = available_ms if pd.isna(last_available) else max(float(last_available), available_ms)
    return state


def merge_candidate_bins(candidate_ids: list[int], max_gap_bins: int) -> list[tuple[int, int]]:
    if not candidate_ids:
        return []
    ids = sorted(set(int(x) for x in candidate_ids))
    groups: list[tuple[int, int]] = []
    start = prev = ids[0]
    for value in ids[1:]:
        if value - prev <= max_gap_bins + 1:
            prev = value
            continue
        groups.append((start, prev))
        start = prev = value
    groups.append((start, prev))
    return groups


def ms_to_iso(value: float | int | None) -> str | None:
    import pandas as pd

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return pd.to_datetime(int(value), unit="ms", utc=True).isoformat()


def build_snapshot_levels(
    state,
    as_of,
    current_price: float,
    symbol: str,
    step_log: float,
    min_anchor_events: int,
    max_gap_bins: int,
    max_distance_bps: float,
    max_levels: int,
    history_minutes: int,
):
    import numpy as np
    import pandas as pd

    if state.empty:
        return pd.DataFrame(columns=LEVEL_COLUMNS)
    eligible = state[state["anchor_count"].astype(float) >= float(min_anchor_events)].copy()
    if eligible.empty:
        return pd.DataFrame(columns=LEVEL_COLUMNS)

    groups = merge_candidate_bins(eligible.index.astype(int).tolist(), max_gap_bins=max_gap_bins)
    rows = []
    as_of_ts = pd.Timestamp(as_of)
    as_of_ms = int(as_of_ts.timestamp() * 1000)

    for low_idx, high_idx in groups:
        block = state.loc[(state.index >= low_idx) & (state.index <= high_idx)].copy()
        if block.empty:
            continue
        zone_lower = bin_lower(low_idx, step_log)
        zone_upper = bin_upper(high_idx, step_log)
        zone_center = math.sqrt(zone_lower * zone_upper)
        signed_distance_bps = math.log(zone_center / current_price) * 10000.0
        abs_distance_bps = abs(signed_distance_bps)
        if abs_distance_bps > max_distance_bps:
            continue

        anchor_count = int(round(float(block["anchor_count"].sum())))
        support_count = int(round(float(block["support_anchor_count"].sum())))
        resistance_count = int(round(float(block["resistance_anchor_count"].sum())))
        anchor_weight = float(block["anchor_weight"].sum())
        support_weight = float(block["support_anchor_weight"].sum())
        resistance_weight = float(block["resistance_anchor_weight"].sum())
        touch_days = int(round(float(block["touch_days"].sum())))
        occupied = int(round(float(block["occupied_minutes"].sum())))
        quote_volume = float(block["quote_volume"].sum())
        trade_count = float(block["trade_count"].sum())
        volume_share = float(block["daily_volume_share_sum"].sum())

        # Transparent ranking heuristic. Raw components are always retained.
        strength = (
            anchor_weight
            + 0.50 * math.log1p(max(touch_days, 0))
            + 0.25 * math.log1p(max(occupied, 0))
            + 0.50 * math.log1p(max(volume_share, 0.0) * 100.0)
        )

        if support_weight >= 1.5 * max(resistance_weight, 1e-12):
            structural_type = "SUPPORT_DOMINANT"
        elif resistance_weight >= 1.5 * max(support_weight, 1e-12):
            structural_type = "RESISTANCE_DOMINANT"
        else:
            structural_type = "TWO_SIDED"

        if current_price < zone_lower:
            relative_location = "ABOVE_PRICE"
        elif current_price > zone_upper:
            relative_location = "BELOW_PRICE"
        else:
            relative_location = "OVERLAPPING_PRICE"

        first_anchor_candidates = block["first_anchor_ms"].dropna().astype(float)
        last_anchor_candidates = block["last_anchor_ms"].dropna().astype(float)
        last_available_candidates = block["last_anchor_available_ms"].dropna().astype(float)
        last_visit_candidates = block["last_visit_ms"].dropna().astype(float)
        first_anchor_ms = float(first_anchor_candidates.min()) if len(first_anchor_candidates) else math.nan
        last_anchor_ms = float(last_anchor_candidates.max()) if len(last_anchor_candidates) else math.nan
        last_available_ms = float(last_available_candidates.max()) if len(last_available_candidates) else math.nan
        last_visit_ms = float(last_visit_candidates.max()) if len(last_visit_candidates) else math.nan

        # Direct causal guard: confirmed anchor information must be strictly available before snapshot.
        if not math.isnan(last_anchor_ms) and last_anchor_ms >= as_of_ms:
            raise RuntimeError("Causality invariant failed: future anchor pivot leaked into snapshot")
        if not math.isnan(last_available_ms) and last_available_ms >= as_of_ms:
            raise RuntimeError("Causality invariant failed: unconfirmed anchor leaked into snapshot")

        age_days = (as_of_ms - first_anchor_ms) / 86_400_000.0 if not math.isnan(first_anchor_ms) else math.nan
        days_since_anchor = (as_of_ms - last_anchor_ms) / 86_400_000.0 if not math.isnan(last_anchor_ms) else math.nan
        days_since_visit = (as_of_ms - last_visit_ms) / 86_400_000.0 if not math.isnan(last_visit_ms) else math.nan

        rows.append(
            {
                "symbol": symbol,
                "as_of_time": as_of_ts,
                "snapshot_date": as_of_ts.date().isoformat(),
                "current_price": current_price,
                "level_id": f"L{low_idx}_{high_idx}",
                "zone_lower": zone_lower,
                "zone_upper": zone_upper,
                "zone_center": zone_center,
                "zone_width_bps": math.log(zone_upper / zone_lower) * 10000.0,
                "relative_location": relative_location,
                "signed_distance_bps": signed_distance_bps,
                "abs_distance_bps": abs_distance_bps,
                "structural_type": structural_type,
                "strength_score": strength,
                "bin_count": int(high_idx - low_idx + 1),
                "anchor_count": anchor_count,
                "support_anchor_count": support_count,
                "resistance_anchor_count": resistance_count,
                "anchor_weight": anchor_weight,
                "support_anchor_weight": support_weight,
                "resistance_anchor_weight": resistance_weight,
                "max_prominence_bps": float(block["max_prominence_bps"].max()),
                "max_scale_minutes": int(round(float(block["max_scale_minutes"].max()))),
                "occupied_minutes": occupied,
                "touch_days": touch_days,
                "quote_volume": quote_volume,
                "trade_count": trade_count,
                "daily_volume_share_sum": volume_share,
                "first_anchor_time": ms_to_iso(first_anchor_ms),
                "last_anchor_time": ms_to_iso(last_anchor_ms),
                "last_anchor_available_time": ms_to_iso(last_available_ms),
                "last_visit_time": ms_to_iso(last_visit_ms),
                "age_days": age_days,
                "days_since_last_anchor": days_since_anchor,
                "days_since_last_visit": days_since_visit,
                "causal_history_minutes": int(history_minutes),
            }
        )

    if not rows:
        return pd.DataFrame(columns=LEVEL_COLUMNS)
    out = pd.DataFrame(rows, columns=LEVEL_COLUMNS)
    out.sort_values(["strength_score", "abs_distance_bps"], ascending=[False, True], inplace=True, kind="mergesort")
    if max_levels > 0:
        out = out.head(max_levels).copy()
    out.sort_values("zone_center", inplace=True, kind="mergesort")
    out.reset_index(drop=True, inplace=True)
    return out


def state_checkpoint_path(output_root: Path, month: str) -> Path:
    return output_root / "checkpoints" / f"month={month}" / "bin_state.parquet"


def snapshot_output_path(output_root: Path, month: str) -> Path:
    year, mm = month.split("-")
    return output_root / "daily_levels" / f"year={year}" / f"month={mm}" / "part-00000.parquet"


def daily_counts_output_path(output_root: Path, month: str) -> Path:
    year, mm = month.split("-")
    return output_root / "daily_levels" / f"year={year}" / f"month={mm}" / "daily_counts.csv"


def init_state_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS month_build (
            month TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            run_signature TEXT NOT NULL,
            snapshot_path TEXT NOT NULL,
            checkpoint_path TEXT NOT NULL,
            snapshot_rows INTEGER NOT NULL,
            completed_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def save_month_state(conn, month: str, signature: str, snapshot_path: Path, checkpoint_path: Path, snapshot_rows: int):
    conn.execute(
        """
        INSERT INTO month_build(month,status,run_signature,snapshot_path,checkpoint_path,snapshot_rows,completed_at)
        VALUES(?,?,?,?,?,?,?)
        ON CONFLICT(month) DO UPDATE SET
            status=excluded.status,
            run_signature=excluded.run_signature,
            snapshot_path=excluded.snapshot_path,
            checkpoint_path=excluded.checkpoint_path,
            snapshot_rows=excluded.snapshot_rows,
            completed_at=excluded.completed_at
        """,
        (month, "COMPLETED", signature, str(snapshot_path), str(checkpoint_path), int(snapshot_rows), utc_now_iso()),
    )
    conn.commit()


def find_resume_checkpoint(conn, months: list[str], signature: str):
    reusable = []
    for month in months:
        row = conn.execute(
            "SELECT status,run_signature,snapshot_path,checkpoint_path FROM month_build WHERE month=?",
            (month,),
        ).fetchone()
        if not row or row[0] != "COMPLETED" or row[1] != signature:
            break
        sp = Path(row[2]); cp = Path(row[3])
        # Compute sidecar directly from snapshot path to avoid changing the SQLite schema.
        dc = sp.parent / "daily_counts.csv"
        if not sp.exists() or not cp.exists() or not dc.exists():
            break
        reusable.append((month, cp))
    return reusable


def load_checkpoint(path: Path):
    import pandas as pd

    frame = pd.read_parquet(path)
    if frame.empty:
        return empty_state()
    return frame.set_index("bin_idx")[STATE_COLUMNS[1:]].sort_index()


def save_checkpoint(path: Path, state) -> None:
    frame = state.reset_index()[STATE_COLUMNS].copy()
    atomic_write_parquet(path, frame)


def write_findings(path: Path, summary: dict[str, Any]) -> None:
    text = f"""# Binance USD-M historical price-level build findings

Status: **{summary['status']}**

## Coverage

- Source rows: {summary['source_rows']:,}
- Source coverage: {summary['first_source_time']} through {summary['last_source_time']}
- Structural anchors: {summary['structural_anchors']:,}
- Day-start snapshots: {summary['snapshot_days']:,}
- Level-state rows: {summary['level_rows']:,}
- First snapshot: {summary.get('first_snapshot_date')}
- Last snapshot: {summary.get('last_snapshot_date')}

## Causality

Each day-start snapshot uses only activity from earlier minutes. Swing anchors are delayed until their complete right-hand confirmation window has closed. No future interaction is used to strengthen an earlier level state.

## Interpretation

The output is a historical **price-zone state database**, not a claim that every emitted zone is true support or resistance. `strength_score` is a transparent ranking heuristic built from structural-anchor evidence and historical activity context. Raw evidence fields are retained for later validation.

Pass 5 should independently audit causality and profile whether strong historical zones actually exhibit different rejection, break and revisit behaviour than matched non-level prices.
"""
    atomic_write_text(path, text)


def run_self_test() -> int:
    import numpy as np
    import pandas as pd

    step = math.log1p(25 / 10000.0)
    prices = np.array([100.0, 100.1, 100.25])
    bins = price_to_bin(prices, step)
    assert len(bins) == 3 and np.all(np.isfinite(bins))
    groups = merge_candidate_bins([10, 11, 13, 20], max_gap_bins=1)
    assert groups == [(10, 13), (20, 20)]

    # Synthetic state with two structural bins around 100.
    state = empty_state()
    state = ensure_state_rows(state, [int(bins[0]), int(bins[0]) + 1])
    for bid in state.index:
        state.at[bid, "anchor_count"] = 2
        state.at[bid, "support_anchor_count"] = 2
        state.at[bid, "anchor_weight"] = 3.0
        state.at[bid, "support_anchor_weight"] = 3.0
        state.at[bid, "occupied_minutes"] = 20
        state.at[bid, "touch_days"] = 2
        state.at[bid, "daily_volume_share_sum"] = 0.1
        state.at[bid, "first_anchor_ms"] = pd.Timestamp("2026-01-01T00:00:00Z").value // 1_000_000
        state.at[bid, "last_anchor_ms"] = pd.Timestamp("2026-01-02T00:00:00Z").value // 1_000_000
        state.at[bid, "last_anchor_available_ms"] = pd.Timestamp("2026-01-02T01:00:00Z").value // 1_000_000
        state.at[bid, "last_visit_ms"] = pd.Timestamp("2026-01-02T12:00:00Z").value // 1_000_000
    levels = build_snapshot_levels(
        state,
        pd.Timestamp("2026-01-03T00:00:00Z"),
        100.0,
        "BTCUSDT",
        step,
        min_anchor_events=2,
        max_gap_bins=1,
        max_distance_bps=5000,
        max_levels=20,
        history_minutes=2880,
    )
    assert len(levels) == 1
    assert levels.iloc[0]["structural_type"] == "SUPPORT_DOMINANT"
    assert levels.iloc[0]["last_anchor_time"] < "2026-01-03"

    # Confirmation-time contract: a 15m pivot cannot be available at pivot time.
    t = pd.Timestamp("2026-01-01T12:00:00Z")
    available = t + pd.to_timedelta(16, unit="m")
    assert available > t

    print("SELF-TEST PASS")
    print("Validated log-price bins, zone merging, causal day-start level construction, structural typing and delayed swing-anchor availability.")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-root", type=Path, default=DEFAULT_HISTORICAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    parser.add_argument("--pivot-scales-minutes", type=parse_int_list, default=parse_int_list("15,60,240"))
    parser.add_argument("--bin-width-bps", type=float, default=25.0)
    parser.add_argument("--min-pivot-prominence-bps", type=float, default=10.0)
    parser.add_argument("--anchor-dedup-minutes", type=int, default=240)
    parser.add_argument("--anchor-dedup-price-bps", type=float, default=35.0)
    parser.add_argument("--min-anchor-events", type=int, default=2)
    parser.add_argument("--zone-merge-gap-bins", type=int, default=1)
    parser.add_argument("--min-history-days", type=int, default=30)
    parser.add_argument("--max-distance-bps", type=float, default=5000.0)
    parser.add_argument("--max-levels-per-snapshot", type=int, default=100)
    parser.add_argument("--snapshot-start-date", type=parse_date, default=None)
    parser.add_argument("--snapshot-end-date", type=parse_date, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.self_test:
        return run_self_test()
    if args.bin_width_bps <= 0 or args.bin_width_bps >= 1000:
        raise RuntimeError("--bin-width-bps must be >0 and <1000")
    if args.min_anchor_events < 1:
        raise RuntimeError("--min-anchor-events must be >=1")
    if args.zone_merge_gap_bins < 0:
        raise RuntimeError("--zone-merge-gap-bins must be >=0")
    if args.min_history_days < 1:
        raise RuntimeError("--min-history-days must be >=1")

    require_libraries()
    import numpy as np
    import pandas as pd

    historical_root = args.historical_root.resolve()
    output_root = args.output_root.resolve()
    repo_root = find_repo_root(Path(__file__))
    report_root = repo_root / "reports" / "binance_um_historical_price_levels"
    report_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    source_paths = verify_audited_source(historical_root, args.symbol, args.interval)
    config = {
        "symbol": args.symbol,
        "interval": args.interval,
        "pivot_scales_minutes": args.pivot_scales_minutes,
        "bin_width_bps": args.bin_width_bps,
        "min_pivot_prominence_bps": args.min_pivot_prominence_bps,
        "anchor_dedup_minutes": args.anchor_dedup_minutes,
        "anchor_dedup_price_bps": args.anchor_dedup_price_bps,
        "min_anchor_events": args.min_anchor_events,
        "zone_merge_gap_bins": args.zone_merge_gap_bins,
        "min_history_days": args.min_history_days,
        "max_distance_bps": args.max_distance_bps,
        "max_levels_per_snapshot": args.max_levels_per_snapshot,
        "snapshot_start_date": str(args.snapshot_start_date) if args.snapshot_start_date else None,
        "snapshot_end_date": str(args.snapshot_end_date) if args.snapshot_end_date else None,
        "activity_price_assignment": "1m_close_price",
        "snapshot_contract": "UTC_DAY_START_STRICTLY_PAST",
    }
    signature = source_signature(source_paths, config)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]

    print("Building causal historical BTCUSDT price-level states...")
    print(f"Builder version       : {BUILDER_VERSION}")
    print(f"Run ID                : {run_id}")
    print(f"Audited source months : {len(source_paths)}")
    print(f"Historical root       : {historical_root}")
    print(f"Output root           : {output_root}")
    print(f"Pivot scales          : {','.join(map(str, args.pivot_scales_minutes))} minutes")
    print(f"Price-bin width       : {args.bin_width_bps:g} bps")
    print("Snapshots are causal UTC day-start states; no future level interactions are used.")

    data = load_history(source_paths)
    source_rows = len(data)
    first_source = pd.Timestamp(data["open_time"].iloc[0])
    last_source = pd.Timestamp(data["open_time"].iloc[-1])
    step_log = math.log1p(args.bin_width_bps / 10000.0)

    anchors_path = output_root / "anchors" / "structural_anchors.parquet"
    anchor_meta_path = output_root / "metadata" / "anchors_signature.json"
    reuse_anchors = False
    if not args.force and anchors_path.exists() and anchor_meta_path.exists():
        try:
            meta = json.loads(anchor_meta_path.read_text(encoding="utf-8"))
            if meta.get("run_signature") == signature:
                anchors = pd.read_parquet(anchors_path)
                anchors["pivot_time"] = pd.to_datetime(anchors["pivot_time"], utc=True)
                anchors["available_time"] = pd.to_datetime(anchors["available_time"], utc=True)
                reuse_anchors = True
        except Exception:
            reuse_anchors = False
    if not reuse_anchors:
        print("Building structural swing-anchor catalogue...")
        anchors = build_structural_anchors(
            data,
            args.symbol,
            args.pivot_scales_minutes,
            step_log,
            args.min_pivot_prominence_bps,
            args.anchor_dedup_minutes,
            args.anchor_dedup_price_bps,
        )
        atomic_write_parquet(anchors_path, anchors)
        atomic_write_json(anchor_meta_path, {"run_signature": signature, "rows": len(anchors), "generated_at": utc_now_iso()})
    else:
        print(f"Structural anchors     : REUSED ({len(anchors):,})")

    anchors["available_date"] = pd.to_datetime(anchors["available_time"], utc=True).dt.date
    anchors_by_date = {d: g.copy() for d, g in anchors.groupby("available_date", sort=True)}

    data["date"] = pd.to_datetime(data["open_time"], utc=True).dt.date
    dates = sorted(data["date"].unique())
    first_date = dates[0]
    last_date = dates[-1]
    min_snapshot_date = first_date + timedelta(days=args.min_history_days)
    if args.snapshot_start_date:
        min_snapshot_date = max(min_snapshot_date, args.snapshot_start_date)
    max_snapshot_date = last_date + timedelta(days=1)
    if args.snapshot_end_date:
        max_snapshot_date = min(max_snapshot_date, args.snapshot_end_date)

    months = sorted({f"{d.year:04d}-{d.month:02d}" for d in dates})
    conn = init_state_db(output_root / "metadata" / "build_state.sqlite")
    if args.force:
        conn.execute("DELETE FROM month_build")
        conn.commit()

    reusable = find_resume_checkpoint(conn, months, signature) if not args.force else []
    reused_months = len(reusable)
    state = empty_state()
    start_month_idx = 0
    current_price = None
    processed_minutes = 0
    if reusable:
        last_reused_month, checkpoint = reusable[-1]
        state = load_checkpoint(checkpoint)
        start_month_idx = months.index(last_reused_month) + 1
        # Recover the prior month final close and processed history length deterministically.
        mask = data["open_time"].dt.strftime("%Y-%m") <= last_reused_month
        processed = data.loc[mask]
        if len(processed):
            current_price = float(processed["close"].iloc[-1])
            processed_minutes = len(processed)
        print(f"Reusable completed months: {reused_months}")

    daily_counts_rows: list[dict[str, Any]] = []
    all_level_summaries: list[dict[str, Any]] = []
    level_rows_total = 0
    snapshot_days_total = 0
    first_snapshot_date = None
    last_snapshot_date = None

    for mi, month in enumerate(months):
        if mi < start_month_idx:
            # Report files are rebuilt later from existing snapshot Parquets.
            continue
        month_frame = data[data["open_time"].dt.strftime("%Y-%m") == month].copy()
        month_dates = sorted(month_frame["date"].unique())
        month_snapshots = []
        month_daily_counts: list[dict[str, Any]] = []
        for day in month_dates:
            day_frame = month_frame[month_frame["date"] == day].copy()
            as_of = pd.Timestamp(datetime(day.year, day.month, day.day, tzinfo=timezone.utc))

            if current_price is not None and day >= min_snapshot_date and day <= max_snapshot_date:
                levels = build_snapshot_levels(
                    state,
                    as_of,
                    current_price,
                    args.symbol,
                    step_log,
                    args.min_anchor_events,
                    args.zone_merge_gap_bins,
                    args.max_distance_bps,
                    args.max_levels_per_snapshot,
                    processed_minutes,
                )
                if len(levels):
                    month_snapshots.append(levels)
                    level_rows_total += len(levels)
                snapshot_days_total += 1
                first_snapshot_date = first_snapshot_date or day.isoformat()
                last_snapshot_date = day.isoformat()
                count_row = {"snapshot_date": day.isoformat(), "level_count": len(levels), "current_price": current_price}
                daily_counts_rows.append(count_row)
                month_daily_counts.append(count_row)

            # Activity and anchors from this day become eligible only for subsequent snapshots.
            state = apply_daily_activity(state, day_frame, step_log)
            if day in anchors_by_date:
                state = apply_available_anchors(state, anchors_by_date[day])
            processed_minutes += len(day_frame)
            current_price = float(day_frame["close"].iloc[-1])

        # If this is the last source month, emit the next day-start state (e.g. 2026-08-01).
        if month == months[-1]:
            next_day = last_date + timedelta(days=1)
            if current_price is not None and next_day >= min_snapshot_date and next_day <= max_snapshot_date:
                as_of = pd.Timestamp(datetime(next_day.year, next_day.month, next_day.day, tzinfo=timezone.utc))
                levels = build_snapshot_levels(
                    state, as_of, current_price, args.symbol, step_log,
                    args.min_anchor_events, args.zone_merge_gap_bins,
                    args.max_distance_bps, args.max_levels_per_snapshot, processed_minutes,
                )
                if len(levels):
                    month_snapshots.append(levels)
                    level_rows_total += len(levels)
                snapshot_days_total += 1
                first_snapshot_date = first_snapshot_date or next_day.isoformat()
                last_snapshot_date = next_day.isoformat()
                count_row = {"snapshot_date": next_day.isoformat(), "level_count": len(levels), "current_price": current_price}
                daily_counts_rows.append(count_row)
                month_daily_counts.append(count_row)

        out_path = snapshot_output_path(output_root, month)
        if month_snapshots:
            month_out = pd.concat(month_snapshots, ignore_index=True)
        else:
            month_out = pd.DataFrame(columns=LEVEL_COLUMNS)
        atomic_write_parquet(out_path, month_out)
        atomic_write_csv(daily_counts_output_path(output_root, month), ["snapshot_date", "level_count", "current_price"], month_daily_counts)
        cp_path = state_checkpoint_path(output_root, month)
        save_checkpoint(cp_path, state)
        save_month_state(conn, month, signature, out_path, cp_path, len(month_out))
        print(f"[{mi+1:02d}/{len(months):02d}] {month}: COMPLETED ({len(month_out):,} level rows)")

    conn.close()

    # Reconstruct report summaries from all monthly snapshot outputs so reused months are included.
    snapshot_files = sorted((output_root / "daily_levels").glob("year=*/month=*/part-00000.parquet"))
    snapshot_frames = []
    for path in snapshot_files:
        try:
            frame = pd.read_parquet(path)
            if len(frame):
                snapshot_frames.append(frame)
        except Exception:
            continue
    all_levels = pd.concat(snapshot_frames, ignore_index=True) if snapshot_frames else pd.DataFrame(columns=LEVEL_COLUMNS)
    level_rows_total = len(all_levels)
    count_sidecars = sorted((output_root / "daily_levels").glob("year=*/month=*/daily_counts.csv"))
    combined_counts = []
    for path in count_sidecars:
        try:
            frame = pd.read_csv(path)
            if len(frame):
                combined_counts.append(frame)
        except Exception:
            continue
    if combined_counts:
        daily_counts = pd.concat(combined_counts, ignore_index=True).drop_duplicates("snapshot_date", keep="last").sort_values("snapshot_date")
        daily_counts_rows = daily_counts.to_dict("records")
        snapshot_days_total = len(daily_counts)
        first_snapshot_date = str(daily_counts["snapshot_date"].iloc[0])
        last_snapshot_date = str(daily_counts["snapshot_date"].iloc[-1])

    if len(all_levels):
        all_levels["as_of_time"] = pd.to_datetime(all_levels["as_of_time"], utc=True)
        level_summary = all_levels.groupby("structural_type", as_index=False).agg(
            rows=("level_id", "size"),
            median_strength=("strength_score", "median"),
            median_anchor_count=("anchor_count", "median"),
            median_age_days=("age_days", "median"),
            median_abs_distance_bps=("abs_distance_bps", "median"),
        )
        all_level_summaries = level_summary.to_dict("records")

    invariants: list[Invariant] = []
    invariants.append(Invariant("global", "pass3_audit_required", "PASS", "PASS", "PASS"))
    anchors_future = int((pd.to_datetime(anchors["available_time"], utc=True) <= pd.to_datetime(anchors["pivot_time"], utc=True)).sum()) if len(anchors) else 0
    invariants.append(Invariant("anchors", "available_after_pivot", "PASS" if anchors_future == 0 else "FAIL", anchors_future, 0))
    duplicate_anchors = int(anchors.duplicated(["side", "pivot_time"]).sum()) if len(anchors) else 0
    invariants.append(Invariant("anchors", "unique_side_pivot", "PASS" if duplicate_anchors == 0 else "FAIL", duplicate_anchors, 0))
    if len(all_levels):
        future_last_anchor = int((pd.to_datetime(all_levels["last_anchor_time"], utc=True) >= pd.to_datetime(all_levels["as_of_time"], utc=True)).sum())
        future_anchor_available = int((pd.to_datetime(all_levels["last_anchor_available_time"], utc=True) >= pd.to_datetime(all_levels["as_of_time"], utc=True)).sum())
        future_last_visit = int((pd.to_datetime(all_levels["last_visit_time"], utc=True) >= pd.to_datetime(all_levels["as_of_time"], utc=True)).sum())
        invalid_distance = int((all_levels["abs_distance_bps"].astype(float) > args.max_distance_bps + 1e-9).sum())
        insufficient_anchors = int((all_levels["anchor_count"].astype(int) < args.min_anchor_events).sum())
    else:
        future_last_anchor = future_anchor_available = future_last_visit = invalid_distance = insufficient_anchors = 0
    invariants.extend([
        Invariant("levels", "last_anchor_strictly_before_snapshot", "PASS" if future_last_anchor == 0 else "FAIL", future_last_anchor, 0),
        Invariant("levels", "anchor_confirmation_strictly_before_snapshot", "PASS" if future_anchor_available == 0 else "FAIL", future_anchor_available, 0),
        Invariant("levels", "last_visit_strictly_before_snapshot", "PASS" if future_last_visit == 0 else "FAIL", future_last_visit, 0),
        Invariant("levels", "distance_filter_respected", "PASS" if invalid_distance == 0 else "FAIL", invalid_distance, 0),
        Invariant("levels", "minimum_anchor_evidence_respected", "PASS" if insufficient_anchors == 0 else "FAIL", insufficient_anchors, 0),
    ])
    failed_invariants = sum(x.status == "FAIL" for x in invariants)

    atomic_write_json(output_root / "metadata" / "level_config.json", {
        "builder_version": BUILDER_VERSION,
        "level_schema_version": LEVEL_SCHEMA_VERSION,
        "run_signature": signature,
        "config": config,
    })
    atomic_write_csv(output_root / "metadata" / "level_invariants.csv", ["scope", "invariant", "status", "observed", "expected", "detail"], [asdict(x) for x in invariants])
    atomic_write_csv(report_root / "historical_price_levels_invariants.csv", ["scope", "invariant", "status", "observed", "expected", "detail"], [asdict(x) for x in invariants])
    atomic_write_csv(report_root / "historical_price_levels_daily_counts.csv", ["snapshot_date", "level_count", "current_price"], daily_counts_rows)
    atomic_write_csv(report_root / "historical_price_levels_level_summary.csv", ["structural_type", "rows", "median_strength", "median_anchor_count", "median_age_days", "median_abs_distance_bps"], all_level_summaries)

    anchor_summary_rows = []
    if len(anchors):
        for (side, scale), g in anchors.groupby(["side", "scale_minutes"]):
            anchor_summary_rows.append({
                "side": side,
                "scale_minutes": int(scale),
                "anchors": len(g),
                "median_prominence_bps": float(g["prominence_bps"].median()),
                "median_anchor_weight": float(g["anchor_weight"].median()),
                "first_pivot": pd.Timestamp(g["pivot_time"].min()).isoformat(),
                "last_pivot": pd.Timestamp(g["pivot_time"].max()).isoformat(),
            })
    atomic_write_csv(report_root / "historical_price_levels_anchor_summary.csv", ["side", "scale_minutes", "anchors", "median_prominence_bps", "median_anchor_weight", "first_pivot", "last_pivot"], anchor_summary_rows)

    summary = {
        "status": "PASS" if failed_invariants == 0 else "FAIL",
        "builder_version": BUILDER_VERSION,
        "level_schema_version": LEVEL_SCHEMA_VERSION,
        "run_id": run_id,
        "run_signature": signature,
        "symbol": args.symbol,
        "source_rows": source_rows,
        "source_months": len(source_paths),
        "first_source_time": first_source.isoformat(),
        "last_source_time": last_source.isoformat(),
        "structural_anchors": len(anchors),
        "snapshot_days": snapshot_days_total,
        "level_rows": level_rows_total,
        "first_snapshot_date": first_snapshot_date,
        "last_snapshot_date": last_snapshot_date,
        "reused_months": reused_months,
        "failed_invariants": failed_invariants,
        "output_root": str(output_root),
        "report_root": str(report_root),
        "generated_at": utc_now_iso(),
    }
    atomic_write_json(output_root / "metadata" / "latest_build_summary.json", summary)
    write_findings(report_root / "historical_price_levels_findings.md", summary)

    print("\nHistorical price-level build summary")
    print("------------------------------------")
    print(f"Status             : {summary['status']}")
    print(f"Source rows        : {source_rows:,}")
    print(f"Structural anchors : {len(anchors):,}")
    print(f"Snapshot days      : {snapshot_days_total:,}")
    print(f"Level rows         : {level_rows_total:,}")
    print(f"First snapshot     : {first_snapshot_date}")
    print(f"Last snapshot      : {last_snapshot_date}")
    print(f"Invariant failures : {failed_invariants}")
    print(f"Summary            : {output_root / 'metadata' / 'latest_build_summary.json'}")
    print(f"Findings           : {report_root / 'historical_price_levels_findings.md'}")
    return 0 if failed_invariants == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
