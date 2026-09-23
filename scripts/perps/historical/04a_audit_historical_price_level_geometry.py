#!/usr/bin/env python3
"""Audit geometry/selectivity of causal historical BTCUSDT price-level states.

Pass 4A is a diagnostic layer between the causal level builder and outcome validation.
It does NOT rebuild levels and does NOT test bounce/rejection profitability. It asks whether
Pass 4 emitted appropriately selective price zones or whether adjacent qualifying bins have
chained into overly broad components.

Default input:
    D:\\stream-data\\historical\\binance\\futures_um\\BTCUSDT\\historical_price_levels_v1

Expected Pass-4 inputs:
    daily_levels/year=YYYY/month=MM/part-00000.parquet
    daily_levels/year=YYYY/month=MM/daily_counts.csv
    metadata/level_config.json
    metadata/latest_build_summary.json
    metadata/level_invariants.csv

Reports:
    reports/binance_um_historical_price_level_geometry/

The audit has two kinds of checks:
  * integrity invariants: objective consistency checks. Any FAIL makes the audit fail.
  * geometry warnings: transparent heuristic diagnostics only. They do not invalidate data;
    they indicate that Pass 4 should be reviewed/retuned before outcome research.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

AUDITOR_VERSION = "1.0.0"
DEFAULT_LEVEL_ROOT = Path(r"D:\stream-data\historical\binance\futures_um\BTCUSDT\historical_price_levels_v1")
REQUIRED_LEVEL_COLUMNS = {
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
    "occupied_minutes",
    "touch_days",
    "quote_volume",
    "trade_count",
    "first_anchor_time",
    "last_anchor_time",
    "last_anchor_available_time",
    "last_visit_time",
    "age_days",
    "days_since_last_anchor",
    "days_since_last_visit",
}


@dataclass
class Check:
    scope: str
    check: str
    status: str
    observed: object
    expected: object
    detail: str = ""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def find_repo_root(path: Path) -> Path:
    current = path.resolve().parent
    for candidate in [current, *current.parents]:
        if (candidate / "scripts").exists() and (candidate / "reports").exists():
            return candidate
    # In artifact/self-test environments, fall back to the script parent.
    return path.resolve().parent


def require_libraries() -> None:
    missing = []
    for name in ("numpy", "pandas", "pyarrow"):
        try:
            __import__(name)
        except Exception:
            missing.append(name)
    if missing:
        raise RuntimeError("Missing required Python packages: " + ", ".join(missing))


def short_temp(path: Path, suffix: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.parent / f".{suffix}-{uuid.uuid4().hex[:8]}.tmp"


def atomic_write_text(path: Path, text: str) -> None:
    tmp = short_temp(path, "txt")
    try:
        tmp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: object) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def atomic_write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        fieldnames = seen
    tmp = short_temp(path, "csv")
    try:
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def signature_for_inputs(level_root: Path, files: list[Path], config: dict) -> str:
    h = hashlib.sha256()
    h.update(AUDITOR_VERSION.encode())
    h.update(json.dumps(config, sort_keys=True, default=str).encode())
    for path in sorted(files):
        rel = str(path.relative_to(level_root)).replace("\\", "/")
        h.update(rel.encode())
        h.update(str(path.stat().st_size).encode())
        h.update(sha256_file(path).encode())
    return h.hexdigest()


def init_state_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS geometry_audit (
            audit_key TEXT PRIMARY KEY,
            input_signature TEXT NOT NULL,
            status TEXT NOT NULL,
            run_id TEXT NOT NULL,
            completed_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def quantile_rows(frame, columns: list[str]) -> list[dict]:
    import numpy as np
    rows: list[dict] = []
    quantiles = [("p50", 0.50), ("p75", 0.75), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99)]
    for col in columns:
        values = frame[col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            rows.append({"metric": col, "rows": 0})
            continue
        row = {
            "metric": col,
            "rows": int(len(values)),
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
        for label, q in quantiles:
            row[label] = float(values.quantile(q))
        rows.append(row)
    return rows


def parse_level_id(level_id: str) -> tuple[int, int] | None:
    value = str(level_id)
    if not value.startswith("L") or "_" not in value:
        return None
    try:
        low, high = value[1:].split("_", 1)
        return int(low), int(high)
    except Exception:
        return None


def log_interval_iou(a_low: float, a_high: float, b_low: float, b_high: float) -> float:
    if min(a_low, a_high, b_low, b_high) <= 0:
        return 0.0
    a0, a1, b0, b1 = map(math.log, (a_low, a_high, b_low, b_high))
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    if union <= 0:
        return 0.0
    return inter / union


def bps_center_distance(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return math.inf
    return abs(math.log(a / b) * 10000.0)


def load_daily_counts(files: list[Path]):
    import pandas as pd
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        if not {"snapshot_date", "level_count", "current_price"}.issubset(frame.columns):
            raise RuntimeError(f"Unexpected daily-count schema: {path}")
        frames.append(frame[["snapshot_date", "level_count", "current_price"]])
    if not frames:
        raise RuntimeError("No daily_counts.csv files were found")
    out = pd.concat(frames, ignore_index=True)
    out["snapshot_date"] = out["snapshot_date"].astype(str)
    out = out.drop_duplicates("snapshot_date", keep="last").sort_values("snapshot_date").reset_index(drop=True)
    return out


def load_levels(files: list[Path]):
    import pandas as pd
    frames = []
    for path in files:
        frame = pd.read_parquet(path)
        missing = sorted(REQUIRED_LEVEL_COLUMNS.difference(frame.columns))
        if missing:
            raise RuntimeError(f"Missing required level columns in {path}: {missing}")
        frames.append(frame)
    if not frames:
        raise RuntimeError("No Pass-4 daily level Parquet files were found")
    out = pd.concat(frames, ignore_index=True)
    out["snapshot_date"] = out["snapshot_date"].astype(str)
    out["as_of_time"] = pd.to_datetime(out["as_of_time"], utc=True)
    for col in ("first_anchor_time", "last_anchor_time", "last_anchor_available_time", "last_visit_time"):
        out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    return out


def integrity_checks(levels, daily_counts, level_config: dict, build_summary: dict, pass4_invariants) -> list[Check]:
    import numpy as np
    import pandas as pd

    checks: list[Check] = []
    failed_pass4 = int((pass4_invariants["status"].astype(str) == "FAIL").sum()) if len(pass4_invariants) else 0
    checks.append(Check("upstream", "pass4_invariants", "PASS" if failed_pass4 == 0 else "FAIL", failed_pass4, 0))
    checks.append(Check("upstream", "pass4_build_status", "PASS" if build_summary.get("status") == "PASS" else "FAIL", build_summary.get("status"), "PASS"))

    dupes = int(levels.duplicated(["snapshot_date", "level_id"]).sum())
    checks.append(Check("levels", "unique_snapshot_level_id", "PASS" if dupes == 0 else "FAIL", dupes, 0))

    invalid_bounds = int(((levels["zone_lower"].astype(float) <= 0) | (levels["zone_upper"].astype(float) <= levels["zone_lower"].astype(float))).sum())
    checks.append(Check("levels", "valid_zone_bounds", "PASS" if invalid_bounds == 0 else "FAIL", invalid_bounds, 0))

    invalid_bins = int((levels["bin_count"].astype(int) < 1).sum())
    checks.append(Check("levels", "positive_bin_count", "PASS" if invalid_bins == 0 else "FAIL", invalid_bins, 0))

    id_mismatch = 0
    for row in levels[["level_id", "bin_count"]].itertuples(index=False):
        parsed = parse_level_id(row.level_id)
        if parsed is None or parsed[1] - parsed[0] + 1 != int(row.bin_count):
            id_mismatch += 1
    checks.append(Check("levels", "level_id_matches_bin_count", "PASS" if id_mismatch == 0 else "FAIL", id_mismatch, 0))

    center_expected = np.sqrt(levels["zone_lower"].astype(float).to_numpy() * levels["zone_upper"].astype(float).to_numpy())
    center_error = np.abs(center_expected - levels["zone_center"].astype(float).to_numpy()) / np.maximum(center_expected, 1e-12)
    center_bad = int((center_error > 1e-10).sum())
    checks.append(Check("levels", "geometric_zone_center", "PASS" if center_bad == 0 else "FAIL", center_bad, 0))

    config = level_config.get("config", level_config)
    bin_width = float(config.get("bin_width_bps", math.nan))
    if math.isfinite(bin_width) and bin_width > 0:
        per_bin_log_bps = math.log1p(bin_width / 10000.0) * 10000.0
        expected_width = levels["bin_count"].astype(float).to_numpy() * per_bin_log_bps
        width_error = np.abs(levels["zone_width_bps"].astype(float).to_numpy() - expected_width)
        width_bad = int((width_error > 1e-7).sum())
        checks.append(Check("levels", "zone_width_matches_bin_count", "PASS" if width_bad == 0 else "FAIL", width_bad, 0, f"configured_bin_width_bps={bin_width}"))
    else:
        checks.append(Check("levels", "zone_width_matches_bin_count", "FAIL", "missing_config", "finite bin_width_bps"))

    inconsistent_price = 0
    for _, g in levels.groupby("snapshot_date", sort=False):
        if len(g) and (g["current_price"].astype(float).max() - g["current_price"].astype(float).min()) > 1e-9:
            inconsistent_price += 1
    checks.append(Check("snapshots", "single_current_price_per_day", "PASS" if inconsistent_price == 0 else "FAIL", inconsistent_price, 0))

    # Zones produced by one snapshot should be disjoint after component merging.
    overlaps = 0
    for _, g in levels.groupby("snapshot_date", sort=False):
        g = g.sort_values("zone_lower")
        prev_high = None
        for row in g.itertuples(index=False):
            low = float(row.zone_lower)
            high = float(row.zone_upper)
            if prev_high is not None and low < prev_high - 1e-12:
                overlaps += 1
            prev_high = max(prev_high or -math.inf, high)
    checks.append(Check("snapshots", "non_overlapping_zones", "PASS" if overlaps == 0 else "FAIL", overlaps, 0))

    actual_counts = levels.groupby("snapshot_date").size().rename("actual")
    reconciled = daily_counts.set_index("snapshot_date").join(actual_counts, how="left").fillna({"actual": 0})
    count_mismatch = int((reconciled["level_count"].astype(int) != reconciled["actual"].astype(int)).sum())
    checks.append(Check("snapshots", "daily_counts_reconcile", "PASS" if count_mismatch == 0 else "FAIL", count_mismatch, 0))

    reported_rows = int(build_summary.get("level_rows", -1))
    checks.append(Check("global", "level_rows_reconcile_build_summary", "PASS" if reported_rows == len(levels) else "FAIL", len(levels), reported_rows))
    reported_days = int(build_summary.get("snapshot_days", -1))
    checks.append(Check("global", "snapshot_days_reconcile_build_summary", "PASS" if reported_days == len(daily_counts) else "FAIL", len(daily_counts), reported_days))

    future_anchor = int((levels["last_anchor_available_time"].notna() & (levels["last_anchor_available_time"] >= levels["as_of_time"])).sum())
    checks.append(Check("causality", "anchor_available_before_snapshot", "PASS" if future_anchor == 0 else "FAIL", future_anchor, 0))
    future_visit = int((levels["last_visit_time"].notna() & (levels["last_visit_time"] >= levels["as_of_time"])).sum())
    checks.append(Check("causality", "last_visit_before_snapshot", "PASS" if future_visit == 0 else "FAIL", future_visit, 0))

    finite_cols = ["zone_width_bps", "abs_distance_bps", "strength_score", "anchor_count", "bin_count"]
    nonfinite = 0
    for col in finite_cols:
        values = levels[col].astype(float).to_numpy()
        nonfinite += int((~np.isfinite(values)).sum())
    checks.append(Check("levels", "finite_core_geometry_fields", "PASS" if nonfinite == 0 else "FAIL", nonfinite, 0))

    return checks


def geometry_by_type(levels) -> list[dict]:
    rows = []
    total = max(len(levels), 1)
    temp = levels.copy()
    temp["anchors_per_bin"] = temp["anchor_count"].astype(float) / temp["bin_count"].astype(float)
    for key, g in temp.groupby("structural_type", sort=True):
        rows.append({
            "structural_type": key,
            "rows": int(len(g)),
            "row_share": float(len(g) / total),
            "median_zone_width_bps": float(g["zone_width_bps"].median()),
            "p95_zone_width_bps": float(g["zone_width_bps"].quantile(0.95)),
            "max_zone_width_bps": float(g["zone_width_bps"].max()),
            "median_bin_count": float(g["bin_count"].median()),
            "p95_bin_count": float(g["bin_count"].quantile(0.95)),
            "median_anchor_count": float(g["anchor_count"].median()),
            "p95_anchor_count": float(g["anchor_count"].quantile(0.95)),
            "median_anchors_per_bin": float(g["anchors_per_bin"].median()),
            "median_strength_score": float(g["strength_score"].median()),
            "median_abs_distance_bps": float(g["abs_distance_bps"].median()),
            "median_age_days": float(g["age_days"].median()),
        })
    return rows


def geometry_by_year(levels, daily_counts) -> list[dict]:
    import pandas as pd
    temp = levels.copy()
    temp["year"] = temp["snapshot_date"].str[:4].astype(int)
    counts = daily_counts.copy()
    counts["year"] = counts["snapshot_date"].str[:4].astype(int)
    rows = []
    years = sorted(counts["year"].unique())
    for year in years:
        g = temp[temp["year"] == year]
        c = counts[counts["year"] == year]
        rows.append({
            "year": int(year),
            "snapshot_days": int(len(c)),
            "level_rows": int(len(g)),
            "mean_levels_per_day": float(c["level_count"].mean()),
            "median_levels_per_day": float(c["level_count"].median()),
            "p90_levels_per_day": float(c["level_count"].quantile(0.90)),
            "max_levels_per_day": int(c["level_count"].max()),
            "zero_level_days": int((c["level_count"] == 0).sum()),
            "zero_level_day_share": float((c["level_count"] == 0).mean()),
            "median_zone_width_bps": float(g["zone_width_bps"].median()) if len(g) else math.nan,
            "p95_zone_width_bps": float(g["zone_width_bps"].quantile(0.95)) if len(g) else math.nan,
            "median_bin_count": float(g["bin_count"].median()) if len(g) else math.nan,
            "median_anchor_count": float(g["anchor_count"].median()) if len(g) else math.nan,
            "two_sided_share": float((g["structural_type"] == "TWO_SIDED").mean()) if len(g) else math.nan,
            "median_abs_distance_bps": float(g["abs_distance_bps"].median()) if len(g) else math.nan,
        })
    return rows


def proximity_coverage(levels, daily_counts, thresholds: list[float]) -> list[dict]:
    import pandas as pd
    rows = []
    all_days = daily_counts[["snapshot_date"]].copy()
    scopes = {
        "ANY": None,
        "ABOVE_PRICE": "ABOVE_PRICE",
        "BELOW_PRICE": "BELOW_PRICE",
    }
    for threshold in thresholds:
        for scope_name, location in scopes.items():
            g = levels[levels["abs_distance_bps"].astype(float) <= threshold]
            if location is not None:
                g = g[g["relative_location"] == location]
            counts = g.groupby("snapshot_date").size().rename("nearby_count")
            merged = all_days.join(counts, on="snapshot_date").fillna({"nearby_count": 0})
            rows.append({
                "distance_bps": float(threshold),
                "scope": scope_name,
                "snapshot_days": int(len(merged)),
                "days_with_level": int((merged["nearby_count"] > 0).sum()),
                "day_share_with_level": float((merged["nearby_count"] > 0).mean()),
                "mean_level_count": float(merged["nearby_count"].mean()),
                "median_level_count": float(merged["nearby_count"].median()),
                "p90_level_count": float(merged["nearby_count"].quantile(0.90)),
                "max_level_count": int(merged["nearby_count"].max()),
            })
    return rows


def persistence_diagnostics(levels, iou_threshold: float, center_threshold_bps: float):
    import pandas as pd
    dates = sorted(levels["snapshot_date"].unique())
    by_date = {d: levels[levels["snapshot_date"] == d].copy() for d in dates}
    rows = []
    run_presence: dict[str, list[pd.Timestamp]] = {}
    for row in levels[["snapshot_date", "level_id"]].drop_duplicates().itertuples(index=False):
        run_presence.setdefault(str(row.level_id), []).append(pd.Timestamp(row.snapshot_date))

    for prev_date, curr_date in zip(dates[:-1], dates[1:]):
        prev = by_date[prev_date]
        curr = by_date[curr_date]
        exact = len(set(prev["level_id"]) & set(curr["level_id"]))
        prev_overlap = 0
        prev_iou = 0
        prev_center = 0
        split_components = 0
        curr_overlap_counts = [0] * len(curr)
        for prow in prev.itertuples(index=False):
            overlaps = 0
            has_iou = False
            has_center = False
            for j, crow in enumerate(curr.itertuples(index=False)):
                iou = log_interval_iou(float(prow.zone_lower), float(prow.zone_upper), float(crow.zone_lower), float(crow.zone_upper))
                if iou > 0:
                    overlaps += 1
                    curr_overlap_counts[j] += 1
                if iou >= iou_threshold:
                    has_iou = True
                if bps_center_distance(float(prow.zone_center), float(crow.zone_center)) <= center_threshold_bps:
                    has_center = True
            if overlaps > 0:
                prev_overlap += 1
            if overlaps > 1:
                split_components += 1
            if has_iou:
                prev_iou += 1
            if has_center:
                prev_center += 1
        merge_components = sum(x > 1 for x in curr_overlap_counts)
        rows.append({
            "previous_date": prev_date,
            "current_date": curr_date,
            "previous_levels": int(len(prev)),
            "current_levels": int(len(curr)),
            "exact_level_id_matches": int(exact),
            "previous_exact_persistence_share": float(exact / len(prev)) if len(prev) else math.nan,
            "previous_any_overlap_share": float(prev_overlap / len(prev)) if len(prev) else math.nan,
            "previous_iou_persistence_share": float(prev_iou / len(prev)) if len(prev) else math.nan,
            "previous_center_persistence_share": float(prev_center / len(prev)) if len(prev) else math.nan,
            "split_component_count": int(split_components),
            "merge_component_count": int(merge_components),
        })

    run_lengths = []
    for level_id, level_dates in run_presence.items():
        values = sorted(set(level_dates))
        if not values:
            continue
        run = 1
        for a, b in zip(values[:-1], values[1:]):
            if (b - a).days == 1:
                run += 1
            else:
                run_lengths.append({"level_id": level_id, "consecutive_days": run})
                run = 1
        run_lengths.append({"level_id": level_id, "consecutive_days": run})
    return rows, run_lengths


def warning_checks(levels, by_type: list[dict], by_year: list[dict], args) -> list[Check]:
    import numpy as np
    checks: list[Check] = []
    p95_width = float(levels["zone_width_bps"].quantile(0.95))
    max_width = float(levels["zone_width_bps"].max())
    wide_share = float((levels["zone_width_bps"].astype(float) >= args.wide_zone_bps).mean())
    giant_bin_share = float((levels["bin_count"].astype(int) >= args.giant_bin_count).mean())
    giant_anchor_share = float((levels["anchor_count"].astype(int) >= args.giant_anchor_count).mean())

    checks.append(Check("geometry", "p95_zone_width", "WARN" if p95_width > args.warn_p95_zone_width_bps else "PASS", p95_width, f"<= {args.warn_p95_zone_width_bps}", "heuristic threshold"))
    checks.append(Check("geometry", "max_zone_width", "WARN" if max_width > args.warn_max_zone_width_bps else "PASS", max_width, f"<= {args.warn_max_zone_width_bps}", "heuristic threshold"))
    checks.append(Check("geometry", "wide_zone_row_share", "WARN" if wide_share > args.warn_wide_row_share else "PASS", wide_share, f"<= {args.warn_wide_row_share}", f"wide_zone_bps={args.wide_zone_bps:g}; heuristic threshold"))
    checks.append(Check("geometry", "giant_bin_component_share", "WARN" if giant_bin_share > args.warn_giant_component_share else "PASS", giant_bin_share, f"<= {args.warn_giant_component_share}", f"giant_bin_count={args.giant_bin_count}; heuristic threshold"))
    checks.append(Check("geometry", "giant_anchor_component_share", "WARN" if giant_anchor_share > args.warn_giant_component_share else "PASS", giant_anchor_share, f"<= {args.warn_giant_component_share}", f"giant_anchor_count={args.giant_anchor_count}; heuristic threshold"))

    type_map = {row["structural_type"]: row for row in by_type}
    two = type_map.get("TWO_SIDED")
    dominant_medians = [type_map[k]["median_anchor_count"] for k in ("SUPPORT_DOMINANT", "RESISTANCE_DOMINANT") if k in type_map and type_map[k]["median_anchor_count"] > 0]
    if two and dominant_medians:
        ratio = float(two["median_anchor_count"] / max(dominant_medians))
        checks.append(Check("geometry", "two_sided_anchor_median_ratio", "WARN" if ratio > args.warn_two_sided_anchor_ratio else "PASS", ratio, f"<= {args.warn_two_sided_anchor_ratio}", "TWO_SIDED median anchors / strongest dominant median; heuristic threshold"))

    if by_year:
        recent = by_year[-1]
        median_levels = float(recent["median_levels_per_day"])
        zero_share = float(recent["zero_level_day_share"])
        checks.append(Check("selectivity", "latest_year_median_levels_per_day", "WARN" if median_levels < args.warn_latest_year_min_median_levels else "PASS", median_levels, f">= {args.warn_latest_year_min_median_levels}", f"year={recent['year']}; heuristic threshold"))
        checks.append(Check("selectivity", "latest_year_zero_level_day_share", "WARN" if zero_share > args.warn_latest_year_max_zero_share else "PASS", zero_share, f"<= {args.warn_latest_year_max_zero_share}", f"year={recent['year']}; heuristic threshold"))
    return checks


def findings_markdown(summary: dict, quantiles: list[dict], by_type: list[dict], proximity: list[dict], warnings: list[Check]) -> str:
    qmap = {x["metric"]: x for x in quantiles}
    lines = [
        "# Historical price-level geometry audit findings",
        "",
        f"Integrity status: **{summary['integrity_status']}**",
        f"Geometry review: **{summary['geometry_review']}**",
        "",
        "This is a geometry/selectivity diagnostic. Geometry warnings are heuristic prompts, not data-integrity failures and not evidence about future bounce/break performance.",
        "",
        "## Coverage",
        "",
        f"- Level-state rows: {summary['level_rows']:,}",
        f"- Snapshot days: {summary['snapshot_days']:,}",
        f"- First snapshot: {summary['first_snapshot_date']}",
        f"- Last snapshot: {summary['last_snapshot_date']}",
    ]
    width = qmap.get("zone_width_bps", {})
    bins = qmap.get("bin_count", {})
    anchors = qmap.get("anchor_count", {})
    lines += [
        "",
        "## Zone geometry",
        "",
        f"- Median zone width: {width.get('p50', math.nan):.2f} bps",
        f"- P95 zone width: {width.get('p95', math.nan):.2f} bps",
        f"- Maximum zone width: {width.get('max', math.nan):.2f} bps",
        f"- Median merged-bin count: {bins.get('p50', math.nan):.2f}",
        f"- P95 merged-bin count: {bins.get('p95', math.nan):.2f}",
        f"- Median anchor count: {anchors.get('p50', math.nan):.2f}",
        f"- P95 anchor count: {anchors.get('p95', math.nan):.2f}",
        "",
        "## Structural-type diagnostics",
        "",
    ]
    for row in by_type:
        lines.append(
            f"- {row['structural_type']}: {row['row_share']:.1%} of rows; median width {row['median_zone_width_bps']:.1f} bps; median bins {row['median_bin_count']:.1f}; median anchors {row['median_anchor_count']:.1f}."
        )
    any_prox = [x for x in proximity if x["scope"] == "ANY"]
    if any_prox:
        lines += ["", "## Proximity coverage", ""]
        for row in any_prox:
            lines.append(f"- Within ±{row['distance_bps']:.0f} bps: a candidate level exists on {row['day_share_with_level']:.1%} of snapshot days; mean count {row['mean_level_count']:.2f}.")
    warn_only = [x for x in warnings if x.status == "WARN"]
    lines += ["", "## Heuristic warnings", ""]
    if warn_only:
        for check in warn_only:
            lines.append(f"- **{check.check}**: observed `{check.observed}`, expected `{check.expected}`. {check.detail}")
    else:
        lines.append("- No configured geometry warning threshold was triggered.")
    lines += [
        "",
        "## Decision",
        "",
    ]
    if summary["integrity_status"] != "PASS":
        lines.append("Do not use these levels for Pass 5 until objective integrity failures are fixed.")
    elif summary["geometry_review"] == "REVIEW_REQUIRED":
        lines.append("Review the largest-zone, by-year and persistence reports before Pass 5. If warnings are driven by broad chained components rather than intentional historical structure, retune Pass 4 merging/selectivity and rebuild levels first.")
    else:
        lines.append("No major geometry warning was detected under the configured diagnostics. The level database is ready for Pass 5 outcome validation, subject to manual review of the largest-zone report.")
    return "\n".join(lines) + "\n"


def run_self_test() -> int:
    import numpy as np
    import pandas as pd

    # Level-id / bin-count contract.
    assert parse_level_id("L10_12") == (10, 12)
    assert parse_level_id("bad") is None
    assert abs(log_interval_iou(100, 110, 105, 115) - (math.log(110) - math.log(105)) / (math.log(115) - math.log(100))) < 1e-12
    assert bps_center_distance(100.0, 100.0) == 0.0

    dates = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
    rows = []
    for i, dt in enumerate(dates):
        # Persistent two-bin level plus a one-day second level.
        rows.append({
            "symbol": "BTCUSDT", "as_of_time": dt, "snapshot_date": dt.date().isoformat(), "current_price": 100.0,
            "level_id": "L10_11", "zone_lower": 90.0, "zone_upper": 91.0, "zone_center": math.sqrt(90*91),
            "zone_width_bps": math.log(91/90)*10000, "relative_location": "BELOW_PRICE", "signed_distance_bps": -950,
            "abs_distance_bps": 950.0, "structural_type": "TWO_SIDED", "strength_score": 10.0, "bin_count": 2,
            "anchor_count": 20, "support_anchor_count": 10, "resistance_anchor_count": 10, "anchor_weight": 20.0,
            "support_anchor_weight": 10.0, "resistance_anchor_weight": 10.0, "occupied_minutes": 100,
            "touch_days": 5, "quote_volume": 1000.0, "trade_count": 1000.0,
            "first_anchor_time": dt-pd.Timedelta(days=20), "last_anchor_time": dt-pd.Timedelta(days=2),
            "last_anchor_available_time": dt-pd.Timedelta(days=2)+pd.Timedelta(hours=1), "last_visit_time": dt-pd.Timedelta(days=1),
            "age_days": 20.0, "days_since_last_anchor": 2.0, "days_since_last_visit": 1.0,
        })
        if i == 1:
            rows.append({**rows[-1], "level_id": "L20_20", "zone_lower": 110.0, "zone_upper": 111.0,
                         "zone_center": math.sqrt(110*111), "zone_width_bps": math.log(111/110)*10000,
                         "relative_location": "ABOVE_PRICE", "signed_distance_bps": 1000, "abs_distance_bps": 1000.0,
                         "structural_type": "RESISTANCE_DOMINANT", "bin_count": 1, "anchor_count": 2})
    levels = pd.DataFrame(rows)
    daily = pd.DataFrame({"snapshot_date": [d.date().isoformat() for d in dates], "level_count": [1,2,1], "current_price": [100,100,100]})
    persistence, runs = persistence_diagnostics(levels, 0.5, 50.0)
    assert len(persistence) == 2
    assert persistence[0]["exact_level_id_matches"] == 1
    assert max(x["consecutive_days"] for x in runs if x["level_id"] == "L10_11") == 3
    prox = proximity_coverage(levels, daily, [1000])
    any_row = next(x for x in prox if x["scope"] == "ANY")
    assert any_row["days_with_level"] == 3

    # Windows-safe short temporary writer smoke test.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        target = Path(td) / ("a" * 80) / "summary.json"
        atomic_write_json(target, {"ok": True})
        assert json.loads(target.read_text())["ok"] is True

    print("SELF-TEST PASS")
    print("Validated level-id/bin geometry, overlap/persistence logic, proximity coverage, and short-path atomic writers.")
    return 0


def parse_float_list(value: str) -> list[float]:
    try:
        values = [float(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated numeric values") from exc
    if not values or any(x <= 0 for x in values):
        raise argparse.ArgumentTypeError("All values must be positive")
    return sorted(set(values))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level-root", type=Path, default=DEFAULT_LEVEL_ROOT)
    parser.add_argument("--report-root", type=Path, default=None)
    parser.add_argument("--proximity-bps", type=parse_float_list, default=parse_float_list("100,250,500,1000,2000,5000"))
    parser.add_argument("--persistence-iou-threshold", type=float, default=0.50)
    parser.add_argument("--persistence-center-bps", type=float, default=50.0)
    parser.add_argument("--largest-zones", type=int, default=100)

    # Heuristic warning thresholds. These NEVER turn an integrity PASS into FAIL.
    parser.add_argument("--wide-zone-bps", type=float, default=500.0)
    parser.add_argument("--warn-p95-zone-width-bps", type=float, default=500.0)
    parser.add_argument("--warn-max-zone-width-bps", type=float, default=2000.0)
    parser.add_argument("--warn-wide-row-share", type=float, default=0.10)
    parser.add_argument("--giant-bin-count", type=int, default=20)
    parser.add_argument("--giant-anchor-count", type=int, default=100)
    parser.add_argument("--warn-giant-component-share", type=float, default=0.10)
    parser.add_argument("--warn-two-sided-anchor-ratio", type=float, default=10.0)
    parser.add_argument("--warn-latest-year-min-median-levels", type=float, default=1.0)
    parser.add_argument("--warn-latest-year-max-zero-share", type=float, default=0.50)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.self_test:
        return run_self_test()
    if not (0 < args.persistence_iou_threshold <= 1):
        raise RuntimeError("--persistence-iou-threshold must be in (0,1]")
    if args.persistence_center_bps <= 0:
        raise RuntimeError("--persistence-center-bps must be positive")
    if args.largest_zones < 1:
        raise RuntimeError("--largest-zones must be >=1")

    require_libraries()
    import numpy as np
    import pandas as pd

    level_root = args.level_root.resolve()
    script_path = Path(__file__).resolve()
    repo_root = find_repo_root(script_path)
    report_root = (args.report_root.resolve() if args.report_root else repo_root / "reports" / "binance_um_historical_price_level_geometry")
    report_root.mkdir(parents=True, exist_ok=True)

    config_path = level_root / "metadata" / "level_config.json"
    build_summary_path = level_root / "metadata" / "latest_build_summary.json"
    pass4_invariants_path = level_root / "metadata" / "level_invariants.csv"
    for path in (config_path, build_summary_path, pass4_invariants_path):
        if not path.exists():
            raise RuntimeError(f"Required Pass-4 metadata file is missing: {path}")
    level_files = sorted((level_root / "daily_levels").glob("year=*/month=*/part-00000.parquet"))
    count_files = sorted((level_root / "daily_levels").glob("year=*/month=*/daily_counts.csv"))
    if not level_files:
        raise RuntimeError(f"No daily level Parquet files found under {level_root / 'daily_levels'}")
    if not count_files:
        raise RuntimeError(f"No daily count files found under {level_root / 'daily_levels'}")

    level_config = json.loads(config_path.read_text(encoding="utf-8"))
    build_summary = json.loads(build_summary_path.read_text(encoding="utf-8"))
    pass4_invariants = pd.read_csv(pass4_invariants_path)
    audit_config = {
        "auditor_version": AUDITOR_VERSION,
        "proximity_bps": args.proximity_bps,
        "persistence_iou_threshold": args.persistence_iou_threshold,
        "persistence_center_bps": args.persistence_center_bps,
        "largest_zones": args.largest_zones,
        "warning_thresholds": {
            "wide_zone_bps": args.wide_zone_bps,
            "warn_p95_zone_width_bps": args.warn_p95_zone_width_bps,
            "warn_max_zone_width_bps": args.warn_max_zone_width_bps,
            "warn_wide_row_share": args.warn_wide_row_share,
            "giant_bin_count": args.giant_bin_count,
            "giant_anchor_count": args.giant_anchor_count,
            "warn_giant_component_share": args.warn_giant_component_share,
            "warn_two_sided_anchor_ratio": args.warn_two_sided_anchor_ratio,
            "warn_latest_year_min_median_levels": args.warn_latest_year_min_median_levels,
            "warn_latest_year_max_zero_share": args.warn_latest_year_max_zero_share,
        },
    }
    input_files = level_files + count_files + [config_path, build_summary_path, pass4_invariants_path]
    input_signature = signature_for_inputs(level_root, input_files, audit_config)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    state_path = level_root / "metadata" / "geometry_audit_state.sqlite"
    conn = init_state_db(state_path)

    expected_reports = [
        report_root / "historical_price_level_geometry_summary.csv",
        report_root / "historical_price_level_geometry_by_type.csv",
        report_root / "historical_price_level_geometry_by_year.csv",
        report_root / "historical_price_level_proximity_coverage.csv",
        report_root / "historical_price_level_persistence.csv",
        report_root / "historical_price_level_persistence_runs.csv",
        report_root / "historical_price_level_largest_zones.csv",
        report_root / "historical_price_level_geometry_warnings.csv",
        report_root / "historical_price_level_geometry_invariants.csv",
        report_root / "historical_price_level_geometry_summary.json",
        report_root / "historical_price_level_geometry_config.json",
        report_root / "historical_price_level_geometry_findings.md",
    ]
    prior = conn.execute("SELECT input_signature, status, run_id FROM geometry_audit WHERE audit_key='GLOBAL'").fetchone()
    if prior and not args.force and prior[0] == input_signature and prior[1] == "COMPLETED" and all(p.exists() for p in expected_reports):
        print("Historical price-level geometry audit already completed for the current Pass-4 inputs/configuration.")
        print(f"Auditor version : {AUDITOR_VERSION}")
        print(f"Prior run ID    : {prior[2]}")
        print(f"Report root     : {report_root}")
        conn.close()
        return 0

    print("Auditing historical price-level geometry and selectivity...")
    print(f"Auditor version     : {AUDITOR_VERSION}")
    print(f"Run ID              : {run_id}")
    print(f"Pass-4 level root   : {level_root}")
    print(f"Monthly level files : {len(level_files)}")
    print(f"Report root         : {report_root}")
    print("No levels will be rebuilt and no bounce/break outcomes will be evaluated.")

    levels = load_levels(level_files)
    daily_counts = load_daily_counts(count_files)
    levels["anchors_per_bin"] = levels["anchor_count"].astype(float) / levels["bin_count"].astype(float)

    integrity = integrity_checks(levels, daily_counts, level_config, build_summary, pass4_invariants)
    failed_integrity = sum(x.status == "FAIL" for x in integrity)

    quantiles = quantile_rows(levels, [
        "zone_width_bps", "bin_count", "anchor_count", "anchors_per_bin", "strength_score",
        "abs_distance_bps", "age_days", "days_since_last_anchor", "days_since_last_visit",
    ])
    by_type = geometry_by_type(levels)
    by_year = geometry_by_year(levels, daily_counts)
    proximity = proximity_coverage(levels, daily_counts, args.proximity_bps)
    persistence, persistence_runs = persistence_diagnostics(levels, args.persistence_iou_threshold, args.persistence_center_bps)
    warnings = warning_checks(levels, by_type, by_year, args)
    warning_count = sum(x.status == "WARN" for x in warnings)

    largest_cols = [
        "snapshot_date", "current_price", "level_id", "structural_type", "zone_lower", "zone_upper", "zone_center",
        "zone_width_bps", "bin_count", "anchor_count", "support_anchor_count", "resistance_anchor_count",
        "anchors_per_bin", "strength_score", "abs_distance_bps", "age_days", "days_since_last_anchor", "days_since_last_visit",
    ]
    largest = levels.sort_values(["zone_width_bps", "bin_count", "anchor_count"], ascending=[False, False, False]).head(args.largest_zones)[largest_cols]
    largest_rows = largest.to_dict(orient="records")

    # Persistence run summary is kept at one row per exact level-id run.
    persistence_runs = sorted(persistence_runs, key=lambda x: (-x["consecutive_days"], x["level_id"]))

    integrity_status = "PASS" if failed_integrity == 0 else "FAIL"
    geometry_review = "REVIEW_REQUIRED" if warning_count else "NO_MAJOR_GEOMETRY_WARNING"
    summary = {
        "integrity_status": integrity_status,
        "geometry_review": geometry_review,
        "auditor_version": AUDITOR_VERSION,
        "run_id": run_id,
        "input_signature": input_signature,
        "pass4_builder_version": level_config.get("builder_version"),
        "pass4_run_id": build_summary.get("run_id"),
        "level_rows": int(len(levels)),
        "snapshot_days": int(len(daily_counts)),
        "first_snapshot_date": str(daily_counts["snapshot_date"].min()),
        "last_snapshot_date": str(daily_counts["snapshot_date"].max()),
        "structural_types": {str(k): int(v) for k, v in levels["structural_type"].value_counts().sort_index().items()},
        "failed_integrity_checks": int(failed_integrity),
        "geometry_warning_count": int(warning_count),
        "largest_zone_width_bps": float(levels["zone_width_bps"].max()),
        "p95_zone_width_bps": float(levels["zone_width_bps"].quantile(0.95)),
        "median_zone_width_bps": float(levels["zone_width_bps"].median()),
        "max_bin_count": int(levels["bin_count"].max()),
        "p95_bin_count": float(levels["bin_count"].quantile(0.95)),
        "max_anchor_count": int(levels["anchor_count"].max()),
        "p95_anchor_count": float(levels["anchor_count"].quantile(0.95)),
        "generated_at": utc_now_iso(),
        "report_root": str(report_root),
    }

    # Write reports atomically.
    atomic_write_csv(report_root / "historical_price_level_geometry_summary.csv", quantiles)
    atomic_write_csv(report_root / "historical_price_level_geometry_by_type.csv", by_type)
    atomic_write_csv(report_root / "historical_price_level_geometry_by_year.csv", by_year)
    atomic_write_csv(report_root / "historical_price_level_proximity_coverage.csv", proximity)
    atomic_write_csv(report_root / "historical_price_level_persistence.csv", persistence)
    atomic_write_csv(report_root / "historical_price_level_persistence_runs.csv", persistence_runs)
    atomic_write_csv(report_root / "historical_price_level_largest_zones.csv", largest_rows)
    atomic_write_csv(report_root / "historical_price_level_geometry_warnings.csv", [asdict(x) for x in warnings])
    atomic_write_csv(report_root / "historical_price_level_geometry_invariants.csv", [asdict(x) for x in integrity])
    atomic_write_json(report_root / "historical_price_level_geometry_summary.json", summary)
    atomic_write_json(report_root / "historical_price_level_geometry_config.json", audit_config)
    atomic_write_text(report_root / "historical_price_level_geometry_findings.md", findings_markdown(summary, quantiles, by_type, proximity, warnings))
    atomic_write_json(level_root / "metadata" / "latest_geometry_audit_summary.json", summary)

    status = "COMPLETED" if failed_integrity == 0 else "FAILED"
    conn.execute(
        "INSERT INTO geometry_audit(audit_key,input_signature,status,run_id,completed_at) VALUES('GLOBAL',?,?,?,?) "
        "ON CONFLICT(audit_key) DO UPDATE SET input_signature=excluded.input_signature,status=excluded.status,run_id=excluded.run_id,completed_at=excluded.completed_at",
        (input_signature, status, run_id, utc_now_iso()),
    )
    conn.commit()
    conn.close()

    print("\nHistorical price-level geometry audit summary")
    print("---------------------------------------------")
    print(f"Integrity status       : {integrity_status}")
    print(f"Geometry review        : {geometry_review}")
    print(f"Level-state rows       : {len(levels):,}")
    print(f"Snapshot days          : {len(daily_counts):,}")
    print(f"Median zone width      : {summary['median_zone_width_bps']:.2f} bps")
    print(f"P95 zone width         : {summary['p95_zone_width_bps']:.2f} bps")
    print(f"Maximum zone width     : {summary['largest_zone_width_bps']:.2f} bps")
    print(f"P95 merged-bin count   : {summary['p95_bin_count']:.2f}")
    print(f"Maximum merged bins    : {summary['max_bin_count']}")
    print(f"Geometry warnings      : {warning_count}")
    print(f"Failed invariants      : {failed_integrity}")
    print(f"Findings               : {report_root / 'historical_price_level_geometry_findings.md'}")
    print(f"Largest zones          : {report_root / 'historical_price_level_largest_zones.csv'}")

    return 0 if failed_integrity == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
