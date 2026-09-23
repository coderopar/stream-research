#!/usr/bin/env python3
"""Phase 7M.6C — frozen-pilot replacement-execution attribution audit.

Scientific role
---------------
This phase is deliberately cheap and outcome-blind.  It NEVER reads raw market
files and NEVER reads future continuation labels.  It operates only on signed
Phase 7M.6B candidate_metrics + level_fate checkpoints from the four-day pilot.

Primary question
----------------
After an opposing price level was depleted and then visibly refilled, how much
of the newly arrived replacement quantity can we conservatively say was reached
by later aggressive executions?

At each refilled price level:

    replacement_execution_lower_bound_qty
        = min(observed_refill_qty,
              max(0, strict_post_refill_aggressive_trade_qty
                     - pre_refill_residual_qty))

The subtraction gives the entire pre-refill residual queue the benefit of the
doubt.  Only aggressive executed quantity beyond that residual is attributed to
replacement liquidity, and the result is capped by observed replacement amount.
This is a price-level lower bound, NOT individual-order identity.

The primary common-support feasibility analysis matches high vs low conservative
replacement-execution share exactly on candidate_date + direction and controls
for momentum, aggression, initial opposing destruction, exact sequence refill
amount, and refill breadth.  A refill-amount-only control set is reported as a
secondary robustness diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:
    pa = None
    pq = None


BUILDER_VERSION = "1.0.0"
SCHEMA_VERSION = 1
SCHEMA_NAME = "prospective_25bp_refill_redepletion_attribution_audit_v1"
SOURCE_SCHEMA = "prospective_25bp_refill_redepletion_sequence_v1"
PRIMARY_HORIZON_SEC = 3
EXPECTED_HORIZONS = (1, 3, 5, 10)
EXPECTED_PILOT_DATES = 4
EXPECTED_PILOT_CANDIDATES = 446
TOTAL_CANDIDATE_DAYS = 62
DISCOVERY_END_DATE = "2026-06-17"
VALIDATION_START_DATE = "2026-06-18"
FULL_DISCOVERY_DAYS = 31
FULL_VALIDATION_DAYS = 31

# Reuse Phase 7M.6B's already-preregistered feasibility gates.  These are not
# tuned using any future outcome.
MIN_PROJECTED_FULL_PRIMARY_PAIRS = 250
MIN_PROJECTED_SPLIT_PAIRS = 100
MIN_FEATURE_POSITIVE_SHARE = 0.10
MAX_FEATURE_POSITIVE_SHARE = 0.90
BALANCE_SMD_GATE = 0.20

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()
DEFAULT_7M6B_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / SOURCE_SCHEMA
DEFAULT_7M6B_SCRIPT = REPO_ROOT / "scripts" / "perps" / "research" / "07m6b_build_refill_redepletion_sequence_pilot.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / SCHEMA_NAME
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "perps_prospective_25bp_refill_redepletion_attribution_audit"

REQUIRED_LEVEL_COLUMNS = (
    "candidate_id", "candidate_date", "direction", "observation_horizon_sec",
    "price_tick", "price", "first_refill_seen", "pre_refill_residual_qty",
    "observed_refill_qty", "observed_refill_quote",
    "strict_post_refill_aggressive_trade_qty",
    "strict_post_refill_aggressive_trade_quote",
    "same_exchange_ms_post_refill_aggressive_trade_qty",
    "strict_trade_time_complete", "anchor_found", "sequence_valid",
)
REQUIRED_METRIC_COLUMNS = (
    "candidate_id", "candidate_date", "direction", "observation_horizon_sec",
    "candidate_second", "anchor_found", "anchor_offset_ms", "sequence_valid",
    "opposing__refilled_level_count", "opposing__observed_refill_quote",
    "opposing__post_refill_aggressive_trade_to_refill_ratio",
    "opposing__refilled_level_post_trade_share",
    "opposing__strict_trade_time_complete_share",
    "control__directional_mid_move_bps_3s",
    "control__directional_aggressive_quote_3s",
    "control__opposing_peak_initial_queue_loss_share_3s",
    "control__opposing_refill_quote_3s_frozen",
    "is_reaccel_context",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def require_stack() -> None:
    if pa is None or pq is None:
        raise SystemExit("Phase 7M.6C requires pandas, numpy and pyarrow in the project .venv.")


def safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return math.nan
    return v if math.isfinite(v) else math.nan


def bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    return s.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False}).fillna(False).astype(bool)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7m6c-", suffix=path.suffix or ".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        Path(tmp).write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path: Path, obj: Any) -> None:
    atomic_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def atomic_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7m6c-", suffix=".csv", dir=str(path.parent))
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_parquet(path: Path, df: pd.DataFrame) -> None:
    require_stack()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7m6c-", suffix=".parquet", dir=str(path.parent))
    os.close(fd)
    try:
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp, compression="zstd")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_module(path: Path, name: str):
    if not path.exists():
        raise RuntimeError(f"Missing required Phase 7M.6B script: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def parquet_columns(path: Path) -> set[str]:
    return set(pq.ParquetFile(path).schema_arrow.names)


def assert_columns(path: Path, required: Sequence[str], kind: str) -> None:
    names = parquet_columns(path)
    missing = [c for c in required if c not in names]
    if missing:
        raise RuntimeError(f"{kind} checkpoint {path} is missing required columns: {missing}")


def checkpoint_paths(root: Path, day: str) -> dict[str, Path]:
    return {
        "metrics": root / "candidate_metrics" / f"candidate_date={day}" / "part.parquet",
        "levels": root / "level_fate" / f"candidate_date={day}" / "part.parquet",
        "manifest": root / "checkpoints" / f"candidate_date={day}" / "manifest.json",
    }


def verify_source_contract(root: Path) -> tuple[dict[str, Any], list[str], pd.DataFrame]:
    meta_path = root / "metadata" / "latest_summary.json"
    if not meta_path.exists():
        raise RuntimeError(f"Missing Phase 7M.6B summary: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if str(meta.get("schema_name")) != SOURCE_SCHEMA:
        raise RuntimeError(f"Expected 7M.6B schema {SOURCE_SCHEMA!r}, found {meta.get('schema_name')!r}")
    if bool(meta.get("future_outcomes_loaded")):
        raise RuntimeError("Phase 7M.6B summary says future outcomes were loaded; audit refuses to proceed.")
    days = [str(x) for x in meta.get("processed_dates", [])]
    if len(days) != EXPECTED_PILOT_DATES or len(set(days)) != EXPECTED_PILOT_DATES:
        raise RuntimeError(f"Expected exactly {EXPECTED_PILOT_DATES} frozen pilot dates; found {days}")

    manifest_rows = []
    total_manifest_candidates = 0
    for day in days:
        p = checkpoint_paths(root, day)
        for key in ("metrics", "levels", "manifest"):
            if not p[key].exists():
                raise RuntimeError(f"Missing required 7M.6B {key} checkpoint for {day}: {p[key]}")
        m = json.loads(p["manifest"].read_text(encoding="utf-8"))
        if str(m.get("status")) != "PASS":
            raise RuntimeError(f"7M.6B checkpoint {day} status is not PASS")
        if str(m.get("schema_name")) != SOURCE_SCHEMA:
            raise RuntimeError(f"7M.6B checkpoint {day} schema mismatch")
        if bool(m.get("future_outcomes_loaded")):
            raise RuntimeError(f"7M.6B checkpoint {day} is not future-clean")
        contract = m.get("contract", {}) or {}
        if bool(contract.get("future_outcomes_loaded")):
            raise RuntimeError(f"7M.6B checkpoint {day} contract is not future-clean")
        if contract.get("order_identity_claim") is not False:
            raise RuntimeError(f"7M.6B checkpoint {day} does not explicitly disclaim order identity")
        total_manifest_candidates += int(m.get("candidate_count", 0))
        manifest_rows.append({
            "candidate_date": day,
            "builder_version": str(m.get("builder_version")),
            "candidate_count": int(m.get("candidate_count", 0)),
            "status": str(m.get("status")),
            "future_outcomes_loaded": bool(m.get("future_outcomes_loaded")),
            "input_signature": str(m.get("input_signature", "")),
            "strict_post_refill_trade": str(contract.get("strict_post_refill_trade", "")),
            "same_ms_policy": str(contract.get("same_ms_policy", "")),
            "order_identity_claim": contract.get("order_identity_claim"),
        })
    if total_manifest_candidates != EXPECTED_PILOT_CANDIDATES:
        raise RuntimeError(
            f"Expected {EXPECTED_PILOT_CANDIDATES} pilot candidates across manifests; found {total_manifest_candidates}"
        )
    return meta, days, pd.DataFrame(manifest_rows)


def read_pilot_checkpoints(root: Path, days: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_parts = []
    level_parts = []
    for day in days:
        p = checkpoint_paths(root, day)
        assert_columns(p["metrics"], REQUIRED_METRIC_COLUMNS, "candidate_metrics")
        assert_columns(p["levels"], REQUIRED_LEVEL_COLUMNS, "level_fate")
        m = pq.read_table(p["metrics"], columns=list(REQUIRED_METRIC_COLUMNS)).to_pandas()
        l = pq.read_table(p["levels"], columns=list(REQUIRED_LEVEL_COLUMNS)).to_pandas()
        m["candidate_id"] = m["candidate_id"].astype(str)
        l["candidate_id"] = l["candidate_id"].astype(str)
        metrics_parts.append(m)
        level_parts.append(l)
    metrics = pd.concat(metrics_parts, ignore_index=True, sort=False)
    levels = pd.concat(level_parts, ignore_index=True, sort=False)
    return metrics, levels


def derive_level_attribution(levels: pd.DataFrame) -> pd.DataFrame:
    x = levels.copy()
    for c in (
        "price", "pre_refill_residual_qty", "observed_refill_qty", "observed_refill_quote",
        "strict_post_refill_aggressive_trade_qty", "strict_post_refill_aggressive_trade_quote",
        "same_exchange_ms_post_refill_aggressive_trade_qty",
    ):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x["first_refill_seen"] = bool_series(x["first_refill_seen"])
    x["strict_trade_time_complete"] = bool_series(x["strict_trade_time_complete"])
    x["anchor_found"] = bool_series(x["anchor_found"])
    x["sequence_valid"] = bool_series(x["sequence_valid"])

    eligible = (
        x["first_refill_seen"]
        & x["strict_trade_time_complete"]
        & x["anchor_found"]
        & x["sequence_valid"]
        & x["observed_refill_qty"].notna() & (x["observed_refill_qty"] > 0)
        & x["pre_refill_residual_qty"].notna() & (x["pre_refill_residual_qty"] >= 0)
        & x["strict_post_refill_aggressive_trade_qty"].notna() & (x["strict_post_refill_aggressive_trade_qty"] >= 0)
        & x["price"].notna() & (x["price"] > 0)
    )
    x["replacement_execution_lower_bound_eligible"] = eligible
    x["replacement_execution_lower_bound_qty"] = np.nan
    raw_lb = (
        x.loc[eligible, "strict_post_refill_aggressive_trade_qty"]
        - x.loc[eligible, "pre_refill_residual_qty"]
    ).clip(lower=0)
    capped = np.minimum(raw_lb.to_numpy(float), x.loc[eligible, "observed_refill_qty"].to_numpy(float))
    x.loc[eligible, "replacement_execution_lower_bound_qty"] = capped
    x["replacement_execution_lower_bound_quote"] = x["replacement_execution_lower_bound_qty"] * x["price"]
    x["replacement_execution_lower_bound_share_of_refill"] = np.where(
        eligible,
        x["replacement_execution_lower_bound_qty"] / x["observed_refill_qty"],
        np.nan,
    )
    x["replacement_execution_lower_bound_positive"] = np.where(
        eligible,
        x["replacement_execution_lower_bound_qty"] > 1e-12,
        False,
    )
    # How much of the original post-refill trade measure survives the stronger
    # conservative attribution requirement.  Diagnostic only.
    x["raw_post_refill_trade_qty"] = x["strict_post_refill_aggressive_trade_qty"]
    x["raw_trade_excess_over_residual_qty"] = np.nan
    x.loc[eligible, "raw_trade_excess_over_residual_qty"] = raw_lb.to_numpy(float)
    return x


def aggregate_candidate_attribution(metrics: pd.DataFrame, level_attr: pd.DataFrame) -> pd.DataFrame:
    # Reproduce the 7M.6B refill denominator from all valid refilled levels, then
    # compute the conservative execution lower bound only where strict trade-time
    # attribution is complete.  If any refilled level is attribution-incomplete,
    # the candidate lower-bound ratio remains NaN rather than silently treating
    # unknown execution as zero.
    x = level_attr.copy()
    all_refilled = (
        bool_series(x["first_refill_seen"])
        & bool_series(x["anchor_found"])
        & bool_series(x["sequence_valid"])
        & (pd.to_numeric(x["observed_refill_qty"], errors="coerce") > 0)
        & (pd.to_numeric(x["observed_refill_quote"], errors="coerce") > 0)
    )
    r = x[all_refilled].copy()
    rows = []
    if len(r):
        for keys, g in r.groupby(["candidate_id", "candidate_date", "direction", "observation_horizon_sec"], sort=True):
            cid, day, direction, h = keys
            refill_quote = float(pd.to_numeric(g["observed_refill_quote"], errors="coerce").fillna(0).sum())
            raw_trade_quote = float(pd.to_numeric(g["strict_post_refill_aggressive_trade_quote"], errors="coerce").fillna(0).sum())
            eligible_mask = bool_series(g["replacement_execution_lower_bound_eligible"])
            ge = g[eligible_mask].copy()
            n_all = int(len(g))
            n_eligible = int(len(ge))
            fully_attributable = bool(n_all > 0 and n_eligible == n_all)
            lb_quote = float(pd.to_numeric(ge["replacement_execution_lower_bound_quote"], errors="coerce").fillna(0).sum()) if n_eligible else 0.0
            pos = int(bool_series(ge["replacement_execution_lower_bound_positive"]).sum()) if n_eligible else 0
            rows.append({
                "candidate_id": str(cid), "candidate_date": str(day), "direction": int(direction),
                "observation_horizon_sec": int(h),
                "opposing__all_refilled_level_count_from_levels": n_all,
                "opposing__attribution_eligible_refilled_level_count": n_eligible,
                "opposing__attribution_complete_for_all_refilled_levels": fully_attributable,
                "opposing__attribution_observed_refill_quote": refill_quote,
                "opposing__replacement_execution_lower_bound_quote": lb_quote if fully_attributable else math.nan,
                "opposing__replacement_execution_lower_bound_to_refill_ratio": (lb_quote / refill_quote) if fully_attributable and refill_quote > 0 else math.nan,
                "opposing__refilled_level_replacement_execution_lower_bound_share": (pos / n_all) if fully_attributable and n_all > 0 else math.nan,
                "opposing__raw_post_refill_aggressive_trade_quote_from_levels": raw_trade_quote,
                "opposing__raw_post_refill_aggressive_trade_to_refill_ratio_from_levels": raw_trade_quote / refill_quote if refill_quote > 0 else math.nan,
            })
    a = pd.DataFrame(rows)

    out = metrics.copy()
    out["candidate_id"] = out["candidate_id"].astype(str)
    if len(a):
        out = out.merge(
            a,
            on=["candidate_id", "candidate_date", "direction", "observation_horizon_sec"],
            how="left", validate="one_to_one",
        )
    else:
        for c in (
            "opposing__all_refilled_level_count_from_levels",
            "opposing__attribution_eligible_refilled_level_count",
            "opposing__attribution_complete_for_all_refilled_levels",
            "opposing__attribution_observed_refill_quote",
            "opposing__replacement_execution_lower_bound_quote",
            "opposing__replacement_execution_lower_bound_to_refill_ratio",
            "opposing__refilled_level_replacement_execution_lower_bound_share",
            "opposing__raw_post_refill_aggressive_trade_quote_from_levels",
            "opposing__raw_post_refill_aggressive_trade_to_refill_ratio_from_levels",
        ):
            out[c] = np.nan
    return out


def candidate_level_refill_parity(candidate_attr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = candidate_attr.copy()
    a = pd.to_numeric(x["opposing__attribution_observed_refill_quote"], errors="coerce")
    b = pd.to_numeric(x["opposing__observed_refill_quote"], errors="coerce")
    level_n = pd.to_numeric(x["opposing__all_refilled_level_count_from_levels"], errors="coerce")
    metric_n = pd.to_numeric(x["opposing__refilled_level_count"], errors="coerce")
    mask = a.notna() & b.notna()
    diff = (a - b).abs()
    tol = np.maximum(1e-6, 1e-10 * np.maximum(a.abs().fillna(0), b.abs().fillna(0)))
    bad = mask & (diff > tol)
    positive_metric_missing = (b > 0) & a.isna()
    count_mask = level_n.notna() & metric_n.notna()
    count_bad = count_mask & (level_n.astype(float) != metric_n.astype(float))
    detail = x.loc[mask, ["candidate_id", "candidate_date", "direction", "observation_horizon_sec"]].copy()
    detail["level_sum_refill_quote"] = a[mask].to_numpy(float)
    detail["candidate_metric_refill_quote"] = b[mask].to_numpy(float)
    detail["abs_difference"] = diff[mask].to_numpy(float)
    summary = pd.DataFrame([{
        "rows_compared": int(mask.sum()),
        "rows_outside_float_tolerance": int(bad.sum()),
        "positive_candidate_refill_missing_level_sum": int(positive_metric_missing.sum()),
        "refilled_level_count_mismatches": int(count_bad.sum()),
        "max_abs_difference": float(diff[mask].max()) if mask.any() else math.nan,
    }])
    return summary, detail


def qsafe(s: pd.Series, q: float) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    return float(x.quantile(q)) if len(x) else math.nan


def build_7m3_parity_audit(primary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = primary.copy()
    x["sequence_refill_quote"] = pd.to_numeric(x["opposing__observed_refill_quote"], errors="coerce")
    x["frozen_7m3_refill_quote"] = pd.to_numeric(x["control__opposing_refill_quote_3s_frozen"], errors="coerce")
    mask = x["sequence_refill_quote"].notna() & x["frozen_7m3_refill_quote"].notna()
    z = x.loc[mask, [
        "candidate_id", "candidate_date", "direction", "anchor_offset_ms",
        "sequence_refill_quote", "frozen_7m3_refill_quote",
    ]].copy()
    z["abs_difference"] = (z["sequence_refill_quote"] - z["frozen_7m3_refill_quote"]).abs()
    denom = z["frozen_7m3_refill_quote"].abs()
    z["abs_relative_error_to_frozen"] = np.where(denom > 0, z["abs_difference"] / denom, np.nan)
    sym_denom = z["sequence_refill_quote"].abs() + z["frozen_7m3_refill_quote"].abs()
    z["symmetric_abs_relative_difference"] = np.where(sym_denom > 0, 2.0 * z["abs_difference"] / sym_denom, 0.0)
    both_pos = (z["sequence_refill_quote"] > 0) & (z["frozen_7m3_refill_quote"] > 0)
    z["sequence_to_frozen_ratio"] = np.where(both_pos, z["sequence_refill_quote"] / z["frozen_7m3_refill_quote"], np.nan)
    z["abs_anchor_offset_ms"] = pd.to_numeric(z["anchor_offset_ms"], errors="coerce").abs()

    pearson = float(z["sequence_refill_quote"].corr(z["frozen_7m3_refill_quote"], method="pearson")) if len(z) >= 3 else math.nan
    spearman = float(z["sequence_refill_quote"].corr(z["frozen_7m3_refill_quote"], method="spearman")) if len(z) >= 3 else math.nan
    summary = pd.DataFrame([{
        "pilot_rows": int(len(primary)),
        "rows_both_finite": int(len(z)),
        "rows_both_positive": int(both_pos.sum()),
        "pearson_corr": pearson,
        "spearman_corr": spearman,
        "median_sequence_refill_quote": qsafe(z["sequence_refill_quote"], 0.5),
        "median_frozen_7m3_refill_quote": qsafe(z["frozen_7m3_refill_quote"], 0.5),
        "median_abs_refill_quote_difference": qsafe(z["abs_difference"], 0.5),
        "median_abs_relative_error_to_frozen": qsafe(z["abs_relative_error_to_frozen"], 0.5),
        "median_symmetric_abs_relative_difference": qsafe(z["symmetric_abs_relative_difference"], 0.5),
        "median_sequence_to_frozen_ratio": qsafe(z.loc[both_pos, "sequence_to_frozen_ratio"], 0.5),
        "p25_sequence_to_frozen_ratio": qsafe(z.loc[both_pos, "sequence_to_frozen_ratio"], 0.25),
        "p75_sequence_to_frozen_ratio": qsafe(z.loc[both_pos, "sequence_to_frozen_ratio"], 0.75),
        "median_abs_anchor_offset_ms": qsafe(z["abs_anchor_offset_ms"], 0.5),
        "p95_abs_anchor_offset_ms": qsafe(z["abs_anchor_offset_ms"], 0.95),
        "max_abs_anchor_offset_ms": float(pd.to_numeric(z["abs_anchor_offset_ms"], errors="coerce").max()) if len(z) else math.nan,
    }])
    return summary, z.sort_values("symmetric_abs_relative_difference", ascending=False, kind="mergesort").reset_index(drop=True)


def build_information_loss_audit(primary: pd.DataFrame) -> pd.DataFrame:
    raw = pd.to_numeric(primary["opposing__post_refill_aggressive_trade_to_refill_ratio"], errors="coerce")
    lb = pd.to_numeric(primary["opposing__replacement_execution_lower_bound_to_refill_ratio"], errors="coerce")
    mask = raw.notna() & lb.notna()
    zraw = raw[mask]
    zlb = lb[mask]
    return pd.DataFrame([{
        "rows_compared": int(mask.sum()),
        "raw_positive_share": float((zraw > 0).mean()) if len(zraw) else math.nan,
        "lower_bound_positive_share": float((zlb > 0).mean()) if len(zlb) else math.nan,
        "pearson_corr_raw_vs_lower_bound": float(zraw.corr(zlb, method="pearson")) if len(zraw) >= 3 else math.nan,
        "spearman_corr_raw_vs_lower_bound": float(zraw.corr(zlb, method="spearman")) if len(zraw) >= 3 else math.nan,
        "median_raw_ratio": qsafe(zraw, 0.5),
        "median_lower_bound_ratio": qsafe(zlb, 0.5),
        "median_lower_bound_minus_raw": qsafe(zlb - zraw, 0.5),
    }])


def build_matching_feasibility(primary: pd.DataFrame, phase7m6b: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = primary.copy()
    d = d[
        bool_series(d["anchor_found"])
        & bool_series(d["sequence_valid"])
        & (pd.to_numeric(d["opposing__observed_refill_quote"], errors="coerce") > 0)
        & (pd.to_numeric(d["opposing__strict_trade_time_complete_share"], errors="coerce") >= 1.0 - 1e-12)
    ].copy()

    d["__momentum"] = pd.to_numeric(d["control__directional_mid_move_bps_3s"], errors="coerce")
    d["__aggression"] = np.log1p(pd.to_numeric(d["control__directional_aggressive_quote_3s"], errors="coerce").clip(lower=0))
    d["__destruction"] = pd.to_numeric(d["control__opposing_peak_initial_queue_loss_share_3s"], errors="coerce")
    d["__sequence_refill"] = np.log1p(pd.to_numeric(d["opposing__observed_refill_quote"], errors="coerce").clip(lower=0))
    d["__refill_breadth"] = np.log1p(pd.to_numeric(d["opposing__refilled_level_count"], errors="coerce").clip(lower=0))

    control_sets = {
        "REFILL_AMOUNT_BREADTH_PRIMARY": ["__momentum", "__aggression", "__destruction", "__sequence_refill", "__refill_breadth"],
        "REFILL_AMOUNT_ONLY_ROBUSTNESS": ["__momentum", "__aggression", "__destruction", "__sequence_refill"],
    }
    features = [
        ("PRIMARY", "opposing__replacement_execution_lower_bound_to_refill_ratio"),
        ("SECONDARY", "opposing__refilled_level_replacement_execution_lower_bound_share"),
        ("REFERENCE_ONLY", "opposing__post_refill_aggressive_trade_to_refill_ratio"),
    ]

    pair_parts = []
    rows = []
    pilot_days = max(1, int(d["candidate_date"].nunique()))
    scale = TOTAL_CANDIDATE_DAYS / pilot_days
    disc_pilot_days = max(1, int(d[d["candidate_date"].astype(str) <= DISCOVERY_END_DATE]["candidate_date"].nunique()))
    val_pilot_days = max(1, int(d[d["candidate_date"].astype(str) >= VALIDATION_START_DATE]["candidate_date"].nunique()))

    for control_name, controls in control_sets.items():
        for role, feature in features:
            z = d.copy()
            for c in [feature, *controls]:
                z = z[pd.to_numeric(z[c], errors="coerce").notna()].copy()
            pairs = phase7m6b.greedy_pairs(z, feature, controls)
            repaired = phase7m6b.repair_pairs(pairs, controls) if len(pairs) else pairs
            mx, vals = phase7m6b.pair_balance(repaired, controls) if len(repaired) else (math.nan, {})
            projected = int(round(len(repaired) * scale))
            disc = int((repaired["candidate_date"].astype(str) <= DISCOVERY_END_DATE).sum()) if len(repaired) else 0
            val = int((repaired["candidate_date"].astype(str) >= VALIDATION_START_DATE).sum()) if len(repaired) else 0
            proj_disc = int(round(disc * FULL_DISCOVERY_DAYS / disc_pilot_days))
            proj_val = int(round(val * FULL_VALIDATION_DAYS / val_pilot_days))
            pos_share = float((pd.to_numeric(z[feature], errors="coerce") > 0).mean()) if len(z) else math.nan
            feasible = bool(
                len(repaired) > 0
                and math.isfinite(mx) and mx <= BALANCE_SMD_GATE
                and projected >= MIN_PROJECTED_FULL_PRIMARY_PAIRS
                and proj_disc >= MIN_PROJECTED_SPLIT_PAIRS
                and proj_val >= MIN_PROJECTED_SPLIT_PAIRS
                and math.isfinite(pos_share)
                and MIN_FEATURE_POSITIVE_SHARE <= pos_share <= MAX_FEATURE_POSITIVE_SHARE
            )
            row = {
                "feature_role": role,
                "feature": feature,
                "control_set": control_name,
                "pilot_eligible_candidates": int(len(z)),
                "pilot_days": pilot_days,
                "pilot_pairs_before_repair": int(len(pairs)),
                "pilot_pairs_after_repair": int(len(repaired)),
                "pilot_max_abs_smd": mx,
                "projected_full_pairs": projected,
                "projected_discovery_pairs": proj_disc,
                "projected_validation_pairs": proj_val,
                "positive_feature_share": pos_share,
                "balance_gate": BALANCE_SMD_GATE,
                "projected_full_pair_gate": MIN_PROJECTED_FULL_PRIMARY_PAIRS,
                "projected_split_pair_gate": MIN_PROJECTED_SPLIT_PAIRS,
                "positive_share_min_gate": MIN_FEATURE_POSITIVE_SHARE,
                "positive_share_max_gate": MAX_FEATURE_POSITIVE_SHARE,
                "feasible_before_outcomes": feasible,
            }
            for c, v in vals.items():
                row[f"smd__{c}"] = v
            rows.append(row)
            if len(repaired):
                q = repaired.copy()
                q["feature_role"] = role
                q["feature"] = feature
                q["control_set"] = control_name
                pair_parts.append(q)
    return pd.DataFrame(rows), pd.concat(pair_parts, ignore_index=True, sort=False) if pair_parts else pd.DataFrame()


def write_partitioned(df: pd.DataFrame, root: Path) -> None:
    if df.empty:
        return
    for day, g in df.groupby("candidate_date", sort=True):
        atomic_parquet(root / f"candidate_date={day}" / "part.parquet", g.reset_index(drop=True))


def findings_text(summary: dict[str, Any], gate: pd.DataFrame, parity: pd.DataFrame, info: pd.DataFrame) -> str:
    primary = gate[
        (gate["feature_role"] == "PRIMARY")
        & (gate["control_set"] == "REFILL_AMOUNT_BREADTH_PRIMARY")
    ]
    p = primary.iloc[0].to_dict() if len(primary) else {}
    pr = parity.iloc[0].to_dict() if len(parity) else {}
    ir = info.iloc[0].to_dict() if len(info) else {}
    lines = [
        "# Phase 7M.6C — Conservative replacement-execution attribution audit",
        "",
        f"- Status: **{summary['status']}**",
        f"- Pilot dates: **{', '.join(summary['processed_dates'])}**",
        f"- Primary 3s refill-eligible candidates: **{summary['primary_refill_eligible']}**",
        "- Future outcomes loaded: **NO**",
        "- Raw market replay performed: **NO**",
        "",
        "## Primary attribution",
        "",
        "At each refilled opposing price, the audit gives the entire pre-refill residual queue the benefit of the doubt. Only later aggressive executed quantity beyond that residual is attributed to replacement liquidity, capped by observed refill amount.",
        "",
        "`replacement_execution_lower_bound = min(refill, max(0, strict_post_refill_aggressive_trade - pre_refill_residual))`",
        "",
        "This is a conservative price-level lower bound; it does not identify individual orders or participants.",
        "",
        "## Primary common-support gate",
        "",
        "Controls: exact candidate_date + direction, 3s momentum, aggressive flow, initial opposing destruction, exact sequence refill amount, and refill breadth.",
    ]
    if p:
        lines += [
            f"- Pilot eligible: **{int(p.get('pilot_eligible_candidates', 0))}**",
            f"- Matched pairs after repair: **{int(p.get('pilot_pairs_after_repair', 0))}**",
            f"- Max |SMD|: **{safe_float(p.get('pilot_max_abs_smd')):.3f}**",
            f"- Projected full pairs: **{int(p.get('projected_full_pairs', 0))}**",
            f"- Projected discovery / validation pairs: **{int(p.get('projected_discovery_pairs', 0))} / {int(p.get('projected_validation_pairs', 0))}**",
            f"- Positive feature share: **{safe_float(p.get('positive_feature_share')):.2%}**",
            f"- Feasible before outcomes: **{bool(p.get('feasible_before_outcomes'))}**",
        ]
    lines += ["", "## 7M.6B vs frozen 7M.3 refill parity", ""]
    if pr:
        lines += [
            f"- Pearson / Spearman correlation: **{safe_float(pr.get('pearson_corr')):.3f} / {safe_float(pr.get('spearman_corr')):.3f}**",
            f"- Median sequence refill quote: **{safe_float(pr.get('median_sequence_refill_quote')):,.2f}**",
            f"- Median frozen 7M.3 refill quote: **{safe_float(pr.get('median_frozen_7m3_refill_quote')):,.2f}**",
            f"- Median absolute relative error to frozen: **{safe_float(pr.get('median_abs_relative_error_to_frozen')):.2%}**",
            f"- Median symmetric relative difference: **{safe_float(pr.get('median_symmetric_abs_relative_difference')):.2%}**",
            f"- Median sequence/frozen ratio (both positive): **{safe_float(pr.get('median_sequence_to_frozen_ratio')):.3f}**",
            f"- Median / p95 absolute anchor offset: **{safe_float(pr.get('median_abs_anchor_offset_ms')):.1f} / {safe_float(pr.get('p95_abs_anchor_offset_ms')):.1f} ms**",
        ]
    lines += [
        "",
        "Parity is diagnostic rather than outcome-tuned: this phase does not invent a numeric parity threshold after seeing the pilot. Review the relative-error and anchor-offset tables before authorizing the full replay.",
        "",
        "## Strength lost by conservative attribution",
        "",
    ]
    if ir:
        lines += [
            f"- Raw post-refill-trade positive share: **{safe_float(ir.get('raw_positive_share')):.2%}**",
            f"- Conservative replacement-execution positive share: **{safe_float(ir.get('lower_bound_positive_share')):.2%}**",
            f"- Spearman raw vs conservative ratio: **{safe_float(ir.get('spearman_corr_raw_vs_lower_bound')):.3f}**",
        ]
    lines += [
        "",
        "No P75, P100, MFE, MAE, first-passage, or continuation outcome is read anywhere in this phase.",
    ]
    return "\n".join(lines) + "\n"


def run_self_test(phase7m6b_script: Path) -> int:
    # The arithmetic/matcher self-test does not require parquet I/O, so it can
    # run even in lightweight environments without pyarrow.
    phase7m6b = load_module(phase7m6b_script, "phase7m6b_for_7m6c_selftest")
    # Formula examples.
    levels = pd.DataFrame([
        {"candidate_id": "a", "candidate_date": "2026-06-01", "direction": 1, "observation_horizon_sec": 3, "price_tick": 1000, "price": 100.0, "first_refill_seen": True, "pre_refill_residual_qty": 2.0, "observed_refill_qty": 10.0, "observed_refill_quote": 1000.0, "strict_post_refill_aggressive_trade_qty": 8.0, "strict_post_refill_aggressive_trade_quote": 800.0, "same_exchange_ms_post_refill_aggressive_trade_qty": 0.0, "strict_trade_time_complete": True, "anchor_found": True, "sequence_valid": True},
        {"candidate_id": "b", "candidate_date": "2026-06-01", "direction": -1, "observation_horizon_sec": 3, "price_tick": 1000, "price": 100.0, "first_refill_seen": True, "pre_refill_residual_qty": 10.0, "observed_refill_qty": 5.0, "observed_refill_quote": 500.0, "strict_post_refill_aggressive_trade_qty": 8.0, "strict_post_refill_aggressive_trade_quote": 800.0, "same_exchange_ms_post_refill_aggressive_trade_qty": 0.0, "strict_trade_time_complete": True, "anchor_found": True, "sequence_valid": True},
    ])
    la = derive_level_attribution(levels)
    assert abs(float(la.loc[0, "replacement_execution_lower_bound_qty"]) - 6.0) < 1e-12
    assert abs(float(la.loc[1, "replacement_execution_lower_bound_qty"]) - 0.0) < 1e-12

    # Ensure the imported matcher remains callable with the preregistered gates.
    rows = []
    for i in range(90):
        rows.append({
            "candidate_id": f"x{i}", "candidate_date": "2026-06-01", "direction": 1,
            "f": float(i), "a": float(i % 7), "b": float(i % 5), "c": float(i % 3), "d": float(i % 11), "e": float(i % 13),
        })
    d = pd.DataFrame(rows)
    pairs = phase7m6b.greedy_pairs(d, "f", ["a", "b", "c", "d", "e"])
    assert isinstance(pairs, pd.DataFrame)
    print("PASS: Phase 7M.6C self-test (conservative lower-bound arithmetic + frozen matcher compatibility).")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 7M.6C frozen-pilot conservative replacement-execution attribution audit")
    p.add_argument("--phase7m6b-root", type=Path, default=DEFAULT_7M6B_ROOT)
    p.add_argument("--phase7m6b-script", type=Path, default=DEFAULT_7M6B_SCRIPT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test(args.phase7m6b_script)
    require_stack()

    print("Phase 7M.6C — Frozen-pilot conservative replacement-execution attribution audit")
    print("[1/6] Verifying signed Phase 7M.6B pilot checkpoints...")
    source_meta, days, manifests = verify_source_contract(args.phase7m6b_root)
    phase7m6b = load_module(args.phase7m6b_script, "phase7m6b_for_7m6c")
    for name, expected in (
        ("BALANCE_SMD_GATE", BALANCE_SMD_GATE),
        ("MIN_PROJECTED_FULL_PRIMARY_PAIRS", MIN_PROJECTED_FULL_PRIMARY_PAIRS),
        ("MIN_PROJECTED_SPLIT_PAIRS", MIN_PROJECTED_SPLIT_PAIRS),
        ("MIN_FEATURE_POSITIVE_SHARE", MIN_FEATURE_POSITIVE_SHARE),
        ("MAX_FEATURE_POSITIVE_SHARE", MAX_FEATURE_POSITIVE_SHARE),
    ):
        actual = safe_float(getattr(phase7m6b, name, math.nan))
        if not math.isfinite(actual) or abs(actual - float(expected)) > 1e-12:
            raise RuntimeError(f"Phase 7M.6B matcher/gate {name} changed: expected {expected}, found {actual}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)
    atomic_csv(args.report_root / "source_checkpoint_manifest.csv", manifests)

    print("[2/6] Loading only candidate_metrics + level_fate checkpoints (no raw replay)...")
    metrics, levels = read_pilot_checkpoints(args.phase7m6b_root, days)
    primary_metrics = metrics[pd.to_numeric(metrics["observation_horizon_sec"], errors="coerce") == PRIMARY_HORIZON_SEC].copy()
    if len(primary_metrics) != EXPECTED_PILOT_CANDIDATES:
        raise RuntimeError(f"Expected {EXPECTED_PILOT_CANDIDATES} primary candidate rows; found {len(primary_metrics)}")

    print("[3/6] Computing conservative replacement-execution lower bounds...")
    level_attr = derive_level_attribution(levels)
    candidate_attr = aggregate_candidate_attribution(metrics, level_attr)
    refill_sum_summary, refill_sum_detail = candidate_level_refill_parity(candidate_attr)
    refill_reconstruction_bad = (
        int(refill_sum_summary.loc[0, "rows_outside_float_tolerance"])
        + int(refill_sum_summary.loc[0, "positive_candidate_refill_missing_level_sum"])
        + int(refill_sum_summary.loc[0, "refilled_level_count_mismatches"])
    )
    if refill_reconstruction_bad != 0:
        raise RuntimeError(
            "Derived level refill sums/counts do not reproduce 7M.6B candidate refill totals; refusing attribution audit."
        )
    write_partitioned(level_attr, args.output_root / "level_attribution")
    write_partitioned(candidate_attr, args.output_root / "candidate_attribution")
    atomic_csv(args.report_root / "candidate_level_refill_aggregation_invariant.csv", refill_sum_summary)

    primary = candidate_attr[pd.to_numeric(candidate_attr["observation_horizon_sec"], errors="coerce") == PRIMARY_HORIZON_SEC].copy()
    refill_eligible = primary[pd.to_numeric(primary["opposing__observed_refill_quote"], errors="coerce") > 0].copy()

    print("[4/6] Auditing relative parity and information loss...")
    parity_summary, parity_detail = build_7m3_parity_audit(primary)
    info = build_information_loss_audit(refill_eligible)
    atomic_csv(args.report_root / "frozen_7m3_relative_parity_audit.csv", parity_summary)
    atomic_csv(args.report_root / "frozen_7m3_relative_parity_by_candidate.csv", parity_detail)
    atomic_csv(args.report_root / "frozen_7m3_relative_parity_worst20.csv", parity_detail.head(20))
    atomic_csv(args.report_root / "conservative_attribution_information_loss.csv", info)

    print("[5/6] Re-running pre-outcome common-support feasibility with refill amount + breadth controls...")
    gate, pairs = build_matching_feasibility(primary, phase7m6b)
    atomic_csv(args.report_root / "precompute_feasibility_gate.csv", gate)
    atomic_parquet(args.output_root / "pilot_matched_pairs_outcome_blind.parquet", pairs)

    primary_gate = gate[
        (gate["feature_role"] == "PRIMARY")
        & (gate["feature"] == "opposing__replacement_execution_lower_bound_to_refill_ratio")
        & (gate["control_set"] == "REFILL_AMOUNT_BREADTH_PRIMARY")
    ]
    primary_feasible = bool(primary_gate["feasible_before_outcomes"].iloc[0]) if len(primary_gate) else False

    print("[6/6] Writing frozen audit summary/invariants...")
    inv = pd.DataFrame([
        {"invariant": "future_outcomes_loaded", "pass": True, "actual": False, "required": False},
        {"invariant": "raw_market_replay_performed", "pass": True, "actual": False, "required": False},
        {"invariant": "signed_source_checkpoints", "pass": True, "actual": len(days), "required": EXPECTED_PILOT_DATES},
        {"invariant": "primary_candidate_count", "pass": len(primary) == EXPECTED_PILOT_CANDIDATES, "actual": len(primary), "required": EXPECTED_PILOT_CANDIDATES},
        {"invariant": "level_refill_sums_counts_reproduce_7m6b", "pass": refill_reconstruction_bad == 0, "actual": refill_reconstruction_bad, "required": 0},
        {"invariant": "same_ms_trades_excluded_from_strict_source_measure", "pass": manifests["same_ms_policy"].astype(str).str.len().gt(0).all(), "actual": manifests["same_ms_policy"].nunique(), "required": "explicit source contract"},
        {"invariant": "order_identity_not_claimed", "pass": (manifests["order_identity_claim"] == False).all(), "actual": manifests["order_identity_claim"].unique().tolist(), "required": False},
        {"invariant": "primary_controls_include_refill_amount_and_breadth", "pass": True, "actual": "momentum|aggression|destruction|sequence_refill|refill_breadth", "required": "amount+breadth"},
        {"invariant": "primary_common_support_feasible_before_outcomes", "pass": primary_feasible, "actual": primary_feasible, "required": True},
    ])
    atomic_csv(args.report_root / "analysis_invariants.csv", inv)

    status = "PRIMARY_ATTRIBUTION_FEASIBLE_FOR_FULL_REPLAY" if primary_feasible else "PRIMARY_ATTRIBUTION_NOT_FEASIBLE_ON_PILOT"
    summary = {
        "builder_version": BUILDER_VERSION,
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": status,
        "processed_dates": days,
        "primary_candidate_count": int(len(primary)),
        "primary_refill_eligible": int(len(refill_eligible)),
        "primary_common_support_feasible_before_outcomes": primary_feasible,
        "future_outcomes_loaded": False,
        "raw_market_replay_performed": False,
        "primary_feature": "opposing__replacement_execution_lower_bound_to_refill_ratio",
        "primary_control_set": "exact date+direction + momentum + aggression + initial destruction + sequence refill amount + refill breadth",
        "parity_numeric_gate": None,
        "parity_note": "Relative parity is diagnostic and intentionally not threshold-tuned after seeing pilot values.",
        "source_7m6b_summary_sha256": sha256_text(json.dumps(source_meta, sort_keys=True, default=str)),
    }
    atomic_json(args.output_root / "metadata" / "latest_summary.json", summary)
    atomic_json(args.output_root / "metadata" / "run_config.json", {
        "generated_at": utc_now_iso(),
        "source_root": str(args.phase7m6b_root),
        "source_script": str(args.phase7m6b_script),
        "processed_dates": days,
        "future_outcomes_loaded": False,
        "raw_market_replay_performed": False,
        "formula": "min(observed_refill_qty, max(0, strict_post_refill_aggressive_trade_qty - pre_refill_residual_qty))",
    })
    atomic_text(args.report_root / "findings.md", findings_text(summary, gate, parity_summary, info))

    print(f"Audit status            : {status}")
    print(f"Pilot candidates        : {len(primary):,}")
    print(f"3s refill eligible      : {len(refill_eligible):,}")
    if len(primary_gate):
        r = primary_gate.iloc[0]
        print(f"Primary pilot pairs     : {int(r['pilot_pairs_after_repair']):,}")
        print(f"Projected full pairs    : {int(r['projected_full_pairs']):,}")
        print(f"Projected D / V pairs   : {int(r['projected_discovery_pairs']):,} / {int(r['projected_validation_pairs']):,}")
        print(f"Primary max |SMD|       : {safe_float(r['pilot_max_abs_smd']):.3f}")
        print(f"Primary positive share  : {safe_float(r['positive_feature_share']):.2%}")
    print(f"Reports                 : {args.report_root}")
    print("Future outcomes loaded  : NO")
    print("Raw market replay       : NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
