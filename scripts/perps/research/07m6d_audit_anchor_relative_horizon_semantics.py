#!/usr/bin/env python3
"""Phase 7M.6D — checkpoint-only anchor-relative horizon semantics audit.

Scientific role
---------------
Phase 7M.6C found that the four-day Phase 7M.6B pilot systematically
under-counted 3-second refill relative to frozen Phase 7M.3, while the exact
frozen book state was usually re-found late inside the integer candidate
second. Phase 7M.6B then ended each horizon at candidate_second + h seconds.

This audit asks one narrow, outcome-blind question:

    Does measuring a true 3.000 seconds AFTER the recovered frozen-book anchor
    materially improve refill parity and conservative replacement-execution
    feasibility?

It DOES NOT replay raw market data and DOES NOT read future continuation labels.
It reconstructs the corrected 3-second sequence entirely from already-paid-for
Phase 7M.6B checkpoints:

    candidate_metrics  -> anchor timestamp + controls
    level_fate         -> frozen opposing levels / initial quantities
    event_tape         -> exact opposing depth changes + saved aggressive trades

The existing boundary-relative 3-second result is retained side-by-side as a
control. No outcome threshold, matcher caliper, or feasibility gate is changed.
"""

from __future__ import annotations

import argparse
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


BUILDER_VERSION = "1.0.1"
SCHEMA_VERSION = 1
SCHEMA_NAME = "prospective_25bp_refill_redepletion_anchor_timing_audit_v1"
SOURCE_SCHEMA = "prospective_25bp_refill_redepletion_sequence_v1"
PRIMARY_HORIZON_SEC = 3
TRADE_TAPE_HORIZON_SEC = 10
EXPECTED_PILOT_DATES = 4
EXPECTED_PILOT_CANDIDATES = 446

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()
DEFAULT_7M6B_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / SOURCE_SCHEMA
DEFAULT_7M6B_SCRIPT = REPO_ROOT / "scripts" / "perps" / "research" / "07m6b_build_refill_redepletion_sequence_pilot.py"
DEFAULT_7M6C_SCRIPT = REPO_ROOT / "scripts" / "perps" / "research" / "07m6c_audit_refill_redepletion_replacement_execution_attribution.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / SCHEMA_NAME
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "perps_prospective_25bp_refill_redepletion_anchor_timing_audit"

REQUIRED_METRIC_COLUMNS = (
    "candidate_id", "candidate_date", "direction", "observation_horizon_sec",
    "candidate_second", "anchor_found", "anchor_ts_recv_ns", "anchor_offset_ms",
    "sequence_valid", "invalid_reason",
    "opposing__refilled_level_count", "opposing__observed_refill_quote",
    "opposing__strict_post_refill_aggressive_trade_quote",
    "opposing__post_refill_aggressive_trade_to_refill_ratio",
    "opposing__refilled_level_post_trade_share",
    "opposing__strict_trade_time_complete_share",
    "control__directional_mid_move_bps_3s",
    "control__directional_aggressive_quote_3s",
    "control__opposing_peak_initial_queue_loss_share_3s",
    "control__opposing_refill_quote_3s_frozen",
    "is_reaccel_context",
)

REQUIRED_LEVEL_COLUMNS = (
    "candidate_id", "candidate_date", "direction", "observation_horizon_sec",
    "price_tick", "price", "initial_qty",
    # Required by the frozen 7M.6C conservative attribution helper when the
    # existing boundary-relative 3s result is reconstructed as a control.
    "pre_refill_residual_qty", "observed_refill_qty", "observed_refill_quote",
    "strict_post_refill_aggressive_trade_qty",
    "strict_post_refill_aggressive_trade_quote",
    "same_exchange_ms_post_refill_aggressive_trade_qty",
    "first_refill_seen", "strict_trade_time_complete",
    "anchor_found", "sequence_valid",
)

REQUIRED_TAPE_COLUMNS = (
    "candidate_id", "candidate_date", "direction", "event_kind", "ts_recv_ns",
    "price_tick", "price", "observation_horizon_sec",
    "transaction_time_ms", "exchange_event_ms", "old_qty", "new_qty", "delta_qty",
    "quantity", "buyer_is_maker", "trade_time_ms", "timing_class",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def require_stack() -> None:
    if pa is None or pq is None:
        raise SystemExit("Phase 7M.6D requires pandas, numpy and pyarrow in the project .venv.")


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
    fd, tmp = tempfile.mkstemp(prefix="tmp7m6d-", suffix=path.suffix or ".tmp", dir=str(path.parent))
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
    fd, tmp = tempfile.mkstemp(prefix="tmp7m6d-", suffix=".csv", dir=str(path.parent))
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
    fd, tmp = tempfile.mkstemp(prefix="tmp7m6d-", suffix=".parquet", dir=str(path.parent))
    os.close(fd)
    try:
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp, compression="zstd")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_module(path: Path, name: str):
    if not path.exists():
        raise RuntimeError(f"Missing required existing phase script: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def checkpoint_paths(root: Path, day: str) -> dict[str, Path]:
    return {
        "metrics": root / "candidate_metrics" / f"candidate_date={day}" / "part.parquet",
        "levels": root / "level_fate" / f"candidate_date={day}" / "part.parquet",
        "tape": root / "event_tape" / f"candidate_date={day}" / "part.parquet",
        "manifest": root / "checkpoints" / f"candidate_date={day}" / "manifest.json",
    }


def assert_columns(path: Path, required: Sequence[str], kind: str) -> None:
    names = set(pq.ParquetFile(path).schema_arrow.names)
    missing = [c for c in required if c not in names]
    if missing:
        raise RuntimeError(f"{kind} checkpoint {path} is missing required columns: {missing}")


def verify_source_contract(root: Path) -> tuple[dict[str, Any], list[str], pd.DataFrame]:
    meta_path = root / "metadata" / "latest_summary.json"
    if not meta_path.exists():
        raise RuntimeError(f"Missing Phase 7M.6B summary: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if str(meta.get("schema_name")) != SOURCE_SCHEMA:
        raise RuntimeError(f"Expected 7M.6B schema {SOURCE_SCHEMA!r}, found {meta.get('schema_name')!r}")
    if bool(meta.get("future_outcomes_loaded")):
        raise RuntimeError("Phase 7M.6B summary is not future-clean.")
    days = [str(x) for x in meta.get("processed_dates", [])]
    if len(days) != EXPECTED_PILOT_DATES or len(set(days)) != EXPECTED_PILOT_DATES:
        raise RuntimeError(f"Expected exactly {EXPECTED_PILOT_DATES} pilot dates; found {days}")

    rows = []
    for day in days:
        p = checkpoint_paths(root, day)
        for key in ("metrics", "levels", "tape", "manifest"):
            if not p[key].exists():
                raise RuntimeError(f"Missing Phase 7M.6B {key} checkpoint for {day}: {p[key]}")
        m = json.loads(p["manifest"].read_text(encoding="utf-8"))
        if str(m.get("status")) != "PASS":
            raise RuntimeError(f"Phase 7M.6B checkpoint {day} is not PASS")
        if str(m.get("schema_name")) != SOURCE_SCHEMA:
            raise RuntimeError(f"Phase 7M.6B checkpoint {day} schema mismatch")
        if bool(m.get("future_outcomes_loaded")):
            raise RuntimeError(f"Phase 7M.6B checkpoint {day} is not future-clean")
        contract = m.get("contract", {}) or {}
        if bool(contract.get("future_outcomes_loaded")):
            raise RuntimeError(f"Phase 7M.6B checkpoint {day} contract is not future-clean")
        rows.append({
            "candidate_date": day,
            "builder_version": str(m.get("builder_version")),
            "candidate_count": int(m.get("candidate_count", 0)),
            "status": str(m.get("status")),
            "future_outcomes_loaded": bool(m.get("future_outcomes_loaded")),
            "input_signature": str(m.get("input_signature", "")),
            "same_ms_policy": str(contract.get("same_ms_policy", "")),
            "order_identity_claim": contract.get("order_identity_claim"),
        })
    return meta, days, pd.DataFrame(rows)


def read_day_source(root: Path, day: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = checkpoint_paths(root, day)
    assert_columns(p["metrics"], REQUIRED_METRIC_COLUMNS, "candidate_metrics")
    assert_columns(p["levels"], REQUIRED_LEVEL_COLUMNS, "level_fate")
    assert_columns(p["tape"], REQUIRED_TAPE_COLUMNS, "event_tape")

    metrics = pq.read_table(p["metrics"], columns=list(REQUIRED_METRIC_COLUMNS)).to_pandas()
    levels = pq.read_table(p["levels"], columns=list(REQUIRED_LEVEL_COLUMNS)).to_pandas()
    tape = pq.read_table(p["tape"], columns=list(REQUIRED_TAPE_COLUMNS)).to_pandas()

    metrics["candidate_id"] = metrics["candidate_id"].astype(str)
    levels["candidate_id"] = levels["candidate_id"].astype(str)
    tape["candidate_id"] = tape["candidate_id"].astype(str)

    m3 = metrics[pd.to_numeric(metrics["observation_horizon_sec"], errors="coerce") == PRIMARY_HORIZON_SEC].copy()
    l3 = levels[pd.to_numeric(levels["observation_horizon_sec"], errors="coerce") == PRIMARY_HORIZON_SEC].copy()
    l10 = levels[pd.to_numeric(levels["observation_horizon_sec"], errors="coerce") == TRADE_TAPE_HORIZON_SEC].copy()
    return m3, l3, l10, tape


def reconstruct_anchor_relative_candidate(
    base: pd.Series,
    levels10: pd.DataFrame,
    tape: pd.DataFrame,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Rebuild exactly 3 seconds after the recovered anchor from saved tape."""
    cid = str(base["candidate_id"])
    day = str(base["candidate_date"])
    direction = int(base["direction"])
    candidate_second = int(base["candidate_second"])
    anchor_found = bool(base["anchor_found"])
    sequence_valid = bool(base["sequence_valid"])
    anchor_raw = safe_float(base["anchor_ts_recv_ns"])

    common = {
        "builder_version": BUILDER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "candidate_id": cid,
        "candidate_date": day,
        "direction": direction,
        "candidate_second": candidate_second,
        "observation_horizon_sec": PRIMARY_HORIZON_SEC,
        "timing_basis": "RECOVERED_ANCHOR_PLUS_3S",
        "anchor_found": anchor_found,
        "anchor_ts_recv_ns": anchor_raw,
        "anchor_offset_ms": safe_float(base["anchor_offset_ms"]),
        "sequence_valid": sequence_valid,
        "control__directional_mid_move_bps_3s": safe_float(base["control__directional_mid_move_bps_3s"]),
        "control__directional_aggressive_quote_3s": safe_float(base["control__directional_aggressive_quote_3s"]),
        "control__opposing_peak_initial_queue_loss_share_3s": safe_float(base["control__opposing_peak_initial_queue_loss_share_3s"]),
        "control__opposing_refill_quote_3s_frozen": safe_float(base["control__opposing_refill_quote_3s_frozen"]),
        "is_reaccel_context": bool(base["is_reaccel_context"]),
    }

    unavailable = {
        **common,
        "invalid_reason": str(base.get("invalid_reason", "")) or "ANCHOR_RELATIVE_SOURCE_UNAVAILABLE",
        "anchor_relative_endpoint_ts_recv_ns": math.nan,
        "effective_boundary_relative_window_ms": math.nan,
        "opposing__refilled_level_count": math.nan,
        "opposing__observed_refill_quote": math.nan,
        "opposing__strict_post_refill_aggressive_trade_quote": math.nan,
        "opposing__same_exchange_ms_post_refill_aggressive_trade_quote": math.nan,
        "opposing__post_refill_aggressive_trade_to_refill_ratio": math.nan,
        "opposing__refilled_level_post_trade_share": math.nan,
        "opposing__minimum_nontrade_queue_reduction_quote": math.nan,
        "opposing__minimum_nontrade_reduction_to_refill_ratio": math.nan,
        "opposing__strict_trade_time_complete_share": math.nan,
        "opposing__replacement_execution_lower_bound_quote": math.nan,
        "opposing__replacement_execution_lower_bound_to_refill_ratio": math.nan,
        "opposing__refilled_level_replacement_execution_lower_bound_share": math.nan,
    }

    if not anchor_found or not sequence_valid or not math.isfinite(anchor_raw):
        return unavailable, []

    anchor_ns = int(anchor_raw)
    boundary_start_ns = candidate_second * 1_000_000_000
    boundary_endpoint_ns = (candidate_second + PRIMARY_HORIZON_SEC) * 1_000_000_000
    endpoint_ns = anchor_ns + PRIMARY_HORIZON_SEC * 1_000_000_000
    anchor_offset_ms = (anchor_ns - boundary_start_ns) / 1e6

    # The 7M.6B anchor search was frozen to the candidate second. If an existing
    # checkpoint violates that, do not extrapolate beyond the tape contract.
    if anchor_ns < boundary_start_ns or anchor_ns > boundary_start_ns + 1_000_000_000:
        unavailable["invalid_reason"] = "ANCHOR_OUTSIDE_FROZEN_CANDIDATE_SECOND"
        return unavailable, []
    if endpoint_ns > boundary_start_ns + TRADE_TAPE_HORIZON_SEC * 1_000_000_000:
        unavailable["invalid_reason"] = "ANCHOR_RELATIVE_ENDPOINT_EXCEEDS_SAVED_TAPE"
        return unavailable, []

    lv = levels10[levels10["candidate_id"] == cid].copy()
    if lv.empty:
        unavailable["invalid_reason"] = "FROZEN_LEVELS_MISSING_IN_SAVED_LEVEL_FATE"
        return unavailable, []

    ct = tape[tape["candidate_id"] == cid].copy()
    depth = ct[ct["event_kind"].astype(str) == "DEPTH_CHANGE"].copy()
    if len(depth):
        depth["ts_recv_ns"] = pd.to_numeric(depth["ts_recv_ns"], errors="coerce")
        depth = depth[(depth["ts_recv_ns"] > anchor_ns) & (depth["ts_recv_ns"] < endpoint_ns)].copy()

    trades = ct[
        (ct["event_kind"].astype(str) == "AGGRESSIVE_TRADE_AT_REFILLED_LEVEL")
        & (pd.to_numeric(ct["observation_horizon_sec"], errors="coerce") == TRADE_TAPE_HORIZON_SEC)
    ].copy()
    if len(trades):
        trades["ts_recv_ns"] = pd.to_numeric(trades["ts_recv_ns"], errors="coerce")
        trades["trade_time_ms"] = pd.to_numeric(trades["trade_time_ms"], errors="coerce")
        trades["quantity"] = pd.to_numeric(trades["quantity"], errors="coerce")

    level_rows: list[dict[str, Any]] = []
    for lr in lv.itertuples(index=False):
        tick = int(lr.price_tick)
        price = safe_float(lr.price)
        initial = safe_float(lr.initial_qty)
        if not (math.isfinite(price) and price > 0 and math.isfinite(initial) and initial > 0):
            continue

        le = depth[depth["price_tick"].astype("Int64") == tick].copy() if len(depth) else pd.DataFrame()
        if len(le):
            le = le.sort_values("ts_recv_ns", kind="mergesort")

        current = initial
        minq = initial
        depleted = False
        first_refill_seen = False
        first_refill_recv: int | None = None
        first_refill_tx = math.nan
        first_refill_exchange = math.nan
        pre_refill_residual = math.nan
        refill_qty = 0.0
        refill_updates = 0
        gross_post_refill_decrease = 0.0
        post_refill_decrease_updates = 0

        for er in le.itertuples(index=False):
            oldq = safe_float(er.old_qty)
            newq = safe_float(er.new_qty)
            if not (math.isfinite(oldq) and math.isfinite(newq)):
                continue
            delta = newq - oldq
            current = newq
            if delta < -1e-12:
                minq = min(minq, newq)
                depleted = True
                if first_refill_seen:
                    gross_post_refill_decrease += -delta
                    post_refill_decrease_updates += 1
            elif delta > 1e-12 and depleted:
                refill_qty += delta
                refill_updates += 1
                if not first_refill_seen:
                    first_refill_seen = True
                    first_refill_recv = int(er.ts_recv_ns)
                    first_refill_tx = safe_float(er.transaction_time_ms)
                    first_refill_exchange = safe_float(er.exchange_event_ms)
                    pre_refill_residual = oldq

        strict_trade_qty = 0.0
        same_ms_trade_qty = 0.0
        before_trade_qty = 0.0
        strict_trade_count = 0
        trade_time_complete = True
        if first_refill_seen and first_refill_recv is not None:
            tg = trades[trades["price_tick"].astype("Int64") == tick].copy() if len(trades) else pd.DataFrame()
            if len(tg):
                tg = tg[(tg["ts_recv_ns"] > first_refill_recv) & (tg["ts_recv_ns"] < endpoint_ns)].copy()
                if len(tg):
                    tt = pd.to_numeric(tg["trade_time_ms"], errors="coerce")
                    trade_time_complete = bool(tt.notna().all() and math.isfinite(first_refill_tx))
                    if math.isfinite(first_refill_tx):
                        strict = tg[tt > first_refill_tx]
                        same = tg[tt == first_refill_tx]
                        before = tg[tt < first_refill_tx]
                        strict_trade_qty = float(pd.to_numeric(strict["quantity"], errors="coerce").fillna(0).sum())
                        same_ms_trade_qty = float(pd.to_numeric(same["quantity"], errors="coerce").fillna(0).sum())
                        before_trade_qty = float(pd.to_numeric(before["quantity"], errors="coerce").fillna(0).sum())
                        strict_trade_count = int(len(strict))
                    else:
                        trade_time_complete = False

        refill_quote = refill_qty * price
        strict_trade_quote = strict_trade_qty * price
        same_ms_quote = same_ms_trade_qty * price
        nontrade_lb_qty = max(0.0, gross_post_refill_decrease - strict_trade_qty)
        nontrade_lb_quote = nontrade_lb_qty * price

        lb_eligible = bool(
            first_refill_seen
            and refill_qty > 0
            and trade_time_complete
            and math.isfinite(pre_refill_residual)
            and pre_refill_residual >= 0
        )
        if lb_eligible:
            replacement_lb_qty = min(refill_qty, max(0.0, strict_trade_qty - pre_refill_residual))
            replacement_lb_quote = replacement_lb_qty * price
            replacement_lb_share = replacement_lb_qty / refill_qty if refill_qty > 0 else math.nan
        else:
            replacement_lb_qty = math.nan
            replacement_lb_quote = math.nan
            replacement_lb_share = math.nan

        level_rows.append({
            "builder_version": BUILDER_VERSION,
            "schema_version": SCHEMA_VERSION,
            "candidate_id": cid,
            "candidate_date": day,
            "direction": direction,
            "observation_horizon_sec": PRIMARY_HORIZON_SEC,
            "timing_basis": "RECOVERED_ANCHOR_PLUS_3S",
            "anchor_ts_recv_ns": anchor_ns,
            "anchor_relative_endpoint_ts_recv_ns": endpoint_ns,
            "price_tick": tick,
            "price": price,
            "initial_qty": initial,
            "min_qty": minq,
            "final_observed_qty": current,
            "depleted_before_refill": depleted,
            "first_refill_seen": first_refill_seen,
            "first_refill_ts_recv_ns": first_refill_recv,
            "first_refill_transaction_time_ms": first_refill_tx,
            "first_refill_exchange_event_ms": first_refill_exchange,
            "pre_refill_residual_qty": pre_refill_residual,
            "observed_refill_qty": refill_qty,
            "observed_refill_quote": refill_quote,
            "refill_update_count": refill_updates,
            "post_refill_gross_queue_reduction_qty": gross_post_refill_decrease,
            "post_refill_decrease_update_count": post_refill_decrease_updates,
            "strict_post_refill_aggressive_trade_qty": strict_trade_qty,
            "strict_post_refill_aggressive_trade_quote": strict_trade_quote,
            "same_exchange_ms_post_refill_aggressive_trade_qty": same_ms_trade_qty,
            "same_exchange_ms_post_refill_aggressive_trade_quote": same_ms_quote,
            "receive_after_but_exchange_before_trade_qty": before_trade_qty,
            "strict_post_refill_aggressive_trade_count": strict_trade_count,
            "strict_trade_time_complete": trade_time_complete,
            "minimum_nontrade_queue_reduction_qty": nontrade_lb_qty,
            "minimum_nontrade_queue_reduction_quote": nontrade_lb_quote,
            "replacement_execution_lower_bound_eligible": lb_eligible,
            "replacement_execution_lower_bound_qty": replacement_lb_qty,
            "replacement_execution_lower_bound_quote": replacement_lb_quote,
            "replacement_execution_lower_bound_share_of_refill": replacement_lb_share,
            "replacement_execution_lower_bound_positive": bool(lb_eligible and replacement_lb_qty > 1e-12),
        })

    lm = pd.DataFrame(level_rows)
    if lm.empty:
        unavailable["invalid_reason"] = "NO_OPPOSING_FROZEN_LEVEL_ROWS_RECONSTRUCTED"
        return unavailable, level_rows

    refilled = lm[bool_series(lm["first_refill_seen"]) & (pd.to_numeric(lm["observed_refill_quote"], errors="coerce") > 0)].copy()
    refill_quote = float(pd.to_numeric(refilled.get("observed_refill_quote", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    strict_trade_quote = float(pd.to_numeric(refilled.get("strict_post_refill_aggressive_trade_quote", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    same_ms_quote = float(pd.to_numeric(refilled.get("same_exchange_ms_post_refill_aggressive_trade_quote", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    nontrade_quote = float(pd.to_numeric(refilled.get("minimum_nontrade_queue_reduction_quote", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    n_refilled = int(len(refilled))
    n_traded = int((pd.to_numeric(refilled.get("strict_post_refill_aggressive_trade_quote", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0).sum()) if n_refilled else 0
    trade_complete_share = float(bool_series(refilled["strict_trade_time_complete"]).mean()) if n_refilled else math.nan

    lb_eligible_mask = bool_series(refilled["replacement_execution_lower_bound_eligible"]) if n_refilled else pd.Series(dtype=bool)
    fully_attributable = bool(n_refilled > 0 and int(lb_eligible_mask.sum()) == n_refilled)
    if fully_attributable:
        lb_quote = float(pd.to_numeric(refilled["replacement_execution_lower_bound_quote"], errors="coerce").fillna(0).sum())
        lb_positive_levels = int(bool_series(refilled["replacement_execution_lower_bound_positive"]).sum())
        lb_ratio = lb_quote / refill_quote if refill_quote > 0 else math.nan
        lb_level_share = lb_positive_levels / n_refilled if n_refilled > 0 else math.nan
    else:
        lb_quote = math.nan
        lb_ratio = math.nan
        lb_level_share = math.nan

    metric = {
        **common,
        "invalid_reason": "",
        "anchor_relative_endpoint_ts_recv_ns": endpoint_ns,
        "effective_boundary_relative_window_ms": max(0.0, (boundary_endpoint_ns - anchor_ns) / 1e6),
        "opposing__refilled_level_count": n_refilled,
        "opposing__observed_refill_quote": refill_quote,
        "opposing__strict_post_refill_aggressive_trade_quote": strict_trade_quote,
        "opposing__same_exchange_ms_post_refill_aggressive_trade_quote": same_ms_quote,
        "opposing__post_refill_aggressive_trade_to_refill_ratio": strict_trade_quote / refill_quote if refill_quote > 0 else math.nan,
        "opposing__refilled_level_post_trade_share": n_traded / n_refilled if n_refilled > 0 else math.nan,
        "opposing__minimum_nontrade_queue_reduction_quote": nontrade_quote,
        "opposing__minimum_nontrade_reduction_to_refill_ratio": nontrade_quote / refill_quote if refill_quote > 0 else math.nan,
        "opposing__strict_trade_time_complete_share": trade_complete_share,
        "opposing__replacement_execution_lower_bound_quote": lb_quote,
        "opposing__replacement_execution_lower_bound_to_refill_ratio": lb_ratio,
        "opposing__refilled_level_replacement_execution_lower_bound_share": lb_level_share,
    }
    return metric, level_rows


def reconstruct_all_days(root: Path, days: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    corrected_metrics: list[dict[str, Any]] = []
    corrected_levels: list[dict[str, Any]] = []
    boundary_m3_parts = []
    boundary_l3_parts = []

    for day in days:
        m3, l3, l10, tape = read_day_source(root, day)
        boundary_m3_parts.append(m3)
        boundary_l3_parts.append(l3)
        for _, base in m3.iterrows():
            metric, levels = reconstruct_anchor_relative_candidate(base, l10, tape)
            corrected_metrics.append(metric)
            corrected_levels.extend(levels)

    return (
        pd.DataFrame(corrected_metrics),
        pd.DataFrame(corrected_levels),
        pd.concat(boundary_m3_parts, ignore_index=True, sort=False),
        pd.concat(boundary_l3_parts, ignore_index=True, sort=False),
    )


def parity_summary(df: pd.DataFrame, sequence_col: str, frozen_col: str, label: str) -> dict[str, Any]:
    x = pd.to_numeric(df[sequence_col], errors="coerce")
    f = pd.to_numeric(df[frozen_col], errors="coerce")
    valid = x.notna() & f.notna()
    both_pos = valid & (x > 0) & (f > 0)
    denom = f.abs().replace(0, np.nan)
    rel = (x - f).abs() / denom
    sym = (x - f).abs() / ((x.abs() + f.abs()) / 2).replace(0, np.nan)
    return {
        "timing_basis": label,
        "rows_compared": int(valid.sum()),
        "pearson_corr": float(x[valid].corr(f[valid], method="pearson")) if int(valid.sum()) >= 3 else math.nan,
        "spearman_corr": float(x[valid].corr(f[valid], method="spearman")) if int(valid.sum()) >= 3 else math.nan,
        "median_sequence_refill_quote": float(x[valid].median()) if valid.any() else math.nan,
        "median_frozen_refill_quote": float(f[valid].median()) if valid.any() else math.nan,
        "median_abs_relative_error_to_frozen": float(rel[valid].median()) if valid.any() else math.nan,
        "median_symmetric_abs_relative_difference": float(sym[valid].median()) if valid.any() else math.nan,
        "median_sequence_to_frozen_ratio": float((x[both_pos] / f[both_pos]).median()) if both_pos.any() else math.nan,
        "both_positive_rows": int(both_pos.sum()),
    }


def build_timing_comparison(boundary: pd.DataFrame, corrected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["candidate_id", "candidate_date", "direction"]
    b = boundary[keys + [
        "anchor_offset_ms", "opposing__observed_refill_quote",
        "control__opposing_refill_quote_3s_frozen",
        "opposing__post_refill_aggressive_trade_to_refill_ratio",
    ]].copy().rename(columns={
        "opposing__observed_refill_quote": "boundary_3s_refill_quote",
        "opposing__post_refill_aggressive_trade_to_refill_ratio": "boundary_3s_post_refill_trade_ratio",
    })
    c = corrected[keys + [
        "effective_boundary_relative_window_ms", "opposing__observed_refill_quote",
        "opposing__post_refill_aggressive_trade_to_refill_ratio",
        "opposing__replacement_execution_lower_bound_to_refill_ratio",
    ]].copy().rename(columns={
        "opposing__observed_refill_quote": "anchor_3s_refill_quote",
        "opposing__post_refill_aggressive_trade_to_refill_ratio": "anchor_3s_post_refill_trade_ratio",
        "opposing__replacement_execution_lower_bound_to_refill_ratio": "anchor_3s_replacement_execution_lower_bound_ratio",
    })
    d = b.merge(c, on=keys, how="left", validate="one_to_one")
    for col in (
        "anchor_offset_ms", "boundary_3s_refill_quote", "anchor_3s_refill_quote",
        "control__opposing_refill_quote_3s_frozen", "effective_boundary_relative_window_ms",
    ):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d["anchor_minus_boundary_refill_quote"] = d["anchor_3s_refill_quote"] - d["boundary_3s_refill_quote"]
    d["anchor_to_boundary_refill_ratio"] = d["anchor_3s_refill_quote"] / d["boundary_3s_refill_quote"].replace(0, np.nan)
    d["boundary_abs_error_to_frozen"] = (d["boundary_3s_refill_quote"] - d["control__opposing_refill_quote_3s_frozen"]).abs()
    d["anchor_abs_error_to_frozen"] = (d["anchor_3s_refill_quote"] - d["control__opposing_refill_quote_3s_frozen"]).abs()
    comparable = d["boundary_abs_error_to_frozen"].notna() & d["anchor_abs_error_to_frozen"].notna()
    d["anchor_parity_abs_error_improved"] = np.where(
        comparable,
        d["anchor_abs_error_to_frozen"] < d["boundary_abs_error_to_frozen"],
        False,
    )

    p_boundary = parity_summary(d, "boundary_3s_refill_quote", "control__opposing_refill_quote_3s_frozen", "INTEGER_SECOND_BOUNDARY_3S")
    p_anchor = parity_summary(d, "anchor_3s_refill_quote", "control__opposing_refill_quote_3s_frozen", "RECOVERED_ANCHOR_PLUS_3S")
    summary_rows = [p_boundary, p_anchor]
    summary = pd.DataFrame(summary_rows)

    extra = pd.DataFrame([{
        "timing_basis": "ANCHOR_SHIFT_DIAGNOSTIC",
        "rows_compared": int(len(d)),
        "median_anchor_offset_ms": safe_float(d["anchor_offset_ms"].median()),
        "p95_anchor_offset_ms": safe_float(d["anchor_offset_ms"].quantile(0.95)),
        "median_effective_old_post_anchor_window_ms": safe_float(d["effective_boundary_relative_window_ms"].median()),
        "median_anchor_minus_boundary_refill_quote": safe_float(d["anchor_minus_boundary_refill_quote"].median()),
        "share_anchor_refill_gt_boundary": float((d["anchor_3s_refill_quote"] > d["boundary_3s_refill_quote"]).mean()),
        "share_abs_parity_error_improved": float(bool_series(d["anchor_parity_abs_error_improved"]).mean()),
    }])
    return pd.concat([summary, extra], ignore_index=True, sort=False), d


def findings_text(summary: dict[str, Any], timing: pd.DataFrame, gate: pd.DataFrame, info: pd.DataFrame) -> str:
    p_old = timing[timing["timing_basis"] == "INTEGER_SECOND_BOUNDARY_3S"]
    p_new = timing[timing["timing_basis"] == "RECOVERED_ANCHOR_PLUS_3S"]
    diag = timing[timing["timing_basis"] == "ANCHOR_SHIFT_DIAGNOSTIC"]
    g = gate[
        (gate["feature_role"] == "PRIMARY")
        & (gate["feature"] == "opposing__replacement_execution_lower_bound_to_refill_ratio")
        & (gate["control_set"] == "REFILL_AMOUNT_BREADTH_PRIMARY")
    ]
    a = p_old.iloc[0].to_dict() if len(p_old) else {}
    b = p_new.iloc[0].to_dict() if len(p_new) else {}
    dd = diag.iloc[0].to_dict() if len(diag) else {}
    gg = g.iloc[0].to_dict() if len(g) else {}
    ii = info.iloc[0].to_dict() if len(info) else {}
    lines = [
        "# Phase 7M.6D — Anchor-relative 3-second horizon semantics audit",
        "",
        f"- Status: **{summary['status']}**",
        f"- Pilot dates: **{', '.join(summary['processed_dates'])}**",
        f"- Candidates: **{summary['primary_candidate_count']}**",
        "- Raw market replay: **NO**",
        "- Future outcomes loaded: **NO**",
        "",
        "## Question",
        "",
        "Does measuring exactly 3 seconds after the recovered frozen-book anchor improve parity with Phase 7M.3 and restore enough common support for conservative replacement-execution attribution?",
        "",
        "## Timing diagnostic",
        "",
    ]
    if dd:
        lines += [
            f"- Median / p95 anchor offset: **{safe_float(dd.get('median_anchor_offset_ms')):.1f} / {safe_float(dd.get('p95_anchor_offset_ms')):.1f} ms**",
            f"- Median effective old post-anchor window: **{safe_float(dd.get('median_effective_old_post_anchor_window_ms')):.1f} ms**",
            f"- Share with more refill under anchor+3s: **{safe_float(dd.get('share_anchor_refill_gt_boundary')):.2%}**",
            f"- Share whose absolute refill error vs frozen 7M.3 improved: **{safe_float(dd.get('share_abs_parity_error_improved')):.2%}**",
        ]
    lines += ["", "## Refill parity", ""]
    if a and b:
        lines += [
            f"- Boundary 3s median sequence/frozen ratio: **{safe_float(a.get('median_sequence_to_frozen_ratio')):.3f}**",
            f"- Anchor+3s median sequence/frozen ratio: **{safe_float(b.get('median_sequence_to_frozen_ratio')):.3f}**",
            f"- Boundary 3s median absolute relative error: **{safe_float(a.get('median_abs_relative_error_to_frozen')):.2%}**",
            f"- Anchor+3s median absolute relative error: **{safe_float(b.get('median_abs_relative_error_to_frozen')):.2%}**",
            f"- Boundary Pearson / Spearman: **{safe_float(a.get('pearson_corr')):.3f} / {safe_float(a.get('spearman_corr')):.3f}**",
            f"- Anchor+3s Pearson / Spearman: **{safe_float(b.get('pearson_corr')):.3f} / {safe_float(b.get('spearman_corr')):.3f}**",
        ]
    lines += ["", "## Corrected conservative-attribution feasibility", ""]
    if gg:
        lines += [
            f"- Pilot eligible candidates: **{int(gg.get('pilot_eligible_candidates', 0))}**",
            f"- Matched pairs after repair: **{int(gg.get('pilot_pairs_after_repair', 0))}**",
            f"- Max |SMD|: **{safe_float(gg.get('pilot_max_abs_smd')):.3f}**",
            f"- Projected full pairs: **{int(gg.get('projected_full_pairs', 0))}**",
            f"- Projected discovery / validation pairs: **{int(gg.get('projected_discovery_pairs', 0))} / {int(gg.get('projected_validation_pairs', 0))}**",
            f"- Positive feature share: **{safe_float(gg.get('positive_feature_share')):.2%}**",
            f"- Feasible before outcomes: **{bool(gg.get('feasible_before_outcomes'))}**",
        ]
    if ii:
        lines += [
            "",
            "## Attribution strength",
            "",
            f"- Raw post-refill-trade positive share: **{safe_float(ii.get('raw_positive_share')):.2%}**",
            f"- Conservative replacement-execution positive share: **{safe_float(ii.get('lower_bound_positive_share')):.2%}**",
        ]
    lines += [
        "",
        "No parity threshold was invented after seeing these values. The preregistered matching and feasibility gates are reused unchanged from 7M.6B/7M.6C.",
    ]
    return "\n".join(lines) + "\n"


def run_self_test() -> int:
    # Synthetic candidate: anchor occurs 0.9s into the candidate second. A second
    # refill arrives at +3.4s from the integer boundary, so old boundary-3s misses
    # it while anchor+3s includes it.
    base = pd.Series({
        "candidate_id": "x", "candidate_date": "2026-06-01", "direction": 1,
        "candidate_second": 100, "anchor_found": True, "anchor_ts_recv_ns": 100_900_000_000,
        "anchor_offset_ms": 900.0, "sequence_valid": True, "invalid_reason": "",
        "control__directional_mid_move_bps_3s": 1.0,
        "control__directional_aggressive_quote_3s": 1000.0,
        "control__opposing_peak_initial_queue_loss_share_3s": 0.5,
        "control__opposing_refill_quote_3s_frozen": 800.0,
        "is_reaccel_context": False,
    })
    levels = pd.DataFrame([{
        "candidate_id": "x", "candidate_date": "2026-06-01", "direction": 1,
        "observation_horizon_sec": 10, "price_tick": 1000, "price": 100.0, "initial_qty": 10.0,
    }])
    tape = pd.DataFrame([
        {"candidate_id":"x","candidate_date":"2026-06-01","direction":1,"event_kind":"DEPTH_CHANGE","ts_recv_ns":101_100_000_000,"price_tick":1000,"price":100.0,"observation_horizon_sec":np.nan,"transaction_time_ms":101100.0,"exchange_event_ms":101100.0,"old_qty":10.0,"new_qty":2.0,"delta_qty":-8.0,"quantity":np.nan,"buyer_is_maker":np.nan,"trade_time_ms":np.nan,"timing_class":np.nan},
        {"candidate_id":"x","candidate_date":"2026-06-01","direction":1,"event_kind":"DEPTH_CHANGE","ts_recv_ns":101_300_000_000,"price_tick":1000,"price":100.0,"observation_horizon_sec":np.nan,"transaction_time_ms":101300.0,"exchange_event_ms":101300.0,"old_qty":2.0,"new_qty":5.0,"delta_qty":3.0,"quantity":np.nan,"buyer_is_maker":np.nan,"trade_time_ms":np.nan,"timing_class":np.nan},
        {"candidate_id":"x","candidate_date":"2026-06-01","direction":1,"event_kind":"DEPTH_CHANGE","ts_recv_ns":103_400_000_000,"price_tick":1000,"price":100.0,"observation_horizon_sec":np.nan,"transaction_time_ms":103400.0,"exchange_event_ms":103400.0,"old_qty":5.0,"new_qty":10.0,"delta_qty":5.0,"quantity":np.nan,"buyer_is_maker":np.nan,"trade_time_ms":np.nan,"timing_class":np.nan},
        {"candidate_id":"x","candidate_date":"2026-06-01","direction":1,"event_kind":"AGGRESSIVE_TRADE_AT_REFILLED_LEVEL","ts_recv_ns":103_500_000_000,"price_tick":1000,"price":100.0,"observation_horizon_sec":10,"transaction_time_ms":np.nan,"exchange_event_ms":103500.0,"old_qty":np.nan,"new_qty":np.nan,"delta_qty":np.nan,"quantity":7.0,"buyer_is_maker":False,"trade_time_ms":103500.0,"timing_class":"STRICT_AFTER"},
    ])
    metric, level_rows = reconstruct_anchor_relative_candidate(base, levels, tape)
    assert len(level_rows) == 1
    assert abs(float(metric["opposing__observed_refill_quote"]) - 800.0) < 1e-12
    # residual at first refill is 2; later aggressive trade is 7 -> >=5 replacement
    assert abs(float(metric["opposing__replacement_execution_lower_bound_quote"]) - 500.0) < 1e-12
    assert abs(float(metric["effective_boundary_relative_window_ms"]) - 2100.0) < 1e-12
    # Regression guard for v1.0.0 failure: the boundary-control level read must
    # contain every field consumed by Phase 7M.6C derive_level_attribution().
    required_boundary_attr = {
        "price", "pre_refill_residual_qty", "observed_refill_qty",
        "observed_refill_quote", "strict_post_refill_aggressive_trade_qty",
        "strict_post_refill_aggressive_trade_quote",
        "same_exchange_ms_post_refill_aggressive_trade_qty",
        "first_refill_seen", "strict_trade_time_complete",
        "anchor_found", "sequence_valid",
    }
    assert required_boundary_attr.issubset(set(REQUIRED_LEVEL_COLUMNS))
    print("PASS: Phase 7M.6D self-test (anchor-relative window + saved-tape refill/trade reconstruction + boundary-control schema).")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 7M.6D checkpoint-only anchor-relative 3-second horizon audit")
    p.add_argument("--phase7m6b-root", type=Path, default=DEFAULT_7M6B_ROOT)
    p.add_argument("--phase7m6b-script", type=Path, default=DEFAULT_7M6B_SCRIPT)
    p.add_argument("--phase7m6c-script", type=Path, default=DEFAULT_7M6C_SCRIPT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()
    require_stack()

    print("Phase 7M.6D — Checkpoint-only anchor-relative 3-second horizon audit")
    print("[1/7] Verifying signed Phase 7M.6B source checkpoints...")
    source_meta, days, manifests = verify_source_contract(args.phase7m6b_root)
    phase7m6b = load_module(args.phase7m6b_script, "phase7m6b_for_7m6d")
    phase7m6c = load_module(args.phase7m6c_script, "phase7m6c_for_7m6d")

    # Freeze compatibility with the exact preregistered feasibility rules. We do
    # not silently accept a locally edited matcher/gate during this timing audit.
    for name in (
        "BALANCE_SMD_GATE", "MIN_PROJECTED_FULL_PRIMARY_PAIRS", "MIN_PROJECTED_SPLIT_PAIRS",
        "MIN_FEATURE_POSITIVE_SHARE", "MAX_FEATURE_POSITIVE_SHARE",
    ):
        b = safe_float(getattr(phase7m6b, name, math.nan))
        c = safe_float(getattr(phase7m6c, name, math.nan))
        if not (math.isfinite(b) and math.isfinite(c) and abs(b - c) <= 1e-12):
            raise RuntimeError(f"Frozen feasibility gate mismatch for {name}: 7M.6B={b}, 7M.6C={c}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)
    atomic_csv(args.report_root / "source_checkpoint_manifest.csv", manifests)

    print("[2/7] Reconstructing anchor+3s from saved depth/trade event tape (no raw replay)...")
    corrected, corrected_levels, boundary_m3, boundary_l3 = reconstruct_all_days(args.phase7m6b_root, days)
    if len(corrected) != EXPECTED_PILOT_CANDIDATES:
        raise RuntimeError(f"Expected {EXPECTED_PILOT_CANDIDATES} corrected candidate rows; found {len(corrected)}")
    atomic_parquet(args.output_root / "corrected_anchor3s_candidate_metrics.parquet", corrected)
    if len(corrected_levels):
        atomic_parquet(args.output_root / "corrected_anchor3s_level_fate.parquet", corrected_levels)

    print("[3/7] Reconstructing the existing boundary-3s conservative attribution as control...")
    boundary_level_attr = phase7m6c.derive_level_attribution(boundary_l3)
    boundary_attr = phase7m6c.aggregate_candidate_attribution(boundary_m3, boundary_level_attr)
    boundary_primary = boundary_attr[pd.to_numeric(boundary_attr["observation_horizon_sec"], errors="coerce") == PRIMARY_HORIZON_SEC].copy()

    print("[4/7] Comparing boundary-relative vs recovered-anchor-relative refill parity...")
    timing_summary, timing_detail = build_timing_comparison(boundary_primary, corrected)
    atomic_csv(args.report_root / "refill_parity_boundary_vs_anchor.csv", timing_summary)
    atomic_csv(args.report_root / "refill_parity_by_candidate.csv", timing_detail)
    worst = timing_detail.sort_values("boundary_abs_error_to_frozen", ascending=False, kind="mergesort").head(25)
    atomic_csv(args.report_root / "refill_parity_worst25_boundary.csv", worst)

    print("[5/7] Re-running unchanged pre-outcome common-support feasibility on corrected anchor+3s...")
    gate, pairs = phase7m6c.build_matching_feasibility(corrected, phase7m6b)
    atomic_csv(args.report_root / "precompute_feasibility_gate_anchor3s.csv", gate)
    atomic_parquet(args.output_root / "pilot_matched_pairs_anchor3s_outcome_blind.parquet", pairs)
    info = phase7m6c.build_information_loss_audit(
        corrected[pd.to_numeric(corrected["opposing__observed_refill_quote"], errors="coerce") > 0].copy()
    )
    atomic_csv(args.report_root / "conservative_attribution_information_loss_anchor3s.csv", info)

    primary_gate = gate[
        (gate["feature_role"] == "PRIMARY")
        & (gate["feature"] == "opposing__replacement_execution_lower_bound_to_refill_ratio")
        & (gate["control_set"] == "REFILL_AMOUNT_BREADTH_PRIMARY")
    ]
    primary_feasible = bool(primary_gate["feasible_before_outcomes"].iloc[0]) if len(primary_gate) else False

    print("[6/7] Writing timing invariants...")
    anchor = pd.to_numeric(corrected["anchor_offset_ms"], errors="coerce")
    effective = pd.to_numeric(corrected["effective_boundary_relative_window_ms"], errors="coerce")
    source_valid = bool_series(corrected["anchor_found"]) & bool_series(corrected["sequence_valid"])
    endpoint_ok = (~source_valid) | (
        pd.to_numeric(corrected["anchor_relative_endpoint_ts_recv_ns"], errors="coerce").notna()
        & (effective >= -1e-9)
        & (effective <= PRIMARY_HORIZON_SEC * 1000 + 1e-9)
    )
    inv = pd.DataFrame([
        {"invariant": "future_outcomes_loaded", "pass": True, "actual": False, "required": False},
        {"invariant": "raw_market_replay_performed", "pass": True, "actual": False, "required": False},
        {"invariant": "signed_source_checkpoints", "pass": len(days) == EXPECTED_PILOT_DATES, "actual": len(days), "required": EXPECTED_PILOT_DATES},
        {"invariant": "primary_candidate_count", "pass": len(corrected) == EXPECTED_PILOT_CANDIDATES, "actual": len(corrected), "required": EXPECTED_PILOT_CANDIDATES},
        {"invariant": "anchor_relative_endpoint_within_saved_10s_tape", "pass": bool(endpoint_ok.all()), "actual": int((~endpoint_ok).sum()), "required": 0},
        {"invariant": "anchor_offset_inside_frozen_candidate_second", "pass": bool(((anchor[source_valid] >= -1e-9) & (anchor[source_valid] <= 1000.0 + 1e-9)).all()), "actual": f"median={safe_float(anchor[source_valid].median()):.3f}ms", "required": "0..1000ms"},
        {"invariant": "same_ms_trades_remain_excluded", "pass": manifests["same_ms_policy"].astype(str).str.len().gt(0).all(), "actual": manifests["same_ms_policy"].nunique(), "required": "explicit source policy"},
        {"invariant": "order_identity_not_claimed", "pass": (manifests["order_identity_claim"] == False).all(), "actual": manifests["order_identity_claim"].unique().tolist(), "required": False},
        {"invariant": "matching_gates_unchanged", "pass": True, "actual": "7M.6B == 7M.6C", "required": "unchanged"},
        {"invariant": "primary_common_support_feasible_before_outcomes", "pass": primary_feasible, "actual": primary_feasible, "required": True},
    ])
    atomic_csv(args.report_root / "analysis_invariants.csv", inv)

    old = timing_summary[timing_summary["timing_basis"] == "INTEGER_SECOND_BOUNDARY_3S"]
    new = timing_summary[timing_summary["timing_basis"] == "RECOVERED_ANCHOR_PLUS_3S"]
    old_err = safe_float(old["median_abs_relative_error_to_frozen"].iloc[0]) if len(old) else math.nan
    new_err = safe_float(new["median_abs_relative_error_to_frozen"].iloc[0]) if len(new) else math.nan
    parity_direction_improved = bool(math.isfinite(old_err) and math.isfinite(new_err) and new_err < old_err)

    if primary_feasible and parity_direction_improved:
        status = "ANCHOR_TIMING_CORRECTION_SUPPORTED_AND_PRIMARY_FEASIBLE"
    elif primary_feasible:
        status = "PRIMARY_FEASIBLE_BUT_PARITY_NOT_IMPROVED"
    elif parity_direction_improved:
        status = "ANCHOR_TIMING_IMPROVES_PARITY_BUT_PRIMARY_STILL_UNDERPOWERED"
    else:
        status = "ANCHOR_TIMING_DOES_NOT_RESOLVE_PILOT_LIMITATION"

    print("[7/7] Writing audit summary/findings...")
    summary = {
        "builder_version": BUILDER_VERSION,
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": status,
        "processed_dates": days,
        "primary_candidate_count": int(len(corrected)),
        "future_outcomes_loaded": False,
        "raw_market_replay_performed": False,
        "source_schema": SOURCE_SCHEMA,
        "timing_hypothesis": "existing boundary endpoint shortened post-anchor window because frozen anchor was recovered inside candidate second",
        "corrected_timing_basis": "anchor_ts_recv_ns + 3_000_000_000 ns",
        "trade_tape_source_horizon_sec": TRADE_TAPE_HORIZON_SEC,
        "primary_common_support_feasible_before_outcomes": primary_feasible,
        "parity_direction_improved": parity_direction_improved,
        "parity_numeric_gate": None,
        "parity_note": "Parity remains diagnostic; no numeric pass threshold was invented after seeing pilot values.",
    }
    atomic_json(args.output_root / "metadata" / "latest_summary.json", summary)
    atomic_json(args.output_root / "metadata" / "run_config.json", {
        "generated_at": utc_now_iso(),
        "source_root": str(args.phase7m6b_root),
        "source_7m6b_script": str(args.phase7m6b_script),
        "source_7m6c_script": str(args.phase7m6c_script),
        "processed_dates": days,
        "future_outcomes_loaded": False,
        "raw_market_replay_performed": False,
        "corrected_endpoint": "anchor_ts_recv_ns + 3 seconds",
        "strict_trade_rule": "receive after refill + trade_time_ms strictly after refill transaction_time_ms; same-ms excluded",
        "replacement_execution_formula": "min(refill_qty, max(0, strict_post_refill_aggressive_trade_qty - pre_refill_residual_qty))",
    })
    atomic_text(args.report_root / "findings.md", findings_text(summary, timing_summary, gate, info))

    print(f"Audit status            : {status}")
    print(f"Pilot candidates        : {len(corrected):,}")
    diag = timing_summary[timing_summary["timing_basis"] == "ANCHOR_SHIFT_DIAGNOSTIC"]
    if len(diag):
        r = diag.iloc[0]
        print(f"Median anchor offset    : {safe_float(r.get('median_anchor_offset_ms')):.1f} ms")
        print(f"Old effective window    : {safe_float(r.get('median_effective_old_post_anchor_window_ms')):.1f} ms")
        print(f"Parity error improved   : {safe_float(r.get('share_abs_parity_error_improved')):.2%} of comparable candidates")
    if len(old) and len(new):
        print(f"Median refill/frozen    : boundary={safe_float(old.iloc[0].get('median_sequence_to_frozen_ratio')):.3f} | anchor={safe_float(new.iloc[0].get('median_sequence_to_frozen_ratio')):.3f}")
        print(f"Median abs rel error    : boundary={old_err:.2%} | anchor={new_err:.2%}")
    if len(primary_gate):
        g = primary_gate.iloc[0]
        print(f"Primary pilot pairs     : {int(g['pilot_pairs_after_repair']):,}")
        print(f"Projected full pairs    : {int(g['projected_full_pairs']):,}")
        print(f"Projected D / V pairs   : {int(g['projected_discovery_pairs']):,} / {int(g['projected_validation_pairs']):,}")
        print(f"Primary max |SMD|       : {safe_float(g['pilot_max_abs_smd']):.3f}")
        print(f"Primary positive share  : {safe_float(g['positive_feature_share']):.2%}")
    print(f"Reports                 : {args.report_root}")
    print("Future outcomes loaded  : NO")
    print("Raw market replay       : NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
