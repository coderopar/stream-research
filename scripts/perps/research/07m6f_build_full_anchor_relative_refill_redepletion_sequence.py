#!/usr/bin/env python3
"""
Phase 7M.6F — Full corrected anchor-relative refill→re-depletion substrate
============================================================================

SCIENTIFIC QUESTION
-------------------
Across the complete frozen prospective 25-bp candidate universe, can we classify
whether aggressive execution demonstrably penetrates replenished opposing
liquidity while preserving enough outcome-blind common support to later test
whether that physical state is associated with P75 continuation?

THIS PHASE DOES NOT TEST THE OUTCOME.
-------------------------------------
No Phase-7M.3 future outcome file, P75/P100 label, MFE/MAE field, or other future
continuation label is opened here.  The phase ends after the full-data
pre-outcome feasibility gate and freezes the matched-pair manifest if feasible.

FROZEN MEASUREMENT CONTRACT
---------------------------
* Candidate universe: Phase 7M.3 prospective EARLY_25BP_RESET5_RELEASE10_REARM60.
* Exact local-book anchor: the frozen Phase 7M.3 price-level vector re-found
  within the candidate second.
* Observation horizons: anchor_ts_recv_ns + 1/3/5/10 seconds.
* Primary horizon: 3 seconds.
* Opposing side: asks for UP candidates, bids for DOWN candidates.
* Refill: positive absolute-quantity update at the same frozen opposing price
  after prior observed depletion below the anchor quantity.
* Strict post-refill aggressive trade: same price, correct aggressor direction,
  receive timestamp after the refill, and exchange trade_time_ms strictly after
  the refill transaction_time_ms. Same-exchange-millisecond trades stay in the
  ambiguity bucket and are excluded from strict attribution.
* Conservative replacement-execution lower bound at each refilled level:

      min(refill_qty,
          max(0, strict_post_refill_aggressive_trade_qty
                 - pre_refill_residual_qty))

* Primary binary exposure at 3s:

      replacement_penetrated =
          opposing__replacement_execution_lower_bound_quote > 0

  This is the physical zero boundary. It is not tuned.
* Public aggregated depth does not identify individual orders/participants.
  "Penetrated" means execution necessarily exceeded the entire pre-refill
  residual at >=1 refilled price under the conservative queue accounting.

FULL PRE-OUTCOME GATE — FROZEN BEFORE OUTCOMES
-----------------------------------------------
1:1 matching without replacement.
Exact block: candidate_date + direction.
Continuous controls:
  1. 3s directional momentum
  2. log1p(3s directional aggressive quote)
  3. 3s opposing initial destruction
  4. log1p(anchor-relative 3s opposing refill amount)
  5. log1p(anchor-relative 3s refill breadth)
Frozen calipers/repair from Phase 7M.6B:
  |z difference| <= 0.75 per control
  Euclidean z distance <= 1.00
  repair until every |SMD| <= 0.20
Mandatory actual full-data support:
  >= 250 matched pairs overall
  >= 100 discovery pairs
  >= 100 validation pairs
  binary prevalence within 10%..90%

If any mandatory full-data gate fails, this phase reports NOT TESTED and future
outcomes must remain unopened. No caliper loosening, refill-breadth removal, or
binary-threshold change is permitted to force a pass.

REUSE / COMPUTE PLAN
--------------------
The four Phase 7M.6B pilot dates are reconstructed from their saved signed
candidate_metrics/level_fate/event_tape checkpoints. They do not raw-replay.
Saved 7M.6B tape is long enough for corrected anchor+1s/+3s/+5s. It ends at the
old boundary+10s, so corrected anchor+10s on those reused dates is explicitly
marked unavailable rather than silently shortened. The primary 3s study is
fully reusable. The remaining candidate dates raw-replay with a corrected
candidate+11s capture envelope, which guarantees anchor+10s coverage because
anchors are restricted to the candidate second.

PER-DATE V2 OUTPUTS
-------------------
data/derived/perps/research/prospective_25bp_refill_redepletion_sequence_v2/
  candidate_metrics/candidate_date=YYYY-MM-DD/part.parquet
  level_fate/candidate_date=YYYY-MM-DD/part.parquet
  event_tape/candidate_date=YYYY-MM-DD/part.parquet
  checkpoints/candidate_date=YYYY-MM-DD/manifest.json

All writes are atomic. Valid v2 checkpoints are signature-checked and reused on
restart. Raw-date level/tape rows are streamed candidate-by-candidate to atomic Parquet writers; a full-day level/tape DataFrame is never materialized. Completed candidate depth-event tapes are streamed into one shared uncompressed day spool so high-candidate days do not retain the entire day's Python event graph in memory. Candidate mechanics are maintained incrementally in compact per-level/per-horizon state and therefore do not reread the depth tape to derive metrics.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
except Exception:
    pa = None
    pc = None
    pq = None


BUILDER_VERSION = "1.0.8"
CHECKPOINT_CONTRACT_BUILDER_VERSION = "1.0.0"
COMPATIBLE_CHECKPOINT_BUILDERS = {"1.0.0", "1.0.1", "1.0.2", "1.0.3", "1.0.4", "1.0.5", "1.0.6", "1.0.7", "1.0.8"}
SCHEMA_VERSION = 2
SCHEMA_NAME = "prospective_25bp_refill_redepletion_sequence_v2"
SOURCE_7M6B_SCHEMA = "prospective_25bp_refill_redepletion_sequence_v1"
SOURCE_7M6E_SCHEMA = "prospective_25bp_binary_replacement_penetration_feasibility_v1"
FROZEN_PHASE7M3_SCHEMA = "prospective_25bp_exact_participant_interaction_v1"
FROZEN_POLICY = "EARLY_25BP_RESET5_RELEASE10_REARM60"
EXPECTED_CANDIDATES = 7_067
EXPECTED_DAYS = 62
EXPECTED_HORIZONS = (1, 3, 5, 10)
PRIMARY_HORIZON_SEC = 3
MAX_HORIZON_SEC = 10
PRICE_SCALE = 10.0
DISCOVERY_END_DATE = "2026-06-17"
VALIDATION_START_DATE = "2026-06-18"
EXPECTED_PILOT_DATES = (
    "2026-05-21",
    "2026-06-09",
    "2026-06-30",
    "2026-07-16",
)

# Frozen 7M.6B / 7M.6E matching and support contract.
MATCH_Z_CALIPER = 0.75
MATCH_EUCLIDEAN_CALIPER = 1.00
BALANCE_SMD_GATE = 0.20
MIN_FULL_PRIMARY_PAIRS = 250
MIN_SPLIT_PAIRS = 100
MIN_FEATURE_POSITIVE_SHARE = 0.10
MAX_FEATURE_POSITIVE_SHARE = 0.90
MIN_ANCHOR_MATCH_RATE = 0.98
MIN_SEQUENCE_VALID_RATE = 0.95
MIN_STRICT_TRADE_TIME_COVERAGE = 0.95

# Operational-only streaming bounds. These do not change replay semantics.
DEPTH_STREAM_BATCH_SIZE_7M6F = 8192
SNAPSHOT_STREAM_BATCH_SIZE_7M6F = 64
RAW_REPLAY_HEARTBEAT_SEC = 30.0
PARQUET_ROW_WRITE_CHUNK_7M6F = 2048

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()
DEFAULT_M3_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / FROZEN_PHASE7M3_SCHEMA
DEFAULT_7M6B_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / SOURCE_7M6B_SCHEMA
DEFAULT_7M6E_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / SOURCE_7M6E_SCHEMA
DEFAULT_CANONICAL_ROOT = REPO_ROOT / "data" / "canonical" / "market_events_v1"
DEFAULT_7M6B_SCRIPT = REPO_ROOT / "scripts" / "perps" / "research" / "07m6b_build_refill_redepletion_sequence_pilot.py"
DEFAULT_7M6E_SCRIPT = REPO_ROOT / "scripts" / "perps" / "research" / "07m6e_audit_binary_replacement_penetration_feasibility.py"
DEFAULT_PHASE6A_SCRIPT = REPO_ROOT / "scripts" / "perps" / "research" / "06a_build_execution_foundation.py"
DEFAULT_PHASE7E_SCRIPT = REPO_ROOT / "scripts" / "perps" / "research" / "07e_build_raw_depth_continuation_mechanics.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / SCHEMA_NAME
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "perps_prospective_25bp_refill_redepletion_sequence_v2"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def require_stack() -> None:
    if pa is None or pq is None:
        raise SystemExit("Phase 7M.6F requires pandas, numpy and pyarrow in the project .venv.")


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


def load_module(path: Path, name: str):
    if not path.exists():
        raise RuntimeError(f"Missing required repository script: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7m6f-", suffix=path.suffix or ".tmp", dir=str(path.parent))
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
    fd, tmp = tempfile.mkstemp(prefix="tmp7m6f-", suffix=".csv", dir=str(path.parent))
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
    fd, tmp = tempfile.mkstemp(prefix="tmp7m6f-", suffix=".parquet", dir=str(path.parent))
    os.close(fd)
    try:
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp, compression="zstd")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def level_fate_arrow_schema() -> Any:
    require_stack()
    return pa.schema([
        ("builder_version", pa.string()), ("schema_version", pa.int64()),
        ("candidate_id", pa.string()), ("candidate_date", pa.string()), ("direction", pa.int64()),
        ("observation_horizon_sec", pa.int64()), ("timing_basis", pa.string()),
        ("anchor_ts_recv_ns", pa.int64()), ("anchor_relative_endpoint_ts_recv_ns", pa.int64()),
        ("price_tick", pa.int64()), ("price", pa.float64()), ("initial_qty", pa.float64()),
        ("min_qty", pa.float64()), ("final_observed_qty", pa.float64()),
        ("depleted_before_refill", pa.bool_()), ("first_refill_seen", pa.bool_()),
        ("first_refill_ts_recv_ns", pa.int64()), ("first_refill_transaction_time_ms", pa.float64()),
        ("first_refill_exchange_event_ms", pa.float64()), ("pre_refill_residual_qty", pa.float64()),
        ("observed_refill_qty", pa.float64()), ("observed_refill_quote", pa.float64()),
        ("refill_update_count", pa.int64()), ("post_refill_gross_queue_reduction_qty", pa.float64()),
        ("post_refill_decrease_update_count", pa.int64()),
        ("strict_post_refill_aggressive_trade_qty", pa.float64()),
        ("strict_post_refill_aggressive_trade_quote", pa.float64()),
        ("same_exchange_ms_post_refill_aggressive_trade_qty", pa.float64()),
        ("same_exchange_ms_post_refill_aggressive_trade_quote", pa.float64()),
        ("receive_after_but_exchange_before_trade_qty", pa.float64()),
        ("strict_post_refill_aggressive_trade_count", pa.int64()),
        ("strict_trade_time_complete", pa.bool_()),
        ("minimum_nontrade_queue_reduction_qty", pa.float64()),
        ("minimum_nontrade_queue_reduction_quote", pa.float64()),
        ("replacement_execution_lower_bound_eligible", pa.bool_()),
        ("replacement_execution_lower_bound_qty", pa.float64()),
        ("replacement_execution_lower_bound_quote", pa.float64()),
        ("replacement_execution_lower_bound_share_of_refill", pa.float64()),
        ("replacement_execution_lower_bound_positive", pa.bool_()),
        ("anchor_found", pa.bool_()), ("sequence_valid", pa.bool_()), ("invalid_reason", pa.string()),
    ])


def event_tape_arrow_schema() -> Any:
    require_stack()
    # Union of raw DEPTH_CHANGE and AGGRESSIVE_TRADE_AT_REFILLED_LEVEL fields.
    return pa.schema([
        ("candidate_id", pa.string()), ("candidate_date", pa.string()), ("direction", pa.int64()),
        ("observation_horizon_sec", pa.float64()), ("event_kind", pa.string()),
        ("ts_recv_ns", pa.int64()), ("relative_to_anchor_ms", pa.float64()),
        ("transaction_time_ms", pa.float64()), ("exchange_event_ms", pa.float64()),
        ("first_update_id", pa.int64()), ("final_update_id", pa.int64()),
        ("previous_final_update_id", pa.int64()), ("book_side", pa.string()),
        ("price_tick", pa.int64()), ("price", pa.float64()),
        ("old_qty", pa.float64()), ("new_qty", pa.float64()), ("delta_qty", pa.float64()),
        ("quantity", pa.float64()), ("buyer_is_maker", pa.bool_()),
        ("trade_time_ms", pa.float64()), ("timing_class", pa.string()),
    ])


class AtomicParquetRowWriter:
    """Stream small row batches to a temp Parquet file and atomically commit it."""
    def __init__(self, path: Path, schema: Any):
        require_stack()
        self.path = path
        self.schema = schema
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix="tmp7m6f-stream-", suffix=".parquet", dir=str(self.path.parent))
        os.close(fd)
        self.tmp = Path(tmp)
        self.writer = pq.ParquetWriter(str(self.tmp), self.schema, compression="zstd")
        self.row_count = 0
        self.committed = False

    def write_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            return
        # Bound transient Arrow/Python duplication even for a very busy candidate.
        # Scientific row content/order is unchanged; only physical write batching differs.
        n = len(rows)
        for start in range(0, n, PARQUET_ROW_WRITE_CHUNK_7M6F):
            chunk = rows[start : start + PARQUET_ROW_WRITE_CHUNK_7M6F]
            table = pa.Table.from_pylist(chunk, schema=self.schema)
            self.writer.write_table(table)
            self.row_count += int(table.num_rows)
            del table, chunk

    def write_table(self, table: Any) -> None:
        if table is None or int(table.num_rows) == 0:
            return
        if table.schema != self.schema:
            table = table.cast(self.schema, safe=False)
        self.writer.write_table(table)
        self.row_count += int(table.num_rows)

    def commit(self) -> None:
        if self.committed:
            return
        self.writer.close()
        self.writer = None
        os.replace(self.tmp, self.path)
        self.committed = True

    def abort(self) -> None:
        try:
            if self.writer is not None:
                self.writer.close()
        finally:
            self.writer = None
            if self.tmp.exists():
                self.tmp.unlink()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_signature(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(paths, key=lambda x: str(x)):
        st = p.stat()
        rows.append({"path": str(p), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)})
    return rows


def contract_payload() -> dict[str, Any]:
    return {
        "builder_version": CHECKPOINT_CONTRACT_BUILDER_VERSION,
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "policy": FROZEN_POLICY,
        "candidate_count": EXPECTED_CANDIDATES,
        "horizons_sec": EXPECTED_HORIZONS,
        "primary_horizon_sec": PRIMARY_HORIZON_SEC,
        "timing_basis": "recovered exact frozen-book anchor + horizon",
        "anchor_search": "within frozen candidate second only",
        "raw_capture_envelope": "candidate_second through candidate_second+11s",
        "refill_definition": "positive absolute-quantity update at same frozen opposing price after prior depletion below anchor qty",
        "strict_post_refill_trade": "same price + correct aggressor + receive after refill + trade_time_ms strictly after refill transaction_time_ms",
        "same_ms_policy": "same exchange millisecond excluded from strict; retained as ambiguity",
        "replacement_execution_lower_bound": "min(refill_qty,max(0,strict_post_refill_aggressive_trade_qty-pre_refill_residual_qty))",
        "primary_binary_exposure": "opposing__replacement_execution_lower_bound_quote > 0 at anchor+3s",
        "binary_threshold": "physical zero boundary only; not tuned",
        "order_identity_claim": False,
        "future_outcomes_loaded": False,
        "matching_exact": ["candidate_date", "direction"],
        "matching_controls": ["momentum", "aggression", "initial_destruction", "anchor_relative_refill_amount", "refill_breadth"],
        "matching_z_caliper": MATCH_Z_CALIPER,
        "matching_euclidean_caliper": MATCH_EUCLIDEAN_CALIPER,
        "balance_smd_gate": BALANCE_SMD_GATE,
        "full_pair_gate": MIN_FULL_PRIMARY_PAIRS,
        "split_pair_gate": MIN_SPLIT_PAIRS,
    }


def checkpoint_paths(root: Path, day: str) -> dict[str, Path]:
    return {
        "metrics": root / "candidate_metrics" / f"candidate_date={day}" / "part.parquet",
        "levels": root / "level_fate" / f"candidate_date={day}" / "part.parquet",
        "tape": root / "event_tape" / f"candidate_date={day}" / "part.parquet",
        "manifest": root / "checkpoints" / f"candidate_date={day}" / "manifest.json",
    }


def source_7m6b_paths(root: Path, day: str) -> dict[str, Path]:
    return checkpoint_paths(root, day)


def verify_7m6e_authorization(root: Path) -> tuple[dict[str, Any], list[str]]:
    p = root / "metadata" / "latest_summary.json"
    if not p.exists():
        raise RuntimeError(f"Missing Phase 7M.6E precompute authorization summary: {p}")
    m = json.loads(p.read_text(encoding="utf-8"))
    if str(m.get("status")) != "BINARY_PENETRATION_FEASIBLE_FOR_FULL_REPLAY":
        raise RuntimeError(f"Phase 7M.6E did not authorize full replay: status={m.get('status')!r}")
    if not bool(m.get("full_replay_authorized_by_this_gate")):
        raise RuntimeError("Phase 7M.6E summary does not set full_replay_authorized_by_this_gate=true.")
    if bool(m.get("future_outcomes_loaded")) or bool(m.get("raw_market_replay_performed")):
        raise RuntimeError("Phase 7M.6E authorization is not the expected outcome-blind checkpoint-only gate.")
    days = sorted(map(str, m.get("processed_dates", [])))
    if tuple(days) != tuple(sorted(EXPECTED_PILOT_DATES)):
        raise RuntimeError(f"Unexpected Phase 7M.6E pilot dates: {days}")
    return m, days


def v2_signature_raw(day: str, ids: Sequence[str], level_path: Path, day_files: Any) -> str:
    payload = {
        "mode": "RAW_CORRECTED_ANCHOR_RELATIVE",
        "day": day,
        "candidate_ids": sorted(map(str, ids)),
        "level_audit": file_signature([level_path]),
        "depth": file_signature(day_files.depth),
        "snapshots": file_signature(day_files.snapshots),
        "trades": file_signature(day_files.trades),
        "contract": contract_payload(),
    }
    return sha256_text(json.dumps(payload, sort_keys=True, default=str))


def v2_signature_pilot_reuse(day: str, ids: Sequence[str], source_paths: dict[str, Path]) -> str:
    for k in ("metrics", "levels", "tape", "manifest"):
        if not source_paths[k].exists():
            raise RuntimeError(f"Missing reusable Phase 7M.6B pilot checkpoint {k}: {source_paths[k]}")
    payload = {
        "mode": "REUSE_7M6B_SAVED_EVENT_TAPE",
        "day": day,
        "candidate_ids": sorted(map(str, ids)),
        "source_checkpoint_files": file_signature([source_paths["metrics"], source_paths["levels"], source_paths["tape"], source_paths["manifest"]]),
        "contract": contract_payload(),
    }
    return sha256_text(json.dumps(payload, sort_keys=True, default=str))


def can_reuse_v2(paths: dict[str, Path], signature: str) -> bool:
    if not all(paths[k].exists() for k in ("metrics", "levels", "tape", "manifest")):
        return False
    try:
        m = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        str(m.get("status")) == "PASS"
        and str(m.get("builder_version")) in COMPATIBLE_CHECKPOINT_BUILDERS
        and str(m.get("schema_name")) == SCHEMA_NAME
        and str(m.get("input_signature")) == signature
        and bool(m.get("future_outcomes_loaded")) is False
    )


def unavailable_metric(base: Mapping[str, Any], h: int, reason: str, *, timing_basis: str, horizon_complete: bool = False) -> dict[str, Any]:
    return {
        "builder_version": BUILDER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "candidate_id": str(base.get("candidate_id", "")),
        "candidate_date": str(base.get("candidate_date", "")),
        "direction": int(base.get("direction", 0)),
        "candidate_second": int(base.get("candidate_second", 0)),
        "observation_horizon_sec": int(h),
        "timing_basis": timing_basis,
        "anchor_found": bool(base.get("anchor_found", False)),
        "anchor_ts_recv_ns": safe_float(base.get("anchor_ts_recv_ns")),
        "anchor_offset_ms": safe_float(base.get("anchor_offset_ms")),
        "anchor_relative_endpoint_ts_recv_ns": math.nan,
        "horizon_complete": bool(horizon_complete),
        "sequence_valid": bool(base.get("sequence_valid", False)),
        "invalid_reason": reason,
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
        "replacement_penetration_eligible": False,
        "replacement_penetrated": math.nan,
        "control__directional_mid_move_bps_3s": safe_float(base.get("control__directional_mid_move_bps_3s", base.get("directional_mid_move_bps"))),
        "control__directional_aggressive_quote_3s": safe_float(base.get("control__directional_aggressive_quote_3s", base.get("directional_aggressive_quote"))),
        "control__opposing_peak_initial_queue_loss_share_3s": safe_float(base.get("control__opposing_peak_initial_queue_loss_share_3s", base.get("opposing__peak_initial_queue_loss_share"))),
        "control__opposing_refill_quote_3s_frozen": safe_float(base.get("control__opposing_refill_quote_3s_frozen", base.get("opposing__same_price_refill_after_depletion_quote"))),
        "is_reaccel_context": bool(base.get("is_reaccel_context", False)),
    }


def build_horizon_from_events(
    *,
    base: Mapping[str, Any],
    opposing_levels: pd.DataFrame,
    depth_events: pd.DataFrame,
    trades: pd.DataFrame,
    h: int,
    trades_need_direction_filter: bool,
    timing_basis: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build one anchor-relative horizon from already-captured causal events."""
    cid = str(base.get("candidate_id", ""))
    day = str(base.get("candidate_date", ""))
    direction = int(base.get("direction", 0))
    cs = int(base.get("candidate_second", 0))
    anchor_found = bool(base.get("anchor_found", False))
    seq_valid = bool(base.get("sequence_valid", False))
    anchor_raw = safe_float(base.get("anchor_ts_recv_ns"))
    if not anchor_found or not math.isfinite(anchor_raw):
        reason = str(base.get("invalid_reason", "")) or "ANCHOR_UNAVAILABLE"
        return unavailable_metric(base, h, reason, timing_basis=timing_basis), [], []

    anchor_ns = int(anchor_raw)
    endpoint_ns = anchor_ns + int(h) * 1_000_000_000
    depth = depth_events.copy()
    if len(depth):
        depth["ts_recv_ns"] = pd.to_numeric(depth["ts_recv_ns"], errors="coerce")
        depth = depth[(depth["ts_recv_ns"] > anchor_ns) & (depth["ts_recv_ns"] < endpoint_ns)].copy()

    tg_all = trades.copy()
    if len(tg_all):
        tg_all["ts_recv_ns"] = pd.to_numeric(tg_all["ts_recv_ns"], errors="coerce")
        tg_all["trade_time_ms"] = pd.to_numeric(tg_all.get("trade_time_ms"), errors="coerce")
        tg_all["quantity"] = pd.to_numeric(tg_all.get("quantity"), errors="coerce")

    level_rows: list[dict[str, Any]] = []
    tape_trade_rows: list[dict[str, Any]] = []
    for lr in opposing_levels.itertuples(index=False):
        tick = int(lr.price_tick)
        price = safe_float(lr.price)
        initial = safe_float(lr.initial_qty)
        if not (math.isfinite(price) and price > 0 and math.isfinite(initial) and initial > 0):
            continue
        le = depth[pd.to_numeric(depth["price_tick"], errors="coerce") == tick].copy() if len(depth) else pd.DataFrame()
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
                    first_refill_tx = safe_float(getattr(er, "transaction_time_ms", math.nan))
                    first_refill_exchange = safe_float(getattr(er, "exchange_event_ms", math.nan))
                    pre_refill_residual = oldq

        strict_trade_qty = 0.0
        same_ms_trade_qty = 0.0
        before_trade_qty = 0.0
        strict_trade_count = 0
        trade_time_complete = True
        if first_refill_seen and first_refill_recv is not None:
            tg = tg_all[pd.to_numeric(tg_all["price_tick"], errors="coerce") == tick].copy() if len(tg_all) else pd.DataFrame()
            if len(tg):
                tg = tg[(tg["ts_recv_ns"] > first_refill_recv) & (tg["ts_recv_ns"] < endpoint_ns)].copy()
                if trades_need_direction_filter and len(tg):
                    maker = bool_series(tg["buyer_is_maker"])
                    correct = (~maker) if direction > 0 else maker
                    tg = tg[correct].copy()
                if len(tg):
                    tt = pd.to_numeric(tg["trade_time_ms"], errors="coerce")
                    trade_time_complete = bool(tt.notna().all() and math.isfinite(first_refill_tx))
                    if math.isfinite(first_refill_tx):
                        strict = tg[tt > first_refill_tx].copy()
                        same = tg[tt == first_refill_tx].copy()
                        before = tg[tt < first_refill_tx].copy()
                        strict_trade_qty = float(pd.to_numeric(strict["quantity"], errors="coerce").fillna(0).sum())
                        same_ms_trade_qty = float(pd.to_numeric(same["quantity"], errors="coerce").fillna(0).sum())
                        before_trade_qty = float(pd.to_numeric(before["quantity"], errors="coerce").fillna(0).sum())
                        strict_trade_count = int(len(strict))
                        for timing_class, frame in (("STRICT_AFTER", strict), ("SAME_EXCHANGE_MS", same), ("EXCHANGE_BEFORE_RECEIVE_AFTER", before)):
                            for rr in frame.itertuples(index=False):
                                tape_trade_rows.append({
                                    "candidate_id": cid,
                                    "candidate_date": day,
                                    "direction": direction,
                                    "observation_horizon_sec": int(h),
                                    "event_kind": "AGGRESSIVE_TRADE_AT_REFILLED_LEVEL",
                                    "ts_recv_ns": int(rr.ts_recv_ns),
                                    "relative_to_anchor_ms": (int(rr.ts_recv_ns) - anchor_ns) / 1e6,
                                    "price_tick": tick,
                                    "price": safe_float(rr.price),
                                    "quantity": safe_float(rr.quantity),
                                    "buyer_is_maker": bool(rr.buyer_is_maker),
                                    "trade_time_ms": safe_float(rr.trade_time_ms),
                                    "exchange_event_ms": safe_float(getattr(rr, "exchange_event_ms", math.nan)),
                                    "timing_class": timing_class,
                                })
                    else:
                        trade_time_complete = False

        refill_quote = refill_qty * price
        strict_trade_quote = strict_trade_qty * price
        same_ms_quote = same_ms_trade_qty * price
        nontrade_lb_qty = max(0.0, gross_post_refill_decrease - strict_trade_qty)
        nontrade_lb_quote = nontrade_lb_qty * price
        lb_eligible = bool(
            anchor_found
            and seq_valid
            and first_refill_seen
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
            "observation_horizon_sec": int(h),
            "timing_basis": timing_basis,
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
            "anchor_found": anchor_found,
            "sequence_valid": seq_valid,
            "invalid_reason": str(base.get("invalid_reason", "")),
        })

    lm = pd.DataFrame(level_rows)
    if lm.empty:
        return unavailable_metric(base, h, "NO_OPPOSING_FROZEN_LEVEL_ROWS_RECONSTRUCTED", timing_basis=timing_basis), level_rows, tape_trade_rows

    refilled = lm[bool_series(lm["first_refill_seen"]) & (pd.to_numeric(lm["observed_refill_quote"], errors="coerce") > 0)].copy()
    refill_quote = float(pd.to_numeric(refilled.get("observed_refill_quote", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    strict_trade_quote = float(pd.to_numeric(refilled.get("strict_post_refill_aggressive_trade_quote", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    same_ms_quote = float(pd.to_numeric(refilled.get("same_exchange_ms_post_refill_aggressive_trade_quote", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    nontrade_quote = float(pd.to_numeric(refilled.get("minimum_nontrade_queue_reduction_quote", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    n_refilled = int(len(refilled))
    n_traded = int((pd.to_numeric(refilled.get("strict_post_refill_aggressive_trade_quote", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0).sum()) if n_refilled else 0
    time_share = float(bool_series(refilled["strict_trade_time_complete"]).mean()) if n_refilled else math.nan
    eligible_mask = bool_series(refilled["replacement_execution_lower_bound_eligible"]) if n_refilled else pd.Series(dtype=bool)
    fully_attributable = bool(n_refilled > 0 and int(eligible_mask.sum()) == n_refilled)
    if fully_attributable:
        lb_quote = float(pd.to_numeric(refilled["replacement_execution_lower_bound_quote"], errors="coerce").fillna(0).sum())
        lb_positive_levels = int(bool_series(refilled["replacement_execution_lower_bound_positive"]).sum())
        lb_ratio = lb_quote / refill_quote if refill_quote > 0 else math.nan
        lb_level_share = lb_positive_levels / n_refilled if n_refilled else math.nan
        penetration_eligible = True
        penetrated = int(lb_quote > 1e-12)
    else:
        lb_quote = math.nan
        lb_ratio = math.nan
        lb_level_share = math.nan
        penetration_eligible = False
        penetrated = math.nan

    metric = {
        "builder_version": BUILDER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "candidate_id": cid,
        "candidate_date": day,
        "direction": direction,
        "candidate_second": cs,
        "observation_horizon_sec": int(h),
        "timing_basis": timing_basis,
        "anchor_found": anchor_found,
        "anchor_ts_recv_ns": anchor_ns,
        "anchor_offset_ms": safe_float(base.get("anchor_offset_ms")),
        "anchor_relative_endpoint_ts_recv_ns": endpoint_ns,
        "horizon_complete": True,
        "sequence_valid": seq_valid,
        "invalid_reason": str(base.get("invalid_reason", "")),
        "opposing__refilled_level_count": n_refilled,
        "opposing__observed_refill_quote": refill_quote,
        "opposing__strict_post_refill_aggressive_trade_quote": strict_trade_quote,
        "opposing__same_exchange_ms_post_refill_aggressive_trade_quote": same_ms_quote,
        "opposing__post_refill_aggressive_trade_to_refill_ratio": strict_trade_quote / refill_quote if refill_quote > 0 else math.nan,
        "opposing__refilled_level_post_trade_share": n_traded / n_refilled if n_refilled else math.nan,
        "opposing__minimum_nontrade_queue_reduction_quote": nontrade_quote,
        "opposing__minimum_nontrade_reduction_to_refill_ratio": nontrade_quote / refill_quote if refill_quote > 0 else math.nan,
        "opposing__strict_trade_time_complete_share": time_share,
        "opposing__replacement_execution_lower_bound_quote": lb_quote,
        "opposing__replacement_execution_lower_bound_to_refill_ratio": lb_ratio,
        "opposing__refilled_level_replacement_execution_lower_bound_share": lb_level_share,
        "replacement_penetration_eligible": penetration_eligible,
        "replacement_penetrated": penetrated,
        "control__directional_mid_move_bps_3s": safe_float(base.get("control__directional_mid_move_bps_3s", base.get("directional_mid_move_bps"))),
        "control__directional_aggressive_quote_3s": safe_float(base.get("control__directional_aggressive_quote_3s", base.get("directional_aggressive_quote"))),
        "control__opposing_peak_initial_queue_loss_share_3s": safe_float(base.get("control__opposing_peak_initial_queue_loss_share_3s", base.get("opposing__peak_initial_queue_loss_share"))),
        "control__opposing_refill_quote_3s_frozen": safe_float(base.get("control__opposing_refill_quote_3s_frozen", base.get("opposing__same_price_refill_after_depletion_quote"))),
        "is_reaccel_context": bool(base.get("is_reaccel_context", False)),
    }
    return metric, level_rows, tape_trade_rows



def init_compact_depth_states(opposing_levels: pd.DataFrame) -> dict[int, dict[int, dict[str, Any]]]:
    """Initialize exact per-level state for each frozen anchor-relative horizon.

    This is a lossless streaming replacement for replaying the saved depth-event
    tape after the day finishes.  Each horizon keeps exactly the sufficient
    statistics consumed by ``build_horizon_from_events``.
    """
    out: dict[int, dict[int, dict[str, Any]]] = {int(h): {} for h in EXPECTED_HORIZONS}
    for lr in opposing_levels.itertuples(index=False):
        tick = int(lr.price_tick)
        price = safe_float(lr.price)
        initial = safe_float(lr.initial_qty)
        if not (math.isfinite(price) and price > 0 and math.isfinite(initial) and initial > 0):
            continue
        for h in EXPECTED_HORIZONS:
            out[int(h)][tick] = {
                "price_tick": tick,
                "price": price,
                "initial_qty": initial,
                "current_qty": initial,
                "min_qty": initial,
                "depleted": False,
                "first_refill_seen": False,
                "first_refill_ts_recv_ns": None,
                "first_refill_transaction_time_ms": math.nan,
                "first_refill_exchange_event_ms": math.nan,
                "pre_refill_residual_qty": math.nan,
                "observed_refill_qty": 0.0,
                "refill_update_count": 0,
                "post_refill_gross_queue_reduction_qty": 0.0,
                "post_refill_decrease_update_count": 0,
            }
    return out


def update_compact_depth_states(
    tracker: Any,
    *,
    ts_ns: int,
    side_name: str,
    tick: int,
    old_qty: float,
    new_qty: float,
    transaction_time_ms: Any,
    exchange_event_ms: Any,
) -> None:
    """Apply one captured opposing-level depth change to every open horizon."""
    if not bool(getattr(tracker, "anchor_found", False)):
        return
    anchor_raw = getattr(tracker, "anchor_ts_recv_ns", None)
    if anchor_raw is None:
        return
    anchor_ns = int(anchor_raw)
    if int(ts_ns) <= anchor_ns or side_name != tracker.expected_side:
        return
    states = getattr(tracker, "_compact_depth_states", None)
    if not states:
        return
    tick_i = int(tick)
    oq = safe_float(old_qty)
    nq = safe_float(new_qty)
    if not (math.isfinite(oq) and math.isfinite(nq)):
        return
    delta = nq - oq
    txn = safe_float(transaction_time_ms)
    exch = safe_float(exchange_event_ms)
    for h in EXPECTED_HORIZONS:
        if int(ts_ns) >= anchor_ns + int(h) * 1_000_000_000:
            continue
        st = states[int(h)].get(tick_i)
        if st is None:
            continue
        st["current_qty"] = nq
        if delta < -1e-12:
            st["min_qty"] = min(float(st["min_qty"]), nq)
            st["depleted"] = True
            if bool(st["first_refill_seen"]):
                st["post_refill_gross_queue_reduction_qty"] = float(st["post_refill_gross_queue_reduction_qty"]) + (-delta)
                st["post_refill_decrease_update_count"] = int(st["post_refill_decrease_update_count"]) + 1
        elif delta > 1e-12 and bool(st["depleted"]):
            st["observed_refill_qty"] = float(st["observed_refill_qty"]) + delta
            st["refill_update_count"] = int(st["refill_update_count"]) + 1
            if not bool(st["first_refill_seen"]):
                st["first_refill_seen"] = True
                st["first_refill_ts_recv_ns"] = int(ts_ns)
                st["first_refill_transaction_time_ms"] = txn
                st["first_refill_exchange_event_ms"] = exch
                st["pre_refill_residual_qty"] = oq


def compact_refill_ticks(tracker: Any) -> set[int]:
    """Return exact max-horizon refill ticks from compact state."""
    if not bool(getattr(tracker, "anchor_found", False)) or not bool(getattr(tracker, "sequence_valid", False)):
        return set()
    states = getattr(tracker, "_compact_depth_states", {}) or {}
    hstates = states.get(int(MAX_HORIZON_SEC), {})
    return {
        int(tick)
        for tick, st in hstates.items()
        if bool(st.get("first_refill_seen", False)) and float(st.get("observed_refill_qty", 0.0) or 0.0) > 0.0
    }


def build_horizon_from_compact(
    *,
    base: Mapping[str, Any],
    opposing_levels: pd.DataFrame,
    compact_states: Mapping[int, Mapping[int, Mapping[str, Any]]],
    trades_by_tick: Mapping[int, pd.DataFrame],
    h: int,
    trades_need_direction_filter: bool,
    timing_basis: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build one horizon from incrementally maintained exact depth state.

    The returned scientific fields are intentionally identical to
    ``build_horizon_from_events``; only the computational path differs.
    """
    cid = str(base.get("candidate_id", ""))
    day = str(base.get("candidate_date", ""))
    direction = int(base.get("direction", 0))
    cs = int(base.get("candidate_second", 0))
    anchor_found = bool(base.get("anchor_found", False))
    seq_valid = bool(base.get("sequence_valid", False))
    anchor_raw = safe_float(base.get("anchor_ts_recv_ns"))
    if not anchor_found or not math.isfinite(anchor_raw):
        reason = str(base.get("invalid_reason", "")) or "ANCHOR_UNAVAILABLE"
        return unavailable_metric(base, h, reason, timing_basis=timing_basis), [], []

    anchor_ns = int(anchor_raw)
    endpoint_ns = anchor_ns + int(h) * 1_000_000_000
    hstates = compact_states.get(int(h), {})
    level_rows: list[dict[str, Any]] = []
    tape_trade_rows: list[dict[str, Any]] = []

    for lr in opposing_levels.itertuples(index=False):
        tick = int(lr.price_tick)
        price = safe_float(lr.price)
        initial = safe_float(lr.initial_qty)
        if not (math.isfinite(price) and price > 0 and math.isfinite(initial) and initial > 0):
            continue
        st = hstates.get(tick)
        if st is None:
            current = initial
            minq = initial
            depleted = False
            first_refill_seen = False
            first_refill_recv = None
            first_refill_tx = math.nan
            first_refill_exchange = math.nan
            pre_refill_residual = math.nan
            refill_qty = 0.0
            refill_updates = 0
            gross_post_refill_decrease = 0.0
            post_refill_decrease_updates = 0
        else:
            current = safe_float(st.get("current_qty", initial))
            minq = safe_float(st.get("min_qty", initial))
            depleted = bool(st.get("depleted", False))
            first_refill_seen = bool(st.get("first_refill_seen", False))
            first_refill_recv = st.get("first_refill_ts_recv_ns")
            first_refill_recv = int(first_refill_recv) if first_refill_recv is not None else None
            first_refill_tx = safe_float(st.get("first_refill_transaction_time_ms"))
            first_refill_exchange = safe_float(st.get("first_refill_exchange_event_ms"))
            pre_refill_residual = safe_float(st.get("pre_refill_residual_qty"))
            refill_qty = safe_float(st.get("observed_refill_qty", 0.0))
            refill_updates = int(st.get("refill_update_count", 0) or 0)
            gross_post_refill_decrease = safe_float(st.get("post_refill_gross_queue_reduction_qty", 0.0))
            post_refill_decrease_updates = int(st.get("post_refill_decrease_update_count", 0) or 0)

        strict_trade_qty = 0.0
        same_ms_trade_qty = 0.0
        before_trade_qty = 0.0
        strict_trade_count = 0
        trade_time_complete = True
        if first_refill_seen and first_refill_recv is not None:
            tg0 = trades_by_tick.get(tick)
            if tg0 is not None and len(tg0):
                tg = tg0[(pd.to_numeric(tg0["ts_recv_ns"], errors="coerce") > first_refill_recv) &
                         (pd.to_numeric(tg0["ts_recv_ns"], errors="coerce") < endpoint_ns)].copy()
                if trades_need_direction_filter and len(tg):
                    maker = bool_series(tg["buyer_is_maker"])
                    correct = (~maker) if direction > 0 else maker
                    tg = tg[correct].copy()
                if len(tg):
                    tt = pd.to_numeric(tg["trade_time_ms"], errors="coerce")
                    trade_time_complete = bool(tt.notna().all() and math.isfinite(first_refill_tx))
                    if math.isfinite(first_refill_tx):
                        strict = tg[tt > first_refill_tx].copy()
                        same = tg[tt == first_refill_tx].copy()
                        before = tg[tt < first_refill_tx].copy()
                        strict_trade_qty = float(pd.to_numeric(strict["quantity"], errors="coerce").fillna(0).sum())
                        same_ms_trade_qty = float(pd.to_numeric(same["quantity"], errors="coerce").fillna(0).sum())
                        before_trade_qty = float(pd.to_numeric(before["quantity"], errors="coerce").fillna(0).sum())
                        strict_trade_count = int(len(strict))
                        for timing_class, frame in (("STRICT_AFTER", strict), ("SAME_EXCHANGE_MS", same), ("EXCHANGE_BEFORE_RECEIVE_AFTER", before)):
                            for rr in frame.itertuples(index=False):
                                tape_trade_rows.append({
                                    "candidate_id": cid,
                                    "candidate_date": day,
                                    "direction": direction,
                                    "observation_horizon_sec": int(h),
                                    "event_kind": "AGGRESSIVE_TRADE_AT_REFILLED_LEVEL",
                                    "ts_recv_ns": int(rr.ts_recv_ns),
                                    "relative_to_anchor_ms": (int(rr.ts_recv_ns) - anchor_ns) / 1e6,
                                    "price_tick": tick,
                                    "price": safe_float(rr.price),
                                    "quantity": safe_float(rr.quantity),
                                    "buyer_is_maker": bool(rr.buyer_is_maker),
                                    "trade_time_ms": safe_float(rr.trade_time_ms),
                                    "exchange_event_ms": safe_float(getattr(rr, "exchange_event_ms", math.nan)),
                                    "timing_class": timing_class,
                                })
                    else:
                        trade_time_complete = False

        refill_quote = refill_qty * price
        strict_trade_quote = strict_trade_qty * price
        same_ms_quote = same_ms_trade_qty * price
        nontrade_lb_qty = max(0.0, gross_post_refill_decrease - strict_trade_qty)
        nontrade_lb_quote = nontrade_lb_qty * price
        lb_eligible = bool(
            anchor_found and seq_valid and first_refill_seen and refill_qty > 0
            and trade_time_complete and math.isfinite(pre_refill_residual)
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
            "observation_horizon_sec": int(h),
            "timing_basis": timing_basis,
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
            "anchor_found": anchor_found,
            "sequence_valid": seq_valid,
            "invalid_reason": str(base.get("invalid_reason", "")),
        })

    lm = pd.DataFrame(level_rows)
    if lm.empty:
        return unavailable_metric(base, h, "NO_OPPOSING_FROZEN_LEVEL_ROWS_RECONSTRUCTED", timing_basis=timing_basis), level_rows, tape_trade_rows

    refilled = lm[bool_series(lm["first_refill_seen"]) & (pd.to_numeric(lm["observed_refill_quote"], errors="coerce") > 0)].copy()
    refill_quote = float(pd.to_numeric(refilled.get("observed_refill_quote", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    strict_trade_quote = float(pd.to_numeric(refilled.get("strict_post_refill_aggressive_trade_quote", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    same_ms_quote = float(pd.to_numeric(refilled.get("same_exchange_ms_post_refill_aggressive_trade_quote", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    nontrade_quote = float(pd.to_numeric(refilled.get("minimum_nontrade_queue_reduction_quote", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    n_refilled = int(len(refilled))
    n_traded = int((pd.to_numeric(refilled.get("strict_post_refill_aggressive_trade_quote", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0).sum()) if n_refilled else 0
    time_share = float(bool_series(refilled["strict_trade_time_complete"]).mean()) if n_refilled else math.nan
    eligible_mask = bool_series(refilled["replacement_execution_lower_bound_eligible"]) if n_refilled else pd.Series(dtype=bool)
    fully_attributable = bool(n_refilled > 0 and int(eligible_mask.sum()) == n_refilled)
    if fully_attributable:
        lb_quote = float(pd.to_numeric(refilled["replacement_execution_lower_bound_quote"], errors="coerce").fillna(0).sum())
        lb_positive_levels = int(bool_series(refilled["replacement_execution_lower_bound_positive"]).sum())
        lb_ratio = lb_quote / refill_quote if refill_quote > 0 else math.nan
        lb_level_share = lb_positive_levels / n_refilled if n_refilled else math.nan
        penetration_eligible = True
        penetrated = int(lb_quote > 1e-12)
    else:
        lb_quote = math.nan
        lb_ratio = math.nan
        lb_level_share = math.nan
        penetration_eligible = False
        penetrated = math.nan

    metric = {
        "builder_version": BUILDER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "candidate_id": cid,
        "candidate_date": day,
        "direction": direction,
        "candidate_second": cs,
        "observation_horizon_sec": int(h),
        "timing_basis": timing_basis,
        "anchor_found": anchor_found,
        "anchor_ts_recv_ns": anchor_ns,
        "anchor_offset_ms": safe_float(base.get("anchor_offset_ms")),
        "anchor_relative_endpoint_ts_recv_ns": endpoint_ns,
        "horizon_complete": True,
        "sequence_valid": seq_valid,
        "invalid_reason": str(base.get("invalid_reason", "")),
        "opposing__refilled_level_count": n_refilled,
        "opposing__observed_refill_quote": refill_quote,
        "opposing__strict_post_refill_aggressive_trade_quote": strict_trade_quote,
        "opposing__same_exchange_ms_post_refill_aggressive_trade_quote": same_ms_quote,
        "opposing__post_refill_aggressive_trade_to_refill_ratio": strict_trade_quote / refill_quote if refill_quote > 0 else math.nan,
        "opposing__refilled_level_post_trade_share": n_traded / n_refilled if n_refilled else math.nan,
        "opposing__minimum_nontrade_queue_reduction_quote": nontrade_quote,
        "opposing__minimum_nontrade_reduction_to_refill_ratio": nontrade_quote / refill_quote if refill_quote > 0 else math.nan,
        "opposing__strict_trade_time_complete_share": time_share,
        "opposing__replacement_execution_lower_bound_quote": lb_quote,
        "opposing__replacement_execution_lower_bound_to_refill_ratio": lb_ratio,
        "opposing__refilled_level_replacement_execution_lower_bound_share": lb_level_share,
        "replacement_penetration_eligible": penetration_eligible,
        "replacement_penetrated": penetrated,
        "control__directional_mid_move_bps_3s": safe_float(base.get("control__directional_mid_move_bps_3s", base.get("directional_mid_move_bps"))),
        "control__directional_aggressive_quote_3s": safe_float(base.get("control__directional_aggressive_quote_3s", base.get("directional_aggressive_quote"))),
        "control__opposing_peak_initial_queue_loss_share_3s": safe_float(base.get("control__opposing_peak_initial_queue_loss_share_3s", base.get("opposing__peak_initial_queue_loss_share"))),
        "control__opposing_refill_quote_3s_frozen": safe_float(base.get("control__opposing_refill_quote_3s_frozen", base.get("opposing__same_price_refill_after_depletion_quote"))),
        "is_reaccel_context": bool(base.get("is_reaccel_context", False)),
    }
    return metric, level_rows, tape_trade_rows

def reconstruct_pilot_day_from_7m6b(
    *, day: str, day_clock: pd.DataFrame, source_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    sp = source_7m6b_paths(source_root, day)
    source_manifest = json.loads(sp["manifest"].read_text(encoding="utf-8"))
    if str(source_manifest.get("status")) != "PASS" or bool(source_manifest.get("future_outcomes_loaded")):
        raise RuntimeError(f"Phase 7M.6B source checkpoint is not reusable for {day}")

    metrics = pq.read_table(sp["metrics"]).to_pandas()
    levels = pq.read_table(sp["levels"]).to_pandas()
    tape = pq.read_table(sp["tape"]).to_pandas()
    for d in (metrics, levels, tape):
        if "candidate_id" in d.columns:
            d["candidate_id"] = d["candidate_id"].astype(str)

    m3 = metrics[pd.to_numeric(metrics["observation_horizon_sec"], errors="coerce") == PRIMARY_HORIZON_SEC].copy()
    l10 = levels[pd.to_numeric(levels["observation_horizon_sec"], errors="coerce") == MAX_HORIZON_SEC].copy()
    m3_by_id = m3.set_index("candidate_id", drop=False)
    clock_by_id = day_clock.set_index(day_clock["candidate_id"].astype(str), drop=False)

    metric_rows: list[dict[str, Any]] = []
    level_rows: list[dict[str, Any]] = []
    for cid in day_clock["candidate_id"].astype(str):
        c = clock_by_id.loc[cid]
        if cid not in m3_by_id.index:
            base = c.to_dict()
            base.update({"candidate_id": cid, "candidate_date": day, "anchor_found": False, "sequence_valid": False})
            for h in EXPECTED_HORIZONS:
                metric_rows.append(unavailable_metric(base, h, "SOURCE_7M6B_PRIMARY_METRIC_MISSING", timing_basis="REUSED_7M6B_EVENT_TAPE_ANCHOR_RELATIVE"))
            continue
        base = m3_by_id.loc[cid]
        if isinstance(base, pd.DataFrame):
            raise RuntimeError(f"Duplicate 7M.6B primary metric for candidate {cid}")
        base = base.to_dict()
        lv = l10[l10["candidate_id"] == cid].copy()
        ct = tape[tape["candidate_id"] == cid].copy()
        depth = ct[ct["event_kind"].astype(str) == "DEPTH_CHANGE"].copy() if len(ct) else pd.DataFrame()
        trades = ct[
            (ct["event_kind"].astype(str) == "AGGRESSIVE_TRADE_AT_REFILLED_LEVEL")
            & (pd.to_numeric(ct.get("observation_horizon_sec"), errors="coerce") == MAX_HORIZON_SEC)
        ].copy() if len(ct) else pd.DataFrame()

        anchor = safe_float(base.get("anchor_ts_recv_ns"))
        source_tape_end_ns = (int(base.get("candidate_second", c.candidate_second)) + MAX_HORIZON_SEC) * 1_000_000_000
        for h in EXPECTED_HORIZONS:
            if math.isfinite(anchor) and int(anchor) + h * 1_000_000_000 > source_tape_end_ns:
                metric_rows.append(unavailable_metric(
                    base, h, "PILOT_SAVED_TAPE_ENDS_BEFORE_CORRECTED_ANCHOR_HORIZON",
                    timing_basis="REUSED_7M6B_EVENT_TAPE_ANCHOR_RELATIVE", horizon_complete=False,
                ))
                continue
            metric, lev_rows, _ = build_horizon_from_events(
                base=base,
                opposing_levels=lv,
                depth_events=depth,
                trades=trades,
                h=h,
                trades_need_direction_filter=True,
                timing_basis="REUSED_7M6B_EVENT_TAPE_ANCHOR_RELATIVE",
            )
            metric_rows.append(metric)
            level_rows.extend(lev_rows)

    tape_out = tape.copy()
    if len(tape_out):
        tape_out["v2_provenance"] = "REUSED_FROM_SIGNED_7M6B_EVENT_TAPE"
        tape_out["v2_note"] = "source tape preserved; v2 metrics rebuilt anchor-relative"

    primary = pd.DataFrame(metric_rows)
    p3 = primary[pd.to_numeric(primary["observation_horizon_sec"], errors="coerce") == PRIMARY_HORIZON_SEC]
    diag = {
        "candidate_date": day,
        "candidate_count": int(len(day_clock)),
        "build_mode": "REUSED_7M6B_EVENT_TAPE",
        "anchor_found": int(bool_series(p3["anchor_found"]).sum()) if len(p3) else 0,
        "sequence_valid": int(bool_series(p3["sequence_valid"]).sum()) if len(p3) else 0,
        "primary_horizon_complete": int(bool_series(p3["horizon_complete"]).sum()) if len(p3) else 0,
        "filtered_trade_rows": 0,
        "depth_rows": 0,
        "source_tape_rows": int(len(tape)),
        "pilot_10s_corrected_unavailable": int((pd.to_numeric(primary["observation_horizon_sec"], errors="coerce") == 10).sum() - bool_series(primary[pd.to_numeric(primary["observation_horizon_sec"], errors="coerce") == 10]["horizon_complete"]).sum()),
    }
    return pd.DataFrame(metric_rows), pd.DataFrame(level_rows), tape_out, diag



def iter_depth_rows_with_times_chunk_safe(
    phase6a: Any,
    paths: Sequence[Path],
    *,
    symbol: str,
    price_scale: float,
    fast_stats: Any,
    batch_size: int = DEPTH_STREAM_BATCH_SIZE_7M6F,
):
    """Decode depth rows without concatenating nested Arrow chunks.

    Scientific/event semantics are identical to the frozen 7M.6B decoder.  The
    operational difference is that each Parquet RecordBatch is consumed directly:
    ``bid_updates`` and ``ask_updates`` are never passed through
    ``ChunkedArray.combine_chunks()``.  Dense nested book updates therefore do not
    require a second temporary copy of the whole nested level array.
    """
    if pa is None or pc is None or pq is None:
        raise RuntimeError("pyarrow + pyarrow.compute are required for chunk-safe depth streaming")
    cols = [
        "ts_recv_ns", "symbol", "source_record_number",
        "exchange_event_ms", "transaction_time_ms",
        "first_update_id", "final_update_id", "previous_final_update_id",
        "bid_updates", "ask_updates",
    ]
    batch_counter = 0
    for path in paths:
        pf = pq.ParquetFile(path)
        names = set(pf.schema_arrow.names)
        missing = [c for c in cols if c not in names]
        if missing:
            raise RuntimeError(f"Depth parquet missing required 7M.6F columns: {path}: {missing}")

        for batch in pf.iter_batches(columns=cols, batch_size=int(batch_size)):
            decode_t0 = time.perf_counter()
            if batch.num_rows == 0:
                continue

            def raw_col(name: str):
                idx = batch.schema.get_field_index(name)
                if idx < 0:
                    raise RuntimeError(f"Depth RecordBatch missing column {name!r}: {path}")
                return batch.column(idx)

            symbol_arr = raw_col("symbol")
            mask = pc.equal(symbol_arr, pa.scalar(symbol, type=symbol_arr.type))
            match_count_scalar = pc.sum(pc.cast(mask, pa.int64(), safe=False))
            match_count = int(match_count_scalar.as_py() or 0)
            if match_count == 0:
                continue
            all_match = match_count == int(batch.num_rows)

            def selected_col(name: str):
                arr = raw_col(name)
                return arr if all_match else pc.filter(arr, mask)

            def np_int(name: str) -> np.ndarray:
                arr = pc.cast(selected_col(name), pa.int64(), safe=False)
                vals = arr.to_numpy(zero_copy_only=False)
                return np.asarray(vals, dtype="int64")

            n = match_count
            ts = np_int("ts_recv_ns")
            U = np_int("first_update_id")
            u = np_int("final_update_id")
            pu = np_int("previous_final_update_id")
            recno = np_int("source_record_number")
            def np_nullable_int(name: str) -> tuple[np.ndarray, np.ndarray]:
                arr = selected_col(name)
                valid = np.asarray(pc.is_valid(arr).to_numpy(zero_copy_only=False), dtype=bool)
                filled = pc.cast(pc.fill_null(arr, -1), pa.int64(), safe=False)
                vals = np.asarray(filled.to_numpy(zero_copy_only=False), dtype="int64")
                return vals, valid

            exch, exch_valid = np_nullable_int("exchange_event_ms")
            txn, txn_valid = np_nullable_int("transaction_time_ms")

            # IMPORTANT: these are Array objects from this one RecordBatch.  Do
            # not wrap in a Table and do not combine_chunks(); either operation
            # can duplicate very large nested list<struct> buffers.
            b_arr = selected_col("bid_updates")
            a_arr = selected_col("ask_updates")
            boff, bticks, bqty = phase6a._arrow_list_struct_numeric(
                b_arr, price_scale, path, "bid_updates"
            )
            aoff, aticks, aqty = phase6a._arrow_list_struct_numeric(
                a_arr, price_scale, path, "ask_updates"
            )

            try:
                fast_stats.rows_decoded += int(n)
                fast_stats.levels_decoded += int(len(bticks) + len(aticks))
                fast_stats.arrow_batches += 1
                fast_stats.arrow_decode_seconds += float(time.perf_counter() - decode_t0)
            except Exception:
                pass

            for i in range(n):
                b0, b1 = int(boff[i]), int(boff[i + 1])
                a0, a1 = int(aoff[i]), int(aoff[i + 1])
                yield {
                    "ts_recv_ns": int(ts[i]),
                    "source_record_number": int(recno[i]),
                    "exchange_event_ms": int(exch[i]) if bool(exch_valid[i]) else None,
                    "transaction_time_ms": int(txn[i]) if bool(txn_valid[i]) else None,
                    "first_update_id": int(U[i]),
                    "final_update_id": int(u[i]),
                    "previous_final_update_id": int(pu[i]),
                    "_fast_bid_ticks": bticks[b0:b1],
                    "_fast_bid_qty": bqty[b0:b1],
                    "_fast_ask_ticks": aticks[a0:a1],
                    "_fast_ask_qty": aqty[a0:a1],
                }

            batch_counter += 1
            # Encourage deterministic release of temporary NumPy/Arrow buffers
            # during multi-hour replays. This does not alter row ordering/state.
            if batch_counter % 512 == 0:
                gc.collect()
                try:
                    pa.default_memory_pool().release_unused()
                except Exception:
                    pass



def merge_rows_no_copy(iterators: Sequence[Any]) -> Any:
    """K-way merge receive-time-sorted shards without copying every row dict.

    7M.6B's generic helper defensively calls ``dict(row)`` for each message.
    Depth rows here are already newly-created dictionaries, so that extra copy
    only adds allocation/GC pressure. Ordering is identical: ts_recv_ns,
    source_record_number, then shard index.
    """
    import heapq
    heap: list[tuple[int, int, int, dict[str, Any], Any]] = []
    for idx, it0 in enumerate(iterators):
        it = iter(it0)
        try:
            row = next(it)
        except StopIteration:
            continue
        heapq.heappush(heap, (int(row["ts_recv_ns"]), int(row.get("source_record_number", 0) or 0), idx, row, it))
    while heap:
        _ts, _rn, idx, row, it = heapq.heappop(heap)
        yield row
        try:
            nxt = next(it)
        except StopIteration:
            continue
        heapq.heappush(heap, (int(nxt["ts_recv_ns"]), int(nxt.get("source_record_number", 0) or 0), idx, nxt, it))


def targeted_depth_changes_before_apply(
    book: Any,
    row: Mapping[str, Any],
    active: Sequence[Any],
    ts_ns: int,
    phase7m6b: Any,
) -> list[tuple[str, int, float, float]]:
    """Extract before/after quantities only for frozen opposing ticks in live windows."""
    wanted_bid: set[int] = set()
    wanted_ask: set[int] = set()
    for t in active:
        if not bool(getattr(t, "anchor_found", False)):
            continue
        anchor_ns = getattr(t, "anchor_ts_recv_ns", None)
        if anchor_ns is None or int(ts_ns) <= int(anchor_ns) or int(ts_ns) >= int(t.final_end_ns):
            continue
        if t.expected_side == "BID":
            wanted_bid.update(int(x) for x in t.opposing_ticks)
        else:
            wanted_ask.update(int(x) for x in t.opposing_ticks)
    if not wanted_bid and not wanted_ask:
        return []

    if ((wanted_bid and (row.get("_fast_bid_ticks") is None or row.get("_fast_bid_qty") is None))
            or (wanted_ask and (row.get("_fast_ask_ticks") is None or row.get("_fast_ask_qty") is None))):
        # Operational fallback for a non-fast row; preserve both sides exactly.
        full = phase7m6b.depth_changes_before_apply(book, row)
        return [
            x for x in full
            if (x[0] == "BID" and int(x[1]) in wanted_bid)
            or (x[0] == "ASK" and int(x[1]) in wanted_ask)
        ]

    out: list[tuple[str, int, float, float]] = []
    for side_name, side, wanted, ticks_key, qty_key in (
        ("BID", book.bids, wanted_bid, "_fast_bid_ticks", "_fast_bid_qty"),
        ("ASK", book.asks, wanted_ask, "_fast_ask_ticks", "_fast_ask_qty"),
    ):
        if not wanted:
            continue
        ticks = row.get(ticks_key)
        qtys = row.get(qty_key)
        for tick, newq in zip(ticks, qtys):
            ti = int(tick)
            if ti not in wanted:
                continue
            nq = max(0.0, float(newq))
            out.append((side_name, ti, phase7m6b.side_qty(side, ti), nq))
    return out

def build_raw_iters_bounded(
    phase6a: Any, phase7m6b: Any, day_files: Any, symbol: str,
    price_scale: float, fast_stats: Any,
    depth_batch_size: int = DEPTH_STREAM_BATCH_SIZE_7M6F,
    snapshot_batch_size: int = SNAPSHOT_STREAM_BATCH_SIZE_7M6F,
):
    """Bound nested Arrow batches without changing event ordering or book semantics.

    The older 7M.6B helper asks ``iter_parquet_rows`` for 4,096 full-book
    snapshots at once.  A snapshot row contains nested bid/ask level arrays, so
    converting that many rows through ``RecordBatch.to_pylist`` can stall for a
    long time and create a large transient object graph.  Here only the I/O
    batch sizes change: depth rows still use the same Arrow-native decoder and
    snapshots still use the same Phase-6A row decoder / LocalBook contract.
    """
    snap_cols = [
        "ts_recv_ns", "symbol", "source_record_number",
        "last_update_id", "bids", "asks",
    ]
    depth_its = [
        iter_depth_rows_with_times_chunk_safe(
            phase6a, (p,), symbol=symbol, price_scale=price_scale,
            fast_stats=fast_stats, batch_size=int(depth_batch_size),
        )
        for p in day_files.depth
    ]
    snap_its = [
        phase6a.iter_parquet_rows(
            (p,), snap_cols, symbol=symbol,
            batch_size=int(snapshot_batch_size),
        )
        for p in day_files.snapshots
    ]
    return merge_rows_no_copy(snap_its), merge_rows_no_copy(depth_its)


def exact_change_tracking_required(active: Sequence[Any], ts_ns: int) -> bool:
    """Return True only when an anchored candidate can consume this depth diff.

    This is the same performance principle used by the proven 07E/07I4 replay
    path: LocalBook must still consume every authoritative depth row, but the
    expensive Python before/after level extraction is unnecessary unless at
    least one candidate is inside its post-anchor observation window.

    Scientific semantics are unchanged. Candidate mechanics only consume depth
    changes strictly after the recovered anchor and strictly before anchor+10s.
    Unanchored candidates use LocalBook state matching, not extracted changes.
    """
    if not active:
        return False
    for t in active:
        if not bool(getattr(t, "anchor_found", False)):
            continue
        anchor_ns = getattr(t, "anchor_ts_recv_ns", None)
        if anchor_ns is None:
            continue
        if int(ts_ns) > int(anchor_ns) and int(ts_ns) < int(t.final_end_ns):
            return True
    return False


def raw_process_day_v2(
    *, day: str, day_clock: pd.DataFrame, level_audit: pd.DataFrame, day_files: Any,
    phase6a: Any, phase7e: Any, phase7m6b: Any, symbol: str, price_scale: float,
    level_output_path: Path, tape_output_path: Path,
    depth_batch_size: int = DEPTH_STREAM_BATCH_SIZE_7M6F,
    snapshot_batch_size: int = SNAPSHOT_STREAM_BATCH_SIZE_7M6F,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one raw v2 day and stream large level/tape outputs directly to Parquet.

    Scientific calculations are identical to v1.0.0. Only output materialization
    changes so a full-day level/tape Python list and pandas DataFrame never coexist.
    """
    candidate_ids = set(day_clock["candidate_id"].astype(str))
    level_audit = level_audit[level_audit["candidate_id"].isin(candidate_ids)].copy()
    trackers = []
    missing_reason: dict[str, str] = {}
    for r in day_clock.itertuples(index=False):
        cid = str(r.candidate_id)
        lev = level_audit[level_audit["candidate_id"] == cid].copy()
        opp = lev[lev["side_role"] == "OPPOSING"].copy() if not lev.empty else lev.copy()
        if lev.empty:
            missing_reason[cid] = "FROZEN_7M3_LEVEL_AUDIT_MISSING"
            continue
        if opp.empty:
            missing_reason[cid] = "FROZEN_7M3_OPPOSING_LEVELS_MISSING"
            continue
        cs = int(r.candidate_second)
        tr = phase7m6b.CandidateTracker(
            candidate_id=cid, candidate_date=day, direction=int(r.direction), candidate_second=cs,
            all_levels=lev, opposing_levels=opp,
            search_start_ns=cs * 1_000_000_000, search_end_ns=(cs + 1) * 1_000_000_000,
            final_end_ns=(cs + MAX_HORIZON_SEC + 1) * 1_000_000_000,
        )
        tr._compact_depth_states = init_compact_depth_states(opp)
        trackers.append(tr)

    clock_by_id = day_clock.set_index(day_clock["candidate_id"].astype(str), drop=False)
    level_writer = AtomicParquetRowWriter(level_output_path, level_fate_arrow_schema())
    tape_writer = AtomicParquetRowWriter(tape_output_path, event_tape_arrow_schema())
    spool_dir: Path | None = None
    spool_path: Path | None = None
    spool_writer: Any | None = None
    spool_groups_by_candidate: dict[str, list[int]] = {}
    spool_group_count = 0
    spooled_ids: set[str] = set()
    refill_ticks_by_candidate: dict[str, set[int]] = {}

    def cleanup_spool() -> None:
        nonlocal spool_dir, spool_writer
        if spool_writer is not None:
            try:
                spool_writer.close()
            except Exception:
                pass
            spool_writer = None
        if spool_dir is not None:
            shutil.rmtree(spool_dir, ignore_errors=True)
            spool_dir = None

    try:
        if not trackers:
            rows = []
            for cid, reason in missing_reason.items():
                b = clock_by_id.loc[cid].to_dict()
                b.update({"candidate_id": cid, "candidate_date": day, "anchor_found": False, "sequence_valid": False})
                for h in EXPECTED_HORIZONS:
                    rows.append(unavailable_metric(b, h, reason, timing_basis="RAW_REPLAY_RECOVERED_ANCHOR_PLUS_H"))
            metrics_out = pd.DataFrame(rows)
            diag = {
                "candidate_date": day, "candidate_count": len(day_clock), "build_mode": "RAW_CORRECTED_ANCHOR_RELATIVE",
                "anchor_found": 0, "sequence_valid": 0, "primary_horizon_complete": 0,
                "filtered_trade_rows": 0, "depth_rows": 0, "applicable_depth_rows": 0,
                "gap_records": 0, "level_fate_rows": 0, "event_tape_rows": 0,
            }
            level_writer.commit(); tape_writer.commit()
            return metrics_out, diag

        # Keep only live candidates' depth-event dicts in memory. Completed
        # candidates are appended to ONE shared, uncompressed temporary Parquet
        # spool. This avoids hundreds of file open/close + zstd compress/decompress
        # cycles on busy days while retaining the exact audit tape. Scientific
        # mechanics no longer reread this tape; compact per-horizon state is
        # maintained incrementally during replay.
        spool_dir = Path(tempfile.mkdtemp(prefix="tmp7m6f-depth-spool-", dir=str(level_output_path.parent)))
        spool_path = spool_dir / "depth_events.parquet"
        spool_writer = pq.ParquetWriter(str(spool_path), event_tape_arrow_schema(), compression=None)

        def spill_tracker_depth(t: Any) -> None:
            nonlocal spool_group_count
            cid = str(t.candidate_id)
            if cid in spooled_ids:
                return
            # Freeze exact trade-opportunity ticks from the streaming state. The
            # definition is identical to the old max-horizon depletion->refill
            # scan, with the same anchor/sequence-validity requirements.
            refill_ticks_by_candidate[cid] = compact_refill_ticks(t)
            rows = t.depth_events
            groups: list[int] = []
            if rows:
                for start in range(0, len(rows), PARQUET_ROW_WRITE_CHUNK_7M6F):
                    chunk = rows[start : start + PARQUET_ROW_WRITE_CHUNK_7M6F]
                    table = pa.Table.from_pylist(chunk, schema=event_tape_arrow_schema())
                    # Explicit row_group_size=len(table) ensures one addressable
                    # row group per chunk so candidate ordering can be reproduced
                    # later without converting the spool back to pandas.
                    spool_writer.write_table(table, row_group_size=max(1, int(table.num_rows)))
                    groups.append(spool_group_count)
                    spool_group_count += 1
                    del table, chunk
                rows.clear()
            spool_groups_by_candidate[cid] = groups
            spooled_ids.add(cid)

        book = phase6a.LocalBook(price_scale=price_scale)
        fast_stats = phase6a.FastDepthStats()
        gap_records: list[Any] = []
        snapshots, depths = build_raw_iters_bounded(
            phase6a, phase7m6b, day_files, symbol, price_scale, fast_stats,
            depth_batch_size=int(depth_batch_size),
            snapshot_batch_size=int(snapshot_batch_size),
        )
        active = []
        pending = sorted(trackers, key=lambda t: (t.search_start_ns, t.candidate_id))
        pidx = 0
        stop_ns = max(int(t.final_end_ns) for t in trackers)
        depth_rows = 0
        applicable_depth_rows = 0
        replay_t0 = time.perf_counter()

        def try_anchor(t: Any, state_ts_ns: int) -> None:
            if t.anchor_found or state_ts_ns < t.search_start_ns or state_ts_ns > t.search_end_ns:
                return
            if phase7m6b.book_vector_matches(book, t.all_levels, t.direction):
                t.anchor_found = True
                t.anchor_ts_recv_ns = int(state_ts_ns)
                t.final_end_ns = int(state_ts_ns) + MAX_HORIZON_SEC * 1_000_000_000

        last_state_ts = 0
        last_heartbeat = time.monotonic()
        last_heartbeat_perf = time.perf_counter()
        last_heartbeat_depth_rows = 0
        snapshot_rows = 0
        for kind, rec in phase7m6b.merge_event_families(snapshots, depths):
            ts_ns = int(rec["ts_recv_ns"])
            if ts_ns > stop_ns and pidx >= len(pending):
                break
            while pidx < len(pending) and pending[pidx].search_start_ns <= ts_ns:
                t = pending[pidx]
                active.append(t)
                try_anchor(t, last_state_ts if last_state_ts >= t.search_start_ns else t.search_start_ns)
                pidx += 1
            for t in active:
                try_anchor(t, ts_ns)

            if kind == "SNAPSHOT":
                snapshot_rows += 1
                book.apply_snapshot(rec)
                last_state_ts = ts_ns
                for t in active:
                    try_anchor(t, ts_ns)
            else:
                depth_rows += 1
                applicable = bool(phase7e.depth_row_is_applicable(book, rec))
                # Performance-only gate restored from the proven 07E/07I4
                # replay architecture. The LocalBook still consumes every
                # authoritative diff below. We only pay the Python-level
                # before/after extraction cost when at least one anchored
                # candidate can actually consume this row.
                track_changes = applicable and exact_change_tracking_required(active, ts_ns)
                changes = targeted_depth_changes_before_apply(book, rec, active, ts_ns, phase7m6b) if track_changes else []
                if applicable:
                    applicable_depth_rows += 1
                if track_changes:
                    for t in active:
                        if not t.anchor_found or ts_ns <= int(t.anchor_ts_recv_ns or 0) or ts_ns >= int(t.final_end_ns):
                            continue
                        if not bool(getattr(book, "valid", False)):
                            t.sequence_valid = False
                            if not t.invalid_reason:
                                t.invalid_reason = "BOOK_INVALID_DURING_WINDOW"
                            continue
                        wanted = t.opposing_ticks
                        for side_name, tick, old_qty, new_qty in changes:
                            if side_name != t.expected_side or int(tick) not in wanted:
                                continue
                            update_compact_depth_states(
                                t,
                                ts_ns=ts_ns,
                                side_name=side_name,
                                tick=int(tick),
                                old_qty=float(old_qty),
                                new_qty=float(new_qty),
                                transaction_time_ms=rec.get("transaction_time_ms"),
                                exchange_event_ms=rec.get("exchange_event_ms"),
                            )
                            t.depth_events.append({
                                "candidate_id": t.candidate_id, "candidate_date": day, "direction": t.direction,
                                "ts_recv_ns": ts_ns,
                                "relative_to_anchor_ms": (ts_ns - int(t.anchor_ts_recv_ns)) / 1e6,
                                "transaction_time_ms": safe_float(rec.get("transaction_time_ms")),
                                "exchange_event_ms": safe_float(rec.get("exchange_event_ms")),
                                "first_update_id": rec.get("first_update_id"), "final_update_id": rec.get("final_update_id"),
                                "previous_final_update_id": rec.get("previous_final_update_id"),
                                "book_side": side_name, "price_tick": int(tick), "price": float(tick) / price_scale,
                                "old_qty": float(old_qty), "new_qty": float(new_qty),
                                "delta_qty": float(new_qty) - float(old_qty), "event_kind": "DEPTH_CHANGE",
                                "observation_horizon_sec": math.nan,
                            })
                book.apply_depth(rec, gap_records, day)
                last_state_ts = ts_ns
                if not bool(getattr(book, "valid", False)):
                    for t in active:
                        if t.anchor_found and ts_ns < int(t.final_end_ns):
                            t.sequence_valid = False
                            if not t.invalid_reason:
                                t.invalid_reason = "BOOK_SEQUENCE_GAP_OR_INVALID"
                for t in active:
                    try_anchor(t, ts_ns)

            keep = []
            expired = []
            for t in active:
                if not t.anchor_found and ts_ns > t.search_end_ns:
                    t.sequence_valid = False
                    if not t.invalid_reason:
                        t.invalid_reason = "FROZEN_ANCHOR_STATE_NOT_REFOUND_WITHIN_CANDIDATE_SECOND"
                if ts_ns < int(t.final_end_ns):
                    keep.append(t)
                else:
                    expired.append(t)
            active = keep
            for t in expired:
                spill_tracker_depth(t)
            if expired:
                del expired

            now = time.monotonic()
            if RAW_REPLAY_HEARTBEAT_SEC > 0 and now - last_heartbeat >= RAW_REPLAY_HEARTBEAT_SEC:
                anchors_now = int(sum(t.anchor_found for t in trackers))
                perf_now = time.perf_counter()
                elapsed = max(1e-9, perf_now - replay_t0)
                rate = depth_rows / elapsed
                recent_elapsed = max(1e-9, perf_now - last_heartbeat_perf)
                recent_rate = (depth_rows - last_heartbeat_depth_rows) / recent_elapsed
                print(
                    f"      {day}: replaying depth={depth_rows:,} applicable={applicable_depth_rows:,} "
                    f"snapshots={snapshot_rows:,} anchors={anchors_now}/{len(trackers)} active={len(active)} "
                    f"spooled={len(spooled_ids)} book={len(book.bids.qty)}/{len(book.asks.qty)} "
                    f"rate={rate:,.0f} depth/s recent={recent_rate:,.0f}/s",
                    flush=True,
                )
                last_heartbeat_perf = perf_now
                last_heartbeat_depth_rows = depth_rows
                # Do not force a full Python GC every 30s. With direct RecordBatch
                # decoding there is no whole-chunk nested copy to collect, and the
                # forced scans materially reduce throughput on long days.
                last_heartbeat = now

        depth_replay_seconds = float(time.perf_counter() - replay_t0)
        post_replay_t0 = time.perf_counter()

        # If the merged streams ended exactly at the capture boundary, ensure the
        # last active candidates are also spilled before loading trades.
        for t in trackers:
            spill_tracker_depth(t)
        if spool_writer is not None:
            spool_writer.close()
            spool_writer = None

        gc.collect()
        try:
            pa.default_memory_pool().release_unused()
        except Exception:
            pass

        for t in trackers:
            if not t.anchor_found:
                t.sequence_valid = False
                if not t.invalid_reason:
                    t.invalid_reason = "ANCHOR_NOT_FOUND"

        # Depth-event lists have been spilled/cleared. The exact max-horizon
        # refill tick set was frozen from the compact state using the same
        # depletion->positive-update definition as the historical tape scan.
        trade_ticks: set[int] = set()
        for ticks in refill_ticks_by_candidate.values():
            trade_ticks.update(int(x) for x in ticks)
        intervals = phase7m6b.merge_candidate_windows(trackers)
        trades = phase7m6b.load_filtered_trades(day_files.trades, symbol, trade_ticks, intervals)
        trades_by_tick = {int(k): g.copy() for k, g in trades.groupby("price_tick", sort=False)} if len(trades) else {}
        filtered_trade_rows = int(len(trades))
        del trades, trade_ticks, intervals
        gc.collect()

        anchor_found_count = int(sum(t.anchor_found for t in trackers))
        sequence_valid_count = int(sum(t.sequence_valid for t in trackers))
        metric_rows: list[dict[str, Any]] = []
        spool_reader = pq.ParquetFile(spool_path) if spool_path is not None and spool_path.exists() and spool_group_count > 0 else None
        for ti, t in enumerate(trackers, 1):
            c = clock_by_id.loc[t.candidate_id]
            base = c.to_dict()
            base.update({
                "candidate_id": t.candidate_id, "candidate_date": day, "direction": t.direction,
                "candidate_second": t.candidate_second, "anchor_found": t.anchor_found,
                "anchor_ts_recv_ns": t.anchor_ts_recv_ns,
                "anchor_offset_ms": ((int(t.anchor_ts_recv_ns) - t.candidate_second * 1_000_000_000) / 1e6) if t.anchor_ts_recv_ns is not None else math.nan,
                "sequence_valid": t.sequence_valid, "invalid_reason": t.invalid_reason,
            })
            compact_states = getattr(t, "_compact_depth_states", {}) or {}
            for h in EXPECTED_HORIZONS:
                metric, lev_rows, trade_tape = build_horizon_from_compact(
                    base=base, opposing_levels=t.opposing_levels, compact_states=compact_states,
                    trades_by_tick=trades_by_tick, h=h, trades_need_direction_filter=True,
                    timing_basis="RAW_REPLAY_RECOVERED_ANCHOR_PLUS_H",
                )
                metric_rows.append(metric)
                level_writer.write_rows(lev_rows)
                tape_writer.write_rows(trade_tape)
                del lev_rows, trade_tape
            # Preserve the historical candidate-level event-tape ordering: all
            # horizon trade rows for this candidate first, then its raw depth
            # tape exactly once. The shared spool stores candidate chunks as
            # addressable row groups, so no pandas materialization is needed.
            if spool_reader is not None:
                for rg in spool_groups_by_candidate.get(str(t.candidate_id), []):
                    tape_writer.write_table(spool_reader.read_row_group(int(rg)))
            del compact_states
            if ti % 16 == 0:
                gc.collect()
                try:
                    pa.default_memory_pool().release_unused()
                except Exception:
                    pass

        if spool_reader is not None:
            close_spool_reader = getattr(spool_reader, "close", None)
            if callable(close_spool_reader):
                close_spool_reader()
            spool_reader = None
        del trades_by_tick
        gc.collect()
        try:
            pa.default_memory_pool().release_unused()
        except Exception:
            pass

        for cid, reason in missing_reason.items():
            b = clock_by_id.loc[cid].to_dict()
            b.update({"candidate_id": cid, "candidate_date": day, "anchor_found": False, "sequence_valid": False})
            for h in EXPECTED_HORIZONS:
                metric_rows.append(unavailable_metric(b, h, reason, timing_basis="RAW_REPLAY_RECOVERED_ANCHOR_PLUS_H"))

        metrics_out = pd.DataFrame(metric_rows)
        post_replay_seconds = float(time.perf_counter() - post_replay_t0)
        total_day_seconds = float(time.perf_counter() - replay_t0)
        p3 = metrics_out[pd.to_numeric(metrics_out["observation_horizon_sec"], errors="coerce") == PRIMARY_HORIZON_SEC]
        diag = {
            "candidate_date": day, "candidate_count": int(len(day_clock)), "build_mode": "RAW_CORRECTED_ANCHOR_RELATIVE",
            "frozen_level_eligible": int(len(trackers)), "frozen_level_missing": int(len(missing_reason)),
            "anchor_found": anchor_found_count, "sequence_valid": sequence_valid_count,
            "primary_horizon_complete": int(bool_series(p3["horizon_complete"]).sum()) if len(p3) else 0,
            "filtered_trade_rows": filtered_trade_rows, "depth_rows": int(depth_rows),
            "applicable_depth_rows": int(applicable_depth_rows), "gap_records": int(len(gap_records)),
            "depth_rows_decoded": int(getattr(fast_stats, "rows_decoded", 0) or 0),
            "depth_levels_decoded": int(getattr(fast_stats, "levels_decoded", 0) or 0),
            "depth_arrow_batches": int(getattr(fast_stats, "arrow_batches", 0) or 0),
            "depth_arrow_decode_seconds": float(getattr(fast_stats, "arrow_decode_seconds", 0.0) or 0.0),
            "snapshot_rows": int(snapshot_rows),
            "raw_replay_seconds": depth_replay_seconds,
            "post_replay_seconds": post_replay_seconds,
            "total_day_seconds": total_day_seconds,
            "depth_rows_per_second": float(depth_rows / depth_replay_seconds) if depth_replay_seconds > 0 else math.nan,
            "depth_batch_size": int(depth_batch_size),
            "snapshot_batch_size": int(snapshot_batch_size),
            "level_fate_rows": int(level_writer.row_count), "event_tape_rows": int(tape_writer.row_count),
        }
        level_writer.commit()
        tape_writer.commit()
        cleanup_spool()
        return metrics_out, diag
    except BaseException:
        # Includes KeyboardInterrupt: never leave an open temp Parquet writer.
        level_writer.abort()
        tape_writer.abort()
        cleanup_spool()
        raise


def full_binary_gate(primary: pd.DataFrame, phase7m6b: Any, phase7m6e: Any) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    analysis = phase7m6e.make_analysis_frame(primary)
    z = analysis[bool_series(analysis["replacement_penetration_eligible"])].copy()
    for c in ("replacement_penetrated", "__momentum", "__aggression", "__destruction", "__sequence_refill", "__refill_breadth"):
        z = z[pd.to_numeric(z[c], errors="coerce").notna()].copy()
    exp = pd.to_numeric(z["replacement_penetrated"], errors="coerce")
    if not exp.dropna().isin([0, 1]).all():
        raise RuntimeError("Full binary analysis contains non-binary eligible exposure values.")

    controls = ["__momentum", "__aggression", "__destruction", "__sequence_refill", "__refill_breadth"]
    pairs0 = phase7m6e.binary_greedy_pairs(z, "replacement_penetrated", controls, phase7m6b)
    repaired = phase7m6b.repair_pairs(pairs0, controls) if len(pairs0) else pairs0
    mx, vals = phase7m6b.pair_balance(repaired, controls) if len(repaired) else (math.nan, {})
    full_pairs = int(len(repaired))
    discovery_pairs = int((repaired["candidate_date"].astype(str) <= DISCOVERY_END_DATE).sum()) if len(repaired) else 0
    validation_pairs = int((repaired["candidate_date"].astype(str) >= VALIDATION_START_DATE).sum()) if len(repaired) else 0
    positive_share = float((exp == 1).mean()) if len(z) else math.nan
    feasible = bool(
        full_pairs >= MIN_FULL_PRIMARY_PAIRS
        and discovery_pairs >= MIN_SPLIT_PAIRS
        and validation_pairs >= MIN_SPLIT_PAIRS
        and math.isfinite(mx) and mx <= BALANCE_SMD_GATE
        and math.isfinite(positive_share) and MIN_FEATURE_POSITIVE_SHARE <= positive_share <= MAX_FEATURE_POSITIVE_SHARE
    )
    row: dict[str, Any] = {
        "feature_role": "PRIMARY",
        "feature": "replacement_penetrated",
        "feature_definition": "opposing__replacement_execution_lower_bound_quote > 0",
        "control_set": "REFILL_AMOUNT_BREADTH_PRIMARY",
        "eligible_candidates": int(len(z)),
        "penetrated_candidates": int((exp == 1).sum()),
        "not_penetrated_candidates": int((exp == 0).sum()),
        "penetrated_share": positive_share,
        "pairs_before_repair": int(len(pairs0)),
        "pairs_after_repair": full_pairs,
        "max_abs_smd": mx,
        "discovery_pairs": discovery_pairs,
        "validation_pairs": validation_pairs,
        "full_pair_gate": MIN_FULL_PRIMARY_PAIRS,
        "split_pair_gate": MIN_SPLIT_PAIRS,
        "balance_gate": BALANCE_SMD_GATE,
        "feasible_before_outcomes": feasible,
    }
    for c, v in vals.items():
        row[f"smd__{c}"] = v

    subgroup_rows = []
    definitions = [
        ("SPLIT", "DISCOVERY", z["candidate_date"].astype(str) <= DISCOVERY_END_DATE),
        ("SPLIT", "VALIDATION", z["candidate_date"].astype(str) >= VALIDATION_START_DATE),
        ("DIRECTION", "UP", pd.to_numeric(z["direction"], errors="coerce") > 0),
        ("DIRECTION", "DOWN", pd.to_numeric(z["direction"], errors="coerce") < 0),
        ("REACCEL", "REACCEL", bool_series(z.get("is_reaccel_context", pd.Series(False, index=z.index)))),
        ("REACCEL", "NON_REACCEL", ~bool_series(z.get("is_reaccel_context", pd.Series(False, index=z.index)))),
    ]
    for family, label, mask in definitions:
        g = z[mask].copy()
        ge = pd.to_numeric(g["replacement_penetrated"], errors="coerce")
        ids = set(g["candidate_id"].astype(str))
        if len(repaired):
            pg = repaired[
                repaired["penetrated_candidate_id"].astype(str).isin(ids)
                & repaired["not_penetrated_candidate_id"].astype(str).isin(ids)
            ].copy()
        else:
            pg = pd.DataFrame()
        subgroup_rows.append({
            "subgroup_family": family,
            "subgroup": label,
            "eligible_candidates": int(len(g)),
            "penetrated_candidates": int((ge == 1).sum()),
            "not_penetrated_candidates": int((ge == 0).sum()),
            "penetrated_share": float((ge == 1).mean()) if len(g) else math.nan,
            "primary_matched_pairs": int(len(pg)),
            "both_exposure_states_present": bool((ge == 1).any() and (ge == 0).any()),
        })
    return pd.DataFrame([row]), repaired, pd.DataFrame(subgroup_rows)


def parity_summary(primary: pd.DataFrame) -> pd.DataFrame:
    a = pd.to_numeric(primary["opposing__observed_refill_quote"], errors="coerce")
    b = pd.to_numeric(primary["control__opposing_refill_quote_3s_frozen"], errors="coerce")
    m = a.notna() & b.notna() & (b >= 0)
    aa, bb = a[m], b[m]
    pos = bb > 0
    ratio = aa[pos] / bb[pos]
    rel = (aa[pos] - bb[pos]).abs() / bb[pos]
    return pd.DataFrame([{
        "rows_compared": int(m.sum()),
        "positive_frozen_refill_rows": int(pos.sum()),
        "pearson_corr": float(aa.corr(bb, method="pearson")) if len(aa) >= 3 else math.nan,
        "spearman_corr": float(aa.corr(bb, method="spearman")) if len(aa) >= 3 else math.nan,
        "median_sequence_to_frozen_ratio": float(ratio.median()) if len(ratio) else math.nan,
        "median_abs_relative_error": float(rel.median()) if len(rel) else math.nan,
        "median_abs_quote_difference": float((aa - bb).abs().median()) if len(aa) else math.nan,
    }])


def findings_text(summary: dict[str, Any], gate: pd.DataFrame, parity: pd.DataFrame) -> str:
    g = gate.iloc[0].to_dict() if len(gate) else {}
    p = parity.iloc[0].to_dict() if len(parity) else {}
    return "\n".join([
        "# Phase 7M.6F — Full corrected anchor-relative refill→re-depletion substrate",
        "",
        f"- Status: **{summary['status']}**",
        f"- Candidate dates: **{summary['candidate_dates']}**",
        f"- Candidates: **{summary['candidate_count']:,}**",
        f"- V2 checkpoints built from raw: **{summary['raw_built_dates']}**",
        f"- Pilot dates reconstructed from saved tape: **{summary['pilot_reused_dates']}**",
        f"- Existing V2 checkpoints reused: **{summary['v2_reused_dates']}**",
        "- Future outcomes loaded: **NO**",
        "",
        "## Primary full-data pre-outcome gate",
        "",
        f"- Eligible candidates: **{int(g.get('eligible_candidates', 0)):,}**",
        f"- Penetrated / not penetrated: **{int(g.get('penetrated_candidates', 0)):,} / {int(g.get('not_penetrated_candidates', 0)):,}**",
        f"- Matched pairs after repair: **{int(g.get('pairs_after_repair', 0)):,}** (gate {MIN_FULL_PRIMARY_PAIRS})",
        f"- Discovery / validation pairs: **{int(g.get('discovery_pairs', 0)):,} / {int(g.get('validation_pairs', 0)):,}** (gate {MIN_SPLIT_PAIRS} each)",
        f"- Max |SMD|: **{safe_float(g.get('max_abs_smd')):.3f}** (gate {BALANCE_SMD_GATE:.2f})",
        f"- Feasible before outcomes: **{bool(g.get('feasible_before_outcomes', False))}**",
        "",
        "## Refill parity diagnostic",
        "",
        f"- Pearson / Spearman: **{safe_float(p.get('pearson_corr')):.4f} / {safe_float(p.get('spearman_corr')):.4f}**",
        f"- Median sequence/frozen ratio: **{safe_float(p.get('median_sequence_to_frozen_ratio')):.4f}**",
        f"- Median absolute relative error: **{safe_float(p.get('median_abs_relative_error')):.2%}**",
        "",
        "## Interpretation rule",
        "",
        "If the full pre-outcome gate passes, freeze the pair manifest and review it before loading P75. This phase itself never loads future continuation labels.",
        "If the gate fails, classify the primary mechanism as NOT TESTED under the frozen observational design. Do not relax calipers, remove refill breadth, or change the physical zero boundary to force a result.",
        "",
    ])


def run_self_test(phase7m6b_script: Path, phase7m6e_script: Path) -> int:
    # Queue accounting: 2 old + 10 refill + 8 strict executions => 6 replacement lower bound.
    base = {
        "candidate_id": "c1", "candidate_date": "2026-06-01", "direction": 1,
        "candidate_second": 100, "anchor_found": True, "anchor_ts_recv_ns": 100_900_000_000,
        "anchor_offset_ms": 900.0, "sequence_valid": True, "invalid_reason": "",
        "control__directional_mid_move_bps_3s": 2.0,
        "control__directional_aggressive_quote_3s": 1000.0,
        "control__opposing_peak_initial_queue_loss_share_3s": 0.5,
        "control__opposing_refill_quote_3s_frozen": 1000.0,
        "is_reaccel_context": False,
    }
    levels = pd.DataFrame([{"price_tick": 1000, "price": 100.0, "initial_qty": 10.0}])
    depth = pd.DataFrame([
        {"ts_recv_ns": 101_000_000_000, "price_tick": 1000, "old_qty": 10.0, "new_qty": 2.0, "transaction_time_ms": 101000, "exchange_event_ms": 101000},
        {"ts_recv_ns": 101_100_000_000, "price_tick": 1000, "old_qty": 2.0, "new_qty": 12.0, "transaction_time_ms": 101100, "exchange_event_ms": 101100},
    ])
    trades = pd.DataFrame([{
        "ts_recv_ns": 101_200_000_000, "price_tick": 1000, "price": 100.0, "quantity": 8.0,
        "buyer_is_maker": False, "trade_time_ms": 101200, "exchange_event_ms": 101200,
    }])
    metric, lev, test_trade_tape = build_horizon_from_events(
        base=base, opposing_levels=levels, depth_events=depth, trades=trades, h=3,
        trades_need_direction_filter=True, timing_basis="SELF_TEST",
    )
    assert metric["replacement_penetrated"] == 1
    assert abs(metric["opposing__replacement_execution_lower_bound_quote"] - 600.0) < 1e-9
    assert int(metric["anchor_relative_endpoint_ts_recv_ns"]) == 103_900_000_000

    # Compact streaming state must be scientifically identical to replaying the
    # captured depth tape for the same horizon.
    class _CompactSelfTestTracker:
        anchor_found = True
        anchor_ts_recv_ns = 100_900_000_000
        sequence_valid = True
        expected_side = "ASK"

    cst = _CompactSelfTestTracker()
    cst._compact_depth_states = init_compact_depth_states(levels)
    for er in depth.itertuples(index=False):
        update_compact_depth_states(
            cst, ts_ns=int(er.ts_recv_ns), side_name="ASK", tick=int(er.price_tick),
            old_qty=float(er.old_qty), new_qty=float(er.new_qty),
            transaction_time_ms=er.transaction_time_ms, exchange_event_ms=er.exchange_event_ms,
        )
    compact_metric, compact_lev, compact_trade_tape = build_horizon_from_compact(
        base=base, opposing_levels=levels, compact_states=cst._compact_depth_states,
        trades_by_tick={1000: trades.copy()}, h=3, trades_need_direction_filter=True,
        timing_basis="SELF_TEST",
    )
    for key in (
        "opposing__observed_refill_quote",
        "opposing__strict_post_refill_aggressive_trade_quote",
        "opposing__replacement_execution_lower_bound_quote",
        "replacement_penetration_eligible",
        "replacement_penetrated",
    ):
        a, b = metric[key], compact_metric[key]
        if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
            assert (math.isnan(a) and math.isnan(b)) or abs(float(a) - float(b)) < 1e-12
        else:
            assert a == b
    assert len(compact_lev) == len(lev)
    for key in (
        "min_qty", "final_observed_qty", "depleted_before_refill", "first_refill_seen",
        "first_refill_ts_recv_ns", "pre_refill_residual_qty", "observed_refill_qty",
        "refill_update_count", "post_refill_gross_queue_reduction_qty",
        "replacement_execution_lower_bound_qty", "replacement_execution_lower_bound_positive",
    ):
        assert compact_lev[0][key] == lev[0][key]
    assert len(compact_trade_tape) == len(test_trade_tape)

    # Operational regression: raw-date level/tape output streams in bounded batches.
    with tempfile.TemporaryDirectory(prefix="7m6f-selftest-") as td:
        lp = Path(td) / "levels.parquet"
        tp = Path(td) / "tape.parquet"
        lw = AtomicParquetRowWriter(lp, level_fate_arrow_schema())
        tw = AtomicParquetRowWriter(tp, event_tape_arrow_schema())
        lw.write_rows(lev)
        tw.write_rows([{
            "candidate_id": "c1", "candidate_date": "2026-06-01", "direction": 1,
            "observation_horizon_sec": math.nan, "event_kind": "DEPTH_CHANGE",
            "ts_recv_ns": 101_000_000_000, "relative_to_anchor_ms": 100.0,
            "transaction_time_ms": 101000.0, "exchange_event_ms": 101000.0,
            "first_update_id": 1, "final_update_id": 2, "previous_final_update_id": 0,
            "book_side": "ASK", "price_tick": 1000, "price": 100.0,
            "old_qty": 10.0, "new_qty": 2.0, "delta_qty": -8.0,
        }])
        tw.write_rows(test_trade_tape)
        lw.commit(); tw.commit()
        assert pq.ParquetFile(lp).metadata.num_rows == len(lev)
        assert pq.ParquetFile(tp).metadata.num_rows == 1 + len(test_trade_tape)

        # Shared day-spool contract: each write is exactly one addressable row
        # group so candidates can be replayed to the final tape without pandas.
        sp = Path(td) / "shared-depth-spool.parquet"
        sw = pq.ParquetWriter(str(sp), event_tape_arrow_schema(), compression=None)
        try:
            for cid, ts in (("a", 1), ("b", 2)):
                tab = pa.Table.from_pylist([{
                    "candidate_id": cid, "candidate_date": "2026-06-01", "direction": 1,
                    "observation_horizon_sec": math.nan, "event_kind": "DEPTH_CHANGE",
                    "ts_recv_ns": ts, "relative_to_anchor_ms": float(ts),
                    "price_tick": 1000, "price": 100.0, "old_qty": 1.0,
                    "new_qty": 2.0, "delta_qty": 1.0,
                }], schema=event_tape_arrow_schema())
                sw.write_table(tab, row_group_size=1)
        finally:
            sw.close()
        spr = pq.ParquetFile(sp)
        assert spr.metadata.num_row_groups == 2
        assert spr.read_row_group(0).column("candidate_id")[0].as_py() == "a"
        assert spr.read_row_group(1).column("candidate_id")[0].as_py() == "b"

    # Saved boundary+10 tape cannot support anchor+10 when anchor offset is +900ms.
    source_end = (100 + 10) * 1_000_000_000
    assert int(base["anchor_ts_recv_ns"]) + 10 * 1_000_000_000 > source_end
    assert int(base["anchor_ts_recv_ns"]) + 5 * 1_000_000_000 < source_end

    phase7m6b = load_module(phase7m6b_script, "phase7m6b_selftest_for_7m6f")
    phase7m6e = load_module(phase7m6e_script, "phase7m6e_selftest_for_7m6f")
    phase7m6b.MATCH_Z_CALIPER = MATCH_Z_CALIPER
    phase7m6b.MATCH_EUCLIDEAN_CALIPER = MATCH_EUCLIDEAN_CALIPER
    phase7m6b.BALANCE_SMD_GATE = BALANCE_SMD_GATE
    rows = []
    for i in range(40):
        rows.append({
            "candidate_id": f"p{i}", "candidate_date": "2026-06-01", "direction": 1,
            "anchor_found": True, "sequence_valid": True, "horizon_complete": True,
            "opposing__observed_refill_quote": 1000.0 + (i // 2),
            "opposing__refilled_level_count": 1 + ((i // 2) % 3),
            "opposing__strict_trade_time_complete_share": 1.0,
            "opposing__replacement_execution_lower_bound_quote": 10.0 if i % 2 else 0.0,
            "control__directional_mid_move_bps_3s": float((i // 2) % 4),
            "control__directional_aggressive_quote_3s": 100.0 + ((i // 2) % 5),
            "control__opposing_peak_initial_queue_loss_share_3s": 0.5 + ((i // 2) % 3) * 0.01,
            "is_reaccel_context": False,
        })
    a = pd.DataFrame(rows)
    analysis = phase7m6e.make_analysis_frame(a)
    controls = ["__momentum", "__aggression", "__destruction", "__sequence_refill", "__refill_breadth"]
    pairs = phase7m6e.binary_greedy_pairs(analysis, "replacement_penetrated", controls, phase7m6b)
    assert len(pairs) > 0
    assert 1 <= SNAPSHOT_STREAM_BATCH_SIZE_7M6F <= 64
    assert 1024 <= DEPTH_STREAM_BATCH_SIZE_7M6F <= 8192
    import inspect
    decoder_src = inspect.getsource(iter_depth_rows_with_times_chunk_safe)
    assert ".combine_chunks(" not in decoder_src
    assert CHECKPOINT_CONTRACT_BUILDER_VERSION == "1.0.0"
    assert {"1.0.0", "1.0.1", "1.0.2", "1.0.3", "1.0.4", "1.0.5", "1.0.6", "1.0.7"}.issubset(COMPATIBLE_CHECKPOINT_BUILDERS)
    assert "1.0.1" in COMPATIBLE_CHECKPOINT_BUILDERS and "1.0.0" in COMPATIBLE_CHECKPOINT_BUILDERS
    class _GateTracker:
        anchor_found = True
        anchor_ts_recv_ns = 1_000
        final_end_ns = 11_000

    class _UnanchoredGateTracker:
        anchor_found = False
        anchor_ts_recv_ns = None
        final_end_ns = 11_000

    gt = _GateTracker()
    assert not exact_change_tracking_required([], 2_000)
    assert not exact_change_tracking_required([_UnanchoredGateTracker()], 2_000)
    assert not exact_change_tracking_required([gt], 1_000)
    assert exact_change_tracking_required([gt], 1_001)
    assert exact_change_tracking_required([gt], 10_999)
    assert not exact_change_tracking_required([gt], 11_000)
    print("PASS: Phase 7M.6F self-test (anchor-relative horizon, conservative penetration, compact streaming mechanics parity, shared depth-tape spool, chunk-safe 8192-row depth streaming, per-date isolation, targeted active-window change extraction, bounded snapshots, checkpoint reuse, pilot-tape coverage, direct binary matcher).")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 7M.6F full corrected anchor-relative refill/re-depletion builder")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--m3-root", type=Path, default=DEFAULT_M3_ROOT)
    p.add_argument("--phase7m6b-root", type=Path, default=DEFAULT_7M6B_ROOT)
    p.add_argument("--phase7m6e-root", type=Path, default=DEFAULT_7M6E_ROOT)
    p.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    p.add_argument("--phase7m6b-script", type=Path, default=DEFAULT_7M6B_SCRIPT)
    p.add_argument("--phase7m6e-script", type=Path, default=DEFAULT_7M6E_SCRIPT)
    p.add_argument("--phase6a-script", type=Path, default=DEFAULT_PHASE6A_SCRIPT)
    p.add_argument("--phase7e-script", type=Path, default=DEFAULT_PHASE7E_SCRIPT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--force-day", action="append", default=[], help="Force rebuild of one YYYY-MM-DD v2 date; repeatable.")
    p.add_argument("--force-all", action="store_true", help="Force rebuild every v2 date. Expensive; normally leave off.")
    p.add_argument("--price-scale", type=float, default=PRICE_SCALE)
    p.add_argument("--depth-batch-size", type=int, default=DEPTH_STREAM_BATCH_SIZE_7M6F,
                   help="Operational Arrow depth batch size; does not change scientific semantics.")
    p.add_argument("--in-process-raw", action="store_true",
                   help="Disable the default fresh-process-per-raw-date supervisor.")
    p.add_argument("--worker-day", default=None, help=argparse.SUPPRESS)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)



def run_raw_day_worker(args: argparse.Namespace, day: str) -> None:
    """Run one expensive raw date in a fresh Python process.

    This is operational isolation only. The child writes the exact same signed
    v2 checkpoint, then exits so Windows can reclaim all Arrow/Numba/Python
    allocations before the next date starts. A child crash therefore does not
    corrupt prior checkpoints or take the supervising process with it.
    """
    log_dir = Path(args.report_root) / "worker_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{day}.log"
    cmd = [
        sys.executable, str(SCRIPT_PATH),
        "--symbol", str(args.symbol),
        "--m3-root", str(args.m3_root),
        "--phase7m6b-root", str(args.phase7m6b_root),
        "--phase7m6e-root", str(args.phase7m6e_root),
        "--canonical-root", str(args.canonical_root),
        "--phase7m6b-script", str(args.phase7m6b_script),
        "--phase7m6e-script", str(args.phase7m6e_script),
        "--phase6a-script", str(args.phase6a_script),
        "--phase7e-script", str(args.phase7e_script),
        "--output-root", str(args.output_root),
        "--report-root", str(args.report_root),
        "--price-scale", str(args.price_scale),
        "--depth-batch-size", str(args.depth_batch_size),
        "--worker-day", str(day),
        "--force-day", str(day),
    ]
    print(
        f"      {day}: starting isolated raw worker "
        f"(depth_batch={int(args.depth_batch_size):,}, log={log_path})",
        flush=True,
    )
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n===== {utc_now_iso()} isolated worker start =====\n")
        child_env = os.environ.copy()
        # The isolated worker writes to a PIPE on Windows, so its stdout may
        # otherwise fall back to a legacy ANSI code page such as cp1252 even
        # when the parent terminal is UTF-8 capable. Force an explicit UTF-8
        # contract for both child encoding and parent-side decoding.
        child_env["PYTHONUTF8"] = "1"
        child_env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            try:
                print(line, end="", flush=True)
            except (BrokenPipeError, OSError):
                # Keep the worker/log alive even if the console sink disappears.
                pass
        rc = int(proc.wait())
        log.write(f"===== {utc_now_iso()} isolated worker exit={rc} =====\n")
    if rc != 0:
        raise RuntimeError(
            f"Isolated raw worker failed for {day} with exit code {rc}. "
            f"Completed earlier dates remain reusable. See {log_path}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test(args.phase7m6b_script, args.phase7m6e_script)
    require_stack()
    if abs(float(args.price_scale) - PRICE_SCALE) > 1e-12:
        raise SystemExit(f"Phase 7M.6F is frozen to price_scale={PRICE_SCALE}")
    if not (1024 <= int(args.depth_batch_size) <= 8192):
        raise SystemExit("--depth-batch-size must be between 1024 and 8192")

    print("Phase 7M.6F - Full corrected anchor-relative refill->re-depletion substrate")
    print("[1/8] Verifying Phase 7M.6E precompute authorization (before expensive replay)...")
    gate_meta, pilot_dates = verify_7m6e_authorization(args.phase7m6e_root)

    print("[2/8] Verifying frozen Phase 7M.3 causal universe and helper contracts...")
    phase7m6b = load_module(args.phase7m6b_script, "phase7m6b_for_7m6f")
    phase7m6e = load_module(args.phase7m6e_script, "phase7m6e_for_7m6f")
    phase6a = load_module(args.phase6a_script, "phase6a_for_7m6f")
    phase7e = load_module(args.phase7e_script, "phase7e_for_7m6f")
    # Keep frozen 7M.6B matching/trade helper settings. Raw depth decoding itself
    # uses this builder's direct RecordBatch path and an operational batch size.
    phase7m6b.DEPTH_BATCH_SIZE = 1024
    phase7m6b.TRADE_BATCH_SIZE = 32768
    phase7m6b.MATCH_Z_CALIPER = MATCH_Z_CALIPER
    phase7m6b.MATCH_EUCLIDEAN_CALIPER = MATCH_EUCLIDEAN_CALIPER
    phase7m6b.BALANCE_SMD_GATE = BALANCE_SMD_GATE

    _, candidates, mechanics = phase7m6b.load_m3(args.m3_root)
    clock = phase7m6b.derive_candidate_clock(candidates, mechanics)
    dates = sorted(clock["candidate_date"].astype(str).unique().tolist())
    if len(clock) != EXPECTED_CANDIDATES or len(dates) != EXPECTED_DAYS:
        raise RuntimeError(f"Frozen full universe mismatch: candidates={len(clock)} dates={len(dates)}")
    if args.worker_day is not None and str(args.worker_day) not in dates:
        raise RuntimeError(f"--worker-day is not in frozen candidate dates: {args.worker_day}")

    # Canonical discovery is done once. Pilot reuse never reads the raw files.
    all_days = phase6a.discover_days(args.canonical_root, None, None, None)
    day_map = {d.day.isoformat(): d for d in all_days}
    missing_raw = [d for d in dates if d not in day_map and d not in pilot_dates]
    if missing_raw:
        raise RuntimeError(f"Missing canonical raw dates required by full replay: {missing_raw}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_root / "metadata" / "run_config.json", {
        "generated_at": utc_now_iso(),
        "schema_name": SCHEMA_NAME,
        "contract": contract_payload(),
        "candidate_dates": dates,
        "pilot_reuse_dates": pilot_dates,
        "phase7m6e_authorization_status": gate_meta.get("status"),
        "future_outcomes_loaded": False,
    })

    print(f"[3/8] Building/reusing {len(dates)} signed v2 date checkpoints...")
    primary_parts: list[pd.DataFrame] = []
    progress: list[dict[str, Any]] = []
    raw_built = 0
    pilot_reused = 0
    v2_reused = 0
    forced = set(map(str, args.force_day))

    for i, day in enumerate(dates, 1):
        if args.worker_day is not None and day != str(args.worker_day):
            continue
        day_clock = clock[clock["candidate_date"].astype(str) == day].copy()
        ids = day_clock["candidate_id"].astype(str).tolist()
        outp = checkpoint_paths(args.output_root, day)
        is_pilot = day in pilot_dates
        if is_pilot:
            sourcep = source_7m6b_paths(args.phase7m6b_root, day)
            sig = v2_signature_pilot_reuse(day, ids, sourcep)
        else:
            day_files = day_map[day]
            level_path = args.m3_root / "level_audit" / f"candidate_date={day}" / "part.parquet"
            sig = v2_signature_raw(day, ids, level_path, day_files)

        force = args.force_all or day in forced
        if not force and can_reuse_v2(outp, sig):
            metrics = pq.read_table(outp["metrics"]).to_pandas()
            manifest = json.loads(outp["manifest"].read_text(encoding="utf-8"))
            diag = manifest.get("diagnostics", {})
            state = "V2_REUSED"
            v2_reused += 1
        else:
            if is_pilot:
                metrics, levels, tape, diag = reconstruct_pilot_day_from_7m6b(
                    day=day, day_clock=day_clock, source_root=args.phase7m6b_root,
                )
                state = "PILOT_TAPE_REUSED"
                pilot_reused += 1
                source_mode = "REUSED_7M6B_EVENT_TAPE"
            else:
                if args.worker_day is None and not args.in_process_raw:
                    # Expensive replay runs in a fresh process. The child writes
                    # the signed checkpoint and exits; the parent then reads it.
                    run_raw_day_worker(args, day)
                    if not can_reuse_v2(outp, sig):
                        raise RuntimeError(f"Isolated worker returned success but checkpoint failed validation: {day}")
                    metrics = pq.read_table(outp["metrics"]).to_pandas()
                    child_manifest = json.loads(outp["manifest"].read_text(encoding="utf-8"))
                    diag = child_manifest.get("diagnostics", {})
                    state = "RAW_BUILT_ISOLATED"
                    raw_built += 1
                    source_mode = "RAW_CORRECTED_ANCHOR_RELATIVE"
                    # The child has already atomically written metrics/levels/tape/manifest.
                    manifest = child_manifest
                else:
                    day_files = day_map[day]
                    lev = phase7m6b.load_level_audit_for_day(args.m3_root, day, set(ids))
                    metrics, diag = raw_process_day_v2(
                        day=day, day_clock=day_clock, level_audit=lev, day_files=day_files,
                        phase6a=phase6a, phase7e=phase7e, phase7m6b=phase7m6b,
                        symbol=args.symbol, price_scale=float(args.price_scale),
                        level_output_path=outp["levels"], tape_output_path=outp["tape"],
                        depth_batch_size=int(args.depth_batch_size),
                        snapshot_batch_size=SNAPSHOT_STREAM_BATCH_SIZE_7M6F,
                    )
                    state = "RAW_BUILT"
                    raw_built += 1
                    source_mode = "RAW_CORRECTED_ANCHOR_RELATIVE"
                    del lev

            if state != "RAW_BUILT_ISOLATED":
                atomic_parquet(outp["metrics"], metrics)
            if is_pilot:
                atomic_parquet(outp["levels"], levels)
                atomic_parquet(outp["tape"], tape)
            if state != "RAW_BUILT_ISOLATED":
                manifest = {
                    "builder_version": BUILDER_VERSION,
                    "schema_name": SCHEMA_NAME,
                    "schema_version": SCHEMA_VERSION,
                    "candidate_date": day,
                    "candidate_count": len(ids),
                    "candidate_ids_sha256": sha256_text("\n".join(sorted(ids))),
                    "input_signature": sig,
                    "status": "PASS",
                    "generated_at": utc_now_iso(),
                    "future_outcomes_loaded": False,
                    "source_mode": source_mode,
                    "diagnostics": diag,
                    "contract": contract_payload(),
                }
                atomic_json(outp["manifest"], manifest)
            if is_pilot:
                del levels, tape

        p3 = metrics[pd.to_numeric(metrics["observation_horizon_sec"], errors="coerce") == PRIMARY_HORIZON_SEC].copy()
        primary_parts.append(p3)
        progress.append({"candidate_date": day, "candidate_count": len(ids), "state": state, **diag})
        print(
            f"  [{i:02d}/{len(dates):02d}] {day}: candidates={len(ids):4d} {state:<17} "
            f"anchor={diag.get('anchor_found', 0)} valid={diag.get('sequence_valid', 0)} "
            f"trades={int(diag.get('filtered_trade_rows', 0) or 0):,} "
            f"replay_s={safe_float(diag.get('raw_replay_seconds')):.1f} "
            f"post_s={safe_float(diag.get('post_replay_seconds')):.1f} "
            f"depth/s={safe_float(diag.get('depth_rows_per_second')):,.0f}"
        )
        if args.worker_day is not None:
            # Worker contract ends after one signed date checkpoint. Never run
            # full-data aggregation/matching inside the isolated child.
            return 0
        del metrics, p3
        gc.collect()
        try:
            pa.default_memory_pool().release_unused()
        except Exception:
            pass

    atomic_csv(args.report_root / "full_replay_progress.csv", pd.DataFrame(progress))
    primary = pd.concat(primary_parts, ignore_index=True, sort=False)
    if len(primary) != EXPECTED_CANDIDATES or primary["candidate_id"].astype(str).duplicated().any():
        raise RuntimeError(f"Primary 3s full candidate invariant failed: rows={len(primary)} duplicates={primary['candidate_id'].astype(str).duplicated().sum()}")

    print("[4/8] Auditing structural coverage and corrected refill parity...")
    anchor_rate = float(bool_series(primary["anchor_found"]).mean())
    seq_rate = float(bool_series(primary["sequence_valid"]).mean())
    horizon_rate = float(bool_series(primary["horizon_complete"]).mean())
    refill = primary[pd.to_numeric(primary["opposing__observed_refill_quote"], errors="coerce") > 0].copy()
    strict_cov = float(pd.to_numeric(refill["opposing__strict_trade_time_complete_share"], errors="coerce").fillna(1.0).mean()) if len(refill) else 0.0
    structural_pass = bool(
        anchor_rate >= MIN_ANCHOR_MATCH_RATE
        and seq_rate >= MIN_SEQUENCE_VALID_RATE
        and horizon_rate >= MIN_SEQUENCE_VALID_RATE
        and strict_cov >= MIN_STRICT_TRADE_TIME_COVERAGE
    )
    parity = parity_summary(primary)
    atomic_csv(args.report_root / "frozen_7m3_refill_parity_full.csv", parity)

    # Horizon coverage documents the intentional pilot 10s gap without contaminating primary 3s.
    horizon_rows = []
    for h in EXPECTED_HORIZONS:
        parts = []
        for day in dates:
            mp = checkpoint_paths(args.output_root, day)["metrics"]
            t = pq.read_table(mp, columns=["candidate_id", "observation_horizon_sec", "horizon_complete", "invalid_reason"]).to_pandas()
            t = t[pd.to_numeric(t["observation_horizon_sec"], errors="coerce") == h]
            parts.append(t)
        hh = pd.concat(parts, ignore_index=True, sort=False)
        horizon_rows.append({
            "observation_horizon_sec": h,
            "candidate_rows": int(len(hh)),
            "horizon_complete": int(bool_series(hh["horizon_complete"]).sum()),
            "coverage_rate": float(bool_series(hh["horizon_complete"]).mean()) if len(hh) else 0.0,
            "pilot_saved_tape_insufficient": int((hh["invalid_reason"].astype(str) == "PILOT_SAVED_TAPE_ENDS_BEFORE_CORRECTED_ANCHOR_HORIZON").sum()),
        })
    atomic_csv(args.report_root / "horizon_coverage.csv", pd.DataFrame(horizon_rows))

    print("[5/8] Running the actual full-data pre-outcome binary common-support gate...")
    gate, pairs, subgroups = full_binary_gate(primary, phase7m6b, phase7m6e)
    atomic_csv(args.report_root / "full_preoutcome_gate.csv", gate)
    atomic_csv(args.report_root / "binary_penetration_subgroup_feasibility_full.csv", subgroups)
    atomic_csv(args.report_root / "binary_penetration_pair_manifest_full.csv", pairs)
    atomic_parquet(args.output_root / "full_matched_pairs_outcome_blind.parquet", pairs)

    analysis = phase7m6e.make_analysis_frame(primary)
    manifest_cols = [
        "candidate_id", "candidate_date", "direction", "anchor_found", "sequence_valid",
        "opposing__observed_refill_quote", "opposing__refilled_level_count",
        "opposing__replacement_execution_lower_bound_quote",
        "replacement_penetration_eligible", "replacement_penetrated",
        "control__directional_mid_move_bps_3s", "control__directional_aggressive_quote_3s",
        "control__opposing_peak_initial_queue_loss_share_3s", "is_reaccel_context",
    ]
    atomic_parquet(args.output_root / "binary_penetration_candidate_manifest.parquet", analysis[manifest_cols].copy())

    print("[6/8] Writing full invariants...")
    g = gate.iloc[0]
    match_pass = bool(g["feasible_before_outcomes"])
    all_subgroups_both = bool(subgroups["both_exposure_states_present"].all()) if len(subgroups) else False
    inv = pd.DataFrame([
        {"invariant": "future_outcomes_loaded", "pass": True, "actual": False, "required": False},
        {"invariant": "phase7m6e_precompute_gate_authorized_replay", "pass": True, "actual": gate_meta.get("status"), "required": "BINARY_PENETRATION_FEASIBLE_FOR_FULL_REPLAY"},
        {"invariant": "full_candidate_count", "pass": len(primary) == EXPECTED_CANDIDATES, "actual": len(primary), "required": EXPECTED_CANDIDATES},
        {"invariant": "full_candidate_dates", "pass": len(dates) == EXPECTED_DAYS, "actual": len(dates), "required": EXPECTED_DAYS},
        {"invariant": "primary_anchor_match_rate", "pass": anchor_rate >= MIN_ANCHOR_MATCH_RATE, "actual": anchor_rate, "required": MIN_ANCHOR_MATCH_RATE},
        {"invariant": "primary_sequence_valid_rate", "pass": seq_rate >= MIN_SEQUENCE_VALID_RATE, "actual": seq_rate, "required": MIN_SEQUENCE_VALID_RATE},
        {"invariant": "primary_horizon_complete_rate", "pass": horizon_rate >= MIN_SEQUENCE_VALID_RATE, "actual": horizon_rate, "required": MIN_SEQUENCE_VALID_RATE},
        {"invariant": "strict_trade_time_coverage", "pass": strict_cov >= MIN_STRICT_TRADE_TIME_COVERAGE, "actual": strict_cov, "required": MIN_STRICT_TRADE_TIME_COVERAGE},
        {"invariant": "refill_breadth_retained_in_primary_controls", "pass": True, "actual": True, "required": True},
        {"invariant": "binary_physical_zero_boundary_unchanged", "pass": True, "actual": "lower_bound_quote > 0", "required": "lower_bound_quote > 0"},
        {"invariant": "all_key_subgroups_have_both_exposure_states", "pass": all_subgroups_both, "actual": all_subgroups_both, "required": True},
        {"invariant": "full_preoutcome_common_support_gate", "pass": match_pass, "actual": match_pass, "required": True},
        {"invariant": "pair_manifest_frozen_before_outcomes", "pass": True, "actual": str(args.output_root / "full_matched_pairs_outcome_blind.parquet"), "required": "written before future labels"},
    ])
    atomic_csv(args.report_root / "analysis_invariants.csv", inv)

    print("[7/8] Freezing outcome-blind summary and findings...")
    # UP/DOWN/reacceleration representation is reported diagnostically; no new
    # numeric subgroup gate is introduced here. The mandatory full gate is the
    # frozen overall + discovery/validation common-support contract.
    overall_pass = bool(structural_pass and match_pass)
    status = "FULL_PREOUTCOME_GATE_PASS_READY_TO_FREEZE_PAIRS" if overall_pass else "FULL_PREOUTCOME_GATE_FAIL_DO_NOT_LOAD_OUTCOMES"
    summary = {
        "builder_version": BUILDER_VERSION,
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": status,
        "candidate_count": int(len(primary)),
        "candidate_dates": len(dates),
        "raw_built_dates": raw_built,
        "pilot_reused_dates": pilot_reused,
        "v2_reused_dates": v2_reused,
        "primary_anchor_match_rate": anchor_rate,
        "primary_sequence_valid_rate": seq_rate,
        "primary_horizon_complete_rate": horizon_rate,
        "strict_trade_time_coverage": strict_cov,
        "primary_preoutcome_gate_pass": match_pass,
        "full_matched_pairs": int(g["pairs_after_repair"]),
        "discovery_pairs": int(g["discovery_pairs"]),
        "validation_pairs": int(g["validation_pairs"]),
        "max_abs_smd": safe_float(g["max_abs_smd"]),
        "future_outcomes_loaded": False,
        "next_step_if_pass": "Review/freeze full pair manifest, then separately authorize outcome loading.",
        "next_step_if_fail": "Do not load outcomes; do not relax frozen matching or penetration definition to force feasibility.",
    }
    atomic_json(args.output_root / "metadata" / "latest_summary.json", summary)
    atomic_text(args.report_root / "findings.md", findings_text(summary, gate, parity))

    print("[8/8] Complete.")
    print(f"Full status              : {status}")
    print(f"Candidates / dates       : {len(primary):,} / {len(dates)}")
    print(f"Checkpoints              : {raw_built} raw built / {pilot_reused} pilot-tape built / {v2_reused} v2 reused")
    print(f"Anchor / sequence valid  : {anchor_rate:.2%} / {seq_rate:.2%}")
    print(f"Full matched pairs       : {int(g['pairs_after_repair']):,}")
    print(f"Discovery / validation   : {int(g['discovery_pairs']):,} / {int(g['validation_pairs']):,}")
    print(f"Max |SMD|                : {safe_float(g['max_abs_smd']):.3f}")
    print(f"Pre-outcome gate pass    : {match_pass}")
    print(f"Reports                  : {args.report_root}")
    print("Future outcomes loaded   : NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
